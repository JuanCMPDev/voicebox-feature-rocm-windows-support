"""Profile the decoding step of the Qwen3-TTS talker (plan, Fase 3.1) before Fase 3 changes it.

Loads the model as ``PyTorchTTSBackend`` does (BF16, ``device_map="cuda"``), takes the cached voice prompt from a bench
data dir and runs the talker (``Qwen3TTSForConditionalGeneration.generate``, what ``generate_voice_clone`` runs before
the codec) on one phrase of ``scripts/bench_phrases.json`` with its seed, in four ways:

1. ``timing``: plain runs, interleaved with what tasks 3.3 and 3.6 would change (``torch.inference_mode``, an
   above-normal process priority) and followed by a batch of two copies of the phrase.
2. ``light``: host-clock timers (no GPU synchronization added) around each decode-step talker forward, its 28-layer
   backbone, ``code_predictor.generate`` and each code-predictor forward, plus every host read of a GPU value made from
   Python (``bool(tensor)``, ``tensor.item()``): the time the CPU spends waiting for the GPU. It also copies one
   mid-generation step's inputs (KV caches included) for the replays.
3. ``profiler``: ``torch.profiler`` over a window of decode steps, with the same regions as ``record_function`` ranges:
   time inside ATen vs Python, host reads (``aten::_local_scalar_dense``, also those made inside ATen ops) and leaf
   ATen ops per step, an estimate of kernel launches. This ROCm build for Windows has no GPU activity in the profiler.
4. ``replay``: the captured step's units (backbone + codec head, one code-predictor forward, their linear layers alone,
   each sampling chain, 200 small kernels) captured in a HIP graph and replayed: wall time over back-to-back replays is
   the unit's GPU time with no Python or per-op dispatch in between (an upper bound for units of many small kernels,
   since a replay costs some host time per node). HIP events are not used: on this stack they come out shorter than the
   work queued between them.

Before all that, each of ``ENV_VARIANTS`` (AOTriton's experimental attention, TunableOp) is profiled in a child process
of this script, since torch reads those settings at start-up; a ``base`` child gives the matching baseline.

Nothing here changes the model, its inputs or the sampling; the wrappers only time, annotate and copy, and are removed
after each run. Writes ``profile-<phrase>.json``, ``profile-<phrase>.md``, the profiler's Chrome trace and op table to
``--out`` (``perf-loop/bench/_profile`` by default, not tracked by git).

Usage:
    backend/venv/Scripts/python.exe scripts/profile_talker.py --phrase p08
"""

import argparse
import contextlib
import copy
import functools
import json
import logging
import math
import statistics
import sys
from bisect import bisect_left
from collections import Counter
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

logger = logging.getLogger("profile_talker")

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = ROOT / "perf-loop" / "bench" / "_data"
DEFAULT_OUT = ROOT / "perf-loop" / "bench" / "_profile"
DEFAULT_PHRASES = ROOT / "scripts" / "bench_phrases.json"

HF_REPOS = {"1.7B": "Qwen/Qwen3-TTS-12Hz-1.7B-Base"}  # the plan keeps the 1.7B model
# backend.backends.LANGUAGE_CODE_TO_NAME, copied so the script does not import the app.
LANGUAGE_NAMES = {
    "zh": "chinese",
    "en": "english",
    "ja": "japanese",
    "ko": "korean",
    "de": "german",
    "fr": "french",
    "ru": "russian",
    "pt": "portuguese",
    "es": "spanish",
    "it": "italian",
}
PROMPT_FIELDS = ("ref_code", "ref_spk_embedding", "x_vector_only_mode", "icl_mode")

# Regions: record_function names in the profiler run, region keys of the timer records.
R_PREFILL = "vb::talker.prefill"
R_STEP = "vb::talker.step"  # decode-step talker forward: code predictor, backbone and codec head
R_BACKBONE = "vb::talker.backbone"  # Qwen3TTSTalkerModel.forward (28 layers)
R_CP_GENERATE = "vb::cp.generate"  # code_predictor.generate: 15 codes of one frame
R_CP_FORWARD = "vb::cp.forward"  # one code-predictor forward (5 layers + its lm_head)
R_MARK = "vb::mark"  # end of the profiled window

# Where a host read happened (innermost region), as the region keys of the reports.
READ_REGIONS = {
    R_BACKBONE: "backbone",
    R_CP_FORWARD: "cp_forward",
    R_CP_GENERATE: "cp_generate_rest",
    R_STEP: "talker_forward_rest",
    R_PREFILL: "prefill",
}
STEP_REGIONS = ("backbone", "cp_forward", "cp_generate_rest", "talker_forward_rest", "talker_loop")

SYNC_OP = "aten::_local_scalar_dense"  # host read of a GPU value (.item(), bool(tensor)): waits for the GPU
# Leaf ATen ops that launch no GPU work: views, metadata, allocations, host reads (counted apart as syncs).
NO_KERNEL_OPS = frozenset(
    {
        "aten::_has_compatible_shallow_copy_type",
        "aten::_local_scalar_dense",
        "aten::_reshape_alias",
        "aten::_unsafe_view",
        "aten::alias",
        "aten::as_strided",
        "aten::broadcast_tensors",
        "aten::chunk",
        "aten::contiguous",
        "aten::detach",
        "aten::detach_",
        "aten::dim",
        "aten::empty",
        "aten::empty_like",
        "aten::empty_strided",
        "aten::expand",
        "aten::expand_as",
        "aten::flatten",
        "aten::is_complex",
        "aten::is_floating_point",
        "aten::is_nonzero",
        "aten::item",
        "aten::lift",
        "aten::lift_fresh",
        "aten::narrow",
        "aten::numel",
        "aten::permute",
        "aten::reshape",
        "aten::resize_",
        "aten::result_type",
        "aten::select",
        "aten::set_",
        "aten::size",
        "aten::slice",
        "aten::split",
        "aten::split_with_sizes",
        "aten::squeeze",
        "aten::stride",
        "aten::t",
        "aten::to",
        "aten::transpose",
        "aten::unbind",
        "aten::unflatten",
        "aten::unsqueeze",
        "aten::view",
        "aten::view_as",
    }
)

ABOVE_NORMAL_PRIORITY_CLASS = 0x00008000
NORMAL_PRIORITY_CLASS = 0x00000020
DUMMY_SIZE = 8192  # side of the BF16 matmul that keeps the GPU busy while the CPU queues a unit


class ProfileError(Exception):
    """A setup problem the script reports and exits on (no GPU, no prompt, unknown phrase)."""


# --------------------------------------------------------------------------------------------------------------------
# Pure helpers (unit-tested without a GPU)
# --------------------------------------------------------------------------------------------------------------------


def stats(values: Iterable[float]) -> dict[str, float | None]:
    """Mean, median, min and max of ``values`` rounded to 3 decimals; all None when empty."""
    data = [float(v) for v in values]
    if not data:
        return {"mean": None, "median": None, "min": None, "max": None, "n": 0}
    return {
        "mean": round(statistics.fmean(data), 3),
        "median": round(statistics.median(data), 3),
        "min": round(min(data), 3),
        "max": round(max(data), 3),
        "n": len(data),
    }


def load_phrase(phrases_path: Path, phrase_id: str) -> dict[str, Any]:
    """``{"id", "text", "seed", "language"}`` of ``phrase_id`` in a ``bench_phrases.json`` file."""
    data = json.loads(phrases_path.read_text(encoding="utf-8"))
    for phrase in data.get("phrases", []):
        if phrase.get("id") == phrase_id:
            return {
                "id": phrase_id,
                "text": phrase["text"],
                "seed": int(phrase["seed"]),
                "language": data.get("language", "en"),
            }
    raise ProfileError(f"phrase {phrase_id!r} not in {phrases_path}")


def find_prompt_file(data_dir: Path, model_size: str) -> Path:
    """The only cached Qwen voice prompt of ``model_size`` in ``data_dir/cache`` (task 2.1 format)."""
    files = sorted((data_dir / "cache").glob(f"qwen_{model_size}_*.prompt"))
    if len(files) != 1:
        raise ProfileError(
            f"expected one qwen_{model_size}_*.prompt in {data_dir / 'cache'}, found {len(files)}; "
            "run the bench once (it caches the prompt) or pass --prompt-file"
        )
    return files[0]


def validate_cached_prompt(cached: Any) -> dict[str, list]:
    """The cached prompt dict if it has every field as same-length lists; ProfileError otherwise."""
    names = (*PROMPT_FIELDS, "ref_text")
    if not isinstance(cached, dict) or not all(isinstance(cached.get(name), list) for name in names):
        raise ProfileError("cached voice prompt is not the dict of lists written by PyTorchTTSBackend (task 2.1)")
    if len({len(cached[name]) for name in names}) != 1 or not cached["ref_code"]:
        raise ProfileError("cached voice prompt has fields of different lengths or no items")
    return cached


def dummy_repeats(cover_ms: float, dummy_ms: float, margin: float = 2.0, minimum: int = 2) -> int:
    """How many dummy GPU jobs of ``dummy_ms`` keep the GPU busy for ``margin`` times ``cover_ms``."""
    if dummy_ms <= 0:
        return minimum
    return max(minimum, math.ceil(margin * cover_ms / dummy_ms) + 1)


@dataclass(frozen=True)
class Call:
    """One timed span on the host clock (seconds): a region call or a host read.

    ``step`` is the decode step (-1 for the prefill); ``index`` the call's index within its parent (code-predictor
    forwards); for host reads, ``region`` is where the read happened (a key of ``READ_REGIONS`` or ``talker_loop``).
    """

    region: str
    step: int
    index: int
    start: float
    end: float

    @property
    def ms(self) -> float:
        return (self.end - self.start) * 1000.0


def light_steps(calls: list[Call], reads: list[Call], generate_end: float) -> list[dict[str, float]]:
    """Per decode step, the host-clock split of the timer run.

    A step runs from the start of its talker forward to the start of the next one (or ``generate_end``): the forward
    (code predictor, backbone, codec head) plus HF's sampling loop around it (logits processors, sampling, stopping).
    ``host_read_ms`` is the time the CPU waited on host reads during the step; ``cpu_busy_ms`` the rest.
    """
    forwards = sorted((c for c in calls if c.region == R_STEP), key=lambda c: c.start)
    by_step: dict[int, list[Call]] = {}
    for call in calls:
        by_step.setdefault(call.step, []).append(call)
    reads_by_step: dict[int, list[Call]] = {}
    for read in reads:
        reads_by_step.setdefault(read.step, []).append(read)
    rows = []
    for i, forward in enumerate(forwards):
        end = forwards[i + 1].start if i + 1 < len(forwards) else generate_end
        inside = by_step.get(forward.step, [])
        cp_forwards = [c for c in inside if c.region == R_CP_FORWARD]
        step_reads = reads_by_step.get(forward.step, [])
        row: dict[str, float] = {
            "step": forward.step,
            "step_ms": (end - forward.start) * 1000.0,
            "talker_forward_ms": forward.ms,
            "cp_generate_ms": sum(c.ms for c in inside if c.region == R_CP_GENERATE),
            "cp_forward_ms": sum(c.ms for c in cp_forwards),
            "cp_forward_n": len(cp_forwards),
            "backbone_ms": sum(c.ms for c in inside if c.region == R_BACKBONE),
            "host_read_n": len(step_reads),
            "host_read_ms": sum(r.ms for r in step_reads),
        }
        row["cp_generate_rest_ms"] = row["cp_generate_ms"] - row["cp_forward_ms"]
        row["talker_forward_rest_ms"] = row["talker_forward_ms"] - row["cp_generate_ms"] - row["backbone_ms"]
        row["talker_loop_ms"] = row["step_ms"] - row["talker_forward_ms"]
        row["cpu_busy_ms"] = row["step_ms"] - row["host_read_ms"]
        for region in STEP_REGIONS:
            mine = [r for r in step_reads if r.region == region]
            row[f"host_read_n.{region}"] = len(mine)
            row[f"host_read_ms.{region}"] = sum(r.ms for r in mine)
        rows.append(row)
    return rows


LIGHT_KEYS = (
    "step_ms",
    "cpu_busy_ms",
    "host_read_ms",
    "host_read_n",
    "talker_forward_ms",
    "backbone_ms",
    "cp_generate_ms",
    "cp_forward_ms",
    "cp_generate_rest_ms",
    "talker_forward_rest_ms",
    "talker_loop_ms",
    "cp_forward_n",
    *(f"host_read_{kind}.{region}" for region in STEP_REGIONS for kind in ("n", "ms")),
)


def summarize_rows(rows: list[dict[str, float]], keys: Iterable[str]) -> dict[str, dict[str, float | None]]:
    """``stats`` of each key over the rows."""
    return {key: stats(row[key] for row in rows if key in row) for key in keys}


@dataclass(frozen=True)
class Ev:
    """A profiler event on the generation thread, in microseconds.

    ``top``: an ATen op not nested in another ATen op (top-level ops never overlap). ``leaf``: an op with no child.
    ``source``: for host reads, the outermost ATen op that issued it (e.g. ``aten::is_nonzero`` for ``bool(t)``).
    """

    name: str
    start: float
    end: float
    top: bool = False
    leaf: bool = False
    source: str | None = None


def is_kernel_op(event: Ev) -> bool:
    """Whether a leaf ATen op launches GPU work (a kernel or a copy): the per-op launch estimate."""
    return event.leaf and event.name.startswith("aten::") and event.name not in NO_KERNEL_OPS


def events_from_profiler(function_events: Iterable[Any]) -> tuple[list[Ev], list[Ev]]:
    """(annotations, ATen ops) of the thread that ran the ``vb::`` ranges, from ``torch.profiler``'s FunctionEvents."""
    events = list(function_events)
    threads = Counter(e.thread for e in events if e.name.startswith("vb::"))
    if not threads:
        return [], []
    thread = threads.most_common(1)[0][0]
    annotations: list[Ev] = []
    ops: list[Ev] = []
    for event in events:
        if event.thread != thread:
            continue
        start, end = float(event.time_range.start), float(event.time_range.end)
        if event.name.startswith("vb::"):
            annotations.append(Ev(event.name, start, end))
            continue
        if not event.name.startswith("aten::"):
            continue
        parent = event.cpu_parent
        top = parent is None or not parent.name.startswith("aten::")
        source = None
        if event.name == SYNC_OP:
            source = event.name
            node = parent
            while node is not None and node.name.startswith("aten::"):
                source = node.name
                node = node.cpu_parent
        ops.append(Ev(event.name, start, end, top=top, leaf=not event.cpu_children, source=source))
    annotations.sort(key=lambda e: e.start)
    ops.sort(key=lambda e: e.start)
    return annotations, ops


class _OpIndex:
    """Additive measures of ATen ops over time windows: top-level ATen time, launch-op count, host reads."""

    def __init__(self, ops: list[Ev]) -> None:
        self.top = sorted((e for e in ops if e.top), key=lambda e: e.start)
        self.top_starts = [e.start for e in self.top]
        self.kernel = sorted(e.start for e in ops if is_kernel_op(e))
        self.syncs = sorted((e for e in ops if e.name == SYNC_OP), key=lambda e: e.start)
        self.sync_starts = [e.start for e in self.syncs]

    def measure(self, lo: float, hi: float) -> dict[str, float]:
        i, j = bisect_left(self.top_starts, lo), bisect_left(self.top_starts, hi)
        aten_us = sum(max(0.0, min(e.end, hi) - max(e.start, lo)) for e in self.top[i:j])
        k, m = bisect_left(self.sync_starts, lo), bisect_left(self.sync_starts, hi)
        return {
            "ms": (hi - lo) / 1000.0,
            "aten_ms": aten_us / 1000.0,
            "sync_ms": sum(e.end - e.start for e in self.syncs[k:m]) / 1000.0,
            "sync_n": m - k,
            "ops_n": bisect_left(self.kernel, hi) - bisect_left(self.kernel, lo),
        }

    def sync_sources(self, lo: float, hi: float) -> Counter:
        k, m = bisect_left(self.sync_starts, lo), bisect_left(self.sync_starts, hi)
        return Counter(e.source or e.name for e in self.syncs[k:m])


def _add(total: dict[str, float], part: dict[str, float], sign: float = 1.0) -> None:
    for key, value in part.items():
        total[key] = total.get(key, 0.0) + sign * value


def _with_python(measure: dict[str, float]) -> dict[str, float]:
    measure["python_ms"] = measure["ms"] - measure["aten_ms"]
    return measure


PROFILE_KEYS = ("ms", "aten_ms", "python_ms", "sync_ms", "sync_n", "ops_n")


def analyze_profile(annotations: list[Ev], ops: list[Ev]) -> dict[str, Any]:
    """Per decode step of the profiled window: time inside ATen vs Python, host reads, launch ops, split by region.

    Regions: ``backbone`` (28-layer talker model), ``cp_forward`` (the code-predictor forwards), ``cp_generate_rest``
    (the rest of ``code_predictor.generate``: HF setup, sampling, stopping checks), ``talker_forward_rest`` (embeddings
    and codec head around them) and ``talker_loop`` (HF's sampling loop between talker forwards).
    """
    index = _OpIndex(ops)
    step_starts = [a.start for a in annotations if a.name == R_STEP]
    boundaries = sorted(step_starts + [a.start for a in annotations if a.name == R_MARK])
    rows: list[dict[str, Any]] = []
    sources: Counter = Counter()
    op_names: Counter = Counter()
    op_time_us: Counter = Counter()
    kernel_ops = [e for e in ops if is_kernel_op(e)]
    kernel_starts = [e.start for e in kernel_ops]
    for step_index, lo in enumerate(step_starts):
        later = [b for b in boundaries if b > lo]
        if not later:
            continue  # the step's end was not recorded
        hi = later[0]
        inside = [a for a in annotations if lo <= a.start < hi]
        step_forward = next(a for a in inside if a.name == R_STEP)
        backbone: dict[str, float] = {}
        cp_forward: dict[str, float] = {}
        cp_generate: dict[str, float] = {}
        for annotation in inside:
            if annotation.name == R_BACKBONE:
                _add(backbone, index.measure(annotation.start, annotation.end))
            elif annotation.name == R_CP_FORWARD:
                _add(cp_forward, index.measure(annotation.start, annotation.end))
            elif annotation.name == R_CP_GENERATE:
                _add(cp_generate, index.measure(annotation.start, annotation.end))
        regions: dict[str, dict[str, float]] = {
            "backbone": backbone,
            "cp_forward": cp_forward,
            "cp_generate_rest": dict(cp_generate),
            "talker_forward_rest": index.measure(step_forward.start, step_forward.end),
            "talker_loop": index.measure(step_forward.end, hi),
        }
        _add(regions["cp_generate_rest"], cp_forward, -1.0)
        _add(regions["talker_forward_rest"], cp_generate, -1.0)
        _add(regions["talker_forward_rest"], backbone, -1.0)
        zero = dict.fromkeys(("ms", "aten_ms", "sync_ms", "sync_n", "ops_n"), 0.0)
        for name in STEP_REGIONS:
            regions[name] = _with_python({**zero, **regions[name]})
        row = _with_python(index.measure(lo, hi))
        row["step"] = step_index
        row["cp_forward_n"] = sum(1 for a in inside if a.name == R_CP_FORWARD)
        row["regions"] = regions
        rows.append(row)
        sources.update(index.sync_sources(lo, hi))
        i, j = bisect_left(kernel_starts, lo), bisect_left(kernel_starts, hi)
        for event in kernel_ops[i:j]:
            op_names[event.name] += 1
            op_time_us[event.name] += event.end - event.start
    n = len(rows)
    summary: dict[str, Any] = {
        "steps": n,
        "step": summarize_rows(rows, PROFILE_KEYS),
        "cp_forward_n": stats(row["cp_forward_n"] for row in rows),
        "regions": {
            name: summarize_rows([row["regions"][name] for row in rows], PROFILE_KEYS) for name in STEP_REGIONS
        },
        "sync_sources_per_step": {name: round(count / n, 2) for name, count in sources.most_common()} if n else {},
        "top_ops_per_step": [
            {"op": name, "per_step": round(count / n, 2), "cpu_ms_per_step": round(op_time_us[name] / n / 1000.0, 3)}
            for name, count in op_names.most_common(25)
        ]
        if n
        else [],
    }
    return {"summary": summary, "rows": rows}


def unit_value(unit: dict[str, Any], key: str = "gpu_ms") -> float:
    """A replayed unit's ``key``; units whose graph capture failed fall back on their eager time (an upper bound)."""
    value = unit.get(key)
    return float(unit["eager_ms"]) if value is None else float(value)


def estimate_gpu_step(
    *,
    backbone_gpu_ms: float,
    cp_forward_gpu_ms: float,
    cp_sampling_gpu_ms: float,
    talker_sampling_gpu_ms: float,
    tiny_gpu_ms: float,
    cp_forwards: float,
    ops_step: float,
    ops_backbone: float,
    ops_cp_forward: float,
    ops_cp_sampling: float,
    ops_talker_sampling: float,
) -> dict[str, float]:
    """GPU time of one decode step from the replayed units; every other launch op is billed as a tiny kernel."""
    other_ops = max(0.0, ops_step - ops_backbone - ops_cp_forward - cp_forwards * ops_cp_sampling - ops_talker_sampling)
    parts = {
        "backbone_ms": backbone_gpu_ms,
        "cp_forward_ms": cp_forwards * cp_forward_gpu_ms,
        "cp_sampling_ms": cp_forwards * cp_sampling_gpu_ms,
        "talker_sampling_ms": talker_sampling_gpu_ms,
        "other_ops_ms": other_ops * tiny_gpu_ms,
    }
    return {
        **{k: round(v, 3) for k, v in parts.items()},
        "other_ops": round(other_ops, 1),
        "total_ms": round(sum(parts.values()), 3),
    }


def estimate_dispatch_step(
    *,
    backbone_dispatch_ms: float,
    cp_forward_dispatch_ms: float,
    cp_sampling_dispatch_ms: float,
    talker_sampling_dispatch_ms: float,
    tiny_dispatch_ms: float,
    cp_forwards: float,
    other_ops: float,
) -> dict[str, float]:
    """CPU time to queue one decode step with the same modules but no host reads (the floor of a sync-free loop)."""
    parts = {
        "backbone_ms": backbone_dispatch_ms,
        "cp_forward_ms": cp_forwards * cp_forward_dispatch_ms,
        "cp_sampling_ms": cp_forwards * cp_sampling_dispatch_ms,
        "talker_sampling_ms": talker_sampling_dispatch_ms,
        "other_ops_ms": other_ops * tiny_dispatch_ms,
    }
    return {**{k: round(v, 3) for k, v in parts.items()}, "total_ms": round(sum(parts.values()), 3)}


def gemv_figures(gpu_ms: float | None, weight_bytes: int, per_step: float) -> dict[str, float | None]:
    """Effective bandwidth of a unit of linear layers and its GPU ms per decode step."""
    if gpu_ms is None or gpu_ms <= 0:
        return {"gb_per_s": None, "ms_per_step": None, "weight_gb": round(weight_bytes / 1e9, 4)}
    return {
        "gb_per_s": round(weight_bytes / (gpu_ms / 1000.0) / 1e9, 1),
        "ms_per_step": round(gpu_ms * per_step, 3),
        "weight_gb": round(weight_bytes / 1e9, 4),
    }


def _fmt(value: Any, digits: int = 2) -> str:
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def render_markdown(result: dict[str, Any]) -> str:
    """Human summary of a ``profile-<phrase>.json`` result."""
    lines = [f"# Perfil del talker: {result['phrase']['id']} (seed {result['phrase']['seed']})", ""]
    env = result.get("env", {})
    lines += [
        f"torch {env.get('torch')} · HIP {env.get('hip')} · {env.get('device_name')} · "
        f"atención talker {env.get('attn_talker')} / code predictor {env.get('attn_code_predictor')}",
        "",
    ]
    timing = result.get("timing", {})
    if timing:
        lines += [
            "## Tiempo del talker (generate completo)",
            "",
            "| Variante | ms totales (mediana) | pasos | ms/paso |",
            "|---|---|---|---|",
        ]
        for name, entry in timing.items():
            lines.append(
                f"| {name} | {_fmt(entry['ms']['median'], 1)} | {entry['steps']} | {_fmt(entry['ms_per_step'], 2)} |"
            )
        lines.append("")
    light = result.get("light", {}).get("summary", {})
    if light:
        lines += [
            "## Reparto por paso (reloj del host, sin sincronizar)",
            "",
            "| Parte | ms (mediana) | ms (media) |",
            "|---|---|---|",
        ]
        lines += [f"| {key} | {_fmt(e['median'])} | {_fmt(e['mean'])} |" for key, e in light.items()]
        lines.append("")
    prof = result.get("profiler", {}).get("summary", {})
    if prof:
        lines += [
            f"## Perfil CPU ({prof['steps']} pasos)",
            "",
            "| Región | ms | ATen ms | Python ms | lecturas host (n) | espera lecturas ms | ops con kernel |",
            "|---|---|---|---|---|---|---|",
        ]
        for name, region in (("paso completo", prof["step"]), *prof["regions"].items()):
            lines.append(
                f"| {name} | {_fmt(region['ms']['mean'])} | {_fmt(region['aten_ms']['mean'])} | "
                f"{_fmt(region['python_ms']['mean'])} | {_fmt(region['sync_n']['mean'], 1)} | "
                f"{_fmt(region['sync_ms']['mean'])} | {_fmt(region['ops_n']['mean'], 0)} |"
            )
        lines += [
            "",
            "Lecturas host por paso según su origen: "
            + ", ".join(f"{k} {v}" for k, v in prof["sync_sources_per_step"].items()),
            "",
            "Ops hoja por paso: " + ", ".join(f"{o['op']} {o['per_step']:.0f}" for o in prof["top_ops_per_step"]),
            "",
        ]
    replay = result.get("replay", {})
    if replay.get("units"):
        lines += [
            "## Unidades del paso capturado (grafo HIP, reloj del host)",
            "",
            "| Unidad | GPU ms (grafo) | lanzar grafo ms | despacho CPU ms | eager ms | ops | error |",
            "|---|---|---|---|---|---|---|",
        ]
        for name, unit in replay["units"].items():
            lines.append(
                f"| {name} | {_fmt(unit.get('gpu_ms'), 3)} | {_fmt(unit.get('graph_launch_ms'), 3)} | "
                f"{_fmt(unit.get('dispatch_ms'), 3)} | {_fmt(unit.get('eager_ms'), 3)} | {_fmt(unit.get('ops'))} | "
                f"{(unit.get('graph_error') or '').splitlines()[0] if unit.get('graph_error') else ''} |"
            )
        lines.append("")
    estimate = result.get("estimate", {})
    if estimate:
        lines += ["## Estimación por paso", "", "```", json.dumps(estimate, indent=2, ensure_ascii=False), "```", ""]
    variants = result.get("env_variants", {})
    if variants:
        lines += [
            "## Variantes de entorno (un proceso cada una)",
            "",
            "| Variante | ms (mediana) | frames | ms/paso | paso (temporizador) | CPU ocupada | espera lecturas | "
            "ops/paso | GEMV talker GB/s | GEMV cp GB/s |",
            "|---|---|---|---|---|---|---|---|---|---|",
        ]
        for name, entry in variants.items():
            if "error" in entry:
                lines.append(f"| {name} | error: {entry['error']} |  |  |  |  |  |  |  |  |")
                continue
            light_v, gemv = entry["light"], entry["gemv"]
            lines.append(
                f"| {name} | {_fmt(entry['ms']['median'], 1)} | {entry['frames']} | {_fmt(entry['ms_per_step'])} | "
                f"{_fmt(light_v['step_ms'])} | {_fmt(light_v['cpu_busy_ms'])} | {_fmt(light_v['host_read_ms'])} | "
                f"{_fmt(entry['launch_ops_per_step'], 0)} | {_fmt(gemv['backbone_linears']['gb_per_s'], 0)} | "
                f"{_fmt(gemv['cp_linears']['gb_per_s'], 0)} |"
            )
        lines.append("")
    return "\n".join(lines)


# --------------------------------------------------------------------------------------------------------------------
# Model, inputs and runs (need torch and a GPU)
# --------------------------------------------------------------------------------------------------------------------


def manual_seed(torch: Any, seed: int) -> None:
    """``backend.backends.base.manual_seed(seed, "cuda")``."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def load_model(torch: Any, model_size: str) -> Any:
    """``Qwen3TTSModel`` loaded exactly as ``PyTorchTTSBackend._load_model_sync`` does on a GPU."""
    from huggingface_hub import constants as hf_constants
    from qwen_tts import Qwen3TTSModel

    return Qwen3TTSModel.from_pretrained(
        HF_REPOS[model_size],
        cache_dir=hf_constants.HF_HUB_CACHE,
        device_map="cuda",
        torch_dtype=torch.bfloat16,
    )


def model_info(torch: Any, model: Any) -> dict[str, Any]:
    talker = model.model.talker
    cp = talker.code_predictor
    return {
        "torch": torch.__version__,
        "hip": torch.version.hip,
        "device_name": torch.cuda.get_device_name(0),
        "dtype": str(talker.dtype),
        "attn_talker": talker.config._attn_implementation,
        "attn_talker_model": talker.model.config._attn_implementation,
        "attn_code_predictor": cp.config._attn_implementation,
        "attn_code_predictor_model": cp.model.config._attn_implementation,
        "talker_layers": talker.config.num_hidden_layers,
        "code_predictor_layers": cp.config.num_hidden_layers,
        "num_code_groups": talker.config.num_code_groups,
        "generate_kwargs": {
            k: v for k, v in model._merge_generate_kwargs().items() if isinstance(v, int | float | bool | str)
        },
    }


def build_talker_inputs(model: Any, phrase: dict[str, Any], cached: dict[str, list], device: str, batch: int = 1):
    """kwargs of ``model.model.generate`` as ``generate_voice_clone`` builds them for a cached prompt (list of items)."""
    language = LANGUAGE_NAMES.get(phrase["language"], "auto")
    texts = [phrase["text"]] * batch
    languages = [language] * batch
    model._validate_languages(languages)
    prompt = {
        name: [value.to(device) if hasattr(value, "to") else value for value in cached[name]] * batch
        for name in PROMPT_FIELDS
    }
    ref_texts = list(cached["ref_text"]) * batch
    input_ids = model._tokenize_texts([model._build_assistant_text(t) for t in texts])
    ref_ids = [model._tokenize_texts([model._build_ref_text(rt)])[0] if rt else None for rt in ref_texts]
    return {
        "input_ids": input_ids,
        "ref_ids": ref_ids,
        "voice_clone_prompt": prompt,
        "languages": languages,
        "non_streaming_mode": False,
        **model._merge_generate_kwargs(),
    }


def run_generate(torch: Any, model: Any, inputs: dict[str, Any], seed: int) -> tuple[float, list[int]]:
    """Seeded talker run between two GPU synchronizations: (ms, frames per item)."""
    manual_seed(torch, seed)
    torch.cuda.synchronize()
    started = perf_counter()
    codes, _ = model.model.generate(**inputs)
    torch.cuda.synchronize()
    return (perf_counter() - started) * 1000.0, [int(c.shape[0]) for c in codes]


@contextlib.contextmanager
def process_priority_above_normal() -> Iterator[bool]:
    """Windows: the process in ABOVE_NORMAL_PRIORITY_CLASS for the block (task 3.3); no-op elsewhere."""
    if sys.platform != "win32":
        yield False
        return
    import ctypes

    kernel32 = ctypes.windll.kernel32
    handle = kernel32.GetCurrentProcess()
    previous = kernel32.GetPriorityClass(handle) or NORMAL_PRIORITY_CLASS
    changed = bool(kernel32.SetPriorityClass(handle, ABOVE_NORMAL_PRIORITY_CLASS))
    try:
        yield changed
    finally:
        kernel32.SetPriorityClass(handle, previous)


def run_timing(torch: Any, model: Any, inputs: dict, batch_inputs: dict | None, seed: int, rounds: int) -> dict:
    """Interleaved plain / inference_mode / above-normal-priority runs, then the batch of two."""
    variants: dict[str, Callable[[], contextlib.AbstractContextManager]] = {
        "base": contextlib.nullcontext,
        "inference_mode": torch.inference_mode,
        "priority_above_normal": process_priority_above_normal,
    }
    results: dict[str, dict[str, Any]] = {name: {"ms": [], "frames": []} for name in variants}
    for round_index in range(rounds):
        for name, make_context in variants.items():
            with make_context():
                ms, frames = run_generate(torch, model, inputs, seed)
            results[name]["ms"].append(ms)
            results[name]["frames"].append(frames)
            logger.info("timing round %d %s: %.1f ms, frames %s", round_index + 1, name, ms, frames)
    if batch_inputs is not None:
        results["batch2"] = {"ms": [], "frames": []}
        for round_index in range(max(1, rounds - 1)):
            ms, frames = run_generate(torch, model, batch_inputs, seed)
            results["batch2"]["ms"].append(ms)
            results["batch2"]["frames"].append(frames)
            logger.info("timing round %d batch2: %.1f ms, frames %s", round_index + 1, ms, frames)
    out = {}
    for name, entry in results.items():
        frames = entry["frames"][0] if entry["frames"] else []
        ms_stats = stats(entry["ms"])
        steps = max(frames) if frames else 0
        out[name] = {
            "ms": ms_stats,
            "frames": frames,
            "frames_all_runs": entry["frames"],
            "steps": steps,
            "ms_per_step": round(ms_stats["median"] / steps, 3) if steps and ms_stats["median"] else None,
            "ms_per_frame_produced": round(ms_stats["median"] / sum(frames), 3) if frames else None,
        }
    return out


_MISSING = object()


def snapshot(torch: Any, value: Any) -> Any:
    """Copy of a call argument: tensors cloned, KV caches deep-copied, containers walked, the rest as is."""
    if isinstance(value, torch.Tensor):
        return value.detach().clone()
    if isinstance(value, dict):
        return {k: snapshot(torch, v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return type(value)(snapshot(torch, v) for v in value)
    if hasattr(value, "get_seq_length") and hasattr(value, "crop"):
        return copy.deepcopy(value)
    return value


class TalkerProbe:
    """Times (and optionally annotates) the talker's regions during one generation; copies one step's inputs.

    Installs instance-level wrappers on the talker forward, its backbone, ``code_predictor.generate`` / ``forward``,
    both ``_get_logits_processor`` and both ``prepare_inputs_for_generation``; ``functools.wraps`` keeps their
    signatures for HF's ``inspect.signature`` checks. With ``time_reads``, ``torch.Tensor.__bool__`` and ``.item``
    are wrapped too, to time each host read and attribute it to the innermost region. Everything is removed on exit.
    """

    def __init__(
        self,
        torch: Any,
        talker: Any,
        *,
        annotate: bool = False,
        time_reads: bool = False,
        capture_step: int | None = None,
        capture_substep: int = 7,
        profiler: Any = None,
        profile_steps: tuple[int, int] | None = None,
    ) -> None:
        self.torch = torch
        self.talker = talker
        self.annotate = annotate
        self.time_reads = time_reads
        self.capture_step = capture_step
        self.capture_substep = capture_substep
        self.profiler = profiler
        self.profile_steps = profile_steps
        self.profiling = False
        self.calls: list[Call] = []
        self.reads: list[Call] = []
        self.captured: dict[str, Any] = {}
        self.step = -1
        self.cp_index = 0
        self.prefilled = False
        self._stack: list[str] = []
        self._saved: list[tuple[Any, str, Any]] = []

    # -- install / remove ------------------------------------------------------------------------------------------
    def __enter__(self) -> "TalkerProbe":
        talker = self.talker
        cp = talker.code_predictor
        self._install(talker, "forward", self._wrap_talker_forward(talker.forward))
        self._install(talker.model, "forward", self._wrap_backbone(talker.model.forward))
        self._install(cp, "generate", self._wrap_cp_generate(cp.generate))
        self._install(cp, "forward", self._wrap_cp_forward(cp.forward))
        self._install(talker, "_get_logits_processor", self._wrap_processors("talker", talker._get_logits_processor))
        self._install(cp, "_get_logits_processor", self._wrap_processors("cp", cp._get_logits_processor))
        self._install(
            talker, "prepare_inputs_for_generation", self._wrap_talker_prepare(talker.prepare_inputs_for_generation)
        )
        self._install(cp, "prepare_inputs_for_generation", self._wrap_cp_prepare(cp.prepare_inputs_for_generation))
        if self.time_reads:
            tensor = self.torch.Tensor
            self._install(tensor, "__bool__", self._wrap_read(tensor.__bool__))
            self._install(tensor, "item", self._wrap_read(tensor.item))
        return self

    def __exit__(self, *exc: object) -> None:
        if self.profiling:
            self._stop_profiler()
        for obj, name, previous in reversed(self._saved):
            if previous is _MISSING:
                delattr(obj, name)
            else:
                setattr(obj, name, previous)
        self._saved.clear()

    def _install(self, obj: Any, name: str, wrapper: Callable) -> None:
        self._saved.append((obj, name, obj.__dict__.get(name, _MISSING)))
        setattr(obj, name, wrapper)

    # -- helpers ---------------------------------------------------------------------------------------------------
    def _capturing(self) -> bool:
        return self.capture_step is not None and self.step == self.capture_step

    @contextlib.contextmanager
    def _timed(self, region: str, index: int = 0) -> Iterator[None]:
        annotation = self.torch.profiler.record_function(region) if self.annotate else contextlib.nullcontext()
        start = perf_counter()
        self._stack.append(region)
        try:
            with annotation:
                yield
        finally:
            self._stack.pop()
        self.calls.append(Call(region, self.step, index, start, perf_counter()))

    def _stop_profiler(self) -> None:
        with self.torch.profiler.record_function(R_MARK):
            pass
        self.profiler.stop()
        self.profiling = False

    def _profiler_window(self) -> None:
        if self.profiler is None or self.profile_steps is None:
            return
        first, stop = self.profile_steps
        if self.step == first and not self.profiling:
            self.profiler.start()
            self.profiling = True
        elif self.step == stop and self.profiling:
            self._stop_profiler()

    # -- wrappers --------------------------------------------------------------------------------------------------
    def _wrap_read(self, original: Callable) -> Callable:
        @functools.wraps(original)
        def wrapper(tensor, *args, **kwargs):
            start = perf_counter()
            value = original(tensor, *args, **kwargs)
            where = READ_REGIONS.get(self._stack[-1], "other") if self._stack else "talker_loop"
            self.reads.append(Call(where, self.step, 0, start, perf_counter()))
            return value

        return wrapper

    def _wrap_talker_forward(self, original: Callable) -> Callable:
        @functools.wraps(original)
        def wrapper(*args, **kwargs):
            embeds = kwargs.get("inputs_embeds")
            prefill = embeds is not None and embeds.shape[1] > 1
            if not prefill:
                self.step += 1
                self._profiler_window()
            with self._timed(R_PREFILL if prefill else R_STEP):
                out = original(*args, **kwargs)
            if prefill:
                self.prefilled = True
            elif self._capturing():
                self.captured["talker_logits"] = out.logits.detach().clone()
            return out

        return wrapper

    def _wrap_backbone(self, original: Callable) -> Callable:
        @functools.wraps(original)
        def wrapper(*args, **kwargs):
            if self._capturing() and "backbone_kwargs" not in self.captured:
                self.captured["backbone_kwargs"] = snapshot(self.torch, kwargs)
            with self._timed(R_BACKBONE):
                return original(*args, **kwargs)

        return wrapper

    def _wrap_cp_generate(self, original: Callable) -> Callable:
        @functools.wraps(original)
        def wrapper(*args, **kwargs):
            self.cp_index = 0
            with self._timed(R_CP_GENERATE):
                return original(*args, **kwargs)

        return wrapper

    def _wrap_cp_forward(self, original: Callable) -> Callable:
        @functools.wraps(original)
        def wrapper(*args, **kwargs):
            index = self.cp_index
            self.cp_index += 1
            capture = self._capturing() and index == self.capture_substep and "cp_kwargs" not in self.captured
            if capture:
                self.captured["cp_kwargs"] = snapshot(self.torch, kwargs)
            with self._timed(R_CP_FORWARD, index):
                out = original(*args, **kwargs)
            if capture:
                self.captured["cp_logits"] = out.logits.detach().clone()
            return out

        return wrapper

    def _wrap_processors(self, owner: str, original: Callable) -> Callable:
        @functools.wraps(original)
        def wrapper(*args, **kwargs):
            processors = original(*args, **kwargs)
            if owner == "talker" and "talker_processors" not in self.captured:
                self.captured["talker_processors"] = processors
            elif owner == "cp" and self._capturing():
                self.captured["cp_processors"] = processors
            return processors

        return wrapper

    def _wrap_talker_prepare(self, original: Callable) -> Callable:
        @functools.wraps(original)
        def wrapper(input_ids, *args, **kwargs):
            upcoming = self.step + 1
            if self.prefilled and self.capture_step is not None and upcoming == self.capture_step:
                self.captured["talker_input_ids"] = input_ids.detach().clone()
            return original(input_ids, *args, **kwargs)

        return wrapper

    def _wrap_cp_prepare(self, original: Callable) -> Callable:
        @functools.wraps(original)
        def wrapper(input_ids, *args, **kwargs):
            if self._capturing() and self.cp_index == self.capture_substep and "cp_input_ids" not in self.captured:
                self.captured["cp_input_ids"] = input_ids.detach().clone()
            return original(input_ids, *args, **kwargs)

        return wrapper


def run_light(torch: Any, model: Any, inputs: dict, seed: int, capture_step: int) -> tuple[dict, dict]:
    """Timer run: per-step split and host-read waits on the host clock, plus decode step ``capture_step``'s inputs."""
    manual_seed(torch, seed)
    torch.cuda.synchronize()
    started = perf_counter()
    with TalkerProbe(torch, model.model.talker, time_reads=True, capture_step=capture_step) as probe:
        codes, _ = model.model.generate(**inputs)
    torch.cuda.synchronize()
    ended = perf_counter()
    rows = light_steps(probe.calls, probe.reads, ended)
    prefill = [c.ms for c in probe.calls if c.region == R_PREFILL]
    quarter = max(1, len(rows) // 4)
    result = {
        "total_ms": round((ended - started) * 1000.0, 3),
        "frames": [int(c.shape[0]) for c in codes],
        "decode_forwards": len(rows),
        "prefill_ms": round(prefill[0], 3) if prefill else None,
        "summary": summarize_rows(rows, LIGHT_KEYS),
        "step_ms_first_quarter": stats(r["step_ms"] for r in rows[:quarter]),
        "step_ms_last_quarter": stats(r["step_ms"] for r in rows[-quarter:]),
        "step_ms_series": [round(r["step_ms"], 3) for r in rows],
        "capture_step": capture_step,
    }
    return result, probe.captured


def run_profiler(torch: Any, model: Any, inputs: dict, seed: int, window: tuple[int, int], out_dir: Path, tag: str):
    """CPU profile of decode steps ``window[0]`` .. ``window[1] - 1`` with the probe's regions as ranges."""
    from torch.profiler import ProfilerActivity, profile, record_function

    with profile(activities=[ProfilerActivity.CPU]), record_function("vb::prime"):
        torch.ones(1, device="cuda").add_(1)  # the first profiler session pays Kineto's start-up
    prof = profile(activities=[ProfilerActivity.CPU], record_shapes=False, with_stack=False)
    manual_seed(torch, seed)
    torch.cuda.synchronize()
    started = perf_counter()
    with TalkerProbe(torch, model.model.talker, annotate=True, profiler=prof, profile_steps=window):
        model.model.generate(**inputs)
    torch.cuda.synchronize()
    total_ms = (perf_counter() - started) * 1000.0
    trace_path = out_dir / f"trace-{tag}.json"
    prof.export_chrome_trace(str(trace_path))
    table_path = out_dir / f"ops-{tag}.txt"
    table_path.write_text(prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=60), encoding="utf-8")
    annotations, ops = events_from_profiler(prof.events())
    analysis = analyze_profile(annotations, ops)
    analysis["total_ms"] = round(total_ms, 3)
    analysis["window"] = list(window)
    analysis["trace"] = str(trace_path)
    analysis["op_table"] = str(table_path)
    return analysis


def count_unit_ops(torch: Any, fn: Callable[[], Any]) -> int:
    """Launch ops (leaf ATen ops with GPU work) of one call of ``fn``, from a CPU profile."""
    from torch.profiler import ProfilerActivity, profile, record_function

    with profile(activities=[ProfilerActivity.CPU]) as prof, record_function("vb::unit"):
        fn()
    torch.cuda.synchronize()
    _, ops = events_from_profiler(prof.events())
    return sum(1 for e in ops if is_kernel_op(e))


def sampling_unit(torch: Any, processors: Any, input_ids: Any, logits: Any) -> Callable[[], Any]:
    """HF ``_sample``'s token selection for captured logits: float copy, logits processors, softmax, multinomial."""

    def run() -> Any:
        scores = logits[:, -1, :].to(copy=True, dtype=torch.float32)
        scores = processors(input_ids, scores)
        probs = torch.nn.functional.softmax(scores, dim=-1)
        return torch.multinomial(probs, num_samples=1)

    return run


def backbone_unit(talker: Any, captured: dict) -> Callable[[], Any]:
    """The captured step's backbone forward and codec head, with the cache cropped back after each call.

    ``attention_mask`` is dropped: it is all ones, so HF's mask helper returns no mask anyway, but checking that reads
    the GPU (``padding_mask.all()``), which a graph capture does not allow.
    """
    kwargs = dict(captured["backbone_kwargs"])
    kwargs["attention_mask"] = None
    cache = kwargs["past_key_values"]
    length = cache.get_seq_length()

    def run() -> Any:
        out = talker.model(**kwargs)
        logits = talker.codec_head(out.last_hidden_state)
        cache.crop(length)
        return logits

    return run


def cp_forward_unit(code_predictor: Any, captured: dict) -> Callable[[], Any]:
    """The captured code-predictor forward (5 layers + lm head), cache cropped back, no attention mask (see above)."""
    kwargs = dict(captured["cp_kwargs"])
    kwargs["attention_mask"] = None
    cache = kwargs["past_key_values"]
    length = cache.get_seq_length()

    def run() -> Any:
        out = code_predictor(**kwargs)
        cache.crop(length)
        return out.logits

    return run


def linears_unit(torch: Any, modules: list[Any]) -> tuple[Callable[[], Any], int]:
    """Every linear layer in ``modules`` on a one-token input of its width (the GEMVs of a decode step); weight bytes."""
    weight = modules[0].weight
    inputs = {
        width: torch.randn(1, 1, width, device=weight.device, dtype=weight.dtype)
        for width in {m.in_features for m in modules}
    }

    def run() -> None:
        for module in modules:
            module(inputs[module.in_features])

    return run, sum(m.weight.numel() * m.weight.element_size() for m in modules)


def graph_time(torch: Any, fn: Callable[[], Any], calls: int) -> dict[str, Any]:
    """Capture one call of ``fn`` in a HIP graph and time ``calls`` back-to-back replays on the host clock.

    Returns ``gpu_ms`` (wall per replay: the unit's GPU time with no Python or per-op dispatch) and
    ``graph_launch_ms`` (host time of the replay call), or ``graph_error`` if the unit cannot be captured.
    """
    try:
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            for _ in range(2):
                fn()
        torch.cuda.current_stream().wait_stream(side)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            fn()
        torch.cuda.synchronize()
        for _ in range(2):
            graph.replay()
        torch.cuda.synchronize()
        started = perf_counter()
        for _ in range(calls):
            graph.replay()
        launched = perf_counter()
        torch.cuda.synchronize()
        ended = perf_counter()
    except Exception as exc:  # any capture failure is a result, not a crash
        with contextlib.suppress(Exception):
            torch.cuda.synchronize()
        return {"gpu_ms": None, "graph_launch_ms": None, "graph_error": f"{type(exc).__name__}: {exc}"[:200]}
    return {
        "gpu_ms": round((ended - started) * 1000.0 / calls, 4),
        "graph_launch_ms": round((launched - started) * 1000.0 / calls, 4),
        "graph_error": None,
    }


def measure_unit(torch: Any, fn: Callable[[], Any], calls: int, dummy: Callable[[], Any], dummy_ms: float) -> dict:
    """Eager, CPU-dispatch and GPU (graph replay) time per call of ``fn``.

    ``eager_ms``: back-to-back calls, one synchronization at the end. ``dispatch_ms``: host time to issue the calls
    while the GPU is busy with dummy matmuls (no host read inside the unit waits for them). ``gpu_ms``: graph replay.
    """
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    started = perf_counter()
    for _ in range(calls):
        fn()
    torch.cuda.synchronize()
    eager_ms = (perf_counter() - started) * 1000.0 / calls
    repeats = dummy_repeats(eager_ms * calls, dummy_ms)
    for _ in range(repeats):
        dummy()
    started = perf_counter()
    for _ in range(calls):
        fn()
    dispatch_ms = (perf_counter() - started) * 1000.0 / calls
    torch.cuda.synchronize()
    return {
        "calls": calls,
        "eager_ms": round(eager_ms, 4),
        "dispatch_ms": round(dispatch_ms, 4),
        **graph_time(torch, fn, calls),
    }


def make_dummy(torch: Any) -> tuple[Callable[[], None], float]:
    """A long BF16 matmul that keeps the GPU busy, and its wall time in ms."""
    a = torch.randn(DUMMY_SIZE, DUMMY_SIZE, device="cuda", dtype=torch.bfloat16)
    out = torch.empty_like(a)

    def dummy() -> None:
        torch.mm(a, a, out=out)

    for _ in range(3):
        dummy()
    torch.cuda.synchronize()
    started = perf_counter()
    for _ in range(5):
        dummy()
    torch.cuda.synchronize()
    return dummy, (perf_counter() - started) * 1000.0 / 5


def gemv_units(torch: Any, talker: Any) -> dict[str, tuple[Callable[[], None], int]]:
    """The linear layers of one decode step: the backbone's (plus codec head) and one code-predictor forward's."""
    linear = torch.nn.Linear
    cp = talker.code_predictor
    cp_modules = [m for m in cp.model.layers.modules() if isinstance(m, linear)]
    if isinstance(cp.small_to_mtp_projection, linear):
        cp_modules.append(cp.small_to_mtp_projection)
    return {
        "backbone_linears": linears_unit(
            torch, [m for m in talker.model.layers.modules() if isinstance(m, linear)] + [talker.codec_head]
        ),
        "cp_linears": linears_unit(torch, [*cp_modules, cp.lm_head[0]]),
    }


TINY_OPS = 200  # tiny kernels per call of the tiny unit: one graph launch per call would hide their cost


def tiny_unit(torch: Any) -> Callable[[], None]:
    """``TINY_OPS`` in-place adds on a 1024-element tensor: the cost of one small kernel, times ``TINY_OPS``."""
    tiny = torch.zeros(1024, device="cuda")

    def run() -> None:
        for _ in range(TINY_OPS):
            tiny.add_(1.0)

    return run


def run_replays(torch: Any, model: Any, captured: dict, calls: int) -> dict:
    """GPU (graph) and dispatch times of the captured step's units, small kernels and a host-read round trip."""
    talker = model.model.talker
    cp = talker.code_predictor
    dummy, dummy_ms = make_dummy(torch)
    gemv = gemv_units(torch, talker)
    # The talker's sampling chain goes last: its suppress-tokens mask reads the GPU, so its capture fails, and a failed
    # capture should not be able to disturb the other units.
    units: dict[str, Callable[[], Any]] = {
        "tiny_kernels": tiny_unit(torch),
        "backbone+codec_head": backbone_unit(talker, captured),
        "backbone_linears": gemv["backbone_linears"][0],
        "cp_forward": cp_forward_unit(cp, captured),
        "cp_linears": gemv["cp_linears"][0],
        "cp_sampling": sampling_unit(torch, captured["cp_processors"], captured["cp_input_ids"], captured["cp_logits"]),
        "talker_sampling": sampling_unit(
            torch, captured["talker_processors"], captured["talker_input_ids"], captured["talker_logits"]
        ),
    }
    unit_calls: dict[str, int] = {}
    results: dict[str, dict] = {}
    with torch.no_grad():  # as generation runs them (Qwen3TTSForConditionalGeneration.generate is no_grad)
        scalar = torch.ones(16, device="cuda")
        for _ in range(10):
            scalar.sum().item()
        reads = calls * 20
        started = perf_counter()
        for _ in range(reads):
            scalar.sum().item()
        results["host_read_roundtrip"] = {
            "calls": reads,
            "eager_ms": round((perf_counter() - started) * 1000.0 / reads, 4),
        }
        for name, fn in units.items():
            results[name] = measure_unit(torch, fn, unit_calls.get(name, calls), dummy, dummy_ms)
            results[name]["ops"] = count_unit_ops(torch, fn)
            logger.info("replay %s: %s", name, results[name])
    return {
        "dummy_ms": round(dummy_ms, 3),
        "processors": {
            "talker": [type(p).__name__ for p in captured["talker_processors"]],
            "cp": [type(p).__name__ for p in captured["cp_processors"]],
        },
        "weight_bytes": {name: unit[1] for name, unit in gemv.items()},
        "tiny_ops_per_call": TINY_OPS,
        "units": results,
    }


def build_estimate(light: dict, prof: dict, replay: dict) -> dict:
    """Whole-step figures: measured step, CPU busy vs waiting, GPU-busy estimate, GEMV share, sync-free CPU floor."""
    units = replay["units"]
    regions = prof["summary"]["regions"]
    step_prof = prof["summary"]["step"]
    cp_forwards = prof["summary"]["cp_forward_n"]["mean"] or 0.0
    gpu = estimate_gpu_step(
        backbone_gpu_ms=unit_value(units["backbone+codec_head"]),
        cp_forward_gpu_ms=unit_value(units["cp_forward"]),
        cp_sampling_gpu_ms=unit_value(units["cp_sampling"]),
        talker_sampling_gpu_ms=unit_value(units["talker_sampling"]),
        tiny_gpu_ms=unit_value(units["tiny_kernels"]) / TINY_OPS,
        cp_forwards=cp_forwards,
        ops_step=step_prof["ops_n"]["mean"] or 0.0,
        ops_backbone=regions["backbone"]["ops_n"]["mean"] or 0.0,
        ops_cp_forward=regions["cp_forward"]["ops_n"]["mean"] or 0.0,
        ops_cp_sampling=units["cp_sampling"]["ops"],
        ops_talker_sampling=units["talker_sampling"]["ops"],
    )
    dispatch = estimate_dispatch_step(
        backbone_dispatch_ms=units["backbone+codec_head"]["dispatch_ms"],
        cp_forward_dispatch_ms=units["cp_forward"]["dispatch_ms"],
        cp_sampling_dispatch_ms=min(units["cp_sampling"]["dispatch_ms"], units["cp_sampling"]["eager_ms"]),
        talker_sampling_dispatch_ms=min(units["talker_sampling"]["dispatch_ms"], units["talker_sampling"]["eager_ms"]),
        tiny_dispatch_ms=units["tiny_kernels"]["dispatch_ms"] / TINY_OPS,
        cp_forwards=cp_forwards,
        other_ops=gpu["other_ops"],
    )
    weights = replay["weight_bytes"]
    gemv = {
        "backbone": gemv_figures(units["backbone_linears"]["gpu_ms"], weights["backbone_linears"], 1.0),
        "code_predictor": gemv_figures(units["cp_linears"]["gpu_ms"], weights["cp_linears"], cp_forwards),
    }
    gemv_ms = sum(v["ms_per_step"] or 0.0 for v in gemv.values())
    light_summary = light["summary"]
    step_ms = light_summary["step_ms"]["median"]
    return {
        "step_ms_measured": step_ms,
        "cpu_busy_ms": light_summary["cpu_busy_ms"]["median"],
        "host_read_wait_ms": light_summary["host_read_ms"]["median"],
        "host_reads_python_per_step": light_summary["host_read_n"]["median"],
        "host_reads_all_per_step": step_prof["sync_n"]["mean"],
        "host_read_roundtrip_ms": units["host_read_roundtrip"]["eager_ms"],
        "gpu_busy_ms": gpu,
        "gpu_idle_share": round(1.0 - gpu["total_ms"] / step_ms, 3) if step_ms else None,
        "gemv": {**gemv, "ms_per_step": round(gemv_ms, 3)},
        "dispatch_floor_ms": dispatch,
        "launch_ops_per_step": step_prof["ops_n"]["mean"],
        "profiled_step_ms": step_prof["ms"]["mean"],
        "profiler_overhead_factor": round(step_prof["ms"]["mean"] / step_ms, 3) if step_ms else None,
    }


CHILD_RESULT_PREFIX = "VBPROFILE-RESULT "
# Settings read by torch's ROCm build when it starts; each is profiled in a fresh process against a fresh baseline.
_AOTRITON = {"TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL": "1"}
_TUNABLEOP = {"PYTORCH_TUNABLEOP_ENABLED": "1"}
ENV_VARIANTS: dict[str, dict[str, str]] = {
    "base": {},
    # SDPA on gfx1101 falls back to the unfused math kernels unless AOTriton's experimental kernels are enabled.
    "aotriton_experimental": dict(_AOTRITON),
    # Task 3.4: GEMMs tuned on first use (the warm-up runs), results in <out>/tunableop; then the same file read-only,
    # as the plan would ship it, alone and with AOTriton.
    "tunableop_tuning": {**_TUNABLEOP, "PYTORCH_TUNABLEOP_TUNING": "1"},
    "tunableop_readonly": {**_TUNABLEOP, "PYTORCH_TUNABLEOP_TUNING": "0"},
    "aotriton+tunableop_readonly": {**_AOTRITON, **_TUNABLEOP, "PYTORCH_TUNABLEOP_TUNING": "0"},
}


def variant_env(name: str, out_dir: Path, phrase_id: str) -> dict[str, str]:
    """Extra environment of env variant ``name`` (TunableOp's results file under ``out_dir``)."""
    env = dict(ENV_VARIANTS[name])
    if env.get("PYTORCH_TUNABLEOP_ENABLED") == "1":
        tuning_dir = out_dir / "tunableop"
        tuning_dir.mkdir(parents=True, exist_ok=True)
        env["PYTORCH_TUNABLEOP_FILENAME"] = str(tuning_dir / f"{phrase_id}_results%d.csv")
    return env


def child_command(args: argparse.Namespace, variant: str) -> list[str]:
    """Command line of the child process that profiles env variant ``variant``."""
    command = [sys.executable, str(Path(__file__).resolve()), "--child", variant, "--phrase", args.phrase]
    command += ["--phrases", str(args.phrases), "--data-dir", str(args.data_dir), "--out", str(args.out)]
    command += ["--model-size", args.model_size, "--rounds", str(args.rounds), "--replay-calls", str(args.replay_calls)]
    if args.prompt_file is not None:
        command += ["--prompt-file", str(args.prompt_file)]
    return command


def parse_child_result(stdout: str) -> dict[str, Any] | None:
    """The JSON a child process printed after ``CHILD_RESULT_PREFIX``; None if it printed none."""
    lines = [line for line in stdout.splitlines() if line.startswith(CHILD_RESULT_PREFIX)]
    return json.loads(lines[-1][len(CHILD_RESULT_PREFIX) :]) if lines else None


def run_env_variants(args: argparse.Namespace, out_dir: Path, phrase_id: str) -> dict[str, Any]:
    """Profile every env variant in its own process, one after another (never two models on the GPU)."""
    import os
    import subprocess

    results: dict[str, Any] = {}
    for name in ENV_VARIANTS:
        extra = variant_env(name, out_dir, phrase_id)
        logger.info("env variant %s %s ...", name, extra)
        completed = subprocess.run(
            child_command(args, name),
            env={**os.environ, **extra},
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
        )
        log_path = out_dir / f"variant-{phrase_id}-{name}.log"
        log_path.write_text(completed.stdout + "\n" + completed.stderr, encoding="utf-8")
        result = parse_child_result(completed.stdout)
        if result is None:
            logger.warning("env variant %s failed (exit %s), see %s", name, completed.returncode, log_path)
            results[name] = {"env": extra, "error": f"exit {completed.returncode}", "log": str(log_path)}
            continue
        results[name] = {"env": extra, **result}
        logger.info("env variant %s: %s", name, json.dumps(result))
    return results


def prepare(args: argparse.Namespace) -> tuple[Any, dict[str, Any], Path, dict[str, list], Any, float]:
    """torch, phrase, prompt file, cached prompt, loaded model and its load time in seconds."""
    import torch

    if not torch.cuda.is_available():
        raise ProfileError("no CUDA/ROCm GPU visible to torch")
    phrase = load_phrase(args.phrases, args.phrase)
    prompt_file = args.prompt_file or find_prompt_file(args.data_dir, args.model_size)
    cached = validate_cached_prompt(torch.load(prompt_file, map_location="cpu", weights_only=True))
    logger.info("loading Qwen3-TTS %s ...", args.model_size)
    started = perf_counter()
    model = load_model(torch, args.model_size)
    return torch, phrase, prompt_file, cached, model, perf_counter() - started


def run_child(args: argparse.Namespace) -> int:
    """One env variant: timing rounds, the timer run, a short profiler window and the GEMV units; prints JSON."""
    torch, phrase, _, cached, model, load_s = prepare(args)
    out_dir: Path = args.out
    inputs = build_talker_inputs(model, phrase, cached, "cuda")
    for _ in range(2):  # warm-up; with TunableOp, tuning happens here
        run_generate(torch, model, inputs, phrase["seed"])
    runs = [run_generate(torch, model, inputs, phrase["seed"]) for _ in range(args.rounds)]
    steps = runs[0][1][0]
    light, _ = run_light(torch, model, inputs, phrase["seed"], capture_step=steps // 2)
    first = max(1, steps // 3)
    prof = run_profiler(
        torch, model, inputs, phrase["seed"], (first, min(steps, first + 5)), out_dir, f"{phrase['id']}-{args.child}"
    )
    dummy, dummy_ms = make_dummy(torch)
    gemv: dict[str, Any] = {}
    cp_forwards = prof["summary"]["cp_forward_n"]["mean"] or 0.0
    with torch.no_grad():
        for name, (fn, weight_bytes) in gemv_units(torch, model.model.talker).items():
            unit = measure_unit(torch, fn, args.replay_calls, dummy, dummy_ms)
            per_step = cp_forwards if name == "cp_linears" else 1.0
            gemv[name] = {**unit, **gemv_figures(unit["gpu_ms"], weight_bytes, per_step)}
    ms = stats(ms for ms, _ in runs)
    summary = light["summary"]
    result = {
        "variant": args.child,
        "load_s": round(load_s, 2),
        "frames": runs[0][1],
        "frames_all_runs": [frames for _, frames in runs],
        "ms": ms,
        "ms_per_step": round(ms["median"] / steps, 3) if steps else None,
        "light": {
            key: summary[key]["median"]
            for key in (
                "step_ms",
                "cpu_busy_ms",
                "host_read_ms",
                "host_read_n",
                "backbone_ms",
                "cp_forward_ms",
                "cp_generate_rest_ms",
                "talker_loop_ms",
            )
        },
        "launch_ops_per_step": prof["summary"]["step"]["ops_n"]["mean"],
        "launch_ops_backbone": prof["summary"]["regions"]["backbone"]["ops_n"]["mean"],
        "launch_ops_cp_forward": prof["summary"]["regions"]["cp_forward"]["ops_n"]["mean"],
        "gemv": gemv,
    }
    sys.stdout.write(CHILD_RESULT_PREFIX + json.dumps(result) + "\n")
    sys.stdout.flush()
    return 0


def run(args: argparse.Namespace) -> int:
    if args.child:
        return run_child(args)
    out_dir: Path = args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = args.phrase
    env_variants = run_env_variants(args, out_dir, tag) if args.env_variants else {}

    torch, phrase, prompt_file, cached, model, load_s = prepare(args)
    env = model_info(torch, model)
    logger.info("model loaded in %.1f s: %s", load_s, env)

    inputs = build_talker_inputs(model, phrase, cached, "cuda")
    batch_inputs = build_talker_inputs(model, phrase, cached, "cuda", batch=2) if args.batch else None

    warm_ms, frames = run_generate(torch, model, inputs, phrase["seed"])
    logger.info("warm-up: %.1f ms, frames %s", warm_ms, frames)
    if batch_inputs is not None:
        warm_batch_ms, batch_frames = run_generate(torch, model, batch_inputs, phrase["seed"])
        logger.info("warm-up batch2: %.1f ms, frames %s", warm_batch_ms, batch_frames)
    steps = frames[0]

    timing = run_timing(torch, model, inputs, batch_inputs, phrase["seed"], args.rounds)
    light, captured = run_light(torch, model, inputs, phrase["seed"], capture_step=steps // 2)
    logger.info("light: %s", json.dumps(light["summary"]))
    first = max(1, steps // 3)
    window = (first, min(steps, first + args.profile_steps))
    prof = run_profiler(torch, model, inputs, phrase["seed"], window, out_dir, tag)
    logger.info("profiler: %s", json.dumps(prof["summary"], ensure_ascii=False))
    replay = run_replays(torch, model, captured, args.replay_calls)
    estimate = build_estimate(light, prof, replay)
    logger.info("estimate: %s", json.dumps(estimate, ensure_ascii=False))

    result = {
        "schema": "vb-talker-profile/1",
        "phrase": phrase,
        "prompt_file": str(prompt_file),
        "load_s": round(load_s, 2),
        "env": env,
        "warmup": {"ms": round(warm_ms, 3), "frames": frames},
        "timing": timing,
        "light": light,
        "profiler": {"summary": prof["summary"], **{k: prof[k] for k in ("total_ms", "window", "trace", "op_table")}},
        "profiler_rows": prof["rows"],
        "replay": replay,
        "estimate": estimate,
        "env_variants": env_variants,
    }
    json_path = out_dir / f"profile-{tag}.json"
    json_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    md_path = out_dir / f"profile-{tag}.md"
    md_path.write_text(render_markdown(result), encoding="utf-8")
    logger.info("wrote %s and %s", json_path, md_path)
    return 0


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0], allow_abbrev=False)
    parser.add_argument("--phrase", default="p08", help="phrase id in --phrases (default p08: ~6 s, 78 steps)")
    parser.add_argument("--phrases", type=Path, default=DEFAULT_PHRASES)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR, help="bench data dir with the prompt cache")
    parser.add_argument("--prompt-file", type=Path, default=None, help="cached voice prompt (.prompt) to use instead")
    parser.add_argument("--model-size", choices=sorted(HF_REPOS), default="1.7B")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--rounds", type=int, default=3, help="timing rounds (each runs every variant once)")
    parser.add_argument("--profile-steps", type=int, default=20, help="decode steps inside the profiler window")
    parser.add_argument("--replay-calls", type=int, default=20, help="calls per replayed unit")
    parser.add_argument("--no-batch", dest="batch", action="store_false", help="skip the batch-of-two timing")
    parser.add_argument(
        "--no-env-variants",
        dest="env_variants",
        action="store_false",
        help="skip the env variants (base, AOTriton experimental, TunableOp), each run in its own process",
    )
    parser.add_argument("--child", choices=sorted(ENV_VARIANTS), default=None, help=argparse.SUPPRESS)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args(argv)
    try:
        return run(args)
    except ProfileError as exc:
        logger.error("%s", exc)
        return 2


if __name__ == "__main__":
    sys.exit(main())
