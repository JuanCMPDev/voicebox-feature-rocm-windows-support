"""Probe HIP graph support in this torch ROCm build for Windows before task 3.5b relies on it (plan, Fase 3.5).

One question: can ``torch.cuda.CUDAGraph`` capture work once and replay it with new inputs, with the right result?
Four checks, each with a pass flag and its numbers:

1. ``matmul``: ``a @ b`` captured, new values copied into ``a`` and ``b`` before each replay, compared with eager.
2. ``rng``: ``torch.multinomial`` captured; replays draw new samples, and from the same seed as eager draws they give
   (or not) the same ones.
3. ``stacks``: decode steps of the talker's first decoder layer, then of its whole backbone (28 layers, final norm and
   codec head), on a static KV cache. One capture per stack; each step copies a new hidden state and cache position
   into the graph's inputs and replays. Rotary positions and the attention mask (``kv_index <= cache_position``) are
   computed inside the graph from the cache position. Every step's output is compared bit for bit with the same eager
   computation on its own static cache, and with the production path (DynamicCache, no mask, as the talker decodes
   today) within bf16 noise. Host-clock time per step for the three paths.
4. ``code_predictor``: one frame of the code predictor (15 forwards and samplings of task 3.2's loop) as a single
   graph on a 16-slot static cache, replayed with new input embeddings; codes from each frame's seed against the same
   eager computation and against the production loop (``qwen_code_predictor.sample_with_loop``), the RNG state after
   a chain of frames from one seed, and time per frame.

Nothing here touches the app: the model is loaded as ``PyTorchTTSBackend`` loads it (BF16, ``device_map="cuda"``) and
only read, under ``torch.inference_mode`` as ``_generate_sync`` runs it. Writes ``hipgraph-probe.json`` to ``--out``
(``perf-loop/bench/_dev`` by default, not tracked by git).

Usage:
    backend/venv/Scripts/python.exe scripts/probe_hip_graph.py
"""

import argparse
import json
import logging
import statistics
import sys
from collections.abc import Callable
from pathlib import Path
from time import perf_counter
from typing import Any

logger = logging.getLogger("probe_hip_graph")

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "perf-loop" / "bench" / "_dev"
HF_REPO = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"  # the plan keeps the 1.7B model
WARMUP_CALLS = 3  # eager calls on a side stream before a capture, as the torch docs ask
REL_TOLERANCE = 0.05  # static vs dynamic cache, relative L2 error per step: bf16 noise, not a wrong result


def capture(torch: Any, fn: Callable[[], Any]) -> tuple[Any, Any]:
    """``fn`` warmed up on a side stream, then captured in a HIP graph; the graph and ``fn``'s (static) output."""
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for _ in range(WARMUP_CALLS):
            fn()
    torch.cuda.current_stream().wait_stream(side)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = fn()
    torch.cuda.synchronize()
    return graph, out


def diff(torch: Any, actual: Any, expected: Any) -> dict[str, Any]:
    """Bit-for-bit equality, max absolute difference and relative L2 error of ``actual`` against ``expected``."""
    a, b = actual.float(), expected.float()
    return {
        "equal": bool(torch.equal(actual, expected)),
        "max_abs": float((a - b).abs().max()),
        "rel": float((a - b).norm() / b.norm().clamp_min(1e-30)),
    }


def summarize(diffs: list[dict[str, Any]]) -> dict[str, Any]:
    """Per-step diffs folded into: every step equal, worst max-abs and worst relative error."""
    return {
        "all_equal": all(d["equal"] for d in diffs),
        "steps_equal": sum(d["equal"] for d in diffs),
        "max_abs": max((d["max_abs"] for d in diffs), default=0.0),
        "max_rel": max((d["rel"] for d in diffs), default=0.0),
    }


# --------------------------------------------------------------------------------------------------------------------
# Decode steps of a stack of talker decoder layers
# --------------------------------------------------------------------------------------------------------------------


def run_layers(layers: list[Any], hidden: Any, **kwargs: Any) -> Any:
    """``hidden`` through ``layers`` as ``Qwen3TTSTalkerModel.forward`` calls them (one shared set of kwargs)."""
    for layer in layers:
        hidden = layer(hidden, use_cache=True, **kwargs)[0]
    return hidden


def prefill(torch: Any, rotary: Any, layers: list[Any], embeds: Any) -> list[tuple[Any, Any]]:
    """Production prefill of ``embeds`` (no mask: SDPA's causal path) into a DynamicCache; its keys and values."""
    from transformers import DynamicCache

    cache = DynamicCache()
    cache_position = torch.arange(embeds.shape[1], device=embeds.device)
    position_ids = cache_position.view(1, 1, -1).expand(3, embeds.shape[0], -1)
    run_layers(
        layers,
        embeds,
        attention_mask=None,
        position_embeddings=rotary(embeds, position_ids),
        cache_position=cache_position,
        past_key_values=cache,
    )
    return [(layer.keys.clone(), layer.values.clone()) for layer in cache.layers[: len(layers)]]


def dynamic_cache(prefix: list[tuple[Any, Any]]) -> Any:
    """A DynamicCache holding copies of ``prefix``, as the talker's cache is after prefill."""
    from transformers import DynamicCache

    return DynamicCache(ddp_cache_data=[(k.clone(), v.clone()) for k, v in prefix])


def load_static(cache: Any, prefix: list[tuple[Any, Any]]) -> None:
    """Reset a StaticCache in place: ``prefix`` in its first slots, zeros after (tensor addresses are kept)."""
    for layer, (keys, values) in zip(cache.layers, prefix, strict=False):
        if not layer.is_initialized:
            layer.lazy_initialization(keys)
        length = keys.shape[2]
        layer.keys.zero_()
        layer.values.zero_()
        layer.keys[:, :, :length].copy_(keys)
        layer.values[:, :, :length].copy_(values)


def attention_mask(torch: Any, allowed: Any, config: Any, dtype: Any) -> Any:
    """``allowed`` (True where a query may attend) as the layers' attention takes a 4D mask, like HF's mask helpers.

    Boolean for SDPA; additive for eager attention, which adds the mask to the scores: 0 where allowed and the
    dtype's minimum elsewhere.
    """
    if config._attn_implementation != "eager":
        return allowed
    return torch.zeros(allowed.shape, dtype=dtype, device=allowed.device).masked_fill(~allowed, torch.finfo(dtype).min)


def static_step(
    torch: Any,
    rotary: Any,
    layers: list[Any],
    head: Callable[[Any], Any] | None,
    cache: Any,
    x: Any,
    cache_position: Any,
    kv_index: Any,
) -> Callable[[], Any]:
    """One decode step on a static cache whose inputs are all device tensors (``x``, ``cache_position``): capturable.

    The mask lets the query see slots ``0..cache_position`` (the prefix, the previous steps and itself); positions are
    the cache position on the three rotary axes, as the talker's decode step computes them with no padding.
    """
    config = layers[0].self_attn.config

    def run() -> Any:
        mask = attention_mask(torch, (kv_index <= cache_position).view(1, 1, 1, -1), config, x.dtype)
        position_ids = cache_position.view(1, 1, -1).expand(3, x.shape[0], -1)
        hidden = run_layers(
            layers,
            x,
            attention_mask=mask,
            position_embeddings=rotary(x, position_ids),
            cache_position=cache_position,
            past_key_values=cache,
        )
        return head(hidden) if head is not None else hidden

    return run


def dynamic_step(
    rotary: Any,
    layers: list[Any],
    head: Callable[[Any], Any] | None,
    cache: Any,
    x: Any,
    cache_position: Any,
) -> Any:
    """The production decode step: DynamicCache and no mask (HF's mask helper returns None for one query)."""
    position_ids = cache_position.view(1, 1, -1).expand(3, x.shape[0], -1)
    hidden = run_layers(
        layers,
        x,
        attention_mask=None,
        position_embeddings=rotary(x, position_ids),
        cache_position=cache_position,
        past_key_values=cache,
    )
    return head(hidden) if head is not None else hidden


def run_sequence(torch: Any, reset: Callable[[], None], step: Callable[[int], Any], steps: int) -> tuple[list, float]:
    """``reset()``, then ``steps`` decode steps; their outputs (copied) and the wall time per step in ms."""
    reset()
    torch.cuda.synchronize()
    started = perf_counter()
    outputs = [step(n).clone() for n in range(steps)]
    torch.cuda.synchronize()
    return outputs, (perf_counter() - started) * 1000.0 / steps


def check_stack(
    torch: Any,
    talker: Any,
    layers: list[Any],
    head: Callable[[Any], Any] | None,
    prefix: list[tuple[Any, Any]],
    inputs: list[Any],
    cache_len: int,
    rounds: int,
) -> dict[str, Any]:
    """Graph replays vs eager on a static cache vs the production DynamicCache path, for decode steps of ``layers``."""
    from transformers import StaticCache

    rotary = talker.model.rotary_emb
    device = inputs[0].device
    start = prefix[0][0].shape[2]
    steps = len(inputs)
    kv_index = torch.arange(cache_len, device=device)
    positions = [torch.tensor([start + n], device=device) for n in range(steps)]

    def static_path() -> tuple[Any, Any, Any, Callable[[], Any]]:
        cache = StaticCache(config=talker.config, max_cache_len=cache_len)
        load_static(cache, prefix)
        x = torch.zeros_like(inputs[0])
        cache_position = positions[0].clone()
        return (
            cache,
            x,
            cache_position,
            static_step(torch, rotary, layers, head, cache, x, cache_position, kv_index),
        )

    graph_cache, graph_x, graph_pos, graph_fn = static_path()
    eager_cache, eager_x, eager_pos, eager_fn = static_path()
    result: dict[str, Any] = {
        "layers": len(layers),
        "head": head is not None,
        "prefix": start,
        "steps": steps,
        "cache_len": cache_len,
        "cache_layer_types": sorted({type(layer).__name__ for layer in graph_cache.layers}),
    }
    try:
        graph, graph_out = capture(torch, graph_fn)
    except Exception as exc:  # a failed capture is the probe's answer, not a crash
        result.update(captured=False, error=f"{type(exc).__name__}: {exc}"[:500])
        return result
    result["captured"] = True

    def graph_step(n: int) -> Any:
        graph_x.copy_(inputs[n])
        graph_pos.copy_(positions[n])
        graph.replay()
        return graph_out

    def eager_step(n: int) -> Any:
        eager_x.copy_(inputs[n])
        eager_pos.copy_(positions[n])
        return eager_fn()

    holder: dict[str, Any] = {}

    def dynamic_reset() -> None:
        holder["cache"] = dynamic_cache(prefix)

    def dyn_step(n: int) -> Any:
        return dynamic_step(rotary, layers, head, holder["cache"], inputs[n], positions[n])

    paths = {
        "graph": (lambda: load_static(graph_cache, prefix), graph_step),
        "eager_static": (lambda: load_static(eager_cache, prefix), eager_step),
        "eager_dynamic": (dynamic_reset, dyn_step),
    }
    times: dict[str, list[float]] = {name: [] for name in paths}
    outputs: dict[str, list[Any]] = {}
    first_graph: list[Any] | None = None
    for _ in range(rounds):
        for name, (reset, step) in paths.items():
            outputs[name], ms = run_sequence(torch, reset, step, steps)
            times[name].append(ms)
        if first_graph is None:
            first_graph = outputs["graph"]
    graph_vs_static = [diff(torch, g, e) for g, e in zip(outputs["graph"], outputs["eager_static"], strict=True)]
    graph_vs_dynamic = [diff(torch, g, d) for g, d in zip(outputs["graph"], outputs["eager_dynamic"], strict=True)]
    result.update(
        graph_vs_eager_static=summarize(graph_vs_static),
        graph_vs_eager_dynamic=summarize(graph_vs_dynamic),
        graph_rounds_identical=all(torch.equal(a, b) for a, b in zip(first_graph or [], outputs["graph"], strict=True)),
        # A replay that ignored its new inputs would repeat one output: count distinct ones.
        graph_distinct_outputs=len({tuple(o.float().flatten()[:64].tolist()) for o in outputs["graph"]}),
        cache_matches_eager=all(
            torch.equal(g.keys, e.keys) and torch.equal(g.values, e.values)
            for g, e in zip(
                graph_cache.layers[: len(layers)],
                eager_cache.layers[: len(layers)],
                strict=True,
            )
        ),
        ms_per_step={name: round(statistics.median(values), 3) for name, values in times.items()},
    )
    if head is not None:
        result["argmax_agree_graph_vs_dynamic"] = sum(
            int(torch.equal(g.argmax(-1), d.argmax(-1)))
            for g, d in zip(outputs["graph"], outputs["eager_dynamic"], strict=True)
        )
    result["ok"] = bool(
        result["captured"]
        and result["graph_vs_eager_static"]["max_rel"] <= 1e-3
        and result["graph_vs_eager_dynamic"]["max_rel"] <= REL_TOLERANCE
        and result["graph_rounds_identical"]
        and result["graph_distinct_outputs"] == steps
        and result["cache_matches_eager"]
    )
    return result


# --------------------------------------------------------------------------------------------------------------------
# Checks
# --------------------------------------------------------------------------------------------------------------------


def check_matmul(torch: Any, replays: int = 3) -> dict[str, Any]:
    """``a @ b`` in a graph, replayed with new ``a`` and ``b``: each replay against eager, and replays all differ."""
    generator = torch.Generator(device="cuda").manual_seed(0)

    def rand(*shape: int) -> Any:
        return torch.randn(*shape, device="cuda", dtype=torch.bfloat16, generator=generator)

    a, b = rand(16, 2048), rand(2048, 6144)
    try:
        graph, out = capture(torch, lambda: a @ b)
    except Exception as exc:
        return {
            "ok": False,
            "captured": False,
            "error": f"{type(exc).__name__}: {exc}"[:500],
        }
    diffs, seen = [], []
    for _ in range(replays):
        a.copy_(rand(16, 2048))
        b.copy_(rand(2048, 6144))
        graph.replay()
        diffs.append(diff(torch, out, a @ b))
        seen.append(out.clone())
    distinct = all(not torch.equal(seen[i], seen[i + 1]) for i in range(len(seen) - 1))
    summary = summarize(diffs)
    return {
        "ok": summary["max_rel"] <= 1e-3 and distinct,
        "captured": True,
        "replays_differ": distinct,
        **summary,
    }


def check_rng(torch: Any, draws: int = 32, seed: int = 1234) -> dict[str, Any]:
    """``torch.multinomial`` in a graph: replays draw new samples; from one seed, graph and eager draws compared."""
    generator = torch.Generator(device="cuda").manual_seed(1)
    probs = torch.softmax(torch.randn(1, 2048, device="cuda", generator=generator) * 3.0, dim=-1)

    def sample() -> Any:
        return torch.multinomial(probs, num_samples=1)

    try:
        graph, out = capture(torch, sample)
    except Exception as exc:
        return {
            "ok": False,
            "captured": False,
            "error": f"{type(exc).__name__}: {exc}"[:500],
        }
    torch.cuda.manual_seed(seed)
    replayed = []
    for _ in range(draws):
        graph.replay()
        replayed.append(int(out.item()))
    torch.cuda.manual_seed(seed)
    eager = [int(sample().item()) for _ in range(draws)]
    distinct = len(set(replayed))
    return {
        "ok": distinct > 1,
        "captured": True,
        "draws": draws,
        "distinct_replayed": distinct,
        "same_as_eager_from_seed": replayed == eager,
        "replayed_head": replayed[:8],
        "eager_head": eager[:8],
    }


def frame_masks(positions: Any, prompt_length: int) -> list[Any]:
    """Boolean 4D masks of one code-predictor frame on a static cache of ``len(positions)`` slots.

    The prefill's (``prompt_length`` queries, causal) and then one per decode position: each query sees the slots up to
    its own position. Passed as 4D tensors, HF's mask helper uses them as they are (no host read, no ``None`` shortcut).
    """
    cache_len = positions.shape[0]
    masks = [(positions[None, :] <= positions[:prompt_length, None]).view(1, 1, prompt_length, cache_len)]
    masks += [(positions <= p).view(1, 1, 1, cache_len) for p in range(prompt_length, cache_len)]
    return masks


def code_predictor_frame(
    torch: Any, predictor: Any, plan: Any, cache: Any, embeds: Any, positions: Any, masks: list[Any]
) -> Callable[[], Any]:
    """The code predictor's tokens of one frame on a static cache: ``sample_with_loop`` (task 3.2) unrolled.

    Same forwards (prefill on ``embeds``, then one per sampled token with the step's ``generation_steps``), same
    sampling (float32 logits, the plan's HF warpers, softmax, ``torch.multinomial``), but every slot of ``cache`` is
    preallocated and each forward gets a 4D mask (``masks[i]``) hiding the slots after its position. Every input is a
    device tensor, so the whole frame is capturable as one graph. Returns the (batch, tokens) int64 codes.
    """
    prompt_length = embeds.shape[1]
    batch = embeds.shape[0]
    masks = [attention_mask(torch, mask, predictor.config, predictor.dtype) for mask in masks]

    def run() -> Any:
        tokens: list[Any] = []
        step_inputs: dict[str, Any] = {"inputs_embeds": embeds}
        cache_position = positions[:prompt_length]
        for index in range(plan.max_new_tokens):
            outputs = predictor(
                **step_inputs,
                attention_mask=masks[index],
                past_key_values=cache,
                use_cache=True,
                cache_position=cache_position,
                position_ids=cache_position.unsqueeze(0).expand(batch, -1),
                return_dict=True,
            )
            scores = outputs.logits[:, -1, :].to(copy=True, dtype=torch.float32)
            for warper in plan.warpers:
                scores = warper(None, scores)
            if plan.do_sample:
                next_tokens = torch.multinomial(torch.nn.functional.softmax(scores, dim=-1), num_samples=1).squeeze(1)
            else:
                next_tokens = torch.argmax(scores, dim=-1)
            tokens.append(next_tokens)
            if len(tokens) == plan.max_new_tokens:
                break
            position = prompt_length + index
            cache_position = positions[position : position + 1]
            step_inputs = {"input_ids": next_tokens[:, None], "generation_steps": outputs.generation_steps}
        return torch.stack(tokens, dim=1)

    return run


def check_code_predictor(torch: Any, model: Any, frames: int, rounds: int, seed: int) -> dict[str, Any]:
    """One frame of the code predictor (15 forwards and samplings) as a single graph, replayed with new embeddings.

    Codes from each frame's seed are compared with the same eager computation on a static cache (bit for bit) and with
    the production loop of task 3.2 (``qwen_code_predictor.sample_with_loop``, DynamicCache); time per frame for all.
    """
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from transformers import StaticCache

    from backend.utils import qwen_code_predictor

    talker = model.model.talker
    predictor = talker.code_predictor
    settings = model._merge_generate_kwargs()
    plan = qwen_code_predictor._sampling_plan(
        predictor,
        {
            "max_new_tokens": talker.config.num_code_groups - 1,
            "do_sample": settings["subtalker_dosample"],
            "top_p": settings["subtalker_top_p"],
            "top_k": settings["subtalker_top_k"],
            "temperature": settings["subtalker_temperature"],
            "output_hidden_states": True,
            "return_dict_in_generate": True,
        },
    )
    generator = torch.Generator(device="cuda").manual_seed(seed)
    hidden = talker.config.hidden_size
    dtype = predictor.dtype
    inputs = [torch.randn(1, 2, hidden, device="cuda", dtype=dtype, generator=generator) for _ in range(frames)]
    prompt_length = inputs[0].shape[1]
    cache_len = prompt_length + plan.max_new_tokens - 1
    positions = torch.arange(cache_len, device="cuda")
    masks = frame_masks(positions, prompt_length)

    def static_path() -> tuple[Any, Callable[[], Any]]:
        """Input buffer and frame function on a static cache of their own (no reset needed: each frame rewrites every
        slot in order, and the masks hide the slots not yet rewritten)."""
        cache = StaticCache(config=predictor.config, max_cache_len=cache_len)
        embeds = torch.zeros_like(inputs[0])
        return embeds, code_predictor_frame(torch, predictor, plan, cache, embeds, positions, masks)

    graph_embeds, graph_fn = static_path()
    eager_embeds, eager_fn = static_path()
    result: dict[str, Any] = {
        "frames": frames,
        "tokens_per_frame": plan.max_new_tokens,
        "warpers": [type(w).__name__ for w in plan.warpers],
        "cache_len": cache_len,
    }
    try:
        graph, graph_out = capture(torch, graph_fn)
    except Exception as exc:  # a failed capture is the probe's answer, not a crash
        result.update(ok=False, captured=False, error=f"{type(exc).__name__}: {exc}"[:500])
        return result
    result["captured"] = True

    def graph_frame(n: int) -> Any:
        graph_embeds.copy_(inputs[n])
        graph.replay()
        return graph_out

    def eager_frame(n: int) -> Any:
        eager_embeds.copy_(inputs[n])
        return eager_fn()

    def loop_frame(n: int) -> Any:
        return qwen_code_predictor.sample_with_loop(predictor, inputs[n], plan).sequences

    paths = {"graph": graph_frame, "eager_static": eager_frame, "production_loop": loop_frame}
    times: dict[str, list[float]] = {name: [] for name in paths}
    codes: dict[str, list[Any]] = {}
    for _ in range(rounds):
        for name, frame in paths.items():
            codes[name] = []
            elapsed = 0.0
            for n in range(frames):
                torch.cuda.manual_seed(seed + n)
                torch.cuda.synchronize()
                started = perf_counter()
                codes[name].append(frame(n).clone())
                torch.cuda.synchronize()
                elapsed += perf_counter() - started
            times[name].append(elapsed * 1000.0 / frames)
    # As in a generation: one seed, then frame after frame on the same generator. The replays must consume the RNG
    # as the eager loop does, or every later sample (the talker's included) would drift.
    chained: dict[str, list[Any]] = {}
    rng_after: dict[str, Any] = {}
    for name in ("graph", "production_loop"):
        torch.cuda.manual_seed(seed)
        chained[name] = [paths[name](n).clone() for n in range(frames)]
        rng_after[name] = torch.cuda.get_rng_state()
    total = frames * plan.max_new_tokens

    def agree(a: str, b: str) -> int:
        return sum(int((x == y).sum()) for x, y in zip(codes[a], codes[b], strict=True))

    result.update(
        tokens_total=total,
        graph_vs_eager_static_equal=agree("graph", "eager_static"),
        graph_vs_production_equal=agree("graph", "production_loop"),
        eager_static_vs_production_equal=agree("eager_static", "production_loop"),
        frames_identical_to_production=sum(
            int(torch.equal(x, y)) for x, y in zip(codes["graph"], codes["production_loop"], strict=True)
        ),
        distinct_frames=len({tuple(c.flatten().tolist()) for c in codes["graph"]}),
        chained_frames_identical=sum(
            int(torch.equal(x, y)) for x, y in zip(chained["graph"], chained["production_loop"], strict=True)
        ),
        chained_rng_state_matches=bool(torch.equal(rng_after["graph"], rng_after["production_loop"])),
        ms_per_frame={name: round(statistics.median(values), 3) for name, values in times.items()},
    )
    result["ok"] = bool(
        result["graph_vs_eager_static_equal"] == total
        and result["distinct_frames"] == frames
        and result["graph_vs_production_equal"] >= 0.9 * total
        and result["chained_rng_state_matches"]
    )
    return result


def load_model(torch: Any) -> Any:
    """``Qwen3TTSModel`` loaded as ``PyTorchTTSBackend._load_model_sync`` loads it on a GPU."""
    from huggingface_hub import constants as hf_constants
    from qwen_tts import Qwen3TTSModel

    return Qwen3TTSModel.from_pretrained(
        HF_REPO,
        cache_dir=hf_constants.HF_HUB_CACHE,
        device_map="cuda",
        torch_dtype=torch.bfloat16,
    )


def run(args: argparse.Namespace) -> dict[str, Any]:
    import torch

    if not torch.cuda.is_available():
        raise SystemExit("no GPU: this probe needs the ROCm build of torch with a visible GPU")
    report: dict[str, Any] = {
        "torch": torch.__version__,
        "hip": torch.version.hip,
        "device": torch.cuda.get_device_name(0),
    }
    with torch.inference_mode():
        report["matmul"] = check_matmul(torch)
        logger.info("matmul: %s", report["matmul"])
        report["rng"] = check_rng(torch)
        logger.info("rng: %s", report["rng"])
        model = load_model(torch)
        talker = model.model.talker
        report["attn_implementation"] = talker.model.config._attn_implementation
        report["rope_scaling"] = talker.config.rope_scaling
        report["sliding_window"] = talker.config.sliding_window
        generator = torch.Generator(device="cuda").manual_seed(args.seed)
        vocab = talker.config.vocab_size
        embed = talker.model.codec_embedding
        ids = torch.randint(0, vocab, (1, args.prefix + args.steps), device="cuda", generator=generator)
        embeds = embed(ids)
        prompt, inputs = (
            embeds[:, : args.prefix],
            [embeds[:, args.prefix + n : args.prefix + n + 1] for n in range(args.steps)],
        )
        all_layers = list(talker.model.layers)

        def head(hidden: Any) -> Any:
            return talker.codec_head(talker.model.norm(hidden))

        report["stacks"] = {}
        for name, layers, stack_head, pad in (
            ("layer0", all_layers[:1], None, 0),
            ("backbone", all_layers, head, 0),
            (f"backbone_cache+{args.pad}", all_layers, head, args.pad),
        ):
            prefix = prefill(torch, talker.model.rotary_emb, layers, prompt)
            report["stacks"][name] = check_stack(
                torch,
                talker,
                layers,
                stack_head,
                prefix,
                inputs,
                args.prefix + args.steps + pad,
                args.rounds,
            )
            logger.info("%s: %s", name, report["stacks"][name])
        report["code_predictor"] = check_code_predictor(torch, model, args.frames, args.rounds, args.seed)
        logger.info("code_predictor: %s", report["code_predictor"])
    stacks = report["stacks"].values()
    report["viable"] = bool(
        report["matmul"]["ok"]
        and report["rng"]["ok"]
        and all(s.get("ok") for s in stacks)
        and report["code_predictor"]["ok"]
    )
    return report


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0], allow_abbrev=False)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT, help="directory for hipgraph-probe.json")
    parser.add_argument("--prefix", type=int, default=450, help="prefill length (the talker's prompt is ~450)")
    parser.add_argument("--steps", type=int, default=32, help="talker decode steps per sequence")
    parser.add_argument("--pad", type=int, default=2048, help="extra static-cache slots for the padded backbone run")
    parser.add_argument("--frames", type=int, default=32, help="code-predictor frames per sequence")
    parser.add_argument("--rounds", type=int, default=3, help="sequences per path (median time; graph determinism)")
    parser.add_argument("--seed", type=int, default=0, help="seed of the random inputs and of the frames' sampling")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    args = parse_args(argv)
    report = run(args)
    args.out.mkdir(parents=True, exist_ok=True)
    path = args.out / "hipgraph-probe.json"
    path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    logger.info("viable=%s, report in %s", report["viable"], path)
    return 0 if report["viable"] else 1


if __name__ == "__main__":
    sys.exit(main())
