"""
Tests for scripts/profile_talker.py, the talker profiler of the perf loop (plan, Fase 3.1).

Everything here runs without a GPU or torch: the timer split per decode step, the profiler-event analysis (ATen vs
Python time, host reads, launch ops per region), the step estimates, the report, the env-variant plumbing and the
probe's wrappers (installed, attributed and removed) on a fake talker.

Usage:
    python -m pytest backend/tests/test_profile_talker.py -v
"""

import contextlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

_SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "profile_talker.py"
_spec = importlib.util.spec_from_file_location("profile_talker", _SCRIPT)
pt = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(pt)


# ---------------------------------------------------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------------------------------------------------


def test_stats_empty_and_values():
    assert pt.stats([]) == {"mean": None, "median": None, "min": None, "max": None, "n": 0}
    assert pt.stats([1, 2, 6]) == {"mean": 3.0, "median": 2.0, "min": 1.0, "max": 6.0, "n": 3}


def test_load_phrase(tmp_path):
    path = tmp_path / "phrases.json"
    path.write_text(
        json.dumps({"language": "es", "phrases": [{"id": "p01", "text": "Hola.", "seed": "11", "target_s": 3}]}),
        encoding="utf-8",
    )
    assert pt.load_phrase(path, "p01") == {"id": "p01", "text": "Hola.", "seed": 11, "language": "es"}
    with pytest.raises(pt.ProfileError):
        pt.load_phrase(path, "p02")


def test_default_phrase_exists_in_the_bench_set():
    phrase = pt.load_phrase(pt.DEFAULT_PHRASES, "p08")
    assert phrase["language"] == "es"
    assert pt.LANGUAGE_NAMES[phrase["language"]] == "spanish"


def test_find_prompt_file(tmp_path):
    cache = tmp_path / "cache"
    cache.mkdir()
    with pytest.raises(pt.ProfileError):
        pt.find_prompt_file(tmp_path, "1.7B")
    (cache / "abc.prompt").write_bytes(b"old format, other key")
    (cache / "qwen_other_abc.prompt").write_bytes(b"other model size")
    wanted = cache / "qwen_1.7B_abc.prompt"
    wanted.write_bytes(b"x")
    assert pt.find_prompt_file(tmp_path, "1.7B") == wanted
    (cache / "qwen_1.7B_def.prompt").write_bytes(b"x")
    with pytest.raises(pt.ProfileError):
        pt.find_prompt_file(tmp_path, "1.7B")


def test_validate_cached_prompt():
    good = {"ref_code": [1], "ref_spk_embedding": [2], "x_vector_only_mode": [False], "icl_mode": [True]}
    good["ref_text"] = ["hola"]
    assert pt.validate_cached_prompt(good) is good
    with pytest.raises(pt.ProfileError):
        pt.validate_cached_prompt([good])
    with pytest.raises(pt.ProfileError):
        pt.validate_cached_prompt({**good, "ref_text": "hola"})
    with pytest.raises(pt.ProfileError):
        pt.validate_cached_prompt({**good, "ref_text": ["a", "b"]})
    with pytest.raises(pt.ProfileError):
        pt.validate_cached_prompt({name: [] for name in good})


def test_dummy_repeats():
    assert pt.dummy_repeats(100.0, 20.0) == 11  # 2 x 100 / 20 + 1
    assert pt.dummy_repeats(1.0, 20.0) == 2
    assert pt.dummy_repeats(100.0, 0.0) == 2


def test_unit_value_falls_back_on_eager_time():
    assert pt.unit_value({"gpu_ms": 2.5, "eager_ms": 9.0}) == 2.5
    assert pt.unit_value({"gpu_ms": None, "eager_ms": 9.0}) == 9.0


def test_gemv_figures():
    figures = pt.gemv_figures(10.0, 2_000_000_000, 15.0)
    assert figures == {"gb_per_s": 200.0, "ms_per_step": 150.0, "weight_gb": 2.0}
    assert pt.gemv_figures(None, 2_000_000_000, 1.0)["gb_per_s"] is None


# ---------------------------------------------------------------------------------------------------------------------
# Timer run split
# ---------------------------------------------------------------------------------------------------------------------


def _call(region, step, start, end, index=0):
    return pt.Call(region, step, index, start, end)


def test_light_steps_split_and_host_reads():
    calls = [
        _call(pt.R_PREFILL, -1, 0.000, 0.050),
        _call(pt.R_BACKBONE, -1, 0.001, 0.049),
        # step 0: forward 0.100-0.190, next forward at 0.200
        _call(pt.R_CP_FORWARD, 0, 0.105, 0.115, 0),
        _call(pt.R_CP_FORWARD, 0, 0.120, 0.130, 1),
        _call(pt.R_CP_GENERATE, 0, 0.101, 0.140),
        _call(pt.R_BACKBONE, 0, 0.150, 0.185),
        _call(pt.R_STEP, 0, 0.100, 0.190),
        # step 1: forward 0.200-0.280, generation ends at 0.300
        _call(pt.R_CP_GENERATE, 1, 0.201, 0.230),
        _call(pt.R_BACKBONE, 1, 0.240, 0.270),
        _call(pt.R_STEP, 1, 0.200, 0.280),
    ]
    reads = [
        _call("cp_forward", 0, 0.106, 0.107),
        _call("cp_generate_rest", 0, 0.116, 0.119),
        _call("talker_loop", 0, 0.191, 0.195),
        _call("prefill", -1, 0.010, 0.020),
    ]
    rows = pt.light_steps(calls, reads, generate_end=0.300)
    assert [row["step"] for row in rows] == [0, 1]
    first, second = rows
    assert first["step_ms"] == pytest.approx(100.0)
    assert first["talker_forward_ms"] == pytest.approx(90.0)
    assert first["cp_generate_ms"] == pytest.approx(39.0)
    assert first["cp_forward_ms"] == pytest.approx(20.0)
    assert first["cp_forward_n"] == 2
    assert first["cp_generate_rest_ms"] == pytest.approx(19.0)
    assert first["backbone_ms"] == pytest.approx(35.0)
    assert first["talker_forward_rest_ms"] == pytest.approx(16.0)
    assert first["talker_loop_ms"] == pytest.approx(10.0)
    assert first["host_read_n"] == 3
    assert first["host_read_ms"] == pytest.approx(8.0)
    assert first["cpu_busy_ms"] == pytest.approx(92.0)
    assert first["host_read_ms.cp_generate_rest"] == pytest.approx(3.0)
    assert first["host_read_n.backbone"] == 0
    assert second["step_ms"] == pytest.approx(100.0)  # until the end of generate
    assert second["cp_forward_n"] == 0
    assert second["host_read_n"] == 0
    summary = pt.summarize_rows(rows, pt.LIGHT_KEYS)
    assert summary["step_ms"]["n"] == 2
    assert set(summary) == set(pt.LIGHT_KEYS)


# ---------------------------------------------------------------------------------------------------------------------
# Profiler events
# ---------------------------------------------------------------------------------------------------------------------


def _event(name, start, end, thread=1, parent=None):
    """An object with the FunctionEvent attributes the script reads."""
    event = SimpleNamespace(
        name=name,
        time_range=SimpleNamespace(start=start, end=end),
        thread=thread,
        cpu_parent=parent,
        cpu_children=[],
    )
    if parent is not None:
        parent.cpu_children.append(event)
    return event


def test_events_from_profiler_keeps_the_annotated_thread_and_finds_read_sources():
    step = _event(pt.R_STEP, 0, 100)
    nonzero = _event("aten::is_nonzero", 10, 20, parent=step)
    item = _event("aten::item", 11, 19, parent=nonzero)
    read = _event(pt.SYNC_OP, 12, 18, parent=item)
    linear = _event("aten::linear", 30, 40, parent=step)
    mm = _event("aten::mm", 31, 39, parent=linear)
    other_thread = _event("aten::add", 50, 60, thread=2)
    not_aten = _event("cudaDeviceSynchronize", 70, 80)
    annotations, ops = pt.events_from_profiler([step, nonzero, item, read, linear, mm, other_thread, not_aten])
    assert [a.name for a in annotations] == [pt.R_STEP]
    by_name = {op.name: op for op in ops}
    assert set(by_name) == {"aten::is_nonzero", "aten::item", pt.SYNC_OP, "aten::linear", "aten::mm"}
    assert by_name["aten::is_nonzero"].top
    assert not by_name["aten::is_nonzero"].leaf
    assert by_name[pt.SYNC_OP].source == "aten::is_nonzero"
    assert by_name["aten::mm"].leaf
    assert not by_name["aten::mm"].top
    assert pt.events_from_profiler([not_aten]) == ([], [])


def test_is_kernel_op():
    assert pt.is_kernel_op(pt.Ev("aten::mm", 0, 1, leaf=True))
    assert not pt.is_kernel_op(pt.Ev("aten::mm", 0, 1, leaf=False))
    assert not pt.is_kernel_op(pt.Ev("aten::view", 0, 1, leaf=True))
    assert not pt.is_kernel_op(pt.Ev(pt.SYNC_OP, 0, 1, leaf=True))
    assert not pt.is_kernel_op(pt.Ev("vb::talker.step", 0, 1, leaf=True))


def _profile_fixture():
    """Two decode steps (0-1000 us, 1000-2000 us) closed by a mark; times in microseconds."""
    a = pt.Ev
    annotations = [
        a(pt.R_STEP, 0, 900),
        a(pt.R_CP_GENERATE, 10, 500),
        a(pt.R_CP_FORWARD, 20, 220),
        a(pt.R_CP_FORWARD, 250, 450),
        a(pt.R_BACKBONE, 600, 850),
        a(pt.R_STEP, 1000, 1900),
        a(pt.R_CP_GENERATE, 1010, 1500),
        a(pt.R_CP_FORWARD, 1020, 1220),
        a(pt.R_CP_FORWARD, 1250, 1450),
        a(pt.R_BACKBONE, 1600, 1850),
        a(pt.R_MARK, 2000, 2000),
    ]
    ops = []
    for base in (0, 1000):
        ops += [
            a("aten::mm", base + 30, base + 130, top=True, leaf=True),  # cp forward 1
            a("aten::view", base + 140, base + 150, top=True, leaf=True),  # no kernel
            a("aten::mm", base + 260, base + 360, top=True, leaf=True),  # cp forward 2
            a("aten::is_nonzero", base + 460, base + 490, top=True),  # cp loop host read
            a(pt.SYNC_OP, base + 465, base + 485, leaf=True, source="aten::is_nonzero"),
            a("aten::mm", base + 610, base + 810, top=True, leaf=True),  # backbone
            a("aten::add", base + 860, base + 870, top=True, leaf=True),  # talker forward rest
            a("aten::softmax", base + 920, base + 940, top=True, leaf=True),  # talker loop
        ]
    return annotations, sorted(ops, key=lambda e: e.start)


def test_analyze_profile_regions_add_up():
    annotations, ops = _profile_fixture()
    result = pt.analyze_profile(annotations, ops)
    rows = result["rows"]
    assert len(rows) == 2
    row = rows[0]
    assert row["ms"] == pytest.approx(1.0)
    assert row["aten_ms"] == pytest.approx(0.47)  # 100 + 10 + 100 + 30 + 200 + 10 + 20 us
    assert row["python_ms"] == pytest.approx(0.53)
    assert row["sync_n"] == 1
    assert row["sync_ms"] == pytest.approx(0.02)
    assert row["ops_n"] == 5  # 3 mm, add, softmax; the view and the read launch nothing
    assert row["cp_forward_n"] == 2
    regions = row["regions"]
    assert regions["cp_forward"]["ms"] == pytest.approx(0.4)
    assert regions["cp_forward"]["ops_n"] == 2
    assert regions["cp_generate_rest"]["ms"] == pytest.approx(0.09)
    assert regions["cp_generate_rest"]["sync_n"] == 1
    assert regions["cp_generate_rest"]["ops_n"] == 0
    assert regions["backbone"]["aten_ms"] == pytest.approx(0.2)
    assert regions["talker_forward_rest"]["ms"] == pytest.approx(0.9 - 0.49 - 0.25)
    assert regions["talker_forward_rest"]["ops_n"] == 1
    assert regions["talker_loop"]["ms"] == pytest.approx(0.1)
    assert regions["talker_loop"]["ops_n"] == 1
    total = sum(regions[name]["ms"] for name in pt.STEP_REGIONS)
    assert total == pytest.approx(row["ms"])
    assert sum(regions[name]["ops_n"] for name in pt.STEP_REGIONS) == row["ops_n"]
    summary = result["summary"]
    assert summary["steps"] == 2
    assert summary["sync_sources_per_step"] == {"aten::is_nonzero": 1.0}
    assert summary["top_ops_per_step"][0] == {"op": "aten::mm", "per_step": 3.0, "cpu_ms_per_step": 0.4}
    assert summary["regions"]["backbone"]["ops_n"]["mean"] == 1.0


def test_analyze_profile_skips_a_step_without_an_end():
    annotations, ops = _profile_fixture()
    without_mark = [a for a in annotations if a.name != pt.R_MARK]
    assert len(pt.analyze_profile(without_mark, ops)["rows"]) == 1


# ---------------------------------------------------------------------------------------------------------------------
# Estimates and report
# ---------------------------------------------------------------------------------------------------------------------


def test_estimate_gpu_step():
    estimate = pt.estimate_gpu_step(
        backbone_gpu_ms=20.0,
        cp_forward_gpu_ms=2.0,
        cp_sampling_gpu_ms=0.2,
        talker_sampling_gpu_ms=1.0,
        tiny_gpu_ms=0.002,
        cp_forwards=15,
        ops_step=8000,
        ops_backbone=2000,
        ops_cp_forward=5000,
        ops_cp_sampling=20,
        ops_talker_sampling=40,
    )
    assert estimate["other_ops"] == pytest.approx(660.0)
    assert estimate["cp_forward_ms"] == pytest.approx(30.0)
    assert estimate["cp_sampling_ms"] == pytest.approx(3.0)
    assert estimate["other_ops_ms"] == pytest.approx(1.32)
    assert estimate["total_ms"] == pytest.approx(55.32)


def test_estimate_gpu_step_never_counts_negative_other_ops():
    estimate = pt.estimate_gpu_step(
        backbone_gpu_ms=1.0,
        cp_forward_gpu_ms=0.0,
        cp_sampling_gpu_ms=0.0,
        talker_sampling_gpu_ms=0.0,
        tiny_gpu_ms=1.0,
        cp_forwards=0,
        ops_step=10,
        ops_backbone=20,
        ops_cp_forward=0,
        ops_cp_sampling=0,
        ops_talker_sampling=0,
    )
    assert estimate["other_ops"] == 0.0
    assert estimate["total_ms"] == 1.0


def test_estimate_dispatch_step():
    estimate = pt.estimate_dispatch_step(
        backbone_dispatch_ms=16.0,
        cp_forward_dispatch_ms=3.0,
        cp_sampling_dispatch_ms=0.1,
        talker_sampling_dispatch_ms=1.0,
        tiny_dispatch_ms=0.002,
        cp_forwards=15,
        other_ops=500,
    )
    assert estimate["total_ms"] == pytest.approx(16.0 + 45.0 + 1.5 + 1.0 + 1.0)


def _stat(value):
    return pt.stats([value])


def test_render_markdown_lists_every_section():
    region = {key: _stat(1.0) for key in pt.PROFILE_KEYS}
    result = {
        "phrase": {"id": "p08", "seed": 1108},
        "env": {"torch": "2.9.1", "attn_talker": "sdpa"},
        "timing": {"base": {"ms": _stat(7800.0), "steps": 78, "ms_per_step": 100.0}},
        "light": {"summary": {"step_ms": _stat(96.0)}},
        "profiler": {
            "summary": {
                "steps": 20,
                "step": region,
                "regions": {name: region for name in pt.STEP_REGIONS},
                "sync_sources_per_step": {"aten::is_nonzero": 33.0},
                "top_ops_per_step": [{"op": "aten::copy_", "per_step": 1915.0, "cpu_ms_per_step": 7.0}],
            }
        },
        "replay": {
            "units": {
                "cp_forward": {"gpu_ms": 2.1, "graph_launch_ms": 1.7, "dispatch_ms": 3.2, "eager_ms": 3.4, "ops": 353},
                "talker_sampling": {"gpu_ms": None, "eager_ms": 1.0, "graph_error": "AcceleratorError: x\nmore"},
            }
        },
        "estimate": {"step_ms_measured": 96.0},
        "env_variants": {
            "base": {
                "ms": _stat(7800.0),
                "frames": [78],
                "ms_per_step": 100.0,
                "light": {"step_ms": 96.0, "cpu_busy_ms": 81.0, "host_read_ms": 14.0},
                "launch_ops_per_step": 8037.0,
                "gemv": {"backbone_linears": {"gb_per_s": 178.0}, "cp_linears": {"gb_per_s": 152.0}},
            },
            "tunableop_tuning": {"error": "exit 1"},
        },
    }
    text = pt.render_markdown(result)
    for heading in ("Tiempo del talker", "Reparto por paso", "Perfil CPU (20 pasos)", "grafo HIP", "Variantes"):
        assert heading in text
    assert "aten::is_nonzero 33.0" in text
    assert "AcceleratorError: x |" in text  # first line of the error only: one table row per unit
    assert "error: exit 1" in text


# ---------------------------------------------------------------------------------------------------------------------
# Env variants plumbing
# ---------------------------------------------------------------------------------------------------------------------


def test_variant_env_points_tunableop_at_the_output_dir(tmp_path):
    assert pt.variant_env("base", tmp_path, "p08") == {}
    assert pt.variant_env("aotriton_experimental", tmp_path, "p08") == {"TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL": "1"}
    tuning = pt.variant_env("tunableop_tuning", tmp_path, "p08")
    readonly = pt.variant_env("tunableop_readonly", tmp_path, "p08")
    assert tuning["PYTORCH_TUNABLEOP_TUNING"] == "1"
    assert readonly["PYTORCH_TUNABLEOP_TUNING"] == "0"
    assert tuning["PYTORCH_TUNABLEOP_FILENAME"] == readonly["PYTORCH_TUNABLEOP_FILENAME"]
    assert Path(tuning["PYTORCH_TUNABLEOP_FILENAME"]).parent == tmp_path / "tunableop"
    assert (tmp_path / "tunableop").is_dir()
    order = list(pt.ENV_VARIANTS)
    assert order.index("tunableop_tuning") < order.index("tunableop_readonly")  # read-only needs the tuned file


def test_child_command_and_result_round_trip(tmp_path):
    args = pt.parse_args(["--phrase", "p01", "--out", str(tmp_path), "--rounds", "2"])
    command = pt.child_command(args, "aotriton_experimental")
    assert command[1] == str(_SCRIPT)
    assert command[command.index("--child") + 1] == "aotriton_experimental"
    assert command[command.index("--rounds") + 1] == "2"
    assert "--prompt-file" not in command
    child_args = pt.parse_args(command[2:])
    assert child_args.child == "aotriton_experimental"
    assert child_args.phrase == "p01"
    for name in ("phrases", "data_dir", "prompt_file", "model_size", "out", "rounds", "replay_calls"):
        assert hasattr(child_args, name)
    stdout = "noise\n" + pt.CHILD_RESULT_PREFIX + json.dumps({"variant": "x", "ms": 1}) + "\nmore noise\n"
    assert pt.parse_child_result(stdout) == {"variant": "x", "ms": 1}
    assert pt.parse_child_result("nothing here") is None


def test_parse_args_defaults():
    args = pt.parse_args([])
    assert args.phrase == "p08"
    assert args.out == pt.DEFAULT_OUT
    assert args.data_dir == pt.DEFAULT_DATA_DIR
    assert args.model_size == "1.7B"
    assert args.batch
    assert args.env_variants
    assert args.child is None
    with pytest.raises(SystemExit):
        pt.parse_args(["--child", "unknown"])


def test_main_reports_setup_errors(monkeypatch):
    def broken(_args):
        raise pt.ProfileError("no GPU")

    monkeypatch.setattr(pt, "run", broken)
    assert pt.main(["--no-env-variants"]) == 2


# ---------------------------------------------------------------------------------------------------------------------
# Probe wrappers on a fake talker
# ---------------------------------------------------------------------------------------------------------------------


class FakeTensor:
    def __init__(self, value=1, shape=(1, 1)):
        self.value = value
        self.shape = shape

    def __bool__(self):
        return bool(self.value)

    def item(self):
        return self.value

    def detach(self):
        return self

    def clone(self):
        return FakeTensor(self.value, self.shape)


class FakeCache:
    def __init__(self, length=0):
        self.length = length

    def get_seq_length(self):
        return self.length

    def crop(self, length):
        self.length = length


class FakeOut:
    def __init__(self):
        self.logits = FakeTensor()


class FakeCodePredictor:
    """generate(): logits processors, then per sub-step prepare inputs, forward (mask read) and a stopping read."""

    sub_steps = 3

    def generate(self, **kwargs):
        self._get_logits_processor()
        input_ids = FakeTensor(shape=(1, 0))
        out = None
        for index in range(self.sub_steps):
            self.prepare_inputs_for_generation(input_ids)
            out = self.forward(input_ids=input_ids, attention_mask=FakeTensor(), past_key_values=FakeCache(index))
            bool(FakeTensor())
        return out

    def forward(self, **kwargs):
        bool(FakeTensor())
        return FakeOut()

    def _get_logits_processor(self):
        return ["cp-processors"]

    def prepare_inputs_for_generation(self, input_ids, **kwargs):
        return {}


class FakeBackbone:
    def forward(self, **kwargs):
        return FakeOut()


class FakeTalker:
    """Prefill forward with 5 embeddings, then one decode forward per step: code predictor, then backbone."""

    def __init__(self):
        self.model = FakeBackbone()
        self.code_predictor = FakeCodePredictor()

    def forward(self, inputs_embeds=None, **kwargs):
        if inputs_embeds is None:
            self.code_predictor.generate()
        self.model.forward(past_key_values=FakeCache(10), attention_mask=FakeTensor(), position_ids=FakeTensor(7))
        return FakeOut()

    def _get_logits_processor(self):
        return ["talker-processors"]

    def prepare_inputs_for_generation(self, input_ids, **kwargs):
        return {}

    def generate(self, steps):
        self._get_logits_processor()
        input_ids = FakeTensor(shape=(1, 0))
        self.prepare_inputs_for_generation(input_ids)
        self.forward(inputs_embeds=FakeTensor(shape=(1, 5)))
        for _ in range(steps):
            bool(FakeTensor())  # HF's "unfinished sequences" check
            self.prepare_inputs_for_generation(input_ids)
            self.forward(input_ids=input_ids)


class RecordingProfiler:
    """record_function, start and stop, recorded in order."""

    def __init__(self):
        self.events = []

    def record_function(self, name):
        @contextlib.contextmanager
        def scope():
            self.events.append(name)
            yield

        return scope()

    def start(self):
        self.events.append("start")

    def stop(self):
        self.events.append("stop")


def _fake_torch(recorder=None):
    profiler = recorder or RecordingProfiler()
    return SimpleNamespace(Tensor=FakeTensor, profiler=SimpleNamespace(record_function=profiler.record_function))


def test_probe_times_regions_attributes_reads_and_restores_everything():
    talker = FakeTalker()
    original_bool = FakeTensor.__dict__["__bool__"]
    original_item = FakeTensor.__dict__["item"]
    with pt.TalkerProbe(_fake_torch(), talker, time_reads=True, capture_step=1, capture_substep=2) as probe:
        assert "forward" in vars(talker)
        talker.generate(steps=3)
    for obj in (talker, talker.model, talker.code_predictor):
        assert "forward" not in vars(obj)
        assert "prepare_inputs_for_generation" not in vars(obj)
    assert "generate" not in vars(talker.code_predictor)
    assert FakeTensor.__dict__["__bool__"] is original_bool
    assert FakeTensor.__dict__["item"] is original_item

    assert sorted({c.step for c in probe.calls if c.region == pt.R_STEP}) == [0, 1, 2]
    assert [c.step for c in probe.calls if c.region == pt.R_PREFILL] == [-1]
    assert [c.index for c in probe.calls if c.region == pt.R_CP_FORWARD and c.step == 0] == [0, 1, 2]
    # Per step: 3 reads in the code-predictor forwards, 3 stopping checks in its generate, 1 in the talker loop.
    step1 = sorted(r.region for r in probe.reads if r.step == 1)
    assert step1 == sorted(["cp_forward"] * 3 + ["cp_generate_rest"] * 3 + ["talker_loop"])
    rows = pt.light_steps(probe.calls, probe.reads, generate_end=probe.calls[-1].end)
    assert [row["host_read_n"] for row in rows] == [7, 7, 6]  # the last step has no next "unfinished" check

    captured = probe.captured
    assert captured["talker_processors"] == ["talker-processors"]
    assert captured["cp_processors"] == ["cp-processors"]
    assert isinstance(captured["backbone_kwargs"]["past_key_values"], FakeCache)
    assert captured["backbone_kwargs"]["position_ids"].value == 7
    assert captured["cp_kwargs"]["past_key_values"].get_seq_length() == 2  # sub-step 2's cache
    assert {"cp_logits", "cp_input_ids", "talker_logits", "talker_input_ids"} <= set(captured)


def test_probe_annotates_and_opens_the_profiler_window():
    recorder = RecordingProfiler()
    talker = FakeTalker()
    with pt.TalkerProbe(_fake_torch(recorder), talker, annotate=True, profiler=recorder, profile_steps=(1, 2)) as probe:
        talker.generate(steps=4)
    events = recorder.events
    start, stop = events.index("start"), events.index("stop")
    assert events[start + 1] == pt.R_STEP  # step 1 opens right after the profiler starts
    assert events[stop - 1] == pt.R_MARK  # the window ends with a mark, before step 2's range
    assert events[start:stop].count(pt.R_STEP) == 1
    assert not probe.profiling


def test_probe_stops_an_open_profiler_window_on_exit():
    recorder = RecordingProfiler()
    talker = FakeTalker()
    with pt.TalkerProbe(_fake_torch(recorder), talker, profiler=recorder, profile_steps=(1, 99)):
        talker.generate(steps=3)
    assert recorder.events[-2:] == [pt.R_MARK, "stop"]


def test_snapshot_copies_tensors_and_caches():
    cache = FakeCache(4)
    tensor = FakeTensor(3)
    copied = pt.snapshot(_fake_torch(), {"a": tensor, "b": [tensor, 5], "c": cache, "d": None})
    assert copied["a"] is not tensor
    assert copied["a"].value == 3
    assert copied["b"][1] == 5
    assert copied["c"] is not cache
    assert copied["c"].get_seq_length() == 4
    assert copied["d"] is None
