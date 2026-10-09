"""
Tests for the code predictor's HIP graph (perf-loop plan, Fase 3.5, task 3.5b): one batch-1 frame of the 3.2 loop
captured when the model loads and replayed on every talker step (backend/utils/qwen_code_predictor.py), and how
PyTorchTTSBackend captures and frees it.

Runs without a GPU. The graph itself needs one, so ``_capture`` is replaced by a fake whose replay runs the captured
frame again into the same output buffer, as a real replay relaunches the captured kernels on the same buffers. What
runs for real, on a tiny code predictor with random weights on the CPU (the real qwen_tts classes): the frame that gets
captured (the loop on a fixed input buffer), the copy in and the clone out, the dispatch between graph, loop and HF
generate, the fallbacks and the release.

Usage:
    python -m pytest backend/tests/test_qwen_code_predictor_graph.py -v
"""

import asyncio
import contextlib
import logging
import sys
import threading
import types
from typing import Any

import pytest
import torch

from backend import backends
from backend.backends import pytorch_backend
from backend.services import task_queue
from backend.tests.test_qwen_code_predictor import (
    _CODE_GROUPS,
    _TALKER_ARGUMENTS,
    _TALKER_HIDDEN,
    _embeds,
    _host_reads,
    _predictor,
)
from backend.utils import perf_log, qwen_code_predictor, qwen_warmup

_HIP = "7.2.53211-158bd99533"
# generate_config.json of the 1.7B Base model, as Qwen3TTSModel._merge_generate_kwargs returns it.
_SUBTALKER_SETTINGS = {
    "subtalker_dosample": True,
    "subtalker_top_k": 50,
    "subtalker_top_p": 1.0,
    "subtalker_temperature": 0.9,
}


def _qwen_model(predictor: Any, settings: dict[str, Any] | None = None) -> Any:
    """``Qwen3TTSModel`` layout around ``predictor``: the talker's config and embeddings, and the generate settings."""
    embedding = torch.nn.Embedding(8, _TALKER_HIDDEN).to(predictor.dtype)
    talker = types.SimpleNamespace(
        code_predictor=predictor,
        config=types.SimpleNamespace(hidden_size=_TALKER_HIDDEN, num_code_groups=_CODE_GROUPS),
        get_input_embeddings=lambda: embedding,
    )
    merged = {"max_new_tokens": 4096, "do_sample": True, **_SUBTALKER_SETTINGS, **(settings or {})}
    return types.SimpleNamespace(
        model=types.SimpleNamespace(talker=talker),
        _merge_generate_kwargs=lambda: dict(merged),
    )


def _talker_call(predictor: Any, embeds: torch.Tensor, **overrides: Any) -> Any:
    """``code_predictor.generate`` as the talker calls it on each step."""
    return predictor.generate(inputs_embeds=embeds, **{**_TALKER_ARGUMENTS, **overrides})


class FakeGraph:
    """``torch.cuda.CUDAGraph`` stand-in: a replay runs the captured frame again into the fixed output buffer."""

    def __init__(self, fn, output: torch.Tensor):
        self.fn = fn
        self.output = output
        self.replays = 0
        self.resets = 0
        self.fail_next = False

    def replay(self) -> None:
        if self.fail_next:
            self.fail_next = False
            raise RuntimeError("hipErrorLaunchFailure")
        self.replays += 1
        self.output.copy_(self.fn())

    def reset(self) -> None:
        self.resets += 1


@pytest.fixture
def graphs(monkeypatch) -> list[FakeGraph]:
    """Replaces the CUDA capture: each capture returns a :class:`FakeGraph` of the frame (recorded here)."""
    captured: list[FakeGraph] = []

    def capture(fn, device):
        assert device == torch.device("cpu")
        output = fn()
        captured.append(FakeGraph(fn, output))
        return captured[-1], output

    monkeypatch.setattr(qwen_code_predictor, "_capture", capture)
    return captured


@pytest.fixture
def loop_calls(monkeypatch) -> list[int]:
    """Batch sizes of the calls that ran the loop outside a graph (the captured frame runs it too, not counted)."""
    calls: list[int] = []
    in_frame = threading.local()
    real_loop = qwen_code_predictor.sample_with_loop
    real_frame = qwen_code_predictor.FrameGraph._frame

    def counting(predictor, inputs_embeds, plan):
        if not getattr(in_frame, "active", False):
            calls.append(inputs_embeds.shape[0])
        return real_loop(predictor, inputs_embeds, plan)

    def frame(self):
        in_frame.active = True
        try:
            return real_frame(self)
        finally:
            in_frame.active = False

    monkeypatch.setattr(qwen_code_predictor, "sample_with_loop", counting)
    monkeypatch.setattr(qwen_code_predictor.FrameGraph, "_frame", frame)
    return calls


def _captured(attn: str = "sdpa", dtype: torch.dtype = torch.float32, settings: dict | None = None):
    """A tiny predictor with the loop installed and its frame captured; its model and the HF generate."""
    predictor = _predictor(attn, dtype)
    hf_generate = predictor.generate
    model = _qwen_model(predictor, settings)
    assert qwen_code_predictor.install(model) is True
    assert qwen_code_predictor.capture_graph(model) is True
    return predictor, model, hf_generate


class TestReplaySamplesAsTheLoop:
    """Frame after frame from one seed, the graph gives the loop's codes and leaves the RNG where the loop does."""

    @pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
    @pytest.mark.parametrize("attn", ["sdpa", "eager"])
    @pytest.mark.parametrize(
        "settings",
        [
            {},  # the talker's subtalker settings
            {"subtalker_top_p": 0.8, "subtalker_top_k": 0, "subtalker_temperature": 1.0},
            {"subtalker_dosample": False},
        ],
    )
    def test_chained_frames(self, graphs, loop_calls, attn, dtype, settings):
        predictor, model, hf_generate = _captured(attn, dtype, settings)
        arguments = qwen_code_predictor.talker_arguments(model)
        frames = [_embeds(1, dtype, seed=seed) for seed in range(4)]

        torch.manual_seed(11)
        expected = [hf_generate(inputs_embeds=f, **arguments).sequences for f in frames]
        expected_rng = torch.get_rng_state()
        torch.manual_seed(11)
        got = [predictor.generate(inputs_embeds=f, **arguments) for f in frames]
        rng = torch.get_rng_state()

        for result, codes in zip(got, expected, strict=True):
            assert result.sequences.shape == (1, _CODE_GROUPS - 1)
            assert result.sequences.dtype == torch.long
            assert torch.equal(result.sequences, codes)
        assert torch.equal(rng, expected_rng)
        assert graphs[0].replays == len(frames)
        assert loop_calls == []
        if arguments["do_sample"]:
            assert len({tuple(r.sequences.flatten().tolist()) for r in got}) > 1

    def test_returns_a_copy_of_the_output_buffer(self, graphs):
        predictor, _, _ = _captured(settings={"subtalker_dosample": False})
        first = _talker_call(predictor, _embeds(1, torch.float32, seed=1)).sequences
        kept = first.clone()
        _talker_call(predictor, _embeds(1, torch.float32, seed=2))
        assert first.data_ptr() != graphs[0].output.data_ptr()
        assert torch.equal(first, kept)

    def test_output_carries_no_hidden_states_or_cache(self, graphs):
        """They stay inside the graph; the talker only reads ``.sequences``."""
        predictor, _, _ = _captured()
        result = _talker_call(predictor, _embeds(1, torch.float32))
        assert result.hidden_states is None
        assert result.past_key_values is None


class TestFrameIsCapturable:
    """A capture records one frame's kernels; replaying them is only right if every frame launches the same ones."""

    def test_no_host_reads(self, graphs):
        predictor, _, _ = _captured()
        graph = predictor.generate.frame_graph
        graph.embeds.copy_(_embeds(1, torch.float32))
        assert _host_reads(graph._frame) == 0

    def test_same_ops_and_shapes_for_any_input(self, graphs):
        predictor, _, _ = _captured()
        graph = predictor.generate.frame_graph

        def trace(seed: int) -> list[tuple[str, str]]:
            graph.embeds.copy_(_embeds(1, torch.float32, seed=seed))
            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU], record_shapes=True) as prof:
                graph._frame()
            return [(event.name, str(event.input_shapes)) for event in prof.events()]

        first, second = trace(1), trace(2)
        assert len(first) > 100
        assert first == second


class TestDispatch:
    """Which calls replay the graph, which run the loop and which go to HF generate."""

    def test_batch_of_two_runs_the_loop(self, graphs, loop_calls):
        predictor, _, hf_generate = _captured()
        embeds = _embeds(2, torch.float32)
        torch.manual_seed(5)
        expected = hf_generate(inputs_embeds=embeds, **_TALKER_ARGUMENTS).sequences
        torch.manual_seed(5)
        result = _talker_call(predictor, embeds)
        assert torch.equal(result.sequences, expected)
        assert loop_calls == [2]
        assert graphs[0].replays == 0

    @pytest.mark.parametrize(
        ("transposed", "overrides"),
        [
            (False, {"top_k": 10}),  # another plan
            (False, {"return_dict_in_generate": False}),
            (True, {}),  # not contiguous
        ],
    )
    def test_calls_the_graph_was_not_captured_for_run_the_loop(self, graphs, loop_calls, transposed, overrides):
        predictor, _, hf_generate = _captured()
        embeds = _embeds(1, torch.float32)
        if transposed:
            embeds = embeds.transpose(1, 2).contiguous().transpose(1, 2)
            assert not embeds.is_contiguous()
        torch.manual_seed(5)
        expected = hf_generate(inputs_embeds=embeds, **{**_TALKER_ARGUMENTS, **overrides})
        torch.manual_seed(5)
        result = _talker_call(predictor, embeds, **overrides)
        assert torch.equal(getattr(result, "sequences", result), getattr(expected, "sequences", expected))
        assert loop_calls == [1]
        assert graphs[0].replays == 0

    def test_matches_only_its_own_frame(self, graphs):
        predictor, _, _ = _captured()
        graph = predictor.generate.frame_graph
        plan = graph.plan
        embeds = _embeds(1, torch.float32)
        assert graph.matches(plan, embeds)
        assert not graph.matches(predictor.generate.plan_for((), {"inputs_embeds": embeds, "top_k": 10}), embeds)
        assert not graph.matches(plan, _embeds(2, torch.float32))  # batch of two
        assert not graph.matches(plan, embeds[:, :1])  # another shape
        assert not graph.matches(plan, embeds.to(torch.bfloat16))  # another dtype
        assert not graph.matches(plan, embeds.to("meta"))  # another device
        assert not graph.matches(plan, embeds.transpose(1, 2).contiguous().transpose(1, 2))  # not contiguous
        graph.release()
        assert not graph.matches(plan, embeds)

    def test_settings_the_loop_does_not_cover_go_to_generate(self, graphs, loop_calls):
        predictor, _, hf_generate = _captured()
        embeds = _embeds(1, torch.float32)
        torch.manual_seed(5)
        expected = hf_generate(inputs_embeds=embeds, **_TALKER_ARGUMENTS, repetition_penalty=1.05).sequences
        torch.manual_seed(5)
        result = _talker_call(predictor, embeds, repetition_penalty=1.05)
        assert torch.equal(result.sequences, expected)
        assert loop_calls == []
        assert graphs[0].replays == 0

    def test_without_a_graph_the_loop_runs(self, graphs, loop_calls):
        predictor = _predictor()
        qwen_code_predictor.install(_qwen_model(predictor))
        _talker_call(predictor, _embeds(1, torch.float32))
        assert loop_calls == [1]
        assert graphs == []

    def test_concurrent_callers_get_their_own_codes(self, graphs):
        """``/generate/stream`` skips the GPU queue: two threads may step at once and share the one graph."""
        predictor, _, hf_generate = _captured(settings={"subtalker_dosample": False})  # greedy: no RNG to share
        arguments = {**_TALKER_ARGUMENTS, "do_sample": False}
        inputs = [_embeds(1, torch.float32, seed=seed) for seed in range(4)]
        expected = [hf_generate(inputs_embeds=e, **arguments).sequences for e in inputs]
        errors: list[str] = []

        def worker(index: int) -> None:
            for _ in range(5):
                codes = predictor.generate(inputs_embeds=inputs[index], **arguments).sequences
                if not torch.equal(codes, expected[index]):
                    errors.append(f"thread {index} got another input's codes")

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(len(inputs))]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        assert errors == []
        assert graphs[0].replays == 5 * len(inputs)


class TestCapture:
    def test_captures_the_talker_plan_once(self, graphs):
        predictor = _predictor()
        model = _qwen_model(predictor)
        qwen_code_predictor.install(model)
        assert qwen_code_predictor.capture_graph(model) is True
        graph = predictor.generate.frame_graph
        assert qwen_code_predictor.capture_graph(model) is True
        assert predictor.generate.frame_graph is graph
        assert len(graphs) == 1
        assert graph.plan is predictor.generate.plan_for((), {"inputs_embeds": graph.embeds, **_TALKER_ARGUMENTS})
        assert tuple(graph.embeds.shape) == (1, 2, _TALKER_HIDDEN)
        assert graph.embeds.dtype == predictor.dtype
        assert not graph.embeds.is_inference()

    def test_talker_arguments_are_the_talker_call(self):
        assert qwen_code_predictor.talker_arguments(_qwen_model(_predictor())) == _TALKER_ARGUMENTS

    def test_logs_the_capture(self, graphs, caplog):
        predictor = _predictor()
        model = _qwen_model(predictor)
        qwen_code_predictor.install(model)
        with caplog.at_level(logging.INFO, logger=qwen_code_predictor.__name__):
            qwen_code_predictor.capture_graph(model)
        assert "captured as a HIP graph" in caplog.text
        assert qwen_code_predictor.CODE_PREDICTOR_GRAPH_ENV in caplog.text

    def test_capture_buffer_is_writable_under_inference_mode_and_without(self, graphs):
        predictor, _, _ = _captured()
        with torch.inference_mode():
            _talker_call(predictor, _embeds(1, torch.float32))
        with torch.no_grad():
            _talker_call(predictor, _embeds(1, torch.float32))
        assert graphs[0].replays == 2

    @pytest.mark.parametrize("kind", ["empty", "no code predictor", "loop not installed"])
    def test_without_the_loop_nothing_is_captured(self, graphs, kind, caplog):
        model = {
            "empty": lambda: types.SimpleNamespace(),
            "no code predictor": lambda: types.SimpleNamespace(model=types.SimpleNamespace(talker=None)),
            "loop not installed": lambda: _qwen_model(_predictor()),
        }[kind]()
        with caplog.at_level(logging.INFO, logger=qwen_code_predictor.__name__):
            assert qwen_code_predictor.capture_graph(model) is False
        assert "no graph to capture" in caplog.text
        assert graphs == []

    def test_talker_settings_the_loop_does_not_cover_capture_nothing(self, graphs):
        predictor = _predictor()
        predictor.generation_config.repetition_penalty = 1.05
        model = _qwen_model(predictor)
        qwen_code_predictor.install(model)
        assert qwen_code_predictor.capture_graph(model) is False
        assert predictor.generate.frame_graph is None
        assert graphs == []


class TestFailures:
    def test_failed_capture_keeps_the_loop(self, monkeypatch, loop_calls, caplog):
        def failing(_fn, _device):
            raise RuntimeError("operation not permitted when stream is capturing")

        monkeypatch.setattr(qwen_code_predictor, "_capture", failing)
        predictor = _predictor()
        model = _qwen_model(predictor)
        qwen_code_predictor.install(model)
        with caplog.at_level(logging.WARNING, logger=qwen_code_predictor.__name__):
            assert qwen_code_predictor.capture_graph(model) is False
        assert "keeps its loop" in caplog.text
        assert predictor.generate.frame_graph is None
        _talker_call(predictor, _embeds(1, torch.float32))
        assert loop_calls == [1]

    def test_failed_replay_falls_back_to_the_loop_for_good(self, graphs, loop_calls, caplog):
        predictor, _, hf_generate = _captured()
        graph = predictor.generate.frame_graph
        graphs[0].fail_next = True
        embeds = _embeds(1, torch.float32)
        torch.manual_seed(3)
        expected = hf_generate(inputs_embeds=embeds, **_TALKER_ARGUMENTS).sequences

        torch.manual_seed(3)
        with caplog.at_level(logging.WARNING, logger=qwen_code_predictor.__name__):
            result = _talker_call(predictor, embeds)
        assert torch.equal(result.sequences, expected)  # the failed frame comes from the loop
        assert "graph replay failed" in caplog.text
        assert predictor.generate.frame_graph is None
        assert graphs[0].resets == 1
        assert graph.embeds is None

        _talker_call(predictor, embeds)
        assert loop_calls == [1, 1]
        assert graphs[0].replays == 0

    def test_replay_after_release_returns_nothing(self, graphs):
        predictor, _, _ = _captured()
        graph = predictor.generate.frame_graph
        graph.release()
        assert graph.replay(_embeds(1, torch.float32)) is None


class TestRelease:
    def test_frees_graph_and_buffers(self, graphs, loop_calls):
        predictor, model, _ = _captured()
        graph = predictor.generate.frame_graph
        qwen_code_predictor.release_graph(model)
        assert predictor.generate.frame_graph is None
        assert graphs[0].resets == 1
        assert (graph.graph, graph.sequences, graph.embeds) == (None, None, None)

        qwen_code_predictor.release_graph(model)  # idempotent
        graph.release()
        assert graphs[0].resets == 1

        _talker_call(predictor, _embeds(1, torch.float32))
        assert loop_calls == [1]

    @pytest.mark.parametrize("installed", [False, True])
    def test_models_without_a_graph_are_left_alone(self, installed):
        qwen_code_predictor.release_graph(None)
        qwen_code_predictor.release_graph(types.SimpleNamespace())
        predictor = _predictor()
        model = _qwen_model(predictor)
        if installed:
            qwen_code_predictor.install(model)
        qwen_code_predictor.release_graph(model)
        assert getattr(predictor.generate, "frame_graph", None) is None

    def test_blas_workspaces_go_with_the_last_graph(self, graphs, monkeypatch):
        """The graphs' kernels write to the BLAS workspace of their capture stream: dropped only when none is left."""
        monkeypatch.setattr(qwen_code_predictor, "_live_graphs", set())
        dropped: list[int] = []
        monkeypatch.setattr(qwen_code_predictor, "_drop_blas_workspaces", lambda: dropped.append(1))
        monkeypatch.setattr(torch.cuda, "synchronize", lambda _device=None: None)
        first, second = _captured()[1], _captured()[1]
        for model in (first, second):
            model.model.talker.code_predictor.generate.frame_graph._device = types.SimpleNamespace(type="cuda")
        assert len(qwen_code_predictor._live_graphs) == 2

        qwen_code_predictor.release_graph(first)
        assert dropped == []
        qwen_code_predictor.release_graph(second)
        assert dropped == [1]
        assert qwen_code_predictor._live_graphs == set()

    def test_cpu_graphs_drop_no_workspace(self, graphs, monkeypatch):
        monkeypatch.setattr(qwen_code_predictor, "_live_graphs", set())
        dropped: list[int] = []
        monkeypatch.setattr(qwen_code_predictor, "_drop_blas_workspaces", lambda: dropped.append(1))
        _, model, _ = _captured()
        qwen_code_predictor.release_graph(model)
        assert dropped == []
        assert qwen_code_predictor._live_graphs == set()

    def test_failed_capture_leaves_no_live_graph(self, monkeypatch):
        monkeypatch.setattr(qwen_code_predictor, "_live_graphs", set())

        def failing(_fn, _device):
            raise RuntimeError("hipErrorStreamCaptureInvalidated")

        monkeypatch.setattr(qwen_code_predictor, "_capture", failing)
        model = _qwen_model(_predictor())
        qwen_code_predictor.install(model)
        assert qwen_code_predictor.capture_graph(model) is False
        assert qwen_code_predictor._live_graphs == set()

    def test_failed_reset_keeps_the_workspaces(self, graphs, monkeypatch):
        """A graph that may still exist keeps the workspace its kernels write to."""
        monkeypatch.setattr(qwen_code_predictor, "_live_graphs", set())
        dropped: list[int] = []
        monkeypatch.setattr(qwen_code_predictor, "_drop_blas_workspaces", lambda: dropped.append(1))
        monkeypatch.setattr(torch.cuda, "synchronize", lambda _device=None: None)
        predictor, model, _ = _captured()
        predictor.generate.frame_graph._device = types.SimpleNamespace(type="cuda")

        def failing_reset():
            raise RuntimeError("hipErrorIllegalAddress")

        graphs[0].reset = failing_reset
        qwen_code_predictor.release_graph(model)
        assert dropped == []
        assert len(qwen_code_predictor._live_graphs) == 1

    def test_release_failure_is_logged(self, graphs, caplog):
        predictor, model, _ = _captured()

        def failing_reset():
            raise RuntimeError("hipErrorIllegalAddress")

        graphs[0].reset = failing_reset
        with caplog.at_level(logging.WARNING, logger=qwen_code_predictor.__name__):
            qwen_code_predictor.release_graph(model)
        assert "releasing the code predictor graph failed" in caplog.text
        assert predictor.generate.frame_graph is None


class TestEnabled:
    """The decision: (device, variant, hip, environment) -> capture the graph or not."""

    @pytest.mark.parametrize(("variant", "hip"), [("rocm", None), (None, _HIP), ("rocm", _HIP)])
    def test_rocm_gpu_enables_it_by_default(self, variant, hip):
        assert qwen_code_predictor.graph_enabled("cuda", variant, hip, environ={}) is True

    @pytest.mark.parametrize(("variant", "hip"), [(None, None), ("cpu", None), ("cuda", None), (None, "")])
    def test_non_rocm_builds_are_untouched(self, variant, hip):
        assert qwen_code_predictor.graph_enabled("cuda", variant, hip, environ={}) is False

    @pytest.mark.parametrize("device", ["cpu", "xpu", "mps"])
    def test_non_gpu_devices_are_untouched(self, device):
        assert qwen_code_predictor.graph_enabled(device, "rocm", _HIP, environ={}) is False

    @pytest.mark.parametrize("value", ["0", "false", " OFF ", "no"])
    def test_variable_disables_it(self, value):
        environ = {qwen_code_predictor.CODE_PREDICTOR_GRAPH_ENV: value}
        assert qwen_code_predictor.graph_enabled("cuda", "rocm", _HIP, environ=environ) is False

    def test_no_loop_no_graph(self):
        environ = {qwen_code_predictor.CODE_PREDICTOR_LOOP_ENV: "0", qwen_code_predictor.CODE_PREDICTOR_GRAPH_ENV: "1"}
        assert qwen_code_predictor.graph_enabled("cuda", "rocm", _HIP, environ=environ) is False

    def test_reads_process_environment_by_default(self, monkeypatch):
        monkeypatch.delenv(qwen_code_predictor.CODE_PREDICTOR_LOOP_ENV, raising=False)
        monkeypatch.setenv(qwen_code_predictor.CODE_PREDICTOR_GRAPH_ENV, "0")
        assert qwen_code_predictor.graph_enabled("cuda", "rocm", _HIP) is False
        monkeypatch.delenv(qwen_code_predictor.CODE_PREDICTOR_GRAPH_ENV)
        assert qwen_code_predictor.graph_enabled("cuda", "rocm", _HIP) is True


class TestBackend:
    """PyTorchTTSBackend captures after installing the loop and before the warm-up, and frees the graph on unload."""

    @pytest.fixture
    def backend(self, monkeypatch, graphs):
        monkeypatch.setattr(pytorch_backend, "get_torch_device", lambda **_: "cuda")
        monkeypatch.setattr(pytorch_backend, "model_load_progress", lambda *_: contextlib.nullcontext())
        monkeypatch.setattr(pytorch_backend, "empty_device_cache", lambda _device: None)
        for name in (
            perf_log.PERF_LOG_ENV,
            qwen_code_predictor.CODE_PREDICTOR_LOOP_ENV,
            qwen_code_predictor.CODE_PREDICTOR_GRAPH_ENV,
            qwen_warmup.WARMUP_ENV,
            "VOICEBOX_BACKEND_VARIANT",
        ):
            monkeypatch.delenv(name, raising=False)

        events: list[str] = []

        def warm_up(model):
            graph = getattr(model.model.talker.code_predictor.generate, "frame_graph", None)
            events.append("warm_up with graph" if graph is not None else "warm_up")
            return {}

        monkeypatch.setattr(qwen_warmup, "warm_up", warm_up)
        predictor = _predictor()
        fake_qwen_tts = types.ModuleType("qwen_tts")
        fake_qwen_tts.Qwen3TTSModel = types.SimpleNamespace(from_pretrained=lambda *_a, **_k: _qwen_model(predictor))
        monkeypatch.setitem(sys.modules, "qwen_tts", fake_qwen_tts)

        def _load(device: str = "cuda") -> tuple[pytorch_backend.PyTorchTTSBackend, list[str]]:
            loaded = pytorch_backend.PyTorchTTSBackend()
            loaded.device = device
            loaded._is_model_cached = lambda _size: True
            loaded._load_model_sync("1.7B")
            return loaded, events

        return _load

    def test_rocm_dev_mode_captures_before_the_warm_up(self, backend, monkeypatch, graphs):
        monkeypatch.setattr(torch.version, "hip", _HIP)
        loaded, events = backend()
        assert loaded.model.model.talker.code_predictor.generate.frame_graph is not None
        assert len(graphs) == 1
        assert events == ["warm_up with graph"]

    def test_packaged_rocm_binary_captures(self, backend, monkeypatch, graphs):
        monkeypatch.setattr(torch.version, "hip", None)
        monkeypatch.setenv("VOICEBOX_BACKEND_VARIANT", "rocm")
        backend()
        assert len(graphs) == 1

    @pytest.mark.parametrize(
        "variable", [qwen_code_predictor.CODE_PREDICTOR_GRAPH_ENV, qwen_code_predictor.CODE_PREDICTOR_LOOP_ENV]
    )
    def test_variables_turn_it_off(self, backend, monkeypatch, graphs, variable):
        monkeypatch.setattr(torch.version, "hip", _HIP)
        monkeypatch.setenv(variable, "0")
        _, events = backend()
        assert graphs == []
        assert events == ["warm_up"]

    @pytest.mark.parametrize(
        ("hip", "variant", "device"), [(None, None, "cuda"), (None, "cuda", "cuda"), (_HIP, None, "cpu")]
    )
    def test_other_platforms_are_untouched(self, backend, monkeypatch, graphs, hip, variant, device):
        monkeypatch.setattr(torch.version, "hip", hip)
        if variant is not None:
            monkeypatch.setenv("VOICEBOX_BACKEND_VARIANT", variant)
        loaded, _ = backend(device)
        assert graphs == []
        assert not hasattr(loaded.model.model.talker.code_predictor.generate, "frame_graph")

    def test_unload_frees_the_graph(self, backend, monkeypatch, graphs):
        monkeypatch.setattr(torch.version, "hip", _HIP)
        loaded, _ = backend()
        graph = loaded.model.model.talker.code_predictor.generate.frame_graph
        loaded.unload_model()
        assert loaded.model is None
        assert graphs[0].resets == 1
        assert graph.embeds is None

    def test_unload_without_a_graph(self, backend, monkeypatch, graphs):
        monkeypatch.setattr(torch.version, "hip", None)
        loaded, _ = backend()
        loaded.unload_model()
        assert loaded.model is None

    @pytest.mark.asyncio
    async def test_the_apps_unload_waits_for_the_gpu_turn(self, backend, monkeypatch, graphs):
        """An unload from the app (unload_backend) releases the graph only after the queued generation running ends."""
        monkeypatch.setattr(torch.version, "hip", _HIP)
        monkeypatch.setattr(task_queue, "_gpu_lock", None)
        loaded, _ = backend()

        generation_turn = task_queue.gpu_lock()
        await generation_turn.acquire()
        unload = asyncio.create_task(backends.unload_backend(loaded))
        await asyncio.sleep(0.01)
        assert not unload.done()
        assert loaded.model is not None
        assert graphs[0].resets == 0

        generation_turn.release()
        await asyncio.wait_for(unload, timeout=5)
        assert loaded.model is None
        assert graphs[0].resets == 1
