"""
Tests for backend/utils/qwen_code_predictor.py, the Qwen3-TTS code predictor's own sampling loop in place of HF
``generate`` (perf-loop plan, Fase 3, task 3.2), and for how PyTorchTTSBackend installs it.

Runs without a GPU: a tiny code predictor with random weights on the CPU (the real qwen_tts classes) is sampled by HF
``generate`` and by the loop from the same seed, with the arguments the talker passes on every step.

Usage:
    python -m pytest backend/tests/test_qwen_code_predictor.py -v
"""

import contextlib
import logging
import sys
import types
from typing import Any

import pytest
import torch

from backend.backends import pytorch_backend
from backend.utils import perf_log, qwen_code_predictor, qwen_warmup

_HIP = "7.2.53211-158bd99533"
_TALKER_HIDDEN = 48
_VOCAB = 64
_CODE_GROUPS = 16
# What Qwen3TTSTalkerForConditionalGeneration.forward passes on every step, with the subtalker defaults of the 1.7B
# Base model's generation_config.json.
_TALKER_ARGUMENTS = {
    "max_new_tokens": _CODE_GROUPS - 1,
    "do_sample": True,
    "top_p": 1.0,
    "top_k": 50,
    "temperature": 0.9,
    "output_hidden_states": True,
    "return_dict_in_generate": True,
}


def _predictor(attn: str = "sdpa", dtype: torch.dtype = torch.float32) -> Any:
    """A 2-layer ``Qwen3TTSTalkerCodePredictorModelForConditionalGeneration`` with random weights, in eval mode."""
    configs = pytest.importorskip("qwen_tts.core.models.configuration_qwen3_tts")
    modeling = pytest.importorskip("qwen_tts.core.models.modeling_qwen3_tts")
    predictor_config = configs.Qwen3TTSTalkerCodePredictorConfig(
        vocab_size=_VOCAB,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        num_code_groups=_CODE_GROUPS,
        initializer_range=0.5,  # spread the logits, so temperature, top-k and top-p change the distribution
        attn_implementation=attn,
    )
    talker_config = configs.Qwen3TTSTalkerConfig(
        hidden_size=_TALKER_HIDDEN,
        num_code_groups=_CODE_GROUPS,
        code_predictor_config=predictor_config.to_dict(),
    )
    torch.manual_seed(0)
    predictor = modeling.Qwen3TTSTalkerCodePredictorModelForConditionalGeneration(predictor_config, talker_config)
    return predictor.to(dtype).eval()


def _model(predictor: Any) -> Any:
    """``Qwen3TTSModel`` layout around ``predictor``: ``model.model.talker.code_predictor``."""
    return types.SimpleNamespace(model=types.SimpleNamespace(talker=types.SimpleNamespace(code_predictor=predictor)))


def _embeds(batch: int, dtype: torch.dtype, seed: int = 123) -> torch.Tensor:
    """The talker's ``torch.cat((past_hidden, last_id_hidden), dim=1)``: two positions of the talker's width."""
    generator = torch.Generator().manual_seed(seed)
    return torch.randn(batch, 2, _TALKER_HIDDEN, generator=generator).to(dtype)


def _sample(generate, seed: int, embeds: torch.Tensor, arguments: dict[str, Any]) -> tuple[Any, torch.Tensor]:
    """``generate(inputs_embeds=embeds, **arguments)`` from ``seed``; returns the result and the RNG state after it."""
    torch.manual_seed(seed)
    result = generate(inputs_embeds=embeds, **arguments)
    return result, torch.get_rng_state()


def _sequences(result: Any) -> torch.Tensor:
    return result if isinstance(result, torch.Tensor) else result.sequences


def _host_reads(run) -> int:
    """GPU-to-host reads (``aten::_local_scalar_dense``) ``run`` makes, as the CPU profiler records them.

    Reads under ``aten::_assert_async`` are left out: ``torch.multinomial`` checks its input with it, which on the CPU
    reads the value but on a GPU is a device-side assert with no read (perf-loop/notes/3.1-profile.md).
    """
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as profiler:
        run()
    reads = 0
    for event in profiler.events():
        if event.name != "aten::_local_scalar_dense":
            continue
        parent = event.cpu_parent
        while parent is not None and parent.name != "aten::_assert_async":
            parent = parent.cpu_parent
        reads += parent is None
    return reads


@pytest.fixture
def loop_calls(monkeypatch) -> list[int]:
    """Counts the calls that went through :func:`qwen_code_predictor.sample_with_loop`."""
    calls: list[int] = []
    real = qwen_code_predictor.sample_with_loop

    def counting(*args, **kwargs):
        calls.append(1)
        return real(*args, **kwargs)

    monkeypatch.setattr(qwen_code_predictor, "sample_with_loop", counting)
    return calls


class TestSameTokensAsGenerate:
    """From the same seed the loop samples the tokens HF generate samples, and leaves the RNG where generate does."""

    @pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
    @pytest.mark.parametrize("attn", ["sdpa", "eager"])
    def test_with_the_talker_arguments(self, loop_calls, attn, dtype):
        predictor = _predictor(attn, dtype)
        hf_generate = predictor.generate
        assert qwen_code_predictor.install(_model(predictor)) is True

        for batch in (1, 2):
            embeds = _embeds(batch, dtype)
            for seed in (0, 1, 1234):
                expected, expected_rng = _sample(hf_generate, seed, embeds, _TALKER_ARGUMENTS)
                result, rng = _sample(predictor.generate, seed, embeds, _TALKER_ARGUMENTS)

                assert result.sequences.shape == (batch, _CODE_GROUPS - 1)
                assert result.sequences.dtype == expected.sequences.dtype == torch.long
                assert torch.equal(result.sequences, expected.sequences)
                assert torch.equal(rng, expected_rng)
        assert len(loop_calls) == 6

    def test_tokens_differ_between_seeds(self):
        """Guards the test above: equal tokens are not just what this tiny model always samples."""
        predictor = _predictor()
        qwen_code_predictor.install(_model(predictor))
        embeds = _embeds(1, torch.float32)
        tokens = {
            tuple(_sample(predictor.generate, seed, embeds, _TALKER_ARGUMENTS)[0].sequences[0].tolist())
            for seed in range(4)
        }
        assert len(tokens) == 4

    @pytest.mark.parametrize(
        "settings",
        [
            {"top_p": 0.8},
            {"top_k": 5, "top_p": 0.9, "temperature": 0.7},
            {"temperature": 1.0, "top_k": 0},
            {"temperature": 1.3, "top_k": None, "top_p": None},
            {"do_sample": False},
            {"max_new_tokens": 3},
            {"return_dict_in_generate": False},
            {"output_hidden_states": False},
        ],
    )
    def test_with_other_sampling_settings(self, loop_calls, settings):
        predictor = _predictor()
        hf_generate = predictor.generate
        qwen_code_predictor.install(_model(predictor))
        arguments = {**_TALKER_ARGUMENTS, **settings}
        embeds = _embeds(2, torch.float32)

        expected, expected_rng = _sample(hf_generate, 7, embeds, arguments)
        result, rng = _sample(predictor.generate, 7, embeds, arguments)

        assert type(result) is type(expected)
        assert torch.equal(_sequences(result), _sequences(expected))
        assert _sequences(result).shape == (2, arguments["max_new_tokens"])
        assert torch.equal(rng, expected_rng)
        assert loop_calls == [1]

    def test_warpers_follow_generate(self):
        logits_process = pytest.importorskip("transformers.generation.logits_process")
        predictor = _predictor()

        plan = qwen_code_predictor._sampling_plan(predictor, _TALKER_ARGUMENTS)
        assert [type(warper) for warper in plan.warpers] == [
            logits_process.TemperatureLogitsWarper,
            logits_process.TopKLogitsWarper,
        ]
        assert plan.warpers[0].temperature == 0.9
        assert plan.warpers[1].top_k == 50
        assert (plan.max_new_tokens, plan.do_sample) == (_CODE_GROUPS - 1, True)

        greedy = qwen_code_predictor._sampling_plan(predictor, {**_TALKER_ARGUMENTS, "do_sample": False})
        assert greedy.warpers == ()
        assert greedy.do_sample is False

    def test_hidden_states_match(self):
        predictor = _predictor()
        hf_generate = predictor.generate
        qwen_code_predictor.install(_model(predictor))
        embeds = _embeds(1, torch.float32)

        expected, _ = _sample(hf_generate, 3, embeds, _TALKER_ARGUMENTS)
        result, _ = _sample(predictor.generate, 3, embeds, _TALKER_ARGUMENTS)

        assert len(result.hidden_states) == len(expected.hidden_states) == _CODE_GROUPS - 1
        for step, expected_step in zip(result.hidden_states, expected.hidden_states, strict=True):
            assert len(step) == len(expected_step)
            for state, expected_state in zip(step, expected_step, strict=True):
                assert torch.equal(state, expected_state)
        assert result.past_key_values.get_seq_length() == expected.past_key_values.get_seq_length()


class TestNoHostReads:
    def test_loop_reads_nothing_back_and_generate_does(self):
        predictor = _predictor()
        hf_generate = predictor.generate
        qwen_code_predictor.install(_model(predictor))
        embeds = _embeds(1, torch.float32)

        hf_reads = _host_reads(lambda: hf_generate(inputs_embeds=embeds, **_TALKER_ARGUMENTS))
        loop_reads = _host_reads(lambda: predictor.generate(inputs_embeds=embeds, **_TALKER_ARGUMENTS))

        # generate: the 2D mask's .all() and the unfinished-sequences check, once per token
        assert hf_reads >= 2 * (_CODE_GROUPS - 1)
        assert loop_reads == 0


class RecordingGenerate:
    """Stands in for HF generate: records its calls and returns a marker."""

    def __init__(self):
        self.calls: list[tuple[tuple, dict]] = []

    def __call__(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        return "hf"


class TestFallsBackToGenerate:
    """Calls the loop does not reproduce go to the original generate, untouched."""

    @pytest.fixture
    def predictor(self):
        predictor = _predictor()
        predictor.generate = RecordingGenerate()
        qwen_code_predictor.install(_model(predictor))
        return predictor

    @pytest.mark.parametrize(
        "extra",
        [
            {"repetition_penalty": 1.05},
            {"attention_mask": torch.ones(1, 2, dtype=torch.long)},
            {"eos_token_id": 3},
            {"min_p": 0.1},
            {"num_beams": 2},
            {"temperature": torch.tensor(0.9)},
        ],
    )
    def test_settings_it_does_not_cover(self, predictor, loop_calls, extra):
        embeds = _embeds(1, torch.float32)
        assert predictor.generate(inputs_embeds=embeds, **{**_TALKER_ARGUMENTS, **extra}) == "hf"
        ((args, kwargs),) = predictor.generate.original.calls
        assert args == ()
        assert kwargs["inputs_embeds"] is embeds
        assert loop_calls == []

    @pytest.mark.parametrize(
        ("field", "value"),
        [
            ("eos_token_id", 3),
            ("repetition_penalty", 1.05),
            ("suppress_tokens", [1]),
            ("cache_implementation", "static"),
        ],
    )
    def test_model_generation_config_it_does_not_cover(self, predictor, loop_calls, field, value):
        setattr(predictor.generation_config, field, value)
        assert predictor.generate(inputs_embeds=_embeds(1, torch.float32), **_TALKER_ARGUMENTS) == "hf"
        assert loop_calls == []

    def test_inputs_it_does_not_cover(self, predictor, loop_calls):
        embeds = _embeds(1, torch.float32)
        calls = [
            ((), {"inputs_embeds": embeds[:, :1], **_TALKER_ARGUMENTS}),
            ((), {"inputs_embeds": embeds[0], **_TALKER_ARGUMENTS}),
            ((), {**_TALKER_ARGUMENTS}),
            ((embeds,), {**_TALKER_ARGUMENTS}),
            ((), {"inputs_embeds": embeds, **_TALKER_ARGUMENTS, "max_new_tokens": 0}),
        ]
        for args, kwargs in calls:
            assert predictor.generate(*args, **kwargs) == "hf"
        assert len(predictor.generate.original.calls) == len(calls)
        assert loop_calls == []

    def test_invalid_settings_reach_generate(self, predictor, loop_calls):
        """generate reports invalid settings itself (here, a negative temperature)."""
        assert (
            predictor.generate(inputs_embeds=_embeds(1, torch.float32), **{**_TALKER_ARGUMENTS, "temperature": -1.0})
            == "hf"
        )
        assert loop_calls == []

    def test_logs_the_first_fallback_only(self, predictor, caplog):
        embeds = _embeds(1, torch.float32)
        with caplog.at_level(logging.INFO, logger=qwen_code_predictor.__name__):
            for _ in range(3):
                predictor.generate(inputs_embeds=embeds, **_TALKER_ARGUMENTS, repetition_penalty=1.05)
        assert caplog.text.count("using generate") == 1


class TestInstall:
    def test_plan_is_built_once_per_set_of_arguments(self, monkeypatch):
        predictor = _predictor()
        qwen_code_predictor.install(_model(predictor))
        built: list[dict] = []
        real = qwen_code_predictor._sampling_plan

        def counting(model, arguments):
            built.append(dict(arguments))
            return real(model, arguments)

        monkeypatch.setattr(qwen_code_predictor, "_sampling_plan", counting)
        embeds = _embeds(1, torch.float32)
        for _ in range(3):
            predictor.generate(inputs_embeds=embeds, **_TALKER_ARGUMENTS)
        predictor.generate(inputs_embeds=embeds, **{**_TALKER_ARGUMENTS, "top_k": 10})
        assert len(built) == 2

    def test_keeps_the_original_and_installs_once(self, caplog):
        predictor = _predictor()
        hf_generate = predictor.generate
        model = _model(predictor)
        with caplog.at_level(logging.INFO, logger=qwen_code_predictor.__name__):
            assert qwen_code_predictor.install(model) is True
            installed = predictor.generate
            assert qwen_code_predictor.install(model) is True
        assert predictor.generate is installed
        assert installed.original == hf_generate
        assert caplog.text.count("samples in its own loop") == 1
        assert qwen_code_predictor.CODE_PREDICTOR_LOOP_ENV in caplog.text

    @pytest.mark.parametrize(
        "model",
        [
            types.SimpleNamespace(),
            types.SimpleNamespace(model=types.SimpleNamespace()),
            types.SimpleNamespace(model=types.SimpleNamespace(talker=types.SimpleNamespace())),
            _model(types.SimpleNamespace()),
        ],
    )
    def test_models_without_code_predictor_are_left_alone(self, model, caplog):
        with caplog.at_level(logging.INFO, logger=qwen_code_predictor.__name__):
            assert qwen_code_predictor.install(model) is False
        assert "sampling loop not installed" in caplog.text


class TestEnabled:
    """The decision: (device, variant, hip, environment) -> install the loop or not."""

    @pytest.mark.parametrize(("variant", "hip"), [("rocm", None), (None, _HIP), ("rocm", _HIP), ("cpu", _HIP)])
    def test_rocm_gpu_enables_it_by_default(self, variant, hip):
        assert qwen_code_predictor.loop_enabled("cuda", variant, hip, environ={}) is True

    @pytest.mark.parametrize(("variant", "hip"), [(None, None), ("cpu", None), ("cuda", None), (None, "")])
    def test_non_rocm_builds_are_untouched(self, variant, hip):
        assert qwen_code_predictor.loop_enabled("cuda", variant, hip, environ={}) is False

    @pytest.mark.parametrize("device", ["cpu", "xpu", "mps", object()])
    def test_non_gpu_or_other_devices_are_untouched(self, device):
        assert qwen_code_predictor.loop_enabled(device, "rocm", _HIP, environ={}) is False

    @pytest.mark.parametrize("value", ["0", "false", " OFF ", "no"])
    def test_variable_disables_it(self, value):
        environ = {qwen_code_predictor.CODE_PREDICTOR_LOOP_ENV: value}
        assert qwen_code_predictor.loop_enabled("cuda", "rocm", _HIP, environ=environ) is False

    @pytest.mark.parametrize("value", ["1", "true", ""])
    def test_other_values_keep_it_on(self, value):
        environ = {qwen_code_predictor.CODE_PREDICTOR_LOOP_ENV: value}
        assert qwen_code_predictor.loop_enabled("cuda", None, _HIP, environ=environ) is True

    def test_reads_process_environment_by_default(self, monkeypatch):
        monkeypatch.setenv(qwen_code_predictor.CODE_PREDICTOR_LOOP_ENV, "0")
        assert qwen_code_predictor.loop_enabled("cuda", "rocm", _HIP) is False
        monkeypatch.delenv(qwen_code_predictor.CODE_PREDICTOR_LOOP_ENV)
        assert qwen_code_predictor.loop_enabled("cuda", "rocm", _HIP) is True


class FakePredictor:
    """Code predictor stand-in: only ``generate``, which records its calls."""

    def __init__(self):
        self.calls: list[dict] = []

    def generate(self, **kwargs):
        self.calls.append(kwargs)
        return "hf"


class TestBackendLoad:
    """PyTorchTTSBackend._load_model_sync installs the loop only on a ROCm GPU, before the warm-up."""

    @pytest.fixture
    def load(self, monkeypatch):
        monkeypatch.setattr(pytorch_backend, "get_torch_device", lambda **_: "cuda")
        monkeypatch.setattr(pytorch_backend, "model_load_progress", lambda *_: contextlib.nullcontext())
        monkeypatch.delenv(perf_log.PERF_LOG_ENV, raising=False)
        monkeypatch.delenv(qwen_code_predictor.CODE_PREDICTOR_LOOP_ENV, raising=False)
        monkeypatch.delenv(qwen_warmup.WARMUP_ENV, raising=False)
        monkeypatch.delenv("VOICEBOX_BACKEND_VARIANT", raising=False)

        warmed_up_with_loop: list[bool] = []

        def warm_up(model):
            generate = model.model.talker.code_predictor.generate
            warmed_up_with_loop.append(getattr(generate, "original", None) is not None)
            return {}

        monkeypatch.setattr(qwen_warmup, "warm_up", warm_up)

        def from_pretrained(*_args, **_kwargs):
            return _model(FakePredictor())

        fake_qwen_tts = types.ModuleType("qwen_tts")
        fake_qwen_tts.Qwen3TTSModel = types.SimpleNamespace(from_pretrained=from_pretrained)
        monkeypatch.setitem(sys.modules, "qwen_tts", fake_qwen_tts)

        def _load(device: str = "cuda") -> tuple[Any, list[bool]]:
            backend = pytorch_backend.PyTorchTTSBackend()
            backend.device = device
            backend._is_model_cached = lambda _size: True
            backend._load_model_sync("1.7B")
            return backend.model.model.talker.code_predictor, warmed_up_with_loop

        return _load

    def test_rocm_dev_mode_installs_it_before_the_warm_up(self, load, monkeypatch):
        monkeypatch.setattr(torch.version, "hip", _HIP)
        predictor, warmed_up_with_loop = load()
        assert getattr(predictor.generate, "original", None) is not None
        assert warmed_up_with_loop == [True]

    def test_packaged_rocm_binary_installs_it(self, load, monkeypatch):
        monkeypatch.setattr(torch.version, "hip", None)
        monkeypatch.setenv("VOICEBOX_BACKEND_VARIANT", "rocm")
        predictor, _ = load()
        assert getattr(predictor.generate, "original", None) is not None

    @pytest.mark.parametrize(
        ("hip", "variant", "device"), [(None, None, "cuda"), (None, "cuda", "cuda"), (_HIP, None, "cpu")]
    )
    def test_other_platforms_are_untouched(self, load, monkeypatch, hip, variant, device):
        monkeypatch.setattr(torch.version, "hip", hip)
        if variant is not None:
            monkeypatch.setenv("VOICEBOX_BACKEND_VARIANT", variant)
        predictor, _ = load(device)
        assert getattr(predictor.generate, "original", None) is None
        assert predictor.generate(inputs_embeds=None) == "hf"

    def test_variable_turns_it_off(self, load, monkeypatch):
        monkeypatch.setattr(torch.version, "hip", _HIP)
        monkeypatch.setenv(qwen_code_predictor.CODE_PREDICTOR_LOOP_ENV, "0")
        predictor, warmed_up_with_loop = load()
        assert getattr(predictor.generate, "original", None) is None
        assert warmed_up_with_loop == [False]
