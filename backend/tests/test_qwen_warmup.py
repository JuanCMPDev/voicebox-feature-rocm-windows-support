"""
Tests for backend/utils/qwen_warmup.py, the ROCm warm-up of the Qwen3-TTS model at load time (perf-loop plan, Fase 1,
task 1.4), and for how PyTorchTTSBackend runs it.

Runs without a GPU: the model is a stand-in with the attribute layout of ``Qwen3TTSModel`` that records what the
warm-up asks of it, and ``torch.backends.miopen`` is a stub that records the ``immediate`` flag. One test drives the
real ``Qwen3TTSModel.generate_voice_clone`` (no weights) to check the dummy prompt fits it.

Usage:
    python -m pytest backend/tests/test_qwen_warmup.py -v
"""

import contextlib
import dataclasses
import logging
import sys
import types
from typing import Any

import numpy as np
import pytest
import torch

from backend.backends import pytorch_backend
from backend.utils import miopen_codec, perf_log, qwen_warmup

_HIP = "7.2.53211-158bd99533"
_QUANTIZERS = 16
_HIDDEN = 64
_EXPECTED_LENGTHS = [300, 32, 64, 96, 128, 160, 192, 224, 256, 288, 320]


class FakeMiopen:
    """Stands in for ``torch.backends.miopen``: records every write to ``immediate``."""

    def __init__(self):
        self._immediate = False
        self.writes: list[bool] = []

    @property
    def immediate(self) -> bool:
        return self._immediate

    @immediate.setter
    def immediate(self, value: bool) -> None:
        self.writes.append(value)
        self._immediate = value


@dataclasses.dataclass
class StubPromptItem:
    """Same fields as ``qwen_tts.VoiceClonePromptItem``."""

    ref_code: Any
    ref_spk_embedding: Any
    x_vector_only_mode: bool
    icl_mode: bool
    ref_text: str | None = None


class FakeTokenizer:
    """``Qwen3TTSTokenizer`` (12 Hz): ``decode`` records the codes and the MIOpen flag it ran with."""

    def __init__(self, miopen: FakeMiopen | None = None, model_type: str = qwen_warmup.CODEC_12HZ):
        self.miopen = miopen
        decoder_config = types.SimpleNamespace(num_quantizers=_QUANTIZERS)
        self.model = types.SimpleNamespace(
            config=types.SimpleNamespace(model_type=model_type, decoder_config=decoder_config),
            decoder=torch.nn.Identity(),
        )
        self.codes: list[torch.Tensor] = []
        self.flags: list[bool | None] = []
        self.error: Exception | None = None

    def decode(self, encoded):
        (item,) = encoded
        self.codes.append(item["audio_codes"])
        self.flags.append(None if self.miopen is None else self.miopen.immediate)
        self.model.decoder(torch.zeros(1))
        if self.error is not None:
            raise self.error
        return [np.zeros(int(item["audio_codes"].shape[0]) * 1920, dtype=np.float32)], 24000

    def lengths(self) -> list[int]:
        return [int(c.shape[0]) for c in self.codes]


class FakeInner:
    """``Qwen3TTSForConditionalGeneration``: talker config, ``generate`` (one frame per call) and the codec."""

    def __init__(self, tokenizer: FakeTokenizer, tts_model_type: str = "base"):
        self.speech_tokenizer = tokenizer
        self.tts_model_type = tts_model_type
        self.device = torch.device("cpu")
        self.talker = types.SimpleNamespace(
            config=types.SimpleNamespace(hidden_size=_HIDDEN, num_code_groups=_QUANTIZERS)
        )
        self.generate_calls: list[dict] = []

    def generate(self, **kwargs):
        self.generate_calls.append(kwargs)
        return [torch.ones(1, _QUANTIZERS, dtype=torch.long)], [torch.zeros(1, _HIDDEN)]

    def get_supported_languages(self):
        return ["auto", "spanish"]


class FakeQwenModel:
    """``Qwen3TTSModel``: ``generate_voice_clone`` runs the talker, then decodes reference + generated codes."""

    def __init__(self, miopen: FakeMiopen | None = None, tokenizer: FakeTokenizer | None = None):
        self.model = FakeInner(tokenizer or FakeTokenizer(miopen))
        self.device = torch.device("cpu")
        self.clone_calls: list[dict] = []

    def generate_voice_clone(self, text, language=None, voice_clone_prompt=None, **kwargs):
        self.clone_calls.append(
            {"text": text, "language": language, "voice_clone_prompt": voice_clone_prompt, **kwargs}
        )
        codes, _ = self.model.generate(voice_clone_prompt=voice_clone_prompt, **kwargs)
        ref_code = voice_clone_prompt[0].ref_code
        self.model.speech_tokenizer.decode([{"audio_codes": torch.cat([ref_code, codes[0]])}])
        return [np.zeros(1920, dtype=np.float32)], 24000

    def create_voice_clone_prompt(self, *args, **kwargs):
        return []


@pytest.fixture
def stub_qwen_tts(monkeypatch):
    """A ``qwen_tts`` module with ``VoiceClonePromptItem``; tests add ``Qwen3TTSModel`` when they need it."""
    module = types.ModuleType("qwen_tts")
    module.VoiceClonePromptItem = StubPromptItem
    monkeypatch.setitem(sys.modules, "qwen_tts", module)
    return module


class TestEnabled:
    """The decision: (device, variant, hip, environment) -> warm up at load or not."""

    @pytest.mark.parametrize(("variant", "hip"), [("rocm", None), (None, _HIP), ("rocm", _HIP), ("cpu", _HIP)])
    def test_rocm_gpu_enables_it_by_default(self, variant, hip):
        assert qwen_warmup.warmup_enabled("cuda", variant, hip, environ={}) is True

    @pytest.mark.parametrize(("variant", "hip"), [(None, None), ("cpu", None), ("cuda", None), (None, "")])
    def test_non_rocm_builds_are_untouched(self, variant, hip):
        assert qwen_warmup.warmup_enabled("cuda", variant, hip, environ={}) is False

    @pytest.mark.parametrize("device", ["cpu", "xpu", "mps", object()])
    def test_non_gpu_or_other_devices_are_untouched(self, device):
        assert qwen_warmup.warmup_enabled(device, "rocm", _HIP, environ={}) is False

    @pytest.mark.parametrize("value", ["0", "false", " OFF ", "no"])
    def test_variable_disables_it(self, value):
        environ = {qwen_warmup.WARMUP_ENV: value}
        assert qwen_warmup.warmup_enabled("cuda", "rocm", _HIP, environ=environ) is False

    @pytest.mark.parametrize("value", ["1", "true", ""])
    def test_other_values_keep_it_on(self, value):
        environ = {qwen_warmup.WARMUP_ENV: value}
        assert qwen_warmup.warmup_enabled("cuda", None, _HIP, environ=environ) is True

    def test_reads_process_environment_by_default(self, monkeypatch):
        monkeypatch.setenv(qwen_warmup.WARMUP_ENV, "0")
        assert qwen_warmup.warmup_enabled("cuda", "rocm", _HIP) is False
        monkeypatch.delenv(qwen_warmup.WARMUP_ENV)
        assert qwen_warmup.warmup_enabled("cuda", "rocm", _HIP) is True


class TestCodec:
    def test_lengths_are_the_plan_buckets(self):
        assert list(qwen_warmup.CODEC_WARMUP_STEPS) == _EXPECTED_LENGTHS

    def test_decodes_zero_codes_at_every_length(self):
        model = FakeQwenModel()
        tokenizer = model.model.speech_tokenizer
        assert qwen_warmup.warm_up_codec(model) == len(_EXPECTED_LENGTHS)

        assert tokenizer.lengths() == _EXPECTED_LENGTHS
        for codes in tokenizer.codes:
            assert codes.shape[1] == _QUANTIZERS
            assert codes.dtype == torch.long
            assert codes.device == model.device
            assert not codes.any()

    def test_other_codecs_are_skipped(self, caplog):
        model = FakeQwenModel(tokenizer=FakeTokenizer(model_type="qwen3_tts_tokenizer_25hz"))
        with caplog.at_level(logging.INFO, logger=qwen_warmup.__name__):
            assert qwen_warmup.warm_up_codec(model) == 0
        assert model.model.speech_tokenizer.codes == []
        assert "codec warm-up skipped" in caplog.text

    @pytest.mark.parametrize(
        "model",
        [
            types.SimpleNamespace(),
            types.SimpleNamespace(model=types.SimpleNamespace(speech_tokenizer=types.SimpleNamespace())),
        ],
    )
    def test_models_without_codec_are_skipped(self, model):
        assert qwen_warmup.warm_up_codec(model) == 0


class TestTalker:
    def test_one_step_with_a_dummy_icl_prompt(self, stub_qwen_tts):
        model = FakeQwenModel()
        assert qwen_warmup.warm_up_talker(model) is True

        (call,) = model.clone_calls
        assert call["text"] == qwen_warmup.TALKER_WARMUP_TEXT
        assert call["language"] == "auto"
        assert call["max_new_tokens"] == 2
        assert set(call) == {"text", "language", "voice_clone_prompt", "max_new_tokens"}
        (item,) = call["voice_clone_prompt"]
        assert isinstance(item, StubPromptItem)
        assert (item.icl_mode, item.x_vector_only_mode) == (True, False)
        assert item.ref_text == qwen_warmup.TALKER_WARMUP_TEXT
        assert item.ref_code.shape == (qwen_warmup.TALKER_WARMUP_REF_STEPS, _QUANTIZERS)
        assert item.ref_code.dtype == torch.long
        assert not item.ref_code.any()
        assert item.ref_spk_embedding.shape == (_HIDDEN,)
        assert not item.ref_spk_embedding.any()

    def test_its_decode_reuses_a_codec_warmup_length(self, stub_qwen_tts):
        model = FakeQwenModel()
        qwen_warmup.warm_up_talker(model)
        assert model.model.speech_tokenizer.lengths() == [qwen_warmup.TALKER_WARMUP_REF_STEPS + 1]
        assert qwen_warmup.TALKER_WARMUP_REF_STEPS + 1 in qwen_warmup.CODEC_WARMUP_STEPS

    @pytest.mark.parametrize("model_type", ["custom_voice", "voice_design", None])
    def test_non_base_models_are_skipped(self, stub_qwen_tts, model_type):
        model = FakeQwenModel()
        model.model.tts_model_type = model_type
        assert qwen_warmup.warm_up_talker(model) is False
        assert model.clone_calls == []

    def test_models_without_generate_voice_clone_are_skipped(self, stub_qwen_tts):
        model = types.SimpleNamespace(model=FakeInner(FakeTokenizer()), device=None)
        assert qwen_warmup.warm_up_talker(model) is False
        assert model.model.generate_calls == []

    def test_real_wrapper_accepts_the_dummy_prompt(self):
        """Drives the real ``Qwen3TTSModel.generate_voice_clone`` (no weights) with the warm-up's prompt."""
        qwen_tts = pytest.importorskip("qwen_tts")

        class FakeProcessor:
            def __call__(self, text, return_tensors="pt", padding=True):
                return {"input_ids": torch.arange(len(text.split()) + 12).unsqueeze(0)}

        tokenizer = FakeTokenizer()
        inner = FakeInner(tokenizer)
        model = qwen_tts.Qwen3TTSModel(model=inner, processor=FakeProcessor(), generate_defaults={"top_k": 50})

        assert qwen_warmup.warm_up_talker(model) is True

        (call,) = inner.generate_calls
        assert call["max_new_tokens"] == 2
        assert call["top_k"] == 50
        assert call["languages"] == ["auto"]
        assert call["ref_ids"] is not None
        assert call["ref_ids"][0] is not None
        prompt = call["voice_clone_prompt"]
        assert prompt["icl_mode"] == [True]
        assert prompt["x_vector_only_mode"] == [False]
        assert prompt["ref_code"][0].shape == (qwen_warmup.TALKER_WARMUP_REF_STEPS, _QUANTIZERS)
        assert tokenizer.lengths() == [qwen_warmup.TALKER_WARMUP_REF_STEPS + 1]


class TestWarmUp:
    def test_codec_first_then_talker(self, stub_qwen_tts):
        model = FakeQwenModel()
        timings = qwen_warmup.warm_up(model)
        assert model.model.speech_tokenizer.lengths() == [*_EXPECTED_LENGTHS, qwen_warmup.TALKER_WARMUP_REF_STEPS + 1]
        assert set(timings) == {"codec", "talker"}
        assert all(isinstance(v, float) for v in timings.values())

    def test_logs_each_part(self, stub_qwen_tts, caplog):
        ticks = iter([0.0, 2.5, 10.0, 10.5])
        with caplog.at_level(logging.INFO, logger=qwen_warmup.__name__):
            timings = qwen_warmup.warm_up(FakeQwenModel(), clock=lambda: next(ticks))
        assert timings == {"codec": 2.5, "talker": 0.5}
        assert "codec decoder at 11 lengths in 2.5 s, one talker step in 0.5 s" in caplog.text

    def test_a_failing_part_does_not_stop_the_other_or_raise(self, stub_qwen_tts, caplog):
        model = FakeQwenModel()
        tokenizer = model.model.speech_tokenizer
        tokenizer.error = RuntimeError("miopenStatusUnknownError")
        with caplog.at_level(logging.WARNING, logger=qwen_warmup.__name__):
            timings = qwen_warmup.warm_up(model)
        assert timings == {"codec": None, "talker": None}
        assert tokenizer.lengths() == [300, qwen_warmup.TALKER_WARMUP_REF_STEPS + 1]
        assert len(model.model.generate_calls) == 1
        assert "codec warm-up failed" in caplog.text
        assert "talker warm-up failed" in caplog.text

    def test_skipped_parts_report_none(self, stub_qwen_tts):
        assert qwen_warmup.warm_up(types.SimpleNamespace()) == {"codec": None, "talker": None}


class TestBackendLoad:
    """PyTorchTTSBackend._load_model_sync warms up only on a ROCm GPU, after the immediate-mode wrapper and VBPERF-free."""

    @pytest.fixture
    def load(self, monkeypatch, stub_qwen_tts):
        miopen = FakeMiopen()
        monkeypatch.setattr(torch.backends, "miopen", miopen)
        monkeypatch.setattr(pytorch_backend, "get_torch_device", lambda **_: "cuda")
        monkeypatch.setattr(pytorch_backend, "model_load_progress", lambda *_: contextlib.nullcontext())
        monkeypatch.delenv(perf_log.PERF_LOG_ENV, raising=False)
        monkeypatch.delenv(miopen_codec.CODEC_IMMEDIATE_ENV, raising=False)
        monkeypatch.delenv(qwen_warmup.WARMUP_ENV, raising=False)
        monkeypatch.delenv("VOICEBOX_BACKEND_VARIANT", raising=False)

        loaded: list[FakeQwenModel] = []

        def from_pretrained(*_args, **_kwargs):
            loaded.append(FakeQwenModel(miopen))
            return loaded[-1]

        stub_qwen_tts.Qwen3TTSModel = types.SimpleNamespace(from_pretrained=from_pretrained)

        def _load(device: str = "cuda") -> tuple[pytorch_backend.PyTorchTTSBackend, FakeQwenModel, FakeMiopen]:
            backend = pytorch_backend.PyTorchTTSBackend()
            backend.device = device
            backend._is_model_cached = lambda _size: True
            backend._load_model_sync("1.7B")
            assert backend.model is loaded[-1]
            return backend, backend.model, miopen

        return _load

    def test_rocm_dev_mode_warms_up_in_immediate_mode(self, load, monkeypatch):
        monkeypatch.setattr(torch.version, "hip", _HIP)
        backend, model, miopen = load()
        tokenizer = model.model.speech_tokenizer
        assert tokenizer.lengths() == [*_EXPECTED_LENGTHS, qwen_warmup.TALKER_WARMUP_REF_STEPS + 1]
        assert tokenizer.flags == [True] * (len(_EXPECTED_LENGTHS) + 1)
        assert len(model.model.generate_calls) == 1
        assert miopen.immediate is False
        assert backend.is_loaded()
        assert backend._current_model_size == "1.7B"

    def test_packaged_rocm_binary_warms_up(self, load, monkeypatch):
        monkeypatch.setattr(torch.version, "hip", None)
        monkeypatch.setenv("VOICEBOX_BACKEND_VARIANT", "rocm")
        _, model, _ = load()
        assert model.model.speech_tokenizer.lengths()[: len(_EXPECTED_LENGTHS)] == _EXPECTED_LENGTHS
        assert len(model.model.generate_calls) == 1

    def test_warm_up_logs_no_vbperf_lines(self, load, monkeypatch, caplog):
        monkeypatch.setenv(perf_log.PERF_LOG_ENV, "1")
        monkeypatch.setattr(perf_log, "cuda_synchronize", lambda: None)
        monkeypatch.setattr(perf_log, "emit_env", lambda _info: None)
        monkeypatch.setattr(torch.version, "hip", _HIP)
        caplog.set_level(logging.INFO, logger=perf_log.logger.name)

        def vbperf_lines() -> list[str]:
            return [r.getMessage() for r in caplog.records if r.getMessage().startswith(perf_log.LINE_PREFIX)]

        _, model, _ = load()
        assert vbperf_lines() == []
        assert len(model.model.speech_tokenizer.codes) == len(_EXPECTED_LENGTHS) + 1

        model.model.speech_tokenizer.decode([{"audio_codes": torch.zeros(3, _QUANTIZERS, dtype=torch.long)}])
        (line,) = vbperf_lines()
        assert '"stage": "codec"' in line

    def test_warm_up_runs_in_find_mode_when_immediate_is_off(self, load, monkeypatch):
        monkeypatch.setattr(torch.version, "hip", _HIP)
        monkeypatch.setenv(miopen_codec.CODEC_IMMEDIATE_ENV, "0")
        _, model, miopen = load()
        tokenizer = model.model.speech_tokenizer
        assert tokenizer.lengths()[: len(_EXPECTED_LENGTHS)] == _EXPECTED_LENGTHS
        assert set(tokenizer.flags) == {False}
        assert miopen.writes == []

    @pytest.mark.parametrize(
        ("hip", "variant", "device"), [(None, None, "cuda"), (None, "cuda", "cuda"), (_HIP, None, "cpu")]
    )
    def test_other_platforms_are_untouched(self, load, monkeypatch, hip, variant, device):
        monkeypatch.setattr(torch.version, "hip", hip)
        if variant is not None:
            monkeypatch.setenv("VOICEBOX_BACKEND_VARIANT", variant)
        _, model, _ = load(device)
        assert model.model.speech_tokenizer.codes == []
        assert model.model.generate_calls == []
        assert model.clone_calls == []

    def test_variable_turns_it_off(self, load, monkeypatch):
        monkeypatch.setattr(torch.version, "hip", _HIP)
        monkeypatch.setenv(qwen_warmup.WARMUP_ENV, "0")
        _, model, _ = load()
        assert model.model.speech_tokenizer.codes == []
        assert model.clone_calls == []

    def test_a_failing_warm_up_still_loads_the_model(self, load, monkeypatch, caplog):
        monkeypatch.setattr(torch.version, "hip", _HIP)

        def failing_warm_up_codec(_model):
            raise RuntimeError("out of memory")

        monkeypatch.setattr(qwen_warmup, "warm_up_codec", failing_warm_up_codec)
        with caplog.at_level(logging.WARNING, logger=qwen_warmup.__name__):
            backend, model, miopen = load()
        assert backend.is_loaded()
        assert "codec warm-up failed" in caplog.text
        assert len(model.model.generate_calls) == 1
        assert miopen.immediate is False
