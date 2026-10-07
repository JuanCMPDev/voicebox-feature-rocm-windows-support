"""
Tests for backend/utils/miopen_codec.py, MIOpen's immediate mode around the Qwen3-TTS codec decoder only (perf-loop
plan, Fase 1, task 1.1), and for how PyTorchTTSBackend applies it when the model loads.

Runs without a GPU: the ``immediate`` flag is a stub object (plus one check against the real torch attribute) and the
Qwen model is a stand-in with the same attribute layout.

Usage:
    python -m pytest backend/tests/test_miopen_codec.py -v
"""

import contextlib
import logging
import sys
import threading
import types

import pytest
import torch

from backend.backends import pytorch_backend
from backend.utils import miopen_codec, perf_log

_HIP = "7.2.53211-158bd99533"
_TIMEOUT_S = 5.0


class FakeMiopen:
    """Stands in for ``torch.backends.miopen``: records every write to ``immediate``."""

    def __init__(self, immediate: bool = False):
        self._immediate = immediate
        self.writes: list[bool] = []

    @property
    def immediate(self) -> bool:
        return self._immediate

    @immediate.setter
    def immediate(self, value: bool) -> None:
        self.writes.append(value)
        self._immediate = value


class FakeDecoder(torch.nn.Module):
    def forward(self, codes):
        return codes


class FakeTokenizer:
    """Mimics Qwen3TTSTokenizer: ``decode`` runs ``model.decoder`` once and records the flag it saw."""

    def __init__(self, miopen: FakeMiopen | None = None):
        self.miopen = miopen
        self.model = types.SimpleNamespace(decoder=FakeDecoder(), dtype=torch.bfloat16)
        self.result = (["wav"], 24000)
        self.calls: list[tuple] = []
        self.flag_seen: list[bool] = []
        self.error: Exception | None = None

    def decode(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        self.flag_seen.append(self._flag())
        self.model.decoder(torch.zeros(1))
        if self.error is not None:
            raise self.error
        return self.result

    def encode(self, *args, **kwargs):
        self.flag_seen.append(self._flag())
        return "codes"

    def _flag(self) -> bool:
        return (self.miopen or torch.backends.miopen).immediate


class FakeInner:
    def __init__(self, miopen: FakeMiopen | None = None):
        self.speech_tokenizer = FakeTokenizer(miopen)
        self.result = ([torch.zeros(39, 16, dtype=torch.long)], [torch.zeros(39, 8)])

    def generate(self, *args, **kwargs):
        return self.result


class FakeQwenModel:
    """``Qwen3TTSModel`` layout: ``model.speech_tokenizer`` and ``create_voice_clone_prompt``."""

    def __init__(self, miopen: FakeMiopen | None = None):
        self.model = FakeInner(miopen)
        self.prompt_flag_seen: list[bool] = []
        self.prompt_calls: list[tuple] = []
        self.prompt_result = [{"ref_code": "codes"}]

    def create_voice_clone_prompt(self, *args, **kwargs):
        self.prompt_calls.append((args, kwargs))
        self.prompt_flag_seen.append(self.model.speech_tokenizer._flag())
        return self.prompt_result


class TestEnabled:
    """The decision: (device, variant, hip, environment) -> wrap the codec decoder or not."""

    @pytest.mark.parametrize(("variant", "hip"), [("rocm", None), (None, _HIP), ("rocm", _HIP), ("cpu", _HIP)])
    def test_rocm_gpu_enables_it_by_default(self, variant, hip):
        assert miopen_codec.codec_immediate_enabled("cuda", variant, hip, environ={}) is True

    @pytest.mark.parametrize(("variant", "hip"), [(None, None), ("cpu", None), ("cuda", None), (None, "")])
    def test_non_rocm_builds_are_untouched(self, variant, hip):
        assert miopen_codec.codec_immediate_enabled("cuda", variant, hip, environ={}) is False

    @pytest.mark.parametrize("device", ["cpu", "xpu", "mps", object()])
    def test_non_gpu_or_other_devices_are_untouched(self, device):
        assert miopen_codec.codec_immediate_enabled(device, "rocm", _HIP, environ={}) is False

    @pytest.mark.parametrize("value", ["0", "false", " OFF ", "no"])
    def test_variable_disables_it(self, value):
        environ = {miopen_codec.CODEC_IMMEDIATE_ENV: value}
        assert miopen_codec.codec_immediate_enabled("cuda", "rocm", _HIP, environ=environ) is False

    @pytest.mark.parametrize("value", ["1", "true", ""])
    def test_other_values_keep_it_on(self, value):
        environ = {miopen_codec.CODEC_IMMEDIATE_ENV: value}
        assert miopen_codec.codec_immediate_enabled("cuda", None, _HIP, environ=environ) is True

    def test_reads_process_environment_by_default(self, monkeypatch):
        monkeypatch.setenv(miopen_codec.CODEC_IMMEDIATE_ENV, "0")
        assert miopen_codec.codec_immediate_enabled("cuda", "rocm", _HIP) is False
        monkeypatch.delenv(miopen_codec.CODEC_IMMEDIATE_ENV)
        assert miopen_codec.codec_immediate_enabled("cuda", "rocm", _HIP) is True


class TestWrapper:
    def test_flag_is_true_only_inside_decode(self):
        miopen = FakeMiopen()
        model = FakeQwenModel(miopen)
        tokenizer = model.model.speech_tokenizer
        assert miopen_codec.use_immediate_mode_for_codec(model, miopen) is True
        assert miopen.writes == []

        tokenizer.decode([{"audio_codes": "c"}])
        assert tokenizer.flag_seen == [True]
        assert miopen.immediate is False
        assert miopen.writes == [True, False]

    def test_previous_value_is_restored(self):
        miopen = FakeMiopen(immediate=True)
        model = FakeQwenModel(miopen)
        miopen_codec.use_immediate_mode_for_codec(model, miopen)
        model.model.speech_tokenizer.decode([])
        assert miopen.immediate is True
        assert miopen.writes == [True, True]

    def test_flag_is_restored_after_an_exception(self):
        miopen = FakeMiopen()
        model = FakeQwenModel(miopen)
        tokenizer = model.model.speech_tokenizer
        tokenizer.error = RuntimeError("miopenStatusUnknownError")
        miopen_codec.use_immediate_mode_for_codec(model, miopen)

        with pytest.raises(RuntimeError, match="miopenStatusUnknownError"):
            tokenizer.decode([])
        assert tokenizer.flag_seen == [True]
        assert miopen.immediate is False

        tokenizer.error = None
        tokenizer.decode([])
        assert tokenizer.flag_seen == [True, True]
        assert miopen.immediate is False

    def test_arguments_and_result_pass_through(self):
        miopen = FakeMiopen()
        model = FakeQwenModel(miopen)
        tokenizer = model.model.speech_tokenizer
        miopen_codec.use_immediate_mode_for_codec(model, miopen)
        encoded = [{"audio_codes": torch.zeros(5, 16, dtype=torch.long)}]

        result = tokenizer.decode(encoded, 7, flag="x")
        assert result is tokenizer.result
        ((args, kwargs),) = tokenizer.calls
        assert args[0] is encoded
        assert args[1:] == (7,)
        assert kwargs == {"flag": "x"}
        assert tokenizer.decode.__name__ == "decode"

    def test_reference_and_talker_stay_out_of_immediate_mode(self):
        miopen = FakeMiopen()
        model = FakeQwenModel(miopen)
        miopen_codec.use_immediate_mode_for_codec(model, miopen)

        prompt = model.create_voice_clone_prompt(ref_audio="ref.wav", ref_text="hola", x_vector_only_mode=False)
        model.model.generate(input_ids=["ids"])
        model.model.speech_tokenizer.encode("ref.wav")

        assert prompt is model.prompt_result
        assert model.prompt_calls == [((), {"ref_audio": "ref.wav", "ref_text": "hola", "x_vector_only_mode": False})]
        assert model.prompt_flag_seen == [False]
        assert model.model.speech_tokenizer.flag_seen == [False]
        assert miopen.writes == []

    def test_prompt_waits_for_a_decode_in_another_thread(self):
        """The flag is process-wide: a voice prompt must not run while another thread decodes in immediate mode."""
        miopen = FakeMiopen()
        model = FakeQwenModel(miopen)
        tokenizer = model.model.speech_tokenizer
        in_decode, release, prompt_done = threading.Event(), threading.Event(), threading.Event()
        original_decode = tokenizer.decode

        def blocking_decode(*args, **kwargs):
            in_decode.set()
            assert release.wait(_TIMEOUT_S)
            return original_decode(*args, **kwargs)

        tokenizer.decode = blocking_decode
        miopen_codec.use_immediate_mode_for_codec(model, miopen)

        decoder = threading.Thread(target=tokenizer.decode, args=([],))
        prompter = threading.Thread(target=lambda: (model.create_voice_clone_prompt(), prompt_done.set()))
        decoder.start()
        try:
            assert in_decode.wait(_TIMEOUT_S)
            prompter.start()
            assert not prompt_done.wait(0.2)
            assert miopen.immediate is True
        finally:
            release.set()
            decoder.join(_TIMEOUT_S)
            if prompter.ident is not None:
                prompter.join(_TIMEOUT_S)

        assert prompt_done.is_set()
        assert tokenizer.flag_seen == [True]
        assert model.prompt_flag_seen == [False]
        assert miopen.immediate is False

    def test_model_without_codec_is_left_alone(self, caplog):
        miopen = FakeMiopen()
        model = types.SimpleNamespace(model=types.SimpleNamespace())
        with caplog.at_level(logging.WARNING, logger=miopen_codec.__name__):
            assert miopen_codec.use_immediate_mode_for_codec(model, miopen) is False
        assert "no codec decode" in caplog.text
        assert miopen.writes == []

    def test_defaults_to_torch_backends_miopen(self):
        model = FakeQwenModel()
        tokenizer = model.model.speech_tokenizer
        before = torch.backends.miopen.immediate
        miopen_codec.use_immediate_mode_for_codec(model)
        tokenizer.decode([])
        assert tokenizer.flag_seen == [True]
        assert torch.backends.miopen.immediate == before


@pytest.fixture
def perf_enabled(monkeypatch, caplog):
    monkeypatch.setenv(perf_log.PERF_LOG_ENV, "1")
    monkeypatch.setattr(perf_log, "cuda_synchronize", lambda: None)
    caplog.set_level(logging.INFO, logger=perf_log.logger.name)

    def _codec_lines() -> list[str]:
        return [
            r.getMessage() for r in caplog.records if r.name == perf_log.logger.name and '"codec"' in r.getMessage()
        ]

    return _codec_lines


@pytest.mark.parametrize("perf_first", [False, True])
def test_composes_with_perf_log_in_either_order(perf_enabled, perf_first):
    miopen = FakeMiopen()
    model = FakeQwenModel(miopen)
    tokenizer = model.model.speech_tokenizer
    if perf_first:
        perf_log.instrument_qwen_model(model)
        miopen_codec.use_immediate_mode_for_codec(model, miopen)
    else:
        miopen_codec.use_immediate_mode_for_codec(model, miopen)
        perf_log.instrument_qwen_model(model)

    encoded = [{"audio_codes": torch.zeros(40, 16, dtype=torch.long)}]
    assert tokenizer.decode(encoded) is tokenizer.result
    assert tokenizer.calls[0][0][0] is encoded
    assert tokenizer.flag_seen == [True]
    assert miopen.immediate is False
    (line,) = perf_enabled()
    assert '"T": 40' in line
    assert '"chunks": 1' in line


class TestBackendLoad:
    """PyTorchTTSBackend._load_model_sync wraps the decoder only on a ROCm GPU, with or without VBPERF."""

    @pytest.fixture
    def load(self, monkeypatch):
        miopen = FakeMiopen()
        monkeypatch.setattr(torch.backends, "miopen", miopen)
        monkeypatch.setattr(pytorch_backend, "get_torch_device", lambda **_: "cuda")
        monkeypatch.setattr(pytorch_backend, "model_load_progress", lambda *_: contextlib.nullcontext())
        monkeypatch.delenv(perf_log.PERF_LOG_ENV, raising=False)
        monkeypatch.delenv(miopen_codec.CODEC_IMMEDIATE_ENV, raising=False)
        monkeypatch.delenv("VOICEBOX_BACKEND_VARIANT", raising=False)

        loaded: list[FakeQwenModel] = []

        def from_pretrained(*_args, **_kwargs):
            loaded.append(FakeQwenModel(miopen))
            return loaded[-1]

        fake_qwen_tts = types.ModuleType("qwen_tts")
        fake_qwen_tts.Qwen3TTSModel = types.SimpleNamespace(from_pretrained=from_pretrained)
        monkeypatch.setitem(sys.modules, "qwen_tts", fake_qwen_tts)

        def _load(device: str = "cuda") -> tuple[FakeQwenModel, FakeMiopen]:
            backend = pytorch_backend.PyTorchTTSBackend()
            backend.device = device
            backend._is_model_cached = lambda _size: True
            backend._load_model_sync("1.7B")
            assert backend.model is loaded[-1]
            return backend.model, miopen

        return _load

    def test_rocm_dev_mode_wraps_decode(self, load, monkeypatch):
        monkeypatch.setattr(torch.version, "hip", _HIP)
        model, miopen = load()
        model.model.speech_tokenizer.decode([])
        assert model.model.speech_tokenizer.flag_seen == [True]
        assert miopen.immediate is False

    def test_packaged_rocm_binary_wraps_decode(self, load, monkeypatch):
        monkeypatch.setattr(torch.version, "hip", None)
        monkeypatch.setenv("VOICEBOX_BACKEND_VARIANT", "rocm")
        model, _ = load()
        model.model.speech_tokenizer.decode([])
        assert model.model.speech_tokenizer.flag_seen == [True]

    def test_wraps_decode_with_vbperf_on(self, load, monkeypatch, perf_enabled):
        monkeypatch.setenv(perf_log.PERF_LOG_ENV, "1")
        monkeypatch.setattr(perf_log, "emit_env", lambda _info: None)
        monkeypatch.setattr(torch.version, "hip", _HIP)
        model, miopen = load()
        model.model.speech_tokenizer.decode([{"audio_codes": torch.zeros(3, 16, dtype=torch.long)}])
        assert model.model.speech_tokenizer.flag_seen == [True]
        assert miopen.immediate is False
        assert len(perf_enabled()) == 1

    @pytest.mark.parametrize(
        ("hip", "variant", "device"), [(None, None, "cuda"), (None, "cuda", "cuda"), (_HIP, None, "cpu")]
    )
    def test_other_platforms_are_untouched(self, load, monkeypatch, hip, variant, device):
        monkeypatch.setattr(torch.version, "hip", hip)
        if variant is not None:
            monkeypatch.setenv("VOICEBOX_BACKEND_VARIANT", variant)
        model, miopen = load(device)
        model.model.speech_tokenizer.decode([])
        assert model.model.speech_tokenizer.flag_seen == [False]
        assert miopen.writes == []

    def test_variable_turns_it_off(self, load, monkeypatch):
        monkeypatch.setattr(torch.version, "hip", _HIP)
        monkeypatch.setenv(miopen_codec.CODEC_IMMEDIATE_ENV, "0")
        model, miopen = load()
        model.model.speech_tokenizer.decode([])
        assert model.model.speech_tokenizer.flag_seen == [False]
        assert miopen.writes == []
