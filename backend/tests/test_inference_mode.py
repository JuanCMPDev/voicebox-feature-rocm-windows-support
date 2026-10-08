"""
Tests for PyTorchTTSBackend running Qwen3-TTS generation under ``torch.inference_mode()`` on ROCm (perf-loop plan,
Fase 3, task 3.3), and for inference mode leaving the sampled tokens as they are.

Runs without a GPU: the backend gets a fake model that records the autograd state it runs under, and a tiny code
predictor with random weights on the CPU (the real qwen_tts classes) samples from the same seed with and without it.

Usage:
    python -m pytest backend/tests/test_inference_mode.py -v
"""

import asyncio

import numpy as np
import pytest
import torch

from backend.backends import pytorch_backend
from backend.tests.test_qwen_code_predictor import _TALKER_ARGUMENTS, _embeds, _model, _predictor
from backend.utils import qwen_code_predictor

_HIP = "7.2.53211-158bd99533"
_SAMPLE_RATE = 24000


class FakeModel:
    """``Qwen3TTSModel`` stand-in: ``generate_voice_clone`` records the autograd state and returns a short wave."""

    def __init__(self):
        self.modes: list[dict] = []

    def generate_voice_clone(self, **kwargs):
        self.modes.append(
            {
                "inference_mode": torch.is_inference_mode_enabled(),
                "grad": torch.is_grad_enabled(),
                "text": kwargs["text"],
            }
        )
        return [np.arange(4, dtype=np.float32)], _SAMPLE_RATE


@pytest.fixture
def generate(monkeypatch):
    """Runs ``PyTorchTTSBackend.generate`` on a loaded fake model; returns its output and the recorded modes."""
    monkeypatch.delenv("VOICEBOX_BACKEND_VARIANT", raising=False)

    def _generate(device: str) -> tuple[tuple, list[dict]]:
        backend = pytorch_backend.PyTorchTTSBackend.__new__(pytorch_backend.PyTorchTTSBackend)
        backend.model = FakeModel()
        backend.model_size = backend._current_model_size = "1.7B"
        backend.device = device
        result = asyncio.run(backend.generate("Hola.", voice_prompt=[], language="es", seed=None))
        return result, backend.model.modes

    return _generate


class TestBackendGenerate:
    def test_rocm_dev_mode_generates_in_inference_mode(self, generate, monkeypatch):
        monkeypatch.setattr(torch.version, "hip", _HIP)
        (audio, sample_rate), modes = generate("cuda")
        assert modes == [{"inference_mode": True, "grad": False, "text": "Hola."}]
        assert np.array_equal(audio, np.arange(4, dtype=np.float32))
        assert sample_rate == _SAMPLE_RATE

    def test_packaged_rocm_binary_generates_in_inference_mode(self, generate, monkeypatch):
        monkeypatch.setattr(torch.version, "hip", None)
        monkeypatch.setenv("VOICEBOX_BACKEND_VARIANT", "rocm")
        _, modes = generate("cuda")
        assert modes[0]["inference_mode"] is True

    @pytest.mark.parametrize(
        ("hip", "variant", "device"),
        [(None, None, "cuda"), (None, "cuda", "cuda"), (None, "cpu", "cpu"), (_HIP, None, "cpu"), (_HIP, None, "xpu")],
    )
    def test_other_platforms_run_as_before(self, generate, monkeypatch, hip, variant, device):
        """No inference mode and no grad-mode change either: the model's own no_grad is all there is, as before."""
        monkeypatch.setattr(torch.version, "hip", hip)
        if variant is not None:
            monkeypatch.setenv("VOICEBOX_BACKEND_VARIANT", variant)
        (audio, _), modes = generate(device)
        assert modes == [{"inference_mode": False, "grad": True, "text": "Hola."}]
        assert np.array_equal(audio, np.arange(4, dtype=np.float32))

    def test_mode_ends_with_the_generation(self, generate, monkeypatch):
        monkeypatch.setattr(torch.version, "hip", _HIP)
        generate("cuda")
        generate("cuda")
        assert not torch.is_inference_mode_enabled()


class TestSameTokens:
    """Inference mode only drops autograd bookkeeping: from the same seed the code predictor samples the same tokens."""

    @pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
    @pytest.mark.parametrize("use_loop", [False, True])
    def test_code_predictor_samples_the_same(self, dtype, use_loop):
        predictor = _predictor("sdpa", dtype)
        if use_loop:
            assert qwen_code_predictor.install(_model(predictor)) is True
        for batch in (1, 2):
            embeds = _embeds(batch, dtype)
            for seed in (0, 1234):
                torch.manual_seed(seed)
                with torch.no_grad():
                    expected = predictor.generate(inputs_embeds=embeds, **_TALKER_ARGUMENTS).sequences
                expected_rng = torch.get_rng_state()

                torch.manual_seed(seed)
                with torch.inference_mode():
                    result = predictor.generate(inputs_embeds=embeds, **_TALKER_ARGUMENTS).sequences
                assert torch.equal(result, expected)
                assert torch.equal(torch.get_rng_state(), expected_rng)
