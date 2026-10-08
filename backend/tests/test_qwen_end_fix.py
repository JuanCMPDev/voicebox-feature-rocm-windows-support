"""
Tests for backend/utils/qwen_end_fix.py, the end-of-phrase fix of Qwen3-TTS (perf-loop task 4.4), and for how
PyTorchTTSBackend applies it.

Runs without a GPU: the processor runs on CPU tensors; the backend gets stand-in models that record what they are asked;
some tests drive HF's own ``generate`` loop on a tiny GPT-2 with random weights to check where the processor runs in it
and that it holds each row's end back on its own.

Usage:
    python -m pytest backend/tests/test_qwen_end_fix.py -v
"""

import asyncio
import contextlib
import sys
import types
from typing import Any

import numpy as np
import pytest
import torch

from backend.backends import pytorch_backend
from backend.utils import miopen_codec, perf_log, qwen_code_predictor, qwen_end_fix, qwen_warmup

_HIP = "7.2.53211-158bd99533"
_SAMPLE_RATE = 24000
_EOS = 5
_VOCAB = 8
_ON: dict[str, str] = {}
_OFF = {qwen_end_fix.END_FIX_ENV: "0"}

# With 7 other tokens at 0, an EOS logit of 3 gives p = e^3 / (e^3 + 7) = 0.74; -3 gives 0.007.
_LIKELY, _UNLIKELY = 3.0, -3.0


def _scores(eos_logits: list[float], vocab: int = _VOCAB) -> torch.Tensor:
    """(rows, vocab) float32 scores: 0 everywhere but ``codec_eos``, which gets each row's logit."""
    scores = torch.zeros(len(eos_logits), vocab)
    scores[:, _EOS] = torch.tensor(eos_logits)
    return scores


def _ids(rows: int, length: int = 0) -> torch.Tensor:
    return torch.zeros(rows, length, dtype=torch.long)


def _item(frames: int | None, ref_text: str | None) -> Any:
    """``VoiceClonePromptItem`` stand-in: ``ref_code`` of ``frames`` steps (None for none) and ``ref_text``."""
    ref_code = None if frames is None else torch.zeros(frames, 16, dtype=torch.long)
    return types.SimpleNamespace(ref_code=ref_code, ref_text=ref_text)


class TestEnabled:
    @pytest.mark.parametrize("value", ["0", "false", " OFF ", "no"])
    def test_variable_turns_it_off(self, value):
        assert qwen_end_fix.end_fix_enabled({qwen_end_fix.END_FIX_ENV: value}) is False

    @pytest.mark.parametrize("environ", [{}, {qwen_end_fix.END_FIX_ENV: "1"}, {qwen_end_fix.END_FIX_ENV: ""}])
    def test_on_by_default(self, environ):
        assert qwen_end_fix.end_fix_enabled(environ) is True

    def test_reads_process_environment_by_default(self, monkeypatch):
        monkeypatch.setenv(qwen_end_fix.END_FIX_ENV, "0")
        assert qwen_end_fix.end_fix_enabled() is False
        monkeypatch.delenv(qwen_end_fix.END_FIX_ENV)
        assert qwen_end_fix.end_fix_enabled() is True


class TestGuardFrames:
    def test_reference_rhythm_times_the_ratio(self):
        # 330 frames for 473 characters: 0.698 frames per character.
        item = _item(330, "x" * 473)
        assert qwen_end_fix.guard_frames(["y" * 44, "y" * 166], [item, item], ratio=1.0) == [31, 116]
        assert qwen_end_fix.guard_frames(["y" * 44], [item], ratio=1.5) == [47]

    def test_default_ratio(self):
        assert qwen_end_fix.guard_frames(["y" * 10], [_item(20, "x" * 10)]) == [20 * qwen_end_fix.GUARD_RATIO]

    def test_one_item_serves_every_text(self):
        assert qwen_end_fix.guard_frames(["ab", "abcd"], [_item(10, "abcde")], ratio=1.0) == [4, 8]

    @pytest.mark.parametrize(
        "item", [_item(None, "texto"), _item(10, None), _item(10, ""), "item", types.SimpleNamespace()]
    )
    def test_rows_without_reference_codes_or_text_get_no_guard(self, item):
        assert qwen_end_fix.guard_frames(["hola", "adiós"], [_item(10, "abcde"), item], ratio=1.0) == [8, 0]

    @pytest.mark.parametrize("items", [[], [_item(10, "ab")] * 3, {"ref_code": [1]}, None])
    def test_items_that_do_not_match_the_texts_give_no_guards(self, items):
        assert qwen_end_fix.guard_frames(["hola", "adiós"], items) == [0, 0]


class TestPhraseEnd:
    def test_yields_non_streaming_and_sets_this_threads_guards(self):
        assert qwen_end_fix.current_guards() is None
        with qwen_end_fix.phrase_end(["ab", "abcd"], [_item(10, "abcde")], environ=_ON) as kwargs:
            assert kwargs == {"non_streaming_mode": True}
            assert qwen_end_fix.current_guards() == qwen_end_fix.guard_frames(["ab", "abcd"], [_item(10, "abcde")])
        assert qwen_end_fix.current_guards() is None

    def test_off_yields_nothing_and_sets_nothing(self):
        with qwen_end_fix.phrase_end(["ab"], [_item(10, "abcde")], environ=_OFF) as kwargs:
            assert kwargs == {}
            assert qwen_end_fix.current_guards() is None

    def test_guards_are_per_thread(self):
        import threading

        seen: list = []
        with qwen_end_fix.phrase_end(["ab"], [_item(10, "abcde")], environ=_ON):
            thread = threading.Thread(target=lambda: seen.append(qwen_end_fix.current_guards()))
            thread.start()
            thread.join()
        assert seen == [None]

    def test_restored_after_an_error(self):
        with pytest.raises(RuntimeError), qwen_end_fix.phrase_end(["ab"], [_item(10, "abcde")], environ=_ON):
            raise RuntimeError("out of memory")
        assert qwen_end_fix.current_guards() is None


class TestEosHoldProcessor:
    def _run(self, processor, steps: list[list[float]], prompt: int = 0) -> list[torch.Tensor]:
        """The processor's output at each step; ``steps[i]`` holds each row's EOS logit at step ``i``."""
        out = []
        for i, eos_logits in enumerate(steps):
            out.append(processor(_ids(len(eos_logits), prompt + i), _scores(eos_logits)).clone())
        return out

    @staticmethod
    def _removed(scores: torch.Tensor) -> list[bool]:
        return torch.isneginf(scores[:, _EOS]).tolist()

    def test_holds_each_rows_first_likely_eos_and_only_that(self):
        processor = qwen_end_fix.EosHoldProcessor(_EOS, hold_frames=1, likely=0.1)
        out = self._run(processor, [[_LIKELY, _UNLIKELY], [_LIKELY, _LIKELY], [_LIKELY, _LIKELY]])
        assert [self._removed(s) for s in out] == [[True, False], [False, True], [False, False]]

    def test_hold_frames_counts_steps_per_row(self):
        processor = qwen_end_fix.EosHoldProcessor(_EOS, hold_frames=2, likely=0.1)
        out = self._run(processor, [[_LIKELY, _UNLIKELY], [_LIKELY, _LIKELY], [_LIKELY, _LIKELY], [_LIKELY, _LIKELY]])
        assert [self._removed(s) for s in out] == [[True, False], [True, True], [False, True], [False, False]]

    def test_without_guards_unlikely_eos_is_left_alone_and_does_not_count(self):
        processor = qwen_end_fix.EosHoldProcessor(_EOS, hold_frames=1, likely=0.1)
        out = self._run(processor, [[_UNLIKELY], [_UNLIKELY], [_LIKELY], [_LIKELY]])
        assert [self._removed(s) for s in out] == [[False], [False], [True], [False]]
        assert torch.equal(out[0], _scores([_UNLIKELY]))

    def test_guard_removes_unlikely_eos_before_it_and_not_after_row_by_row(self):
        processor = qwen_end_fix.EosHoldProcessor(_EOS, hold_frames=0, likely=0.1, guards=[2, 0, 4])
        out = self._run(processor, [[_UNLIKELY] * 3] * 5)
        assert [self._removed(s) for s in out] == [
            [True, False, True],
            [True, False, True],
            [False, False, True],
            [False, False, True],
            [False, False, False],
        ]

    def test_guard_leaves_likely_eos_to_the_hold(self):
        processor = qwen_end_fix.EosHoldProcessor(_EOS, hold_frames=1, likely=0.1, guards=[10])
        out = self._run(processor, [[_UNLIKELY], [_LIKELY], [_UNLIKELY], [_LIKELY]])
        # step 0: unlikely, guarded; 1: first likely, held; 2: unlikely, guarded; 3: likely, allowed.
        assert [self._removed(s) for s in out] == [[True], [True], [True], [False]]

    def test_steps_count_from_the_first_call(self):
        """With a prompt in ``input_ids`` (HF's decoder-only text models), steps start at the first new token."""
        processor = qwen_end_fix.EosHoldProcessor(_EOS, hold_frames=0, likely=0.1, guards=[2])
        out = self._run(processor, [[_UNLIKELY]] * 3, prompt=7)
        assert [self._removed(s) for s in out] == [[True], [True], [False]]

    @pytest.mark.parametrize("guards", [[3], [3, 3, 3], None])
    def test_guards_of_another_batch_size_are_ignored(self, guards):
        processor = qwen_end_fix.EosHoldProcessor(_EOS, hold_frames=0, likely=0.1, guards=guards)
        (out,) = self._run(processor, [[_UNLIKELY, _UNLIKELY]])
        assert self._removed(out) == [False, False]

    def test_only_eos_changes(self):
        processor = qwen_end_fix.EosHoldProcessor(_EOS, hold_frames=1, likely=0.1, guards=[0, 0, 0, 5])
        scores = torch.randn(4, _VOCAB, generator=torch.Generator().manual_seed(3))
        scores[:, _EOS] = torch.tensor([_LIKELY, _UNLIKELY, _LIKELY, _UNLIKELY])
        expected = scores.clone()
        expected[[0, 2, 3], _EOS] = float("-inf")
        result = processor(_ids(4), scores)
        assert torch.equal(result, expected)
        assert result.dtype == torch.float32

    @pytest.mark.parametrize(("temperature", "top_k"), [(None, None), (0.9, None), (0.9, 2), (0.7, 3), (1.3, 50)])
    def test_probability_is_hfs_sampling_distribution(self, temperature, top_k):
        """softmax after HF's temperature and top-k warpers, as the talker samples from."""
        from transformers.generation.logits_process import TemperatureLogitsWarper, TopKLogitsWarper

        generator = torch.Generator().manual_seed(7)
        scores = torch.randn(4, _VOCAB, generator=generator) * 2
        processor = qwen_end_fix.EosHoldProcessor(_EOS, temperature=temperature, top_k=top_k)
        expected = scores.clone()
        if temperature is not None:
            expected = TemperatureLogitsWarper(temperature)(None, expected)
        if top_k is not None:
            expected = TopKLogitsWarper(top_k=top_k)(None, expected)
        torch.testing.assert_close(processor.eos_probability(scores), torch.softmax(expected, -1)[:, _EOS])

    def test_top_k_can_make_eos_likely(self):
        """EOS second best of eight: below 0.2 over the whole vocabulary, above it among the top two."""
        scores = torch.zeros(1, _VOCAB)
        scores[0, 0], scores[0, _EOS] = 2.0, 1.0
        plain = qwen_end_fix.EosHoldProcessor(_EOS, hold_frames=1, likely=0.2)
        top_two = qwen_end_fix.EosHoldProcessor(_EOS, hold_frames=1, likely=0.2, top_k=2)
        assert self._removed(plain(_ids(1), scores.clone())) == [False]
        assert self._removed(top_two(_ids(1), scores.clone())) == [True]

    def test_no_hold_and_no_guard_changes_nothing(self):
        processor = qwen_end_fix.EosHoldProcessor(_EOS, hold_frames=0, likely=0.1)
        for eos_logits in ([_LIKELY, _UNLIKELY], [_UNLIKELY, _LIKELY]):
            scores = _scores(eos_logits)
            assert torch.equal(processor(_ids(2), scores.clone()), scores)


class FakeTalker:
    """Talker stand-in: ``generate`` records its arguments."""

    def __init__(self):
        self.calls: list[dict] = []

    def generate(self, *args, **kwargs):
        self.calls.append({"args": args, **kwargs})
        return "talker result"


def _qwen_model(talker: Any, eos: Any = _EOS) -> Any:
    """``Qwen3TTSModel`` layout: ``model.model.talker`` and ``model.model.config.talker_config.codec_eos_token_id``."""
    config = types.SimpleNamespace(talker_config=types.SimpleNamespace(codec_eos_token_id=eos))
    return types.SimpleNamespace(model=types.SimpleNamespace(talker=talker, config=config))


# What Qwen3TTSForConditionalGeneration.generate passes to the talker, with the 1.7B Base model's settings.
_TALKER_KWARGS = {"max_new_tokens": 8192, "min_new_tokens": 2, "do_sample": True, "top_k": 50, "temperature": 0.9}


class TestInstall:
    def test_talker_generate_gets_the_processor(self):
        talker = FakeTalker()
        assert qwen_end_fix.install(_qwen_model(talker)) is True

        assert talker.generate(inputs_embeds="embeds", **_TALKER_KWARGS) == "talker result"

        (call,) = talker.calls
        processors = call.pop("logits_processor")
        assert call == {"args": (), "inputs_embeds": "embeds", **_TALKER_KWARGS}
        (processor,) = processors
        assert isinstance(processor, qwen_end_fix.EosHoldProcessor)
        assert processor.eos_token_id == _EOS
        assert (processor.hold_frames, processor.likely) == (qwen_end_fix.HOLD_FRAMES, qwen_end_fix.EOS_LIKELY)
        assert (processor.temperature, processor.top_k) == (0.9, 50)
        assert processor.guards is None

    def test_guards_come_from_the_phrase_end_call(self):
        talker = FakeTalker()
        qwen_end_fix.install(_qwen_model(talker))
        items = [_item(10, "abcde"), _item(20, "abcde")]
        with qwen_end_fix.phrase_end(["ab", "abcd"], items, environ=_ON):
            talker.generate(**_TALKER_KWARGS)
        talker.generate(**_TALKER_KWARGS)
        inside, outside = (call["logits_processor"][0] for call in talker.calls)
        assert inside.guards == qwen_end_fix.guard_frames(["ab", "abcd"], items)
        assert outside.guards is None

    def test_goes_after_the_calls_own_processors(self):
        talker = FakeTalker()
        qwen_end_fix.install(_qwen_model(talker))
        own = object()
        talker.generate(logits_processor=[own], **_TALKER_KWARGS)
        processors = talker.calls[0]["logits_processor"]
        assert processors[0] is own
        assert isinstance(processors[1], qwen_end_fix.EosHoldProcessor)

    def test_greedy_calls_use_the_plain_distribution(self):
        talker = FakeTalker()
        qwen_end_fix.install(_qwen_model(talker))
        talker.generate(**{**_TALKER_KWARGS, "do_sample": False})
        (processor,) = talker.calls[0]["logits_processor"]
        assert (processor.temperature, processor.top_k) == (None, None)

    def test_each_call_gets_its_own_processor(self):
        talker = FakeTalker()
        qwen_end_fix.install(_qwen_model(talker))
        talker.generate(**_TALKER_KWARGS)
        talker.generate(**_TALKER_KWARGS)
        first, second = (call["logits_processor"][0] for call in talker.calls)
        assert first is not second

    def test_installing_twice_adds_one_processor(self):
        talker = FakeTalker()
        model = _qwen_model(talker)
        assert qwen_end_fix.install(model) is True
        wrapper = talker.generate
        assert qwen_end_fix.install(model) is True
        assert talker.generate is wrapper
        talker.generate(**_TALKER_KWARGS)
        assert len(talker.calls[0]["logits_processor"]) == 1
        assert wrapper.original.__self__ is talker

    @pytest.mark.parametrize(
        "model",
        [
            types.SimpleNamespace(),
            _qwen_model(types.SimpleNamespace()),
            _qwen_model(FakeTalker(), eos=None),
        ],
    )
    def test_models_without_talker_generate_or_eos_are_left_alone(self, model):
        talker = getattr(getattr(model, "model", None), "talker", None)
        before = getattr(talker, "generate", None)
        assert qwen_end_fix.install(model) is False
        assert getattr(talker, "generate", None) == before


_FILLER = 3
_PAD = 0


class _Script:
    """Adds a bias that makes ``codec_eos`` the greedy choice of each row from its given step on, filler before.

    ``eos_bias`` over the filler's 50 sets how likely EOS is then: +50 makes it ~1, +1 makes it ~0.73. It goes in the
    call's own processors, so it runs after HF's (minimum length) and before the end fix.
    """

    def __init__(self, prompt_length: int, eos_from: list[int], eos_bias: float = 50.0):
        self.prompt_length = prompt_length
        self.eos_from = eos_from
        self.eos_bias = eos_bias

    def __call__(self, input_ids, scores):
        step = input_ids.shape[1] - self.prompt_length
        bias = torch.zeros_like(scores)
        bias[:, _FILLER] = 50.0
        for row, start in enumerate(self.eos_from):
            bias[row, _EOS] = 50.0 + self.eos_bias if step >= start else -100.0
        return scores + bias


class TestHFGenerate:
    """The installed processor inside HF's real ``generate`` loop (tiny GPT-2, greedy), row by row."""

    @pytest.fixture
    def talker(self):
        transformers = pytest.importorskip("transformers")
        config = transformers.GPT2Config(
            vocab_size=_VOCAB, n_positions=32, n_embd=8, n_layer=1, n_head=2, bos_token_id=1, eos_token_id=_EOS
        )
        torch.manual_seed(0)
        return transformers.GPT2LMHeadModel(config).eval()

    def _generate(self, talker, eos_from: list[int], eos_bias: float = 50.0, **kwargs) -> list[list[int]]:
        from transformers import LogitsProcessorList

        prompt = torch.ones(len(eos_from), 2, dtype=torch.long)
        with torch.no_grad():
            out = talker.generate(
                prompt,
                attention_mask=torch.ones_like(prompt),
                do_sample=False,
                max_new_tokens=8,
                eos_token_id=_EOS,
                pad_token_id=_PAD,
                logits_processor=LogitsProcessorList([_Script(prompt.shape[1], eos_from, eos_bias)]),
                **kwargs,
            )
        return out[:, prompt.shape[1] :].tolist()

    def test_without_the_fix_each_row_ends_where_the_model_says(self, talker):
        assert self._generate(talker, [0, 2]) == [[_EOS, _PAD, _PAD], [_FILLER, _FILLER, _EOS]]

    @pytest.mark.parametrize("hold_frames", [1, 2])
    def test_each_row_ends_hold_frames_later(self, talker, monkeypatch, hold_frames):
        monkeypatch.setattr(qwen_end_fix, "HOLD_FRAMES", hold_frames)
        assert qwen_end_fix.install(_qwen_model(talker)) is True
        row0 = [_FILLER] * hold_frames + [_EOS]
        row1 = [_FILLER] * (2 + hold_frames) + [_EOS]
        assert self._generate(talker, [0, 2]) == [row0 + [_PAD] * (len(row1) - len(row0)), row1]

    def test_unlikely_eos_waits_for_each_rows_guard(self, talker, monkeypatch):
        """EOS the greedy choice from the start but below ``EOS_LIKELY``: only the guard keeps it out, per row."""
        monkeypatch.setattr(qwen_end_fix, "EOS_LIKELY", 0.9)
        qwen_end_fix.install(_qwen_model(talker))
        items = [_item(10, "x" * 10)] * 2
        with qwen_end_fix.phrase_end(["abc", ""], items, environ=_ON):
            out = self._generate(talker, [0, 0], eos_bias=1.0)
        guards = qwen_end_fix.guard_frames(["abc", ""], items)
        assert guards[0] > 0
        assert guards[1] == 0
        assert out == [[_FILLER] * guards[0] + [_EOS], [_EOS] + [_PAD] * guards[0]]

    def test_runs_after_hfs_minimum_length(self, talker, monkeypatch):
        """``min_new_tokens`` bans EOS first; a banned EOS is not likely, so the hold comes on the next step."""
        monkeypatch.setattr(qwen_end_fix, "HOLD_FRAMES", 1)
        qwen_end_fix.install(_qwen_model(talker))
        assert self._generate(talker, [0], min_new_tokens=1) == [[_FILLER, _FILLER, _EOS]]

    def test_never_goes_past_max_new_tokens(self, talker, monkeypatch):
        monkeypatch.setattr(qwen_end_fix, "HOLD_FRAMES", 20)
        qwen_end_fix.install(_qwen_model(talker))
        assert self._generate(talker, [0]) == [[_FILLER] * 8]


class FakeVoiceCloneModel:
    """``Qwen3TTSModel`` stand-in: records each ``generate_voice_clone`` call, with the guards set while it runs."""

    def __init__(self):
        self.calls: list[dict] = []

    def generate_voice_clone(self, **kwargs):
        self.calls.append({**kwargs, "guards": qwen_end_fix.current_guards()})
        texts = kwargs["text"] if isinstance(kwargs["text"], list) else [kwargs["text"]]
        return [np.zeros(4, dtype=np.float32) for _ in texts], _SAMPLE_RATE


def _loaded_backend(device: str) -> pytorch_backend.PyTorchTTSBackend:
    backend = pytorch_backend.PyTorchTTSBackend.__new__(pytorch_backend.PyTorchTTSBackend)
    backend.model = FakeVoiceCloneModel()
    backend.model_size = backend._current_model_size = "1.7B"
    backend.device = device
    return backend


class TestBackendGenerate:
    """``generate`` and ``generate_batch`` apply the fix on every device, row by row, unless it is turned off."""

    _ITEM_A = _item(330, "x" * 473)
    _ITEM_B = _item(100, "x" * 50)

    @pytest.fixture(autouse=True)
    def _environment(self, monkeypatch):
        monkeypatch.delenv(qwen_end_fix.END_FIX_ENV, raising=False)
        monkeypatch.delenv("VOICEBOX_BACKEND_VARIANT", raising=False)
        monkeypatch.setattr(pytorch_backend, "manual_seed", lambda seed, device: None)

    def _call(self, backend, batch: bool) -> tuple[dict, list[str], list]:
        if batch:
            texts, items = ["Hola.", "Adiós, hasta mañana."], [self._ITEM_A, self._ITEM_B]
            asyncio.run(backend.generate_batch(texts, [[item] for item in items], language="es", seed=1))
        else:
            texts, items = ["Hola."], [self._ITEM_A]
            asyncio.run(backend.generate(texts[0], voice_prompt=items, language="es", seed=1))
        (call,) = backend.model.calls
        return call, texts, items

    @pytest.mark.parametrize("batch", [False, True])
    @pytest.mark.parametrize(("device", "hip"), [("cuda", _HIP), ("cuda", None), ("cpu", None), ("xpu", None)])
    def test_applied_on_every_device(self, monkeypatch, batch, device, hip):
        monkeypatch.setattr(torch.version, "hip", hip)
        call, texts, items = self._call(_loaded_backend(device), batch)
        assert call["non_streaming_mode"] is True
        assert call["guards"] == qwen_end_fix.guard_frames(texts, items)
        assert all(guard > 0 for guard in call["guards"])
        assert call["language"] == "spanish"
        assert qwen_end_fix.current_guards() is None

    @pytest.mark.parametrize("batch", [False, True])
    def test_turned_off_passes_what_it_passed_before(self, monkeypatch, batch):
        monkeypatch.setattr(torch.version, "hip", _HIP)
        monkeypatch.setenv(qwen_end_fix.END_FIX_ENV, "0")
        call, _, _ = self._call(_loaded_backend("cuda"), batch)
        assert call.pop("guards") is None
        assert set(call) == {"text", "voice_clone_prompt", "language", "instruct"}


class FakeQwenModel:
    """``Qwen3TTSModel`` with a talker that records its ``generate`` calls; no codec, so the ROCm warm-up skips it."""

    def __init__(self):
        self.talker = FakeTalker()
        self.model = _qwen_model(self.talker).model
        self.device = torch.device("cpu")


class TestBackendLoad:
    """``_load_model_sync`` installs the stopping rule on every device, unless ``VOICEBOX_QWEN_END_FIX=0``."""

    @pytest.fixture
    def load(self, monkeypatch):
        module = types.ModuleType("qwen_tts")
        module.Qwen3TTSModel = types.SimpleNamespace(from_pretrained=lambda *_a, **_k: FakeQwenModel())
        monkeypatch.setitem(sys.modules, "qwen_tts", module)
        monkeypatch.setattr(pytorch_backend, "model_load_progress", lambda *_: contextlib.nullcontext())
        monkeypatch.delenv(qwen_end_fix.END_FIX_ENV, raising=False)
        monkeypatch.delenv(perf_log.PERF_LOG_ENV, raising=False)
        monkeypatch.delenv("VOICEBOX_BACKEND_VARIANT", raising=False)
        # On a ROCm GPU, leave out the other load-time steps: they need a real model.
        monkeypatch.setenv(miopen_codec.CODEC_IMMEDIATE_ENV, "0")
        monkeypatch.setenv(qwen_code_predictor.CODE_PREDICTOR_LOOP_ENV, "0")
        monkeypatch.setenv(qwen_warmup.WARMUP_ENV, "0")

        def _load(device: str) -> FakeQwenModel:
            monkeypatch.setattr(pytorch_backend, "get_torch_device", lambda **_: device)
            backend = pytorch_backend.PyTorchTTSBackend()
            backend._is_model_cached = lambda _size: True
            backend._load_model_sync("1.7B")
            return backend.model

        return _load

    @pytest.mark.parametrize(("device", "hip"), [("cuda", _HIP), ("cuda", None), ("cpu", None)])
    def test_installed_on_every_device(self, load, monkeypatch, device, hip):
        monkeypatch.setattr(torch.version, "hip", hip)
        model = load(device)
        model.model.talker.generate(**_TALKER_KWARGS)
        (call,) = model.talker.calls
        (processor,) = call["logits_processor"]
        assert isinstance(processor, qwen_end_fix.EosHoldProcessor)

    def test_variable_leaves_the_talker_as_it_was(self, load, monkeypatch):
        monkeypatch.setattr(torch.version, "hip", _HIP)
        monkeypatch.setenv(qwen_end_fix.END_FIX_ENV, "0")
        model = load("cuda")
        model.model.talker.generate(**_TALKER_KWARGS)
        (call,) = model.talker.calls
        assert "logits_processor" not in call
