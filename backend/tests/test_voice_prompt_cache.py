"""
Tests for the Qwen voice prompt cache (perf-loop plan, Fase 2, task 2.1): PyTorchTTSBackend.create_voice_prompt stores
the prompt as a dict of CPU tensors, lists, bools and strings that ``torch.load(weights_only=True)`` reads back, and
rebuilds the ``VoiceClonePromptItem`` list on a hit; ``backend/utils/cache.py`` discards unreadable files.

Runs without a GPU: the model is a stand-in for ``Qwen3TTSModel`` whose ``create_voice_clone_prompt`` returns CPU
tensors and counts its calls, and the cache directory is a temporary one. One test drives the real
``Qwen3TTSModel.generate_voice_clone`` (no weights) to check that cached items reach the talker as fresh ones do.

Usage:
    python -m pytest backend/tests/test_voice_prompt_cache.py -v
"""

import asyncio
import dataclasses
import logging
import pathlib
import sys
import types
from typing import Any

import pytest
import torch

from backend.backends import pytorch_backend
from backend.utils import cache, perf_log

_QUANTIZERS = 16
_HIDDEN = 64
_REF_STEPS = 30
_REF_TEXT = "Texto de referencia."
_OTHER_SIZE = "other-size"


@dataclasses.dataclass
class StubPromptItem:
    """Same fields as ``qwen_tts.VoiceClonePromptItem``."""

    ref_code: Any
    ref_spk_embedding: Any
    x_vector_only_mode: bool
    icl_mode: bool
    ref_text: str | None = None


def _items(seed: int = 0, ref_text: str = _REF_TEXT, item_type: type = StubPromptItem) -> list:
    """One ICL prompt item like ``create_voice_clone_prompt`` builds: long codes, a bfloat16 speaker embedding."""
    generator = torch.Generator().manual_seed(seed)
    return [
        item_type(
            ref_code=torch.randint(0, 2048, (_REF_STEPS, _QUANTIZERS), generator=generator),
            ref_spk_embedding=torch.randn(_HIDDEN, generator=generator).to(torch.bfloat16),
            x_vector_only_mode=False,
            icl_mode=True,
            ref_text=ref_text,
        )
    ]


def _assert_same_items(actual: list, expected: list) -> None:
    assert len(actual) == len(expected)
    for got, want in zip(actual, expected, strict=True):
        assert torch.equal(got.ref_code, want.ref_code)
        assert got.ref_code.dtype == want.ref_code.dtype
        assert torch.equal(got.ref_spk_embedding, want.ref_spk_embedding)
        assert got.ref_spk_embedding.dtype == want.ref_spk_embedding.dtype
        assert (got.x_vector_only_mode, got.icl_mode, got.ref_text) == (
            want.x_vector_only_mode,
            want.icl_mode,
            want.ref_text,
        )


class FakeQwenModel:
    """``Qwen3TTSModel``: ``create_voice_clone_prompt`` returns fresh CPU items and records each call."""

    def __init__(self):
        self.prompt_calls: list[dict] = []

    def create_voice_clone_prompt(self, ref_audio, ref_text, x_vector_only_mode):
        self.prompt_calls.append({"ref_audio": ref_audio, "ref_text": ref_text, "xvec": x_vector_only_mode})
        return _items(ref_text=ref_text)

    def _prompt_items_to_voice_clone_prompt(self, items):
        """Same as ``Qwen3TTSModel._prompt_items_to_voice_clone_prompt``."""
        return dict(
            ref_code=[it.ref_code for it in items],
            ref_spk_embedding=[it.ref_spk_embedding for it in items],
            x_vector_only_mode=[it.x_vector_only_mode for it in items],
            icl_mode=[it.icl_mode for it in items],
        )


@pytest.fixture
def stub_qwen_tts(monkeypatch):
    """A ``qwen_tts`` module whose ``VoiceClonePromptItem`` is :class:`StubPromptItem`."""
    module = types.ModuleType("qwen_tts")
    module.VoiceClonePromptItem = StubPromptItem
    monkeypatch.setitem(sys.modules, "qwen_tts", module)
    return module


@pytest.fixture
def cache_dir(tmp_path, monkeypatch) -> pathlib.Path:
    """Empty memory cache and a temporary disk cache."""
    directory = tmp_path / "cache"
    directory.mkdir()
    monkeypatch.setattr(cache, "_memory_cache", {})
    monkeypatch.setattr(cache, "_get_cache_dir", lambda: directory)
    return directory


@pytest.fixture
def reference(tmp_path) -> str:
    """A reference audio file (only its bytes matter: they go into the cache key)."""
    path = tmp_path / "sample.wav"
    path.write_bytes(b"RIFF fake reference audio")
    return str(path)


def _restart() -> None:
    """What a server restart does to the cache: memory is gone, disk stays."""
    cache._memory_cache.clear()


class TestCacheFormat:
    def test_entry_holds_cpu_copies_lists_bools_and_strings(self):
        items = _items()
        entry = pytorch_backend._voice_prompt_to_cache(FakeQwenModel(), items)

        assert set(entry) == set(pytorch_backend._PROMPT_CACHE_FIELDS)
        assert all(isinstance(column, list) and len(column) == 1 for column in entry.values())
        assert entry["x_vector_only_mode"] == [False]
        assert entry["icl_mode"] == [True]
        assert entry["ref_text"] == [_REF_TEXT]
        for name in ("ref_code", "ref_spk_embedding"):
            (tensor,) = entry[name]
            original = getattr(items[0], name)
            assert tensor.device.type == "cpu"
            assert torch.equal(tensor, original)
            assert tensor.dtype == original.dtype
            assert tensor.untyped_storage().data_ptr() != original.untyped_storage().data_ptr()

    def test_disk_round_trip_with_weights_only(self, cache_dir, stub_qwen_tts):
        items = _items()
        cache.cache_voice_prompt("key", pytorch_backend._voice_prompt_to_cache(FakeQwenModel(), items))
        _restart()

        assert torch.load(cache_dir / "key.prompt", weights_only=True)["ref_text"] == [_REF_TEXT]
        loaded = cache.get_cached_voice_prompt("key")
        rebuilt = pytorch_backend._voice_prompt_from_cache(loaded, "cpu")

        assert all(isinstance(item, StubPromptItem) for item in rebuilt)
        _assert_same_items(rebuilt, items)

    def test_rebuilt_items_are_on_the_backend_device(self, stub_qwen_tts):
        entry = pytorch_backend._voice_prompt_to_cache(FakeQwenModel(), _items())
        (item,) = pytorch_backend._voice_prompt_from_cache(entry, "meta")
        assert item.ref_code.device.type == "meta"
        assert item.ref_spk_embedding.device.type == "meta"
        assert item.ref_spk_embedding.dtype == torch.bfloat16
        assert entry["ref_code"][0].device.type == "cpu"

    def test_x_vector_only_items_keep_their_missing_codes(self, stub_qwen_tts):
        (item,) = _items()
        item.ref_code, item.x_vector_only_mode, item.icl_mode, item.ref_text = None, True, False, None
        entry = pytorch_backend._voice_prompt_to_cache(FakeQwenModel(), [item])
        (rebuilt,) = pytorch_backend._voice_prompt_from_cache(entry, "cpu")
        assert rebuilt.ref_code is None
        assert (rebuilt.x_vector_only_mode, rebuilt.icl_mode, rebuilt.ref_text) == (True, False, None)

    @pytest.mark.parametrize(
        "cached",
        [
            None,
            [],
            torch.zeros(3),
            {"ref_audio": "sample.wav", "ref_text": _REF_TEXT},
            {"prompt": torch.zeros(3)},
        ],
        ids=["none", "list", "tensor", "mlx-dict", "legacy-dict"],
    )
    def test_other_formats_are_misses(self, stub_qwen_tts, cached):
        assert pytorch_backend._voice_prompt_from_cache(cached, "cpu") is None

    @pytest.mark.parametrize(
        ("field", "value"),
        [
            ("ref_text", [_REF_TEXT, _REF_TEXT]),
            ("icl_mode", True),
            ("ref_spk_embedding", [[0.0] * _HIDDEN]),
            ("ref_code", ["codes"]),
        ],
        ids=["length-mismatch", "not-a-list", "embedding-not-tensor", "codes-not-tensor"],
    )
    def test_malformed_entries_are_misses(self, stub_qwen_tts, field, value):
        entry = pytorch_backend._voice_prompt_to_cache(FakeQwenModel(), _items())
        entry[field] = value
        assert pytorch_backend._voice_prompt_from_cache(entry, "cpu") is None

    def test_empty_entry_is_a_miss(self, stub_qwen_tts):
        entry = {name: [] for name in pytorch_backend._PROMPT_CACHE_FIELDS}
        assert pytorch_backend._voice_prompt_from_cache(entry, "cpu") is None


class TestUnreadableFiles:
    """``get_cached_voice_prompt`` deletes what ``torch.load(weights_only=True)`` cannot read and reports a miss."""

    def test_legacy_dataclass_list_is_discarded(self, cache_dir, caplog):
        path = cache_dir / "key.prompt"
        torch.save(_items(), path)
        with caplog.at_level(logging.WARNING, logger=cache.__name__):
            assert cache.get_cached_voice_prompt("key") is None
        assert not path.exists()
        assert "key.prompt" in caplog.text
        assert "key" not in cache._memory_cache

    def test_corrupt_file_is_discarded(self, cache_dir, caplog):
        path = cache_dir / "key.prompt"
        path.write_bytes(b"\x00 not a torch file")
        with caplog.at_level(logging.WARNING, logger=cache.__name__):
            assert cache.get_cached_voice_prompt("key") is None
        assert not path.exists()
        assert "Discarding unreadable voice prompt cache key.prompt" in caplog.text

    def test_failing_delete_still_reports_a_miss(self, cache_dir, monkeypatch, caplog):
        path = cache_dir / "key.prompt"
        path.write_bytes(b"\x00 not a torch file")

        def locked(self, missing_ok=False):
            raise PermissionError("file in use")

        monkeypatch.setattr(pathlib.Path, "unlink", locked)
        with caplog.at_level(logging.WARNING, logger=cache.__name__):
            assert cache.get_cached_voice_prompt("key") is None
        assert "Failed to delete cache file" in caplog.text

    def test_missing_file_is_a_silent_miss(self, cache_dir, caplog):
        with caplog.at_level(logging.WARNING, logger=cache.__name__):
            assert cache.get_cached_voice_prompt("absent") is None
        assert caplog.text == ""


class TestCreateVoicePrompt:
    """``PyTorchTTSBackend.create_voice_prompt`` with the cache: compute once, then memory, then disk after a restart."""

    @pytest.fixture
    def backend(self, cache_dir, stub_qwen_tts, monkeypatch):
        monkeypatch.delenv(perf_log.PERF_LOG_ENV, raising=False)
        backend = pytorch_backend.PyTorchTTSBackend()
        backend.device = "cpu"
        backend.model = FakeQwenModel()
        backend._current_model_size = backend.model_size = "1.7B"
        return backend

    @staticmethod
    def _create(backend, reference: str, use_cache: bool = True, ref_text: str = _REF_TEXT):
        return asyncio.run(backend.create_voice_prompt(reference, ref_text, use_cache=use_cache))

    def test_first_call_computes_and_writes_a_weights_only_file(self, backend, reference, cache_dir):
        items, cached = self._create(backend, reference)

        assert cached is False
        _assert_same_items(items, _items())
        assert backend.model.prompt_calls == [{"ref_audio": reference, "ref_text": _REF_TEXT, "xvec": False}]
        key = pytorch_backend._prompt_cache_key("1.7B", reference, _REF_TEXT)
        assert key == f"qwen_1.7B_{cache.get_cache_key(reference, _REF_TEXT)}"
        entry = torch.load(cache_dir / f"{key}.prompt", weights_only=True)
        assert entry["ref_text"] == [_REF_TEXT]

    def test_second_call_hits_the_memory_cache(self, backend, reference):
        first, _ = self._create(backend, reference)
        second, cached = self._create(backend, reference)

        assert cached is True
        assert len(backend.model.prompt_calls) == 1
        _assert_same_items(second, first)

    def test_after_a_restart_it_hits_the_disk_cache(self, backend, reference):
        first, _ = self._create(backend, reference)
        _restart()
        second, cached = self._create(backend, reference)

        assert cached is True
        assert len(backend.model.prompt_calls) == 1
        assert all(isinstance(item, StubPromptItem) for item in second)
        _assert_same_items(second, first)

    def test_without_cache_it_always_computes_and_writes_nothing(self, backend, reference, cache_dir):
        self._create(backend, reference, use_cache=False)
        _, cached = self._create(backend, reference, use_cache=False)

        assert cached is False
        assert len(backend.model.prompt_calls) == 2
        assert list(cache_dir.glob("*.prompt")) == []

    def test_each_model_size_has_its_own_entry(self, backend, reference):
        self._create(backend, reference)
        backend._current_model_size = backend.model_size = _OTHER_SIZE
        _, cached = self._create(backend, reference)

        assert cached is False
        assert len(backend.model.prompt_calls) == 2

    def test_reference_text_is_part_of_the_key(self, backend, reference):
        self._create(backend, reference)
        _, cached = self._create(backend, reference, ref_text="Otro texto.")

        assert cached is False
        assert backend.model.prompt_calls[-1]["ref_text"] == "Otro texto."

    def test_a_legacy_file_is_replaced_by_a_readable_one(self, backend, reference, cache_dir):
        key = pytorch_backend._prompt_cache_key("1.7B", reference, _REF_TEXT)
        torch.save(_items(seed=1), cache_dir / f"{key}.prompt")

        _, cached = self._create(backend, reference)
        assert cached is False
        _restart()
        _, cached = self._create(backend, reference)

        assert cached is True
        assert len(backend.model.prompt_calls) == 1

    def test_a_corrupt_file_is_recomputed(self, backend, reference, cache_dir):
        self._create(backend, reference)
        (path,) = cache_dir.glob("*.prompt")
        path.write_bytes(b"truncated")
        _restart()

        items, cached = self._create(backend, reference)
        assert cached is False
        assert len(backend.model.prompt_calls) == 2
        _assert_same_items(items, _items())
        assert torch.load(path, weights_only=True)["ref_text"] == [_REF_TEXT]

    def test_vbperf_reference_line_reports_the_hit(self, backend, reference, monkeypatch, caplog):
        monkeypatch.setenv(perf_log.PERF_LOG_ENV, "1")
        monkeypatch.setattr(perf_log, "cuda_synchronize", lambda: None)
        caplog.set_level(logging.INFO, logger=perf_log.logger.name)

        self._create(backend, reference)
        self._create(backend, reference)

        lines = [r.getMessage() for r in caplog.records if r.getMessage().startswith(perf_log.LINE_PREFIX)]
        assert len(lines) == 2
        assert all('"stage": "reference"' in line for line in lines)
        assert '"cached": false' in lines[0]
        assert '"cached": true' in lines[1]


class TestRealWrapper:
    def test_cached_items_reach_the_talker_like_fresh_ones(self, cache_dir):
        """Real ``Qwen3TTSModel`` (no weights): the talker gets the same inputs, ref_ids included, from a disk hit."""
        qwen_tts = pytest.importorskip("qwen_tts")

        class FakeProcessor:
            def __call__(self, text, return_tensors="pt", padding=True):
                return {"input_ids": torch.tensor([[ord(char) for char in text]])}

        class FakeSpeechTokenizer:
            def decode(self, encoded):
                return [torch.zeros(int(item["audio_codes"].shape[0]) * 4).numpy() for item in encoded], 24000

        class FakeInner:
            tts_model_type = "base"
            device = torch.device("cpu")

            def __init__(self):
                self.speech_tokenizer = FakeSpeechTokenizer()
                self.generate_calls: list[dict] = []

            def generate(self, **kwargs):
                self.generate_calls.append(kwargs)
                return [torch.ones(2, _QUANTIZERS, dtype=torch.long)], None

            def get_supported_languages(self):
                return ["auto", "spanish"]

        inner = FakeInner()
        model = qwen_tts.Qwen3TTSModel(model=inner, processor=FakeProcessor())
        fresh = _items(item_type=qwen_tts.VoiceClonePromptItem)

        cache.cache_voice_prompt("key", pytorch_backend._voice_prompt_to_cache(model, fresh))
        _restart()
        rebuilt = pytorch_backend._voice_prompt_from_cache(cache.get_cached_voice_prompt("key"), "cpu")
        assert all(isinstance(item, qwen_tts.VoiceClonePromptItem) for item in rebuilt)

        wavs = [
            model.generate_voice_clone(text="Hola.", language="spanish", voice_clone_prompt=prompt)[0][0]
            for prompt in (fresh, rebuilt)
        ]

        fresh_call, cached_call = inner.generate_calls
        assert cached_call["ref_ids"] is not None
        assert torch.equal(cached_call["ref_ids"][0], fresh_call["ref_ids"][0])
        assert torch.equal(cached_call["input_ids"][0], fresh_call["input_ids"][0])
        assert cached_call["languages"] == fresh_call["languages"]
        fresh_prompt, cached_prompt = fresh_call["voice_clone_prompt"], cached_call["voice_clone_prompt"]
        for name in ("ref_code", "ref_spk_embedding"):
            assert torch.equal(cached_prompt[name][0], fresh_prompt[name][0])
            assert cached_prompt[name][0].dtype == fresh_prompt[name][0].dtype
        for name in ("x_vector_only_mode", "icl_mode"):
            assert cached_prompt[name] == fresh_prompt[name]
        assert wavs[0].shape == wavs[1].shape
