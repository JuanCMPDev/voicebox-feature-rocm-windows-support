"""/health walks the HF cache at most once a minute, and off the event loop.

No GPU and no models: scan_cache_dir, the TTS backend and the GPU probes are stubs; the route and its cache are the
real ones.

Usage:
    python -m pytest backend/tests/test_health_cache_scan.py -q
"""

import asyncio
import sys
import threading
import time
from types import SimpleNamespace

import huggingface_hub
import pytest
from huggingface_hub import constants as hf_constants

from backend.routes import health

REPO_ID = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"


@pytest.fixture(autouse=True)
def _empty_cache():
    health._model_downloaded_cache.clear()
    yield
    health._model_downloaded_cache.clear()


class _FakeScan:
    """Stand-in for huggingface_hub.scan_cache_dir that counts its calls."""

    def __init__(self, repo_ids: list[str], *, error: Exception | None = None):
        self.repo_ids = repo_ids
        self.error = error
        self.calls = 0
        self.threads: list[int] = []

    def __call__(self):
        self.calls += 1
        self.threads.append(threading.get_ident())
        if self.error is not None:
            raise self.error
        return SimpleNamespace(repos=[SimpleNamespace(repo_id=repo_id) for repo_id in self.repo_ids])


@pytest.fixture
def fake_scan(monkeypatch):
    scan = _FakeScan([REPO_ID])
    monkeypatch.setattr(huggingface_hub, "scan_cache_dir", scan)
    return scan


@pytest.fixture
def no_gpu(monkeypatch):
    """The rest of /health without a GPU, a loaded model or optional GPU packages."""

    class _Model:
        def is_loaded(self) -> bool:
            return False

    monkeypatch.setattr(health.tts, "get_tts_model", lambda: _Model())
    monkeypatch.setattr(health, "get_backend_type", lambda: "pytorch")
    monkeypatch.setattr(health, "is_amd_gpu_windows", lambda: False)
    monkeypatch.setattr(health.torch.cuda, "is_available", lambda: False)
    monkeypatch.setitem(sys.modules, "intel_extension_for_pytorch", None)
    monkeypatch.setitem(sys.modules, "torch_directml", None)


@pytest.mark.asyncio
async def test_health_polls_within_60_s_scan_the_cache_once(fake_scan, no_gpu):
    first = await health.health()
    second = await health.health()

    assert fake_scan.calls == 1
    assert first.model_downloaded is True
    assert second.model_downloaded is True


@pytest.mark.asyncio
async def test_health_scans_again_after_60_s(fake_scan, no_gpu):
    await health.health()
    scanned_at, result = health._model_downloaded_cache[REPO_ID]
    health._model_downloaded_cache[REPO_ID] = (scanned_at - health.MODEL_DOWNLOADED_CACHE_SECONDS - 1, result)

    await health.health()

    assert fake_scan.calls == 2


@pytest.mark.asyncio
async def test_cache_entry_just_under_60_s_is_still_used(fake_scan):
    health._model_downloaded_cache[REPO_ID] = (time.monotonic() - health.MODEL_DOWNLOADED_CACHE_SECONDS + 5, None)

    assert await health._model_downloaded(REPO_ID) is None
    assert fake_scan.calls == 0


@pytest.mark.asyncio
async def test_scan_runs_off_the_event_loop(monkeypatch, tmp_path):
    # An empty HF cache: if the scan blocked the loop and failed, the fallback must not find the real model.
    monkeypatch.setattr(hf_constants, "HF_HUB_CACHE", str(tmp_path))
    loop_ran = threading.Event()

    def slow_scan():
        # Only returns promptly if the loop keeps running other coroutines while the scan is in progress.
        assert loop_ran.wait(timeout=5)
        return SimpleNamespace(repos=[SimpleNamespace(repo_id=REPO_ID)])

    async def other_coroutine():
        loop_ran.set()

    monkeypatch.setattr(huggingface_hub, "scan_cache_dir", slow_scan)

    downloaded, _ = await asyncio.wait_for(asyncio.gather(health._model_downloaded(REPO_ID), other_coroutine()), 10)

    assert downloaded is True


@pytest.mark.asyncio
async def test_scan_runs_in_a_worker_thread(fake_scan):
    await health._model_downloaded(REPO_ID)

    assert len(fake_scan.threads) == 1
    assert fake_scan.threads[0] != threading.get_ident()


@pytest.mark.asyncio
async def test_repo_missing_from_the_scan_is_unknown_as_before(monkeypatch):
    monkeypatch.setattr(huggingface_hub, "scan_cache_dir", _FakeScan(["other/model"]))

    assert await health._model_downloaded(REPO_ID) is None


@pytest.mark.parametrize(
    ("files", "expected"),
    [
        (["snapshots/abc/model.safetensors"], True),
        (["snapshots/abc/config.json"], False),
        (None, None),
    ],
)
def test_fallback_when_the_scan_fails_looks_for_weights(monkeypatch, tmp_path, files, expected):
    monkeypatch.setattr(huggingface_hub, "scan_cache_dir", _FakeScan([], error=OSError("corrupt cache")))
    monkeypatch.setattr(hf_constants, "HF_HUB_CACHE", str(tmp_path))
    if files is not None:
        repo_dir = tmp_path / ("models--" + REPO_ID.replace("/", "--"))
        for name in files:
            path = repo_dir / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"")

    assert health._scan_model_downloaded(REPO_ID) is expected
