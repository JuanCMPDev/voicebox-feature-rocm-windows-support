"""One GPU turn for queued generations and /transcribe, and the queued -> generating -> completed life of a generation.

No GPU and no models: the generation jobs, Whisper and the TTS engine are stubs; the queue, the lock, the routes and
the history rows are the real ones (on a temporary SQLite database).

Usage:
    python -m pytest backend/tests/test_gpu_queue.py -q
"""

import asyncio
import contextlib
import io

import numpy as np
import pytest
import pytest_asyncio
from fastapi import HTTPException, UploadFile
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from backend import backends
from backend.database import Base, Generation as DBGeneration, VoiceProfile as DBVoiceProfile
from backend.models import GenerationRequest
from backend.routes import generations, transcription
from backend.services import generation, task_queue
from backend.utils import audio as audio_utils, chunked_tts


@pytest_asyncio.fixture
async def gpu_queue():
    """A fresh queue and GPU lock; the worker is stopped afterwards, so a failed test cannot leave it running."""
    task_queue.init_queue(force=True)
    yield
    worker = task_queue._generation_worker_task
    worker.cancel()
    with contextlib.suppress(asyncio.CancelledError, TimeoutError):
        await asyncio.wait_for(worker, timeout=1)


def _job(
    events: list[str],
    name: str,
    *,
    started: asyncio.Event | None = None,
    release: asyncio.Event | None = None,
    finished: asyncio.Event | None = None,
):
    """Stand-in for run_generation: logs when it starts and ends, and can be held between the two."""

    async def run() -> None:
        events.append(f"{name} start")
        if started is not None:
            started.set()
        if release is not None:
            await release.wait()
        events.append(f"{name} end")
        if finished is not None:
            finished.set()

    return run()


class _FakeWhisper:
    model_size = "base"

    def __init__(self):
        self.events: list[str] = []
        self.error: Exception | None = None
        self.reached_gpu = asyncio.Event()

    def is_loaded(self) -> bool:
        return True

    async def transcribe(self, audio_path: str, language: str | None, model_size: str) -> str:
        self.events.append("transcribe start")
        await asyncio.sleep(0)
        if self.error is not None:
            raise self.error
        self.events.append("transcribe end")
        return "hola"


@pytest.fixture
def whisper(monkeypatch):
    """Patch /transcribe's Whisper; ``reached_gpu`` is set right before the route asks for the GPU."""
    fake = _FakeWhisper()

    def get_whisper_model() -> _FakeWhisper:
        fake.reached_gpu.set()
        return fake

    monkeypatch.setattr(transcription.transcribe, "get_whisper_model", get_whisper_model)
    monkeypatch.setattr(audio_utils, "load_audio", lambda path, **kwargs: (np.zeros(1600, dtype=np.float32), 16000))
    return fake


def _transcribe_request():
    upload = UploadFile(file=io.BytesIO(b"RIFF fake wav"), filename="reference.wav")
    return transcription.transcribe_audio(file=upload, language=None, model=None)


@pytest.mark.asyncio
@pytest.mark.usefixtures("gpu_queue")
async def test_transcription_waits_for_the_running_generation_only(whisper):
    events = whisper.events
    first_started, release_first, second_finished = asyncio.Event(), asyncio.Event(), asyncio.Event()
    task_queue.enqueue_generation("gen-1", _job(events, "gen-1", started=first_started, release=release_first))
    task_queue.enqueue_generation("gen-2", _job(events, "gen-2", finished=second_finished))
    await asyncio.wait_for(first_started.wait(), timeout=1)

    request = asyncio.create_task(_transcribe_request())
    await asyncio.wait_for(whisper.reached_gpu.wait(), timeout=1)
    await asyncio.sleep(0.05)
    assert events == ["gen-1 start"]  # the transcription does not share the GPU with the running generation

    release_first.set()
    response = await asyncio.wait_for(request, timeout=1)
    await asyncio.wait_for(second_finished.wait(), timeout=1)

    assert response.text == "hola"
    # It runs as soon as gen-1 ends, ahead of gen-2, which was queued before it arrived.
    assert events == ["gen-1 start", "gen-1 end", "transcribe start", "transcribe end", "gen-2 start", "gen-2 end"]


@pytest.mark.asyncio
@pytest.mark.usefixtures("gpu_queue")
async def test_transcription_with_an_idle_gpu_runs_at_once(whisper):
    response = await asyncio.wait_for(_transcribe_request(), timeout=1)

    assert response.text == "hola"
    assert not task_queue.gpu_lock().locked()


@pytest.mark.asyncio
@pytest.mark.usefixtures("gpu_queue")
async def test_failed_transcription_hands_the_gpu_back(whisper):
    whisper.error = RuntimeError("whisper exploded")
    finished = asyncio.Event()

    with pytest.raises(HTTPException) as excinfo:
        await asyncio.wait_for(_transcribe_request(), timeout=1)
    task_queue.enqueue_generation("gen-1", _job(whisper.events, "gen-1", finished=finished))
    await asyncio.wait_for(finished.wait(), timeout=1)

    assert excinfo.value.status_code == 500
    assert whisper.events == ["transcribe start", "gen-1 start", "gen-1 end"]


@pytest.mark.asyncio
@pytest.mark.usefixtures("gpu_queue")
async def test_generation_cancelled_while_waiting_for_the_gpu_never_runs():
    events: list[str] = []
    finished = asyncio.Event()
    lock = task_queue.gpu_lock()
    await lock.acquire()  # a transcription holds the GPU

    task_queue.enqueue_generation("gen-1", _job(events, "gen-1"))
    task_queue.enqueue_generation("gen-2", _job(events, "gen-2", finished=finished))
    await asyncio.sleep(0.05)
    assert events == []
    assert task_queue.cancel_generation("gen-1") == "queued"

    lock.release()
    await asyncio.wait_for(finished.wait(), timeout=1)

    assert events == ["gen-2 start", "gen-2 end"]


class _LoadedBackend:
    def is_loaded(self) -> bool:
        return True


@pytest.fixture
def session_factory(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'test.db'}", connect_args={"check_same_thread": False})
    Base.metadata.create_all(bind=engine)
    factory = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    with factory() as db:
        db.add(DBVoiceProfile(id="profile-1", name="Test Profile"))
        db.commit()
    yield factory
    engine.dispose()


@pytest.fixture
def statuses(monkeypatch, session_factory):
    """Run the real run_generation on a stub engine and record every status update of the history row."""
    seen: list[str] = []
    real_update = generation.history.update_generation_status

    async def update_generation_status(generation_id, status, db, **kwargs):
        seen.append(status)
        return await real_update(generation_id, status, db, **kwargs)

    async def load_engine_model(engine, model_size):
        return None

    async def create_voice_prompt_for_profile(*args, **kwargs):
        return {}

    async def generate_chunked(model, text, voice_prompt, **kwargs):
        return np.zeros(2400, dtype=np.float32), 24000

    monkeypatch.setattr(generation.history, "update_generation_status", update_generation_status)
    monkeypatch.setattr(backends, "get_tts_backend_for_engine", lambda engine: _LoadedBackend())
    monkeypatch.setattr(backends, "load_engine_model", load_engine_model)
    monkeypatch.setattr(backends, "engine_needs_trim", lambda engine: False)
    monkeypatch.setattr(generation.profiles, "create_voice_prompt_for_profile", create_voice_prompt_for_profile)
    monkeypatch.setattr(chunked_tts, "generate_chunked", generate_chunked)
    monkeypatch.setattr(audio_utils, "normalize_audio", lambda audio: audio)
    monkeypatch.setattr(generation, "get_db", lambda: iter([session_factory()]))
    monkeypatch.setattr(generation, "_save_generate", lambda **kwargs: "generations/gen.wav")
    monkeypatch.setattr(generation, "_notify_speak_end", lambda *args, **kwargs: None)
    return seen


def _status(db, generation_id: str) -> str:
    db.expire_all()
    return db.query(DBGeneration).filter_by(id=generation_id).one().status


async def _wait_terminal(db, generation_id: str, timeout_s: float = 2.0) -> str:
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout_s
    while (status := _status(db, generation_id)) not in ("completed", "failed") and loop.time() < deadline:
        await asyncio.sleep(0.01)
    return status


def _request() -> GenerationRequest:
    return GenerationRequest(profile_id="profile-1", text="Hola.", language="es")


@pytest.mark.asyncio
@pytest.mark.usefixtures("gpu_queue")
async def test_generation_row_goes_queued_generating_completed(statuses, session_factory):
    lock = task_queue.gpu_lock()
    await lock.acquire()  # the GPU is busy, so the new generation waits for its turn

    with session_factory() as db:
        response = await generations.generate_speech(_request(), db)
        await asyncio.sleep(0.05)
        assert response.status == "queued"
        assert _status(db, response.id) == "queued"
        assert statuses == []

        lock.release()
        assert await _wait_terminal(db, response.id) == "completed"

    assert statuses == ["generating", "completed"]


@pytest.mark.asyncio
@pytest.mark.usefixtures("gpu_queue")
async def test_queued_generation_can_be_cancelled(statuses, session_factory):
    lock = task_queue.gpu_lock()
    await lock.acquire()

    with session_factory() as db:
        response = await generations.generate_speech(_request(), db)
        result = await generations.cancel_generation(response.id, db)
        lock.release()
        await asyncio.sleep(0.05)
        status = _status(db, response.id)

    assert result == {"message": "Queued generation cancelled"}
    assert status == "failed"
    assert statuses == ["failed"]  # written by the cancel route; the job itself never ran
