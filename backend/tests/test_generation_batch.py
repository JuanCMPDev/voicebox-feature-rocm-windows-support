"""
Opt-in batching of queued Qwen generations (perf-loop plan, Fase 3.6): what the queue groups into one model call and
what it leaves alone, and the path a batch takes through run_generation and the PyTorch backend.

No GPU and no models: the jobs, the TTS engine and the Qwen model are stubs; the queue, the GPU lock, run_generation,
the routes and the history rows (on a temporary SQLite database) are the real ones.

Usage:
    python -m pytest backend/tests/test_generation_batch.py -q
"""

import asyncio
import contextlib

import numpy as np
import pytest
import pytest_asyncio
import torch
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from backend import backends
from backend.backends import pytorch_backend
from backend.database import Base, Generation as DBGeneration, VoiceProfile as DBVoiceProfile
from backend.models import GenerationRequest
from backend.routes import generations
from backend.services import generation, task_queue
from backend.utils import audio as audio_utils, chunked_tts, perf_log, qwen_end_fix

_HIP = "7.2.53211-158bd99533"
_SAMPLE_RATE = 24000
# Two sentences, 160 characters: one chunk with the default max_chunk_chars (800), two with 100.
_TWO_SENTENCES = "Esta es la primera frase del texto, bastante larga para pasar de cien caracteres. " + (
    "Y esta es la segunda frase, que va en otro trozo si el límite es cien."
)


@pytest_asyncio.fixture
async def gpu_queue():
    """A fresh queue and GPU lock; the worker is stopped afterwards, so a failed test cannot leave it running."""
    task_queue.init_queue(force=True)
    yield
    worker = task_queue._generation_worker_task
    worker.cancel()
    with contextlib.suppress(asyncio.CancelledError, TimeoutError):
        await asyncio.wait_for(worker, timeout=1)


async def _drain(timeout_s: float = 2.0) -> None:
    """Wait until the worker is done with every job enqueued so far, task_done included."""
    await asyncio.wait_for(task_queue._generation_queue.join(), timeout=timeout_s)


async def _until(condition, timeout_s: float = 1.0) -> None:
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout_s
    while not condition():
        assert loop.time() < deadline, "condition not reached"
        await asyncio.sleep(0.005)


class _Runner:
    """``run_requests`` stand-in: records each call and answers every request with ``"<request> audio"``."""

    def __init__(self, error: Exception | None = None):
        self.calls: list[list[str]] = []
        self.error = error

    async def __call__(self, requests: list[str]) -> list[str]:
        self.calls.append(list(requests))
        if self.error is not None:
            raise self.error
        return [f"{request} audio" for request in requests]


def _job(
    log: list[str],
    name: str,
    runner: _Runner,
    *,
    hold: asyncio.Event | None = None,
    error: Exception | None = None,
):
    """Stand-in for run_generation: prepares, then asks its batch for its audio, or generates alone without one."""

    async def run() -> None:
        log.append(f"{name} prepare")
        if hold is not None:
            await hold.wait()
        if error is not None:
            raise error
        batch = task_queue.current_batch()
        if batch is None:
            log.append(f"{name} alone")
            return
        try:
            audio = await batch.generate(name, runner)
        except asyncio.CancelledError:  # run_generation also ends quietly when cancelled
            log.append(f"{name} cancelled")
            return
        except Exception as e:
            log.append(f"{name} failed: {e}")
            return
        log.append(f"{name} got {audio}")

    return run()


def _enqueue(log: list[str], runner: _Runner, name: str, key, **kwargs) -> None:
    task_queue.enqueue_generation(name, _job(log, name, runner, **kwargs), batch_key=key)


@pytest.mark.asyncio
@pytest.mark.usefixtures("gpu_queue")
class TestQueueGrouping:
    async def test_jobs_queued_with_the_same_key_run_as_one_batch(self):
        log, runner = [], _Runner()
        lock = task_queue.gpu_lock()
        await lock.acquire()  # the GPU is busy, so all three wait in the queue
        for name in ("a", "b", "c"):
            _enqueue(log, runner, name, "k")
        lock.release()
        await _drain()

        assert runner.calls == [["a", "b", "c"]]
        # Each job prepares in queue order before the one call, then finishes with its own audio.
        assert log == ["a prepare", "b prepare", "c prepare", "a got a audio", "b got b audio", "c got c audio"]

    async def test_a_job_alone_in_the_queue_runs_without_a_batch(self):
        log, runner = [], _Runner()
        _enqueue(log, runner, "a", "k")
        await _drain()

        assert runner.calls == []
        assert log == ["a prepare", "a alone"]

    async def test_jobs_without_a_key_never_batch(self):
        log, runner = [], _Runner()
        lock = task_queue.gpu_lock()
        await lock.acquire()
        for name in ("a", "b", "c"):
            _enqueue(log, runner, name, None)
        lock.release()
        await _drain()

        assert runner.calls == []
        assert log == ["a prepare", "a alone", "b prepare", "b alone", "c prepare", "c alone"]

    async def test_only_jobs_with_the_same_key_batch_and_the_rest_keep_their_order(self):
        log, runner = [], _Runner()
        lock = task_queue.gpu_lock()
        await lock.acquire()
        for name, key in (("a", "k1"), ("b", "k2"), ("c", "k1"), ("d", None), ("e", "k1"), ("f", "k2")):
            _enqueue(log, runner, name, key)
        lock.release()
        await _drain()

        assert runner.calls == [["a", "c", "e"], ["b", "f"]]
        assert log == [
            "a prepare",
            "c prepare",
            "e prepare",
            "a got a audio",
            "c got c audio",
            "e got e audio",
            "b prepare",
            "f prepare",
            "b got b audio",
            "f got f audio",
            "d prepare",
            "d alone",
        ]

    async def test_a_batch_takes_at_most_max_batch_size_jobs(self):
        log, runner = [], _Runner()
        names = [f"j{i}" for i in range(task_queue.MAX_BATCH_SIZE + 2)]
        lock = task_queue.gpu_lock()
        await lock.acquire()
        for name in names:
            _enqueue(log, runner, name, "k")
        lock.release()
        await _drain()

        assert runner.calls == [names[: task_queue.MAX_BATCH_SIZE], names[task_queue.MAX_BATCH_SIZE :]]

    async def test_job_cancelled_in_the_queue_is_left_out(self):
        log, runner = [], _Runner()
        lock = task_queue.gpu_lock()
        await lock.acquire()
        for name in ("a", "b", "c"):
            _enqueue(log, runner, name, "k")
        assert task_queue.cancel_generation("b") == "queued"
        lock.release()
        await _drain()

        assert runner.calls == [["a", "c"]]
        assert not any(line.startswith("b ") for line in log)

    async def test_jobs_prepare_one_at_a_time_and_stay_cancellable_meanwhile(self):
        log, runner = [], _Runner()
        hold = asyncio.Event()
        lock = task_queue.gpu_lock()
        await lock.acquire()
        _enqueue(log, runner, "a", "k", hold=hold)
        _enqueue(log, runner, "b", "k")
        _enqueue(log, runner, "c", "k")
        lock.release()

        await _until(lambda: "a prepare" in log)
        await asyncio.sleep(0.05)
        assert log == ["a prepare"]  # b does not start (load model, voice prompt) while a is still preparing
        assert task_queue.cancel_generation("b") == "queued"
        hold.set()
        await _drain()

        assert runner.calls == [["a", "c"]]
        assert log == ["a prepare", "c prepare", "a got a audio", "c got c audio"]

    async def test_job_failing_before_generating_does_not_hold_the_batch(self, monkeypatch):
        forced: list[str] = []

        async def force_fail(generation_id: str, error: str) -> None:
            forced.append(generation_id)

        monkeypatch.setattr(task_queue, "_force_fail_if_active", force_fail)
        log, runner = [], _Runner()
        lock = task_queue.gpu_lock()
        await lock.acquire()
        _enqueue(log, runner, "a", "k")
        _enqueue(log, runner, "b", "k", error=RuntimeError("profile has no samples"))
        _enqueue(log, runner, "c", "k")
        lock.release()
        await _drain()

        assert runner.calls == [["a", "c"]]
        assert log == ["a prepare", "b prepare", "c prepare", "a got a audio", "c got c audio"]
        assert forced == ["b"]  # its row is failed by the worker, as for a single job that raises

    async def test_job_cancelled_while_waiting_for_the_batch_is_left_out(self):
        log, runner = [], _Runner()
        hold = asyncio.Event()
        lock = task_queue.gpu_lock()
        await lock.acquire()
        _enqueue(log, runner, "a", "k")
        _enqueue(log, runner, "b", "k")
        _enqueue(log, runner, "c", "k", hold=hold)
        lock.release()

        await _until(lambda: "c prepare" in log)
        assert task_queue.cancel_generation("b") == "running"
        await _until(lambda: "b cancelled" in log)
        hold.set()
        await _drain()

        assert runner.calls == [["a", "c"]]
        assert "a got a audio" in log
        assert "c got c audio" in log
        assert not any(line.startswith("b got") for line in log)

    async def test_failed_batch_call_fails_every_job(self):
        log, runner = [], _Runner(error=RuntimeError("out of memory"))
        lock = task_queue.gpu_lock()
        await lock.acquire()
        _enqueue(log, runner, "a", "k")
        _enqueue(log, runner, "b", "k")
        lock.release()
        await _drain()

        assert log == ["a prepare", "b prepare", "a failed: out of memory", "b failed: out of memory"]

    async def test_queue_still_serves_single_jobs_after_a_batch(self):
        log, runner = [], _Runner()
        lock = task_queue.gpu_lock()
        await lock.acquire()
        _enqueue(log, runner, "a", "k")
        _enqueue(log, runner, "b", "k")
        lock.release()
        await _drain()
        _enqueue(log, runner, "c", "k")
        await _drain()

        assert runner.calls == [["a", "b"]]
        assert log[-2:] == ["c prepare", "c alone"]
        assert not task_queue.gpu_lock().locked()


@pytest.mark.asyncio
class TestGenerationBatch:
    async def _run(self, run_requests, names: list[str]) -> list:
        batch = task_queue.GenerationBatch()
        tasks = [asyncio.create_task(batch.generate(name, run_requests)) for name in names]
        await asyncio.sleep(0)
        await batch.run()
        return await asyncio.gather(*tasks, return_exceptions=True)

    async def test_each_request_gets_its_own_outcome(self):
        async def run_requests(requests):
            return ["a audio", ValueError("b failed alone")]

        a, b = await self._run(run_requests, ["a", "b"])
        assert a == "a audio"
        assert isinstance(b, ValueError)

    async def test_wrong_number_of_results_fails_every_request(self):
        async def run_requests(requests):
            return ["only one"]

        outcomes = await self._run(run_requests, ["a", "b"])
        assert all(isinstance(outcome, RuntimeError) for outcome in outcomes)


class _FakeBatchBackend:
    """TTS engine stand-in with batch support; records how it was asked to generate."""

    def __init__(self, *, supports: bool = True, error: Exception | None = None):
        self.supports = supports
        self.error = error
        self.batch_calls: list[dict] = []

    def is_loaded(self) -> bool:
        return True

    def supports_batch(self) -> bool:
        return self.supports

    async def generate_batch(self, texts, voice_prompts, language="en", seed=None, instruct=None):
        self.batch_calls.append(
            {
                "texts": list(texts),
                "voice_prompts": list(voice_prompts),
                "language": language,
                "seed": seed,
                "instruct": instruct,
                "gen_id": perf_log._generation_id.get(),
            }
        )
        if self.error is not None:
            raise self.error
        return [np.full(4, index, dtype=np.float32) for index in range(len(texts))], _SAMPLE_RATE


class TestBatchKey:
    @pytest.fixture
    def engine(self, monkeypatch):
        engine = _FakeBatchBackend()
        monkeypatch.setattr(backends, "get_tts_backend_for_engine", lambda name: engine)
        monkeypatch.setenv("VOICEBOX_QWEN_BATCH", "1")
        return engine

    @staticmethod
    def _key(**overrides):
        arguments = {
            "engine": "qwen",
            "model_size": "1.7B",
            "profile_id": "profile-1",
            "language": "es",
            "instruct": None,
            "text": "Hola.",
            "max_chunk_chars": None,
        }
        arguments.update(overrides)
        return generation.batch_key(**arguments)

    @pytest.mark.usefixtures("engine")
    def test_key_groups_by_engine_size_profile_language_and_instruct(self):
        assert self._key() == ("qwen", "1.7B", "profile-1", "es", None)
        assert self._key(text="Otra frase.") == self._key()
        assert self._key(profile_id="profile-2") != self._key()
        assert self._key(language="en") != self._key()
        assert self._key(model_size="other-size") != self._key()
        assert self._key(instruct="despacio") != self._key()

    @pytest.mark.usefixtures("engine")
    @pytest.mark.parametrize("value", [None, "", "0", "false", "off"])
    def test_without_the_variable_nothing_batches(self, monkeypatch, value):
        if value is None:
            monkeypatch.delenv("VOICEBOX_QWEN_BATCH", raising=False)
        else:
            monkeypatch.setenv("VOICEBOX_QWEN_BATCH", value)
        assert self._key() is None

    @pytest.mark.usefixtures("engine")
    @pytest.mark.parametrize("value", ["1", "true", " ON ", "yes"])
    def test_variable_turns_batching_on(self, monkeypatch, value):
        monkeypatch.setenv("VOICEBOX_QWEN_BATCH", value)
        assert self._key() is not None

    @pytest.mark.usefixtures("engine")
    @pytest.mark.parametrize("engine_name", ["qwen_custom_voice", "kokoro", "chatterbox", "luxtts"])
    def test_other_engines_never_batch(self, engine_name):
        assert self._key(engine=engine_name) is None

    def test_backend_without_batch_support_never_batches(self, engine, monkeypatch):
        engine.supports = False
        assert self._key() is None
        monkeypatch.setattr(backends, "get_tts_backend_for_engine", lambda name: object())
        assert self._key() is None

    @pytest.mark.usefixtures("engine")
    def test_text_of_several_chunks_never_batches(self):
        assert self._key(text=_TWO_SENTENCES) is not None
        assert self._key(text=_TWO_SENTENCES, max_chunk_chars=100) is None

    @pytest.mark.usefixtures("engine")
    def test_backend_lookup_error_runs_the_job_alone(self, monkeypatch):
        def broken(name):
            raise RuntimeError("no backend")

        monkeypatch.setattr(backends, "get_tts_backend_for_engine", broken)
        assert self._key() is None


def _request(name: str, backend, seed: int | None, *, trim_fn=None) -> generation._BatchRequest:
    gen_kwargs = {"language": "es", "seed": seed, "instruct": None, "trim_fn": trim_fn}
    return generation._BatchRequest(name, backend, f"texto {name}", [f"prompt {name}"], gen_kwargs)


class TestGenerateTogether:
    @pytest.fixture
    def single_calls(self, monkeypatch) -> list[dict]:
        """Every ``generate_chunked`` call, i.e. every job generated on its own."""
        calls: list[dict] = []

        async def generate_chunked(backend, text, voice_prompt, **kwargs):
            calls.append(
                {
                    "backend": backend,
                    "text": text,
                    "voice_prompt": voice_prompt,
                    "kwargs": kwargs,
                    "gen_id": perf_log._generation_id.get(),
                }
            )
            if text == "texto broken":
                raise RuntimeError("broken alone")
            return np.zeros(2, dtype=np.float32), _SAMPLE_RATE

        monkeypatch.setattr(chunked_tts, "generate_chunked", generate_chunked)
        return calls

    def test_single_request_takes_the_single_job_call(self, single_calls):
        backend = _FakeBatchBackend()
        request = _request("a", backend, 11)

        outcomes = asyncio.run(generation._generate_together([request]))

        assert backend.batch_calls == []
        assert single_calls == [
            {
                "backend": backend,
                "text": "texto a",
                "voice_prompt": ["prompt a"],
                "kwargs": request.gen_kwargs,
                "gen_id": "a",
            }
        ]
        assert len(outcomes) == 1
        assert outcomes[0][1] == _SAMPLE_RATE

    def test_several_requests_take_one_batch_call_seeded_with_the_first(self, single_calls):
        backend = _FakeBatchBackend()

        outcomes = asyncio.run(generation._generate_together([_request("a", backend, 11), _request("b", backend, 12)]))

        assert single_calls == []
        assert backend.batch_calls == [
            {
                "texts": ["texto a", "texto b"],
                "voice_prompts": [["prompt a"], ["prompt b"]],
                "language": "es",
                "seed": 11,
                "instruct": None,
                "gen_id": "a,b",
            }
        ]
        assert [audio.tolist() for audio, _ in outcomes] == [[0.0] * 4, [1.0] * 4]
        assert all(sample_rate == _SAMPLE_RATE for _, sample_rate in outcomes)
        assert perf_log._generation_id.get() is None

    def test_batch_output_goes_through_trim(self, single_calls):
        backend = _FakeBatchBackend()

        def trim(audio, sample_rate):
            return audio[:2]

        requests = [_request("a", backend, 11, trim_fn=trim), _request("b", backend, 12, trim_fn=trim)]
        outcomes = asyncio.run(generation._generate_together(requests))

        assert [len(audio) for audio, _ in outcomes] == [2, 2]

    def test_failed_batch_call_generates_each_job_alone(self, single_calls):
        backend = _FakeBatchBackend(error=RuntimeError("out of memory"))
        requests = [_request("a", backend, 11), _request("broken", backend, 12), _request("c", backend, None)]

        outcomes = asyncio.run(generation._generate_together(requests))

        assert len(backend.batch_calls) == 1
        assert [(call["text"], call["kwargs"]["seed"], call["gen_id"]) for call in single_calls] == [
            ("texto a", 11, "a"),
            ("texto broken", 12, "broken"),
            ("texto c", None, "c"),
        ]
        assert outcomes[0][1] == _SAMPLE_RATE
        assert isinstance(outcomes[1], RuntimeError)
        assert outcomes[2][1] == _SAMPLE_RATE

    def test_backend_without_batch_generates_each_job_alone(self, single_calls):
        class SingleOnly:
            pass

        backend = SingleOnly()
        outcomes = asyncio.run(generation._generate_together([_request("a", backend, 1), _request("b", backend, 2)]))

        assert [call["text"] for call in single_calls] == ["texto a", "texto b"]
        assert len(outcomes) == 2


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
def stub_engine(monkeypatch, session_factory):
    """Run the real routes and run_generation on a fake batch-capable engine.

    Records each row's status updates and every job generated on its own (``generate_chunked``).
    """
    engine = _FakeBatchBackend()
    engine.statuses = []
    engine.single_calls = []
    real_update = generation.history.update_generation_status

    async def update_generation_status(generation_id, status, db, **kwargs):
        engine.statuses.append((generation_id, status))
        return await real_update(generation_id, status, db, **kwargs)

    async def load_engine_model(engine_name, model_size):
        return None

    async def create_voice_prompt_for_profile(profile_id, db, **kwargs):
        return [f"prompt of {profile_id}"]

    async def generate_chunked(model, text, voice_prompt, **kwargs):
        engine.single_calls.append({"text": text, "seed": kwargs.get("seed")})
        return np.zeros(2400, dtype=np.float32), _SAMPLE_RATE

    monkeypatch.setattr(generation.history, "update_generation_status", update_generation_status)
    monkeypatch.setattr(backends, "get_tts_backend_for_engine", lambda name: engine)
    monkeypatch.setattr(backends, "load_engine_model", load_engine_model)
    monkeypatch.setattr(backends, "engine_needs_trim", lambda name: False)
    monkeypatch.setattr(generation.profiles, "create_voice_prompt_for_profile", create_voice_prompt_for_profile)
    monkeypatch.setattr(chunked_tts, "generate_chunked", generate_chunked)
    monkeypatch.setattr(audio_utils, "normalize_audio", lambda audio: audio)
    monkeypatch.setattr(generation, "get_db", lambda: iter([session_factory()]))
    monkeypatch.setattr(generation, "_save_generate", lambda **kwargs: "generations/gen.wav")
    monkeypatch.setattr(generation, "_notify_speak_end", lambda *args, **kwargs: None)
    return engine


def _status(db, generation_id: str) -> str:
    db.expire_all()
    return db.query(DBGeneration).filter_by(id=generation_id).one().status


async def _post_while_gpu_busy(session_factory, requests: list[GenerationRequest]) -> list[str]:
    """POST /generate for each request while the GPU is busy, then free it; returns the ids once all are done."""
    lock = task_queue.gpu_lock()
    await lock.acquire()
    with session_factory() as db:
        ids = [(await generations.generate_speech(request, db)).id for request in requests]
        lock.release()
        await _until(lambda: all(_status(db, gen_id) in ("completed", "failed") for gen_id in ids), timeout_s=2)
        assert [_status(db, gen_id) for gen_id in ids] == ["completed"] * len(ids)
    return ids


def _generation_request(text: str, seed: int, **kwargs) -> GenerationRequest:
    return GenerationRequest(profile_id="profile-1", text=text, language="es", seed=seed, **kwargs)


@pytest.mark.asyncio
@pytest.mark.usefixtures("gpu_queue")
class TestRoutes:
    async def test_requests_queued_together_take_one_model_call(self, stub_engine, session_factory, monkeypatch):
        monkeypatch.setenv("VOICEBOX_QWEN_BATCH", "1")
        texts = ["Hola.", "Buenos días.", "Adiós."]

        ids = await _post_while_gpu_busy(
            session_factory, [_generation_request(text, seed) for text, seed in zip(texts, (11, 12, 13), strict=True)]
        )

        assert stub_engine.single_calls == []
        assert stub_engine.batch_calls == [
            {
                "texts": texts,
                "voice_prompts": [["prompt of profile-1"]] * 3,
                "language": "es",
                "seed": 11,
                "instruct": None,
                "gen_id": ",".join(ids),
            }
        ]
        for gen_id in ids:
            assert [status for row, status in stub_engine.statuses if row == gen_id] == ["generating", "completed"]

    async def test_without_the_variable_each_request_runs_alone(self, stub_engine, session_factory, monkeypatch):
        monkeypatch.delenv("VOICEBOX_QWEN_BATCH", raising=False)
        texts, seeds = ["Hola.", "Buenos días.", "Adiós."], [11, 12, 13]

        await _post_while_gpu_busy(
            session_factory, [_generation_request(t, s) for t, s in zip(texts, seeds, strict=True)]
        )

        assert stub_engine.batch_calls == []
        assert stub_engine.single_calls == [{"text": t, "seed": s} for t, s in zip(texts, seeds, strict=True)]

    async def test_engine_without_batch_support_runs_each_request_alone(
        self, stub_engine, session_factory, monkeypatch
    ):
        monkeypatch.setenv("VOICEBOX_QWEN_BATCH", "1")
        stub_engine.supports = False

        await _post_while_gpu_busy(session_factory, [_generation_request("Hola.", 1), _generation_request("Sí.", 2)])

        assert stub_engine.batch_calls == []
        assert [call["text"] for call in stub_engine.single_calls] == ["Hola.", "Sí."]

    async def test_text_of_several_chunks_runs_alone(self, stub_engine, session_factory, monkeypatch):
        monkeypatch.setenv("VOICEBOX_QWEN_BATCH", "1")

        await _post_while_gpu_busy(
            session_factory,
            [
                _generation_request("Hola.", 1),
                _generation_request(_TWO_SENTENCES, 2, max_chunk_chars=100),
                _generation_request("Adiós.", 3),
            ],
        )

        assert [call["texts"] for call in stub_engine.batch_calls] == [["Hola.", "Adiós."]]
        assert stub_engine.single_calls == [{"text": _TWO_SENTENCES, "seed": 2}]


class FakeVoiceCloneModel:
    """``Qwen3TTSModel`` stand-in: records each ``generate_voice_clone`` call and returns one wave per text."""

    def __init__(self):
        self.calls: list[dict] = []

    def generate_voice_clone(self, **kwargs):
        self.calls.append({**kwargs, "inference_mode": torch.is_inference_mode_enabled()})
        return [np.full(4, index, dtype=np.float32) for index in range(len(kwargs["text"]))], _SAMPLE_RATE


def _loaded_backend(device: str) -> pytorch_backend.PyTorchTTSBackend:
    backend = pytorch_backend.PyTorchTTSBackend.__new__(pytorch_backend.PyTorchTTSBackend)
    backend.model = FakeVoiceCloneModel()
    backend.model_size = backend._current_model_size = "1.7B"
    backend.device = device
    return backend


class TestPyTorchBackend:
    @pytest.fixture(autouse=True)
    def _no_variant(self, monkeypatch):
        monkeypatch.delenv("VOICEBOX_BACKEND_VARIANT", raising=False)
        monkeypatch.delenv(qwen_end_fix.END_FIX_ENV, raising=False)

    @pytest.mark.parametrize(
        ("hip", "variant", "device", "expected"),
        [
            (_HIP, None, "cuda", True),
            (None, "rocm", "cuda", True),
            (None, None, "cuda", False),
            (None, "cuda", "cuda", False),
            (_HIP, None, "cpu", False),
            (None, "rocm", "cpu", False),
            (_HIP, None, "xpu", False),
        ],
    )
    def test_batches_only_on_a_rocm_gpu(self, monkeypatch, hip, variant, device, expected):
        monkeypatch.setattr(torch.version, "hip", hip)
        if variant is not None:
            monkeypatch.setenv("VOICEBOX_BACKEND_VARIANT", variant)
        assert _loaded_backend(device).supports_batch() is expected

    @pytest.fixture
    def seeds(self, monkeypatch) -> list[tuple]:
        calls: list[tuple] = []
        monkeypatch.setattr(pytorch_backend, "manual_seed", lambda seed, device: calls.append((seed, device)))
        return calls

    def test_generate_batch_makes_one_call_with_lists(self, monkeypatch, seeds):
        monkeypatch.setattr(torch.version, "hip", _HIP)
        backend = _loaded_backend("cuda")

        wavs, sample_rate = asyncio.run(
            backend.generate_batch(["Hola.", "Adiós."], [["item a"], ["item b"]], language="es", seed=7)
        )

        assert backend.model.calls == [
            {
                "text": ["Hola.", "Adiós."],
                "voice_clone_prompt": ["item a", "item b"],
                "language": "spanish",
                "instruct": None,
                "non_streaming_mode": True,  # perf-loop task 4.4 (backend/utils/qwen_end_fix.py)
                "inference_mode": True,
            }
        ]
        assert seeds == [(7, "cuda")]  # seeded once for the whole batch
        assert [wav.tolist() for wav in wavs] == [[0.0] * 4, [1.0] * 4]
        assert sample_rate == _SAMPLE_RATE
        assert not torch.is_inference_mode_enabled()

    def test_generate_batch_without_seed_leaves_the_rng_alone(self, monkeypatch, seeds):
        monkeypatch.setattr(torch.version, "hip", _HIP)
        asyncio.run(_loaded_backend("cuda").generate_batch(["Hola.", "Sí."], [["a"], ["b"]], language="es"))
        assert seeds == []

    @pytest.mark.parametrize("prompts", [[["a", "b"], ["c"]], [{}, ["c"]], [[], ["c"]]])
    def test_generate_batch_needs_one_prompt_item_per_text(self, seeds, prompts):
        backend = _loaded_backend("cuda")
        with pytest.raises(ValueError, match="one voice prompt item per text"):
            asyncio.run(backend.generate_batch(["Hola.", "Sí."], prompts, language="es", seed=1))
        assert backend.model.calls == []
        assert seeds == []
