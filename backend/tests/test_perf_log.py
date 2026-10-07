"""
Tests for backend/utils/perf_log.py, the VBPERF per-stage instrumentation (perf-loop/contracts.md, section 4).

Everything runs without a GPU: a fake clock and a fake GPU synchronizer replace ``perf_counter`` and
``torch.cuda.synchronize``, and the Qwen model is a stand-in with the same attribute layout.

Usage:
    python -m pytest backend/tests/test_perf_log.py -v
"""

import asyncio
import importlib.util
import io
import json
import logging
from pathlib import Path

import numpy as np
import pytest
import torch

from backend.utils import perf_log

_BENCH = Path(__file__).resolve().parents[2] / "scripts" / "bench_tts.py"
_spec = importlib.util.spec_from_file_location("bench_tts_for_perf_log", _BENCH)
bench_tts = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(bench_tts)


class FakeTiming:
    """Clock that advances ``step_s`` per reading and a synchronizer; both record the call order."""

    def __init__(self, step_s: float = 0.25):
        self.step_s = step_s
        self.now = 100.0
        self.calls: list[str] = []

    def clock(self) -> float:
        self.calls.append("clock")
        self.now += self.step_s
        return self.now

    def sync(self) -> None:
        self.calls.append("sync")


@pytest.fixture
def enabled(monkeypatch):
    monkeypatch.setenv(perf_log.PERF_LOG_ENV, "1")


@pytest.fixture
def timing(monkeypatch):
    fake = FakeTiming()
    monkeypatch.setattr(perf_log, "perf_counter", fake.clock)
    monkeypatch.setattr(perf_log, "cuda_synchronize", fake.sync)
    return fake


@pytest.fixture
def lines(caplog):
    caplog.set_level(logging.INFO, logger=perf_log.logger.name)

    def _events() -> list[dict]:
        out = []
        for record in caplog.records:
            message = record.getMessage()
            if record.name == perf_log.logger.name and message.startswith(perf_log.LINE_PREFIX):
                out.append(json.loads(message[len(perf_log.LINE_PREFIX) :]))
        return out

    return _events


class TestLineFormat:
    def test_prefix_and_single_line_json(self):
        payload = {"gen_id": "g1", "stage": "codec", "ms": 1.5, "path": "C:\\x\ny", "dtype": torch.bfloat16}
        line = perf_log.format_line(payload)
        assert line.startswith("VBPERF {")
        assert "\n" not in line
        decoded = json.loads(line[len("VBPERF ") :])
        assert decoded == {**payload, "dtype": "torch.bfloat16"}
        assert list(decoded) == ["gen_id", "stage", "ms", "path", "dtype"]

    def test_emit_tags_the_current_generation(self, lines):
        perf_log.emit("normalize", ms=2.0)
        token = perf_log.set_generation_id("gen-7")
        try:
            perf_log.emit("save", ms=3.0)
        finally:
            perf_log.reset_generation_id(token)
        assert lines() == [
            {"gen_id": None, "stage": "normalize", "ms": 2.0},
            {"gen_id": "gen-7", "stage": "save", "ms": 3.0},
        ]

    def test_env_line_has_no_generation_id(self, lines):
        token = perf_log.set_generation_id("gen-7")
        try:
            perf_log.emit_env({"torch": "2.9.1", "hip": "7.2.1"})
        finally:
            perf_log.reset_generation_id(token)
        assert lines() == [{"stage": "env", "torch": "2.9.1", "hip": "7.2.1"}]


class TestStageTimer:
    def test_syncs_before_each_mark_and_logs_ms(self, lines):
        fake = FakeTiming(step_s=0.25)
        timer = perf_log.StageTimer("codec", clock=fake.clock, sync=fake.sync, enabled=True)
        with timer:
            assert fake.calls == ["sync", "clock"]
            timer.fields.update(T=369, chunks=2)
        assert fake.calls == ["sync", "clock", "sync", "clock"]
        assert timer.ms == pytest.approx(250.0)
        assert lines() == [{"gen_id": None, "stage": "codec", "ms": 250.0, "T": 369, "chunks": 2}]

    def test_steps_yield_steps_per_s(self, lines):
        fake = FakeTiming(step_s=9.1)
        timer = perf_log.StageTimer("talker", clock=fake.clock, sync=fake.sync, enabled=True).start()
        assert timer.stop(steps=39) == pytest.approx(9100.0)
        assert lines() == [{"gen_id": None, "stage": "talker", "ms": 9100.0, "steps": 39, "steps_per_s": 4.286}]

    def test_unknown_steps_have_no_rate(self, lines):
        fake = FakeTiming()
        with perf_log.StageTimer("talker", clock=fake.clock, sync=fake.sync, enabled=True) as timer:
            timer.fields["steps"] = None
        assert lines() == [{"gen_id": None, "stage": "talker", "ms": 250.0, "steps": None}]

    def test_failed_block_logs_nothing(self, lines):
        fake = FakeTiming()
        with (
            pytest.raises(RuntimeError),
            perf_log.StageTimer("save", clock=fake.clock, sync=fake.sync, enabled=True),
        ):
            raise RuntimeError("disk full")
        assert lines() == []

    def test_stop_without_start_is_a_no_op(self, lines):
        fake = FakeTiming()
        timer = perf_log.StageTimer("save", clock=fake.clock, sync=fake.sync, enabled=True)
        assert timer.stop() is None
        assert fake.calls == []
        assert lines() == []

    @pytest.mark.parametrize("value", [None, "", "0", "true"])
    def test_off_unless_variable_is_1(self, monkeypatch, timing, lines, value):
        if value is None:
            monkeypatch.delenv(perf_log.PERF_LOG_ENV, raising=False)
        else:
            monkeypatch.setenv(perf_log.PERF_LOG_ENV, value)
        assert not perf_log.is_enabled()
        with perf_log.stage("total") as timer:
            timer.fields["x"] = 1
        assert timer.stop() is None
        assert timing.calls == []
        assert lines() == []

    def test_stage_uses_module_clock_and_sync_when_enabled(self, enabled, timing, lines):
        with perf_log.stage("normalize"):
            pass
        assert timing.calls == ["sync", "clock", "sync", "clock"]
        assert lines() == [{"gen_id": None, "stage": "normalize", "ms": 250.0}]


@pytest.mark.asyncio
async def test_generation_id_reaches_worker_threads(enabled, timing, lines):
    def work():
        with perf_log.stage("talker") as timer:
            timer.fields["steps"] = 10

    token = perf_log.set_generation_id("gen-thread")
    try:
        await asyncio.to_thread(work)
    finally:
        perf_log.reset_generation_id(token)
    await asyncio.to_thread(work)
    assert [e["gen_id"] for e in lines()] == ["gen-thread", None]


class FakeDecoder(torch.nn.Module):
    def forward(self, codes):
        return codes


class FakeTokenizer:
    """Mimics Qwen3TTSTokenizer.decode: chunked_decode runs ``model.decoder`` once per 300 steps."""

    def __init__(self):
        self.model = type("Codec", (), {})()
        self.model.decoder = FakeDecoder()
        self.model.dtype = torch.bfloat16
        self.result = (["wav"], 24000)

    def decode(self, encoded):
        for item in encoded:
            codes = item["audio_codes"]
            for start in range(0, codes.shape[0], 300):
                self.model.decoder(codes[start : start + 300])
        return self.result


class FakeInner:
    def __init__(self):
        self.speech_tokenizer = FakeTokenizer()
        self.config = type("Config", (), {"_attn_implementation": "sdpa"})()
        self.talker = type("Talker", (), {})()
        self.talker.config = type("Config", (), {"_attn_implementation": "eager"})()
        self.calls: list[tuple] = []
        self.result = ([torch.zeros(39, 16, dtype=torch.long)], [torch.zeros(39, 8)])

    def generate(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        return self.result


class FakeQwenModel:
    def __init__(self):
        self.model = FakeInner()


class TestQwenInstrumentation:
    def test_talker_and_codec_lines_with_untouched_calls(self, enabled, timing, lines):
        model = FakeQwenModel()
        inner, tokenizer = model.model, model.model.speech_tokenizer
        perf_log.instrument_qwen_model(model)

        token = perf_log.set_generation_id("gen-1")
        try:
            out = inner.generate(input_ids=["ids"], languages=["Spanish"])
            codes = torch.cat([torch.zeros(330, 16, dtype=torch.long), out[0][0]])
            wavs = tokenizer.decode([{"audio_codes": codes}])
        finally:
            perf_log.reset_generation_id(token)

        assert out is inner.result
        assert wavs is tokenizer.result
        assert inner.calls == [((), {"input_ids": ["ids"], "languages": ["Spanish"]})]
        assert lines() == [
            {"gen_id": "gen-1", "stage": "talker", "ms": 250.0, "steps": 39, "steps_per_s": 156.0},
            {"gen_id": "gen-1", "stage": "codec", "ms": 250.0, "T": 369, "chunks": 2},
        ]

    def test_each_decode_counts_only_its_own_chunks(self, enabled, timing, lines):
        model = FakeQwenModel()
        perf_log.instrument_qwen_model(model)
        tokenizer = model.model.speech_tokenizer
        tokenizer.decode([{"audio_codes": torch.zeros(650, 16, dtype=torch.long)}])
        tokenizer.decode([{"audio_codes": torch.zeros(32, 16, dtype=torch.long)}])
        assert [(e["T"], e["chunks"]) for e in lines()] == [(650, 3), (32, 1)]

    def test_wrappers_are_inert_when_variable_is_cleared(self, monkeypatch, enabled, timing, lines):
        model = FakeQwenModel()
        perf_log.instrument_qwen_model(model)
        monkeypatch.delenv(perf_log.PERF_LOG_ENV)
        assert model.model.generate() is model.model.result
        assert timing.calls == []
        assert lines() == []

    def test_unexpected_shapes_give_null_fields(self, enabled, timing, lines):
        model = FakeQwenModel()
        model.model.result = (None, None)
        model.model.speech_tokenizer.model.decoder = None
        model.model.speech_tokenizer.decode = lambda encoded: (["wav"], 24000)
        perf_log.instrument_qwen_model(model)
        model.model.generate()
        model.model.speech_tokenizer.decode({"audio_codes": torch.zeros(1, 4, 16)})
        assert lines() == [
            {"gen_id": None, "stage": "talker", "ms": 250.0, "steps": None},
            {"gen_id": None, "stage": "codec", "ms": 250.0, "T": None, "chunks": None},
        ]

    def test_model_without_talker_or_codec_is_left_alone(self, enabled, caplog):
        caplog.set_level(logging.WARNING, logger=perf_log.logger.name)
        model = type("Bare", (), {"model": object()})()
        perf_log.instrument_qwen_model(model)
        assert "VBPERF: model has no talker generate / codec decode" in caplog.text


class TestEnvInfo:
    def test_miopen_env_lists_known_and_set_variables(self):
        environ = {"MIOPEN_FIND_MODE": "FAST", "MIOPEN_ENABLE_LOGGING_CMD": "1", "PATH": "C:\\bin"}
        env = perf_log.miopen_env(environ)
        assert set(perf_log.MIOPEN_KNOWN_VARS) <= set(env)
        assert env["MIOPEN_FIND_MODE"] == "FAST"
        assert env["MIOPEN_USER_DB_PATH"] is None
        assert env["MIOPEN_ENABLE_LOGGING_CMD"] == "1"
        assert "PATH" not in env

    @pytest.mark.parametrize(
        ("environ", "platform", "expected"),
        [
            ({"MIOPEN_USER_DB_PATH": "D:\\cache\\miopen"}, "win32", str(Path("D:\\cache\\miopen"))),
            ({"MIOPEN_USER_DB_PATH": "~/miopen"}, "linux", str(Path.home() / "miopen")),
            ({"MIOPEN_USER_DB_PATH": "rel/db"}, "win32", str(Path.cwd() / "rel" / "db")),
            ({}, "win32", str(Path.home() / ".miopen" / "db")),
            ({"MIOPEN_USER_DB_PATH": ""}, "linux", str(Path.home() / ".config" / "miopen")),
        ],
    )
    def test_miopen_user_db_path(self, environ, platform, expected):
        assert perf_log.miopen_user_db_path(environ, platform=platform) == expected

    @pytest.mark.parametrize("hip", ["7.2.53211", None])
    def test_qwen_env_info(self, monkeypatch, hip):
        monkeypatch.setattr(torch.version, "hip", hip)
        monkeypatch.setenv("MIOPEN_FIND_MODE", "FAST")
        monkeypatch.setenv("MIOPEN_USER_DB_PATH", "D:\\db")
        info = perf_log.qwen_env_info(FakeQwenModel(), "cuda")
        assert list(info)[:10] == [
            "torch",
            "hip",
            "cudnn_benchmark",
            "miopen_immediate",
            "attn_implementation",
            "attn_implementation_talker",
            "codec_dtype",
            "miopen_env",
            "miopen_user_db_path",
            "code_path",
        ]
        assert info["torch"] == torch.__version__
        assert info["hip"] == hip
        assert info["cudnn_benchmark"] is False
        assert info["attn_implementation"] == "sdpa"
        assert info["attn_implementation_talker"] == "eager"
        assert info["codec_dtype"] == "torch.bfloat16"
        assert info["miopen_env"]["MIOPEN_FIND_MODE"] == "FAST"
        assert info["miopen_user_db_path"] == (str(Path("D:\\db")) if hip else None)
        assert Path(info["code_path"]) == Path(perf_log.__file__).resolve().parents[1]
        assert info["device"] == "cuda"
        json.loads(perf_log.format_line({"stage": "env", **info})[len("VBPERF ") :])

    def test_env_info_tolerates_a_bare_model(self):
        info = perf_log.qwen_env_info(object(), "cpu")
        assert info["attn_implementation"] is None
        assert info["codec_dtype"] is None


def test_bench_reads_the_emitted_lines(caplog, enabled, timing):
    """The lines, as the server's colored log handler writes them, parse with scripts/bench_tts.py."""
    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    handler.setFormatter(logging.Formatter("\x1b[32m%(levelname)s\x1b[0m:     %(message)s"))
    caplog.set_level(logging.INFO, logger=perf_log.logger.name)
    perf_log.logger.addHandler(handler)
    try:
        model = FakeQwenModel()
        perf_log.instrument_qwen_model(model)
        perf_log.emit_env(perf_log.qwen_env_info(model, "cuda"))
        token = perf_log.set_generation_id("gen-9")
        try:
            with perf_log.stage("reference") as timer:
                timer.fields["cached"] = False
            model.model.generate()
            model.model.speech_tokenizer.decode([{"audio_codes": torch.zeros(369, 16, dtype=torch.long)}])
        finally:
            perf_log.reset_generation_id(token)
    finally:
        perf_log.logger.removeHandler(handler)

    env, stages = bench_tts.collect_vbperf(bench_tts.parse_vbperf_lines(stream.getvalue().splitlines()))
    assert env["code_path"] == str(perf_log.CODE_PATH)
    assert set(stages) == {"gen-9"}
    assert stages["gen-9"]["reference"] == {"ms": 250.0, "cached": False}
    assert stages["gen-9"]["talker"] == {"ms": 250.0, "steps": 39, "steps_per_s": 156.0}
    assert stages["gen-9"]["codec"] == {"ms": 250.0, "T": 369, "chunks": 2}


class _FakeTaskManager:
    def __init__(self):
        self.completed: list[str] = []

    def complete_generation(self, generation_id: str) -> None:
        self.completed.append(generation_id)


class _FakeDB:
    def close(self) -> None:
        pass


class _LoadedBackend:
    def is_loaded(self) -> bool:
        return True


@pytest.mark.asyncio
async def test_run_generation_tags_every_stage_with_its_id(monkeypatch, enabled, timing, lines):
    from backend import backends
    from backend.services import generation
    from backend.utils import audio as audio_utils, chunked_tts

    statuses: list[str] = []

    async def fake_status(*args, **kwargs):
        statuses.append(kwargs.get("status") or args[1])

    async def fake_load(engine, model_size):
        return None

    async def fake_prompt(*args, **kwargs):
        with perf_log.stage("reference"):
            return {}

    async def fake_generate_chunked(model, text, voice_prompt, **kwargs):
        def work():
            with perf_log.stage("talker") as timer:
                timer.fields["steps"] = 10
            return np.zeros(2400, dtype=np.float32), 24000

        return await asyncio.to_thread(work)

    task_manager = _FakeTaskManager()
    monkeypatch.setattr(backends, "get_tts_backend_for_engine", lambda engine: _LoadedBackend())
    monkeypatch.setattr(backends, "load_engine_model", fake_load)
    monkeypatch.setattr(backends, "engine_needs_trim", lambda engine: False)
    monkeypatch.setattr(generation.profiles, "create_voice_prompt_for_profile", fake_prompt)
    monkeypatch.setattr(generation.history, "update_generation_status", fake_status)
    monkeypatch.setattr(chunked_tts, "generate_chunked", fake_generate_chunked)
    monkeypatch.setattr(audio_utils, "normalize_audio", lambda audio: audio)
    monkeypatch.setattr(generation, "get_db", lambda: iter([_FakeDB()]))
    monkeypatch.setattr(generation, "get_task_manager", lambda: task_manager)
    monkeypatch.setattr(generation, "_save_generate", lambda **kwargs: "generations/gen-42.wav")
    monkeypatch.setattr(generation, "_notify_speak_end", lambda *args, **kwargs: None)

    await generation.run_generation(
        generation_id="gen-42",
        profile_id="profile-1",
        text="hola",
        language="es",
        engine="qwen",
        model_size="1.7B",
        seed=1,
        normalize=True,
        mode="generate",
    )

    events = lines()
    assert statuses[-1] == "completed"
    assert [e["stage"] for e in events] == ["reference", "talker", "normalize", "save", "total"]
    assert {e["gen_id"] for e in events} == {"gen-42"}
    assert events[-1]["ms"] > events[0]["ms"]
    assert task_manager.completed == ["gen-42"]
    perf_log.emit("after", ms=0.0)
    assert lines()[-1]["gen_id"] is None
