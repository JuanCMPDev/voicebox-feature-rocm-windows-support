"""
Tests for scripts/bench_tts.py, the perf-loop benchmark (perf-loop/contracts.md, section 2).

Everything here runs without a GPU or a server: VBPERF parsing, summary maths, server command and environment,
the isolated bench data dir, the /generate polling loop against a fake client and the pass lifecycle (``hot`` reusing
the previous pass's server) against fake servers.

Usage:
    python -m pytest backend/tests/test_bench_tts.py -v
"""

import importlib.util
import re
from pathlib import Path

import pytest

_BENCH = Path(__file__).resolve().parents[2] / "scripts" / "bench_tts.py"
_spec = importlib.util.spec_from_file_location("bench_tts", _BENCH)
bench_tts = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(bench_tts)


def test_self_test_passes():
    assert bench_tts.run_self_test() == 0


def test_self_test_reports_failures(monkeypatch):
    def _broken(failures):
        failures.append("deliberate")

    monkeypatch.setattr(bench_tts, "SELF_TESTS", (_broken,))
    assert bench_tts.run_self_test() == 1


def test_cli_self_test_exit_code():
    assert bench_tts.main(["--self-test"]) == 0


def test_cli_requires_config_tag_and_out():
    with pytest.raises(SystemExit) as exc:
        bench_tts.main(["--tag", "x"])
    assert exc.value.code == 2


def test_bench_never_imports_backend():
    source = _BENCH.read_text(encoding="utf-8")
    assert not re.search(r"^\s*(from|import)\s+backend\b", source, re.M)


def test_vbperf_stages_attach_to_their_generation():
    lines = [
        '\x1b[32mINFO\x1b[0m:     VBPERF {"gen_id": "g1", "stage": "talker", "ms": 8000.0, "steps": 40, '
        '"steps_per_s": 5.0}',
        'VBPERF {"gen_id": "g2", "stage": "codec", "ms": 900.0, "T": 300, "chunks": 1}',
        'VBPERF {"stage": "env", "miopen_user_db_path": "C:/x", "code_path": "E:/wt/backend"}',
        "INFO:     127.0.0.1:5000 - GET /history/g1 HTTP/1.1 200 OK",
    ]
    env, stages = bench_tts.collect_vbperf(bench_tts.parse_vbperf_lines(lines))
    assert env["code_path"] == "E:/wt/backend"
    assert stages["g1"] == {"talker": {"ms": 8000.0, "steps": 40, "steps_per_s": 5.0}}
    assert stages["g2"]["codec"]["T"] == 300

    rec = bench_tts.new_phrase_record({"id": "p01", "seed": 1101, "text": "hola"})
    rec.update(gen_id="g1", status="completed", wall_s=16.0, audio_s=3.2, wav="warm/p01.wav")
    bench_tts.finalize_phrase(rec, stages.get("g1"))
    assert rec["steps"] == 40
    assert rec["rtf"] == 5.0


def test_steps_fall_back_to_audio_duration_without_vbperf():
    rec = bench_tts.new_phrase_record({"id": "p01", "seed": 1101, "text": "hola"})
    rec.update(wall_s=12.0, audio_s=6.0)
    bench_tts.finalize_phrase(rec, None)
    assert rec["steps"] == 75
    assert rec["stages"] == {}


def test_fresh_mode_points_miopen_at_the_bench_dir(tmp_path):
    fresh = tmp_path / "miopen-db"
    env = bench_tts.build_server_env({"MIOPEN_USER_DB_PATH": "C:/user"}, extra_env={}, fresh_db_dir=fresh)
    assert env["MIOPEN_USER_DB_PATH"] == str(fresh)
    assert env["MIOPEN_CUSTOM_CACHE_DIR"] == str(fresh)

    persist = bench_tts.build_server_env({"MIOPEN_USER_DB_PATH": "C:/user"}, extra_env={}, fresh_db_dir=None)
    assert persist["MIOPEN_USER_DB_PATH"] == "C:/user"
    assert "MIOPEN_CUSTOM_CACHE_DIR" not in persist


def test_server_command_uses_dev_entry_point(tmp_path):
    cmd = bench_tts.build_server_cmd("python", 17599, tmp_path)
    assert cmd == ["python", "-m", "backend.main", "--port", "17599", "--data-dir", str(tmp_path)]


def test_summary_counts_find_db_growth_per_pass():
    def _pass(name, before, after, walls):
        phrases = []
        for i, wall in enumerate(walls):
            rec = bench_tts.new_phrase_record({"id": f"p{i + 1:02d}", "seed": i, "text": "x"})
            rec.update(status="completed", wall_s=wall, audio_s=4.0, wav=f"{name}/p{i + 1:02d}.wav")
            bench_tts.finalize_phrase(rec, None)
            phrases.append(rec)
        return {"name": name, "ufdb_lines_before": before, "ufdb_lines_after": after, "phrases": phrases}

    passes = [_pass("cold", 0, 30, [140.0, 160.0]), _pass("warm", 30, 30, [16.0, 18.0])]
    summary = bench_tts.compute_summary(passes, {"cold": {"load_s": 40.0}, "warm": {"load_s": 12.0}}, 2, 30)
    assert summary["cold_mean_s"] == 150.0
    assert summary["warm_mean_s"] == 17.0
    assert summary["ufdb_lines_added_cold"] == 30
    assert summary["ufdb_lines_added_warm"] == 0
    assert summary["load_s"] == 40.0
    assert summary["stage_ms_mean"] == {}
    assert summary["talker_steps_per_s_mean"] is None
    for key in ("hot_mean_s", "hot_median_s", "rtf_hot_mean", "ufdb_lines_added_hot", "stage_ms_mean_hot"):
        assert summary[key] is None
    assert summary["talker_steps_per_s_mean_hot"] is None


def _pass_with_talker(name, walls_and_talker_ms, before=0, after=0):
    phrases = []
    for i, (wall, talker_ms) in enumerate(walls_and_talker_ms):
        rec = bench_tts.new_phrase_record({"id": f"p{i + 1:02d}", "seed": i, "text": "x"})
        rec.update(status="completed", wall_s=wall, audio_s=4.0, wav=f"{name}/p{i + 1:02d}.wav")
        talker = {"ms": talker_ms, "steps": 50, "steps_per_s": 50 / (talker_ms / 1000)}
        bench_tts.finalize_phrase(rec, {"talker": talker})
        phrases.append(rec)
    return {"name": name, "ufdb_lines_before": before, "ufdb_lines_after": after, "phrases": phrases}


def test_summary_reports_hot_pass_apart_from_warm():
    warm = _pass_with_talker("warm", [(60.0, 12500.0), (70.0, 12500.0)], 30, 30)
    hot = _pass_with_talker("hot", [(15.0, 10000.0), (17.0, 12500.0), (25.0, 10000.0)], 30, 32)
    server = {"warm": {"load_s": 12.0, "pid": 7}, "hot": {"load_s": None, "pid": 7, "same_process_as": "warm"}}
    summary = bench_tts.compute_summary([warm, hot], server, 3, 32)
    assert summary["hot_mean_s"] == 19.0
    assert summary["hot_median_s"] == 17.0
    assert summary["rtf_hot_mean"] == round((15.0 + 17.0 + 25.0) / 4.0 / 3, 3)
    assert summary["ufdb_lines_added_hot"] == 2
    # stage_ms_mean and talker_steps_per_s_mean stay on warm; the *_hot variants use the hot pass.
    assert summary["stage_ms_mean"] == {"talker": 12500.0}
    assert summary["talker_steps_per_s_mean"] == 4.0
    assert summary["stage_ms_mean_hot"] == {"talker": 10833.333}
    assert summary["talker_steps_per_s_mean_hot"] == 4.667
    assert summary["warm_mean_s"] == 65.0
    assert summary["load_s"] == 12.0


@pytest.mark.parametrize("spec", ["hot", "hot,warm", "warm,hot,hot"])
def test_hot_pass_cannot_come_first_or_repeat(spec):
    with pytest.raises(bench_tts.BenchError):
        bench_tts.parse_passes(spec)


def test_hot_pass_follows_any_other_pass():
    assert bench_tts.parse_passes("cold,warm,hot") == ["cold", "warm", "hot"]
    assert bench_tts.parse_passes("warm,hot") == ["warm", "hot"]


def test_batch_reference_prefers_hot_then_warm():
    cold, warm, hot = ({"name": n, "phrases": []} for n in ("cold", "warm", "hot"))
    assert bench_tts.batch_reference_pass([cold, warm, hot])["name"] == "hot"
    assert bench_tts.batch_reference_pass([cold, warm])["name"] == "warm"
    assert bench_tts.batch_reference_pass([cold])["name"] == "cold"
    assert bench_tts.batch_reference_pass([]) is None


def test_server_pid_is_the_one_uvicorn_reports():
    lines = ["INFO:     Started server process [22116]", "INFO:     Waiting for application startup."]
    assert bench_tts.server_pid_from_log(lines) == 22116
    assert bench_tts.server_pid_from_log(["no pid here"]) is None


def test_hot_pass_reuses_the_previous_server_process(tmp_path):
    code, doc, servers = bench_tts.run_with_fake_servers(
        tmp_path, "cold,warm,hot", "--scenario", "batch", "--concurrent", "2"
    )
    assert code == 0
    assert doc["errors"] == []
    # Two processes only (cold, warm); warm stays up for hot and the batch, then stops once.
    assert [(s.name, s.starts, s.stops) for s in servers] == [("cold", 1, 1), ("warm", 1, 1)]
    assert servers[1].posts_at_stop == 6
    srv = doc["server"]
    assert srv["cold"]["pid"] == 101
    assert srv["warm"]["pid"] == srv["hot"]["pid"] == 102
    assert srv["hot"]["same_process_as"] == "warm"
    assert srv["hot"]["load_s"] is None
    hot = next(p for p in doc["passes"] if p["name"] == "hot")
    assert hot["server_log"] == "logs/server-warm.log"
    assert [r["wav"] for r in hot["phrases"]] == ["hot/p01.wav", "hot/p02.wav"]
    assert all((tmp_path / "out" / r["wav"]).is_file() for r in hot["phrases"])
    summary = doc["summary"]
    assert summary["hot_mean_s"] is not None
    assert summary["stage_ms_mean"] == {"talker": 1500.0}
    assert summary["stage_ms_mean_hot"] == {"talker": 3500.0}
    batch = doc["scenarios"]["batch"]
    assert batch["ref_pass"] == "hot"
    assert batch["sum_individual_s"] == round(sum(r["wall_s"] for r in hot["phrases"]), 3)


def test_hot_pass_after_cold_shares_the_cold_process(tmp_path):
    code, doc, servers = bench_tts.run_with_fake_servers(tmp_path, "cold,hot")
    assert code == 0
    assert len(servers) == 1
    assert doc["server"]["hot"] == {"startup_s": None, "load_s": None, "pid": 101, "same_process_as": "cold"}
    assert doc["summary"]["warm_mean_s"] is None
    assert doc["scenarios"]["batch"] is None
