"""
Tests for scripts/compare_wavs.py, the perf-loop quality comparator (perf-loop/contracts.md, section 3).

Everything here runs without a GPU or models: SNR, WER and text normalization maths, the profile's reference audio
lookup, and the comparison flow with stand-in speaker and Whisper models. The real models are replaced by stubs, and
``snr,dur`` runs check that no model is ever loaded.

Usage:
    python -m pytest backend/tests/test_compare_wavs.py -v
"""

import contextlib
import importlib.util
import json
import math
import re
import sqlite3
from pathlib import Path

import numpy as np
import pytest

_COMPARE = Path(__file__).resolve().parents[2] / "scripts" / "compare_wavs.py"
_spec = importlib.util.spec_from_file_location("compare_wavs", _COMPARE)
compare_wavs = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(compare_wavs)

PROFILE_ID = "prof-1"


def test_self_test_passes():
    assert compare_wavs.run_self_test() == 0


def test_self_test_reports_failures(monkeypatch):
    def _broken(failures):
        failures.append("deliberate")

    monkeypatch.setattr(compare_wavs, "SELF_TESTS", (_broken,))
    assert compare_wavs.run_self_test() == 1


def test_cli_self_test_exit_code():
    assert compare_wavs.main(["--self-test"]) == 0


def test_cli_requires_config_ref_cand_and_out():
    with pytest.raises(SystemExit) as exc:
        compare_wavs.main(["--ref", "x", "--cand", "y"])
    assert exc.value.code == 2


def test_comparator_never_imports_backend():
    source = _COMPARE.read_text(encoding="utf-8")
    assert not re.search(r"^\s*(from|import)\s+backend\b", source, re.M)


def test_snr_is_999_for_identical_audio_and_matches_known_noise():
    tone = 0.5 * np.sin(2 * np.pi * 220 * np.arange(48000) / 24000)
    assert compare_wavs.snr_db(tone, tone.copy()) == 999.0
    noise = np.random.default_rng(0).normal(0.0, 0.01, tone.size)
    assert compare_wavs.snr_db(tone, tone + noise) == pytest.approx(10 * math.log10(0.125 / 1e-4), abs=0.5)


def test_snr_uses_the_common_length():
    tone = 0.5 * np.sin(2 * np.pi * 220 * np.arange(24000) / 24000)
    assert compare_wavs.snr_db(tone, np.concatenate([tone, np.ones(500)])) == 999.0
    assert compare_wavs.snr_db(tone, tone[:-300]) == 999.0
    assert compare_wavs.snr_db(np.zeros(0), tone) == -999.0


@pytest.mark.parametrize(
    ("raw", "normalized"),
    [
        ("¿Qué TAL,   señor?", "qué tal señor"),
        ("canción", "canción"),
        ("canció́n", "canción"),
        ("«Sí» —dijo— (vale).", "sí dijo vale"),
        ("pa­labra", "palabra"),
    ],
)
def test_text_normalization(raw, normalized):
    assert compare_wavs.normalize_text(raw) == normalized


def test_wer_counts_word_edits_after_normalization():
    assert compare_wavs.word_error_rate("Mi vecino riega las plantas.", "mi vecino riega las plantas") == 0.0
    assert compare_wavs.word_error_rate("el gato come pescado", "el perro come pescado") == 0.25
    assert compare_wavs.word_error_rate("el gato come", "el gato negro come") == pytest.approx(1 / 3)
    assert compare_wavs.word_error_rate("uno dos tres", "") == 1.0


def test_metrics_parsing():
    assert compare_wavs.parse_metrics("all") == ["snr", "spk", "wer", "dur"]
    assert compare_wavs.parse_metrics("dur,snr") == ["snr", "dur"]
    with pytest.raises(compare_wavs.CompareError):
        compare_wavs.parse_metrics("snr,mos")


def test_self_comparison_gives_exact_zero_deltas_with_one_model_call_per_audio(tmp_path):
    compare_wavs.make_fake_runs(tmp_path)
    reference = compare_wavs.make_audio(0.3 * np.sin(np.arange(48000) / 7.0), 24000)
    embed = compare_wavs._StubEmbedder()
    transcribe = compare_wavs._StubTranscriber({}, default="la mañana amaneció tranquila")
    doc = compare_wavs.compare_dirs(
        tmp_path / "ref",
        tmp_path / "ref",
        "warm",
        ["snr", "spk", "wer", "dur"],
        texts=compare_wavs.SELFTEST_TEXTS,
        reference=reference,
        embed=embed,
        transcribe=transcribe,
    )
    summary = doc["summary"]
    assert summary["n"] == 3
    assert summary["snr_db_min"] == 999.0
    assert summary["spk_cos_delta_mean"] == 0.0
    assert summary["wer_delta"] == 0.0
    assert summary["dur_dev_max"] == 0.0
    # p01 and p03 hold the same samples: two distinct phrase audios plus the profile reference.
    assert embed.calls == 3
    assert transcribe.calls == 2


def test_comparison_reports_differences_and_the_phrase_intersection(tmp_path):
    compare_wavs.make_fake_runs(tmp_path)
    reference = compare_wavs.make_audio(0.3 * np.sin(np.arange(48000) / 7.0), 24000)
    doc = compare_wavs.compare_dirs(
        tmp_path / "ref",
        tmp_path / "cand",
        "warm",
        ["snr", "spk", "dur"],
        texts={},
        reference=reference,
        embed=compare_wavs._StubEmbedder(),
    )
    assert [p["id"] for p in doc["phrases"]] == ["p01", "p02"]
    assert (doc["only_ref"], doc["only_cand"]) == (["p03"], ["p04"])
    p01, p02 = doc["phrases"]
    assert p01["snr_db"] == 999.0
    assert p02["snr_db"] == pytest.approx(31.0, abs=0.5)
    assert p02["dur_dev"] == pytest.approx(0.1, abs=1e-6)
    assert p02["wer_ref"] is None
    assert doc["summary"]["wer_delta"] is None
    assert doc["summary"]["spk_cos_delta_mean"] != 0.0


def _write_config(tmp_path, data_dir):
    config = tmp_path / "config.json"
    bench = {"profile_id": PROFILE_ID, "data_dir": str(data_dir), "language": "es"}
    config.write_text(json.dumps({"bench": bench, "compare": {"whisper_model": "small"}}), encoding="utf-8")
    return config


def _make_data_dir(tmp_path):
    data_dir = tmp_path / "data"
    sample = data_dir / "profiles" / PROFILE_ID / "s1.wav"
    compare_wavs._write_wav(sample, compare_wavs._tone(2.0, freq=180.0))
    with contextlib.closing(sqlite3.connect(data_dir / "voicebox.db")) as conn:
        conn.execute("CREATE TABLE profile_samples (id TEXT, profile_id TEXT, audio_path TEXT)")
        conn.execute("INSERT INTO profile_samples VALUES ('s1', ?, ?)", (PROFILE_ID, f"profiles\\{PROFILE_ID}\\s1.wav"))
        conn.commit()
    return data_dir, sample


def test_snr_and_dur_never_load_a_model(tmp_path, monkeypatch):
    def _no_model(*_args, **_kwargs):
        raise AssertionError("a model was loaded for snr,dur")

    monkeypatch.setattr(compare_wavs.QwenSpeakerEmbedder, "load", _no_model)
    monkeypatch.setattr(compare_wavs.WhisperTranscriber, "load", _no_model)
    monkeypatch.setattr(compare_wavs, "resolve_device", _no_model)
    compare_wavs.make_fake_runs(tmp_path)
    out = tmp_path / "out.json"
    argv = ["--config", str(_write_config(tmp_path, tmp_path / "missing-data"))]
    argv += ["--ref", str(tmp_path / "ref"), "--cand", str(tmp_path / "cand"), "--metrics", "snr,dur"]
    assert compare_wavs.main([*argv, "--out", str(out)]) == 0
    doc = json.loads(out.read_text(encoding="utf-8"))
    assert doc["schema"] == "vb-compare/1"
    assert doc["summary"]["n"] == 2
    assert doc["summary"]["spk_cos_ref_mean"] is None
    assert doc["models"]["whisper"] is None


def test_all_metrics_cli_with_stub_models_compares_a_run_with_itself(tmp_path, monkeypatch):
    loaded = []
    monkeypatch.setattr(compare_wavs.QwenSpeakerEmbedder, "load", lambda self: loaded.append(self.repo_id))
    monkeypatch.setattr(
        compare_wavs.QwenSpeakerEmbedder, "__call__", lambda self, audio: np.array([1.0, len(audio.samples)])
    )
    monkeypatch.setattr(compare_wavs.WhisperTranscriber, "load", lambda self: loaded.append(self.repo_id))
    monkeypatch.setattr(compare_wavs.WhisperTranscriber, "__call__", lambda self, audio: "texto")
    compare_wavs.make_fake_runs(tmp_path)
    data_dir, sample = _make_data_dir(tmp_path)
    out = tmp_path / "out.json"
    argv = ["--config", str(_write_config(tmp_path, data_dir)), "--ref", str(tmp_path / "ref")]
    argv += ["--cand", str(tmp_path / "ref"), "--metrics", "all", "--device", "cpu", "--out", str(out)]
    assert compare_wavs.main(argv) == 0
    assert loaded == ["Qwen/Qwen3-TTS-12Hz-1.7B-Base", "openai/whisper-small"]
    doc = json.loads(out.read_text(encoding="utf-8"))
    assert doc["reference_audio"] == [str(sample)]
    assert doc["models"] == {
        "speaker": "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
        "speaker_device": "cpu",
        "whisper": "openai/whisper-small",
        "whisper_device": "cpu",
    }
    summary = doc["summary"]
    assert summary["n"] == 3
    assert summary["snr_db_min"] == 999.0
    assert summary["spk_cos_delta_mean"] == 0.0
    assert summary["wer_delta"] == 0.0


def test_reference_audio_comes_from_the_bench_data_dir(tmp_path):
    data_dir, sample = _make_data_dir(tmp_path)
    assert compare_wavs.profile_reference_paths(data_dir, PROFILE_ID) == [sample]
    with pytest.raises(compare_wavs.CompareError):
        compare_wavs.profile_reference_paths(data_dir, "unknown")
    with pytest.raises(compare_wavs.CompareError):
        compare_wavs.profile_reference_paths(tmp_path / "nowhere", PROFILE_ID)


def test_model_load_failure_is_a_clear_error(tmp_path, monkeypatch):
    def _fail(self):
        raise OSError("no network and not cached")

    monkeypatch.setattr(compare_wavs.WhisperTranscriber, "load", _fail)
    compare_wavs.make_fake_runs(tmp_path)
    argv = ["--config", str(_write_config(tmp_path, tmp_path / "data")), "--ref", str(tmp_path / "ref")]
    argv += ["--cand", str(tmp_path / "ref"), "--metrics", "wer", "--device", "cpu", "--out", str(tmp_path / "o.json")]
    assert compare_wavs.main(argv) == 1
    assert not (tmp_path / "o.json").exists()
