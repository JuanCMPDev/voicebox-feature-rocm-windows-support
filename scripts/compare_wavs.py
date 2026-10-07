"""Quality metrics for the Voicebox perf loop (perf-loop/contracts.md, section 3).

Compares two bench runs phrase by phrase, ``<ref>/<pass>/<id>.wav`` against ``<cand>/<pass>/<id>.wav`` for every id
present in both:

- ``snr``: waveform SNR over the common length; 999 dB when the samples are identical.
- ``spk``: cosine between Qwen3-TTS's own ``extract_speaker_embedding`` of each WAV and that of the profile's
  reference audio (read from ``config.bench.data_dir``), for ref and for cand.
- ``wer``: word error rate of a Whisper transcription (greedy, temperature 0) against the requested text, both
  normalized the same way.
- ``dur``: ``|dur_cand / dur_ref - 1|``.

Models are only loaded when a requested metric needs them. Every distinct audio is embedded and transcribed once per
run, so a run compared with itself gives exactly zero deltas. Like ``bench_tts.py``, this script never imports
anything from ``backend/`` (contracts, section 5): it loads ``qwen_tts``, ``transformers`` and ``torch`` directly,
so the evaluator does not change when the code it evaluates changes.

Usage:
    python scripts/compare_wavs.py --config perf-loop/config.json --ref perf-loop/bench/<tagA>
        --cand perf-loop/bench/<tagB> [--pass warm] [--metrics snr,dur|all] [--device auto|cpu] --out <file.json>
    python scripts/compare_wavs.py --self-test
"""

import argparse
import contextlib
import hashlib
import json
import logging
import math
import os
import sqlite3
import statistics
import sys
import tempfile
import unicodedata
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path, PureWindowsPath
from types import SimpleNamespace
from typing import Any

import numpy as np
import soundfile as sf

logger = logging.getLogger("compare_wavs")

SCHEMA = "vb-compare/1"
ROOT = Path(__file__).resolve().parents[1]
PHRASES_FILE = ROOT / "scripts" / "bench_phrases.json"

ALL_METRICS = ("snr", "spk", "wer", "dur")
DEFAULT_METRICS = "snr,dur"
DEFAULT_PASS = "warm"
DEFAULT_DATA_DIR = "perf-loop/bench/_data"
SNR_CAP_DB = 999.0
WHISPER_SR = 16000
DEFAULT_WHISPER_MODEL = "small"
DEFAULT_MODEL_SIZE = "1.7B"
QWEN_REPOS = {"1.7B": "Qwen/Qwen3-TTS-12Hz-1.7B-Base", "0.6B": "Qwen/Qwen3-TTS-12Hz-0.6B-Base"}
QWEN_WEIGHTS_FILE = "model.safetensors"
SPEAKER_WEIGHTS_PREFIX = "speaker_encoder."

# Decimals kept in the JSON output; summaries are computed before rounding.
ROUND_DIGITS = {
    "snr_db": 3,
    "spk_cos_ref": 6,
    "spk_cos_cand": 6,
    "spk_cos_delta": 6,
    "wer_ref": 6,
    "wer_cand": 6,
    "wer_delta": 6,
    "dur_ref_s": 4,
    "dur_cand_s": 4,
    "dur_dev": 6,
    "snr_db_min": 3,
    "snr_db_mean": 3,
    "spk_cos_ref_mean": 6,
    "spk_cos_cand_mean": 6,
    "spk_cos_delta_mean": 6,
    "wer_ref_mean": 6,
    "wer_cand_mean": 6,
    "dur_dev_max": 6,
}
SUMMARY_KEYS = (
    "snr_db_min",
    "snr_db_mean",
    "spk_cos_ref_mean",
    "spk_cos_cand_mean",
    "spk_cos_delta_mean",
    "wer_ref_mean",
    "wer_cand_mean",
    "wer_delta",
    "dur_dev_max",
)


class CompareError(Exception):
    """Fatal comparison error; the message goes to stderr and the exit code is non-zero."""


@dataclass(frozen=True, eq=False)
class Audio:
    """Mono float32 samples and their rate; ``key`` identifies the exact samples a model would receive."""

    samples: np.ndarray
    sr: int
    key: str

    @property
    def duration_s(self) -> float:
        return len(self.samples) / self.sr if self.sr else 0.0


Embedder = Callable[[Audio], np.ndarray]
Transcriber = Callable[[Audio], str]


def _is_number(value: Any) -> bool:
    return isinstance(value, int | float) and not isinstance(value, bool)


def _round(value: float | None, ndigits: int) -> float | None:
    if not _is_number(value) or not math.isfinite(value):
        return None
    return round(float(value), ndigits)


def _now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        raise CompareError(f"cannot read {path}: {e}") from e


def _write_json_atomic(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def parse_metrics(spec: str | None) -> list[str]:
    """Parse ``--metrics`` (``all`` or a comma-separated subset of snr, spk, wer, dur) into canonical order."""
    names = {m.strip().lower() for m in (spec or "").split(",") if m.strip()}
    if "all" in names:
        return list(ALL_METRICS)
    if not names:
        raise CompareError("--metrics is empty")
    unknown = sorted(names - set(ALL_METRICS))
    if unknown:
        raise CompareError(f"unknown metrics: {', '.join(unknown)} (valid: {', '.join(ALL_METRICS)} or all)")
    return [m for m in ALL_METRICS if m in names]


# Audio


def make_audio(samples: np.ndarray, sr: int) -> Audio:
    mono = np.ascontiguousarray(samples, dtype=np.float32)
    digest = hashlib.sha256(f"{int(sr)}:".encode() + mono.tobytes()).hexdigest()
    return Audio(mono, int(sr), digest)


def read_wav(path: Path) -> Audio:
    """Read a WAV as mono float32 (channels averaged, as librosa does), keeping its sample rate."""
    data, sr = sf.read(str(path), dtype="float32", always_2d=True)
    mono = data[:, 0] if data.shape[1] == 1 else data.mean(axis=1)
    return make_audio(mono, sr)


def resample(samples: np.ndarray, sr: int, target_sr: int) -> np.ndarray:
    """Resample with librosa's default (soxr_hq), the same call qwen_tts and the app's audio loader make."""
    if int(sr) == int(target_sr):
        return samples
    import librosa  # lazy: heavy import

    return librosa.resample(y=np.asarray(samples, dtype=np.float32), orig_sr=int(sr), target_sr=int(target_sr))


# Metrics


def snr_db(ref: np.ndarray, cand: np.ndarray) -> float:
    """``10*log10(sum(ref^2) / sum((ref - cand)^2))`` over the common length, within +-999 dB (999 if identical)."""
    n = min(len(ref), len(cand))
    if n == 0:
        return SNR_CAP_DB if len(ref) == len(cand) else -SNR_CAP_DB
    r = np.asarray(ref[:n], dtype=np.float64)
    noise = float(np.sum(np.square(r - np.asarray(cand[:n], dtype=np.float64))))
    if noise == 0.0:
        return SNR_CAP_DB
    signal = float(np.sum(np.square(r)))
    if signal == 0.0:
        return -SNR_CAP_DB
    return max(-SNR_CAP_DB, min(SNR_CAP_DB, 10.0 * math.log10(signal / noise)))


def normalize_text(text: str) -> str:
    """Lower case, punctuation and symbols as spaces, each accent once per letter, NFC, single spaces.

    Accents are kept ("qué" and "que" are different words); NFC makes precomposed and combining forms equal, and a
    repeated combining mark on the same letter (a duplicated tilde) counts once.
    """
    out: list[str] = []
    marks: set[str] = set()
    after_letter = False
    for ch in unicodedata.normalize("NFD", text.lower()):
        cat = unicodedata.category(ch)
        if cat.startswith("M"):
            if after_letter and ch not in marks:
                marks.add(ch)
                out.append(ch)
            continue
        if cat == "Cf":  # soft hyphens and zero-width characters sit inside a word
            continue
        marks = set()
        after_letter = cat[0] in "LN"
        out.append(ch if after_letter else " ")
    return " ".join(unicodedata.normalize("NFC", "".join(out)).split())


def word_errors(ref_words: list[str], hyp_words: list[str]) -> int:
    """Word-level Levenshtein distance (substitutions + deletions + insertions)."""
    prev = list(range(len(hyp_words) + 1))
    for i, ref_word in enumerate(ref_words, 1):
        cur = [i] + [0] * len(hyp_words)
        for j, hyp_word in enumerate(hyp_words, 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ref_word != hyp_word))
        prev = cur
    return prev[-1]


def word_error_rate(reference: str, hypothesis: str) -> float:
    """WER of ``hypothesis`` against ``reference`` after :func:`normalize_text` on both."""
    ref_words = normalize_text(reference).split()
    hyp_words = normalize_text(hypothesis).split()
    if not ref_words:
        return 0.0 if not hyp_words else 1.0
    return word_errors(ref_words, hyp_words) / len(ref_words)


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    a64 = np.asarray(a, dtype=np.float64).ravel()
    b64 = np.asarray(b, dtype=np.float64).ravel()
    denom = float(np.linalg.norm(a64) * np.linalg.norm(b64))
    if denom == 0.0:
        raise ValueError("speaker embedding with zero norm")
    return float(np.clip(np.dot(a64, b64) / denom, -1.0, 1.0))


def duration_deviation(ref_s: float, cand_s: float) -> float:
    if ref_s <= 0:
        raise ValueError("the reference WAV is empty")
    return abs(cand_s / ref_s - 1.0)


# Reference audio of the profile


def resolve_data_dir(bench_cfg: dict) -> Path:
    """``config.bench.data_dir`` (relative to the worktree), the same data dir the bench gives the server."""
    data_dir = Path(bench_cfg.get("data_dir") or DEFAULT_DATA_DIR)
    if not data_dir.is_absolute():
        data_dir = ROOT / data_dir
    return data_dir.resolve()


def rebase_storage_path(stored: str, data_dir: Path) -> Path | None:
    """Where a DB-stored path lives inside ``data_dir`` (the server's ``data`` rebasing, as in bench_tts.py)."""
    path = PureWindowsPath(stored) if "\\" in stored else Path(stored)
    parts = path.parts
    if path.anchor:
        for idx, part in enumerate(parts):
            if part == "data":
                return data_dir.joinpath(*parts[idx + 1 :])
        return None
    if parts and parts[0] == "data":
        parts = parts[1:]
    return data_dir.joinpath(*parts)


def profile_reference_paths(data_dir: Path, profile_id: str) -> list[Path]:
    """Audio files of the profile's samples inside the bench data dir, in the order the app lists them."""
    db_path = data_dir / "voicebox.db"
    if not db_path.is_file():
        raise CompareError(f"the bench database does not exist: {db_path}")
    try:
        with contextlib.closing(sqlite3.connect(db_path)) as conn:
            query = "SELECT id, audio_path FROM profile_samples WHERE profile_id = ? ORDER BY rowid"
            rows = conn.execute(query, (profile_id,)).fetchall()
    except sqlite3.Error as e:
        raise CompareError(f"cannot read the samples of profile {profile_id} from {db_path}: {e}") from e
    if not rows:
        raise CompareError(f"profile {profile_id} has no samples in {db_path}")
    paths = []
    for sample_id, stored in rows:
        target = rebase_storage_path(str(stored), data_dir) if stored else None
        if target is None or not target.is_file():
            raise CompareError(f"sample {sample_id} of profile {profile_id} is not in the bench data dir: {stored}")
        paths.append(target)
    return paths


def load_reference_audio(paths: list[Path], target_sr: int = 24000) -> Audio:
    """The profile's reference audio: its only sample, or every sample at ``target_sr`` one after another."""
    clips = [read_wav(p) for p in paths]
    if not clips:
        raise CompareError("the profile has no reference audio")
    if len(clips) == 1:
        return clips[0]
    return make_audio(np.concatenate([resample(c.samples, c.sr, target_sr) for c in clips]), target_sr)


# Models


def hf_file(repo_id: str, filename: str) -> str:
    """Local path of a Hugging Face Hub file: the cached copy if there is one, else a one-time download."""
    from huggingface_hub import hf_hub_download, try_to_load_from_cache  # lazy: only for model metrics

    cached = try_to_load_from_cache(repo_id, filename)
    if isinstance(cached, str):
        return cached
    return hf_hub_download(repo_id, filename)


def speaker_repo(model_size: str | None) -> str:
    size = str(model_size or DEFAULT_MODEL_SIZE)
    if size not in QWEN_REPOS:
        raise CompareError(f"unknown Qwen model size {size!r} (valid: {', '.join(QWEN_REPOS)})")
    return QWEN_REPOS[size]


def whisper_repo(model: str | None) -> str:
    name = str(model or DEFAULT_WHISPER_MODEL)
    return name if "/" in name else f"openai/whisper-{name}"


def resolve_device(spec: str) -> str:
    """``cpu``, or for ``auto`` the GPU when torch sees one (CUDA or ROCm) and the CPU otherwise."""
    if spec == "cpu":
        return "cpu"
    import torch  # lazy: heavy import

    return "cuda" if torch.cuda.is_available() else "cpu"


class QwenSpeakerEmbedder:
    """Speaker embeddings from Qwen3-TTS's own ``extract_speaker_embedding``, on CPU in FP32.

    Only the ``speaker_encoder.*`` weights of the checkpoint are loaded (12 M parameters instead of the whole 1.7 B
    model) and the model's own method runs on them, so the mel front end and the encoder are the app's. It stays on
    the CPU on purpose: on ROCm every new WAV length would start a MIOpen Find (the very cost this loop measures)
    and write to the Find-Db the app uses, and FP32 on the CPU gives the same embedding for the same audio every run.
    """

    def __init__(self, repo_id: str):
        self.repo_id = repo_id
        self.sample_rate = 24000
        self._owner: Any = None
        self._extract: Any = None

    def load(self) -> None:
        import torch  # lazy: heavy import
        from qwen_tts.core.models import Qwen3TTSConfig, Qwen3TTSForConditionalGeneration
        from qwen_tts.core.models.modeling_qwen3_tts import Qwen3TTSSpeakerEncoder
        from safetensors import safe_open

        config = Qwen3TTSConfig.from_json_file(hf_file(self.repo_id, "config.json"))
        encoder = Qwen3TTSSpeakerEncoder(config.speaker_encoder_config)
        state = {}
        with safe_open(hf_file(self.repo_id, QWEN_WEIGHTS_FILE), framework="pt", device="cpu") as weights:
            for name in weights.keys():  # noqa: SIM118 - safe_open handles are not iterable, only .keys() is
                if name.startswith(SPEAKER_WEIGHTS_PREFIX):
                    state[name.removeprefix(SPEAKER_WEIGHTS_PREFIX)] = weights.get_tensor(name)
        encoder.load_state_dict(state, strict=True)
        encoder.float().eval()
        # extract_speaker_embedding reads only these three attributes of the full model.
        self._owner = SimpleNamespace(speaker_encoder=encoder, device=torch.device("cpu"), dtype=torch.float32)
        self._extract = Qwen3TTSForConditionalGeneration.extract_speaker_embedding
        self.sample_rate = int(config.speaker_encoder_config.sample_rate)
        logger.info("speaker encoder of %s loaded on cpu (%d tensors)", self.repo_id, len(state))

    def __call__(self, audio: Audio) -> np.ndarray:
        if self._extract is None:
            self.load()
        wav = np.array(resample(audio.samples, audio.sr, self.sample_rate), dtype=np.float32)
        embedding = self._extract(self._owner, audio=wav, sr=self.sample_rate)
        return embedding.double().cpu().numpy()


class WhisperTranscriber:
    """Whisper transcription with deterministic decoding: greedy, one beam, temperature 0.

    Same approach as the app's STT backend (WhisperProcessor + WhisperForConditionalGeneration on 16 kHz audio,
    language forced), loaded here so the evaluator does not depend on backend/.
    """

    def __init__(self, repo_id: str, *, language: str, device: str):
        self.repo_id = repo_id
        self.language = language
        self.device = device
        self._torch: Any = None
        self._processor: Any = None
        self._model: Any = None

    def load(self) -> None:
        import torch  # lazy: heavy import
        from transformers import WhisperForConditionalGeneration, WhisperProcessor

        self._torch = torch
        self._processor = WhisperProcessor.from_pretrained(self.repo_id)
        self._model = WhisperForConditionalGeneration.from_pretrained(self.repo_id).to(self.device).eval()
        logger.info("Whisper %s loaded on %s", self.repo_id, self.device)

    def __call__(self, audio: Audio) -> str:
        if self._model is None:
            self.load()
        wav = resample(audio.samples, audio.sr, WHISPER_SR)
        features = self._processor(wav, sampling_rate=WHISPER_SR, return_tensors="pt").input_features
        with self._torch.inference_mode():
            ids = self._model.generate(
                features.to(self.device),
                language=self.language,
                task="transcribe",
                do_sample=False,
                num_beams=1,
                temperature=0.0,
            )
        return self._processor.batch_decode(ids, skip_special_tokens=True)[0].strip()


def _load_model(label: str, load: Callable[[], None]) -> None:
    try:
        load()
    except Exception as e:  # any loader failure (download, weights, config) makes the requested metric impossible
        raise CompareError(f"cannot load the {label}: {type(e).__name__}: {e}") from e


# Comparison


def wav_ids(folder: Path) -> set[str]:
    if not folder.is_dir():
        return set()
    return {p.stem for p in folder.glob("*.wav") if p.is_file()}


def match_phrases(ref_folder: Path, cand_folder: Path) -> tuple[list[str], list[str], list[str]]:
    """Return (ids in both, ids only in ref, ids only in cand), each sorted."""
    ref_ids, cand_ids = wav_ids(ref_folder), wav_ids(cand_folder)
    return (sorted(ref_ids & cand_ids), sorted(ref_ids - cand_ids), sorted(cand_ids - ref_ids))


def new_phrase_record(phrase_id: str) -> dict:
    """Phrase entry of the output with every field present (null when not requested or not computed)."""
    return {
        "id": phrase_id,
        "snr_db": None,
        "spk_cos_ref": None,
        "spk_cos_cand": None,
        "spk_cos_delta": None,
        "wer_ref": None,
        "wer_cand": None,
        "wer_delta": None,
        "hyp_ref": None,
        "hyp_cand": None,
        "dur_ref_s": None,
        "dur_cand_s": None,
        "dur_dev": None,
        "error": None,
    }


def summarize(records: list[dict], metrics: list[str]) -> dict:
    """Summary over the phrases compared without error (unrounded; see :func:`round_fields`)."""
    ok = [r for r in records if not r.get("error")]

    def _values(key: str) -> list[float]:
        return [r[key] for r in ok if _is_number(r.get(key))]

    def _mean(key: str) -> float | None:
        values = _values(key)
        return statistics.fmean(values) if values else None

    summary: dict[str, Any] = {"n": len(ok), **dict.fromkeys(SUMMARY_KEYS)}
    if not ok:
        return summary
    if "snr" in metrics:
        summary["snr_db_min"] = min(_values("snr_db"))
        summary["snr_db_mean"] = _mean("snr_db")
    if "spk" in metrics:
        summary["spk_cos_ref_mean"] = _mean("spk_cos_ref")
        summary["spk_cos_cand_mean"] = _mean("spk_cos_cand")
        summary["spk_cos_delta_mean"] = _mean("spk_cos_delta")
    if "wer" in metrics:
        summary["wer_ref_mean"] = _mean("wer_ref")
        summary["wer_cand_mean"] = _mean("wer_cand")
        summary["wer_delta"] = _mean("wer_delta")
    if "dur" in metrics:
        summary["dur_dev_max"] = max(_values("dur_dev"))
    return summary


def round_fields(record: dict) -> dict:
    return {k: _round(v, ROUND_DIGITS[k]) if k in ROUND_DIGITS else v for k, v in record.items()}


class _Memo:
    """Runs a model once per distinct audio (same samples and rate) and reuses the result."""

    def __init__(self, fn: Callable[[Audio], Any]):
        self.fn = fn
        self.results: dict[str, Any] = {}

    def __call__(self, audio: Audio) -> Any:
        if audio.key not in self.results:
            self.results[audio.key] = self.fn(audio)
        return self.results[audio.key]


def _measure_phrase(
    rec: dict,
    ref_audio: Audio,
    cand_audio: Audio,
    metrics: list[str],
    *,
    text: str | None,
    ref_embedding: np.ndarray | None,
    embed: _Memo | None,
    transcribe: _Memo | None,
) -> None:
    if "dur" in metrics:
        rec["dur_ref_s"] = ref_audio.duration_s
        rec["dur_cand_s"] = cand_audio.duration_s
        rec["dur_dev"] = duration_deviation(ref_audio.duration_s, cand_audio.duration_s)
    if "snr" in metrics:
        if ref_audio.sr != cand_audio.sr:
            raise ValueError(f"sample rates differ ({ref_audio.sr} vs {cand_audio.sr} Hz)")
        rec["snr_db"] = snr_db(ref_audio.samples, cand_audio.samples)
    if "spk" in metrics:
        assert embed is not None
        assert ref_embedding is not None
        rec["spk_cos_ref"] = cosine(embed(ref_audio), ref_embedding)
        rec["spk_cos_cand"] = cosine(embed(cand_audio), ref_embedding)
        rec["spk_cos_delta"] = rec["spk_cos_cand"] - rec["spk_cos_ref"]
    if "wer" in metrics:
        assert transcribe is not None
        if text is None:
            raise ValueError(f"{rec['id']} is not in scripts/bench_phrases.json; no text to score WER against")
        rec["hyp_ref"] = transcribe(ref_audio)
        rec["hyp_cand"] = transcribe(cand_audio)
        rec["wer_ref"] = word_error_rate(text, rec["hyp_ref"])
        rec["wer_cand"] = word_error_rate(text, rec["hyp_cand"])
        rec["wer_delta"] = rec["wer_cand"] - rec["wer_ref"]


def compare_dirs(
    ref_dir: Path,
    cand_dir: Path,
    pass_name: str,
    metrics: list[str],
    *,
    texts: dict[str, str],
    reference: Audio | None = None,
    embed: Embedder | None = None,
    transcribe: Transcriber | None = None,
) -> dict:
    """Compare the WAVs of one pass of two bench runs and return the output document (contracts, section 3).

    ``embed`` and ``transcribe`` are only needed (and only called) for ``spk`` and ``wer``; each one runs once per
    distinct audio, so identical WAVs always get identical metrics.
    """
    if "spk" in metrics and (embed is None or reference is None):
        raise CompareError("the spk metric needs a speaker model and the profile's reference audio")
    if "wer" in metrics and transcribe is None:
        raise CompareError("the wer metric needs a transcription model")
    ids, only_ref, only_cand = match_phrases(ref_dir / pass_name, cand_dir / pass_name)
    errors: list[str] = []
    if only_ref or only_cand:
        logger.warning("phrases in only one run (not compared): ref %s, cand %s", only_ref, only_cand)
    if not ids:
        errors.append(f"no phrase has a WAV in both {ref_dir / pass_name} and {cand_dir / pass_name}")

    embed_memo = _Memo(embed) if "spk" in metrics and embed is not None else None
    transcribe_memo = _Memo(transcribe) if "wer" in metrics and transcribe is not None else None
    ref_embedding = embed_memo(reference) if embed_memo is not None and reference is not None else None

    records: list[dict] = []
    for phrase_id in ids:
        rec = new_phrase_record(phrase_id)
        try:
            ref_audio = read_wav(ref_dir / pass_name / f"{phrase_id}.wav")
            cand_audio = read_wav(cand_dir / pass_name / f"{phrase_id}.wav")
            _measure_phrase(
                rec,
                ref_audio,
                cand_audio,
                metrics,
                text=texts.get(phrase_id),
                ref_embedding=ref_embedding,
                embed=embed_memo,
                transcribe=transcribe_memo,
            )
        except Exception as e:  # one unreadable WAV or failed inference must not hide the other phrases
            rec["error"] = f"{type(e).__name__}: {e}"
            errors.append(f"{phrase_id}: {rec['error']}")
            logger.error("%s: %s", phrase_id, rec["error"])
        else:
            logger.info(
                "%s: snr %s dB, spk %s -> %s, wer %s -> %s, dur_dev %s",
                phrase_id,
                _round(rec["snr_db"], 2),
                _round(rec["spk_cos_ref"], 4),
                _round(rec["spk_cos_cand"], 4),
                _round(rec["wer_ref"], 3),
                _round(rec["wer_cand"], 3),
                _round(rec["dur_dev"], 4),
            )
        records.append(rec)

    return {
        "schema": SCHEMA,
        "ref": ref_dir.name,
        "cand": cand_dir.name,
        "pass": pass_name,
        "metrics": list(metrics),
        "ref_dir": str(ref_dir),
        "cand_dir": str(cand_dir),
        "created_at": _now_iso(),
        "only_ref": only_ref,
        "only_cand": only_cand,
        "phrases": [round_fields(r) for r in records],
        "summary": round_fields(summarize(records, metrics)),
        "errors": errors,
    }


def run(args: argparse.Namespace) -> int:
    """Compare two bench runs as the CLI asks and write ``--out``; 0 if at least one phrase was compared."""
    metrics = parse_metrics(args.metrics)
    config = _read_json(Path(args.config))
    bench_cfg: dict = config.get("bench") or {}
    compare_cfg: dict = config.get("compare") or {}
    ref_dir, cand_dir = Path(args.ref).resolve(), Path(args.cand).resolve()
    for label, folder in (("--ref", ref_dir), ("--cand", cand_dir)):
        if not (folder / args.pass_name).is_dir():
            raise CompareError(f"{label}: {folder / args.pass_name} does not exist")
    phrase_file = _read_json(PHRASES_FILE)
    texts = {str(p["id"]): str(p["text"]) for p in phrase_file.get("phrases") or []}
    language = str(bench_cfg.get("language") or phrase_file.get("language") or "es")

    models: dict[str, str | None] = {"speaker": None, "speaker_device": None, "whisper": None, "whisper_device": None}
    reference = embed = transcribe = None
    reference_paths: list[Path] = []
    if "spk" in metrics:
        profile_id = str(bench_cfg.get("profile_id") or "")
        if not profile_id:
            raise CompareError("config.bench.profile_id is required for the spk metric")
        reference_paths = profile_reference_paths(resolve_data_dir(bench_cfg), profile_id)
        reference = load_reference_audio(reference_paths)
        embedder = QwenSpeakerEmbedder(speaker_repo(bench_cfg.get("model_size")))
        _load_model("speaker encoder", embedder.load)
        embed = embedder
        models.update(speaker=embedder.repo_id, speaker_device="cpu")
    if "wer" in metrics:
        transcriber = WhisperTranscriber(
            whisper_repo(compare_cfg.get("whisper_model")), language=language, device=resolve_device(args.device)
        )
        _load_model("Whisper model", transcriber.load)
        transcribe = transcriber
        models.update(whisper=transcriber.repo_id, whisper_device=transcriber.device)

    logger.info("comparing %s against %s (pass %s, metrics %s)", cand_dir, ref_dir, args.pass_name, ",".join(metrics))
    doc = compare_dirs(
        ref_dir, cand_dir, args.pass_name, metrics, texts=texts, reference=reference, embed=embed, transcribe=transcribe
    )
    doc["language"] = language
    doc["models"] = models
    doc["reference_audio"] = [str(p) for p in reference_paths]
    out = Path(args.out)
    _write_json_atomic(out, doc)
    summary = doc["summary"]
    logger.info(
        "wrote %s: n=%s, snr min %s dB, spk delta %s, wer delta %s, dur dev max %s",
        out,
        summary["n"],
        summary["snr_db_min"],
        summary["spk_cos_delta_mean"],
        summary["wer_delta"],
        summary["dur_dev_max"],
    )
    return 0 if summary["n"] > 0 else 1


# Self-test (no GPU, no models)


class _StubEmbedder:
    """Deterministic stand-in for the speaker model: a few statistics of the samples; counts its calls."""

    def __init__(self) -> None:
        self.calls = 0

    def __call__(self, audio: Audio) -> np.ndarray:
        self.calls += 1
        x = audio.samples.astype(np.float64)
        return np.array([1.0, float(np.mean(np.abs(x))), float(np.std(x)), len(x) / 1e5])


class _StubTranscriber:
    """Stand-in for Whisper: returns the text registered for an audio key (or a fixed one); counts its calls."""

    def __init__(self, by_key: dict[str, str], default: str = "texto desconocido") -> None:
        self.by_key = by_key
        self.default = default
        self.calls = 0

    def __call__(self, audio: Audio) -> str:
        self.calls += 1
        return self.by_key.get(audio.key, self.default)


def _expect(failures: list[str], label: str, got: Any, want: Any) -> None:
    if got != want:
        failures.append(f"{label}: got {got!r}, want {want!r}")


def _expect_close(failures: list[str], label: str, got: Any, want: float, tol: float) -> None:
    if not _is_number(got) or abs(got - want) > tol:
        failures.append(f"{label}: got {got!r}, want {want!r} +- {tol}")


def _tone(seconds: float, sr: int = 24000, freq: float = 220.0, amp: float = 0.5) -> np.ndarray:
    t = np.arange(round(seconds * sr)) / sr
    return (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def _write_wav(path: Path, samples: np.ndarray, sr: int = 24000, subtype: str = "PCM_16") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), samples, sr, subtype=subtype)


def _selftest_snr(failures: list[str]) -> None:
    sr = 24000
    ref = _tone(2.0, sr).astype(np.float64)  # 440 whole periods: mean power exactly amp^2 / 2 = 0.125
    _expect(failures, "snr: identical", snr_db(ref, ref.copy()), SNR_CAP_DB)
    sigma = 0.005
    alternating = sigma * np.where(np.arange(len(ref)) % 2 == 0, 1.0, -1.0)
    expected = 10 * math.log10(0.125 / sigma**2)
    _expect_close(failures, "snr: +-sigma noise", snr_db(ref, ref + alternating), expected, 0.5)
    gaussian = np.random.default_rng(1234).normal(0.0, 0.01, len(ref))
    expected_gaussian = 10 * math.log10(0.125 / 0.01**2)
    _expect_close(failures, "snr: gaussian noise", snr_db(ref, ref + gaussian), expected_gaussian, 0.5)

    longer = np.concatenate([ref, np.random.default_rng(1).normal(0.0, 0.3, 4000)])
    _expect(failures, "snr: cand longer, same common part", snr_db(ref, longer), SNR_CAP_DB)
    _expect(failures, "snr: cand shorter, same common part", snr_db(ref, ref[:-500]), SNR_CAP_DB)
    _expect(failures, "snr: ref shorter, same common part", snr_db(ref[:-500], ref), SNR_CAP_DB)
    trimmed = (ref + gaussian)[:-700]
    _expect_close(failures, "snr: different lengths with noise", snr_db(ref, trimmed), expected_gaussian, 0.5)
    _expect(failures, "snr: both empty", snr_db(np.zeros(0), np.zeros(0)), SNR_CAP_DB)
    _expect(failures, "snr: nothing in common", snr_db(np.zeros(0), ref), -SNR_CAP_DB)
    _expect(failures, "snr: silent ref", snr_db(np.zeros(100), np.ones(100)), -SNR_CAP_DB)

    pcm = np.round(ref * 32767) / 32767
    _expect_close(failures, "snr: 16-bit quantization", snr_db(ref, pcm), 10 * math.log10(0.125 * 12 * 32767**2), 1.0)


def _selftest_text(failures: list[str]) -> None:
    _expect(failures, "norm: case, punctuation, spaces", normalize_text("  ¡Hola,   MUNDO!  "), "hola mundo")
    _expect(failures, "norm: NFC", normalize_text("cancio\u0301n"), "canci\u00f3n")
    _expect(failures, "norm: duplicated accent", normalize_text("cancio\u0301\u0301n"), "canci\u00f3n")
    _expect(failures, "norm: precomposed + combining", normalize_text("canci\u00f3\u0301n"), "canci\u00f3n")
    _expect(failures, "norm: keeps ñ", normalize_text("NIÑO"), "ni\u00f1o")
    _expect(failures, "norm: keeps accents", normalize_text("Qué"), "qu\u00e9")
    _expect(failures, "norm: quotes and dashes", normalize_text("«Sí» —dijo— (vale)."), "s\u00ed dijo vale")
    _expect(failures, "norm: soft hyphen", normalize_text("pa\u00adlabra"), "palabra")
    _expect(failures, "norm: whitespace", normalize_text("tab\tnueva\nlínea"), "tab nueva l\u00ednea")
    _expect(failures, "norm: digits", normalize_text("Él vino a las 3."), "\u00e9l vino a las 3")
    _expect(failures, "norm: stray accent", normalize_text("hola \u0301 mundo"), "hola mundo")
    _expect(failures, "norm: empty", normalize_text(" ¿? "), "")

    _expect(failures, "wer: identical", word_error_rate("el gato come pescado", "el gato come pescado"), 0.0)
    _expect(failures, "wer: substitution", word_error_rate("el gato come pescado", "el perro come pescado"), 0.25)
    _expect(failures, "wer: deletion", word_error_rate("el gato come pescado", "el gato pescado"), 0.25)
    _expect(failures, "wer: insertion", word_error_rate("el gato come", "el gato negro come"), 1 / 3)
    _expect(failures, "wer: normalized", word_error_rate("¿Qué hora es?", "Qué  hora es."), 0.0)
    _expect(failures, "wer: accent counts", word_error_rate("¿Qué hora es?", "que hora es"), 1 / 3)
    _expect(failures, "wer: empty hypothesis", word_error_rate("uno dos tres", ""), 1.0)
    _expect(failures, "wer: empty both", word_error_rate("", " . "), 0.0)
    _expect(failures, "wer: empty reference", word_error_rate("", "algo"), 1.0)
    _expect(failures, "wer: reversed", word_errors(["a", "b", "c"], ["c", "b", "a"]), 2)
    _expect(failures, "wer: many errors", word_error_rate("a b", "c d e f"), 2.0)


def _selftest_basics(failures: list[str]) -> None:
    _expect(failures, "metrics: all", parse_metrics("all"), ["snr", "spk", "wer", "dur"])
    _expect(failures, "metrics: canonical order", parse_metrics("dur, SNR"), ["snr", "dur"])
    _expect(failures, "metrics: default", parse_metrics(DEFAULT_METRICS), ["snr", "dur"])
    for bad in ("", "snr,foo", " , "):
        try:
            parse_metrics(bad)
        except CompareError:
            pass
        else:
            failures.append(f"metrics: {bad!r} should fail")

    vec = np.array([0.3, -1.2, 2.0])
    _expect_close(failures, "cosine: same", cosine(vec, vec), 1.0, 1e-12)
    _expect_close(failures, "cosine: opposite", cosine(vec, -vec), -1.0, 1e-12)
    _expect_close(failures, "cosine: orthogonal", cosine(np.array([1.0, 0.0]), np.array([0.0, 2.0])), 0.0, 1e-12)
    _expect_close(failures, "dur: deviation", duration_deviation(2.0, 2.1), 0.05, 1e-12)
    _expect(failures, "dur: equal", duration_deviation(3.12, 3.12), 0.0)
    _expect(failures, "whisper repo", whisper_repo("small"), "openai/whisper-small")
    _expect(failures, "whisper repo: full id", whisper_repo("org/model"), "org/model")
    _expect(failures, "speaker repo", speaker_repo(None), "Qwen/Qwen3-TTS-12Hz-1.7B-Base")
    _expect(failures, "device: cpu without torch", resolve_device("cpu"), "cpu")

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        stereo = np.stack([_tone(0.5), _tone(0.5, freq=330.0)], axis=1)
        _write_wav(tmp_path / "stereo.wav", stereo)
        audio = read_wav(tmp_path / "stereo.wav")
        _expect(failures, "wav: stereo to mono", (audio.samples.ndim, len(audio.samples), audio.sr), (1, 12000, 24000))
        _expect(failures, "wav: duration", audio.duration_s, 0.5)
        _write_wav(tmp_path / "a.wav", _tone(0.25))
        _write_wav(tmp_path / "b.wav", _tone(0.25))
        _write_wav(tmp_path / "c.wav", _tone(0.25), sr=16000)
        same = (read_wav(tmp_path / "a.wav").key, read_wav(tmp_path / "b.wav").key)
        _expect(failures, "wav: same samples, same key", same[0] == same[1], True)
        _expect(failures, "wav: other rate, other key", read_wav(tmp_path / "c.wav").key == same[0], False)


def make_fake_runs(root: Path) -> None:
    """Write two fake bench runs under ``root``.

    ``ref/warm`` has p01-p03; ``cand/warm`` has p01 (identical), p02 (Gaussian noise of sigma 0.01 on a
    0.5-amplitude tone, about 31 dB, and 10 % longer) and p04.
    """
    tone = _tone(1.0)
    longer = _tone(1.5, freq=330.0)
    noisy = longer + np.random.default_rng(7).normal(0.0, 0.01, len(longer)).astype(np.float32)
    noisy_longer = np.concatenate([noisy, np.zeros(3600, dtype=np.float32)])
    for name, samples in (("p01", tone), ("p02", longer), ("p03", tone)):
        _write_wav(root / "ref" / "warm" / f"{name}.wav", samples)
    for name, samples in (("p01", tone), ("p02", noisy_longer), ("p04", tone)):
        _write_wav(root / "cand" / "warm" / f"{name}.wav", samples)


SELFTEST_TEXTS = {
    "p01": "La mañana amaneció tranquila.",
    "p02": "Mi vecino riega las plantas cada tarde.",
    "p03": "El tren salió puntual.",
}


def _selftest_compare(failures: list[str]) -> None:
    with tempfile.TemporaryDirectory() as tmp, _quiet_logger():
        root = Path(tmp)
        make_fake_runs(root)
        reference = make_audio(_tone(2.0, freq=180.0), 24000)
        ref_p02 = read_wav(root / "ref" / "warm" / "p02.wav")
        cand_p02 = read_wav(root / "cand" / "warm" / "p02.wav")
        transcripts = {ref_p02.key: "mi vecino riega las plantas cada tarde", cand_p02.key: "mi vecino riega plantas"}
        transcripts[read_wav(root / "ref" / "warm" / "p01.wav").key] = "La mañana amaneció, tranquila"

        embed, transcribe = _StubEmbedder(), _StubTranscriber(transcripts)
        same = compare_dirs(
            root / "ref",
            root / "ref",
            "warm",
            list(ALL_METRICS),
            texts=SELFTEST_TEXTS,
            reference=reference,
            embed=embed,
            transcribe=transcribe,
        )
        summary = same["summary"]
        _expect(failures, "self: n", summary["n"], 3)
        _expect(failures, "self: snr", (summary["snr_db_min"], summary["snr_db_mean"]), (SNR_CAP_DB, SNR_CAP_DB))
        _expect(failures, "self: spk delta", summary["spk_cos_delta_mean"], 0.0)
        _expect(failures, "self: wer delta", summary["wer_delta"], 0.0)
        _expect(failures, "self: dur dev", summary["dur_dev_max"], 0.0)
        _expect(failures, "self: spk ref = cand", summary["spk_cos_ref_mean"], summary["spk_cos_cand_mean"])
        _expect(failures, "self: one embedding per distinct audio", embed.calls, 3)
        _expect(failures, "self: one transcription per distinct audio", transcribe.calls, 2)
        _expect(failures, "self: wer p01", same["phrases"][0]["wer_ref"], 0.0)
        _expect(failures, "self: no errors", same["errors"], [])

        diff = compare_dirs(
            root / "ref",
            root / "cand",
            "warm",
            list(ALL_METRICS),
            texts=SELFTEST_TEXTS,
            reference=reference,
            embed=_StubEmbedder(),
            transcribe=_StubTranscriber(transcripts),
        )
        _expect(failures, "diff: ids", [p["id"] for p in diff["phrases"]], ["p01", "p02"])
        _expect(failures, "diff: only ref / cand", (diff["only_ref"], diff["only_cand"]), (["p03"], ["p04"]))
        p01, p02 = diff["phrases"]
        _expect(
            failures, "diff: identical phrase", (p01["snr_db"], p01["spk_cos_delta"], p01["dur_dev"]), (999.0, 0, 0)
        )
        _expect_close(failures, "diff: noisy snr", p02["snr_db"], 10 * math.log10(0.125 / 1e-4), 0.5)
        _expect_close(failures, "diff: dur dev", p02["dur_dev"], 0.1, 1e-6)
        _expect(failures, "diff: wer ref/cand", (p02["wer_ref"], p02["wer_cand"]), (0.0, round(3 / 7, 6)))
        _expect(failures, "diff: summary wer delta", diff["summary"]["wer_delta"], round(3 / 7 / 2, 6))
        _expect(failures, "diff: summary snr min", diff["summary"]["snr_db_min"], p02["snr_db"])
        _expect(failures, "diff: summary dur", diff["summary"]["dur_dev_max"], p02["dur_dev"])
        spk_delta = diff["summary"]["spk_cos_delta_mean"]
        _expect(failures, "diff: spk delta is a number", _is_number(spk_delta) and spk_delta != 0, True)

        light = compare_dirs(root / "ref", root / "cand", "warm", ["snr", "dur"], texts={})
        nulls = [light["phrases"][0][k] for k in ("spk_cos_ref", "spk_cos_cand", "wer_ref", "wer_cand", "hyp_ref")]
        _expect(failures, "snr,dur: model fields null", nulls, [None] * 5)
        light_summary = light["summary"]
        model_keys = ("spk_cos_ref_mean", "spk_cos_cand_mean", "spk_cos_delta_mean", "wer_ref_mean", "wer_delta")
        _expect(failures, "snr,dur: model summary null", [light_summary[k] for k in model_keys], [None] * 5)
        _expect(failures, "snr,dur: n", light_summary["n"], 2)
        for key in ("n", *SUMMARY_KEYS):
            if key not in light_summary:
                failures.append(f"summary: missing {key}")
        for key in ("id", "snr_db", "spk_cos_ref", "spk_cos_cand", "wer_ref", "wer_cand", "dur_ref_s", "dur_cand_s"):
            if key not in light["phrases"][0]:
                failures.append(f"phrase: missing {key}")
        json.dumps(diff, allow_nan=False)

        try:
            compare_dirs(root / "ref", root / "ref", "warm", ["spk"], texts={})
        except CompareError:
            pass
        else:
            failures.append("compare: spk without a model should fail")

        _write_wav(root / "cand" / "warm" / "p03.wav", _tone(1.0), sr=16000)
        broken = compare_dirs(root / "ref", root / "cand", "warm", ["snr", "dur"], texts={})
        bad = next(p for p in broken["phrases"] if p["id"] == "p03")
        _expect(failures, "rates differ: phrase error", bool(bad["error"]) and "sample rates" in bad["error"], True)
        _expect(failures, "rates differ: excluded from n", broken["summary"]["n"], 2)

        empty = compare_dirs(root / "ref", root / "nothing", "warm", ["snr"], texts={})
        _expect(failures, "no common phrases: n", (empty["summary"]["n"], bool(empty["errors"])), (0, True))


def _selftest_reference(failures: list[str]) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        data_dir = Path(tmp) / "data"
        profile_dir = data_dir / "profiles" / "prof-1"
        _write_wav(profile_dir / "s1.wav", _tone(0.5))
        _write_wav(profile_dir / "s2.wav", _tone(0.25, freq=330.0))
        with contextlib.closing(sqlite3.connect(data_dir / "voicebox.db")) as conn:
            conn.execute("CREATE TABLE profile_samples (id TEXT, profile_id TEXT, audio_path TEXT)")
            rows = [
                ("s1", "prof-1", "profiles\\prof-1\\s1.wav"),
                ("s2", "prof-1", str(Path("/srv/old/data/profiles/prof-1/s2.wav"))),
                ("x", "other", "profiles\\other\\x.wav"),
            ]
            conn.executemany("INSERT INTO profile_samples VALUES (?, ?, ?)", rows)
            conn.commit()
        paths = profile_reference_paths(data_dir, "prof-1")
        _expect(failures, "reference: paths", paths, [profile_dir / "s1.wav", profile_dir / "s2.wav"])
        combined = load_reference_audio(paths)
        _expect(failures, "reference: concatenated", (len(combined.samples), combined.sr), (18000, 24000))
        single = load_reference_audio(paths[:1])
        _expect(failures, "reference: single sample", single.key, read_wav(profile_dir / "s1.wav").key)
        _expect(
            failures,
            "rebase: data prefix",
            rebase_storage_path("data/profiles/a.wav", data_dir),
            data_dir / "profiles" / "a.wav",
        )
        _expect(failures, "rebase: outside data", rebase_storage_path(str(Path("/x/y.wav")), data_dir), None)
        for profile in ("other", "missing"):
            try:
                profile_reference_paths(data_dir, profile)
            except CompareError:
                pass
            else:
                failures.append(f"reference: profile {profile!r} should fail")
        _expect(failures, "data dir: relative to worktree", resolve_data_dir({}), (ROOT / DEFAULT_DATA_DIR).resolve())


def _selftest_cli(failures: list[str]) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        make_fake_runs(root)
        config = root / "config.json"
        config.write_text(json.dumps({"bench": {"profile_id": "prof-1"}}), encoding="utf-8")
        out = root / "out" / "compare.json"
        argv = ["--config", str(config), "--ref", str(root / "ref"), "--cand", str(root / "ref"), "--out", str(out)]
        with _quiet_logger():
            code = main([*argv, "--metrics", "snr,dur"])
        doc = json.loads(out.read_text(encoding="utf-8"))
        _expect(failures, "cli: exit code", code, 0)
        _expect(
            failures,
            "cli: schema and pass",
            (doc["schema"], doc["pass"], doc["metrics"]),
            (SCHEMA, "warm", ["snr", "dur"]),
        )
        _expect(failures, "cli: ref name", (doc["ref"], doc["cand"]), ("ref", "ref"))
        _expect(failures, "cli: self snr", (doc["summary"]["n"], doc["summary"]["snr_db_min"]), (3, SNR_CAP_DB))
        _expect(
            failures,
            "cli: models unused",
            doc["models"],
            dict.fromkeys(("speaker", "speaker_device", "whisper", "whisper_device")),
        )
        with _quiet_logger():
            missing = main([*argv[:4], "--cand", str(root / "nothing"), "--out", str(out)])
            wrong_pass = main([*argv, "--pass", "cold"])
        _expect(failures, "cli: missing cand folder", missing, 1)
        _expect(failures, "cli: missing pass folder", wrong_pass, 1)


@contextlib.contextmanager
def _quiet_logger() -> Any:
    """Hide the INFO and ERROR lines of deliberate runs so the self-test output stays readable."""
    level = logger.level
    logger.setLevel(logging.CRITICAL)
    try:
        yield
    finally:
        logger.setLevel(level)


SELF_TESTS: tuple[Callable[[list[str]], None], ...] = (
    _selftest_snr,
    _selftest_text,
    _selftest_basics,
    _selftest_compare,
    _selftest_reference,
    _selftest_cli,
)


def run_self_test() -> int:
    """Run every self-check without GPU or models; return 0 if all pass."""
    failures: list[str] = []
    for check in SELF_TESTS:
        before = len(failures)
        try:
            check(failures)
        except Exception as e:  # a crash in one check must not hide the others
            failures.append(f"{check.__name__}: crashed with {type(e).__name__}: {e}")
        status = "ok" if len(failures) == before else "FAILED"
        logger.info("self-test %s: %s", check.__name__.removeprefix("_selftest_"), status)
    for failure in failures:
        logger.error("self-test: %s", failure)
    if failures:
        logger.error("self-test failed (%d problem(s))", len(failures))
        return 1
    logger.info("self-test passed")
    return 0


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Voicebox quality metrics between two bench runs (contracts, §3)")
    parser.add_argument("--config", help="perf-loop/config.json")
    parser.add_argument("--ref", help="reference bench run (perf-loop/bench/<tag>)")
    parser.add_argument("--cand", help="candidate bench run (perf-loop/bench/<tag>)")
    parser.add_argument("--pass", dest="pass_name", default=DEFAULT_PASS, help=f"pass folder (default {DEFAULT_PASS})")
    parser.add_argument(
        "--metrics",
        default=DEFAULT_METRICS,
        help=f"comma-separated among {', '.join(ALL_METRICS)}, or all (default {DEFAULT_METRICS})",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu"),
        default="auto",
        help="Whisper device: auto uses the GPU if there is one (the speaker encoder always runs on the CPU)",
    )
    parser.add_argument("--out", help="output JSON file")
    parser.add_argument("--self-test", action="store_true", help="check the metrics without GPU or models")
    return parser


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [compare] %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stderr,
    )
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    if args.self_test:
        return run_self_test()
    missing = [opt for opt in ("config", "ref", "cand", "out") if not getattr(args, opt)]
    if missing:
        parser.error("missing " + ", ".join(f"--{m}" for m in missing))
    try:
        return run(args)
    except CompareError as e:
        logger.error("%s", e)
        return 1
    except KeyboardInterrupt:
        logger.error("interrupted")
        return 130


if __name__ == "__main__":
    sys.exit(main())
