"""Qwen3-TTS benchmark for the Voicebox perf loop (perf-loop/contracts.md, section 2).

Starts the development server (``python -m backend.main``) on its own port and data dir, runs the phrases from
``scripts/bench_phrases.json`` over HTTP and writes ``<out>/bench.json`` with timings, WAVs, per-stage VBPERF data
and MIOpen Find-Db line counts. ``cold`` and ``warm`` each use a brand-new server process; ``hot`` repeats the
same phrases in the process of the previous pass (no restart, no reload), so every length is already seen.

The script never imports anything from ``backend/`` (contracts, section 5): the evaluator must not change when the
code it evaluates changes. It only needs the standard library.

Usage:
    python scripts/bench_tts.py --config perf-loop/config.json --tag <tag> --out perf-loop/bench/<tag>
        [--phrases all|p01,p02,...] [--passes cold,warm,hot] [--miopen-db fresh|persist]
        [--env KEY=VALUE]... [--scenario contention|batch] [--concurrent N]
    python scripts/bench_tts.py --self-test
"""

import argparse
import contextlib
import hashlib
import http.client
import io
import json
import logging
import os
import re
import shutil
import signal
import socket
import sqlite3
import statistics
import subprocess
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
import wave
from collections.abc import Callable, Iterable
from datetime import datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger("bench_tts")

SCHEMA = "vb-bench/1"
ROOT = Path(__file__).resolve().parents[1]
PHRASES_FILE = ROOT / "scripts" / "bench_phrases.json"

VBPERF_PREFIX = "VBPERF "
CODEC_FRAMES_PER_AUDIO_S = 12.5
VALID_PASSES = ("cold", "warm", "hot")
# Passes that reuse the server process of the pass right before them instead of starting a new one.
REUSE_PASSES = frozenset({"hot"})
VALID_SCENARIOS = ("contention", "batch")
TERMINAL_STATUSES = frozenset({"completed", "failed"})
# Fields summed when one generation reports the same stage more than once (e.g. several text chunks).
ADDITIVE_STAGE_FIELDS = ("ms", "steps", "chunks", "T")

POLL_INTERVAL_S = 0.2
DEFAULT_START_TIMEOUT_S = 600.0
DEFAULT_PHRASE_TIMEOUT_S = 1800.0
DEFAULT_BATCH_N = 4
SHUTDOWN_GRACE_S = 30.0
PORT_RELEASE_TIMEOUT_S = 60.0
POST_TIMEOUT_S = 120.0
LOG_TAIL_LINES = 40

IS_WINDOWS = os.name == "nt"
CREATE_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0x08000000)
DIRTY_CMD = ("git", "status", "--porcelain", "--untracked-files=all", "--", ".", ":(exclude)perf-loop")
NET_ERRORS = (OSError, http.client.HTTPException)

_ANSI_RE = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")
# Uvicorn's startup line; on Windows the venv python.exe is a launcher, so Popen.pid is not the server's PID.
_SERVER_PID_RE = re.compile(r"Started server process \[(\d+)\]")


class BenchError(Exception):
    """Fatal benchmark error; the message goes to stderr and the exit code is non-zero."""


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------


def _is_number(value: Any) -> bool:
    return isinstance(value, int | float) and not isinstance(value, bool)


def _round(value: float | None, ndigits: int = 3) -> float | None:
    return round(value, ndigits) if _is_number(value) else None


def _mean(values: list[float]) -> float | None:
    return round(statistics.fmean(values), 3) if values else None


def _median(values: list[float]) -> float | None:
    return round(statistics.median(values), 3) if values else None


def _is_within(path: Path, parent: Path) -> bool:
    try:
        path.resolve().relative_to(parent.resolve())
    except ValueError:
        return False
    return True


def _now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        raise BenchError(f"cannot read {path}: {e}") from e


def _write_json_atomic(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(tmp, path)


# ---------------------------------------------------------------------------
# CLI parsing
# ---------------------------------------------------------------------------


def select_phrases(phrases: list[dict], spec: str | None) -> list[dict]:
    """Return the requested phrases in file order (``all`` or a comma-separated id list)."""
    spec = (spec or "all").strip()
    if spec == "all":
        return list(phrases)
    wanted = [p.strip() for p in spec.split(",") if p.strip()]
    if not wanted:
        raise BenchError("--phrases is empty")
    known = {p["id"] for p in phrases}
    unknown = [w for w in wanted if w not in known]
    if unknown:
        raise BenchError(f"unknown phrase ids: {', '.join(unknown)}")
    chosen = set(wanted)
    return [p for p in phrases if p["id"] in chosen]


def parse_passes(spec: str | None) -> list[str]:
    """Parse ``--passes`` (execution order, each name at most once; ``hot`` needs a pass before it)."""
    names = [p.strip() for p in (spec or "").split(",") if p.strip()]
    if not names:
        raise BenchError("--passes is empty")
    bad = [n for n in names if n not in VALID_PASSES]
    if bad:
        raise BenchError(f"unknown passes: {', '.join(bad)} (valid: {', '.join(VALID_PASSES)})")
    if len(set(names)) != len(names):
        raise BenchError("--passes repeats a pass")
    if names[0] in REUSE_PASSES:
        raise BenchError(f"--passes cannot start with {names[0]}: it reuses the server of the previous pass")
    return names


def parse_env_args(items: Iterable[str] | None) -> dict[str, str]:
    """Parse repeated ``--env KEY=VALUE`` options."""
    env: dict[str, str] = {}
    for item in items or []:
        key, sep, value = item.partition("=")
        key = key.strip()
        if not sep or not key:
            raise BenchError(f"--env expects KEY=VALUE, got {item!r}")
        env[key] = value
    return env


# ---------------------------------------------------------------------------
# Server command and environment
# ---------------------------------------------------------------------------


def build_server_cmd(python: str | Path, port: int, data_dir: Path) -> list[str]:
    """Command that starts the development server, as the contract prescribes."""
    return [str(python), "-m", "backend.main", "--port", str(port), "--data-dir", str(data_dir)]


def build_server_env(base_env: dict[str, str], *, extra_env: dict[str, str], fresh_db_dir: Path | None) -> dict:
    """Inherited env + fresh MIOpen dirs (only in ``fresh`` mode) + config/CLI variables (these win)."""
    env = dict(base_env)
    # Only affects how the server's own stdout/stderr are written to the log, not what it computes.
    env.setdefault("PYTHONUNBUFFERED", "1")
    env.setdefault("PYTHONIOENCODING", "utf-8")
    if fresh_db_dir is not None:
        env["MIOPEN_USER_DB_PATH"] = str(fresh_db_dir)
        env["MIOPEN_CUSTOM_CACHE_DIR"] = str(fresh_db_dir)
    env.update({str(k): str(v) for k, v in extra_env.items()})
    return env


def resolve_server_python(root: Path, config: dict) -> Path:
    """Python from ``config.python`` (relative to the worktree), or the one running this script."""
    configured = config.get("python")
    if configured:
        candidate = Path(configured)
        if not candidate.is_absolute():
            candidate = root / candidate
        if candidate.is_file():
            return candidate
        logger.warning("config.python %s not found; using %s", candidate, sys.executable)
    return Path(sys.executable)


def git_state(root: Path) -> tuple[str, bool]:
    """Return (HEAD commit, dirty) where files under perf-loop/ do not count as dirty."""
    try:
        head = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True, check=True
        ).stdout.strip()
        status = subprocess.run(list(DIRTY_CMD), cwd=root, capture_output=True, text=True, check=True).stdout
    except (OSError, subprocess.CalledProcessError) as e:
        raise BenchError(f"git failed in {root}: {e}") from e
    return head, bool(status.strip())


# ---------------------------------------------------------------------------
# VBPERF log lines (contracts, section 4)
# ---------------------------------------------------------------------------


def parse_vbperf_lines(lines: Iterable[str]) -> list[dict]:
    """Extract the JSON payloads of ``VBPERF {...}`` lines; anything before the prefix is ignored."""
    events: list[dict] = []
    for raw in lines:
        idx = raw.find(VBPERF_PREFIX)
        if idx < 0:
            continue
        payload = _ANSI_RE.sub("", raw[idx + len(VBPERF_PREFIX) :]).strip()
        try:
            obj = json.loads(payload)
        except ValueError:
            logger.debug("ignoring malformed VBPERF line: %s", raw.rstrip())
            continue
        if isinstance(obj, dict):
            events.append(obj)
    return events


def merge_stage(existing: dict | None, new: dict) -> dict:
    """Combine two events of the same stage and generation (sums durations and counters)."""
    if existing is None:
        return dict(new)
    merged = {**existing, **new}
    for key in ADDITIVE_STAGE_FIELDS:
        a, b = existing.get(key), new.get(key)
        if _is_number(a) and _is_number(b):
            merged[key] = a + b
    steps, ms = merged.get("steps"), merged.get("ms")
    if "steps_per_s" in merged and _is_number(steps) and _is_number(ms) and ms > 0:
        merged["steps_per_s"] = round(steps / (ms / 1000.0), 3)
    return merged


def collect_vbperf(events: Iterable[dict]) -> tuple[dict | None, dict[str, dict[str, dict]]]:
    """Return (last ``stage=env`` payload or None, {gen_id: {stage: payload}})."""
    env_payload: dict | None = None
    stages: dict[str, dict[str, dict]] = {}
    for event in events:
        stage = event.get("stage")
        if stage == "env":
            env_payload = dict(event)
            continue
        gen_id = event.get("gen_id")
        if not isinstance(stage, str) or not gen_id:
            continue
        payload = {k: v for k, v in event.items() if k not in ("gen_id", "stage")}
        per_gen = stages.setdefault(str(gen_id), {})
        per_gen[stage] = merge_stage(per_gen.get(stage), payload)
    return env_payload, stages


# ---------------------------------------------------------------------------
# MIOpen Find-Db
# ---------------------------------------------------------------------------


def default_miopen_db_dir() -> Path:
    """MIOpen's default user Find-Db on Windows (``%USERPROFILE%\\.miopen\\db``)."""
    return Path.home() / ".miopen" / "db"


def resolve_miopen_db_dir(env_payload: dict | None, server_env: dict[str, str]) -> Path:
    """Effective Find-Db: VBPERF env line, else the variable passed to the server, else MIOpen's default."""
    if env_payload and env_payload.get("miopen_user_db_path"):
        return Path(str(env_payload["miopen_user_db_path"]))
    if server_env.get("MIOPEN_USER_DB_PATH"):
        return Path(server_env["MIOPEN_USER_DB_PATH"])
    return default_miopen_db_dir()


def count_ufdb_lines(db_dir: Path) -> int | None:
    """Non-empty lines across every ``*.ufdb.txt`` under ``db_dir`` (0 if the folder does not exist yet)."""
    if not db_dir.is_dir():
        return 0
    total = 0
    try:
        for path in sorted(db_dir.rglob("*.ufdb.txt")):
            with path.open("rb") as fh:
                total += sum(1 for line in fh if line.strip())
    except OSError as e:
        logger.warning("cannot read the Find-Db in %s: %s", db_dir, e)
        return None
    return total


# ---------------------------------------------------------------------------
# WAV, per-phrase metrics and summary
# ---------------------------------------------------------------------------


def wav_duration_s(data: bytes) -> float | None:
    """Duration of a RIFF/WAVE byte string (any sample format), or None if it cannot be parsed."""
    if len(data) < 12 or data[:4] != b"RIFF" or data[8:12] != b"WAVE":
        return None
    pos, rate, block_align, data_size = 12, 0, 0, None
    while pos + 8 <= len(data):
        chunk_id = data[pos : pos + 4]
        size = int.from_bytes(data[pos + 4 : pos + 8], "little")
        body = pos + 8
        if chunk_id == b"fmt " and size >= 16:
            rate = int.from_bytes(data[body + 4 : body + 8], "little")
            block_align = int.from_bytes(data[body + 12 : body + 14], "little")
        elif chunk_id == b"data":
            data_size = min(size, len(data) - body)
            break
        pos = body + size + (size & 1)
    if not rate or not block_align or data_size is None:
        return None
    return (data_size // block_align) / rate


def new_phrase_record(phrase: dict) -> dict:
    """Phrase entry of bench.json with every field present (null when there is no data)."""
    return {
        "id": phrase["id"],
        "seed": phrase.get("seed"),
        "text_chars": len(phrase["text"]),
        "gen_id": None,
        "status": None,
        "wall_s": None,
        "audio_s": None,
        "steps": None,
        "rtf": None,
        "wav": None,
        "stages": {},
        "error": None,
    }


def finalize_phrase(rec: dict, stages: dict | None) -> None:
    """Attach VBPERF stages and derive ``steps`` (talker or audio_s x 12.5) and ``rtf`` (wall_s / audio_s)."""
    rec["stages"] = stages or {}
    audio_s, wall_s = rec.get("audio_s"), rec.get("wall_s")
    talker_steps = (rec["stages"].get("talker") or {}).get("steps")
    if _is_number(talker_steps):
        rec["steps"] = int(talker_steps)
    elif _is_number(audio_s):
        rec["steps"] = round(audio_s * CODEC_FRAMES_PER_AUDIO_S)
    else:
        rec["steps"] = None
    if _is_number(wall_s) and _is_number(audio_s) and audio_s > 0:
        rec["rtf"] = round(wall_s / audio_s, 3)
    else:
        rec["rtf"] = None


def _pass_values(pass_rec: dict | None, key: str) -> list[float]:
    if not pass_rec:
        return []
    return [x[key] for x in pass_rec.get("phrases", []) if _is_number(x.get(key))]


def _ufdb_added(pass_rec: dict | None) -> int | None:
    if not pass_rec:
        return None
    before, after = pass_rec.get("ufdb_lines_before"), pass_rec.get("ufdb_lines_after")
    if _is_number(before) and _is_number(after):
        return after - before
    return None


def _stage_means(pass_rec: dict | None) -> tuple[dict[str, float | None], float | None]:
    """Mean ms per VBPERF stage and mean talker steps/s over the phrases of one pass."""
    stage_values: dict[str, list[float]] = {}
    talker_sps: list[float] = []
    for rec in (pass_rec or {}).get("phrases", []):
        stages = rec.get("stages") or {}
        for name, payload in stages.items():
            if isinstance(payload, dict) and _is_number(payload.get("ms")):
                stage_values.setdefault(name, []).append(payload["ms"])
        sps = (stages.get("talker") or {}).get("steps_per_s")
        if _is_number(sps):
            talker_sps.append(sps)
    return {name: _mean(values) for name, values in stage_values.items()}, _mean(talker_sps)


def compute_summary(passes: list[dict], server: dict, phrases_n: int, ufdb_total: int | None) -> dict:
    """Summary block of bench.json.

    ``stage_ms_mean`` and ``talker_steps_per_s_mean`` use the warm pass (or the last pass if there is no warm one);
    ``stage_ms_mean_hot`` and ``talker_steps_per_s_mean_hot`` use the hot pass and are null without one.
    Per-pass means, medians, RTF and Find-Db growth are null for passes that were not measured.
    """
    by_name = {p["name"]: p for p in passes}
    hot = by_name.get("hot")
    stage_ms, talker_sps = _stage_means(by_name.get("warm") or (passes[-1] if passes else None))
    stage_ms_hot, talker_sps_hot = _stage_means(hot) if hot else (None, None)

    summary: dict[str, Any] = {}
    for name in VALID_PASSES:
        pass_rec = by_name.get(name)
        summary[f"{name}_mean_s"] = _mean(_pass_values(pass_rec, "wall_s"))
        summary[f"{name}_median_s"] = _median(_pass_values(pass_rec, "wall_s"))
        summary[f"rtf_{name}_mean"] = _mean(_pass_values(pass_rec, "rtf"))
    summary["talker_steps_per_s_mean"] = talker_sps
    summary["talker_steps_per_s_mean_hot"] = talker_sps_hot
    for name in VALID_PASSES:
        summary[f"ufdb_lines_added_{name}"] = _ufdb_added(by_name.get(name))
    summary["ufdb_lines_total"] = ufdb_total
    summary["stage_ms_mean"] = stage_ms
    summary["stage_ms_mean_hot"] = stage_ms_hot
    summary["load_s"] = (server.get(passes[0]["name"]) or {}).get("load_s") if passes else None
    summary["phrases_n"] = phrases_n
    return summary


def batch_reference_pass(passes: list[dict]) -> dict | None:
    """Pass whose individual ``wall_s`` the batch scenario compares against: hot, else warm, else the last one."""
    by_name = {p["name"]: p for p in passes}
    return by_name.get("hot") or by_name.get("warm") or (passes[-1] if passes else None)


def assemble_bench_json(
    *,
    tag: str,
    commit: str | None,
    dirty: bool,
    code_path: str,
    started_at: str,
    args: dict,
    env_checks: dict | None,
    server: dict,
    passes: list[dict],
    scenarios: dict,
    summary: dict,
    errors: list[str],
) -> dict:
    """Top-level bench.json document (contracts, section 2)."""
    return {
        "schema": SCHEMA,
        "tag": tag,
        "commit": commit,
        "dirty": dirty,
        "code_path": code_path,
        "started_at": started_at,
        "finished_at": _now_iso(),
        "args": args,
        "env_checks": env_checks,
        "server": server,
        "passes": passes,
        "scenarios": scenarios,
        "summary": summary,
        "errors": errors,
    }


# ---------------------------------------------------------------------------
# Bench data dir
# ---------------------------------------------------------------------------


def _db_has_profile(db_path: Path, profile_id: str) -> bool:
    try:
        with contextlib.closing(sqlite3.connect(db_path)) as conn:
            row = conn.execute("SELECT 1 FROM profiles WHERE id = ?", (profile_id,)).fetchone()
    except sqlite3.Error:
        return False
    return row is not None


def _sample_paths(db_path: Path, profile_id: str) -> list[str]:
    try:
        with contextlib.closing(sqlite3.connect(db_path)) as conn:
            query = "SELECT audio_path FROM profile_samples WHERE profile_id = ?"
            rows = conn.execute(query, (profile_id,)).fetchall()
    except sqlite3.Error as e:
        raise BenchError(f"cannot read the samples of profile {profile_id} from {db_path}: {e}") from e
    return [str(r[0]) for r in rows if r[0]]


def rebase_storage_path(stored: str, data_dir: Path) -> Path | None:
    """Where the server will look for a DB-stored path inside ``data_dir`` (mirrors its ``data`` rebasing)."""
    path = Path(stored)
    parts = path.parts
    if path.is_absolute():
        for idx, part in enumerate(parts):
            if part == "data":
                return data_dir.joinpath(*parts[idx + 1 :])
        return None
    if parts and parts[0] == "data":
        parts = parts[1:]
    return data_dir.joinpath(*parts)


def _copy_source_db(source: Path, data_dir: Path) -> None:
    src_db = source / "voicebox.db"
    if not src_db.is_file():
        raise BenchError(f"source database not found: {src_db}")
    for suffix in ("-wal", "-shm"):
        (data_dir / f"voicebox.db{suffix}").unlink(missing_ok=True)
    tmp = data_dir / "voicebox.db.tmp"
    shutil.copyfile(src_db, tmp)
    os.replace(tmp, data_dir / "voicebox.db")
    src_wal = source / "voicebox.db-wal"
    if src_wal.is_file():
        shutil.copyfile(src_wal, data_dir / "voicebox.db-wal")
    logger.info("copied %s into the bench data dir", src_db)


def _sync_tree(src: Path, dst: Path) -> int:
    """Copy files that are missing or differ (size/mtime) from ``src`` into ``dst``; never deletes."""
    copied = 0
    for path in sorted(src.rglob("*")):
        if not path.is_file():
            continue
        target = dst / path.relative_to(src)
        st = path.stat()
        if target.is_file():
            tst = target.stat()
            if tst.st_size == st.st_size and int(tst.st_mtime) == int(st.st_mtime):
                continue
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
        copied += 1
    return copied


def prepare_data_dir(root: Path, bench_cfg: dict) -> Path:
    """Create or update the bench data dir from ``source_data_dir`` (read only) and return its absolute path.

    Copies ``voicebox.db`` when the bench copy is missing or lacks the profile, and syncs the profile folder
    (its samples). The voice-prompt cache and any Find-Db the app keeps there persist between runs.
    """
    try:
        source = Path(bench_cfg["source_data_dir"]).resolve()
        profile_id = str(bench_cfg["profile_id"])
    except KeyError as e:
        raise BenchError(f"config.bench.{e.args[0]} is required") from e
    data_dir = Path(bench_cfg.get("data_dir") or "perf-loop/bench/_data")
    if not data_dir.is_absolute():
        data_dir = root / data_dir
    data_dir = data_dir.resolve()
    if not _is_within(data_dir, root):
        raise BenchError(f"config.bench.data_dir must live inside the worktree ({root}); got {data_dir}")
    if _is_within(data_dir, source) or _is_within(source, data_dir):
        raise BenchError(f"the bench data dir {data_dir} overlaps source_data_dir {source}")

    data_dir.mkdir(parents=True, exist_ok=True)
    db_path = data_dir / "voicebox.db"
    if not db_path.is_file() or not _db_has_profile(db_path, profile_id):
        _copy_source_db(source, data_dir)
        if not _db_has_profile(db_path, profile_id):
            raise BenchError(f"profile {profile_id} not found in {source / 'voicebox.db'}")

    src_profile = source / "profiles" / profile_id
    if src_profile.is_dir():
        copied = _sync_tree(src_profile, data_dir / "profiles" / profile_id)
        if copied:
            logger.info("copied %d profile file(s) into the bench data dir", copied)

    for stored in _sample_paths(db_path, profile_id):
        target = rebase_storage_path(stored, data_dir)
        if target is None:
            raise BenchError(f"profile sample {stored} is outside any data dir; cannot isolate it")
        if not target.is_file():
            raise BenchError(f"profile sample missing in the bench data dir: {target}")
    return data_dir


# ---------------------------------------------------------------------------
# HTTP
# ---------------------------------------------------------------------------


def build_multipart(fields: dict[str, str], files: dict[str, tuple[str, bytes, str]]) -> tuple[bytes, str]:
    """Encode a multipart/form-data body; returns (body, content type)."""
    boundary = f"----vbbench{uuid.uuid4().hex}"
    parts: list[bytes] = []
    for name, value in fields.items():
        parts.append(f'--{boundary}\r\nContent-Disposition: form-data; name="{name}"\r\n\r\n{value}\r\n'.encode())
    for name, (filename, content, ctype) in files.items():
        head = (
            f"--{boundary}\r\n"
            f'Content-Disposition: form-data; name="{name}"; filename="{filename}"\r\n'
            f"Content-Type: {ctype}\r\n\r\n"
        )
        parts.append(head.encode() + content + b"\r\n")
    parts.append(f"--{boundary}--\r\n".encode())
    return b"".join(parts), f"multipart/form-data; boundary={boundary}"


class HttpClient:
    """Minimal JSON/bytes HTTP client on top of urllib (no proxies)."""

    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip("/")
        self._opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))

    def request(
        self,
        method: str,
        path: str,
        *,
        body: bytes | None = None,
        headers: dict[str, str] | None = None,
        timeout: float = 30.0,
    ) -> tuple[int, bytes]:
        req = urllib.request.Request(self.base_url + path, data=body, method=method, headers=headers or {})
        try:
            with self._opener.open(req, timeout=timeout) as resp:
                return resp.status, resp.read()
        except urllib.error.HTTPError as e:
            return e.code, e.read()

    def json_request(self, method: str, path: str, payload: Any = None, *, timeout: float = 30.0) -> tuple[int, Any]:
        body = json.dumps(payload).encode() if payload is not None else None
        headers = {"Content-Type": "application/json"} if body is not None else {}
        status, raw = self.request(method, path, body=body, headers=headers, timeout=timeout)
        try:
            data = json.loads(raw) if raw else None
        except ValueError:
            data = raw.decode("utf-8", "replace")
        return status, data


def port_in_use(port: int, host: str = "127.0.0.1") -> bool:
    """True if something accepts connections on the port or it cannot be bound."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(1.0)
        if sock.connect_ex((host, port)) == 0:
            return True
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        try:
            sock.bind((host, port))
        except OSError:
            return True
    return False


def wait_port_free(port: int, timeout_s: float) -> None:
    deadline = time.perf_counter() + timeout_s
    while port_in_use(port):
        if time.perf_counter() > deadline:
            raise BenchError(f"port {port} still busy {timeout_s:.0f} s after stopping the server")
        time.sleep(0.5)


def run_generations(
    client: HttpClient,
    bodies: list[dict],
    *,
    timeout_s: float,
    ensure_alive: Callable[[], None],
    on_posted: Callable[[], None] | None = None,
    poll_s: float = POLL_INTERVAL_S,
) -> list[dict]:
    """POST every body to /generate back to back, then poll /history/{id} until each one is terminal.

    ``wall_s`` of each generation runs from just before its POST until the poll that sees a terminal status.
    """
    results: list[dict] = []
    for body in bodies:
        res = {"gen_id": None, "status": None, "wall_s": None, "error": None, "duration": None}
        t0 = time.perf_counter()
        try:
            status, data = client.json_request("POST", "/generate", body, timeout=POST_TIMEOUT_S)
        except NET_ERRORS as e:
            ensure_alive()
            status, data = 0, str(e)
        if status == 200 and isinstance(data, dict) and data.get("id"):
            res["gen_id"] = str(data["id"])
        else:
            res.update(status="failed", error=f"POST /generate returned {status}: {data}")
        results.append(res)
        res["_t0"] = t0
    if on_posted is not None:
        on_posted()

    pending = [r for r in results if r["status"] is None]
    while pending:
        for res in list(pending):
            try:
                status, info = client.json_request("GET", f"/history/{res['gen_id']}", timeout=30)
            except NET_ERRORS:
                status, info = 0, None
            elapsed = time.perf_counter() - res["_t0"]
            if status == 200 and isinstance(info, dict) and info.get("status") in TERMINAL_STATUSES:
                res.update(
                    status=info["status"], wall_s=elapsed, error=info.get("error"), duration=info.get("duration")
                )
                pending.remove(res)
            elif elapsed > timeout_s:
                with contextlib.suppress(*NET_ERRORS):
                    client.request("POST", f"/generate/{res['gen_id']}/cancel", body=b"", timeout=10)
                res.update(status="timeout", error=f"no terminal status after {timeout_s:.0f} s")
                pending.remove(res)
        if pending:
            ensure_alive()
            time.sleep(poll_s)
    for res in results:
        res.pop("_t0", None)
    return results


# ---------------------------------------------------------------------------
# Server process
# ---------------------------------------------------------------------------


def server_pid_from_log(lines: Iterable[str]) -> int | None:
    """PID that uvicorn reports in ``Started server process [N]`` (the last one if there are several)."""
    pid = None
    for line in lines:
        match = _SERVER_PID_RE.search(line)
        if match:
            pid = int(match.group(1))
    return pid


def _kill_tree(proc: subprocess.Popen) -> None:
    if proc.poll() is not None:
        return
    if IS_WINDOWS:
        subprocess.run(["taskkill", "/PID", str(proc.pid), "/T", "/F"], capture_output=True, check=False)
    else:
        with contextlib.suppress(ProcessLookupError, PermissionError):
            os.killpg(proc.pid, signal.SIGKILL)
    try:
        proc.wait(timeout=15)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=15)


class ServerProcess:
    """One ``python -m backend.main`` process with stdout+stderr captured to a log file."""

    def __init__(self, *, cmd: list[str], env: dict[str, str], cwd: Path, log_path: Path, port: int):
        self.cmd = cmd
        self.env = env
        self.cwd = cwd
        self.log_path = log_path
        self.port = port
        self.client = HttpClient(f"http://127.0.0.1:{port}")
        self.proc: subprocess.Popen | None = None
        self.pid: int | None = None
        self.started_at = 0.0
        self._log_fh: Any = None

    def start(self) -> None:
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._log_fh = self.log_path.open("wb")
        kwargs: dict[str, Any] = {"creationflags": CREATE_NO_WINDOW} if IS_WINDOWS else {"start_new_session": True}
        self.started_at = time.perf_counter()
        self.proc = subprocess.Popen(
            self.cmd,
            cwd=str(self.cwd),
            env=self.env,
            stdin=subprocess.DEVNULL,
            stdout=self._log_fh,
            stderr=subprocess.STDOUT,
            **kwargs,
        )

    def alive(self) -> bool:
        return self.proc is not None and self.proc.poll() is None

    def ensure_alive(self) -> None:
        if not self.alive():
            code = self.proc.returncode if self.proc is not None else None
            raise BenchError(f"the server exited (code {code}); last log lines:\n{self.log_tail()}")

    def log_lines(self) -> list[str]:
        try:
            return self.log_path.read_bytes().decode("utf-8", "replace").splitlines()
        except OSError:
            return []

    def log_tail(self, n: int = LOG_TAIL_LINES) -> str:
        return "\n".join(self.log_lines()[-n:])

    def wait_healthy(self, timeout_s: float) -> float:
        """Wait for GET /health, record the server PID and return seconds since the process started.

        The PID is the one uvicorn logs (the real interpreter behind the venv launcher), else Popen's.
        """
        deadline = self.started_at + timeout_s
        while True:
            self.ensure_alive()
            with contextlib.suppress(*NET_ERRORS):
                status, data = self.client.json_request("GET", "/health", timeout=10)
                if status == 200 and isinstance(data, dict):
                    elapsed = time.perf_counter() - self.started_at
                    self.pid = server_pid_from_log(self.log_lines()) or (self.proc.pid if self.proc else None)
                    return elapsed
            if time.perf_counter() > deadline:
                raise BenchError(f"/health did not answer within {timeout_s:.0f} s; last log lines:\n{self.log_tail()}")
            time.sleep(0.5)

    def load_model(self, model_size: str, timeout_s: float) -> float:
        """Load the Qwen model (and any warm-up the app runs at load) via POST /models/load; return seconds."""
        t0 = time.perf_counter()
        path = f"/models/load?model_size={urllib.parse.quote(model_size)}"
        try:
            status, raw = self.client.request("POST", path, body=b"", timeout=timeout_s)
        except NET_ERRORS as e:
            self.ensure_alive()
            raise BenchError(f"model load failed: {e}") from e
        load_s = time.perf_counter() - t0
        if status != 200:
            raise BenchError(f"POST /models/load returned {status}: {raw[:500]!r}")
        status, health = self.client.json_request("GET", "/health", timeout=60)
        if status != 200 or not isinstance(health, dict) or not health.get("model_loaded"):
            raise BenchError(f"the model is not loaded after /models/load: {health}")
        return load_s

    def stop(self, port_timeout_s: float = PORT_RELEASE_TIMEOUT_S) -> None:
        """Ask for an orderly shutdown, kill the process tree if it lingers, and wait for the port."""
        if self.proc is None:
            return
        try:
            if self.proc.poll() is None:
                with contextlib.suppress(*NET_ERRORS):
                    self.client.request("POST", "/shutdown", body=b"", timeout=5)
                try:
                    self.proc.wait(timeout=SHUTDOWN_GRACE_S)
                except subprocess.TimeoutExpired:
                    logger.warning("server did not exit after /shutdown; killing its process tree")
                    _kill_tree(self.proc)
        finally:
            if self._log_fh is not None:
                self._log_fh.close()
                self._log_fh = None
        wait_port_free(self.port, port_timeout_s)


# ---------------------------------------------------------------------------
# Benchmark run
# ---------------------------------------------------------------------------


class BenchRun:
    """State of one benchmark invocation; ``run()`` returns the process exit code."""

    def __init__(self, args: argparse.Namespace):
        self.root = ROOT
        self.tag: str = args.tag
        self.out = Path(args.out).resolve()
        self.config: dict = _read_json(Path(args.config))
        self.bench_cfg: dict = self.config.get("bench") or {}
        phrase_file = _read_json(PHRASES_FILE)
        self.all_phrases: list[dict] = phrase_file.get("phrases") or []
        self.phrases = select_phrases(self.all_phrases, args.phrases)
        self.passes = parse_passes(args.passes)
        self.miopen_db: str = args.miopen_db
        self.scenario: str | None = args.scenario
        if args.concurrent is not None and args.concurrent < 1:
            raise BenchError("--concurrent must be >= 1")
        self.concurrent: int | None = args.concurrent
        self.extra_env = {str(k): str(v) for k, v in (self.bench_cfg.get("env") or {}).items()}
        self.extra_env.update(parse_env_args(args.env))

        self.port = int(self.bench_cfg.get("port", 17599))
        self.user_app_port = self.bench_cfg.get("user_app_port")
        self.profile_id = str(self.bench_cfg.get("profile_id") or "")
        self.language = str(self.bench_cfg.get("language") or phrase_file.get("language") or "es")
        self.engine = str(self.bench_cfg.get("engine") or "qwen")
        self.model_size = str(self.bench_cfg.get("model_size") or "1.7B")
        self.start_timeout_s = float(self.bench_cfg.get("server_start_timeout_s") or DEFAULT_START_TIMEOUT_S)
        self.phrase_timeout_s = float(self.bench_cfg.get("phrase_timeout_s") or DEFAULT_PHRASE_TIMEOUT_S)
        self.python = resolve_server_python(self.root, self.config)

        self.started_at = _now_iso()
        self.commit: str | None = None
        self.dirty = True
        self.data_dir: Path | None = None
        self.fresh_dir: Path | None = None
        self.server_env: dict[str, str] = {}
        self.env_payload: dict | None = None
        self.server_info: dict[str, dict] = {}
        self.pass_records: list[dict] = []
        self.scenarios: dict[str, dict | None] = {"contention": None, "batch": None}
        self.errors: list[str] = []
        self._active: ServerProcess | None = None
        # Pass records run on ``_active``; their VBPERF stages are attached once that server stops.
        self._served: list[dict] = []

    # -- top level -----------------------------------------------------------

    def run(self) -> int:
        self.commit, self.dirty = git_state(self.root)
        if self.dirty:
            logger.warning("the worktree has uncommitted changes outside perf-loop/; bench.json will say dirty=true")
        if port_in_use(self.port):
            raise BenchError(f"port {self.port} is busy (another server or a previous bench is still running)")
        if self.user_app_port and self._responds(int(self.user_app_port)):
            logger.warning("the user's app answers on port %s; it may compete for the GPU", self.user_app_port)
        if not self.profile_id:
            raise BenchError("config.bench.profile_id is required")

        self.data_dir = prepare_data_dir(self.root, self.bench_cfg)
        self.out.mkdir(parents=True, exist_ok=True)
        if self.miopen_db == "fresh":
            self.fresh_dir = self.out / "miopen-db"
            if self.fresh_dir.exists():
                shutil.rmtree(self.fresh_dir)
            self.fresh_dir.mkdir(parents=True)
        self.server_env = build_server_env(dict(os.environ), extra_env=self.extra_env, fresh_db_dir=self.fresh_dir)
        logger.info(
            "bench %s: %d phrase(s), passes %s, Find-Db %s, server %s",
            self.tag,
            len(self.phrases),
            ",".join(self.passes),
            self.miopen_db,
            " ".join(build_server_cmd(self.python, self.port, self.data_dir)),
        )
        return self._execute()

    def _execute(self) -> int:
        """Run the passes (a pass followed by ``hot`` keeps its server for it), then write bench.json."""
        try:
            for idx, name in enumerate(self.passes):
                is_last = idx == len(self.passes) - 1
                keep_alive = not is_last and self.passes[idx + 1] in REUSE_PASSES
                self._run_pass(name, scenario=self.scenario if is_last else None, keep_alive=keep_alive)
        except BenchError as e:
            self.errors.append(str(e))
            logger.error("%s", e)
        except NET_ERRORS as e:
            self.errors.append(f"HTTP error talking to the server: {e}")
            logger.error("HTTP error talking to the server: %s", e)
        finally:
            if self._active is not None:
                self._stop_server(self._active)
            self._write_bench_json()
        return 0 if self._all_ok() else 1

    def _all_ok(self) -> bool:
        if self.errors or len(self.pass_records) != len(self.passes):
            return False
        for pass_rec in self.pass_records:
            if len(pass_rec["phrases"]) != len(self.phrases):
                return False
            if any(rec.get("status") != "completed" or not rec.get("wav") for rec in pass_rec["phrases"]):
                return False
        return True

    def _responds(self, port: int) -> bool:
        try:
            HttpClient(f"http://127.0.0.1:{port}").request("GET", "/", timeout=3)
        except NET_ERRORS:
            return False
        return True

    # -- passes ----------------------------------------------------------------

    def _generate_body(self, phrase: dict) -> dict:
        return {
            "profile_id": self.profile_id,
            "text": phrase["text"],
            "language": self.language,
            "seed": phrase["seed"],
            "engine": self.engine,
            "model_size": self.model_size,
        }

    def _current_env_payload(self, server: ServerProcess) -> dict | None:
        env_payload, _ = collect_vbperf(parse_vbperf_lines(server.log_lines()))
        return env_payload

    def _new_server(self, name: str) -> ServerProcess:
        assert self.data_dir is not None
        return ServerProcess(
            cmd=build_server_cmd(self.python, self.port, self.data_dir),
            env=self.server_env,
            cwd=self.root,
            log_path=self.out / "logs" / f"server-{name}.log",
            port=self.port,
        )

    def _run_pass(self, name: str, *, scenario: str | None, keep_alive: bool = False) -> None:
        """Run one pass. ``hot`` reuses the live server of the previous pass; ``keep_alive`` leaves it running."""
        reuse = name in REUSE_PASSES
        server = self._active
        if reuse and (server is None or not self.pass_records):
            raise BenchError(f"the {name} pass needs the server of the previous pass, which is not running")
        prev_name = self.pass_records[-1]["name"] if reuse else None
        pass_dir = self.out / name
        if pass_dir.is_dir():
            for old in pass_dir.glob("*.wav"):
                old.unlink()
        pass_rec: dict[str, Any] = {
            "name": name,
            "miopen_db_path": None,
            "ufdb_lines_before": None,
            "ufdb_lines_after": None,
            "server_log": self.pass_records[-1]["server_log"] if reuse else f"logs/server-{name}.log",
            "phrases": [],
        }
        self.pass_records.append(pass_rec)
        info: dict[str, Any] = {"startup_s": None, "load_s": None, "pid": None, "same_process_as": prev_name}
        self.server_info[name] = info

        if reuse:
            assert server is not None
            logger.info("pass %s: reusing the %s server (pid %s), no restart or reload", name, prev_name, server.pid)
            server.ensure_alive()
        else:
            server = self._new_server(name)
            logger.info("pass %s: starting the server", name)
            self._active = server
            server.start()
        self._served.append(pass_rec)
        try:
            if not reuse:
                info["startup_s"] = _round(server.wait_healthy(self.start_timeout_s))
                if self.engine == "qwen":
                    info["load_s"] = _round(server.load_model(self.model_size, self.start_timeout_s))
                logger.info(
                    "pass %s: server ready in %.1f s, model loaded in %s s", name, info["startup_s"], info["load_s"]
                )
            info["pid"] = server.pid

            db_dir = resolve_miopen_db_dir(self._current_env_payload(server), self.server_env)
            pass_rec["miopen_db_path"] = str(db_dir)
            pass_rec["ufdb_lines_before"] = count_ufdb_lines(db_dir)

            for phrase in self.phrases:
                pass_rec["phrases"].append(self._run_phrase(server, phrase, name))

            end_dir = resolve_miopen_db_dir(self._current_env_payload(server), self.server_env)
            if end_dir != db_dir:
                logger.warning("pass %s: the Find-Db path changed during the pass (%s -> %s)", name, db_dir, end_dir)
                pass_rec["miopen_db_path"] = str(end_dir)
            pass_rec["ufdb_lines_after"] = count_ufdb_lines(end_dir)

            if scenario == "contention":
                self.scenarios["contention"] = self._scenario_contention(server)
            elif scenario == "batch":
                self.scenarios["batch"] = self._scenario_batch(server)
        finally:
            # On an error the server is stopped by _execute, which then finalizes this pass too.
            if not keep_alive:
                self._stop_server(server)

    def _stop_server(self, server: ServerProcess) -> None:
        """Stop the server and attach the VBPERF data of its log to every pass that ran on it."""
        try:
            server.stop()
        except BenchError as e:
            self.errors.append(str(e))
            logger.error("%s", e)
        finally:
            served: list[dict] = []
            if self._active is server:
                self._active = None
                served, self._served = self._served, []
            env_payload, stages = collect_vbperf(parse_vbperf_lines(server.log_lines()))
            if env_payload is not None:
                self.env_payload = env_payload
            for pass_rec in served:
                for rec in pass_rec["phrases"]:
                    finalize_phrase(rec, stages.get(rec["gen_id"]) if rec.get("gen_id") else None)

    def _run_phrase(self, server: ServerProcess, phrase: dict, pass_name: str) -> dict:
        rec = new_phrase_record(phrase)
        (res,) = run_generations(
            server.client,
            [self._generate_body(phrase)],
            timeout_s=self.phrase_timeout_s,
            ensure_alive=server.ensure_alive,
        )
        rec.update(gen_id=res["gen_id"], status=res["status"], wall_s=_round(res["wall_s"]), error=res["error"])
        if res["status"] == "completed":
            rel = f"{pass_name}/{phrase['id']}.wav"
            audio_s = self._download_wav(server, res["gen_id"], self.out / rel)
            if audio_s is None:
                audio_s = res.get("duration")
            rec.update(wav=rel, audio_s=_round(audio_s))
            logger.info("%s %s: %.2f s (audio %s s)", pass_name, phrase["id"], rec["wall_s"], rec["audio_s"])
        else:
            logger.error("%s %s: %s (%s)", pass_name, phrase["id"], res["status"], res["error"])
        return rec

    def _download_wav(self, server: ServerProcess, gen_id: str, dest: Path) -> float | None:
        try:
            status, raw = server.client.request("GET", f"/audio/{gen_id}", timeout=POST_TIMEOUT_S)
        except NET_ERRORS as e:
            server.ensure_alive()
            raise BenchError(f"downloading the audio of {gen_id} failed: {e}") from e
        if status != 200:
            raise BenchError(f"GET /audio/{gen_id} returned {status}: {raw[:300]!r}")
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(raw)
        return wav_duration_s(raw)

    # -- scenarios -------------------------------------------------------------

    def _reference_audio(self, client: HttpClient) -> tuple[str, bytes]:
        status, samples = client.json_request("GET", f"/profiles/{self.profile_id}/samples")
        if status != 200 or not isinstance(samples, list) or not samples:
            raise BenchError(f"cannot list the samples of profile {self.profile_id}: {status} {samples}")
        sample = samples[0]
        status, raw = client.request("GET", f"/samples/{sample['id']}", timeout=POST_TIMEOUT_S)
        if status != 200:
            raise BenchError(f"GET /samples/{sample['id']} returned {status}")
        suffix = Path(str(sample.get("audio_path") or "reference.wav")).suffix or ".wav"
        return f"reference{suffix}", raw

    def _scenario_contention(self, server: ServerProcess) -> dict:
        """Two generations of the longest selected phrase plus a /transcribe of the profile's reference audio.

        The transcription is sent right after both generations are queued, so it always meets a running one.
        """
        phrase = max(self.phrases, key=lambda p: len(p["text"]))
        result: dict[str, Any] = {
            "phrase": phrase["id"],
            "transcribe_wait_s": None,
            "transcribe_status": None,
            "generation_s_max": None,
            "generation_s": [],
            "error": None,
        }
        ref_name, ref_audio = self._reference_audio(server.client)
        body, ctype = build_multipart({"language": self.language}, {"file": (ref_name, ref_audio, "audio/wav")})
        transcribe: dict[str, Any] = {}

        def _transcribe() -> None:
            t0 = time.perf_counter()
            try:
                status, raw = server.client.request(
                    "POST", "/transcribe", body=body, headers={"Content-Type": ctype}, timeout=self.phrase_timeout_s
                )
                transcribe.update(status=status, wait_s=time.perf_counter() - t0, raw=raw)
            except NET_ERRORS as e:
                transcribe.update(status=None, error=str(e))

        worker = threading.Thread(target=_transcribe, name="bench-transcribe", daemon=True)
        logger.info("scenario contention: 2 x %s + /transcribe", phrase["id"])
        try:
            gens = run_generations(
                server.client,
                [self._generate_body(phrase), self._generate_body(phrase)],
                timeout_s=self.phrase_timeout_s,
                ensure_alive=server.ensure_alive,
                on_posted=worker.start,
            )
        finally:
            if worker.ident is not None:
                worker.join(timeout=self.phrase_timeout_s)

        walls = [g["wall_s"] for g in gens if g["status"] == "completed" and _is_number(g["wall_s"])]
        result["generation_s"] = [_round(g["wall_s"]) for g in gens]
        problems = [
            f"generation {g['gen_id']}: {g['status']} ({g['error']})" for g in gens if g["status"] != "completed"
        ]
        if len(walls) == len(gens):
            result["generation_s_max"] = _round(max(walls))
        result["transcribe_status"] = transcribe.get("status")
        if transcribe.get("status") == 200:
            result["transcribe_wait_s"] = _round(transcribe["wait_s"])
        else:
            detail = transcribe.get("error") or (transcribe.get("raw") or b"")[:300]
            problems.append(f"/transcribe: {transcribe.get('status')} {detail!r}")
        if problems:
            result["error"] = "; ".join(problems)
            self.errors.append(f"contention scenario: {result['error']}")
        logger.info(
            "scenario contention: transcribe %s s, generation max %s s",
            result["transcribe_wait_s"],
            result["generation_s_max"],
        )
        return result

    def _scenario_batch(self, server: ServerProcess) -> dict:
        """Queue N distinct phrases at once and compare against their individual times in the hot pass (else warm).

        Hot runs in an already warmed process with every length seen, so the batch does not win just by being
        compared against a freshly started server.
        """
        ref_pass = batch_reference_pass(self.pass_records)
        assert ref_pass is not None
        walls = {
            r["id"]: r["wall_s"]
            for r in ref_pass["phrases"]
            if r.get("status") == "completed" and _is_number(r.get("wall_s"))
        }
        n_requested = self.concurrent or DEFAULT_BATCH_N
        chosen = [p for p in self.phrases if p["id"] in walls][:n_requested]
        result: dict[str, Any] = {
            "n": len(chosen),
            "n_requested": n_requested,
            "phrases": [p["id"] for p in chosen],
            "ref_pass": ref_pass["name"],
            "total_s": None,
            "sum_individual_s": _round(sum(walls[p["id"]] for p in chosen)) if chosen else None,
            "throughput_gain": None,
            "wall_s": {},
            "error": None,
        }
        if not chosen:
            result["error"] = "no completed phrases to batch"
            self.errors.append(f"batch scenario: {result['error']}")
            return result
        if len(chosen) < n_requested:
            logger.warning("scenario batch: only %d phrase(s) available (asked for %d)", len(chosen), n_requested)

        batch_dir = self.out / "batch"
        if batch_dir.is_dir():
            for old in batch_dir.glob("*.wav"):
                old.unlink()
        logger.info("scenario batch: %d phrases at once", len(chosen))
        t0 = time.perf_counter()
        gens = run_generations(
            server.client,
            [self._generate_body(p) for p in chosen],
            timeout_s=self.phrase_timeout_s,
            ensure_alive=server.ensure_alive,
        )
        total = time.perf_counter() - t0
        problems = []
        for phrase, gen in zip(chosen, gens, strict=True):
            result["wall_s"][phrase["id"]] = _round(gen["wall_s"])
            if gen["status"] == "completed":
                self._download_wav(server, gen["gen_id"], self.out / "batch" / f"{phrase['id']}.wav")
            else:
                problems.append(f"{phrase['id']}: {gen['status']} ({gen['error']})")
        if problems:
            result["error"] = "; ".join(problems)
            self.errors.append(f"batch scenario: {result['error']}")
        else:
            result["total_s"] = _round(total)
            result["throughput_gain"] = _round(result["sum_individual_s"] / total) if total > 0 else None
        logger.info("scenario batch: total %s s, gain %s", result["total_s"], result["throughput_gain"])
        return result

    # -- output ------------------------------------------------------------------

    def _write_bench_json(self) -> None:
        if self.commit is not None:
            try:
                end_commit, end_dirty = git_state(self.root)
            except BenchError as e:
                logger.warning("%s", e)
            else:
                if end_commit != self.commit:
                    self.errors.append(f"HEAD moved during the run ({self.commit[:10]} -> {end_commit[:10]})")
                self.dirty = self.dirty or end_dirty
        ufdb_total = None
        if self.pass_records:
            last_dir = self.pass_records[-1].get("miopen_db_path")
            if last_dir:
                ufdb_total = count_ufdb_lines(Path(last_dir))
        code_path = (self.env_payload or {}).get("code_path") or str(self.root / "backend")
        doc = assemble_bench_json(
            tag=self.tag,
            commit=self.commit,
            dirty=self.dirty,
            code_path=str(code_path),
            started_at=self.started_at,
            args={
                "phrases": [p["id"] for p in self.phrases],
                "passes": self.passes,
                "miopen_db": self.miopen_db,
                "env": self.extra_env,
                "scenario": self.scenario,
                "concurrent": self.concurrent,
            },
            env_checks=self.env_payload,
            server=self.server_info,
            passes=self.pass_records,
            scenarios=self.scenarios,
            summary=compute_summary(self.pass_records, self.server_info, len(self.phrases), ufdb_total),
            errors=self.errors,
        )
        _write_json_atomic(self.out / "bench.json", doc)
        logger.info("wrote %s", self.out / "bench.json")


# ---------------------------------------------------------------------------
# Self-test (no GPU, no server)
# ---------------------------------------------------------------------------


class _FakeClient:
    """Stand-in for HttpClient: /generate returns ids, /history walks a scripted list of statuses."""

    def __init__(self, statuses: list[str], fail_post: bool = False):
        self.statuses = statuses
        self.fail_post = fail_post
        self.polls: dict[str, int] = {}
        self.calls: list[tuple[str, str]] = []

    def json_request(self, method: str, path: str, payload: Any = None, *, timeout: float = 30.0) -> tuple[int, Any]:
        self.calls.append((method, path))
        if method == "POST" and path == "/generate":
            if self.fail_post:
                return 404, {"detail": "Profile not found"}
            gen_id = f"gen-{len(self.polls)}"
            self.polls[gen_id] = 0
            return 200, {"id": gen_id, "status": "generating"}
        if method == "GET" and path.startswith("/history/"):
            gen_id = path.rsplit("/", 1)[1]
            idx = min(self.polls[gen_id], len(self.statuses) - 1)
            self.polls[gen_id] += 1
            status = self.statuses[idx]
            return 200, {
                "id": gen_id,
                "status": status,
                "duration": 3.0,
                "error": "boom" if status == "failed" else None,
            }
        return 404, None

    def request(self, method: str, path: str, **_: Any) -> tuple[int, bytes]:
        self.calls.append((method, path))
        if method == "GET" and path.startswith("/audio/"):
            return 200, _wav_bytes(2.0)
        return 200, b"{}"

    def generate_posts(self) -> int:
        return self.calls.count(("POST", "/generate"))


class _FakeServer:
    """Stand-in for ServerProcess: no process; a _FakeClient and a log with one VBPERF talker line per generation.

    The talker of the i-th generation (0-based) takes ``1000 * (i + 1)`` ms, so passes on one server differ.
    """

    def __init__(self, name: str, pid: int, db_dir: Path):
        self.name = name
        self.pid: int | None = None
        self.client = _FakeClient(["completed"])
        self.starts = 0
        self.stops = 0
        self.posts_at_stop: int | None = None
        self._pid = pid
        self._db_dir = db_dir

    def start(self) -> None:
        self.starts += 1

    def wait_healthy(self, timeout_s: float) -> float:
        self.pid = server_pid_from_log(self.log_lines())
        return 0.5

    def load_model(self, model_size: str, timeout_s: float) -> float:
        return 1.5

    def ensure_alive(self) -> None:
        if self.stops:
            raise BenchError(f"fake server {self.name} is stopped")

    def log_lines(self) -> list[str]:
        env = {"stage": "env", "miopen_user_db_path": str(self._db_dir), "code_path": "E:/wt/backend"}
        lines = [f"INFO:     Started server process [{self._pid}]", VBPERF_PREFIX + json.dumps(env)]
        for idx, gen_id in enumerate(self.client.polls):
            ms = 1000.0 * (idx + 1)
            event = {"gen_id": gen_id, "stage": "talker", "ms": ms, "steps": 10, "steps_per_s": ms / 100}
            lines.append(VBPERF_PREFIX + json.dumps(event))
        return lines

    def stop(self) -> None:
        self.stops += 1
        self.posts_at_stop = self.client.generate_posts()


class _FakeBenchRun(BenchRun):
    """BenchRun whose servers are _FakeServer objects with PIDs 101, 102, ..."""

    def __init__(self, args: argparse.Namespace, db_dir: Path):
        super().__init__(args)
        self.db_dir = db_dir
        self.fake_servers: list[_FakeServer] = []

    def _new_server(self, name: str) -> Any:
        server = _FakeServer(name, 101 + len(self.fake_servers), self.db_dir)
        self.fake_servers.append(server)
        return server


def run_with_fake_servers(tmp: Path, passes: str, *extra_args: str) -> tuple[int, dict, list[_FakeServer]]:
    """Run the pass lifecycle on p01,p02 against fake servers under ``tmp``; return (exit code, bench.json, servers)."""
    config = tmp / "config.json"
    config.write_text(json.dumps({"bench": {"profile_id": "prof-1", "port": 1}}), encoding="utf-8")
    argv = ["--config", str(config), "--tag", "fake", "--out", str(tmp / "out"), "--phrases", "p01,p02"]
    run = _FakeBenchRun(build_arg_parser().parse_args([*argv, "--passes", passes, *extra_args]), tmp / "miopen-db")
    run.data_dir = tmp / "data"
    code = run._execute()
    return code, _read_json(run.out / "bench.json"), run.fake_servers


def _expect(failures: list[str], label: str, got: Any, want: Any) -> None:
    if got != want:
        failures.append(f"{label}: got {got!r}, want {want!r}")


def _wav_bytes(seconds: float, rate: int = 24000, channels: int = 1, width: int = 2) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(channels)
        w.setsampwidth(width)
        w.setframerate(rate)
        w.writeframes(b"\x00" * int(seconds * rate) * channels * width)
    return buf.getvalue()


def _selftest_vbperf(failures: list[str]) -> None:
    lines = [
        "INFO:     Started server process [1234]",
        '\x1b[32mINFO\x1b[0m:     VBPERF {"gen_id": "g1", "stage": "talker", "ms": 9100.0, "steps": 39, '
        '"steps_per_s": 4.29}',
        '2026-10-07 10:00:00,123 INFO backend: VBPERF {"gen_id": "g1", "stage": "codec", "ms": 1500.0, "T": 369, '
        '"chunks": 2}\r',
        'VBPERF {"gen_id": "g1", "stage": "reference", "ms": 1830.0}',
        "VBPERF {not json}",
        "VBPERF [1, 2, 3]",
        'VBPERF {"stage": "env", "torch": "2.9.1", "miopen_user_db_path": "C:/db", "code_path": "E:/wt/backend"}',
        'VBPERF {"gen_id": "g2", "stage": "talker", "ms": 1000.0, "steps": 10, "steps_per_s": 10.0}',
        'VBPERF {"gen_id": "g2", "stage": "talker", "ms": 1000.0, "steps": 30, "steps_per_s": 30.0}',
        'VBPERF {"stage": "talker", "ms": 1.0}',
    ]
    events = parse_vbperf_lines(lines)
    _expect(failures, "vbperf: valid events", len(events), 7)
    env, stages = collect_vbperf(events)
    _expect(failures, "vbperf: env path", (env or {}).get("miopen_user_db_path"), "C:/db")
    _expect(failures, "vbperf: env code_path", (env or {}).get("code_path"), "E:/wt/backend")
    _expect(failures, "vbperf: gens", sorted(stages), ["g1", "g2"])
    _expect(failures, "vbperf: g1 stages", sorted(stages.get("g1", {})), ["codec", "reference", "talker"])
    _expect(failures, "vbperf: g1 talker", stages["g1"]["talker"], {"ms": 9100.0, "steps": 39, "steps_per_s": 4.29})
    _expect(failures, "vbperf: g1 codec T", stages["g1"]["codec"].get("T"), 369)
    _expect(failures, "vbperf: merged g2", stages["g2"]["talker"], {"ms": 2000.0, "steps": 40, "steps_per_s": 20.0})


def _selftest_phrase_and_summary(failures: list[str]) -> None:
    with_stages = new_phrase_record({"id": "p01", "seed": 1101, "text": "x" * 44})
    with_stages.update(gen_id="g1", status="completed", wall_s=151.3, audio_s=3.1, wav="cold/p01.wav")
    finalize_phrase(with_stages, {"talker": {"ms": 9100.0, "steps": 39, "steps_per_s": 4.3}})
    _expect(failures, "phrase: text_chars", with_stages["text_chars"], 44)
    _expect(failures, "phrase: steps from talker", with_stages["steps"], 39)
    _expect(failures, "phrase: rtf", with_stages["rtf"], round(151.3 / 3.1, 3))

    no_stages = new_phrase_record({"id": "p02", "seed": 1102, "text": "y" * 50})
    no_stages.update(wall_s=10.0, audio_s=6.0)
    finalize_phrase(no_stages, None)
    _expect(failures, "phrase: steps from audio", no_stages["steps"], 75)
    _expect(failures, "phrase: empty stages", no_stages["stages"], {})

    failed = new_phrase_record({"id": "p03", "seed": 1, "text": "z"})
    finalize_phrase(failed, None)
    _expect(failures, "phrase: failed steps", (failed["steps"], failed["rtf"]), (None, None))

    def _ph(pid: str, wall: float, audio: float, stages: dict | None = None) -> dict:
        rec = new_phrase_record({"id": pid, "seed": 1, "text": pid})
        rec.update(status="completed", wall_s=wall, audio_s=audio, wav=f"x/{pid}.wav")
        finalize_phrase(rec, stages)
        return rec

    cold = {
        "name": "cold",
        "ufdb_lines_before": 10,
        "ufdb_lines_after": 47,
        "phrases": [_ph("p01", 150.0, 3.0), _ph("p02", 130.0, 4.0)],
    }
    warm = {
        "name": "warm",
        "ufdb_lines_before": 47,
        "ufdb_lines_after": 47,
        "phrases": [
            _ph("p01", 15.0, 3.0, {"talker": {"ms": 9000.0, "steps": 40, "steps_per_s": 4.0}, "codec": {"ms": 1000.0}}),
            _ph(
                "p02", 17.0, 4.0, {"talker": {"ms": 11000.0, "steps": 50, "steps_per_s": 5.0}, "codec": {"ms": 2000.0}}
            ),
            _ph("p03", 19.0, 5.0),
        ],
    }
    hot = {
        "name": "hot",
        "ufdb_lines_before": 47,
        "ufdb_lines_after": 48,
        "phrases": [
            _ph("p01", 6.0, 3.0, {"talker": {"ms": 4000.0, "steps": 40, "steps_per_s": 10.0}, "codec": {"ms": 500.0}}),
            _ph("p02", 8.0, 4.0, {"talker": {"ms": 5000.0, "steps": 60, "steps_per_s": 12.0}}),
            _ph("p03", 13.0, 5.0),
        ],
    }
    server = {
        "cold": {"load_s": 41.2, "startup_s": 9.0, "pid": 11, "same_process_as": None},
        "warm": {"load_s": 12.0, "startup_s": 8.0, "pid": 22, "same_process_as": None},
        "hot": {"load_s": None, "startup_s": None, "pid": 22, "same_process_as": "warm"},
    }
    summary = compute_summary([cold, warm, hot], server, 3, 48)
    _expect(failures, "summary: warm mean", summary["warm_mean_s"], 17.0)
    _expect(failures, "summary: warm median", summary["warm_median_s"], 17.0)
    _expect(failures, "summary: cold mean", summary["cold_mean_s"], 140.0)
    _expect(failures, "summary: cold median", summary["cold_median_s"], 140.0)
    _expect(failures, "summary: rtf warm", summary["rtf_warm_mean"], round((5.0 + 4.25 + 3.8) / 3, 3))
    _expect(failures, "summary: hot mean", summary["hot_mean_s"], 9.0)
    _expect(failures, "summary: hot median", summary["hot_median_s"], 8.0)
    _expect(failures, "summary: rtf hot", summary["rtf_hot_mean"], round((2.0 + 2.0 + 2.6) / 3, 3))
    _expect(failures, "summary: talker sps (warm)", summary["talker_steps_per_s_mean"], 4.5)
    _expect(failures, "summary: talker sps hot", summary["talker_steps_per_s_mean_hot"], 11.0)
    _expect(failures, "summary: ufdb cold", summary["ufdb_lines_added_cold"], 37)
    _expect(failures, "summary: ufdb warm", summary["ufdb_lines_added_warm"], 0)
    _expect(failures, "summary: ufdb hot", summary["ufdb_lines_added_hot"], 1)
    _expect(failures, "summary: ufdb total", summary["ufdb_lines_total"], 48)
    _expect(failures, "summary: stage means (warm)", summary["stage_ms_mean"], {"talker": 10000.0, "codec": 1500.0})
    _expect(failures, "summary: stage means hot", summary["stage_ms_mean_hot"], {"talker": 4500.0, "codec": 500.0})
    _expect(failures, "summary: load_s", summary["load_s"], 41.2)
    _expect(failures, "summary: phrases_n", summary["phrases_n"], 3)
    _expect(failures, "batch ref: hot first", (batch_reference_pass([cold, warm, hot]) or {}).get("name"), "hot")
    _expect(failures, "batch ref: else warm", (batch_reference_pass([warm, cold]) or {}).get("name"), "warm")
    _expect(failures, "batch ref: else last", (batch_reference_pass([cold]) or {}).get("name"), "cold")

    only_cold = compute_summary([cold], {"cold": {"load_s": 5.0}}, 2, None)
    _expect(failures, "summary: no warm -> null", (only_cold["warm_mean_s"], only_cold["rtf_warm_mean"]), (None, None))
    _expect(failures, "summary: no warm stages", only_cold["stage_ms_mean"], {})
    _expect(failures, "summary: no warm ufdb", only_cold["ufdb_lines_added_warm"], None)
    hot_fields = (
        "hot_mean_s",
        "hot_median_s",
        "rtf_hot_mean",
        "ufdb_lines_added_hot",
        "stage_ms_mean_hot",
        "talker_steps_per_s_mean_hot",
    )
    _expect(failures, "summary: no hot -> null", [only_cold[k] for k in hot_fields], [None] * len(hot_fields))
    _expect(failures, "summary: empty", compute_summary([], {}, 0, None)["load_s"], None)

    doc = assemble_bench_json(
        tag="t",
        commit="abc",
        dirty=False,
        code_path="E:/wt/backend",
        started_at=_now_iso(),
        args={},
        env_checks=None,
        server=server,
        passes=[cold, warm, hot],
        scenarios={"contention": None, "batch": None},
        summary=summary,
        errors=[],
    )
    for key in ("schema", "tag", "commit", "dirty", "code_path", "passes", "summary", "env_checks", "scenarios"):
        if key not in doc:
            failures.append(f"bench.json: missing {key}")
    _expect(failures, "bench.json: schema", doc["schema"], SCHEMA)
    for pass_rec in doc["passes"]:
        for rec in pass_rec["phrases"]:
            missing = [k for k in ("id", "wall_s", "audio_s", "steps", "wav", "stages") if k not in rec]
            _expect(failures, f"bench.json: {pass_rec['name']}/{rec['id']} keys", missing, [])
    json.dumps(doc)


def _selftest_cli_and_commands(failures: list[str]) -> None:
    phrases = [{"id": f"p{i:02d}", "text": "t" * i, "seed": i} for i in range(1, 5)]
    _expect(failures, "phrases: all", [p["id"] for p in select_phrases(phrases, "all")], ["p01", "p02", "p03", "p04"])
    _expect(failures, "phrases: file order", [p["id"] for p in select_phrases(phrases, "p03, p01")], ["p01", "p03"])
    for bad_spec in ("p09", ","):
        try:
            select_phrases(phrases, bad_spec)
        except BenchError:
            pass
        else:
            failures.append(f"phrases: {bad_spec!r} should fail")
    _expect(failures, "passes: default", parse_passes("cold,warm"), ["cold", "warm"])
    _expect(failures, "passes: warm only", parse_passes("warm"), ["warm"])
    _expect(failures, "passes: with hot", parse_passes("cold, warm, hot"), ["cold", "warm", "hot"])
    _expect(failures, "passes: hot after cold", parse_passes("cold,hot,warm"), ["cold", "hot", "warm"])
    for bad_spec in ("hot", "hot,warm", "warm,hot,hot", "tepid", "warm,warm", ""):
        try:
            parse_passes(bad_spec)
        except BenchError:
            pass
        else:
            failures.append(f"passes: {bad_spec!r} should fail")
    _expect(failures, "env args", parse_env_args(["A=1", "B=x=y", "C="]), {"A": "1", "B": "x=y", "C": ""})
    try:
        parse_env_args(["NOVALUE"])
    except BenchError:
        pass
    else:
        failures.append("env args: missing '=' should fail")

    data_dir = Path("E:/wt/perf-loop/bench/_data")
    _expect(
        failures,
        "server cmd",
        build_server_cmd("py.exe", 17599, data_dir),
        ["py.exe", "-m", "backend.main", "--port", "17599", "--data-dir", str(data_dir)],
    )
    base = {"PATH": "x", "MIOPEN_USER_DB_PATH": "C:/user-db", "PYTHONUNBUFFERED": "0"}
    persist = build_server_env(base, extra_env={"VOICEBOX_PERF_LOG": "1"}, fresh_db_dir=None)
    _expect(failures, "env persist: keeps user db", persist.get("MIOPEN_USER_DB_PATH"), "C:/user-db")
    _expect(failures, "env persist: no cache dir", persist.get("MIOPEN_CUSTOM_CACHE_DIR"), None)
    _expect(failures, "env persist: perf log", persist.get("VOICEBOX_PERF_LOG"), "1")
    _expect(failures, "env persist: setdefault", persist.get("PYTHONUNBUFFERED"), "0")
    _expect(failures, "env: base untouched", "VOICEBOX_PERF_LOG" in base, False)
    fresh_dir = Path("E:/wt/perf-loop/bench/t/miopen-db")
    fresh = build_server_env(base, extra_env={"MIOPEN_FIND_MODE": "FAST"}, fresh_db_dir=fresh_dir)
    _expect(failures, "env fresh: user db", fresh.get("MIOPEN_USER_DB_PATH"), str(fresh_dir))
    _expect(failures, "env fresh: cache dir", fresh.get("MIOPEN_CUSTOM_CACHE_DIR"), str(fresh_dir))
    _expect(failures, "env fresh: extra", fresh.get("MIOPEN_FIND_MODE"), "FAST")
    override = build_server_env({}, extra_env={"MIOPEN_USER_DB_PATH": "D:/x"}, fresh_db_dir=fresh_dir)
    _expect(failures, "env: --env wins", override.get("MIOPEN_USER_DB_PATH"), "D:/x")
    _expect(failures, "dirty cmd", DIRTY_CMD[-2:], (".", ":(exclude)perf-loop"))

    _expect(failures, "miopen: from env line", resolve_miopen_db_dir({"miopen_user_db_path": "A"}, {}), Path("A"))
    _expect(failures, "miopen: from server env", resolve_miopen_db_dir(None, {"MIOPEN_USER_DB_PATH": "B"}), Path("B"))
    _expect(failures, "miopen: default", resolve_miopen_db_dir({"torch": "x"}, {}), default_miopen_db_dir())

    log = ["INFO:     Started server process [7872]", "INFO:     Waiting for application startup."]
    _expect(failures, "server pid: from uvicorn", server_pid_from_log(log), 7872)
    _expect(failures, "server pid: missing", server_pid_from_log(["INFO:     Uvicorn running on ..."]), None)

    body, ctype = build_multipart({"language": "es"}, {"file": ("reference.wav", b"RIFFdata", "audio/wav")})
    boundary = ctype.split("boundary=", 1)[1]
    _expect(failures, "multipart: ends", body.endswith(f"--{boundary}--\r\n".encode()), True)
    _expect(failures, "multipart: field", b'name="language"\r\n\r\nes\r\n' in body, True)
    _expect(failures, "multipart: file", b'filename="reference.wav"' in body and b"RIFFdata" in body, True)


def _selftest_files(failures: list[str]) -> None:
    _expect(failures, "wav: mono 16-bit", wav_duration_s(_wav_bytes(3.0)), 3.0)
    _expect(failures, "wav: stereo 24 kHz", wav_duration_s(_wav_bytes(1.5, channels=2)), 1.5)
    _expect(failures, "wav: garbage", wav_duration_s(b"not a wav file at all"), None)

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        db_dir = tmp_path / "db"
        _expect(failures, "ufdb: missing dir", count_ufdb_lines(db_dir), 0)
        (db_dir / "sub").mkdir(parents=True)
        (db_dir / "gfx1101_30.HIP.x.ufdb.txt").write_text("a=1\nb=2\n\n", encoding="utf-8")
        (db_dir / "sub" / "other.ufdb.txt").write_text("c=3\n", encoding="utf-8")
        (db_dir / "gfx1101_30.HIP.x.ufdb.txt.time").write_text("123\n", encoding="utf-8")
        (db_dir / "gfx1101_30.udb.txt").write_text("ignored\n", encoding="utf-8")
        _expect(failures, "ufdb: count", count_ufdb_lines(db_dir), 3)

        _selftest_data_dir(failures, tmp_path)


def _selftest_data_dir(failures: list[str], tmp_path: Path) -> None:
    profile_id = "prof-1"
    worktree = tmp_path / "wt"
    source = tmp_path / "src" / "data"
    sample = source / "profiles" / profile_id / "s1.wav"
    sample.parent.mkdir(parents=True)
    sample.write_bytes(_wav_bytes(0.5))
    worktree.mkdir()
    with contextlib.closing(sqlite3.connect(source / "voicebox.db")) as conn:
        conn.execute("CREATE TABLE profiles (id TEXT PRIMARY KEY, name TEXT)")
        conn.execute("CREATE TABLE profile_samples (id TEXT, profile_id TEXT, audio_path TEXT)")
        conn.execute("INSERT INTO profiles VALUES (?, ?)", (profile_id, "Voz"))
        conn.execute("INSERT INTO profile_samples VALUES ('s1', ?, ?)", (profile_id, str(sample)))
        conn.commit()

    def _digest(folder: Path) -> str:
        h = hashlib.sha256()
        for path in sorted(folder.rglob("*")):
            h.update(str(path.relative_to(folder)).encode())
            if path.is_file():
                h.update(path.read_bytes())
        return h.hexdigest()

    before = _digest(source)
    cfg = {"data_dir": "perf-loop/bench/_data", "source_data_dir": str(source), "profile_id": profile_id}
    data_dir = prepare_data_dir(worktree, cfg)
    _expect(failures, "data dir: location", data_dir, (worktree / "perf-loop" / "bench" / "_data").resolve())
    _expect(failures, "data dir: db copied", (data_dir / "voicebox.db").is_file(), True)
    _expect(failures, "data dir: sample copied", (data_dir / "profiles" / profile_id / "s1.wav").is_file(), True)
    marker = data_dir / "cache" / "keep.prompt"
    marker.parent.mkdir(parents=True)
    marker.write_bytes(b"cache")
    prepare_data_dir(worktree, cfg)
    _expect(failures, "data dir: cache persists", marker.is_file(), True)
    _expect(failures, "data dir: source untouched", _digest(source), before)
    _expect(
        failures,
        "rebase: absolute",
        rebase_storage_path(str(source / "profiles" / "a.wav"), data_dir),
        data_dir / "profiles" / "a.wav",
    )
    _expect(
        failures,
        "rebase: relative",
        rebase_storage_path("data/profiles/a.wav", data_dir),
        data_dir / "profiles" / "a.wav",
    )
    bad_cfgs = (
        {**cfg, "data_dir": str(tmp_path / "outside")},
        {**cfg, "data_dir": str(source)},
        {**cfg, "source_data_dir": str(worktree)},
    )
    for bad_cfg in bad_cfgs:
        try:
            prepare_data_dir(worktree, bad_cfg)
        except BenchError:
            pass
        else:
            failures.append(f"data dir: {bad_cfg['data_dir']} should be rejected")
    try:
        prepare_data_dir(worktree, {**cfg, "profile_id": "missing", "data_dir": "other"})
    except BenchError:
        pass
    else:
        failures.append("data dir: an unknown profile should be rejected")


def _selftest_generations(failures: list[str]) -> None:
    def _alive() -> None:
        return None

    client: Any = _FakeClient(["generating", "generating", "completed"])
    (res,) = run_generations(client, [{"text": "a"}], timeout_s=60, ensure_alive=_alive, poll_s=0.0)
    _expect(
        failures, "generate: completed", (res["status"], res["gen_id"], res["duration"]), ("completed", "gen-0", 3.0)
    )
    _expect(failures, "generate: polled until done", client.polls.get("gen-0"), 3)
    _expect(failures, "generate: wall_s", _is_number(res["wall_s"]) and res["wall_s"] >= 0, True)

    two: Any = _FakeClient(["generating", "failed"])
    posted: list[int] = []
    gens = run_generations(
        two,
        [{"text": "a"}, {"text": "b"}],
        timeout_s=60,
        ensure_alive=_alive,
        on_posted=lambda: posted.append(len(two.polls)),
        poll_s=0.0,
    )
    _expect(failures, "generate: on_posted after all POSTs", posted, [2])
    _expect(failures, "generate: failed", [(g["status"], g["error"]) for g in gens], [("failed", "boom")] * 2)

    client = _FakeClient(["completed"], fail_post=True)
    (res,) = run_generations(client, [{"text": "a"}], timeout_s=60, ensure_alive=_alive, poll_s=0.0)
    _expect(failures, "generate: POST error", (res["status"], res["gen_id"]), ("failed", None))

    client = _FakeClient(["generating"])
    (res,) = run_generations(client, [{"text": "a"}], timeout_s=-1.0, ensure_alive=_alive, poll_s=0.0)
    _expect(failures, "generate: timeout", res["status"], "timeout")
    _expect(failures, "generate: timeout cancels", ("POST", "/generate/gen-0/cancel") in client.calls, True)

    def _dead() -> None:
        raise BenchError("server died")

    client = _FakeClient(["generating"])
    try:
        run_generations(client, [{"text": "a"}], timeout_s=60, ensure_alive=_dead, poll_s=0.0)
    except BenchError:
        pass
    else:
        failures.append("generate: a dead server should abort the wait")


@contextlib.contextmanager
def _quiet_bench_logger() -> Any:
    """Hide the INFO lines of a fake bench run so the self-test output stays readable."""
    level = logger.level
    logger.setLevel(logging.WARNING)
    try:
        yield
    finally:
        logger.setLevel(level)


def _selftest_hot_lifecycle(failures: list[str]) -> None:
    with tempfile.TemporaryDirectory() as tmp, _quiet_bench_logger():
        tmp_path = Path(tmp)
        code, doc, servers = run_with_fake_servers(
            tmp_path, "cold,warm,hot", "--scenario", "batch", "--concurrent", "2"
        )
        _expect(failures, "hot: exit code", (code, doc["errors"]), (0, []))
        started = [(s.name, s.starts, s.stops) for s in servers]
        _expect(failures, "hot: one process per non-hot pass", started, [("cold", 1, 1), ("warm", 1, 1)])
        _expect(failures, "hot: warm kept alive for hot + batch", [s.posts_at_stop for s in servers], [2, 6])
        server = doc["server"]
        pids = [(server[p]["pid"], server[p]["same_process_as"]) for p in ("cold", "warm", "hot")]
        _expect(failures, "hot: pid and same_process_as", pids, [(101, None), (102, None), (102, "warm")])
        _expect(failures, "hot: no reload", (server["hot"]["load_s"], server["warm"]["load_s"]), (None, 1.5))
        by_name = {p["name"]: p for p in doc["passes"]}
        _expect(failures, "hot: pass order", list(by_name), ["cold", "warm", "hot"])
        _expect(failures, "hot: shares the warm log", by_name["hot"]["server_log"], "logs/server-warm.log")
        hot_wavs = [r["wav"] for r in by_name["hot"]["phrases"]]
        _expect(failures, "hot: wavs", hot_wavs, ["hot/p01.wav", "hot/p02.wav"])
        _expect(failures, "hot: wav files", all((tmp_path / "out" / w).is_file() for w in hot_wavs), True)
        summary = doc["summary"]
        _expect(failures, "hot: hot_mean_s", _is_number(summary["hot_mean_s"]), True)
        # Warm server log: gen-0/1 are warm, gen-2/3 hot, gen-4/5 the batch.
        _expect(failures, "hot: stage_ms_mean on warm", summary["stage_ms_mean"], {"talker": 1500.0})
        _expect(failures, "hot: stage_ms_mean_hot", summary["stage_ms_mean_hot"], {"talker": 3500.0})
        _expect(failures, "hot: talker sps hot", summary["talker_steps_per_s_mean_hot"], 35.0)
        batch = doc["scenarios"]["batch"] or {}
        hot_sum = _round(sum(r["wall_s"] for r in by_name["hot"]["phrases"]))
        _expect(
            failures, "hot: batch ref_pass", (batch.get("ref_pass"), batch.get("sum_individual_s")), ("hot", hot_sum)
        )


SELF_TESTS: tuple[Callable[[list[str]], None], ...] = (
    _selftest_vbperf,
    _selftest_phrase_and_summary,
    _selftest_cli_and_commands,
    _selftest_files,
    _selftest_generations,
    _selftest_hot_lifecycle,
)


def run_self_test() -> int:
    """Run every self-check; return 0 if all pass."""
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


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Voicebox Qwen3-TTS benchmark (perf-loop/contracts.md, section 2)")
    parser.add_argument("--config", help="perf-loop/config.json")
    parser.add_argument("--tag", help="label written to bench.json")
    parser.add_argument("--out", help="output directory (perf-loop/bench/<tag>)")
    parser.add_argument("--phrases", default="all", help="all or a comma-separated list of ids (default: all)")
    parser.add_argument(
        "--passes",
        default="cold,warm",
        help="passes in execution order among cold, warm and hot; hot reuses the previous pass's server "
        "(default: cold,warm)",
    )
    parser.add_argument("--miopen-db", choices=("fresh", "persist"), default="persist")
    parser.add_argument("--env", action="append", default=[], metavar="KEY=VALUE", help="extra server env (repeatable)")
    parser.add_argument("--scenario", choices=VALID_SCENARIOS, default=None)
    parser.add_argument("--concurrent", type=int, default=None, help=f"batch size (default {DEFAULT_BATCH_N})")
    parser.add_argument("--self-test", action="store_true", help="check parsing and summaries without GPU or server")
    return parser


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [bench] %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stderr,
    )
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    if args.self_test:
        return run_self_test()
    missing = [opt for opt in ("config", "tag", "out") if not getattr(args, opt)]
    if missing:
        parser.error("missing " + ", ".join(f"--{m}" for m in missing))
    try:
        return BenchRun(args).run()
    except BenchError as e:
        logger.error("%s", e)
        return 1
    except KeyboardInterrupt:
        logger.error("interrupted")
        return 130


if __name__ == "__main__":
    sys.exit(main())
