#!/usr/bin/env python3
"""loopctl.py: la parte determinista del loop /tts-loop (plan de velocidad de Qwen3-TTS).

El agente decide y escribe código. loopctl hace todo lo que no debe depender del
criterio de un modelo: estado del backlog, ramas, banco, comparación de audio,
puertas de calidad, decisiones humanas y guardarraíles (hooks de Claude Code).

Solo biblioteca estándar, Python >= 3.10. Pensado para Windows (Git Bash o
PowerShell); también funciona en Linux/macOS.

Uso rápido (desde la raíz del worktree):
    python .claude/skills/tts-loop/loopctl.py status
    python .claude/skills/tts-loop/loopctl.py next
    python .claude/skills/tts-loop/loopctl.py --help
"""
from __future__ import annotations

import argparse
import copy
import datetime as dt
import hashlib
import html
import json
import os
import re
import secrets
import shutil
import socket
import sqlite3
import stat
import subprocess
import sys
import tempfile
import time
import traceback
import xml.etree.ElementTree as ET
from pathlib import Path

VERSION = "1.1"
IS_WIN = os.name == "nt"
SKILL_DIR = Path(__file__).resolve().parent
LOOP = "perf-loop"
STATE_REL = f"{LOOP}/backlog.json"
CONFIG_REL = f"{LOOP}/config.json"
TOKEN_REL = f"{LOOP}/.human-token.json"
MARKER_REL = f"{LOOP}/.iteration"
STOP_REL = f"{LOOP}/STOP"  # si existe, `next` se detiene antes de la siguiente iteración
TESTS_BASELINE_REL = f"{LOOP}/tests-baseline.json"
# Códigos de salida de pytest con resultados completos: 0 (todo pasa) y 1 (algún test falla).
# 2 = interrumpido (p. ej. un error al recoger un módulo), 3 = error interno, 4 = uso, 5 = ningún test.
PYTEST_COMPLETE_RC = (0, 1)
PYTEST_RC_WHY = {
    2: "pytest se interrumpió (casi siempre, un error al importar o recoger un módulo de test)",
    3: "error interno de pytest",
    4: "error de uso de pytest (revisa tests.command en config.json)",
    5: "pytest no encontró ningún test",
    124: "los tests superaron el tiempo límite (tests.timeout_s)",
    127: "no se pudo lanzar el comando de tests",
}
MANAGED_FILES = [
    f"{LOOP}/backlog.json",
    f"{LOOP}/progress.md",
    f"{LOOP}/results.jsonl",
    f"{LOOP}/RESULTS.md",
    f"{LOOP}/tests-baseline.json",
    f"{LOOP}/baseline-report.md",
]
MANAGED_DIRS = [f"{LOOP}/notes"]
# Archivos que ningún agente edita (los escribe loopctl o el usuario a mano).
PROTECTED = MANAGED_FILES + [
    CONFIG_REL,
    f"{LOOP}/contracts.md",
    f"{LOOP}/plan.md",
    f"{LOOP}/informe.md",
    f"{LOOP}/run-loop.ps1",
    f"{LOOP}/.gitignore",
    TOKEN_REL,
    MARKER_REL,
    STOP_REL,
]
PROTECTED_PREFIXES = [".claude/", f"{LOOP}/bench/", f"{LOOP}/logs/", "backend/venv/"]
SUBAGENT_LOOPCTL_OK = {"show", "status", "brief", "next", "doctor", "note", "profiles"}
STALE_HEARTBEAT_S = 300
TERMINAL = {"passed", "rejected", "skipped"}
STOPPING = {"awaiting_human", "blocked"}
CAP_SNR = 999.0
DETACHED_PROCESS = 0x00000008
CREATE_NEW_PROCESS_GROUP = 0x00000200
CREATE_BREAKAWAY_FROM_JOB = 0x01000000
CREATE_NO_WINDOW = 0x08000000


class LoopError(Exception):
    pass


# ---------------------------------------------------------------------------
# Utilidades
# ---------------------------------------------------------------------------

def now() -> str:
    return dt.datetime.now().astimezone().isoformat(timespec="seconds")


def sh(args, cwd=None, check=True, timeout=None, env=None, input_text=None):
    p = subprocess.run(
        [str(a) for a in args], cwd=cwd, text=True, encoding="utf-8", errors="replace",
        capture_output=True, timeout=timeout, env=env, input=input_text,
    )
    if check and p.returncode != 0:
        msg = (p.stderr or p.stdout or "").strip()
        raise LoopError(f"Falló: {' '.join(map(str, args))}\n{msg[-3000:]}")
    return p


def shb(args, cwd=None, check=True, env=None, input_bytes=None):
    """Como sh() pero en bytes: en Windows el modo texto convierte \n <-> \r\n y rompe parches."""
    p = subprocess.run([str(a) for a in args], cwd=cwd, capture_output=True, env=env, input=input_bytes)
    if check and p.returncode != 0:
        raise LoopError(f"Falló: {' '.join(map(str, args))}\n{p.stderr.decode('utf-8', 'replace')[-3000:]}")
    return p


def git(root, *args, check=True, env=None, input_text=None) -> str:
    return sh(["git", "-C", str(root), *args], check=check, env=env, input_text=input_text).stdout.strip()


def read_json(path, default=None):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except FileNotFoundError:
        return default
    except json.JSONDecodeError as e:
        raise LoopError(f"JSON inválido en {path}: {e}") from e


def write_json(path, obj) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
    for _ in range(5):  # en Windows un lector puede tener el archivo abierto un instante
        try:
            os.replace(tmp, path)
            return
        except PermissionError:
            time.sleep(0.2)
    os.replace(tmp, path)


def sha256_file(p) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def norm(p) -> str:
    return os.path.normcase(os.path.abspath(str(p)))


def inside(child, parent) -> bool:
    c, p = norm(child), norm(parent)
    return c == p or c.startswith(p.rstrip("\\/") + os.sep)


def relposix(root, p) -> str:
    return Path(os.path.relpath(os.path.abspath(str(p)), os.path.abspath(str(root)))).as_posix()


def find_root(start=None) -> Path:
    env = os.environ.get("TTS_LOOP_ROOT")
    if env:
        return Path(env).resolve()
    start = Path(start or os.getcwd())
    p = sh(["git", "-C", str(start), "rev-parse", "--show-toplevel"], check=False)
    if p.returncode == 0 and p.stdout.strip():
        return Path(p.stdout.strip()).resolve()
    return start.resolve()


def fmt(v, nd=2) -> str:
    if v is None:
        return "—"
    if isinstance(v, bool):
        return "sí" if v else "no"
    if isinstance(v, float):
        return f"{v:.{nd}f}"
    return str(v)


def mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def tail(path, n=8) -> str:
    try:
        lines = Path(path).read_text(encoding="utf-8", errors="replace").splitlines()
    except FileNotFoundError:
        return ""
    return "\n".join(lines[-n:])


# ---------------------------------------------------------------------------
# Estado
# ---------------------------------------------------------------------------

class Ctx:
    def __init__(self, root=None, require=True, dry=False):
        self.root = Path(root) if root else find_root()
        self.state_path = self.root / STATE_REL
        self.state = read_json(self.state_path)
        if self.state is None and require:
            raise LoopError(
                "No hay loop inicializado en este directorio (falta perf-loop/backlog.json). "
                "Desde el repo principal ejecuta `/tts-loop init`."
            )
        if dry and self.state is not None:
            self.state = copy.deepcopy(self.state)
        self.dry = dry
        self.config = read_json(self.root / CONFIG_REL, {}) or {}

    @property
    def loop(self):
        return self.state["loop"]

    @property
    def tasks(self):
        return self.state["tasks"]

    def task(self, tid):
        for t in self.tasks:
            if t["id"] == tid:
                return t
        raise LoopError(f"No existe la tarea {tid}")

    def save(self):
        if self.dry:
            return
        self.loop["updated_at"] = now()
        write_json(self.state_path, self.state)

    def prefix(self):
        return self.loop.get("branch_prefix") or self.config.get("branch_prefix") or "perf/qwen"

    def phase_branch(self, n):
        return self.loop.get("phase_branches", {}).get(str(n)) or f"{self.prefix()}-f{n}"

    def task_branch(self, tid):
        return f"{self.prefix()}-task/{tid}"

    def python(self):
        p = self.config.get("python")
        if p:
            cand = Path(p) if Path(p).is_absolute() else self.root / p
            if cand.exists():
                return str(cand)
        for c in (self.root / "backend/venv/Scripts/python.exe", self.root / "backend/venv/bin/python"):
            if c.exists():
                return str(c)
        return sys.executable

    def bench_dir(self, tag):
        return self.root / LOOP / "bench" / tag

    def phase_title(self, n):
        return self.state.get("phases", {}).get(str(n), {}).get("title", f"Fase {n}")

    def defaults(self):
        return self.state.get("defaults", {})

    def max_attempts(self, t):
        return t.get("max_attempts") or self.defaults().get("max_attempts", {}).get(t["kind"], 3)


def progress(ctx, text):
    if ctx.dry:
        return
    p = ctx.root / LOOP / "progress.md"
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "a", encoding="utf-8", newline="\n") as f:
        f.write(f"- {now()} · {text}\n")


# ---------------------------------------------------------------------------
# Git
# ---------------------------------------------------------------------------

def current_branch(root):
    return git(root, "rev-parse", "--abbrev-ref", "HEAD")


def head_sha(root, ref="HEAD"):
    return git(root, "rev-parse", ref)


def branch_exists(root, b):
    return sh(["git", "-C", str(root), "rev-parse", "--verify", "--quiet", f"refs/heads/{b}"],
              check=False).returncode == 0


def porcelain(root, untracked=False):
    # Sin strip(): la primera columna de --porcelain puede ser un espacio significativo.
    out = sh(["git", "-C", str(root), "status", "--porcelain",
              "--untracked-files=" + ("all" if untracked else "no")]).stdout
    res = []
    for line in out.splitlines():
        if not line.strip():
            continue
        code, path = line[:2], line[3:]
        if " -> " in path:
            path = path.split(" -> ", 1)[1]
        res.append((code, path.strip('"')))
    return res


def dirty_code(root):
    return [p for c, p in porcelain(root) if not p.startswith(LOOP + "/")]


def untracked_outside(root):
    return [p for c, p in porcelain(root, untracked=True) if c == "??" and not p.startswith(LOOP + "/")]


def code_changed_since(root, commit):
    return code_changed_since_ref(root, commit, "HEAD")


def code_changed_since_ref(root, commit, ref):
    if not commit:
        return []
    out = git(root, "diff", "--name-only", commit, ref, "--", ".", f":(exclude){LOOP}", check=False)
    return [x for x in out.splitlines() if x.strip()]


def managed_paths(root):
    paths = [p for p in MANAGED_FILES if (root / p).exists()]
    for d in MANAGED_DIRS:
        if (root / d).is_dir() and any(f.is_file() for f in (root / d).rglob("*")):
            paths.append(d)
    return paths


def commit_state(ctx, msg) -> bool:
    if ctx.dry:
        return False
    paths = managed_paths(ctx.root)
    if not paths:
        return False
    git(ctx.root, "add", "-A", "--", *paths)
    if sh(["git", "-C", str(ctx.root), "diff", "--cached", "--quiet", "--", *paths], check=False).returncode == 0:
        return False
    git(ctx.root, "commit", "-q", "--no-verify", "-m", msg, "--", *paths)
    return True


def snapshot_managed(root):
    snap = {}
    for p in MANAGED_FILES:
        f = root / p
        if f.is_file():
            snap[p] = f.read_bytes()
    for d in MANAGED_DIRS:
        base = root / d
        if base.is_dir():
            for f in base.rglob("*"):
                if f.is_file():
                    snap[f.relative_to(root).as_posix()] = f.read_bytes()
    return snap


def restore_managed(root, snap):
    for p, data in snap.items():
        f = root / p
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_bytes(data)


def switch_carrying_state(ctx, target, msg):
    """Cambia de rama llevando consigo el estado del loop (perf-loop/) más reciente."""
    d = dirty_code(ctx.root)
    if d:
        raise LoopError("No puedo cambiar de rama con código sin commit: " + ", ".join(d[:10]))
    ctx.save()
    commit_state(ctx, msg)
    snap = snapshot_managed(ctx.root)
    git(ctx.root, "switch", "-q", target)
    restore_managed(ctx.root, snap)
    # ctx.state no se recarga: el snapshot es su serialización y así siguen valiendo
    # las referencias (t, L) que tenga el llamador.
    commit_state(ctx, msg)


def set_marker(ctx, tid):
    if ctx.dry:
        return
    cur = read_json(ctx.root / MARKER_REL)
    if cur and cur.get("task") == tid:
        return
    write_json(ctx.root / MARKER_REL, {"task": tid, "since": now(), "stop_blocks": {}})


def clear_marker(ctx):
    try:
        (ctx.root / MARKER_REL).unlink()
    except FileNotFoundError:
        pass


# ---------------------------------------------------------------------------
# Procesos
# ---------------------------------------------------------------------------

def pid_alive(pid) -> bool:
    if not pid:
        return False
    if IS_WIN:
        import ctypes
        k = ctypes.windll.kernel32
        h = k.OpenProcess(0x1000, False, int(pid))  # PROCESS_QUERY_LIMITED_INFORMATION
        if not h:
            return False
        code = ctypes.c_ulong()
        ok = k.GetExitCodeProcess(h, ctypes.byref(code))
        k.CloseHandle(h)
        return bool(ok) and code.value == 259  # STILL_ACTIVE
    try:
        os.kill(int(pid), 0)  # en Windows esto mataría el proceso; aquí solo POSIX
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def kill_tree(p):
    try:
        if IS_WIN:
            subprocess.run(["taskkill", "/PID", str(p.pid), "/T", "/F"], capture_output=True)
        else:
            import signal
            os.killpg(p.pid, signal.SIGKILL)
    except Exception:
        try:
            p.kill()
        except Exception:
            pass


def spawn_detached(args, cwd, log_path):
    log = open(log_path, "ab")
    kw = dict(cwd=str(cwd), stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, close_fds=True)
    if IS_WIN:
        last = None
        for flags in (DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP | CREATE_BREAKAWAY_FROM_JOB,
                      DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP):
            try:
                return subprocess.Popen(args, creationflags=flags, **kw)
            except OSError as e:
                last = e
        raise LoopError(f"No pude lanzar el proceso en segundo plano: {last}")
    return subprocess.Popen(args, start_new_session=True, **kw)


def keep_awake(on: bool):
    if not IS_WIN:
        return
    try:
        import ctypes
        es_continuous, es_system_required = 0x80000000, 0x00000001
        flags = es_continuous | (es_system_required if on else 0)
        ctypes.windll.kernel32.SetThreadExecutionState(flags)
    except Exception:
        pass


def port_in_use(port, host="127.0.0.1") -> bool:
    try:
        with socket.create_connection((host, int(port)), timeout=0.5):
            return True
    except OSError:
        return False


# ---------------------------------------------------------------------------
# Selección de la siguiente tarea
# ---------------------------------------------------------------------------

def deps_status(ctx, t):
    for d in t.get("requires", []):
        s = ctx.task(d)["status"]
        if s in ("rejected", "skipped"):
            return "dead", d
        if s != "passed":
            return "wait", d
    for d in t.get("depends_on", []):
        if ctx.task(d)["status"] not in TERMINAL:
            return "wait", d
    return "ok", None


def stop(reason, message, task=None, **kw):
    return {"stop": True, "reason": reason, "message": message, "task": task, **kw}


def human_message(ctx, t):
    h = t.get("human") or {}
    k = h.get("kind")
    tid = t["id"]
    if t["status"] == "blocked":
        why = (t.get("result") or {}).get("reason", "")
        return (f"Tarea {tid} bloqueada: {why}. Opciones: `/tts-loop retry {tid}` (más intentos) "
                f"o `/tts-loop skip {tid} <motivo>`.")
    if k == "start":
        return (f"La tarea {tid} ({t['title']}) necesita tu visto bueno antes de empezar. {h.get('message', '')} "
                f"`/tts-loop approve {tid}` o `/tts-loop skip {tid} <motivo>`.")
    if k == "ab":
        page = ctx.root / h["page"] if h.get("page") else "?"
        return (f"Escucha ciega pendiente para {tid}: abre {page} en el navegador y después escribe "
                f"`/tts-loop ab {tid} <código>`.")
    if k == "manual":
        return f"Tarea manual {tid}: {h.get('message', '')} Cuando termines, `/tts-loop approve {tid}`."
    return (f"La tarea {tid} espera tu revisión. {h.get('message', '')} "
            f"`/tts-loop approve {tid}` o `/tts-loop reject {tid} <motivo>`.")


def bench_status(ctx, tag):
    if not tag:
        return None
    st = read_json(ctx.bench_dir(tag) / "status.json")
    if st and st.get("state") in ("starting", "running") and not worker_alive(st):
        st["state"] = "crashed"
    return st


def worker_alive(st) -> bool:
    age = time.time() - float(st.get("heartbeat_ts") or 0)
    if age < 90:
        return True
    if age > STALE_HEARTBEAT_S:  # en Windows un PID reutilizado parecería vivo para siempre
        return False
    return pid_alive(st.get("pid"))


def resume_action(ctx, t):
    if t.get("last_tag") and t["last_tag"] in (t.get("consumed_tags") or []):
        return t.get("resume_hint") or ("implement" if t.get("implementer", True) else "verify")
    st = bench_status(ctx, t.get("last_tag"))
    if st:
        s = st.get("state")
        if s in ("starting", "running"):
            return "wait"
        if s in ("crashed", "error"):
            return "verify_failed"
        if s == "done":
            return "decide" if not code_changed_since(ctx.root, st.get("commit")) else "verify"
    if t["kind"] == "manual" and not t.get("manual_done"):
        return "await"
    return "implement" if t.get("implementer", True) else "verify"


def stop_file_sel(ctx):
    if not (ctx.root / STOP_REL).exists():
        return None
    cur = ctx.loop.get("current")
    extra = f" La tarea {cur} queda como está y se retomará al seguir." if cur else ""
    return stop("stop_file", "Hay un archivo perf-loop/STOP: el loop se detiene antes de la siguiente iteración."
                + extra + " Bórralo para seguir (run-loop.ps1 lo quita solo al arrancar).", cur)


def select(ctx, honor_stop=True):
    stopping = honor_stop and (ctx.root / STOP_REL).exists()
    sel = _select(ctx, mark=not stopping)
    if not stopping:
        return sel
    if sel.get("stop"):  # una parada que pide algo al usuario no se tapa con la del archivo STOP
        sel["message"] += " (Además hay un perf-loop/STOP: bórralo cuando quieras seguir.)"
        return sel
    return stop_file_sel(ctx)


def _select(ctx, mark=True):
    L = ctx.loop
    if L.get("done"):
        return stop("done", "Backlog completado. `loopctl export` saca una rama limpia sin el andamiaje del loop.")
    cur = L.get("current")
    if cur:
        t = ctx.task(cur)
        if t["status"] in STOPPING:
            return stop(t["status"], human_message(ctx, t), cur)
        if t["status"] == "in_progress":
            if mark:
                set_marker(ctx, cur)
            return {"stop": False, "task": cur, "action": resume_action(ctx, t), "title": t["title"]}
        L["current"] = None
    for t in ctx.tasks:
        if t["status"] in STOPPING:
            return stop(t["status"], human_message(ctx, t), t["id"])
    changed = False
    while True:
        ph = L["phase"]
        phase_tasks = [t for t in ctx.tasks if t["phase"] == ph]
        for t in phase_tasks:
            if t["status"] != "pending":
                continue
            if ctx.config.get("skip_optional") and t.get("optional"):
                t["status"] = "skipped"
                t["result"] = {"kind": "skip", "reason": "skip_optional en config", "at": now()}
                changed = True
                continue
            st, dep = deps_status(ctx, t)
            if st == "dead":
                t["status"] = "skipped"
                t["result"] = {"kind": "skip", "reason": f"requiere {dep}, que no se superó", "at": now()}
                progress(ctx, f"{t['id']} se omite: requiere {dep}, que no se superó")
                changed = True
        pending = [t for t in phase_tasks if t["status"] == "pending"]
        elig = [t for t in pending if deps_status(ctx, t)[0] == "ok"]
        if elig:
            t = elig[0]
            if t.get("human_start") and not t.get("start_approved"):
                t["status"] = "awaiting_human"
                t["human"] = {"kind": "start", "message": t.get("human_start_message", ""), "at": now()}
                L["current"] = t["id"]
                ctx.save()
                progress(ctx, f"{t['id']} espera visto bueno para empezar")
                return stop("awaiting_human", human_message(ctx, t), t["id"])
            if changed:
                ctx.save()
            return {"stop": False, "task": t["id"], "action": "begin", "title": t["title"]}
        if pending:
            if changed:
                ctx.save()
            ids = ", ".join(f"{t['id']} (espera {deps_status(ctx, t)[1]})" for t in pending)
            return stop("blocked", f"Hay tareas pendientes con dependencias sin cumplir: {ids}")
        nxt = ph + 1
        if not any(t["phase"] == nxt for t in ctx.tasks):
            L["done"] = True
            ctx.save()
            progress(ctx, "Backlog completado")
            return stop("done", "Backlog completado. `loopctl export` saca una rama limpia sin el andamiaje del loop.")
        L["phase"] = nxt
        changed = True
        progress(ctx, f"Fase {ph} cerrada; empieza la fase {nxt}: {ctx.phase_title(nxt)}")


# ---------------------------------------------------------------------------
# Banco y puertas
# ---------------------------------------------------------------------------

def bench_cmd(ctx, b, tag):
    cfg = ctx.config.get("bench", {})
    ph = b.get("phrases", "all")
    if ph == "smoke":
        ph = ",".join(cfg.get("smoke_phrases", ["p01", "p02"]))
    elif isinstance(ph, list):
        ph = ",".join(ph)
    cmd = [ctx.python(), "scripts/bench_tts.py", "--config", CONFIG_REL, "--tag", tag,
           "--out", f"{LOOP}/bench/{tag}", "--phrases", ph,
           "--passes", ",".join(b.get("passes", ["cold", "warm"])),
           "--miopen-db", b.get("miopen_db", "persist")]
    env = {**(cfg.get("env") or {}), **(b.get("env") or {})}
    for k, v in env.items():
        cmd += ["--env", f"{k}={v}"]
    if b.get("scenario"):
        cmd += ["--scenario", b["scenario"]]
    if b.get("concurrent"):
        cmd += ["--concurrent", str(b["concurrent"])]
    return cmd


def compare_cmd(ctx, ref, cand, metrics, out_rel):
    return [ctx.python(), "scripts/compare_wavs.py", "--config", CONFIG_REL,
            "--ref", f"{LOOP}/bench/{ref}", "--cand", f"{LOOP}/bench/{cand}",
            "--pass", "warm", "--metrics", metrics, "--out", out_rel]


def fill(cmd, ctx):
    return [ctx.python() if c == "{python}" else c for c in cmd]


def tests_cmd(ctx, junit_rel):
    cmd = ctx.config.get("tests", {}).get("command") or [
        "{python}", "-m", "pytest", "backend/tests", "-q", "-p", "no:cacheprovider"]
    return fill(cmd, ctx) + [f"--junitxml={junit_rel}"]


def lint_cmd(ctx, files):
    cmd = ctx.config.get("lint", {}).get("command") or [
        "{python}", "-m", "ruff", "check", "--no-cache", "--output-format", "concise",
        "--select", "E9,F63,F7,F82"]
    return fill(cmd, ctx) + files


def changed_py(ctx, t):
    out = git(ctx.root, "diff", "--name-only", "--diff-filter=AMR", t["base"], "HEAD", "--", "*.py", check=False)
    return [x for x in out.splitlines() if x.strip() and (ctx.root / x).exists()]


def quality_def(ctx, t):
    qt = t.get("quality_type")
    if qt is None:
        return {}
    return ctx.defaults().get("quality", {}).get(str(qt), {})


def compare_plan(ctx, t, tag):
    g = t.get("gate") or {}
    L = ctx.loop
    kinds = list(g.get("compare") or [])
    if g.get("bench"):
        for k in quality_def(ctx, t).get("compare", []):
            if k not in kinds:
                kinds.append(k)
    plan = []
    for k in kinds:
        if k == "prev":
            plan.append(("prev", L.get("quality_ref"), tag, "snr,dur"))
        elif k == "anchor":
            plan.append(("anchor", L.get("anchor"), tag, "all"))
        elif k == "self":
            plan.append(("self", L.get("quality_ref"), L.get("quality_ref"), "all"))
    if t["kind"] == "baseline":
        plan.append(("aa", tag, f"{tag}-aa", "all"))
    return plan


def plan_steps(ctx, t, tag):
    py = ctx.python()
    g = t.get("gate") or {}
    bcfg = ctx.config.get("bench", {})
    b_timeout = int(bcfg.get("timeout_s", 10800))
    c_timeout = int(ctx.config.get("compare", {}).get("timeout_s", 3600))
    steps = []
    b = g.get("bench")
    if b:
        for i, pre in enumerate(b.get("pre_runs") or []):
            sub = {**b, **pre}
            sub.pop("pre_runs", None)
            sfx = pre.get("suffix", f"pre{i + 1}")
            steps.append({"name": f"bench-{sfx}", "kind": "bench", "cmd": bench_cmd(ctx, sub, f"{tag}-{sfx}"),
                          "timeout": b_timeout})
        main = {k: v for k, v in b.items() if k != "pre_runs"}
        steps.append({"name": "bench", "kind": "bench", "cmd": bench_cmd(ctx, main, tag), "timeout": b_timeout})
        if t["kind"] == "baseline":
            steps.append({"name": "bench-aa", "kind": "bench", "cmd": bench_cmd(ctx, main, f"{tag}-aa"),
                          "timeout": b_timeout})
    for kind, ref, cand, metrics in compare_plan(ctx, t, tag):
        name = f"compare-{kind}"
        if not ref:
            steps.append({"name": name, "skip_reason": "todavía no hay referencia"})
            continue
        out_rel = f"{LOOP}/bench/{tag}/{name}.json"
        steps.append({"name": name, "kind": "compare", "cmd": compare_cmd(ctx, ref, cand, metrics, out_rel),
                      "timeout": c_timeout, "ref": ref, "cand": cand})
    for i, c in enumerate(g.get("selftest") or []):
        steps.append({"name": f"selftest-{i + 1}", "kind": "cmd", "cmd": [py, *c], "timeout": 1800})
    if g.get("tests"):
        steps.append({"name": "tests", "kind": "tests", "fatal": False,
                      "cmd": tests_cmd(ctx, f"{LOOP}/bench/{tag}/junit.xml"),
                      "timeout": int(ctx.config.get("tests", {}).get("timeout_s", 1800))})
    if g.get("lint", True):
        files = changed_py(ctx, t)
        if files:
            steps.append({"name": "lint", "kind": "lint", "fatal": False, "cmd": lint_cmd(ctx, files),
                          "timeout": 600})
    return steps


def parse_junit(path, rc):
    res = {"rc": rc, "passed": 0, "failed": [], "skipped": 0, "passed_ids": [], "skipped_ids": []}
    if not Path(path).exists():
        res["missing"] = True
        return res
    try:
        root = ET.parse(path).getroot()
    except ET.ParseError as e:
        res["missing"] = True
        res["error"] = str(e)
        return res
    for tc in root.iter("testcase"):
        name = f"{tc.get('classname', '')}::{tc.get('name', '')}"
        if tc.find("failure") is not None or tc.find("error") is not None:
            res["failed"].append(name)
        elif tc.find("skipped") is not None:
            res["skipped"] += 1
            res["skipped_ids"].append(name)
        else:
            res["passed"] += 1
            res["passed_ids"].append(name)
    return res


def test_modules_changed(ctx, since):
    """Nombres (sin .py) de los archivos de test añadidos o cambiados desde `since` hasta HEAD."""
    if not since:
        return set()
    out = git(ctx.root, "diff", "--name-only", "--diff-filter=AMR", since, "HEAD", "--", ".", f":(exclude){LOOP}",
              check=False)
    stems = set()
    for x in out.splitlines():
        name = x.strip().rsplit("/", 1)[-1]
        if name.endswith(".py") and (name.startswith("test_") or name.endswith("_test.py")):
            stems.add(name[:-3])
    return stems


def tests_in_modules(ids, stems):
    """Ids de junit (p. ej. «backend.tests.test_x::test_y» o «::tests.test_x») que pertenecen a esos módulos."""
    if not stems:
        return []
    return [i for i in ids if stems & set(re.split(r"[.:/\\]+", i))]


LINT_LINE = re.compile(r"^\S.*:\d+:\d+: [A-Z]+\d+")


def run_step(ctx, step, bd, st, sp):
    logs = bd / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    log = logs / f"{step['name']}.log"
    env = dict(os.environ, PYTHONUTF8="1", PYTHONIOENCODING="utf-8", TTS_LOOP_TAG=bd.name)
    kw = {"creationflags": CREATE_NO_WINDOW} if IS_WIN else {"start_new_session": True}
    t0 = time.time()
    with open(log, "ab") as lf:
        lf.write(("$ " + " ".join(map(str, step["cmd"])) + "\n").encode("utf-8"))
        lf.flush()
        try:
            p = subprocess.Popen([str(c) for c in step["cmd"]], cwd=str(ctx.root), stdout=lf,
                                 stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, env=env, **kw)
        except OSError as e:
            lf.write(f"[loopctl] no se pudo lanzar: {e}\n".encode("utf-8"))
            return 127, 0.0
        timeout = step.get("timeout") or 3600
        while True:
            try:
                rc = p.wait(timeout=10)
                break
            except subprocess.TimeoutExpired:
                st["heartbeat_ts"] = time.time()
                write_json(sp, st)
                if time.time() - t0 > timeout:
                    kill_tree(p)
                    rc = 124
                    lf.write(f"\n[loopctl] timeout tras {timeout}s\n".encode("utf-8"))
                    break
    secs = time.time() - t0
    if step.get("kind") == "tests":
        write_json(bd / "tests.json", parse_junit(bd / "junit.xml", rc))
    if step.get("kind") == "lint":
        errs = [ln for ln in tail(log, 400).splitlines() if LINT_LINE.match(ln)]
        write_json(bd / "lint.json", {"rc": rc, "errors": len(errs) if rc != 0 else 0, "sample": errs[:20]})
    return rc, secs


def run_worker(root, tid, tag):
    ctx = Ctx(root)
    t = ctx.task(tid)
    bd = ctx.bench_dir(tag)
    sp = bd / "status.json"
    st = read_json(sp, {}) or {}
    st.update(state="running", pid=os.getpid(), heartbeat_ts=time.time())
    st.setdefault("steps", [])
    write_json(sp, st)
    keep_awake(True)
    try:
        steps = plan_steps(ctx, t, tag)
        st["plan"] = [s["name"] for s in steps]
        write_json(sp, st)
        for step in steps:
            if step.get("skip_reason"):
                st["steps"].append({"name": step["name"], "skipped": step["skip_reason"]})
                continue
            st["step"] = step["name"]
            st["step_started_ts"] = time.time()
            write_json(sp, st)
            rc, secs = run_step(ctx, step, bd, st, sp)
            st["steps"].append({"name": step["name"], "rc": rc, "seconds": round(secs, 1),
                                "fatal": step.get("fatal", True), "log": f"logs/{step['name']}.log"})
            write_json(sp, st)
            if rc != 0 and step.get("fatal", True):
                break
        st["step"] = "gate"
        write_json(sp, st)
        gate = evaluate_gate(ctx, t, tag, st)
        write_json(bd / "gate.json", gate)
        st.update(state="done", verdict=gate["verdict"], finished_at=now())
    except Exception as e:  # el worker nunca debe morir sin dejar rastro
        st.update(state="error", error=(f"{e}\n{traceback.format_exc()}")[-4000:], finished_at=now())
    finally:
        st["heartbeat_ts"] = time.time()
        write_json(sp, st)
        keep_awake(False)


OPS = {
    ">=": lambda v, x: v >= x,
    "<=": lambda v, x: v <= x,
    "==": lambda v, x: v == x,
    "abs<=": lambda v, x: abs(v) <= x,
    "contains": lambda v, x: str(x).lower() in str(v).lower().replace("\\", "/"),
}


def dig(d, path):
    cur = d
    for part in path.split("."):
        if isinstance(cur, dict) and part in cur:
            cur = cur[part]
        else:
            return None
    return cur


def phrase_times(bench, pass_name):
    for p in (bench or {}).get("passes", []) or []:
        if p.get("name") == pass_name:
            return {x["id"]: float(x["wall_s"]) for x in p.get("phrases", []) if x.get("wall_s") is not None}
    return {}


SPEEDUP_PASSES = ("hot", "warm")  # el speedup se mide en régimen estable (hot) si ambas mediciones lo tienen


def derived_metrics(bench, ref):
    d = {}
    if not bench:
        return d
    warm, cold, hot = phrase_times(bench, "warm"), phrase_times(bench, "cold"), phrase_times(bench, "hot")
    # cold y warm son procesos nuevos: su diferencia aísla el Find de MIOpen (fase 1).
    common = sorted(set(warm) & set(cold))
    if common:
        sw = sum(warm[i] for i in common)
        sc = sum(cold[i] for i in common)
        d["cold_warm_ratio"] = sc / sw if sw else None
        d["find_overhead_s_mean"] = (sc - sw) / len(common)
    # warm (proceso nuevo, Find-Db ya poblada) frente a hot (mismo proceso): lo que cuesta un reinicio.
    c_wh = sorted(set(warm) & set(hot))
    if c_wh:
        d["restart_overhead_s_mean"] = sum(warm[i] - hot[i] for i in c_wh) / len(c_wh)
    if ref:
        for pname in SPEEDUP_PASSES:
            cur, rp = phrase_times(bench, pname), phrase_times(ref, pname)
            c2 = sorted(set(cur) & set(rp))
            if c2 and sum(rp[i] for i in c2) > 0:
                d["speedup"] = 1 - sum(cur[i] for i in c2) / sum(rp[i] for i in c2)
                d["speedup_n"] = len(c2)
                d["speedup_pass"] = pname
                break
    sc_ = bench.get("scenarios") or {}
    con = sc_.get("contention") or {}
    if con.get("transcribe_wait_s") is not None and con.get("generation_s_max"):
        d["transcribe_wait_ratio"] = con["transcribe_wait_s"] / con["generation_s_max"]
    bat = sc_.get("batch") or {}
    if bat.get("throughput_gain") is not None:
        d["batch_throughput_gain"] = bat["throughput_gain"]
    return d


def validate_bench(ctx, b, tag, bd, want_passes=None):
    errs = []
    if b.get("schema") != "vb-bench/1":
        errs.append("bench.json: 'schema' debe ser 'vb-bench/1'")
    if b.get("tag") != tag:
        errs.append(f"bench.json: tag {b.get('tag')!r} no coincide con {tag!r}")
    for k in ("commit", "dirty", "code_path", "passes", "summary"):
        if k not in b:
            errs.append(f"bench.json: falta '{k}'")
    names = [p.get("name") for p in b.get("passes") or []]
    for w in want_passes or []:
        if w not in names:
            errs.append(f"bench.json: falta la pasada {w} (se pidió --passes {','.join(want_passes)})")
    if "hot" in names:
        i = names.index("hot")
        srv = b.get("server") or {}
        h = srv.get("hot") or {}
        if i == 0:
            errs.append("bench.json: la pasada hot no puede ser la primera (reutiliza el proceso de la anterior)")
        elif h.get("same_process_as") != names[i - 1]:
            errs.append(f"bench.json: server.hot.same_process_as debe ser {names[i - 1]!r}")
        elif h.get("pid") is None or h.get("pid") != (srv.get(names[i - 1]) or {}).get("pid"):
            errs.append("bench.json: la pasada hot no corrió en el mismo proceso que la anterior (server.<pasada>.pid)")
        if (b.get("summary") or {}).get("hot_mean_s") is None:
            errs.append("bench.json: falta summary.hot_mean_s")
    if b.get("dirty"):
        errs.append("bench.json: se midió con el árbol de trabajo sucio")
    cp = b.get("code_path")
    if cp and not inside(cp if Path(cp).is_absolute() else ctx.root / cp, ctx.root):
        errs.append(f"el servidor importó `backend` desde fuera del worktree: {cp}")
    for p in b.get("passes") or []:
        if not p.get("phrases"):
            errs.append(f"bench.json: la pasada {p.get('name')} no tiene frases")
        for x in p.get("phrases") or []:
            for k in ("id", "wall_s", "audio_s", "steps", "wav"):
                if k not in x:
                    errs.append(f"bench.json: {p.get('name')}/{x.get('id')} sin '{k}'")
            if x.get("wav") and not (bd / x["wav"]).exists():
                errs.append(f"falta el WAV {x['wav']}")
    return errs[:20]


def validate_stages(bench):
    errs = []
    for p in (bench or {}).get("passes", []) or []:
        if p.get("name") not in ("warm", "hot"):
            continue
        for x in p.get("phrases") or []:
            stg = x.get("stages") or {}
            for k in ("reference", "talker", "codec"):
                if not isinstance((stg.get(k) or {}).get("ms"), (int, float)):
                    errs.append(f"{p.get('name')}/{x.get('id')}: falta stages.{k}.ms")
            if not isinstance((stg.get("talker") or {}).get("steps_per_s"), (int, float)):
                errs.append(f"{p.get('name')}/{x.get('id')}: falta stages.talker.steps_per_s")
    if not (bench or {}).get("env_checks"):
        errs.append("bench.json: falta env_checks (línea VBPERF stage=env)")
    return errs[:20]


def validate_compare(c, name):
    errs = []
    if not c:
        return [f"falta {name}.json"]
    if c.get("schema") != "vb-compare/1":
        errs.append(f"{name}: 'schema' debe ser 'vb-compare/1'")
    if not isinstance(c.get("phrases"), list):
        errs.append(f"{name}: falta la lista 'phrases'")
    s = c.get("summary") or {}
    for k in ("n", "snr_db_min"):
        if k not in s:
            errs.append(f"{name}: falta summary.{k}")
    return errs


def validate_phrases(ctx):
    d = read_json(ctx.root / "scripts/bench_phrases.json")
    if d is None:
        return ["falta scripts/bench_phrases.json"]
    errs = []
    ph = d.get("phrases") or []
    if d.get("language") != "es":
        errs.append("bench_phrases.json: 'language' debe ser 'es'")
    if len(ph) != 12:
        errs.append(f"se esperan 12 frases y hay {len(ph)}")
    ids = [x.get("id") for x in ph]
    if len(set(ids)) != len(ids):
        errs.append("hay ids repetidos")
    for x in ph:
        if not re.fullmatch(r"p\d{2}", str(x.get("id", ""))):
            errs.append(f"id inválido: {x.get('id')!r} (formato pNN)")
        if not isinstance(x.get("seed"), int):
            errs.append(f"{x.get('id')}: 'seed' debe ser entero")
        ts = x.get("target_s")
        if not isinstance(ts, (int, float)) or not 3 <= ts <= 12:
            errs.append(f"{x.get('id')}: 'target_s' debe estar entre 3 y 12")
        if not str(x.get("text", "")).strip():
            errs.append(f"{x.get('id')}: texto vacío")
    lens = [len(str(x.get("text", ""))) for x in ph]
    if len(set(lens)) != len(lens):
        errs.append("las frases deben tener longitudes de texto distintas")
    for s in ctx.config.get("bench", {}).get("smoke_phrases", []):
        if s not in ids:
            errs.append(f"smoke_phrases incluye {s}, que no está en el conjunto")
    return errs


def expand_freeze(ctx):
    files = []
    for p in ctx.state.get("freeze_paths", []):
        f = ctx.root / p
        if f.is_dir():
            files += [x.relative_to(ctx.root).as_posix() for x in sorted(f.rglob("*")) if x.is_file()]
        elif f.is_file():
            files.append(p)
    return files


def venv_pkg_dir(root):
    for c in [root / "backend/venv/Lib/site-packages/qwen_tts",
              *sorted((root / "backend/venv/lib").glob("python*/site-packages/qwen_tts"))]:
        if c.is_dir():
            return c
    return None


def venv_pkg_hash(root):
    d = venv_pkg_dir(root)
    if not d:
        return None
    h = hashlib.sha256()
    for f in sorted(d.rglob("*.py")):
        h.update(f.relative_to(d).as_posix().encode())
        h.update(sha256_file(f).encode())
    return h.hexdigest()


def frozen_mismatches(ctx):
    out = []
    vp = ctx.loop.get("venv_pkg_hash")
    if vp and venv_pkg_hash(ctx.root) != vp:
        out.append("backend/venv/…/qwen_tts (paquete del venv compartido)")
    for p, h in (ctx.loop.get("frozen") or {}).items():
        f = ctx.root / p
        if not f.exists():
            out.append(f"{p} (borrado)")
        elif sha256_file(f) != h:
            out.append(p)
    return out


def anchor_mismatches(ctx):
    out = []
    for p, h in (ctx.loop.get("anchor_wavs") or {}).items():
        f = ctx.root / p
        if not f.exists() or sha256_file(f) != h:
            out.append(p)
    return out


def bench_ids(bench, pass_name="warm"):
    return set(phrase_times(bench, pass_name))


def evaluate_gate(ctx, t, tag, st):
    bd = ctx.bench_dir(tag)
    g = t.get("gate") or {}
    L = ctx.loop
    qd = quality_def(ctx, t)
    fails, warns, checks = [], [], []
    steps = st.get("steps", [])
    for s in steps:
        if s.get("skipped"):
            warns.append(f"paso {s['name']} omitido: {s['skipped']}")
        elif s["name"] not in ("tests", "lint") and s.get("rc") not in (0, None):
            fails.append(f"el paso {s['name']} terminó con código {s['rc']} (ver {s.get('log')})")
    ran_compares = {s["name"][8:] for s in steps if s["name"].startswith("compare-") and not s.get("skipped")}
    data = {"noise": L.get("noise") or {}}
    bench = None
    if g.get("bench"):
        bench = read_json(bd / "bench.json")
        if bench is None:
            if not any(s["name"] == "bench" and s.get("rc") not in (0, None) for s in steps):
                fails.append("el banco no escribió bench.json")
        else:
            fails += validate_bench(ctx, bench, tag, bd, want_passes=g["bench"].get("passes"))
            bc = str(bench.get("commit", ""))
            if bc and not str(st.get("commit", "")).startswith(bc[:12]):
                fails.append(f"bench.json dice commit {bc[:10]} pero se lanzó en {str(st.get('commit'))[:10]}")
    data["bench"] = bench
    for k in ran_compares:
        data[f"compare_{k}"] = read_json(bd / f"compare-{k}.json")
    perf_ref = read_json(ctx.bench_dir(L["perf_ref"]) / "bench.json") if L.get("perf_ref") else None
    data["derived"] = derived_metrics(bench, perf_ref if L.get("perf_ref") != tag else None)

    for v in g.get("validate") or []:
        if v == "phrases":
            fails += validate_phrases(ctx)
        elif v == "bench_schema" and bench is None and g.get("bench") is None:
            fails.append("validate bench_schema sin banco configurado")
        elif v == "stages":
            fails += validate_stages(bench)
        elif v == "compare_schema":
            for k in ran_compares:
                fails += validate_compare(data.get(f"compare_{k}"), f"compare-{k}")
        elif v == "files":
            for f in g.get("files") or []:
                if not (ctx.root / f).exists():
                    fails.append(f"falta {f}")
            for f, pat in (g.get("files_match") or {}).items():
                fp = ctx.root / f
                txt = fp.read_text(encoding="utf-8", errors="replace") if fp.exists() else ""
                if not re.search(pat, txt, re.M):
                    fails.append(f"{f} no contiene {pat!r}")

    defs = []
    if g.get("bench"):
        defs += qd.get("checks", [])
    defs += g.get("checks", [])
    for c in defs:
        label = c.get("label") or f"{c['metric']} {c['op']} {c['value']}"
        root_key = c["metric"].split(".")[0]
        if root_key.startswith("compare_") and root_key[8:] not in ran_compares:
            if c.get("required"):
                fails.append(f"comprobación obligatoria sin datos (no se calculó {root_key}): {label}")
            else:
                warns.append(f"comprobación omitida (no se calculó {root_key}): {label}")
            continue
        v = dig(data, c["metric"])
        if v is None:
            fails.append(f"métrica ausente: {c['metric']} ({label})")
            checks.append({"label": label, "metric": c["metric"], "value": None, "ok": False})
            continue
        try:
            ok = bool(OPS[c["op"]](v, c["value"]))
        except Exception as e:
            ok = False
            fails.append(f"no pude evaluar {label}: {e}")
        checks.append({"label": label, "metric": c["metric"], "value": v, "op": c["op"],
                       "target": c["value"], "ok": ok})
        if not ok:
            fails.append(f"{label}: valor {fmt(v, 4) if isinstance(v, float) else v}")

    tests_info = None
    if g.get("tests"):
        tj = read_json(bd / "tests.json")
        rc = (tj or {}).get("rc")
        if not tj:
            fails.append("no hay resultados de tests (junit.xml); mira logs/tests.log")
        elif rc is not None and rc not in PYTEST_COMPLETE_RC:
            # Un pytest interrumpido deja un junit.xml con el error de colección como único «fallo»: si lo
            # diéramos por bueno, la regla «ningún test nuevo fallando» no comprobaría nada.
            why = PYTEST_RC_WHY.get(rc, f"pytest terminó con código {rc}")
            tests_info = {"passed": tj.get("passed"), "failed": len(tj.get("failed", [])), "rc": rc,
                          "incomplete": True}
            culprits = ", ".join(tj.get("failed", [])[:3]) or "ver el log"
            if g["tests"] == "record":
                hint = ("el usuario decide cómo corregir tests.command en config.json (p. ej. --ignore del módulo "
                        "roto) y vuelve a lanzar la tarea")
            else:
                hint = ("si el módulo lo añadió o tocó esta tarea, arréglalo; si ya fallaba antes, bloquea la tarea "
                        "para que el usuario corrija tests.command")
            fails.append(f"la batería de tests no terminó: {why} ({culprits}); {tj.get('passed', 0)} tests pasaron. "
                         f"Mira logs/tests.log: {hint}")
        elif tj.get("missing"):
            fails.append("no hay resultados de tests (junit.xml); mira logs/tests.log")
        elif g["tests"] == "record":
            tests_info = {"passed": tj.get("passed"), "failed": len(tj.get("failed", [])), "record": True}
            if tj.get("failed"):
                warns.append(f"{len(tj['failed'])} tests ya fallan en el código de partida; quedan como referencia: "
                             f"{', '.join(tj['failed'][:5])}")
                mine = tests_in_modules(tj["failed"], test_modules_changed(ctx, L.get("scaffold_commit")))
                if mine:
                    warns.append(f"ojo: {len(mine)} de esos fallos están en tests que añadió o cambió el loop "
                                 f"({', '.join(mine[:5])}); las tareas que vuelvan a tocar esos archivos tendrán que "
                                 "dejarlos pasando")
            if not tj.get("passed"):
                fails.append("la batería de tests no tiene ningún test que pase: una lista de partida así no "
                             "protege nada; revisa tests.command en config.json")
        else:
            bl = read_json(ctx.root / TESTS_BASELINE_REL)
            if bl is None:
                warns.append("no hay perf-loop/tests-baseline.json: cualquier fallo cuenta como nuevo")
            elif bl.get("stale"):
                rt = next((x["id"] for x in ctx.tasks if (x.get("gate") or {}).get("tests") == "record"), "0.0")
                fails.append("la lista de tests de partida está pendiente de registrarse de nuevo "
                             f"({bl.get('stale_reason', 'tarea de tests reabierta')}): hace falta cerrar antes la "
                             f"{rt} (`/tts-loop retry {rt}` si está bloqueada; `/tts-loop reopen {rt}` si se omitió)")
            base = set((bl or {}).get("failed", []))
            # Un test que ya fallaba deja de contar como «de partida» si esta tarea toca su archivo.
            touched = tests_in_modules(tj.get("failed", []), test_modules_changed(ctx, t.get("base")))
            new = [x for x in tj.get("failed", []) if x not in base or x in touched]
            gone = sorted(set((bl or {}).get("passed_ids", [])) - set(tj.get("passed_ids", [])) - set(tj.get("failed", [])))
            tests_info = {"passed": tj.get("passed"), "failed": len(tj.get("failed", [])), "new_failures": new,
                          "no_longer_run": gone}
            if new:
                fails.append(f"{len(new)} tests nuevos fallando: {', '.join(new[:5])}")
            if gone:
                fails.append(f"{len(gone)} tests que pasaban en la línea base ya no se ejecutan o se omiten: "
                             f"{', '.join(gone[:5])}")
    lint = read_json(bd / "lint.json")
    if lint and lint.get("errors"):
        fails.append(f"ruff encontró {lint['errors']} errores graves (E9/F63/F7/F82): {lint.get('sample', [])[:3]}")

    mism = frozen_mismatches(ctx)
    if mism:
        fails.append("el arnés de evaluación congelado cambió: " + ", ".join(mism))
    if "anchor" in ran_compares:
        am = anchor_mismatches(ctx)
        if am:
            fails.append("los WAV de la línea base cambiaron: " + ", ".join(am[:5]))

    snr_prev = dig(data, "compare_prev.summary.snr_db_min")
    if t.get("quality_type") == 3 and snr_prev is not None and snr_prev >= CAP_SNR:
        fails.append("cambio de tipo 3 pero la onda salió idéntica: ¿se está midiendo el código nuevo?")
    sp_ = data["derived"].get("speedup")
    sp_pass = data["derived"].get("speedup_pass") or "warm"
    nz = L.get("noise") or {}
    noise = nz.get(f"{sp_pass}_rel_diff", nz.get("warm_rel_diff"))
    factor = float(ctx.config.get("noise", {}).get("speedup_factor", 2.0))
    uses_speedup = any("speedup" in c["metric"] for c in defs)
    if sp_ is not None and noise and abs(sp_) < factor * noise and uses_speedup:
        warns.append(f"speedup {sp_:.3f} dentro de {factor}× el ruido A/A de {sp_pass} ({noise:.3f}): "
                     "decisión poco fiable")
    if sp_ is not None and sp_pass != "hot" and uses_speedup:
        warns.append("speedup medido en warm (proceso nuevo) porque la referencia o la medición no tienen pasada "
                     "hot: incluye el coste de reinicio por longitud")

    h = g.get("human") or qd.get("human")
    basis = g.get("ab_basis", "prev")
    if h == "ab_if_changed":
        snr = dig(data, f"compare_{basis}.summary.snr_db_min")
        h = "ab" if (snr is None or snr < 50) else None
    if t["kind"] == "baseline":
        h = "approve"
    if t["kind"] == "phase_gate" and ctx.config.get("pause_at_phase_gates", True) and not h:
        h = "approve"
    verdict = "fail" if fails else ("needs_human" if h else "pass")

    anchor_bench = read_json(ctx.bench_dir(L["anchor"]) / "bench.json") if L.get("anchor") else None
    update_refs = bool(
        g.get("update_refs", True) and bench and bench_ids(bench)
        and (anchor_bench is None or bench_ids(bench) >= bench_ids(anchor_bench))
    )
    s = (bench or {}).get("summary") or {}
    metrics = {
        "hot_mean_s": s.get("hot_mean_s"),
        "warm_mean_s": s.get("warm_mean_s"), "cold_mean_s": s.get("cold_mean_s"),
        "rtf_hot_mean": s.get("rtf_hot_mean"),
        "rtf_warm_mean": s.get("rtf_warm_mean"), "talker_steps_per_s_mean": s.get("talker_steps_per_s_mean"),
        "restart_overhead_s_mean": data["derived"].get("restart_overhead_s_mean"),
        "ufdb_lines_added_cold": s.get("ufdb_lines_added_cold"),
        "ufdb_lines_added_warm": s.get("ufdb_lines_added_warm"),
        "snr_prev_min": snr_prev,
        "snr_anchor_min": dig(data, "compare_anchor.summary.snr_db_min"),
        "spk_delta_anchor": dig(data, "compare_anchor.summary.spk_cos_delta_mean"),
        "wer_delta_anchor": dig(data, "compare_anchor.summary.wer_delta"),
        "dur_dev_anchor": dig(data, "compare_anchor.summary.dur_dev_max"),
        "speedup": data["derived"].get("speedup"),
        "speedup_pass": data["derived"].get("speedup_pass"),
        "cold_warm_ratio": data["derived"].get("cold_warm_ratio"),
    }
    return {
        "task": t["id"], "tag": tag, "at": now(), "commit": st.get("commit"), "verdict": verdict,
        "human": h if verdict == "needs_human" else None, "failures": fails, "warnings": warns,
        "checks": checks, "tests": tests_info, "derived": data["derived"], "metrics": metrics,
        "update_refs": update_refs,
    }


def render_gate(gate) -> str:
    if not gate:
        return "Sin puerta evaluada."
    lines = [f"ESTADO: done · tarea {gate['task']} · {gate['tag']} · veredicto: {gate['verdict'].upper()}"]
    if gate.get("human"):
        lines.append(f"Decisión humana requerida: {gate['human']} → ejecuta `loopctl await {gate['task']}`")
    for c in gate.get("checks", []):
        mark = "OK " if c.get("ok") else "NO "
        v = c.get("value")
        lines.append(f"  {mark}{c['label']}  (valor {fmt(v, 4) if isinstance(v, float) else v})")
    for f in gate.get("failures", []):
        lines.append(f"  ✗ {f}")
    for w in gate.get("warnings", []):
        lines.append(f"  ! {w}")
    m = {k: v for k, v in (gate.get("metrics") or {}).items() if v is not None}
    if m:
        lines.append("  métricas: " + ", ".join(f"{k}={fmt(v, 3)}" for k, v in m.items()))
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Resultados
# ---------------------------------------------------------------------------

def record_result(ctx, t, gate):
    if ctx.dry:
        return
    row = {"at": now(), "task": t["id"], "title": t["title"], "phase": t["phase"], "tag": t.get("last_tag"),
           "result": (t.get("result") or {}).get("kind"), "verdict": (gate or {}).get("verdict"),
           "commit": (gate or {}).get("commit"), **((gate or {}).get("metrics") or {})}
    p = ctx.root / LOOP / "results.jsonl"
    with open(p, "a", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
    write_results_md(ctx)


def write_results_md(ctx):
    rows = []
    p = ctx.root / LOOP / "results.jsonl"
    if p.exists():
        for line in p.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rows.append(json.loads(line))
    hdr = ("| Tarea | Etiqueta | Resultado | Hot medio (s) | Warm medio (s) | Cold medio (s) | Reinicio (s) "
           "| RTF hot | Talker pasos/s | ufdb + cold/warm | SNR mín. vs ant. (dB) | Δ sim. locutor | Δ WER "
           "| Desv. duración | Speedup |")
    out = ["# Resultados del banco", "",
           "Tabla generada por loopctl a partir de `results.jsonl` (no editar a mano).", "",
           "Hot = misma frase en el mismo proceso (régimen estable); warm = proceso nuevo con la Find-Db ya poblada "
           "(tras un reinicio); cold = primera vez. Reinicio = warm − hot por frase. Speedup en hot (en warm si "
           "falta la pasada hot, marcado con *).", "", hdr,
           "|" + "---|" * 15]
    for r in rows:
        uf = f"{fmt(r.get('ufdb_lines_added_cold'))}/{fmt(r.get('ufdb_lines_added_warm'))}"
        sp = r.get("speedup")
        star = "*" if isinstance(sp, (int, float)) and r.get("speedup_pass") not in (None, "hot") else ""
        out.append(
            f"| {r['task']} | {r.get('tag') or '—'} | {r.get('result')} | {fmt(r.get('hot_mean_s'))} "
            f"| {fmt(r.get('warm_mean_s'))} | {fmt(r.get('cold_mean_s'))} | {fmt(r.get('restart_overhead_s_mean'))} "
            f"| {fmt(r.get('rtf_hot_mean'))} | {fmt(r.get('talker_steps_per_s_mean'))} "
            f"| {uf} | {fmt(r.get('snr_prev_min'), 1)} | {fmt(r.get('spk_delta_anchor'), 4)} "
            f"| {fmt(r.get('wer_delta_anchor'), 3)} | {fmt(r.get('dur_dev_anchor'), 3)} "
            f"| {('%+.1f %%' % (sp * 100) + star) if isinstance(sp, (int, float)) else '—'} |"
        )
    (ctx.root / LOOP / "RESULTS.md").write_text("\n".join(out) + "\n", encoding="utf-8", newline="\n")


def freeze(ctx, t):
    L = ctx.loop
    tag = t["last_tag"]
    bd = ctx.bench_dir(tag)
    missing = [p for p in ctx.state.get("freeze_paths", [])
               if p != TESTS_BASELINE_REL and not p.endswith("/") and not (ctx.root / p).exists()]
    if missing:
        raise LoopError(f"No se puede congelar la línea base: faltan {missing}")
    bl = read_json(ctx.root / TESTS_BASELINE_REL)
    if bl is None or bl.get("stale"):
        record_tests_baseline(ctx, tag)
    L["frozen"] = {p: sha256_file(ctx.root / p) for p in expand_freeze(ctx)}
    L["frozen_at"] = now()
    L["venv_pkg_hash"] = venv_pkg_hash(ctx.root)
    L["anchor"] = L["quality_ref"] = L["perf_ref"] = tag
    L["anchor_wavs"] = {f.relative_to(ctx.root).as_posix(): sha256_file(f) for f in sorted((bd / "warm").glob("*.wav"))}
    a = read_json(bd / "bench.json")
    b = read_json(ctx.bench_dir(f"{tag}-aa") / "bench.json")
    noise = {}
    for pname in ("warm", "hot"):
        wa, wb = phrase_times(a, pname), phrase_times(b, pname)
        common = sorted(set(wa) & set(wb))
        if common:
            sa, sb = sum(wa[i] for i in common), sum(wb[i] for i in common)
            noise[f"{pname}_rel_diff"] = abs(sa - sb) / sa if sa else None
            noise[f"{pname}_rel_diff_max_phrase"] = max(
                (abs(wa[i] - wb[i]) / wa[i] for i in common if wa[i]), default=None)
    aa = read_json(bd / "compare-aa.json") or {}
    noise["aa_snr_db_min"] = dig(aa, "summary.snr_db_min")
    noise["aa_spk_delta"] = dig(aa, "summary.spk_cos_delta_mean")
    noise["aa_wer_delta"] = dig(aa, "summary.wer_delta")
    L["noise"] = noise
    rep = [f"# Línea base congelada ({tag})", "", f"Congelada el {now()}.", "",
           "## Ruido A/A (dos pasadas del mismo código)", ""]
    for k, v in noise.items():
        rep.append(f"- {k}: {fmt(v, 4)}")
    if noise.get("aa_snr_db_min") is not None and noise["aa_snr_db_min"] < 50:
        rep += ["", "**Atención**: el mismo código no reproduce la onda a > 50 dB. Las puertas de tipo 1 "
                "(SNR > 50 dB) no son fiables hasta recalibrarlas; revisa umbrales en backlog.json."]
    rep += ["", "## Archivos congelados", ""] + [f"- `{p}`" for p in L["frozen"]]
    (ctx.root / LOOP / "baseline-report.md").write_text("\n".join(rep) + "\n", encoding="utf-8", newline="\n")
    progress(ctx, f"Línea base {tag} congelada ({len(L['frozen'])} archivos)")


def record_tests_baseline(ctx, tag):
    tj = read_json(ctx.bench_dir(tag) / "tests.json", {}) or {}
    write_json(ctx.root / TESTS_BASELINE_REL,
               {"recorded_at": now(), "tag": tag, "rc": tj.get("rc"), "failed": sorted(tj.get("failed", [])),
                "passed": tj.get("passed"), "passed_ids": sorted(tj.get("passed_ids", []))})


def close_task(ctx, t, result, reason, gate, by_human):
    L = ctx.loop
    status = {"pass": "passed", "noop": "passed", "reject": "rejected", "skip": "skipped"}[result]
    d = dirty_code(ctx.root)
    if d:
        raise LoopError("Hay cambios de código sin commit; haz commit o descártalos antes de cerrar: "
                        + ", ".join(d[:10]))
    if result == "pass" and (t.get("gate") or {}).get("tests") == "record" and t.get("last_tag"):
        record_tests_baseline(ctx, t["last_tag"])
    if t["kind"] == "baseline" and result == "pass":
        freeze(ctx, t)
    elif result == "pass" and gate and gate.get("update_refs"):
        L["quality_ref"] = t["last_tag"]
        L["perf_ref"] = t["last_tag"]
    t["status"] = status
    t["result"] = {"kind": result, "reason": reason, "at": now(), "tag": t.get("last_tag"),
                   "verdict": (gate or {}).get("verdict"), "by_human": by_human}
    t.pop("human", None)
    record_result(ctx, t, gate)
    if L.get("current") == t["id"]:
        L["current"] = None
    clear_marker(ctx)
    progress(ctx, f"{t['id']} cerrada: {result}" + (f" — {reason}" if reason else ""))
    ctx.save()
    pb = ctx.phase_branch(t["phase"])
    msg = f"loop({t['id']}): {result}"
    on_task_branch = t.get("branch") and current_branch(ctx.root) == t["branch"]
    if on_task_branch and result in ("pass", "noop"):
        commit_state(ctx, msg)
        git(ctx.root, "switch", "-q", pb)
        git(ctx.root, "merge", "-q", "--ff-only", t["branch"])
    elif on_task_branch:
        switch_carrying_state(ctx, pb, msg + f" (rama {t['branch']} conservada)")
    else:
        commit_state(ctx, msg)
    if result in ("reject", "skip") and t.get("reopened"):
        restore_previous_pass(ctx, t, f"{result}: {reason}")


def restore_previous_pass(ctx, t, why):
    """Una tarea reabierta cuyo nuevo intento se rechaza u omite conserva la versión anterior, que ya está en la
    rama de fase: vuelve a «passed» con su resultado de entonces, para que las dependencias y export cuadren."""
    r = t["reopened"][-1]
    prev = r.get("prev_result") or {}
    t["status"] = "passed"
    t["result"] = {**prev, "restored": True, "restored_at": now(), "restored_why": why}
    if r.get("prev_tag"):
        t["last_tag"] = r["prev_tag"]
    ctx.save()
    progress(ctx, f"{t['id']}: el nuevo intento no sigue adelante ({why}); se conserva la versión anterior "
                  f"({r.get('prev_tag') or '—'}), que ya estaba en la rama de fase")
    commit_state(ctx, f"loop({t['id']}): se conserva la versión anterior")


# ---------------------------------------------------------------------------
# A/B ciego
# ---------------------------------------------------------------------------

AB_HTML = """<!doctype html>
<html lang="es"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Escucha ciega __TAG__</title>
<style>
:root{color-scheme:light dark;--bg:#f6f5f2;--fg:#1d1d1b;--muted:#6b6a66;--card:#fff;--line:#dedcd6;--acc:#2f5d50}
@media (prefers-color-scheme:dark){:root{--bg:#151514;--fg:#ecebe7;--muted:#a3a29c;--card:#1f1f1d;--line:#34332f;--acc:#7fb8a6}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);font:16px/1.5 system-ui,-apple-system,Segoe UI,sans-serif}
main{max-width:760px;margin:0 auto;padding:32px 16px 64px}h1{font-size:1.5rem;margin:0 0 4px}p{color:var(--muted)}
.pair{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:16px;margin:16px 0}
.pair h2{font-size:1rem;margin:0 0 4px}.text{font-style:italic;color:var(--muted);margin:0 0 12px}
.row{display:flex;gap:12px;align-items:center;flex-wrap:wrap;margin:6px 0}.row b{width:1.5em}
audio{flex:1;min-width:0;width:100%}.choice{display:flex;gap:20px;margin-top:10px;flex-wrap:wrap}
label{cursor:pointer;display:inline-flex;align-items:center;gap:6px}.text{overflow-wrap:anywhere}
button{background:var(--acc);color:var(--bg);border:0;border-radius:8px;padding:10px 16px;font:inherit;cursor:pointer}
code,#out{font:15px ui-monospace,Consolas,monospace}#out{width:100%;padding:10px;border:1px solid var(--line);border-radius:8px;background:var(--card);color:var(--fg)}
input[type=radio]{width:18px;height:18px;margin:0;accent-color:var(--acc)}
</style></head><body><main>
<h1>Escucha ciega · tarea __TASK__</h1>
<p>__N__ pares. En cada par, una versión es la línea base y la otra el cambio; el orden es aleatorio.
Usa auriculares, escucha las dos completas y elige la que suene mejor (o «Igual»). No abras key.json.</p>
__PAIRS__
<p><button id="go">Generar comando</button></p>
<p><input id="out" readonly placeholder="Responde todos los pares y pulsa el botón"></p>
<p>Copia el comando y pégalo en Claude Code.</p>
</main><script>
document.getElementById('go').onclick=function(){
 var n=__N__,s='';for(var i=1;i<=n;i++){var r=document.querySelector('input[name=p'+i+']:checked');
 if(!r){alert('Falta el par '+i);return;}s+=r.value;}
 var o=document.getElementById('out');o.value='/tts-loop ab __TASK__ '+s;o.select();
 try{navigator.clipboard.writeText(o.value);}catch(e){}
};
</script></body></html>
"""


def build_ab(ctx, t):
    tag = t["last_tag"]
    basis = ctx.loop.get("anchor")
    if not basis:
        raise LoopError("No hay línea base con la que comparar.")
    n = int(ctx.config.get("ab", {}).get("pairs", 6))
    cand_dir, base_dir = ctx.bench_dir(tag) / "warm", ctx.bench_dir(basis) / "warm"
    ids = sorted({f.stem for f in cand_dir.glob("*.wav")} & {f.stem for f in base_dir.glob("*.wav")})
    if not ids:
        raise LoopError("No hay WAV comunes entre el candidato y la línea base.")
    n = min(n, len(ids))
    chosen = [ids[int(i * len(ids) / n)] for i in range(n)]
    texts = {}
    ph = read_json(ctx.root / "scripts/bench_phrases.json", {}) or {}
    for x in ph.get("phrases", []):
        texts[x.get("id")] = x.get("text", "")
    rng = secrets.SystemRandom()
    pairs, blocks = [], []
    audio_dir = ctx.bench_dir(tag) / "ab" / "audio"
    for i, pid in enumerate(chosen, 1):
        side = rng.choice("AB")
        pairs.append({"n": i, "id": pid, "cand_side": side})
        names = {"cand": f"{secrets.token_hex(6)}.wav", "base": f"{secrets.token_hex(6)}.wav"}
        audio_dir.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(cand_dir / f"{pid}.wav", audio_dir / names["cand"])
        shutil.copyfile(base_dir / f"{pid}.wav", audio_dir / names["base"])
        pairs[-1]["files"] = names
        cand_src, base_src = f"audio/{names['cand']}", f"audio/{names['base']}"
        a_src, b_src = (cand_src, base_src) if side == "A" else (base_src, cand_src)
        blocks.append(
            f'<section class="pair"><h2>Par {i}</h2><p class="text">{html.escape(texts.get(pid, ""))}</p>'
            f'<div class="row"><b>A</b><audio controls preload="none" src="{a_src}"></audio></div>'
            f'<div class="row"><b>B</b><audio controls preload="none" src="{b_src}"></audio></div>'
            f'<div class="choice"><label><input type="radio" name="p{i}" value="A"> A</label>'
            f'<label><input type="radio" name="p{i}" value="B"> B</label>'
            f'<label><input type="radio" name="p{i}" value="="> Igual</label></div></section>'
        )
    ab = ctx.bench_dir(tag) / "ab"
    ab.mkdir(parents=True, exist_ok=True)
    write_json(ab / "key.json", {"tag": tag, "basis": basis, "pairs": pairs, "created_at": now()})
    page = (AB_HTML.replace("__TAG__", html.escape(tag)).replace("__TASK__", html.escape(t["id"]))
            .replace("__N__", str(n)).replace("__PAIRS__", "\n".join(blocks)))
    (ab / "index.html").write_text(page, encoding="utf-8")
    return str(ab / "index.html")


# ---------------------------------------------------------------------------
# Decisiones humanas
# ---------------------------------------------------------------------------

def require_human(ctx, cmd, tid, a, expect_args=None):
    if getattr(a, "i_am_human", False) or os.environ.get("TTS_LOOP_HUMAN") == "1":
        return "terminal"
    tp = ctx.root / TOKEN_REL
    tok = read_json(tp)
    msg = (f"«{cmd} {tid}» es una decisión humana. Escríbela tú en Claude Code (`/tts-loop {cmd} {tid} …`) "
           f"o ejecútala desde tu terminal con --i-am-human.")
    if not tok:
        raise LoopError(msg)
    age = time.time() - float(tok.get("ts", 0))
    if tok.get("cmd") != cmd or tok.get("id") != tid or age > 1800:
        raise LoopError(msg + " (no hay una orden tuya reciente para esto)")
    if expect_args is not None and expect_args.strip().upper() not in (tok.get("args") or "").upper():
        raise LoopError(msg + " (el código no coincide con el que escribiste)")
    tp.unlink()
    return "prompt"


# ---------------------------------------------------------------------------
# Comandos
# ---------------------------------------------------------------------------

def cmd_status(a):
    ctx = Ctx(dry=True)
    L = ctx.loop
    sel = select(ctx)
    if a.json:
        print(json.dumps({"loop": {k: L.get(k) for k in ("phase", "current", "anchor", "quality_ref", "perf_ref",
                                                          "frozen_at", "done")},
                          "next": sel, "stop": sel.get("stop"), "stop_reason": sel.get("reason"),
                          "tasks": [{k: t.get(k) for k in ("id", "phase", "kind", "title", "status", "attempts",
                                                           "last_tag", "branch")} for t in ctx.tasks]},
                         ensure_ascii=False, indent=2))
        return 0
    try:
        br = current_branch(ctx.root)
    except LoopError:
        br = "?"
    print(f"Loop tts · rama {br} · fase {L['phase']} ({ctx.phase_title(L['phase'])})")
    print(f"Refs: línea base {L.get('anchor') or '—'} · calidad {L.get('quality_ref') or '—'} · "
          f"rendimiento {L.get('perf_ref') or '—'} · arnés congelado: {'sí' if L.get('frozen') else 'no'}")
    print()
    print(f"{'id':6} {'estado':15} {'int':>3}  {'último':10} título")
    for t in ctx.tasks:
        gate = read_json(ctx.bench_dir(t["last_tag"]) / "gate.json") if t.get("last_tag") else None
        v = (gate or {}).get("verdict", "")
        print(f"{t['id']:6} {t['status']:15} {t.get('attempts', 0):>3}  {v:10} {t['title']}")
    print()
    if sel.get("stop"):
        print(f"PARADA ({sel['reason']}): {sel['message']}")
    else:
        print(f"Siguiente: {sel['task']} · acción {sel['action']}")
    return 0


def cmd_brief(a):
    try:
        root = find_root()
        if not (root / STATE_REL).exists():
            if (SKILL_DIR / "scaffold").is_dir():
                print("tts-loop: aquí no hay loop todavía (repo principal). Usa `/tts-loop init` para crear el worktree.")
            else:
                print("tts-loop: no hay loop en este directorio.")
            return 0
        ctx = Ctx(root, dry=True)
        L = ctx.loop
        try:
            br = current_branch(root)
        except LoopError:
            br = "?"
        sel = select(ctx)
        lines = [f"[tts-loop] worktree {root} · rama {br} · fase {L['phase']} ({ctx.phase_title(L['phase'])})"]
        cur = L.get("current")
        if cur:
            t = ctx.task(cur)
            st = bench_status(ctx, t.get("last_tag"))
            bs = f" · verificación {t['last_tag']}: {st.get('state')}" if st else ""
            lines.append(f"Tarea actual: {cur} «{t['title']}» · {t['status']} · intento {t.get('attempts', 0)}"
                         f"/{ctx.max_attempts(t) + t.get('extra_attempts', 0)}{bs}")
        if sel.get("stop"):
            lines.append(f"Parada: {sel['reason']} — {sel['message']}")
        else:
            lines.append(f"Siguiente acción: {sel['action']} ({sel['task']})")
        lines.append(f"Refs: base {L.get('anchor') or '—'} · calidad {L.get('quality_ref') or '—'} · "
                     f"rendimiento {L.get('perf_ref') or '—'} · arnés congelado: {'sí' if L.get('frozen') else 'no'}")
        prog = tail(root / LOOP / "progress.md", 6)
        if prog:
            lines.append("Últimas entradas de progress.md:\n" + prog)
        print("\n".join(lines))
    except Exception as e:
        print(f"tts-loop: no pude leer el estado ({e})")
    return 0


def cmd_next(a):
    ctx = Ctx()
    sel = select(ctx)
    if a.json:
        print(json.dumps(sel, ensure_ascii=False, indent=2))
    elif sel.get("stop"):
        print(f"PARADA ({sel['reason']}): {sel['message']}")
    else:
        print(f"Siguiente: {sel['task']} «{sel['title']}» · acción: {sel['action']}")
    return 0


def cmd_begin(a):
    ctx = Ctx()
    sel = select(ctx)
    if sel.get("stop"):
        raise LoopError(sel["message"])
    if sel["task"] != a.id:
        raise LoopError(f"La siguiente tarea elegible es {sel['task']}, no {a.id}.")
    if sel["action"] != "begin":
        raise LoopError(f"{a.id} ya está en curso (acción pendiente: {sel['action']}).")
    d = dirty_code(ctx.root)
    if d:
        raise LoopError("Hay cambios sin commit fuera de perf-loop/: " + ", ".join(d[:10]))
    t = ctx.task(a.id)
    L = ctx.loop
    ph = t["phase"]
    pb = f"{ctx.prefix()}-f{ph}" if str(ph) not in L.get("phase_branches", {}) else L["phase_branches"][str(ph)]
    if current_branch(ctx.root) != pb:
        if not branch_exists(ctx.root, pb):
            known = sorted((int(k), v) for k, v in L.get("phase_branches", {}).items()
                           if int(k) < ph and branch_exists(ctx.root, v))
            if not known:
                raise LoopError(f"No encuentro ninguna rama de fase anterior a la {ph}.")
            prev = known[-1][1]
            ctx.save()
            commit_state(ctx, f"loop: cierre de la fase {ph - 1}")
            git(ctx.root, "branch", pb, prev)
            progress(ctx, f"Rama de fase {pb} creada desde {prev}")
        switch_carrying_state(ctx, pb, f"loop: estado antes de {a.id}")
    L.setdefault("phase_branches", {})[str(ph)] = pb
    tb, n = ctx.task_branch(t["id"]), 2
    while branch_exists(ctx.root, tb):
        tb = f"{ctx.task_branch(t['id'])}-r{n}"
        n += 1
    git(ctx.root, "switch", "-q", "-c", tb)
    t.update(status="in_progress", branch=tb, base=head_sha(ctx.root), started_at=now())
    t.setdefault("attempts", 0)
    L["current"] = t["id"]
    ctx.save()
    set_marker(ctx, t["id"])
    progress(ctx, f"{t['id']} empieza en {tb} (base {t['base'][:8]})")
    nxt = "implement" if t.get("implementer", True) else ("await" if t["kind"] == "manual" else "verify")
    print(json.dumps({"task": t["id"], "branch": tb, "base": t["base"], "next_action": nxt},
                     ensure_ascii=False, indent=2))
    return 0


def current_task(ctx, tid):
    t = ctx.task(tid)
    if ctx.loop.get("current") != tid or t["status"] != "in_progress":
        raise LoopError(f"{tid} no es la tarea en curso (en curso: {ctx.loop.get('current') or 'ninguna'}; "
                        f"{tid} está {t['status']}).")
    return t


def running_workers(ctx):
    out = []
    for t in ctx.tasks:
        st = bench_status(ctx, t.get("last_tag"))
        if st and st.get("state") in ("starting", "running"):
            out.append(t["last_tag"])
    return out


def cmd_verify(a):
    ctx = Ctx()
    t = current_task(ctx, a.id)
    if t["kind"] == "manual" and not t.get("manual_done"):
        raise LoopError("Tarea manual: primero `loopctl await` y la aprobación del usuario.")
    busy = running_workers(ctx)
    if busy:
        raise LoopError(f"Ya hay una verificación en curso ({', '.join(busy)}). Usa `loopctl wait {a.id}`.")
    d = dirty_code(ctx.root)
    if d:
        raise LoopError("Haz commit del código antes de medir. Sin commit: " + ", ".join(d[:10]))
    if current_branch(ctx.root) != t["branch"]:
        raise LoopError(f"Estás en {current_branch(ctx.root)}; la tarea vive en {t['branch']}.")
    mism = frozen_mismatches(ctx)
    if mism:
        raise LoopError("El arnés de evaluación congelado cambió: " + ", ".join(mism))
    limit = ctx.max_attempts(t) + t.get("extra_attempts", 0)
    if t.get("attempts", 0) >= limit:
        raise LoopError(f"Intentos agotados ({t['attempts']}/{limit}). Cierra con `finish {a.id} reject` "
                        f"(experimento o medida) o `block {a.id}` (obligatoria).")
    loose = untracked_outside(ctx.root)
    if loose:
        raise LoopError("Hay archivos sin seguimiento fuera de perf-loop/ (se medirían pero no llegarían a la rama): "
                        + ", ".join(loose[:10]) + ". Haz commit de los que formen parte del cambio y borra el resto.")
    t["attempts"] = t.get("attempts", 0) + 1
    tag = f"{t['id']}-a{t['attempts']}"
    bd = ctx.bench_dir(tag)
    k = 2
    while bd.exists():
        tag = f"{t['id']}-a{t['attempts']}.{k}"
        bd = ctx.bench_dir(tag)
        k += 1
    bd.mkdir(parents=True)
    t["last_tag"] = tag
    t.setdefault("tags", []).append(tag)
    ctx.save()
    progress(ctx, f"{t['id']}: verificación {tag} lanzada")
    commit_state(ctx, f"loop({t['id']}): verificación {tag}")  # el banco mide un árbol limpio
    set_marker(ctx, t["id"])
    status = {"tag": tag, "task": t["id"], "state": "starting", "commit": head_sha(ctx.root),
              "branch": t["branch"], "started_at": now(), "steps": [], "heartbeat_ts": time.time()}
    write_json(bd / "status.json", status)
    if a.foreground:
        run_worker(ctx.root, t["id"], tag)
        print(render_gate(read_json(bd / "gate.json")))
        return 0
    spawn_detached([sys.executable, str(Path(__file__).resolve()), "--root", str(ctx.root), "_worker", t["id"], tag],
                   ctx.root, bd / "worker.log")
    print(json.dumps({"tag": tag, "started": True, "bench_dir": relposix(ctx.root, bd),
                      "next": f"loopctl wait {t['id']}  (Bash con timeout 600000)"}, ensure_ascii=False, indent=2))
    return 0


def cmd_worker(a):
    run_worker(find_root(), a.id, a.tag)
    return 0


def cmd_wait(a):
    ctx = Ctx()
    t = ctx.task(a.id)
    tag = a.tag or t.get("last_tag")
    if not tag:
        raise LoopError("No hay ninguna verificación lanzada para esta tarea.")
    deadline = time.time() + a.max
    while True:
        st = bench_status(ctx, tag)
        if not st:
            raise LoopError(f"No encuentro {tag}/status.json")
        s = st.get("state")
        if s == "done":
            print(render_gate(read_json(ctx.bench_dir(tag) / "gate.json")))
            return 0
        if s in ("error", "crashed"):
            print(f"ESTADO: {s} · {tag}")
            print(st.get("error") or "el proceso del banco desapareció sin terminar")
            print(tail(ctx.bench_dir(tag) / "worker.log", 15))
            print("Puedes relanzar con `loopctl verify <id>` (gasta un intento) o `--foreground`.")
            return 1
        if time.time() >= deadline:
            step = st.get("step", "?")
            el = time.time() - float(st.get("step_started_ts") or time.time())
            done = ", ".join(f"{x['name']}({x.get('rc', 'omitido')})" for x in st.get("steps", []))
            print(f"ESTADO: running · {tag} · paso {step} ({el / 60:.1f} min) · hechos: {done or 'ninguno'}")
            log = ctx.bench_dir(tag) / "logs" / f"{step}.log"
            lt = tail(log, 6)
            if lt:
                print(lt)
            print(f"Sigue en marcha: vuelve a ejecutar `loopctl wait {a.id}`.")
            return 0
        time.sleep(max(1.0, min(15.0, deadline - time.time())))


def cmd_gate(a):
    ctx = Ctx()
    t = ctx.task(a.id)
    tag = a.tag or t.get("last_tag")
    st = read_json(ctx.bench_dir(tag) / "status.json") if tag else None
    if not st or st.get("state") != "done":
        raise LoopError("No hay una verificación terminada que reevaluar.")
    gate = evaluate_gate(ctx, t, tag, st)
    write_json(ctx.bench_dir(tag) / "gate.json", gate)
    print(render_gate(gate))
    return 0


def last_gate(ctx, t):
    return read_json(ctx.bench_dir(t["last_tag"]) / "gate.json") if t.get("last_tag") else None


def cmd_finish(a):
    ctx = Ctx()
    t = current_task(ctx, a.id)
    gate = last_gate(ctx, t)
    if a.result == "pass":
        if gate is None:
            raise LoopError("No hay puerta evaluada: ejecuta `loopctl verify` primero.")
        if gate["verdict"] == "needs_human":
            raise LoopError("La puerta pide una decisión humana: ejecuta `loopctl await`.")
        if gate["verdict"] != "pass":
            raise LoopError(f"La puerta dio '{gate['verdict']}': no se puede cerrar como pass.")
        changed = code_changed_since(ctx.root, gate.get("commit"))
        if changed:
            raise LoopError("Hay cambios de código posteriores a la medición: vuelve a verificar. " + ", ".join(changed[:5]))
    elif a.result == "noop":
        if not t.get("allow_noop"):
            raise LoopError(f"{t['id']} no admite noop: debe pasar su puerta (`loopctl verify`).")
        changed = code_changed_since(ctx.root, t["base"])
        if changed:
            raise LoopError("noop exige que no haya cambios de código: " + ", ".join(changed[:5]))
        if not a.reason:
            raise LoopError("noop necesita --reason con la evidencia")
        gate = None
    elif a.result == "reject":
        if (t.get("gate") or {}).get("tests") == "record":
            raise LoopError("La lista de tests de partida es obligatoria para todas las puertas siguientes: si no se "
                            f"puede registrar, bloquea la tarea (`loopctl block {t['id']} --reason …`) para que "
                            "decida el usuario.")
        if not (t["kind"] in ("experiment", "measure") or t.get("optional")):
            raise LoopError("Una tarea obligatoria que no pasa se bloquea (`loopctl block`), no se rechaza.")
        if not a.reason:
            raise LoopError("reject necesita --reason")
    close_task(ctx, t, a.result, a.reason, gate, False)
    print(f"{t['id']} cerrada como {a.result}. Rama actual: {current_branch(ctx.root)}")
    return 0


def cmd_await(a):
    ctx = Ctx()
    t = current_task(ctx, a.id)
    if t["kind"] == "manual" and not t.get("manual_done"):
        h = {"kind": "manual", "message": t.get("human_message", "Sigue la lista de la tarea."), "at": now()}
    else:
        gate = last_gate(ctx, t)
        if not gate or gate["verdict"] != "needs_human":
            raise LoopError("La puerta no pide decisión humana (o no hay puerta).")
        h = {"kind": gate["human"], "at": now(), "tag": gate["tag"]}
        if gate["human"] == "ab":
            h["page"] = relposix(ctx.root, build_ab(ctx, t))
        else:
            m = {k: v for k, v in (gate.get("metrics") or {}).items() if v is not None}
            h["message"] = ("Métricas: " + ", ".join(f"{k}={fmt(v, 3)}" for k, v in m.items())
                            + f". Revisa la rama {t['branch']}, perf-loop/RESULTS.md"
                            + (" y perf-loop/bench/" + gate["tag"] + "/" if gate.get("tag") else "") + ".")
            if t["kind"] == "baseline":
                h["message"] += (" Al aprobar se congela el arnés (scripts/bench_*, compare_wavs, frases, tests) "
                                 "y esta medición pasa a ser la línea base.")
    t["status"] = "awaiting_human"
    t["human"] = h
    clear_marker(ctx)
    ctx.save()
    progress(ctx, f"{t['id']} espera decisión humana ({h['kind']})")
    commit_state(ctx, f"loop({t['id']}): espera decisión humana")
    print(human_message(ctx, t))
    return 0


def cmd_approve(a):
    ctx = Ctx()
    select(ctx, honor_stop=False)  # p. ej. una tarea que pide visto bueno para empezar pasa a awaiting_human
    t = ctx.task(a.id)
    if t["status"] != "awaiting_human":
        raise LoopError(f"{a.id} no espera aprobación (estado {t['status']}).")
    h = t.get("human") or {}
    if h.get("kind") == "ab" and not a.override:
        raise LoopError("Esta puerta es una escucha ciega: usa `/tts-loop ab <id> <código>`.")
    require_human(ctx, "approve", a.id, a)
    k = h.get("kind")
    if k == "start":
        t["start_approved"] = True
        t["status"] = "pending"
        t.pop("human", None)
        ctx.loop["current"] = None
        ctx.save()
        progress(ctx, f"{t['id']}: el usuario aprueba empezar")
        commit_state(ctx, f"loop({t['id']}): aprobado el inicio")
        print(f"{t['id']} aprobada para empezar.")
        return 0
    if k == "manual":
        t["manual_done"] = True
        t["status"] = "in_progress"
        t.pop("human", None)
        if (t.get("gate") or {}).get("bench"):
            ctx.save()
            progress(ctx, f"{t['id']}: pasos manuales hechos; falta medir")
            commit_state(ctx, f"loop({t['id']}): pasos manuales hechos")
            print(f"{t['id']}: pasos manuales registrados. Siguiente: `loopctl verify {t['id']}`.")
            return 0
        close_task(ctx, t, "pass", a.note or "hecho por el usuario", None, True)
        print(f"{t['id']} cerrada.")
        return 0
    t["status"] = "in_progress"
    close_task(ctx, t, "pass", a.note or "aprobado por el usuario", last_gate(ctx, t), True)
    print(f"{t['id']} aprobada. Rama actual: {current_branch(ctx.root)}")
    return 0


def block_task(ctx, t, reason):
    t["status"] = "blocked"
    t["result"] = {"kind": "blocked", "reason": reason, "at": now()}
    t.pop("human", None)
    clear_marker(ctx)
    ctx.save()
    progress(ctx, f"{t['id']} bloqueada: {reason}")
    commit_state(ctx, f"loop({t['id']}): bloqueada")


def cmd_reject(a):
    ctx = Ctx()
    select(ctx, honor_stop=False)
    t = ctx.task(a.id)
    if t["status"] not in ("awaiting_human", "in_progress"):
        raise LoopError(f"{a.id} no está en curso ni esperando decisión.")
    require_human(ctx, "reject", a.id, a)
    kind = (t.get("human") or {}).get("kind")
    t["status"] = "in_progress"
    if kind == "start":
        close_task(ctx, t, "skip", a.reason or "el usuario no aprueba empezar", None, True)
    elif t["kind"] in ("experiment", "measure") or t.get("optional"):
        close_task(ctx, t, "reject", a.reason or "rechazada por el usuario", last_gate(ctx, t), True)
    else:
        block_task(ctx, t, "rechazada por el usuario: " + (a.reason or ""))
    print(f"{t['id']}: {t['status']}")
    return 0


def cmd_ab(a):
    ctx = Ctx()
    t = ctx.task(a.id)
    h = t.get("human") or {}
    if t["status"] != "awaiting_human" or h.get("kind") != "ab":
        raise LoopError(f"{a.id} no tiene una escucha ciega pendiente.")
    page = ctx.root / h["page"]
    if not a.answer:
        print(f"Abre en el navegador: {page}\nDespués: `/tts-loop ab {a.id} <código>`")
        return 0
    require_human(ctx, "ab", a.id, a, expect_args=a.answer)
    key = read_json(page.parent / "key.json")
    ans = a.answer.strip().upper().replace(" ", "")
    if len(ans) != len(key["pairs"]) or any(c not in "AB=" for c in ans):
        raise LoopError(f"El código debe tener {len(key['pairs'])} caracteres A, B o =.")
    base_pref = sum(1 for c, p in zip(ans, key["pairs"], strict=True) if c in "AB" and c != p["cand_side"])
    cand_pref = sum(1 for c, p in zip(ans, key["pairs"], strict=True) if c == p["cand_side"])
    mx = int(ctx.config.get("ab", {}).get("max_base_preferred", 3))
    res = {"answer": ans, "base_preferred": base_pref, "cand_preferred": cand_pref, "ties": ans.count("="),
           "pairs": len(ans), "max_base_preferred": mx, "at": now()}
    write_json(page.parent / "result.json", res)
    gate = last_gate(ctx, t) or {}
    gate["ab"] = res
    write_json(ctx.bench_dir(t["last_tag"]) / "gate.json", gate)
    summary = f"A/B: base preferida {base_pref}/{len(ans)}, cambio {cand_pref}, iguales {res['ties']}"
    t["status"] = "in_progress"
    if base_pref <= mx:
        close_task(ctx, t, "pass", summary, gate, True)
    elif t["kind"] in ("experiment", "measure") or t.get("optional"):
        close_task(ctx, t, "reject", summary + " (se prefiere la base)", gate, True)
    else:
        block_task(ctx, t, summary + " (se prefiere la base)")
    print(f"{summary} → {t['status']}")
    return 0


def cmd_skip(a):
    ctx = Ctx()
    t = ctx.task(a.id)
    if t["status"] in TERMINAL:
        raise LoopError(f"{a.id} ya está cerrada ({t['status']}).")
    require_human(ctx, "skip", a.id, a)
    t["status"] = "in_progress"
    close_task(ctx, t, "skip", a.reason or "omitida por el usuario", None, True)
    print(f"{a.id} omitida." if t["status"] == "skipped" else
          f"{a.id}: se omite el nuevo intento y se conserva la versión anterior (estado {t['status']}).")
    return 0


def cmd_retry(a):
    ctx = Ctx()
    t = ctx.task(a.id)
    require_human(ctx, "retry", a.id, a)
    extra = ctx.max_attempts(t)
    prev = (t.get("result") or {}).get("reason")
    if prev:
        t.setdefault("notes", []).append({"at": now(), "text": f"reintento tras: {prev}"})
    if t["status"] == "blocked":
        t["status"] = "in_progress"
        t["extra_attempts"] = t.get("extra_attempts", 0) + extra
        st = bench_status(ctx, t.get("last_tag")) if t.get("last_tag") else None
        if t.get("last_tag"):
            t.setdefault("consumed_tags", []).append(t["last_tag"])
        env_failure = bool(st and st.get("state") in ("crashed", "error"))
        t["resume_hint"] = "verify" if (env_failure or not t.get("implementer", True)) else "implement"
        t.pop("result", None)
        ctx.loop["current"] = t["id"]
        if t.get("branch") and current_branch(ctx.root) != t["branch"] and branch_exists(ctx.root, t["branch"]):
            switch_carrying_state(ctx, t["branch"], f"loop({t['id']}): reintento")
        ctx.save()
    elif t["status"] in ("rejected", "skipped"):
        if t["phase"] != ctx.loop["phase"]:
            raise LoopError("Esa fase ya está cerrada; no se puede reabrir la tarea.")
        t["status"] = "pending"
        t["extra_attempts"] = t.get("extra_attempts", 0) + extra
        t.pop("result", None)
        ctx.save()
    else:
        raise LoopError(f"retry solo vale para tareas bloqueadas, rechazadas u omitidas (estado {t['status']}).")
    if ctx.loop.pop("done", None):
        ctx.save()
    progress(ctx, f"{t['id']}: el usuario concede {extra} intentos más")
    commit_state(ctx, f"loop({t['id']}): reintento")
    print(f"{t['id']} reabierta con {extra} intentos más.")
    return 0


def cmd_reopen(a):
    """Reabre una tarea ya superada de la fase en curso para rehacerla sobre la punta de la rama de fase.

    Sus commits anteriores siguen en la rama de fase; el nuevo intento parte de ahí. Si hay otra tarea recién
    empezada y sin trabajo (como tras cortar el runner justo en `begin`), vuelve a pendiente y su rama vacía se borra.
    """
    ctx = Ctx()
    L = ctx.loop
    t = ctx.task(a.id)
    if t["status"] != "passed":
        raise LoopError(f"reopen solo vale para tareas superadas ({a.id} está {t['status']}); para bloqueadas, "
                        "rechazadas u omitidas usa retry.")
    if t["phase"] != L["phase"]:
        raise LoopError("Esa fase ya está cerrada; no se puede reabrir la tarea.")
    if t["kind"] == "baseline" or (t["phase"] == 0 and L.get("frozen")):
        raise LoopError("El arnés ya está congelado: reabrir una tarea de la fase 0 cambiaría la línea base.")
    if not (a.reason or "").strip():
        raise LoopError("reopen necesita --reason: el implementador lo leerá como la instrucción del nuevo intento.")
    busy = running_workers(ctx)
    if busy:
        raise LoopError(f"Hay una verificación en curso ({', '.join(busy)}); espera a que termine.")
    d = dirty_code(ctx.root)
    if d:
        raise LoopError("Haz commit de los cambios fuera de perf-loop/ antes de reabrir: " + ", ".join(d[:10]))
    pb = ctx.phase_branch(t["phase"])
    if not branch_exists(ctx.root, pb):
        raise LoopError(f"No encuentro la rama de fase {pb}.")
    reset = None
    cur = L.get("current")
    if cur and cur != t["id"]:
        c = ctx.task(cur)
        cb = c.get("branch")
        # «Recién empezada» = su rama sigue exactamente en la base (begin no hace commits en la rama de la tarea).
        moved = bool(cb and branch_exists(ctx.root, cb) and head_sha(ctx.root, cb) != c.get("base"))
        if c["status"] != "in_progress" or c.get("last_tag") or c.get("attempts") or moved:
            raise LoopError(f"La tarea {cur} está en curso con trabajo hecho (o esperando una decisión); ciérrala, "
                            f"bloquéala u omítela antes de reabrir {t['id']}."
                            + (f" Su rama {cb} tiene commits posteriores a su base." if moved else ""))
        loose = untracked_outside(ctx.root)
        if loose:
            raise LoopError(f"Hay archivos sin seguimiento fuera de perf-loop/ (¿trabajo de la tarea {cur}?): "
                            + ", ".join(loose[:10]) + ". Haz commit de ellos o bórralos antes de reabrir.")
        reset = c
    require_human(ctx, "reopen", a.id, a)

    reason = a.reason.strip()
    t.setdefault("reopened", []).append({"at": now(), "reason": reason, "prev_tag": t.get("last_tag"),
                                         "prev_branch": t.get("branch"), "prev_result": t.get("result")})
    notes = t.setdefault("notes", [])
    notes.append({"at": now(), "text": f"Reabierta por el usuario: {reason}"})
    del notes[:-30]
    t["status"] = "pending"
    t["extra_attempts"] = t.get("attempts", 0)  # límite = intentos ya gastados + los de una tarea nueva
    t["last_tag"] = None  # la medición anterior ya no vale para cerrar este intento
    for k in ("result", "human", "resume_hint", "manual_done", "branch", "base", "started_at"):
        t.pop(k, None)  # rama y base anteriores quedan en reopened[-1]
    L.pop("done", None)
    stale_tests = (t.get("gate") or {}).get("tests") == "record"
    if stale_tests:
        bp = ctx.root / TESTS_BASELINE_REL
        old = read_json(bp, {}) or {}
        write_json(bp, {**old, "stale": True, "stale_reason": f"{t['id']} reabierta: {reason}", "stale_at": now()})
    dropped = None
    if reset:
        dropped = reset.get("branch")
        for k in ("branch", "base", "started_at", "resume_hint", "manual_done"):
            reset.pop(k, None)
        reset["status"] = "pending"
        L["current"] = None
        clear_marker(ctx)
    msg = f"loop({t['id']}): reabierta por el usuario"
    if current_branch(ctx.root) != pb:
        switch_carrying_state(ctx, pb, msg)
    if dropped and branch_exists(ctx.root, dropped) and current_branch(ctx.root) != dropped:
        git(ctx.root, "branch", "-D", dropped)
    later = [x["id"] for x in ctx.tasks if x["phase"] == t["phase"] and x["status"] == "passed"
             and ctx.tasks.index(x) > ctx.tasks.index(t)]
    ctx.save()
    progress(ctx, f"{t['id']} reabierta por el usuario: {reason}"
             + (f"; {reset['id']} vuelve a pendiente (rama vacía {dropped} borrada)" if reset else ""))
    commit_state(ctx, msg)
    out = [f"{t['id']} reabierta: vuelve a pendiente con {ctx.max_attempts(t)} intentos más y partirá de la punta "
           f"de {pb}."]
    if stale_tests:
        out.append("La lista de tests de partida queda pendiente: se vuelve a registrar al cerrar esta tarea.")
    if reset:
        out.append(f"{reset['id']} vuelve a pendiente" + (f" y su rama vacía {dropped} se ha borrado." if dropped else "."))
    if later:
        out.append(f"Tareas posteriores de la fase ya cerradas que no se reabren: {', '.join(later)}.")
    print(" ".join(out))
    return 0


def cmd_refreeze(a):
    ctx = Ctx()
    require_human(ctx, "refreeze", "harness", a)
    if not ctx.loop.get("frozen"):
        raise LoopError("Todavía no hay arnés congelado.")
    before = frozen_mismatches(ctx)
    ctx.loop["frozen"] = {p: sha256_file(ctx.root / p) for p in expand_freeze(ctx)}
    ctx.save()
    progress(ctx, f"Arnés recongelado por el usuario ({a.reason or 'sin motivo'}); cambiados: {', '.join(before) or 'ninguno'}")
    commit_state(ctx, "loop: arnés recongelado")
    print("Arnés recongelado.")
    return 0


def cmd_block(a):
    ctx = Ctx()
    t = current_task(ctx, a.id)
    block_task(ctx, t, a.reason)
    print(f"{t['id']} bloqueada. El loop se detiene hasta que decidas (`retry` o `skip`).")
    return 0


def cmd_pause(a):
    ctx = Ctx()
    clear_marker(ctx)
    progress(ctx, "Loop en pausa a petición del usuario")
    print("Pausa registrada; la tarea en curso se retomará en la próxima iteración.")
    return 0


def cmd_note(a):
    ctx = Ctx()
    text = " ".join(a.text).strip()
    if not text:
        raise LoopError("Nota vacía.")
    if a.id != "-":
        t = ctx.task(a.id)
        notes = t.setdefault("notes", [])
        notes.append({"at": now(), "text": text[:2000]})
        del notes[:-30]
        ctx.save()
    progress(ctx, f"nota {a.id}: {text[:500]}")
    print("Nota registrada.")
    return 0


def render_checks(ctx, t):
    out = []
    g = t.get("gate") or {}
    defs = (quality_def(ctx, t).get("checks", []) if g.get("bench") else []) + g.get("checks", [])
    for c in defs:
        out.append(f"- {c.get('label') or c['metric']}  (`{c['metric']} {c['op']} {c['value']}`)")
    if g.get("tests"):
        out.append("- Sin tests nuevos fallando respecto a la línea base")
    if g.get("validate"):
        out.append(f"- Validaciones: {', '.join(g['validate'])}")
    if g.get("selftest"):
        out.append(f"- Autotests: {g['selftest']}")
    h = g.get("human") or quality_def(ctx, t).get("human")
    if h:
        out.append(f"- Decisión humana: {h}")
    return out


def cmd_show(a):
    ctx = Ctx()
    t = ctx.task(a.id)
    L = ctx.loop
    qt = t.get("quality_type")
    lines = [f"# Tarea {t['id']}: {t['title']}", "",
             f"Fase {t['phase']} ({ctx.phase_title(t['phase'])}) · tipo `{t['kind']}` · calidad tipo {qt if qt is not None else '—'}"
             f" · intento {t.get('attempts', 0) + 1}/{ctx.max_attempts(t) + t.get('extra_attempts', 0)}"
             f" · estado {t['status']}"]
    if t.get("branch"):
        lines.append(f"Rama `{t['branch']}` · base `{str(t.get('base', ''))[:10]}` (punta de `{ctx.phase_branch(t['phase'])}`)")
    if t.get("reopened") and t["status"] != "passed":
        r = t["reopened"][-1]
        lines += ["", "## Reabierta por el usuario (prioridad de este intento)", "",
                  f"{r['reason']}",
                  "",
                  f"El trabajo anterior ({r.get('prev_tag') or '—'}, rama `{r.get('prev_branch') or '—'}`) ya está en la "
                  "rama de fase y este intento parte de ahí: cambia solo lo que pide el motivo; el resto de la ficha "
                  "describe la tarea completa."]
    lines += ["", "## Qué hacer", ""] + [str(x) for x in t.get("spec", [])]
    if t.get("files_hint"):
        lines += ["", "## Archivos de partida", ""] + [f"- `{f}`" for f in t["files_hint"]]
    if t.get("plan_first"):
        lines += ["", f"**Plan primero**: antes de tocar código escribe `perf-loop/notes/{t['id']}.md` con el enfoque, "
                      "los puntos exactos del código y cómo verificarás la equivalencia; después implementa."]
    lines += ["", "## Aceptación (la comprueba loopctl verify; no la evalúes tú)", ""]
    lines += [f"- {x}" for x in t.get("acceptance", [])] + render_checks(ctx, t)
    cons = ctx.state.get("constraints", []) + t.get("constraints", [])
    if cons:
        lines += ["", "## Restricciones (no negociables)", ""] + [f"- {c}" for c in cons]
    if t.get("tags"):
        lines += ["", "## Intentos anteriores", ""]
        for tag in t["tags"]:
            g = read_json(ctx.bench_dir(tag) / "gate.json")
            if g:
                lines.append(f"- {tag}: {g['verdict']}")
                lines += [f"  - ✗ {f}" for f in g.get("failures", [])[:8]]
            else:
                st = bench_status(ctx, tag) or {}
                why = st.get("error") or ("el proceso del banco murió sin terminar" if st.get("state") == "crashed" else "")
                lines.append(f"- {tag}: {st.get('state', '?')} {why[:300]}")
    if t.get("notes"):
        lines += ["", "## Notas", ""] + [f"- {n['at']}: {n['text']}" for n in t["notes"][-10:]]
    lines += ["", "## Contexto", "",
              "- Plan completo: `perf-loop/plan.md`; informe de causas: `perf-loop/informe.md`.",
              "- Contratos del banco, comparador y log VBPERF: `perf-loop/contracts.md`.",
              f"- Referencias actuales: línea base `{L.get('anchor') or '—'}`, calidad `{L.get('quality_ref') or '—'}`, "
              f"rendimiento `{L.get('perf_ref') or '—'}` (WAV y bench.json en `perf-loop/bench/<etiqueta>/`)."]
    if a.for_ == "implementer":
        lines += ["", "## Al terminar", "",
                  f"- Commit en `{t.get('branch')}` con rutas explícitas: `git add <rutas> && git commit -m \"perf({t['id']}): …\"`.",
                  "- No ejecutes el banco completo ni cambies de rama: eso lo hace el orquestador con loopctl.",
                  "- Devuelve el informe JSON que pide tu definición de agente."]
    if a.for_ == "reviewer" and t.get("base"):
        stat = git(ctx.root, "diff", "--stat", t["base"], "HEAD", "--", ".", f":(exclude){LOOP}", check=False)
        mod_tests = git(ctx.root, "diff", "--name-only", "--diff-filter=MD", t["base"], "HEAD", "--",
                        "backend/tests", check=False)
        lines += ["", "## Diff a revisar", "", f"`git diff {t['base'][:10]}..HEAD -- . ':(exclude)perf-loop'`",
                  "", "```", stat or "(sin cambios de código)", "```"]
        if mod_tests.strip():
            lines += ["", "**Tests existentes modificados o borrados** (justificación obligatoria):", ""] + \
                     [f"- `{x}`" for x in mod_tests.splitlines()]
    print("\n".join(lines))
    return 0


def cmd_results(a):
    ctx = Ctx()
    write_results_md(ctx)
    print((ctx.root / LOOP / "RESULTS.md").read_text(encoding="utf-8"))
    return 0


def cmd_profiles(a):
    d = Path(a.data_dir).expanduser().resolve()
    db = d / "voicebox.db"
    if not db.exists():
        raise LoopError(f"No encuentro {db}")
    con = sqlite3.connect(f"file:{db.as_posix()}?mode=ro", uri=True)
    rows = con.execute(
        "select p.id, p.name, p.language, p.voice_type, count(s.id), "
        "coalesce(sum(length(s.reference_text)),0) from profiles p "
        "left join profile_samples s on s.profile_id = p.id group by p.id order by p.name").fetchall()
    con.close()
    if a.json:
        print(json.dumps([{"id": r[0], "name": r[1], "language": r[2], "voice_type": r[3], "samples": r[4]}
                          for r in rows], ensure_ascii=False, indent=2))
    else:
        for r in rows:
            print(f"{r[0]}  {r[1]}  · idioma {r[2]} · tipo {r[3]} · muestras {r[4]}")
    return 0


def cmd_doctor(a):
    root = find_root()
    rows = []

    def add(level, msg):
        rows.append((level, msg))

    add("OK", f"Python {sys.version.split()[0]} ({sys.executable})")
    gv = sh(["git", "--version"], check=False)
    add("OK" if gv.returncode == 0 else "FAIL", gv.stdout.strip() or "git no está en el PATH")
    in_loop = (root / STATE_REL).exists()
    if not in_loop:
        add("INFO", f"{root} no es un worktree del loop.")
        add("OK" if (SKILL_DIR / "scaffold").is_dir() else "FAIL", "scaffold/ junto a loopctl.py")
        add("OK" if (root / "backend/venv").exists() else "FAIL", "backend/venv en el repo principal")
    else:
        ctx = Ctx(root)
        br = current_branch(root)
        add("OK" if br.startswith(ctx.prefix()) else "WARN", f"rama actual {br}")
        py = ctx.python()
        add("OK" if py != sys.executable or "venv" in py else "WARN", f"python del venv: {py}")
        d = dirty_code(root)
        add("WARN" if d else "OK", "árbol limpio" if not d else f"cambios sin commit: {', '.join(d[:5])}")
        b = ctx.config.get("bench", {})
        add("OK" if b.get("profile_id") else "WARN",
            f"perfil de voz para el banco: {b.get('profile_id') or 'sin definir (config.json → bench.profile_id)'}")
        src = b.get("source_data_dir")
        add("OK" if src and Path(src).exists() else "WARN", f"datos de origen del perfil: {src or 'sin definir'}")
        port = b.get("port", 17599)
        add("WARN" if port_in_use(port) else "OK", f"puerto del banco {port} {'ocupado' if port_in_use(port) else 'libre'}")
        up = b.get("user_app_port", 17493)
        if port_in_use(up):
            add("WARN", f"la app de Voicebox está abierta (puerto {up}): competirá por la GPU durante el banco")
        try:
            free = shutil.disk_usage(root).free / 2**30
            add("OK" if free > 10 else "WARN", f"espacio libre {free:.0f} GiB")
        except OSError:
            pass
        settings = read_json(root / ".claude/settings.json", {}) or {}
        add("OK" if "loopctl.py" in json.dumps(settings.get("hooks", {})) else "FAIL", "hooks del loop en .claude/settings.json")
        hook_script = root / ".claude/skills/tts-loop/loopctl.py"
        if hook_script.exists():
            hp = sh([py, str(hook_script), "hook", "pre-bash"], check=False, timeout=60,
                    env=dict(os.environ, CLAUDE_PROJECT_DIR=str(root)),
                    input_text=json.dumps({"tool_input": {"command": "git push origin main"}}))
            add("OK" if hp.returncode == 2 else "FAIL",
                "el hook pre-bash bloquea `git push` con el python del venv" if hp.returncode == 2
                else f"el hook pre-bash no bloquea (código {hp.returncode}): {(hp.stderr or '').strip()[:200]}")
        else:
            add("FAIL", "falta .claude/skills/tts-loop/loopctl.py en el worktree")
        mism = frozen_mismatches(ctx)
        add("FAIL" if mism else "OK", "arnés congelado intacto" if not mism else f"arnés modificado: {mism}")
        tb = read_json(root / TESTS_BASELINE_REL)
        if tb is not None:
            if tb.get("stale"):
                add("INFO", f"lista de tests de partida pendiente de registrarse de nuevo ({tb.get('stale_reason', '')})")
            elif not tb.get("passed"):
                add("WARN", "la lista de tests de partida no tiene ningún test que pase: la regla «ningún test "
                            "nuevo fallando» no protege nada"
                    + ("" if ctx.loop.get("frozen") else " (revisa tests.command y reabre la tarea que la registra)"))
        if (root / STOP_REL).exists():
            add("WARN", "existe perf-loop/STOP: el loop no avanzará hasta que lo borres")
        busy = running_workers(ctx)
        if busy:
            add("INFO", f"verificación en curso: {', '.join(busy)}")
        add("OK" if shutil.which("claude") else "INFO", "CLI `claude` en el PATH (necesario para run-loop.ps1)")
        if not a.quick:
            p = sh([py, "-c", "import torch;print(torch.__version__, torch.version.hip, torch.cuda.is_available())"],
                   check=False, timeout=300)
            add("OK" if p.returncode == 0 and "True" in p.stdout else "WARN",
                f"torch: {(p.stdout or p.stderr).strip()[-200:]}")
    for level, msg in rows:
        print(f"[{level:4}] {msg}")
    return 1 if any(r[0] == "FAIL" for r in rows) else 0


def real_changes(root):
    out = git(root, "diff", "--ignore-cr-at-eol", "--numstat", "HEAD", check=False)
    res = []
    for line in out.splitlines():
        parts = line.split("\t")
        if len(parts) == 3:
            a_, d_, p = parts
            if a_ == "-" or int(a_) + int(d_) > 0:
                res.append(p)
    return res


def link_dir(src, dst):
    if IS_WIN:
        import _winapi
        _winapi.CreateJunction(str(src), str(dst))
    else:
        os.symlink(str(src), str(dst), target_is_directory=True)


def cmd_init(a):
    root = find_root()
    if (root / STATE_REL).exists():
        raise LoopError("Este directorio ya es un worktree del loop.")
    scaffold = SKILL_DIR / "scaffold"
    if not scaffold.is_dir():
        raise LoopError(f"Falta {scaffold}")
    head = head_sha(root)
    branch = current_branch(root)
    wt = Path(a.worktree).expanduser().resolve() if a.worktree else root.parent / f"{root.name}-perf"
    prefix = a.prefix
    f0 = f"{prefix}-f0"
    src = Path(a.source_data_dir).expanduser().resolve() if a.source_data_dir else root / "data"
    real = real_changes(root)
    carry = list(a.carry or [])
    if a.carry_backend:
        carry += [p for p in real if p.startswith("backend/") and p not in carry]
    problems = []
    if wt.exists():
        problems.append(f"ya existe {wt}")
    if branch_exists(root, f0):
        problems.append(f"ya existe la rama {f0}")
    venv = root / "backend" / "venv"
    if not venv.exists():
        problems.append("no existe backend/venv en el repo principal")
    for p in carry:
        if not (root / p).is_file():
            problems.append(f"--carry {p}: no es un archivo")
    profiles = []
    if (src / "voicebox.db").exists():
        con = sqlite3.connect(f"file:{(src / 'voicebox.db').as_posix()}?mode=ro", uri=True)
        profiles = [{"id": r[0], "name": r[1], "language": r[2]} for r in
                    con.execute("select id, name, language from profiles order by name").fetchall()]
        con.close()
    new_backend = [x for x in git(root, "ls-files", "--others", "--exclude-standard", "--", "backend",
                                  check=False).splitlines() if x.strip()]
    report = {"repo": str(root), "rama_actual": branch, "commit_base": head, "worktree": str(wt),
              "rama_fase0": f0, "cambios_reales_sin_commit": real,
              "archivos_nuevos_sin_seguimiento_en_backend (solo con --carry)": new_backend, "se_traen": carry,
              "datos_origen": str(src), "perfiles": profiles, "perfil_elegido": a.profile_id,
              "problemas": problems}
    if a.dry_run or problems:
        print(json.dumps(report, ensure_ascii=False, indent=2))
        if problems and not a.dry_run:
            raise LoopError("No se puede inicializar: " + "; ".join(problems))
        return 0
    if a.profile_id and profiles and a.profile_id not in {p["id"] for p in profiles}:
        raise LoopError(f"El perfil {a.profile_id} no está en {src}/voicebox.db")

    git(root, "worktree", "add", "-q", "-b", f0, str(wt), head)
    link_dir(venv, wt / "backend" / "venv")
    # Git puede ver el junction/symlink como archivo y no aplicarle el patrón «venv/» del .gitignore.
    excl = Path(git(root, "rev-parse", "--git-common-dir"))
    excl = (excl if excl.is_absolute() else root / excl) / "info" / "exclude"
    excl.parent.mkdir(parents=True, exist_ok=True)
    cur = excl.read_text(encoding="utf-8") if excl.exists() else ""
    if "/backend/venv" not in cur.split():
        with open(excl, "a", encoding="utf-8", newline="\n") as f:
            f.write(("" if cur.endswith("\n") or not cur else "\n") + "/backend/venv\n")
    carry_commit = None
    if carry:
        for p in carry:
            data = (root / p).read_bytes()
            blob = shb(["git", "-C", str(root), "show", f"HEAD:{p}"], check=False)
            if blob.returncode == 0 and b"\r\n" not in blob.stdout:
                data = data.replace(b"\r\n", b"\n")  # el árbol de trabajo tiene CRLF y el repo no
            (wt / p).parent.mkdir(parents=True, exist_ok=True)
            (wt / p).write_bytes(data)
        git(wt, "add", "--", *carry)
        git(wt, "commit", "-q", "--no-verify", "-m",
            f"chore(perf-loop): trae {len(carry)} cambios locales sin commit\n\n" + "\n".join(f"- {p}" for p in carry))
        carry_commit = head_sha(wt)
    # La plantilla guarda la configuración de Claude Code en scaffold/dot-claude/ y aquí pasa a .claude/.
    shutil.copytree(scaffold / LOOP, wt / LOOP, dirs_exist_ok=True)
    shutil.copytree(scaffold / "dot-claude", wt / ".claude", dirs_exist_ok=True)
    sk = wt / ".claude" / "skills" / "tts-loop"
    sk.mkdir(parents=True, exist_ok=True)
    for f in ("SKILL.md", "README.md", "loopctl.py"):
        if (SKILL_DIR / f).exists():
            shutil.copy2(SKILL_DIR / f, sk / f)
    cfg = read_json(wt / CONFIG_REL, {}) or {}
    cfg.update(main_repo=str(root), worktree=str(wt), branch_prefix=prefix, created_at=now())
    cfg.setdefault("bench", {}).update(source_data_dir=str(src), profile_id=a.profile_id)
    if a.models_dir:
        cfg["bench"].setdefault("env", {})["VOICEBOX_MODELS_DIR"] = str(Path(a.models_dir).expanduser().resolve())
    write_json(wt / CONFIG_REL, cfg)
    git(wt, "add", "-f", "--", ".claude")
    git(wt, "add", "--", LOOP)
    git(wt, "commit", "-q", "--no-verify", "-m", "chore(perf-loop): andamiaje del loop agéntico /tts-loop")
    scaffold_commit = head_sha(wt)
    st = read_json(wt / STATE_REL)
    for t in st["tasks"]:
        t.setdefault("status", "pending")
        t.setdefault("attempts", 0)
    st["loop"] = {"schema": "tts-loop/1", "created_at": now(), "base_commit": head, "base_branch": branch,
                  "carry_commit": carry_commit, "scaffold_commit": scaffold_commit, "branch_prefix": prefix,
                  "phase": 0, "phase_branches": {"0": f0}, "current": None, "anchor": None,
                  "quality_ref": None, "perf_ref": None, "frozen": {}, "noise": {}}
    write_json(wt / STATE_REL, st)
    ctx = Ctx(wt)
    progress(ctx, f"Loop inicializado desde {branch}@{head[:10]} en {wt}")
    commit_state(ctx, "loop: estado inicial")
    print(json.dumps({**report, "creado": True, "scaffold_commit": scaffold_commit,
                      "siguiente": f"Abre Claude Code en {wt} y ejecuta /tts-loop"}, ensure_ascii=False, indent=2))
    return 0


def is_link_or_junction(p) -> bool:
    p = Path(p)
    if p.is_symlink():
        return True
    isj = getattr(os.path, "isjunction", None)  # Python >= 3.12
    if isj and isj(p):
        return True
    try:
        tag = getattr(os.lstat(p), "st_reparse_tag", 0)
    except OSError:
        return False
    return bool(tag) and tag == getattr(stat, "IO_REPARSE_TAG_MOUNT_POINT", 0xA0000003)


def cmd_install(a):
    """Copia el skill a <repo>/.claude/skills/tts-loop para que Claude Code lo cargue como /tts-loop."""
    root = find_root()
    if (root / STATE_REL).exists():
        raise LoopError("Esto es un worktree del loop; instala desde el repo principal.")
    dest = root / ".claude" / "skills" / "tts-loop"
    if norm(dest) == norm(SKILL_DIR):
        print(f"Ya está instalado en {dest}.")
        return 0
    if dest.exists() and not a.force:
        raise LoopError(f"Ya existe {dest}. Repite con --force para sobrescribirlo.")
    dest.mkdir(parents=True, exist_ok=True)
    for f in ("SKILL.md", "README.md", "loopctl.py"):
        shutil.copy2(SKILL_DIR / f, dest / f)
    shutil.copytree(SKILL_DIR / "scaffold", dest / "scaffold", dirs_exist_ok=True)
    print(f"Instalado en {dest}. Abre Claude Code en {root} y escribe /tts-loop init.")
    return 0


def cmd_teardown(a):
    root = find_root()
    if (root / STATE_REL).exists():
        raise LoopError("Ejecuta teardown desde el repo principal, no desde el worktree que vas a borrar.")
    wt = Path(a.worktree).expanduser().resolve() if a.worktree else root.parent / f"{root.name}-perf"
    listed = git(root, "worktree", "list", "--porcelain")
    if not any(norm(line[9:]) == norm(wt) for line in listed.splitlines() if line.startswith("worktree ")):
        raise LoopError(f"{wt} no es un worktree de este repo.")
    if not a.yes:
        raise LoopError(f"Esto borra {wt} (incluidos los WAV de perf-loop/bench). Repite con --yes si es lo que quieres.")
    venv = wt / "backend" / "venv"
    if venv.exists() or is_link_or_junction(venv):
        if not is_link_or_junction(venv):
            raise LoopError(f"{venv} no es un enlace: no lo borro. Revísalo a mano.")
        if IS_WIN and not venv.is_symlink():
            os.rmdir(venv)  # en un junction, rmdir quita el enlace y no toca el destino
        else:
            os.unlink(venv)
    if not (root / "backend" / "venv").exists():
        raise LoopError("El venv del repo principal ya no está; me detengo sin tocar nada más.")
    git(root, "worktree", "remove", "--force", str(wt))
    print(f"Worktree {wt} eliminado; el venv del repo principal sigue intacto. Las ramas {a.prefix}-* se conservan "
          f"(bórralas con git branch -D si ya no las quieres).")
    return 0


def cmd_export(a):
    ctx = Ctx()
    L = ctx.loop
    name = a.branch or f"{ctx.prefix()}-export"
    if branch_exists(ctx.root, name) and not a.force:
        raise LoopError(f"Ya existe {name} (usa --force para rehacerla).")
    parent = L.get("carry_commit") or L["base_commit"]
    prev = head_sha(ctx.root, L["scaffold_commit"])
    commit = parent
    idx = Path(git(ctx.root, "rev-parse", "--git-path", "index.tts-export"))
    if not idx.is_absolute():
        idx = ctx.root / idx
    env = dict(os.environ, GIT_INDEX_FILE=str(idx))
    try:
        git(ctx.root, "read-tree", parent, env=env)
        made = []
        for n in sorted(int(k) for k in L.get("phase_branches", {})):
            b = L["phase_branches"][str(n)]
            if not branch_exists(ctx.root, b):
                continue
            tip = head_sha(ctx.root, b)
            diff = shb(["git", "-C", str(ctx.root), "diff", "--binary", prev, tip, "--", ".",
                        f":(exclude){LOOP}", ":(exclude).claude"]).stdout  # bytes: sin strip ni CRLF
            if diff.strip():
                shb(["git", "-C", str(ctx.root), "apply", "--cached", "--whitespace=nowarn"], env=env,
                    input_bytes=diff)
                tree = git(ctx.root, "write-tree", env=env)
                done = [t["id"] for t in ctx.tasks if t["phase"] == n and t["status"] == "passed"]
                msg = f"perf(qwen-tts): fase {n} · {ctx.phase_title(n)}\n\nTareas superadas: {', '.join(done) or '—'}"
                commit = git(ctx.root, "commit-tree", tree, "-p", commit, "-m", msg)
                made.append(n)
            prev = tip
    finally:
        try:
            idx.unlink()
        except FileNotFoundError:
            pass
    git(ctx.root, "branch", "-f", name, commit)
    print(f"Rama {name} creada sobre {parent[:10]} con {len(made)} commits (fases {made}), "
          f"sin .claude/ ni perf-loop/.")
    return 0


# ---------------------------------------------------------------------------
# Hooks de Claude Code
# ---------------------------------------------------------------------------

WRITE_HINT = re.compile(
    r"(\btee\b|\bsed\s+-i|\bcp\b|\bmv\b|\bSet-Content\b|\bAdd-Content\b|\bOut-File\b|\btruncate\b|\btouch\b"
    r"|\brm\b|\bdel\b|\bRemove-Item\b|\bCopy-Item\b|\bMove-Item\b|open\(|write_text|write_bytes|\bdd\b|\bln\b"
    r"|\bunlink\b|\bshutil\b|\bos\.remove\b|\bri\b|\berase\b|\bNew-Item\b|\bni\b)", re.I)
HARMLESS_REDIRECT = re.compile(r"\d?>&\d|[12]?>{1,2}\s*(/dev/null|NUL|\$null)\b", re.I)
REDIRECT_TARGET = re.compile(r">{1,2}\s*[\"']?([^\s\"'|;&<>]+)")
VENV_EXE = re.compile(r"(\./)?backend/venv/(Scripts|bin)/[\w.\-]+")


def bash_touches_guarded(cmd, guarded):
    """Devuelve la ruta protegida que el comando escribiría, o None."""
    c = HARMLESS_REDIRECT.sub(" ", cmd.replace("\\", "/"))
    c = VENV_EXE.sub(" ", c)  # ejecutar el python/ruff del venv no es escribir en el venv
    for m in REDIRECT_TARGET.finditer(c):
        for g in guarded:
            if g in m.group(1):
                return g
    hits = [g for g in guarded if g in c]
    if hits and (WRITE_HINT.search(c) or re.search(r"\bgit\s+add\b", c)):
        return hits[0]
    return None
BASH_RULES = [
    (r"\bgit\s+push\b", "push prohibido: el loop trabaja solo en ramas locales."),
    (r"\bgit\s+(checkout|switch|merge|rebase|reset|stash|cherry-pick|revert|worktree|filter-branch|filter-repo"
     r"|update-ref|replace|gc|prune|clean|am|apply|pull|fetch|remote|submodule|config|tag|notes|commit-tree"
     r"|read-tree|write-tree|update-index|symbolic-ref)\b",
     "las operaciones de rama, historia y configuración de git son de loopctl; tú solo add/commit/diff/log/show/status."),
    (r"\bgit\s+restore\b.*(\s\.(\s|$)|\s:/|--source|\s-s\b)", "git restore solo con rutas concretas."),
    (r"\bgit\s+branch\s+(?!(--show-current|--list|-a|-r|-v|-vv|--contains|--merged|--no-merged)\b)\S",
     "no crees, borres ni muevas ramas: lo hace loopctl."),
    (r"\bgit\s+add\s+(.*\s)?(-A|--all|-u|--update|\.|:/|\*|-f|--force)(\s|$)",
     "añade rutas explícitas (sin -A, -u, -f, «.» ni comodines)."),
    (r"\bgit\s+commit\b.*(\s-a\b|\s-am\b|\s-ma\b|\s--all\b|--amend|--no-verify|\s-n\b)",
     "commit sin -a/--all/--amend/--no-verify."),
    (r"\brm\s+(-[a-zA-Z]*[rR]|--recursive)|Remove-Item\b.*-Recurse|\brmdir\b|\brd\s+/s|\bdel\s+/s|shutil\.rmtree",
     "borrado recursivo prohibido."),
    (r"\b(pip3?|uv|conda|mamba|poetry)\b.*\b(install|uninstall|add|remove|sync)\b|-m\s+pip\s+(install|uninstall)",
     "no instales ni quites paquetes: el venv es el del repo principal y lo comparte tu app."),
    (r"\b17493\b", "no toques la app de Voicebox del usuario (puerto 17493); el banco usa su propio puerto."),
    (r"cudnn\.benchmark\s*=\s*True", "cudnn.benchmark=True fuerza la búsqueda exhaustiva de MIOpen (ver «Qué no hacer»)."),
    (r"\b(taskkill|Stop-Process|pkill|killall)\b.*\b(python|voicebox|claude|node)\b",
     "no mates procesos por nombre; el banco gestiona su propio servidor."),
    (r"\bclaude\s+(-p|--print|-w|--worktree|--dangerously|--permission-mode|--resume|--continue)\b|--dangerously-skip-permissions",
     "no lances otras sesiones de Claude desde el loop."),
]
SIMPLE_LOOPCTL = re.compile(r"^\s*(\S*python(\.exe)?|py)\s+\S*loopctl\.py\s+[^;&|`<>$\n]*$", re.I)


def hook_input():
    try:
        raw = sys.stdin.buffer.read().decode("utf-8", "replace")  # en Windows stdin sería cp1252
    except Exception:
        raw = ""
    try:
        return json.loads(raw) if raw.strip() else {}
    except json.JSONDecodeError:
        return {}


def hook_root(data):
    return Path(os.environ.get("CLAUDE_PROJECT_DIR") or data.get("cwd") or os.getcwd())


def deny(reason):
    sys.stderr.write(f"[tts-loop] Bloqueado: {reason}\n")
    sys.exit(2)


def main_repo_forms(cfg):
    mr = cfg.get("main_repo")
    if not mr:
        return []
    p = os.path.abspath(mr)
    forms = {p, p.replace("\\", "/")}
    m = re.match(r"^([A-Za-z]):[\\/](.*)$", p)
    if m:
        forms.add(f"/{m.group(1).lower()}/{m.group(2)}".replace("\\", "/"))
        forms.add(f"/mnt/{m.group(1).lower()}/{m.group(2)}".replace("\\", "/"))
    return [f.rstrip("\\/") for f in forms]


def hook_pre_bash(data):
    root = hook_root(data)
    if not (root / STATE_REL).exists():
        return 0
    cmd = (data.get("tool_input") or {}).get("command") or ""
    if re.search(r"--i-am-human|TTS_LOOP_HUMAN|human-token", cmd, re.I):
        deny("las decisiones humanas solo las toma el usuario.")
    if re.search(r"loopctl\.py[\"']?\s+(--root\s+\S+\s+)?(hook|_worker|teardown)\b", cmd):
        deny("los subcomandos internos de loopctl (hook, _worker, teardown) no se lanzan a mano.")
    agent_type = str(data.get("agent_type") or "")
    m = re.search(r"loopctl\.py[\"']?\s+(--root\s+\S+\s+)?([\w-]+)", cmd)
    if agent_type.startswith("tts-") and m and m.group(2) not in SUBAGENT_LOOPCTL_OK:
        deny(f"«loopctl {m.group(2)}» lo ejecuta el orquestador, no {agent_type}. Devuélvelo en tu informe.")
    if SIMPLE_LOOPCTL.match(cmd):
        return 0
    c = cmd.replace("\\", "/")
    for pat, why in BASH_RULES:
        if re.search(pat, cmd, re.I):
            deny(why)
    cfg = read_json(root / CONFIG_REL, {}) or {}
    for f in main_repo_forms(cfg):
        ff = f.replace("\\", "/")
        if re.search(re.escape(ff) + r"([\\/\"' ]|$)", c, re.I):
            deny(f"no trabajes en el repo principal ({cfg.get('main_repo')}); todo ocurre en el worktree.")
    st = read_json(root / STATE_REL, {}) or {}
    guarded = PROTECTED + PROTECTED_PREFIXES + list((st.get("loop") or {}).get("frozen", {}) or {})
    hit = bash_touches_guarded(cmd, guarded)
    if hit:
        deny(f"{hit} lo gestiona loopctl o es parte del arnés congelado; no se toca desde Bash.")
    if re.search(r"\bgit\s+commit\b", cmd):
        try:
            br = current_branch(root)
        except LoopError:
            br = ""
        prefix = (st.get("loop") or {}).get("branch_prefix") or "perf/qwen"
        if not br.startswith(prefix + "-task/"):
            deny(f"los commits de código van en la rama de la tarea ({prefix}-task/…); ahora estás en {br}.")
    return 0


def hook_pre_edit(data):
    root = hook_root(data)
    if not (root / STATE_REL).exists():
        return 0
    ti = data.get("tool_input") or {}
    fp = ti.get("file_path") or ti.get("notebook_path") or ti.get("path")
    if not fp:
        return 0
    p = Path(fp)
    if not p.is_absolute():
        p = Path(data.get("cwd") or root) / p
    if not inside(p, root):
        extra = [data.get("scratchpad_dir"), tempfile.gettempdir()]
        if any(x and inside(p, x) for x in extra):
            return 0
        deny(f"{fp} está fuera del worktree del loop ({root}).")
    r = relposix(root, p)
    rl = r.lower()
    if rl in {x.lower() for x in PROTECTED} or any(rl.startswith(x.lower()) for x in PROTECTED_PREFIXES):
        deny(f"{r} lo gestiona loopctl o es configuración del loop; no se edita desde el agente.")
    st = read_json(root / STATE_REL, {}) or {}
    if rl in {x.lower() for x in ((st.get("loop") or {}).get("frozen") or {})}:
        deny(f"{r} es parte del arnés de evaluación congelado tras la línea base. "
             "Si de verdad hay que cambiarlo, explícalo con `loopctl block` y que el usuario decida.")
    new_parts = [ti.get("new_string"), ti.get("content"), ti.get("new_source")]
    old_parts = [ti.get("old_string")]
    for e in ti.get("edits") or []:
        new_parts.append(e.get("new_string"))
        old_parts.append(e.get("old_string"))
    new = "\n".join(x for x in new_parts if x)
    old = "\n".join(x for x in old_parts if x)
    if ti.get("content") is not None and p.exists():
        try:
            old = p.read_text(encoding="utf-8", errors="replace")
        except OSError:
            pass
    if re.search(r"cudnn\.benchmark\s*=\s*True", new):
        deny("cudnn.benchmark=True fuerza la búsqueda exhaustiva de MIOpen (ver «Qué no hacer»).")
    if re.search(r"--i-am-human|TTS_LOOP_HUMAN|human-token", new, re.I):
        deny("las decisiones humanas solo las toma el usuario.")
    if rl.startswith("backend/") and new.count("0.6B") > old.count("0.6B"):
        deny("el plan excluye pasar al modelo 0.6B (otra red).")
    return 0


def hook_stop(data):
    root = hook_root(data)
    if not (root / STATE_REL).exists() or data.get("stop_hook_active"):
        return 0
    m = read_json(root / MARKER_REL)
    if not m:
        return 0
    st = read_json(root / STATE_REL, {}) or {}
    tid = m.get("task")
    t = next((x for x in st.get("tasks", []) if x["id"] == tid), None)
    if not t or t.get("status") != "in_progress":
        return 0
    sid = str(data.get("session_id") or "?")
    blocks = m.setdefault("stop_blocks", {})
    if blocks.get(sid, 0) >= 1:
        return 0
    blocks[sid] = blocks.get(sid, 0) + 1
    write_json(root / MARKER_REL, m)
    reason = (f"La tarea {tid} del loop sigue en curso. Antes de terminar, déjala en un estado claro: "
              f"`loopctl finish {tid} pass|reject|noop`, `loopctl await {tid}` si necesita al usuario, "
              f"`loopctl block {tid} --reason …` si estás atascado, o `loopctl pause` si el usuario pidió parar. "
              f"Si hay una verificación corriendo, sigue con `loopctl wait {tid}`.")
    print(json.dumps({"decision": "block", "reason": reason}, ensure_ascii=False))
    return 0


PROMPT_CMD = re.compile(r"^\s*/tts-loop\s+(approve|reject|ab|skip|retry|reopen|refreeze)\b\s*(\S+)?\s*(.*)$",
                        re.I | re.S)


def hook_prompt(data):
    root = hook_root(data)
    if not (root / STATE_REL).exists():
        return 0
    lines = (data.get("prompt") or "").strip().splitlines()
    m = PROMPT_CMD.match(lines[0]) if lines else None
    if not m:
        return 0
    cmd = m.group(1).lower()
    tid = "harness" if cmd == "refreeze" else (m.group(2) or "")
    args = m.group(3) if cmd != "refreeze" else " ".join(x for x in (m.group(2), m.group(3)) if x)
    write_json(root / TOKEN_REL, {"cmd": cmd, "id": tid, "args": args.strip(), "ts": time.time(),
                                  "session_id": data.get("session_id")})
    print(f"tts-loop: decisión humana registrada ({cmd} {tid}). Ejecuta el loopctl correspondiente.")
    return 0


def hook_session_start(data):
    root = hook_root(data)
    if (root / STATE_REL).exists():
        os.chdir(root)
        cmd_brief(None)
    return 0


def cmd_hook(a):
    data = hook_input()
    fn = {"pre-bash": hook_pre_bash, "pre-edit": hook_pre_edit, "stop": hook_stop,
          "prompt": hook_prompt, "session-start": hook_session_start}[a.event]
    try:
        return fn(data)
    except SystemExit:
        raise
    except Exception as e:  # un fallo del hook no debe tumbar la sesión
        sys.stderr.write(f"[tts-loop] hook {a.event} falló: {e}\n")
        return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser():
    p = argparse.ArgumentParser(prog="loopctl", description="Control determinista del loop /tts-loop.",
                                allow_abbrev=False)
    p.add_argument("--root", help="raíz del worktree (por defecto, la del repo git actual)")
    sp = p.add_subparsers(dest="cmd", required=True)

    def add(name, fn, help_):
        s = sp.add_parser(name, help=help_, allow_abbrev=False)
        s.set_defaults(fn=fn)
        return s

    s = add("init", cmd_init, "crea el worktree, las ramas y el andamiaje (desde el repo principal)")
    s.add_argument("--worktree")
    s.add_argument("--prefix", default="perf/qwen")
    s.add_argument("--carry", nargs="*", help="archivos con cambios sin commit que se traen al worktree")
    s.add_argument("--carry-backend", action="store_true", help="trae los cambios reales sin commit de backend/")
    s.add_argument("--source-data-dir", help="data dir con el perfil de voz (por defecto ./data)")
    s.add_argument("--profile-id")
    s.add_argument("--models-dir", help="carpeta de modelos de la app (VOICEBOX_MODELS_DIR), si usa una propia")
    s.add_argument("--dry-run", action="store_true")
    s = add("profiles", cmd_profiles, "lista los perfiles de voz de un data dir")
    s.add_argument("--data-dir", required=True)
    s.add_argument("--json", action="store_true")
    s = add("status", cmd_status, "estado del backlog")
    s.add_argument("--json", action="store_true")
    add("brief", cmd_brief, "resumen corto para inyectar en el contexto")
    s = add("doctor", cmd_doctor, "comprueba el entorno")
    s.add_argument("--quick", action="store_true")
    s = add("next", cmd_next, "siguiente tarea o motivo de parada")
    s.add_argument("--json", action="store_true")
    s = add("begin", cmd_begin, "empieza una tarea (crea su rama)")
    s.add_argument("id")
    s = add("show", cmd_show, "ficha de la tarea")
    s.add_argument("id")
    s.add_argument("--for", dest="for_", choices=["implementer", "reviewer", "human"], default="human")
    s = add("verify", cmd_verify, "lanza banco + comparación + tests + puerta")
    s.add_argument("id")
    s.add_argument("--foreground", action="store_true")
    s = add("_worker", cmd_worker, argparse.SUPPRESS)
    s.add_argument("id")
    s.add_argument("tag")
    s = add("wait", cmd_wait, "espera a la verificación en curso (máx. --max s por llamada)")
    s.add_argument("id")
    s.add_argument("--tag")
    s.add_argument("--max", type=float, default=540)
    s = add("gate", cmd_gate, "reevalúa la puerta con los datos ya medidos")
    s.add_argument("id")
    s.add_argument("--tag")
    s = add("finish", cmd_finish, "cierra la tarea en curso")
    s.add_argument("id")
    s.add_argument("result", choices=["pass", "reject", "noop"])
    s.add_argument("--reason", default="")
    s = add("await", cmd_await, "deja la tarea esperando una decisión humana")
    s.add_argument("id")
    s = add("block", cmd_block, "bloquea la tarea en curso y detiene el loop")
    s.add_argument("id")
    s.add_argument("--reason", required=True)
    add("pause", cmd_pause, "pausa: permite terminar la sesión con la tarea a medias")
    s = add("note", cmd_note, "añade una nota a progress.md (id o «-»)")
    s.add_argument("id")
    s.add_argument("text", nargs="+")
    add("results", cmd_results, "regenera RESULTS.md")
    s = add("install", cmd_install, "copia el skill a .claude/skills/tts-loop del repo actual")
    s.add_argument("--force", action="store_true")
    s = add("teardown", cmd_teardown, "borra el worktree sin tocar el venv compartido (desde el repo principal)")
    s.add_argument("--worktree")
    s.add_argument("--prefix", default="perf/qwen")
    s.add_argument("--yes", action="store_true")
    s = add("export", cmd_export, "rama limpia con un commit por fase, sin .claude/ ni perf-loop/")
    s.add_argument("--branch")
    s.add_argument("--force", action="store_true")
    for name, fn in (("approve", cmd_approve), ("reject", cmd_reject), ("skip", cmd_skip),
                     ("retry", cmd_retry)):
        s = add(name, fn, f"decisión humana: {name}")
        s.add_argument("id")
        s.add_argument("--i-am-human", action="store_true")
        if name == "approve":
            s.add_argument("--note", default="")
            s.add_argument("--override", action="store_true", help="aprueba sin escucha ciega")
        if name in ("reject", "skip"):
            s.add_argument("--reason", default="")
    s = add("ab", cmd_ab, "decisión humana: resultado de la escucha ciega")
    s.add_argument("id")
    s.add_argument("answer", nargs="?")
    s.add_argument("--i-am-human", action="store_true")
    s = add("reopen", cmd_reopen, "decisión humana: reabre una tarea superada de la fase en curso")
    s.add_argument("id")
    s.add_argument("--reason", default="")
    s.add_argument("--i-am-human", action="store_true")
    s = add("refreeze", cmd_refreeze, "decisión humana: acepta cambios en el arnés congelado")
    s.add_argument("--reason", default="")
    s.add_argument("--i-am-human", action="store_true")
    s = add("hook", cmd_hook, argparse.SUPPRESS)
    s.add_argument("event", choices=["pre-bash", "pre-edit", "stop", "prompt", "session-start"])
    return p


def main(argv=None):
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass
    args = build_parser().parse_args(argv)
    if args.root:
        os.environ["TTS_LOOP_ROOT"] = str(Path(args.root).resolve())
        try:
            os.chdir(args.root)
        except OSError:
            pass
    try:
        return args.fn(args) or 0
    except LoopError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1
    except BrokenPipeError:  # p. ej. `loopctl status | head`
        try:
            sys.stdout = open(os.devnull, "w")
        except OSError:
            pass
        return 0


if __name__ == "__main__":
    sys.exit(main())
