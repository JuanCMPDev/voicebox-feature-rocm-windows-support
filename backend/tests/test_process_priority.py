"""
Tests for backend/utils/process_priority.py, the server process at above normal priority on Windows with ROCm
(perf-loop plan, Fase 3, task 3.3), and for how main.py (``python -m backend.main``) and server.py (packaged binary)
call it at startup.

Runs without a GPU: the priority change is a stub except in one Windows-only test, which restores the original class.

Usage:
    python -m pytest backend/tests/test_process_priority.py -v
"""

import logging
import os
import runpy
import subprocess
import sys
import types
from pathlib import Path

import pytest
import uvicorn

from backend import config, database
from backend.utils import miopen_db, process_priority

_BACKEND = Path(__file__).resolve().parents[1]
_HIP = "7.2.53211-158bd99533"


class Recorder:
    """Stand-in for :func:`process_priority.set_above_normal_priority`: counts calls, optionally raises."""

    def __init__(self, error: BaseException | None = None):
        self.calls = 0
        self.error = error

    def __call__(self) -> None:
        self.calls += 1
        if self.error is not None:
            raise self.error


@pytest.fixture
def no_real_priority(monkeypatch) -> Recorder:
    """Replaces the ctypes call, so no test changes the priority of the test process by accident."""
    recorder = Recorder(AssertionError("the real priority change must not run"))
    monkeypatch.setattr(process_priority, "set_above_normal_priority", recorder)
    return recorder


class TestEnabled:
    @pytest.mark.parametrize(("variant", "hip"), [("rocm", None), (None, _HIP), ("cpu", _HIP), ("rocm", _HIP)])
    def test_windows_rocm_enables_it_by_default(self, variant, hip):
        assert process_priority.priority_enabled(variant, hip, "win32", {}) is True

    @pytest.mark.parametrize("platform", ["linux", "darwin", "cygwin"])
    def test_other_systems_are_untouched(self, platform):
        assert process_priority.priority_enabled("rocm", _HIP, platform, {}) is False

    @pytest.mark.parametrize(("variant", "hip"), [(None, None), ("cpu", None), ("cuda", None), ("", "")])
    def test_non_rocm_builds_are_untouched(self, variant, hip):
        assert process_priority.priority_enabled(variant, hip, "win32", {}) is False

    @pytest.mark.parametrize("value", ["0", "false", "No", " off "])
    def test_variable_disables_it(self, value):
        environ = {process_priority.PRIORITY_ENV: value}
        assert process_priority.priority_enabled("rocm", None, "win32", environ) is False

    @pytest.mark.parametrize("value", ["1", "true", "", "yes"])
    def test_other_values_keep_it_on(self, value):
        environ = {process_priority.PRIORITY_ENV: value}
        assert process_priority.priority_enabled("rocm", None, "win32", environ) is True

    def test_reads_process_environment_by_default(self, monkeypatch):
        monkeypatch.setenv(process_priority.PRIORITY_ENV, "0")
        assert process_priority.priority_enabled("rocm", _HIP, "win32") is False


class TestRaise:
    def test_windows_rocm_raises_it(self, caplog):
        recorder = Recorder()
        with caplog.at_level(logging.INFO, logger=process_priority.__name__):
            assert process_priority.raise_process_priority("rocm", None, "win32", {}, recorder) is True
        assert recorder.calls == 1
        assert "above normal priority" in caplog.text

    def test_defaults_to_the_ctypes_call(self, monkeypatch):
        recorder = Recorder()
        monkeypatch.setattr(process_priority, "set_above_normal_priority", recorder)
        assert process_priority.raise_process_priority(None, _HIP, "win32", {}) is True
        assert recorder.calls == 1

    @pytest.mark.parametrize("platform", ["linux", "darwin"])
    def test_never_called_off_windows(self, platform, no_real_priority):
        recorder = Recorder(AssertionError("called off Windows"))
        assert process_priority.raise_process_priority("rocm", _HIP, platform, {}, recorder) is False
        assert process_priority.raise_process_priority("rocm", _HIP, platform, {}) is False
        assert recorder.calls == no_real_priority.calls == 0

    @pytest.mark.parametrize(("variant", "hip"), [(None, None), ("cpu", None), ("cuda", None)])
    def test_never_called_off_rocm(self, variant, hip, no_real_priority):
        assert process_priority.raise_process_priority(variant, hip, "win32", {}) is False
        assert no_real_priority.calls == 0

    def test_never_called_when_disabled(self, no_real_priority):
        environ = {process_priority.PRIORITY_ENV: "0"}
        assert process_priority.raise_process_priority("rocm", _HIP, "win32", environ) is False
        assert no_real_priority.calls == 0

    @pytest.mark.parametrize(
        "error", [OSError(5, "Access is denied"), AttributeError("module 'ctypes' has no attribute 'WinDLL'")]
    )
    def test_failure_is_logged_not_raised(self, error, caplog):
        recorder = Recorder(error)
        with caplog.at_level(logging.WARNING, logger=process_priority.__name__):
            assert process_priority.raise_process_priority("rocm", _HIP, "win32", {}, recorder) is False
        assert recorder.calls == 1
        assert "could not raise the server process priority" in caplog.text


@pytest.mark.skipif(sys.platform != "win32", reason="SetPriorityClass is Windows only")
def test_ctypes_call_sets_above_normal_on_windows():
    """The real call, on this process: the pseudo-handle reaches SetPriorityClass intact. The class is restored."""
    import ctypes
    from ctypes import wintypes

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.GetCurrentProcess.restype = wintypes.HANDLE
    kernel32.GetPriorityClass.argtypes = (wintypes.HANDLE,)
    kernel32.GetPriorityClass.restype = wintypes.DWORD
    kernel32.SetPriorityClass.argtypes = (wintypes.HANDLE, wintypes.DWORD)
    kernel32.SetPriorityClass.restype = wintypes.BOOL
    process = kernel32.GetCurrentProcess()
    original = kernel32.GetPriorityClass(process)
    assert original != 0
    try:
        process_priority.set_above_normal_priority()
        assert kernel32.GetPriorityClass(process) == process_priority.ABOVE_NORMAL_PRIORITY_CLASS
    finally:
        assert kernel32.SetPriorityClass(process, original)
    assert kernel32.GetPriorityClass(process) == original


def test_module_does_not_import_torch():
    """server.py imports it before backend.main pulls in torch, so it must stay stdlib-only."""
    code = "import sys, backend.utils.process_priority; sys.exit('torch' in sys.modules)"
    result = subprocess.run([sys.executable, "-c", code], cwd=_BACKEND.parent, capture_output=True, timeout=60)
    assert result.returncode == 0, result.stderr.decode(errors="replace")


@pytest.fixture
def entry_point(monkeypatch, tmp_path):
    """Runs an entry point's ``__main__`` with stubbed app, torch and uvicorn; records ``raise_process_priority``."""
    for name in (*miopen_db.MIOPEN_DB_VARS, "VOICEBOX_BACKEND_VARIANT"):
        monkeypatch.setenv(name, "x")
        monkeypatch.delenv(name)
    monkeypatch.setattr(config, "_data_dir", config._data_dir)
    monkeypatch.setattr(database, "init_db", lambda: None)
    monkeypatch.setattr(logging, "basicConfig", lambda **_kwargs: None)

    calls: list[tuple] = []
    order: list[str] = []

    def record(variant, hip=None, *_args, **_kwargs):
        calls.append((variant, hip))
        order.append("priority")
        return False

    monkeypatch.setattr(process_priority, "raise_process_priority", record)
    monkeypatch.setattr(uvicorn, "run", lambda *_a, **_k: order.append("uvicorn"))

    def run(which: str, hip: str | None, executable: str = sys.executable) -> tuple[list[tuple], list[str]]:
        stub_torch = types.ModuleType("torch")
        stub_torch.version = types.SimpleNamespace(hip=hip)
        monkeypatch.setitem(sys.modules, "torch", stub_torch)
        argv = ["--port", "17599", "--data-dir", str(tmp_path / "data")]
        if which == "main":
            stub_app = types.ModuleType("backend.app")
            stub_app.app = object()
            monkeypatch.setitem(sys.modules, "backend.app", stub_app)
            monkeypatch.delitem(sys.modules, "backend.main", raising=False)
            monkeypatch.setattr(sys, "argv", ["main.py", *argv])
            runpy.run_module("backend.main", run_name="__main__")
        else:
            stub_main = types.ModuleType("backend.main")
            stub_main.app = object()
            monkeypatch.setitem(sys.modules, "backend.main", stub_main)
            monkeypatch.setattr(sys, "executable", executable)
            monkeypatch.setattr(sys, "argv", [executable, *argv])
            runpy.run_path(str(_BACKEND / "server.py"), run_name="__main__")
        return calls, order

    return run


class TestEntryPoints:
    """Both entry points ask for the priority once, before uvicorn starts; the function itself decides."""

    @pytest.mark.parametrize("hip", [_HIP, None])
    def test_dev_main_passes_torch_hip(self, entry_point, hip):
        calls, order = entry_point("main", hip)
        assert calls == [(None, hip)]
        assert order == ["priority", "uvicorn"]

    @pytest.mark.parametrize(
        ("executable", "variant"),
        [
            ("C:\\voicebox\\voicebox-server-rocm.exe", "rocm"),
            ("C:\\voicebox\\voicebox-server-cuda.exe", "cuda"),
            ("C:\\voicebox\\voicebox-server.exe", "cpu"),
        ],
    )
    def test_packaged_server_passes_its_variant(self, entry_point, executable, variant):
        calls, order = entry_point("server", None, executable)
        assert calls == [(variant, None)]
        assert order == ["priority", "uvicorn"]
        assert os.environ["VOICEBOX_BACKEND_VARIANT"] == variant
