"""
Tests for backend/utils/miopen_db.py, MIOpen's Find-Db and kernel cache in {data_dir}/cache/miopen on ROCm (perf-loop
plan, Fase 1, task 1.2), and for how server.py (packaged binary) and main.py (``python -m backend.main``) apply it.

Runs without a GPU: the entry points run with ``backend.main`` / ``backend.app``, ``torch`` and ``uvicorn.run``
replaced by stubs that record the environment MIOpen would see.

Usage:
    python -m pytest backend/tests/test_miopen_db.py -v
"""

import argparse
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
from backend.utils import miopen_db

_BACKEND = Path(__file__).resolve().parents[1]
_HIP = "7.2.53211-158bd99533"
_VARS = miopen_db.MIOPEN_DB_VARS
_USER_DB = "D:\\mine\\miopen"


def _folder(data_dir: Path) -> Path:
    return data_dir.resolve() / "cache" / "miopen"


@pytest.fixture
def clean_env(monkeypatch):
    """Unset the MIOpen and variant variables; whatever the test or the code under test sets is undone afterwards."""
    for name in (*_VARS, "VOICEBOX_BACKEND_VARIANT"):
        # setenv before delenv makes monkeypatch record the original value, so later writes are undone too.
        monkeypatch.setenv(name, "x")
        monkeypatch.delenv(name)
    monkeypatch.setattr(config, "_data_dir", config._data_dir)
    return monkeypatch


class TestMiopenDbDir:
    def test_absolute_data_dir(self, tmp_path):
        assert miopen_db.miopen_db_dir(tmp_path) == tmp_path.resolve() / "cache" / "miopen"
        assert miopen_db.miopen_db_dir(str(tmp_path)) == tmp_path.resolve() / "cache" / "miopen"

    def test_relative_data_dir_resolves_against_cwd(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        assert miopen_db.miopen_db_dir("data") == tmp_path.resolve() / "data" / "cache" / "miopen"

    @pytest.mark.parametrize("data_dir", ["data", "nested/../data", "."])
    def test_matches_config_cache_dir(self, data_dir, tmp_path, monkeypatch):
        """Same folder the app's own cache uses once ``config.set_data_dir`` ran with the same ``--data-dir``."""
        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr(config, "_data_dir", config._data_dir)
        config.set_data_dir(data_dir)
        assert miopen_db.miopen_db_dir(data_dir) == config.get_cache_dir() / "miopen"


class TestDataDirFromArgv:
    @pytest.mark.parametrize(
        ("argv", "expected"),
        [
            ([], None),
            (["--port", "8000"], None),
            (["--data-dir", "C:\\Users\\a b\\voicebox"], "C:\\Users\\a b\\voicebox"),
            (["--data-dir=C:\\data"], "C:\\data"),
            (["--data-dir=a=b"], "a=b"),
            (["--data-dir", "D:\\x", "--port", "17493", "--parent-pid", "42"], "D:\\x"),
            (["--host", "0.0.0.0", "--data-dir", "rel/data"], "rel/data"),
            (["--data-dir", "a", "--data-dir", "b"], "b"),
            (["--data-dir=a", "--data-dir", "b"], "b"),
            (["--data-dir"], None),
            (["--data-dirs", "x"], None),
        ],
    )
    def test_parses_flag(self, argv, expected):
        assert miopen_db.data_dir_from_argv(argv) == expected

    @pytest.mark.parametrize(
        "argv",
        [
            [],
            ["--data-dir", "C:\\data", "--port", "8000"],
            ["--data-dir=C:\\data"],
            ["--data-dir", "a", "--data-dir", "b"],
            ["--port", "1", "--data-dir=x", "--parent-pid", "2"],
        ],
    )
    def test_agrees_with_argparse(self, argv):
        parser = argparse.ArgumentParser()
        parser.add_argument("--port", type=int)
        parser.add_argument("--data-dir", default=None)
        parser.add_argument("--parent-pid", type=int)
        assert miopen_db.data_dir_from_argv(argv) == parser.parse_args(argv).data_dir


class TestApplyMiopenDbDefaults:
    @pytest.mark.parametrize(("variant", "hip"), [("rocm", None), (None, _HIP), ("rocm", _HIP)])
    def test_rocm_sets_both_and_creates_folder(self, variant, hip, tmp_path, caplog):
        data_dir = tmp_path / "data"
        environ = {"PATH": "C:\\bin"}
        with caplog.at_level(logging.INFO, logger=miopen_db.__name__):
            changes = miopen_db.apply_miopen_db_defaults(data_dir, variant, hip, environ=environ)
        folder = _folder(data_dir)
        assert changes == dict.fromkeys(_VARS, str(folder))
        assert environ == {"PATH": "C:\\bin", **changes}
        assert folder.is_dir()
        assert str(folder) in caplog.text

    @pytest.mark.parametrize(
        ("variant", "hip"),
        [(None, None), ("cpu", None), ("cuda", None), (None, ""), ("rocm-like", None)],
    )
    def test_non_rocm_paths_are_untouched(self, variant, hip, tmp_path, caplog):
        environ = {"PATH": "C:\\bin"}
        with caplog.at_level(logging.INFO, logger=miopen_db.__name__):
            assert miopen_db.apply_miopen_db_defaults(tmp_path / "data", variant, hip, environ=environ) == {}
        assert environ == {"PATH": "C:\\bin"}
        assert not (tmp_path / "data").exists()
        assert caplog.text == ""

    def test_user_value_wins(self, tmp_path):
        environ = {"MIOPEN_USER_DB_PATH": _USER_DB}
        changes = miopen_db.apply_miopen_db_defaults(tmp_path, "rocm", None, environ=environ)
        assert changes == {"MIOPEN_CUSTOM_CACHE_DIR": str(_folder(tmp_path))}
        assert environ == {"MIOPEN_USER_DB_PATH": _USER_DB, **changes}
        assert _folder(tmp_path).is_dir()

    @pytest.mark.parametrize("value", [_USER_DB, ""])
    def test_user_values_for_both_leave_everything_alone(self, value, tmp_path):
        environ = dict.fromkeys(_VARS, value)
        assert miopen_db.apply_miopen_db_defaults(tmp_path, None, _HIP, environ=environ) == {}
        assert environ == dict.fromkeys(_VARS, value)
        assert not (tmp_path / "cache").exists()

    def test_second_call_is_noop(self, tmp_path, caplog):
        environ: dict[str, str] = {}
        first = miopen_db.apply_miopen_db_defaults(tmp_path, "rocm", None, environ=environ)
        with caplog.at_level(logging.INFO, logger=miopen_db.__name__):
            assert miopen_db.apply_miopen_db_defaults(tmp_path / "other", None, _HIP, environ=environ) == {}
        assert environ == first
        assert caplog.text == ""

    def test_unusable_folder_keeps_miopen_default(self, tmp_path, caplog):
        (tmp_path / "cache").write_text("not a folder", encoding="utf-8")
        environ: dict[str, str] = {}
        with caplog.at_level(logging.INFO, logger=miopen_db.__name__):
            assert miopen_db.apply_miopen_db_defaults(tmp_path, "rocm", None, environ=environ) == {}
        assert environ == {}
        assert "keeping MIOpen's default" in caplog.text

    def test_defaults_to_process_environment(self, clean_env, tmp_path):
        clean_env.setenv("MIOPEN_CUSTOM_CACHE_DIR", _USER_DB)
        changes = miopen_db.apply_miopen_db_defaults(tmp_path, "rocm", None)
        assert changes == {"MIOPEN_USER_DB_PATH": str(_folder(tmp_path))}
        assert os.environ["MIOPEN_USER_DB_PATH"] == str(_folder(tmp_path))
        assert os.environ["MIOPEN_CUSTOM_CACHE_DIR"] == _USER_DB


def test_module_does_not_import_torch():
    """server.py imports it before backend.main pulls in torch, so it must stay stdlib-only."""
    code = "import sys, backend.utils.miopen_db; sys.exit('torch' in sys.modules)"
    result = subprocess.run([sys.executable, "-c", code], cwd=_BACKEND.parent, capture_output=True, timeout=60)
    assert result.returncode == 0, result.stderr.decode(errors="replace")


def _miopen_vars() -> dict[str, str | None]:
    return {name: os.environ.get(name) for name in _VARS}


def _run_server(monkeypatch, executable: str, argv: list[str]) -> dict:
    """Run server.py's module level (not ``__main__``) and record the MIOpen variables when it imports the app."""
    seen: dict = {}

    def main_module_getattr(name: str):
        if name != "app":
            raise AttributeError(name)
        seen.update(_miopen_vars())
        return object()

    stub_main = types.ModuleType("backend.main")
    stub_main.__getattr__ = main_module_getattr
    monkeypatch.setitem(sys.modules, "backend.main", stub_main)
    monkeypatch.setattr(sys, "executable", executable)
    monkeypatch.setattr(sys, "argv", [executable, *argv])
    monkeypatch.setattr(logging, "basicConfig", lambda **_kwargs: None)
    runpy.run_path(str(_BACKEND / "server.py"), run_name="voicebox_server_under_test")
    assert seen, "server.py did not import the app from backend.main"
    return seen


def _run_dev_main(monkeypatch, argv: list[str], hip: str | None) -> dict:
    """Run ``python -m backend.main`` with stubbed app and torch; record the MIOpen variables at ``uvicorn.run``."""
    seen: dict = {}

    def fake_uvicorn_run(*_args, **_kwargs):
        seen.update(_miopen_vars())

    stub_app = types.ModuleType("backend.app")
    stub_app.app = object()
    stub_torch = types.ModuleType("torch")
    stub_torch.version = types.SimpleNamespace(hip=hip)
    monkeypatch.setitem(sys.modules, "backend.app", stub_app)
    monkeypatch.setitem(sys.modules, "torch", stub_torch)
    monkeypatch.delitem(sys.modules, "backend.main", raising=False)
    monkeypatch.setattr(uvicorn, "run", fake_uvicorn_run)
    monkeypatch.setattr(database, "init_db", lambda: None)
    monkeypatch.setattr(sys, "argv", ["main.py", *argv])
    runpy.run_module("backend.main", run_name="__main__")
    assert seen, "main.py did not reach uvicorn.run"
    return seen


class TestServerEntryPoint:
    """server.py (packaged binary): the variables are set before ``from backend.main import app`` imports torch."""

    @pytest.mark.parametrize("flag", ["separate", "equals"])
    def test_rocm_binary_uses_data_dir(self, flag, clean_env, tmp_path):
        data_dir = tmp_path / "app data"
        argv = ["--data-dir", str(data_dir)] if flag == "separate" else [f"--data-dir={data_dir}"]
        seen = _run_server(clean_env, "C:\\voicebox\\voicebox-server-rocm.exe", [*argv, "--port", "17493"])
        assert seen == dict.fromkeys(_VARS, str(_folder(data_dir)))
        assert _folder(data_dir).is_dir()

    def test_rocm_binary_without_data_dir_uses_default(self, clean_env, tmp_path):
        clean_env.setattr(config, "_data_dir", tmp_path / "default")
        seen = _run_server(clean_env, "C:\\voicebox\\voicebox-server-rocm.exe", [])
        assert seen == dict.fromkeys(_VARS, str(_folder(tmp_path / "default")))

    def test_user_value_wins(self, clean_env, tmp_path):
        clean_env.setenv("MIOPEN_USER_DB_PATH", _USER_DB)
        seen = _run_server(clean_env, "C:\\voicebox\\voicebox-server-rocm.exe", ["--data-dir", str(tmp_path)])
        assert seen == {"MIOPEN_USER_DB_PATH": _USER_DB, "MIOPEN_CUSTOM_CACHE_DIR": str(_folder(tmp_path))}

    @pytest.mark.parametrize(
        "executable", ["C:\\voicebox\\voicebox-server-cuda.exe", "C:\\voicebox\\voicebox-server.exe"]
    )
    def test_other_binaries_are_untouched(self, executable, clean_env, tmp_path):
        seen = _run_server(clean_env, executable, ["--data-dir", str(tmp_path)])
        assert seen == dict.fromkeys(_VARS)
        assert not (tmp_path / "cache").exists()


class TestDevEntryPoint:
    """``python -m backend.main`` (what the bench runs): torch.version.hip decides, before ``uvicorn.run``."""

    def test_rocm_torch_uses_data_dir(self, clean_env, tmp_path):
        data_dir = tmp_path / "data"
        seen = _run_dev_main(clean_env, ["--port", "17599", "--data-dir", str(data_dir)], hip=_HIP)
        assert seen == dict.fromkeys(_VARS, str(_folder(data_dir)))
        assert _folder(data_dir).is_dir()
        assert config.get_data_dir() == data_dir.resolve()

    def test_without_data_dir_uses_default(self, clean_env, tmp_path):
        clean_env.setattr(config, "_data_dir", tmp_path / "default")
        seen = _run_dev_main(clean_env, [], hip=_HIP)
        assert seen == dict.fromkeys(_VARS, str(_folder(tmp_path / "default")))

    def test_user_value_wins(self, clean_env, tmp_path):
        clean_env.setenv("MIOPEN_CUSTOM_CACHE_DIR", _USER_DB)
        seen = _run_dev_main(clean_env, ["--data-dir", str(tmp_path)], hip=_HIP)
        assert seen == {"MIOPEN_USER_DB_PATH": str(_folder(tmp_path)), "MIOPEN_CUSTOM_CACHE_DIR": _USER_DB}

    def test_non_rocm_torch_is_untouched(self, clean_env, tmp_path):
        seen = _run_dev_main(clean_env, ["--data-dir", str(tmp_path)], hip=None)
        assert seen == dict.fromkeys(_VARS)
        assert not (tmp_path / "cache").exists()


def test_binary_and_dev_pick_the_same_folder(clean_env, tmp_path):
    """Same relative ``--data-dir`` from the same working directory: both entry points use one folder."""
    clean_env.chdir(tmp_path)
    from_binary = _run_server(clean_env, "C:\\voicebox\\voicebox-server-rocm.exe", ["--data-dir", "data"])
    for name in (*_VARS, "VOICEBOX_BACKEND_VARIANT"):
        clean_env.delenv(name)
    from_dev = _run_dev_main(clean_env, ["--data-dir", "data"], hip=_HIP)
    assert from_binary == from_dev == dict.fromkeys(_VARS, str(_folder(tmp_path / "data")))
