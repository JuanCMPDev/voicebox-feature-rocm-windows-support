"""
Tests for backend/utils/miopen_env.py, the ROCm-only MIOpen defaults that keep the naive conv solvers out of Find
(perf-loop plan, Fase 1, variante B), and for how backend/app.py applies them around ``import torch``.

Runs without a GPU: the decision takes the environment, the backend variant and ``torch.version.hip`` as arguments.

Usage:
    python -m pytest backend/tests/test_miopen_env.py -v
"""

import ast
import logging
import os
import subprocess
import sys
from pathlib import Path

import pytest

from backend.utils import miopen_env

_BACKEND = Path(__file__).resolve().parents[1]
_NAIVE_VARS = {
    "MIOPEN_DEBUG_CONV_DIRECT_NAIVE_CONV_FWD": "0",
    "MIOPEN_DEBUG_CONV_DIRECT_NAIVE_CONV_BWD": "0",
}


class TestMiopenEnvDefaults:
    """The pure decision: (environment, variant, hip) -> variables to set."""

    @pytest.mark.parametrize(
        ("variant", "hip"),
        [("rocm", None), (None, "7.2.53211"), ("rocm", "7.2.53211")],
    )
    def test_rocm_disables_naive_solvers(self, variant, hip):
        assert miopen_env.miopen_env_defaults({}, variant, hip) == _NAIVE_VARS

    @pytest.mark.parametrize(
        ("variant", "hip"),
        [(None, None), ("cpu", None), ("cuda", None), (None, ""), ("ROCm-like", None)],
    )
    def test_non_rocm_paths_are_untouched(self, variant, hip):
        assert miopen_env.miopen_env_defaults({}, variant, hip) == {}

    def test_user_values_win(self):
        environ = {"MIOPEN_DEBUG_CONV_DIRECT_NAIVE_CONV_FWD": "1", "PATH": "C:\\bin"}
        assert miopen_env.miopen_env_defaults(environ, "rocm", None) == {
            "MIOPEN_DEBUG_CONV_DIRECT_NAIVE_CONV_BWD": "0",
        }

    def test_user_empty_value_is_respected(self):
        environ = dict.fromkeys(_NAIVE_VARS, "")
        assert miopen_env.miopen_env_defaults(environ, None, "7.2.53211") == {}

    def test_does_not_mutate_environment(self):
        environ = {"PATH": "C:\\bin"}
        miopen_env.miopen_env_defaults(environ, "rocm", None)
        assert environ == {"PATH": "C:\\bin"}


class TestApplyMiopenEnvDefaults:
    """Applying the decision with setdefault semantics."""

    def test_sets_variables_on_rocm(self, caplog):
        environ = {"PATH": "C:\\bin"}
        with caplog.at_level(logging.INFO, logger=miopen_env.__name__):
            changes = miopen_env.apply_miopen_env_defaults(None, "7.2.53211", environ=environ)
        assert changes == _NAIVE_VARS
        assert environ == {"PATH": "C:\\bin", **_NAIVE_VARS}
        assert "MIOPEN_DEBUG_CONV_DIRECT_NAIVE_CONV_FWD=0" in caplog.text

    def test_keeps_user_values(self):
        environ = {"MIOPEN_DEBUG_CONV_DIRECT_NAIVE_CONV_FWD": "1"}
        miopen_env.apply_miopen_env_defaults("rocm", None, environ=environ)
        assert environ == {
            "MIOPEN_DEBUG_CONV_DIRECT_NAIVE_CONV_FWD": "1",
            "MIOPEN_DEBUG_CONV_DIRECT_NAIVE_CONV_BWD": "0",
        }

    def test_second_call_is_noop(self, caplog):
        environ: dict[str, str] = {}
        miopen_env.apply_miopen_env_defaults("rocm", None, environ=environ)
        with caplog.at_level(logging.INFO, logger=miopen_env.__name__):
            assert miopen_env.apply_miopen_env_defaults("rocm", "7.2.53211", environ=environ) == {}
        assert environ == _NAIVE_VARS
        assert caplog.text == ""

    @pytest.mark.parametrize("variant", [None, "cpu", "cuda"])
    def test_non_rocm_leaves_environment_alone(self, variant, caplog):
        environ = {"PATH": "C:\\bin"}
        with caplog.at_level(logging.INFO, logger=miopen_env.__name__):
            assert miopen_env.apply_miopen_env_defaults(variant, None, environ=environ) == {}
        assert environ == {"PATH": "C:\\bin"}
        assert caplog.text == ""

    def test_defaults_to_process_environment(self, monkeypatch):
        # setenv before delenv makes monkeypatch record the original value, so the write below is undone.
        monkeypatch.setenv("MIOPEN_DEBUG_CONV_DIRECT_NAIVE_CONV_FWD", "1")
        monkeypatch.delenv("MIOPEN_DEBUG_CONV_DIRECT_NAIVE_CONV_FWD")
        monkeypatch.setenv("MIOPEN_DEBUG_CONV_DIRECT_NAIVE_CONV_BWD", "1")
        changes = miopen_env.apply_miopen_env_defaults("rocm", None)
        assert changes == {"MIOPEN_DEBUG_CONV_DIRECT_NAIVE_CONV_FWD": "0"}
        assert os.environ["MIOPEN_DEBUG_CONV_DIRECT_NAIVE_CONV_FWD"] == "0"
        assert os.environ["MIOPEN_DEBUG_CONV_DIRECT_NAIVE_CONV_BWD"] == "1"


def test_module_does_not_import_torch():
    """app.py imports it before torch, so it must stay stdlib-only."""
    code = "import sys, backend.utils.miopen_env; sys.exit('torch' in sys.modules)"
    result = subprocess.run([sys.executable, "-c", code], cwd=_BACKEND.parent, capture_output=True, timeout=60)
    assert result.returncode == 0, result.stderr.decode(errors="replace")


def _apply_call(node: ast.stmt) -> ast.Call | None:
    if (
        isinstance(node, ast.Expr)
        and isinstance(node.value, ast.Call)
        and isinstance(node.value.func, ast.Name)
        and node.value.func.id == "apply_miopen_env_defaults"
    ):
        return node.value
    return None


def test_app_applies_defaults_around_torch_import():
    """backend/app.py: once before ``import torch`` (packaged binary) and once right after with torch.version.hip."""
    body = ast.parse((_BACKEND / "app.py").read_text(encoding="utf-8")).body
    torch_index = next(
        i for i, node in enumerate(body) if isinstance(node, ast.Import) and any(a.name == "torch" for a in node.names)
    )
    calls = [(i, call) for i, node in enumerate(body) if (call := _apply_call(node)) is not None]
    assert [i for i, _ in calls] == [torch_index - 1, torch_index + 1]
    before, after = (call for _, call in calls)
    assert "VOICEBOX_BACKEND_VARIANT" in ast.unparse(before.args[0])
    assert "VOICEBOX_BACKEND_VARIANT" in ast.unparse(after.args[0])
    after_hip = next(kw.value for kw in after.keywords if kw.arg == "hip")
    assert "torch.version" in ast.unparse(after_hip)
    assert "hip" in ast.unparse(after_hip)
