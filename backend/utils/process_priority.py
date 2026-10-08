"""ROCm on Windows only: run the server process at «above normal» priority (perf-loop plan, Fase 3, task 3.3).

The Qwen3-TTS talker is bound by the CPU thread that dispatches its kernels (perf-loop/notes/3.1-profile.md), so a
desktop app or a background task at normal priority can take that thread's time slices mid-generation. Above normal
(``ABOVE_NORMAL_PRIORITY_CLASS``, one class below high) lets the server win those ties and needs no elevation. ``server.py`` (packaged binary) and ``main.py`` (``python -m backend.main``) call
:func:`raise_process_priority` once at startup. CPU, CUDA, DirectML and MLX servers keep normal priority: a CPU-bound
server above normal would slow the desktop down.

Stdlib only, like ``miopen_db``: nothing here needs torch.
"""

import logging
import os
import sys
from collections.abc import Callable, Mapping

logger = logging.getLogger(__name__)

PRIORITY_ENV = "VOICEBOX_ROCM_PROCESS_PRIORITY"
ROCM_VARIANT = "rocm"
WINDOWS = "win32"
ABOVE_NORMAL_PRIORITY_CLASS = 0x00008000
_DISABLED_VALUES = frozenset({"0", "false", "no", "off"})


def priority_enabled(
    variant: str | None,
    hip: str | None,
    platform: str = sys.platform,
    environ: Mapping[str, str] = os.environ,
) -> bool:
    """Whether the server process should run at above normal priority.

    Args:
        variant: ``VOICEBOX_BACKEND_VARIANT``; ``"rocm"`` in the packaged ROCm binary, unset in development.
        hip: ``torch.version.hip``; None on CPU, CUDA, DirectML and MLX builds.
        platform: ``sys.platform``.
        environ: Process environment, where ``VOICEBOX_ROCM_PROCESS_PRIORITY=0`` keeps normal priority.

    Returns:
        True only on Windows with a ROCm build and unless the user disabled it.
    """
    if platform != WINDOWS or (variant != ROCM_VARIANT and not hip):
        return False
    return environ.get(PRIORITY_ENV, "1").strip().lower() not in _DISABLED_VALUES


def set_above_normal_priority() -> None:
    """``SetPriorityClass(GetCurrentProcess(), ABOVE_NORMAL_PRIORITY_CLASS)`` through ctypes. Windows only.

    The handle types are declared so the ``GetCurrentProcess`` pseudo-handle (-1) reaches ``SetPriorityClass`` as a
    64-bit handle instead of a truncated ``int``.

    Raises:
        OSError: Windows refused the change.
    """
    import ctypes
    from ctypes import wintypes

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.GetCurrentProcess.argtypes = ()
    kernel32.GetCurrentProcess.restype = wintypes.HANDLE
    kernel32.SetPriorityClass.argtypes = (wintypes.HANDLE, wintypes.DWORD)
    kernel32.SetPriorityClass.restype = wintypes.BOOL
    if not kernel32.SetPriorityClass(kernel32.GetCurrentProcess(), ABOVE_NORMAL_PRIORITY_CLASS):
        raise ctypes.WinError(ctypes.get_last_error())


def raise_process_priority(
    variant: str | None,
    hip: str | None,
    platform: str = sys.platform,
    environ: Mapping[str, str] = os.environ,
    set_priority: Callable[[], None] | None = None,
) -> bool:
    """Run this process at above normal priority on Windows with ROCm. Never raises: on failure it stays as it was.

    Args:
        variant: ``VOICEBOX_BACKEND_VARIANT``; ``"rocm"`` in the packaged ROCm binary, unset in development.
        hip: ``torch.version.hip``; None on CPU, CUDA, DirectML and MLX builds.
        platform: ``sys.platform``.
        environ: Process environment, where ``VOICEBOX_ROCM_PROCESS_PRIORITY=0`` keeps normal priority.
        set_priority: Applies the change; defaults to :func:`set_above_normal_priority`.

    Returns:
        True if the priority was raised.
    """
    if not priority_enabled(variant, hip, platform, environ):
        return False
    try:
        (set_priority or set_above_normal_priority)()
    except Exception:  # OSError from Windows or a missing ctypes piece: normal priority still works
        logger.warning("ROCm: could not raise the server process priority, keeping it as it was", exc_info=True)
        return False
    logger.info("ROCm: server process runs at above normal priority (%s=0 disables it)", PRIORITY_ENV)
    return True
