"""MIOpen environment defaults for ROCm, applied by ``backend/app.py`` before the first convolution.

On a codec length MIOpen has not seen, its default Find mode (hybrid) times every applicable solver, and the naive
direct-convolution ones take minutes per new length. With them disabled Find only times the GEMM solvers, which cover
every codec shape, so a new length costs seconds and the Find-Db keeps recording the best measured kernel.

Stdlib only: ``app.py`` imports this before ``torch``.
"""

import logging
import os
from collections.abc import Mapping, MutableMapping

logger = logging.getLogger(__name__)

ROCM_VARIANT = "rocm"
NAIVE_CONV_DEFAULTS = {
    "MIOPEN_DEBUG_CONV_DIRECT_NAIVE_CONV_FWD": "0",
    "MIOPEN_DEBUG_CONV_DIRECT_NAIVE_CONV_BWD": "0",
}


def miopen_env_defaults(environ: Mapping[str, str], variant: str | None, hip: str | None) -> dict[str, str]:
    """Decide which MIOpen variables to add to the environment.

    Args:
        environ: Current process environment.
        variant: ``VOICEBOX_BACKEND_VARIANT``; ``"rocm"`` in the packaged ROCm binary, unset in development.
        hip: ``torch.version.hip``; None before torch is imported and on CPU, CUDA, DirectML or MLX builds.

    Returns:
        The variables to set, without the ones already present (a user value always wins); empty off ROCm.
    """
    if variant != ROCM_VARIANT and not hip:
        return {}
    return {name: value for name, value in NAIVE_CONV_DEFAULTS.items() if name not in environ}


def apply_miopen_env_defaults(
    variant: str | None,
    hip: str | None,
    environ: MutableMapping[str, str] = os.environ,
) -> dict[str, str]:
    """Set the MIOpen defaults decided by :func:`miopen_env_defaults` with ``setdefault``.

    MIOpen reads these variables at the first convolution, so this works both before and right after
    ``import torch``. Calling it again is a no-op.

    Args:
        variant: ``VOICEBOX_BACKEND_VARIANT`` or None.
        hip: ``torch.version.hip`` or None.
        environ: Environment to update. Defaults to ``os.environ``.

    Returns:
        The variables that were set.
    """
    changes = miopen_env_defaults(environ, variant, hip)
    for name, value in changes.items():
        environ.setdefault(name, value)
    if changes:
        logger.info(
            "ROCm: disabling MIOpen naive convolution solvers (%s)",
            ", ".join(f"{name}={value}" for name, value in changes.items()),
        )
    return changes
