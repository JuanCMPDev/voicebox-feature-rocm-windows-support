"""ROCm only: keep MIOpen's Find-Db and kernel cache in the app's data dir, at ``{data_dir}/cache/miopen``.

By default MIOpen keeps them in ``%USERPROFILE%\\.miopen``, outside the app: they do not follow the data dir and the
app cannot clear them. ``server.py`` (packaged binary) and ``main.py`` (``python -m backend.main``) set
``MIOPEN_USER_DB_PATH`` and ``MIOPEN_CUSTOM_CACHE_DIR`` with setdefault semantics, so a value the user set wins. MIOpen
reads them at the first convolution, so they only need to be set before a model runs.

Stdlib only: ``server.py`` imports this before ``torch``.
"""

import logging
import os
from collections.abc import MutableMapping, Sequence
from pathlib import Path

logger = logging.getLogger(__name__)

ROCM_VARIANT = "rocm"
DATA_DIR_FLAG = "--data-dir"
MIOPEN_DB_VARS = ("MIOPEN_USER_DB_PATH", "MIOPEN_CUSTOM_CACHE_DIR")


def miopen_db_dir(data_dir: str | Path) -> Path:
    """Folder for MIOpen's Find-Db and kernel cache: ``{data_dir}/cache/miopen``.

    ``data_dir`` is resolved the same way as in ``config.set_data_dir``, so this equals
    ``config.get_cache_dir() / "miopen"`` once the data dir is set.
    """
    return Path(data_dir).resolve() / "cache" / "miopen"


def data_dir_from_argv(argv: Sequence[str]) -> str | None:
    """Value of ``--data-dir`` in ``argv`` before argparse runs (``--data-dir X`` or ``--data-dir=X``; last one wins).

    Args:
        argv: Command-line arguments without the program name (``sys.argv[1:]``).

    Returns:
        The value, or None if the flag is absent or has no value.
    """
    value = None
    args = iter(argv)
    for arg in args:
        if arg == DATA_DIR_FLAG:
            value = next(args, None)
        elif arg.startswith(f"{DATA_DIR_FLAG}="):
            value = arg.partition("=")[2]
    return value


def apply_miopen_db_defaults(
    data_dir: str | Path,
    variant: str | None,
    hip: str | None,
    environ: MutableMapping[str, str] = os.environ,
) -> dict[str, str]:
    """Point MIOpen's Find-Db and kernel cache at ``{data_dir}/cache/miopen`` on ROCm, creating the folder.

    Variables already in ``environ`` are left alone (setdefault), even if empty. If the folder cannot be created,
    nothing is set and MIOpen keeps its default location. Calling it again is a no-op.

    Args:
        data_dir: App data directory (``--data-dir``, or the default from ``config.get_data_dir()``).
        variant: ``VOICEBOX_BACKEND_VARIANT``; ``"rocm"`` in the packaged ROCm binary, unset in development.
        hip: ``torch.version.hip``; None before torch is imported and on CPU, CUDA, DirectML or MLX builds.
        environ: Environment to update. Defaults to ``os.environ``.

    Returns:
        The variables that were set; empty off ROCm.
    """
    if variant != ROCM_VARIANT and not hip:
        return {}
    missing = [name for name in MIOPEN_DB_VARS if name not in environ]
    if not missing:
        return {}
    try:
        folder = miopen_db_dir(data_dir)
        folder.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        logger.warning(
            "ROCm: cannot create the MIOpen cache folder under %s, keeping MIOpen's default: %s", data_dir, e
        )
        return {}
    changes = dict.fromkeys(missing, str(folder))
    for name, value in changes.items():
        environ.setdefault(name, value)
    logger.info("ROCm: MIOpen Find-Db and kernel cache in %s (%s)", folder, ", ".join(changes))
    return changes
