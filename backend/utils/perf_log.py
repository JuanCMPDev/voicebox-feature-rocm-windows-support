"""Per-stage timing lines (``VBPERF {json}``) for the TTS perf loop, active only with ``VOICEBOX_PERF_LOG=1``.

Each stage logs one ``VBPERF {"gen_id": ..., "stage": ..., "ms": ...}`` line through ``logging`` (stderr), which
Tauri forwards as ``server-log`` and ``scripts/bench_tts.py`` parses (perf-loop/contracts.md, section 4). The GPU is
synchronized before every timestamp so a stage includes the kernels it queued.

Stages: ``reference`` (voice prompt, ``cached``), ``talker`` (``steps``, ``steps_per_s``), ``codec`` (``T``,
``chunks``), ``normalize`` and ``effects`` (only when they run), ``save`` (WAVs and versions, effects included) and
``total`` (the whole generation). ``env`` is logged once per model load with the settings that decide MIOpen's
behaviour. Without the variable nothing is synchronized, timed, wrapped or logged.
"""

import contextvars
import functools
import json
import logging
import os
import sys
from collections.abc import Callable, Mapping
from pathlib import Path
from time import perf_counter
from typing import Any

logger = logging.getLogger(__name__)

PERF_LOG_ENV = "VOICEBOX_PERF_LOG"
LINE_PREFIX = "VBPERF "
CODE_PATH = Path(__file__).resolve().parents[1]
# Always reported (None when unset) so env_checks shows which MIOpen knobs were left at their defaults.
MIOPEN_KNOWN_VARS = (
    "MIOPEN_FIND_MODE",
    "MIOPEN_FIND_ENFORCE",
    "MIOPEN_USER_DB_PATH",
    "MIOPEN_CUSTOM_CACHE_DIR",
    "MIOPEN_SYSTEM_DB_PATH",
    "MIOPEN_DEBUG_CONV_DIRECT_NAIVE_CONV_FWD",
    "MIOPEN_DEBUG_CONV_DIRECT_NAIVE_CONV_BWD",
    "MIOPEN_DEBUG_CONV_GEMM",
    "MIOPEN_LOG_LEVEL",
)

# Generation the current task works for; asyncio.to_thread copies it into the worker thread.
_generation_id: contextvars.ContextVar[str | None] = contextvars.ContextVar("vbperf_generation_id", default=None)


def is_enabled() -> bool:
    """Whether VBPERF instrumentation is on (``VOICEBOX_PERF_LOG=1``)."""
    return os.environ.get(PERF_LOG_ENV) == "1"


def set_generation_id(generation_id: str | None) -> contextvars.Token:
    """Tag the VBPERF lines of the current task, and of the threads it starts, with ``generation_id``."""
    return _generation_id.set(generation_id)


def reset_generation_id(token: contextvars.Token) -> None:
    """Undo :func:`set_generation_id`."""
    _generation_id.reset(token)


def format_line(payload: Mapping[str, Any]) -> str:
    """``VBPERF `` plus ``payload`` as single-line JSON (values JSON cannot encode are stringified)."""
    return LINE_PREFIX + json.dumps(payload, default=str)


def emit(stage: str, **fields: Any) -> None:
    """Log the VBPERF line of ``stage`` for the current generation."""
    logger.info("%s", format_line({"gen_id": _generation_id.get(), "stage": stage, **fields}))


def emit_env(info: Mapping[str, Any]) -> None:
    """Log the ``env`` line (no generation id: it describes the process, not a generation)."""
    logger.info("%s", format_line({"stage": "env", **info}))


def cuda_synchronize() -> None:
    """Wait for the GPU work queued so far; no-op without a CUDA/ROCm device."""
    import torch  # lazy: heavy import

    if torch.cuda.is_available():
        torch.cuda.synchronize()


class StageTimer:
    """Times one stage and logs its VBPERF line when it stops; inert when instrumentation is off.

    ``fields`` collects extra values for the line; a numeric ``steps`` field also yields ``steps_per_s``. As a
    context manager the line is logged only if the block finishes without raising.
    """

    def __init__(
        self,
        stage: str,
        *,
        clock: Callable[[], float] | None = None,
        sync: Callable[[], None] | None = None,
        enabled: bool | None = None,
    ) -> None:
        self.stage = stage
        self.fields: dict[str, Any] = {}
        self.ms: float | None = None
        self._clock = clock or perf_counter
        self._sync = sync or cuda_synchronize
        self._enabled = is_enabled() if enabled is None else enabled
        self._started_at: float | None = None

    def start(self) -> "StageTimer":
        """Synchronize the GPU and take the start timestamp."""
        if self._enabled:
            self._sync()
            self._started_at = self._clock()
        return self

    def stop(self, **fields: Any) -> float | None:
        """Synchronize the GPU, take the end timestamp and log the line; returns the elapsed ms or None."""
        started_at, self._started_at = self._started_at, None  # closed even if the sync below raises
        if started_at is None:
            return None
        self._sync()
        elapsed_ms = (self._clock() - started_at) * 1000.0
        self.ms = elapsed_ms
        payload = {"ms": round(elapsed_ms, 3), **self.fields, **fields}
        steps = payload.get("steps")
        if isinstance(steps, int | float) and elapsed_ms > 0:
            payload["steps_per_s"] = round(steps / (elapsed_ms / 1000.0), 3)
        emit(self.stage, **payload)
        return elapsed_ms

    def __enter__(self) -> "StageTimer":
        return self.start()

    def __exit__(self, exc_type, exc, tb) -> None:
        if exc_type is None:
            self.stop()
        else:
            self._started_at = None


def stage(name: str) -> StageTimer:
    """Timer for stage ``name``: ``with perf_log.stage("save"):`` or ``.start()`` ... ``.stop()``."""
    return StageTimer(name)


def miopen_env(environ: Mapping[str, str] | None = None) -> dict[str, str | None]:
    """Every ``MIOPEN_*`` variable the process sees, plus the known ones as None when unset."""
    environ = os.environ if environ is None else environ
    values: dict[str, str | None] = {name: environ.get(name) for name in MIOPEN_KNOWN_VARS}
    values.update({name: value for name, value in sorted(environ.items()) if name.upper().startswith("MIOPEN_")})
    return values


def miopen_user_db_path(environ: Mapping[str, str] | None = None, *, platform: str | None = None) -> str:
    """MIOpen's user Find-Db folder: ``MIOPEN_USER_DB_PATH`` if set, else MIOpen's default for the platform."""
    environ = os.environ if environ is None else environ
    configured = environ.get("MIOPEN_USER_DB_PATH")
    if configured:
        return str(Path(configured).expanduser().absolute())
    if (platform or sys.platform) == "win32":
        return str(Path.home() / ".miopen" / "db")
    return str(Path.home() / ".config" / "miopen")


def _attn_implementation(module: Any) -> str | None:
    return getattr(getattr(module, "config", None), "_attn_implementation", None)


def qwen_env_info(model: Any, device: str) -> dict[str, Any]:
    """Payload of the ``env`` line for a loaded ``Qwen3TTSModel``."""
    import torch  # lazy: heavy import

    inner = getattr(model, "model", None)
    codec = getattr(getattr(inner, "speech_tokenizer", None), "model", None)
    codec_dtype = getattr(codec, "dtype", None)
    hip = torch.version.hip
    return {
        "torch": torch.__version__,
        "hip": hip,
        "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
        "miopen_immediate": getattr(getattr(torch.backends, "miopen", None), "immediate", None),
        "attn_implementation": _attn_implementation(inner),
        "attn_implementation_talker": _attn_implementation(getattr(inner, "talker", None)),
        "codec_dtype": None if codec_dtype is None else str(codec_dtype),
        "miopen_env": miopen_env(),
        "miopen_user_db_path": miopen_user_db_path() if hip else None,
        "code_path": str(CODE_PATH),
        "device": device,
        "backend_variant": os.environ.get("VOICEBOX_BACKEND_VARIANT"),
    }


def _talker_steps(result: Any) -> int | None:
    """Codec frames the talker produced (EOS excluded), summed over the batch."""
    try:
        return sum(int(codes.shape[0]) for codes in result[0])
    except (TypeError, IndexError, AttributeError):
        return None


def _codec_steps(encoded: Any) -> int | None:
    """Code steps handed to the codec (reference + generated), summed over a list of ``{"audio_codes": (T, Q)}``."""
    total = 0
    for item in encoded if isinstance(encoded, list) else [encoded]:
        codes = item.get("audio_codes") if isinstance(item, dict) else None
        shape = getattr(codes, "shape", None)
        if shape is None or len(shape) != 2:
            return None
        total += int(shape[0])
    return total


def instrument_qwen_model(model: Any) -> None:
    """Log a ``talker`` line per ``model.model.generate`` call and a ``codec`` line per ``speech_tokenizer.decode``.

    The wrappers only add GPU synchronization and timestamps around the original calls: arguments, results and RNG
    use are untouched. ``chunked_decode`` runs the codec decoder once per chunk, so a forward pre-hook that only
    counts calls gives ``chunks``.
    """
    import torch  # lazy: heavy import

    inner = getattr(model, "model", None)
    tokenizer = getattr(inner, "speech_tokenizer", None)
    if not callable(getattr(inner, "generate", None)) or not callable(getattr(tokenizer, "decode", None)):
        logger.warning("VBPERF: model has no talker generate / codec decode to instrument")
        return

    original_generate = inner.generate
    original_decode = tokenizer.decode
    decoder = getattr(getattr(tokenizer, "model", None), "decoder", None)
    counts_chunks = isinstance(decoder, torch.nn.Module)
    decoder_calls = [0]

    def _count_decoder_call(_module, _args) -> None:
        decoder_calls[0] += 1

    if counts_chunks:
        decoder.register_forward_pre_hook(_count_decoder_call)

    @functools.wraps(original_generate)
    def timed_generate(*args, **kwargs):
        with stage("talker") as timer:
            result = original_generate(*args, **kwargs)
            timer.fields["steps"] = _talker_steps(result)
        return result

    @functools.wraps(original_decode)
    def timed_decode(encoded, *args, **kwargs):
        calls_before = decoder_calls[0]
        with stage("codec") as timer:
            result = original_decode(encoded, *args, **kwargs)
            timer.fields["T"] = _codec_steps(encoded)
            timer.fields["chunks"] = decoder_calls[0] - calls_before if counts_chunks else None
        return result

    inner.generate = timed_generate
    tokenizer.decode = timed_decode
