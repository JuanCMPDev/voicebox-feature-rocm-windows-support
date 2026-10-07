"""ROCm only: run the Qwen3-TTS codec decoder in MIOpen's immediate mode, and nothing else.

In its default Find mode MIOpen times every applicable solver for each convolution shape it has not seen, and the codec
decoder gets a new shape with every new output length: minutes per new phrase on gfx1101. Immediate mode
(``torch.backends.miopen.immediate``) picks the solver from the Find-Db or MIOpen's heuristics without timing, which
for the decoder means GEMM kernels in about a second. It also changes the kernels of the speaker encoder and the codec
encoder, and with them the voice prompt and the sampled tokens, so it is switched on only around
``speech_tokenizer.decode``, which runs after sampling: the waveform then differs from Find mode only by rounding.
"""

import functools
import logging
import os
import threading
from collections.abc import Mapping
from typing import Any

logger = logging.getLogger(__name__)

CODEC_IMMEDIATE_ENV = "VOICEBOX_CODEC_MIOPEN_IMMEDIATE"
ROCM_VARIANT = "rocm"
_DISABLED_VALUES = frozenset({"0", "false", "no", "off"})

# torch.backends.miopen.immediate is process-wide, not per thread. Requests outside the generation queue (e.g. the
# streaming endpoint) can build a voice prompt while another thread decodes, so the prompt and the immediate-mode
# decode hold the same lock: the speaker encoder and codec encoder never run while the flag is on.
_immediate_lock = threading.RLock()


def codec_immediate_enabled(
    device: Any,
    variant: str | None,
    hip: str | None,
    environ: Mapping[str, str] = os.environ,
) -> bool:
    """Whether the codec decoder should run in MIOpen's immediate mode.

    Args:
        device: Torch device the TTS model runs on; ROCm GPUs show up as ``"cuda"``.
        variant: ``VOICEBOX_BACKEND_VARIANT``; ``"rocm"`` in the packaged ROCm binary, unset in development.
        hip: ``torch.version.hip``; None on CPU, CUDA, DirectML and MLX builds.
        environ: Process environment, where ``VOICEBOX_CODEC_MIOPEN_IMMEDIATE=0`` turns the feature off.

    Returns:
        True only on a ROCm GPU and unless the user disabled it.
    """
    if device != "cuda" or (variant != ROCM_VARIANT and not hip):
        return False
    return environ.get(CODEC_IMMEDIATE_ENV, "1").strip().lower() not in _DISABLED_VALUES


def use_immediate_mode_for_codec(model: Any, miopen: Any = None) -> bool:
    """Wrap ``model.model.speech_tokenizer.decode`` so it runs with ``miopen.immediate = True``.

    The previous value is restored when decode returns or raises. Arguments and results pass through untouched, and
    the wrapper composes with :func:`backend.utils.perf_log.instrument_qwen_model` in either order. The model's
    ``create_voice_clone_prompt`` is wrapped too, only to take the same lock (see ``_immediate_lock``).

    Args:
        model: Loaded ``Qwen3TTSModel``.
        miopen: Object holding the ``immediate`` flag; defaults to ``torch.backends.miopen``.

    Returns:
        True if the decoder was wrapped, False if the model has no codec decode.
    """
    tokenizer = getattr(getattr(model, "model", None), "speech_tokenizer", None)
    original_decode = getattr(tokenizer, "decode", None)
    if not callable(original_decode):
        logger.warning("ROCm: model has no codec decode, MIOpen immediate mode not applied")
        return False
    if miopen is None:
        import torch  # lazy: heavy import

        miopen = torch.backends.miopen

    @functools.wraps(original_decode)
    def decode_in_immediate_mode(*args, **kwargs):
        with _immediate_lock:
            previous = miopen.immediate
            miopen.immediate = True
            try:
                return original_decode(*args, **kwargs)
            finally:
                miopen.immediate = previous

    tokenizer.decode = decode_in_immediate_mode

    original_prompt = getattr(model, "create_voice_clone_prompt", None)
    if callable(original_prompt):

        @functools.wraps(original_prompt)
        def prompt_outside_immediate_mode(*args, **kwargs):
            with _immediate_lock:
                return original_prompt(*args, **kwargs)

        model.create_voice_clone_prompt = prompt_outside_immediate_mode

    logger.info("ROCm: codec decoder runs in MIOpen immediate mode (%s=0 disables it)", CODEC_IMMEDIATE_ENV)
    return True
