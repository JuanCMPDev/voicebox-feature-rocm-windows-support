"""ROCm only: warm the Qwen3-TTS model up while it loads, so the first generation of the process does not pay for it.

In a new process the first codec decode of each length loads MIOpen kernels (and, with the decoder in Find mode,
``VOICEBOX_CODEC_MIOPEN_IMMEDIATE=0``, runs MIOpen's Find for each length the Find-Db lacks), and the first talker step
loads the rocBLAS/hipBLASLt GEMM kernels.
:func:`warm_up` moves that to the load: it decodes all-zero codes at the lengths the codec decoder will see once they
are padded to multiples of 32 (300, its chunk size, and 32 to 320 in steps of 32) and runs one talker step on a short
text. ``PyTorchTTSBackend._load_model_sync`` calls it, so a generation that triggers the load stays in
``loading_model`` meanwhile.

Nothing the warm-up computes is kept: its outputs are discarded, no weight, config or setting changes, and it runs
through the same (already wrapped) ``speech_tokenizer.decode`` as generation, so the decoder picks the same kernels.
Its sampling consumes RNG, which a seeded generation re-seeds before it samples; the backend publishes the model only
after the warm-up, so no generation samples while it runs. Any failure is logged and swallowed: the model works without
warm-up.
"""

import logging
import os
from collections.abc import Callable, Mapping
from time import perf_counter
from typing import Any

logger = logging.getLogger(__name__)

WARMUP_ENV = "VOICEBOX_QWEN_ROCM_WARMUP"
ROCM_VARIANT = "rocm"
_DISABLED_VALUES = frozenset({"0", "false", "no", "off"})

CODEC_12HZ = "qwen3_tts_tokenizer_12hz"
# Codec decoder lengths (code steps): its 300-step chunk, then 32..320 in steps of 32 (plan, Fase 1).
CODEC_WARMUP_STEPS = (300, *range(32, 321, 32))
TALKER_WARMUP_TEXT = "Hola."
# With max_new_tokens=2 the talker emits one frame, so generate_voice_clone decodes 31 + 1 = 32 steps: a length the
# codec warm-up has already run.
TALKER_WARMUP_REF_STEPS = 31
TALKER_WARMUP_MAX_NEW_TOKENS = 2


def warmup_enabled(
    device: Any,
    variant: str | None,
    hip: str | None,
    environ: Mapping[str, str] = os.environ,
) -> bool:
    """Whether the model should be warmed up when it loads.

    Args:
        device: Torch device the TTS model runs on; ROCm GPUs show up as ``"cuda"``.
        variant: ``VOICEBOX_BACKEND_VARIANT``; ``"rocm"`` in the packaged ROCm binary, unset in development.
        hip: ``torch.version.hip``; None on CPU, CUDA, DirectML and MLX builds.
        environ: Process environment, where ``VOICEBOX_QWEN_ROCM_WARMUP=0`` turns the warm-up off.

    Returns:
        True only on a ROCm GPU and unless the user disabled it.
    """
    if device != "cuda" or (variant != ROCM_VARIANT and not hip):
        return False
    return environ.get(WARMUP_ENV, "1").strip().lower() not in _DISABLED_VALUES


def warm_up_codec(model: Any, steps: tuple[int, ...] = CODEC_WARMUP_STEPS) -> int:
    """Decode all-zero codes of each length in ``steps`` with ``model.model.speech_tokenizer.decode``.

    Args:
        model: Loaded ``Qwen3TTSModel``.
        steps: Code steps of each decode.

    Returns:
        How many lengths were decoded; 0 if the model has no 12 Hz codec decoder.
    """
    tokenizer = getattr(getattr(model, "model", None), "speech_tokenizer", None)
    config = getattr(getattr(tokenizer, "model", None), "config", None)
    quantizers = getattr(getattr(config, "decoder_config", None), "num_quantizers", None)
    if (
        getattr(config, "model_type", None) != CODEC_12HZ
        or not isinstance(quantizers, int)
        or not callable(getattr(tokenizer, "decode", None))
    ):
        logger.info("ROCm warm-up: no 12 Hz codec decoder, codec warm-up skipped")
        return 0
    import torch  # lazy: heavy import

    device = getattr(model, "device", None)
    for length in steps:
        codes = torch.zeros(length, quantizers, dtype=torch.long, device=device)
        tokenizer.decode([{"audio_codes": codes}])
    return len(steps)


def warm_up_talker(
    model: Any,
    text: str = TALKER_WARMUP_TEXT,
    ref_steps: int = TALKER_WARMUP_REF_STEPS,
) -> bool:
    """Run one talker step on ``text`` with a dummy voice-clone prompt, through ``model.generate_voice_clone``.

    The prompt is in ICL mode, like the ones ``create_voice_clone_prompt`` builds for generation: ``ref_steps`` all-zero
    reference codes, a zero speaker embedding and ``text`` as reference transcript. ``max_new_tokens=2`` is a prefill
    plus one decoding step, with its code-predictor run; the rest of the sampling settings are the model's defaults.

    Args:
        model: Loaded ``Qwen3TTSModel`` (Base).
        text: Text to synthesize and reference transcript.
        ref_steps: Code steps of the dummy reference.

    Returns:
        True if the step ran; False if the model is not a Base model with ``generate_voice_clone``.
    """
    inner = getattr(model, "model", None)
    talker_config = getattr(getattr(inner, "talker", None), "config", None)
    hidden_size = getattr(talker_config, "hidden_size", None)
    code_groups = getattr(talker_config, "num_code_groups", None)
    if (
        getattr(inner, "tts_model_type", None) != "base"
        or not callable(getattr(model, "generate_voice_clone", None))
        or not isinstance(hidden_size, int)
        or not isinstance(code_groups, int)
    ):
        logger.info("ROCm warm-up: not a Base model with generate_voice_clone, talker warm-up skipped")
        return False
    import torch  # lazy: heavy import
    from qwen_tts import VoiceClonePromptItem  # lazy: heavy import

    device = getattr(model, "device", None)
    prompt = VoiceClonePromptItem(
        ref_code=torch.zeros(ref_steps, code_groups, dtype=torch.long, device=device),
        ref_spk_embedding=torch.zeros(hidden_size, device=device),
        x_vector_only_mode=False,
        icl_mode=True,
        ref_text=text,
    )
    model.generate_voice_clone(
        text=text,
        language="auto",
        voice_clone_prompt=[prompt],
        max_new_tokens=TALKER_WARMUP_MAX_NEW_TOKENS,
    )
    return True


def warm_up(model: Any, clock: Callable[[], float] = perf_counter) -> dict[str, float | None]:
    """Warm up the codec decoder, then the talker; log how long each took. Never raises.

    The codec goes first so the decode inside the talker step hits a length it has already run.

    Args:
        model: Loaded ``Qwen3TTSModel``, with its decoder already wrapped as generation runs it.
        clock: Time source in seconds.

    Returns:
        Seconds each part took (``"codec"``, ``"talker"``), None for a part that was skipped or failed.
    """
    parts: tuple[tuple[str, Callable[[Any], int | bool]], ...] = (("codec", warm_up_codec), ("talker", warm_up_talker))
    timings: dict[str, float | None] = {}
    for name, run in parts:
        started = clock()
        try:
            ran = run(model)
        except Exception:
            logger.warning("ROCm warm-up: %s warm-up failed, the first generation will pay for it", name, exc_info=True)
            ran = False
        timings[name] = clock() - started if ran else None
    logger.info(
        "ROCm warm-up: codec decoder at %d lengths in %s, one talker step in %s (%s=0 disables it)",
        len(CODEC_WARMUP_STEPS),
        _seconds(timings["codec"]),
        _seconds(timings["talker"]),
        WARMUP_ENV,
    )
    return timings


def _seconds(value: float | None) -> str:
    return "-" if value is None else f"{value:.1f} s"
