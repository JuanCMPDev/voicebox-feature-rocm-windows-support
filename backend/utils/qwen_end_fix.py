"""Qwen3-TTS: let the talker finish the last word before it ends a phrase (perf-loop task 4.4).

Short phrases ended cut. Measured on the bench phrases (perf-loop/notes/4.4.md), the talker ends a phrase in two bad
ways: with ``codec_eos`` at probability ~1 right after a frame that still has voice, so the decay of the last syllable,
which the next frame would carry, is never generated (the codec decoder is causal); or with a ``codec_eos`` drawn by
chance at a probability of 1-7 % in the middle of the last word. Two changes, one to the talker's inputs and one to its
stopping rule; ``VOICEBOX_QWEN_END_FIX=0`` turns both off and leaves generation as it was:

- Inputs: ``generate_voice_clone(non_streaming_mode=True)``, the layout ``generate_voice_design`` and
  ``generate_custom_voice`` use by default: the whole text (reference and target) first, then the reference codes,
  instead of the text laid over the first reference codes ("simulated streaming"). With it the talker more often
  generates a decaying frame before it ends.
- Stopping rule: :func:`install` gives the talker's ``generate`` an :class:`EosHoldProcessor`, which, per row, removes
  ``codec_eos`` at the first :data:`HOLD_FRAMES` steps where it is likely (probability at least :data:`EOS_LIKELY`
  under the sampling distribution), so the model generates that frame itself, and, while the row is shorter than
  :data:`GUARD_RATIO` times the frames the reference's own rhythm gives its text, at the steps where it is not likely,
  so the phrase cannot end on a chance draw. Past that length unlikely ``codec_eos`` is allowed again, as before, so a
  row that never makes it likely still ends.

Nothing else changes: same weights, same sampling settings (``top_k``, ``top_p``, ``temperature``,
``repetition_penalty``, ``subtalker_*``, ``max_new_tokens``, which still bounds every row), same reference.
"""

import contextlib
import functools
import logging
import math
import os
import threading
from collections.abc import Iterator, Mapping, Sequence
from typing import Any

logger = logging.getLogger(__name__)

END_FIX_ENV = "VOICEBOX_QWEN_END_FIX"
_DISABLED_VALUES = frozenset({"0", "false", "no", "off"})

# Steps per row at which a likely codec_eos is held back (one 80 ms frame each).
HOLD_FRAMES = 1
# Probability of codec_eos, under the sampling distribution, from which it counts as likely.
EOS_LIKELY = 0.1
# Length, in frames the reference's rhythm gives the text, up to which an unlikely codec_eos is removed.
GUARD_RATIO = 1.5
# Extra generate_voice_clone arguments.
GENERATE_KWARGS: dict[str, Any] = {"non_streaming_mode": True}

# Per-row guards of the generate_voice_clone call running on this thread (set by :func:`phrase_end`).
_call = threading.local()


def end_fix_enabled(environ: Mapping[str, str] = os.environ) -> bool:
    """Whether the end-of-phrase fix applies; on by default, ``VOICEBOX_QWEN_END_FIX=0`` turns it off."""
    return environ.get(END_FIX_ENV, "1").strip().lower() not in _DISABLED_VALUES


def guard_frames(texts: Sequence[str], items: Sequence[Any], ratio: float = GUARD_RATIO) -> list[int]:
    """Per text, the length in frames up to which the talker may not end it on an unlikely ``codec_eos``.

    ``ratio`` times the frames the text gets at the reference's rhythm: frames of the item's ``ref_code`` per
    character of its ``ref_text``, times the characters of the text. 0 (no guard) for a row whose item has no
    reference codes or transcript. A single item serves every text, as ``generate_voice_clone`` does.

    Args:
        texts: Texts of one ``generate_voice_clone`` call, in order.
        items: Their ``VoiceClonePromptItem`` (one per text, or one for all).
        ratio: Multiple of the reference-rhythm length.

    Returns:
        One guard per text.
    """
    if not isinstance(items, list | tuple):
        return [0] * len(texts)
    if len(items) == 1 and len(texts) > 1:
        items = list(items) * len(texts)
    if len(items) != len(texts):
        return [0] * len(texts)
    guards = []
    for text, item in zip(texts, items, strict=True):
        ref_text = getattr(item, "ref_text", None)
        shape = getattr(getattr(item, "ref_code", None), "shape", None)
        if not isinstance(ref_text, str) or not ref_text or not shape or not isinstance(text, str):
            guards.append(0)
            continue
        guards.append(math.ceil(ratio * int(shape[0]) * len(text) / len(ref_text)))
    return guards


@contextlib.contextmanager
def phrase_end(
    texts: Sequence[str],
    items: Sequence[Any],
    environ: Mapping[str, str] = os.environ,
) -> Iterator[dict[str, Any]]:
    """Context for one ``generate_voice_clone`` call: yields its extra arguments; the talker gets each row's guard.

    With the fix off it yields no arguments and the talker no guards, so the call runs as before.

    Args:
        texts: Texts of the call, in order.
        items: Their voice prompt items (one per text, or one for all).
        environ: Process environment, where ``VOICEBOX_QWEN_END_FIX=0`` turns the fix off.
    """
    if not end_fix_enabled(environ):
        yield {}
        return
    previous = getattr(_call, "guards", None)
    _call.guards = guard_frames(texts, items)
    try:
        yield dict(GENERATE_KWARGS)
    finally:
        _call.guards = previous


def current_guards() -> list[int] | None:
    """Per-row guards of the :func:`phrase_end` call running on this thread, or None outside one."""
    return getattr(_call, "guards", None)


class EosHoldProcessor:
    """Logits processor for the talker's ``codec_eos``: holds back the first likely one, rules out chance ones.

    At every step it computes, per row, the probability of ``eos_token_id`` under the distribution the talker samples
    from (``scores`` after the earlier processors, divided by ``temperature`` and cut to the ``top_k`` best, as HF's
    warpers do next). It sets ``codec_eos`` to ``-inf`` in a row:

    - where it is likely (probability at least ``likely``), at the first ``hold_frames`` such steps of the row;
    - where it is not likely, while the row has generated fewer frames than its entry in ``guards``.

    The rest of the scores, and every other row and step, are left as they are. Only GPU ops, no host reads.

    Args:
        eos_token_id: The talker's ``codec_eos`` id.
        hold_frames: Steps per row at which a likely ``codec_eos`` is held back.
        likely: Probability from which ``codec_eos`` counts as likely.
        temperature: Sampling temperature, or None when the talker does not sample.
        top_k: Sampling top-k, or None when the talker does not sample or keeps every token.
        guards: Per row, frames generated before which an unlikely ``codec_eos`` is removed; None (or a length
            other than the batch's) for no guard.
    """

    def __init__(
        self,
        eos_token_id: int,
        hold_frames: int = HOLD_FRAMES,
        likely: float = EOS_LIKELY,
        temperature: float | None = None,
        top_k: int | None = None,
        guards: Sequence[int] | None = None,
    ):
        self.eos_token_id = eos_token_id
        self.hold_frames = hold_frames
        self.likely = likely
        self.temperature = temperature
        self.top_k = top_k
        self.guards = None if guards is None else [int(g) for g in guards]
        self.held: Any = None  # per-row count of held steps, created at the first call
        self._guard: Any = None  # guards as a tensor on the scores' device
        self._start: int | None = None  # input_ids length at the first call: the first generated frame

    def eos_probability(self, scores: Any) -> Any:
        """Per-row probability of ``codec_eos`` under the sampling distribution, (batch,) float32."""
        import torch  # lazy: heavy import

        logits = scores.float()
        if self.temperature is not None and self.temperature != 1.0:
            logits = logits / self.temperature
        if self.top_k is not None and 0 < self.top_k < logits.shape[-1]:
            kth = torch.topk(logits, self.top_k, dim=-1).values[..., -1:]
            logits = logits.masked_fill(logits < kth, float("-inf"))
        return torch.softmax(logits, dim=-1)[:, self.eos_token_id]

    def __call__(self, input_ids: Any, scores: Any) -> Any:
        import torch  # lazy: heavy import

        if self.held is None:
            rows = scores.shape[0]
            guards = self.guards if self.guards is not None and len(self.guards) == rows else [0] * rows
            self.held = torch.zeros(rows, dtype=torch.int64, device=scores.device)
            self._guard = torch.tensor(guards, dtype=torch.int64, device=scores.device)
            self._start = input_ids.shape[1]
        step = input_ids.shape[1] - self._start  # frames this call has generated so far
        likely = self.eos_probability(scores) >= self.likely
        can_hold = self.held < self.hold_frames
        remove = torch.where(likely, can_hold, self._guard > step)
        self.held += likely & can_hold
        scores[:, self.eos_token_id].masked_fill_(remove, float("-inf"))
        return scores


def install(model: Any) -> bool:
    """Make ``model``'s talker ``generate`` run with an :class:`EosHoldProcessor`. Installing twice is a no-op.

    The processor goes after the logits processors the call already has; HF runs it after its own (repetition
    penalty, suppressed tokens, minimum length) and before temperature and top-k. Its temperature and top-k are the
    call's, so it sees the distribution the talker samples from; its guards are the ones :func:`phrase_end` set on
    this thread (none outside it). The original ``generate`` stays available as ``generate.original``.

    Args:
        model: Loaded ``Qwen3TTSModel``.

    Returns:
        True if the processor is installed, False if the model has no talker ``generate`` or ``codec_eos`` id.
    """
    inner = getattr(model, "model", None)
    talker = getattr(inner, "talker", None)
    original_generate = getattr(talker, "generate", None)
    talker_config = getattr(getattr(inner, "config", None), "talker_config", None)
    eos_token_id = getattr(talker_config, "codec_eos_token_id", None)
    if not callable(original_generate) or not isinstance(eos_token_id, int):
        logger.info("Qwen end fix: model has no talker generate or codec_eos id, end rule not installed")
        return False
    if getattr(original_generate, "end_hold", False):
        return True

    @functools.wraps(original_generate)
    def generate_with_end_hold(*args, **kwargs):
        from transformers import LogitsProcessorList  # lazy: heavy import

        sampling = bool(kwargs.get("do_sample", True))
        processor = EosHoldProcessor(
            eos_token_id,
            hold_frames=HOLD_FRAMES,
            likely=EOS_LIKELY,
            temperature=kwargs.get("temperature") if sampling else None,
            top_k=kwargs.get("top_k") if sampling else None,
            guards=current_guards(),
        )
        kwargs["logits_processor"] = LogitsProcessorList([*(kwargs.get("logits_processor") or []), processor])
        return original_generate(*args, **kwargs)

    generate_with_end_hold.original = original_generate
    generate_with_end_hold.end_hold = True
    talker.generate = generate_with_end_hold
    logger.info(
        "Qwen end fix: the talker holds back its first likely codec_eos and rules out chance ones (%s=0 disables it)",
        END_FIX_ENV,
    )
    return True
