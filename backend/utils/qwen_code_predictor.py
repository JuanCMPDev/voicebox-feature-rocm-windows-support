"""ROCm only: sample the Qwen3-TTS code predictor's tokens in a plain loop instead of HF ``generate``.

Every talker step calls ``talker.code_predictor.generate`` for the 15 residual codebooks of the frame
(``Qwen3TTSTalkerForConditionalGeneration.forward``). Around its 15 forwards HF ``generate`` rebuilds its config, inputs,
attention mask, processors and stopping criteria, and reads the GPU back twice per token (the 2D mask's ``.all()`` and
the ``unfinished_sequences.max() == 0`` loop check): ~18 ms and 30 host waits of a ~100 ms step on gfx1101
(perf-loop/notes/3.1-profile.md). :func:`install` replaces it with a loop that does what ``generate`` does with no host
reads:

- The same forwards, through the module itself: a prefill on the input embeddings, then one forward per token on the
  token just sampled, on a ``DynamicCache`` built as ``generate`` builds it, with the same ``cache_position`` and
  ``position_ids``. No attention mask: with a cache, an all-ones 2D mask and no mask both give the same causal mask
  (none at all under SDPA), so attention gets the same inputs without the ``.all()``.
- The same sampling: last-position logits copied to float32, HF's own temperature, top-k and top-p warpers in HF's order
  and under HF's conditions, then softmax and one ``torch.multinomial`` per token (argmax without sampling). The RNG is
  consumed the same way, so the tokens are the ones ``generate`` would sample.
- Exactly ``max_new_tokens`` tokens: without an EOS token ``generate`` only stops at that length.

Calls with anything else in the effective generation config (EOS, repetition penalty, beams, other processors, extra
arguments) go to the original ``generate`` unchanged.
"""

import copy
import functools
import logging
import os
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)

CODE_PREDICTOR_LOOP_ENV = "VOICEBOX_QWEN_CODE_PREDICTOR_LOOP"
ROCM_VARIANT = "rocm"
_DISABLED_VALUES = frozenset({"0", "false", "no", "off"})

# Arguments the talker passes to code_predictor.generate; any other one (attention_mask, logits_processor, streamer,
# generation_config, ...) sends the call to HF generate.
_LOOP_ARGUMENTS = frozenset(
    {
        "inputs_embeds",
        "max_new_tokens",
        "do_sample",
        "top_p",
        "top_k",
        "temperature",
        "output_hidden_states",
        "return_dict_in_generate",
    }
)
# Generation settings that may differ from GenerationConfig() defaults: the loop reproduces them (max_length is
# overridden by max_new_tokens, the last two are metadata). Anything else non-default sends the call to HF generate.
_LOOP_SETTINGS = frozenset(
    {
        "do_sample",
        "temperature",
        "top_k",
        "top_p",
        "max_new_tokens",
        "max_length",
        "output_hidden_states",
        "return_dict_in_generate",
        "_from_model_config",
        "transformers_version",
    }
)


def loop_enabled(
    device: Any,
    variant: str | None,
    hip: str | None,
    environ: Mapping[str, str] = os.environ,
) -> bool:
    """Whether the code predictor should sample in :func:`install`'s loop instead of HF ``generate``.

    Args:
        device: Torch device the TTS model runs on; ROCm GPUs show up as ``"cuda"``.
        variant: ``VOICEBOX_BACKEND_VARIANT``; ``"rocm"`` in the packaged ROCm binary, unset in development.
        hip: ``torch.version.hip``; None on CPU, CUDA, DirectML and MLX builds.
        environ: Process environment, where ``VOICEBOX_QWEN_CODE_PREDICTOR_LOOP=0`` turns the loop off.

    Returns:
        True only on a ROCm GPU and unless the user disabled it.
    """
    if device != "cuda" or (variant != ROCM_VARIANT and not hip):
        return False
    return environ.get(CODE_PREDICTOR_LOOP_ENV, "1").strip().lower() not in _DISABLED_VALUES


@dataclass(frozen=True)
class _SamplingPlan:
    """What ``generate`` would do for one set of arguments, when the loop can do the same."""

    max_new_tokens: int
    do_sample: bool
    warpers: tuple[Any, ...]
    output_hidden_states: bool
    return_dict_in_generate: bool


def _sampling_plan(predictor: Any, arguments: Mapping[str, Any]) -> _SamplingPlan | None:
    """The loop's plan for ``generate(inputs_embeds=..., **arguments)``, or None if ``generate`` would do more.

    The effective config is built as ``generate`` builds it without a ``generation_config`` argument: a copy of the
    model's ``generation_config`` updated with the arguments. The warpers are HF's classes, appended in the order and
    under the conditions of ``GenerationMixin._get_logits_processor``.
    """
    from transformers import GenerationConfig  # lazy: heavy import
    from transformers.generation.logits_process import (  # lazy: heavy import
        TemperatureLogitsWarper,
        TopKLogitsWarper,
        TopPLogitsWarper,
    )

    base = getattr(predictor, "generation_config", None)
    if not isinstance(base, GenerationConfig):
        return None
    config = copy.deepcopy(base)
    if config.update(**arguments):
        return None
    if not set(config.to_diff_dict()) <= _LOOP_SETTINGS:
        return None
    max_new_tokens = config.max_new_tokens
    if not isinstance(max_new_tokens, int) or isinstance(max_new_tokens, bool) or max_new_tokens < 1:
        return None

    warpers: list[Any] = []
    if config.do_sample:
        if config.temperature is not None and config.temperature != 1.0:
            warpers.append(TemperatureLogitsWarper(config.temperature))
        if config.top_k is not None and config.top_k != 0:
            warpers.append(TopKLogitsWarper(top_k=config.top_k, min_tokens_to_keep=1))
        if config.top_p is not None and config.top_p < 1.0:
            warpers.append(TopPLogitsWarper(top_p=config.top_p, min_tokens_to_keep=1))
    return _SamplingPlan(
        max_new_tokens=max_new_tokens,
        do_sample=bool(config.do_sample),
        warpers=tuple(warpers),
        output_hidden_states=bool(config.output_hidden_states),
        return_dict_in_generate=bool(config.return_dict_in_generate),
    )


def sample_with_loop(predictor: Any, inputs_embeds: Any, plan: _SamplingPlan) -> Any:
    """Sample ``plan.max_new_tokens`` tokens as HF ``generate``'s sampling loop does, without reading the GPU.

    Args:
        predictor: ``Qwen3TTSTalkerCodePredictorModelForConditionalGeneration``.
        inputs_embeds: Prompt embeddings, (batch, positions, hidden) with at least 2 positions.
        plan: From :func:`_sampling_plan`.

    Returns:
        What ``generate`` returns: the (batch, max_new_tokens) int64 tokens, inside a ``GenerateDecoderOnlyOutput``
        when ``return_dict_in_generate`` is set.
    """
    import torch  # lazy: heavy import
    from transformers import DynamicCache  # lazy: heavy import
    from transformers.generation.utils import GenerateDecoderOnlyOutput  # lazy: heavy import

    batch_size, prompt_length = inputs_embeds.shape[:2]
    cache = DynamicCache(config=predictor.config.get_text_config(decoder=True))
    # One arange for every position of the call; each forward takes a view, so no step launches a kernel for it.
    positions = torch.arange(prompt_length + plan.max_new_tokens - 1, device=inputs_embeds.device)
    logits_device = predictor.device
    output_kwargs = {"output_hidden_states": True} if plan.output_hidden_states else {}
    hidden_states: list[Any] | None = [] if plan.output_hidden_states and plan.return_dict_in_generate else None

    tokens: list[Any] = []
    # The prefill takes the embeddings; each later forward takes the token just sampled and the generation_steps the
    # previous forward returned (which picks the codebook's embedding and lm_head).
    step_inputs: dict[str, Any] = {"inputs_embeds": inputs_embeds}
    cache_position = positions[:prompt_length]
    with torch.no_grad():
        while True:
            outputs = predictor(
                **step_inputs,
                past_key_values=cache,
                use_cache=True,
                cache_position=cache_position,
                position_ids=cache_position.unsqueeze(0).expand(batch_size, -1),
                return_dict=True,
                **output_kwargs,
            )
            if hidden_states is not None:
                hidden_states.append(outputs.hidden_states)

            scores = outputs.logits[:, -1, :].to(copy=True, dtype=torch.float32, device=logits_device)
            for warper in plan.warpers:
                scores = warper(None, scores)  # temperature, top-k and top-p do not look at input_ids
            if plan.do_sample:
                probs = torch.nn.functional.softmax(scores, dim=-1)
                next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
            else:
                next_tokens = torch.argmax(scores, dim=-1)
            tokens.append(next_tokens)
            if len(tokens) == plan.max_new_tokens:
                break

            position = prompt_length + len(tokens) - 1
            cache_position = positions[position : position + 1]
            step_inputs = {"input_ids": next_tokens[:, None], "generation_steps": outputs.generation_steps}

    sequences = torch.stack(tokens, dim=1)
    if not plan.return_dict_in_generate:
        return sequences
    return GenerateDecoderOnlyOutput(
        sequences=sequences,
        hidden_states=None if hidden_states is None else tuple(hidden_states),
        past_key_values=cache,
    )


def install(model: Any) -> bool:
    """Make ``model.model.talker.code_predictor.generate`` sample with :func:`sample_with_loop` when it can.

    Calls the loop does not reproduce (see :func:`_sampling_plan`) go to the original ``generate``, which stays
    available as ``generate.original``. Installing twice is a no-op.

    Args:
        model: Loaded ``Qwen3TTSModel``.

    Returns:
        True if the loop is installed, False if the model has no code predictor ``generate``.
    """
    talker = getattr(getattr(model, "model", None), "talker", None)
    predictor = getattr(talker, "code_predictor", None)
    original_generate = getattr(predictor, "generate", None)
    if not callable(original_generate):
        logger.info("ROCm: model has no code predictor generate, sampling loop not installed")
        return False
    if getattr(original_generate, "original", None) is not None:
        return True

    plans: dict[tuple, _SamplingPlan | None] = {}
    fallback_logged = [False]

    def plan_for(args: tuple, kwargs: dict[str, Any]) -> _SamplingPlan | None:
        inputs_embeds = kwargs.get("inputs_embeds")
        if args or not set(kwargs) <= _LOOP_ARGUMENTS or getattr(inputs_embeds, "ndim", None) != 3:
            return None
        if inputs_embeds.shape[1] < 2:
            return None
        arguments = {name: value for name, value in kwargs.items() if name != "inputs_embeds"}
        if not all(value is None or isinstance(value, bool | int | float) for value in arguments.values()):
            return None
        # The plan only depends on the arguments and the model's generation_config, so it is built once per
        # combination instead of copying the config on every talker step.
        key = (id(predictor.generation_config), *sorted(arguments.items()))
        if key not in plans:
            try:
                plans[key] = _sampling_plan(predictor, arguments)
            except Exception:  # generate validates the same settings and reports the error itself
                plans[key] = None
        return plans[key]

    @functools.wraps(original_generate)
    def generate_with_loop(*args, **kwargs):
        plan = plan_for(args, kwargs)
        if plan is None:
            if not fallback_logged[0]:
                fallback_logged[0] = True
                logger.info("ROCm: code predictor call with settings the sampling loop does not cover, using generate")
            return original_generate(*args, **kwargs)
        return sample_with_loop(predictor, kwargs["inputs_embeds"], plan)

    generate_with_loop.original = original_generate
    predictor.generate = generate_with_loop
    logger.info(
        "ROCm: code predictor samples in its own loop instead of HF generate (%s=0 disables it)",
        CODE_PREDICTOR_LOOP_ENV,
    )
    return True
