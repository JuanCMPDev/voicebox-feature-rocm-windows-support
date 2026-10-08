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

On top of the loop, :func:`capture_graph` (task 3.5b, perf-loop/notes/3.5b.md) records one frame of batch size 1 (the
loop itself, its 15 forwards and samplings, run on a fixed input buffer) as a single CUDA (HIP) graph when the model
loads. The talker's batch-1 calls then copy their embeddings into the buffer and replay the graph: the same kernels on
the same shapes and weights, with none of the per-op Python and dispatch time. Batches of 2 or more, other settings and
any failure of the graph go to the loop.
"""

import copy
import functools
import logging
import os
import threading
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from time import perf_counter
from typing import Any

logger = logging.getLogger(__name__)

CODE_PREDICTOR_LOOP_ENV = "VOICEBOX_QWEN_CODE_PREDICTOR_LOOP"
CODE_PREDICTOR_GRAPH_ENV = "VOICEBOX_QWEN_CP_GRAPH"
ROCM_VARIANT = "rocm"
_DISABLED_VALUES = frozenset({"0", "false", "no", "off"})
# Eager frames on a side stream before the capture, as the torch docs ask (lazy library handles and workspaces).
GRAPH_WARMUP_CALLS = 3

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


def graph_enabled(
    device: Any,
    variant: str | None,
    hip: str | None,
    environ: Mapping[str, str] = os.environ,
) -> bool:
    """Whether the loop's batch-1 frame should be captured as a HIP graph when the model loads (:func:`capture_graph`).

    Args:
        device: Torch device the TTS model runs on; ROCm GPUs show up as ``"cuda"``.
        variant: ``VOICEBOX_BACKEND_VARIANT``; ``"rocm"`` in the packaged ROCm binary, unset in development.
        hip: ``torch.version.hip``; None on CPU, CUDA, DirectML and MLX builds.
        environ: Process environment, where ``VOICEBOX_QWEN_CP_GRAPH=0`` turns the graph off.

    Returns:
        True only where :func:`loop_enabled` is (the graph replays the loop) and unless the user disabled it.
    """
    if not loop_enabled(device, variant, hip, environ):
        return False
    return environ.get(CODE_PREDICTOR_GRAPH_ENV, "1").strip().lower() not in _DISABLED_VALUES


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


def _capture(fn: Callable[[], Any], device: Any) -> tuple[Any, Any]:
    """``fn`` run eagerly on a side stream, then captured as a CUDA (HIP) graph; the graph and ``fn``'s output.

    The graph gets its own memory pool, so the output and every tensor ``fn`` allocates keep their addresses for the
    replays: the output is a fixed buffer each replay overwrites. The capture runs on the warm-up's stream, so its GEMMs
    use the BLAS workspace torch gave that stream during the warm-up (~108 MiB on gfx1101) instead of a second one.
    ``thread_local`` capture mode only checks this thread's CUDA calls, so a request on another thread
    (``/generate/stream`` does not take the GPU queue) neither fails nor breaks the capture.
    """
    import torch  # lazy: heavy import

    side = torch.cuda.Stream(device=device)
    side.wait_stream(torch.cuda.current_stream(device))
    with torch.cuda.stream(side):
        for _ in range(GRAPH_WARMUP_CALLS):
            fn()
    torch.cuda.current_stream(device).wait_stream(side)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=side, capture_error_mode="thread_local"):
        output = fn()
    torch.cuda.synchronize(device)
    return graph, output


# FrameGraphs holding a captured graph. The BLAS workspaces their kernels write to may only be dropped when none is left.
_live_graphs: set[int] = set()
_live_graphs_lock = threading.Lock()


def _drop_blas_workspaces() -> None:
    """Free the BLAS workspaces torch keeps per (handle, stream), once no captured graph uses one.

    Torch never frees them, so the capture stream's would outlive the model; each stream gets a new one at its next
    GEMM.
    """
    import torch  # lazy: heavy import

    clear = getattr(torch._C, "_cuda_clearCublasWorkspaces", None)
    if clear is not None:
        clear()


class FrameGraph:
    """One batch-1 frame of the code predictor, :func:`sample_with_loop` as it is, captured as a single graph.

    The captured frame is the loop run on ``embeds``, a fixed input buffer: a fresh ``DynamicCache``, the 15 forwards
    with no attention mask, HF's warpers, softmax and ``torch.multinomial``. A replay launches those kernels again with
    the same shapes and weights, reading whatever :meth:`replay` copied into the buffer, so it samples the codes the
    loop samples. Each ``multinomial`` registers its Philox increment at capture, and every replay advances the CUDA
    generator by their total: the RNG is consumed as in the loop.

    Thread-safe: a lock serializes "copy the input in, replay, clone the output out". Every caller in this app queues
    its work on the device's default stream, which keeps that order on the GPU as well.

    Args:
        predictor: ``Qwen3TTSTalkerCodePredictorModelForConditionalGeneration``.
        plan: The plan of the calls this graph replays (from ``generate.plan_for``).
        embeds: Fixed input buffer, shaped, typed and placed as those calls' ``inputs_embeds``.
    """

    def __init__(self, predictor: Any, plan: _SamplingPlan, embeds: Any):
        self.predictor = predictor
        self.plan = plan
        self.embeds = embeds
        self.graph: Any = None
        self.sequences: Any = None  # the graph's output buffer
        self._device = embeds.device
        self._lock = threading.Lock()

    def _frame(self) -> Any:
        result = sample_with_loop(self.predictor, self.embeds, self.plan)
        return result.sequences if self.plan.return_dict_in_generate else result

    def capture(self) -> None:
        """Capture the frame (raises if the capture fails; the caller then keeps the loop)."""
        import torch  # lazy: heavy import

        with _live_graphs_lock:  # from here on its capture stream's workspace must stay
            _live_graphs.add(id(self))
        try:
            # Outside inference mode: the buffer stays a normal tensor, which replays may write with or without it.
            with torch.inference_mode(False), torch.no_grad():
                graph, sequences = _capture(self._frame, self._device)
        except BaseException:
            with _live_graphs_lock:
                _live_graphs.discard(id(self))
            raise
        with self._lock:
            self.graph, self.sequences = graph, sequences

    def matches(self, plan: _SamplingPlan, inputs_embeds: Any) -> bool:
        """Whether a call with ``plan`` and ``inputs_embeds`` is the frame this graph replays."""
        embeds = self.embeds
        return (
            plan is self.plan
            and self.graph is not None
            and embeds is not None
            and getattr(inputs_embeds, "shape", None) == embeds.shape
            and getattr(inputs_embeds, "dtype", None) == embeds.dtype
            and getattr(inputs_embeds, "device", None) == embeds.device
            and inputs_embeds.is_contiguous()
        )

    def replay(self, inputs_embeds: Any) -> Any:
        """The frame's (1, max_new_tokens) int64 codes for ``inputs_embeds``; None if the graph was released."""
        with self._lock:
            if self.graph is None:
                return None
            self.embeds.copy_(inputs_embeds)
            self.graph.replay()
            return self.sequences.clone()  # the next replay overwrites the output buffer

    def release(self) -> None:
        """Free the graph, its memory pool, the fixed buffers and the BLAS workspace of its capture stream.

        Idempotent; logs instead of raising.
        """
        with self._lock:
            graph, self.graph = self.graph, None
            self.sequences = None
            self.embeds = None
            if graph is None:
                return
            on_gpu = self._device.type == "cuda"
            try:
                import torch  # lazy: heavy import

                if on_gpu:
                    torch.cuda.synchronize(self._device)  # no replay may still be reading the pool
                graph.reset()
            except Exception:
                logger.warning("ROCm: releasing the code predictor graph failed", exc_info=True)
                return
            with _live_graphs_lock:
                _live_graphs.discard(id(self))
                if on_gpu and not _live_graphs:
                    try:
                        _drop_blas_workspaces()
                    except Exception:
                        logger.warning("ROCm: freeing the BLAS workspaces failed", exc_info=True)


def _graph_output(sequences: Any, plan: _SamplingPlan) -> Any:
    """What ``generate`` returns, from a replay's codes.

    Hidden states and cache stay inside the graph: the talker only reads ``.sequences``.
    """
    if not plan.return_dict_in_generate:
        return sequences
    from transformers.generation.utils import GenerateDecoderOnlyOutput  # lazy: heavy import

    return GenerateDecoderOnlyOutput(sequences=sequences, hidden_states=None, past_key_values=None)


def install(model: Any) -> bool:
    """Make ``model.model.talker.code_predictor.generate`` sample with :func:`sample_with_loop` when it can.

    Calls the loop does not reproduce (see :func:`_sampling_plan`) go to the original ``generate``, which stays
    available as ``generate.original``. Calls a captured :class:`FrameGraph` covers (``generate.frame_graph``, set by
    :func:`capture_graph`) replay it instead of running the loop. Installing twice is a no-op.

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
        inputs_embeds = kwargs["inputs_embeds"]
        graph = generate_with_loop.frame_graph
        if graph is not None and graph.matches(plan, inputs_embeds):
            try:
                sequences = graph.replay(inputs_embeds)
            except Exception:
                logger.warning(
                    "ROCm: code predictor graph replay failed, sampling in the loop for the rest of the model's life",
                    exc_info=True,
                )
                generate_with_loop.frame_graph = None
                graph.release()
            else:
                if sequences is not None:
                    return _graph_output(sequences, plan)
        return sample_with_loop(predictor, inputs_embeds, plan)

    generate_with_loop.original = original_generate
    generate_with_loop.plan_for = plan_for
    generate_with_loop.frame_graph = None
    predictor.generate = generate_with_loop
    logger.info(
        "ROCm: code predictor samples in its own loop instead of HF generate (%s=0 disables it)",
        CODE_PREDICTOR_LOOP_ENV,
    )
    return True


def _installed_generate(model: Any) -> Any:
    """The :func:`install` wrapper on ``model``'s code predictor, or None."""
    talker = getattr(getattr(model, "model", None), "talker", None)
    generate = getattr(getattr(talker, "code_predictor", None), "generate", None)
    return generate if callable(getattr(generate, "plan_for", None)) else None


def talker_arguments(model: Any) -> dict[str, Any]:
    """The arguments the talker passes to ``code_predictor.generate`` on each step of ``generate_voice_clone``.

    ``Qwen3TTSTalkerForConditionalGeneration.forward`` passes ``num_code_groups - 1`` new tokens, the ``subtalker_*``
    settings ``generate_voice_clone`` takes from ``Qwen3TTSModel._merge_generate_kwargs`` (the model's
    ``generate_config.json`` unless the caller overrides them; the app does not), and asks for hidden states in a dict.
    """
    settings = model._merge_generate_kwargs()
    return {
        "max_new_tokens": model.model.talker.config.num_code_groups - 1,
        "do_sample": settings["subtalker_dosample"],
        "top_p": settings["subtalker_top_p"],
        "top_k": settings["subtalker_top_k"],
        "temperature": settings["subtalker_temperature"],
        "output_hidden_states": True,
        "return_dict_in_generate": True,
    }


def capture_graph(model: Any, clock: Callable[[], float] = perf_counter) -> bool:
    """Capture the code predictor's batch-1 frame as a HIP graph for :func:`install`'s wrapper to replay. Never raises.

    The frame is captured for the talker's own arguments (:func:`talker_arguments`, through the wrapper's plan cache, so
    the talker's calls get the very plan the graph holds) and for its ``(1, 2, hidden_size)`` input embeddings. If the
    capture fails, the code predictor keeps sampling in the loop for the model's life.

    Args:
        model: Loaded ``Qwen3TTSModel`` with the loop installed. The caller holds the GPU (load runs in the GPU queue).
        clock: Time source in seconds.

    Returns:
        True if the wrapper now replays a graph.
    """
    generate = _installed_generate(model)
    if generate is None:
        logger.info("ROCm: code predictor sampling loop not installed, no graph to capture")
        return False
    if generate.frame_graph is not None:
        return True
    started = clock()
    graph = None
    try:
        import torch  # lazy: heavy import

        talker = model.model.talker
        weight = talker.get_input_embeddings().weight  # the talker's step embeddings are in its dtype and device
        with torch.inference_mode(False):
            embeds = torch.zeros((1, 2, talker.config.hidden_size), dtype=weight.dtype, device=weight.device)
        plan = generate.plan_for((), {"inputs_embeds": embeds, **talker_arguments(model)})
        if plan is None:
            logger.info("ROCm: the talker's code predictor settings are not the loop's, no graph to capture")
            return False
        graph = FrameGraph(talker.code_predictor, plan, embeds)
        graph.capture()
    except Exception:
        logger.warning(
            "ROCm: capturing the code predictor frame as a HIP graph failed, the code predictor keeps its loop",
            exc_info=True,
        )
        if graph is not None:
            graph.release()
        return False
    generate.frame_graph = graph
    logger.info(
        "ROCm: code predictor frame captured as a HIP graph in %.1f s (%s=0 disables it)",
        clock() - started,
        CODE_PREDICTOR_GRAPH_ENV,
    )
    return True


def release_graph(model: Any) -> None:
    """Free the code predictor graph of ``model`` (graph, memory pool, fixed buffers), if it has one."""
    generate = _installed_generate(model)
    graph = getattr(generate, "frame_graph", None)
    if graph is None:
        return
    generate.frame_graph = None
    graph.release()
    logger.info("ROCm: code predictor graph released")
