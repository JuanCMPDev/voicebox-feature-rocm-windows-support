"""
PyTorch backend implementation for TTS and STT.
"""

from typing import Any, Optional, List, Tuple
import asyncio
import contextlib
import logging
import os
import torch
import numpy as np

from ..utils import miopen_codec, perf_log, qwen_code_predictor, qwen_end_fix, qwen_warmup

logger = logging.getLogger(__name__)

from . import TTSBackend, STTBackend, LANGUAGE_CODE_TO_NAME, WHISPER_HF_REPOS
from .base import (
    is_model_cached,
    get_torch_device,
    empty_device_cache,
    manual_seed,
    combine_voice_prompts as _combine_voice_prompts,
    model_load_progress,
)
from ..utils.cache import get_cache_key, get_cached_voice_prompt, cache_voice_prompt
from ..utils.audio import load_audio

# A Qwen voice prompt in the voice prompt cache (memory and disk) is the dict Qwen3TTSModel's
# _prompt_items_to_voice_clone_prompt builds from create_voice_clone_prompt's items, plus each item's ref_text, with the
# tensors on the CPU: only tensors, lists, bools, strings and None, so torch.load(weights_only=True) reads it back.
# Reads rebuild the items, because generate_voice_clone only tokenizes ref_text (which ICL mode needs) for items.
_PROMPT_CACHE_FIELDS = ("ref_code", "ref_spk_embedding", "x_vector_only_mode", "icl_mode", "ref_text")


def _prompt_cache_key(model_size: str, audio_path: str, reference_text: str) -> str:
    """Cache key of a voice prompt; the speaker embedding depends on the model, so each size has its own entry."""
    return f"qwen_{model_size}_{get_cache_key(audio_path, reference_text)}"


def _tensor_to(value: Any, device: Any, copy: bool = False) -> Any:
    """``value`` moved to ``device`` if it is a tensor; anything else as is."""
    if isinstance(value, torch.Tensor):
        return value.detach().to(device, copy=copy)
    return value


def _voice_prompt_to_cache(model: Any, items: list) -> dict[str, list]:
    """Cache entry for the items of ``model.create_voice_clone_prompt``: CPU copies of their tensors, values unchanged."""
    prompt = model._prompt_items_to_voice_clone_prompt(items)
    cached = {name: [_tensor_to(value, "cpu", copy=True) for value in values] for name, values in prompt.items()}
    cached["ref_text"] = [item.ref_text for item in items]
    return cached


def _voice_prompt_from_cache(cached: Any, device: Any) -> list | None:
    """Items of a cache entry written by :func:`_voice_prompt_to_cache`, tensors on ``device``; None for anything else.

    Anything else is a miss: an entry from another format or backend is recomputed and overwritten, never returned.
    """
    if not isinstance(cached, dict) or not all(isinstance(cached.get(name), list) for name in _PROMPT_CACHE_FIELDS):
        return None
    columns = [cached[name] for name in _PROMPT_CACHE_FIELDS]
    if len({len(column) for column in columns}) != 1 or not columns[0]:
        return None
    codes, embeddings = cached["ref_code"], cached["ref_spk_embedding"]
    if not all(isinstance(code, torch.Tensor | None) for code in codes) or not all(
        isinstance(embedding, torch.Tensor) for embedding in embeddings
    ):
        return None
    from qwen_tts import VoiceClonePromptItem  # lazy: heavy import

    return [
        VoiceClonePromptItem(
            ref_code=_tensor_to(code, device),
            ref_spk_embedding=_tensor_to(embedding, device),
            x_vector_only_mode=x_vector_only_mode,
            icl_mode=icl_mode,
            ref_text=ref_text,
        )
        for code, embedding, x_vector_only_mode, icl_mode, ref_text in zip(*columns, strict=True)
    ]


def _generation_mode(device: str) -> contextlib.AbstractContextManager:
    """Context for a generation: ``torch.inference_mode()`` on a ROCm GPU, a no-op context anywhere else.

    ``generate_voice_clone`` already runs under ``no_grad``; inference mode also drops the version counter and view
    tracking autograd keeps there, CPU time on every op of a talker step bound by kernel dispatch (perf-loop plan,
    Fase 3, task 3.3). Same kernels, same inputs, same arithmetic. CPU, CUDA, DirectML and MLX run as before.
    """
    variant = os.environ.get("VOICEBOX_BACKEND_VARIANT")
    if device == "cuda" and (variant == "rocm" or getattr(torch.version, "hip", None)):
        return torch.inference_mode()
    return contextlib.nullcontext()


class PyTorchTTSBackend:
    """PyTorch-based TTS backend using Qwen3-TTS."""

    def __init__(self, model_size: str = "1.7B"):
        self.model = None
        self.model_size = model_size
        self.device = self._get_device()
        self._current_model_size = None

    def _get_device(self) -> str:
        """Get the best available device."""
        return get_torch_device(allow_xpu=True, allow_directml=True)

    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        return self.model is not None

    def _get_model_path(self, model_size: str) -> str:
        """
        Get the HuggingFace Hub model ID.

        Args:
            model_size: Model size (1.7B or 0.6B)

        Returns:
            HuggingFace Hub model ID
        """
        hf_model_map = {
            "1.7B": "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
            "0.6B": "Qwen/Qwen3-TTS-12Hz-0.6B-Base",
        }

        if model_size not in hf_model_map:
            raise ValueError(f"Unknown model size: {model_size}")

        return hf_model_map[model_size]

    def _is_model_cached(self, model_size: str) -> bool:
        return is_model_cached(self._get_model_path(model_size))

    async def load_model_async(self, model_size: Optional[str] = None):
        """
        Lazy load the TTS model with automatic downloading from HuggingFace Hub.

        Args:
            model_size: Model size to load (1.7B or 0.6B)
        """
        if model_size is None:
            model_size = self.model_size

        # If already loaded with correct size, return
        if self.model is not None and self._current_model_size == model_size:
            return

        # Unload existing model if different size requested
        if self.model is not None and self._current_model_size != model_size:
            self.unload_model()

        # Run blocking load in thread pool
        await asyncio.to_thread(self._load_model_sync, model_size)

    # Alias for compatibility
    load_model = load_model_async

    def _load_model_sync(self, model_size: str):
        """Synchronous model loading."""
        model_name = f"qwen-tts-{model_size}"
        is_cached = self._is_model_cached(model_size)

        with model_load_progress(model_name, is_cached):
            from qwen_tts import Qwen3TTSModel

            model_path = self._get_model_path(model_size)
            logger.info("Loading TTS model %s on %s...", model_size, self.device)

            # Route both HF Hub and Transformers through a single cache root.
            # On Windows local setups, model assets can otherwise split between
            # .hf-cache/hub and .hf-cache/transformers, causing speech_tokenizer
            # and preprocessor_config.json to fail to resolve during load.
            from huggingface_hub import constants as hf_constants
            tts_cache_dir = hf_constants.HF_HUB_CACHE

            if self.device == "cpu":
                self.model = Qwen3TTSModel.from_pretrained(
                    model_path,
                    cache_dir=tts_cache_dir,
                    torch_dtype=torch.float32,
                    low_cpu_mem_usage=False,
                )
            else:
                self.model = Qwen3TTSModel.from_pretrained(
                    model_path,
                    cache_dir=tts_cache_dir,
                    device_map=self.device,
                    torch_dtype=torch.bfloat16,
                )

        self._current_model_size = model_size
        self.model_size = model_size
        logger.info("TTS model %s loaded successfully", model_size)

        # Every device: the talker's stopping rule holds back a phrase's first likely codec_eos and rules out chance
        # ones, so phrases do not end before the last word does (perf-loop task 4.4; generate and generate_batch also
        # pass the non-streaming prompt and each row's guard). Before the warm-up, so the warm-up's talker step already
        # runs the rule's processor and pays its kernels' first launch; the warm-up keeps its own prompt, without
        # guards.
        if qwen_end_fix.end_fix_enabled():
            qwen_end_fix.install(self.model)

        if miopen_codec.codec_immediate_enabled(
            self.device,
            os.environ.get("VOICEBOX_BACKEND_VARIANT"),
            getattr(torch.version, "hip", None),
        ):
            miopen_codec.use_immediate_mode_for_codec(self.model)

        # ROCm: the code predictor samples each frame's 15 codebook tokens in a loop without host reads instead of HF
        # generate (same forwards, same sampling). Before the warm-up, so the warm-up runs the loop generation uses.
        if qwen_code_predictor.loop_enabled(
            self.device,
            os.environ.get("VOICEBOX_BACKEND_VARIANT"),
            getattr(torch.version, "hip", None),
        ):
            qwen_code_predictor.install(self.model)

        # ROCm: the loop's batch-1 frame captured once as a HIP graph, which each talker step then replays (same
        # kernels, one launch). Before the warm-up, so its talker step already replays the graph.
        if qwen_code_predictor.graph_enabled(
            self.device,
            os.environ.get("VOICEBOX_BACKEND_VARIANT"),
            getattr(torch.version, "hip", None),
        ):
            qwen_code_predictor.capture_graph(self.model)

        # ROCm: pay the first decode of each codec length and the first talker step here, while a generation that
        # triggered the load still shows loading_model. After the immediate-mode wrapper, so the warm-up decodes run
        # as generation does; before VBPERF, so it logs no talker/codec lines.
        if qwen_warmup.warmup_enabled(
            self.device,
            os.environ.get("VOICEBOX_BACKEND_VARIANT"),
            getattr(torch.version, "hip", None),
        ):
            qwen_warmup.warm_up(self.model)

        if perf_log.is_enabled():
            perf_log.instrument_qwen_model(self.model)
            perf_log.emit_env(perf_log.qwen_env_info(self.model, self.device))

    def unload_model(self):
        """Unload the model to free memory."""
        if self.model is not None:
            # The code predictor graph (ROCm) holds its own memory pool; free it now rather than when the GC breaks
            # the model's reference cycles, so empty_device_cache below returns that memory too.
            qwen_code_predictor.release_graph(self.model)
            del self.model
            self.model = None
            self._current_model_size = None

            empty_device_cache(self.device)

            logger.info("TTS model unloaded")

    async def create_voice_prompt(
        self,
        audio_path: str,
        reference_text: str,
        use_cache: bool = True,
    ) -> Tuple[dict, bool]:
        """
        Create voice prompt from reference audio.

        Args:
            audio_path: Path to reference audio file
            reference_text: Transcript of reference audio
            use_cache: Whether to use cached prompt if available

        Returns:
            Tuple of (voice_prompt, was_cached); voice_prompt is the list of
            VoiceClonePromptItem that generate_voice_clone takes
        """
        await self.load_model_async(None)

        with perf_log.stage("reference") as perf_timer:
            # Check cache if enabled
            cache_key = None
            if use_cache:
                cache_key = _prompt_cache_key(self._current_model_size or self.model_size, audio_path, reference_text)
                cached_items = _voice_prompt_from_cache(get_cached_voice_prompt(cache_key), self.device)
                if cached_items is not None:
                    perf_timer.fields["cached"] = True
                    return cached_items, True

            def _create_prompt_sync():
                """Run synchronous voice prompt creation in thread pool."""
                # Inference runs with the process's default HF_HUB_OFFLINE
                # state. Forcing offline here (issue #462) regressed online
                # users whose libraries issue legitimate metadata lookups
                # during voice-prompt creation.
                return self.model.create_voice_clone_prompt(
                    ref_audio=str(audio_path),
                    ref_text=reference_text,
                    x_vector_only_mode=False,
                )

            # Run blocking operation in thread pool
            voice_prompt_items = await asyncio.to_thread(_create_prompt_sync)

            # Cache if enabled
            if cache_key is not None:
                cache_voice_prompt(cache_key, _voice_prompt_to_cache(self.model, voice_prompt_items))

            perf_timer.fields["cached"] = False
            return voice_prompt_items, False

    async def combine_voice_prompts(
        self,
        audio_paths: List[str],
        reference_texts: List[str],
    ) -> Tuple[np.ndarray, str]:
        return await _combine_voice_prompts(audio_paths, reference_texts)

    async def generate(
        self,
        text: str,
        voice_prompt: dict,
        language: str = "en",
        seed: Optional[int] = None,
        instruct: Optional[str] = None,
    ) -> Tuple[np.ndarray, int]:
        """
        Generate audio from text using voice prompt.

        Args:
            text: Text to synthesize
            voice_prompt: Voice prompt dictionary from create_voice_prompt
            language: Language code (en or zh)
            seed: Random seed for reproducibility
            instruct: Natural language instruction for speech delivery control

        Returns:
            Tuple of (audio_array, sample_rate)
        """
        # Load model
        await self.load_model_async(None)

        def _generate_sync():
            """Run synchronous generation in thread pool."""
            # Set seed if provided
            if seed is not None:
                manual_seed(seed, self.device)

            # See _create_prompt_sync comment — inference runs with the
            # process's default HF_HUB_OFFLINE state (issue #462).
            with _generation_mode(self.device), qwen_end_fix.phrase_end([text], voice_prompt) as end_kwargs:
                wavs, sample_rate = self.model.generate_voice_clone(
                    text=text,
                    voice_clone_prompt=voice_prompt,
                    language=LANGUAGE_CODE_TO_NAME.get(language, "auto"),
                    instruct=instruct,
                    **end_kwargs,
                )
            return wavs[0], sample_rate

        # Run blocking inference in thread pool to avoid blocking event loop
        audio, sample_rate = await asyncio.to_thread(_generate_sync)

        return audio, sample_rate

    def supports_batch(self) -> bool:
        """Whether queued jobs may be batched into :meth:`generate_batch`: on a ROCm GPU only (perf-loop plan, 3.6)."""
        variant = os.environ.get("VOICEBOX_BACKEND_VARIANT")
        return self.device == "cuda" and (variant == "rocm" or bool(getattr(torch.version, "hip", None)))

    async def generate_batch(
        self,
        texts: list[str],
        voice_prompts: list[list],
        language: str = "en",
        seed: int | None = None,
        instruct: str | None = None,
    ) -> tuple[list[np.ndarray], int]:
        """Generate several texts in one ``generate_voice_clone`` call, one voice prompt per text.

        The talker steps every sequence at once, so the per-step overhead is paid once for the whole batch. The
        sampling settings are the model's, as in :meth:`generate`, but the draws are not: every sequence samples from
        one RNG stream seeded once with ``seed``, and the prompts are left-padded to a common length, so a text's
        audio differs from what :meth:`generate` gives for the same text and seed.

        Args:
            texts: Texts to synthesize.
            voice_prompts: For each text, the voice prompt :meth:`create_voice_prompt` returned (one item).
            language: Language code shared by every text.
            seed: Seed for the batch's RNG stream, or None to leave the RNG as it is.
            instruct: Instruction shared by every text (the Base model ignores it).

        Returns:
            Tuple of (audio arrays aligned with ``texts``, sample_rate).

        Raises:
            ValueError: If a voice prompt is not a single prompt item.
        """
        items = []
        for prompt in voice_prompts:
            if not isinstance(prompt, list) or len(prompt) != 1:
                raise ValueError("generate_batch takes one voice prompt item per text")
            items.extend(prompt)

        await self.load_model_async(None)

        def _generate_batch_sync():
            if seed is not None:
                manual_seed(seed, self.device)
            with _generation_mode(self.device), qwen_end_fix.phrase_end(list(texts), items) as end_kwargs:
                wavs, sample_rate = self.model.generate_voice_clone(
                    text=list(texts),
                    voice_clone_prompt=items,
                    language=LANGUAGE_CODE_TO_NAME.get(language, "auto"),
                    instruct=instruct,
                    **end_kwargs,
                )
            return list(wavs), sample_rate

        return await asyncio.to_thread(_generate_batch_sync)


class PyTorchSTTBackend:
    """PyTorch-based STT backend using Whisper."""

    def __init__(self, model_size: str = "base"):
        self.model = None
        self.processor = None
        self.model_size = model_size
        self.device = self._get_device()

    def _get_device(self) -> str:
        """Get the best available device."""
        return get_torch_device(allow_xpu=True, allow_directml=True)

    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        return self.model is not None

    def _is_model_cached(self, model_size: str) -> bool:
        hf_repo = WHISPER_HF_REPOS.get(model_size, f"openai/whisper-{model_size}")
        return is_model_cached(hf_repo)

    async def load_model_async(self, model_size: Optional[str] = None):
        """
        Lazy load the Whisper model.

        Args:
            model_size: Model size (tiny, base, small, medium, large)
        """
        if model_size is None:
            model_size = self.model_size

        if self.model is not None and self.model_size == model_size:
            return

        await asyncio.to_thread(self._load_model_sync, model_size)

    # Alias for compatibility
    load_model = load_model_async

    def _load_model_sync(self, model_size: str):
        """Synchronous model loading."""
        progress_model_name = f"whisper-{model_size}"
        is_cached = self._is_model_cached(model_size)

        with model_load_progress(progress_model_name, is_cached):
            from transformers import WhisperProcessor, WhisperForConditionalGeneration

            model_name = WHISPER_HF_REPOS.get(model_size, f"openai/whisper-{model_size}")
            logger.info("Loading Whisper model %s on %s...", model_size, self.device)

            self.processor = WhisperProcessor.from_pretrained(model_name)
            self.model = WhisperForConditionalGeneration.from_pretrained(model_name)

        self.model.to(self.device)
        self.model_size = model_size
        logger.info("Whisper model %s loaded successfully", model_size)

    def unload_model(self):
        """Unload the model to free memory."""
        if self.model is not None:
            del self.model
            del self.processor
            self.model = None
            self.processor = None

            empty_device_cache(self.device)

            logger.info("Whisper model unloaded")

    async def transcribe(
        self,
        audio_path: str,
        language: Optional[str] = None,
        model_size: Optional[str] = None,
    ) -> str:
        """
        Transcribe audio to text.

        Args:
            audio_path: Path to audio file
            language: Optional language hint
            model_size: Optional model size override

        Returns:
            Transcribed text
        """
        await self.load_model_async(model_size)

        def _transcribe_sync():
            """Run synchronous transcription in thread pool."""
            # Load audio
            audio, _sr = load_audio(audio_path, sample_rate=16000)

            # Inference runs with the process's default HF_HUB_OFFLINE
            # state — forcing offline here (issue #462) broke online users
            # whose `get_decoder_prompt_ids` / tokenizer calls issue
            # legitimate metadata lookups.
            # Process audio
            inputs = self.processor(
                audio,
                sampling_rate=16000,
                return_tensors="pt",
            )
            inputs = inputs.to(self.device)

            # Generate transcription
            # If language is provided, force it; otherwise let Whisper auto-detect
            generate_kwargs = {}
            if language:
                forced_decoder_ids = self.processor.get_decoder_prompt_ids(
                    language=language,
                    task="transcribe",
                )
                generate_kwargs["forced_decoder_ids"] = forced_decoder_ids

            with torch.no_grad():
                predicted_ids = self.model.generate(
                    inputs["input_features"],
                    **generate_kwargs,
                )

            # Decode
            transcription = self.processor.batch_decode(
                predicted_ids,
                skip_special_tokens=True,
            )[0]

            return transcription.strip()

        # Run blocking transcription in thread pool
        return await asyncio.to_thread(_transcribe_sync)
