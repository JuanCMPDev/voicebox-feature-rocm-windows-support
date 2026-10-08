"""
Tests for scripts/probe_hip_graph.py, the HIP graph probe of the perf loop (plan, Fase 3.5; task 3.5a).

The graphs themselves need a GPU; what runs here, on CPU with tiny random-weight modules, is what the probe captures:
the static-cache decode step of talker layers must give what the production step (DynamicCache, no mask) gives, with
garbage in the slots it has not reached; and the unrolled code-predictor frame must sample the same codes and consume
the RNG as task 3.2's loop does.

Usage:
    python -m pytest backend/tests/test_probe_hip_graph.py -v
"""

import importlib.util
from pathlib import Path
from typing import Any

import pytest
import torch

from backend.utils import qwen_code_predictor

_SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "probe_hip_graph.py"
_spec = importlib.util.spec_from_file_location("probe_hip_graph", _SCRIPT)
probe = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(probe)

_HIDDEN = 32
_CODE_GROUPS = 16
_PROMPT = 6
_STEPS = 4


def _configs(attn: str = "sdpa") -> tuple[Any, Any]:
    """Tiny talker and code-predictor configs with the real model's layout (mRoPE in 3 sections, GQA)."""
    configs = pytest.importorskip("qwen_tts.core.models.configuration_qwen3_tts")
    layout = {
        "intermediate_size": 64,
        "num_hidden_layers": 2,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "head_dim": 8,
        "attn_implementation": attn,
    }
    predictor_config = configs.Qwen3TTSTalkerCodePredictorConfig(
        vocab_size=64, hidden_size=_HIDDEN, num_code_groups=_CODE_GROUPS, initializer_range=0.5, **layout
    )
    talker_config = configs.Qwen3TTSTalkerConfig(
        hidden_size=_HIDDEN,
        num_code_groups=_CODE_GROUPS,
        code_predictor_config=predictor_config.to_dict(),
        rope_scaling={"mrope_section": [2, 1, 1], "interleaved": True, "rope_type": "default"},
        **layout,
    )
    return talker_config, predictor_config


def _talker_layers(attn: str = "sdpa") -> tuple[Any, list[Any], Any, Any]:
    """Config, decoder layers, rotary embedding and a final norm + head of a tiny talker backbone, in eval mode."""
    modeling = pytest.importorskip("qwen_tts.core.models.modeling_qwen3_tts")
    config, _ = _configs(attn)
    torch.manual_seed(0)
    layers = [modeling.Qwen3TTSTalkerDecoderLayer(config, i).eval() for i in range(config.num_hidden_layers)]
    rotary = modeling.Qwen3TTSTalkerRotaryEmbedding(config)
    norm = modeling.Qwen3TTSRMSNorm(_HIDDEN, eps=config.rms_norm_eps)
    head = torch.nn.Linear(_HIDDEN, 64, bias=False)

    def norm_head(hidden: torch.Tensor) -> torch.Tensor:
        return head(norm(hidden))

    return config, layers, rotary, norm_head


@pytest.mark.parametrize("attn", ["sdpa", "eager"])
@pytest.mark.parametrize("with_head", [False, True])
def test_static_step_matches_the_production_decode_step(attn, with_head):
    transformers = pytest.importorskip("transformers")
    config, layers, rotary, norm_head = _talker_layers(attn)
    head = norm_head if with_head else None
    generator = torch.Generator().manual_seed(1)
    prompt = torch.randn(1, _PROMPT, _HIDDEN, generator=generator)
    inputs = [torch.randn(1, 1, _HIDDEN, generator=generator) for _ in range(_STEPS)]
    cache_len = _PROMPT + _STEPS + 3  # spare slots past the last step: the mask has to hide them

    with torch.inference_mode():
        prefix = probe.prefill(torch, rotary, layers, prompt)
        static = transformers.StaticCache(config=config, max_cache_len=cache_len)
        probe.load_static(static, prefix)
        for layer in static.layers[: len(layers)]:  # garbage in every slot not written yet
            layer.keys[:, :, _PROMPT:] = 50.0
            layer.values[:, :, _PROMPT:] = -50.0
        x = torch.zeros(1, 1, _HIDDEN)
        cache_position = torch.tensor([_PROMPT])
        step = probe.static_step(torch, rotary, layers, head, static, x, cache_position, torch.arange(cache_len))
        dynamic = probe.dynamic_cache(prefix)
        for n, hidden in enumerate(inputs):
            x.copy_(hidden)  # what the probe does before each replay
            cache_position.fill_(_PROMPT + n)
            expected = probe.dynamic_step(rotary, layers, head, dynamic, hidden, torch.tensor([_PROMPT + n]))
            torch.testing.assert_close(step(), expected, rtol=1e-5, atol=1e-5)


def test_static_step_sees_its_own_slot_and_not_the_next():
    """With the mask off by one either way the step must differ from production (the test above can fail)."""
    transformers = pytest.importorskip("transformers")
    config, layers, rotary, _ = _talker_layers()
    generator = torch.Generator().manual_seed(2)
    prompt = torch.randn(1, _PROMPT, _HIDDEN, generator=generator)
    hidden = torch.randn(1, 1, _HIDDEN, generator=generator)
    cache_len = _PROMPT + 2
    with torch.inference_mode():
        prefix = probe.prefill(torch, rotary, layers, prompt)
        expected = probe.dynamic_step(
            rotary, layers, None, probe.dynamic_cache(prefix), hidden, torch.tensor([_PROMPT])
        )
        for shift in (-1, 1):
            static = transformers.StaticCache(config=config, max_cache_len=cache_len)
            probe.load_static(static, prefix)
            for layer in static.layers[: len(layers)]:
                layer.keys[:, :, _PROMPT + 1 :] = 50.0
            kv_index = torch.arange(cache_len) + shift  # shifts which slots the mask lets through
            step = probe.static_step(
                torch, rotary, layers, None, static, hidden.clone(), torch.tensor([_PROMPT]), kv_index
            )
            assert not torch.allclose(step(), expected, rtol=1e-5, atol=1e-5)


def test_load_static_resets_in_place():
    transformers = pytest.importorskip("transformers")
    config, layers, rotary, _ = _talker_layers()
    prompt = torch.randn(1, _PROMPT, _HIDDEN, generator=torch.Generator().manual_seed(3))
    with torch.inference_mode():
        prefix = probe.prefill(torch, rotary, layers, prompt)
        static = transformers.StaticCache(config=config, max_cache_len=_PROMPT + 5)
        probe.load_static(static, prefix)
        pointers = [(layer.keys.data_ptr(), layer.values.data_ptr()) for layer in static.layers[: len(layers)]]
        for layer in static.layers[: len(layers)]:
            layer.keys.fill_(7.0)
            layer.values.fill_(7.0)
        probe.load_static(static, prefix)
    for layer, (keys, values), pointer in zip(static.layers, prefix, pointers, strict=False):
        assert (layer.keys.data_ptr(), layer.values.data_ptr()) == pointer  # a captured graph keeps its addresses
        assert torch.equal(layer.keys[:, :, :_PROMPT], keys)
        assert torch.equal(layer.values[:, :, :_PROMPT], values)
        assert not layer.keys[:, :, _PROMPT:].any()
        assert not layer.values[:, :, _PROMPT:].any()


def test_attention_mask_follows_the_attention_implementation():
    allowed = torch.tensor([True, True, False]).view(1, 1, 1, 3)
    sdpa, eager = _configs("sdpa")[0], _configs("eager")[0]
    assert probe.attention_mask(torch, allowed, sdpa, torch.bfloat16) is allowed
    additive = probe.attention_mask(torch, allowed, eager, torch.bfloat16)
    assert additive.dtype == torch.bfloat16
    assert additive.flatten().tolist() == [0.0, 0.0, torch.finfo(torch.bfloat16).min]


def test_frame_masks():
    masks = probe.frame_masks(torch.arange(5), prompt_length=2)
    assert [tuple(m.shape) for m in masks] == [(1, 1, 2, 5), (1, 1, 1, 5), (1, 1, 1, 5), (1, 1, 1, 5)]
    assert masks[0][0, 0].tolist() == [[True, False, False, False, False], [True, True, False, False, False]]
    assert masks[1][0, 0, 0].tolist() == [True, True, True, False, False]
    assert masks[3][0, 0, 0].tolist() == [True] * 5


def _predictor(attn: str) -> Any:
    modeling = pytest.importorskip("qwen_tts.core.models.modeling_qwen3_tts")
    talker_config, predictor_config = _configs(attn)
    torch.manual_seed(0)
    predictor = modeling.Qwen3TTSTalkerCodePredictorModelForConditionalGeneration(predictor_config, talker_config)
    return predictor.eval()


@pytest.mark.parametrize("attn", ["sdpa", "eager"])
@pytest.mark.parametrize(
    "arguments",
    [
        {"do_sample": True, "top_p": 1.0, "top_k": 50, "temperature": 0.9},  # the talker's subtalker settings
        {"do_sample": True, "top_p": 0.8, "top_k": 0, "temperature": 1.0},
        {"do_sample": False},
    ],
)
def test_code_predictor_frame_samples_as_the_production_loop(attn, arguments):
    transformers = pytest.importorskip("transformers")
    predictor = _predictor(attn)
    plan = qwen_code_predictor._sampling_plan(
        predictor,
        {
            "max_new_tokens": _CODE_GROUPS - 1,
            "output_hidden_states": True,
            "return_dict_in_generate": True,
            **arguments,
        },
    )
    generator = torch.Generator().manual_seed(4)
    frames = [torch.randn(1, 2, _HIDDEN, generator=generator) for _ in range(3)]
    cache_len = 2 + plan.max_new_tokens - 1
    positions = torch.arange(cache_len)
    with torch.inference_mode():
        cache = transformers.StaticCache(config=predictor.config, max_cache_len=cache_len)
        embeds = torch.zeros(1, 2, _HIDDEN)
        frame = probe.code_predictor_frame(
            torch, predictor, plan, cache, embeds, positions, probe.frame_masks(positions, 2)
        )
        # One seed, then frame after frame on the same cache and generator, as the probe replays them.
        torch.manual_seed(11)
        expected = [qwen_code_predictor.sample_with_loop(predictor, f, plan).sequences for f in frames]
        state_expected = torch.get_rng_state()
        torch.manual_seed(11)
        got = []
        for f in frames:
            embeds.copy_(f)
            got.append(frame().clone())
        state_got = torch.get_rng_state()
    for g, e in zip(got, expected, strict=True):
        assert g.shape == (1, _CODE_GROUPS - 1)
        assert torch.equal(g, e)
    assert torch.equal(state_got, state_expected)
    if arguments["do_sample"]:
        assert len({tuple(g.flatten().tolist()) for g in got}) > 1


def test_diff_and_summarize():
    a = torch.tensor([1.0, 2.0, 3.0])
    b = torch.tensor([1.0, 2.0, 3.5])
    same = probe.diff(torch, a, a)
    other = probe.diff(torch, a, b)
    assert same == {"equal": True, "max_abs": 0.0, "rel": 0.0}
    assert other["equal"] is False
    assert other["max_abs"] == pytest.approx(0.5)
    assert other["rel"] == pytest.approx(0.5 / float(b.norm()))
    summary = probe.summarize([same, other])
    assert summary == {"all_equal": False, "steps_equal": 1, "max_abs": other["max_abs"], "max_rel": other["rel"]}
    assert probe.summarize([])["all_equal"] is True


def test_parse_args_defaults():
    args = probe.parse_args([])
    assert (args.prefix, args.steps, args.frames, args.rounds, args.seed) == (450, 32, 32, 3, 0)
    assert args.out == probe.DEFAULT_OUT
