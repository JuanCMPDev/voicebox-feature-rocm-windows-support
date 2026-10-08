"""Dev check for 3.5b (not tracked): per-forward code-predictor logits of the 3.2 loop vs the 3.5a probe's static
frame (A, StaticCache + 4D mask) and vs graph B's eager frame, on the real frame inputs of a phrase.
"""

import importlib.util
import json
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main(phrase_id: str = "p08") -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    import torch
    from transformers import StaticCache

    from backend.utils import qwen_code_predictor as qcp

    profile = _load("profile_talker")
    probe = _load("probe_hip_graph")
    model = profile.load_model(torch, "1.7B")
    cached = profile.validate_cached_prompt(
        torch.load(profile.find_prompt_file(profile.DEFAULT_DATA_DIR, "1.7B"), map_location="cpu", weights_only=True)
    )
    qcp.install(model)
    predictor = model.model.talker.code_predictor
    generate = predictor.generate
    plan = generate.plan_for((), {"inputs_embeds": torch.zeros(1, 2, 2048), **qcp.talker_arguments(model)})

    phrase = profile.load_phrase(profile.DEFAULT_PHRASES, phrase_id)
    inputs = profile.build_talker_inputs(model, phrase, cached, "cuda")
    frames: list = []
    real = generate

    def recording(*args, **kwargs):
        frames.append(kwargs["inputs_embeds"].clone())
        return real(*args, **kwargs)

    predictor.generate = recording
    profile.manual_seed(torch, phrase["seed"])
    with torch.inference_mode():
        model.model.generate(**inputs)
    predictor.generate = real

    logits: list = []
    handle = predictor.register_forward_hook(lambda _m, _a, out: logits.append(out.logits[:, -1].clone()))
    positions = torch.arange(16, device="cuda")
    cache = StaticCache(config=predictor.config, max_cache_len=16)
    embeds = torch.zeros_like(frames[0])
    static_fn = probe.code_predictor_frame(
        torch, predictor, plan, cache, embeds, positions, probe.frame_masks(positions, 2)
    )
    stats = {"frames": len(frames), "forwards": 0, "A_equal": 0, "A_max_abs": 0.0, "A_codes_equal": 0}
    with torch.inference_mode():
        for n, frame in enumerate(frames):
            torch.cuda.manual_seed(1000 + n)
            logits.clear()
            loop_codes = qcp.sample_with_loop(predictor, frame, plan).sequences
            loop_logits = list(logits)
            torch.cuda.manual_seed(1000 + n)
            logits.clear()
            embeds.copy_(frame)
            a_codes = static_fn()
            a_logits = list(logits)
            for x, y in zip(loop_logits, a_logits, strict=True):
                stats["forwards"] += 1
                stats["A_equal"] += int(torch.equal(x, y))
                stats["A_max_abs"] = max(stats["A_max_abs"], float((x.float() - y.float()).abs().max()))
            stats["A_codes_equal"] += int(torch.equal(loop_codes, a_codes))
    handle.remove()
    logging.info("%s", stats)
    out = ROOT / "perf-loop" / "bench" / "_dev" / "logits-3.5b.json"
    out.write_text(json.dumps(stats, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main(*sys.argv[1:]))
