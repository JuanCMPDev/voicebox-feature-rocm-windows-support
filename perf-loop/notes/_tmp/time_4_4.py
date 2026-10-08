"""Dev timing for 4.4 (not part of the app): talker steps/s with and without each part of the end fix.

Same setup as diag_4_4.py (model loaded as the dev server does, bench prompt and seeds). For each configuration and
phrase it times ``model.model.generate`` between two GPU synchronizations, as the VBPERF ``talker`` line does, and
reports frames / seconds.

Usage:
    python perf-loop/notes/_tmp/time_4_4.py [p01,...] [rounds]
"""

import json
import logging
import os
import sys
from pathlib import Path
from time import perf_counter

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

PHRASES = ROOT / "scripts" / "bench_phrases.json"
DATA = ROOT / "perf-loop" / "bench" / "_data"


def main(argv: list[str]) -> int:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    os.environ["VOICEBOX_QWEN_END_FIX"] = (
        "0"  # the configurations below add each part themselves
    )
    wanted = argv[0].split(",") if argv else [f"p{i:02d}" for i in range(1, 13)]
    rounds = int(argv[1]) if len(argv) > 1 else 1

    import torch
    from transformers import LogitsProcessorList

    from backend.backends import LANGUAGE_CODE_TO_NAME
    from backend.backends.base import manual_seed
    from backend.backends.pytorch_backend import (
        PyTorchTTSBackend,
        _generation_mode,
        _voice_prompt_from_cache,
    )
    from backend.utils import qwen_end_fix

    phrases = json.loads(PHRASES.read_text(encoding="utf-8"))
    language = LANGUAGE_CODE_TO_NAME.get(phrases["language"], "auto")
    rows = [p for p in phrases["phrases"] if p["id"] in wanted]

    backend = PyTorchTTSBackend("1.7B")
    backend._load_model_sync("1.7B")
    model = backend.model
    prompt_file = next((DATA / "cache").glob("qwen_1.7B_*.prompt"))
    items = _voice_prompt_from_cache(
        torch.load(prompt_file, map_location="cpu", weights_only=True), backend.device
    )
    settings = model._merge_generate_kwargs()
    eos = model.model.config.talker_config.codec_eos_token_id

    use_processor = [False]
    guards_now: list = [None]
    talker = model.model.talker
    original_talker_generate = talker.generate

    def talker_generate(*args, **kwargs):
        if use_processor[0]:
            kwargs["logits_processor"] = LogitsProcessorList(
                [
                    qwen_end_fix.EosHoldProcessor(
                        eos,
                        temperature=settings["temperature"],
                        top_k=settings["top_k"],
                        guards=guards_now[0],
                    )
                ]
            )
        return original_talker_generate(*args, **kwargs)

    talker.generate = talker_generate
    timings: list = []
    original_generate = model.model.generate

    def generate(*args, **kwargs):
        torch.cuda.synchronize()
        started = perf_counter()
        result = original_generate(*args, **kwargs)
        torch.cuda.synchronize()
        timings.append(
            (
                perf_counter() - started,
                int(result[0][0].shape[0]),
                int(kwargs["input_ids"][0].shape[1]),
            )
        )
        return result

    model.model.generate = generate

    configs = {
        "stream": (False, False),
        "nonstream": (True, False),
        "stream+rule": (False, True),
        "fix": (True, True),
    }
    results: dict = {name: [] for name in configs}
    for _ in range(rounds):
        for name, (nonstream, processor) in configs.items():
            use_processor[0] = processor
            for phrase in rows:
                guards_now[0] = (
                    qwen_end_fix.guard_frames([phrase["text"]], items)
                    if processor
                    else None
                )
                timings.clear()
                manual_seed(phrase["seed"], backend.device)
                with _generation_mode(backend.device):
                    model.generate_voice_clone(
                        text=phrase["text"],
                        voice_clone_prompt=items,
                        language=language,
                        instruct=None,
                        non_streaming_mode=nonstream,
                    )
                seconds, frames, _ = timings[0]
                results[name].append(
                    {"id": phrase["id"], "frames": frames, "s": round(seconds, 4)}
                )
    summary = {}
    for name, runs in results.items():
        rates = [r["frames"] / r["s"] for r in runs]
        summary[name] = {
            "steps_per_s_mean": round(sum(rates) / len(rates), 3),
            "frames": sum(r["frames"] for r in runs),
            "seconds": round(sum(r["s"] for r in runs), 3),
            "steps_per_s_total": round(
                sum(r["frames"] for r in runs) / sum(r["s"] for r in runs), 3
            ),
        }
        logging.info("TIMING %s %s", name, summary[name])
    out = ROOT / "perf-loop" / "bench" / "_dev" / "4.4-timing.json"
    out.write_text(
        json.dumps({"summary": summary, "runs": results}, indent=2), encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
