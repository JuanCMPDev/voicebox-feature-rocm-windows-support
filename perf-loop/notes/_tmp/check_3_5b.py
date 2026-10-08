"""Dev check for 3.5b (not tracked): talker codes and speed with the 3.2 loop vs graph B (the loop captured) vs graph A
(the 3.5a probe's StaticCache + 4D mask frame), on real phrases with the bench's cached voice prompt.

Usage:
    backend/venv/Scripts/python.exe perf-loop/notes/_tmp/check_3_5b.py --phrases p01,p02,p08 --rounds 2
"""

import argparse
import importlib.util
import json
import logging
import sys
from pathlib import Path
from time import perf_counter

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("--phrases", default="p01,p02,p08")
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--variants", default="loop,B,A")
    parser.add_argument("--out", type=Path, default=ROOT / "perf-loop" / "bench" / "_dev" / "check-3.5b.json")
    args = parser.parse_args()

    import torch
    from transformers import StaticCache

    from backend.utils import qwen_code_predictor as qcp

    profile = _load("profile_talker")
    probe = _load("probe_hip_graph")

    model = profile.load_model(torch, "1.7B")
    cached = profile.validate_cached_prompt(
        torch.load(profile.find_prompt_file(profile.DEFAULT_DATA_DIR, "1.7B"), map_location="cpu", weights_only=True)
    )
    assert qcp.install(model)
    generate = model.model.talker.code_predictor.generate
    predictor = model.model.talker.code_predictor

    # Graph B: the module's own capture.
    started = perf_counter()
    assert qcp.capture_graph(model), "capture B failed"
    graph_b = generate.frame_graph
    capture_b_s = perf_counter() - started
    plan = graph_b.plan

    # Graph A: the probe's static frame on the same plan, through the same FrameGraph plumbing.
    class StaticFrameGraph(qcp.FrameGraph):
        def __init__(self, embeds):
            super().__init__(predictor, plan, embeds)
            positions = torch.arange(16, device=embeds.device)
            cache = StaticCache(config=predictor.config, max_cache_len=16)
            self._fn = probe.code_predictor_frame(
                torch, predictor, plan, cache, embeds, positions, probe.frame_masks(positions, 2)
            )

        def _frame(self):
            return self._fn()

    graph_a = None
    if "A" in args.variants.split(","):
        with torch.inference_mode(False):
            embeds_a = torch.zeros_like(graph_b.embeds)
        graph_a = StaticFrameGraph(embeds_a)
        started = perf_counter()
        graph_a.capture()
        capture_a_s = perf_counter() - started
    else:
        capture_a_s = None

    variants = {"loop": None, "B": graph_b, "A": graph_a}
    report = {"capture_s": {"B": capture_b_s, "A": capture_a_s}, "phrases": {}}
    for phrase_id in args.phrases.split(","):
        phrase = profile.load_phrase(profile.DEFAULT_PHRASES, phrase_id)
        inputs = profile.build_talker_inputs(model, phrase, cached, "cuda")
        result = {}
        codes_ref = None
        for name in args.variants.split(","):
            generate.frame_graph = variants[name]
            times, codes_runs, rng = [], [], []
            for _ in range(args.rounds):
                profile.manual_seed(torch, phrase["seed"])
                torch.cuda.synchronize()
                t0 = perf_counter()
                with torch.inference_mode():
                    codes, _ = model.model.generate(**inputs)
                torch.cuda.synchronize()
                times.append(perf_counter() - t0)
                codes_runs.append([c.clone() for c in codes])
                rng.append(torch.cuda.get_rng_state())
            frames = int(codes_runs[0][0].shape[0])
            if codes_ref is None:
                codes_ref = codes_runs[0]
                rng_ref = rng[0]
            same = all(
                len(run) == len(codes_ref) and all(torch.equal(a, b) for a, b in zip(run, codes_ref, strict=True))
                for run in codes_runs
            )
            first_diff = None
            if not same:
                a, b = codes_runs[0][0], codes_ref[0]
                n = min(a.shape[0], b.shape[0])
                rows = (a[:n] != b[:n]).any(-1).nonzero()
                first_diff = int(rows[0]) if rows.numel() else n
            result[name] = {
                "frames": frames,
                "s": [round(t, 3) for t in times],
                "steps_per_s": round(frames / min(times), 2),
                "codes_equal_loop": same,
                "rng_equal_loop": all(torch.equal(r, rng_ref) for r in rng),
                "first_diff_frame": first_diff,
            }
            logging.info("%s %s: %s", phrase_id, name, result[name])
        report["phrases"][phrase_id] = result
    generate.frame_graph = graph_b
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    logging.info("report in %s", args.out)
    qcp.release_graph(model)
    if graph_a is not None:
        graph_a.release()
    return 0


if __name__ == "__main__":
    sys.exit(main())
