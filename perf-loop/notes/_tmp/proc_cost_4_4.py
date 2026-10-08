"""Dev check for 4.4 (not part of the app): the end rule's processor alone on the GPU, no model.

1. With ``torch.cuda.set_sync_debug_mode("error")`` it runs 300 steps as in the talker (batch 1 and 4): any host read
   (``.item()``, ``.tolist()``, ``bool`` of a tensor, a blocking copy) raises. The first call of each ``generate`` copies
   the guards to the GPU once; it runs before the check.
2. Per-call cost over 3000 calls: host time (what a dispatch-bound talker step pays) and GPU time, next to HF's own
   temperature + top-k warpers and sampling on the same scores, which every talker step already runs.

Usage:
    python perf-loop/notes/_tmp/proc_cost_4_4.py
"""

import json
import logging
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

VOCAB = 3072
EOS = 2150
CALLS = 3000


def _time(torch, fn) -> dict:
    for i in range(50):
        fn(i)
    torch.cuda.synchronize()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    started = time.perf_counter()
    start.record()
    for i in range(CALLS):
        fn(i)
    end.record()
    host = time.perf_counter() - started
    torch.cuda.synchronize()
    return {"host_us": round(host / CALLS * 1e6, 1), "gpu_us": round(start.elapsed_time(end) / CALLS * 1e3, 1)}


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    import torch
    from transformers.generation.logits_process import TemperatureLogitsWarper, TopKLogitsWarper

    from backend.utils import qwen_end_fix

    device = torch.device("cuda")
    report: dict = {"device": torch.cuda.get_device_name(0), "hip": torch.version.hip}
    for rows in (1, 4):
        scores = torch.randn(rows, VOCAB, device=device)
        ids = torch.zeros(rows, 4096, dtype=torch.long, device=device)

        def make():
            return qwen_end_fix.EosHoldProcessor(EOS, temperature=0.9, top_k=50, top_p=1.0, guards=[60] * rows)

        processor = make()
        processor(ids[:, :100], scores.clone())
        torch.cuda.synchronize()
        torch.cuda.set_sync_debug_mode("error")
        try:
            for n in range(101, 401):
                processor(ids[:, :n], scores.clone())
        finally:
            torch.cuda.set_sync_debug_mode(0)
        torch.cuda.synchronize()

        timed = make()
        work = scores.clone()
        timed(ids[:, :100], work)
        temperature, top_k = TemperatureLogitsWarper(0.9), TopKLogitsWarper(top_k=50)

        def hf_step(_i):
            probs = torch.softmax(top_k(None, temperature(None, scores)), dim=-1)
            return torch.multinomial(probs, num_samples=1)

        report[f"rows_{rows}"] = {
            "sync_check": "no host sync in 300 steps",
            "processor": _time(torch, lambda i: timed(ids[:, : 101 + i % 3000], work)),
            "hf_warpers_and_sampling": _time(torch, hf_step),
        }
        logging.info("rows=%d %s", rows, report[f"rows_{rows}"])
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
