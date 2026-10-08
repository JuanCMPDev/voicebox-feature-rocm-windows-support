"""Dev check for 3.5b (not tracked): GPU memory of the code predictor graph, and that release_graph gives it back."""

import importlib.util
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    import torch

    from backend.utils import qwen_code_predictor as qcp

    spec = importlib.util.spec_from_file_location("profile_talker", ROOT / "scripts" / "profile_talker.py")
    profile = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(profile)

    def mib(value: int) -> float:
        return round(value / 2**20, 1)

    model = profile.load_model(torch, "1.7B")
    qcp.install(model)
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    before = (torch.cuda.memory_allocated(), torch.cuda.memory_reserved())
    assert qcp.capture_graph(model)
    generate = model.model.talker.code_predictor.generate
    with torch.inference_mode():
        embeds = torch.randn(1, 2, 2048, device="cuda", dtype=torch.bfloat16)
        args = qcp.talker_arguments(model)
        torch.cuda.manual_seed(1)
        graph_codes = generate(inputs_embeds=embeds, **args).sequences
    torch.cuda.synchronize()
    captured = (torch.cuda.memory_allocated(), torch.cuda.memory_reserved())
    qcp.release_graph(model)
    torch.cuda.empty_cache()
    released = (torch.cuda.memory_allocated(), torch.cuda.memory_reserved())
    with torch.inference_mode():
        torch.cuda.manual_seed(1)
        loop_codes = generate(inputs_embeds=embeds, **args).sequences
    logging.info(
        "allocated/reserved MiB: before %s, with graph %s, after release %s; loop after release == graph: %s",
        [mib(v) for v in before],
        [mib(v) for v in captured],
        [mib(v) for v in released],
        bool(torch.equal(graph_codes, loop_codes)),
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
