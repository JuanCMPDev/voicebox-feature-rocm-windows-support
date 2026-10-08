"""Dev check for 3.5b (not tracked): where the memory that stays after release_graph comes from."""

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

    def mem(label: str) -> None:
        torch.cuda.synchronize()
        logging.info(
            "%-40s allocated %8.1f MiB, reserved %8.1f MiB",
            label,
            torch.cuda.memory_allocated() / 2**20,
            torch.cuda.memory_reserved() / 2**20,
        )

    model = profile.load_model(torch, "1.7B")
    qcp.install(model)
    generate = model.model.talker.code_predictor.generate
    args = qcp.talker_arguments(model)
    embeds = torch.randn(1, 2, 2048, device="cuda", dtype=torch.bfloat16)
    torch.cuda.empty_cache()
    mem("loaded")
    with torch.inference_mode():
        generate(inputs_embeds=embeds, **args)
    mem("one loop frame (default stream)")
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side), torch.inference_mode():
        generate(inputs_embeds=embeds, **args)
    torch.cuda.current_stream().wait_stream(side)
    mem("one loop frame (side stream)")
    torch._C._cuda_clearCublasWorkspaces()
    torch.cuda.empty_cache()
    mem("after clearCublasWorkspaces")
    assert qcp.capture_graph(model)
    mem("captured")
    qcp.release_graph(model)
    torch.cuda.empty_cache()
    mem("released + empty_cache")
    torch._C._cuda_clearCublasWorkspaces()
    torch.cuda.empty_cache()
    mem("+ clearCublasWorkspaces")
    with torch.inference_mode():
        generate(inputs_embeds=embeds, **args)
    mem("one loop frame after that")
    return 0


if __name__ == "__main__":
    sys.exit(main())
