"""Dev diagnosis for 4.4 (not part of the app): where the talker emits codec_eos on the bench phrases.

Loads the model as the dev server does (``PyTorchTTSBackend._load_model_sync``: code predictor loop, HIP graph,
warm-up), takes the bench profile's cached voice prompt and, per phrase and seed of scripts/bench_phrases.json, runs
``generate_voice_clone`` with the arguments ``PyTorchTTSBackend.generate`` passes. A logits processor that only reads
the scores records, per talker step, codec_eos's probability under the sampling distribution (temperature, then top-k,
as HF applies them after the processors) and its rank. Nothing it does changes the scores or the RNG.

Usage:
    python perf-loop/notes/_tmp/diag_4_4.py <configs> [p01,p02,...] [seed offsets: 0,1000,...] [batch]
    configs: comma-separated mode[:hold_frames:eos_likely[:ratio]]; mode is one of
      stream (today), nonstream (non_streaming_mode=True), minframes (stream + EOS banned before ratio x estimate),
      stream-hold / nonstream-hold (+ backend.utils.qwen_end_fix.EosHoldProcessor)
    batch: also p02-p04 in one call seeded with p02's seed, as the bench's batch scenario.

Writes perf-loop/bench/_dev/4.4-diag-<config>/<warm|batch>[-s<offset>]/<id>.wav and diag.json.
"""

import json
import logging
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

PHRASES = ROOT / "scripts" / "bench_phrases.json"
DATA = ROOT / "perf-loop" / "bench" / "_data"
REF = ROOT / "perf-loop" / "bench" / "3.G-a1" / "warm"


def tail_silence_ms(
    audio, sr: int, frame_ms: int = 10, threshold_db: float = -40.0
) -> int:
    import numpy as np

    pcm = np.clip(audio, -1.0, 1.0)
    pcm = (pcm * 32767).astype(np.int16).astype(np.float64) / 32768.0
    frame = int(sr * frame_ms / 1000)
    n = len(pcm) // frame
    threshold = 10 ** (threshold_db / 20)
    silent = 0
    for i in range(n - 1, -1, -1):
        rms = math.sqrt(float(np.mean(pcm[i * frame : (i + 1) * frame] ** 2)))
        if rms >= threshold:
            break
        silent += 1
    return silent * frame_ms


def frame_dbfs(audio, frame: int = 1920, last: int = 6) -> list[float]:
    import numpy as np

    n = len(audio) // frame
    out = []
    for i in range(max(0, n - last), n):
        rms = math.sqrt(float(np.mean(audio[i * frame : (i + 1) * frame] ** 2)))
        out.append(round(20 * math.log10(max(rms, 1e-9)), 1))
    return out


def wav_seconds(path: Path) -> float | None:
    import wave

    if not path.is_file():
        return None
    with wave.open(str(path), "rb") as w:
        return w.getnframes() / w.getframerate()


def parse_config(text: str) -> dict:
    parts = text.split(":")
    mode = parts[0]
    hold = mode.endswith("hold")
    config = {
        "mode": mode,
        "nonstream": mode.startswith("nonstream"),
        "hold": hold,
        "hold_frames": int(parts[1]) if len(parts) > 1 else 2,
        "eos_likely": float(parts[2]) if len(parts) > 2 else 0.1,
        "ratio": float(parts[3]) if len(parts) > 3 else (0.0 if hold else 0.8),
    }
    config["tag"] = (
        f"{mode}-k{config['hold_frames']}-p{config['eos_likely']}" if hold else mode
    )
    if hold and config["ratio"]:
        config["tag"] += f"-g{config['ratio']}"
    return config


def main(argv: list[str]) -> int:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    import os

    os.environ["VOICEBOX_QWEN_END_FIX"] = (
        "0"  # the configs below choose the mode and add the processor themselves
    )
    configs = [parse_config(c) for c in (argv[0] if argv else "stream").split(",")]
    wanted = (
        argv[1].split(",") if len(argv) > 1 else ["p01", "p02", "p03", "p04", "p05"]
    )
    offsets = [int(x) for x in argv[2].split(",")] if len(argv) > 2 else [0]
    batch = len(argv) > 3 and argv[3] == "batch"

    import soundfile as sf
    import torch
    from transformers import LogitsProcessor, LogitsProcessorList

    from backend.backends import LANGUAGE_CODE_TO_NAME
    from backend.backends.base import manual_seed
    from backend.backends.pytorch_backend import (
        PyTorchTTSBackend,
        _generation_mode,
        _voice_prompt_from_cache,
    )
    from backend.utils.audio import normalize_audio
    from backend.utils.qwen_end_fix import EosHoldProcessor

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
    ref_frames = int(items[0].ref_code.shape[0])
    ref_text = items[0].ref_text
    frames_per_char = ref_frames / len(ref_text)
    settings = model._merge_generate_kwargs()
    temperature, top_k = settings["temperature"], settings["top_k"]
    eos = model.model.config.talker_config.codec_eos_token_id
    logging.info(
        "ref: %d frames, %d chars (%.3f frames/char)",
        ref_frames,
        len(ref_text),
        frames_per_char,
    )

    steps: list = []
    held_steps: list = []
    min_frames = [0]
    config: dict = {}

    class Recorder(LogitsProcessor):
        def __call__(self, input_ids, scores):
            s = scores.float() / temperature
            kth = torch.topk(s, top_k, dim=-1).values[..., -1:]
            s = s.masked_fill(s < kth, float("-inf"))
            p = torch.softmax(s, dim=-1)[:, eos]
            rank = (scores > scores[:, eos : eos + 1]).sum(-1)
            steps.append((p.clone(), rank.clone()))
            return scores

    class BanEos(LogitsProcessor):
        def __call__(self, input_ids, scores):
            if input_ids.shape[1] < min_frames[0]:
                scores[:, eos] = float("-inf")
            return scores

    class HeldRecorder(LogitsProcessor):
        def __call__(self, input_ids, scores):
            held_steps.append(torch.isneginf(scores[:, eos]).clone())
            return scores

    from backend.utils.qwen_end_fix import guard_frames

    guards_now: list = [None]
    talker = model.model.talker
    original_talker_generate = talker.generate

    def talker_generate(*args, **kwargs):
        procs = (
            [BanEos(), Recorder()] if config["mode"] == "minframes" else [Recorder()]
        )
        if config["hold"]:
            procs += [
                EosHoldProcessor(
                    eos,
                    config["hold_frames"],
                    config["eos_likely"],
                    temperature,
                    top_k,
                    guards=guards_now[0],
                ),
                HeldRecorder(),
            ]
        kwargs["logits_processor"] = LogitsProcessorList(procs)
        return original_talker_generate(*args, **kwargs)

    talker.generate = talker_generate
    codes_seen: list = []
    original_generate = model.model.generate

    def generate(*args, **kwargs):
        result = original_generate(*args, **kwargs)
        codes_seen.append(result[0])
        return result

    model.model.generate = generate

    # Groups of phrases generated together: each phrase alone, plus (with "batch") p02-p04 in one call seeded with
    # p02's seed, as the bench's batch scenario runs them.
    groups = [[phrase] for phrase in rows]
    if batch:
        groups.append(
            [p for p in phrases["phrases"] if p["id"] in ("p02", "p03", "p04")]
        )

    for current in configs:
        config.clear()
        config.update(current)
        out_dir = ROOT / "perf-loop" / "bench" / "_dev" / f"4.4-diag-{config['tag']}"
        results = []
        for offset in offsets:
            for group in groups:
                steps.clear()
                codes_seen.clear()
                held_steps.clear()
                min_frames[0] = (
                    math.ceil(config["ratio"] * frames_per_char * len(group[0]["text"]))
                    if config["mode"] == "minframes"
                    else 0
                )
                guards_now[0] = (
                    guard_frames([p["text"] for p in group], items, config["ratio"])
                    if config["hold"] and config["ratio"]
                    else None
                )
                manual_seed(group[0]["seed"] + offset, backend.device)
                with _generation_mode(backend.device):
                    wavs, sr = model.generate_voice_clone(
                        text=[p["text"] for p in group]
                        if len(group) > 1
                        else group[0]["text"],
                        voice_clone_prompt=items * len(group),
                        language=language,
                        instruct=None,
                        non_streaming_mode=config["nonstream"],
                    )
                pass_name = ("batch" if len(group) > 1 else "warm") + (
                    f"-s{offset}" if offset else ""
                )
                (out_dir / pass_name).mkdir(parents=True, exist_ok=True)
                for r, phrase in enumerate(group):
                    audio = normalize_audio(wavs[r])
                    n_frames = int(codes_seen[0][r].shape[0])
                    p_eos = [float(p[r]) for p, _ in steps[: n_frames + 1]]
                    ranks = [int(k[r]) for _, k in steps[: n_frames + 1]]
                    held = [
                        i
                        for i, h in enumerate(held_steps[: n_frames + 1])
                        if i >= 2 and bool(h[r]) and p_eos[i] >= config["eos_likely"]
                    ]
                    estimate = frames_per_char * len(phrase["text"])
                    sf.write(
                        str(out_dir / pass_name / f"{phrase['id']}.wav"),
                        audio,
                        sr,
                        subtype="PCM_16",
                    )
                    row = {
                        "id": phrase["id"],
                        "pass": pass_name,
                        "seed": group[0]["seed"] + offset,
                        "frames": n_frames,
                        "frames_over_estimate": round(n_frames / estimate, 3),
                        "p_eos_last6": [round(x, 4) for x in p_eos[-6:]],
                        "rank_eos_last6": ranks[-6:],
                        "p_eos_at_eos": round(p_eos[n_frames], 4)
                        if len(p_eos) > n_frames
                        else None,
                        "audio_s": round(len(audio) / sr, 3),
                        "tail_silence_ms": tail_silence_ms(audio, sr),
                        "last_frames_dbfs": frame_dbfs(audio, last=8),
                        "held_at": held,
                        "p_eos_at_held": [round(p_eos[i], 4) for i in held],
                        "guard": guards_now[0][r] if guards_now[0] else None,
                    }
                    logging.info("%s %s", config["tag"], row)
                    results.append(row)
        tails = [r["tail_silence_ms"] for r in results]
        summary = {
            "n": len(results),
            "tail_ge_100": sum(t >= 100 for t in tails),
            "tail_min": min(tails),
            "tail_median": sorted(tails)[len(tails) // 2],
            "cut": [
                f"{r['pass']}/{r['id']}:{r['tail_silence_ms']}"
                for r in results
                if r["tail_silence_ms"] < 100
            ],
            "eos_below_0.1": [
                f"{r['pass']}/{r['id']}:{r['p_eos_at_eos']}"
                for r in results
                if r["p_eos_at_eos"] is not None and r["p_eos_at_eos"] < 0.1
            ],
        }
        logging.info("SUMMARY %s %s", config["tag"], summary)
        (out_dir / "diag.json").write_text(
            json.dumps(
                {
                    **config,
                    "ref_frames": ref_frames,
                    "ref_chars": len(ref_text),
                    "summary": summary,
                    "rows": results,
                },
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
