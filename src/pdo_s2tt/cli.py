"""Command-line entry point for a single WAV utterance."""

import argparse
from dataclasses import asdict
import json
from pathlib import Path

from .audio import packets, read_wav
from .model import PDOS2TT


def default_checkpoint() -> Path:
    return Path(__file__).resolve().parents[2] / "checkpoints" / "pdo_s2tt.pt"


def main() -> None:
    parser = argparse.ArgumentParser(description="PDO streaming S2TT inference")
    parser.add_argument("audio", help="16 kHz WAV audio")
    parser.add_argument("--target", required=True, choices=["zh", "de", "es", "ja", "fr"])
    parser.add_argument("--checkpoint", default=str(default_checkpoint()))
    parser.add_argument("--base-model", default="Qwen/Qwen3-ASR-1.7B")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--jsonl", action="store_true", help="emit complete event objects")
    args = parser.parse_args()

    model = PDOS2TT(args.checkpoint, args.target, args.base_model, args.device)
    observed = []
    for event in model.stream(packets(read_wav(args.audio))):
        observed.append(event)
        if args.jsonl:
            print(json.dumps(asdict(event), ensure_ascii=False))
        else:
            print(f"[{event.audio_time:6.2f}s] {event.target_prefix}", flush=True)
    if not observed:
        raise RuntimeError("the audio produced no streaming events")


if __name__ == "__main__":
    main()
