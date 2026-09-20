#!/usr/bin/env python3
"""Run the released PDO policy on a prepared FLEURS manifest."""

from __future__ import annotations

import argparse
from dataclasses import asdict, is_dataclass
import json
from pathlib import Path
import time

from pdo_s2tt.audio import packets, read_wav
from pdo_s2tt.languages import LANGUAGE_NAMES
from pdo_s2tt.model import PDOS2TT


def event_dict(event) -> dict:
    if is_dataclass(event):
        return asdict(event)
    if isinstance(event, dict):
        return dict(event)
    return dict(vars(event))


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", default="zh", choices=sorted(LANGUAGE_NAMES))
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument(
        "--base-model",
        help="base model ID or local directory (auto-detects checkpoints/Qwen3-ASR-1.7B)",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--limit", type=int, help="run only the first N examples")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    manifest = args.manifest or Path(f"data/fleurs/en-{args.target}/test.jsonl")
    output = args.output or Path(f"results/fleurs/en-{args.target}/predictions.jsonl")
    checkpoint = args.checkpoint or repo_root / "checkpoints" / "pdo_s2tt.pt"
    rows = read_jsonl(manifest)
    if args.limit is not None:
        rows = rows[: args.limit]
    if not rows or any(row.get("target_lang") != args.target for row in rows):
        raise ValueError("manifest is empty or does not match --target")

    output.parent.mkdir(parents=True, exist_ok=True)
    completed = {}
    if output.exists():
        saved = read_jsonl(output)
        if any(row.get("target_lang") != args.target for row in saved):
            raise ValueError("existing predictions were produced for another target language")
        completed = {row["id"]: row for row in saved}
        if len(completed) != sum(1 for _ in output.open(encoding="utf-8")):
            raise ValueError("prediction file contains duplicate IDs")
        unknown = set(completed) - {row["id"] for row in rows}
        if unknown:
            raise ValueError("existing prediction IDs do not match this manifest")
    pending = [row for row in rows if row["id"] not in completed]
    if not pending:
        print(f"Already complete: {len(completed)} predictions in {output}")
        return

    model = PDOS2TT(checkpoint, args.target, args.base_model, args.device)
    with output.open("a", encoding="utf-8", newline="\n") as stream:
        for index, row in enumerate(pending, 1):
            model.reset()
            audio_path = Path(row["audio"])
            if not audio_path.is_absolute():
                audio_path = manifest.parent / audio_path
            pcm = read_wav(audio_path)
            started = time.perf_counter()
            events = []
            for packet, final in packets(pcm):
                emitted = list(model.accept_pcm(packet, final=final))
                cumulative_compute = time.perf_counter() - started
                for event in emitted:
                    item = event_dict(event)
                    item.setdefault("input_audio_end_time", item.get("audio_time", 0.0))
                    item.setdefault("wall_time_sec", cumulative_compute)
                    item.setdefault("display_wall_time_sec", cumulative_compute)
                    item.setdefault("display_target_prefix", item.get("target_prefix", ""))
                    events.append(item)
            elapsed = time.perf_counter() - started
            if not events:
                raise RuntimeError(f"no translation events for {row['id']}")
            hypothesis = str(events[-1].get("target_prefix", "")).strip()
            result = {
                "id": row["id"],
                "target_lang": args.target,
                "target_language": LANGUAGE_NAMES[args.target],
                "audio": str(audio_path),
                "audio_duration_sec": float(row["audio_duration_sec"]),
                "source_text": row["source_text"],
                "reference": row["target_text"],
                "translation": hypothesis,
                "hypothesis": hypothesis,
                "events": events,
                "event_count": len(events),
                "compute_time_sec": elapsed,
                "compute_rtf": elapsed / float(row["audio_duration_sec"]),
            }
            stream.write(json.dumps(result, ensure_ascii=False) + "\n")
            stream.flush()
            print(f"[{len(completed) + index}/{len(rows)}] {row['id']}", flush=True)
    print(f"Saved {len(rows)} predictions: {output}")


if __name__ == "__main__":
    main()
