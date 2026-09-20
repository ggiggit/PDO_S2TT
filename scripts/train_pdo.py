#!/usr/bin/env python3
"""Train PDO from the released SFT checkpoint with torchrun."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import random
import time

import torch
import torch.distributed as dist

from pdo_s2tt.training.model import TrainingModel
from pdo_s2tt.training.data import LANGUAGES, validate_released_manifest
from pdo_s2tt.training.trainer import capture_proximal, rollout, update


def read_manifest(path: Path) -> list[dict]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    for row in rows:
        audio = Path(row["audio"])
        row["audio"] = str(audio if audio.is_absolute() else path.parent / audio)
    return rows


def distributed_environment(allow_single_gpu: bool) -> tuple[int, int, int]:
    world = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world > 1:
        if world != 4:
            raise ValueError("the released full recipe uses exactly four GPUs")
        torch.cuda.set_device(local_rank)
    elif not allow_single_gpu:
        raise ValueError("launch the full recipe with torchrun --nproc-per-node=4")
    return rank, world, local_rank


def save_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def model_fingerprint(model: TrainingModel) -> str:
    digest = hashlib.sha256()
    for name, parameter in model.named_parameters:
        value = parameter.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(str(tuple(value.shape)).encode("ascii"))
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("data/fleurs/train/train.jsonl"))
    parser.add_argument("--sft-checkpoint", type=Path, default=Path("checkpoints/pdo_s2tt_sft.pt"))
    parser.add_argument("--output", type=Path, default=Path("results/training"))
    parser.add_argument("--base-model")
    parser.add_argument("--seed", type=int, default=52)
    parser.add_argument("--max-rounds", type=int)
    parser.add_argument("--resume", type=Path, help="resume from training_state.pt")
    parser.add_argument("--smoke", action="store_true", help="one utterance per GPU and one update")
    args = parser.parse_args()

    rank, world, local_rank = distributed_environment(args.smoke)
    device = f"cuda:{local_rank}" if world > 1 else "cuda:0"
    rows = read_manifest(args.manifest)
    if not args.smoke:
        validate_released_manifest(
            rows,
            Path(__file__).resolve().parents[1]
            / "references"
            / "fleurs_train_targets.jsonl.gz",
        )

    model = TrainingModel(
        args.sft_checkpoint,
        base_model=args.base_model,
        device=device,
        learning_rate=1e-6,
    )
    start_round, updates = 0, 0
    if args.resume is not None:
        state = torch.load(args.resume, map_location="cpu", weights_only=True)
        start_round, updates = model.load_training_state(state)
    # PEFT 0.19.1 must load adapters before torch.distributed is initialized;
    # otherwise its tensor-parallel probe imports an unavailable integration.
    if world > 1:
        dist.init_process_group("nccl", device_id=torch.device(device))
    total_rounds = math.ceil(len(rows) / 32)
    if args.smoke:
        total_rounds = 1
    if args.max_rounds is not None:
        total_rounds = min(total_rounds, args.max_rounds)
    started = time.time()
    if rank == 0:
        args.output.mkdir(parents=True, exist_ok=True)
        save_json(args.output / "config.json", {
            "algorithm": "PDO",
            "mode": "smoke" if args.smoke else "full",
            "languages": sorted(LANGUAGES),
            "group_size": 4,
            "temperature": 0.2,
            "audio_update_seconds": 2.0,
            "input_chunk_seconds": 0.1,
            "global_rollout_batch": world if args.smoke else 32,
            "global_optimizer_minibatch": world if args.smoke else 8,
            "learning_rate": 1e-6,
            "weight_decay": 0.01,
            "gradient_clip": 1.0,
            "seed": args.seed,
        })
    if dist.is_initialized():
        dist.barrier()

    if not 0 <= start_round <= total_rounds:
        raise ValueError("resume round lies outside the requested schedule")
    for outer in range(start_round, total_rounds):
        global_rows = rows[32 * outer:32 * (outer + 1)]
        if args.smoke:
            global_rows = rows[:world]
        local_rows = global_rows[rank::world]
        local = []
        for index, row in enumerate(local_rows):
            seed = args.seed * 10_000_000 + outer * 100_000 + rank * 10_000 + index * 100
            local.append(rollout(model, row, seed))
        all_events = [event for item in local for event in item["events"]]
        capture_proximal(model, all_events)

        update_count = 1 if args.smoke else math.ceil(len(global_rows) / 8)
        receipts = []
        for minibatch in range(update_count):
            selected = local[minibatch * 2:(minibatch + 1) * 2]
            events = [event for item in selected for event in item["events"]]
            global_utterances = min(8, len(global_rows) - minibatch * 8)
            receipts.append(update(model, events, global_utterances))
            updates += 1

        fingerprint = model_fingerprint(model)
        if dist.is_initialized():
            replicas = [None] * world
            dist.all_gather_object(replicas, fingerprint)
            if len(set(replicas)) != 1:
                raise RuntimeError("distributed replicas diverged after the synchronized update")

        if rank == 0:
            checkpoint = args.output / "pdo_s2tt.pt"
            policy = model.state_dict(rounds=outer + 1, adam_updates=updates)
            checkpoint_temporary = checkpoint.with_suffix(".tmp")
            torch.save(policy, checkpoint_temporary)
            checkpoint_temporary.replace(checkpoint)
            training_state = args.output / "training_state.pt"
            temporary = training_state.with_suffix(".tmp")
            torch.save({
                "format": "pdo-s2tt-training-state-v1",
                "rounds": outer + 1,
                "adam_updates": updates,
                "policy": policy,
                "optimizer": model.optimizer.state_dict(),
            }, temporary)
            temporary.replace(training_state)
            save_json(args.output / "progress.json", {
                "completed_rounds": outer + 1,
                "completed_updates": updates,
                "total_rounds": total_rounds,
                "elapsed_seconds": time.time() - started,
                "last_updates": receipts,
                "model_fingerprint": fingerprint,
                "checkpoint": str(checkpoint),
            })
            print(f"round {outer + 1}/{total_rounds}; Adam updates {updates}", flush=True)
        if dist.is_initialized():
            dist.barrier()

    if rank == 0:
        save_json(args.output / "complete.json", {
            "status": "complete",
            "rounds": total_rounds,
            "adam_updates": updates,
            "elapsed_seconds": time.time() - started,
            "checkpoint": str(args.output / "pdo_s2tt.pt"),
        })
    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
