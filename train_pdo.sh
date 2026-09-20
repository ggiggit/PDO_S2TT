#!/usr/bin/env bash
set -euo pipefail

# Full paper recipe: 13,000 five-direction FLEURS units, 407 rollout rounds,
# four trajectories per utterance, and 1,625 synchronized AdamW updates.
torchrun --standalone --nproc-per-node=4 scripts/train_pdo.py \
  --manifest data/fleurs/train/train.jsonl \
  --sft-checkpoint checkpoints/pdo_s2tt_sft.pt \
  --output results/training
