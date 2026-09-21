#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

# Full public recipe: 13,000 five-direction FLEURS units, 407 rollout rounds,
# four trajectories per utterance, and 1,625 synchronized AdamW updates.
output=results/training
arguments=(
  --manifest data/fleurs/train/train.jsonl
  --sft-checkpoint checkpoints/pdo_s2tt_sft.pt
  --output "$output"
)

if [[ -f "$output/training_state.pt" ]]; then
  echo "Resuming from $output/training_state.pt"
  arguments+=(--resume "$output/training_state.pt")
fi

torchrun --standalone --nproc-per-node=4 scripts/train_pdo.py "${arguments[@]}"
