#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

run_direction() {
  local target="$1"
  local data_dir="data/fleurs/en-${target}"
  local result_dir="results/fleurs/en-${target}"

  python scripts/prepare_fleurs.py --target "${target}" --output "${data_dir}"
  python scripts/infer_fleurs.py \
    --target "${target}" \
    --manifest "${data_dir}/test.jsonl" \
    --output "${result_dir}/predictions.jsonl"
  python scripts/evaluate_fleurs.py \
    --target "${target}" \
    --manifest "${data_dir}/test.jsonl" \
    --predictions "${result_dir}/predictions.jsonl" \
    --output "${result_dir}/metrics.json"
}

# En→Zh is the default reproduction. Uncomment the remaining directions to
# reproduce the complete five-direction table.
run_direction zh
# run_direction de
# run_direction es
# run_direction ja
# run_direction fr

# Run this after all five directions finish.
# python scripts/evaluate_macro.py
