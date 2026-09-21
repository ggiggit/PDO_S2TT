# Lightweight reproduction receipts

This directory keeps the reviewable text record of the independent public-code
reproduction:

- [`train.log`](train.log): complete sanitized 407-round training log;
- [`test.log`](test.log): compact five-direction FLEURS TEST receipt;
- [`receipt.json`](receipt.json): machine-readable completion, artifact, and
  macro-metric record.

The fresh inference checkpoint, resumable optimizer state, per-language evaluator
logs, complete trajectories, and metric JSON files are hosted separately at
[`hf-wzx1205/PDO_S2TT-reproduction`](https://huggingface.co/hf-wzx1205/PDO_S2TT-reproduction).
They are intentionally not stored with the paper checkpoint.
