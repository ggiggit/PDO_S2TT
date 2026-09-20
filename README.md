<div align="center">

# PDO_S2TT

### Persistent Delivery Optimization for Streaming Speech-to-Text Translation with Revisions

Official inference, evaluation, and PDO training repository for the **ICASSP 2027 submission**.

[![Model](https://img.shields.io/badge/Model-Hugging%20Face-FFD21E.svg?logo=huggingface&logoColor=black)](https://huggingface.co/hf-wzx1205/PDO_S2TT)
[![License](https://img.shields.io/badge/License-Apache%202.0-4C8BF5.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.12-3776AB.svg?logo=python&logoColor=white)](https://www.python.org/)
[![Directions](https://img.shields.io/badge/Directions-5-2E8B57.svg)](#released-fleurs-test-results)

</div>

PDO translates a growing English speech stream directly into a revisable target-language display. This release reproduces the paper's FLEURS TEST results for **En→Zh, En→De, En→Es, En→Ja, and En→Fr**, and includes the reinforcement-learning stage from the released SFT initialization.

## Highlights

- **Direct streaming S2TT:** speech prefixes are translated without exposing an intermediate transcript.
- **Revision-aware delivery:** the model may wait, append, or revise the complete visible translation as speech arrives.
- **One-command reproduction:** download FLEURS TEST, run all streaming stages, report every paper metric, and verify released results.
- **Reproducible RL stage:** download the released SFT initialization and run the paper's G=4 multilingual PDO recipe on four GPUs.

## Model checkpoint

> [!IMPORTANT]
> **Both released checkpoints are hosted at [🤗 `hf-wzx1205/PDO_S2TT`](https://huggingface.co/hf-wzx1205/PDO_S2TT).**
>
> - [`pdo_s2tt.pt`](https://huggingface.co/hf-wzx1205/PDO_S2TT/blob/main/pdo_s2tt.pt): paper inference checkpoint.
> - [`pdo_s2tt_sft.pt`](https://huggingface.co/hf-wzx1205/PDO_S2TT/blob/main/pdo_s2tt_sft.pt): SFT initialization for reproducing PDO training.
>
> The Qwen3-ASR-1.7B base model is fetched automatically on first use. For an offline run, download or copy it to `checkpoints/Qwen3-ASR-1.7B`; both inference and training detect that directory automatically.

## Quick start: reproduce En→Zh

Requirements: Linux, Python 3.12, CUDA, and a GPU with at least 16 GB memory.

```bash
git clone https://github.com/ggiggit/PDO_S2TT.git
cd PDO_S2TT

python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .

hf download hf-wzx1205/PDO_S2TT pdo_s2tt.pt --local-dir checkpoints
bash run_fleurs.sh
```

The default recipe performs the complete experiment:

1. downloads the official English FLEURS TEST split;
2. builds the 647-example En→Zh manifest;
3. runs revision-capable streaming inference;
4. reports BLEU, COMET, chrF++, FTL, FRD, LAAL-CU, RTF, and revision-aware metrics;
5. checks hardware-independent results against the released values.

FLEURS TEST audio is approximately 277 MB. The first run also downloads `Qwen/Qwen3-ASR-1.7B` and `Unbabel/wmt22-comet-da`. Data and predictions are written to `data/` and `results/`; both are ignored by Git.

## Reproduce all five directions

[`run_fleurs.sh`](run_fleurs.sh) contains one `run_direction` call for every supported direction. En→Zh is enabled by default; uncomment the other four calls to run all directions. Then uncomment the final macro command.

```bash
run_direction zh
# run_direction de
# run_direction es
# run_direction ja
# run_direction fr

# python scripts/evaluate_macro.py
```

The macro command reads the five metric files and their saved trajectories, then writes `results/fleurs/macro.json`. Quality and revision means are direction-level macro averages; latency percentiles are computed over all 3,235 trajectories.

## Released FLEURS TEST results

Quality and latency:

| Direction | BLEU ↑ | COMET ↑ | chrF++ ↑ | FTL ↓ | FRD ↓ | LAAL-CU mean / P90 ↓ | RTF ↓ |
|:--|--:|--:|--:|--:|--:|--:|--:|
| En→Zh | 38.16 | 86.31 | 26.92 | 2.00 | 2.33 | 3.53 / 6.61 | .39 |
| En→De | 29.57 | 84.22 | 57.03 | 2.00 | 2.42 | 3.08 / 5.65 | .57 |
| En→Es | 23.35 | 83.57 | 50.88 | 2.00 | 2.44 | 2.51 / 4.44 | .57 |
| En→Ja | 28.66 | 88.50 | 26.47 | 2.00 | 2.47 | 3.67 / 6.23 | .57 |
| En→Fr | 38.16 | 84.26 | 62.06 | 2.00 | 2.44 | 2.46 / 4.93 | .59 |
| **Five-direction macro** | **31.58** | **85.37** | **44.67** | **2.00** | **2.42** | **3.05 / 5.66** | **.54** |

Revision-aware delivery:

| Direction | Normalized erasure ↓ | Age-weighted erasure ↓ | First stable unit (s) ↓ | Mean finalization (s) ↓ |
|:--|--:|--:|--:|--:|
| En→Zh | 1.113 | 2.369 | 3.337 | 7.744 |
| En→De | .923 | 1.966 | 2.957 | 7.355 |
| En→Es | .641 | 1.397 | 2.743 | 6.803 |
| En→Ja | 1.345 | 2.709 | 3.136 | 8.026 |
| En→Fr | .660 | 1.467 | 2.715 | 6.890 |
| **Five-direction macro** | **.936** | **1.982** | **2.978** | **7.363** |

Times are in seconds unless noted otherwise. FRD and RTF include measured computation and therefore depend on hardware; the table reports measurements on one RTX 4090. All five directions have 100% output coverage.

## Run the steps separately

```bash
# 1. Download FLEURS and prepare one direction.
python scripts/prepare_fleurs.py --target zh

# 2. Run streaming inference.
python scripts/infer_fleurs.py \
  --target zh \
  --manifest data/fleurs/en-zh/test.jsonl \
  --output results/fleurs/en-zh/predictions.jsonl

# 3. Evaluate the saved trajectories.
python scripts/evaluate_fleurs.py \
  --target zh \
  --manifest data/fleurs/en-zh/test.jsonl \
  --predictions results/fleurs/en-zh/predictions.jsonl \
  --output results/fleurs/en-zh/metrics.json

# 4. After all five directions are complete, compute the macro result.
python scripts/evaluate_macro.py
```

Use `--skip-comet` for a faster deterministic-only evaluation; the output then marks the full reproduction check as incomplete. For a one-example smoke test, add `--limit 1` to inference and evaluation. Inference is resumable: rerunning the command skips completed utterances.

Chinese BLEU uses SacreBLEU's `zh` tokenizer, Japanese uses `ja-mecab`, and the remaining languages use `13a`.

## Reproduce PDO training

The public training release starts from the SFT policy used by PDO; it does not include the preceding supervised-training stages. The full recipe uses four GPUs with at least 24 GB each and processes all 13,000 FLEURS TRAIN direction examples once.

```bash
# Download the five-direction FLEURS TRAIN manifest and English audio.
python scripts/prepare_fleurs_train.py

# Download the released SFT initialization.
hf download hf-wzx1205/PDO_S2TT pdo_s2tt_sft.pt --local-dir checkpoints

# Run 407 G=4 rollout rounds and 1,625 synchronized AdamW updates.
bash train_pdo.sh
```

The final inference checkpoint is written to `results/training/pdo_s2tt.pt`. It can be passed directly to `scripts/infer_fleurs.py`:

```bash
python scripts/infer_fleurs.py \
  --target zh \
  --manifest data/fleurs/en-zh/test.jsonl \
  --checkpoint results/training/pdo_s2tt.pt \
  --output results/trained/en-zh/predictions.jsonl
```

On an offline machine, place the base model at `checkpoints/Qwen3-ASR-1.7B` or pass `--base-model /path/to/Qwen3-ASR-1.7B` to `scripts/train_pdo.py`.

Before committing a full run, use the same real streaming and backward path on one utterance per GPU:

```bash
torchrun --standalone --nproc-per-node=4 scripts/train_pdo.py \
  --manifest data/fleurs/train/train.jsonl \
  --sft-checkpoint checkpoints/pdo_s2tt_sft.pt \
  --output results/smoke \
  --smoke
```

The paper recipe is fixed as follows:

| Setting | Value |
|:--|:--|
| Directions | En→Zh/De/Es/Ja/Fr |
| Training units | 2,600 recordings × 5 directions |
| Acoustic update / input packet | 2.0 s / 0.1 s |
| Samples per utterance | 4 |
| Sampling temperature | 0.2 |
| Rollout batch / optimizer minibatch | 32 / 8 utterances |
| Trainable parameters | text-decoder LoRA + private history adapter |
| Optimizer | AdamW, LR `1e-6`, weight decay `.01` |
| Gradient clipping | 1.0 |
| Schedule | 407 rollout rounds / 1,625 updates / one TRAIN pass |

On four RTX 4090 GPUs, a complete run takes approximately 10 hours. The trainer writes resumable state and an inference checkpoint after every rollout round.

For every displayed draft, the process term scores only the prefix that remains unchanged in all later drafts. The terminal term is language-aware sentence BLEU. The complete future return `G_t = Φ_T − Φ_(t−1)` is standardized across the four trajectories at each event; no value model or direct latency reward is used. [`reward.py`](src/pdo_s2tt/training/reward.py) contains the complete reward definition.

As a reproduction check, the released SFT and PDO checkpoints give the following five-direction FLEURS TEST macro results. Independent full training runs are stochastic; small deviations are expected.

| Checkpoint | BLEU ↑ | COMET ↑ | chrF++ ↑ | FTL ↓ | FRD ↓ | LAAL-CU mean / P90 ↓ |
|:--|--:|--:|--:|--:|--:|--:|
| Released SFT initialization | 30.77 | 85.61 | 44.07 | 2.00 | 2.45 | 3.42 / 6.38 |
| Released PDO | 31.58 | 85.37 | 44.67 | 2.00 | 2.42 | 3.05 / 5.66 |

### DEV checkpoint selection used in the paper

Checkpoint selection did not use the PDO reward or a weighted quality-latency score. We first retained checkpoints with no empty outputs, COMET within 0.30 of the best candidate from the same run, BLEU within 1.0 of that run's best candidate, and COMET no more than 0.50 below the selected History-SFT checkpoint. Among the remaining checkpoints, we selected lexicographically by lower LAAL-CU mean, lower LAAL-CU P90, lower FRD, and finally higher COMET. A candidate was promoted only if it passed the quality filters and improved either LAAL-CU mean by at least 0.20 s or LAAL-CU P90 by at least 0.50 s relative to History-SFT.

## Translate one WAV file

Input audio must be mono 16 kHz WAV. Other WAV encodings are converted to 16-bit PCM before streaming.

```bash
pdo-s2tt path/to/audio.wav --target zh
```

The command prints each complete visible translation as it is revised. Add `--jsonl` to emit the full event trace.

## FLEURS data

The preparation scripts download English TSV/audio archives from the official [`google/fleurs`](https://huggingface.co/datasets/google/fleurs) repository. Test references are aligned by the official FLEURS/FLORES sentence IDs. For training, `references/fleurs_train_targets.jsonl.gz` freezes the exact 13,000 direction targets and order used by the paper: human FLEURS translations where available and the original frozen teacher translations for missing language directions. Target-language audio is not required.

FLEURS is distributed under CC BY 4.0. Its FLORES-derived text remains subject to the corresponding FLORES attribution and share-alike terms.

## License

The PDO_S2TT inference code and released checkpoint are licensed under the [Apache License 2.0](LICENSE). The `Qwen/Qwen3-ASR-1.7B` base model is distributed separately under its own Apache-2.0 license. FLEURS-derived references are not relicensed by this repository and remain subject to CC BY 4.0 and the applicable FLORES terms.

Copyright 2026 PDO_S2TT Authors.
