# Inference and evaluation

This guide reproduces the paper checkpoint on the official English FLEURS TEST
split. It covers En→Zh, En→De, En→Es, En→Ja, and En→Fr and reports every metric
used in the paper.

## Requirements

- Linux and Python 3.12
- CUDA and a GPU with at least 16 GB memory
- `git`, `ffmpeg`, and a recent Hugging Face CLI

```bash
git clone https://github.com/ggiggit/PDO_S2TT.git
cd PDO_S2TT

python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
```

## Download the paper checkpoint

```bash
mkdir -p checkpoints
hf download hf-wzx1205/PDO_S2TT pdo_s2tt.pt \
  --revision 62c27b03aa1d7da1b739e43e815796029ee243f5 \
  --local-dir checkpoints
```

The Qwen3-ASR-1.7B base model and WMT22-COMET-DA are downloaded automatically
on first use. For offline evaluation, place the base model at
`checkpoints/Qwen3-ASR-1.7B`.

## Reproduce En→Zh

```bash
bash scripts/run_fleurs.sh
```

The default launcher downloads FLEURS TEST, builds the 647-example En→Zh
manifest, runs revision-capable streaming inference, evaluates all metrics, and
checks hardware-independent values against the released result.

FLEURS TEST audio is approximately 277 MB. Every direction reuses the same
English audio under `data/fleurs/audio/test`.

## Reproduce all five directions

The launcher contains all five calls; En→Zh is enabled and the other four are
commented to make the first run small:

```bash
run_direction zh
# run_direction de
# run_direction es
# run_direction ja
# run_direction fr

# python scripts/evaluate_macro.py
```

Uncomment the four directions and the macro command in
[`scripts/run_fleurs.sh`](../scripts/run_fleurs.sh). The macro evaluator reads
all five metric files and trajectories, writes `results/fleurs/macro.json`, and
computes pooled latency percentiles over all 3,235 records.

## Run each stage manually

```bash
# 1. Download and prepare one direction.
python scripts/prepare_fleurs.py --target zh

# 2. Run streaming inference.
python scripts/infer_fleurs.py \
  --target zh \
  --manifest data/fleurs/en-zh/test.jsonl \
  --output results/fleurs/en-zh/predictions.jsonl

# 3. Score final translations and the complete display trajectory.
python scripts/evaluate_fleurs.py \
  --target zh \
  --manifest data/fleurs/en-zh/test.jsonl \
  --predictions results/fleurs/en-zh/predictions.jsonl \
  --output results/fleurs/en-zh/metrics.json

# 4. Aggregate after all five directions complete.
python scripts/evaluate_macro.py
```

Inference is resumable: rerunning skips completed utterances. Add `--limit 1`
to inference and evaluation for a one-example smoke test. Add `--skip-comet` for
a quick deterministic-only check; the output then marks full reproduction as
incomplete.

## Metric conventions

| Metric | Definition |
|:--|:--|
| BLEU | SacreBLEU `zh` for Chinese, `ja-mecab` for Japanese, `13a` otherwise |
| COMET | Pinned WMT22-COMET-DA revision |
| chrF++ | Corpus chrF with word order 2 |
| FTL | Source-audio position of the first non-empty visible translation |
| FRD | Computation-aware wall-clock time of that first visible translation |
| LAAL-CU | Length-adaptive lag using the earliest source time after which each final prefix remains unchanged |
| RTF | Total inference compute time divided by source-audio duration |
| Normalized erasure | Erased units divided by final-output length |
| Age-weighted erasure | Each erased unit weighted by its visible lifetime |
| First stable unit | First time a final target unit appears and remains unchanged |
| Mean finalization | Mean final-prefix stabilization time over target units |

Revision units are characters for Chinese/Japanese and normalized words for
German/Spanish/French. Quality and revision macro means are direction-level
averages; latency P90 is pooled over all trajectories.

## Translate one WAV file

```bash
pdo-s2tt path/to/audio.wav --target zh
```

Input should be mono 16 kHz WAV. Other WAV encodings are converted to 16-bit
PCM before streaming. The command prints each complete visible translation as
it is revised; add `--jsonl` for the full event trace.

## Data and licenses

Preparation downloads English TSV/audio from
[`google/fleurs`](https://huggingface.co/datasets/google/fleurs). TEST references
are aligned through official FLEURS/FLORES sentence IDs. Audio and generated
results are written to ignored `data/` and `results/` directories.

FLEURS is CC BY 4.0. FLORES-derived text remains subject to the corresponding
attribution and share-alike terms.
