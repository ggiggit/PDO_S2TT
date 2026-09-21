<div align="center">

# PDO_S2TT

### Persistent Delivery Optimization for Streaming Speech-to-Text Translation with Revisions

Official repository for the **ICASSP 2027 submission**

[![Paper model](https://img.shields.io/badge/🤗_Paper_model-PDO__S2TT-FFD21E)](https://huggingface.co/hf-wzx1205/PDO_S2TT)
[![Reproduction](https://img.shields.io/badge/🤗_Reproduction-artifacts-FFD21E)](https://huggingface.co/hf-wzx1205/PDO_S2TT-reproduction)
[![Tests](https://github.com/ggiggit/PDO_S2TT/actions/workflows/tests.yml/badge.svg)](https://github.com/ggiggit/PDO_S2TT/actions/workflows/tests.yml)
[![License](https://img.shields.io/badge/License-Apache--2.0-4C8BF5)](LICENSE)

**Direct streaming S2TT · Revision-aware training · Five target languages**

</div>

PDO trains a streaming speech-to-text translation model to emit content that
appears early **and remains unchanged**. The model translates growing English
speech directly into revisable Chinese, German, Spanish, Japanese, or French,
without exposing an intermediate transcript.

## Choose your path

| I want to… | Start here | What it contains |
|:--|:--|:--|
| **Run the model** | [Inference & evaluation](docs/inference.md) | Install, checkpoint download, one WAV, FLEURS TEST, all metrics |
| **Train PDO** | [Training reproduction](docs/training.md) | SFT initialization, four-GPU command, resume, exact recipe, DEV selection |
| **Match paper formulas to code** | [Paper ↔ code](docs/paper-to-code.md) | Eq. (1)–(4), reward units, baseline, sampling, loss, LAAL-CU |

## Checkpoints

| Release | Use | Link |
|:--|:--|:--|
| **Paper release** | Paper inference model + SFT training initialization | [🤗 `hf-wzx1205/PDO_S2TT`](https://huggingface.co/hf-wzx1205/PDO_S2TT) |
| **Independent reproduction** | Fresh checkpoint + optimizer state + training/test logs + trajectories | [🤗 `hf-wzx1205/PDO_S2TT-reproduction`](https://huggingface.co/hf-wzx1205/PDO_S2TT-reproduction) |

> [!IMPORTANT]
> Reproduction artifacts are deliberately separate from the paper checkpoint, so
> the independently sampled model cannot be mistaken for the submitted result.

## Quick inference

Linux, Python 3.12, CUDA, and a GPU with at least 16 GB memory are recommended.

```bash
git clone https://github.com/ggiggit/PDO_S2TT.git
cd PDO_S2TT

python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -e .

mkdir -p checkpoints
hf download hf-wzx1205/PDO_S2TT pdo_s2tt.pt \
  --revision 62c27b03aa1d7da1b739e43e815796029ee243f5 \
  --local-dir checkpoints

bash scripts/run_fleurs.sh
```

The default command downloads official FLEURS TEST and reproduces all 647 En→Zh
examples. The other four directions and macro command are already included and
commented in [`scripts/run_fleurs.sh`](scripts/run_fleurs.sh).

## Quick training

```bash
python scripts/prepare_fleurs_train.py

hf download hf-wzx1205/PDO_S2TT pdo_s2tt_sft.pt \
  --revision 62c27b03aa1d7da1b739e43e815796029ee243f5 \
  --local-dir checkpoints

bash scripts/train_pdo.sh
```

The frozen recipe uses four 24 GB GPUs, 407 rollout rounds, and 1,625 AdamW
updates. It saves an inference checkpoint and resumable optimizer state every
round. See the [training guide](docs/training.md) before starting a full run.

## Core result

FLEURS TEST, five-direction macro, 647 examples per direction:

| BLEU ↑ | COMET ↑ | chrF++ ↑ | FTL ↓ | FRD ↓ | LAAL-CU mean / P90 ↓ | Coverage |
|--:|--:|--:|--:|--:|--:|--:|
| **31.58** | **85.37** | **44.67** | **2.00 s** | **2.42 s** | **3.05 / 5.66 s** | **100%** |

The public training path was also independently run from SFT initialization through
all five TEST directions: 407/407 rounds, 1,625/1,625 updates, and 3,235/3,235
non-empty outputs. See the [reproduction record](docs/reproduction.md) and
[downloadable artifacts](https://huggingface.co/hf-wzx1205/PDO_S2TT-reproduction).

## Repository layout

```text
src/                       model, streaming runtime, PDO, metrics
scripts/                   inference, evaluation, and training entry points
docs/                      the three focused guides above
artifacts/reproduction/    lightweight training/test receipts
references/                frozen FLEURS identities and TRAIN targets
tests/                     reward, policy, evaluator, inventory tests
```

## Citation

```bibtex
@inproceedings{pdo_s2tt_2027,
  title     = {Persistent Delivery Optimization for Streaming Speech-to-Text Translation with Revisions},
  booktitle = {IEEE International Conference on Acoustics, Speech and Signal Processing},
  year      = {2027}
}
```

Author metadata will be added after anonymous review. Code and PDO checkpoints are
released under [Apache-2.0](LICENSE); upstream models and datasets retain their own
licenses.
