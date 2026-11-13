# Persistent Delivery Optimization for Streaming Speech-to-Text Translation with Revisions

Official code and checkpoints for
[*Persistent Delivery Optimization for Streaming Speech-to-Text Translation with Revisions*](https://arxiv.org/abs/2609.26427),
submitted to **ICASSP 2027**.

[![arXiv](https://img.shields.io/badge/arXiv-2609.26427-b31b1b.svg)](https://arxiv.org/abs/2609.26427)
[![Hugging Face](https://img.shields.io/badge/🤗_Hugging_Face-model_weights-FFD21E)](https://huggingface.co/hf-wzx1205/PDO_S2TT)
[![ModelScope](https://img.shields.io/badge/ModelScope-model_weights-624AFF)](https://modelscope.cn/models/wanzixiang/PDO_S2TT)
[![Tests](https://github.com/ggiggit/PDO_S2TT/actions/workflows/tests.yml/badge.svg)](https://github.com/ggiggit/PDO_S2TT/actions/workflows/tests.yml)
[![License](https://img.shields.io/badge/License-Apache--2.0-4C8BF5)](LICENSE)

**Direct streaming S2TT · Revision-aware training · Five target languages**

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

## Model weights

The released weights are mirrored on
[Hugging Face](https://huggingface.co/hf-wzx1205/PDO_S2TT) and
[ModelScope](https://modelscope.cn/models/wanzixiang/PDO_S2TT).

| File | Use | SHA-256 | Download |
|:--|:--|:--|:--|
| `pdo_s2tt.pt` | Latest five-language finite-window PDO checkpoint | `194fba85b1a87547b1d180d0d98416cdd7ee694dceb6f8a53577fec9a37a08da` | [Download](https://huggingface.co/hf-wzx1205/PDO_S2TT/blob/main/pdo_s2tt.pt) |
| `pdo_s2tt_sft.pt` | Latest five-language finite-window SFT initialization | `10e6cb0c9b7f739bfc60ac3456dadc55c487dce57ee131d6e2d437d714ca1f78` | [Download](https://huggingface.co/hf-wzx1205/PDO_S2TT/blob/main/pdo_s2tt_sft.pt) |

Both checkpoints use one shared five-language model with a 24-second acoustic
window, 12 two-second events, six-event retirement, and a 128-token visible
history cap.

## Quick inference

Linux, Python 3.12, CUDA, and a GPU with at least 16 GB memory are recommended.

```bash
git clone https://github.com/ggiggit/PDO_S2TT.git
cd PDO_S2TT

python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -e .

mkdir -p checkpoints
hf download hf-wzx1205/PDO_S2TT pdo_s2tt.pt --local-dir checkpoints

bash scripts/run_fleurs.sh
```

The default command downloads official FLEURS TEST and reproduces all 647 En→Zh
examples. The other four directions and macro command are already included and
commented in [`scripts/run_fleurs.sh`](scripts/run_fleurs.sh).

## Quick training

```bash
python scripts/prepare_fleurs_train.py

hf download hf-wzx1205/PDO_S2TT pdo_s2tt_sft.pt --local-dir checkpoints

bash scripts/train_pdo.sh
```

The frozen recipe uses four 24 GB GPUs, 407 rollout rounds, and 1,625 AdamW
updates. It saves an inference checkpoint and resumable optimizer state every
round. See the [training guide](docs/training.md) before starting a full run.

<details>
<summary><strong>Independent public-code verification</strong></summary>

The released training path was independently run from the SFT initialization
through all 407 rounds and five-direction TEST evaluation. Detailed logs,
optimizer state, predictions, and metrics are kept in the
[reproduction record](docs/reproduction.md), with large artifacts hosted in a
[separate archive](https://huggingface.co/hf-wzx1205/PDO_S2TT-reproduction).

</details>

## Results

The latest finite-window PDO checkpoint was evaluated on the same 647-example
FLEURS TEST population in each of five directions (3,235 sessions total):

| BLEU | COMET ×100 | chrF++ | FRD | stable LAAL-CU | normalized erasure | RTF |
|--:|--:|--:|--:|--:|--:|--:|
| 31.2355 | 85.1594 | 44.4709 | 2.2782 s | 3.0057 s | 0.9211 | 0.3213 |

FRD and RTF depend on hardware and runtime. Quality, stable LAAL-CU, and
revision metrics use the same definitions as the paper. The checkpoint kept
its acoustic state bounded in all five directions.

For reference, Table 2 from the original paper checkpoint is retained below:

![Table 2: in-domain FLEURS TEST results](assets/table2.png)

## Repository layout

```text
src/                       model, streaming runtime, PDO, metrics
scripts/                   inference, evaluation, and training entry points
docs/                      the three focused guides above
artifacts/reproduction/    lightweight training/test receipts
references/                frozen FLEURS identities and TRAIN targets
tests/                     reward, policy, evaluator, inventory tests
```

## License

Code and PDO checkpoints are released under [Apache-2.0](LICENSE). Upstream models
and datasets retain their own licenses.
