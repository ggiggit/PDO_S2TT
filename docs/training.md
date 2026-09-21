# PDO training reproduction

The public training release starts from the released SFT policy and reproduces
the paper's PDO reinforcement-learning stage. It does not redistribute the
preceding supervised-training pipeline.

The exact mapping from each paper equation and metric to source code is in the
dedicated [paper-to-code guide](paper-to-code.md).

## Installation

Requirements: Linux, Python 3.12, CUDA, four GPUs with at least 24 GB each,
`git`, `ffmpeg`, and a recent Hugging Face CLI.

```bash
git clone https://github.com/ggiggit/PDO_S2TT.git
cd PDO_S2TT

python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
```

## Full run

```bash
# Prepare 2,600 English recordings × five target directions.
python scripts/prepare_fleurs_train.py

# Download the SFT initialization.
mkdir -p checkpoints
hf download hf-wzx1205/PDO_S2TT pdo_s2tt_sft.pt \
  --revision 62c27b03aa1d7da1b739e43e815796029ee243f5 \
  --local-dir checkpoints

# Run PDO on four GPUs.
bash scripts/train_pdo.sh
```

The trainer writes `results/training/training_state.pt` and
`results/training/pdo_s2tt.pt` after every rollout round. The launcher resumes
automatically when the training-state file exists. A full four-RTX-4090 run
takes approximately ten hours.

Use the real streaming/backward path on one utterance per GPU before committing
to the full run:

```bash
torchrun --standalone --nproc-per-node=4 scripts/train_pdo.py \
  --manifest data/fleurs/train/train.jsonl \
  --sft-checkpoint checkpoints/pdo_s2tt_sft.pt \
  --output results/smoke \
  --smoke
```

## Frozen recipe

| Setting | Value |
|:--|:--|
| Directions | En→Zh/De/Es/Ja/Fr |
| Training inventory | 2,600 recordings × 5 directions |
| Acoustic update / input packet | 2.0 s / 0.1 s |
| Samples per utterance | 4 |
| Behavior-policy temperature | 0.2 |
| Rollout batch / optimizer minibatch | 32 / 8 utterances |
| Trainable parameters | Text-decoder LoRA + private history adapter |
| Optimizer | AdamW, LR `1e-6`, weight decay `.01` |
| PPO clip / truncated correction | `.2` / `c_max=2` |
| Gradient clipping | 1.0 |
| Default seed | 52 |
| Schedule | 407 rollout rounds / 1,625 updates / one TRAIN pass |

Before model loading, the run verifies the released order, IDs, directions,
target text, and 2,600-recording × five-direction grouping. An altered or merely
similar manifest is rejected.

## Exact reward and optimization contract

### Text units

- Chinese and Japanese use CJK characters while retaining contiguous non-CJK
  alphanumeric strings.
- German, Spanish, and French use case-folded Unicode word units. Internal
  apostrophes and hyphens are preserved; surrounding punctuation is discarded.
- Process reward uses LCS recall on these units. Terminal reward uses normalized
  effective-order sentence BLEU with `tokenize="none"`.
- Reporting tokenization is deliberately separate from reward tokenization.

### Return and baseline

For every displayed draft, the process term scores only the prefix that remains
unchanged in all later drafts. Eq. (2) integrates persistent-prefix LCS recall
over source-time intervals and adds final sentence BLEU. Eq. (3) uses
`G_t = Φ_T − Φ_(t−1)`.

Returns are standardized at each event across the four trajectories using the
population variance. Variance ≤ `1e-16` produces four zero weights. No learned
value model or direct latency reward is used.

### Behavior and proximal policies

Each utterance owns four persistent RNG lanes. Sampling applies the released
wait/forbidden-token and no-repeat masks, then samples from
`softmax(logits / .2)` with `torch.multinomial`; each sampled draft becomes the
private history for the next update. The round-start policy supplies both the
behavior distribution and frozen proximal snapshot. Token masks and behavior
log-probabilities are stored with each action.

Eq. (4) applies PPO clipping `ε=.2` and truncated importance correction
`c_max=2`. Tokens in one draft share the event weight; probability ratios remain
token-specific. Loss is normalized by trajectory rather than generated token.

The Qwen3-ASR backbone is frozen. Only decoder LoRA and the private-history
module are optimized. Dropout is disabled during on-policy rollout and update.

## DEV checkpoint selection used in the paper

Selection did not use PDO reward or a weighted quality-latency score. Candidates
first had to satisfy all of the following:

- no empty outputs;
- COMET within 0.30 of the best candidate from that run;
- BLEU within 1.0 of the best candidate from that run;
- COMET no more than 0.50 below the selected History-SFT checkpoint.

Remaining checkpoints were ordered lexicographically by lower LAAL-CU mean,
lower LAAL-CU P90, lower FRD, and finally higher COMET. Promotion also required
at least 0.20 s lower mean LAAL-CU or 0.50 s lower P90 than History-SFT.

## Controlled RL baselines in Table 3

All methods share History-SFT initialization, state/action interface, K=4
rollouts, temperature, trainable parameters, optimizer, data order, and update
framework.

- **Current-draft RL** replaces only the persistent prefix in the process term
  with the currently visible draft and retains the full return.
- **Hibiki-Zero-style RL** uses intermediate-plus-final sentence BLEU with
  `α=.5`.
- **HPO-style RL** uses quality-gated final BLEU/LAAL-CU with quality threshold
  `.33` and latency weight `.5`.

The supported public launcher trains PDO. These matched-control definitions are
included to make the reported comparison unambiguous.
