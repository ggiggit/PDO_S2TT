# Paper-to-code guide

This page maps the compact paper formulation to the exact released
implementation. Training commands and hyperparameters are documented separately
in [`training.md`](training.md).

## Equation map

| Paper component | Code | What the implementation does |
|:--|:--|:--|
| Eq. (1): persistent prefix | [`persistent_prefixes`](../src/pdo_s2tt/training/reward.py#L63) | Finds the longest prefix of each displayed draft that survives in every later draft |
| Eq. (2): PDO utility | [`trajectory_ledger`](../src/pdo_s2tt/training/reward.py#L88) | Integrates persistent-prefix LCS recall over source time and adds terminal sentence BLEU |
| Eq. (3): trajectory return | [`full_trajectory_returns`](../src/pdo_s2tt/training/reward.py#L130) | Computes `G_t = Φ_T − Φ_(t−1)` for every display event |
| Eq. (4): policy loss | [`pdo_loss`](../src/pdo_s2tt/training/policy.py#L229) | Applies clipped PPO ratios and truncated behavior correction to event-weighted token log-probabilities |

## Training mechanics

| Detail | Code | Released behavior |
|:--|:--|:--|
| Reward units and normalization | [`units`](../src/pdo_s2tt/training/reward.py#L26) | CJK characters for Zh/Ja; normalized words for De/Es/Fr |
| Group-relative baseline | [`_standardize`](../src/pdo_s2tt/training/reward.py#L140) | Per-event population standardization over four trajectories |
| Zero-variance handling | [`_standardize`](../src/pdo_s2tt/training/reward.py#L140) | Variance ≤ `1e-16` produces four zero weights |
| Behavior-policy sampling | [`sample_group`](../src/pdo_s2tt/training/policy.py#L137) | Four persistent RNG lanes at temperature `.2` |
| Saved behavior log-probabilities | [`action_logps`](../src/pdo_s2tt/training/policy.py#L213) | Stores masked action probabilities used by the off-policy correction |
| Closed-loop rollout | [`rollout`](../src/pdo_s2tt/training/trainer.py#L37) | Each sampled complete draft becomes private history for the next acoustic update |
| Synchronized update | [`update`](../src/pdo_s2tt/training/trainer.py#L248) | Updates decoder LoRA and private-history adapter across distributed ranks |
| Hyperparameters and seeds | [`train_pdo.py`](../scripts/train_pdo.py#L49) | Freezes optimizer, clipping, batch, seed, and schedule defaults |
| Training inventory | [`validate_released_manifest`](../src/pdo_s2tt/training/data.py#L21) | Validates all 13,000 multilingual TRAIN units before loading the model |

## Evaluation mechanics

| Paper quantity | Code | Released behavior |
|:--|:--|:--|
| FTL / FRD / LAAL-CU | [`simultaneous_metrics.py`](../src/pdo_s2tt/simultaneous_metrics.py) | Separates source-consumption time from compute-aware delivery and uses stable final-prefix timestamps |
| Erasure and finalization | [`revision_record`](../src/pdo_s2tt/evaluation.py#L140) | Computes normalized erasure, age-weighted erasure, first-stable time, and mean finalization |
| Corpus quality | [`evaluate_fleurs.py`](../scripts/evaluate_fleurs.py) | BLEU, COMET, and chrF++ with language-specific SacreBLEU tokenizers |
| Five-direction macro | [`evaluate_macro.py`](../scripts/evaluate_macro.py) | Direction-level quality/revision means and pooled trajectory latency percentiles |

## Important implementation choices

- The process term uses LCS recall on normalized reward units. The terminal term
  uses effective-order sentence BLEU with `tokenize="none"`.
- The reward baseline is entirely group-relative; there is no learned value model.
- Four trajectories are sampled with persistent, deterministic RNG lanes. A draft
  is conditioned on the previous draft from the same lane.
- Eq. (4) uses PPO clip `ε=.2` and truncated importance correction `c_max=2`.
  Tokens retain token-specific probability ratios but share their display event's
  return weight.
- The Qwen3-ASR backbone stays frozen. PDO updates text-decoder LoRA and the
  private-history module only.
- No direct latency reward is added. Earlier stable delivery emerges from scoring
  only content that persists through the future revision sequence.

For the complete frozen recipe, seed construction, DEV selection criterion, and
matched RL controls, continue to the [training guide](training.md).
