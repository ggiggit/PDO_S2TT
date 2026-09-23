# Independent reproduction record

A clean public-code run was started from the released SFT initialization, trained
to the exact public stopping point, and evaluated on all five FLEURS TEST
directions. This is a stochastic verification run, not the paper checkpoint.

## Artifact repositories

| Purpose | Repository |
|:--|:--|
| Paper inference checkpoint and SFT initialization | [`hf-wzx1205/PDO_S2TT`](https://huggingface.co/hf-wzx1205/PDO_S2TT) |
| Fresh-run checkpoint, optimizer state, logs, predictions, and metrics | [`hf-wzx1205/PDO_S2TT-reproduction`](https://huggingface.co/hf-wzx1205/PDO_S2TT-reproduction/tree/b63f1c648fdebc85783b1324b3d007d36044e5b4) |

The two repositories are intentionally separate so the stochastic reproduction
checkpoint cannot be mistaken for the model used in the paper tables.

## Training receipt

| Item | Verified value |
|:--|:--|
| Initialization | Released `pdo_s2tt_sft.pt` |
| Seed | 52 |
| Hardware | 4 × RTX 4090 |
| Schedule | 407/407 rollout rounds, 1,625/1,625 AdamW updates |
| Wall time | 35,046.17 s (9 h 44 min) |
| Final model fingerprint | `5afc75eab69c19b33e4b551b0c5710f9088f723b86963efcb63938a0945ae2de` |
| Inference checkpoint SHA-256 | `bd6bc20f5a8dddd33991822bc17015b985932193abb0f7496d4e5a4df2081129` |
| Inference checkpoint size | 73,080,509 bytes |
| Training-state SHA-256 | `c3e94d3be46593cbf32bf66a5f5e55fd481484020cd1217ba3dfc4a466ae0e79` |
| Training-state size | 251,768,225 bytes |

The training state contains the learned policy, AdamW optimizer, rollout-round
counter, and Adam-update counter. The complete sanitized training and test logs
are mirrored in [`artifacts/reproduction`](../artifacts/reproduction); large
binaries and trajectories live only on Hugging Face.

## Five-direction TEST result

| Checkpoint | BLEU ↑ | COMET ↑ | chrF++ ↑ | FTL ↓ | FRD ↓ | LAAL-CU mean / P90 ↓ | Coverage |
|:--|--:|--:|--:|--:|--:|--:|--:|
| Paper PDO | 31.580 | 85.373 | 44.673 | 2.000 | 2.419 | 3.049 / 5.660 | 100% |
| Fresh reproduction | 31.838 | 85.344 | 44.871 | 2.000 | 2.422 | 2.981 / 5.479 | 100% |

| Checkpoint | RTF ↓ | Norm. erasure ↓ | Age erasure ↓ | First stable ↓ | Mean finalization ↓ |
|:--|--:|--:|--:|--:|--:|
| Paper PDO | .537 | .936 | 1.982 | 2.978 s | 7.364 s |
| Fresh reproduction | .530 | .947 | 2.000 | 2.966 s | 7.317 s |

Small differences are consistent with stochastic on-policy sampling. The fresh
run recovered the same quality/latency operating point, produced 3,235/3,235
non-empty translations, and completed without traceback, CUDA OOM, NaN, or
non-finite-value failure.

The saved trajectories were rescored on 2026-09-23 with evaluator commit
`7636709`. This corrected Japanese kana and accented-Latin tokenization in
LAAL-CU; quality, FTL, FRD, RTF, and revision metrics are unchanged.

## Integrity checks

- 647 unique predictions per direction; each ID set exactly matches its manifest.
- 100% non-empty final translations in all five directions.
- All metric and receipt JSON files parse successfully.
- Chinese/Japanese reward units are characters; corpus BLEU uses `zh` /
  `ja-mecab`; other directions use `13a`.
- Logs contain no credentials or internal machine paths.
- Local and hosted LFS hashes match for both checkpoint files.
