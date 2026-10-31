# V3 sparse-LGB Classification continuation: six temporal windows

## Frozen comparison

The continuation applies the previously fixed 25% sparse-LGB / 75% C2_B probability blend only when `signal_stream` is present. The existing CatBoost missing-stream expert remains unchanged. Four new fits (cutoffs 59, 63, 71, 75) complete the six four-cycle windows; predictions from the already verified 55/67 pilot are reused.

The model, threshold (0.50), feature construction, training horizon, blend weight and evaluation windows were frozen before the four fits. The full Train file and cached controls/pilot inputs are hash-bound in `experiments/optimization_v3/classification_sparse_full/freeze.json`. No Test or Kaggle access occurred.

## Primary window results

| Window | Rows | Champion Macro F1 | Fixed blend Macro F1 | Delta | Equal-cycle delta | AP delta | Brier delta |
|---|---:|---:|---:|---:|---:|---:|---:|
| W1 | 4114 | 0.802090 | 0.803160 | +0.001070 | +0.000915 | +0.002263 | -0.001569 |
| W2 | 6493 | 0.849458 | 0.851618 | +0.002160 | +0.002174 | +0.002639 | -0.002088 |
| W3 | 7078 | 0.828527 | 0.832515 | +0.003989 | +0.003924 | +0.006505 | -0.002433 |
| W4 | 4950 | 0.840297 | 0.841418 | +0.001120 | +0.000565 | +0.008415 | -0.001615 |
| W5 | 7462 | 0.812698 | 0.814497 | +0.001799 | +0.001610 | +0.006577 | -0.001636 |
| W6 | 8491 | 0.813426 | 0.816599 | +0.003174 | +0.003176 | +0.009310 | -0.002333 |

## Pooled validation and H12

| View | Rows | Champion Macro F1 | Fixed blend Macro F1 | Delta | Equal-cycle delta | AP delta | Brier delta |
|---|---:|---:|---:|---:|---:|---:|---:|
| Six-window primary | 38588 | 0.827646 | 0.829944 | +0.002297 | +0.002061 | +0.006245 | -0.002002 |
| H12 fixed-origin pooled | 38588 | 0.798984 | 0.802966 | +0.003982 | +0.003250 | +0.009220 | -0.002630 |
| H12 early (56-67) | 17685 | 0.792671 | 0.795032 | +0.002361 | +0.002178 | +0.003870 | -0.002305 |
| H12 late (68-79) | 20903 | 0.803724 | 0.808961 | +0.005237 | +0.004322 | +0.009630 | -0.002905 |

## Routing and subgroup checks

The independent verifier checks that present-stream rows receive the fixed blend and missing-stream rows preserve the champion probability exactly. Results include class-wise precision/recall/F1, confusion matrices, stream-present/missing splits, and text_load quartiles. Quartile edges are computed from cycles 47-55 feature values only.

- Missing branch invariant: **True**.
- Parent primary-gain gate (>= 0.0005): **True** (+0.002297).
- Parent H12-harm gate (>= -0.001): **True** (+0.003982).
- Parent window-support gate (>= 4/6 nonnegative): **True** (6/6).

The full subgroup and per-class details are in `fullsix.json`; the independently recomputed receipt is `verification.json`.

## Limits

These Train-cycle windows have already informed model development, and H12 overlaps the pooled primary cycle range while using fixed-origin predictions. The results are comparative evidence for parent review, not a fresh holdout estimate and not authorization to change the official model or submit to Kaggle.
