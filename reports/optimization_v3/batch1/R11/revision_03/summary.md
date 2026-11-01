# R11 — Cumulative ordinal median pilot

**Verdict: ADVANCE** — revision-03 W1+W4/H12 Train pilot; no Test/Kaggle access.

Recipe: nine binary LightGBM heads estimate strict survival thresholds y>10,...,90; row-wise PAVA enforces nonincreasing probabilities; the median is linearly interpolated at the 0.5 crossing and rank50-mapped within cycle before the frozen 0.75/0.25 blend.

| View | n | Control MAE | R11 MAE | Gain (control − R11) |
|---|---:|---:|---:|---:|
| W1 | 4,114 | 15.882112 | 15.820867 | +0.061245 |
| W4 | 4,950 | 11.455005 | 11.448039 | +0.006967 |
| primary | 9,064 | 13.464396 | 13.432793 | +0.031602 |
| H12 | 38,588 | 14.047533 | 14.038716 | +0.008817 |

Fit accounting: 2 temporal adapter attempts; 18 successful LightGBM head fits; 23.74 fit seconds.
SVD components receive no semantic interpretation. Validation windows are development data and are not an untouched holdout.

See `results.json`, `verification.json`, `subgroups.json`, `gate.json`, `attempts.jsonl`, and `experiments/optimization_v3/batch1/R11/revision_03/freeze.json` for provenance.
