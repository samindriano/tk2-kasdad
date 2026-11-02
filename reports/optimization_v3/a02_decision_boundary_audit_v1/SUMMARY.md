# A02 decision-boundary audit

**Status:** COMPLETED_ZERO_FIT
**F1 lane decision:** NO_SUPPORTED_F1_LANE
**QA:** PASS
**Interpretation:** adaptive diagnostic using reused validation rows; no holdout or causal claim.

## Main observed-arm label changes

All labels use the frozen rule `p >= 0.50`, so exact 0.50 predicts positive. `helpful` means a baseline error became correct; `harmful` means a correct baseline label became wrong. Error-count change is additive; Macro F1 is reported only when class support passes the fixed rule.

| Scope | n rows | learned-present | observed flips / helpful / harmful / net | shuffle flips / helpful / harmful / net | Δ positive rate (observed) |
|---|---:|---:|---:|---:|---:|
| full_six_primary / all_rows | 38588 | 32288 | 404 / 231 / 173 / 58 | 315 / 143 / 172 / -29 | 0.00420 |
| full_six_primary / learned_present | 32288 | 32288 | 404 / 231 / 173 / 58 | 315 / 143 / 172 / -29 | 0.00502 |
| full_six_primary / W1_fallback | 4114 | 0 | 0 / 0 / 0 / 0 | 0 / 0 / 0 / 0 | 0.00000 |
| new_origin_aggregate / all_rows | 21033 | 19715 | 262 / 147 / 115 / 32 | 216 / 96 / 120 / -24 | 0.00437 |
| late / all_rows | 15953 | 14920 | 181 / 92 / 89 / 3 | 139 / 55 / 84 / -29 | 0.00520 |
| inherited_H12_L / all_rows | 20903 | 19528 | 250 / 140 / 110 / 30 | 173 / 93 / 80 / 13 | 0.00354 |

W1 is unchanged Sparse25 fallback; it contains 4,114 rows (3,722 present-stream and 392 missing-stream). H12-L overlaps primary. All table rows compare the same query rows across the baseline, observed, and shuffle arms.

## Per-origin first-four windows (observed arm)

| Origin window | n | learned-present | label changes | helpful | harmful | net errors reduced | Δ positive rate |
|---|---:|---:|---:|---:|---:|---:|---:|
| E59 | 6493 | 6124 | 71 | 45 | 26 | 19 | 0.00046 |
| E63 | 7078 | 6636 | 97 | 61 | 36 | 25 | 0.00579 |
| E67 | 4950 | 4608 | 55 | 33 | 22 | 11 | 0.00707 |
| E71 | 7462 | 6955 | 94 | 41 | 53 | -12 | 0.00643 |
| E75 | 8491 | 7965 | 87 | 51 | 36 | 15 | 0.00412 |

## Fixed decision bands (full-six learned-present, observed)

| Baseline p band | n | flips 0→1 | flips 1→0 | helpful | harmful | net errors reduced | Macro F1 Δ | F1 support |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| [.45,.475) | 491 | 106 | 0 | 54 | 52 | 2 | 0.15790 | True |
| [.475,.50) | 440 | 177 | 0 | 103 | 74 | 29 | 0.24039 | True |
| [.50,.525) | 393 | 0 | 86 | 56 | 30 | 26 | 0.19245 | True |
| [.525,.55] | 400 | 0 | 35 | 18 | 17 | 1 | 0.07972 | True |

All four full-primary bands contain flips: below 0.50 they are 0→1, at/above 0.50 they are 1→0. The net error improvement is concentrated in the middle two bands, while the origin-level pattern is mixed. The late view has only 3 net errors reduced despite a positive Δ predicted-positive rate; E71 increases the positive rate while net errors worsen. This shows why positive-rate growth alone does not establish useful correction.

Per-origin shuffle counts and full detail are in `flip_summary.csv`; band-by-origin arm counts are in `baseline_band_summary.csv`; descriptor context is in `band_descriptor_summary.csv`.

## Boundary verification

Both observed and shuffled corrections stay within ±0.05 (max observed |Δp| 0.050000000000, shuffle 0.050000000000; tiny excess over .05 is floating representation). Label changes outside baseline p ∈ [0.45,0.55] are 0; exact 0.50 is positive. All hard-label changes are limited to learned-present rows; W1 fallback and missing-stream rows are unchanged.

The confidence taper in the prior review note is rejected as a Macro F1/Kaggle lane: Sol's independent check found zero hard-label changes on the four saved views (6,134 probability values changed in full-six primary). Such a transform may be considered only if probability quality becomes a separate objective; no taper run or submission was made here.

## Reliability descriptors and decision

Descriptor summaries condition on fixed baseline bands and compare unchanged rows with useful and harmful flips. A band is supported for this comparison only at n ≥ 100 and at least 20 label changes; Macro F1 has a separate n ≥ 200 and ≥30 per-class support check. Unsupported rows remain visible in the CSVs.

No unique descriptor/band passed the predeclared replication rule requiring at least 20 useful and 20 harmful flips in a supported band for at least three origin windows, a same-direction contrast in both new-origin and late aggregates, and supported aggregate Macro F1 comparisons. Result: **NO_SUPPORTED_F1_LANE**. Do not create another Macro F1 card from these reused rows.

## Integrity and limits

Official Train SHA-256: `875552a68a3d24b0f3e1966d6ddb5671e6f54305a1c526e2f0a76747942ed91c`. Fit ledger SHA-256 before/after: `6fcc4b4d427e3ff03139e0e9ee429f529f7c7bf6ad91ce0eb9a52d55e2503147` / `6fcc4b4d427e3ff03139e0e9ee429f529f7c7bf6ad91ce0eb9a52d55e2503147`. Fit accounting remains 66 starts, 64 saved, 2 historical failures; estimator/preprocessor fit calls: 0.

This diagnostic did not access Test/SampleSubmission or Kaggle, did not write row-level predictions, and did not change the frozen A02 recipe, gates, metrics, public score receipt, or champion. All evaluated rows have already been inspected; any pattern is descriptive, not fresh validation.
