# Ordered supervised token-statistics pilot

## Question

Do compact summaries of token-level target associations learned strictly from earlier cycles add predictive value beyond the existing representations? Each report is encoded before any labels from its own cycle enter the token dictionary. These statistics do not reveal anonymous token meaning and do not establish causality.

Exactly four official-Train fits were run: Classification and Regression at cutoffs 55 and 67. The model, token support (3), smoothing (30), LightGBM parameters (400 trees, 15 leaves, minimum child 40, lambda 3), rank rule, blend weights, and thresholds were fixed before these fits.

## Classification

The initial C2_B-present/CB-missing control and the latest fixed sparse25-present/CB-missing control are both reported. The direct candidate replaces only the present branch. The fixed candidate blends 25% new supervised probability with 75% of the latest sparse25 present expert; the CB missing branch is preserved exactly.

| View | n | Latest Macro F1 | Direct Macro F1 | Direct Δ | Fixed blend Macro F1 | Blend Δ vs latest | AP Δ vs latest | Brier Δ vs latest |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| W1 | 4,114 | 0.803160 | 0.749902 | -0.053258 | 0.801063 | -0.002097 | +0.003403 | +0.000325 |
| W4 | 4,950 | 0.841418 | 0.774984 | -0.066434 | 0.841907 | +0.000489 | -0.008927 | +0.001506 |
| H12_E | 17,685 | 0.795032 | 0.721928 | -0.073104 | 0.790689 | -0.004342 | -0.000990 | +0.002085 |
| H12_L | 20,903 | 0.808961 | 0.738486 | -0.070475 | 0.809289 | +0.000328 | -0.008004 | +0.001918 |
| primary_pilot | 9,064 | 0.823972 | 0.763567 | -0.060406 | 0.823293 | -0.000679 | -0.002967 | +0.000970 |
| H12 | 38,588 | 0.802966 | 0.730933 | -0.072033 | 0.801033 | -0.001932 | -0.004642 | +0.001995 |

Classification advance gate versus latest sparse25: **FAIL**; primary Δ=-0.000679, H12 Δ=-0.001932, minimum individual-view Δ=-0.004342.
Per-class precision/recall/F1, confusion matrices, present/missing groups, and both initial/latest control comparisons are in `pilot.json`.

## Regression

Regression compares raw supervised predictions, fixed within-cycle rank50, and the exact 25% rank50 + 75% three-component champion blend. It reports MAE, RMSE, signed bias, equal-cycle MAE, per-cycle metrics, tails, EF0/EF1, and signal-stream availability.

| View | n | Champion MAE | Raw MAE | Rank50 MAE | Fixed blend MAE | Blend MAE gain | Bias shift |
|---|---:|---:|---:|---:|---:|---:|---:|
| W1 | 4,114 | 15.8821 | 16.7258 | 16.8424 | 15.9969 | -0.1148 | -0.048 |
| W4 | 4,950 | 11.4550 | 12.4842 | 12.4580 | 11.5731 | -0.1181 | +0.022 |
| H12_E | 17,685 | 16.9266 | 17.6359 | 17.7166 | 16.9867 | -0.0601 | +0.032 |
| H12_L | 20,903 | 11.6117 | 13.0237 | 12.9014 | 11.7981 | -0.1864 | +0.052 |
| primary_pilot | 9,064 | 13.4644 | 14.4094 | 14.4480 | 13.5810 | -0.1166 | -0.010 |
| H12 | 38,588 | 14.0475 | 15.1375 | 15.1082 | 14.1761 | -0.1285 | +0.043 |

Regression fixed-blend advance gate: **FAIL**; primary gain=-0.1166, H12 gain=-0.1285, maximum W1/W4/H12 view harm=+0.1864.

## Leakage and scope checks

- All four full audit states record per-cycle `labels_previously_seen` counts; each equals the number of fit rows with a strictly earlier cycle.
- The synthetic same-cycle perturbation check confirms current-cycle labels do not alter features for their own cycle, but do alter later-cycle features when shared tokens have support.
- Fit fingerprints bind only Train rows through each cutoff; transformations are fit past-only.
- No Test/Kaggle read or call; these known Train-cycle views are development evidence, not an untouched holdout.
- Anonymous target-stat token values are evidence of past target association, not semantic interpretation or causal effects.

Full per-class, per-cycle, and subgroup results and fit lineages are in `pilot.json`; independent checks are in `verification.json`.
