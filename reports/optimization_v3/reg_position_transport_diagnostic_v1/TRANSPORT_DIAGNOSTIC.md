# REG_POSITION_TRANSPORT_DIAGNOSTIC_V1

Zero-fit descriptive analysis of frozen REG_POSITION_L1_V1 outputs. This report preserves the pilot's RETIRE verdict and all failed gates. It does not score a new policy or identify a causal transport effect.

Reproduce with: python scripts/reg_position_transport_diagnostic_v1.py --source-root "C:\Users\Sam\OneDrive\Documents\KASDAD\TP 2" --worktree-root "C:\Users\Sam\OneDrive\Documents\KASDAD\tk2-regression-frontier" --output-root "C:\Users\Sam\OneDrive\Documents\KASDAD\tk2-regression-frontier\reports\optimization_v3\reg_position_transport_diagnostic_v1"
Freeze: 1982463561610456c516d3ac3c913f0be38aca6bd2955e2813960f3e50e03ca5. Frozen source/input files verified: 45. Pilot artifacts hashed before/after: 25. Six-fit ledger SHA-256: 9b68164678ccad74286dedc77193e9d9b76c75da1ac8884f0d9465cbcf03cb4b.

## Paired W6 cohort

E67 and E75 use the exact same 8,491 official record IDs, cycles 76–79, and targets, paired by record_id. The origin-specific R06 baseline differs on those same rows.

| Quantity | Value |
|---|---:|
| E67 R06 MAE | 11.653450 |
| E75 R06 MAE | 9.921613 |
| Mean signed R06 change E75−E67 | +0.949615 |
| Mean absolute R06 change | 4.434065 |

| Arm | Head sign agreement | Mean abs head E67→E75 | Mean abs delta E67→E75 | Mean abs final-prediction change | Candidate MAE E67→E75 |
|---|---:|---:|---:|---:|---:|
| BAG | 0.810 | 4.648→3.000 | 1.127→0.732 | 4.634 | 11.627625→9.785707 |
| REAL | 0.825 | 4.526→3.066 | 1.101→0.751 | 4.670 | 11.632168→9.770188 |
| SHUFFLED | 0.812 | 4.601→2.924 | 1.116→0.712 | 4.638 | 11.629671→9.787883 |

Per-cycle R06/candidate MAE, gain, residual, raw-head, and delta distributions are in paired_w6_by_cycle.csv. Paired row-level change magnitudes and sign agreement are in paired_w6_origin_changes.csv.

## Residual distributions

Residual = target − origin-specific final mapped R06. Deciles are fixed as floor(clip(R06,0,100)/10), capped at 9. Donors are detailed by source cutoff and actual cycle; query cells cover E67 W4/W5/W6 and E75 W6. Every cell contains n, median, p10 and p90; cells under n=100 are UNKNOWN with quantiles withheld.

| Query view | Supported matched deciles | Median of within-decile median shifts | Range |
|---|---:|---:|---:|
| E67_W4 | 10/10 | +0.120 | -1.282 to +2.665 |
| E67_W5 | 10/10 | +0.269 | -1.685 to +5.343 |
| E67_W6 | 10/10 | -0.082 | -0.669 to +4.880 |
| E75_W6 | 10/10 | -0.299 | -2.001 to +3.029 |

There are 95 detailed donor/query cutoff-cycle-decile cells below n=100. They remain UNKNOWN. Marginal changes are descriptive; donor history, fit population, target horizon, and R06 baseline move jointly.

## Saved losses and existing correction effect

| Origin | Arm | Saved fit residual MAE* | Query MAE | Mean delta | Help/harm/tie, present rows | 0.25-head cap rate | MAE change vs R06 |
|---|---|---:|---:|---:|---:|---:|---:|
| E67 | BAG | 9.622720 | 11.566679 | -0.1172 | 9486/10039/3 | 0.425% | +0.036970 |
| E67 | REAL | 9.474247 | 11.564997 | -0.0952 | 9592/9931/5 | 0.486% | +0.035288 |
| E67 | SHUFFLED | 9.626777 | 11.573594 | -0.0961 | 9520/10005/3 | 0.446% | +0.043885 |
| E75 | BAG | 9.951727 | 9.785707 | -0.0646 | 4520/3445/0 | 0.025% | -0.135906 |
| E75 | REAL | 9.863899 | 9.770188 | -0.0712 | 4557/3408/0 | 0.213% | -0.151424 |
| E75 | SHUFFLED | 9.993680 | 9.787883 | -0.0513 | 4502/3463/0 | 0.000% | -0.133730 |

*Saved in-sample residual fit loss is descriptive only. Raw-head quantiles/sign agreement, final-bound clipping, missing counts, and train-query loss gaps are in delta_error_effect.csv and train_query_loss.csv. Help/harm compare each already-saved prediction to the same origin's R06 errors; no new prediction policy was evaluated.

## LightGBM block importances

Feature indices are verified against frozen unigram/position widths and model n_features_in_: BAG unigrams first, optional REAL/SHUFFLED position-third suffix next, normalized R06 baseline score last.

| Origin | Arm | Block | Features | Split share | Gain share |
|---|---|---|---:|---:|---:|
| E67 | BAG | BAG_UNIGRAM | 5675 | 88.84% | 86.04% |
| E67 | BAG | POSITION_THIRD | 0 | 0.00% | 0.00% |
| E67 | BAG | BASELINE_SCORE | 1 | 11.16% | 13.96% |
| E67 | REAL | BAG_UNIGRAM | 5675 | 49.91% | 54.44% |
| E67 | REAL | POSITION_THIRD | 17025 | 41.34% | 34.12% |
| E67 | REAL | BASELINE_SCORE | 1 | 8.75% | 11.45% |
| E67 | SHUFFLED | BAG_UNIGRAM | 5675 | 66.93% | 74.70% |
| E67 | SHUFFLED | POSITION_THIRD | 17025 | 23.45% | 12.49% |
| E67 | SHUFFLED | BASELINE_SCORE | 1 | 9.62% | 12.81% |
| E75 | BAG | BAG_UNIGRAM | 5723 | 89.96% | 87.13% |
| E75 | BAG | POSITION_THIRD | 0 | 0.00% | 0.00% |
| E75 | BAG | BASELINE_SCORE | 1 | 10.04% | 12.87% |
| E75 | REAL | BAG_UNIGRAM | 5723 | 47.59% | 48.52% |
| E75 | REAL | POSITION_THIRD | 17169 | 43.32% | 39.32% |
| E75 | REAL | BASELINE_SCORE | 1 | 9.09% | 12.16% |
| E75 | SHUFFLED | BAG_UNIGRAM | 5723 | 69.45% | 75.18% |
| E75 | SHUFFLED | POSITION_THIRD | 17169 | 21.64% | 12.67% |
| E75 | SHUFFLED | BASELINE_SCORE | 1 | 8.91% | 12.15% |

Block importance is descriptive, not causal: correlated features and different block widths/model capacities prevent attributing gains to one block or claiming generalization.

## Interpretation and one recommendation

All three heads improve E75 W6, while the pilot failed E67_H12 and E67 W4 aggregate gates; actual position adds only a small E67_H12 contrast versus BAG/SHUFFLED. The paired diagnostic measures changes in baseline, head, correction and residual distributions, but cannot isolate which jointly changed history component explains the transfer pattern.

Recommendation: ONE_SPECIFIC_NEXT_PILOT_FOR_SOL_REVIEW — use only the separately authorized, fixed REAL E79 exploratory benchmark. Keep RETIRE_REG_POSITION_L1_V1 and every pilot gate/failure unchanged. The exploratory benchmark is not a qualified model and does not authorize upload; use BAG/SHUFFLED pilot results as fixed counterevidence. Diagnostic observations do not change the benchmark recipe or selection.

## Reproducibility assets

INPUT_HASHES.json records freeze/source/input plus pilot model, receipt, prediction, report, and ledger hashes. Computed tables: paired_w6_by_cycle.csv, paired_w6_origin_changes.csv, residual_distribution_cells.csv, paired_decile_comparisons.csv, block_importance.csv, delta_error_effect.csv, train_query_loss.csv.

No vectorizer/model fit, model.predict call, new candidate policy, Test/Sample/Kaggle access, dependency install, main-tree write, or pilot-ledger mutation occurred.
