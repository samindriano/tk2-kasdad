# Regression Round 1 — Comparison Tables

Protocol: `samuel-regression-round1-v1`
Run: `round1_v1`; 64 completed fits; status `ROUND1_COMPLETE_REVIEW_NO_FINAL_MODEL`.
The reported primary metric is raw, unclipped MAE. These are validation comparisons for review; they do not select a final model.

## Primary pooled metrics

Pooled statistics use all primary validation rows (cycles 56–79, once each). Equal-cycle MAE weights each of those 24 cycles equally; mean-window MAE weights each of the six windows equally.

| Recipe | n | Pooled MAE | Pooled RMSE | Signed bias | Equal-cycle MAE | Mean-window MAE |
| --- | --- | --- | --- | --- | --- | --- |
| R0_A | 38,588 | 25.0000 | 28.8671 | -0.0069 | 25.0000 | 25.0000 |
| R0_B | 38,588 | 15.5616 | 19.5610 | 0.0468 | 15.7032 | 15.6278 |
| R0_C | 38,588 | 15.3114 | 19.2570 | -0.1382 | 15.4389 | 15.3781 |
| R1_A | 38,588 | 15.6226 | 19.1821 | 0.6254 | 15.7343 | 15.6683 |
| R1_B | 38,588 | 15.5710 | 19.0908 | 0.5482 | 15.6962 | 15.6345 |
| R1_B_ADD | 38,588 | 15.6015 | 19.1328 | 0.5266 | 15.7260 | 15.6608 |
| R1_C | 38,588 | 15.1490 | 19.0575 | -0.2857 | 15.2541 | 15.1985 |
| R1_D | 38,588 | 13.3498 | 17.3576 | -0.0371 | 13.5897 | 13.6220 |

Signed bias is mean(prediction − actual); positive means overprediction. Negative MAE differences in the comparison table indicate lower MAE for the candidate.

## Primary MAE by validation window

All six windows are reported separately. They cover non-overlapping validation cycles 56–79.

| Recipe | W1 | W2 | W3 | W4 | W5 | W6 |
| --- | --- | --- | --- | --- | --- | --- |
| R0_A | 25.0000 | 24.9999 | 25.0000 | 25.0000 | 24.9999 | 25.0000 |
| R0_B | 16.9696 | 15.7896 | 16.7414 | 14.4496 | 14.6167 | 15.2001 |
| R0_C | 16.7349 | 15.3696 | 16.5437 | 14.2514 | 14.3991 | 14.9697 |
| R1_A | 16.6113 | 15.7118 | 16.7795 | 14.7717 | 14.8428 | 15.2926 |
| R1_B | 16.7900 | 15.7824 | 16.7189 | 14.6648 | 14.6621 | 15.1889 |
| R1_B_ADD | 16.7823 | 15.8104 | 16.7254 | 14.6920 | 14.6937 | 15.2607 |
| R1_C | 16.3860 | 15.1531 | 16.3275 | 14.1140 | 14.2760 | 14.9347 |
| R1_D | 16.8834 | 12.9446 | 13.9938 | 13.0620 | 12.6885 | 12.1599 |

## Secondary H12 windows

These longer-horizon checks are secondary. H12_E shares cycles 56–67 with primary W1–W3; H12_L shares cycles 68–79 with primary W4–W6. They are overlapping validation views, not independent evidence.

| Window | Recipe | n | MAE | RMSE | Signed bias | Equal-cycle MAE |
| --- | --- | --- | --- | --- | --- | --- |
| H12_E | R0_A | 17,685 | 25.0000 | 28.8670 | -0.0017 | 25.0000 |
| H12_E | R0_B | 17,685 | 17.0462 | 21.3436 | 1.1842 | 17.0872 |
| H12_E | R0_C | 17,685 | 16.7388 | 20.9966 | 0.9229 | 16.7682 |
| H12_E | R1_A | 17,685 | 16.6445 | 20.4841 | 2.0260 | 16.6640 |
| H12_E | R1_B | 17,685 | 16.9045 | 20.7363 | 2.0227 | 16.9113 |
| H12_E | R1_B_ADD | 17,685 | 16.9021 | 20.7419 | 2.0329 | 16.9115 |
| H12_E | R1_C | 17,685 | 16.4321 | 20.6039 | 0.7251 | 16.4472 |
| H12_E | R1_D | 17,685 | 17.7956 | 22.5640 | 1.8422 | 17.6168 |
| H12_L | R0_A | 20,903 | 24.9999 | 28.8672 | -0.0031 | 25.0000 |
| H12_L | R0_B | 20,903 | 15.0050 | 18.5258 | -0.8636 | 15.0047 |
| H12_L | R0_C | 20,903 | 14.7735 | 18.2025 | -0.9852 | 14.7675 |
| H12_L | R1_A | 20,903 | 15.1500 | 18.2999 | -0.1674 | 15.1847 |
| H12_L | R1_B | 20,903 | 15.0481 | 18.1174 | -0.3598 | 15.0702 |
| H12_L | R1_B_ADD | 20,903 | 15.0857 | 18.1738 | -0.4161 | 15.1095 |
| H12_L | R1_C | 20,903 | 14.6706 | 18.1063 | -1.1611 | 14.6584 |
| H12_L | R1_D | 20,903 | 13.7004 | 17.1336 | -1.0569 | 13.5599 |

## Primary error by subgroup

Subgroups use validation labels only. Region boundaries are the frozen thirds [0, 33.333…), [33.333…, 66.667…), [66.667, 100]; the extreme tails are actual threat_rank <= 5 and >= 95. These tails are not TP1 target terciles.

| Recipe | Subgroup | n | MAE | RMSE | Signed bias | Over | Under | Equal |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| R0_A | Entry friction = 0 | 26,048 | 23.3122 | 27.3649 | 12.5924 | 69.5% | 30.5% | 0.0% |
| R0_A | Entry friction = 1 | 12,540 | 28.5057 | 31.7613 | -26.1781 | 9.5% | 90.5% | 0.0% |
| R0_A | Low region: threat_rank < 33.333 | 12,843 | 33.3510 | 34.7074 | 33.3510 | 100.0% | 0.0% | 0.0% |
| R0_A | Middle region: 33.333 <= threat_rank < 66.667 | 12,844 | 8.3231 | 9.6109 | 0.0681 | 50.2% | 49.8% | 0.0% |
| R0_A | High region: threat_rank >= 66.667 | 12,901 | 33.2896 | 34.6605 | -33.2896 | 0.0% | 100.0% | 0.0% |
| R0_A | Low tail: threat_rank <= 5 | 1,928 | 47.4920 | 47.5137 | 47.4920 | 100.0% | 0.0% | 0.0% |
| R0_A | High tail: threat_rank >= 95 | 1,955 | 47.4726 | 47.4951 | -47.4726 | 0.0% | 100.0% | 0.0% |
| R0_B | Entry friction = 0 | 26,048 | 17.2494 | 21.2671 | 0.7814 | 53.6% | 46.4% | 0.0% |
| R0_B | Entry friction = 1 | 12,540 | 12.0556 | 15.4254 | -1.4790 | 40.8% | 59.2% | 0.0% |
| R0_B | Low region: threat_rank < 33.333 | 12,843 | 17.5470 | 21.3350 | 16.1655 | 87.0% | 13.0% | 0.0% |
| R0_B | Middle region: 33.333 <= threat_rank < 66.667 | 12,844 | 13.5299 | 16.8324 | -2.6197 | 44.1% | 55.9% | 0.0% |
| R0_B | High region: threat_rank >= 66.667 | 12,901 | 15.6077 | 20.2307 | -13.3447 | 17.3% | 82.7% | 0.0% |
| R0_B | Low tail: threat_rank <= 5 | 1,928 | 22.7228 | 24.6664 | 22.7228 | 100.0% | 0.0% | 0.0% |
| R0_B | High tail: threat_rank >= 95 | 1,955 | 16.8440 | 19.9072 | -16.8440 | 0.0% | 100.0% | 0.0% |
| R0_C | Entry friction = 0 | 26,048 | 17.0554 | 21.0132 | 0.1148 | 52.7% | 47.3% | 0.0% |
| R0_C | Entry friction = 1 | 12,540 | 11.6888 | 14.9640 | -0.6636 | 42.4% | 57.6% | 0.0% |
| R0_C | Low region: threat_rank < 33.333 | 12,843 | 16.8720 | 20.4641 | 15.4674 | 86.8% | 13.2% | 0.0% |
| R0_C | Middle region: 33.333 <= threat_rank < 66.667 | 12,844 | 13.6900 | 17.0190 | -2.7491 | 43.4% | 56.6% | 0.0% |
| R0_C | High region: threat_rank >= 66.667 | 12,901 | 15.3721 | 20.0978 | -13.0741 | 18.0% | 82.0% | 0.0% |
| R0_C | Low tail: threat_rank <= 5 | 1,928 | 21.8879 | 23.2807 | 21.8879 | 100.0% | 0.0% | 0.0% |
| R0_C | High tail: threat_rank >= 95 | 1,955 | 16.7871 | 19.8292 | -16.7871 | 0.0% | 100.0% | 0.0% |
| R1_A | Entry friction = 0 | 26,048 | 17.2858 | 20.8085 | 2.2243 | 56.9% | 43.1% | 0.0% |
| R1_A | Entry friction = 1 | 12,540 | 12.1679 | 15.2598 | -2.6957 | 36.8% | 63.2% | 0.0% |
| R1_A | Low region: threat_rank < 33.333 | 12,843 | 19.1184 | 21.8596 | 18.8058 | 95.3% | 4.7% | 0.0% |
| R1_A | Middle region: 33.333 <= threat_rank < 66.667 | 12,844 | 11.2020 | 14.1755 | -2.0835 | 42.5% | 57.5% | 0.0% |
| R1_A | High region: threat_rank >= 66.667 | 12,901 | 16.5437 | 20.6115 | -14.7764 | 13.5% | 86.5% | 0.0% |
| R1_A | Low tail: threat_rank <= 5 | 1,928 | 26.9520 | 27.5791 | 26.9520 | 100.0% | 0.0% | 0.0% |
| R1_A | High tail: threat_rank >= 95 | 1,955 | 18.6124 | 21.2143 | -18.6124 | 0.0% | 100.0% | 0.0% |
| R1_B | Entry friction = 0 | 26,048 | 17.1426 | 20.6977 | 2.1506 | 57.2% | 42.8% | 0.0% |
| R1_B | Entry friction = 1 | 12,540 | 12.3065 | 15.2203 | -2.7805 | 36.8% | 63.2% | 0.0% |
| R1_B | Low region: threat_rank < 33.333 | 12,843 | 18.7049 | 21.6904 | 18.2446 | 94.4% | 5.6% | 0.0% |
| R1_B | Middle region: 33.333 <= threat_rank < 66.667 | 12,844 | 11.8857 | 14.8381 | -1.6691 | 45.2% | 54.8% | 0.0% |
| R1_B | High region: threat_rank >= 66.667 | 12,901 | 16.1203 | 20.0643 | -14.8613 | 12.2% | 87.8% | 0.0% |
| R1_B | Low tail: threat_rank <= 5 | 1,928 | 25.2989 | 26.0940 | 25.2989 | 100.0% | 0.0% | 0.0% |
| R1_B | High tail: threat_rank >= 95 | 1,955 | 20.0021 | 21.8814 | -20.0021 | 0.0% | 100.0% | 0.0% |
| R1_B_ADD | Entry friction = 0 | 26,048 | 17.1638 | 20.7177 | 2.1969 | 57.1% | 42.9% | 0.0% |
| R1_B_ADD | Entry friction = 1 | 12,540 | 12.3564 | 15.3254 | -2.9431 | 36.0% | 64.0% | 0.0% |
| R1_B_ADD | Low region: threat_rank < 33.333 | 12,843 | 18.7403 | 21.7239 | 18.2816 | 94.4% | 5.6% | 0.0% |
| R1_B_ADD | Middle region: 33.333 <= threat_rank < 66.667 | 12,844 | 11.8870 | 14.8209 | -1.7637 | 44.5% | 55.5% | 0.0% |
| R1_B_ADD | High region: threat_rank >= 66.667 | 12,901 | 16.1750 | 20.1602 | -14.8686 | 12.0% | 88.0% | 0.0% |
| R1_B_ADD | Low tail: threat_rank <= 5 | 1,928 | 25.3259 | 26.1335 | 25.3259 | 100.0% | 0.0% | 0.0% |
| R1_B_ADD | High tail: threat_rank >= 95 | 1,955 | 19.8283 | 21.8071 | -19.8283 | 0.0% | 100.0% | 0.0% |
| R1_C | Entry friction = 0 | 26,048 | 16.8175 | 20.7446 | -0.0871 | 52.3% | 47.7% | 0.0% |
| R1_C | Entry friction = 1 | 12,540 | 11.6832 | 14.9566 | -0.6983 | 42.3% | 57.7% | 0.0% |
| R1_C | Low region: threat_rank < 33.333 | 12,843 | 16.5109 | 20.0960 | 14.9824 | 86.1% | 13.9% | 0.0% |
| R1_C | Middle region: 33.333 <= threat_rank < 66.667 | 12,844 | 13.6056 | 16.9300 | -2.7381 | 43.3% | 56.7% | 0.0% |
| R1_C | High region: threat_rank >= 66.667 | 12,901 | 15.3298 | 19.9733 | -13.0437 | 18.0% | 82.0% | 0.0% |
| R1_C | Low tail: threat_rank <= 5 | 1,928 | 20.7920 | 22.1918 | 20.7821 | 99.6% | 0.4% | 0.0% |
| R1_C | High tail: threat_rank >= 95 | 1,955 | 16.7240 | 19.6659 | -16.7240 | 0.0% | 100.0% | 0.0% |
| R1_D | Entry friction = 0 | 26,048 | 14.5735 | 18.6084 | 0.1672 | 51.4% | 48.6% | 0.0% |
| R1_D | Entry friction = 1 | 12,540 | 10.8081 | 14.4169 | -0.4616 | 44.4% | 55.6% | 0.0% |
| R1_D | Low region: threat_rank < 33.333 | 12,843 | 14.1209 | 18.0938 | 12.0181 | 80.2% | 19.8% | 0.0% |
| R1_D | Middle region: 33.333 <= threat_rank < 66.667 | 12,844 | 13.0297 | 16.4861 | -2.5482 | 41.5% | 58.5% | 0.0% |
| R1_D | High region: threat_rank >= 66.667 | 12,901 | 12.9009 | 17.4549 | -9.5382 | 25.7% | 74.3% | 0.0% |
| R1_D | Low tail: threat_rank <= 5 | 1,928 | 18.2214 | 20.3089 | 18.1913 | 99.1% | 0.9% | 0.0% |
| R1_D | High tail: threat_rank >= 95 | 1,955 | 11.5148 | 15.9226 | -11.2520 | 10.0% | 90.0% | 0.0% |

## Primary impact_load groups

These summaries aggregate per-level validation errors for each observed impact_load value; they are descriptive associations, not causal effects.

| Recipe | Distinct impact_load levels | Groups n<30 / n>=30 | Supported-group n range | Supported-group MAE min / median / max | Supported-group bias min / median / max |
| --- | --- | --- | --- | --- | --- |
| R0_A | 79 | 31/48 | 30–6,000 | 14.5294 / 23.6579 / 41.9542 | -35.3179 / 9.8966 / 41.9542 |
| R0_B | 79 | 31/48 | 30–6,000 | 7.6777 / 17.9210 / 34.6626 | -18.3394 / -0.1529 / 31.8602 |
| R0_C | 79 | 31/48 | 30–6,000 | 7.6777 / 16.9800 / 29.6249 | -13.4008 / -0.6075 / 19.9028 |
| R1_A | 79 | 31/48 | 30–6,000 | 7.8119 / 16.0143 / 20.7178 | -10.2332 / 2.2142 / 9.7744 |
| R1_B | 79 | 31/48 | 30–6,000 | 6.3071 / 15.7597 / 21.0533 | -5.6863 / 1.4024 / 10.3344 |
| R1_B_ADD | 79 | 31/48 | 30–6,000 | 6.4141 / 16.3414 / 20.9993 | -5.9075 / 1.1606 / 7.8502 |
| R1_C | 79 | 31/48 | 30–6,000 | 6.2963 / 15.9029 / 20.9587 | -9.3054 / -1.2429 / 5.0140 |
| R1_D | 79 | 31/48 | 30–6,000 | 7.4549 / 13.5731 / 19.7761 | -11.1334 / 0.0360 / 6.0667 |

Per-level rows are summarized to keep sparse one-row groups from dominating the report. Group support is counted on pooled primary validation rows; the 30-row marker matches the frozen lookup support threshold and is descriptive here.

## Raw versus clipped primary metrics

Clipping is evaluated separately against [0, 100]. Raw metrics remain the primary comparison.

| Recipe | Raw MAE | Clipped MAE | Δ MAE | Raw RMSE | Clipped RMSE | Raw bias | Clipped bias | Raw below 0 | Raw above 100 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| R0_A | 25.0000 | 25.0000 | 0.0000 | 28.8671 | 28.8671 | -0.0069 | -0.0069 | 0 | 0 |
| R0_B | 15.5616 | 15.5616 | 0.0000 | 19.5610 | 19.5610 | 0.0468 | 0.0468 | 0 | 0 |
| R0_C | 15.3114 | 15.3114 | 0.0000 | 19.2570 | 19.2570 | -0.1382 | -0.1382 | 0 | 0 |
| R1_A | 15.6226 | 15.6226 | 0.0000 | 19.1821 | 19.1821 | 0.6254 | 0.6254 | 0 | 0 |
| R1_B | 15.5710 | 15.5710 | 0.0000 | 19.0908 | 19.0908 | 0.5482 | 0.5482 | 0 | 0 |
| R1_B_ADD | 15.6015 | 15.6015 | 0.0000 | 19.1328 | 19.1328 | 0.5266 | 0.5266 | 0 | 0 |
| R1_C | 15.1490 | 15.1490 | 0.0000 | 19.0575 | 19.0575 | -0.2857 | -0.2857 | 0 | 0 |
| R1_D | 13.3498 | 13.3479 | -0.0019 | 17.3576 | 17.3571 | -0.0371 | -0.0383 | 10 | 62 |

## Frozen pairwise comparisons

A negative pooled MAE delta favors the listed candidate on this validation aggregate; counts show the number of primary windows/cycles with lower candidate MAE. This table supports review only and makes no final-model selection.

| Candidate | Control | Pooled Δ MAE | Windows improved | Cycles improved |
| --- | --- | --- | --- | --- |
| R0_B | R0_A | -9.4384 | 6/6 | 24/24 |
| R0_C | R0_B | -0.2502 | 6/6 | 24/24 |
| R1_A | R0_C | 0.3112 | 1/6 | 4/24 |
| R1_B | R1_A | -0.0516 | 4/6 | 16/24 |
| R1_B | R1_B_ADD | -0.0305 | 5/6 | 15/24 |
| R1_C | R1_B | -0.4220 | 6/6 | 24/24 |
| R1_D | R1_C | -1.7991 | 5/6 | 21/24 |

## Fit parameter and fitted-state audit summary

Summaries below are derived from the eight per-window `fit_audits/*.json` files, including all six primary and two H12 fits. Exact fitted states remain in those source audit files; hashes are counted here but not repeated.

| Recipe | Feature preset | Actual parameter summary | Fitted-state summary | Fit time range |
| --- | --- | --- | --- | --- |
| R0_A | two | global target median; minimum group support 30 | fit rows 12,020–42,117; transformed features 0–0; 6 distinct fold state hashes; global fit median 50.000–50.034 | 0.057–0.215 s |
| R0_B | two | impact_load median; fallback global median; minimum group support 30 | fit rows 12,020–42,117; transformed features 1–1; 6 distinct fold state hashes; global fit median 50.000–50.034; joint supported groups 31–50 | 0.115–0.321 s |
| R0_C | two | impact_load x entry_friction median; hierarchical fallback; minimum group support 30 | fit rows 12,020–42,117; transformed features 2–2; 6 distinct fold state hashes; global fit median 50.000–50.034; joint supported groups 33–56, impact supported groups 31–50, EF supported groups 2–2 | 0.253–0.631 s |
| R1_A | two | LinearRegression; intercept=True; StandardScaler(mean=True, std=True) | fit rows 12,020–42,117; transformed features 2–2; 6 distinct fold state hashes; fitted intercept 50.034–50.040 | 0.166–0.368 s |
| R1_B | two | loss=squared_error; lr=0.05; iterations=200; leaves=15; min_leaf_n=30; L2=1.0; max_bins=255; early_stopping=False; interaction_cst=None; seed=42 | fit rows 12,020–42,117; transformed features 2–2; 6 distinct fold state hashes; configured max_iter=200 (no early stopping) | 0.526–2.150 s |
| R1_B_ADD | two | loss=squared_error; lr=0.05; iterations=200; leaves=15; min_leaf_n=30; L2=1.0; max_bins=255; early_stopping=False; interaction_cst=no_interactions; seed=42 | fit rows 12,020–42,117; transformed features 2–2; 6 distinct fold state hashes; configured max_iter=200 (no early stopping) | 0.357–0.816 s |
| R1_C | two | loss=absolute_error; lr=0.05; iterations=200; leaves=15; min_leaf_n=30; L2=1.0; max_bins=255; early_stopping=False; interaction_cst=None; seed=42 | fit rows 12,020–42,117; transformed features 2–2; 6 distinct fold state hashes; configured max_iter=200 (no early stopping) | 0.680–1.458 s |
| R1_D | expanded | loss=absolute_error; lr=0.05; iterations=200; leaves=15; min_leaf_n=30; L2=1.0; max_bins=255; early_stopping=False; interaction_cst=None; seed=42; native categorical inputs=3 | fit rows 12,020–42,117; transformed features 39–39; 6 distinct fold state hashes; configured max_iter=200 (no early stopping) | 1.837–4.692 s |

Source: `fit_audits/W1.json` through `W6.json`, plus `H12_E.json` and `H12_L.json`. HGB settings shown are actual estimator attributes saved after fitting; the R1_D categorical count reflects its fitted native categorical mask.

## Scope and interpretation

- No official-test predictions were loaded or plotted.
- Primary and H12 windows overlap in validation rows, so H12 results are not independent replications.
- Tail definitions use fixed `threat_rank <= 5` and `>= 95`; they are distinct from TP1 tercile partitions.
- All tables are descriptive evidence for the Round 1 review. No final model is selected.
