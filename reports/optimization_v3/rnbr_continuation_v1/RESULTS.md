# R-NBR three-origin continuation — corrected post-fit results

**Status:** `EVALUATED_DIAGNOSTIC_ONLY`; this tranche does not qualify a global model.

**Freeze:** `f469cfcda87cf1452d3e54d725eb390f299153b54e9a343bbad2bf2dc01d593b`  
**Fit source commit:** `c1f23d4427faac014c7cba2a8c510c92b94505e0`  
**Post-fit adapter commit:** `9361d718a026b1f7bfbce17fe034fdf6475e63ab`

> The initial frozen-verifier output is preserved in `frozen_verifier_output_pre_correction_*` and superseded. It overlaid E67 predictions onto shared W5/W6 maps before computing the primary views. The corrected results below use isolated per-view maps and the frozen origins/gates.

## Main views

| View | n | R06 MAE | R-NBR observed | Shuffled | R22_ORIGINAL | Gain vs R06 | Gain vs shuffle | Gain vs R22 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| W1 | 4114 | 15.816833 | 15.816833 | 15.816833 | 15.743656 | +0.000000 | +0.000000 | -0.073178 |
| W2 | 6493 | 11.857170 | 11.337376 | 11.842248 | 11.833929 | +0.519793 | +0.504871 | +0.496553 |
| W3 | 7078 | 11.926889 | 11.456377 | 11.936401 | 12.084101 | +0.470512 | +0.480024 | +0.627724 |
| W4 | 4950 | 11.396232 | 11.508788 | 11.477601 | 11.390307 | -0.112556 | -0.031187 | -0.118481 |
| W5 | 7462 | 10.281664 | 10.169007 | 10.301404 | 10.333223 | +0.112657 | +0.132396 | +0.164216 |
| W6 | 8491 | 9.921613 | 9.742037 | 9.901416 | 9.880454 | +0.179576 | +0.159379 | +0.138416 |
| new_origin_aggregate | 21033 | 11.321680 | 10.962913 | 11.327278 | 11.385703 | +0.358767 | +0.364365 | +0.422789 |
| full_primary | 38588 | 11.502413 | 11.281786 | 11.511458 | 11.519691 | +0.220627 | +0.229672 | +0.237905 |
| late_W5_W6 | 15953 | 10.090026 | 9.941752 | 10.088510 | 10.092236 | +0.148274 | +0.146758 | +0.150484 |
| inherited_H12_L_E67 | 20903 | 11.529709 | 11.577980 | 11.582025 | 11.511220 | -0.048271 | +0.004046 | -0.066760 |

## Gate summary

- `full_primary_gain_vs_R06_ge_0_03`: **True**
- `full_primary_gain_vs_shuffle_ge_0_01`: **True**
- `new_origin_gain_vs_R06_nonnegative`: **True**
- `new_origin_gain_vs_shuffle_nonnegative`: **True**
- `learned_primary_windows_nonnegative_at_least_3_of_5`: **True**
- `late_W5_W6_gain_vs_R06_positive`: **True**
- `inherited_H12_L_harm_vs_R06_le_0_05`: **True**
- `each_learned_window_harm_vs_R06_lt_0_25`: **True**
- `all_supported_subgroup_guards_pass`: **True**

## Supported subgroup harm vs R06

Harm is candidate MAE minus origin-matched R06 MAE; positive values are worse. Minimum support is the frozen n≥100 rule.

| View | Missing stream n / harm | Target ≤5 n / harm | Target ≥95 n / harm |
|---|---:|---:|---:|
| W1 | 392 / +0.0000 | 205 / +0.0000 | 213 / +0.0000 |
| W2 | 369 / +0.0000 | 322 / -1.8483 | 325 / -0.6141 |
| W3 | 442 / +0.0000 | 353 / -1.0776 | 360 / -0.5978 |
| W4 | 342 / +0.0000 | 244 / -0.9117 | 249 / -0.0397 |
| W5 | 507 / +0.0000 | 362 / -0.3550 | 374 / -0.2271 |
| W6 | 526 / +0.0000 | 442 / -0.6507 | 434 / -0.1484 |
| new_origin_aggregate | 1318 / +0.0000 | 1037 / -1.0647 | 1059 / -0.4719 |
| full_primary | 2578 / +0.0000 | 1928 / -0.8372 | 1955 / -0.2936 |
| late_W5_W6 | 1033 / +0.0000 | 804 / -0.5176 | 808 / -0.1849 |
| inherited_H12_L_E67 | 1375 / +0.0000 | 1048 / -0.9569 | 1057 / -0.0314 |

Archived R06-vs-legacy W1 missing-stream guard remains failed: harm **+0.395556 MAE** on n=392; continuation does not clear this historical risk.

## Interpretation

Results are diagnostic and origin comparisons are descriptive. The fit identities and recipes are unchanged; corrected scoring changes only map assembly so each declared window uses its frozen origin. No full-Train refit, Test access, Kaggle access, or additional fit occurred.

## Reproduction

The six official bundles remain under the ignored continuation run directory. The original frozen-verifier files and outputs are preserved; this report uses `scripts/optimization_v3_rnbr_continuation_corrected_eval.py`.
