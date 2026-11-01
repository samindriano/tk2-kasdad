# N03 — Past-only router over R22/R06

**Status:** `RETIRED_PERFORMANCE`. Pilot fit the frozen router at cutoffs 67 and 75 only.

## LATE views

| View | n | Router MAE | R22 A MAE | Fixed B MAE | Constant control MAE | Gain vs A | Gain vs constant |
|---|---:|---:|---:|---:|---:|---:|---:|
| W4_67_to_68_71 | 4,950 | 11.369630 | 11.390307 | 11.346119 | 11.346119 | +0.020677 | -0.023511 |
| W6_75_to_76_79 | 8,491 | 9.905745 | 9.880454 | 9.908256 | 9.908256 | -0.025291 | +0.002511 |
| late_primary | 13,441 | 10.444859 | 10.436497 | 10.437787 | 10.437787 | -0.008362 | -0.007072 |
| late_h12 | 20,903 | 11.483243 | 11.511220 | 11.449377 | 11.449377 | +0.027977 | -0.033866 |

## Gates and interpretation

- `late_primary_gain_vs_A_ge_0_03`: FAIL
- `late_h12_harm_vs_A_le_0_05`: PASS
- `each_primary_window_harm_le_0_5`: PASS
- `supported_missing_and_target_tail_harm_le_0_25_vs_A_and_legacy`: FAIL
- `late_primary_gain_vs_past_only_constant_ge_0_01`: FAIL
- `at_least_one_primary_row_routed_to_B`: PASS

The router uses only prior OOF rows to learn the loss-difference head and imputer. The B arm is fixed in advance; missing-stream rows always receive exact R22 arm A. R06_ORIGINAL is reported as a secondary cache comparator and does not set the control. Group masks use target values only after prediction for evaluation.
