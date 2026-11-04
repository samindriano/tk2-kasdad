# REG_POSITION_L1_V1 results

**Decision:** RETIRE_PILOT_NO_CANDIDATE_PASSED

This is adaptive development evidence with overlapping E67/E75 outcomes. It is not untouched validation or submission authorization.

## Primary and window metrics

| View | n | R06 MAE | BAG gain | REAL gain | SHUFFLED gain |
|---|---:|---:|---:|---:|---:|
| E67_H12 | 20903 | 11.529709 | -0.036970 | -0.035288 | -0.043885 |
| E67_W4 | 4950 | 11.396232 | -0.120112 | -0.110751 | -0.128718 |
| E67_W5 | 7462 | 11.477448 | -0.053272 | -0.049600 | -0.064606 |
| E67_W6 | 8491 | 11.653450 | +0.025824 | +0.021282 | +0.023779 |
| E75_W6 | 8491 | 9.921613 | +0.135906 | +0.151424 | +0.133730 |
| LATE_UNIQUE_E67_W4_PLUS_E75_W6 | 13441 | 10.464680 | +0.041621 | +0.054871 | +0.037077 |

## Fixed subgroup gates

- **REAL:** 61 PASS, 4 FAIL, 0 UNKNOWN group checks; competition gate: FAIL.
- **BAG:** 59 PASS, 6 FAIL, 0 UNKNOWN group checks; competition gate: FAIL.
- **SHUFFLED:** 59 PASS, 6 FAIL, 0 UNKNOWN group checks; competition gate: FAIL.

## REAL mechanism comparison

- E67_H12_gain_vs_BAG_ge_0_01: FAIL (+0.001682; threshold 0.01).
- E67_H12_gain_vs_SHUFFLED_ge_0_01: FAIL (+0.008597; threshold 0.01).
- E75_W6_gain_vs_BAG_ge_0_01: PASS (+0.015518; threshold 0.01).
- E75_W6_gain_vs_SHUFFLED_ge_0_01: PASS (+0.017695; threshold 0.01).

REAL mechanism gate: FAIL.

## Fit accounting

- Starts: 6; saved: 6; failed: 0; retries: 0; isolated cap: 6.
- Query targets were read only after all six saved artifacts passed independent prelabel checks.
- Classification labels/probabilities, Test, Sample Submission, Kaggle, and the shared main ledger were not accessed or written.
- Reported in-sample training MAE is descriptive only; selection uses the frozen development gates.

## Charts

- REG_POSITION_L1_V1_PRIMARY_GAINS.png compares MAE gains across all frozen views.
- REG_POSITION_L1_V1_GROUP_HARM.png shows fixed subgroup harm and marks unsupported denominators UNKNOWN.

A later tranche requires Sol review and its own source/freeze/accounting amendment. No E79 fit or upload is authorized by this pilot.
