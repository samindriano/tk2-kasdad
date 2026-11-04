# REG_POSITION_DIRECT_L1_V1

Status: **FAIL**

This is an adaptive development pilot. E67 and E75 query views overlap; scores are engineering gates, not significance tests.

## Primary views

| View | n | R06 MAE | BAG MAE | REAL MAE | SHUFFLED MAE | REAL gain vs R06 | REAL gain vs BAG | REAL gain vs SHUFFLED |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| E67_H12 | 20903 | 11.529709 | 11.519753 | 11.508149 | 11.519058 | 0.021560 | 0.011604 | 0.010909 |
| E67_W4 | 4950 | 11.396232 | 11.392147 | 11.375229 | 11.389084 | 0.021003 | 0.016918 | 0.013854 |
| E67_W5 | 7462 | 11.477448 | 11.465856 | 11.478174 | 11.473069 | -0.000726 | -0.012318 | -0.005106 |
| E67_W6 | 8491 | 11.653450 | 11.641508 | 11.611980 | 11.635246 | 0.041470 | 0.029528 | 0.023266 |
| E75_W6 | 8491 | 9.921613 | 9.977921 | 9.986419 | 9.973275 | -0.064806 | -0.008499 | -0.013145 |
| LATE_UNIQUE_E67_W4_PLUS_E75_W6 | 13441 | 10.464680 | 10.498746 | 10.497885 | 10.494683 | -0.033205 | 0.000862 | -0.003201 |

## Gate checks

- **REAL_gain_vs_R06_E67_H12: FAIL**
  - Observed 0.021560; threshold 0.030000
- **REAL_gain_vs_R06_E75_W6: FAIL**
  - Observed -0.064806; threshold 0.030000
- **REAL_gain_vs_BAG_E67_H12: PASS**
  - Observed 0.011604; threshold 0.010000
- **REAL_gain_vs_SHUFFLED_E67_H12: PASS**
  - Observed 0.010909; threshold 0.010000
- **REAL_gain_vs_BAG_E75_W6: FAIL**
  - Observed -0.008499; threshold 0.010000
- **REAL_gain_vs_SHUFFLED_E75_W6: FAIL**
  - Observed -0.013145; threshold 0.010000
- **REAL_E67_W4_harm: PASS**
  - Observed -0.021003; threshold 0.100000
- **REAL_E67_W5_harm: PASS**
  - Observed 0.000726; threshold 0.100000
- **REAL_E67_W6_harm: PASS**
  - Observed -0.041470; threshold 0.100000
- **protected_groups: PASS**
  - Unsupported group/view cells: 0 (reported UNKNOWN)
  - Supported harm failures: 0

## Interpretation and overlap

The late unique view uses E67 W4 rows for cycles 68-71 plus E75 W6 rows for cycles 76-79. E75 W6 overlaps E67 W6 exactly, so the E67 W6 copy is omitted from the unique union.

Unsupported protected groups remain UNKNOWN. No arm is promoted or uploaded automatically.

The verifier read Regression Train query targets only after all 12 fit units and six raw prediction bundles passed prelabel verification. Test, Sample, and Kaggle data were not accessed.
