# CLS A02 position-kernel pilot results

Exploratory result only. The original prelabel boundary was violated by the disclosed E67 cache-member access; this run is not outcome-blind and is not scientifically qualified.

| View | N | Champion Macro F1 | REAL Macro F1 | REAL gain | SHUFFLED Macro F1 | REAL − SHUFFLED |
|---|---:|---:|---:|---:|---:|---:|
| UNIQUE_PRIMARY | 20903 | 0.821941 | 0.821435 | -0.000506 | 0.821738 | -0.000303 |
| E67_H12 | 20903 | 0.811327 | 0.811061 | -0.000266 | 0.811313 | -0.000252 |
| E67_W4 | 4950 | 0.844337 | 0.844139 | -0.000199 | 0.845133 | -0.000995 |
| E67_W5 | 7462 | 0.806246 | 0.805360 | -0.000886 | 0.805758 | -0.000399 |
| E67_W6 | 8491 | 0.792507 | 0.792741 | +0.000234 | 0.792436 | +0.000304 |
| E75_W6 | 8491 | 0.818884 | 0.818501 | -0.000383 | 0.818303 | +0.000199 |

## Decision gates

- Scientific performance gates: **FAIL** (qualification remains **FAIL** because of the recorded protocol deviation).
- Exploratory eligibility: **NOT_ELIGIBLE_EXPLORATORY**.
- Supported subgroup cells: 48; guard failures: 0; unsupported cells remain UNKNOWN.
- E67/E75 overlapping W6 rows: 8491; unique primary pool rows: 20903.
- The operational missing-stream threshold was previously learned from global OOF, so these paired temporal views are adaptive development evidence, not a fresh holdout.
- No Kaggle/Test/E79 action was taken.
