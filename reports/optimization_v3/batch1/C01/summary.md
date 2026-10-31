# C01 Cache-only Route Summary

- Decision: **RETIRE**.
- Primary Macro F1: 0.829944 sparse25 → 0.827770 C01; gain -0.002174 on 38,588 rows.
- H12 Macro F1: 0.802966 → 0.800791; gain -0.002175 on 38,588 rows.
- W1–W6 gains: -0.008463, -0.002400, -0.002154, +0.002272, -0.000581, -0.003044; 1/6 windows nonnegative.
- Supported subgroup guard failures: 12.
- Six temporal windows were recomposed from frozen Train-fold caches; no predictor or preprocessing fit was run.
- Temporal validation outcomes are reused development evidence, not untouched holdout evidence.
- Classification Test, SampleSubmission, and Kaggle were not accessed.
