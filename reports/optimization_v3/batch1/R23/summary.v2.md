# R23 Pilot Summary

- Decision: **RETIRE**
- Primary MAE: 13.464396 baseline → 13.372310 candidate; gain +0.092085 on 9,064 rows.
- H12 MAE: 14.047533 baseline → 13.936034 candidate; gain +0.111498 on 38,588 rows.
- W1 / W4 gains: +0.244875 / -0.034900.
- Retirement trigger: the supported-group harm cap (0.25 MAE) failed for missing-stream W1 (+0.488615, n=392), target ≥95 W4 (+0.263713, n=249), and missing-stream primary (+0.344021, n=734).
- Temporal development outcomes are reused, not an untouched holdout. Test/Sample/Kaggle were not read.
