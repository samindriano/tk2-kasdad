# C09 — Pilot summary

- Verdict: **RETIRE**
- Status: `PILOT_COMPLETE_RETIRE`
- Frozen identity: `75da6a9f265778d2dd32dcc3764c6299308f45075fff9d1dbc7691ca31c1c877`
- Verification: `PASS_WITH_CONTROL_SCOPE_RECONCILIATION`
- Predictor fits: 2 successful / 2 attempts; convergence warnings: 0

| View | n | Baseline Macro F1 | C09 Macro F1 | Delta |
|---|---:|---:|---:|---:|
| W1 | 4114 | 0.803160 | 0.800414 | -0.002746 |
| W4 | 4950 | 0.841418 | 0.843888 | +0.002471 |
| primary_pilot | 9064 | 0.823972 | 0.824083 | +0.000110 |
| H12_E | 17685 | 0.795032 | 0.794524 | -0.000508 |
| H12_L | 20903 | 0.808961 | 0.809157 | +0.000196 |
| H12 | 38588 | 0.802966 | 0.802798 | -0.000168 |

Supported group harms and denominators are in `results.json`; small groups remain descriptive.

- Route: present stream uses 0.75 C09 + 0.25 frozen sparse binary; missing stream uses frozen CatBoost; threshold 0.50.
- Limitation: temporal Train-cycle development evidence, not an untouched holdout.
- Test/Sample/Kaggle access: none.
