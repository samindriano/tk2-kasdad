# C10 — Separate trigram context pilot

- Verdict: **RETIRE**
- Freeze: `415f00250a3d73abd6de9b6816784294f9a126816e15cb61ea3519fe00068045`
- Independent verification: **PASS**
- Predictor fits: 2 successful / 2 attempts; support-only TF-IDF probes: 4
- Threshold: 0.50; present route 0.75 C2_B + 0.25 C10; missing route frozen CatBoost.

| View | n | Baseline Macro F1 | C10 Macro F1 | Delta |
|---|---:|---:|---:|---:|
| W1 | 4114 | 0.803160 | 0.801648 | -0.001512 |
| W4 | 4950 | 0.841418 | 0.843292 | +0.001874 |
| primary | 9064 | 0.823972 | 0.824315 | +0.000343 |
| H12_E | 17685 | 0.795032 | 0.795515 | +0.000483 |
| H12_L | 20903 | 0.808961 | 0.810203 | +0.001241 |
| H12 | 38588 | 0.802966 | 0.803844 | +0.000878 |

Supported subgroup guard details and exact denominators are in `verification.json`.
Temporal windows reuse Train outcomes and are development evidence, not an untouched holdout.
No Test, SampleSubmission, or Kaggle access occurred.
