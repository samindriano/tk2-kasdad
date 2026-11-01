# R06 — Character sparse L1 boosting

**Verdict: ADVANCE** — fixed STANDARD temporal Train pilot; no Test/Kaggle access.

Recipe: `char_wb` TF-IDF 3–5 grams, `min_df=3`, cap 30,000, sublinear TF; fit-only structured/OHE plus missing-stream indicator; LightGBM `regression_l1`.
Policy: 0.75 × frozen three-component champion + 0.25 × within-cycle rank50(raw R06), clipped to [0,100].

| View | n | Control MAE | R06 MAE | Gain (control − R06) |
|---|---:|---:|---:|---:|
| W1 | 4,114 | 15.882112 | 15.816833 | +0.065279 |
| W4 | 4,950 | 11.455005 | 11.396232 | +0.058773 |
| primary | 9,064 | 13.464396 | 13.402670 | +0.061726 |
| H12 | 38,588 | 14.047533 | 13.944273 | +0.103260 |

Control and candidate use the same exact row positions per view. Vocabulary, IDF, numeric scaling, and categorical encoding are fit separately from the past-only rows at each cutoff. Anonymous character fragments are not interpreted semantically.

Independent QA confirms two saved fits, zero failed attempts, exact row/ID alignment, reconstructed predictions, and all pilot/subgroup guards passing. Global pilot budget is 70/180 predictor fits.

See `results.json`, `verification.json`, `independent_qa.json`, `subgroups.json`, `gate.json`, `preflight_revision_02.json`, `budget_reconciliation.json`, and `experiments/optimization_v3/batch1/R06/freeze_revision_02.json` for exact provenance, support, fit accounting, memory estimate, and diagnostics.
