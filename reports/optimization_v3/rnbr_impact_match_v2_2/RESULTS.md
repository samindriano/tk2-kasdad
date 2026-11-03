# RNBR exact-impact-match V2 diagnostic

Decision: **DIAGNOSTIC_ONLY_GATES_FAIL**. This result is diagnostic only.

- Freeze: `6214df286f4059555a60cff581d7db778eb0cbd6eeb2df44f0df31e389c5a5ff`
- Source commit: `edc416cbb32776ad72f083f9f2d14ad033a911a4`
- Four fit accounting: 4 started, 4 saved, 0 failed.
- No official Test, SampleSubmission, Kaggle, or Classification labels/probabilities were accessed.

| View | n | R06 MAE | Exact-impact observed MAE | Matched shuffle MAE | Original R-NBR MAE | Gain vs R06 |
|---|---:|---:|---:|---:|---:|---:|
| W4 | 4,950 | 11.396232 | 11.535668 | 11.533785 | 11.508788 | -0.139436 |
| W6 | 8,491 | 9.921613 | 9.730564 | 9.892274 | 9.742037 | +0.191049 |
| POOLED_W4_W6 | 13,441 | 10.464680 | 10.395341 | 10.496803 | 10.392689 | +0.069339 |
| H12_E67 | 20,903 | 11.529709 | 11.586566 | 11.624332 | 11.577980 | -0.056857 |

## Gates

- FAIL `W4_gain_vs_R06_nonnegative`
- PASS `W6_gain_vs_R06_nonnegative`
- PASS `pooled_gain_vs_R06_at_least_0_03`
- FAIL `H12_gain_vs_R06_nonnegative`
- PASS `pooled_gain_vs_matched_shuffle_at_least_0_01`
- PASS `H12_gain_vs_matched_shuffle_nonnegative`
- FAIL `pooled_gain_vs_original_RNBR_at_least_0_03`
- FAIL `H12_gain_vs_original_RNBR_nonnegative`
- PASS `W4_target_le_5`
- PASS `W4_target_ge_95`
- PASS `W4_missing_exact`
- PASS `W4_impact_missing_exact`
- PASS `W6_target_le_5`
- PASS `W6_target_ge_95`
- PASS `W6_missing_exact`
- PASS `W6_impact_missing_exact`
- PASS `POOLED_W4_W6_target_le_5`
- PASS `POOLED_W4_W6_target_ge_95`
- PASS `POOLED_W4_W6_missing_exact`
- PASS `POOLED_W4_W6_impact_missing_exact`
- PASS `H12_E67_target_le_5`
- PASS `H12_E67_target_ge_95`
- PASS `H12_E67_missing_exact`
- PASS `H12_E67_impact_missing_exact`

All numerical detail, coverage, subgroup values, movement audit, and counterevidence are in `results.json`.

