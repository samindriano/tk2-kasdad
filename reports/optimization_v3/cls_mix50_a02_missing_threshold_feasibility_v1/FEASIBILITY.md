# Feasibility: threshold transfer for A02 on missing streams

**Scope:** read-only feasibility only. No threshold was selected/scored, no fit or model inference ran, and no Test/Kaggle access occurred.

## Support available in the already-loaded Train OOF diagnostics

The candidate question is whether a threshold selected for saved original A02 probabilities `pA` on missing-stream rows can transfer across time while present rows remain the fixed MIX50 rule.

| Train cycles | Missing rows | Class 0 | Class 1 | n≥200 and each class≥30 |
|---|---:|---:|---:|---|
| 68–71 (E67 W4) | 342 | 202 | 140 | Yes |
| 68–75 (E67 W4+W5) | 849 | 533 | 316 | Yes |
| 68–79 stitched (E67 W4/W5 + E75 W6) | 1,375 | 888 | 487 | Yes |

The minimum support rule is feasible for all three denominators. This is only a support check; it does not establish a transferable threshold or any Macro F1 gain.

## Existing work that does and does not transfer

The existing MIX50 temporal selector varies the present-stream MIX50 cutoff while keeping missing decisions at `pP >= 0.50`. Its selected thresholds and receipts therefore do not test `pA` on missing rows. The fixed-0.50 zero-fit audit already measured the exact recomposition `present=MIX50; missing=pA`, but it did not score an alternate missing threshold.

C01 used sparse25 present plus C1_B missing. C20 used its inherited q80 present route and CatBoost missing. V2 calibration used the older C2_B-present/CatBoost-missing recipe. Those results have different routes and comparators, so they are not direct evidence for threshold transfer on this exact policy.

## Feasible protocol and limits

A later exploratory transfer could choose a `pA` cutoff on an earlier missing-only window, then apply that frozen cutoff to a later window while holding present predictions at `MIX50 >= 0.50`. However, labels for cycles 68–79 have already been opened in prior adaptive development work. A new calculation on those windows would be retrospective sensitivity evidence, not a fresh holdout or confirmatory transfer test.

Missing-only Macro F1 is not the full objective. Any authorized threshold comparison must report the full-row Macro F1/confusion on each window with all present rows included, alongside missing-only counts and the existing supported subgroup guards. A threshold that improves the missing subset can still lower the whole-view score. No evaluation threshold, tie rule, window, or pass gate is selected here.

For exact lineage reuse, the Train OOF probability bundle is `experiments/runs/optimization_v3_posoffset_a02_endgame_v1/policy_predictions.npz` (SHA-256 `25d1d2b91348a57d33b3b0e5ce78da29840d115b75e24f9502cc83c1b63ef351`), bound to official Train SHA-256 `875552a68a3d24b0f3e1966d6ddb5671e6f54305a1c526e2f0a76747942ed91c`. The existing loader/source checks IDs, cycles, missing mask and MIX50 formula; Train labels are joined by ID and cycle. A new receipt would be required because current selector receipts encode missing `pP`, not missing `pA`.

**Feasibility verdict:** enough supported missing rows exist for a clearly labeled retrospective transfer diagnostic if Root authorizes it. The available windows cannot provide untouched confirmation, and the full-row objective must remain primary. No implementation or threshold scoring is authorized by this feasibility note.

Reference evidence: `reports/optimization_v3/cls_missing_current_mix50_audit_v1/RESULTS.json`, `reports/optimization_v3/cls_mix50_temporal_decision_v1/SELECTOR_DIAGNOSTIC.md`, `reports/optimization_v3/posoffset_a02_endgame_v1/RECIPE.json`, and `reports/optimization_v3/posoffset_a02_endgame_v1/POLICY_METRICS.json`.
