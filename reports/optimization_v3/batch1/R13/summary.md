# R13 — prerequisite skip

**Decision: SKIPPED_PREREQUISITE.** No CatBoostRanker implementation, predictor fit, or budget reservation was made.

The card requires evidence that R12 improved average within-cycle Spearman in both origins while missing its aggregate MAE gate. The saved R12 artifacts contain no per-origin Spearman result, while primary MAE improved by 0.245801 on 9,064 rows and H12 MAE by 0.284490 on 38,588 rows; both aggregate MAE checks passed. R12 retired for a separate supported subgroup guard. Under the declared prerequisite, the ranker was skipped.

An independent read-only audit and main recheck matched the R12 metrics and gate. No official Test or Kaggle was accessed. See results.json, gate.json, and verification.json.
