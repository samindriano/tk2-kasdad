# R19 — EF-stratified OOF median offsets

**Verdict: RETIRE.** The cache-only pilot applied half the historical median residual as a fixed offset for `entry_friction` 0/1 groups. The route improved the late primary MAE by **0.012735**, below the required +0.03. Supported low-tail MAE worsened by +0.413 on the primary and +0.421 on H12_L, above the +0.25 guard.

## Evidence

- Row-matched current-champion control: `v2_reg_sparse_lgb25`, supported by the pre-existing independent full-six report and pinned in revision 2. Primary W4+W6: **10.477525** on 13,441 rows; H12_L: **11.611686** on 20,903 rows.
- Candidate: primary **10.464790** (gain +0.012735); W4 gain +0.017305; W6 gain +0.010071; H12_L gain +0.017895. H12_E used the exact champion fallback.
- At fit cutoff 67, offsets were +0.465496 for EF0 (n=8,991) and +0.321694 for EF1 (n=4,580). At cutoff 75 they were +0.494587 for EF0 (n=8,606) and +0.495084 for EF1 (n=3,806). Unknown EF used zero; none occurred in these groups.
- Supported low-tail harm was +0.413007 on the primary (n=686) and +0.421027 on H12_L (n=1,048). Supported missing-stream harm was +0.050456 / +0.056686. High-tail MAE improved.
- **0 predictor fits**; precomputed pinned predictions only. No Test, SampleSubmission, or Kaggle access.

## Provenance and QA notes

The first freeze compared the late view with full-six Ridge25 control values; the row-matched baseline check stopped before offsets or candidate metrics. Its freeze is preserved. Revision 2 pins the already verified sparse-LGB25 W4/W6/H12_L metrics, then scores once with a frozen recipe. The first revision 2 wrapper omitted result persistence, so the same deterministic scorer was repeated once to save the report. Its pinned verifier had an indexing defect; the post-freeze independent re-audit reproduced the baseline and all five candidate views. A preparatory worker had transiently loaded target/prediction arrays before freeze without displaying or using values; strict outcome blindness therefore cannot be claimed.

The frozen runner's H12_L guard used 0.5 rather than the card's 0.05. The observed H12_L harm (−0.017895) passes the canonical stricter limit, so the **RETIRE** decision is unchanged. See [report amendment](report_amendment_01.json) and [canonical gate reconciliation](gate.json). No source or frozen artifact was overwritten.

Artifacts: [revision 2 recipe](../../../../experiments/optimization_v3/batch1/R19/revision_02/recipe.json), [revision 2 freeze](../../../../experiments/optimization_v3/batch1/R19/revision_02/freeze.json), [results](results.json), [independent QA](verification.json), [persistence receipt](persistence.json).
