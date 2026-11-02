# Proposed card — R-NBR Regression neighbor transfer

**State: READY FOR PROTOCOL REVIEW / NOT RUN.** The required first-four-cycle donor predictions and target-free top-neighbor geometry are verified. No new model results exist. Current fit capacity is zero. The future result must remain diagnostic-only because a present-stream-only correction cannot change R06's inherited W1 missing-stream subgroup failure.

## Question

Among present-stream queries, does a local weighted-median transfer of strictly past R06 OOF residuals improve MAE over the unchanged R06 prediction, and does the gain exceed a matched within-cycle/prediction-bin shuffle control?

## Frozen populations and source boundary

- Use only Regression Train `threat_rank` for donor residuals. Do not read or join Classification labels. The query target is opened only after predictions are frozen for metric evaluation.
- Query blocks are the first four cycles at each existing cutoff: cutoff55→56–59; 59→60–63; 63→64–67; 67→68–71; 71→72–75; 75→76–79.
- Reuse the already verified R06 OOF `prediction` (the final champion+R06 blend) as the base and as each donor's prediction. A donor residual is `threat_rank_donor − R06_OOF_prediction_donor`; every donor cycle must be strictly less than the query cycle and must use the R06 prediction from its earlier cutoff.
- The cutoff55 query has no earlier R06 OOF residual donors and is a frozen unchanged-R06 fallback. For cutoffs59–75, donor rows are the present-stream rows in cycles56 through the cutoff.
- Reuse only the verified A02 `neighbor_source_rows`, `neighbor_ids`, and `neighbor_similarities` arrays. Source rows are target-free geometry; do not load A02 `labels`, class predictions, corrections, residuals, or model objects. Require exact ID, cycle, donor-bank, and geometry-hash checks before using any saved neighbor row.

## Candidate recipe

1. Use the at-most-32 positive-cosine neighbors already saved for each query. Similarity weight is cosine squared, matching the fixed text-neighbor geometry convention.
2. Calculate the weighted median of the neighbors' Regression R06 OOF residuals. Sort by residual value; if cumulative weight lands exactly at 50% between two residual values, use their midpoint. Resolve equal residual sorting by source row ID. Weighted median is the candidate because absolute error is the objective; weighted mean residual transport is not an alternate search arm.
3. Use a fixed shrinkage scale `alpha = 0.25` and cap the correction at `[-5, +5]` threat-rank points. These are predeclared conservative design values, not values selected from Regression outcomes.
4. Apply `clip(R06_prediction + alpha * weighted_median_residual, 0, 100)`. Do not rank-map again after this addition. R06's existing component rank50 is already computed once, on each complete scored cycle with average ties.
5. Missing-stream queries, zero-neighbor queries, or queries with fewer than five positive neighbors receive zero correction and exactly the original R06 prediction. The same fallback applies if any donor integrity check fails; fail the run if alignment is invalid rather than silently repairing it.

## Matched control

Use identical queries, neighbors, weights, scale, cap, and fallback. Before constructing the control memory, permute donor residuals within exact `cycle_code × decile(R06 donor prediction)` cells using fixed seed 42. This preserves each cell's donor residual distribution and temporal/base-prediction mix while breaking the residual-to-neighbor assignment. No shuffle is allowed across cycles or deciles. Candidate and control are both compared with unchanged R06.

## Evaluation and interpretation

Report candidate and control MAE, delta versus R06, and candidate gain over the shuffled control for each of the six windows, pooled primary, and H12. Include every protected legacy subgroup with denominator and harm, missing-stream groups, tail groups, and all negative results. Do not use these reused Train views as untouched validation or as a Kaggle selection rule. Keep R06's original verdict and guard thresholds intact; because the missing W1 fallback is unchanged, this experiment cannot clear R06's existing missing-stream failure.

## Cost and blocked action

Cutoff55 requires no target-memory fit because it deterministically falls back. Cutoffs59, 63, 67, 71, and 75 require one observed Regression target-memory predictor and one matched shuffled memory each: **10 task fits total**. The neighbor geometry is reused from verified saved arrays, so no new text-vectorizer fit is needed. Under the current A02 accounting convention, each target-memory/index predictor counts as a task fit even though it is a weighted lookup without gradient optimization.

Current Research Next cap: 66/66 starts, no pending fits. **Do not run this card unless a later explicit budget amendment authorizes the required slots and the final freeze is committed first.**

## Integrity references

- Reproduction script: [`scripts/optimization_v3_rnbr_preflight.py`](../../../scripts/optimization_v3_rnbr_preflight.py)
- Verified inventory: [`INVENTORY.json`](INVENTORY.json)
- Historical R06 recipe: `experiments/optimization_v3/batch1/R06/recipe_revision_02.json`
- Historical counterevidence: `reports/optimization_v3/regression_residual/FINDINGS.md` and `reports/optimization_v3/regression_diagnostics/FINDINGS.md`
