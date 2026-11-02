# Proposed next card — confidence-conditioned attenuation

**Status:** PROPOSAL_FOR_REVIEW_ONLY. Do not execute as part of the completed audit.

## Question

Can tapering the frozen A02 residual correction only when Sparse25 is highly positive-confidence reduce wrong exact-endpoint outputs while retaining the observed gains through the mid-confidence range?

## Evidence motivating the proposal

In the full-six primary maximum-cosine × Sparse25-confidence table, 37/50 cells meet the fixed support rule. Brier improvement over Sparse25 is positive in 27 supported cells and negative in 10. The supported-cell pattern is broadly repeated in the new-origin (17/24 positive) and late (10/16 positive) views: gains concentrate around mid-confidence probabilities, while high-confidence bands are weaker or negative. In contrast, age × broad-similarity has only 9/18 supported full-primary cells and no consistent monotonic age pattern within similarity. This comparison motivates a reliability/confidence diagnostic ahead of a neighbor-age weighting study; it does not establish the mechanism.

## Frozen diagnostic recipe

Use the existing saved A02 observed and matched-shuffle probabilities only. Define the current correction as `delta = p_A02 - p_Sparse25`, and apply a parameter-free attenuation:

`g(p) = 1` for `p <= 0.8`; `g(p) = (1-p)/0.2` for `0.8 < p < 1`; and `g(p) = 0` for `p >= 1`.

`p_candidate = clip(p_Sparse25 + g(p_Sparse25) * delta, 0, 1)`.

The one-sided taper uses the already fixed 0.8 band boundary: the observed lower-confidence bins showed positive or near-zero Brier change, while the two supported high-confidence bands showed negative average Brier change. This formula is an explicitly post-result proposal, not an outcome-blind discovery. Do not tune its formula or parameters on these results. Compare it with unchanged Sparse25, unchanged A02, and the correspondingly attenuated matched shuffle. Preserve threshold 0.50 and report Macro F1, confusion counts, AUC/AP, Brier, both log-loss conventions, exact/wrong endpoints, and the original protected subgroup gates. Report full-six primary, new-origin, late, and H12-L separately; H12-L remains overlapping.

## Cost, boundaries, and decision limits

- Estimator fits: **0**. This is deterministic postprocessing on the already verified, ignored row-level outputs; no model or preprocessor training is needed.
- Compute: one vectorized pass over existing rows; low CPU and memory. No new source data is read.
- No Test/SampleSubmission, Kaggle, submission, champion, or notebook access/change.
- All validation outcomes in these views have already been inspected. Any rerun on them is diagnostic only and cannot qualify or promote the candidate. A confirmatory decision requires a newly authorized labeled temporal holdout; none is available in this audit.
- Failure meaning: if the fixed attenuation does not preserve the primary and protected subgroup gates, stop the confidence-shrinkage lane. Do not tune another multiplier on the same rows. If gates pass, request a separate review for a prospective validation source before making any model-selection claim.
