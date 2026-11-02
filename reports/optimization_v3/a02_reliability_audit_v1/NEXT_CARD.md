# Review note — confidence-conditioned attenuation

**Status:** REJECTED_AS_MACRO_F1_OR_KAGGLE_LANE. This is not an active experiment card.

## Question

Could tapering the frozen A02 residual correction when Sparse25 is highly positive-confidence improve probability quality?

## Evidence motivating the proposal

In the full-six primary maximum-cosine × Sparse25-confidence table, 37/50 cells meet the fixed support rule. Brier improvement over Sparse25 is positive in 27 supported cells and negative in 10. The supported-cell pattern is broadly repeated in the new-origin (17/24 positive) and late (10/16 positive) views: gains concentrate around mid-confidence probabilities, while high-confidence bands are weaker or negative. In contrast, age × broad-similarity has only 9/18 supported full-primary cells and no consistent monotonic age pattern within similarity. This comparison motivates a reliability/confidence diagnostic ahead of a neighbor-age weighting study; it does not establish the mechanism.

## Optional probability-quality observation only

Use the existing saved A02 observed and matched-shuffle probabilities only. Define the current correction as `delta = p_A02 - p_Sparse25`, and apply a parameter-free attenuation:

`g(p) = 1` for `p <= 0.8`; `g(p) = (1-p)/0.2` for `0.8 < p < 1`; and `g(p) = 0` for `p >= 1`.

`p_candidate = clip(p_Sparse25 + g(p_Sparse25) * delta, 0, 1)`.

The one-sided taper uses the already fixed 0.8 band boundary: the observed lower-confidence bins showed positive or near-zero Brier change, while the two supported high-confidence bands showed negative average Brier change. This formula is an explicitly post-result idea, not an outcome-blind discovery. It was not run, and no extra taper experiment or submission is authorized or recommended. If probability calibration becomes a separate objective, review it as such; it cannot be presented as a Macro F1 improvement.

## Cost, boundaries, and decision limits

- Estimator fits: **0**. This is deterministic postprocessing on the already verified, ignored row-level outputs; no model or preprocessor training is needed.
- Compute: one vectorized pass over existing rows; low CPU and memory. No new source data is read.
- No Test/SampleSubmission, Kaggle, submission, champion, or notebook access/change.
- The threshold-label-preservation claim is verified directly in the decision-boundary follow-up report. Do not rerun or tune the taper as a route to Macro F1.
- No Test/SampleSubmission, Kaggle, submission, model, gate, champion, or notebook change is authorized by this note.
