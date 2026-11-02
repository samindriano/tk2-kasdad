# A02 reliability audit — frozen analysis plan

This plan is locked before opening the saved per-group outcome table for this audit. It uses only persisted predictions, saved neighbor geometry/bank artifacts, and official Classification Train labels joined by verified score-row identity. No model/preprocessor fit, Test/SampleSubmission access, Kaggle/UI access, or submission generation is allowed.

## Prior knowledge disclosed

The continuation aggregate anchors and one historical supported-subgroup Macro F1 guard flag were already known from prior reports. Therefore this is not outcome-blind research. This audit asks where the frozen correction helps or harms and whether one mechanism merits a future protocol proposal; it cannot reverse historical gates or establish untouched validation.

## Population and views

- Verify the official Train SHA-256 against the frozen value before reading target labels.
- Reconstruct each saved learned origin from its exact manifest and persisted arrays: 59, 63, 67, 71, and 75 when available. Require `origin_i < cycle_i <= E < query_cycle` for every neighbor; join `y_true` only by verified `score_rows` + `score_record_ids` + cycle. NPZ `labels` are predicted labels and must never be treated as `y_true`.
- Build unique-row views: full-six primary (38,588, W1 unchanged Sparse25 fallback plus five learned origins), new-origin aggregate (21,033; W2/W3/W5), late (15,953; W5/W6), and inherited H12-L (20,903) separately. H12-L is not independent; overlapping queries are not double-counted in row-level pooled summaries.
- Missing-stream rows are an exact Sparse25 equality control and are excluded from geometry/reliability grouping. Report their row count and equality checks. Learned-origin groups use present-stream rows and expose denominator and cycle coverage.

## Feature-available descriptors

Using only the saved query-to-bank geometry/weights and frozen historical OOF residual bank, compute: maximum cosine; `sum(s^2)`; ESS `(sum(w)^2 / sum(w^2))`; positive-weight neighbor count; frozen shrink factor `mass/(4+mass)`; weighted residual mean and dispersion; absolute raw correction; cap boundary at `|raw| >= 0.05` and strict clip at `|raw| > 0.05` as separate flags; Sparse25 probability; weighted mean bank age `E-bank_cycle` and query age `query_cycle-bank_cycle`; weight mass from the last four fit cycles; represented cycle count; largest cycle concentration; and weighted residual sign consistency. Preserve zero text, no positive weights, ties, and single-effective-neighbor cases as explicit categories/flags; do not silently drop them. Fixed age bins apply to `E-bank_cycle`; query age is a companion recency descriptor on the same fixed edges.

## Fixed descriptive bins and support

No bin edges will be selected after inspecting outcomes. Use left-closed/right-open intervals, with the last interval right-closed:

- max cosine: `[0,.2), [.2,.4), [.4,.6), [.6,.8), [.8,1]`;
- ESS: `[0,4), [4,8), [8,16), [16,32), [32,+inf)`;
- Sparse25 probability: `[0,.1), [.1,.2), ...,[.9,1]`;
- weighted mean neighbor age: `[0,4), [4,8), [8,16), [16,32), [32,+inf)`;
- sign consistency: `[0,.5), [.5,.6), [.6,.7), [.7,.8), [.8,.9), [.9,1]`.

For any F1 comparison, label a bin supported only if pooled `n >= 200` and both true classes have at least 30 rows. Otherwise report n/class counts and mark F1 unsupported; do not merge bins post hoc. Broad similarity bands are the fixed-edge coarsening `[0,.4), [.4,.6), [.6,1]`. Two-way contrasts are limited to detailed similarity × baseline-probability and age × broad-similarity; every reported cell uses the same support rule.

## Outcomes and contrasts

- Threshold is frozen at 0.50. In supported bins, report n, class counts, cycle coverage, Macro F1, TP/FP/FN/TN, candidate-vs-Sparse25 and candidate-vs-matched-shuffle; F1 is not additive across bins.
- For additive summaries report paired per-row Brier improvement and classification-error improvement (baseline loss minus candidate loss; positive favors A02), plus log-loss deltas. State the clipping used for endpoint log-loss: show both the previously reported `1e-15` diagnostic and scikit-learn's native epsilon convention where comparable; do not alter probabilities.
- Compare cap-hit/uncapped and incorrect exact-endpoint groups descriptively, separating wrong extreme confidence from ranking/threshold effects. Compare observed assignment with matched shuffle in identical groups. Inspect direction by learned origin and within broad baseline-confidence bands.
- Local-recency is observational only: compare age groups within supported broad similarity/confidence strata; it does not identify a causal effect of neighbor age.

## Decision rule

Recommend exactly one future card only if saved evidence distinguishes (A) reliability-dependent local shrinkage beyond the frozen `mass/(4+mass)` rule from (B) neighbor-age weighting within similar neighbors. Otherwise return `STOP_LOW_VALUE_SEARCH`. Any card is a proposal only: include its minimal estimator-fit count including matched control/inner selection, origin boundaries, features available at prediction time, gates, resource estimate, and failure meaning. The current campaign remains at the consumed 66/66 fit cap; this audit authorizes zero fits and zero submissions.
