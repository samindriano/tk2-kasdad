# A02 decision-boundary audit — fixed zero-fit plan

**Status:** adaptive diagnostic, not outcome-blind. This follow-up was initiated after reviewing the previous A02 audit, and after the reviewer identified that the probability-only taper cannot cross the 0.50 label threshold under the frozen ±0.05 correction bound. The band edges and support rules below are fixed before this follow-up's label-flip tables are computed.

## Source and boundaries

- Read only official Classification Train (`data/raw/classification/Train.csv`) and the four persisted A02 audit row artifacts: full-six primary, new-origin aggregate, late, and inherited H12-L. Verify frozen source SHA, row/cycle/target alignment, unique row IDs, cache/report hashes, and historical QA.
- Reuse official Train labels by exact row-index join and validate them against saved cycles and prior ID/cycle QA. Never use NPZ prediction labels as truth without this join.
- W1 is the unchanged Sparse25 fallback (4,114 rows); disclose it and exclude it from learned-neighbor descriptor analysis. In full-six primary, 32,288 rows are learned-present; 2,186 learned-origin rows are missing-stream/no-correction. H12-L overlaps full-six primary and is reported separately.
- Do not read Test/SampleSubmission, query Kaggle or browser state, fit/refit any model or preprocessor, transform text, alter probabilities, tune a policy, or create a submission.

## Fixed calculations

- Use hard labels `p >= 0.50`, including exact 0.50 as positive.
- Verify `|p_arm - p_baseline| <= 0.05` for both observed and matched-shuffle arms. Count every baseline-to-arm label change and assert all changes lie in the inclusive baseline interval `[0.45, 0.55]`; report exact boundary cases separately. No epsilon-based label decisions.
- Use fixed baseline-probability bands `[0.45,0.475)`, `[0.475,0.50)`, `[0.50,0.525)`, `[0.525,0.55]`. Do not search edges or add bins.
- For each band and arm, count 0→1 and 1→0 changes. Decompose into baseline-wrong→arm-correct and baseline-correct→arm-wrong; report net error-count and error-rate change, positive-prediction-rate change, confusion counts, Macro F1, and class counts. F1 is non-additive.
- Report full-six, each origin's declared first-four window, new-origin aggregate, late aggregate, and H12-L separately. Use the same query rows for baseline, observed, and shuffle within each view. H12-L is overlapping evidence. W1 fallback is explicitly reported as unchanged.
- Within each fixed band, compare descriptor summaries among unchanged, useful-flip, and harmful-flip rows for ESS, similarity mass, maximum cosine, residual SD, residual sign consistency, cycle-weight concentration, and weighted origin/query age. No post-result descriptor bins or features used to route predictions.
- Boundary descriptor support: at least 100 rows in band and 20 label changes for an arm. Macro F1 support is assessed separately using at least 200 rows, at least 30 positives, and at least 30 negatives. Mark all unsupported cells explicitly and show unchanged-row context.
- Evaluate whether useful-vs-harmful flips and descriptor contrasts repeat across origins after conditioning on baseline-probability band, rather than reflecting only an increase in predicted-positive rate. All subgroup patterns are descriptive, not causal and not inference-time features.
- A follow-on Macro F1 mechanism card requires at least 20 useful and 20 harmful flips in the same supported band for at least 3/5 origin windows, with the same median descriptor contrast direction in the new-origin and late aggregates. These per-origin descriptor strata must meet the fixed n≥100 and ≥20 changed-label support; Macro F1 class support is reported separately and is required in full-primary, new-origin, and late aggregate comparisons, but not imposed on smaller origin windows. If no unique descriptor/band meets these conditions, report `NO_SUPPORTED_F1_LANE`; do not lower support or choose an edge post hoc.

## Decision and budget

- Zero estimator/preprocessor fits, no budget reserve or release. The ledger SHA-256 and 66-start / 64-saved / 2-historical-failure accounting must be identical before and after the audit.
- Retain no next Macro F1 mechanism unless label-changing groups meet support and show a repeatable useful-vs-harmful pattern across origins. Otherwise report `NO_SUPPORTED_F1_LANE`.
- Probability-quality observations are separate from Macro F1. The high-confidence taper proposal cannot be promoted as a Macro F1/Kaggle lane if it produces no label changes.
- Output only a compact summary, flip and band-descriptor tables, QA, and relevant metric-only tests. Commit only this follow-up and the correction to the prior summary/card; no push. Stop at the review checkpoint.
