# A02 Reliability and Temporal Mechanism Audit

**Status:** COMPLETED_ZERO_FIT

**Decision:** one diagnostic proposal is justified: confidence-conditioned attenuation. This is a research proposal only; it does not qualify A02, change a gate, or change a champion.

## Scope and integrity

This audit reconstructed frozen A02 observed and within-cycle shuffled outputs from saved predictions, persisted neighbor geometry, the frozen residual bank, and official Classification Train labels joined by verified row ID and cycle. It did not fit or refit a model or preprocessor, read Test/SampleSubmission, access Kaggle, alter probabilities, or create a submission. The analysis plan records that continuation anchors and a historical subgroup flag were already known; this was not outcome-blind work.

All five stored origins (59, 63, 67, 71, 75) passed source and cache checks. Each origin-specific first-four-cycle diagnostic used only its declared four-cycle query window. The full-six primary view contains 38,588 unique rows across cycles 56–79. Of these, 36,010 have a present stream, but only 32,288 are learned-present rows with an A02 neighbor correction. W1 contributes 4,114 unchanged Sparse25 fallback rows, including 3,722 present-stream rows; 2,186 additional rows in learned-origin windows have a missing stream and no correction. The full view therefore contains 6,300 unchanged rows. The new-origin view contains 21,033 unique rows. The late view contains 15,953. H12-L contains 20,903 rows and overlaps the primary view completely, so it is reported separately and is not independent evidence.

## Main results

| View | n | Sparse25 Macro F1 | A02 Macro F1 | Matched-shuffle Macro F1 | A02 minus Sparse25 | A02 minus shuffle |
|---|---:|---:|---:|---:|---:|---:|
| Full-six primary | 38,588 | 0.829944 | 0.832045 | 0.829277 | +0.002101 | +0.002767 |
| New origins (W2/W3/W5) | 21,033 | 0.835171 | 0.837187 | 0.834268 | +0.002016 | +0.002919 |
| Late (W5/W6) | 15,953 | 0.815790 | 0.816958 | 0.813424 | +0.001168 | +0.003534 |
| Inherited H12-L, overlapping | 20,903 | 0.808961 | 0.811159 | 0.809773 | +0.002198 | +0.001386 |

Across the five individual first-four-cycle origin windows, A02 beat the matched shuffle on Macro F1 in 5/5, Brier score in 5/5, and classification error in 5/5. It beat Sparse25 on Macro F1 in 4/5; origin 71 was the exception (−0.000823). A02 improved Brier score in all five, but its threshold and subgroup effects were not uniform. These are reused development windows, not untouched holdouts.

On full-six primary rows, A02 also improves ROC AUC (0.912054 vs 0.910761 baseline and 0.910818 shuffle), average precision (0.840892 vs 0.838412 and 0.835700), Brier score (0.113481 vs 0.114276 and 0.114260), and classification error (0.153675 vs 0.155178 and 0.155929). **Log loss gets worse:** native-epsilon log loss is 0.414164 vs 0.364051 baseline and 0.406963 shuffle. This is not a reversal of the Macro F1 result; it identifies a probability-extremity weakness.

The A02 arm has 4,006 exact 0/1 probability outputs among the **32,288 learned-present primary rows**. Of these, 62 are wrong (16 positive rows assigned 0 and 46 negative rows assigned 1). Under native-epsilon log loss, those 62 wrong endpoints account for a −0.061986 contribution to the learned-present mean improvement vs Sparse25, while the other 32,226 learned-present rows contribute +0.002094. The separate full-six native-epsilon log loss (0.414164 vs 0.364051 baseline and 0.406963 shuffle) is computed across all 38,588 rows, including unchanged W1 and missing-stream rows. This endpoint analysis is diagnostic; probabilities were not edited.

## Reliability versus recency

The predeclared maximum-cosine × Sparse25-confidence grid uses only the 32,288 learned-present primary rows and has 37/50 supported cells (support requires n ≥ 200 and at least 30 examples in each class). Brier improvement over Sparse25 is positive in 27/37 supported cells and negative in 10/37. It is strongest in mid-confidence bands: the n-weighted improvement over supported cells is +0.00235 for Sparse25 probabilities 0.2–0.3, +0.00358 for 0.3–0.4, +0.00319 for 0.4–0.5, and +0.00267 for 0.5–0.6. It turns negative for 0.8–0.9 (−0.00086) and 0.9–1.0 (−0.00105). The same broad taper is visible in the new-origin grid (24/50 supported cells, 17 positive and 7 negative) and in late windows (16/50, 10 positive and 6 negative). These are descriptive strata, not causal estimates.

The age × broad-similarity grid is less decisive: only 9/18 cells are supported in full-six primary, with 8 positive and 1 negative Brier contrasts; the supported signs vary across new-origin windows. Within broad similarity bands, age effects do not follow a consistent monotonic direction. This evidence is weaker for an age-weighting mechanism than for confidence-conditioned attenuation.

**Initial research judgment:** the confidence-by-similarity pattern appeared more promising than age weighting for a probability-quality diagnostic. Independent review noted that the proposed high-confidence taper should not change any 0.50-threshold labels given the frozen ±0.05 correction bound; the decision-boundary follow-up directly verifies this and decomposes the actual label changes. The taper is not a Macro F1/Kaggle improvement lane; retain it only as an optional probability-quality observation. No metric, gate, feature allowlist, or historical verdict changes.

## Visuals

- [Paired Brier change by maximum cosine and Sparse25 confidence](reliability_similarity_confidence.png): strongest gains sit mainly at mid confidence; high-confidence cells are weaker or negative. Unsupported cells display counts without a color effect.
- [Neighbor age within broad similarity bands](age_within_similarity.png): contrasts vary across similarity bands and do not support one consistent recency trend.

Both plots are descriptive, have no confidence intervals, and use saved development predictions.

## QA and budget

`QA.json` reports PASS: source/freeze/cache hashes and historical verifiers matched; all five observed/shuffle bundles and six C16 cache routes reconstructed; row/ID/cycle joins, OOF temporal boundaries, exact missing-stream fallback, group denominators, 36 F1 anchor reconstructions, confusion metrics, and additive-loss calculations passed. The fit-call guard observed 0 calls. The official ledger remains 66 starts / 64 saved / 2 historical failures, with the 66-fit cap consumed exactly. Row-level artifacts remain under the Git-ignored run root; tracked reports contain no raw text or record IDs.

Code checks also passed: `py_compile`, `pytest` (6 passed), and `git diff --check`.

## Caveats and handoff

- This audit reuses validation rows; H12-L overlaps primary. No independent holdout was created.
- F1 is non-additive across subgroups. Group direction counts are descriptive and have no confidence intervals.
- The full top-32 neighborhood was not recomputed; saved frozen verifier tie-order checks were used. See `QA.json` for the complete known-unknowns list.
- Historical C16 gates and subgroup verdicts remain as recorded in their original reports. This audit does not promote C16 or select a new champion.
- Kaggle status/public scores were not queried in this audit.

`NEXT_CARD.md` records the review disposition of the earlier proposal. It is not an active experiment card.
