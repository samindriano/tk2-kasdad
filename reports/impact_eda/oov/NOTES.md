# Fit-vocabulary coverage: raw target and outer-error EDA

**Question.** Pada panjang laporan dan cycle sebanding, apakah rendahnya cakupan kosakata TF-IDF historis berkaitan dengan heterogenitas threat_rank dan error model?

## Definition and provenance

Coverage is the fraction of TF-IDF analyzer unigram/bigram occurrences that appear in the model's fit-only feature vocabulary. OOV terms include terms filtered by `min_df` or the `max_features` cap; this is a model-vocabulary measure, not semantic novelty. The exact fitted vocabulary and IDF were rebuilt from each fold's training `signal_stream` and both hashes matched the frozen Stage 5 fit audit. The saved outer predictions, row positions, rank weights and Stage 5 job hashes were reconciled before errors were calculated. Only official Regression Train was used (SHA-256 `756523ff8156a9cd776e52249f7e2a8f23dcec432b8d10570e55ff6db0e0f082`); no Test data, submission score or new model fits were used.

## Keep the two questions distinct

The first table summarizes the observed `threat_rank` distribution by fixed coverage bands. A lower median or narrower/wider IQR is a raw-data association. It does not say vocabulary coverage causes the target distribution. The second table compares historical frozen outer-model error over the same groups; it is retrospective model-error evidence, not an EDA available before model review. Cycles 56–67 are the discovery narrative and 68–79 are a sensitivity period. All cycles have appeared in earlier development and the later period is not a holdout.

For the length/cycle comparison, each cycle was divided into four equal-count analyzer-token-length strata. Within each stratum, the lower and upper OOV quartiles were compared. A contrast is retained only if each extreme-OOV group has at least 20 rows; tail rates require at least 5 examples per comparison group. `stats.json` gives each matched cell, its denominator, eligible-row coverage, target IQR/tail counts, and error MAE/within-cell percentile contrasts. Any pooled number is the support-weighted average of these cells; inspect the supported-cell counts and coverage beside it.

## Observed pattern

In cycles 56–67, 48/48 matched cycle × length cells meet the minimum (coverage 100.0% of defined present-stream rows). The high-OOV group has a weighted within-cell target-IQR difference of +7.06 target points (36/48 cells positive), and a +2.10 error-MAE difference (34/48 positive). Its mean absolute-error percentile is +6.21 points higher (35/48 positive). This supports testing a coverage scalar, not an assumed performance gain.

In cycles 68–79, corresponding differences are +4.14 target-IQR points (29/48 positive), +2.26 absolute-error MAE (41/48 positive), and +6.69 within-cell error-percentile points (40/48 positive). This is a useful sensitivity pattern, but later cycles have also featured repeatedly in development.

Tail evidence is sparse after length/cycle matching: at least five examples in each low/high OOV group for both target tails occur in only 1/48 discovery cells and 5/48 sensitivity cells. Do not infer a tail-specific mechanism from these matched cells. The fixed low-coverage band (coverage ≤0.25) contains just 11 discovery rows and 0 sensitivity rows, so its displayed raw bin is descriptive only.

The figure is a compact unadjusted descriptive view. The matched contrasts, not a visual slope across all coverage bands, address whether target or error differences persist within cycle and comparable text length. Residual analysis still uses the same outer validation labels that informed earlier experiments, so adaptive selection bias remains.

## Interpretation boundary

This analysis only checks whether the planned scalar is worth its single controlled ablation. It does not establish causal effect, generalization to cycle 80–91, meaning of anonymized text, or an untouched test result. It does not investigate EF, impact, scope, telemetry, module or organization spread. No target label enters the vocabulary, analyzer, coverage calculation, or length strata.
