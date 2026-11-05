# V1 Replay Superseded

The V1 freeze, results, and independent QA receipt are preserved as an audit
trail. Do not use this replay's operational full-row metrics for the four
matched-count policies.

Post-QA code review found that those four policies initialized predictions to
all-negative and then assigned top-K positives only among stream-present rows.
They therefore omitted the frozen `pA/CB >= 0.53230976129477` fallback on
stream-missing rows. Their full-row confusion counts, positive totals, and
full-row metrics are invalid. The first independent verifier repeated the same
omission, so its `PASS` did not detect this routing defect.

The defect does not affect the fixed-threshold policy results, present-only
metrics for any policy, present-only matched top-K membership, present-only
ROC AUC/AP, or ID/overlap calculations that explicitly filter to present rows.
Those quantities remain archived here but are superseded pending the corrected
replay.

No fit, Test access, E79 access, Kaggle access, or new candidate inference was
performed. The corrected replay will use a separate output namespace and will
retain the same frozen cached arrays, thresholds, views, and tie rules.
