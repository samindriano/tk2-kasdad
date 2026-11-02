# A02 continuation — result checkpoint

**Scientific gates:** PASS
**Risk review:** no hold flag
**Conditional final eligible:** True

The three continuation origins (59, 63, 71) produce 21,033 previously used development rows. They add new origin-specific predictions, not an untouched holdout. The 38,588-row full-six view includes the exact Sparse25 fallback through the C16 cache bridge at W1 and the already verified A02 pilot predictions at W4/W6.

| View | Rows | Sparse25 (C16 cache bridge) Macro F1 | A02 observed Macro F1 | Δ vs Sparse25 | Δ vs matched shuffle |
|---|---:|---:|---:|---:|---:|
| W1 | 4,114 | 0.803160 | 0.803160 | +0.000000 | +0.000000 |
| W2 | 6,493 | 0.851618 | 0.854591 | +0.002972 | +0.002715 |
| W3 | 7,078 | 0.832515 | 0.836849 | +0.004333 | +0.003922 |
| W4 | 4,950 | 0.841418 | 0.844405 | +0.002988 | +0.003314 |
| W5 | 7,462 | 0.814497 | 0.813673 | -0.000823 | +0.003025 |
| W6 | 8,491 | 0.816599 | 0.819604 | +0.003005 | +0.004041 |
| new_origin_aggregate | 21,033 | 0.835171 | 0.837187 | +0.002016 | +0.002919 |
| full_six_primary | 38,588 | 0.829944 | 0.832045 | +0.002101 | +0.002767 |
| late | 15,953 | 0.815790 | 0.816958 | +0.001168 | +0.003534 |
| inherited_H12_L | 20,903 | 0.808961 | 0.811159 | +0.002198 | +0.001386 |

H12-L is inherited pilot evidence and reuses development rows. Full per-cycle, subgroup, confusion, precision/recall, probability, and endpoint diagnostics are in `results.json`, `per_cycle.csv`, and `groups.csv`.

The endpoint log-loss diagnostic reports manual clipping at 1e-15 separately from scikit-learn's default dtype epsilon. Predictions were not clipped or recalibrated for analysis. The route remains a predictive association; it is not causal.

The continuation evaluation did not access Test or SampleSubmission. After all frozen gates passed with no risk-review hold, the authorized conditional origin-79 full-Train fit and Test inference were completed; see `FINAL_HANDOFF.md` for lineage and boundaries.

## Submission status — user-reported manual upload

Samuel later pasted a Kaggle submissions-page screenshot in the Sol chat showing `a02_continuation.csv` as **Complete** with public Macro F1 **0.82930**. The screenshot showed account Samuel Indriano and relative time “now”; it did not show a submission message, exact timestamp, submission reference, or server-side file hash. This is recorded as user-reported evidence only, not live/API verification.

The previously reported Sparse25 public Macro F1 is 0.82908, so A02 is the **best observed public result among the reported entries**, by +0.00022. This small leaderboard difference is not a private-score or generalization guarantee and does not automatically select the final academic champion. The local CSV hash was rechecked against the unchanged preparation manifest. The manifest intentionally remains a historical `PREPARED_VALIDATED_NOT_SUBMITTED` snapshot; the later upload receipt supersedes that status. The single A02 upload is consumed; do not upload it again. See `submission_receipts.jsonl` for the receipt and unknown fields.
