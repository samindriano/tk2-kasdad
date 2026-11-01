# C15 — prequential two-score meta-head pilot

**Decision: RETIRE (valid evidence; independent QA passed).** C15 fits a frozen logistic meta-head on out-of-fold scores from C2_B and the sparse-binary expert, and applies it only to present-stream rows. The missing-stream CatBoost route and 0.50 threshold remain unchanged. Early H12_E uses the exact frozen-route fallback because no causal sparse score exists at cutoff 55.

## Matched evaluation

Controls are the accepted `v3_cls_sparse25` route recomputed on the same late rows. W4 and W6 are summed before computing the primary Macro F1. The candidate misses the required +0.0005 primary improvement and harms positive recall in the supported high-text-load tail; the H12_L improvement does not override those failures.

| View | Rows | Control Macro F1 | Candidate Macro F1 | Gain |
|---|---:|---:|---:|---:|
| W4 + W6 primary | 13,441 | 0.828147 | 0.826808 | −0.001339 |
| H12_L | 20,903 | 0.808961 | 0.812243 | +0.003281 |
| W4 | 4,950 | 0.841418 | 0.840136 | −0.001282 |
| W6 | 8,491 | 0.816599 | 0.815125 | −0.001474 |
| H12_E fallback | 17,685 | 0.795032 | 0.795032 | 0.000000 |

In `text_load_q4_high_tail`, positive-class recall fell by 0.024653 on W4 (n=1,662; 649 positives) and 0.021440 on the primary (n=3,621; 1,306 positives), exceeding the allowed 0.02 harm in both views. Primary subgroup Macro F1 also fell 0.007954. The gate therefore fails both the primary-gain and supported-subgroup checks.

## Execution and scope

Two predictor fits were saved at cutoffs 67 and 75; cutoff 55 is an exact fallback (zero fit). One earlier shared-runner dispatch failed before any predictor fit and was recorded; its two-fit reservation was reused. Earlier prefit revisions and defects remain preserved under the experiment directory; no candidate score was computed until the active revision 05 freeze. Independent verification rebuilt metrics from saved predictions and labels, and the shared runner verifier passed. No Test, SampleSubmission, or Kaggle data was accessed. The training ledger records 62/180 reserved pilot predictor fits after C15.

Evidence: [results](results.json), [gate](gate.json), [independent verification](verification.json), [machine summary](summary.json).
