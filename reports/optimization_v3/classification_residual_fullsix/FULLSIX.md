# V3 Classification — OOF residual Head A, six-window continuation

## Frozen scope

Only Head A (C2_B probability plus expanded structured covariates) was continued. Cutoffs 59, 63, 71, and 75 are four newly fitted heads; already verified cutoffs 55 and 67 are reused byte-for-byte. Head B was not fitted again. The comparison is against the current C2_B-present / CB_MISSING-missing champion and the separately verified fixed sparse25 challenger. Threshold and head shrinkage remain 0.50; no weight, route, threshold, or feature selection used these outer labels.

The correction heads train only on stream-present OOF rows from cycles 52 through each cutoff, using exact nonoverlapping four-cycle blocks scored by a base fit that ends before those rows. The missing-stream branch keeps the cached CatBoost prediction exactly. Official Test and Kaggle were not read.

## Primary and fixed-origin comparison

| View | Rows | Champion F1 | Residual A F1 (Δ) | Sparse25 F1 (Δ) | Residual A Brier | Sparse25 Brier |
|---|---:|---:|---:|---:|---:|---:|
| W1 | 4114 | 0.802090 | 0.802226 (+0.000136) | 0.803160 (+0.001070) | 0.135544 | 0.132820 |
| W2 | 6493 | 0.849458 | 0.842963 (-0.006495) | 0.851618 (+0.002160) | 0.111242 | 0.106916 |
| W3 | 7078 | 0.828527 | 0.826656 (-0.001870) | 0.832515 (+0.003989) | 0.118959 | 0.116409 |
| W4 | 4950 | 0.840297 | 0.840099 (-0.000198) | 0.841418 (+0.001120) | 0.112946 | 0.113077 |
| W5 | 7462 | 0.812698 | 0.814096 (+0.001398) | 0.814497 (+0.001799) | 0.115825 | 0.115885 |
| W6 | 8491 | 0.813426 | 0.809562 (-0.003864) | 0.816599 (+0.003174) | 0.110632 | 0.108428 |
| primary_six_window_pooled | 38588 | 0.827646 | 0.826217 (-0.001430) | 0.829944 (+0.002297) | 0.116219 | 0.114276 |
| H12_E | 17685 | 0.792671 | 0.797176 (+0.004505) | 0.795032 (+0.002361) | 0.135260 | 0.133411 |
| H12_L | 20903 | 0.803724 | 0.806477 (+0.002753) | 0.808961 (+0.005237) | 0.121428 | 0.120838 |
| H12_fixed_origin_pooled | 38588 | 0.798984 | 0.802740 (+0.003756) | 0.802966 (+0.003982) | 0.127767 | 0.126600 |

## Error movement and subgroup robustness

Each view reports per-class precision-relevant counts, per-cycle metrics, stream-present/missing metrics, fit-only entry_friction and text_load quartiles, and paired error transitions. Quartile edges are computed separately at each prediction cutoff using feature values from cycles 47 through that cutoff; labels are not used to define subgroup boundaries.

In the pooled primary view, residual A fixes 333 champion errors and breaks 382 correct champion predictions. Sparse25 fixes 384 and breaks 283. Residual A and sparse25 disagree on 892 pooled-primary predictions.

H12 uses fixed pilot origins 55 (cycles 56–67) and 67 (cycles 68–79); it overlaps the six-window validation chronology. It is a stability view, not an independent holdout. Primary pooled coverage is 38,588; H12 is 38,588.

## Interpretation limits

These temporal validation outcomes are development evidence and overlap prior campaign work. A six-window or H12 gain does not establish generalization to the official Test. Subgroup contrasts are descriptive and can have small support. The residual head is a prespecified fixed recipe, not a tuned final model. No final route or model replacement is selected in this report.

## Kuartil entry_friction dengan ties

`entry_friction` memiliki dua nilai pada data fit. Pada semua cutoff, raw quartile edges adalah `[0, 0, 1]`; aturan `digitize` yang sama mempertahankan observasi, sementara kuartil dengan batas sama dapat kosong. Sel kosong tercatat sebagai `n=0` pada JSON; tidak ada cutpoint yang digeser atau subgroup yang direkayasa.
