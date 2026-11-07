# CLS_DAN_V1 evaluation recovery R2

One additional exact query-label vector read was used after the R2 markers were committed. The original consumed R1 attempt and failure receipt remain unchanged.

## Main metrics

| View | Rows | Champion Macro F1 | DAN Macro F1 | DAN gain | LINEAR Macro F1 | LINEAR gain | DAN − LINEAR |
|---|---:|---:|---:|---:|---:|---:|---:|
| PRIMARY_UNIQUE | 20903 | 0.821941 | 0.815061 | -0.006879 | 0.819439 | -0.002502 | -0.004378 |
| E67_W4 | 4950 | 0.844337 | 0.840953 | -0.003385 | 0.842355 | -0.001982 | -0.001403 |
| E67_W5 | 7462 | 0.806246 | 0.800447 | -0.005799 | 0.802456 | -0.003790 | -0.002009 |
| E67_W6 | 8491 | 0.792507 | 0.784619 | -0.007887 | 0.789946 | -0.002560 | -0.005327 |
| E67_H12 | 20903 | 0.811327 | 0.805545 | -0.005783 | 0.808568 | -0.002759 | -0.003024 |
| E75_W6 | 8491 | 0.818884 | 0.808369 | -0.010515 | 0.817239 | -0.001645 | -0.008871 |

## Gate verdicts

- **exploratory_eligibility_original_rule: FAIL**
- **mechanism_contrast: FAIL**
- **scientific_original_frozen: FAIL**
- **scientific_r2_amended: FAIL**

The original frozen scientific verdict preserves E75 LINEAR-minus-champion >= 0. The separately reported R2-amended scientific verdict uses E75 DAN-minus-LINEAR >= 0 under the later root adjudication; it does not replace the original verdict.

The exploratory eligibility rule is unchanged and remains the operative criterion for any E79 review. Neither scientific verdict authorizes more fitting, Test inference, or Kaggle action.

## Limits

E67/E75 temporal views overlap; E75 training includes cycles 68-75. The primary unique view uses E67 predictions for cycles 68-75 and E75 for cycles 76-79. The comparison does not isolate a causal nonlinear effect. Missing-stream rows use exact champion fallback.

## Read accounting and QA

The first R2 `evaluate` invocation stopped in the pre-reader intent guard: the target reader was not entered, no target values were read, and no metrics were computed. The committed pre-reader amendment records this. After amendment, one target-free production validator check passed; the successful scorer then read the exact frozen 20,903-row/ID vector once. Campaign accounting is two actual query-vector reads total: the original consumed R1 read plus this R2 read. R2 made one technical pre-reader failure and one successful scoring invocation. No labels were persisted.

The independent sklearn verifier passed all 6 frozen views. Row counts, unique IDs, probability ranges, exact missing fallback, expanded missing coverage of zero, and protected subgroup support checks passed. New fits: 0; prediction/model changes: 0; Test/SampleSubmission/Kaggle access: none.

### Supported subgroup diagnostics

| View | Supported cells | Unsupported cells | Worst DAN Macro F1 harm | Worst DAN positive-recall harm |
|---|---:|---:|---:|---:|
| PRIMARY_UNIQUE | 8 | 0 | +0.0119 | +0.0183 |
| E67_W4 | 8 | 0 | +0.0164 | +0.0148 |
| E67_W5 | 8 | 0 | +0.0081 | +0.0331 |
| E67_W6 | 8 | 0 | +0.0141 | +0.0361 |
| E67_H12 | 8 | 0 | +0.0099 | +0.0231 |
| E75_W6 | 8 | 0 | +0.0155 | +0.0232 |

All observed subgroup cells in the six real views met minimum support (200 rows and 30 per class); the synthetic test separately covered unsupported-cell serialization. Exploratory subgroup tolerance failed in E67 W5 and W6: W5 fit-only text-load quartile 3 had +0.0331 recall harm; W6 present-stream rows had +0.0262, entry-friction group 0 had +0.0281, fit-only quartile 1 +0.0308, and quartile 3 +0.0361. E67 W6 entry-friction group 0 also exceeded the scientific F1-harm threshold (+0.0121).

### Direction by cycle

| Origin | Cycles | DAN gain positive | Negative | Ties |
|---|---:|---:|---:|---:|
| E67 | 12 | 2 | 10 | 0 |
| E75 | 4 | 0 | 4 | 0 |

### Gate interpretation

Primary DAN gain was -0.006879; E67 H12 -0.005783; E75 W6 -0.010515; primary DAN-minus-LINEAR -0.004378. These all failed their frozen thresholds. The original frozen E75 LINEAR gain was -0.001645; the separate R2-amended E75 DAN-minus-LINEAR contrast was -0.008871; both were negative. The unchanged exploratory rule failed its performance and subgroup components, so CLS_DAN_V1 is not eligible for an E79 review under the campaign rule.

## Recommended next step

Keep the four saved fits and all historical receipts unchanged. Do not promote CLS_DAN_V1 or start E79; return these negative results to the root for review. Any different classifier mechanism requires separate authorization and a new frozen protocol.
