# Regression Round2 — temporal generalization risk

Feature checks are descriptive and were run after all experiment recipes and interactions were selected. They did not select a recipe, modify preprocessing or consume official Test labels. The independent initial-submission track predicts Test using older frozen recipes; its scores are recorded separately.

## Observed supplied-feature differences

| Feature | Train missing | Test missing | Unknown Test / nonmissing Test | Outside Train numeric range |
|---|---:|---:|---:|---:|
| impact_load | 0.00% | 0.00% | n/a | 1 |
| entry_friction | 0.00% | 0.00% | n/a | 0 |
| cycle_code | 0.00% | 0.00% | n/a | 29421 |
| scope_pressure | 0.00% | 0.00% | n/a | 6284 |
| text_load | 0.00% | 0.00% | n/a | 178 |
| scope_trace | 8.36% | 8.27% | n/a | 5772 |
| org_spread | 8.61% | 8.39% | n/a | 5741 |
| module_spread | 8.22% | 8.50% | n/a | 5752 |
| telemetry_flag | 11.27% | 11.11% | n/a | n/a |
| org_code | 5.31% | 23.27% | 4732/22575 (20.96%) | n/a |
| module_code | 7.99% | 25.14% | 8099/22024 (36.77%) | n/a |
| pattern_code | 16.17% | 8.61% | 189/26889 (0.70%) | n/a |

Train cycle **47–79**, Test **80–91**: all **29,421** Test rows have cycles above the Train range. Numeric tree splits can distinguish observed cycle regions, but do not imply a learned linear trend beyond the training range. Removing cycle was experimentally tested using internal windows and did not dominate full R1-D. We do not select the no-cycle recipe from the observed Test range.

## A coherent novel scope-support cohort

- train: any scope/spread count zero **0 / 50,608**; all three zero **0**; pressure below Train minimum **0**; joint any-zero and low-pressure **0**.
- test: any scope/spread count zero **6,278 / 29,421**; all three zero **4,835**; pressure below Train minimum **6,283**; joint any-zero and low-pressure **6,278**.

The three count fields are scope_trace, org_spread and module_spread. All **6,278 any-zero Test rows** also have scope_pressure below the Train minimum; 4,835 have all three count fields zero. No Train count is zero. The parsers accept zero as a valid number and report no malformed nonmissing tokens. Low support is not automatically missingness, and should not be silently imputed as the training median. The existing parsers are unchanged.

Scope pressure has descriptive KS distance **0.213757**; scope_trace **0.213844** among finite parsed values. Scope_pressure spans 68.4148–756.3290 in Train versus -0.8996–779.5182 in Test. KS is a feature-distribution distance, not a Test error or causal drift estimate.

This joint pattern is an observed out-of-support cohort. Genuine low scope, different collection/representation conventions, and another source-process change remain competing explanations; the dataset does not establish which one. No source mechanism, corruption label or future target behavior is inferred. In internal validation, removing the scope/spread group raised primary MAE by about 0.04156 and H12 by0.03888; that result does not guarantee its future contribution under novel support.

## Observed history/horizon behavior

- Full recent-8 gives the lowest primary MAE13.217154, but pooled H12 rises to15.794561 versus15.577240 expanding. More recent training is not universally safer for a 12-cycle forecast.
- Removing categories gives the lowest pooled H12 MAE15.053467, but late-H12 and high-tail worsen. The early gain recurs across expanding/recent/matched histories; this is not a universal unknown-category solution.
- The no-cycle recent-8 recipe moderates the primary cost of removing cycle and approximately matches expanding full D on pooled H12, while still trailing the full recent model on primary.
- Known/unknown category cohorts change error ordering between primary and H12. Similar cohorts remain when no category predictor is used, so novelty is not a proven encoder-failure mechanism.

## What remains unknown

There are no Test labels in the research analysis. We cannot establish Test MAE by cohort, stationarity of conditional targets, timestamp-certified feature availability, target-generation mechanism, or the cause of new numeric support. H12 shares validation records with primary, has only two history/era views, and all windows have already informed development decisions. Matched-size rows still differ in category mix/support and temporal dependence. No p-values or independent-replication claims are made.

## Implications for review

Retain the original structured controls and the two contrasting candidates. A future Round3 should preregister ranking/conditional-spread diagnostics and one modeling hypothesis with explicit EF/tail/H12 guardrails. Any text experiment needs its own train-only vocabulary and missing-stream checks. The novel zero-scope cohort motivates a source-semantics question and future cohort monitoring, not transductive tuning. Kaggle observations remain a separate external benchmark and do not authorize additional submissions or model changes.
