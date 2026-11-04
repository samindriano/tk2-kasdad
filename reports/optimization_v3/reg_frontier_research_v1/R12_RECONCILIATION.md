# R12 present-only reconciliation — zero-fit review

Date: 2026-11-04
Scope: existing frozen R12 pilot and documented present-only policy
Decision: **AGGREGATE SIGNAL REMAINS; PRESENT-ROW COMPARISON UNKNOWN; NO-GO FOR CONTINUATION**

## Summary

The post-hoc R12 present-only policy's published aggregate gains do not disappear when compared with the current R06 headline gains on the same declared STANDARD views. Because both reports use the same legacy champion as their baseline, subtracting the rounded gains gives an implied whole-view R12-policy margin over R06 of about **0.223696 MAE on primary** and **0.202910 on H12**.

This does not establish incremental signal on present rows specifically. The present-only policy falls back to the legacy champion for missing-stream rows, while current R06 uses its own prediction there. No ID-aligned R12 and R06 prediction arrays are available in the pinned run inventory, so present-only row, cycle, and supported-group comparisons against current R06 remain UNKNOWN. The original R12 gate failure remains RETIRE; the post-hoc policy does not retroactively pass it.

## Frozen recipe and target boundary

R12's frozen signature is dbc69182a35d02032763d463c630c49ff52b2d803ac487202aefd465b25d78d3. The freeze and recipe hashes match between the active worktree and the pinned artifact root at HEAD bcf8b8a6ddd22e755bd21f8c34b30fadda604407. The saved R12 verification is PASS and records validation labels were not passed to the adapter.

The frozen recipe declares one stable contiguous training query per exact fit cycle; labels and sparse rows are sorted in the same cycle order. Fit cutoff 55 uses cycles 47–55 and scores complete cycles 56–67; cutoff 67 uses cycles 47–67 and scores complete cycles 68–79. Training grades are clip(floor(threat_rank / 10), 0, 9), formed from the fit prefix. Candidate scores are converted to 100 × average predicted-score rank / complete-cycle size before inverse-mapping to original rows. The mapping uses predictions across a complete future cycle and does not need future target labels; operationally it requires the full cycle's score rows to be available together. This checks the frozen protocol and stored verifier; the runner source was not replayed in this zero-fit review.

The original fixed policy is clip(0.75 × exact three-component champion + 0.25 × whole-cycle R12 percentile, 0, 100). The later present-only diagnostic retains that full-cycle ranking step, then uses the R12 blend for stream-present rows and the exact champion for missing-stream rows. Evidence: reports/optimization_v3/batch1/R12/recipe.json keys query_definition, group_order, target_grade, score_inverse_mapping, evaluation_views and policy; reports/optimization_v3/batch1/R12/verification.json; reports/optimization_v3/postscreen_review/REVIEW.md:48,55–63.

## Baseline accounting

All figures below are previously reported aggregate gains against the same exact legacy champion on the declared matching views. The R12 present-only values are post-hoc rounded report values; the R06 values are the frozen R06 pilot summary. The final column is arithmetic on those reported gains, not a row-level recomputation.

| View | Rows / origins | R12 present-only gain vs legacy champion | Current R06 gain vs same champion | Implied R12 policy gain vs current R06 |
|---|---|---:|---:|---:|
| Primary | 9,064; cutoff 55 W1 + cutoff 67 W4 | +0.285422 | +0.061726 | ≈ +0.223696 MAE |
| H12 | 38,588; cutoff 55 cycles 56–67 + cutoff 67 cycles 68–79 | +0.306170 | +0.103260 | ≈ +0.202910 MAE |

R12's original policy reported primary MAE 13.218595 versus champion 13.464396 and H12 MAE 13.763042 versus 14.047533. Current R06 reported 13.402670 primary and 13.944273 H12 on the same declared view sizes. Evidence: reports/optimization_v3/batch1/R12/summary.md; reports/optimization_v3/batch1/R06/summary.md:5–17; reports/optimization_v3/postscreen_review/REVIEW.md:52–59.

The difference in the table is a whole-view policy comparison. It does not isolate the present subset: the R12 policy uses champion fallback on missing, whereas R06 does not. Exact ID-level alignment and matched present-only MAE versus R06 are UNKNOWN because the aligned arrays are absent.

## Original missing guard and post-hoc selection

The original R12 pilot failed its supported missing-stream guards: MAE harm was +0.768759 on W1 (n=392), +0.489276 on primary (n=734), and +0.324499 on H12 (n=2,578), all above the +0.25 limit. These remain the frozen R12 verdict. Evidence: reports/optimization_v3/postscreen_review/REVIEW.md:46–48; reports/optimization_v3/postscreen_review/CARD_REVIEW.md:47.

The later present-only calculation chose its fallback after the original pilot results were visible. Missing-stream harm versus the legacy champion becomes zero by construction. The review reported no tail or missing harm versus that legacy comparator on the checked views, but those claims do not establish present-row safety versus current R06. That comparison and its per-cycle/support-group harms are UNKNOWN. This policy is post-hoc evidence, not an independent validation result or a retroactive gate pass. Evidence: reports/optimization_v3/postscreen_review/REVIEW.md:53–63.

## Saved-artifact inventory

The R12 freeze signature, recipe hash E76861B798A212F7B1B2BF70764DB49C67A957ADFA345D7173AF203DF939069D, and verification hash 72BC362D4520977AB0A80AAE517C56C26D9F5EB60657FE95F3491999EC2EF78E match between the active worktree and the pinned artifact root. R06 summary and verification hashes also match. No R12 run directory exists under experiments/runs in that pinned root, and the R12-named inventory contains only the original freeze and pilot report files; no R12_PRESENT_ROUTE artifact or R12 continuation bundle for cutoffs 59, 63, 71, 75, or 79 was found.

The pinned root has an E79 predictions bundle for the separate reg_position_real_e79_benchmark_v1 position benchmark: predictions.npz SHA-256 AA81A7D17CC0DFE9DA1DE35235DE87DCE4B89D4F4B34FCB34ED921478B10C4E0; its INFERENCE_MANIFEST.json SHA-256 is 15FE5C6446B7CA2EB3B24FE72E9B4A85C51A6C4847495B192AF7E76143474802. These hashes identify that separate position artifact, not an R12 continuation. The inventory did not open bundle contents.

## Decision and smallest next action

**NO-GO for a new R12 fit from this reconciliation.** The aggregate policy comparison is promising, but the exact present-row and group comparison to current R06 cannot be established from aggregate summaries. No predictor fit, prediction call, learned preprocessing fit, or row-level/per-cycle metric recomputation was performed; table differences above are arithmetic on rounded published gains.

If the exact archived R12 and R06 aligned outputs are later located, the smallest next step is a zero-fit recomputation of the already documented fixed present-only policy on the identical rows, with the whole-cycle rank computed before masking and the same saved development labels. Report per-origin/per-cycle and supported-group metrics against both the legacy champion and current R06. Stop if row identity or origin matching fails, or if the present-only policy does not retain its margin under those matched comparisons.

Only if that zero-fit reconciliation verifies signal and all frozen guards pass would a separately frozen continuation be worth proposing. The existing R12 review's minimal continuation outline uses four predictor fits at cutoffs 59, 63, 71, and 75; count preprocessing separately as four word-TFIDF fits and four structured preprocessing fits. This is a cost estimate, not authorization. No E79 fit, Test/Sample/Kaggle access, upload, dependency probe, or ledger change occurred in this review.
