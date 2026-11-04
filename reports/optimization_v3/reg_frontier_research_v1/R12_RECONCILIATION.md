# R12 present-only reconciliation — zero-fit review

Date: 2026-11-04
Scope: existing frozen R12 pilot and documented present-only policy
Decision: **AGGREGATE SIGNAL REMAINS; PRESENT-ROW COMPARISON UNKNOWN; NO-GO FOR CONTINUATION**

> **Historical snapshot notice (2026-11-04):** The initial review below checked a narrower sibling artifact root and incorrectly recorded the aligned R12/R06 arrays as absent. The amendment at the end of this file corrects that inventory and supersedes the initial UNKNOWN conclusions about present-row, cycle, and supported-group comparisons. The original review text is retained as an audit trail.

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

## Amendment — corrected inventory and matched zero-fit reconciliation

Date: 2026-11-04

Status: **matched present-row signal verified; continuation remains unapproved**

### Inventory correction

The earlier “arrays absent” statement was based on the sibling artifact root and was incomplete. The authoritative MAIN checkout is `C:\Users\Sam\OneDrive\Documents\KASDAD\TP 2`, at the time of this check on `main` / HEAD `253f4d949e537d4a53c7ced094d62acfaf23b043`. Its saved aligned bundles are under `experiments/runs/optimization_v3_batch1/R12` and `.../R06`. No files in that MAIN checkout were changed.

SHA-256 verification was performed before loading the NPZ arrays; each JSON sidecar's `npz_sha256` matches its paired NPZ. The official Train source hash matches the frozen source hash.

| Artifact | Size (bytes) | SHA-256 |
|---|---:|---|
| `data/raw/regression/Train.csv` | 24,540,489 | `756523FF8156A9CD776E52249F7E2A8F23DCEC432B8D10570E55FF6DB0E0F082` |
| `R12/R12_fit55.json` | 237,988 | `1E500421498DA10B668AAD66C30FE94BC7D4F71251FB093D6A3AFCBFCA232E7D` |
| `R12/R12_fit55.npz` | 1,272,265 | `561CF2F2FC53FCB3D02440E345D002145C20154E690B90D6CA812C11D008CDC5` |
| `R12/R12_fit67.json` | 279,284 | `2C76981096929D4BD7DE347A8966DAFD2941B26A524CF5EBFB94F33D12E74436` |
| `R12/R12_fit67.npz` | 1,497,611 | `FA43A9E09B22698F0120D25CC7499E30E3269EAD1D78241B59D21775E1BEA119` |
| `R06/R06_fit55.json` | 236,186 | `68E60624D79957075C755AC1C35BD40733D8B983CD2EC931B8186243DE0100B0` |
| `R06/R06_fit55.npz` | 1,173,693 | `ED52CF9C8958B693E49E4C2999015BABE35AA8F350FEA28EB313E492F19E5DD6` |
| `R06/R06_fit67.json` | 277,172 | `FAA11F25BFD37FDED8DB5F84244CE306D6F2027DC153FB143560304FCF91DA1E` |
| `R06/R06_fit67.npz` | 1,387,095 | `4BDA0AC6838388EA46CA768E0998F7B0F9A9A1B6DEFB2B1E5399F6FF83154804` |

The R12 freeze signature remains `dbc69182a35d02032763d463c630c49ff52b2d803ac487202aefd465b25d78d3`; the R06 freeze revision signature is `02ae1b4e6e6424a8ac7f66fd0f1b955e478f6265c4f071989d9d8c5ef532c0a8`.

### Alignment and saved-prediction reconstruction

All four NPZ files were loaded with `allow_pickle=False`. Saved IDs are Unicode (`<U16`), so no object-array deserialization was needed. At each cutoff:

- R12 and R06 `score_rows` and `score_record_ids` match exactly; row IDs and cycle codes match MAIN Train at those positions. The saved R12 and R06 legacy `baseline_prediction` arrays also match exactly.
- There are 17,685 aligned validation rows at cutoff 55 and 20,903 at cutoff 67, for 38,588 unique IDs across the combined H12 view. The four-cycle W1 (cycles 56–59) contains 4,114 rows; W4 (cycles 68–71) contains 4,950.
- R12's saved `audit__rank_percentile`, recomputed from `raw_candidate` over each complete cycle, matches exactly (maximum absolute difference 0). Rebuilding the saved R12 prediction from its exact legacy baseline and full-cycle percentile also matches exactly.
- R06's clipped rank50 intermediate and final saved prediction reconstruct exactly (maximum absolute difference 0), including the frozen clip on the rank50 intermediate before the final blend.
- Sidecars identify fit cutoffs 55/67, fit sizes 12,020/29,705, score sizes 17,685/20,903, one stable exact-cycle query per fit cycle, and `validation_data_or_labels_passed_to_fit=False`. The reports' verification status is PASS. No model was loaded or refit and no new prediction call was made.

The only Train fields read for these rows were `record_id`, `cycle_code`, `threat_rank`, and `signal_stream`; the raw values were not printed. Present means non-null `signal_stream` with non-empty stripped text; null, empty, or whitespace-only values are missing.

### Fixed route and matched comparison

This is a zero-fit replay of the already documented post-hoc policy: retain the saved R12 blend on present rows and use the exact saved legacy champion on missing rows. R12's percentile is still computed over each entire cycle before the availability mask. Current R06 is its exact saved prediction. Gain below means `MAE(R06) - MAE(fixed R12 route)`; positive favors the R12 route. The target is the saved Train `threat_rank` on the same row IDs.

| View | n | Fixed route MAE | R06 MAE | Legacy MAE | Gain vs R06 |
|---|---:|---:|---:|---:|---:|
| W1 (cycles 56–59) | 4,114 | 15.290446 | 15.816833 | 15.882112 | +0.526387 |
| W4 (cycles 68–71) | 4,950 | 11.424105 | 11.396232 | 11.455005 | −0.027873 |
| Primary (W1 + W4) | 9,064 | 13.178974 | 13.402670 | 13.464396 | +0.223696 |
| E55_H12 (cycles 56–67) | 17,685 | 16.393490 | 16.798196 | 16.926611 | +0.404706 |
| E67_H12 (cycles 68–79) | 20,903 | 11.497528 | 11.529709 | 11.611686 | +0.032180 |
| H12 | 38,588 | 13.741363 | 13.944273 | 14.047533 | +0.202910 |

The W1/W4 and combined views are matched on the same row positions for both routes. W1 and Primary improve against R06, while W4 has a small net loss.

| View and availability | n | Fixed route MAE | R06 MAE | Legacy MAE | Gain vs R06 | Original R12 missing harm vs legacy |
|---|---:|---:|---:|---:|---:|---:|
| W1 present | 3,722 | 15.443915 | 15.984081 | 16.097895 | +0.540166 | — |
| W1 missing | 392 | 13.833276 | 14.228832 | 13.833276 | +0.395556 | +0.768759 |
| W4 present | 4,608 | 11.243322 | 11.219893 | 11.276515 | −0.023429 | — |
| W4 missing | 342 | 13.859922 | 13.772165 | 13.859922 | −0.087757 | +0.168932 |
| Primary present | 8,330 | 13.120226 | 13.348621 | 13.430798 | +0.228396 | — |
| Primary missing | 734 | 13.845691 | 14.016053 | 13.845691 | +0.170361 | +0.489276 |
| H12 present | 36,010 | 13.699570 | 13.905967 | 14.027659 | +0.206397 | — |
| H12 missing | 2,578 | 14.325132 | 14.479334 | 14.325132 | +0.154202 | +0.324499 |

The fixed route retains aggregate present-row gain versus current R06 on Primary and H12. On the missing subsets, the fallback is legacy exactly by construction; its positive margins over R06 are measured comparisons, not a newly trained missing expert.

### Per-cycle results

| Cycle | n | Present n | Fixed route gain vs R06, whole cycle | R12 gain vs R06, present rows |
|---:|---:|---:|---:|---:|
| 56 | 564 | 479 | +0.280127 | +0.310296 |
| 57 | 962 | 864 | +0.587939 | +0.608610 |
| 58 | 1,205 | 1,105 | +0.577394 | +0.589256 |
| 59 | 1,383 | 1,274 | +0.539557 | +0.537597 |
| 60 | 1,478 | 1,393 | +0.368896 | +0.384082 |
| 61 | 1,632 | 1,541 | +0.435008 | +0.445824 |
| 62 | 1,500 | 1,412 | +0.507194 | +0.546162 |
| 63 | 1,883 | 1,778 | +0.055083 | +0.063460 |
| 64 | 1,807 | 1,698 | +0.477480 | +0.478195 |
| 65 | 1,567 | 1,480 | +0.288969 | +0.276299 |
| 66 | 1,751 | 1,631 | +0.666610 | +0.707981 |
| 67 | 1,953 | 1,827 | +0.199251 | +0.202501 |
| 68 | 577 | 525 | −0.243765 | −0.255455 |
| 69 | 1,079 | 997 | −0.045307 | −0.049911 |
| 70 | 1,605 | 1,511 | +0.173809 | +0.192319 |
| 71 | 1,689 | 1,575 | −0.134635 | −0.136305 |
| 72 | 1,811 | 1,687 | −0.000193 | −0.005021 |
| 73 | 1,880 | 1,738 | +0.037143 | +0.050143 |
| 74 | 1,814 | 1,679 | −0.070435 | −0.089503 |
| 75 | 1,957 | 1,851 | +0.031290 | +0.031529 |
| 76 | 1,822 | 1,704 | +0.015063 | −0.002866 |
| 77 | 2,335 | 2,163 | +0.066311 | +0.069414 |
| 78 | 2,146 | 2,015 | +0.112906 | +0.101329 |
| 79 | 2,188 | 2,083 | +0.175100 | +0.164176 |

The late-origin weakness is localized rather than uniform. Present rows in cycles 68, 69, 71, 72, 74, and 76 are below current R06; W4 as a whole is also below R06.

### Target-defined support groups

These are descriptive groups formed from the already visible Train outcome (`threat_rank`): low `<=5`, middle `5<y<95`, high `>=95`. They were not used for the route or any fitting. Gains are against R06 under the same fixed route.

| View | Availability | Target group | n | Gain vs R06 |
|---|---|---|---:|---:|
| Primary | Present | `<=5` | 430 | +0.539478 |
| Primary | Present | `5<y<95` | 7,478 | +0.204381 |
| Primary | Present | `>=95` | 422 | +0.336972 |
| Primary | Missing | `<=5` | 19 | −0.304907 |
| Primary | Missing | `5<y<95` | 675 | +0.172044 |
| Primary | Missing | `>=95` | 40 | +0.367726 |
| H12 | Present | `<=5` | 1,843 | +0.657397 |
| H12 | Present | `5<y<95` | 32,321 | +0.157586 |
| H12 | Present | `>=95` | 1,846 | +0.610751 |
| H12 | Missing | `<=5` | 85 | −0.219884 |
| H12 | Missing | `5<y<95` | 2,384 | +0.151599 |
| H12 | Missing | `>=95` | 109 | +0.502846 |

All present target groups have at least 100 rows. The H12 missing high group (n=109) meets that support mark; missing low groups (Primary n=19, H12 n=85) do not. The supported present tail gains are positive, but this does not remove the W4/cycle-level losses.

### Continuation decision and scope boundary

The corrected zero-fit evidence verifies a positive Primary and H12 aggregate margin against current R06, including present rows. It does **not** retroactively pass R12's original frozen gate: original R12 missing-stream harm still exceeded +0.25 on W1 (+0.768759, n=392), Primary (+0.489276, n=734), and H12 (+0.324499, n=2,578). The fallback was selected after pilot outcomes were visible, and these are reused Train development labels, so the rescue comparison is post hoc rather than independent validation. W4 and several late present cycles also show losses.

Recommendation: the evidence is sufficient to justify a **separately frozen prospective proposal** for R12_PRESENT_ROUTE if Sol/user chooses to spend the remaining experiment budget. This memo does not authorize a fit, predict call, or resume; the four-origin continuation outline in the prior review remains a cost proposal only.

The authoritative MAIN R12 run directory contains only fit55/fit67 bundles. No R12 continuation bundles or `R12_PRESENT_ROUTE` output were found for cutoffs 59, 63, 71, 75, or 79. A separate `tk2-regression-frontier` E79 `reg_position_real_e79_benchmark_v1` bundle was identified by inventory/hash only: predictions SHA-256 `AA81A7D17CC0DFE9DA1DE35235DE87DCE4B89D4F4B34FCB34ED921478B10C4E0`, manifest SHA-256 `15FE5C6446B7CA2EB3B24FE72E9B4A85C51A6C4847495B192AF7E76143474802`. It is not an R12 continuation and its contents were not opened.

No fit, prediction call, model load, Test/SampleSubmission/Kaggle access, upload, dependency probe, policy grid, or immutable-artifact change occurred. This amendment is limited to the four saved R12/R06 bundles, the pinned Train rows at their saved score positions, and this isolated reconciliation report.
