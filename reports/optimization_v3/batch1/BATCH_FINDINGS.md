# V3 Batch 1 — rolling findings

Snapshot: 2026-11-01, after C11 pilot resume, R08 and C20 completion, and independent QA. This is a rolling index built from the canonical queue and per-card reports present in the repository; it does not claim that missing reports were executed or validated.

Queue closure: **23/45 cards (51.1%)** now have terminal pilot decisions with saved gates; this counts R14, C11, R08, and C20 and excludes cards still awaiting QA or missing a gate. This is pilot-card progress, not the later conditional continuations and finalist stage.

For each row, metrics are copied from that card’s `results.json`; statuses come from its saved `verification.json` and `gate.json`. `NO_REPORT` and `PARTIAL` remain unresolved. Gains use each card’s declared pilot view, so raw gains from different populations are not directly ranked.

| Order | Card | Evidence status | Gate | Primary (rows; gain) | H12 (rows; gain) | Fits (saved / failed) | Evidence |
|---:|:---:|---|---|---:|---:|---:|---|
| 1 | **C03** | PARTIAL: verification only | RETIRE | - | - | ? / ? | [QA](C03/verification.json), [gate](C03/gate.json) |
| 2 | **R01** | PARTIAL: verification only | RETIRE | - | - | ? / ? | [QA](R01/verification.json), [gate](R01/gate.json) |
| 3 | **C02** | PARTIAL: verification only | RETIRE | - | - | ? / ? | [QA](C02/verification.json), [gate](C02/gate.json) |
| 4 | **R02** | RETIRE; QA PASS | RETIRE | 9064; +0.022196 | 38588; +0.010084 | 2 / ? | [results](R02/results.json), [QA](R02/verification.json), [gate](R02/gate.json) |
| 5 | **C07** | PILOT_SAVED_AWAITING_INDEPENDENT_QA; QA PASS | RETIRE | 9064; -0.000181 | 38588; +0.000216 | 2 / ? | [results](C07/results.json), [QA](C07/verification.json), [gate](C07/gate.json) |
| 6 | **R18** | PILOT_VERIFIED; gate missing | NO_GATE | 13441; +0.014361 | ?; ? | 0 / ? | [results](R18/results.json), [QA](R18/verification.json) |
| 7 | **C09** | PILOT_COMPLETE_RETIRE; gate missing | NO_GATE | ?; ? | 38588; ? | ? / ? | [results](C09/results.json), [QA](C09/verification.json) |
| 8 | **R03** | PILOT_SAVED_AWAITING_INDEPENDENT_QA; gate missing | NO_GATE | 9064; +0.013836 | 38588; +0.013678 | 2 / ? | [results](R03/results.json), [QA](R03/verification.json) |
| 9 | **C04** | PILOT_COMPLETE_RETIRE; QA INDEPENDENT_QA_PASS | RETIRE | 9064; -0.000212 | 38588; +0.000810 | 2 / 0 | [results](C04/results.json), [QA](C04/verification.json), [gate](C04/gate.json) |
| 10 | **R16** | PILOT_COMPLETE_RETIRE; QA INDEPENDENT_QA_PASS | FAIL_VALID_EVIDENCE | 9064; +0.015729 | 38588; +0.017173 | 2 / 0 | [results](R16/results.json), [QA](R16/verification.json), [gate](R16/gate.json) |
| 11 | **C18** | PILOT_COMPLETE_RETIRE; QA INDEPENDENT_QA_PASS | FAIL_VALID_EVIDENCE | 9064; -0.000045 | 38588; -0.000496 | 2 / 0 | [results](C18/results.json), [QA](C18/verification.json), [gate](C18/gate.json) |
| 12 | **R21** | PILOT_COMPLETE_ADVANCE; QA PASS | ADVANCE | 9064; +0.030828 | 38588; +0.044280 | 2 / ? | [results](R21/results.json), [QA](R21/verification.json), [gate](R21/gate.json) |
| 13 | **C16** | PILOT_COMPLETE_ADVANCE; QA INDEPENDENT_QA_PASS | PASS | 9064; +0.001971 | 38588; +0.002837 | 2 / 0 | [results](C16/results.json), [QA](C16/verification.json), [gate](C16/gate.json) |
| 14 | **R24** | PILOT_COMPLETE_RETIRE; QA INDEPENDENT_QA_PASS | FAIL_VALID_EVIDENCE | 9064; +0.003289 | 38588; +0.002348 | 2 / ? | [results](R24/results.json), [QA](R24/verification.json), [gate](R24/gate.json) |
| 15 | **C13** | PILOT_COMPLETE_RETIRE; QA INDEPENDENT_QA_PASS | FAIL_VALID_EVIDENCE | 9064; -0.002459 | 38588; -0.002541 | 2 / 0 | [results](C13/results.json), [QA](C13/verification.json), [gate](C13/gate.json) |
| 16 | **R09** | PILOT_COMPLETE_RETIRE; QA PASS | RETIRE | 9064; -0.036425 | 38588; -0.081655 | 2 / ? | [results](R09/results.json), [QA](R09/verification.json), [gate](R09/gate.json) |
| 17 | **C06** | PILOT_SAVED_AWAITING_INDEPENDENT_QA; QA PASS | RETIRE | 9064; -0.000576 | 38588; +0.000092 | 2 / ? | [results](C06/results.json), [QA](C06/verification.json), [gate](C06/gate.json) |
| 18 | **R12** | PARTIAL: verification only | RETIRE | - | - | ? / ? | [QA](R12/verification.json), [gate](R12/gate.json) |
| 19 | **C01** | CACHE_SIX_COMPLETE; QA PASS | RETIRE | 38588; -0.002174 | 38588; -0.002175 | ? / ? | [results](C01/results.json), [QA](C01/verification.json), [gate](C01/gate.json) |
| 20 | **R17** | PILOT_VERIFIED; gate missing | NO_GATE | 13441; +0.032732 | ?; ? | 2 / ? | [results](R17/results.json), [QA](R17/verification.json) |
| 21 | **C10** | PILOT_COMPLETE_RETIRE; gate missing | NO_GATE | 9064; +0.000343 | 38588; +0.000878 | 2 / ? | [results](C10/results.json), [QA](C10/verification.json) |
| 22 | **R05** | RETIRE; QA PASS | RETIRE | ?; ? | ?; ? | 2 / ? | [results](R05/results.json), [QA](R05/verification.json), [gate](R05/gate.json) |
| 23 | **C19** | PILOT_VERIFIED; QA PASS | RETIRE | 9064; +0.000375 | 38588; +0.004152 | ? / ? | [results](C19/results.json), [QA](C19/verification.json), [gate](C19/gate.json) |
| 24 | **R25** | RETIRE; QA PASS | RETIRE | ?; ? | ?; ? | 2 / ? | [results](R25/results.json), [QA](R25/verification.json), [gate](R25/gate.json) |
| 25 | **C08** | PILOT_VERIFIED; QA PASS | RETIRE | 9064; +0.000128 | 38588; -0.000911 | ? / ? | [results](C08/results.json), [QA](C08/verification.json), [gate](C08/gate.json) |
| 26 | **R15** | PILOT_VERIFIED; QA PASS | RETIRE | 13441; +0.007600 | ?; ? | ? / ? | [results](R15/results.json), [QA](R15/verification.json), [gate](R15/gate.json) |
| 27 | **C17** | PILOT_COMPLETE_ADVANCE; QA INDEPENDENT_QA_PASS | PASS | 9064; +0.001541 | 38588; +0.001411 | 2 / 0 | [results](C17/results.json), [QA](C17/verification.json), [gate](C17/gate.json) |
| 28 | **R23** | NO_REPORT | NO_GATE | - | - | ? / ? | — |
| 29 | **C14** | PILOT_COMPLETE_RETIRE; QA INDEPENDENT_QA_PASS | FAIL_VALID_EVIDENCE | 9064; -0.006724 | 38588; -0.002919 | 2 / 0 | [results](C14/results.json), [QA](C14/verification.json), [gate](C14/gate.json) |
| 30 | **R04** | RETIRE; QA PASS | RETIRE | 9064; +0.021887 | 38588; +0.014318 | 2 / ? | [results](R04/results.json), [QA](R04/verification.json), [gate](R04/gate.json) |
| 31 | **C05** | PILOT_COMPLETE_RETIRE; QA INDEPENDENT_QA_PASS | RETIRE | 9064; -0.001260 | 38588; -0.000284 | 2 / 0 | [results](C05/results.json), [QA](C05/verification.json), [gate](C05/gate.json) |
| 32 | **R14** | PILOT_COMPLETE_RETIRE; QA INDEPENDENT_QA_PASS | RETIRE | 9064; +0.043122 | 38588; +0.026910 | 2 / 0 | [results](R14/results.json), [QA](R14/verification.json), [gate](R14/gate.json), [amendment](R14/report_amendment_01.json) |
| 33 | **C11** | PILOT_COMPLETE_ADVANCE; QA INDEPENDENT_QA_PASS | ADVANCE | 9064; +0.000661 | 38588; -0.000310 | 2 / 0 | [results](C11/results.json), [QA](C11/verification.json), [gate](C11/gate.json), [attempts](C11/attempts.jsonl), [partial checkpoint](C11/partial_status.json) |
| 34 | **R08** | PILOT_COMPLETE_RETIRE; QA PASS | RETIRE | 9064; -0.097574 | 38588; -0.121789 | 2 / 0 | [results](R08/results.json), [QA](R08/verification.json), [gate](R08/gate.json) |
| 35 | **C20** | CACHE_SIX_COMPLETE; QA INDEPENDENT_QA_PASS | RETIRE | 38588; -0.001573 | 38588; -0.004157 | 0 / 0 | [results](C20/results.json), [QA](C20/verification.json), [gate](C20/gate.json), [summary](C20/summary.md) |
| 36 | **R19** | NO_REPORT | NO_GATE | - | - | ? / ? | — |
| 37 | **C15** | NO_REPORT | NO_GATE | - | - | ? / ? | — |
| 38 | **R22** | NO_REPORT | NO_GATE | - | - | ? / ? | — |
| 39 | **R20** | NO_REPORT | NO_GATE | - | - | ? / ? | — |
| 40 | **C12** | NO_REPORT | NO_GATE | - | - | ? / ? | — |
| 41 | **R06** | NO_REPORT | NO_GATE | - | - | ? / ? | — |
| 42 | **R07** | NO_REPORT | NO_GATE | - | - | ? / ? | — |
| 43 | **R10** | NO_REPORT | NO_GATE | - | - | ? / ? | — |
| 44 | **R13** | NO_REPORT | NO_GATE | - | - | ? / ? | — |
| 45 | **R11** | NO_REPORT | NO_GATE | - | - | ? / ? | — |

## C05 — latest completed lane

- **Verdict: RETIRE.** The frozen C05 recipe changes only `min_df` from 3 to 2 while retaining `max_features=60,000` and word n-grams `(1,2)`. It preserves the frozen `.75 C2_B + .25 sparse expert` route on present streams, frozen CatBoost fallback on missing streams, and threshold 0.50.
- Official Classification Train SHA-256: `875552a68a3d24b0f3e1966d6ddb5671e6f54305a1c526e2f0a76747942ed91c`. Freeze signature: `d1635493beccf512b3f9d35c60ab994e3b4f9484073de8720840fb8d6843b36a`.
- Primary W1+W4: 9,064 rows; control 0.8239725; candidate 0.8227129; gain −0.0012596. The fixed requirement was +0.0005, so this gate failed.
- H12: 38,588 rows; control 0.8029656; candidate 0.8026813; gain −0.0002842, within the −0.001 harm limit. W1 gain −0.0022796 and W4 −0.0004333, both within the −0.005 view harm limit. Supported subgroup guards passed.
- Two predictor fits were saved (cutoffs 55 and 67), with zero failed attempts and 22.262 seconds total fit time. Peak memory was estimated at 2.04 GB / 2.78 GB, not observed.
- Independent verification passed and was repeated from the saved bundles; the repeated verification, gate, and fit accounting exactly matched the published reports. No Test data or Kaggle was used. C05 does not advance to full-six.
- Audit note: `attempts.jsonl` contains the four fit lifecycle events; the frozen runner’s intended `PREFLIGHT_PASS` append is unreachable. The complete preflight snapshot remains in `results.json`. [Report amendment](C05/report_amendment_01.json) records this logging-only limitation without changing metrics or refitting.
- Next eligible lane: **C11**, subject to a fresh support, data, RAM, budget, and single-job preflight.

## R14 — completed saved-output verification

- **Verdict: RETIRE.** The saved EF1-only specialist passed the primary gain gate (+0.043122 MAE on the exact 9,064-row W1+W4 primary), H12 harm guard (−0.026910), and W1/W4 harm guards. It failed the supported tail/missing guard: maximum harm was **+0.555867** on the supported H12_L `threat_rank >= 95` subgroup (n=1,057), above the +0.25 limit.
- Both frozen predictor fits (cutoffs 55 and 67) were reused; no refit occurred. Independent row, ID, fit-population, target-digest, incumbent-route, and candidate-route checks passed. No Test, SampleSubmission, or Kaggle access was used.
- The first post-fit verifier stopped before publishing due to two report-layer defects: it compared the 9,064-row pilot primary to a 38,588-row V2 full-six metric (11.559390), and its subgroup-harm aggregation had an invalid nested lookup. Amendment 01 documents these corrections. The exact row-matched R14 primary incumbent is 13.464396 on 9,064 rows; the V2 full-six value remains provenance only. H12 remains 14.047533 on 38,588 rows.
- Active verifier: `scripts/verify_optimization_v3_batch1_r14_v2.py`; frozen recipe and runner remain unchanged. See `reports/optimization_v3/batch1/R14/report_amendment_01.json` for source hashes and the defect record.
- Next eligible lane: **C11**, subject to its own support, RAM, dependency, budget, and single-training-job preflight.

## C11 — pilot complete; full-six continuation remains

- C11's frozen word plus `char_wb(3,5)` sparse recipe passed independent QA after resuming the previously interrupted cutoff 67. The preserved cutoff-55 bundle was hash-verified and reused; it was not refit. Cutoff 67 saved in 197.03 seconds. `attempts.jsonl` retains the original interruption, verified resume, and successful save events.
- Pilot verdict: **ADVANCE**. On the exact 9,064-row W1+W4 primary, Macro F1 moved from 0.8239725 to 0.8246337 (gain **+0.0006612**, above the +0.0005 gate). H12 moved from 0.8029656 to 0.8026553 (gain **−0.0003103**, within the 0.001 harm limit). W1 and W4 gains were +0.0003693 and +0.0008795; all declared pilot and supported subgroup guards passed.
- Both pilot predictor fits are saved and reconciled under the existing reservation; global pilot reservations remain 58/180. No Test, SampleSubmission, or Kaggle access occurred. `partial_status.json` is retained as the historical interruption checkpoint; `results.json`, `verification.json`, and `gate.json` contain the completed resumed pilot.
- C11 remains in the **ADVANCE** pool. Under the campaign handbook, continuation selection happens after the 45-card screening pass; the four new cutoffs (59, 63, 71, 75) are deferred until that selection. Preserve and reuse cutoffs 55 and 67, and keep the frozen model/route unchanged.

## R08 — completed RETIRE

- The frozen R08 recipe passed a fresh official preflight at approximately 5.4 GiB available, with estimated peaks 2.04/2.09 GiB, the 2 GiB reserve intact, and the single training slot free. Two fits completed (cutoffs 55 and 67); the shared reservation advanced from 58 to 60/180.
- On the exact 9,064-row W1+W4 primary, MAE changed from 13.464396 to 13.561970 (gain **−0.097574**). H12 changed from 14.047533 to 14.169321 (gain **−0.121789**). W1/W4 harms passed their individual limits, but primary, H12, and supported tail/missing gates failed. The independent verifier returned PASS; C20 is not continued. No Test, SampleSubmission, or Kaggle access occurred.

## C20 — completed RETIRE, cache-only

- C20 used the inherited all-fit-row q80 rule (`cycle_code <= fit_end`, linear interpolation), including missing-stream rows; `text_load == q80` stays with C2_B. The missing-stream branch is CatBoost, high-volume present rows use sparse25, and other present rows use C2_B. Six-window support totals 38,588 rows. No new models were fit.
- Full-six Macro F1 moved from 0.829944 to 0.828371 (gain **−0.001573**); H12 moved from 0.802966 to 0.798808 (gain **−0.004157**). All six individual window harms were within −0.005, but 0/6 windows were nonnegative; full-primary, H12, and window-count gates failed. Supported subgroup guards passed. Independent QA passed; decision **RETIRE**.
- Process note: while resolving the inherited q80 definition, the historical volume-route findings table (including its old metrics) appeared in search output before C20 was frozen. No C20 candidate metric was computed before its freeze, and no Test/Kaggle data were read, but this discovery was not strictly outcome-blind. The frozen, predeclared route was not changed after C20 scoring.
- Next eligible pilot is **R19** (queue order 36), subject to its fixed LATE chronology/support preflight; do not select C11 continuation yet.

## Ledger and scope notes

- The active budget ledger used by the runner is `experiments/runs/optimization_v3_batch1/batch_ledger.jsonl`; it records 60 of 180 candidate predictor fits reserved after R08. C20 added zero fits. Runtime fit bundles remain ignored by Git.
- The tracked `reports/optimization_v3/batch1/batch_ledger.jsonl` is an older partial reservation snapshot (8 entries); it is not a complete mirror of the runtime ledger and must not be used to infer remaining global budget.
- V3 remains active. This file is an interim evidence index; it does not close the remaining conditional queue or authorize Test/Kaggle access for pilots.
