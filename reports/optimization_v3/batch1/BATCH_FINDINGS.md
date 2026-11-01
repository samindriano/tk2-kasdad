# V3 Batch 1 — rolling findings

Snapshot: 2026-11-01, after R14 saved-output verification v2. This is a rolling index built from the canonical queue and per-card reports present in the repository; it does not claim that missing reports were executed or validated.

Queue closure: **20/45 cards (44.4%)** now have terminal pilot decisions with saved gates; this counts R14 and excludes cards still awaiting QA or missing a gate. This is pilot-card progress, not the later conditional continuations and finalist stage.

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
| 33 | **C11** | PARTIAL: cutoff 55 saved; cutoff 67 resource-interrupted | NO_GATE | - | - | 1 saved / 1 interrupted | [partial status](C11/partial_status.json), [attempts](C11/attempts.jsonl) |
| 34 | **R08** | IMPLEMENTED; SYNTHETIC_QA_PASS; OFFICIAL_PREFLIGHT_BLOCKED_RESOURCE | NO_GATE | - | - | 0 / 0 | [preparation status](R08/preparation_status.json) |
| 35 | **C20** | NO_REPORT | NO_GATE | - | - | ? / ? | — |
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

## C11 — partial checkpoint; not scored

- C11's frozen word plus `char_wb(3,5)` recipe passed Train/control/support/budget preflight. The analyzer-based estimate was 2.25 GiB at cutoff 55 and 3.49 GiB at cutoff 67; independent profiling estimated 2.43/3.62 GiB. Both are estimates, not peak model measurements.
- The cutoff-55 fit completed and its ignored bundle is preserved (113.30 seconds). Cutoff 67 was started, then interrupted by MAIN after system free memory fell to 0.49 GiB, below the frozen 2 GiB reserve; process working set was 2.22 GiB. No cutoff-67 bundle exists. The interruption is logged separately in `C11/attempts.jsonl`; no partial candidate metric was computed and no model conclusion is made.
- C11's two-fit reservation is 2/180 (global total now 58/180); reconciliation is pending until the second fit and report QA are complete. No Test, SampleSubmission, or Kaggle access occurred.
- Next action: retain cutoff 55; rerun current memory and single-slot preflight and resume cutoff 67 only when the full frozen estimate fits available RAM minus the 2 GiB reserve. The runner's resume path must reuse cutoff 55 and must not refit it.

## R08 — implementation ready; official pilot not started

- R08's adapter and runner are implemented in its own namespace. Seven targeted tests pass, including the synthetic two-cutoff saved-bundle/independent-verification path, fit/target/scalar digests, tied within-cycle ranks, raw character-count behavior, and zero/low-support groups.
- Official Regression Train SHA matches the frozen card. Fit/score populations are 12,020/17,685 at cutoff 55 and 29,705/20,903 at cutoff 67; primary W1+W4 is 9,064 rows and H12 is 38,588. The train-only scalar profile is recorded in `R08/preparation_status.json`.
- The shared fit budget is available (58 reserved; 60 if R08 reserves its two fits, out of 180). The live resource check is blocked: 2.30 GiB physical RAM was available, against estimated R08 peaks of 2.04/2.09 GiB plus the required 2 GiB reserve. These are estimates, not observed model peaks.
- No R08 recipe/freeze or fit reservation was written, and no official candidate fit or candidate metric was computed. No Test, SampleSubmission, or Kaggle access occurred. R08 is not a terminal lane decision and the closure count remains 20/45.
- Next action: keep this implementation and tests; recheck RAM and training-slot ownership before freezing. Resume C11's saved cutoff 67 first if its frozen resource preflight passes; then rerun the R08 preflight and freeze only when its own resource gate passes.

## Ledger and scope notes

- The active budget ledger used by the runner is `experiments/runs/optimization_v3_batch1/batch_ledger.jsonl`; it records 58 of 180 candidate predictor fits reserved after C11. C05 and R14 are reconciled; C11 reconciliation is pending. Runtime fit bundles remain ignored by Git.
- The tracked `reports/optimization_v3/batch1/batch_ledger.jsonl` is an older partial reservation snapshot (8 entries); it is not a complete mirror of the runtime ledger and must not be used to infer remaining global budget.
- V3 remains active. This file is an interim evidence index; it does not close the remaining conditional queue or authorize Test/Kaggle access for pilots.
