# Saved-output reconciliation 01

Date: 2026-11-01
Source checkout: `f54f50dface534dfbea26743626320ff35260f4d`
Scope: resolve twelve stale, partial, versioned, or gate-file-missing entries in the rolling V3 pilot index.

This reconciliation uses existing frozen identities, fit receipts, saved bundles, result files, verification files, and predeclared gates. It ran no estimator refits, accessed no Test/SampleSubmission/Kaggle data, and changed no queue, model, EDA, notebook, or per-card frozen output. The independent saved-output audits replayed row/ID alignment, metrics, route or calibration reconstruction, subgroup denominators, and the declared gates against the official Train files.

The official Classification Train identity used by classification cards is SHA-256 `875552a68a3d24b0f3e1966d6ddb5671e6f54305a1c526e2f0a76747942ed91c`; the official Regression Train identity used by regression cards is `756523ff8156a9cd776e52249f7e2a8f23dcec432b8d10570e55ff6db0e0f082`. Each per-card `freeze.json` remains the source of its full recipe, source/input hashes, feature and fold definitions, and freeze signature.

| Card | Reconciled decision | Primary view (rows; gain) | H12 view (rows; gain) | New predictor fits (failed; fit seconds) | Gate evidence |
|---|---|---:|---:|---:|---|
| C03 | RETIRE; independent saved-output QA PASS | 9,064; −0.000319 Macro F1 | 38,588; −0.000471 Macro F1 | 2 (0; 21.287) | Primary-gain threshold fails; H12, window, and supported subgroup guards pass. |
| R01 | RETIRE; independent saved-output QA PASS | 9,064; +0.034940 MAE gain | 38,588; +0.076438 MAE gain | 2 (0; 129.369) | Primary/H12/window checks pass, but W4 `target_tail=high_ge95` has +0.460013 MAE harm (n=249), above +0.25. |
| C02 | RETIRE; independent saved-output QA PASS | 9,064; −0.000534 Macro F1 | 38,588; +0.000089 Macro F1 | 2 (0; 22.696) | Primary-gain threshold fails; H12, window, and supported subgroup guards pass. |
| C07 | RETIRE; independent saved-output QA PASS | 9,064; −0.000181 Macro F1 | 38,588; +0.000216 Macro F1 | 2 (0; 25.532) | Primary-gain threshold fails; other declared guards pass. |
| R18 | RETIRE; independent saved-output QA PASS | 13,441 late W4+W6; +0.014361 MAE gain | 20,903 H12_L; +0.027931 MAE gain | 0 (0; not applicable) | Frozen design applies past-only OOF offsets/fallbacks without predictor fits. Primary minimum +0.03 fails; supported low-tail harms are +0.592 W4, +0.455 W6, and +0.591 H12_L. |
| C09 | RETIRE; row-scope reconciliation and independent saved-output QA PASS | 9,064; +0.000110 Macro F1 | 38,588; −0.000168 Macro F1 | 2 (0; 10.664) | Correct W1+W4 row-matched primary gain is below +0.0005; H12/window and supported subgroup guards pass. The original 9,064-vs-38,588 comparison remains preserved as an invalid historical assertion. |
| R03 | RETIRE; independent saved-output QA PASS | 9,064; +0.013836 MAE gain | 38,588; +0.013678 MAE gain | 2 (0; duration UNKNOWN) | Primary minimum +0.03 fails; H12, window, and supported tail/missing guards pass. No attempt-time log was present. |
| C06 | RETIRE; independent saved-output QA PASS | 9,064; −0.000576 Macro F1 | 38,588; +0.000092 Macro F1 | 2 (0; 22.371) | Primary-gain threshold fails; other declared guards pass. Freeze amendment 01 preserves the pre-fit serialization failure record. |
| R12 | RETIRE; independent saved-output QA PASS | 9,064; +0.245801 MAE gain | 38,588; +0.284490 MAE gain | 2 (0; 45.521) | Aggregate gates pass, but supported missing-stream harms exceed +0.25: W1 +0.768759 (n=392), primary +0.489276 (n=734), H12 +0.324499 (n=2,578). |
| R17 | RETIRE; independent saved-output QA PASS | 13,441 late W4+W6; +0.032732 MAE gain | 20,903 H12_L; +0.031992 MAE gain | 2 (0; 44.640) | Primary minimum passes; supported low-tail harms fail +0.25: W4 +0.593 (n=244), W6 +0.969 (n=442), H12_L +0.596 (n=1,048). |
| C10 | RETIRE; independent saved-output QA PASS | 9,064; +0.000343 Macro F1 | 38,588; +0.000878 Macro F1 | 2 (0; 30.051) | Primary-gain threshold fails; H12/window and supported subgroup guards pass. Gate is embedded in `results.json` and reconstructed by `verification.json`. |
| R23 | RETIRE; independent saved-output QA PASS, version 2 | 9,064; +0.092085 MAE gain | 38,588; +0.111498 MAE gain | 2 (0; 1,038.187) | Aggregate gates pass, but supported harms fail +0.25: missing-stream W1 +0.488615 (n=392), target ≥95 W4 +0.263713 (n=249), missing-stream primary +0.344021 (n=734). |

## Evidence locations and lifecycle differences

- C03, R01, C02, and R12 use the saved `pilot.json` schema instead of a separate `results.json`; each already has a matching freeze signature, independent `verification.json`, and `gate.json`. The index now reads those actual artifacts rather than treating the rows as unrun.
- C07 and C06 results had retained the lifecycle string `PILOT_SAVED_AWAITING_INDEPENDENT_QA`. The independent saved-bundle audits now pass; the original result and gate files are unchanged.
- C09's result and recovery manifest preserve the original scope-check failure. The corrected row-matched metrics above are independently reconstructed from frozen prediction caches. Its frozen shared-helper SHA `d21ac914d0d1f516a93e431116556dabdbe7313a21fd2b1b04a73371bd56f420` exactly matches the helper in result commit `9e9b7fb` and the parent of `5840f4c`; the only later helper diff is budget-reservation bookkeeping in `reserve_card_budget`, not prediction or metric code. This reconciles the source hash drift without rewriting the historical freeze or runner failure.
- R03, R17, R18, and C10 lack a separate `gate.json`, or retain a stale lifecycle status, but their saved verifier/summary includes a reproducible gate reconstruction and terminal RETIRE decision. R18's zero predictor fits are part of its frozen design, not a missing fit.
- R23's complete package is versioned as `freeze.v2.json`, `results.v2.json`, `verification.v2.json`, `gate.v2.json`, and `summary.v2.md`; the earlier `NO_REPORT` index entry was stale.
- The shared helper hash `d21ac914…` is recoverable in historical Git content; current SHA `746e12d4…` differs only by later reservation bookkeeping. Some historical freeze identities also contain queue hashes predating the unrelated R10/R13 queue-status update. Frozen recipes and card-specific source identities remain preserved.

## Budget-ledger reconciliation

The tracked `reports/optimization_v3/batch1/batch_ledger.jsonl` and ignored runtime `experiments/runs/optimization_v3_batch1/batch_ledger.jsonl` overlap on the first five reservations. A deduplicated union adds legacy-only reservations C06 (2), R12 (2), and C01 (0), bringing the conservative runtime reservation total to **94/180**. The runtime ledger was updated by appending migration entries that record the source ledger SHA-256 `4c73a5604e6638f44241fbc040a95034e0045e1b18461eacdc1c5451b797f68f`; neither ledger was rewritten. No fit was run for this accounting repair.

After these reconciliations, **43/45** cards have terminal saved pilot decisions and R10/R13 remain the two explicit prerequisite skips. This closes the initial screening inventory; it does not complete the conditional continuation or finalist stages.
