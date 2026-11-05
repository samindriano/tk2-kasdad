# REG final wave follow-up V2 — source and execution contract

## Purpose and scope

This bounded follow-up asks two separate questions using already-scored regression predictions:

1. Does a predeclared half-life-eight weighted affine L1 fit improve on an unweighted affine L1 fit when both are trained on the same causal prefix predictions?
2. Does restoring the original direct prediction on the exact saved missing-stream rows avoid any harm from applying the previously fitted global affine candidate there?

This is reused temporal development evidence, not an untouched holdout. The prior public best remains ref `57994383`, MAE `11.04552`; that score is context and cannot select a recipe, clear a local gate, or establish private performance. Candidate A's final full-OOF fit is in-sample. No new base estimator, preprocessing fit, model family, half-life search, threshold search, or score-driven recipe change is in scope.

## Frozen prediction and row identity

- Official regression Train SHA-256: `756523ff8156a9cd776e52249f7e2a8f23dcec432b8d10570e55ff6db0e0f082`.
- OOF support is exactly 20,903 unique physical rows and record IDs.
- For cycles 68–75, preserve the direct-position predictions from E67 (12,412 rows). For cycles 76–79, use E75 (8,491 rows). The E67 late-cycle copy is excluded from the stitched support. Join by original physical row, record ID, cycle, origin, and original bundle index. Do not average origin predictions.
- E67's underlying direct model is fit through cycle 67; E75's is fit through cycle 75. This origin transition is a limitation in the D2 temporal comparison.
- E79 application uses only the exact saved inference bundle, its signed identity/missing mask, and round-trip parsed saved candidate CSVs. Do not read Test.csv or SampleSubmission.csv; no Test labels are allowed.

## Models and fixed chronology

For each diagnostic origin, fit both calibrators on the same causal direct OOF history:

| Stage | Fit history | Score window | Prediction origin |
|---|---|---|---|
| D1 | cycles 68–71 | 72–75 | E67 |
| D2 | cycles 68–75 | 76–79 | E75 |
| FINAL | unique stitched cycles 68–79 | none | E67 for 68–75, E75 for 76–79 |

The global fit minimizes `sum(abs(y - a - b*p))`, with free intercept and `b >= 0`. H8 minimizes `sum(w_i * abs(y_i - a - b*p_i))` on the identical rows, with `raw_i = 2**(-(t-cycle_i)/8)` and weights normalized to arithmetic mean one over the exact fit rows. Record min, max, sum, mean, effective sample size, cycle counts, and weight digest. No tail weights, fit-time clipping, coefficient constraints beyond nonnegative slope, or outcome-driven rescue are allowed. Use sparse HiGHS with two threads and the pinned R17 numerical conventions. Clip `a+b*p` to `[0,100]` only when predicting.

LP order and cap: `D1_GLOBAL`, `D1_RECENCY_H8`, `D2_GLOBAL`, `D2_RECENCY_H8`, `FINAL_RECENCY_H8`; five starts maximum. D1 and D2 each report raw direct, causal global on all rows, causal H8 on all rows, and causal global with direct restored on its missing rows. The present-only diagnostic uses the same per-prefix causal global coefficients as the all-row diagnostic. The current global coefficients (`a=2.169196428141562`, `b=0.9744451489746011`, public ref `57994383`) are permitted only as a labeled post-selected descriptive operational reference; never use them as a causal held-out comparator or gate.

Candidate A is the one final H8 fit on all 20,903 stitched rows at reference cycle 79, later applied to all 29,421 saved E79 direct predictions, including the 1,062 rows formerly routed to fallback. Candidate B uses the existing global-affine saved prediction on present rows and restores the raw direct saved prediction on the exact 1,062-row missing mask; it consumes zero new LP starts. At most two local exploratory CSVs may be produced after separate root review. Neither candidate is a champion or upload authorization.

## Reporting and review screens

Report D1 and D2 independently and pool their unique evaluation IDs (W5+W6, 15,953 rows) using pooled absolute-error sums and denominators. Never average the two window MAEs. Report MAE and denominator overall, per cycle, for present/missing, and for `y <= 5` / `y >= 95`; groups with fewer than 200 rows are `UNKNOWN_UNSUPPORTED`. Include prediction changes, monotonicity, range, clipping counts, coefficients, and differences against causal global and raw direct. A supported missing/tail subgroup MAE harm above `0.25` is an explicit risk flag.

Frozen screens guide root review only:

- H8 screen: pooled MAE gain over causal global at least `0.02`, with no D1/D2 MAE harm above `0.03`.
- Candidate B screen: pooled causal global-present-only MAE no worse than causal global all-row, with no D1/D2 harm above `0.03`.

Preserve all failed screens. They do not authorize retrospective PASS, public-score selection, or private-performance claims. Public benchmark access/upload is outside this runner's authorization.

## Durability, permissions, and resource gates

- R3 preserves the R1 artifacts, the R2 freeze/identity-preflight/synthetic-QA receipts, and `SOURCE_VERIFICATION_FAILED_R2.json`. R2's independent verifier found that two pinned helpers are resolved from sibling worktrees while the checkout copies have identical bytes; R2 therefore remains a recorded failed verification. R3 uses `PREFIT_FREEZE_R3.json`, `SOURCE_PREPARATION_BLOCKED_R3.json`, `IDENTITY_PREFLIGHT_R3.json`, and `SYNTHETIC_QA_R3.json`, and its lineage binds R1 and R2 plus their zero-official-access receipts.
- The frozen source commit must remain an ancestor of live `HEAD` on `codex/reg-position-failure-mechanism-v1`. Descendant commits are accepted only when every changed path is a report-only `.md` or `.txt` under this experiment's report directory and is not a pinned source. All pinned source/dependency/protocol/test/QA bytes and input hashes must still match the freeze; tracked or staged executable/non-report changes and unexpected untracked files fail closed. Root receipts continue to bind the original frozen source commit, freeze digest, and fit IDs.
- Prefit freeze binds committed source revision, protocol, script, tests, verifier, synthetic QA, all pinned input files, saved E79 source identities, prior global affine state/receipt, and exact stitched OOF identity before official labels are read.
- No official label parse or LP starts before an exact `ROOT_SOURCE_PASS.json`. Each LP reservation/start/state binds that pass receipt, freeze, source/input hashes, fit ID, cycle/row/ID/prediction/target/weight hashes, and source commit.
- `ROOT_SOURCE_PASS.json` must be a signed `PASS` receipt for this experiment, exact frozen source commit, prefit-freeze SHA, complete source/input hash maps, and exactly the first four diagnostic fit IDs. Target access rechecks the approval, freeze, expected branch, source ancestry, pinned source/input hashes, and report-only descendant boundary immediately before parsing.
- A later `ROOT_PRODUCTION_PASS.json` must bind that source pass, the signed diagnostics file, prefit freeze, source/input hash maps, exactly `FINAL_RECENCY_H8`, candidates `A` and `B`, and a two-CSV maximum.
- Five unique LP IDs maximum. A start without a provable terminal immutable state is a terminal blocker; never retry it. A resource stop before LP start consumes no start. A verified state saved before its ledger completion event is reconciled from the exact signed state with zero refit. A reservation without a start may resume only the same bound unstarted slot.
- State, prediction, and CSV files are published atomically with no-overwrite semantics; JSONL ledger events are append-only and fsynced.
- Use the shared advisory lock `%TEMP%/kasdad_single_heavy_job.lock` for the whole official run, compatible with the CLS heavy-job lane. Never delete a lock or terminate another process. Check process ownership and actual lock state before work; conflict blocks without polling.
- Before selected label access/reservation and before every LP, measure available physical RAM and require at least 4 GiB and enough for the 512 MiB estimated incremental peak plus the 2 GiB reserve. The 512 MiB value is an estimate, not a measured peak. Use one heavy job, threadpool 2 and HiGHS 2.
- Independent verification must recompute OOF identities, receipt/source bindings, unweighted and H8-weighted LP objectives, prediction equations, metrics, pooling, and candidate CSV round-trip checks without importing or calling the builder.

## Required source-preparation verification

Candidate-blind synthetic QA covers known weighted affine recovery; unit weights against the pinned unweighted objective and predictions (not arbitrary coefficient equality for a nonunique solution); H8 weight formula and scale invariance; shuffled original-index stitch; duplicate/missing identities; support boundaries; present-only route; causal/descriptive comparator separation; finite range and CSV round-trip; mismatched source/receipt rejection before labels/start; ledger cap/write-once/recovery after persisted fit before completion without refit.

Source preparation and identity preflight must report exact paths/hashes, frozen commit/live checkpoint/branch, row identities, measured available RAM, estimate-vs-measurement distinction, actual lock/process status, and UNKNOWNs. The independent verifier resolves internal sources to this checkout and checks the two external helper paths against their pinned checkout suffixes and frozen bytes. A previously saved successful identity-preflight receipt is immutable: a rerun performs fresh resource/process/lock checks, validates the saved receipt against the exact freeze and checkpoint, then reuses it without rewriting when the RAM snapshot differs. It must read zero Train target rows, start zero official LPs, access no Test/Sample, create no candidate CSV, and make no Kaggle request. Run source commands only:

```powershell
python scripts/reg_final_wave_followup_v2.py --synthetic-qa
python -m pytest tests/test_reg_final_wave_followup_v2.py
python scripts/reg_final_wave_followup_v2.py --prepare-freeze
python scripts/reg_final_wave_followup_v2.py --identity-preflight
python scripts/verify_reg_final_wave_followup_v2.py --verify-prefit
```

Official diagnostics, final production LP, and candidate CSVs require later separately bound root authorizations.
