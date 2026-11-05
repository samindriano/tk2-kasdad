# Root source review R1 — REWORK before official CLS fits

Reviewed ade4a677048776ff423a7416b9274f75b7688b6a. Root reran `python -m pytest -q tests/test_pos_frequency_ridge_v1.py`: **14 passed in5.07s**. No official fit/target/Test access by root. No ROOT_SOURCE_PASS issued; source readiness is not official execution readiness. Preserve original source-preflight and source-readiness receipts as historical preparation.

Numerical review: union-of-three-thirds DF, mean-one clipped inverse-sqrt DF penalty, repeated token weights, weighted objective/gradient and frozen bag logit agree with the handoff. Pure numerical tests support the math. Hypothesis remains unproven. Source-preflight validates inputs; it does not exercise the actual runner persistence/lock paths listed below.

## Required narrow execution repairs

1. Cross-lane job exclusion is absent. `_run_one_fit` locks only RUN_ROOT/FIT_RUN.lock; there is no process scan or shared global lock. REG now uses OS advisory lock at tempfile.gettempdir()/kasdad_single_heavy_job.lock. Acquire that SAME shared lock before memory-heavy input/penalty work, labels, reservation and fit; retain local lock if useful. Fresh resource checks remain. Do not delete locks, kill jobs, poll or invent ownership. Synthetic test must hold shared lock in another process and prove no labels/reservation/fit starts; release then show it proceeds synthetically.

2. Durable postpublish recovery is missing. `fit_resume_decision` rejects FIT_STARTED without FIT_SAVED, even if gamma AND source/input-bound receipt are safely published. `_ensure_penalty_map` similarly rejects PREPROCESSING_MAP_STARTED before inspecting a valid artifact+receipt. Add validation/reconciliation that appends missing terminal event only when BOTH files are intact and bind exact source/root-pass/Train/prefix IDs/matrices/offset/penalty formula and vector. Never rerun the optimizer/map in that case. A start with missing/incomplete/corrupt proof remains blocked, no retry. Add actual runner-path fault injection after artifact+receipt publish before terminal event, then resume with zero optimizer calls. Current synthetic test manually appends FIT_SAVED then calls a helper; it does not test this crash case.

3. `_append_campaign_reservation` runs before the resource gate. Move reservation behind fresh RAM/shared-job gates, before target/fit starts as appropriate. Prove low RAM and held lock yield no official reservation/map/label/fit event. Keep cap4; no global historical ledger changes.

4. On normal saved-fit reuse, validate receipt against current exact source/root pass/preflight/Train, prefix rows/IDs, base/position matrices, bag offset and repeated penalty vector, and match terminal ledger event digest. Current path checks only artifact byte hash. Add a wrong-source/wrong-offset/penalty receipt mutation test. Immutable persistence must not overwrite conflicting existing artifact/receipt bytes.

Keep changes inside the existing namespace; no algorithm/seed/penalty/optimizer changes, no broad refactor. Preserve old preflight file or explicitly version successor report/manifest and bind ROOT_SOURCE_PASS to the active digest after repair. Commit repaired executable/tests, run focused synthetic QA/source-only verifier, callback exact source-ready commit and hashes. Resource below4GiB remains a safe runtime stop; no polling/schedules or user question.

## E79 scope

Luna supplied exact async reply reference call_b28a2f4712a445bb815d11b1b3d76673, wording 'Ya, otorisasi satu benchmark E79'. This is recorded for later scope reconciliation. It does not change this source review into a fit approval or remove input/source QA. Complete the current four-origin-arm pilot preparation first; no old POS_MOMENTS/bigram source substitution. A separately pinned current-recipe E79 amendment will be reviewed by root if that stage is taken. Do not ask Sam again about this internal staging issue.
