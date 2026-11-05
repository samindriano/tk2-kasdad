# CLS_POS_FREQUENCY_RIDGE_V1 — source readiness R2

Status: **source and identity checks passed; official fitting remains blocked** until a committed `ROOT_SOURCE_PASS.json` is issued. The fresh RAM gate passed during preflight, but that does not authorize a fit by itself.

## Changes in this checkpoint

- The Classification runner now acquires the shared OS advisory lock at `%TEMP%/kasdad_single_heavy_job.lock`, matching the active Regression follow-up lane. It holds the shared lock and the local campaign lock before a fresh RAM gate, campaign reservation, matrix work, label access, or fitting.
- Campaign reservation now follows the shared-lock and RAM gates. A held shared lock or failed RAM check produces no reservation or downstream work.
- Penalty-map receipts bind the current source commit and hashes, root-source-pass and R2 preflight digests, official Train hash, exact prefix row and ID hashes, positional-matrix hash, vocabulary width, seed, and frozen penalty formula. Published artifacts and receipts use exclusive creation and are never overwritten.
- Gamma receipts also bind the current source/preflight/root-pass/Train identity, prefix rows and IDs, base and positional matrices, fixed bag state and offset, penalty-map receipt, and repeated penalty vector. A saved event must match both receipt and artifact digests.
- If a process stops after both artifact and receipt are durable but before the terminal event, resume validates the complete lineage and appends only the missing terminal event. Missing, partial, corrupt, or conflicting files remain blocked; optimizer and map computations are not retried.
- The original `SOURCE_PREFLIGHT.json` and `SOURCE_READINESS.md` remain unchanged as historical R1 evidence. This report and `SOURCE_PREFLIGHT_R2.json` are their versioned successors.

## Target-free preflight

`SOURCE_PREFLIGHT_R2.json` was generated from code commit `721edc30344e62f7365baf91e533080ffe0bf7cc` and has SHA-256 `065f81e4d20da6559f11983c9cbf0a372e75d16537450bed7999619964383b73`.

- Official Train SHA-256: `875552a68a3d24b0f3e1966d6ddb5671e6f54305a1c526e2f0a76747942ed91c` (50,608 rows, cycles 47–79).
- E67 prefix/query: 29,705 / 20,903 rows; E75 prefix/query: 42,117 / 8,491 rows. Prefix and query IDs/cycles matched the frozen manifests.
- Both origin manifests, all 16 matrix files, prediction bundles, C2_B bag artifacts/receipts, 19 historical source pins, and the shared historical budget-ledger hash matched their frozen values.
- Shared historical budget ledger was not modified. Test, SampleSubmission, Kaggle, and target values were not accessed.
- Available RAM was 6,744,518,656 bytes (6.28 GiB); the preflight's 2 GiB estimate left 4,597,035,008 bytes projected free. Both initial and projected reserve checks passed at this snapshot.
- `ROOT_SOURCE_PASS.json` is absent, and no official fit, E79 fit, or inference was started.

## Verification

- `python -m pytest tests/test_pos_frequency_ridge_v1.py -q`: **24 passed**. This includes a separate-process shared-lock contention test, held-lock and low-RAM no-reservation checks, actual `_run_one_fit` fault injection after gamma/receipt publication, map publication fault injection, no-recompute resume, complete saved-receipt binding mutations for source/offset/penalty, missing-receipt blocking, and exclusive-write preservation.
- `python scripts/optimization_v3_cls_pos_frequency_ridge_v1.py --synthetic-qa`: **PASS**. Synthetic optimization converged; map recovery computed the map once; gamma recovery appended the missing terminal event and validated the saved digest without another optimizer call. No official inputs or labels were opened.
- `python scripts/verify_optimization_v3_cls_pos_frequency_ridge_v1.py --source-preflight reports/optimization_v3/cls_pos_frequency_ridge_v1/SOURCE_PREFLIGHT_R2.json`: **PASS_SOURCE_PREFLIGHT_INDEPENDENT**.
- `python -m py_compile ...` and `git diff --check`: passed for the runner, verifier, model module, and tests.

The repaired executable and tests are committed locally in `721edc30344e62f7365baf91e533080ffe0bf7cc`. This source-readiness checkpoint has not been pushed. No source review pass has been issued, and no official project reservation or fit slot was consumed; temporary ledgers used only synthetic rows. The existing four-arm E67/E75 pilot recipe remains unchanged. The prior user authorization for one E79 benchmark is recorded, but this R1 repair checkpoint does not execute it.
