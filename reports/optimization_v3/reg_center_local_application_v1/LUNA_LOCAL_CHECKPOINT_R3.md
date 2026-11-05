# Luna local checkpoint — FULLREG center-local application R3

Date: 2026-11-06 (Asia/Jakarta)

## Status

`SOURCE_READY_PENDING_ROOT_SOURCE_PASS`. R3 source preparation is complete and
independently verified twice, including receipt readback. The root pass does
not exist. No real target array was loaded, no real fit started, and no
application/submission occurred. Stop here pending root review and the signed
pass defined in `ROOT_SOURCE_PASS_CONTRACT_R3.md`.

R2 artifacts are preserved and pinned as inputs to R3. The R2 initial
source-verification call passed, but a subsequent idempotence replay exposed a
readback helper defect. R3 repairs that code path and its verifier passed twice
against the same immutable receipt. The original blocked R1 receipt remains
unchanged, SHA256
`a3c9377ba5a802c4f0ea84ef3589fcaf20a8932016e2d37e6f0368d1fc74eb0f`.

## R3 immutable receipts

| Artifact | Status | SHA256 |
|---|---|---|
| `experiments/reg_center_local_application_v1/PREFIT_FREEZE_R3.json` | `SOURCE_FROZEN_PENDING_NON_TARGET_PREFLIGHT` | `752ecaec45734f7261eeb6b0a67f62d22a87158e99376b6bc2881486fae01538` |
| `reports/optimization_v3/reg_center_local_application_v1/IDENTITY_PREFLIGHT_R3.json` | `PASS_NON_TARGET_IDENTITY_PREFLIGHT_PENDING_ROOT_SOURCE_PASS` | `ca908690b4b1e73c68669e346b11b847b1848399ae3c69eaa0fee0cf4fdd9c74` |
| `reports/optimization_v3/reg_center_local_application_v1/SOURCE_READY_VERIFICATION_R3.json` | `PASS_INDEPENDENT_SOURCE_READY_VERIFICATION` | `38307fd9cedfeffa415d39c652c8ba38483cf1d215a0b7decc2068aa61d88bf0` |

R3 source/input inventory digest: `e2730d458ed8d2cce11d6c0bb52f4f6acbacacddaaeae26996a70006e2be6bb9`.

R3 pins the three R2 receipt files at these hashes: freeze
`87c87b6e29e9a1638b615921e3dd0178388caceef7bd202bccdfe639993199d6`, preflight
`68e08cde4687d66ddebb9c2b9bc3c6cff2f52ed64340e5e479815a70516902b0`, and
source verification
`8b13e5d233ff22c6e110ee50212066d9d679a73f61b4917f94b83925495b875f`.

The frozen FULLREG producer commit is `ac0f84905d35080837cda6bd0da5394073fce2c1`,
verified as an ancestor of checkout HEAD `2ef76694f2129042e8d7133b1568c4c5aa3d5257`.
R3 binds current source/input bytes and producer ancestry; it does not require
current HEAD equality. The pinned HINGE OOF bundle is
`e75f70cccd1bb231499060b84155af876173214d984461e39e4c1364e4c5b2bf`; parent
final report is
`fec7182eace04b425b7dbf1d871dbb39f14a1080e479ba95243803e4b1cdae73`; final
fit state is
`a81b2e8e709a33e9a15036ab040c87c92faabd08f16b48ff7f42916fd23ad99a`.

## Resolved population and causal checks

- OOF universe: 20,903 unique record IDs, physical rows, and origin/bundle
  index pairs.
- Present and eligible: 19,528 rows; missing and excluded: 1,375.
- Present rows with `b(raw_DIRECT) > 0`: 4,125; present `b == 0` rows: 15,403,
  which have no choice-dependent effect on delta.
- Origin 67: 12,412 rows, 11,563 present, cycles 68–75.
- Origin 75: 8,491 rows, 7,965 present, cycles 76–79.
- Every origin is strictly earlier than its target cycle.
- Preflight loads only `physical_rows`, `record_ids`, `prediction_origins`,
  `original_bundle_indices`, `cycles`, `direct_predictions`, and
  `missing_mask`. The archive's `targets` member was not indexed.
- The fixed coefficients are intercept `1.7288079036674198`, low slope
  `0.9933285599956824`, and high slope `0.957778072288735`.

## Verification

- `python -m py_compile scripts/reg_center_local_application_v1.py scripts/verify_reg_center_local_application_v1.py tests/test_reg_center_local_application_v1.py` — PASS.
- `python -m unittest tests.test_reg_center_local_application_v1 -v` — 4 tests PASS.
- Synthetic CLI run covered target-blind freeze/preflight, independent source
  verification, one isolated synthetic scalar fit, durable state/receipt,
  readback/resume without reloading targets, exact weighted-median ties,
  missing-route bit preservation, and fit-state/receipt tamper rejection.
- A synthetic malformed `targets.npy` member remained unread in the
  non-target preflight path.
- Real R3 `--freeze`, `--preflight`, and `--verify-source` — PASS. Repeating
  `--verify-source` returned the same signed receipt and SHA256 — PASS.
- No real fit, E79/Test/SampleSubmission read, CSV, upload, shared RAM lock, or
  Kaggle action occurred. Luna1's last resource note said POSDF replay had not
  started and RAM available was 3.10 GiB; this lane did not consume the lock.
- No files were staged or committed. Existing worktree changes remain intact.

## Next action after the communication hold

Root reviews R3 receipts, source inventory, policy, pass contract, runner,
independent verifier, tests, and the preserved R2/R1 history. If accepted, root
creates `experiments/reg_center_local_application_v1/ROOT_SOURCE_PASS_R3.json`
with the exact signed keys from the contract. Only then may the one scalar fit
and its readback verification run. A separate root decision is required for
any E79/Test application, CSV, or upload.

No message was sent to Sol during the temporary communication hold; this file
is the local checkpoint.
