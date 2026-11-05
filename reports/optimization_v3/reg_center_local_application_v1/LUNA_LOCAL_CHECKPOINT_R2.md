# Luna local checkpoint — FULLREG center-local application R2

Date: 2026-11-06 (Asia/Jakarta)

## Status

`SOURCE_READY_PENDING_ROOT_SOURCE_PASS`. Preparation and independent
verification are complete. The root pass does not exist. No real target array
was loaded and no real fit started. The next execution boundary is the root's
review and signed pass defined in `ROOT_SOURCE_PASS_CONTRACT_R2.md`.

The earlier `SOURCE_PREP_BLOCKED_R1.json` is preserved byte-for-byte at
SHA256 `a3c9377ba5a802c4f0ea84ef3589fcaf20a8932016e2d37e6f0368d1fc74eb0f`.
R2 records the population resolution without modifying that historical
receipt.

## Immutable source receipts

| Artifact | Status | SHA256 |
|---|---|---|
| `experiments/reg_center_local_application_v1/PREFIT_FREEZE_R2.json` | `SOURCE_FROZEN_PENDING_NON_TARGET_PREFLIGHT` | `87c87b6e29e9a1638b615921e3dd0178388caceef7bd202bccdfe639993199d6` |
| `reports/optimization_v3/reg_center_local_application_v1/IDENTITY_PREFLIGHT_R2.json` | `PASS_NON_TARGET_IDENTITY_PREFLIGHT_PENDING_ROOT_SOURCE_PASS` | `68e08cde4687d66ddebb9c2b9bc3c6cff2f52ed64340e5e479815a70516902b0` |
| `reports/optimization_v3/reg_center_local_application_v1/SOURCE_READY_VERIFICATION_R2.json` | `PASS_INDEPENDENT_SOURCE_READY_VERIFICATION` | `8b13e5d233ff22c6e110ee50212066d9d679a73f61b4917f94b83925495b875f` |

Source/input inventory digest: `312c4a62c1feb6e76494af9fe6fb4d0b938fac597d90e20bee2be3dd4d38cc90`.

The frozen producer commit is `ac0f84905d35080837cda6bd0da5394073fce2c1`,
verified as an ancestor of checkout HEAD `2ef76694f2129042e8d7133b1568c4c5aa3d5257`.
The gate binds file and producer hashes and does not require current HEAD to
equal the producer commit.

Pinned producer artifacts include final report
`fec7182eace04b425b7dbf1d871dbb39f14a1080e479ba95243803e4b1cdae73`, final
fit-state `a81b2e8e709a33e9a15036ab040c87c92faabd08f16b48ff7f42916fd23ad99a`,
and `HINGE_FINAL.npz`
`e75f70cccd1bb231499060b84155af876173214d984461e39e4c1364e4c5b2bf`.
The frozen FULLREG coefficients are intercept `1.7288079036674198`, low slope
`0.9933285599956824`, and high slope `0.957778072288735`.

## Population and source boundary

- OOF universe: 20,903 unique record IDs and physical rows.
- Present and fit-eligible: 19,528.
- Missing and excluded: 1,375.
- Present `b(raw_DIRECT) > 0` rows informing delta: 4,125; present `b == 0`
  rows: 15,403 and choice-independent.
- Origin 67: 12,412 rows, 11,563 present, cycles 68–75.
- Origin 75: 8,491 rows, 7,965 present, cycles 76–79.
- All 20,903 `(prediction_origin, original_bundle_index)` pairs are unique and
  every origin strictly precedes its target cycle.
- Only these NPZ members were loaded during preparation: `physical_rows`,
  `record_ids`, `prediction_origins`, `original_bundle_indices`, `cycles`,
  `direct_predictions`, and `missing_mask`. The `targets` member was present
  but not indexed. E79/Test/SampleSubmission were not read.

At an authorized fit, use the exact lower weighted median of
`(y - g(raw_DIRECT)) / b` with weights `b`, over present rows with `b > 0`,
and bounds `[-2, 0]`. Missing rows remain excluded and must preserve raw DIRECT
float64 bits at any separately authorized application. The R2 source pass
contract allows one scalar fit only and explicitly denies E79/Test/CSV/Kaggle.

## Validation

- `python -m py_compile scripts/reg_center_local_application_v1.py scripts/verify_reg_center_local_application_v1.py tests/test_reg_center_local_application_v1.py` — PASS.
- `python -m unittest tests.test_reg_center_local_application_v1 -v` — 4 tests PASS.
- Synthetic end-to-end CLI exercised freeze, preflight, independent verification,
  one synthetic fit, durable receipt, readback, resume without a second target
  read, and fit-state/receipt tamper rejection before target access.
- A malformed synthetic `targets.npy` member was ignored by the non-target
  loader, confirming preflight does not parse target data.
- Real `--freeze`, `--preflight`, and `--verify-source` — PASS.
- No shared RAM lock acquired and no heavy fit launched; Luna1 reported the
  POSDF replay was not yet active and had 3.10 GiB free RAM at its last check.
- No real fit, E79/Test read, CSV, or Kaggle action occurred. No root pass was
  authored. No files were staged or committed; pre-existing worktree changes
  remain untouched.

## Root's next action

Review the three immutable R2 receipts, source inventory, policy, root-pass
contract, runner, independent verifier, and four synthetic tests. If accepted,
create `experiments/reg_center_local_application_v1/ROOT_SOURCE_PASS_R2.json`
with the exact signed fields in the contract. Only then may the one fit command
run:

```powershell
python scripts/reg_center_local_application_v1.py --fit
python scripts/reg_center_local_application_v1.py --verify-fit
```

Any later E79/Test application, CSV, or upload requires a separate root
decision. This checkpoint is local; no callback was sent while Sol's temporary
communication hold is active.
