# Sol handoff: R06-native hinge pilot R1

Status: source and inputs are ready for root review; no official LP was run.

## Frozen evidence

- Repository: `C:\Users\Sam\OneDrive\Documents\KASDAD\tk2-reg-position-failure-mechanism-v1`
- Freeze: `experiments/reg_r06_native_hinge_v1/PREFIT_FREEZE_R1.json`
  - SHA-256: `350de1cfa1cd122ff8b09222496eee14e5bfae81170c47b8ad1d7db4fb98f984`
- Non-target preflight: `reports/optimization_v3/reg_r06_native_hinge_v1/IDENTITY_PREFLIGHT_R1.json`
  - SHA-256: `307607f1df4ac039bc75085a826524ba0fe24b2da81d547887058f06887366b4`
- Synthetic CLI and receipt E2E: `reports/optimization_v3/reg_r06_native_hinge_v1/SYNTHETIC_QA_R1.json`
- Independent source-ready verification: `reports/optimization_v3/reg_r06_native_hinge_v1/DIAGNOSTICS_VERIFICATION_R1.json`
  - SHA-256: `468ffe9683e90c82a836d6540c9804ef169efdd7eaba66249db2fae1329f3823`
- Protocol: `reports/optimization_v3/reg_r06_native_hinge_v1/PROTOCOL_R1.md`

The source-ready verifier returned `PASS_SOURCE_READY_PENDING_ROOT_RELEASE`. The targeted test file passed all 7 tests. Non-target preflight reports 20,903 unique IDs, 4,950 D1 fit rows, 7,462 D1 score rows, 12,412 D2 fit rows, and 8,491 D2 score rows. Target arrays were not loaded.

## Source and budget decision

Use exact R06 fit67 predictions for cycles 68–75 and exact R06 fit75 predictions for cycles 76–79. Both archive and sidecar hashes match the existing target-free inventory. Fit71 is explicitly excluded. The saved 20,903-row HINGE_FINAL bundle supplies the cached labels, IDs, cycles, missing mask, and original DIRECT predictions. Missing DIRECT rows were verified bit-identical to same-row raw R06.

The prior R4 ledger already contains three LP starts (D1, D2, FINAL). It remains untouched. The new pilot has a separate ledger path and cap of three starts: R06_HINGE_D1, R06_HINGE_D2, and conditional R06_HINGE_FINAL. No new ledger currently exists.

Comparators are frozen as: (a) same-history R06 with the old DIRECT-trained HINGE_FINAL coefficients, primary; (b) original DIRECT predictions with those coefficients, reported separately; (c) raw R06. The final fit is allowed only when both stated diagnostic gates pass. This is an explicit conservative interpretation of the two gates; unsupported groups (n < 100) remain UNKNOWN and prevent a pass. No fresh holdout claim is made.

## Root source-pass request

After reviewing the freeze, preflight, synthetic receipt, source hashes, and this protocol, the root may create `experiments/reg_r06_native_hinge_v1/ROOT_SOURCE_PASS_R1.json` with a canonical signature over its body and these exact bindings:

- `experiment_id`: `REG_R06_NATIVE_HINGE_V1`
- `verdict`: `PASS_SOURCE_AND_INPUTS`
- `prefit_freeze_sha256`: `350de1cfa1cd122ff8b09222496eee14e5bfae81170c47b8ad1d7db4fb98f984`
- `identity_preflight_sha256`: `307607f1df4ac039bc75085a826524ba0fe24b2da81d547887058f06887366b4`
- `source_commit`: the freeze value
- `source_hashes`: exact object in the freeze
- `input_hashes`: exact object in the freeze
- `authorized_fit_ids`: `[R06_HINGE_D1, R06_HINGE_D2]`
- `conditional_final_fit_id`: `R06_HINGE_FINAL`

Only after that root pass should the executor load the cached target array or start an LP. This handoff does not authorize a Test label read, final model selection, CSV creation, or Kaggle access.
