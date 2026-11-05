# Fixed-p50 hinge final application R1

## Scope and decision status

This source prepares one exploratory application of the fixed-knot-50 monotone hinge to the pinned E79 raw DIRECT predictions. It uses the existing 20,903 unique OOF rows and can consume only the already-accounted third cumulative slot, `HINGE_FINAL`, after a separate new root source pass.

R4 did not meet the qualified scientific threshold: its qualified gate is **FAIL**. The two exploratory windows had positive gains of `0.007355896` and `0.005485018`, while p90 absolute error worsened by `0.01998969392750638` and `0.017396558419765284`; high-target subgroup harm was `0.1324871651977233` and `0.02145241117350949`. The final application remains exploratory and pending the user's later benchmark-batch selection. It makes no promotion or upload claim.

## Immutable source preparation

The frozen source set is the R1 runner, independent verifier, tests, this protocol, synthetic QA receipt, `.gitattributes`, and the existing R4 helper sources. Generate the synthetic QA receipt before freezing source hashes. Commit the reviewed source set locally, then create the immutable freeze:

```powershell
python tests/test_reg_monotone_hinge_p50_final_application_v1.py
python scripts/reg_monotone_hinge_p50_final_application_v1.py --synthetic-qa
python scripts/reg_monotone_hinge_p50_final_application_v1.py --freeze-final-application
```

The freeze must show exactly two existing R4 starts, no `HINGE_FINAL` reservation, a total cap of three, the inherited Train raw-byte hash, and a separate preflight allowlist. Hashing Train bytes verifies the frozen source pin; it does not parse target values. The freeze does not read target labels, Test/Sample, or CSV content, and does not reserve a fit or create a candidate CSV.

## Non-target identity preflight

Before the real preflight, coordinate with Luna1 about the shared heavy-job lock and current RAM. Then run:

```powershell
python scripts/reg_monotone_hinge_p50_final_application_v1.py --identity-preflight
```

The preflight obtains the shared lock and checks the existing R4 source pass against every pinned input hash. It rehashes the exact R1 preflight allowlist, including the raw Train file, verifies the 20,903-row OOF identity, and reads only `record_id`, `prediction`, `baseline`, and `missing` from the pinned E79 inference archive. It validates the signed E79 manifest and QA and checks the public helper API contracts. It must report zero target parses, zero LP reservations/starts, no R4 ledger modification, no CSV creation, and no Test/Sample access.

If the preflight cannot verify any pin, source, row identity, resource boundary, ledger prefix, or API contract, stop with the failure evidence. Do not fill missing evidence or alter a frozen artifact to make the preflight pass.

## Required new root source pass

The new pass is a separate review artifact at `experiments/reg_monotone_hinge_p50_v1/FINAL_APPLICATION_ROOT_SOURCE_PASS_R1.json`. It must bind the exact R1 source commit, source hashes, R1 freeze hash, identity-preflight hash, full input hashes, R4 diagnostic pass/hash, and the unchanged two-event historical ledger prefix. It must authorize exactly:

- `pass_kind`: `FINAL_APPLICATION`;
- `authorized_fit_ids`: `["HINGE_FINAL"]`;
- `authorized_final_lp_starts`: `1`;
- `prior_official_lp_starts`: `2`;
- `total_official_lp_start_cap`: `3`;
- candidate CSV creation after that fit;
- no Test/Sample access and no Kaggle upload.

The R4 diagnostic pass authorizes D1/D2 only. It cannot substitute for this pass. This R1 workflow does not create the root pass.

## Post-pass execution contract

Only after the new pass is independently checked by the runner may `--run-final-application` be used. It reads the selected 20,903 Train target values, starts `HINGE_FINAL` once, and appends its events after the exact preserved R4 ledger prefix. The fit is the unweighted L1 hinge `a + sL*min(p,50) + sH*max(p-50,0)`, with both slopes in `[0,2]`, two solver threads, and the existing 180-second limit. A failed or ambiguous started slot is terminal; no retry or counter reset is permitted.

For application, present raw DIRECT values are transformed by the fitted hinge and clipped to `[0,100]`. The 1,062 missing rows retain raw DIRECT float64 bits exactly. The independently verified CSV has exactly 29,421 unique ordered IDs and columns `record_id,threat_rank`. No D1/D2 refit, GLOBAL control refit, base-model fit, Test/Sample read, benchmark-batch selection, Kaggle upload, or push is in scope.
