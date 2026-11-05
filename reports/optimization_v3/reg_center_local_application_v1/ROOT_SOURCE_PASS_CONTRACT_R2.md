# Root source-pass contract — FULLREG center-local R2

This is a review contract, not an authorization. No `ROOT_SOURCE_PASS_R2.json`
is created by source preparation. The root must independently review the source,
freeze, preflight, and synthetic QA before issuing a pass.

## Exact pass object

The root-created JSON must contain exactly these keys, including `signature`:

```json
{
  "verdict": "PASS",
  "experiment_id": "REG_CENTER_LOCAL_APPLICATION_V1_R2",
  "pass_kind": "ONE_LOCAL_SCALAR_FIT",
  "producer_source_commit": "<producer commit from the frozen final application report>",
  "producer_final_report_sha256": "<sha256 from PREFIT_FREEZE_R2.json>",
  "preparation_freeze_sha256": "<sha256 of PREFIT_FREEZE_R2.json>",
  "identity_preflight_sha256": "<sha256 of IDENTITY_PREFLIGHT_R2.json>",
  "source_inventory_sha256": "<source_inventory_sha256 from both receipts>",
  "authorized_fit_ids": ["LOCAL_CENTER_FINAL"],
  "calibrator_fit_cap": 1,
  "authorized_calibrator_fits": 1,
  "base_model_fits": 0,
  "preprocessing_fits": 0,
  "official_lp_starts": 0,
  "fit_total_oof_rows": 20903,
  "fit_present_rows": 19528,
  "excluded_missing_rows": 1375,
  "target_array_access_authorized": true,
  "e79_access_authorized": false,
  "test_or_sample_access_authorized": false,
  "csv_creation_authorized": false,
  "kaggle_upload_authorized": false,
  "signature": "<canonical sha256 of all preceding keys>"
}
```

`signature` is SHA256 over UTF-8 JSON of the object without the `signature`
field, serialized with sorted keys, `ensure_ascii=false`, `allow_nan=false`,
and separators `(',', ':')`. The implementation exposes the canonical
serialization in `scripts/reg_center_local_application_v1.py`.

## Bound action

The pass opens exactly one closed-form scalar fit on the pinned OOF archive.
The total universe is 20,903 unique rows. Exactly 19,528 present rows are
eligible; the 1,375 missing rows are excluded. Within present rows, only
`b(raw_DIRECT) > 0` rows determine the scalar. Rows with `b == 0` are
choice-independent. The fixed FULLREG hinge, triangle center 50/half-width 10,
and bounds `[-2, 0]` are immutable.

The pass grants no base or preprocessing fit, no LP, no global fit, no old R06
model, and no retry after an ambiguous reservation. It does not grant E79, Test,
SampleSubmission, CSV, or Kaggle access. A later application or submission
requires a separate root decision after the scalar fit and its verification.

## Provenance checks before signing

Confirm that the blocked R1 receipt remains byte-identical and is superseded
only by the explicit population decision in R2. Confirm the R2 source inventory
matches the files on disk, the final HINGE OOF bundle hash and producer report
hash match the preflight, prediction origins and cycles pass the causal checks,
and the independent source verifier reports PASS. Current checkout HEAD need
not equal the historical producer commit; the producer source commit must be an
ancestor, and all relevant source/input bytes are content-pinned.
