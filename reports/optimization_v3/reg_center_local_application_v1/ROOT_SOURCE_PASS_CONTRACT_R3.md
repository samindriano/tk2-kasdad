# Root source-pass contract — FULLREG center-local R3

This contract does not authorize execution. The root must independently review
the R3 source, receipts, prior R2 receipts, and synthetic QA before creating
`experiments/reg_center_local_application_v1/ROOT_SOURCE_PASS_R3.json`.

## Exact pass object

The JSON must contain exactly these keys, including `signature`:

```json
{
  "verdict": "PASS",
  "experiment_id": "REG_CENTER_LOCAL_APPLICATION_V1_R3",
  "pass_kind": "ONE_LOCAL_SCALAR_FIT",
  "producer_source_commit": "<producer commit from the frozen final application report>",
  "producer_final_report_sha256": "<sha256 from PREFIT_FREEZE_R3.json>",
  "preparation_freeze_sha256": "<sha256 of PREFIT_FREEZE_R3.json>",
  "identity_preflight_sha256": "<sha256 of IDENTITY_PREFLIGHT_R3.json>",
  "source_inventory_sha256": "<source_inventory_sha256 from both R3 receipts>",
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
and separators `(',', ':')`.

## Bound action

The pass opens exactly one closed-form scalar fit on the pinned OOF archive.
The universe is 20,903 unique rows; 19,528 present rows are eligible and 1,375
missing rows are excluded. Within present rows, only `b(raw_DIRECT) > 0` rows
determine delta. `b == 0` rows are choice-independent. The FULLREG hinge,
triangle center 50/half-width 10, and bounds `[-2, 0]` are fixed.

The pass grants no base/preprocessing fit, LP, global fit, old R06 model, or
retry after an ambiguous reservation. It grants no E79, Test, SampleSubmission,
CSV, or Kaggle access. A later application or submission requires a separate
root decision after the scalar fit and its verification.

## Review checks

Confirm that both `SOURCE_PREP_BLOCKED_R1.json` and the R2 source receipts are
byte-preserved. Confirm all R3 source/input hashes, the final HINGE bundle and
producer report hashes, the 19,528/1,375 population split, unique OOF IDs and
origin/index pairs, origin/cycle causal boundaries, 4,125 present positive-
basis rows, and `PASS_INDEPENDENT_SOURCE_READY_VERIFICATION`.

The producer commit must be an ancestor of the current checkout. Current HEAD
need not equal the historical producer commit because all relevant source and
input bytes are content-pinned.
