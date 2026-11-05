# E79 application root-pass contract R3

The runner accepts exactly one canonically signed JSON object at
`experiments/reg_center_local_application_v1/E79_APPLICATION_ROOT_PASS_R3.json`.
The signature is SHA256 of the unsigned object serialized as UTF-8 JSON with
sorted keys, no NaN, no insignificant whitespace, and Unicode preserved.

The object must contain exactly these keys:

```text
verdict
experiment_id
pass_kind
application_freeze_sha256
application_preflight_sha256
source_inventory_sha256
local_fit_state_sha256
independent_fit_qa_sha256
e79_archive_sha256
output_csv_path
authorized_actions
base_model_fits
calibrator_fits
e79_prediction_access_authorized
e79_target_values_authorized
test_or_sample_access_authorized
csv_creation_authorized
kaggle_upload_authorized
signature
```

Exact values are derived from the R3 source freeze and preflight receipts:

- `verdict`: `PASS`
- `experiment_id`: `REG_CENTER_LOCAL_E79_APPLICATION_V1_R3`
- `pass_kind`: `FINAL_E79_LOCAL_APPLICATION`
- all SHA256 and output path fields: exact values from the source freeze/preflight
- `authorized_actions`: `READ_E79_ALLOWLIST`, `CREATE_LOCAL_CSV`,
  `VERIFY_LOCAL_CSV`, in that order
- `base_model_fits` and `calibrator_fits`: `0`
- `e79_prediction_access_authorized` and `csv_creation_authorized`: `true`
- `e79_target_values_authorized`, `test_or_sample_access_authorized`, and
  `kaggle_upload_authorized`: `false`

No application pass is included in this source-preparation deliverable. The
production E79 archive and CSV remain untouched until the authorized pass is
provided through the project root approval process.
