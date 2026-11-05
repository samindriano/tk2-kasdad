# E79 global-strength gamma continuation

## Purpose and stop boundary

This package prepares exactly one full-Train fit for `E79_GLOBAL_STRENGTH` and
a later, separately gated Test application. It preserves the frozen E79
recipe. This source-preparation checkpoint does not authorize itself: a root
source pass must be reviewed and committed before the official gamma fit, and a
separate application pass must be reviewed and committed before Test features,
the saved A02 Test probability cache, or a CSV are read or created.

At this checkpoint, only the verified Train-only DF map has been reused. The
gamma runner has not read Train target values, fit gamma, read Test or
SampleSubmission, created a submission CSV, or contacted Kaggle.

## Frozen numerical recipe

The full official Train population is 50,608 rows in cycles 47 through 79.
Missing-stream rows remain in the fit population; their positional feature
rows are zero and their `C2_B` offset remains available.

The base matrix is the exact frozen `C2_B` structured representation, saved
A02 TF-IDF matrix, and one missing-stream indicator. The positional matrix is
the frozen three-segment unigram representation. Both are rebuilt from
feature-only Train columns and must match the R3 exact shape, nonzero count,
ordered-ID digest, and logical matrix digest before the target-only read.

For binary `C2_B` with `classes_ == [0, 1]`, the offset is the positive-class
logit

```text
z0 = X_base @ C2_B.coef_[0] + C2_B.intercept_[0]
```

The intercept is added. `pA` is not used as an offset. The optimizer uses
`Xpos @ gamma` as a correction to this frozen logit. For each row `i`, with
`y_i = artifact_signal_i`, the exact frozen objective is

```text
sum_i log(1 + exp(-(2*y_i - 1) * (z0_i + Xpos_i @ gamma)))
    + 0.5 * dot(W, gamma**2)
```

The initial gamma vector is all zeros. The penalty is built from the verified
Train DF map as `v = max(1, aligned_weight)`, `c = mean(v)`, and
`W = full(3 * V, c)`. The fit calls the unchanged
`src/models/pos_frequency_ridge_v1.py` implementation once, with SciPy
L-BFGS-B, analytic gradient, and its frozen options (`maxiter=2000`,
`ftol=1e-12`, `gtol=1e-6`, `maxls=50`). No retry is permitted after the
official attempt marker is written.

For a later Test application, present-stream probability is fixed as

```text
0.5 * sigmoid(z0 + Xpos @ gamma) + 0.5 * frozen A02 pA
```

Missing-stream probability is the exact frozen A02 `pA` fallback, preserved
bit-for-bit. The present threshold is `0.50`; the missing-stream threshold is
`0.53230976129477`; both use `>=`. This E79 policy does not add the older
POS_BAG_OFFSET missing fallback.

## Frozen upstream evidence

`GAMMA_SOURCE_FREEZE.json` binds the gamma runner, independent verifier, pure
recipe helper, this protocol, synthetic tests, numerical optimizer, R3/CGeom
matrix builders, A02/C2_B fitted states, immutable Train and upstream files,
the V2 DF-map source freeze, its map artifact and receipt, and its persisted
independent verification receipt. Source freeze construction requires these
package files already be committed and the DF-map independent verifier to pass.

The source root pass must be committed at
`GAMMA_ROOT_SOURCE_PASS.json`. It must bind the exact gamma source freeze and
its file hash, the reviewed commit containing that freeze, the sole fit ID
`E79_GLOBAL_STRENGTH`, the Train SHA-256, and explicit denials for Test,
SampleSubmission, and Kaggle access. Its JSON `pass_sha256` is the canonical
SHA-256 of the remaining object. The runner fails closed if the pass is absent,
modified, uncommitted, or does not authorize exactly one full-Train gamma fit.
Required fields:

```json
{
  "schema_version": 1,
  "status": "PASS_E79_GAMMA_SOURCE_ROOT_REVIEW",
  "reviewed_head": "<commit containing GAMMA_SOURCE_FREEZE.json>",
  "gamma_source_freeze_sha256": "<freeze_sha256>",
  "gamma_source_freeze_file_sha256": "<stable file SHA-256>",
  "one_full_train_gamma_fit_authorized": true,
  "authorized_fit_ids": ["E79_GLOBAL_STRENGTH"],
  "authorized_train_sha256": "875552a68a3d24b0f3e1966d6ddb5671e6f54305a1c526e2f0a76747942ed91c",
  "authorized_train_rows": 50608,
  "artifact_signal_access_authorized": true,
  "test_inference_authorized": false,
  "sample_submission_access_authorized": false,
  "kaggle_access_authorized": false,
  "pass_sha256": "<canonical SHA-256 of all other fields>"
}
```

The application pass is separately committed at
`GAMMA_APPLICATION_ROOT_PASS.json`. It must bind the gamma fit receipt file,
the A02 Test probability NPZ and receipt, and the official Test CSV hashes. It
must explicitly authorize one Test inference and deny SampleSubmission and
Kaggle access. Its canonical `pass_sha256`, committed file identity, and
reviewed freeze commit are checked before the runner reads Test or A02 Test
probability values.
Required fields:

```json
{
  "schema_version": 1,
  "status": "PASS_E79_GAMMA_APPLICATION_ROOT_REVIEW",
  "reviewed_head": "<commit containing GAMMA_SOURCE_FREEZE.json>",
  "gamma_source_freeze_sha256": "<freeze_sha256>",
  "gamma_source_freeze_file_sha256": "<stable file SHA-256>",
  "gamma_fit_receipt_file_sha256": "<official fit receipt SHA-256>",
  "a02_prediction_npz_sha256": "<A02 Test probability archive SHA-256>",
  "a02_test_receipt_sha256": "<A02 Test receipt SHA-256>",
  "official_test_file_sha256": "<official Test CSV SHA-256>",
  "test_inference_authorized": true,
  "one_test_inference_authorized": true,
  "test_target_access_authorized": false,
  "sample_submission_access_authorized": false,
  "kaggle_access_authorized": false,
  "pass_sha256": "<canonical SHA-256 of all other fields>"
}
```

## Accounting and persistence

The one gamma run requires both the project shared heavy-job lock and the CGeom
local training lock. It is limited to two computational threads. Before the
feature build it requires at least 2 GiB available RAM, based on a 1 GiB
incremental estimate plus a 1 GiB reserve. The reserve is checked again after
the feature build and after the fit. These are conservative gates, not a
measured gamma peak claim.

The official fit writes one attempt marker, one `gamma.npz`, and one
`fit_receipt.json` beneath
`experiments/runs/optimization_v3_cls_pos_global_strength_e79_v3/gamma_v1/`.
The receipt binds source commit/pass/freeze, Train and exact matrix hashes,
C2_B coefficient and intercept hashes, offset, aligned weights, target hash,
gamma hash, optimizer output, fit population, resource observations, and
source-boundary flags. The read-only verifier reads the receipt and gamma
artifact only; it does not reload target values or replay the fit.

Test application has its own one-shot attempt marker, NPZ, receipt, and fixed
CSV path. The independent verifier compares that CSV against persisted IDs and
labels, verifies row count/order/schema, checks finite probabilities and
frozen thresholds, and checks the exact missing-stream A02 fallback. It does
not open Test target values, SampleSubmission, or Kaggle.

## Synthetic end-to-end QA

The synthetic CLI uses tiny generated arrays and a caller-provided temporary
directory. It performs the real `fit_offset` optimizer call, persists gamma
and a receipt, independently reads them back, resumes by read-only verification
without another fit, applies the routing and thresholds to synthetic rows,
creates a synthetic CSV, and independently verifies the CSV. The command does
not load repository datasets or model artifacts.

From the repository root:

```powershell
python -m pytest -q tests/test_optimization_v3_cls_pos_global_strength_e79_gamma_v1.py
$qa = Join-Path $env:TEMP 'kasdad-e79-gamma-synthetic-qa'
python -m scripts.optimization_v3_cls_pos_global_strength_e79_gamma_v1 --synthetic-e2e --output-dir $qa
python -m scripts.verify_optimization_v3_cls_pos_global_strength_e79_gamma_v1 --verify-synthetic-e2e --output-dir $qa
python -m scripts.optimization_v3_cls_pos_global_strength_e79_gamma_v1 --resume-synthetic-e2e --output-dir $qa
```

The synthetic output directory is one-shot. A second `--synthetic-e2e` run
must fail instead of fitting again.

## Future commands after review

After a root reviewer creates, commits, and verifies the exact source pass:

```powershell
python -m scripts.optimization_v3_cls_pos_global_strength_e79_gamma_v1 --verify-freeze
python -m scripts.optimization_v3_cls_pos_global_strength_e79_gamma_v1 --fit-gamma
python -m scripts.verify_optimization_v3_cls_pos_global_strength_e79_gamma_v1 --verify-fit
```

Stop and review the official fit receipt. Only after a distinct committed
application root pass is approved may Test application run:

```powershell
python -m scripts.optimization_v3_cls_pos_global_strength_e79_gamma_v1 --apply-test
python -m scripts.verify_optimization_v3_cls_pos_global_strength_e79_gamma_v1 --verify-application
```

Neither sequence authorizes a Kaggle upload. There is no model-selection
conclusion in this source-only package.
