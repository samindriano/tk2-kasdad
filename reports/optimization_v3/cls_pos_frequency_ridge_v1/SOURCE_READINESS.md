# CLS_POS_FREQUENCY_RIDGE_V1 — source-ready checkpoint

Status: **source preparation and target-free preflight passed; official fit is blocked** until a committed `ROOT_SOURCE_PASS.json` and the RAM gate pass.

## Frozen pilot scope

This implementation prepares the four predictor fits already specified by the source-review handoff: `E67_ALIGNED`, `E67_PERMUTED`, `E75_ALIGNED`, and `E75_PERMUTED`. It reuses the original POS_BAG_OFFSET unigram three-third matrices and same-origin C2_B bag state. Two new prefix-only document-frequency penalty maps are accounted separately. The objective, normalization, permutation seed, optimizer, and no-retry acceptance rule are recorded in [protocol.json](../../../experiments/optimization_v3/cls_pos_frequency_ridge_v1/protocol.json).

The official Train hash is `875552a68a3d24b0f3e1966d6ddb5671e6f54305a1c526e2f0a76747942ed91c` (50,608 rows, cycles 47–79). Target-free identity checks matched the frozen populations: E67 prefix 29,705 / query 20,903; E75 prefix 42,117 / query 8,491. Both matrix manifests, all eight matrices per origin, prediction-bundle byte hashes, bag states/receipts, 19 historical source pins, and the shared-ledger reference matched their frozen hashes.

## QA and current gates

- `python -m pytest tests/test_pos_frequency_ridge_v1.py -q`: **14 passed**.
- `python scripts/optimization_v3_cls_pos_frequency_ridge_v1.py --synthetic-qa`: **PASS**; optimizer converged, objective decreased, missing fallback remained exact, and restart reused the durable receipt without another fit.
- `python scripts/verify_optimization_v3_cls_pos_frequency_ridge_v1.py --source-preflight reports/optimization_v3/cls_pos_frequency_ridge_v1/SOURCE_PREFLIGHT.json`: **PASS** independent re-hash of source, Train, manifests, matrices, and prediction bundles plus Train ID/cycle mapping.
- `python -m py_compile src/models/pos_frequency_ridge_v1.py scripts/optimization_v3_cls_pos_frequency_ridge_v1.py scripts/verify_optimization_v3_cls_pos_frequency_ridge_v1.py`: **PASS**.

The last target-free preflight is [SOURCE_PREFLIGHT.json](SOURCE_PREFLIGHT.json). It opened only Train `record_id` and `cycle_code`; `artifact_signal` values were not read. Test, SampleSubmission, Kaggle, and official fitting were not accessed.

At the latest preflight, available RAM was **3,500,994,560 bytes (about 3.26 GiB)**. The conservative 2 GiB working-set estimate leaves only **1,353,510,912 bytes (about 1.26 GiB)** projected free. Both the required 4 GiB initial minimum and 2 GiB projected reserve failed. This is **NO-GO for any official fit**. The runner rechecks memory before loading matrices and immediately before each fit. `ROOT_SOURCE_PASS.json` is also absent, so official fitting remains source-gated.

The source code is committed locally in `73e813ea68a163545ad965f8496d7e3e07293684`, `27a2f9e6c4acec43675cffaca4883882c0a948ab`, and `7fa1041fde627c50c484a8fa4115756ea780c759`. The independent preflight report binds the latest executable source commit and hashes; the report itself is committed in the following checkpoint commit. No push was made.

## E79 authorization note

Sam replied: **“Ya, otorisasi satu benchmark E79”**, authorizing one Classification E79 benchmark with one gamma fit on all Train and Test inference, without a Kaggle upload. That authorization does not silently rewrite the already frozen four-fit pilot recipe. The E79 fit and inference have not been run in this source-ready pass. Sol should bind any E79 amendment to its own explicit recipe/source review before its official target access.

## Root review and resume

Before any official fit, Sol must commit `ROOT_SOURCE_PASS.json` binding `source_commit`, the exact five `source_hashes` in `SOURCE_PREFLIGHT.json`, and `source_preflight_sha256`. The runner checks that pass, the committed preflight, unchanged Train/cache/shared-ledger hashes, a single active process, two-thread execution, the four-ID cap, and current RAM. It creates a new exclusive ledger under the ignored runtime namespace and does not modify the historical shared budget.

After root pass and a fresh resource pass, the next command is:

```powershell
python scripts/optimization_v3_cls_pos_frequency_ridge_v1.py --run-next-fit E67_ALIGNED
```

Run one fit per invocation. A failed predictor fit consumes its slot and stops the pilot for root review. No fit should be started while available RAM is below 4 GiB.
