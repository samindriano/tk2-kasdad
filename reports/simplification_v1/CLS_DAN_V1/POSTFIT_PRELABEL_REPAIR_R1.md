# CLS_DAN_V1 post-fit prelabel repair R1

## Why this adapter exists

The four authorized fits completed and their saved models reproduced every saved query probability exactly. Prelabel construction then stopped before reading validation labels because the original runner required E67 and E75 champion probabilities to be identical for their overlapping late query IDs. Those are origin-specific temporal baselines trained from different prefixes, so the scores may differ even when the record ID, source row, cycle, and stream availability agree.

The root authorization is recorded in `ROOT_PRELABEL_REPAIR_R1.json`. The original fit runner, freeze, fit receipts, model binaries, and prediction binaries remain unchanged. The adapter bypasses only that one cross-origin probability equality guard, after independently verifying row ID, Train row position, cycle, and availability alignment. It then verifies each origin's entire champion input against that origin's hash-bound feature cache and checks each missing-stream fallback against that same origin's champion.

The first adapter invocation completed the frozen target-free builder and wrote both prelabel files, then stopped at an adapter postcondition that incorrectly expected one guard call. The frozen guard is inside the per-row overlap loop, so it is called 8,491 times for 8,491 aligned E75 query rows. Recovery reuses and independently verifies those existing files; it does not rebuild them. The completion receipt records the initial adapter commit and the corrected per-row count.

## Binary artifact policy

`PRELABEL.npz` contains raw row arrays and remains ignored and untracked. A committed `PREDICTIONS_READY.json` binds the exact NPZ SHA-256 and records that query labels and metrics have not been read. The separately committed `POSTFIT_PRELABEL_R1.json` binds both hashes, the adapter commit, authorization, fit receipts, and per-origin cache checks. The adapter substitutes the original committed-file guard only for this exact NPZ; every other committed-file check is unchanged.

## Validation

Before official prelabel construction, run:

```powershell
python -m pytest tests\test_cls_dan_v1_postfit_repair_r1.py tests\test_cls_dan_v1.py -q
python scripts\cls_dan_v1_postfit_repair_r1.py synthetic --output-dir "$env:TEMP\cls_dan_v1_postfit_repair_r1"
```

The synthetic end-to-end check uses 20,903 generated records, 8,491 overlapping late rows, deliberately different E67/E75 champion probabilities, and the production arm, begin, and scoring functions. It requires exactly one synthetic target-vector read, matching independent metrics, and no Test, SampleSubmission, or Kaggle access. Additional tests reject row, ID, cycle, availability, and prelabel hash mismatches. Synthetic scores are not research results.
