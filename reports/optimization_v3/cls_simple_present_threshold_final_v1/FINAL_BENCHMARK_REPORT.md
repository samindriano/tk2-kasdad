# Final A02 Past-OOF Threshold Benchmark

Status: **completed; experimental benchmark only**
Campaign: `CLS_SIMPLE_PRESENT_THRESHOLD_FINAL_V1`

## Scope and recipe

The authorized benchmark calibrated only the present-stream decision threshold. It did not fit a predictor or preprocessing component. The selected rows were 20,903 unique past-OOF records: 12,412 from E67 cycles 68–75 and 8,491 from E75 cycles 76–79. Labels came from the matching official Train rows. OOF target members were not opened, and Test/SampleSubmission target values were not read during calibration.

The decision rule uses the frozen interval `[0.45, 0.55]`, minimum support of 200 per class, exact Macro F1 comparisons for labels `[0, 1]`, and the frozen tie-breaking order. The resulting present-stream threshold is **0.4854367183801688** (hex `0x1.f11652b6e1ad2p-2`; optimal decision-state interval `[0.48541374896069867, 0.4854367183801688]`). The missing-stream decision remained fixed at `pA >= 0.53230976129477`.

The reported calibration objective is Macro F1 **0.8239633574** (`77977941/94637632`) on those same selected rows. This is an **in-sample calibration objective**, not an untouched validation estimate. Fit confusion counts were TP 5,037, FP 1,498, TN 12,678, and FN 1,690. Present-stream class support was 13,288 class-0 rows and 6,240 class-1 rows.

## Candidate CSV and independent QA

Candidate: `submissions/optimization_v3/cls_simple_present_threshold_final_v1/classification_threshold_final.csv`

- 29,421 rows, with IDs in official Test order and SampleSubmission schema.
- 6,815 positive and 22,606 negative predictions; 1,062 missing-stream rows retained the frozen missing-stream decision.
- CSV SHA-256: `409479edf1614306289144e6c05db85b5a8a86aafadf7a7f87882884b6fbb006`.
- Prediction-vector SHA-256: `05098b8ec8ca726b26c356b98f1e486a04e003c099feaa931991fb068aff76ac`.

Independent QA passed 14 checks, including source/input hashes, root authorization lineage, exact OOF row membership, the threshold optimum, durable fit accounting, Test-ID ordering, missing-mask equality, CSV replay, fixed missing-route invariance, comparisons to prior vectors, and duplicate-vector scan. No local duplicate prediction vectors were found. The final QA receipt is `FINAL_INDEPENDENT_QA.json` (SHA-256 `2ab38c5e3c690433eb912d9be491fee24934c4c0f729e02bfa6ce7be4c3fc205`). The focused test suite passed **5 tests**; `py_compile` passed for the wrapper, verifier, and tests.

The applied candidate differs from the current public champion in 384 decisions (+56 positives), from prior A02 in 149 (+113), and from NO_POS in 131 (+131). None of the prediction vectors is identical. The champion was not changed.

## Kaggle result — separate from internal calibration

One upload was made through the authenticated Kaggle CLI under the single-upload authorization:

- Competition: `tp2-classification-cyber-vulnerability-kasdd-gasal-2026-2027`.
- Submission ref: **58029412**; status: **COMPLETE**.
- Public score: **0.82984**; private score was not returned.
- Message: `Samuel CLS Simplified A02 Past OOF Threshold Benchmark`.

This public score is +0.00033 versus NO_POS ref 58024217 (0.82951), and −0.00241 versus the current public champion ref 58001282 (0.83225). These leaderboard values are reported separately from the in-sample calibration objective. They were not used for further tuning. The champion remains unchanged, and no further upload is authorized by this benchmark.

The live quota was 1 submission today / 24 total / 9 allowed before upload and 2 today / 25 total / 8 allowed after it. Preflight confirmed team count 5, user entry, the competition deadline, no exact-message duplicate, and no local duplicate candidate vector. The upload result and post-upload snapshots are preserved alongside this report.

## Lineage and stop point

The operative pre-fit freeze is `PRE_FIT_FREEZE_V2.json` (SHA-256 `9d16166e279a0bb903de7e72a7a53b31afedf4683c845f4e803ff87657715f45`), source commit `127394921ee485e960dc4f0c7adf093f911a40d8`. V1 remains preserved as superseded evidence; no official fit or label read occurred under V1. The final upload receipt is `KAGGLE_UPLOAD_RESULT.json` and records one authorized upload with status `COMPLETE`.

The candidate is a benchmark, not a newly qualified champion. This milestone ends without another fit, upload, retuning, or champion change.
