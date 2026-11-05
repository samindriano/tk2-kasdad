# CLS Simple Present Threshold — Final A02 Past-OOF Benchmark

Status: bounded final calibrator, authorized by the root decision at
`../tk2-simplification-audit-v1/reports/simplification_v1/CLS_SIMPLE_THRESHOLD_FINAL_AUTHORIZATION.md`.
This is a one-slot, diagnostic benchmark. It does not replace a champion.

## Frozen scope

- One threshold-only calibration fit; zero predictor and preprocessing fits.
- Calibration uses 20,903 unique OOF records: E67 cycles 68–75 (12,412) and
  E75 cycles 76–79 (8,491). The two pools are concatenated only after exact ID
  and Train cycle checks.
- Read `artifact_signal` only for those exact official Train IDs. OOF target
  members are not opened.
- Optimize full-row binary Macro F1, including the fixed missing-stream route,
  with exact rational comparisons over the inclusive interval `[0.45, 0.55]`.
  Scores use `pA >= threshold`; ties choose the value closest to 0.50, then the
  higher value. Present rows need at least 200 examples of each target class.
- Keep the missing route fixed at `pA >= 0.53230976129477`.
- Commit the source and a pre-fit freeze of Train, OOF, canonical E79, Test,
  SampleSubmission, upstream E79 evidence, root authorization, and comparison
  CSV hashes before the sole official calibration fit.
- `PRE_FIT_FREEZE_V2.json` supersedes the committed V1 pre-fit snapshot because
  the audit repo later advanced with an unrelated regression report; the
  authorization file stayed byte-identical and its authorizing commit remains
  an ancestor. V1 is retained and the record confirms no official fit or label
  read happened under it.
- Do not open E79 probabilities until `fits/FINAL.json` durably freezes the
  learned threshold. Then use only the existing canonical E79 `a02_probability`
  and `missing_mask` arrays. Read Test `record_id` and `signal_stream` only;
  read the SampleSubmission header and IDs only. Never read Test targets or
  SampleSubmission target values.
- Present rows use the learned threshold. Missing rows retain the frozen A02
  decision. Do not fit or infer any model or preprocessing component.
- Independently replay the exact optimum and all CSV labels. Compare the
  candidate vector with current public champion, prior A02, and NO_POS. Scan
  local submissions for an identical vector; an identical prior upload blocks
  another upload.
- One Kaggle CLI upload is allowed only after fresh live rules/team/deadline,
  quota, ledger, duplicate, and authentication checks. Message:
  `Samuel CLS Simplified A02 Past OOF Threshold Benchmark`. Never retry an
  ambiguous request. Record `COMPLETE`, pending, or failure honestly.

## Commands

After `PRE_FIT_FREEZE_V2.json` has been created and committed, run from the repo:

```powershell
python scripts/cls_simple_present_threshold_final_v1.py fit
python scripts/cls_simple_present_threshold_final_v1.py apply
python scripts/verify_cls_simple_present_threshold_final_v1.py
```

The test suite creates isolated synthetic files and invokes the same CLI with
the hidden `--synthetic-fixture` switch. That switch is only for tests and
cannot be used to create the production pre-fit freeze. Fixture paths must all
resolve under the OS temporary directory and outside the repository. The test
path retains the shared/lane locks with a 512 MiB start floor and 256 MiB
reserve; the one official calibration uses the unchanged project gate of
1.5 GiB start, 0.75 GiB reserve, and 512 MiB estimated work.

Runtime receipts and ledger live under ignored
`experiments/runs/cls_simple_present_threshold_final_v1/`. The final CSV is
written once under `submissions/optimization_v3/cls_simple_present_threshold_final_v1/`.
The compact final report is `FINAL_BENCHMARK_REPORT.md`; the independent
machine replay is `FINAL_INDEPENDENT_QA.json`.

## Interpretation

The optimized Macro F1 is a calibration objective on the selected past OOF
rows, not an untouched validation result. The E79 CSV is a single experimental
benchmark. Its Kaggle score, if submitted and completed, is reported separately
from internal results. It does not qualify or replace the existing champion.
