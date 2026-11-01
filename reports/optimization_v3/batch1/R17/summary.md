# R17 — OOF nonnegative affine L1 calibration

Verdict: **RETIRE** (late-only pilot; no external promotion).
Freeze signature: `20b12df924c9a33d757df03c622a34ec24eb8da27676b17f9aa488066021c3ac`

| View | Rows | Baseline MAE | Candidate MAE | Gain (positive is better) |
|---|---:|---:|---:|---:|
| primary | 13441 | 10.477525 | 10.444793 | 0.032732 |
| W4 | 4950 | 11.455005 | 11.422473 | 0.032532 |
| W6 | 8491 | 9.907683 | 9.874835 | 0.032849 |
| H12_L | 20903 | 11.611686 | 11.579694 | 0.031992 |
| H12_E | 17685 | 16.926611 | 16.926611 | 0.000000 |

Independent QA: **PASS**.
Calibration LP solves: 2; preprocessing fits: 0.
H12_E (cycles 56–67) is reported separately and uses exact champion fallback.
This LATE pilot alone does not authorize full-Train inference or Kaggle promotion.
