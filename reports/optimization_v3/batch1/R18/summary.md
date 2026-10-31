# R18 — Recent OOF median location offset

Status: **RETIRE** (late-only pilot; no external promotion).
Freeze signature: 0a3cbc56317d2f129735ba00dbbc94a50befcd8c66696894cf71b56f1cf53312

| View | Rows | Baseline MAE | Candidate MAE | Gain (positive is better) |
|---|---:|---:|---:|---:|
| primary | 13441 | 10.477525 | 10.463164 | 0.014361 |
| W4 | 4950 | 11.455005 | 11.433127 | 0.021878 |
| W6 | 8491 | 9.907683 | 9.897704 | 0.009979 |
| H12_L | 20903 | 11.611686 | 11.583755 | 0.027931 |
| H12_E | 17685 | 16.926611 | 16.926611 | 0.000000 |

H12_E is reported separately and uses the exact-champion fallback.
Candidate deltas use only the four complete past OOF cycles fixed in the R18 card.
