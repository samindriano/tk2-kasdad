# REG_R12_CURRENT_ROUTE_LATE_V1

## Decision

**FAIL_CLOSE_RESCUE** — gates and zero-fit matched comparisons are recorded in `RESULTS.json`.

| View | n | New route MAE | Current R06 MAE | Mapping-only MAE | Gain vs R06 | Gain vs mapping-only |
|---|---:|---:|---:|---:|---:|---:|
| W5 | 7462 | 10.441509 | 10.281664 | 10.266731 | -0.159845 | -0.174779 |
| W6 | 8491 | 10.103855 | 9.921613 | 9.940114 | -0.182242 | -0.163741 |
| pooled unique | 15953 | 10.261792 | 10.090026 | 10.092888 | -0.171766 | -0.168904 |

The E71 H8 figure in `RESULTS.json` is diagnostic only, overlaps W5/W6, and is not included in the pooled denominator. Missing rows are exactly current R06 by construction.

The original R12 legacy-baseline route is secondary only. W5/W6 are reused development outcomes; W5 labels enter the origin75 training prefix, and these predictions are not fresh holdout evidence.

No Test, SampleSubmission, Kaggle, upload, dependency installation, policy grid, retry, or R06 refit was performed.

Protocol signature: `aa96bb8d76ffcd8c119a0c0fcbb5b933b60c6c7b4ecdf67265b9a89b6b7f2659`
Prelabel receipt signature: `6126762e42ec9d675e186f81268cc5bfe11b5f87e644501f50063aac1232fe70`
