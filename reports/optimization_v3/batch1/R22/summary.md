# R22 Pilot Summary

- Recipe: XGBoost 3.2.0 pseudo-Huber (`reg:pseudohubererror`, `huber_slope=5.0`) on the frozen structured + TF-IDF/SVD32 design; only the two objective settings differ from R21. The 0.75 champion + 0.25 within-cycle rank50 policy is unchanged.
- Decision: **ADVANCE**
- Primary MAE: 13.464396 baseline → 13.366220 candidate; gain +0.098176 on 9,064 rows.
- H12 MAE: 14.047533 baseline → 13.922659 candidate; gain +0.124873 on 38,588 rows.
- W1 / W4 gains: +0.138456 / +0.064698; H12_E / H12_L gains: +0.153722 / +0.100466.
- Both prescribed fits passed independent receipt, row/ID, fit-only boundary, prediction reconstruction, metric, and gate checks. Cutoffs 55/67 used 12,020/29,705 rows and took 7.67/14.70 s. The prefit peak estimates were 2.40/2.95 GiB and the available-memory preflight passed with the 2 GiB reserve; estimates are not observed peak RSS.
- All pilot gates passed. The largest supported subgroup harm was +0.171021 MAE on W4 `threat_rank >= 95` (n=249), within the +0.25 guard. See [independent QA](independent_qa.json).
- Temporal development outcomes are reused, not an untouched holdout. Test/Sample/Kaggle were not read.
- Full-six continuation is deferred until the 45-card screening pass and profile-based continuation selection under `ONE_CHAT_END_TO_END`.
