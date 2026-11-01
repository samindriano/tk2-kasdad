# V3 expanded benchmark campaign status

Date: 2026-11-01

Campaign: `optimization_v3_expanded_submissions_v1`

Frozen source commit: `62eb622d8b9bc491ed72aa553349dddd5635070f`

Freeze signature: `9237741de7d9c8d21ef2b90ef7f2f8d4074391a34eb4071bfb75d3b06493b165`

## Candidate reconciliation

Positive gains below mean higher Macro F1 for Classification and lower MAE for Regression. Temporal metrics use 38,588 pooled primary rows; H12 is the existing 38,588-row diagnostic, and W5+W6 uses 15,953 rows. These are development results, not untouched holdouts.

| Candidate | Continuation result | Primary gain | H12 gain | W5+W6 gain | Full-Train CSV QA | Kaggle |
|---|---|---:|---:|---:|---|---|
| C16 | Experimental exception; primary, late, and W2 text-load Q4 guards failed | +0.000304 | +0.002837 | -0.001078 | PASS | Not uploaded |
| C17 | Experimental exception; primary, late, and minimum window-count gates failed | -0.000545 | +0.001411 | -0.002063 | PASS | Not uploaded |
| C11 | Experimental exception; primary, late, and minimum window-count gates failed | -0.000479 | -0.000310 | -0.000622 | PASS | Not uploaded |
| R22_ORIGINAL | Qualified; retained as selected Regression candidate | +0.039699 | +0.124873 | +0.021379 | PASS | Not uploaded |
| R21 | Experimental exception; primary, late, and window-count gates failed | -0.006106 | +0.044280 | -0.026890 | PASS | Not uploaded |
| R11 | Continuation integrity blocked; performance gate remains UNKNOWN_NOT_EVALUATED | — | — | — | PASS for the frozen full-Train recipe | Not uploaded |
| R06_ORIGINAL | Experimental exception; subgroup guard failed despite primary/H12/late gains | +0.056977 | +0.103260 | +0.023589 | PASS | Not uploaded |

The verdicts above preserve the frozen gates. The authorized performance exception does not turn failed gates into passes. R11's four continuation bundle writes failed on object-dtype diagnostic controls. Its continuation scores were not opened and the 36-fit recovery was not run; the full-Train CSV was independently checked against the original frozen recipe, and the manifest retains the unknown continuation result.

## Full-Train preparation and QA

All seven CSVs are present locally and independently verified against the official Test IDs and SampleSubmission. Each has exactly 29,421 rows, matching unique IDs in the official order, and the required columns. Classification values are integer 0/1; regression values are finite and within [0,100]. Recipe reconstruction, availability routing, rank50 average ties over complete cycles, blending, clipping, frozen input hashes, and fit lineage passed. Each recipe uses all 50,608 official Train rows; preprocessing was fit from Train only. Official Test was used only for inference and ID/cycle/output QA; Test labels/targets were not read.

The reusable verification result is `FINAL_QA.json`; candidate recipe and CSV hashes are in the per-candidate manifests. CSVs and fitted prediction arrays remain Git-ignored. The verifier is `scripts/verify_optimization_v3_expanded_submissions.py`.

## Fits and resources

- The five new full-Train refits completed serially: C17 (1), C11 (1), R21 (1), R11 (9), R06_ORIGINAL (1), for 13 predictor fits.
- These 13 fits were already included in the original 65-fit reservation; they were not added twice.
- C16's C2_B and CB_MISSING components and R22's three-component champion were reused only after verified full-Train lineage checks; C16's candidate component and R22's XGBoost component were fitted for their recipes. Their two prior off-ledger fits are counted by the frozen budget audit.
- The freeze records an effective total of 177/180 predictor-fit capacity, with 3 remaining. No fit retry was used. R11 continuation recovery would require 36 fits and is not authorized by the remaining capacity.
- Jobs used one training process at a time and two threads. Resource preflight passed for all five new refits.

## Kaggle status and authentication blocker

No live Kaggle rules, team, deadline, shared quota, or complete submission history/idempotency check could be made. No API upload was attempted, no quota was consumed, and no public scores or submission references exist for these seven CSVs. The authorizations remain unused. Local ledger scans are not a substitute for the live shared ledger.

The current runtime inventory is: Kaggle CLI absent, Python Kaggle module absent, standard `~/.kaggle/kaggle.json` absent, and no `KAGGLE_*` environment-variable names. Credential values were not read or searched for. The earlier temporary CLI invocation was recorded as broken because its virtual-environment `Scripts/python.exe` was missing. This is an execution-environment authentication gap; there is no evidence Kaggle rejected a login because no authenticated API request was made.

Authentication worked in an earlier runtime: the historical submission report records Kaggle CLI 2.2.5 using a locally available token, and a 2026-10-31 preflight recorded an authenticated team/rules response. The current process does not have that CLI/token context. Repository evidence does not identify why the runtime or its temporary environment changed, so that platform-level cause remains unknown.

Resume only after a Kaggle API/CLI is authenticated in the active runtime. Then recheck current rules, team, deadline, shared quota, complete ledger, and idempotency before using at most one of the seven authorizations. Follow the authorized message naming each candidate and its actual qualification status; do not infer success until Kaggle returns a submission reference and scoring status.

## Preserved evidence

- Continuation evaluation: `reports/optimization_v3/expanded_benchmark_v1_final_evaluation/CAMPAIGN_SUMMARY.md` and its candidate result/verification JSON files.
- Execution freeze, budget, preflight, and seven full-Train manifests: this directory.
- C16/R22 prior preparation record: `reports/submissions/c16_r22/SUBMISSION_STATUS.md`.
- Historical successful CLI submission evidence: `reports/submissions/INITIAL_SUBMISSIONS.md` and `reports/submissions/KAGGLE_RESULTS.md`.

No model, frozen recipe, historical verdict, notebook, or teammate work was changed. No push was performed.
