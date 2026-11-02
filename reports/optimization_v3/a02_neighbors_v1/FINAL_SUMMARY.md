# Final checkpoint: A02 pilot and C19 upload handoff

**Status:** review checkpoint complete. No continuation, fit, refit, Test/Sample access, or Kaggle upload was performed during report preparation. No candidate verdict or gate was changed.

## Decision summary

- A02 independent reconstruction returns `PILOT_PASS`. This is pilot evidence only; it does not qualify A02 or authorize continuation.
- The four authorized A02 fits and prediction bundles are durable. Verification started no fits and reports zero A02 fit failures.
- C19 full-Train CSV passed local validation. Sam later reported uploading it manually and seeing `Complete`, public Macro F1 0.82760. This is human-reported, not verified through Kaggle API/live access; reference, exact upload time, and server-side file hash are unknown. The historical C19 verdict `RETIRE / prior frozen-gate failure` is preserved.
- Our browser upload attempt remained blocked before file selection. Computer Use stopped before opening Chrome extension settings, so that setting did not change. The one C19 upload authorization is now considered used based on Sam's report; do not retry or submit again.

## A02 frozen scope and source boundaries

Question: whether a past-only lexical-neighbor residual correction improves the frozen sparse25 classification route, compared with sparse25 and a within-cycle shuffled-residual control.

The run used the official Classification Train only (50,608 rows, cycles 47–79; SHA-256 `875552a68a3d24b0f3e1966d6ddb5671e6f54305a1c526e2f0a76747942ed91c`). The freeze signature is `cdac164df87d4f559e36795066cc0c3a91034e5917b0b1fd72cebd1051dae08b`; protocol SHA-256 is `24af9a2e7793690a4421a033fb18de1c7596c7f348c0e365c56f242446e54f4f`. Frozen source commit: `ecf48c32481a3a8d9d6e56e6d1cba3de52f97420`.

The earlier C19/A02 v1 preflight/freeze files remain preserved as superseded, pre-fit history. The A02 v1 coverage map was replaced before any fit; all four A02 fits used v2 only. The separate C19 v1 preflight is likewise retained as historical evidence; C19 inference used the v2 freeze and manifest.

Each of the four task fits used one task estimator. E67 used a 29,705-row text prefix and an OOF bank of 17,685 rows (16,482 with present stream); E75 used a 42,117-row text prefix and an OOF bank of 30,097 rows (28,045 present). Queries were 20,903 rows for E67 and 8,491 for E75. Each component reused the exact frozen C16 OOF cache route. No Test or SampleSubmission path was used by A02.

### Recovery and verifier defects

The failures and repairs are recorded here without altering the frozen runner or verifier:

1. E67 observed was saved, then the prediction writer raised `AttributeError: numpy.ndarray has no attribute to_numpy` because the frozen writer expected a pandas Series for query IDs. The resume harness adapted the already-saved query ID array and completed the other three authorized fits and prediction bundles; no fit was repeated.
2. The first independent verifier pass rejected E67 observed bank IDs because persisted NumPy string IDs carried `str_:` while the verifier expected `str:`. The underlying IDs, rows, cycles, and OOF probabilities matched. The adapter normalizes only this type tag in memory; saved model and prediction bytes remain unchanged.
3. The frozen verifier then rejected neighbor source positions greater than query positions. Official Train row order is not chronological (for example, row 2 can be cycle 77). The in-memory adapter removes only that invalid row-position comparison. The verifier's exact source-bank, ID, and row joins remain; the strict bank-cycle-before-query-cycle check remains.
4. The repair helper additionally checks each cache cutoff and each exact four-cycle OOF block against official Train; all query cycles are after the outer origin; bank and query IDs are disjoint; every neighbor row/ID/cycle maps to the official bank; no self or future-cycle neighbor occurs; and similarity/source-row tie order remains unchanged. It checked 1,759,552 saved neighbor references.
5. The first receipt-writing attempt after successful verifier scoring hit a JSON serialization error on a NumPy `int64` counter. The counter was converted to a Python integer and the verifier was rerun without fits. The final technical receipt was then written successfully.

The frozen verifier source hash is still `40c11d4a66197d0842195aef30b243563264cf0c54341ec90a75be73e9db9541`. The compatibility adapter is [optimization_v3_a02_resume_patch.py](../../../scripts/optimization_v3_a02_resume_patch.py), SHA-256 `52e4d23742e553eb6af95099bf09d5e4402632eb11f8452d9cfb35526a517611`. No model recipe, preprocessing, residual assignment, or fit accounting changed.

## A02 internal metrics

All values below were reconstructed from the four durable bundles using Train labels. `H12-L` is not independent of the primary view. Positive-recall values are included to show the classification tradeoff; A02’s locked headline metric remains Macro F1.

| View | n | Sparse25 Macro F1 | Observed Macro F1 | Shuffled Macro F1 | Observed gain vs sparse25 | Observed gain vs shuffled | Positive recall: sparse25 → observed |
|---|---:|---:|---:|---:|---:|---:|---:|
| W4 | 4,950 | 0.841418 | 0.844405 | 0.841091 | +0.002988 | +0.003314 | 0.791149 → 0.803562 |
| W6 | 8,491 | 0.816599 | 0.819604 | 0.815563 | +0.003005 | +0.004041 | 0.698000 → 0.708000 |
| Primary | 13,441 | 0.828147 | 0.831157 | 0.827502 | +0.003010 | +0.003656 | 0.737652 → 0.748679 |
| H12-L | 20,903 | 0.808961 | 0.811159 | 0.809773 | +0.002198 | +0.001386 | 0.736584 → 0.744314 |

### Probability metrics from saved predictions

These metrics were computed from persisted `baseline_prediction` and `prediction` arrays and the matching official Train labels. No model was fit for this calculation. Lower is better for log loss and Brier score.

| View | Arm | ROC AUC | Average precision | Log loss | Brier |
|---|---|---:|---:|---:|---:|
| W4 | Sparse25 | 0.911881 | 0.825966 | 0.366748 | 0.113077 |
| W4 | Observed | 0.915358 | 0.830957 | 0.419169 | 0.111075 |
| W4 | Shuffled | 0.912282 | 0.820180 | 0.428159 | 0.113072 |
| W6 | Sparse25 | 0.909973 | 0.791327 | 0.345912 | 0.108428 |
| W6 | Observed | 0.911107 | 0.792959 | 0.377625 | 0.107386 |
| W6 | Shuffled | 0.910926 | 0.790781 | 0.350714 | 0.107947 |
| Primary | Sparse25 | 0.911767 | 0.808551 | 0.353585 | 0.110140 |
| Primary | Observed | 0.913905 | 0.811560 | 0.392924 | 0.108745 |
| Primary | Shuffled | 0.912416 | 0.805531 | 0.379235 | 0.109835 |
| H12-L | Sparse25 | 0.896701 | 0.790661 | 0.381605 | 0.120838 |
| H12-L | Observed | 0.899755 | 0.792814 | 0.417247 | 0.119377 |
| H12-L | Shuffled | 0.896945 | 0.786562 | 0.408023 | 0.120769 |

Observed improves AUC, average precision, and Brier score versus sparse25 in all four views. Log loss is worse than sparse25 in all four views and worse than shuffled in W6, primary, and H12-L. This disagreement among probability scores is a material calibration/severe-error diagnostic; no calibration or threshold search was authorized.

### Cycle-level effects

The table reports Macro F1 gain for observed A02 versus each comparator. All cycles and denominators are included.

| Origin | Cycle | n | Gain vs sparse25 | Gain vs shuffled |
|---:|---:|---:|---:|---:|
| 67 | 68 | 577 | +0.000059 | +0.001790 |
| 67 | 69 | 1,079 | +0.003249 | +0.004161 |
| 67 | 70 | 1,605 | −0.000259 | +0.000446 |
| 67 | 71 | 1,689 | +0.007053 | +0.006150 |
| 67 | 72 | 1,811 | +0.005017 | +0.002440 |
| 67 | 73 | 1,880 | +0.000670 | +0.001233 |
| 67 | 74 | 1,814 | +0.002171 | −0.001575 |
| 67 | 75 | 1,957 | +0.002720 | +0.000000 |
| 67 | 76 | 1,822 | +0.003084 | +0.002325 |
| 67 | 77 | 2,335 | +0.002528 | +0.003626 |
| 67 | 78 | 2,146 | +0.000970 | +0.001718 |
| 67 | 79 | 2,188 | −0.001788 | −0.004460 |
| 75 | 76 | 1,822 | +0.003789 | +0.005964 |
| 75 | 77 | 2,335 | +0.000099 | −0.000619 |
| 75 | 78 | 2,146 | +0.011043 | +0.013106 |
| 75 | 79 | 2,188 | −0.001241 | −0.000328 |

Observed beats sparse25 in 13/16 cycles and shuffled in 11/16, ties shuffled in one cycle, and loses to shuffled in four. The pooled gains therefore do not imply a uniform cycle effect.

### Protected subgroup support and historical guards

Each row gives `n (class 0 / class 1)`, Macro F1 change for observed A02 versus sparse25 and shuffled, and the count of historical harm flags fired. The six flags are F1 harm thresholds 0.01/0.02/0.03 and positive-recall harm thresholds 0.02/0.03/0.05. Across all 32 supported subgroup-view rows, **0/192 historical flags fired**. This does not erase the small negative subgroup contrasts shown below.

#### W4

| Group | n (class 0 / class 1) | ΔF1 vs sparse25 | ΔF1 vs shuffled | Flags |
|---|---:|---:|---:|---:|
| Stream missing | 342 (202/140) | +0.000000 | +0.000000 | 0/6 |
| Stream present | 4,608 (2,895/1,713) | +0.003210 | +0.003564 | 0/6 |
| Entry friction 0 | 3,418 (2,211/1,207) | +0.003499 | +0.004033 | 0/6 |
| Entry friction 1 | 1,532 (886/646) | +0.001832 | +0.001758 | 0/6 |
| Text-load Q1 | 1,092 (579/513) | +0.003671 | +0.005498 | 0/6 |
| Text-load Q2 | 963 (625/338) | −0.001698 | −0.003967 | 0/6 |
| Text-load Q3 | 1,130 (808/322) | +0.004599 | +0.004599 | 0/6 |
| Text-load Q4 | 1,765 (1,085/680) | +0.004600 | +0.005583 | 0/6 |

#### W6

| Group | n (class 0 / class 1) | ΔF1 vs sparse25 | ΔF1 vs shuffled | Flags |
|---|---:|---:|---:|---:|
| Stream missing | 526 (355/171) | +0.000000 | +0.000000 | 0/6 |
| Stream present | 7,965 (5,636/2,329) | +0.003214 | +0.004323 | 0/6 |
| Entry friction 0 | 5,738 (4,316/1,422) | +0.002348 | +0.000868 | 0/6 |
| Entry friction 1 | 2,753 (1,675/1,078) | +0.003825 | +0.008405 | 0/6 |
| Text-load Q1 | 2,244 (1,463/781) | +0.007000 | +0.009262 | 0/6 |
| Text-load Q2 | 2,285 (1,674/611) | +0.003227 | +0.006358 | 0/6 |
| Text-load Q3 | 1,969 (1,526/443) | +0.000685 | −0.001664 | 0/6 |
| Text-load Q4 | 1,993 (1,328/665) | −0.000019 | −0.000035 | 0/6 |

#### Primary

| Group | n (class 0 / class 1) | ΔF1 vs sparse25 | ΔF1 vs shuffled | Flags |
|---|---:|---:|---:|---:|
| Stream missing | 868 (557/311) | +0.000000 | +0.000000 | 0/6 |
| Stream present | 12,573 (8,531/4,042) | +0.003226 | +0.003918 | 0/6 |
| Entry friction 0 | 9,156 (6,527/2,629) | +0.002884 | +0.002215 | 0/6 |
| Entry friction 1 | 4,285 (2,561/1,724) | +0.003099 | +0.005887 | 0/6 |
| Text-load Q1 | 3,336 (2,042/1,294) | +0.005673 | +0.007642 | 0/6 |
| Text-load Q2 | 3,248 (2,299/949) | +0.001643 | +0.002879 | 0/6 |
| Text-load Q3 | 3,099 (2,334/765) | +0.002264 | +0.000797 | 0/6 |
| Text-load Q4 | 3,758 (2,413/1,345) | +0.002404 | +0.002750 | 0/6 |

#### H12-L

| Group | n (class 0 / class 1) | ΔF1 vs sparse25 | ΔF1 vs shuffled | Flags |
|---|---:|---:|---:|---:|
| Stream missing | 1,375 (888/487) | +0.000000 | +0.000000 | 0/6 |
| Stream present | 19,528 (13,288/6,240) | +0.002375 | +0.001498 | 0/6 |
| Entry friction 0 | 14,344 (10,221/4,123) | +0.002844 | +0.002110 | 0/6 |
| Entry friction 1 | 6,559 (3,955/2,604) | +0.000796 | −0.000058 | 0/6 |
| Text-load Q1 | 5,484 (3,443/2,041) | +0.002948 | +0.002333 | 0/6 |
| Text-load Q2 | 4,841 (3,334/1,507) | −0.000699 | −0.000330 | 0/6 |
| Text-load Q3 | 4,715 (3,574/1,141) | +0.003143 | +0.001012 | 0/6 |
| Text-load Q4 | 5,863 (3,825/2,038) | +0.003201 | +0.002157 | 0/6 |

## Fit budget and receipts

The Research Next global cap is 60. The live ledger now has 59 starts/consumed, 57 saved, and 2 historical fit failures. A02 accounts for exactly four starts and four saves, with no A02 failure and no retry. Gross reservations total 63, with 4 unused fits released, so active reserved/consumed is 59. The budget helper's `remaining_capacity_after_reservations = -3` uses gross reservations before subtracting releases; the active/net count is 59. No capacity is used or reserved by this report.

The two failed starts are historical and unrelated to A02: `MEASURED_CLASS_ONLY_FULLTRAIN` (missing `baseline_prediction` array) and `CLASSIFICATION_SPARSE_MLP_V1_55` (NumPy `int64` JSON serialization). They remain in the ledger.

| Fit ID | Arm/origin | Text prefix rows | Saved fit receipt SHA-256 |
|---|---|---:|---|
| `A02_E67_OBSERVED` | Observed, E67 | 29,705 | `4a07d9f3827563d04b602dad8fc3a5dc71a6faf6076d18851cbced17817b0e6c` |
| `A02_E67_SHUFFLED` | Shuffled, E67 | 29,705 | `98f01938b25b025cec1ae16710d1f4e286fd4474b686eac37c1421d09c5aacf3` |
| `A02_E75_OBSERVED` | Observed, E75 | 42,117 | `48f444429265cb9892d7793bd5f2c1331d3fa16299bf9793b630d7984b79caa6` |
| `A02_E75_SHUFFLED` | Shuffled, E75 | 42,117 | `380b54cd9f60acbf71871b447a6e5fc09dbc4ded1276ed312ffbb4332349371b` |

## C19 Kaggle upload handoff

**CSV for the one authorized benchmark upload:** `submissions/optimization_v3/c19_benchmark_v2/c19_xgb_svd32.csv`

Absolute path: `C:\Users\Sam\OneDrive\Documents\KASDAD\TP 2\submissions\optimization_v3\c19_benchmark_v2\c19_xgb_svd32.csv`

SHA-256: `4dc88a17d3e5708fb2d9f051161fd340d7cba89af6e120c620efbc0db66d6e35`

The file has 29,421 rows and columns `record_id, artifact_signal`; labels are integer 0/1 and IDs match official Test IDs in order. The manifest records 344 labels different from sparse25 and zero changed labels on the missing-stream route. The pre-upload manifest remains unchanged as the hash-lineage record; its `PREPARED_VALIDATED_NOT_SUBMITTED` status reflects that preparation checkpoint. The append-only receipt now records Sam's manual-upload report: Complete, public Macro F1 0.82760 versus best sparse25 0.82908 (reported delta −0.00148). Exact Kaggle reference, upload time, and server file hash remain UNKNOWN; no API/live verification was performed. Treat the user report as the latest status and the previous browser block as historical.

Competition: `tp2-classification-cyber-vulnerability-kasdd-gasal-2026-2027`. Proposed upload message: `Samuel C19 XGB SVD32 Frozen Benchmark`. Historical C19 status remains `RETIRE / prior frozen-gate failure; experimental benchmark exception only`.

The live preflight snapshot at 2026-11-02 11:27:59 UTC reported rules accepted, team `kasdud lovers`, and 9 uploads remaining that day. The subsequent browser file chooser failed before file selection (`UPLOAD_BLOCKED_BEFORE_FILE_SELECTION`); quota remained 9 and Submit was not clicked. After the user authorized enabling file-URL access, the Computer Use runtime stopped before opening the Chrome extension page because it could not confidently establish the current browser URL. No extension setting changed and our browser session transmitted no data. Sam later reported manual submission complete; the append-only receipt records that report without claiming live/API verification. The authorization is considered consumed; no repeat upload is allowed.

## Evidence files

- [Independent A02 metrics and verifier result](metrics_reconstructed.json)
- [A02 probability metrics computed from saved predictions, no fit](probability_metrics_from_saved_predictions.json)
- [A02 technical resume receipt](technical_resume_receipt.json)
- [A02 independent verification](verification.json)
- [A02 preflight and synthetic QA](preflight.json), [synthetic QA receipt](synthetic_qa.json)
- C19: `reports/optimization_v3/c19_benchmark_v2/submission_manifest.json`, `kaggle_preflight.json`, `submission_receipts.jsonl`, and `verification.json`

## QA and stop point

- Independent A02 verifier: `PASS`, 37 checks; scientific verdict `PILOT_PASS`.
- Budget reconciliation: 59/60 starts consumed; all four A02 fit receipts match the ledger; no new starts during verification or report generation.
- Probability metrics: recomputed solely from persisted A02 probability arrays and official Train labels; no fit performed.
- C19 CSV: prior independent verifier PASS, 29,421 rows, expected schema and IDs; manual completion and public score are user-reported only; server reference/time/hash UNKNOWN.
- No modeling, refit, Test/Sample access, additional experiment, Kaggle submission, or UI retry was performed for this checkpoint.
- Stop here for Sol/user review. The C19 upload allowance is used; no retry or additional submission is authorized. Any A02 continuation requires separate review/authorization; no new work is implied by this report.
