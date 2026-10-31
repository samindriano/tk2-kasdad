# Samuel Classification Campaign — Findings

## Decision

The three prespecified text-representation candidates did not improve the retained C4_HGB route on the full six-window primary comparison. Keep C4_HGB as the classification champion. DC_NBSVM was the best of these candidates by pooled Macro F1, but still scored 0.002506 below C4_HGB; its AP improved while its fixed-threshold Macro F1 and paired correct-count balance did not.

These are adaptive development results: cycles 56–79 have appeared in earlier experiments. The comparison is not an untouched holdout or an independent estimate of generalization. Official Test was not read or predicted, and no Kaggle action was taken.

## Protocol

The present-stream expert candidates retained C2_B's structured feature allowlist, fit-only preprocessing, word TF-IDF parameters, stream-missing indicator, and LogisticRegression settings except for DC_SVC. Missing-stream records continued to use the verified C1_B output. Every route used a fixed 0.50 decision threshold; no outer-label threshold selection was performed.

Three fixed changes were compared:

| Recipe | Single change |
|---|---|
| DC_SVC | LinearSVC on the unchanged C2_B design; fixed decision margin 0. `expit(margin)` is only an uncalibrated monotone score view. |
| DC_NBSVM | An L1-normalized, fit-fold document-presence log-count ratio weights the C2_B word TF-IDF block before the unchanged logistic head. This is NB-SVM-inspired, not the original paper's literal model. |
| DC_WORD_CHAR | Add a fit-only `char_wb` TF-IDF block with 3–5 character n-grams, `min_df=3`, and at most 30,000 features. |

Stage 1 screened all three at fit cutoffs 55 and 67, retaining primary W1/W4 and H12_E/H12_L. None met the registered retirement rule of at least −0.02 Macro F1 on both H12s. Stage 2 completed the remaining fit cutoffs 59, 63, 71, and 75 for all three recipes, adding 12 fits and reusing Stage 1 cutoffs 55 and 67 without refitting. The six primary validation windows cover 38,588 records from cycles 56–79; 36,010 have a present stream and 2,578 use the unchanged C1_B fallback.

## Full-six results

| Route | Pooled Macro F1 | Δ vs C4 | Equal-cycle Macro F1 | Δ vs C4 | AP | Δ AP | Brier Δ vs C4 |
|---|---:|---:|---:|---:|---:|---:|---:|
| C4_HGB | 0.825481 | — | 0.818278 | — | 0.829320 | — | 0.000000 |
| DC_SVC | 0.814421 | −0.011059 | 0.808464 | −0.009813 | 0.807724 | −0.021596 | +0.022615 |
| DC_NBSVM | 0.822975 | −0.002506 | 0.814436 | −0.003841 | 0.836232 | +0.006913 | −0.001222 |
| DC_WORD_CHAR | 0.822696 | −0.002785 | 0.815653 | −0.002625 | 0.828881 | −0.000439 | +0.000889 |

AP is a ranking measure. DC_NBSVM's higher AP did not translate to better Macro F1 at the required fixed 0.50 cutoff. Its Brier score was also slightly lower than C4's. DC_SVC's AP is valid for its monotone margin transform, but its Brier score is not calibration evidence because `expit(margin)` was not calibrated.

Window-level Macro F1 deltas vs C4_HGB:

| Window | DC_SVC | DC_NBSVM | DC_WORD_CHAR |
|---|---:|---:|---:|
| W1 | −0.0049 | −0.0119 | −0.0033 |
| W2 | −0.0108 | +0.0055 | −0.0025 |
| W3 | −0.0090 | +0.0029 | −0.0016 |
| W4 | −0.0171 | −0.0069 | −0.0056 |
| W5 | −0.0074 | −0.0011 | −0.0043 |
| W6 | −0.0158 | −0.0081 | −0.0004 |

DC_NBSVM's positive differences in W2 and W3 are counterexamples to a uniformly negative result, but the losses in four other windows erase those gains. The primary window values also differ from the two H12 values because each primary window uses its own expanding fit cutoff; H12_E/H12_L use the earlier fit55/fit67 model outputs over 12-cycle validation spans.

## Paired present-stream errors

On the same 36,010 present-stream validation records, the candidate-only correct rows versus C4-only correct rows were:

| Candidate | Candidate-only correct | C4-only correct | Net correction difference |
|---|---:|---:|---:|
| DC_SVC | 944 | 1,409 | −465 |
| DC_NBSVM | 820 | 846 | −26 |
| DC_WORD_CHAR | 386 | 491 | −105 |

The fixed blend gate required at least 25 corrections in each direction and a nonnegative candidate net correction count. No candidate passed, so the registered 0.50 probability blend was not evaluated. The missing-stream subgroup is unchanged by design: each route exactly reuses C1_B there, so its metric delta versus C4 is 0.

## Interpretation and next research step

Changing the present-stream linear text representation alone did not beat C4_HGB at the fixed decision policy. DC_NBSVM may carry some ranking information that a future past-only calibration study could examine, but this screen does not justify replacing the route or tuning against these outer cycles. The strongest immediate weakness in the candidate family is not a broad text signal: all three candidates reduced pooled present-stream Macro F1 and had a negative net paired correction count.

The next bounded classification question approved by the main agent is to test CatBoost only as an alternative missing-stream expert while retaining the original C2_B present-stream expert. That work has not started. It will use the same six temporal cutoffs and H12 windows, fixed CatBoost settings, and a preregistered conditional fixed 0.50 blend gate. No new text family, Test prediction, Kaggle score use, or submission is authorized by this report.

## Audit trail and files

- Stage 1 preliminary values, six-fold diagnostics, and chart: `reports/dual_campaign/classification/amendment_01/`.
- Full-six results, window table, paired-correction plots, and independent verifier: `reports/dual_campaign/classification/stage2/`.
- Stage 1 initial freeze: `experiments/dual_campaign/classification/freeze.json`.
- Prescore SVC convergence failure: `reports/dual_campaign/classification/stage1_attempt.json`.
- One-parameter convergence amendment: `experiments/dual_campaign/classification/amendment_01.json` and `amendment_01_freeze.json`.
- Full-six frozen protocol: `experiments/dual_campaign/classification/stage2_protocol.json` and `stage2_freeze.json`.
- Report assembly plot-accessor failures and report-only amendments are recorded alongside their frozen receipts. No model outputs were discarded or overwritten.
- Fitted-model probability caches are in ignored `experiments/runs/dual_cls_v1a/` and `experiments/runs/dual_cls_v1_stage2/`; raw data and model binaries were not added to Git.

### Verification

- Focused synthetic unit tests: 8 passed, 3 subtests passed.
- Source compilation: passed.
- Stage 1 independent metrics/provenance verifier: PASS (6 candidate-fold artifacts).
- Full-six independent metrics/provenance verifier: PASS (12 new fit artifacts; six primary windows; fit-only fingerprints, row identities, routes, metric calculations, blend gate, and chart hashes checked).
- Report-only assembly recovery verifier: PASS; no additional fit or prediction was performed.
