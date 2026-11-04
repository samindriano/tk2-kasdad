# CLS_POS_A02_META_V1 — run and review

**Status:** completed diagnostic-only; not qualified as a champion and not submitted.

## Execution and recovery

The root-approved R2 recipe ran exactly six predeclared LogisticRegression fits: D1_PAIR, D1_SCALAR, D2_PAIR, D2_SCALAR, FINAL_PAIR, FINAL_SCALAR. D1_PAIR completed before a reader bug rejected a scikit-learn 1.8 `FutureWarning` as though it were a convergence warning. Its model, receipt, and ledger entry were preserved unchanged. A separately committed, strict reader recovery accepted only the source-verified `penalty="l2"` deprecation warning; it continued to reject `ConvergenceWarning`, unknown warning classes, invalid iterations, nonfinite parameters, or mismatched hashes. D1_PAIR was reused, not refit.

The RAM gate then paused before D2_SCALAR reservation. At that checkpoint, three fits were complete and none had failed. A committed progress receipt bound all three saved states and the ledger; the continuation then completed the remaining three fits. The final ledger has six unique reservations, six starts, six completions, zero failures, and no retries.

## Internal temporal results

The prespecified pooled W5/W6 evaluation contains **15,953 unique rows**: D1 covers cycles 72–75 (7,462 rows) and D2 covers cycles 76–79 (8,491 rows). The threshold remained 0.50.

| Arm | Macro F1 | Δ vs MIX50 | Precision | Recall | Confusion matrix (TN, FP; FN, TP) |
|---|---:|---:|---:|---:|---|
| MIX50 | 0.811707 | — | 0.753548 | 0.718917 | [[9933, 1146], [1370, 3504]] |
| PAIR | 0.804077 | -0.007629 | 0.770994 | 0.676241 | [[10100, 979], [1578, 3296]] |
| SCALAR | 0.805470 | -0.006237 | 0.776571 | 0.674600 | [[10133, 946], [1586, 3288]] |

Both learned heads reduced Macro F1 against MIX50. PAIR lost 208 true positives while removing 167 false positives; SCALAR lost 216 true positives while removing 200 false positives. The fixed missing-stream route was identical to MIX50 on 1,033 rows. On supported text-load Q4 (4,011 rows), Macro F1 was 0.768042 for MIX50, 0.761556 for PAIR, and 0.761773 for SCALAR; positive recall was 0.6150, 0.5744, and 0.5669, respectively.

By window, PAIR Δ Macro F1 was -0.002473 in cycles 72–75 and -0.012687 in cycles 76–79; SCALAR was -0.001995 and -0.010301. The later window accounts for the larger pooled loss. These results do not qualify either head.

## Candidate CSV and boundaries

- CSV: `submissions/optimization_v3/cls_pos_a02_meta_v1/classification_pair_e79.csv`
- SHA-256: `43f9af4eceb192215f1dada69c73db4c45a8918a7d3c807050f099e0187c1945`
- Rows: 29,421; columns and IDs match the official Test ID order; labels are integer 0/1.
- No Test labels or SampleSubmission were loaded. No Kaggle upload occurred; no public score exists.
- Only the PAIR CSV was created; no SCALAR CSV was created.

## QA

All six fit states, receipts, input/source hashes, and completion events passed independent checks. CSV schema, uniqueness, binary labels, row count, Test ID order, and hash passed. Relevant test suites: **25 passed in 2.99s**. The detailed metrics, subgroups, coefficients, and provenance are in `RESULTS.json`, while this receipt is `INDEPENDENT_QA_R1.json`.
