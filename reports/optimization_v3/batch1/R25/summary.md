# R25 — Unnormalized word counts

**Verdict: RETIRE** — fixed temporal Train pilot, no Test/Kaggle access.

Recipe: word unigram/bigram CountVectorizer, min_df=2, cap=20,000, raw counts (`binary=False`), no IDF/normalization; inherited structured/OHE + missing indicator; sparse LightGBM regression_l1.

| View | n | Control MAE | R25 MAE | Gain (control − R25) |
|---|---:|---:|---:|---:|
| W1 | 4,114 | 15.882112 | 15.870471 | +0.011641 |
| W4 | 4,950 | 11.455005 | 11.432916 | +0.022089 |
| primary | 9,064 | 13.464396 | 13.447049 | +0.017347 |
| H12 | 38,588 | 14.047533 | 14.027201 | +0.020332 |

The comparator is the frozen three-component Regression champion on the exact same rows. Preprocessing is fit-only at each temporal cutoff; H12 reuses development Train windows and is not an untouched holdout.

See `results.json`, `verification.json`, `subgroups.json`, `gate.json`, and `runtime_notes.json` for fit accounting, denominators, support, cache reconstruction, memory, and runtime notes.
