# R05 — Binary word presence

**Verdict: RETIRE** — fixed temporal Train pilot, no Test/Kaggle access.

Recipe: word unigram/bigram CountVectorizer, min_df=2, cap=20,000, binary presence, no IDF/normalization; structured/OHE + missing indicator; sparse LightGBM regression_l1.

| View | n | Control MAE | R05 MAE | Gain (control − R05) |
|---|---:|---:|---:|---:|
| W1 | 4,114 | 15.882112 | 15.864418 | +0.017694 |
| W4 | 4,950 | 11.455005 | 11.442588 | +0.012417 |
| primary | 9,064 | 13.464396 | 13.449584 | +0.014812 |
| H12 | 38,588 | 14.047533 | 14.030725 | +0.016808 |

The comparison uses the frozen three-component Regression control on exactly the same rows. H12 includes both early and late horizons; it is not substituted for the primary W1+W4 pilot. Tokenizer and all structured transforms were fit separately using rows through each cutoff.

See `results.json`, `verification.json`, `subgroups.json`, and `gate.json` for denominators, support, fit accounting, cache reconstruction, and diagnostics.
