# R07 — NMF32 nonnegative text factors

**Verdict: RETIRE** — fixed STANDARD temporal Train pilot; no Test/Kaggle access.

Recipe: fit-only word TF-IDF (1,2), `min_df=2`, cap 20,000, sublinear TF; NMF32 on nonnegative text only (`nndsvda`, CD, Frobenius, max_iter=200, seed=42); append factors and the missing-stream indicator to the structured native-category design; LightGBM uses the original SVD32 `base` parameters.
Policy: 0.75 × frozen three-component champion + 0.25 × within-cycle rank50(raw R07), clipped to [0,100].

| View | n | Control MAE | R07 MAE | Gain (control − R07) |
|---|---:|---:|---:|---:|
| W1 | 4,114 | 15.882112 | 15.947171 | -0.065059 |
| W4 | 4,950 | 11.455005 | 11.554989 | -0.099983 |
| primary | 9,064 | 13.464396 | 13.548528 | -0.084132 |
| H12 | 38,588 | 14.047533 | 14.132629 | -0.085097 |

Control and candidate use the same exact row positions per view. TF-IDF vocabulary/IDF, NMF components, and structured preprocessing are fit separately on past-only rows at each cutoff. NMF factors are predictive representations; no semantic meanings are assigned.

NMF convergence by cutoff: 55: 75/200 iterations, reconstruction error 96.7662, converged=True, fit convergence warnings=0, score-transform convergence warnings=0; 67: 76/200 iterations, reconstruction error 152.277, converged=True, fit convergence warnings=0, score-transform convergence warnings=0.
Resource preflight measured fold-specific sparse TF-IDF matrices (20,000 features; 659,532/1,513,869 fit nnz and 811,475/1,047,040 score nnz). Available memory was 9,461,342,208 bytes; estimated peaks were 779,244,892 and 925,272,828 bytes. Peak estimates are not observed training RSS. Both fits completed in 24.68 seconds total with zero failed attempts.

**Retire reason:** primary MAE worsened by 0.084132 against a required gain of 0.03, and H12 worsened by 0.085097 against the 0.05 harm limit. W1/W4 harm caps and all supported subgroup guards passed; the largest guarded harm was 0.153840 for H12 `threat_rank <= 5` (n=1,928). Independent QA returned PASS. LightGBM emitted warnings that negative category codes would be treated as missing; saved predictions still reconstructed exactly. No Test or Kaggle data was accessed.

See `results.json`, `verification.json`, `independent_qa.json`, `subgroups.json`, `gate.json`, `attempts.jsonl`, `budget_reconciliation.json`, and `experiments/optimization_v3/batch1/R07/freeze.json` for provenance and fit receipts.
