# Learned sparse pair interactions — zero-fit preflight

**Decision: `READY_FOR_SOL_PROTOCOL_REVIEW` — this is not fit permission.** The objective and gradient are small enough to validate and implement without a new library, and a label-free rank-16 resource check passed on saved matrices. No optimizer iteration, official-label read, fit, preprocessing fit, budget reservation, or ledger write occurred. Learned factorization interactions remain a speculative candidate with meaningful negative evidence.

## Novelty and evidence already in the repository

N07 is not the first interaction experiment. It is novel only in the narrower sense of learning a low-rank factorization of pair effects.

- N02 used `sklearn.kernel_approximation.PolynomialCountSketch(degree=2, n_components=1024, random_state=42)` on an L2-normalized copy of the C2_B matrix, then trained Logistic Regression on the original main effects plus the fixed random sketch buckets. The logistic head learned bucket weights; it did not learn feature vectors whose dot products assign pair-specific effects. Source: `src/models/research_next_n02.py` (SHA-256 `638bade6b81b9c07391c5eca8688f906e5cfc000a8b5ee1fb38a2af3400b29fd`).
- The N02 card header still says `READY TO IMPLEMENT; NOT RUN`, but that text is stale: the actual pilot summary records four completed fits and `PILOT_COMPLETE_ADVANCE`, followed by a measured continuation that failed its gate. Use those result artifacts, not the old card header, for status.
- N02 pilot advanced on a small primary improvement: Macro F1 0.824496, +0.000523 over current and +0.000358 over its no-sketch control. The later continuation failed its primary gate: 0.829803 versus 0.829944 current (−0.000141), and +0.000193 versus matched no-sketch, below the required +0.0002. Late W5–W6 was −0.000812; W6 was −0.001768. The original text-load Q4 positive-recall guard violation of +0.023062 remains preserved. Pilot and continuation reports are hash-pinned in the source evidence inventory below.
- C09's hand-built text × entry-friction interaction was retired: primary +0.000110, W1 −0.002746, H12 −0.000168. C14's random structured partitions were retired: primary −0.006724, H12 −0.002919, W1 −0.007256, W4 −0.006180; supported Q2 subgroup Macro F1/positive-recall changes were −0.013664/−0.011968 and Q4 changes were −0.013836/−0.036077. These are counterevidence for interaction-heavy approaches, but neither is a learned FM over all sparse feature pairs.
- No N07 implementation, fit bundle, receipt, or result artifact was found under the project `src/`, `scripts/`, `reports/optimization_v3/`, or `experiments/` trees. The only earlier N07 material is the speculative planning entry in `reports/optimization_v3/research_next_v1/RESEARCH_LANES.md`.

N02 therefore reduces the expected value of N07 without disproving it: a fixed random degree-two representation produced little primary gain and weakened late performance, while learned factors impose a different, low-rank parameterization. N07 must beat a same-matrix main-effects control; a comparison only against a weak baseline would not identify the interaction contribution.

## Formula, backend, and synthetic checks

For sparse row (x_i) and rank-16 factors (V), the pair score is

\[
  s_i^{(2)} = \tfrac12\sum_f\left[(x_iV)_f^2 - (x_i\odot x_i)(V\odot V)_f\right]
           = \sum_{j<k}x_{ij}x_{ik}\langle V_j,V_k\rangle.
\]

Repeated feature counts remain multiplicative. A row with only one active feature has no pair effect. At (V=0), the interaction score and its entire factor gradient are zero, so first-order optimization leaves the interaction block stuck there. The fixed proposal uses `numpy.default_rng(42).normal(0, 0.01, (n_features, 16))`; main coefficients and intercept start at zero.

The prototype also checks binary logistic loss plus L2, with an optional fixed offset argument for tests. Central finite differences cover main weights, factors, intercept, and nonzero offset. Sparse/dense parity and explicit `i < j` enumeration pass. **No optimizer is imported or called.** Current environment: NumPy 2.4.2, SciPy 1.18.0, scikit-learn 1.8.0 installed; `torch`, `fastFM`, and `numba` absent. The permitted small custom objective/gradient is about 300 lines including hash-checked benchmarking, not a training framework or vendored FM package. A future solver could use SciPy L-BFGS-B; convergence remains unverified because this milestone prohibits optimizer iterations.

## Label-free matrix and resource audit

Both saved `fit_base` matrices are C2_B structured + TF-IDF + missing-indicator design matrices built with the frozen positional-pilot preprocessing. Source assembly is `build_c2b_arm_matrices` in `src/models/optimization_v3_positional_text_pilot_v1.py:294–321`. The old preprocessing states and row populations are reused; no vocabulary, IDF, scaling, category, or other learned statistic was recomputed. Only 500 deterministic evenly spaced source rows from each matrix (1,000 distinct rows total) were used for rank-16 pair-score/gradient operations. The three timing repetitions reuse those same rows; they do not read or sample additional matrix rows. Archive SHA-256 and logical CSR hashes matched both saved manifests.

The saved routed prediction packages also passed their existing integrity verifiers: E67 `PASS_PREDICTION_INTEGRITY` (20,903 scored rows; 19,528 present, 1,375 missing/blank) and E75 `PASS_PREDICTION_INTEGRITY` (8,491 scored rows; 7,965 present, 526 missing/blank). Each package contains the `prob_missing_hgb` fallback, matching `score_record_ids`, `score_cycles`, and its exact matrix-manifest hash. Verification hashes: E67 `4f5e61ad5f221e5802a4d7ff61318d631e7c239d357f53f511da6dee5bf9f28c`; E75 `438ac29704ff67ec116db5572898685bd3914c31121007522df90159ace71321`. Reverify these bindings if the pilot is later authorized.

| Origin | Saved FIT shape | NNZ | Sample rows/NNZ | Candidate full evaluation estimate | 200-evaluation candidate + control estimate | Modeled working set* |
|---|---:|---:|---:|---:|---:|---:|
| E67 | 29,705 × 20,285 | 2,259,013 | 500 / 38,111 | 0.268 s | 54.42 s | 118.8 MB |
| E75 | 42,117 × 20,371 | 3,227,773 | 500 / 38,053 | 0.339 s | 68.97 s | 142.4 MB |

Times are three-repeat sample medians scaled linearly by saved-matrix NNZ; the control estimate times the same main-effect score/gradient path. The pair-branch row normalization is also scaled once per candidate matrix. The total approximate four-fit cost is about **123 seconds at the proposed 200 function-evaluation cap**, before general runtime overhead. These are estimates, not full-fit timings. *Modeled memory is an explicit buffer estimate for two sparse matrices, factor parameters/gradient, L-BFGS history (`maxcor=10`), and a 1,024-row working batch; Python/SciPy allocator overhead and OS peak RSS are excluded. It is not a measured peak.* No dense all-pairs matrix is materialized.

## One proposed Classification pilot card

This is the sole candidate card from this preflight; do not start it without a separate protocol/budget decision.

- **Population and inputs:** fit cutoffs 67 and 75 on the exact saved C2_B `fit_base` matrices; score their matching frozen `score_base` rows. Reuse only after archive/logical hashes and row/cycle identity pass. Existing C2_B and current-route predictions are comparison baselines. Reuse the same-origin saved C1_B missing-stream expert only if its stored verification and row alignment still pass; otherwise this four-fit card is `NO_GO`, with no extra fit.
- **Candidate:** binary logistic FM, rank 16, `logit = X_main @ w + b + FM(L2_row_normalize(X_main), V)`. There is no score offset, no probability residual correction, no A02 residual reuse, no probability calibration, and no learned feature crosses outside this FM. The main effects `w` and intercept `b` are jointly learned with `V`; threshold remains 0.50.
- **Loss and regularization:** mean binary cross-entropy plus \(\frac{1}{2Cn}(\|w\|_2^2+\|V\|_F^2+b^2)\), fixed `C=1`, `n=fit_rows`; bias is regularized as the `intercept_scaling=1` synthetic feature in the existing liblinear recipe. The C2_B main block is unchanged; only the interaction branch is row-L2-normalized. No class weights or loss rescaling.
- **Inference policy:** on present-stream rows use the FM expert probability directly; missing/blank rows stay exactly on the verified saved HGB fallback (`prob_missing_hgb`). The frozen C2_B bag route and frozen current route are comparison baselines only; no probability blend or additive logit offset is applied. If the current-route predictions cannot be row-ID/cycle aligned to each view from saved artifacts, stop before any fit.
- **Matched control:** identical C2_B rows, logistic objective, main effects, L2, initialization for `w/b`, solver and routing, but `V` is absent. This tests whether adding the learned pair block helps beyond the same main effects and optimization path. It does not remove the candidate's extra parameter capacity or prove learned-factorization superiority over a capacity-matched random interaction map; N02 is contextual evidence, not a row-matched control for these origins.
- **Solver and limits:** SciPy `minimize(method="L-BFGS-B")`, full-batch objective/gradient accumulated in fixed 1,024-row chunks, `maxiter=200`, `maxfun=200`, `maxcor=10`, `ftol=1e-9`, `gtol=1e-4`; one seed (42), no retry, no parameter/seed/regularization sweep. Run one fit at a time with two BLAS threads, respect the existing 6-GiB job cap and 2-GiB free-memory reserve, and stop if either fit fails convergence or numerical guards.
- **Validation:** thresholded Macro F1 for the full routed output versus the frozen current route; also report the C2_B bag route and matched no-pair control. E67 is the primary temporal view (cycles 68–79, also W4/W5/W6); E75 W6 (cycles 76–79) is a later-fit descriptive robustness view and shares the same W6 rows as E67, so it is not independent replication. Carry forward the frozen N02 pilot gate magnitudes: primary gain ≥0.0005 versus current and ≥0.0002 versus matched control; H12 harm ≤0.001; individual-window harm ≤0.005; supported subgroup Macro-F1 harm ≤0.01 and positive-recall harm ≤0.02 with n≥200 and at least 30 examples of each class. Report all negative views and subgroup values. Never select on Kaggle.
- **Cost:** 2 FM candidate fits + 2 no-pair control fits = **4 new predictor fits**, 0 learned preprocessing fits if saved matrix and fallback reuse checks pass, 0 retries. Current live Research Next reservations already consume the unchanged 109-fit cap; this proposal has no reservation and would need an explicit future budget decision. No ledger rows were added here.

## Reproducibility and evidence inventory

From the repository root:

```powershell
python -m pytest -q tests/test_optimization_v3_learned_pair_preflight_v1.py
python scripts/optimization_v3_learned_pair_preflight_v1.py --benchmark --output reports/optimization_v3/learned_pair_preflight_v1/BENCHMARK.json
```

Synthetic QA: **6 passed**. The benchmark reports `PASS_LABEL_FREE_RESOURCE_PREFLIGHT`, 1,000 distinct source rows, 0 fit/optimizer calls, and 0 target/label files opened. It verifies E67 archive/logical hashes `1f6ebc9b… / 29e2e289…` and E75 `f6dbc38d… / fab4612b…`; both point to frozen input SHA `875552a6…`, source commit `3429fb93…`, and freeze signature `f310d034…`. The unchanged live budget ledger SHA-256 at start and end is `e6bf5051d4f45c3654ae6ce21f31437f30662db03e9c26e63f1051b2c7b4dc2b`.

Implementation hashes: prototype `0741f610eaf3965a4b781636342334ac8d9fab1dd1c58352264892e4d574c554`; synthetic tests `853e5a5201ff6695654dff81ffb249bd2a96382ec9dbfb6d36a83e0a47ea6bd4`; benchmark record `7193f42a2b212689433d4e72a4e6267311e5a7e82a18a8a364b353401a8eba94`.

Prior evidence hashes: N02 pilot summary `00c04c8ec9ce2886afc4c17646b96c4e42b475b15197d25fc95c862742bdfc89`; N02 measured continuation `7682ff435b9f24bdd451481e5cd918441374391435dc4110482d98ec2c6faef2`; C09 results `758b57faf2a6e3105705525fc3366292a89aae5d186e83a1de6f10b21e022a6b`; C14 results `e0926c31f511af6037bd6cb4c9a7915c9845ea26c913b75538fd95cd0dec76e6`.

All artifacts here are confined to the new `learned_pair_preflight_v1` namespace plus its explicitly named test and script. Historical reports, recipes, data, matrices, and the live ledger are unchanged. No Test/Sample/Kaggle access, regression-label access, package installation, notebook work, or push.
