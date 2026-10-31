# V3 Regression — cache-only residual diagnosis

Generated from commit `78d6c67f6a243e91f91a4a23bfb1de2781538f03`; no estimator fits, Kaggle calls, or Test reads were performed.
The six-window primary OOF and H12 arrays are reused development outcomes; H12 overlaps primary and is not replication. No outer-derived subgroup is a validation partition or a model-selection gate.

## Reconstructed incumbent

Recipe: `0.5625 × LGB SVD32 + 0.1875 × Hybrid Ridge + 0.25 × sparse LGB rank50`, clipped to [0,100]. It is reconstructed from existing row-exact arrays; the Ridge25 cache decomposition is checked to 1e-10.

| View | Rows | MAE | Mean signed residual | Median signed residual | p95 absolute error | + / − cycle mean residuals |
|---|---:|---:|---:|---:|---:|---:|
| Primary six windows | 38,588 | 11.5594 | -0.557 | -0.631 | 33.315 | 9 / 15 |
| H12_E | 17,685 | 16.9266 | +1.333 | +1.916 | 44.225 | 12 / 0 |
| H12_L | 20,903 | 11.6117 | -1.296 | -0.889 | 31.370 | 1 / 11 |

## Key diagnostic signals

Primary per-cycle residuals and fit-defined quantile strata are in `diagnostics.json` and `experiments/optimization_v3/regression_diagnostics/tables.json`. Each quantile edge is estimated from rows at or before that outer cutoff; tied edges collapse bins. Pooled minimum group support is 100 and per-window contrast support is 50 per side. Raw category novelty is measured against the fit's observed values, with missing kept separate; it is not identical to a model's rare-category or unknown encoding.
Across the six primary folds, several conditional directions recur with adequate per-fold counts: EF1 minus EF0 mean residual is positive in 6/6 folds; any unseen raw org/module/pattern value minus all-observed-seen is positive in 6/6; and the descriptive low-tail minus high-tail mean residual is positive in 6/6. The latter contrast ranges +13.07 to +32.10 points, with each low/high tail side supported by at least 205 rows per fold. Stream-missing minus present is positive in 5/6 folds (342–526 missing rows/fold). Impact Q5−Q1 is positive in 5/6; text_load Q5−Q1 is negative in 5/6 but reverses in W1. Scope-pressure and parsed scope-trace Q5−Q1 are mixed, so those pooled gradients are not stable enough to state as a consistent direction.

Target tails (≤5, ≥95) are descriptive, selected by the observed target after scoring, and cannot support a route or validation claim. Signal missingness uses null/empty/whitespace-only. EF and other supplied features are observed values.
The tail shape is asymmetric shrinkage (low targets predicted high, high targets predicted low), repeated in all six primary folds and both H12 eras. It motivates a fixed-bin distributional / CDF-median comparison under MAE, but does not establish conditional multimodality or imply that binning will beat the L1 point estimate.

The primary result should be read as error structure in repeatedly reused temporal development support, not a fresh estimate. The six primary folds contain 38,588 unique Train rows; H12_E and H12_L re-score those rows with overlapping temporal views.

## Prior research audit and next hypotheses

The attached machine summary records the covered prior families. V1/R1–R2 already tested structured HGB and feature/history ablations; hybrid word TF-IDF/Ridge; SVD32/128 and LightGBM-SVD32; rank policies; CatBoost; and fixed blend shares, including sparse-LGB promotion in V2. Do not repeat those grids.

Ranked as information-gathering hypotheses, not selected candidates:

1. **Past-only L1 residual corrector:** train the agreed shallow correction head on earlier, strictly cross-fitted incumbent residuals and component predictions plus fit-available cycle/EF/availability/parsed-volume covariates; roll it forward without same-row label reuse. Cycle drift and novelty strata motivate checking stability, but pooled subgroup patterns alone do not validate the corrector.
2. **Fixed-bin distributional regression:** compare a predeclared ten-bin threat_rank classifier plus predicted-CDF median against the L1 point model. The primary low tail is overpredicted (+11.22 mean residual) and high tail underpredicted (−8.85), with the same direction in H12_E (+22.85/−13.32) and H12_L (+9.13/−6.96). This supports testing asymmetric tail calibration as a different objective; it does not demonstrate conditional multimodality, and a ten-bin approximation may lose precision.
3. **Character-level sparse text channel:** compare a fixed character n-gram representation against existing word-token sparse/SVD families, then assess incremental residual complementarity. Character channels are not part of regression's prior V2 family; do not assume they help from classification results.
Cross-fitted supervised token target statistics remain a later alternative, with strict earlier-label encoding and unseen-token fallback; high token sparsity and leakage risk make them lower priority than the three above.

These are recommendations for MAIN to freeze or reject; this lane did not evaluate candidates, scores, or public data.

## Early OOF cache boundary

The raw end-51 LightGBM SVD32 and Hybrid Ridge arrays each contain 6,220 fit rows (47–51) and 5,800 disjoint scored rows (52–55), and both match PASS verification receipts. They are valid predecessor component OOF predictions. No sparse-L1 LightGBM prediction exists at cutoff 51, so they cannot reproduce the exact three-component champion. Further, the historical SVD32 rank mapping at the 55 cutoff was selected using 52–55 labels; applying that selected mapping to those same rows would contaminate a residual-head training target. Begin exact incumbent residual-head development with the 56–59 OOF block (cutoff 55), score 60–63 (cutoff 59), then expand past OOF history forward.

## Exact source and reuse handles

Cache/input SHA256s, fold keys, row counts, and the row-aligned OOF prediction payload are recorded in `diagnostics.json` and the ignored `experiments/runs/optimization_v3_reg_diagnostics/oof_residuals.npz`. Re-run `python scripts/optimization_v3_reg_diagnostics.py` from repository root to reproduce.
