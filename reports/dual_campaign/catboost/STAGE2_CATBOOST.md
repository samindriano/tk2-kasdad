# Stage 2 — CatBoost MAE + structured/SVD32

## Protocol and source boundary

- Frozen candidate: `dual_regression_catboost_svd32_v1`; CatBoost 1.2.8.
- Fixed CPU recipe: MAE, 500 iterations, depth 6, learning rate .05, seed 42, two threads. Native categorical columns; structured parser/imputer, TF-IDF and SVD32 are learned on each temporal fit partition.
- Pilot cutoffs 55 and 67 were evaluated first against frozen LightGBM and Ridge25 predictions. The predeclared gate uses clipped MAE across W1/W4/H12_E/H12_L; full six runs only when mean delta vs Ridge25 <= +0.50 and no view > +1.00.
- No Test access or external data. The cached predictions are derived Train validation vectors; no model binaries are saved.

## Pilot gate

Decision: **continue with all six**; mean delta=+0.100754, max delta=+0.424256.

| View | CatBoost clipped MAE | Ridge25 MAE | Delta |
|---|---:|---:|---:|
| W1 | 15.770882 | 16.011308 | -0.240426 |
| W4 | 12.090357 | 11.666102 | +0.424256 |
| H12_E | 16.964983 | 17.043582 | -0.078600 |
| H12_L | 12.188138 | 11.890354 | +0.297785 |

## Full six-window results

Primary MAE raw/clipped: 11.823789/11.809633; equal-cycle MAE=12.138116.
H12 pooled MAE raw/clipped: 14.392387/14.377381; H12_E=16.964983; H12_L=12.188138.

| Window | CatBoost clipped MAE | LGB clipped MAE | Ridge25 clipped MAE | Δ vs Ridge25 |
|---|---:|---:|---:|---:|
| W1 / n=4,114 | 15.770882 | 16.405781 | 16.011308 | -0.240426 |
| W2 / n=6,493 | 11.473634 | 12.042870 | 12.151934 | -0.678300 |
| W3 / n=7,078 | 12.279010 | 12.354210 | 12.409597 | -0.130587 |
| W4 / n=4,950 | 12.090357 | 11.999389 | 11.666102 | +0.424256 |
| W5 / n=7,462 | 10.865350 | 10.774911 | 10.595422 | +0.269927 |
| W6 / n=8,491 | 10.422217 | 10.303425 | 10.122384 | +0.299833 |
| H12_E / n=17,685 | 16.964983 | 17.473257 | 17.043582 | -0.078600 |
| H12_L / n=20,903 | 12.188138 | 12.361143 | 11.890354 | +0.297785 |

Corrected after row-aligned independent reconciliation: signed-error correlation=0.9478; CatBoost wins 50.7% of paired rows, LGB 49.1%. The initial summary aligned window-ordered predictions with target values in a different order; model predictions and MAE results are unchanged.
The fixed 25% CatBoost blend was not evaluated: corrected correlation 0.9478 exceeds the predeclared 0.93 limit, even though CatBoost wins 50.7% of rows.

Figure: `stage2_catboost_windows.png` shows relative clipped MAE for the available pilot/full windows.

## Interpretation

All folds reuse Train cycles previously used in the campaign. These results are adaptive development evidence, not an untouched estimate. Review raw/clipped metrics, each window and subgroup in `stage2_catboost.json`; no Kaggle submission or model promotion is performed here.
