# Stage4 — text resolution, scope dependence and same-family control

Three fixed recipes were frozen before either new Kaggle public score was available. Rank weights use the same past-only inner grid; no outer weights are selected.

| Recipe | Policy | Primary MAE | H12 MAE |
|---|---|---:|---:|
|svd128|raw|12.415152|14.731770|
|svd128|inner|12.299353|14.648435|
|svd_noscope32|raw|12.435756|14.814872|
|svd_noscope32|inner|12.352186|14.739826|
|structured_ridge|raw|14.659216|15.709524|
|structured_ridge|inner|14.178654|15.205818|

**Text contributes beyond model-family changes:** structured-only Ridge with the exact same alpha/scaler/OHE scores14.178654/15.205818 after inner mapping, versus hybrid Ridge12.551305/13.938115. The control replaces every stream with a constant empty string for fit/predict; the difference includes text content plus variable availability, not only token coefficients.

**SVD128 improves modestly:** primary12.299353 and H1214.648435 versus SVD32inner12.366181/14.714120. Both early and late H12 improve, but the model remains early-H12-sensitive and still trails hybrid Ridge on pooled H12. All HGB settings are unchanged; this tests representation resolution.

**Removing scope is not a clear win:** no-scope SVD32 primary12.352186/H1214.739826 versus full SVD32inner12.366181/14.714120. Primary improves by only.013995 and H12 worsens. This does not support promoting the ablation solely because Test has a novel zero cohort. No zero-to-median replacement is applied.

No final model is chosen. All windows are reused development evidence, H12 overlaps primary, and held-out source-cycle completeness is still unknown. A separate unscored perturbation probe examines prediction sensitivity; it does not estimate actual error on the Test cohort.
