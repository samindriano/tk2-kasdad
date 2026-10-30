# Stage5 — fixed text representation with LightGBM MAE

Hypothesis: combine the established predictive text representation with a stronger tabular estimator. The representation is the same native structured + TF-IDF/SVD32 + missing indicator used by HGB. Only two earlier LightGBM configurations are reused; their parameter ID and mapping weight are jointly chosen from past-only inner labels. This is a model × information interaction, not a new model zoo or a leaderboard grid.

| Policy | Primary MAE | Equal-cycle MAE | H12 MAE | EF0 | EF1 | Low-tail MAE | High-tail MAE |
|---|---:|---:|---:|---:|---:|---:|---:|
|raw|12.047761|12.398071|14.786377|13.051519|9.962761|13.684962|10.581931|
|inner|11.931600|12.283548|14.704041|12.979047|9.755852|12.235391|9.495441|

Inner-selected policy improves all six primary windows and both H12 windows against R1-D expanding. H12_E remains difficult17.473257; H12_L12.361143. Hybrid Ridge remains the best pooled H12 model at13.938115, so this is not a uniform horizon winner. Primary11.931600 improves over structured LightGBM13.064889 and SVD-HGB12.366181. The gain is chiefly extra information × nonlinear modeling; mapping contributes a smaller raw12.047761→11.931600 improvement.

Mean signed error remains negative (−0.418158). Low/high-tail MAEs12.235391/9.495441 improve substantially over R1-D18.221372/11.514815. Missing-stream MAE13.041813 trails stream-present11.852119; no automatic routing or ensemble was added. Inner mapping slightly worsens missing MAE relative to raw12.972384 while improving the much larger present group.

Public scores12.55466/12.34519 were known before Stage5; the decision is explicitly adaptive. Configurations were not chosen by matching public scores and no artificial labels/corrections were assigned to the novel Test cohort. All validation has been reused, H12 is not independent, and early-H12 risk must stay with any provisional recommendation.

Independent Stage5 audit PASS:18 logical model-fit references /15unique model pairs,6inner selections/60weight scores,16outer jobs/4aggregates, fit-only source/target fingerprints and exact saved mappings. LightGBM4.6.0 is recorded in every fitted model state, matching Stage23 runtime; no dependency upgrade occurred.
