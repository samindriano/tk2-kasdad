# Stages 2–3 — fixed text recipes and inner-selected LightGBM

Frozen text Ridge alpha10, hybrid Ridge alpha10 and native-structured HGB+TF-IDF/SVD32. LightGBM MAE has three predeclared complexity settings. Each fit cutoff uses past inner4cycles to choose mapping weight; LightGBM chooses parameter and mapping jointly. No outer labels choose per-fold parameters or mappings. Each recipe also reports fixed raw/.5/fullrank as descriptive controls.

| Recipe/policy | Primary MAE | Equal-cycle MAE | H12 MAE |
|---|---:|---:|---:|
|text_ridge__raw|18.753454|18.984847|19.936100|
|text_ridge__rank50|18.275499|18.468351|19.459078|
|text_ridge__rank100|19.577156|19.825679|20.781912|
|text_ridge__inner|18.274839|18.465759|19.474794|
|hybrid_ridge__raw|13.242685|13.466716|14.706501|
|hybrid_ridge__rank50|12.571665|12.807672|13.994865|
|hybrid_ridge__rank100|12.775901|13.027425|14.192617|
|hybrid_ridge__inner|12.551305|12.790073|13.938115|
|svd_hgb__raw|12.448467|12.756056|14.783460|
|svd_hgb__rank50|12.359996|12.675064|14.743361|
|svd_hgb__rank100|12.741306|13.064635|15.111947|
|svd_hgb__inner|12.366181|12.677743|14.714120|
|lgb_inner__raw|13.122836|13.385072|15.640152|
|lgb_inner__rank50|13.100666|13.360430|15.667312|
|lgb_inner__rank100|13.512056|13.776600|16.091415|
|lgb_inner__inner|13.064889|13.322505|15.604153|

Text-only regression is weak. Hybrid text Ridge improves long-horizon results; SVD-HGB improves primary most. LightGBM structured has only a modest gain and does not improve H12 enough to displace the text recipes. Ranking helps the compressed Ridge scores more than the HGB scores, showing that the usefulness of percentile mapping depends on the underlying representation.

Primary/H12 are reused adaptive development views, not independent confirmation. Different fixed weights cannot be selected from outer labels and presented as untouched validation. Promotions use the frozen inner-selected recipe. Formal EDA remains unchanged. Text is supplied-schema eligible for this authorized experiment, with operational timing still provisional.

Independent promotion audit: exact fit-row fingerprints, inner argmin/rank map,128jobs/32aggregates, EF/tail/stream groups and equal-cycle MAE reconciled. More detailed independent stage verifier is separate. No full redundant model replay.

Next bounded checks:128 latent text dimensions (whether SVD32 loses informative detail), no-scope SVD32 (reduce reliance on novel scope support), and same-family structured-only Ridge to isolate incremental text information. These checks will be frozen separately before fitting. No zero-to-median imputation or fabricated Test targets.
