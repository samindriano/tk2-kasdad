# Optimization V2 — research checkpoint

Starting point: `6bd5250`, 31 October 2026. This campaign owns only new Optimization V2 files. All 666 preexisting tracked files, including EDA, academic notebooks, accepted manifests and teammate work, are protected by a byte-hash inventory in `experiments/optimization_v2/milestone.json`.

## Live competition baseline

Authenticated Kaggle SDK checks: `kasdud lovers` is second of four returned Classification teams at Macro F1 0.82701, 0.00081 below the public leader; first of two returned Regression teams at MAE 11.64134. Retrieved leaderboard rows cover the full returned pages. These positions are a dated snapshot, not a prediction of final/private rank. The rules still allow only assistant-provided data and prohibit copying other groups. No competitor solutions or private information were accessed.

Initial remaining daily allowances: Classification8, Regression6; total team submissions3/9. API deadline2026-11-08 16:55UTC. The new campaign budget is at most two worthwhile submissions per task, not a quota, and the gate preserves at least one shared live slot. No new upload has occurred at this interim checkpoint.

## Prior-experiment audit and implications

| Prior evidence | What is already answered | Decision for this campaign |
|---|---|---|
| `docs/CLASSIFICATION_ROUND1.md`, `reports/classification_round2/FINDINGS.md` | Majority/lookup, structured LR/HGB, text-only/hybrid LR, missing indicators, text weights0.5/2, past-only threshold policies and availability routing | Keep hybrid present and CatBoost missing as control. Do not repeat text-weight or indicator sweeps. |
| `reports/classification_ttr/FINDINGS.md` | Count/TTR/availability scalars add little beyond TF-IDF; structured-only TTR gain does not transfer to hybrid | No further TTR/scalar variants; formal EDA remains untouched. |
| `reports/autonomous_eda/classification/FINDINGS.md`, `classification_volume/FINDINGS.md` | Uniform availability specialists lose; q80 specialist has small internal gain but public0.82640 vs C4 0.82641 | No subset-fit/q80 rerun or threshold adjustment from public. |
| `reports/dual_campaign/classification/FINDINGS.md` | SVC and word+char fail; NB-weighted LR improves AP, not fixed-cutoff F1 | Investigate strictly past-only calibration/score combination once, rather than another lexical parameter grid. |
| `reports/dual_campaign/classification_catboost/results.json` | Structured CatBoost is a stronger missing expert; present expert has not been replaced with native-text boosting | Test one native-text CatBoost family and a low-dimensional text/structured score stack. |
| `docs/REGRESSION_ROUND2.md`, `docs/REGRESSION_RESEARCH.md`, `reports/regression_research/FINDINGS.md` | Structured feature/history ablations, HGB, linear Ridge, native-tabular LGB, text-only/hybrid Ridge, HGB+SVD32/128, no-scope, structured Ridge, LGB+SVD32 and rank policies already tested | No new SVD-dimension, structured-only or old ablation sweep. Test sparse L1 and raw sparse-text boosting instead. |
| `reports/autonomous_eda/FINDINGS.md` | Coverage scalar and new SVD128 fail external improvement | Do not repeat those variants. |
| `reports/dual_campaign/regression/STAGE1_PAIRED_BLEND.md`, `catboost/STAGE2_CATBOOST.md` | Ridge25 improves internal/public, Ridge50 loses public; CatBoost has weak tails/public and original CB/LGB blend gate fails | Use exact Ridge25 incumbent. A fresh two-family sparse representation is higher information than tuning old public blend weights. |

The independent read-only audit also explored CatBoost/Ridge25 convex combinations after observing their outcomes. Shares0.10/0.25/0.50 improve reused primary/H12, but tails worsen as CatBoost share increases. These are explicitly posthoc diagnostics, not preregistered model-selection evidence. They do not override the earlier failed CB/LGB gate and are not being uploaded as a public rescue.

## Completed Classification score-head studies

All comparisons use the current champion, six old temporal cutoffs, and the exact same cached CatBoost missing branch. Base model fit, score-head fit, and outer evaluation use different temporal partitions. H12 reuses primary outcomes; repeated development remains adaptive.

| Present expert | Primary Macro F1 | H12 pooled | Decision |
|---|---:|---:|---|
| Current C2_B | 0.827646 | 0.798984 | Retain |
| Recent-four calibrated C2_B | 0.822238 | 0.796440 | Retire |
| Recent-four calibrated NB-weighted LR | 0.821600 | 0.795069 | Retire |
| Recent-four two-score C2_B/NB stack | 0.823571 | 0.797504 | Retire |
| Expanding OOF calibrated C2_B | 0.826368 | 0.797986 | Retire |
| Expanding OOF C2_B/structured-CatBoost stack | 0.827527 | 0.797434 | Retire; close primary, no H12 improvement |

The first study trained each head on present rows from E−3..E, using forecasts from models fitted only through E−4. The second used disjoint four-cycle OOF blocks52..E, each forecast made before its labels. It added a structured nonlinear score, with expanding calibration support. No threshold or regularization search was performed. Fixed cutoff0.50 remains. The two studies added only C2_B/NB fits through51 and one structuredCatBoost fit through51; all other frozen score vectors were reused.

The recent-head verifier independently recomputed120 metric groups; the expanding-head verifier independently rebuilt12 past-only sklearn heads and reconciled90 metric groups. Missing predictions stayed exactly identical. These failures suggest that improved ranking or an additional structured score does not automatically survive temporal calibration/refit shift. No claim is made about a unique cause.

Method references: [scikit-learn calibration](https://scikit-learn.org/stable/modules/calibration) and [stacking](https://scikit-learn.org/stable/modules/ensemble). Calibration/meta-training uses out-of-fit predictions and only past labels, following the separation required by these methods. Installed sklearn is pinned by each receipt; this is an explicit temporal implementation, not the default random-CV stack.

## Remaining live work

Regression sparse-L1 and sparse-LightGBM fixed blends passed the two-cutoff pilot; eight remaining fits are in progress under a separate continuation freeze. Standalone rank50 policies failed and are not external candidates. SGD reaches its fixed iteration budget: report approximate optimization and warnings rather than claiming convergence.

One native-text CatBoost Classification pilot is being implemented and tested before its real-data freeze. Full-six continuation and external selection depend on the registered gates. Academic/EDA artifacts and old champions remain unchanged.
