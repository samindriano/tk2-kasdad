# Optimization V3 — Classification diagnostics

**Status: descriptive cache audit only. No model fit, Test read, or route selection was performed.**

## Data and method

Official Classification Train: **50,608 rows**, cycles 47–79, SHA-256 `875552a68a3d24b0f3e1966d6ddb5671e6f54305a1c526e2f0a76747942ed91c`. The audited primary predictions cover 38,588 non-overlapping outer rows (36,010 stream-present; 2,578 stream-missing). H12 reuses those outcomes and is a second horizon view, not an independent holdout.

All model scores came from archived past-only prediction caches; comparisons use the frozen 0.50 decision cutoff. Numeric bins use empirical quartile boundaries learned from cycles 47 through each fold cutoff, applied to its outer rows. Unknown category means a nonmissing outer `org_code`, `module_code`, or `pattern_code` absent from that fold’s fit vocabulary. Confidence bins are fixed C2_B probability intervals. Per-window subgroup minimum is n=50; pooled stability requires n≥200 and support in at least four of six primary windows. Fits and labels from outer rows are never used to define features, bins, cutoffs, or a route.

## Temporal error summary

| View | Rows | Champion Macro F1 | Prior C2_B/C1_B route | Missing CB vs C1_B ΔF1 | Present C2_B Macro F1 |
|---|---:|---:|---:|---:|---:|
| Primary six windows | 38,588 | 0.8276 | 0.8255 | +0.0291 | 0.8286 |
| H12 pooled | 38,588 | 0.7990 | 0.7968 | +0.0273 | 0.8005 |

The champion’s primary gain over the former C2_B-present/C1_B-missing route is confined to the missing branch: 70 net additional correct predictions among 2,578 missing-stream rows (208 CB-only correct vs 138 C1_B-only correct). CB improves that branch in W1/W2/W3/W5/W6, but loses W4; H12_E improves and H12_L loses. The C2_B present branch itself is unchanged.

| Window | Present n | C2_B F1 | Raw CB score F1 on present (counterfactual) | CB−C1_B missing ΔF1 | Champion all F1 | Prior route all F1 |
|---|---:|---:|---:|---:|---:|---:|
| W1 | 3,722 | 0.7949 | 0.7692 | +0.0791 | 0.8021 | 0.7936 |
| W2 | 6,124 | 0.8480 | 0.8061 | +0.0399 | 0.8495 | 0.8471 |
| W3 | 6,636 | 0.8316 | 0.7641 | +0.0285 | 0.8285 | 0.8264 |
| W4 | 4,608 | 0.8469 | 0.7284 | -0.0312 | 0.8403 | 0.8426 |
| W5 | 6,955 | 0.8140 | 0.7457 | +0.0077 | 0.8127 | 0.8121 |
| W6 | 7,965 | 0.8147 | 0.7204 | +0.0475 | 0.8134 | 0.8104 |
| H12_E | 16,482 | 0.7912 | 0.7613 | +0.0742 | 0.7927 | 0.7867 |
| H12_L | 19,528 | 0.8080 | 0.6918 | -0.0190 | 0.8037 | 0.8051 |

## Present-stream complementarity and counterexamples

The structured CB_MISSING model was fitted on past data and only routed to missing rows in its original experiment. Its score on present rows is shown strictly as a counterfactual diagnostic. Across 36,010 present rows, direct CB predictions have Macro F1 0.7592 and 8,004 errors, versus C2_B 0.8286 and 5,609 errors. CB fixes 1,979 C2_B errors but breaks 4,374 C2_B-correct rows. It loses every primary window (−2.57 to −11.85 Macro F1 points) and both H12 segments. This strongly argues against treating the cached raw CB score as a generally useful present-stream head input; any future inclusion should be a predeclared ablation, not a presumed gain.

Raw CB also has higher error in every fixed C2_B-confidence band, each supported in all six primary windows: +3.86 pp below .20; +10.86 pp at .20–.40; +2.32 pp at .40–.60; +12.57 pp at .60–.80; +8.95 pp at ≥.80. For C2_B confidence .60–.80 × text_load Q4 it is +20.85 pp (n=849; worse in all six windows). A near-neutral counterexample is confidence .40–.60 × text_load Q2: −0.21 pp pooled, but mixed across windows (2 lower-error, 4 higher-error). No eligible confidence × text_load or confidence × entry_friction cell has raw CB lower error in at least four windows.

Unknown-category rows are sizeable and are not a rescue for raw CB: 17,531 present rows have at least one unseen nonmissing structured category. Raw CB adds +10.23 error pp there, versus +3.25 pp where all three categories are known. Individually, unseen `org_code` has +12.12 pp (n=9,873) and unseen `module_code` +10.22 pp (n=16,173). Unseen `pattern_code` has only 240 rows pooled and two supported windows, so that subgroup is too sparse for a stable conclusion.

Cached present-stream alternatives are also counterevidence against simply adding another fixed text score: DC_NBSVM changes primary Macro F1 by −0.0027 and error by +0.07 pp; on H12 its F1 changes −0.0005 while error improves −0.27 pp. Its fold outcomes are mixed. DC_SVC changes F1 by −0.0119 / −0.0085 and error by +1.29 / +1.15 pp (primary/H12); DC_WORD_CHAR changes F1 by −0.0030 in both views. Their nonzero error complementarity does not imply a stable route gain.

## C2_B residual patterns and support

The C2_B mean signed probability residual is `p − y`: positive is overprediction of the positive class; negative is underprediction. These are held-out descriptive residual patterns, not a selected correction. Confidence bins show a repeatable sign change: in .20–.40, mean residual is +3.72 pp (overprediction in five of six windows); in .60–.80, −3.75 pp (underprediction in five of six). Their fixed-cutoff errors remain 25.47% and 25.76%, respectively. This supports testing a nonlinear conditional residual head more than a global linear score recalibration, while the failed V2 calibration results remain an important warning.

| Fit-defined subgroup | n pooled | C2_B error | Mean C2_B p−y | Residual sign windows (+/−) |
|---|---:|---:|---:|---:|
| C2B_confidence / 0.20-0.40 | 5,693 | 25.47% | +3.72 pp | 5/1 |
| C2B_confidence / 0.40-0.60 | 3,798 | 41.73% | +1.46 pp | 4/2 |
| C2B_confidence / 0.60-0.80 | 3,708 | 25.76% | -3.75 pp | 1/5 |
| C2B_confidence / <0.20 | 16,073 | 5.39% | +0.65 pp | 5/1 |
| C2B_confidence / >=0.80 | 6,738 | 11.16% | +1.39 pp | 4/2 |
| text_load / Q1 | 8,951 | 15.86% | -0.61 pp | 3/3 |
| text_load / Q2 | 9,031 | 14.49% | +1.75 pp | 4/2 |
| text_load / Q3 | 8,511 | 12.69% | +1.33 pp | 4/2 |
| text_load / Q4 | 9,517 | 18.91% | +1.15 pp | 4/2 |
| scope_pressure / Q1 | 10,106 | 17.32% | -0.22 pp | 4/2 |
| scope_pressure / Q2 | 10,134 | 16.80% | +0.56 pp | 4/2 |
| scope_pressure / Q3 | 8,495 | 16.59% | +2.51 pp | 5/1 |
| scope_pressure / Q4 | 7,275 | 10.27% | +1.08 pp | 4/2 |
| any_unknown_category / unseen_any | 17,531 | 17.85% | +0.64 pp | 4/2 |
| any_unknown_category / no_unseen_category | 18,479 | 13.42% | +1.16 pp | 5/1 |

Broad marginal residual signs are weaker than the confidence pattern: text_load Q1 is −0.61 pp with three positive and three negative windows; Q2/Q3/Q4 are +1.75/+1.33/+1.15 pp, each positive in four windows. This is descriptive heterogeneity, not evidence that adding these covariates will improve predictions. `entry_friction` has tied/discrete quartile cutpoints, so its Q labels collapse to two occupied groups; do not read them as four balanced bins.

## Ranked hypotheses for the next frozen pilot

1. **Nonlinear OOF residual head with structured interactions (most justified).** Train only on past-only OOF rows; use current C2_B score plus expanded structured variables (including fit-derived category novelty and cycle as a candidate covariate) to model residual corrections. Include a fixed ablation without the structured CB score because its raw present-stream evidence is strongly negative. The opposite residual signs across C2_B confidence bands motivate nonlinearity; the near-neutral/mixed confidence×text_load cell and H12 reversals are counterexamples that require strict temporal guardrails. This is a hypothesis only, not fitted or validated here.
2. **Sparse word-bigram TF-IDF LightGBM classifier and a pre-fixed small blend with C2_B.** This has a genuinely different feature-learning path from V2’s failed unigram and two-score calibration tests; this diagnostics lane did not score it and leaves it to the separately authorized main lane. Keep its share fixed before fitting and evaluate on the same temporal windows.
3. **Cross-fitted reliability gate by confidence × text_load / category novelty.** Use only if a past-only gate learns a stable benefit on meta-training OOF blocks and is frozen before outer scoring. It ranks below the residual head because raw CB errors are worse in all broad bins, joint cells are mostly negative for CB, and several high-cardinality category cells are sparse. Do not select cells or thresholds from these outer labels.

## Reproducibility and limitation

Run from the repository root with `python scripts/optimization_v3_cls_diagnostics.py`. The script verifies cache row IDs/probability hashes, consumes only official Train and archived predictions, and writes `audit.json` plus an ignored run receipt containing input hashes. No model fitting occurs. All internal windows reuse a single chronology and have already informed prior work; the evidence is adaptive development, not a fresh untouched holdout. The detailed per-fold and subgroup denominator ledger is in `audit.json`.
