# POS frequency ridge: mean-one coefficient/logit diagnostic R1

## Scope and provenance

This is an additive zero-fit decomposition of already-frozen gamma vectors and score-position matrices. It reads no new target labels, creates no predictions, and does not compute counterfactual performance.

The row partition is exact: frozen ALIGNED token weight `w < 1` versus `w >= 1`, repeated across the three positional blocks. No new bins or cut points were selected.

MIX50_ORIGINAL is a code label for the current primary reference; it is not the archived historical A02 vector on missing-stream rows. Scores are unchanged.

The primary comparison baseline uses original uniform POS probability and pA on missing-stream rows. The historical A02 MIX50 archive instead uses pP on missing-stream rows. These are distinct references; previously computed scores are unchanged.

## Frequency partition and coefficient energy

`w < 1` means the aligned mean-one fit applies a lower per-coordinate ridge penalty than uniform weight 1. Coefficient-energy and fitted-penalty quadratics below describe each already-fitted candidate; they are not objective values comparable across different penalties.

| Origin | Partition | Tokens | Median prefix DF | Aligned w median |
|---:|---|---:|---:|---:|
| 67 | w_below_1 | 2,956 | 55.0 | 0.581488 |
| 67 | w_at_least_1 | 2,719 | 10.0 | 1.363710 |
| 75 | w_below_1 | 2,961 | 76.0 | 0.575555 |
| 75 | w_at_least_1 | 2,762 | 14.0 | 1.341002 |

| Origin | Candidate | Total coefficient L2 | w<1 energy share | w<1 fitted penalty quadratic* | w>=1 fitted penalty quadratic* |
|---:|---|---:|---:|---:|---:|
| 67 | UNIFORM | 30.5688 | 76.0% | 709.825 | 224.628 |
| 67 | ALIGNED | 41.6264 | 92.3% | 873.209 | 170.676 |
| 67 | PERMUTED | 36.6452 | 72.8% | 737.398 | 252.243 |
| 75 | UNIFORM | 34.2846 | 72.4% | 850.494 | 324.94 |
| 75 | ALIGNED | 45.2472 | 90.3% | 1020.29 | 253.54 |
| 75 | PERMUTED | 41.0993 | 71.0% | 888.542 | 352.165 |

*Candidate-specific `sum(w_j * gamma_j^2)` without the objective's factor 1/2; descriptive only.

## Position-logit movement: ALIGNED minus UNIFORM

For each scored row, `delta = X_position @ (gamma_ALIGNED - gamma_UNIFORM)`. The two frequency partitions sum back to total delta; the reported reconstruction error checks this numerically.

| Origin | Contribution | Mean signed | Mean absolute | P95 absolute | L1 magnitude share |
|---:|---|---:|---:|---:|---:|
| 67 | total | -0.0554665 | 0.146344 | 0.391095 | 100.0% |
| 67 | w<1 | -0.0556071 | 0.146215 | 0.390814 | 95.6% |
| 67 | w>=1 | +0.000140511 | 0.00665478 | 0.0324907 | 4.4% |
| 75 | total | -0.0433097 | 0.1292 | 0.339264 | 100.0% |
| 75 | w<1 | -0.0435346 | 0.128454 | 0.340184 | 94.4% |
| 75 | w>=1 | +0.000224861 | 0.00757553 | 0.0391254 | 5.6% |

The partition L1 shares add to one by definition. Total absolute movement can be smaller because the two signed contributions cancel across partitions; the JSON reports the cancellation fraction and reconstruction error. These are logit diagnostics, not probability or metric counterfactuals.

## Previously observed outcome evidence

The already-committed R1 results show that ALIGNED lost to both PERMUTED and UNIFORM on the mechanism gate at E67 H12 and E75 W6. Competition Macro F1 gains versus the recorded primary comparator were negative on E67 H12, E75 W6, and the unique pool. The per-window harm and supported-subgroup guards passed. No result is recomputed in this diagnostic.

## Assessment and next hypothesis

If `w<1` accounts for a majority of ALIGNED-versus-UNIFORM logit-change magnitude, that supports the narrow descriptive claim that relaxed frequent-token penalties are a major source of fitted logit movement. It does not explain why the movement is useful: the frozen outcome evidence is counterevidence because aligned weighting did not improve the saved evaluation.

At most one prospective hypothesis is suggested: test a one-sided map that never lowers frequent-token penalties below uniform weight 1 while retaining increased penalties on the rare-token side. This changes average penalty strength, so a future protocol would need a preregistered global-strength control and the same frozen temporal comparisons. This report does not authorize any fit or continuation.

## Limitations

This is a post-result diagnostic chosen after observing failed R1 gates, so it is adaptive rather than confirmatory. The split `w<1`/`w>=1` is exact and inherited from the fitted map, but decomposing a linear score by feature blocks does not isolate causal effects: coefficients are jointly fitted, the partitions may cancel, and the result depends on the saved score matrix and fitted gammas. No token semantics, new target reading, Test scoring, or public leaderboard evidence is involved.

Freeze SHA256: `c1051aac6f6ca7ac9d5440c769d7bc8e9c9b3fcd23bedc65bbafb9013225483b`.
