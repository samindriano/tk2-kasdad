# B01 — residual-centering versus local-label-rate ablation

**Status:** DIAGNOSTIC_GATE_FAILED<br>
**Qualification:** DIAGNOSTIC_ONLY_NOT_QUALIFIED<br>
**Training and inference fits:** 0.<br>
**QA:** source, formula, independent confusion replay, subgroup and ledger checks passed.

B01 replaces only the bank OOF probability centering term with the query Sparse25 probability. It uses the same saved top-32 neighbors and squared-cosine weights, shrinkage 4, ±0.05 correction cap, exact fallback and 0.50 threshold. The label-shuffle control permutes binary bank labels with the verified original assignment. No query labels enter either formula.

## Main results

| View | n | Sparse25 F1 | A02 F1 | B01 F1 | B01 label-shuffle F1 | Δ B01 vs A02 | Δ B01 vs shuffle | B01 positives |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| full_six_primary | 38588 | 0.829944 | 0.832045 | 0.832306 | 0.829964 | +0.000261 | +0.002342 | 13296 |
| new_origin_aggregate | 21033 | 0.835171 | 0.837187 | 0.837706 | 0.835047 | +0.000519 | +0.002660 | 7796 |
| late | 15953 | 0.815790 | 0.816958 | 0.818720 | 0.815854 | +0.001762 | +0.002866 | 4580 |
| inherited_H12_L | 20903 | 0.808961 | 0.811159 | 0.815049 | 0.813635 | +0.003890 | +0.001414 | 6606 |

## Seven diagnostic gates

| Gate | Result |
|---|---|
| G1_full_primary_gain_vs_A02_ge_0005 | FAIL |
| G2_new_origin_gain_vs_A02_ge_0005 | PASS |
| G3_late_and_H12_gains_vs_A02_nonnegative | PASS |
| G4_at_least_4_of_5_origins_nonnegative_vs_A02 | FAIL |
| G5_full_primary_gain_vs_B01_label_shuffle_ge_0002 | PASS |
| G6_no_supported_subgroup_harm_over_fixed_guards | PASS |
| G7_source_geometry_label_separation_zero_fit_ledger_integrity_pass | PASS |

The results are adaptive Train-only diagnostics on repeatedly used development rows. H12-L overlaps primary. A `PASS` only means B01 may return for review; it does not qualify the recipe, authorize Test inference, alter the champion, or authorize Kaggle upload. Descriptive matched-positive-count tables use the frozen root utility and do not select K.

Per-origin, subgroup, confusion, AUC/AP, Brier, both log-loss conventions, endpoint counts and descriptive count-matched results are in `RESULTS.json`. Hashes, zero-fit evidence and integrity checks are in `QA.json`; the recipe and inputs were frozen before score calculation in `FREEZE_V2.json`.

Official Train SHA-256: `875552a68a3d24b0f3e1966d6ddb5671e6f54305a1c526e2f0a76747942ed91c`. Fit ledger SHA-256: `6fcc4b4d427e3ff03139e0e9ee429f529f7c7bf6ad91ce0eb9a52d55e2503147`; 66 starts / 64 saved / 2 historical failures before and after.

The pre-score recipe freeze is recorded in `FREEZE_V2.json`. Completed results are published together as an atomic output bundle.
