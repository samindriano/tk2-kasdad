# CGEOM_V1 — coefficient-informed A02 geometry pilot

**Status:** `CLOSED_GATE_FAIL`  
**Source commit:** `ddd64736155a64e70c5453c921aedcbd910397f8`  
**Freeze:** `fae503bf65ab98ae4847249e808caafc514c34c1e2fbf688a1d4236c7f0b8eaa`  
**Independent scored verification:** `PASS_INDEPENDENT_METRICS_AND_GATES`

## Question and frozen method

CGEOM_V1 tests whether same-origin C2_B LR token-coefficient magnitudes can improve A02 donor retrieval when lexical correction mass is held at the original per-query A02 mass. The permuted arm uses the same multiplier multiset permuted with seed 42. This is an alignment diagnostic, not a formal randomization null or a causal claim.

For each A02 token j, `a_j = 1 + |beta_j| / (1 + |beta_j|)` and `d_j = sqrt(a_j)`. C2_B and A02 vocabularies and IDF arrays were checked for exact identity; coefficients were selected by token string from the C2_B logistic text block. Weighted TF-IDF rows were L2-normalized and top-32 positive cosine donors were searched over the full original bank with source-row-stable ties. The selected squared-cosine weights were rescaled to sum to original A02 `M0(q)`. The correction retained shrinkage 4, clipping ±0.05, the original Sparse25 baseline, 0.50 threshold, and missing/zero-neighbor fallback.

## Primary and history-sensitive results

| View | n | Arm | Macro F1 | Positive recall | Precision | ROC AUC | AP | Log loss (clip 1e-15) | Brier | Confusion [[TN,FP],[FN,TP]] |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---|
| E67_H12 | 20903 | aligned | 0.811820 | 0.744760 | 0.744760 | 0.899910 | 0.792537 | 0.420444 | 0.119278 | [[12459, 1717], [1717, 5010]] |
| E67_H12 | 20903 | permuted | 0.811130 | 0.744017 | 0.743685 | 0.899691 | 0.790287 | 0.430898 | 0.119401 | [[12451, 1725], [1722, 5005]] |
| E67_H12 | 20903 | a02 | 0.811159 | 0.744314 | 0.743540 | 0.899755 | 0.792814 | 0.417247 | 0.119377 | [[12449, 1727], [1720, 5007]] |
| E67_H12 | 20903 | sparse25_baseline | 0.808961 | 0.736584 | 0.743994 | 0.896701 | 0.790661 | 0.381605 | 0.120838 | [[12471, 1705], [1772, 4955]] |
| E67_W4 | 4950 | aligned | 0.845400 | 0.803562 | 0.808800 | 0.915457 | 0.829789 | 0.427158 | 0.110997 | [[2745, 352], [364, 1489]] |
| E67_W4 | 4950 | permuted | 0.844371 | 0.803022 | 0.806941 | 0.915222 | 0.825656 | 0.468466 | 0.111125 | [[2741, 356], [365, 1488]] |
| E67_W4 | 4950 | a02 | 0.844405 | 0.803562 | 0.806609 | 0.915358 | 0.830957 | 0.419169 | 0.111075 | [[2740, 357], [364, 1489]] |
| E67_W4 | 4950 | sparse25_baseline | 0.841418 | 0.791149 | 0.809498 | 0.911881 | 0.825966 | 0.366748 | 0.113077 | [[2752, 345], [387, 1466]] |
| E67_W5 | 7462 | aligned | 0.806064 | 0.723252 | 0.744579 | 0.893558 | 0.781196 | 0.417718 | 0.122247 | [[4499, 589], [657, 1717]] |
| E67_W5 | 7462 | permuted | 0.805530 | 0.721988 | 0.744247 | 0.893134 | 0.779600 | 0.419356 | 0.122461 | [[4499, 589], [660, 1714]] |
| E67_W5 | 7462 | a02 | 0.805176 | 0.722409 | 0.743068 | 0.893292 | 0.781576 | 0.413971 | 0.122400 | [[4495, 593], [659, 1715]] |
| E67_W5 | 7462 | sparse25_baseline | 0.802502 | 0.717355 | 0.740435 | 0.890962 | 0.779854 | 0.387607 | 0.123435 | [[4491, 597], [671, 1703]] |
| E67_W6 | 8491 | aligned | 0.793280 | 0.721600 | 0.699225 | 0.892979 | 0.771235 | 0.418927 | 0.121497 | [[5215, 776], [696, 1804]] |
| E67_W6 | 8491 | permuted | 0.792648 | 0.721200 | 0.698026 | 0.892930 | 0.770391 | 0.419141 | 0.121535 | [[5211, 780], [697, 1803]] |
| E67_W6 | 8491 | a02 | 0.792999 | 0.721200 | 0.698837 | 0.892884 | 0.770734 | 0.419006 | 0.121561 | [[5214, 777], [697, 1803]] |
| E67_W6 | 8491 | sparse25_baseline | 0.791846 | 0.714400 | 0.700667 | 0.889540 | 0.769727 | 0.384991 | 0.123079 | [[5228, 763], [714, 1786]] |
| E75_W6 | 8491 | aligned | 0.819068 | 0.707600 | 0.775537 | 0.911036 | 0.792678 | 0.371440 | 0.107456 | [[5479, 512], [731, 1769]] |
| E75_W6 | 8491 | permuted | 0.819313 | 0.707600 | 0.776218 | 0.911157 | 0.793008 | 0.377640 | 0.107368 | [[5481, 510], [731, 1769]] |
| E75_W6 | 8491 | a02 | 0.819604 | 0.708000 | 0.776656 | 0.911107 | 0.792959 | 0.377625 | 0.107386 | [[5482, 509], [730, 1770]] |
| E75_W6 | 8491 | sparse25_baseline | 0.816599 | 0.698000 | 0.777629 | 0.909973 | 0.791327 | 0.345912 | 0.108428 | [[5492, 499], [755, 1745]] |

### Pooled unique W4 + W6

Denominator: `13441` unique rows = E67 W4 (4950) + E75 W6 (8491); overlap = 0. Origin-specific geometry is kept separate before concatenation.

| Arm | Macro F1 | Positive recall | Precision | ROC AUC | AP | Log loss | Brier |
|---|---:|---:|---:|---:|---:|---:|---:|
| aligned | 0.831211 | 0.748449 | 0.790393 | 0.913905 | 0.810873 | 0.391960 | 0.108760 |
| permuted | 0.830964 | 0.748220 | 0.789959 | 0.913892 | 0.809069 | 0.411089 | 0.108752 |
| a02 | 0.831157 | 0.748679 | 0.790061 | 0.913905 | 0.811560 | 0.392924 | 0.108745 |
| sparse25_baseline | 0.828147 | 0.737652 | 0.791862 | 0.911767 | 0.808551 | 0.353585 | 0.110140 |

## Frozen gates

Overall status: `CLOSED_GATE_FAIL`; gate counts: `{'fail': 2, 'pass': 60, 'total': 62, 'unknown': 0}`.

| Gate | View | Comparator | Gain | Minimum | Status |
|---|---|---|---:|---:|---|
| E67_H12_aligned_vs_A02 | E67_H12 | a02 | 0.000661 | 0.000500 | PASS |
| E75_W6_aligned_vs_A02 | E75_W6 | a02 | -0.000536 | 0.000500 | FAIL |
| E67_H12_aligned_vs_permuted | E67_H12 | permuted | 0.000690 | 0.000200 | PASS |
| E75_W6_aligned_vs_permuted | E75_W6 | permuted | -0.000245 | 0.000000 | FAIL |
| E67_W4_harm_vs_a02 | E67_W4 | a02 | 0.000995 | -0.005000 | PASS |
| E67_W4_harm_vs_permuted | E67_W4 | permuted | 0.001029 | -0.005000 | PASS |
| E67_W5_harm_vs_a02 | E67_W5 | a02 | 0.000889 | -0.005000 | PASS |
| E67_W5_harm_vs_permuted | E67_W5 | permuted | 0.000535 | -0.005000 | PASS |
| E67_W6_harm_vs_a02 | E67_W6 | a02 | 0.000281 | -0.005000 | PASS |
| E67_W6_harm_vs_permuted | E67_W6 | permuted | 0.000632 | -0.005000 | PASS |
| subgroup:E67_H12:availability:missing | E67_H12 |  | nan | nan | PASS |
| subgroup:E67_H12:availability:present | E67_H12 |  | nan | nan | PASS |
| subgroup:E67_H12:text_load_quartile:Q1 | E67_H12 |  | nan | nan | PASS |
| subgroup:E67_H12:text_load_quartile:Q2 | E67_H12 |  | nan | nan | PASS |
| subgroup:E67_H12:text_load_quartile:Q3 | E67_H12 |  | nan | nan | PASS |
| subgroup:E67_H12:text_load_quartile:Q4 | E67_H12 |  | nan | nan | PASS |
| subgroup:E67_H12:entry_friction:0 | E67_H12 |  | nan | nan | PASS |
| subgroup:E67_H12:entry_friction:1 | E67_H12 |  | nan | nan | PASS |
| subgroup:E67_W4:availability:missing | E67_W4 |  | nan | nan | PASS |
| subgroup:E67_W4:availability:present | E67_W4 |  | nan | nan | PASS |
| subgroup:E67_W4:text_load_quartile:Q1 | E67_W4 |  | nan | nan | PASS |
| subgroup:E67_W4:text_load_quartile:Q2 | E67_W4 |  | nan | nan | PASS |
| subgroup:E67_W4:text_load_quartile:Q3 | E67_W4 |  | nan | nan | PASS |
| subgroup:E67_W4:text_load_quartile:Q4 | E67_W4 |  | nan | nan | PASS |
| subgroup:E67_W4:entry_friction:0 | E67_W4 |  | nan | nan | PASS |
| subgroup:E67_W4:entry_friction:1 | E67_W4 |  | nan | nan | PASS |
| subgroup:E67_W5:availability:missing | E67_W5 |  | nan | nan | PASS |
| subgroup:E67_W5:availability:present | E67_W5 |  | nan | nan | PASS |
| subgroup:E67_W5:text_load_quartile:Q1 | E67_W5 |  | nan | nan | PASS |
| subgroup:E67_W5:text_load_quartile:Q2 | E67_W5 |  | nan | nan | PASS |
| subgroup:E67_W5:text_load_quartile:Q3 | E67_W5 |  | nan | nan | PASS |
| subgroup:E67_W5:text_load_quartile:Q4 | E67_W5 |  | nan | nan | PASS |
| subgroup:E67_W5:entry_friction:0 | E67_W5 |  | nan | nan | PASS |
| subgroup:E67_W5:entry_friction:1 | E67_W5 |  | nan | nan | PASS |
| subgroup:E67_W6:availability:missing | E67_W6 |  | nan | nan | PASS |
| subgroup:E67_W6:availability:present | E67_W6 |  | nan | nan | PASS |
| subgroup:E67_W6:text_load_quartile:Q1 | E67_W6 |  | nan | nan | PASS |
| subgroup:E67_W6:text_load_quartile:Q2 | E67_W6 |  | nan | nan | PASS |
| subgroup:E67_W6:text_load_quartile:Q3 | E67_W6 |  | nan | nan | PASS |
| subgroup:E67_W6:text_load_quartile:Q4 | E67_W6 |  | nan | nan | PASS |
| subgroup:E67_W6:entry_friction:0 | E67_W6 |  | nan | nan | PASS |
| subgroup:E67_W6:entry_friction:1 | E67_W6 |  | nan | nan | PASS |
| subgroup:E75_W6:availability:missing | E75_W6 |  | nan | nan | PASS |
| subgroup:E75_W6:availability:present | E75_W6 |  | nan | nan | PASS |
| subgroup:E75_W6:text_load_quartile:Q1 | E75_W6 |  | nan | nan | PASS |
| subgroup:E75_W6:text_load_quartile:Q2 | E75_W6 |  | nan | nan | PASS |
| subgroup:E75_W6:text_load_quartile:Q3 | E75_W6 |  | nan | nan | PASS |
| subgroup:E75_W6:text_load_quartile:Q4 | E75_W6 |  | nan | nan | PASS |
| subgroup:E75_W6:entry_friction:0 | E75_W6 |  | nan | nan | PASS |
| subgroup:E75_W6:entry_friction:1 | E75_W6 |  | nan | nan | PASS |
| subgroup:POOLED_E67_W4_PLUS_E75_W6:availability:missing | POOLED_E67_W4_PLUS_E75_W6 |  | nan | nan | PASS |
| subgroup:POOLED_E67_W4_PLUS_E75_W6:availability:present | POOLED_E67_W4_PLUS_E75_W6 |  | nan | nan | PASS |
| subgroup:POOLED_E67_W4_PLUS_E75_W6:text_load_quartile:E67_Q1 | POOLED_E67_W4_PLUS_E75_W6 |  | nan | nan | PASS |
| subgroup:POOLED_E67_W4_PLUS_E75_W6:text_load_quartile:E67_Q2 | POOLED_E67_W4_PLUS_E75_W6 |  | nan | nan | PASS |
| subgroup:POOLED_E67_W4_PLUS_E75_W6:text_load_quartile:E67_Q3 | POOLED_E67_W4_PLUS_E75_W6 |  | nan | nan | PASS |
| subgroup:POOLED_E67_W4_PLUS_E75_W6:text_load_quartile:E67_Q4 | POOLED_E67_W4_PLUS_E75_W6 |  | nan | nan | PASS |
| subgroup:POOLED_E67_W4_PLUS_E75_W6:text_load_quartile:E75_Q1 | POOLED_E67_W4_PLUS_E75_W6 |  | nan | nan | PASS |
| subgroup:POOLED_E67_W4_PLUS_E75_W6:text_load_quartile:E75_Q2 | POOLED_E67_W4_PLUS_E75_W6 |  | nan | nan | PASS |
| subgroup:POOLED_E67_W4_PLUS_E75_W6:text_load_quartile:E75_Q3 | POOLED_E67_W4_PLUS_E75_W6 |  | nan | nan | PASS |
| subgroup:POOLED_E67_W4_PLUS_E75_W6:text_load_quartile:E75_Q4 | POOLED_E67_W4_PLUS_E75_W6 |  | nan | nan | PASS |
| subgroup:POOLED_E67_W4_PLUS_E75_W6:entry_friction:0 | POOLED_E67_W4_PLUS_E75_W6 |  | nan | nan | PASS |
| subgroup:POOLED_E67_W4_PLUS_E75_W6:entry_friction:1 | POOLED_E67_W4_PLUS_E75_W6 |  | nan | nan | PASS |

## Subgroup guards

Support rule: n ≥ 200 and at least 30 examples per class. Unsupported cells are reported as UNKNOWN, without an estimated harm.

| View | Group | Value | n | Positive/negative | Δ Macro F1 vs A02 | Δ Recall vs A02 | Δ Macro F1 vs permuted | Δ Recall vs permuted | Status |
|---|---|---|---:|---:|---:|---:|---:|---:|---|
| E67_H12 | availability | missing | 1375 | 487/888 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | PASS |
| E67_H12 | availability | present | 19528 | 6240/13288 | 0.000713 | 0.000481 | 0.000745 | 0.000801 | PASS |
| E67_H12 | text_load_quartile | Q1 | 5484 | 2041/3443 | 0.000121 | -0.000980 | 0.000706 | 0.000000 | PASS |
| E67_H12 | text_load_quartile | Q2 | 4841 | 1507/3334 | 0.000800 | 0.001991 | 0.001745 | 0.003318 | PASS |
| E67_H12 | text_load_quartile | Q3 | 4715 | 1141/3574 | 0.000711 | 0.000000 | 0.000111 | 0.000876 | PASS |
| E67_H12 | text_load_quartile | Q4 | 5863 | 2038/3825 | 0.001116 | 0.000981 | 0.000267 | -0.000491 | PASS |
| E67_H12 | entry_friction | 0 | 14344 | 4123/10221 | 0.000733 | 0.000728 | 0.000956 | 0.001455 | PASS |
| E67_H12 | entry_friction | 1 | 6559 | 2604/3955 | 0.000580 | 0.000000 | 0.000263 | -0.000384 | PASS |
| E67_W4 | availability | missing | 342 | 140/202 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | PASS |
| E67_W4 | availability | present | 4608 | 1713/2895 | 0.001074 | 0.000000 | 0.001110 | 0.000584 | PASS |
| E67_W4 | text_load_quartile | Q1 | 1092 | 513/579 | 0.000914 | 0.000000 | 0.000914 | 0.000000 | PASS |
| E67_W4 | text_load_quartile | Q2 | 963 | 338/625 | 0.002260 | 0.002959 | 0.002260 | 0.002959 | PASS |
| E67_W4 | text_load_quartile | Q3 | 1130 | 322/808 | -0.000274 | -0.003106 | 0.000969 | 0.000000 | PASS |
| E67_W4 | text_load_quartile | Q4 | 1765 | 680/1085 | 0.001106 | 0.000000 | 0.000553 | 0.000000 | PASS |
| E67_W4 | entry_friction | 0 | 3418 | 1207/2211 | 0.001238 | 0.000829 | 0.001588 | 0.001657 | PASS |
| E67_W4 | entry_friction | 1 | 1532 | 646/886 | 0.000560 | -0.001548 | -0.000069 | -0.001548 | PASS |
| E67_W5 | availability | missing | 507 | 176/331 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | PASS |
| E67_W5 | availability | present | 6955 | 2198/4757 | 0.000963 | 0.000910 | 0.000580 | 0.001365 | PASS |
| E67_W5 | text_load_quartile | Q1 | 2116 | 740/1376 | 0.000903 | 0.000000 | 0.000452 | 0.000000 | PASS |
| E67_W5 | text_load_quartile | Q2 | 1696 | 585/1111 | 0.000000 | 0.000000 | 0.000882 | 0.003419 | PASS |
| E67_W5 | text_load_quartile | Q3 | 1632 | 384/1248 | 0.001032 | 0.002604 | 0.001343 | 0.005208 | PASS |
| E67_W5 | text_load_quartile | Q4 | 2018 | 665/1353 | 0.001677 | 0.001504 | -0.000225 | -0.001504 | PASS |
| E67_W5 | entry_friction | 0 | 5188 | 1494/3694 | 0.000678 | 0.000669 | 0.000825 | 0.002008 | PASS |
| E67_W5 | entry_friction | 1 | 2274 | 880/1394 | 0.001340 | 0.001136 | 0.000000 | 0.000000 | PASS |
| E67_W6 | availability | missing | 526 | 171/355 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | PASS |
| E67_W6 | availability | present | 7965 | 2329/5636 | 0.000303 | 0.000429 | 0.000680 | 0.000429 | PASS |
| E67_W6 | text_load_quartile | Q1 | 2276 | 788/1488 | -0.001028 | -0.002538 | 0.000866 | 0.000000 | PASS |
| E67_W6 | text_load_quartile | Q2 | 2182 | 584/1598 | 0.000833 | 0.003425 | 0.002202 | 0.003425 | PASS |
| E67_W6 | text_load_quartile | Q3 | 1953 | 435/1518 | 0.001098 | 0.000000 | -0.001466 | -0.002299 | PASS |
| E67_W6 | text_load_quartile | Q4 | 2080 | 693/1387 | 0.000690 | 0.001443 | 0.000455 | 0.000000 | PASS |
| E67_W6 | entry_friction | 0 | 5738 | 1422/4316 | 0.000466 | 0.000703 | 0.000651 | 0.000703 | PASS |
| E67_W6 | entry_friction | 1 | 2753 | 1078/1675 | 0.000000 | 0.000000 | 0.000690 | 0.000000 | PASS |
| E75_W6 | availability | missing | 526 | 171/355 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | PASS |
| E75_W6 | availability | present | 7965 | 2329/5636 | -0.000574 | -0.000429 | -0.000262 | 0.000000 | PASS |
| E75_W6 | text_load_quartile | Q1 | 2244 | 781/1463 | -0.000536 | -0.001280 | -0.000536 | -0.001280 | PASS |
| E75_W6 | text_load_quartile | Q2 | 2285 | 611/1674 | -0.001592 | -0.001637 | -0.000778 | 0.001637 | PASS |
| E75_W6 | text_load_quartile | Q3 | 1969 | 443/1526 | -0.000367 | -0.002257 | -0.000367 | -0.002257 | PASS |
| E75_W6 | text_load_quartile | Q4 | 1993 | 665/1328 | 0.000462 | 0.003008 | 0.000715 | 0.001504 | PASS |
| E75_W6 | entry_friction | 0 | 5738 | 1422/4316 | 0.000191 | 0.001406 | -0.000394 | 0.000000 | PASS |
| E75_W6 | entry_friction | 1 | 2753 | 1078/1675 | -0.001600 | -0.002783 | 0.000000 | 0.000000 | PASS |
| POOLED_E67_W4_PLUS_E75_W6 | availability | missing | 868 | 311/557 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | PASS |
| POOLED_E67_W4_PLUS_E75_W6 | availability | present | 12573 | 4042/8531 | 0.000058 | -0.000247 | 0.000266 | 0.000247 | PASS |
| POOLED_E67_W4_PLUS_E75_W6 | text_load_quartile | E67_Q1 | 1092 | 513/579 | 0.000914 | 0.000000 | 0.000914 | 0.000000 | PASS |
| POOLED_E67_W4_PLUS_E75_W6 | text_load_quartile | E67_Q2 | 963 | 338/625 | 0.002260 | 0.002959 | 0.002260 | 0.002959 | PASS |
| POOLED_E67_W4_PLUS_E75_W6 | text_load_quartile | E67_Q3 | 1130 | 322/808 | -0.000274 | -0.003106 | 0.000969 | 0.000000 | PASS |
| POOLED_E67_W4_PLUS_E75_W6 | text_load_quartile | E67_Q4 | 1765 | 680/1085 | 0.001106 | 0.000000 | 0.000553 | 0.000000 | PASS |
| POOLED_E67_W4_PLUS_E75_W6 | text_load_quartile | E75_Q1 | 2244 | 781/1463 | -0.000536 | -0.001280 | -0.000536 | -0.001280 | PASS |
| POOLED_E67_W4_PLUS_E75_W6 | text_load_quartile | E75_Q2 | 2285 | 611/1674 | -0.001592 | -0.001637 | -0.000778 | 0.001637 | PASS |
| POOLED_E67_W4_PLUS_E75_W6 | text_load_quartile | E75_Q3 | 1969 | 443/1526 | -0.000367 | -0.002257 | -0.000367 | -0.002257 | PASS |
| POOLED_E67_W4_PLUS_E75_W6 | text_load_quartile | E75_Q4 | 1993 | 665/1328 | 0.000462 | 0.003008 | 0.000715 | 0.001504 | PASS |
| POOLED_E67_W4_PLUS_E75_W6 | entry_friction | 0 | 9156 | 2629/6527 | 0.000588 | 0.001141 | 0.000431 | 0.000761 | PASS |
| POOLED_E67_W4_PLUS_E75_W6 | entry_friction | 1 | 4285 | 1724/2561 | -0.000818 | -0.002320 | -0.000036 | -0.000580 | PASS |

## Per-cycle effects

Positive means aligned exceeds the comparator on Macro F1. All cycle denominators are included.

| Origin | Cycle | n | Δ vs A02 | Δ vs permuted |
|---:|---:|---:|---:|---:|
| 67 | 68 | 577 | +0.004332 | +0.004332 |
| 67 | 69 | 1079 | +0.000913 | +0.000000 |
| 67 | 70 | 1605 | +0.002016 | +0.002718 |
| 67 | 71 | 1689 | -0.000885 | -0.000885 |
| 67 | 72 | 1811 | +0.000808 | +0.001484 |
| 67 | 73 | 1880 | +0.001046 | +0.000711 |
| 67 | 74 | 1814 | +0.002261 | +0.002432 |
| 67 | 75 | 1957 | -0.000501 | -0.002478 |
| 67 | 76 | 1822 | -0.000897 | +0.000342 |
| 67 | 77 | 2335 | +0.001420 | +0.001420 |
| 67 | 78 | 2146 | -0.000737 | +0.000254 |
| 67 | 79 | 2188 | +0.001031 | +0.000444 |
| 75 | 76 | 1822 | -0.000539 | -0.001079 |
| 75 | 77 | 2335 | +0.000632 | +0.000632 |
| 75 | 78 | 2146 | -0.002604 | +0.000000 |
| 75 | 79 | 2188 | +0.000000 | -0.000613 |

## Lineage, accounting, and limitations

- Official Classification Train SHA-256: `875552a68a3d24b0f3e1966d6ddb5671e6f54305a1c526e2f0a76747942ed91c`; query labels were opened only after prelabel PASS (receipt `d67f01cf7e5ff63500eab97f3d7ea923f5652d6bf04acf09c9ec454e4a43a56e`).
- Four reserved/saved indexed memory predictors: `CGEOM_V1_E67_ALIGNED, CGEOM_V1_E67_PERMUTED, CGEOM_V1_E75_ALIGNED, CGEOM_V1_E75_PERMUTED`. No classifier, vectorizer, or preprocessing refits; retries: 0.
- Budget ledger SHA-256: `bcedd3b589c3c496176fa24e0f5d4da316907541389cecbd8361ce133a5179e7`; snapshot: `{'active_reservations': 118, 'active_reservations_by_category': {'continuation': 38, 'control_replay': 2, 'pilot': 77, 'technical_retry': 1}, 'effective_category_caps': {'continuation': 38, 'control_replay': 2, 'pilot': 80, 'technical_retry': 1}, 'event_rows': 302, 'fit_failed': 4, 'fit_pending': 0, 'fit_saved': 114, 'fit_starts': 118, 'global_cap': 121}`.
- Independent score verifier SHA-256: `eef28d7010a6bb01d5944985e63a0185075d94a9536847123daa8579eb1d4394`; completion result SHA-256: `db8aa20464aa2f273998e553aab967074372eff630734c7106311b807c203552`.
- All internal rows are reused development labels; E75 W6 overlaps E67 late rows and is history sensitivity rather than independent evidence.
- Permutation preserves the coefficient-derived multiplier multiset, not every association with token frequency; it is not a randomization p-value.
- Passing the gates means only `ADVANCE_TO_SOL_REVIEW`; no continuation, champion change, Test inference, full-Train refit, Kaggle use, or submission follows automatically.
- Any subgroup below support is UNKNOWN. No subgroup estimate is fabricated.

See `RESULTS.json`, `PRELABEL_VERIFICATION.json`, `SCORED_VERIFICATION.json`, and `COMPLETION_RECEIPT.json` for machine-readable details.
