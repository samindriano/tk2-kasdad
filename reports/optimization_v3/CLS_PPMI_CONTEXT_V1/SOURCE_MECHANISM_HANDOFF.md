# PPMI source-mechanism handoff

**Decision: no-go on another fit for this PPMI32 + HGB blend, longer-prefix extension, vocabulary expansion, or HGB-capacity increase.** R2 is a zero-fit readout of the fixed candidate, not a refit recommendation. Evidence below uses committed R2 metrics/subgroups and source/history summaries only; no target, raw CSV, or prediction-array access was made for this handoff.

## What the E67/E75 contrast says

On E67_W6 (8,491 rows), the real candidate is **+0.001997 MacroF1 vs champion**, while shuffled is **−0.006669**. Their **+0.008666 real-minus-shuffled contrast is arithmetically about 77% shuffled-control degradation and 23% real-candidate gain vs champion**. This is a decomposition of the reported scores, not a causal attribution: real and shuffled PPMI produce different factorization axes/geometry and HGB is axis-sensitive.

On the same E75_W6 query rows, real is **−0.002638 vs champion**, shuffled is **−0.002502**, and real-minus-shuffled is **−0.000136**. Both arms lose to champion and are nearly tied. The primary unique view also loses **−0.001268** for real vs champion, despite real beating shuffled by **+0.000888**. E67_H12 is a small counterexample at **+0.000572**, and E67_W4/W6 have positive candidate gains; E67_W5 and E75_W6 lose. The evidence is time-dependent and does not support a stable PPMI advantage.

## Supported-group failures and counterevidence

| View / group | Support (positive / negative) | MacroF1 harm | Positive-recall harm |
|---|---:|---:|---:|
| E75_W6 / `inherited_text_load_quartile=3` | 1,993 (665 / 1,328) | 0.010935 | 0.033083 |
| E67_W6 / `inherited_text_load_quartile=3` | 2,080 (693 / 1,387) | 0.008586 | 0.027417 |
| E67_W5 / `inherited_text_load_quartile=3` | 2,018 (665 / 1,353) | 0.005219 | 0.028571 |
| E75_W6 / `entry_friction=1` | 2,753 (1,078 / 1,675) | 0.005608 | 0.023191 |

The highest harm is concentrated in the same inherited text quartile across views, but it is not uniform: the supported `availability=missing` group in both E67_W6 and E75_W6 (526 rows; 171 positive, 355 negative) has zero measured MacroF1 and recall harm. Do not turn these post-score groups into routing rules.

## Novelty, fit value, and uncertainty

The bounded earlier search found no word-local PPMI implementation; prior SVD was document-TFIDF, so the completed PPMI representation was genuinely distinct. But the long-prefix question is already represented by E75: E67 fit ends at cycle 67, E75 at 75. Both used the same frozen unigram vocabulary, 32 SVD components, and fixed HGB settings; active PPMI terms move only from 5,675 to 5,723. That is not evidence for a vocabulary-expansion mechanism. A capacity-only sweep also lacks a distinct hypothesis and repeats territory with prior SVD-HGB/C19 counterevidence and fixed-recipe decisions against capacity/weight grids.

The two origins are not a causal training-history ablation. Their prefixes, PPMI graphs, SVD axes, structured widths (284 vs 370), and fitted HGB states differ even though the W6 query population is the same. Thus the current result cannot separate real context value from shuffled-geometry loss or prefix-specific state.

**Next decision: no-go; fit cost is zero.** This retires the exact candidate and declines a longer-prefix/vocabulary/HGB-capacity variant. It resolves whether current evidence justifies another same-family fit (it does not), but leaves representation-order versus learned-geometry/history causality unresolved. Reopen only with fresh independent evaluation and a predeclared design that fixes the prefix, vocabulary, structured block, factorization basis, HGB settings, and matched control. No E79, refit, upload, threshold search, or post-hoc routing.

## Evidence references

- Corrected scores, all six views, and subgroup denominators: [CORRECTED_RESULTS_R2.json](../../../experiments/CLS_PPMI_CONTEXT_V1/CORRECTION_R2/CORRECTED_RESULTS_R2.json); [CORRECTED_REPORT_R2.md](../../../experiments/CLS_PPMI_CONTEXT_V1/CORRECTION_R2/CORRECTED_REPORT_R2.md).
- Frozen prefix/representation recipe and active-term counts: [RECIPE_AUDIT.md](CORRECTION_R1/RECIPE_AUDIT.md:27), [PROTOCOL.md](PROTOCOL.md:39).
- Prior novelty and control caveats: [ROOT_CAMPAIGN.md](C:/Users/Sam/OneDrive/Documents/KASDAD/tk2-simplification-audit-v1/reports/simplification_v1/CLS_DUAL_REPRESENTATION_V1/ROOT_CAMPAIGN.md:9), [control caveat](C:/Users/Sam/OneDrive/Documents/KASDAD/tk2-simplification-audit-v1/reports/simplification_v1/CLS_DUAL_REPRESENTATION_V1/ROOT_CAMPAIGN.md:47).
- Prior SVD32/C19 limits and no capacity-grid rationale: [Classification Next decision](../classification_next_v1/DECISION.md:13), [C19 continuation rationale](../classification_next_v1/DECISION.md:25).
