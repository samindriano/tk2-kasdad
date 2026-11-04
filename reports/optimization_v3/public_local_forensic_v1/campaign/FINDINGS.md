# Retrospective findings: local gates and public outcomes

**Decision.** Prior `STOP` / `NONE_JUSTIFIED` wording was too broad if read as “no predictive value” or “scientifically exhausted.” The evidence shows only that no candidate met the continuation queue's unchanged rules at that checkpoint. Historical verdicts stay intact. A gate failure rejects its tested recipe under its baseline, horizon, metric, and risk rule; it does not establish zero signal for the whole family.

## 45-card review

The Nov 1 registry covers 20 Classification and 25 Regression cards: 43 saved terminal decisions plus two prerequisite skips, R10/R13 (unmeasured). Cached six-origin outputs are not new fits; R04's repaired attempt still has a valid final result. The [45-card table](CARD_DISPOSITION.md) preserves original statuses, tested mechanisms, gains, reasons, and saved evidence.

The table separates negative/near-null recipes, positive recipes rejected by risk or horizon guards, unmeasured prerequisite skips, post hoc routes, and public benchmarks. A failed guard does not make a recipe a champion; a skip or untested variant is not positive evidence.

## Three informative cases

| Case | Local baseline → candidate and guard | Public record | What it does and does not show |
|---|---|---|---|
| **REG direct position + R06** | E67 H12 gain **+0.021560 MAE**, E75 W6 **−0.064806**, matched E67 W6 **+0.041470**. R06 same-view baselines: 11.529709 / 9.921613; rounded-gain candidates: about 11.508149 / 9.986419. E71 W5 was not run. | Ref 57988146 `direct_position_real_e79.csv`, COMPLETE, **11.14400** vs R06 **11.16620** (gain **0.02220**). | Public win and mixed local signs coexist; E75 W6 still fails. Exact local-to-upload recipe identity is UNKNOWN in this lane. |
| **R12 present-only route** | Original primary/H12 gains: +0.245801/+0.284490. Missing-stream harms: +0.489276 (n=734), +0.324499 (n=2,578), W1 +0.768759 (n=392). Post hoc present-only route gains +0.285422/+0.306170, selected after outcomes. | Ref 57987865 `r12_current_route_e79.csv`, COMPLETE, **11.18673**, 0.02053 worse than R06. | No local manifest/hash link to the post hoc policy; exact recipe match is UNKNOWN. |
| **C19 XGB/SVD32** | Full-six Sparse25→C19: 0.829944→0.830611 (+0.000667); H12 0.802966→0.807117 (+0.004152), each n=38,588. W6 −0.002599; W2/Q3 (n=1,450) F1/recall harms 0.010461/0.017857. | Ref 57907572 COMPLETE, **0.82760**; −0.00148 vs Sparse25 and −0.00384 vs MIX50 .83144. | Same family was public-tested, but not with the full-six continuation recipe; public result is not a win. |

## Local recipes without a matching public record

The Nov 4 snapshots have no filename/description match for these IDs. This is **no matched record found**, not proof that the exact recipe was never submitted; public identity is UNKNOWN. They are screened recipes, not new variants.

| Recipe and profile | Exact saved local comparison | Failure / public identity | Readout under unchanged gates |
|---|---|---|---|
| **C10 trigram, STANDARD** | Primary gain +0.000342846 (n=9,064); H12 +0.000878127 (n=38,588). | Missed +0.0005 primary bar by 0.000157154; no supported subgroup failure. Snapshot no-match; identity UNKNOWN. [Result](../../batch1/C10/results.json) | Positive but below the frozen bar; no basis for another fit. |
| **R16 half-life-8 recency, STANDARD** | Primary gain +0.015728635 (n=9,064); H12 +0.017173171 (n=38,588). | Below +0.03 MAE materiality threshold; review notes early/late variation. Snapshot no-match; identity UNKNOWN. [Result](../../batch1/R16/results.json) | Positive but below margin; insufficient to resume under the same rule. |
| **R17 affine OOF L1 calibration, LATE** | W4+W6 gain +0.032732104 (n=13,441); H12_L +0.031992251 (n=20,903). | Low-tail harm exceeded +0.25: W4 +0.593373 (n=244), W6 +0.969182 (n=442), H12_L +0.595885 (n=1,048). Snapshot no-match; identity UNKNOWN. [Result](../../batch1/R17/results.json) | Strongest aggregate here, but risk-rejected. No fit without an explicitly changed risk objective. |

## Why local and public ranks differ

Protocols have different denominators: STANDARD W1+W4/H12 = 9,064/38,588; LATE W4+W6/H12_L = 13,441/20,903; CACHE_SIX has its own six-origin policy. Primary and H12 can reuse labels. E67 H12 (cycles 68–79) overlaps E75 W6 (76–79). E79 deployment trains through 79 and predicts 80–91; Train has no untouched 12-cycle future block after all 33 training cycles.

Baselines change too: early Regression used legacy/R1-D, continuation R22/R06, and public comparisons R06 or current MIX50. These gains are not one estimand. Reused development labels and selection among public submissions mean public/private ranking, paired uncertainty, and causal explanations remain unknown.

## Counterevidence and stop point

Counterevidence remains: C14/R07/R08/R20 were strongly negative; recent Classification positional public scores (CGEOM .82884, JointPos .82911, Moments .82979, Bigram .83064, Contrast .82935) were below MIX50 .83144. R22 qualified internally but scored 11.30535 vs R06 11.16620. R06's strong aggregate and best public score coexist with missing-W1 harm +0.395556 MAE (n=392). Pooled score and subgroup guard can rank models differently.

Saved official CLI snapshots contain 19 Classification records (18 COMPLETE, one ERROR) and 18 Regression records (all COMPLETE). The ERROR is `samuel_rnbr_final_e79.csv`; its Regression entry is COMPLETE. Its cause is unknown. Public subset composition, private scores, and server hashes are unavailable. Snapshots: [`Classification`](../KAGGLE_CLASSIFICATION_SNAPSHOT_20261104.json), [`Regression`](../KAGGLE_REGRESSION_SNAPSHOT_20261104.json). Direct-position recipe equivalence remains with the separate REG audit.

**Checkpoint:** the campaign was not scientifically exhausted; no fit qualified under the frozen queue. This retrospective authorizes no fit, model change, Test-label access, or submission. Review the independent REG lineage report before any amendment.

Sources: [`batch_ledger.json`](../../batch1/batch_ledger.json), [`BATCH_FINDINGS.md`](../../batch1/BATCH_FINDINGS.md), [`CARD_REVIEW.md`](../../postscreen_review/CARD_REVIEW.md), [`REVIEW.md`](../../postscreen_review/REVIEW.md), [`Research Next`](../../research_next_v1/CAMPAIGN_SUMMARY.md), [`Classification Next`](../../classification_next_v1/FINAL_SUMMARY.md), [`R22 continuation`](../../final_continuation/R22/R22_CONTINUATION.md), [`R06/public audit`](../../research_next_v1/R06_GATE_VS_PUBLIC.md), R06 same-view values in [`RNBR continuation results`](../../rnbr_continuation_v1/RESULTS.md), and the root [`public/local audit frame`](../ROOT_AUDIT_FRAME.md). This audit made no Kaggle connection.
