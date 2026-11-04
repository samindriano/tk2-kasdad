# CLS representation frontier memo

**Disposition:** `NONE` of the representation mechanisms in this bounded review justifies another fit now. This is a narrow `NOT_EXHAUSTIVE` finding, not evidence that classification improvement is impossible and not a stop decision for the active CLS decision lane.

## Scope and correction

I withdraw my earlier suggestion to test ordered adjacent bigrams by relative position. The `bigram_position_pilot_v1` recipe is already implemented and completed as a six-fit pilot with BAG, REAL, and SHUFFLED controls. A duplicate pilot would repeat that question.

The evidence checkout was `C:\Users\Sam\OneDrive\Documents\KASDAD\TP 2`, branch `main`, HEAD `dcab53691a1885a6f0c34c733e5ee75c6befee9a`. Its status showed three unrelated untracked files under `reports/optimization_v3/public_local_forensic_v1/`; this review did not change that checkout. The memo is being written in the isolated `tk2-reg-position-failure-mechanism-v1` worktree. No fit, scoring, prediction, threshold change, CSV, Test/Sample read, or Kaggle action was performed for this memo.

## Evidence and decision

| Mechanism / evidence | Observed result | Implication for a new fit |
|---|---|---|
| Position-bucketed ordered bigrams | REAL vs MIX50: E67 H12 `+0.000206` Macro F1, below the frozen `+0.000500` gate; E75 W6 `-0.001482` and below BAG and SHUFFLED. Public ref `57981852` scored `0.83064`, below MIX50 ref `57956714` at `0.83144`. | Withdraw as already tested; gate remains failed. |
| Bigram probability and error audit | REAL slightly improves log loss, Brier, and AUC on both W6 views. At threshold `0.50`, useful/harmful flips are E67 `54/60` and E75 `34/45`. E67 W6 and E75 W6 are the same 8,491 rows; these are adaptive, overlapping development results. | This is counterevidence to calling the feature wholly uninformative, but it does not clear the existing Macro F1 gate or justify a new fit. |
| C09 text × `entry_friction` | The C09 card still says `PLANNED_NOT_RUN`, but the actual result summary records `PILOT_COMPLETE_RETIRE`, two successful fits, and verifier status `PASS_WITH_CONTROL_SCOPE_RECONCILIATION`. Its paired `primary_pilot` result is `0.824083` vs `0.823972` (`+0.000110`); H12 is `0.802798` vs `0.802966` (`-0.000168`). W4 improved `+0.002471`, while W1 declined `-0.002746`. | The implementation was tested; the stale card is not evidence of an unrun lane. Preserve the RETIRE verdict. Because the verifier notes control-scope reconciliation and its pilot differs from the current MIX50 evaluation, do not treat these scores as a direct MIX50 comparison. |
| C10 trigram | Authorized public ref `57992538` completed at `0.83021`, `-0.00123` vs MIX50 `0.83144`. It exceeded the older Sparse25 public reference by `0.00113`, but did not replace MIX50. | Trigram expansion is already tested; public performance does not support another related fit in this review. |
| Learned-pair FM V2 | `MECHANISM_FAIL` and `COMPETITION_FAIL`; Macro F1 `0.763788` on E67 H12 and `0.785334` on E75 W6, with deltas vs bag `-0.041319` and `-0.025050`. | Strong negative evidence against repeating generic pairwise interactions. |
| Sparse MLP text hybrid | Frozen gate verdict `NO_GO_GATE_FAILED`; primary delta vs sparse25 `-0.001698`, pooled H12 `-0.000830`, and cutoff-67 H12 `-0.005971`. | Does not support another MLP-style text blend under the reviewed recipe. |
| POS/A02 complementarity audit | POS and A02 are jointly wrong on `88.72%` of MIX50 present-stream errors on E67 H12 and `90.03%` on E75 W6. | Constrains weight-only rescue. It does **not** prove a ceiling or rule out genuinely new information. |

The position audit's own next-lane conclusion is that no additional fit is justified by those diagnostics. Taken with the completed C09, C10, FM, and MLP results, the reviewed evidence does not identify a distinct, untested representation mechanism with a concrete reason to spend another fit now. This conclusion applies only to the artifacts listed below; other mechanisms remain possible and unreviewed.

## Active CLS lane remains separate

This memo does not cancel or block Sol's current `CLS_MIX50_TEMPORAL_DECISION_V1` authorization to Luna1: two past-only diagnostic threshold selectors, one OOF final, zero base-model fits, and one exploratory CSV only if conditional root review permits it. No threshold selector or final was run here, and this memo does not change its scope or gate.

## Evidence index

Hashes are SHA-256 of the reviewed files in the read-only `TP 2` checkout.

| File | SHA-256 |
|---|---|
| `reports/optimization_v3/bigram_position_pilot_v1/RECIPE.json` | `e5dbf18cf2cf79e35425c74db4cb0202501992be5f913a50aef467ce00741c6f` |
| `reports/optimization_v3/bigram_position_pilot_v1/RESULTS.md` | `c2b65893ec36de8f5dde92a20060fc2021bafcb6d1d039c6da27081241a88baf` |
| `reports/optimization_v3/cls_position_error_audit_v1/RESULTS.md` | `0225660950b0521658e583317852ca1322315bb2b6704d327011bdf3f257a26d` |
| `reports/optimization_v3/batch1/C09/summary.md` | `e62da56c82d0d13fdc0d4f6d5f16144d548a67fc5d4db447fd1b21dac7b5734c` |
| `reports/optimization_v3/batch1/C09/verification.json` | `3ac0273626b481ed533b0ffb9ec5bc9497c33f982eca2b30bc432a39b2be24e0` |
| `reports/optimization_v3/research_backlog/cards/C09.md` | `25e6fa39d7d63925dd6dba354392ba6806686b37598a0eee15a3ea101a81e701` |
| `reports/optimization_v3/c10_trigram_e79_exploratory_v1/SOL_KAGGLE_RESULT.md` | `90ac96bae225cae101ffd91a4f20467939dcb85d6b2ab2d5b8b08ff78dbc12a4` |
| `reports/optimization_v3/learned_pair_pilot_v2/POSTFIT_RESULTS.md` | `4f719f8047534fc11cc7dee8191219dffaead83b831deeb3277f4021eb37c564` |
| `reports/optimization_v3/classification_mlp_v1/FINAL_SUMMARY.md` | `1755e2464510f9e7b996b5e89b208ab2e40600261a3440dd7536179d0dd0a3aa` |
| `reports/optimization_v3/posmix_followup_v1/SOL_NEXT_RESEARCH.md` | `22e8328256ae442c722a605fdfbcba67c36297860f0a2b99ff328751d8acaaf4` |
| `reports/optimization_v3/posoffset_a02_endgame_v1/SOL_FINAL_DECISION.md` | `6d51136707453b7e9ab0fa989af93b4f45de710a14a3d4c9249f9cd4aeff3cb5` |

The C09 backlog card and broader research queue are stale relative to completed result artifacts; status claims here follow the result summary and verifier, not the queue. C10's public result is a historical completed score, not independent validation or private-performance evidence.
