# Classification decision audit for MIX50

**Status: complete, read-only.** No new estimator fit, threshold scoring, Test
prediction, CSV, or Kaggle upload was performed for this audit.

## Finding

The fixed `>= 0.50` decision rule on the current MIX50 is a frozen recipe choice,
not a cutoff shown to be optimal for MIX50 by a past-only temporal selection
rule. Earlier TP2 work did test honest temporal cutoff selection for C2_B and
C4 recipes, and it tested past-only probability heads at a fixed cutoff. Those
results are relevant counterevidence, but they do not answer whether a selected
cutoff transfers for the current `0.5*pP + 0.5*pA` MIX50 present branch.

The MIX50 Kaggle receipt records public Macro F1 **0.83144** (submission
57956714). Its frozen policy is present: `0.5*pP + 0.5*pA`; missing:
`pP` exactly; prediction rule: `p >= 0.50`. The Train-side MIX50 audit also
computed errors at `>= 0.50` only. Public score is one aggregate result for
that one submitted cutoff. It cannot compare alternatives or reveal which
public labels were false positives or false negatives.

## Evidence and scope

| Evidence | What was tested | Result and relevance to MIX50 |
|---|---|---|
| `experiments/classification_round1/protocol.json`; `reports/classification_round1/comparison_tables.md`, `FINDINGS.md` | Past-only inner temporal threshold selection on C2_B and earlier recipes; 0.10–0.90 by 0.01, fitted-history validation before each outer window. | This is an honest precedent for the *method*. On primary C2_B, selected cutoff gave Macro F1 0.8188 versus 0.8206 at 0.50 (−0.0019); selected cutoffs had median 0.47 and range 0.34–0.53. It is not a MIX50 result. |
| `experiments/classification_round2/protocol.json`; `reports/classification_round2/comparison_tables.md`, `FINDINGS.md` | Fixed 0.50, inner-pooled, half-shrunk, and equal-cycle thresholds for C2_B/C4 recipes across six fit histories. | C2_B inner-pooled thresholds ranged 0.34–0.53; C4_HGB ranged 0.34–0.53. The fixed C4_HGB route was the primary fixed-policy leader (0.825481); policy winners differed by recipe and horizon. This argues against assuming one cutoff transfers, but it does not test MIX50. |
| `experiments/optimization_v2/classification_calibration/protocol.json`; `reports/optimization_v2/classification_calibration/results.json`, `verification.json` | Past-only logistic probability heads on the earlier C2_B-present + CatBoost-missing route; all used fixed 0.50. No outer-label threshold selection. | Primary Macro F1 was 0.827646 for the old champion, 0.822238 for calibrated-base, 0.821600 for calibrated-NB, and 0.823571 for the two-score stack. This is counterevidence to a generic calibration-head rescue, not a cutoff test for MIX50. |
| `reports/optimization_v3/a02_decision_boundary_audit_v1/SUMMARY.md` | Reused A02 probability corrections at the frozen `>= 0.50` rule; no estimator fits and no alternate cutoff. | Full-six primary had 404 observed label flips (231 helpful, 173 harmful; net 58 fewer errors), while late had only 3 net errors reduced. Verdict was `NO_SUPPORTED_F1_LANE`. It demonstrates allocation tradeoffs at 0.50; it is not a current-MIX50 threshold-selection result. |
| `reports/optimization_v3/posoffset_a02_endgame_v1/RECIPE.json`, `E79_SELECTED_MANIFEST.json`, `KAGGLE_SCORE_20261104.json`; `reports/optimization_v3/posmix_followup_v1/SOL_NEXT_RESEARCH.md`, `ERROR_AUDIT.json` | Current MIX50 recipe and saved Train predictions, evaluated at 0.50. E67 H12 scores cycles 68–79; E75 W6 scores cycles 76–79. | The E67/E75 W6 views are the same 8,491 Train rows under different model origins; these are overlapping/adaptive development evidence. No alternate MIX50 threshold was scored. E67 H12 has 20,903 rows; its W4/W5/W6 partitions have 4,950/7,462/8,491 rows. |

The inspected backlog identifiers are the actual cards: C07 is category
frequency, C08 is ordered category outcome rates, and C15 is a two-score
prequential meta-head. They are planned representation/meta-head lanes; none
constitutes a past-only threshold test for MIX50.

## Why Macro F1 makes the decision rule matter

Changing a scalar cutoff leaves each row's probability and ranking unchanged,
so ROC-AUC, average precision, log loss, and Brier score do not change. It can
change predicted-positive allocation, false positives, false negatives,
class-specific precision/recall, and therefore Macro F1. Macro F1 averages the
two class F1 scores; it does not use the 0.50 cutoff as an optimality rule.
Changing prevalence or probability scale across cycles can change that
precision/recall balance. A probability calibration transform is a different
operation: it changes probabilities and can change proper scoring metrics as
well as decisions at a fixed 0.50.

The saved A02 boundary audit also shows why predicted-positive rate alone is
not evidence of improvement: E71 increased that rate while net errors worsened.
Likewise, the MIX50 audit's lower AUC but better log loss than A02 does not say
which cutoff would maximize future Macro F1.

## Best next mechanism and minimum cost

The best targeted follow-up is **prequential cutoff transfer on exact MIX50
Train OOF probabilities**, with every cutoff chosen on strictly earlier cycles
and evaluated once on later cycles. This is a recipe-specific audit of a known
decision method, not a claim that thresholding is a new technique.

The saved prediction bundle is
`experiments/runs/optimization_v3_posoffset_a02_endgame_v1/policy_predictions.npz`.
It contains Unicode `record_id`s, cycles, exact MIX50 probabilities, missing
masks, and the E67/E75 OOF rows; its published hash matches the zero-fit
MIX50 audit. A future selector can retrieve labels only for its explicitly
chosen Train IDs and keep the evaluation IDs out of selection:

1. Select one global threshold from E67 cycles 68–71, then evaluate E67 cycles
   72–75. Both use the same base origin (fit through cycle 67).
2. Select one threshold from E67 cycles 68–75, then evaluate E75 cycles 76–79.
   This is forward in time, but the base origin changes from E67 to E75 and
   must be reported as a transfer across refit histories.
3. If the diagnostics justify a deployed cutoff, select one final threshold
   from the unique OOF Train rows E67 cycles 68–75 plus E75 cycles 76–79, then
   apply it to saved E79 probabilities. Do not use the older E67 W6 copy in
   this final selection because it duplicates the same IDs under the earlier
   model origin.

Keep the missing branch exactly at the current MIX50 decision. For selection,
maximize overall binary Macro F1 on all selected rows; use unique tied-score
cutpoints plus 0.50 and the two all-positive/all-negative endpoints; ties
within 1e-12 go to the cutoff closest to 0.50, then the lower cutoff. Require
at least 1,000 selection rows and at least 100 examples of each class; otherwise
use 0.50 and record the fallback. Compare the selected policy with 0.50 on the
same later rows. Report pooled unique-row W5+W6 metrics from pooled confusion
counts rather than averaging window F1. Report class errors, cycle signs,
availability, and supported legacy groups; leave under-supported groups
`UNKNOWN`.

**Minimum estimator-fit cost: zero.** This uses already saved Train OOF
probabilities and requires three deterministic threshold selections at most,
not new base-model or preprocessing fits. It is still adaptive development
analysis: these Train outcomes have been seen in earlier audits, so it is not
an untouched holdout or evidence of causal effect. The final Test labels remain
unavailable and must not be inferred from the public leaderboard.

## Decision

The current 0.50 rule was evaluated as part of MIX50, but it was **not shown to
be a temporally selected cutoff for MIX50**. Older C2_B/C4 experiments prove
that past-only cutoff selection has been tried for different recipes; their
mixed/negative transfers are the strongest reason to require a same-recipe
forward check before considering a change. Proceed only with the frozen
prequential protocol above; do not tune from public score.
