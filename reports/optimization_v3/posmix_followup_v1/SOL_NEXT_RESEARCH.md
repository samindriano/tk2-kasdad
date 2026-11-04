# Reopened Classification research: one contrast-only positional pilot

**GO: POS_CONTRAST_V1, four pilot gamma fits and at most one conditional E79
gamma fit.** Sam explicitly reopened research after the completed MIX50 endgame
and asked for deeper score improvement on 4 November 2026. This is new scope;
the old endgame is complete and its verdicts, gates and upload limit remain
historical. Incoming HEAD e707ba7e34c15821cfaae4900aed5ba74ff4e4e9.

## Findings that motivate the experiment

Root executed `python scripts/analyze_posmix_frontier_v1.py`. It checks the
immutable endgame prediction-bundle hash and exact MIX50 formula, reuses only
already evaluated Train outcomes, and leaves the live ledger byte-identical.
The output is `ERROR_AUDIT.json` and `cycle_complementarity.png` beside this file.
No new prediction policy was scored, no Test access or estimator fit occurred.

| View | Jointly wrong POS/A02, present | MIX50 errors, present | Share jointly wrong | Disagreements, present |
|---|---:|---:|---:|---:|
| E67 H12 | 2,777 | 3,130 | 88.72% | 806 |
| E75 W6 | 1,039 | 1,154 | 90.03% | 257 |

At threshold 0.50, a convex probability blend cannot fix a row both components
classify incorrectly on the same side of the threshold. This limits weight-only
search; it does not bound improvements from retraining or new information.
The earlier informal commentary approximated this as 85%; the exact percentages
are **88.72% and 90.03%**. Reused overlapping development outcomes are adaptive
descriptive evidence, not a new holdout or statistical assurance.

MIX50's AUC is lower than A02 on both views (.897805 versus .899755; .907720
versus .911107), while its log loss is better (.381852 versus .419190; .349281
versus .379397). Also, POS/MIX50 and A02 use different missing experts. On E67
MIX50 makes 19 MORE present-stream errors than A02, but 24 FEWER missing-stream
errors; on E75 it makes 7 MORE present and 19 MORE missing errors. Aggregate
comparisons therefore mix predictive/ranking, probability and routing effects.
These findings do not authorize threshold or fallback tuning.

The saved original positional head has non-negligible equal-across-thirds
coefficient freedom: 31.25%/29.58% of squared gamma norm at E67/E75. Its mean
absolute common-direction logit contribution on present rows is .188/.182,
versus .301/.298 for the remaining contrasts. These contributions reconstruct
the saved total to <1e-15. The original shuffled heads also use that direction.
This is evidence the capacity is used, **not evidence it is harmful**.

Luna's independent targeted read-only novelty audit found no tested positional
centering/Helmert projection in the old recipes/source. It confirmed raw joint
position and offset recipes failed their original gates, while FM V2 has strong
negative predictive evidence. Do not reopen FM or relabel old failures.

## Falsifiable hypothesis and exact recipe

Removing equal-across-thirds coefficient freedom may improve transfer of the
positional offset enough to improve the fixed MIX50 combination. It may also
remove useful signal and fail. This is a capacity restriction on existing
information, not a claim of new token semantics or isolated causal position.

Reuse the original frozen E67/E75 C2_B bag model, prefix transforms, REAL and
seed42 SHUFFLED matrices from POSITIONAL_TEXT_PILOT_V1. The stored matrices named
fit_position/score_position are AUGMENTED [base | position]; slice after the
verified same-origin base width first. The positional suffix has three equal
width normalized blocks Xf, Xm, Xl. Compute ONLY:

    Z1 = (Xf - Xl) / sqrt(2)
    Z2 = (Xf - 2*Xm + Xl) / sqrt(6)
    Z = concatenate(Z1, Z2)

No normalization after projection. No new vocabulary, IDF, scaler, feature
selection, intercept, bag fit, sample/class weight or hyperparameter search.
`src/models/positional_contrast_v1.py` implements this transform and coefficient
embedding; its 10 focused tests passed, including objective-compatible L2 norm,
logit equivalence, blank/equal-third behavior and actual synthetic optimizer
persistence/replay. Reuse it; do not duplicate or change the formula.

Fit theta from zero with the unchanged proven offset solver:

    SUM logistic loss(y, z0 + Z@theta) + ||theta||^2 / 2
    z0 = original BAG_LR intercept + Xbase @ original BAG_LR beta
    C=1; float64; L-BFGS-B maxiter2000, ftol1e-12, gtol1e-6, maxls50.

Require optimizer success, finite objective/gradient/coefficients, and final
loss <= zero-theta loss +1e-8. Persist termination details, not merely a PASS.
No retries, solver rescues, warm starts or changed regularization. A numerical
failure consumes its fit ID and is reported to Sol. Identical prefix population
as original bag fit (including blank rows); blank Z contributes no theta gradient.
In-sample original bag offset is sequential additive fitting, not cross-fitting.

The projection has orthonormal rows and enforces gamma_f+gamma_m+gamma_l=0 in
original coordinates while preserving theta's L2 penalty. However, source
sublinear TF is applied separately within thirds: Xf+Xm+Xl is not exactly an
order-invariant bag representation. Do not claim all bag overlap is removed.

## Four fits, fixed views and candidate

Order: E67_REAL, E67_SHUFFLED, E75_REAL, E75_SHUFFLED, new unique
POS_CONTRAST_V1 fit IDs. E67 fits cycles47–67 (29,705 rows) and scores68–79
(20,903); E75 fits47–75 (42,117) and scores76–79 (8,491). Both W6 views contain
the same outcomes. Verify actual row IDs/order/cycles, not only row counts.

For each arm obtain q=expit(z0_query+Z_query@theta). Its FIXED competition policy
on present rows is .5*q+.5*original_A02_probability; missing rows exactly the
original pP HGB route. Threshold >=.50. REAL is the sole competition candidate;
SHUFFLED is a matched-capacity mechanism control and is never a submission.
Do not choose between raw q, residual-transfer policies or new blend weights.
Report q/bag/oldPOS descriptively; qualify the fixed REAL blend versus MIX50.

Freeze source, recipe, all input hashes, gates and source commit before the first
fit. Query outcomes must stay out of fitting and construction; all four fits and
prelabel formula/lineage QA must be durable before new policy metrics are read.
Do not call these development labels fresh or outcome-blind campaign evidence.

All required for competition eligibility (tolerance1e-12):
- REAL blend gain >=.0005 versus original MIX50 on BOTH E67 H12 and E75 W6.
- REAL blend gain >=0 versus SHUFFLED blend on both required views.
- E67 W4/W5/W6 Macro F1 harm versus MIX50 <=.003 each.
- Same fit-defined availability/text-load/entry-friction groups as endgame on all
  five views: support n>=200 and >=30 each class; supported F1 harm<=.01 and
  positive-recall harm<=.02 versus MIX50. Unsupported explicitly UNKNOWN.
- Missing probabilities exactly original pP, valid boundaries/lineage, independent
  formula/metric/guard QA PASS. Report A02/POS comparisons without changing gate.

These are frozen engineering risk/benefit rules, not significance thresholds.
If any gate fails, no automatic full-Train fit or upload; return all results and
failure evidence to Sol. A near miss is not retroactively PASS. Root can evaluate
a separately documented benchmark later, but none is authorized by this card.

## Accounting, QA and execution ownership

Current ledger SHA8380d884fabb3cf19edfd606b2b23c7839b9267d16c6660e579444e8089f2d22,
129 starts/123 saved/6 historical failures/0 pending; prior cap132, category
caps pilot80/continuation49/control_replay2/technical_retry1. Authorize append-only
new amendment: pilot84, continuation50, other categories unchanged; cap137.
This adds exactly FOUR pilot units and ONE conditional E79 continuation unit.
Maximum actual additional starts five, no new retry allowance. Do not consume
old released slots, reset counters, or rewrite past events. Reserve new IDs;
release any unused conditional capacity explicitly. No learned preprocessors
are authorized: transform-only state reuse. A memory/index fit is also a fit.

Luna owns the new runner/verifier/tests and reports/experiments in
positional_contrast_v1 namespace only; root diagnostic namespace posmix_followup_v1
is review input. Keep old source/freezes/receipts/data unchanged. Use small proven
adapters, not a new general framework. One job, two threads, 6GiB RSS limit,
4GiB initial free RAM and 2GiB running reserve, deterministic artifacts.

Before production: test actual transform->offset optimizer->durable receipt->
prediction->metric/guard lifecycle using explicit TEMP-ONLY ledger paths, guarding
the production writer and asserting its unchanged hash. Include tampered IDs,
source/bundle mismatch, exact missing route, interrupted post-save recovery with
no refit, and independently reconstructed metrics/contrast/logits/penalty. Persist
started/failed/saved states and hashes. No large notebook or HTML work.

After all pilot results: independently verify, write concise findings + at most
2 informative PNGs, commit only owned files, SEND result/blocker back to Sol chat
01a19569-16e0-7253-aa14-57a0a3160e03 host local BEFORE final reply, then STOP for
review. Routine prefit implementation fixes authorized with a new source/freeze
before fits; no gate changes. If actual source/input corruption or missing state
cannot be resolved without refitting, report exact evidence to Sol immediately.

## Conditional score push after Sol review

One E79 REAL theta fit is reserved only if the pilot qualifies and Sol approves
the independent verification. Reuse verified E79 bag/structured/TF-IDF state,
build original REAL thirds transform-only, project identically, and fit theta
on all50,608 official Train rows. Reuse A02 E79 and missing pP unchanged.
No Test labels or Test-informed training. Same fixed policy, one CSV29421 exact
official IDs/schema/integer labels, manifest/source/model/Train/Test hashes,
formula and duplicate-label QA, then one possible Kaggle upload by Sol following
independent review. User renewed score-improvement authorization covers this
single eligible push. Check live history/acceptance; never blind retry. No final
leaderboard selection changes, normal push, notebook/EDA or teammate edits.
Public score does not trigger an automatic next variation.
