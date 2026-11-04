# Sol review: retire exact pilot; diagnose history-dependent transport before fitting

**ACCEPT_RESULTS; RETIRE_REG_POSITION_L1_V1; NO_E79_OR_UPLOAD.**
Reviewed clean executor HEAD14f06be9ef414f1004c0cb5c4c02df0fbffb4030.
This rejects the exact pilot, not all position information or Regression research.

## Independent checks and decision

Sol reran `scripts.verify_reg_position_l1_v1.final(source_root, output_root)`:
PASS_INDEPENDENT_METRICS_AND_GATES; frozen decision remains
RETIRE_PILOT_NO_CANDIDATE_PASSED. Root reran the actual focused test file:
9 passed, one sklearn synthetic sparse feature-name warning. An initial test
invocation named an additional nonexistent runner test file and collected no
tests; the corrected invocation is `python -m pytest tests/test_reg_position_l1_v1.py -q`.

Separately, without the production composer, Sol loaded all6 numeric prediction
bundles, joined official Regression Train IDs/cycles/availability, reconstructed
delta=clip(.25*head,-5,5) and clip(R06+delta,0,100), checked exact missing fallback,
and recomputed MAE. This matches all main-view results:

| View | R06 MAE | BAG MAE | REAL MAE | SHUFFLED MAE |
|---|---:|---:|---:|---:|
| E67 H12 |11.529709|11.566679|11.564997|11.573594|
| E75 W6 |9.921613|9.785707|9.770188|9.787883|

REAL E67 gain -.035288 and W4 harm .110751 fail the competition criteria,
along with4 supported subgroup harms. At E75 REAL improves .151424 and beats
BAG/shuffle by .015518/.017695, but E67 mechanism contrasts (.001682/.008597)
fail their .01 minima. These observations do not establish uniformly absent
position information. Much of E75's improvement is also present in BAG and
SHUFFLED, so attribution to actual position is limited. No old gates change.

REAL's mean absolute correction on present rows is1.199 at E67 versus .801
at E75. Mean signed correction is -.095/-.071. REAL changes predictions relative
to BAG by mean absolute .324/.263 over full queries. Substantial perturbation
therefore need not give substantial MAE gain. E67 REAL helps9,592 rows and
harms9,931; E75 helps4,557 and harms3,408. Row counts are not equivalent to MAE.

All6 fit starts are saved in the isolated cap6 ledger; none failed/pending/retried.
Shared Classification ledger was not used. No Test/Sample/Kaggle, new fit during
review, final refit, or upload. Public reference remains R06 11.16620 from the
previous confirmed record; no new public evidence exists for this pilot.

## Next authorized milestone: REG_POSITION_TRANSPORT_DIAGNOSTIC_V1, zero fits

Sam wants continuing research and thorough analysis of failed experiments.
Use a separate new diagnostic namespace, preserve this pilot source/freeze,
models/receipts, and ledger bytes. Luna 2 only, in this isolated worktree. No
new policy, weight/alpha/cap/threshold choice, training, index, Test/Sample/Kaggle,
dependency install, shared-main edit, push, merge, scheduled task or chat polling.
Root stops writing after this scoped commit/handoff.

Question: what changes between E67 and E75 explain the common arm improvement,
and is there enough evidence to design one genuinely different next experiment?
This is descriptive, adaptive analysis, not proof of a cause or a new holdout.

1. Reconcile exact original inputs and source hashes. All model/receipt/query/
   OOF identities must remain valid. Reuse only already scored outcomes; no new
   candidate probability/point predictions or alternate inference policies.
2. Pair the SAME W6 rows76–79 across origins by official record_id. Verify exact
   cohort equality. Report unchanged R06 MAE and head gains per arm/cycle; compare
   raw head/correction distributions, sign agreement and paired change magnitude
   between origins. Distinguish changing baseline, changing head, and changing
   temporal horizon/history. These change jointly, not independently/causally.
3. From the existing donor banks, show training residual distributions by source
   cutoff/cycle and same fixed R06-prediction deciles (floor(clip(p,0,100)/10),
   max9), including n, median,10/90th percentiles. Compare descriptively with
   query residual distributions in the same predeclared bins on E67 W4/W5/W6
   and E75 W6. Query targets are analysis only and cannot change a model/route.
   Minimum n100 for conclusions; sparse cells UNKNOWN. No learned bin edges,
   optimal recency span, fitted trend, or calibrated correction is selected.
4. Inspect existing fitted LightGBM split/gain importances aggregated by BAG,
   position and baseline-score feature blocks, plus saved training-loss summaries
   and query loss. Verify feature index boundaries from actual design/metadata.
   Correlated features and differing capacities prevent causal attribution;
   importance alone is not proof of useful information or generalization.
5. Compare raw residual-head prediction with original R06 error descriptively:
   signs, clipping frequency, helpful/harmful counts and MAE contribution using
   existing fixed delta. Identify whether E67's problem is mostly a common shift,
   conditional transfer failure, or remains unexplained. Do not score a centered,
   rescaled, recency-routed, tail-routed or dropped-block rescue policy.

One concise diagnostic report, computed tables and max2 informative PNGs,
reproducibility command/source/input hashes. No duplicate full pilot report.
Any code uses explicit source/output roots and temp-only tests; assert isolated
production ledger unchanged, never expect another lane's shared ledger to freeze.

Return ONE recommendation with evidence/counterevidence/UNKNOWNs:
- NO_FURTHER_POSITIONAL_REG_FIT_JUSTIFIED, or
- ONE_SPECIFIC_NEXT_PILOT_FOR_SOL_REVIEW with exact hypothesis/controls/population/
  objective/cost/stop gates and confirmed novelty. A direct all-prefix L1 learner
  with position is one possible design question, not approved training. Do not
  claim residual transport caused failure simply because a direct learner differs;
  fit population/target/learner may change too. Do not propose another broad
  rank/neighbor/recency/parameter list or automatically rescue a near miss.

Finish computed diagnostic, local scoped commit, then SEND compact result or
real blocker to Sol01a19569-16e0-7253-aa14-57a0a3160e03 hostlocal BEFORE final
reply and stop for the next Sol decision. No user approval needed for this zero-fit
scope; do not silently idle halfway through or request the same instruction again.
