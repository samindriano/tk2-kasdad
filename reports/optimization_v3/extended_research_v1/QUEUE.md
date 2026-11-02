# Extended research queue — exploratory allowance, no new fit authorization

Sam explicitly asked to keep researching new and even low-value hypotheses while
the current account allowance remains available. Sol selects this next executable
frontier. A finite completed research milestone is still reported honestly; do
not pretend an exhausted account is a scientific stopping rule or promise uptime.
Model-token allowance does not raise the consumed official fit cap66.

Preserve and finish the active B01 benchmark preparation first. A failure of its
upload or preparation does not block independent Train-only research below.
All historical failed gates remain failed. No Kaggle upload, final entry change,
new model/index/preprocessor fit, installation or notebook editing is authorized.

## B02 — label-free plug-in Macro F1 allocation (execute zero-fit)

Hypothesis: B01's improved probability quality can support a decision rule more
appropriate for class-averaged F1 than fixed .50. This differs from old inner
label-tuned threshold grids, score calibration, and output count matching.
Inspect the old threshold studies before claiming novelty; report equivalences.

Main input: B01 probabilities, reconstructed from existing frozen Train-only
geometry. Fixed sensitivity inputs: unchanged A02 and Sparse25, not a search for
the most favorable source. All priors, banks and predictions remain unchanged.
No new fitted state is constructed; this is a deterministic batch decision rule.

For one query batch, freeze missing-stream hard labels at the existing .50 route
and leave W1 fallback unchanged. Sort eligible present probabilities descending.
Only candidate K values at ends of equal-score blocks (plus zero and all) are
allowed; do not split identical scores by ID to manufacture decisions.
Let S be the sum of predicted positive probabilities over all batch rows and N
their count. Fixed missing predictions contribute their soft expected TP/FP.
For a candidate K, combine that fixed contribution with the top-K cumulative
probability sum, giving TP, FP, FN=S-TP and TN=N-S-FP. Optimize

`0.5 * (2*TP/(2*TP+FP+FN) + 2*TN/(2*TN+FP+FN))`.

Zero denominators contribute zero, consistent with explicit two-class reporting.
If maximizing candidates tie within1e-12, choose K nearest the existing .50
positive count, then smaller K. Selection code receives probabilities, masks,
and stable source identifiers only, never y or public score. No empirical
threshold grid, clipping adjustment, label prevalence estimate, or tuning.

This maximizes Macro F1 of expected confusion counts under the estimated
probabilities. It is NOT the exact expected value of Macro F1 (nonlinearity),
not a guarantee of Bayes optimality, and not the positive-class rule threshold
equals half F1. Calibration error and batch composition can make it worse.

Primary batch definition: each of the five trained origins' first-four-cycle
query windows separately; aggregate those fixed decisions with unchanged W1.
For inherited H12-L, origin67's full12-cycle query is a separate batch scenario.
Disclose that the same record can receive a different label in a different batch
scenario; do not treat those scenarios as independent holdouts. Deployment on a
12-cycle batch is closer in horizon to H12 than primary, but only one late H12
origin exists. No cycle-wise versus global optimizer grid or selection.

Compare each source's decoder with its own fixed .50 labels, and report B01 main
candidate against original A02 and Sparse25. Record selected K, implied score
boundary, positive rate, soft objective, actual Macro F1/confusion/guards per
origin. Query labels are joined only after K/labels have been fixed. Maintain
same-row comparisons and missing equality. Include existing B01 label-shuffle
source only as a negative-control diagnostic if it can be reconstructed without
fits; don't let its scores select the candidate.

Eligibility for further review only: B01-decoder full-primary and new-origin
gains versus unchanged B01 each >=.0005, late/H12 nonnegative, >=4/5 trained
origins nonnegative, no window harm>.003, and no supported F1harm>.01 or recall
harm>.03 versus B01 (n>=200, eachclass>=30). Report other source arms regardless,
but do not replace the preselected B01 candidate by their winner. Failure is a
completed negative experiment; no threshold rescue. These are adaptive diagnostic
criteria, not qualification or submission approval.

Tests: exact soft-confusion formula on tiny hand calculations, all-zero/all-one
and tied scores, missing fixed labels, degenerate classes, y-perturbation leaves
K/predictions unchanged, independent confusion metric replay. No fit in tests.
Hash/freeze source/recipe/inputs before score output. Use separate namespace
scripts/optimization_v3_b02_*.py, corresponding tests, reports/optimization_v3/
b02_plugin_allocation_v1/, ignored derived outputs if needed. Atomic durable
publication and no-fit ledger SHA check; one concise report, no notebook.

Primary literature grounding, not dataset-specific guarantees:
- https://arxiv.org/abs/1402.1892 (F1 threshold/batch dependence; do not transplant
  the positive-class half-F1 corollary to binary two-class Macro F1).
- https://proceedings.neurips.cc/paper_files/paper/2014/hash/3644e33a5161ec5f3997a6acb98d4447-Abstract.html
  (plug-in classifiers for non-decomposable measures; assumptions matter).

## R-NBR — Regression neighbor transfer readiness (execute preflight only)

Check the six R06 OOF cache/sidecars and whether exact first-four-cycle donor
predictions, raw feature/ID/cycle alignment, source recipe and saved geometry
exist for past-only banks. Previous location notes are not proof. Compare
Classification and Regression Train feature/ID alignment before reusing any
target-independent geometry. Never join Classification labels into Regression.

Report whether constructing a NEW Regression indexed target memory would count
as new task fits under the A02 accounting convention. Do not call it zero-fit
merely because indexing has no gradient optimizer. No new bank/index fitting or
candidate Regression prediction in this preflight. Missing cache never permits
refitting old experts.

If ready, draft one MAE-appropriate protocol comparing R06 with local correction
and a matched temporal/prediction-bin shuffle. Distinguish weighted median (MAE
functional) from weighted mean residual transport. Fix a minimal recipe, scale,
cap, whole-cycle rank handling, missing fallback, and all controls before any
future fit. Cost every new target-memory/index predictor conservatively. Prior
Regression residual-head failure and horizon bias reversal are counterevidence.
R12 present-only rescue remains an old reserve, not a new discovery.

Use reports/optimization_v3/rnbr_readiness_v1/ for inventory/report/proposed card.
No Test/Sample/Kaggle, models/targets changed, new fitted state or fit budget.

## Positional text — design only reserve

Check whether absolute early/middle/late token placement was actually tested;
ordinary bigrams already encode local order. Only position information is new.
A credible design needs a matched within-document position-permutation control,
preserved token multiset/feature budget and exact tokenizer, alongside ordinary
bag-of-words. Do not attribute bigger-feature-space gains to position. Specify
learner/recipe/control/fit costs and target-free normalization before making an
execution card. Mark unresolved implementation choices as not executable yet.
No fit authorization; this reserve does not bypass cap66.

## Execution and callback

One Luna Git writer; root has finished writing this file and allows Luna to
commit this exact queue file with owned new research paths. Preserve B01 files,
artifacts, raw data, EDA and teammates. No push/merge/rebase/reset. Do B02 and
Regression readiness regardless of one another's scientific outcome, then reserve
design if enough context. No new unreviewed policies beyond B02.

On each completed milestone or genuine blocker, message KASDAD - TK2 - Sol,
local01a19569-16e0-7253-aa14-57a0a3160e03, before the final user-facing report.
Human explicitly authorized bidirectional messaging. Routine acknowledgment is
not completion: continue the remaining independent authorized tasks. Block only
dependent work, preserve checkpoint, no automatic fit retries. At queue end,
send results/counterevidence/next recommendation/fit0/ledger hash/commit and stop
for Sol's next decision; do not claim a positive policy from an adaptive rerun
is confirmed, and do not wait in a loop to consume quota.
