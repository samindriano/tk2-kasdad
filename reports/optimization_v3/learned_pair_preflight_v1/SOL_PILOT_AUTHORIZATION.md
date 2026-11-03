# Sol: four-fit learned-pair Classification pilot

**GO** under Sam's standing measured-research authorization. Root reviewed the
FM identity and sparse factor gradient and reran6 prototype tests:PASS. N02's
negative results reduce expected value; they do not test learned rank16 factors.
No guarantee of public improvement. This is an adaptive mechanism pilot, not
fresh holdout evidence or automatic champion selection.

Use the sole proposed card in REPORT.md, with the following prospective
clarifications taking priority. New namespace learned_pair_pilot_v1.

## Exact recipe and optimizer

E67/E75, existing verified C2_B fit_base/score_base matrices, unchanged main
block X; interaction block Q is a target-free row-L2-normalized copy of X.
No new preprocessing fits. Score z=Xw+b+.5*sum((QV)^2-Q²(V²)),rank16.
Train w,b,V jointly, no offset or probability residuals. Seed42 factors N(0,.01),
w=0,b=0,reset RNG each origin; no seed alternatives. All same-prefix rows;
query targets excluded until durable predictions and integrity verification.
Control has only Xw+b, same regularization/solver/start for w,b, no V.
Same saved C1_B missing expert and threshold>=.50. No A02 correction/blend.

For numerical scale use SUM binary logistic loss plus
(||w||²+||V||²+b²)/(2*C),C=1. Control omits factor penalty. This is exactly
n times the proposed mean loss with penalties1/(C*n), same minimizer. It avoids
interpreting a tiny mean-loss gradient as useful factor convergence. Validate
loss/gradient equivalence under n scaling, including uneven final chunks.
Accumulate raw data loss/gradient across fixed1024-row chunks, then add L2 ONCE,
not once per chunk. Use stable signed-label logaddexp loss and expit gradients.

Float64 SciPy L-BFGS-B,maxiter1000,maxfun2000,maxcor10,maxls50,ftol1e-9,
gtol1e-4. These replace proposed200/200 limits prospectively, not as a retry.
The old~123-second estimate was for200 evaluations; at2000 evaluations simple
linear extrapolation can approach~20 minutes total before overhead, uncertain.
One fixed run/configuration per fit; no optimizer/seed/penalty rescue.

Require solver success and finite state/loss/gradient, final loss<=initial+1e-8.
Report actual nit/nfev/status, loss,gradient infinity norm (sum and mean scales),
factor change/norm and pair-score contribution. Nonconvergence is an optimization
failure/inconclusive mechanism test, not evidence all learned interactions fail.
Persist terminal diagnostic state/receipt; no retry, release unstarted slots.

## Views and prospective gates

E67 score68-79: W4/W5/W6 and H12_E67. E75 score76-79 W6, overlapping history
sensitivity, not replication. Historical full-six primary is not computed here.
Faithful cached C2_B bag is baseline; same-solver no-pair is the mechanism control.
Reproduce cached missing route exactly. Align cached A02/current scores if
available as separately labeled references, never fit inputs or new boundaries.
If reference unavailable, mark UNKNOWN; do not invent an A02 result or refit.

All following required for pilot ADVANCE:
- E67 H12 F1 gain>=0.0005 vs cached bag AND>=0.0002 vs matched no-pair.
- E75 W6 gain>=0 versus BOTH. E67 W4/W5/W6 harm<=0.005 versus either.
- Supported fit-defined text_load Q4,EF and availability groups,n>=200 and
  >=30/class,F1 harm<=0.01 and positive-recall harm<=0.02 versus either, for
  declared windows/aggregates. Missing probabilities exact in all arms.
- Source/row/matrix/receipt/accounting/independent score checks PASS and both
  compared optimizers converged. Unsupported groups explicit.

These are NEW pilot gates, not retroactive N02 passes. Since E67 H12 is the
aggregate here, do not also apply an old early-H12/primary rule to the wrong
origin or denominator. A significant mismatch between same-solver no-pair and
cached LR must be explained with objective/gradient/convergence evidence;
do not mistake a poorly optimized control for interaction gain. Higher capacity
and nonconvex optimization remain confounds even with this control. All cycle
signs,confusion/precision/recall and secondary metrics reported, not used to rescue
failed F1. PASS is mechanism review only; no continuation/refit/upload automatic.

## Accounting and execution

Append cap109->113,pilot68->72; continuation38/control_replay2/retry1 unchanged.
Exactly4 new predictor IDs=E67/E75 x FM/no-pair. No new preprocessing/basefits,
retries or quota reset. Verify current109 starts/106 saves/3 historical failures/
pending0 before append-only amendment. Preserve earlier ledger incident record.
No rewrite/truncation/deletion of ledger events.

Before official fits: synthetic actual optimizer->durable state/predictions->
completed ledger->independent verifier->report with temp-only explicit paths,
production writer guarded,finally hash/restoration,zero repeated fits on resume.
Cover sum-vs-mean/chunk-vs-full gradient equality,stable extremes,one-feature
zero pair,nonzero initialization,factor/main parameter partition,class order,
unsorted cycles/IDs,matrix hash tamper,missing fallback. Keep implementation
small; no framework or dependency installation. Freeze full recipe/input/source/
optimizer/gates on actual clean execution HEAD before fits. Durable parameters
and raw scores precede opening outer targets; verifier checks formula/gradient
without fitting. Fit-prefix Classification labels only; no Regression targets.

One job,two threads,6GiB RSS,4GiB initially free and2GiB running reserve; record
actual resource observations and distinguish samples from OS peak. No Test/
Sample/Kaggle,notebooks/teammate/history edits or push. Scoped local commit.
Callback Sol all arms/views/gates,counterevidence,optimizer/resource/QA,ledger,
hashes and commit BEFORE final reply,thenSTOP. Routine unfit preparation choices
authorized; real new blocker requires compact Sol callback, never auto-rescue.
