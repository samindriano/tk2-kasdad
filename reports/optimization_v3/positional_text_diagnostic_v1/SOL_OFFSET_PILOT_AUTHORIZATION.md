# Sol decision: one frozen-bag positional-offset pilot

**GO for four predictors, no preprocessing fits.** Root reviewed diagnostic
decomposition and reran its6 unit tests:PASS. The drop-block comparison is not
a fair refit ablation and does not establish an incremental positional benefit.
However, holding the original bag coefficients fixed while fitting only the
added channel tests a different, narrow constraint. The original joint-fit
three-third recipe remains retired. This follow-up was selected adaptively;
the existing overlapping Train outcomes are not an untouched holdout.

## Exact POS_BAG_OFFSET_V1 recipe

Reuse saved matrices and faithful bag LR from POSITIONAL_TEXT_PILOT_V1, E67
and E75, real and seed42 shuffled positional blocks. Verify all frozen source,
matrix,preprocessor,estimator and row hashes before use. No new vocabulary,
IDF,transform fitting,scaling,bag coefficients or intercept is learned.

For fit-prefix rows compute z0=original bag intercept+Xbase@original bag beta,
directly in logit space, not logit of rounded/clipped probabilities. Fit only
gamma, starting exactly at zero, minimizing

sum(logaddexp(0,-(2*y-1)*(z0+Xpos@gamma))) + dot(gamma,gamma)/(2*C), C=1.

Gradient=Xpos.T@(expit(z0+Xpos@gamma)-y)+gamma/C. Loss is SUM, not mean.
No class/sample weights or learned intercept. Float64 sparse computations;
scipy L-BFGS-B,maxiter2000,ftol1e-12,gtol1e-6,maxls50. Require success, finite
objective/gradient/coefficients and final loss<=zero-gamma loss+1e-8. Report
termination reason,nit,nfev,gradient infinity norm and loss change. Failure is
consumed FIT_FAILED with no retry, warm-start or solver change. Do not tune C.

Fit on all same-prefix rows as original C2_B; blank position rows have zero
positional gradient contribution. The offsets come from an in-sample fitted
bag model; this is sequential additive fitting, NOT cross-fitted residual
training. Report this limitation without relabeling it leakage across the
outer boundary. Query labels never enter construction or optimization.

Query probability=expit(z0_query+Xpos_query@gamma). Missing/blank stream routes
exactly to the saved C1_B HGB expert. Threshold>=0.50. Use unmodified real and
shuffled blocks from the completed pilot, no new mixing/channel scale. Zero
gamma and empty blocks must reproduce bag probabilities within numerical
tolerance. No A02 residual reuse or champion blending.

## Views, controls and frozen gates

E67 fit47-67->score68-79 with W4/W5/W6 and H12_E67; E75 fit47-75->score76-79.
The W6 outcomes overlap and history comparisons are not independent replication.
Compare real offset versus original bag and shuffled offset. Joint-position
pilot may be a clearly labeled descriptive reference, not a new fitted control.

Keep preceding pilot gates exactly:
- E67 H12_E67 F1 real>=bag+0.003 and real>=shuffled-offset+0.003.
- E75 W6 real gain>=0 versus BOTH; E67 W4/W5/W6 each harm<=0.005 versus either.
- Fit-defined text_load Q4,n>=200 and>=30 each class,F1 harm<=0.01 and positive
  recall harm<=0.02 versus either control per declared window/aggregate.
- Exact missing fallback,all integrity/accounting/source checks PASS.
Report unsupported groups,EF/availability,all per-cycle signs and secondary
metrics. FAIL ends further positional fits; no seed/C/bins/scale/threshold rescue.
PASS is pilot mechanism evidence only, not full-six qualification or upload.

## Budget, QA and stop point

Current105 starts/102 saved/3 historical failures/pending0. Append cap105->109,
pilot64->68,continuation38/control_replay2/retry1 unchanged. Four unique new
predictor IDs=2origins x real/shuffled. No extra fits of any kind. Frozen state
reuse, prediction replay and independent verification consume zero new fits.

Before official fitting: inspect saved fit-row boundaries and dimensions; run
finite-difference gradient tests including nonzero offsets and L2, stable extreme
logits, fixed-base/intercept immutability, gamma0 equality, blank position block,
class order,row-ID tamper and exact fallback. Actual synthetic optimizer->durable
receipt->resume->independent verifier->metric/report/completed ledger path must
pass using explicit temp-only ledger paths. No import-captured live defaults;
production writer guarded during tests,finally restoration+livehash checks.
Preserve the recorded earlier append/removal incident and never rewrite ledger.

Freeze source/input/optimizer/matrix/gate plan on actual clean execution commit
before fits; persist gamma,raw predictions,receipts and termination data before
reports or outer target scoring. Independent verifier checks numerical
objective/gradient and score formulas without refitting any optimizer.

New namespace positional_bag_offset_v1 only. One job,two threads,6GiB RSS,
4GiB initially free and2GiB running reserve. No Test/Sample/Kaggle,install,
notebook/teammate/historical edits,push or fit retries. Local scoped commit.
Callback Sol with all views/gates/counterevidence,optimizer/convergence,QA,
ledger/hashes and commit BEFORE executor final reply, thenSTOP. Routine unfit
implementation choices authorized; actual new blockers go to Sol promptly.
