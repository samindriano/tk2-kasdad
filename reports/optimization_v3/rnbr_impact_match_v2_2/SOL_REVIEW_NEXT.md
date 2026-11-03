# Sol review and next bounded pilot: local target median

**RETIRE exact-impact residual matching; GO for one different functional.**
Sol read the results/postfit wrapper and independently recomputed MAE from safe
observed bundles joined rowwise to official Train IDs/cycles. E67 W4=11.535668,
E67 H12=11.586566 and E75 W6=9.730564 match. Prediction=clip(baseline+delta)
also matched exactly. The postfit suite was rerun:6 passed. Old artifacts,
verifier corrections, ledger and negative results remain preserved.

Exact-impact fails W4/H12 and does not improve original R-NBR sufficiently.
No continuation, full-Train fit, weight search or upload for this recipe.
The callback's opening reference to N04 is unrelated historical context: this
verdict concerns RNBR_IMPACT_MATCH_V2_2, not the old domain-weighting N04 lane.

## Next question: transport the local target functional, not donor residuals

For cosine-squared weights, median(y_donor-p_donor)+p_query is generally not
median(y_donor). Donors with similar text can have different structured
predictions. Residual transport assumes local error corrections remain useful
across those contexts. Exact-impact did not solve the observed instability.

Hypothesis: a shrunk local weighted median of actual past Regression targets
can supply information without subtracting each donor's historical baseline.
Counterargument: text similarity alone may mix different conditional target
distributions; structured R06 may already be better. This is a single bounded
test of that functional, not evidence it will improve public MAE.

Novelty: original R-NBR uses median residuals; exact-impact changes retrieval;
R18/R19 are scalar/group residual offsets; supervised-token statistics encode
tokens rather than retrieve documents. Classification B01 tested a local label
rate, but this is a separate Regression weighted-median functional under MAE.
No new empirical result for this recipe has been computed. All development and
public feedback is already known: the next pilot remains adaptive, not holdout.

## RNBR_TARGET_MEDIAN_V1 recipe and controls

- Official Regression Train only. Origins67 and75, donor present cycles56-E,
  queries E67=68-79, E75=76-79; identical rows/baselines to original R-NBR.
- Reuse original TEXT-ONLY top32 positive-cosine geometry, NOT exact-impact
  geometry. Read safe verified numeric NPZ only; same missing/<5 fallback.
  Verify source rows/IDs/cycles, OOF lineage references and full geometry hashes.
  If geometric reconstruction is needed, use the two standalone replayed TF-IDF
  states already persisted in the impact repair. No fit_transform, Class data,
  full Classification model unpickle or extra preprocessing fit is authorized.
- For each supported query, compute m=weighted_median(y_donor, cosine^2)
  with the frozen stable source-row/half-mass midpoint rule.
  delta=clip(0.25*(m-R06_query),-5,+5).
  prediction=clip(R06_query+delta,0,100). No rerank. Missing/<5 positive
  neighbors exactly R06. Do not use query labels for any construction or route.
- Control: permute donor TARGET values within exact donor cycle x fixed R06
  donor-prediction decile, not impact cells. Fixed decile=min(9,floor(p/10)),
  clip p0..100. RNG42 reset per origin; ascending cycle/decile/source-row.
  Same geometry, median, scale/cap and query fallback. Preserve target range.
  Do not reuse original shuffled RESIDUAL predictions as this new control.

Two origins x observed/shuffled = FOUR new target-memory fits; target lookup
counts as fit even without optimizer. Reuse exact R06 and original R-NBR caches
as additional comparators; exact-impact remains a labeled side comparison only.

## Execution and prospective decision

Append cap83->87,pilot42->46,continuation38/control_replay2/retry1 unchanged.
No retries/refunds/new base or preprocessing fits. Verify current terminal83
starts/80 saves/3 historical failures/pending0 before amendment. Reserve four
new unique IDs, durable start immediately before new target-memory construction.

Preflight source/row/geometry integrity and exact fallback before targets; verify
original comparator loads correctly before fits. Freeze parameter/source/input
hashes and synthetic completed-ledger->verifier->report execution before fitting.
Keep this adapter small; reuse verified pure helpers, avoid another generic
framework. Synthetic cases cover asymmetric donor baselines (distinguishes this
functional from residual median), half-mass ties, unsorted Train cycles, Unicode
IDs, missing/<5 fallback, receipt tamper, view isolation and completed accounting.

Preflight>=95% of present rows with>=5 positive neighbors in each W4/W6/H12
view and>=80% donor source movement in the fixed shuffle. If failed, SKIP before
memory fits without tuning. Reuse existing clean text states; integrity errors
go to Sol, never silently fallback or refit.

Declare pilot ADVANCE only if all:
1. W4 and W6 each gain>=0 vs R06; pooled W4+W6 gain>=0.03,n13,441.
2. Fixed-origin E67 H12 gain>=0 vs R06,n20,903.
3. Pooled gainvs matched target-shuffle>=0.01 and H12 gainvs shuffle>=0.
4. Pooled gainvs original R-NBR>=0.03 and H12 gainvs original>=0.
5. Supported missing/target<=5/target>=95,n>=100,harm<=0.25 vs R06 across
   W4,W6,pooled,H12; missing exact R06. All integrity/accounting checks PASS.

All-row metrics decide; common-supported results are diagnostic. Report every
cycle, denominator, fallback and unsupported group. Preserve old R06 W1 missing
legacy harm0.395556,n392. New pass cannot clear it. Primary/H12 overlap and
post-public selection remain explicit. FAIL retires this functional without
rescue grid; PASS is pilot only, not automatic continuation or upload.

## Ownership and handoff

New namespace rnbr_target_median_v1 only, frozen historical sources untouched.
Save prediction/target-memory/geometry references and receipts before reports;
verify safe readback/resume without new fits. One job,two threads,6GiB plus2GiB
free reserve; no installs,Test/Sample/Kaggle,notebooks/teammate edits or push.
Locally commit scoped scripts/tests/reports. Callback Sol with exact gates,
negative evidence, independent verification, ledger/hash and commit before final
reply; then stop. Sam already authorized measured successor scopes; no routine
approval pause. Genuine new blocker requires a compact Sol callback.
