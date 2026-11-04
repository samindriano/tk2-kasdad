# GO — six-fit CLS_BIGRAM_POSITION_PILOT_V1

Sam requests continued measured research after both exploratory submissions.
Sol reviewed READINESS_REPORT/PROPOSED_CARD and reran the5 focused tests: PASS.
Support is operationally adequate, not evidence of predictive gain. Authorize
exactly six gamma fits, BAG/REAL/SHUFFLED at E67/E75, under the proposed feature
construction and frozen original C2B+unigram-POS logit anchor. No new transform,
base/A02/HGB/index fit, weight/C/seed/degree sweep, Test or submission in pilot.

Luna1 thread01a18b41-4be9-7cb2-b8b5-d4cd45ed8e18 owns MAIN CLS only. Luna2
thread01a1a442-590f-7520-a09a-bcd10f292328 owns isolated REG; never interchange.
Sol callback01a19569-16e0-7253-aa14-57a0a3160e03 hostlocal. No scheduled tasks.

## Binding refinements before first fit

- The source proposal's generic 'unchanged gate' language is replaced by exact
  prospective criteria below. Write resolved recipe and freeze before fits.
- BAG uses complete known bigram counts, original IDF, sublinear TF, L2 row norm;
  REAL/SHUFFLED use three-third counts and joint norm as the proposed card.
  BAG is lower dimensional; REAL-vs-SHUFFLED is the matched-capacity contrast.
- Shuffle pair-occurrence assignments, preserving pair multiset and bin sizes,
  not raw tokens (which would alter the bigram identities). No semantic claims.
- Prefix fit E67<=67,n29705; E75<=75,n42117. Query E67=68–79,n20903;
  E75=76–79,n8491. Same five views and overlapped development outcomes as before.
- Reconstruct frozen z_original on fit/query from verified C2B base matrices,
  BAG coefficients/intercept and ORIGINAL unigram POS gamma/matrix, not moments
  or contrast. Require old query pP replay and all row/ID/cycle/mask identities.
- Fit gamma only from zero: SUM logistic loss(y,z_original+P@gamma)+||gamma||²/2,
  C1; original pure L-BFGS-B maxiter2000,ftol1e-12,gtol1e-6,maxls50.
  Require success/finite values/final objective<=initial+1e-8; no automatic retry.
- Policy present=.5*sigmoid(z_original+P@gamma)+.5*originalA02; missing EXACT pP;
  threshold>=.50. For zero-known-bigram rows return original MIX50 exactly using
  cached probabilities. Validate zero-gamma equivalence to incumbent numerically.
- Only REAL is a potential later candidate; BAG/SHUFFLED are controls.

## Exact prospective gates (tolerance1e-12)

All required: REAL blend gain>=.0005 versus MIX50 on BOTH E67 H12 and E75 W6;
REAL gain>=0 versus BOTH BAG and SHUFFLED on both required views; E67 W4/W5/W6
F1 harm<=.003 versus MIX50. Same fit-defined availability, EF and text-load
groups on all5 views: n>=200 and each class>=30, supported F1 harm<=.01 and
positive recall harm<=.02 versus MIX50. Unsupported explicitly UNKNOWN.
Missing/no-known-bigram fallback and all integrity/independent QA must PASS.
These are engineering criteria, not significance thresholds. Report all arms,
per-cycle/probability metrics and failure evidence; do not redefine old verdicts.

## Accounting, readiness and execution

Verified current shared state: cap143=pilot88+continuation52+control2+retry1;
138 starts/132 saves/6 historical failures, no pending. Append new cap149 and
pilot94, other categories unchanged. Reserve six unique pilot IDs once only,
fixed order E67_BAG/REAL/SHUFFLED then E75_BAG/REAL/SHUFFLED. No E79 reservation,
no released-slot reuse/refunds. Luna2's independent REG ledger is separate.

Before starts, independently verify ALL actual source matrices/formulas/logit/
receipt/hash/row contracts on actual input schemas, plus synthetic actual-entrypoint
six-fit temp-ledger lifecycle, persistence/reload/resume, all guard/scoring paths.
Explicitly test safe Unicode NPZ IDs, parent manifest list/map schemas, raw vs
header-aware array hash conventions, scalar summation reproducibility and actual
callback signatures. Same numeric arrays with different hash contracts must be
validated by their respective producer rules, never compared as equivalent hex.
No source/refit rescue after a consumed failure; recover durable output only.

One fit process,2 threads,6GiB RSS ceiling,4GiB initially free/2GiB reserve.
Coordinate real compute with Luna2; never kill its process/delete locks. Use
direct callback if scheduling needs root; no silent idle or periodic chat polling.
Commit all source/recipe/tests and freeze BEFORE fits; durable model/receipt/
prediction + full prelabel independent QA before new query metrics. All six
fits must be reconciled, including failures. No installs, notebooks, teammate
edits, push, merge or Test/Sample/Kaggle. Routine prefit metadata/schema repairs
authorized with successor source/freeze; substantive data/formula mismatch to Sol.

Finish six fits, independent verification, concise report/max2plots/local scoped
commit and SEND result or exact real blocker to Sol before final reply. Root
decides the next fit or a separate exploratory benchmark; no automatic E79.
Do not stop halfway to request this same approval again. Old failed gates and
negative public results remain preserved. Sol stops writing after handoff.
