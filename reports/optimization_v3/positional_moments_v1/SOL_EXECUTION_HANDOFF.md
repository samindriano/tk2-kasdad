# Luna 1 — POS_MOMENTS_V1 bounded Classification continuation

**GO: four new gamma fits.** Sam explicitly asks Luna 1 to continue after the
contrast pilot. That pilot remains RETIRED under its original gates. No watchdog,
scheduled task or periodic chat polling is authorized. Send direct callbacks.

Luna 1 = thread01a18b41-4be9-7cb2-b8b5-d4cd45ed8e18, main TP 2 worktree.
Sol = thread01a19569-16e0-7253-aa14-57a0a3160e03, hostlocal.
Luna 2 = thread01a1a442-590f-7520-a09a-bcd10f292328, REG only, isolated
tk2-regression-frontier worktree; never edit or take over that lane.

## Decision and falsifiable hypothesis

Original thirds POS gave public .83123 and fixed MIX50 .83144. Contrast-only
removal did not improve MIX50 (E67 H12 -.000138, E75 W6 -.001702). That rejects
the tested restriction, not all positional information. Its failure does not
prove the removed direction is causally helpful. The earlier bounded novelty
audit found no tested continuous per-token position moments.

Test a smoother positional representation while RETAINING an explicit unigram
bag channel: it can express front/back and edges/center variation without hard
third boundaries. This changes representation, not the gamma learner/objective.
It may smooth away useful structure and fail. No promise of public improvement.

## Exact feature definition (not the earlier two-channel sketch)

Use each origin's frozen C2_B word vocabulary/IDF, same unigram tokenization,
same original faithful BAG_LR model and fit/score base matrices as POS_BAG_OFFSET.
No learned transform fits. Verify saved row IDs, documents, model/preprocessor,
vocabulary/IDF and matrix hashes before new fits.

For a tokenized document of n tokens, including OOV occurrences for positions:
- n>1: u_i=2*i/(n-1)-1 for zero-based i; otherwise h1=h2=0.
- h1_i=u_i.
- h2_i=(3*u_i^2-1)/2 minus the mean of that quantity over ALL document positions.
  For n<=1 set both moments exactly zero; n=0 gives an all-zero block.
- For each known unigram t with count c_t, a_t=(1+log(c_t))*frozen IDF_t.
- B_t=a_t/sqrt(sum_known(a_t^2)); zero if no known unigram.
- m1_t=mean(h1_i over occurrences of t); m2_t analogous.
- Form W=[B, B*m1, B*m2], then L2-normalize the entire concatenation ONCE.

Three channels and3*d coefficients match original thirds head dimension, though
feature geometry/normalization differ. Do not call the comparison an exact
information ablation. Unlike the prior two-moment exploratory suggestion, the
explicit B channel here preserves bag reweighting freedom. Freeze THIS formula.

SHUFFLED applies original deterministic_order_shuffle(seed42) to token order
before computing moments; the original normalized token bag/length/IDF and B
are identical. No row ID/target/global RNG input to permutation. Reuse exact
original tokenization/shuffle source; no n-gram or vocabulary/degree/scale sweep.
No raw report/token content in Git. Unknown tokens contribute to position index
and document-wide h2 centering but produce no feature column.

## Learner, populations and fixed policy

Two origins E67/E75; REAL and SHUFFLED arms. Fit all original prefix rows:
E67 cycles47–67,n29705, query68–79,n20903; E75 fit47–75,n42117,
query76–79,n8491. Blank feature rows are zero; query missing uses original pP.

z0 = original BAG_LR intercept + original Xbase @ beta, directly float64 logit.
Learn gamma only from zero, no new intercept, using unchanged pure solver
src/models/positional_bag_offset_v1.py:
SUM logistic loss(y,z0+W@gamma)+||gamma||^2/2, C1,
L-BFGS-B maxiter2000,ftol1e-12,gtol1e-6,maxls50.
Require optimizer success, finite objective/gradient/coefficients, final objective
<=zero-gamma objective+1e-8; report convergence details. No retry/warmstart/rescue.
In-sample bag offset is sequential additive fitting, not OOF residual training.

For each arm q=expit(z0+W@gamma). FIXED comparison/submission policy:
present=.5*q+.5*original_A02; missing=original pP exactly; threshold>=.50.
Only REAL may become a candidate. Raw q, shuffled, two-channel moments, other
weights/thresholds are not alternative deployment choices.

Fit order E67_REAL,E67_SHUFFLED,E75_REAL,E75_SHUFFLED. No query labels in fit or
feature construction. Commit recipe/code/tests and source/input freeze BEFORE
starts; all four model/receipt/prediction artifacts and prelabel integrity PASS
before new policy metrics. These reused overlapping views remain adaptive data.

## Numerical gates, unchanged current comparator

All required, tolerance1e-12:
- REAL blend gain>=.0005 vs existing MIX50 on BOTH E67 H12 and E75 W6.
- REAL blend gain>=0 vs SHUFFLED blend on both required views.
- E67 W4/W5/W6 each Macro F1 harm<=.003 vs MIX50.
- Same endgame availability/text-load/entry-friction groups, prefix-only cuts,
  all5 views, support n>=200 and each class>=30; supported F1 harm<=.01 and
  positive-recall harm<=.02 vs MIX50; unsupported UNKNOWN.
- Missing bit-exact pP and all independent formula/source/boundary/metric QA PASS.

Report per-cycle directions, probability metrics, direct q vs bag/oldPOS and
fixed policy vs A02 descriptively. No new gates picked from outcomes, no public
confidence interval or independent-replication claim. FAIL returns complete
findings and no E79; do not relabel a near miss PASS.

## Accounting and integration readiness

Current shared ledger expected133 starts/127 saves/6 historical failures/0pending,
SHA7816f8c64f760347cd4a5c7596a97ef99c0059a352994be00f45bce4297a986c,
caps137=pilot84+continuation50+control2+retry1. Read live state before amendment.
Authorize append-only cap142: pilot88,continuation51,control2,retry1. Exactly
4 new pilot IDs POS_MOMENTS_V1_E{67,75}_{REAL,SHUFFLED}, plus1 reserved conditional
E79_REAL ID AFTER Sol review. Do not reuse released contrast reservations.
Luna 2 uses its own isolated REG cap6 ledger and must not be counted as shared
fits; report ledgers separately for later root reconciliation. No new retry.

Use small new model/runner/verifier/tests in positional_moments_v1 namespace.
Preserve all old modules/freezes. Prefer pure old solver/tokenizer helpers and
an explicit adapter over monkeypatching an inherited runner. Actual entrypoint
synthetic lifecycle must execute budget call signatures, start/publish/receipt/
reload/prelabel/metrics/guards/accounting/resume using a TEMP-ONLY ledger. Guard
production ledger against any test write. Tests must pass before AND after real
runs without requiring an obsolete live snapshot. Add cases n0/1/2, repetition,
allOOV/Unicode, reversal h1 sign/h2 parity, exact B shuffle invariance, bounds,
zero-gamma equivalence, unchanged fixed bag and missing route, tamper/wrongorigin,
and interrupted-after-save recovery without another fit. Check actual fields in
signed parent lineage before freeze; no missing-key repair after fitting needed.

Do target-free real-data preflight: matrices align, finite, expected3*d width,
nonempty rows norm1, zero rows valid, document hashes, bag/length preservation,
runtime/RAM estimate. Not an outcome gate. Do not claim token semantics.

One production fit process/two threads/6GiB RSS limit,4GiB initially free and
2GiB running reserve. Coordinate compute with Luna 2; never kill/delete its job
or locks. Preparation/tests can run concurrently. If another production fit is
active and existing guards require exclusivity, finish ready state and send a
direct scheduling callback to Sol; do not convert it into fit failure or silent
idle. Sol coordinates via messages, NOT a scheduled task. No wait >60seconds.

Routine PRE-FIT implementation corrections authorized with successor freeze;
preserve failure evidence and bind any existing reservations explicitly without
duplicating slots. Started failed fits stay consumed, with no automatic retry.
Save each optimizer artifact/receipt durably before next fit. If a reporting
error occurs, recover saved outputs without retraining. No old-source edits.

## End-to-end milestone and direct return

Complete implementation, tests, freeze, all4 fits, independent replay, concise
results (max2 useful PNGs), accounting and scoped LOCAL commit. No push,
notebook/EDA/teammate edits, installs, Test/Sample/Kaggle access during pilot.
Send result or real blocker with exact evidence, all gates/counterevidence/QA,
paths/hashes/accounting/commit to Sol chat01a19569-16e0-7253-aa14-57a0a3160e03
hostlocal BEFORE final reply, request the next Sol decision, then stop for that
decision. Do not silently stop mid-pilot or start another family on your own.

If eligible, Sol independently reviews before permitting the one conditional
E79 gamma fit and one possible reviewed CSV/upload under renewed user score-push
authority. No E79 or submission by Luna from this initial protocol alone.
No watchdog, scheduled task, repeated chat polling or unsolicited automation.
