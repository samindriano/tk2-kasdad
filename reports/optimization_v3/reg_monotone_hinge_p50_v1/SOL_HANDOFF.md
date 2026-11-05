# REG next bounded lane — monotone hinge at fixed prediction50

User explicitly requested continued research/decision/benchmark cycles beyond a checkpoint or submission. Root decides between stages; no repeated user approval question. Current best confirmed ref58010862 MAE11.04304, GLOBAL_PRESENT_ONLY_RESTORE_DIRECT_MISSING_E79. H8recency stays retired; old unused fifth LP is cancelled, not repurposed. This lane has a separate cap3 and separate ledger, preserving historical counts.

Status GO_PREPARE_SOURCE_PENDING_SCOPED_NOVELTY_CHECK. If exact same fixed-knot monotone L1 calibration is found previously tested, callback before implementation; no blind duplicate. Root's targeted MAIN search found no piecewise/isotonic/spline calibration implementation; not exhaustive evidence. Luna2 finishes bounded REG novelty check then prepares below. No official LP until ROOT_SOURCE_PASS after actual source review. No user question needed.

## Hypothesis and recipe

Affine calibration already helped but cannot express different low/high score slopes. Test ONE continuous monotone two-segment map at a fixed a-priori prediction knot50 (midpoint of0..100). Not a fitted knot, target bin, rank feature, isotonic fit, or grid. Mathematical form:

 g(p)=a+sL*min(p,50)+sH*max(p-50,0).

Fit a free, 0<=sL<=2,0<=sH<=2, minimizing unweighted SUM absolute error on historical OOF direct predictions. Three parameters only. This contains affine maps within the slope bounds as sL=sH. Use sparse HiGHS LP with nonnegative per-row slacks, two threads and fixed180s solver time limit; no retry, grid, seed or knot search. y and p must be finite0..100. Clip only at application, not inside training objective. A fixed bound2 constrains extrapolation and is a disclosed design choice, not a discovered optimum. Training loss improvement alone does not support generalization.

Fit population ALL same historical OOF rows, including missing rows, matching causalGLOBAL affine controls; final application changes PRESENT only and keeps current missing exactrawDIRECT. This isolates map shape instead of changing training population or fallback simultaneously. No recency weights. No new base model, vocabulary, SVD, positional or other preprocessing fits.

## Reused frozen data and comparators

Reuse signed raw direct E67/E75 bundles and exact row-index stitch from closed REG_FINAL_WAVE_FOLLOWUP_V2 R3. Freeze all historical source/receipt/manifest/row hashes in NEW stage before target access. Do not execute old closed-stage live-HEAD gates or alter their freezes; reuse reviewed pure identity/math helpers or persist explicit new-stage bindings. Verify raw CSV/archive digests using each producer's original serialization schema: direct raw little-endian float64 hash vs tagged global/vector hashes. Test schema distinction prospectively, not via a failed official access.

D1: calibrate E67 predictions cycles68–71 (4950rows), score E67 cycles72–75 (7462). D2: calibrate E67 predictions cycles68–75 (12412), score E75 predictions76–79 (8491). Exact training/score IDs, origin, original bundle indices and cycles must match prior receipts. Pool evaluation only W5+W6,15953unique rows. Final possible fit uses20903unique OOF E67cycles68–75+E75cycles76–79. Prediction-origin/horizon shift at D2 is a limitation. All labels already reused; no fresh holdout/outcome-blind claim.

Mandatory causal comparator: reused exact D1_GLOBAL/D2_GLOBAL coefficient states from closed R3, applied to present only with raw direct missing fallback. Do not use final current-best coefficients fitted on score windows as independent validation. Current full-support coefficient operational reference may be descriptive only. Compare additionally raw direct and causalGLOBAL all rows for context; they do not select a different policy.

## Fit accounting and evaluation gate

Fit IDs HINGE_D1,HINGE_D2,HINGE_FINAL; max3, two diagnostics first, final only after Sol results review. One start/save per ID, failedstartconsumes slot, no retry. Immutable atomic outputs and source/input/row-bound recovery of published artifacts+receipts without optimizer rerun. Old four-diagnostic ledger remains unchanged. New learned preprocessing units0, base fits0, diagnostic calibrator fits2, conditional production1.

Before each LP require training support >=1000 and >=200 historical p<50 and >=200 p>=50; otherwise report unsupported and stop before its start, no moved knot. Report side-support on score windows too. Final requires>=5000total and>=200 eachside. No requirement of artificial target balance for regression.

Report full-row and present-only MAE, exact denominators, coefficients/slopes/bounds activity, error mean, clipping, percycle signs, missing/present/y<=5/y>=95 with n>=200 support elseUNKNOWN. Fixed application missing rows must stay bit-exact current direct. Independent metric replay, objective/constraints solution checks, label/source boundaries, prediction hash and receipts before any candidate promotion.

QUALIFIED: pooled gain>=0.02 versus causalGLOBAL_PRESENT_ONLY and both windows nonnegative; supported subgroup harm<=0.25.
EXPLORATORY_ROOT_REVIEW: if notQUALIFIED, pooled gain>=0 OR (onewindowgain>=0.02 and pooledharm<=0.01), eachwindowharm<=0.03 and supported subgroup harm<=0.25. No retroactive scientificPASS. Otherwise retire exact lane with mechanistic readout and sendrootresults; no automatic adjacent-knot/slope rescue. Public scores never pick parameters.

Conditional final/CSV stage: Sol may authorize one HINGE_FINAL and exactly one CSV with present=clip(g(rawDIRECT),0,100), missing=rawDIRECT, 29421ordered officialIDs, finite0..100, exact1062missing fallback. Test raw labels/Sample/new inference forbidden; hashverified savedarrays only after freeze. No public score expected/guaranteed. No Kaggle upload by Luna; root independently verifies then decides at mostone measured upload under standing authorization.

## Implementation, resource and continuity

Luna2 owns new scripts/model/helper/tests, experiments and reports under reg_monotone_hinge_p50_v1 only; no historical source/protocol/notebook/teammate changes. Keep implementation small and reuse verified pure helpers instead of copying entire old runner/verifier scaffolds. New source freeze independent of closed R3 source checkpoint. No strict liveHEAD==freeze; source ancestor + pinned executable/input bytes, report-only updates safe. Preserve known R1/R2 historical untracked paths; stage only named current files, local commits okay, no push/merge/reset.

Candidate-blind tests: known continuous hinge recovery, equal-slopes affine equivalence, monotonicity/continuity and slope bound violations, weighted-vs-unweighted prohibition, wronghash/roworder rejected before labels, stored-global control exactreplay, causal stage boundaries, no duplicate fitting after postpublish crash, inputhash schemas, CSV route/IDs/range. Use a separate minimal verifier to reconstruct metrics/objectives/CSV, not builder self-attestation.

SharedOSlock %TEMP%/kasdad_single_heavy_job.lock, fresh4GiB available and2GiB reserve, job1/thread2. CLS currently has priority for query evaluation; prepare source/tests/identity while it works. If lock/RAM insufficient return exact resumable checkpoint once, no polls/schedules or process killing. Root controls serialized compute release.

First deliver source-ready localcommit, fixed protocol/source+inputfreeze identity-only, tests/syntheticQA/commands/costUNKNOWNs to Sol01a19569-16e0-7253-aa14-57a0a3160e03. Source review->diagnostics->production/upload decisions are internal Sol/Luna callbacks, not user-review stop points. Every meaningful result/blocker must callbackroot before final. Do not silently idle with unfinished task or ask Sam. After a failed lane root may choose a different grounded hypothesis; do not self-start unbounded experiments.
