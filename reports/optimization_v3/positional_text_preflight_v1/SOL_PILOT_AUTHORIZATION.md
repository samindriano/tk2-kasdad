# Sol authorization: controlled positional Classification pilot

**GO: two-origin pilot, no Test inference/submission.** Sam authorized measured
successor research. Sol reviewed source constructor, proposed recipe, controls
and C2_B/C1_B source parameters. Root reran positional, fixture lifecycle and
historical target-median tests together:13 passed. The explicit test isolation
is active only for its named frozen lifecycle test; existing historical source
and live ledger remain unchanged. Do not claim this is whole-repository QA.

The preflight demonstrates a changed feature channel, not predictive benefit.
Vocabulary overlap is affected by unequal corpus sizes and temporal composition;
do not describe low Jaccard alone as proof of drift or position information.
Deterministic order shuffle is one fixed control realization, not randomization
inference. No confidence interval or independence is supplied by overlapping
W6 outcomes across histories.

## Fixed recipe

Execute the proposed recipe in REPORT.md with these explicit clarifications:

- E67 fit47-67, score68-79; call pooled view H12_E67 (or H12_L), not H12_E
  which historically refers to the early cutoff55 forecast. Report W4/W5/W6.
  E75 fit47-75, score76-79 as history sensitivity, not independent replication.
- Three LR arms each origin: faithful unordered C2_B, C2_B plus position,
  C2_B plus shuffled-position. One shared C1_B missing expert each origin.
  Same C2_B LR and TF-IDF parameters/source allowlist as verified original.
  Missing expert HGB parameters also unchanged. Threshold>=0.50, seed42.
- ALL three LR arms share exactly one fit-prefix structured state, scaler,
  OHE and bag TF-IDF per origin; same column order and numeric matrix values.
  The missing expert has its own faithful native-categorical structured state.
  Fit LR estimators on already transformed matrices, not three independent
  pipelines that silently refit common preprocessing.
- Position block: three counts vectors over ONLY unigram entries in the shared
  C2_B mixed unigram/bigram vocabulary; use their original shared IDF values.
  Nonzero count c -> (1+log(c))*IDF. Concatenate three segments and L2-normalize
  that combined block ONCE per row, preserving zero rows. Concatenate this block
  with unchanged C2_B bag/structured matrix; no further full-matrix normalization
  or selected scaling. Same exact operations and dimensions for shuffle.
- Boundaries=floor(3*i/n), original tokenizer/normalization. Control permutes
  ONLY positional input with the existing seed42/text-digest routine; original
  bag/bigram and structured inputs stay unchanged. No use of IDs for features.
- Train each LR on all prefix rows as original C2_B; route missing rows to the
  shared HGB expert at score time. Missing probabilities identical across arms.
  Do not reuse A02 residuals or blend the new probability with A02 predictions.
  A02 cached scores may be a labeled reference only, never target/fit input.

## Budget and fitting boundaries

Verified previous checkpoint:87 starts,84 saved,3 historical failures,pending0.
Prospective cap87->105; pilot46->64; continuation38/control_replay2/retry1
unchanged. Count ALL18 new fit units under pilot, not just eight predictors:
per origin5 shared learned preprocessing steps +4 estimators. Explicit fit_kind
separates preprocessing from prediction; do not call these18 predictors.

Five preprocessing units each origin: C2_B structured state, scaler, OHE,
TF-IDF, and C1_B structured state. Each canonical structured-state step receipt
must enumerate any nested category/imputation learned statistics, their prefix
boundaries, hashes and actual underlying fit calls. This budget counts those
two aggregate structured-state steps once each; do not conceal a separate
estimator fit inside them. No duplicated preprocessing on candidate/control,
no scoring-data fit, no implicit fit on import/replay/verification.

Reserve18 unique IDs with actual durable start/save and dependent receipts.
If implementation cannot achieve this shared fit plan, report revised cost to
Sol BEFORE fitting; no automatic cap expansion or skipped accounting.
Save learned transforms/matrix hashes once, then reuse for all arms. Outer
scored labels only opened after durable predictions/integrity verification.
Fit-prefix Classification labels allowed for estimators; Regression labels never
used. No retries, base-expert extras, threshold tuning or max_iter rescue.

## Prospective mechanism gates

1. E67 H12_E67 Macro F1 position improves>=0.003 versus BOTH bag and shuffle.
2. E75 W6 position gain>=0 versus BOTH controls. E67 W4/W5/W6 each harm<=0.005
   versus either control; all views, denominators and cycle signs disclosed.
3. For fit-defined text_load Q4,n>=200 and>=30 each class,F1 harm<=0.01 and
   positive-recall harm<=0.02 versus either control, in each declared aggregate/
   window. Unsupported groups explicit. Report EF/availability groups as well.
4. Exact missing expert probabilities, finite aligned outputs, source/PIT/
   feature-matrix/receipt/accounting/independent metric checks all PASS.

ADVANCE means this pilot supports a positional mechanism for further review,
not full-six qualification or outperforming A02. FAIL ends this three-third
recipe without bins/seed/C/vocab/block-weight sweeps. Report precision/recall,
confusion matrices,AUC/AP/logloss/Brier as descriptive additional evidence.
Do not select a failed F1 candidate using another metric.

## Execution and handoff

New namespace positional_text_pilot_v1. Preserve preflight and historical work.
Before actual fits, synthetic actual shared-transform/executor->persistence->
completed accounting->verifier->report must pass, including unsorted row cycles,
empty text, unseen tokens/categories, shuffle-preserved bag, finite matrices,
identical shared preprocessing hashes, class order, missing routing, tamper and
resume with zero repeated fits. Tests must run both before and after fit using
temporary ledgers; never hardcode a live historical ledger into synthetic QA.
Pin recipe/input/source/fit-plan before new outcomes. Real resource preflight:
one job,two threads,6GiB RSS ceiling,4GiB initially free;2GiB free reserve while
running. No install,Test/Sample/Kaggle,notebook/teammate changes or push.
Locally commit scoped implementation/tests/reports. Callback Sol with every
arm/view, gates/counterevidence, convergence, actual component/estimator fit
counts, peak memory, independent QA and commit BEFORE final reply, then STOP.
Only genuine new blockers need interim callbacks; routine fixes within unfit
preparation are authorized. Frozen/consumed outputs are never silently changed.
