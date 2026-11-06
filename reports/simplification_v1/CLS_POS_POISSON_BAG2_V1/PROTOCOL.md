# CLS_POS_POISSON_BAG2_V1 — source freeze proposal

## Status and scope

Status: `SOURCE_PREPARATION_ONLY_WAITING_ROOT_PREFIT_APPROVAL`.

This is one fixed, two-origin diagnostic of prediction averaging for the already
frozen positional-gamma plus A02 recipe. It adds exactly four predictor fits:
two Poisson-weighted replicas at each of E67 and E75. The source package, input
hashes, synthetic pipeline, and independent verifier must be committed and
reviewed before any official fit starts. No official fit, validation-label
read, Test inference, CSV creation, Kaggle access, or upload is authorized by
this source-preparation checkpoint.

The experiment is adaptive: POS_BAG_OFFSET, the E67/E75 development windows,
and the global E79 benchmark have already been seen. Those windows are matched
development evidence, not fresh holdouts. The earlier E79 global-strength
benchmark scored 0.83166; the current full POS/A02 public champion scored
0.83225. This candidate does not replace either historical result or the
champion by definition.

## Hypothesis and fixed recipe

The narrow hypothesis is that averaging predictions from two exchangeable
row-bootstrap perturbations of the positional-gamma fit may reduce estimator
variance. It adds no new information source, does not guarantee a gain, and is
distinct from stronger uniform shrinkage.

For each origin, use the exact saved POS_BAG_OFFSET inputs: the saved C2_B
structured-plus-TF-IDF offset (including its intercept), the exact saved
three-third positional feature matrix, and the already fitted uniform-penalty
POS gamma. Fit two additional gamma vectors with fixed NumPy `default_rng`
Poisson(1) row multiplicities using seeds 17 and 137. Normalize each weight
vector to sum to the fit-row count. Keep all prefix rows, including rows with
an all-zero positional block. Only gamma is optimized; there is no new
intercept, preprocessing, vectorizer, scaler, OHE, DF map, A02 fit, or other
learner fit.

For prefix row `i`, fixed offset `z0_i`, positional vector `x_i`, binary label
`y_i`, normalized multiplicity `a_i`, and `C=1`, optimize from zero:

`sum_i a_i * logaddexp(0, -(2*y_i-1)*(z0_i + x_i @ gamma)) + 0.5 * ||gamma||_2^2`.

Use the inherited L-BFGS-B options: `maxiter=2000`, `ftol=1e-12`,
`gtol=1e-6`, `maxls=50`; one attempt per fit ID, no retries. The original
unweighted gamma is reused and must be replayed faithfully before the replicas
are applied.

On present-stream score rows, form
`pPOSbag2 = mean(sigmoid(z0 + Xpos@gamma_original),
sigmoid(z0 + Xpos@gamma_seed17), sigmoid(z0 + Xpos@gamma_seed137))` and
`pCandidate = 0.5*pPOSbag2 + 0.5*pA02`. The full-champion comparator uses
`pChampion = 0.5*sigmoid(z0 + Xpos@gamma_original) + 0.5*pA02` on the exact same
IDs. On missing-stream rows, both policies use `pA02` bit-exactly.

Retain the frozen decision rule: present rows use `p >= 0.50`; missing rows
use `pA02 >= 0.53230976129477`. The latter threshold was selected from earlier
global OOF evidence. It is fixed for this paired E67/E75 development benchmark,
not origin-learned or past-only. A shared missing route does not make Macro F1
bias cancel: its class denominators are nonlinear and the changed present-row
decisions can alter those denominators. The final E79 threshold remains
unchanged.

## Data boundaries and score construction

- E67 fit: official Train rows from cycles 47–67; score rows: cycles 68–79.
- E75 fit: official Train rows from cycles 47–75; score rows: cycles 76–79.
- Fit labels are read only for the exact frozen prefix row positions and IDs,
  after the source pass and fit reservation. Query labels are not opened until
  all four fit receipts and both prediction bundles are durable and independently
  verified.
- The official A02 `.npz` files include a `labels` member. The runner hashes the
  complete file for lineage but reads only its probability, row-position, ID,
  cycle, and missing-mask members; it must never index or deserialize `labels`.
- The primary unique pool is E67 cycles 68–75 plus E75 cycles 76–79: 20,903
  unique rows. E67 H12 and E75 W6 are also reported as fixed secondary views;
  E67 and E75 overlap on cycles 76–79 and are not independent holdouts.
- Report every per-cycle result and every fixed, fit-defined supported subgroup
  used by the saved POS/A02 comparison. Use the pre-existing row-level
  `entry_friction`, stream-availability, and fit-prefix-defined `text_load`
  groups and their frozen supports; do not redefine groups after scores are
  seen. A subgroup is supported only at `n >= 200` and with at least 30 rows
  in each class.

The existing threshold freeze retains the same global missing threshold for
E67/E75. The examined missing-threshold feasibility note says no threshold was
selected there, and no prefix-only E67/E75 missing-threshold state was found in
the saved files checked for this source preparation. Therefore no threshold
sensitivity result is available or manufactured. If a pre-existing,
origin-specific prefix-only state is found before scoring, it may be reported
as one matched sensitivity without refitting or selecting a threshold; absence
does not block this pilot.

## Frozen review gates

Scientific qualification is distinct from exploratory benchmark eligibility.

Scientific qualification requires all of:

- unique-pool Macro F1 gain over the full champion of at least `+0.0005`;
- E75 W6 Macro F1 gain of at least `0`;
- E67 H12 Macro F1 gain no worse than `-0.0005`;
- all inherited supported temporal and subgroup guards from the frozen
  POS_BAG_OFFSET recipe, evaluated against the matched full-champion route;
- exact missing-stream fallback and independent QA PASS.

Exploratory-benchmark eligibility (not a retroactive old-gate PASS and not
automatic promotion) requires:

- either unique-pool gain `> 0`, or E75 W6 gain `>= +0.0005` with unique-pool
  gain `>= -0.0005`;
- for every supported fixed subgroup, Macro F1 harm no worse than `-0.02` and
  positive-recall harm no worse than `-0.03`;
- exact missing-stream fallback and independent QA PASS.

Report negative results and all denominators. No seed, weight, threshold,
blend-weight, or recipe sweep follows from either verdict. No Test or Kaggle
result may be used to choose among the fixed replicas.

## Resource and persistence contract

The four official fit IDs are fixed in the source and consume exactly four
predictor-fit slots; preprocessing fit count is zero. A committed root prefit
approval bound to the freeze is required before `reserve` or `fit`. Before each
fit, require at least 1 GiB available RAM; after fit require at least 512 MiB
available. Use one shared heavy-job lock and at most two threads. A started fit
is consumed even if it fails; no retry or replacement is allowed. Persist each
gamma, receipt, and ledger event atomically. Keep all outputs in the new
`CLS_POS_POISSON_BAG2_V1` namespace and preserve every prior result, receipt,
ledger, and champion artifact.

The fitted source is not ready for official execution until this protocol,
implementation, source/input freeze, synthetic CLI lifecycle, and independent
verification are committed and Sol records one exact prefit approval. The next
authorized work after that review is the four fixed fits, followed by
prediction persistence and matched validation scoring; still no Test inference
or Kaggle submission.
