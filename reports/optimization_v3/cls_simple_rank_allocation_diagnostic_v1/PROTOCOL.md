# CLS simple rank-allocation diagnostic V1

## Question

Does the frozen POS/A02 half blend add ranking information over A02, or does it
mainly change the number and identity of positive predictions at the fixed
0.50 decision threshold?

This is a zero-fit, Train-development diagnostic over already-saved E67/E75
vectors. It is not a new validation set, causal analysis, candidate-selection
gate, deployment policy, or Kaggle submission authorization.

## Frozen inputs and routes

Use only the cached A02 policy archive and the cached POS archive named in
`PRE_ACCESS_FREEZE.json`; the audit archive is read through these allowlisted
members only: `E67_ids`, `E67_cycles`, `E67_missing`, `E67_y`, `E67_pA`,
`E67_pP`, and the corresponding six `E75_*` members. The frozen root audit
already checked pA/pP ID alignment and verified pA equals its saved CB fallback
on missing-stream rows. No raw Train file, Test, SampleSubmission, E79, target
reconstruction, or Kaggle endpoint is accessed.

For present rows:

- complex score: `0.5 * pP + 0.5 * pA`, positive at `score >= 0.50`;
- simple score: `pA`, positive at `score >= 0.50`.

For missing-stream rows, every operational full-row comparison uses the same
frozen CB/pA fallback and `score >= 0.53230976129477`. This route is not
changed by matched-count diagnostics.

## Matched-count diagnostics

Primary matching is within each complete `cycle_code` and origin. For each
cycle, derive `K_complex` and `K_simple` from the fixed-threshold predictions,
without labels. Select the top `K_complex` present rows by pA and top `K_simple`
present rows by the complex score. Sort descending by score, then ascending by
string `record_id` to resolve exact score ties. Do not choose K from metrics.

Also report an origin-level sensitivity: sum each fixed-threshold K over the
present rows of that origin block and apply the same stable top-K rule. In the
unique pool, retain separate origin blocks (E67 cycles 68–75; E75 cycles
76–79) before combining metrics. This is a declared sensitivity, not a
result-selected policy.

Report confusion counts, Macro F1, positive/negative precision and recall,
positive counts, selected-ID hashes, overlap and Jaccard for the two threshold
policies and both matched-count policies. Report ROC AUC and average precision
for pA and the complex score on present rows only. Never interpret these as
causal effects.

## Fixed views

- E67 W4: origin 67, cycles 68–71.
- E67 W5: origin 67, cycles 72–75.
- E67 W6: origin 67, cycles 76–79.
- E67 H12: origin 67, cycles 68–79.
- E75 W6: origin 75, cycles 76–79.
- UNIQUE: E67 cycles 68–75 plus E75 cycles 76–79; expected 20,903 unique rows.

For every view, report both operational full rows and present-only rows. E75
late cycles overlap the E67 history view; only the declared UNIQUE pool removes
that overlap. Reused labels make all results adaptive development evidence.

## Novelty review

The bounded review reads at most four existing A02 rank/allocation documents,
listed with hashes in `PRE_ACCESS_FREEZE.json`. The B01 summary has descriptive
matched-positive-count tables, but its comparison is B01/A02/Sparse25. It is
not the POS half-blend-versus-A02 comparison specified here. Record an exact
duplicate only if the same POS/A02 matched-count estimand and recipe is found.

## Resource, reproducibility, and outputs

Build the source and freeze its hashes, all input hashes, view definitions,
thresholds, and tie rule before opening cached vectors. Use the per-lane lock,
at least 2.5 GiB available before reading, at most 512 MiB incremental
estimate, and at least 1 GiB available at boundaries. Do not acquire the shared
heavy-fit lock for this small cached-vector job.

Run `python -m scripts.cls_simple_rank_allocation_diagnostic_v1 --freeze`
before calculation, then commit that freeze. Run `--calculate`, run the
standalone `python -m scripts.verify_cls_simple_rank_allocation_diagnostic_v1`
and require PASS, then run `--plots`. Outputs are exclusive-write and never
overwritten. Keep one compact `SUMMARY.md`, one machine-readable `RESULTS.json`,
one independent QA receipt, and no more than two PNG plots.

No estimator, preprocessing, calibration, or inference fit is permitted. Any
matched-count decision rule is diagnostic-only and must not replace the current
threshold policy. Finish with at most one prospectively defined follow-up
recommendation; do not open a model or threshold search.
