# Fixed-p50 hinge residual discovery R1

## Scope

This is a zero-fit descriptive audit of the already evaluated `HINGE_D1` and
`HINGE_D2` cached fit-input and score-row bundles. It does not read raw Train,
Test, SampleSubmission, E79, or Kaggle data. It does not fit or refit a model,
learn a new mapping, alter a threshold, or rescore a newly proposed policy.
Existing saved GLOBAL and HINGE coefficients are applied only to the cached fit
populations where their fitted-row view is requested; saved predictions are
used for score populations and independently reconstructed from those existing
coefficients for verification. The cached `causal_global_all_rows` score array
is inventoried in the source NPZ but is deliberately not loaded, evaluated, or
compared; this audit does not compute counterfactual rescoring.

The novelty scan is bounded to six existing summaries: R4 decision, R4
protocol, the 45-card research plan, the research lanes, R-NBR results, and the
ResearchNext campaign summary. The 45-card plan is treated as proposals, not
executed results. The scan found no clearly novel mechanism and recommends
none pending these descriptive outputs.

## Frozen descriptive rules

- Analyze D1 fit (4,950 rows), D1 score (7,462), D2 fit (12,412), and D2 score
  (8,491) separately. Do not pool away opposite signs or history differences.
- Use raw DIRECT bins `[0,20)`, `[20,40)`, `[40,60)`, `[60,80)`, and `[80,100]`;
  100 belongs to the final bin. Values outside `[0,100]` fail closed.
- Minimum support is 200 present rows per raw bin and 50 present rows per cycle.
  Unsupported cells are `UNKNOWN_UNSUPPORTED`, with null metrics; never zero-fill.
- For each population, bin, and cycle, report all/present/missing counts;
  medians of `y - raw`, `y - GLOBAL-present`, and `y - HINGE-present`; existing
  raw, GLOBAL-present, and HINGE-present MAEs; and HINGE gain over GLOBAL on both
  all-row and present-only denominators. Report target `<=5` and `>=95` support
  as descriptive tail counts only.
- HINGE fit populations include every cached calibration row in the original
  L1 objective. Application of HINGE and GLOBAL mappings is present-only; the
  missing route preserves raw DIRECT bits. Display both denominators.
- Existing raw, GLOBAL-present-only, and HINGE-present-only D1/D2 score
  summaries are reconstructed against the R4 diagnostics receipt. The
  counterfactual GLOBAL-all-rows score record is excluded. No new holdout claim
  follows from fit-population metrics.
- Use a fresh memory gate while holding the shared OS lock: initial available
  memory at least 1.5 GiB, incremental estimate 0.5 GiB, projected and actual
  reserve at least 0.75 GiB, one heavy job, and at most two threads.

## Interpretation boundary

Residual medians describe the cached rows and do not establish that a different
curve will generalize. Keep p90 and upper-target harm visible. D1 and D2 differ
in both temporal horizon and prediction origin. Development labels and the
public score have been selected repeatedly; no public-score uplift or fresh
holdout claim is allowed. At most one future mechanism may be recommended, and
only as a proposal for prospective evaluation under a new authorization.
