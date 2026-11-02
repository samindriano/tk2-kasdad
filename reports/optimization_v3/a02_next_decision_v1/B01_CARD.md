# B01: residual-centering versus local-label-rate ablation

**Scope:** one fixed adaptive zero-fit diagnostic, no promotion or upload.
Owner: Luna execution; Sol owns review. Official Train and saved A02 outputs only.

## Hypothesis and counterevidence

Matching output counts shows A02 has ranking value beyond allocating more
positives. Confidence-compatible neighbor selection is not supported by the
descriptor check. Before altering weights, test whether the residual target
itself adds value over shrunk local label prevalence.

Counterevidence: Sparse25 already extracts strong text/structured information;
raw neighbor label rates may discard context, drift toward class prevalence,
and underperform. Many earlier probability corrections failed. A negative
result is useful evidence for retaining A02's residual centering.

## One fixed formula

For each query use the original saved top-32 neighbors and weights `w=s^2`.
Let `b` be the exact frozen Sparse25 query probability, `M=sum(w)`, and `y_i`
the historical bank labels, verified directly from Train.

- Original A02: `d=clip(sum(w*(y_i-p_i_OOF))/(4+M), -.05, .05)`.
- B01: `d=clip(sum(w*(y_i-b))/(4+M), -.05, .05)`.
- Output: `clip(b+d,0,1)`, threshold `>=.50`. No neighbor/text/weight changes.
- Missing, zero-neighbor and W1 fallback remain exact Sparse25.

B01 before the correction cap is a local label-rate estimate shrunk toward the
query baseline. No new bandwidth or mixing constant is introduced.

Matched control: permute *bank labels* using the original A02 shuffled model's
within-cycle/probability-decile residual-assignment index. Use the same bank rows,
geometry and formula. Do not add shuffled residual to the unpermuted source
probability to reconstruct labels: that can create nonbinary values. Reconstruct
the original bank labels from official Train, validate them, then index by the
verified assignment. No labels from the query can enter the formula.

## Boundaries and cost

Read the five saved observed/shuffled origin states 59/63/67/71/75. Use only their
declared first-four windows for primary; retain E67 12-cycle output as inherited
H12-L, not fresh confirmation. W1 exact fallback; full primary38,588, new21,033,
late15,953, H12-L20,903. Use original official Train hash and existing ID/cycle/OOF
checks. No refit, transform fit, new nearest-neighbor search, Test/Sample/Kaggle,
or new indexed predictor fit. Estimator/preprocessor fit count must remain zero;
ledger hash unchanged, cap66 still consumed. Stream arrays; no large dense text.

## Predeclare diagnostic criteria before new B01 scores

These govern only whether to return B01 for further review, not qualification:

1. Full-primary F1 gain >=.0005 versus unchanged A02.
2. New-origin aggregate gain >=.0005 versus unchanged A02.
3. Late and inherited H12-L gains versus A02 nonnegative.
4. At least four of five learned primary origins nonnegative versus A02.
5. Full-primary gain versus B01's matched label-shuffle control >=.0002.
6. No supported subgroup harm >.01 F1 or >.03 positive recall versus A02;
   support n>=200 and >=30 per actual class. Report Sparse25 contrasts too.
7. All geometry, source, label-target separation and no-fit checks PASS.

Report all F1/TP/FP/FN, positive counts, AUC/AP/Brier/logloss (both epsilon
conventions), endpoint errors, per-origin and subgroup results even if it fails.
Include matched-positive-count controls versus A02 and Sparse25 as descriptive
checks; do not choose K or thresholds from outcomes. No grid, rescue or formula
change after scores. Never claim confirmatory holdout or causal effect.

## Deliver and return

Own only new `scripts/optimization_v3_a02_b01_*.py`, corresponding meaningful
metric/formula/no-label-leakage tests, reports under
`reports/optimization_v3/a02_b01_centering_v1/`, and ignored derived outputs under
the analogous run directory. Preserve all frozen sources/models/predictions.
Prove the formula with tiny arrays, missing/zero-mass and matched-label indexing
tests; independent metric reconstruction and query-label perturbation test.

One concise summary, statistics, QA, provenance and exact ledger before/after;
local commit owned paths only, no push. Send outcome or real blocker to
KASDAD - TK2 - Sol (local 01a19569-16e0-7253-aa14-57a0a3160e03), then stop.
Both a positive or negative outcome completes the task. No submission or budget
increase is authorized by this card. Human authorized this ongoing next-decision
research; routine fixes can proceed without requesting permission, but no fits.
