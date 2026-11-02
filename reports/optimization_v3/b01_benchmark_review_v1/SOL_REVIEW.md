# Sol review: prepare one B01 benchmark, preserve failed qualification

**Decision:** ACCEPT_B01_DIAGNOSTIC; APPROVE_ZERO_FIT_BENCHMARK_PREPARATION_ONLY.
The diagnostic remains DIAGNOSTIC_GATE_FAILED / NOT_QUALIFIED. No gate, prior
prediction, champion or public receipt is changed. Kaggle upload is not authorized
by this preparation decision; return the concrete CSV/manifest for final review.

## Evidence checked directly by Sol

- Read the frozen B01 formula implementation and gate/report outputs.
- Re-ran the focused B01 suite: 10 passed.
- Independently checked 200 actual saved-neighbor queries (40 from each of five
  learned origins), using the equivalent convex expression
  `(4*b + sum(w*y_bank))/(4+sum(w))`, bounded back to `b +/- .05`.
  All agree with the implementation within 1e-15. Source model hashes matched
  fit receipts. Historical Train labels, not query targets, enter the formula.
- Inspected source label permutation: binary bank labels are permuted using the
  verified original assignment; shuffled residuals are not misused as labels.
- Read the QA checks/immutable input and ledger attestations. This review did not
  independently replay every neighbor search or every result field.

## Why preparation is justified despite gate failure

| View | B01 gain versus A02 |
|---|---:|
| Full primary | +0.000261 |
| New-origin aggregate | +0.000519 |
| Late | +0.001762 |
| Inherited H12-L | +0.003890 |

The full-primary hurdle (+.0005) and the required four of five nonnegative origins
fail (three of five improve). These are material counterevidence; retain both
failures. The probability-quality improvement and late evidence nevertheless
justify preparing one low-cost benchmark candidate rather than more fits or a
parameter sweep. This is an explicit post-diagnostic benchmark exception, not a
reinterpretation of a failed scientific qualification as PASS.

The pre-cap local-label estimate is a convex combination of the query baseline
and bank labels. For strictly interior baseline probabilities, positive prior
weight prevents exact 0/1 endpoints; the correction cap keeps the output between
the prior and the uncapped local estimate. This explains a structural probability
advantage without establishing a Macro F1 or future-data advantage. C19's failed
public transfer remains a warning against trusting internal late gains alone.

## Boundaries for Luna preparation

Reuse the already fitted A02 origin-79 state, its exact stored top-32 geometry
and verified full-Train Sparse25 control probabilities. Derive B01's single frozen
observed formula. No new model/index/preprocessor fit, neighbor-policy change,
threshold/calibration search, new target, or budget amendment. Budget remains66.
If geometry needs replay, only unchanged fitted transforms/search on official
Test features is allowed; Test targets are never loaded. Reconstruct the original
A02 prediction hash before deriving B01. A missing dependency is not permission
to refit.

One CSV and source/input/model/geometry/formula manifest; independently verify
ordered Test IDs/schema, 29,421 integer labels, missing-stream equality, formula,
finite probabilities, saved-state replay and exact zero-fit ledger hash. Compare
labels/probabilities against the already uploaded A02, including transition
counts. If labels are identical, stop and do not propose another upload.

No Kaggle/UI access, upload, final-selection change, push, or academic notebook
change during preparation. Preserve all old sources/outputs/failed gates.
Send the completed candidate or real blocker to Sol for a final one-benchmark
decision. All results remain adaptive and do not predict a public score.
