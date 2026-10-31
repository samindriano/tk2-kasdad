# V3 batch1 execution status

The shared runner and transaction primitives are introduced in
`src/models/optimization_v3_batch1.py`. Each card adapter exposes a
`fit_predict(fit_features, fit_target, score_features, card_id, cutoff)` function.
The adapter receives no score-period labels. A separate control loader returns
row-aligned predictions; the adapter returns its final policy prediction and
raw candidate score. Per-origin predictions are stored in exclusive NPZ plus
hash-bound JSON sidecars under the ignored
`experiments/runs/optimization_v3_batch1/` namespace. Existing partial or
mismatched output pairs fail closed and are preserved.

Before any official fit, each card needs an immutable `freeze.json` containing
the recipe, exact feature allowlist, source/input hashes, dependency versions,
and evaluation views. The runner records attempts in append-only JSONL and
resumes only when freeze, row positions, and hashes match. Predictions are
scored only after fitting; `scripts/optimization_v3_batch1_verify.py`
recomputes the declared metrics independently from raw Train labels and saved
row indices.

## C03 pilot

The C03 adapter and execution bridge are implemented in the batch1 namespace.
The bridge loads only official Classification Train, checks the frozen Train
hash and the frozen sparse25/C2_B/CatBoost controls, and reconstructs the
present-stream route as `0.75 * C2_B + 0.25 * candidate`; missing-stream rows
retain the existing CatBoost branch. The route is independently reconstructed
again by the verifier.

Before the fit, the following checks passed:

- `python -m pytest tests/test_optimization_v3_batch1.py -q`: 16 passed.
- `python -m py_compile` passed for the batch1 engine, C03 adapter, dispatcher,
  bridge, and verifier.
- `python scripts/optimization_v3_batch1.py C03 --preflight-only` returned
  `PREFLIGHT_PASS_NO_FREEZE_NO_FIT`; 50,608 official Train rows and the frozen
  SHA matched. Declared views were W1, W4, primary, and H12.
- Independent read-only review passed on exact row identity, source hashes,
  control reconstruction, resource gate, subgroup gate, resume integrity, and
  no-Test boundaries.

The resource estimate was 1.602 GiB with 9.804 GiB available at preflight and a
6 GiB per-job cap. Exactly two new predictor fits ran (cutoffs 55 and 67), with
one training job and two threads. Both fit receipts were saved and the fit count
matches the card.

C03 is **RETIRE** under its predeclared gate. Primary Macro F1 gain was
−0.000319 against a required +0.0005; H12 gain was −0.000471 against a maximum
allowed harm of 0.001. W1 was −0.001562 and W4 was +0.000703. All supported
subgroup guards passed. Independent metric and row verification returned
`PASS`. Full denominators and supporting artifacts are in
[`C03/RESULTS.md`](C03/RESULTS.md), `C03/pilot.json`, `C03/verification.json`,
and `C03/gate.json`.

No Test data or Kaggle result was used. The frozen C03 recipe is retired and
will not enter continuation or full-six evaluation.

## R01 pilot

R01 is also **RETIRE**. The two LinearSVR fits converged without warnings
(14,437 and 14,949 iterations). Primary MAE gain was +0.034940 against a +0.03
minimum, H12 improved by +0.076438, and the W1/W4 window harms stayed within
their limits. The supported W4 high-target tail (`threat_rank >= 95`, n=249)
had +0.460013 MAE harm, above the frozen +0.25 guard. This was the sole failed
gate. Independent QA verified both fits, row IDs, controls, hashes, metrics and
the RETIRE decision.

A supplemental, post-fit entry_friction diagnostic is saved separately. It is
descriptive and does not change the frozen R01 gate. R01 will not enter
continuation or full-six evaluation. See [`R01/RESULTS.md`](R01/RESULTS.md).
No Test or Kaggle result was used.
