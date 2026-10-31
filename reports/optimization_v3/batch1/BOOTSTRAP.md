# V3 batch1 bootstrap and C03 pilot status

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

## C03 readiness checkpoint

The C03 adapter and execution bridge are implemented in the batch1 namespace.
The bridge loads only official Classification Train, checks the frozen Train
hash and the frozen sparse25/C2_B/CatBoost controls, and reconstructs the
present-stream route as `0.75 * C2_B + 0.25 * candidate`; missing-stream rows
retain the existing CatBoost branch. The route is independently reconstructed
again by the verifier.

Before the first fit, the following checks passed:

- `python -m pytest tests/test_optimization_v3_batch1.py -q`: 16 passed.
- `python -m py_compile` passed for the batch1 engine, C03 adapter, dispatcher,
  bridge, and verifier.
- `python scripts/optimization_v3_batch1.py C03 --preflight-only` returned
  `PREFLIGHT_PASS_NO_FREEZE_NO_FIT`; 50,608 official Train rows and the frozen
  SHA matched. Declared views were W1, W4, primary, and H12.
- Independent read-only review passed on exact row identity, source hashes,
  control reconstruction, resource gate, subgroup gate, resume integrity, and
  no-Test boundaries.

The resource estimate was 1.602 GiB with 9.470 GiB available at preflight and a
6 GiB per-job cap. The pilot is fixed at two new predictor fits (cutoffs 55 and
67), with one training job and two threads. The predeclared gate requires a
primary Macro F1 gain of at least 0.0005, H12 harm no worse than 0.001, each
pilot window harm no worse than 0.005, and no supported subgroup guard failure.

At this checkpoint there is still no C03 freeze, official fit, result, Test
read, or new Kaggle call. Do not describe C03 as passed until the pilot has run
and its report has passed independent verification.
