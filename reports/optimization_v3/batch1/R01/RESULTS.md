# R01 — Pilot result

**Decision: RETIRE.** The candidate passed its aggregate pilot MAE gates but
failed the predeclared supported high-target subgroup guard in W4, so it will
not receive continuation or full-six evaluation.

## Frozen experiment

- Candidate: V2 sparse hybrid transforms and LinearSVR with
  `epsilon=0`, `loss=epsilon_insensitive`, `dual=True`, `C=1`, `tol=1e-4`,
  `max_iter=20,000`, seed 42; target `z=(threat_rank-50)/50`, inverse
  `50*z+50`.
- Policy: `0.75 × frozen three-component champion + 0.25 × candidate rank50`,
  clipped to `[0,100]`. This is an auxiliary blend with the full incumbent,
  not a replacement of one incumbent component. SGD regularization equivalence
  is not claimed.
- Official Regression Train only; SHA-256:
  `756523ff8156a9cd776e52249f7e2a8f23dcec432b8d10570e55ff6db0e0f082`.
- Two predictor fits at cutoffs 55 and 67, with 12,020 and 29,705 fit rows.
  Fit times were 28.41 s and 100.96 s. LinearSVR converged without warnings:
  14,437 and 14,949 iterations, below the 20,000 limit.
- Freeze signature:
  `2f034eef306e7e2f9d40def5ff533640e5d4e3e0c6b9f08e55305e68365529d4`.
- Preflight estimated 1.602 GiB peak against a 6 GiB cap; 9.640 GiB was
  available at preflight. One training job, two threads.

## Temporal comparison

| View | Rows | Candidate MAE | Frozen control | MAE gain |
|---|---:|---:|---:|---:|
| W1 (cycles 56–59) | 4,114 | 15.689997 | 15.882112 | +0.192115 |
| W4 (cycles 68–71) | 4,950 | 11.550695 | 11.455005 | −0.095690 |
| Primary (W1 + W4) | 9,064 | 13.429456 | 13.464396 | +0.034940 |
| H12 | 38,588 | 13.971095 | 14.047533 | +0.076438 |

The pilot primary gain exceeded the predeclared `0.03` requirement, H12
improved, and both individual windows stayed within their `0.5` harm allowance.
However, the supported W4 high-target tail (`threat_rank >= 95`, n=249) had MAE
7.814041 versus 7.354028 for the control: **0.460013 MAE harm**, above the
predeclared 0.25 subgroup limit. This is the sole failed gate and determines
RETIRE.

## Verification and limits

The saved independent verifier returned `PASS`, reconstructing metrics and
predictions from the frozen control components and row-aligned Train cache.
The tail subgroup is defined after predictions from observed Train targets; it
is diagnostic, not a feature. A queue-declared `entry_friction` diagnostic was
also computed after the fit because the frozen R01 gate only covered missing
stream and target-tail groups. All four groups were supported; none exceeded
the 0.25 harm limit. This supplemental diagnostic does not alter the frozen
gate or RETIRE decision.

| View | `entry_friction` | Rows | Candidate MAE | Frozen control | Candidate minus control |
|---|---:|---:|---:|---:|---:|
| Primary | 0 | 6,131 | 14.823030 | 14.875833 | −0.052803 |
| Primary | 1 | 2,933 | 10.516398 | 10.513997 | +0.002401 |
| H12 | 0 | 26,048 | 15.487911 | 15.646196 | −0.158286 |
| H12 | 1 | 12,540 | 10.820375 | 10.726800 | +0.093575 |

The entry_friction check is reproducible with
`python scripts/optimization_v3_batch1_r01_supplemental.py` and is recorded in
`entry_friction_diagnostics.json`. No Test data or Kaggle result was used.
These are reused temporal development outcomes, not untouched holdout estimates.

Machine-readable evidence: `recipe.json`, `pilot.json`, `verification.json`,
`gate.json`, `entry_friction_diagnostics.json`, and `attempts.jsonl`; immutable inputs and sources are in
`experiments/optimization_v3/batch1/R01/freeze.json`. Prediction arrays and
hash-bound sidecars remain in the ignored `experiments/runs/` directory.
