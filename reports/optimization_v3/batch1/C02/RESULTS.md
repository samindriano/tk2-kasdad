# C02 — Pilot result

**Decision: RETIRE.** The present-only sparse expert did not meet the
predeclared primary Macro F1 gain requirement.

## Frozen experiment

- Candidate: same sparse binary classifier and word TF-IDF settings as the
  accepted sparse25 expert, with predictor and all learned transforms fitted
  only on nonempty `signal_stream` rows. Present rows use
  `0.75 × C2_B + 0.25 × candidate`; missing rows retain frozen CatBoost. The
  fixed classification threshold is 0.50.
- Official Classification Train only; SHA-256:
  `875552a68a3d24b0f3e1966d6ddb5671e6f54305a1c526e2f0a76747942ed91c`.
- Two predictor fits at cutoffs 55 and 67. Present-only support was 11,194 of
  12,020 fit rows at cutoff 55 and 27,676 of 29,705 at cutoff 67; both classes
  were represented in each fit. The missing indicator was constant zero in
  training and the expert was applied only to present-stream score rows.
- Freeze signature:
  `19b60206b6a0cc36b2f5ad04952473d047c729395018880114d33f97a860fe9e`.
- Preflight estimated 1.759 GiB peak against a 6 GiB cap; 9.584 GiB was
  available at preflight. One training job, two threads.

## Temporal comparison

| View | Rows | Candidate Macro F1 | Frozen control | Gain |
|---|---:|---:|---:|---:|
| W1 (cycles 56–59) | 4,114 | 0.801398 | 0.803160 | −0.001762 |
| W4 (cycles 68–71) | 4,950 | 0.841865 | 0.841418 | +0.000447 |
| Primary (W1 + W4) | 9,064 | 0.823439 | 0.823972 | −0.000534 |
| H12 | 38,588 | 0.803055 | 0.802966 | +0.000089 |

The primary requirement was a gain of at least +0.0005. H12 harm, W1/W4 harm,
and all supported subgroup guards passed, but the observed primary gain was
−0.000534. The primary gate alone requires RETIRE.

## Verification and limits

The independent verifier returned `PASS`. The second QA pass confirmed exactly
two fits, present-only row fingerprints, all 19 frozen input hashes and 23
source hashes, ordered score IDs, and exact reconstruction of the C2_B, sparse25,
and missing-stream CatBoost routes. No Test data or Kaggle result was used.
These are reused temporal development outcomes, not untouched holdout estimates.

The sparse prediction path emitted two scikit-learn feature-name `UserWarning`s;
independent probability, row-alignment, and route checks passed. This warning is
recorded without changing the frozen model or predictions.

Machine-readable evidence: `recipe.json`, `pilot.json`, `verification.json`,
`gate.json`, and `attempts.jsonl`; the immutable sources and inputs are in
`experiments/optimization_v3/batch1/C02/freeze.json`. The recipe file is
explicitly marked as a post-fit summary materialized from that freeze and the
verified fit states. Prediction arrays and hash-bound sidecars remain in the
ignored `experiments/runs/` directory.
