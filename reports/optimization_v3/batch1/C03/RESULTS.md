# C03 — Pilot result

**Decision: RETIRE.** The split unigram/bigram vocabulary did not meet the
predeclared primary gain gate, so it will not receive continuation or full-six
evaluation.

## Frozen experiment

- Candidate: structured features plus separate word TF-IDF blocks (up to
  30,000 unigrams and 10,000 bigrams), routed as `0.75 × C2_B + 0.25 × candidate`
  when `signal_stream` is present; the existing CatBoost route is unchanged for
  missing streams.
- Official Classification Train only; SHA-256:
  `875552a68a3d24b0f3e1966d6ddb5671e6f54305a1c526e2f0a76747942ed91c`.
- Two new predictor fits, at cutoffs 55 and 67. Fit rows were 12,020 and 29,705;
  fit durations were 6.08 s and 15.21 s. The card's expected fit count matched.
- Freeze signature:
  `72844356c2e8f7696d65e85245c49a419e8e7afc3ad4ca90f2cfbeff0df0b614`.
- Preflight estimated 1.602 GiB peak versus a 6 GiB job cap; 9.804 GiB was
  available at the actual preflight. One training job, two threads.

## Temporal comparison

| View | Rows | Candidate Macro F1 | Frozen baseline | Gain |
|---|---:|---:|---:|---:|
| W1 (cycles 56–59) | 4,114 | 0.801598 | 0.803160 | −0.001562 |
| W4 (cycles 68–71) | 4,950 | 0.842121 | 0.841418 | +0.000703 |
| Primary (W1 + W4) | 9,064 | 0.823653 | 0.823972 | −0.000319 |
| H12 | 38,588 | 0.802494 | 0.802966 | −0.000471 |

The candidate improved W4 slightly but declined on W1, the pooled primary
comparison, and H12. The predeclared primary requirement was a gain of at least
0.0005; the pilot observed −0.000319. H12 harm stayed within 0.001, each pilot
window stayed within the 0.005 harm limit, and all supported subgroup guards
passed. These checks do not override the failed primary gate.

## Verification and limits

The independent verifier returned `PASS`, recomputing the metrics and exact
ordered row identities from Train labels and saved score positions. The route
reconstruction matched the frozen baseline policy. No Test data, Kaggle score,
or leaderboard result was used. These are reused temporal development outcomes,
not an untouched holdout estimate.

Machine-readable evidence: `pilot.json`, `verification.json`, `gate.json`, and
`attempts.jsonl`; the immutable recipe, source and input hashes are in
`experiments/optimization_v3/batch1/C03/freeze.json`. Prediction arrays and
hash-bound sidecars remain in the ignored `experiments/runs/` directory.
