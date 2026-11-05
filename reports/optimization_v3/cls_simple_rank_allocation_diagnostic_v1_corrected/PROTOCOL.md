# CLS Simple Rank-Allocation Diagnostic — Corrected Replay

## Scope and lineage

This is a narrow implementation correction to
`../cls_simple_rank_allocation_diagnostic_v1/PROTOCOL.md` (frozen SHA-256
`ce2b6f76ca82a89032ceaf1e6578c96eb2d433be082f9bde8bff357ee15c6284`). It
preserves that diagnostic's question, inputs, views, thresholds, K derivation,
tie rule, estimands, and exclusions. The first replay is archived with
`SUPERSEDED.md`; its matched-policy full-row metrics are invalid because its
missing-stream fallback was omitted.

The corrected output namespace is
`reports/optimization_v3/cls_simple_rank_allocation_diagnostic_v1_corrected/`.
Use the same frozen A02 policy archive, cached POS probabilities, and four
novelty documents as the original protocol. The root audit's pinned source
commit must remain an ancestor; current HEAD may advance only if the checkout
is clean and every pinned input hash still matches.

## Correction and invariants

Each of the four matched-count policies starts with the exact frozen missing
route, `pA >= 0.53230976129477` on stream-missing rows. Top-K changes only
stream-present predictions and derives K only from fixed-threshold predictions
within the declared cycle/origin blocks. Before any metrics are computed, the
producer and independent verifier must assert identical missing-row decision
bits for all six policies. For every full-row policy, predicted positives must
equal present predicted positives plus the fixed number of missing-route
positives; TP/FN, FP/TN, and TP/FP counts must close.

Before the freeze or cached-array access, run the producer's hand-written
`--self-check`. It must cover missing positive and negative labels, K=0, K=n,
equal-score ties resolved by ascending `record_id`, and disjoint origin blocks.
The standalone verifier must independently implement and pass its own fixture;
it may not import or share the producer's initialization helper.

## Resources, QA, and outputs

Use the single-job gate: both the shared
`kasdad_single_heavy_job.lock` and per-lane lock, at least 1.5 GiB available at
start, no more than 512 MiB estimated incremental use, and at least 0.75 GiB at
the boundary. Both calculation and independent verification acquire these
locks before opening arrays. Fit no estimator and access no raw Train labels,
Test, SampleSubmission, E79, or Kaggle.

Run `--self-check`, commit source/protocol, run `--freeze`, commit the freeze,
then run `--calculate`, standalone verifier, and `--plots` only after verifier
PASS. Outputs are exclusive-write: one compact summary, `RESULTS.json`, an
independent QA receipt, and no more than two PNGs. Do not overwrite the
superseded replay.
