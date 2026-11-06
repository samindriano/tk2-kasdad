# E79 source-preparation checkpoint

## Current state

`CLS_POS_POISSON_BAG2_E79_V1` is source-ready only. No official fit has been
started, no cutoff or validation result was opened as a new evaluation, no
official Test inference has run, no candidate CSV exists, and Kaggle was not
contacted. The earlier BAG2 pilot remains unchanged with its original status.

This implementation is a candidate continuation with fixed provenance and
recipe; it is not a qualification decision or champion change. Any future
official execution requires the separate committed root pass bound to the
prefit freeze. That pass must authorize exactly the two listed fit IDs and one
Test application. There is no retry, extra seed, model selection, or public
leaderboard use in this workstream.

## Files

- `src/models/cls_pos_poisson_bag2_e79_v1.py`: fixed IDs, seeds, parity checks,
  threshold policy, artifact identity, and reusable pure helpers.
- `scripts/cls_pos_poisson_bag2_e79_v1.py`: isolated CLI for source freeze,
  guarded two-fit continuation, guarded one-shot application, and synthetic
  lifecycle QA.
- `scripts/verify_cls_pos_poisson_bag2_e79_v1.py`: independent receipt,
  artifact, blend, threshold, CSV, ordered-ID, and baseline replay verification.
- `tests/test_cls_pos_poisson_bag2_e79_v1.py`: synthetic CLI, resume, tamper,
  fixed-recipe, and optimized-Python safety tests.
- `reports/simplification_v1/CLS_POS_POISSON_BAG2_E79_V1/PROTOCOL.md`: exact
  recipe and execution/data boundaries.

The final source commit and source-freeze commit are recorded in Git and in
`PREFIT_FREEZE.json`. The freeze records the pinned official repository HEAD,
old gamma source and data hashes, exact Train identity, and expected A02/Test
application-chain hashes without opening Test or A02 pA values.

## Synthetic QA expected

The `synthetic` command uses deterministic local arrays only. It executes two
small fits, persists and independently reopens the gamma artifacts and receipts,
constructs synthetic present/missing-stream probabilities, checks exact
missing-stream fallback and threshold decisions, creates a synthetic baseline
and candidate CSV, performs independent CSV/ID/baseline verification, and then
resumes without creating a third fit or second application. It also verifies
that a modified gamma artifact or submission CSV fails closed. The same CLI
path is run under `python -O` so runtime safety checks cannot depend on Python
assert statements.

## Next gated action

After review, the root pass must be written, committed, and bound to the exact
`PREFIT_FREEZE.json`. Then run `verify-freeze`, `reserve`, and `fit-all` in
sequence. Verify both persisted fit receipts independently before the one
authorized `apply-test` call. Stop after the CSV and application manifest pass
readback. Do not upload to Kaggle from this script; Kaggle access is explicitly
outside this candidate protocol.

