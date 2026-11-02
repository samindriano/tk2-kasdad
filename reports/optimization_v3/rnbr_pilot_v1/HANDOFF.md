# R-NBR B0 preparation handoff

**Status:** `READY_FOR_FOUR_FIT_BUDGET_REVIEW`. Target-free preflight passed,
the synthetic lifecycle and independent verifier passed, and all 15 focused
tests passed. This is not a pilot result or a model qualification.

## Completed in B0

- Frozen the two-origin R06 residual-neighbor diagnostic and shuffled control
  in `PROTOCOL.md`.
- Added pure, dataset-free correction, weighted-median, fixed-decile shuffle,
  row/ID/cycle validation, donor lineage validation, fit-plan, and incremental
  diagnostic-gate primitives.
- Added a hash-checked synthetic bundle lifecycle and an independently written
  verifier. The lifecycle covers durable save, injected interruption, verified
  resume, post-prediction truth access, independent prediction/metric/gate
  recomputation, and fail-closed corruption handling.
- Added a target-free official-source preflight. It checks target-free Train
  row alignment, R06 lineage and saved cache identities, A02 neighbor geometry
  for E67/E75, pilot window coverage and support, ledger accounting, and that
  the old readiness inventory is unchanged. Primary is E67 W4 + E75 W6,
  n=13,441. H12_L is all E67 cycles 68–79, n=20,903; no E75 data enters that
  view. Every present W4/W6 row has the full 32 saved neighbors.
- Drafted a cap-70 request for these four task fits only. Nothing was reserved.

## Historical context kept separate

The first B0 checkpoint at `dbaa3bb` had a view-provenance error: its H12_L
summary mixed E67 W4 and E75 W6 and held W5 at R06. It is superseded by this
successor; the prior commit and report hashes are retained in Git history and
`H12_SEMANTICS_CORRECTION.md`.

R06 remains unqualified under its original legacy group guard: the archived W1
missing-stream group has n=392 and harm 0.395556 versus legacy, above 0.25.
Its W6 harm is 0.013929. The R-NBR pilot would compare incremental effects
against R06; it cannot rewrite or rescue that historical verdict. R22 and
legacy cached comparisons are documented in `PROTOCOL.md` as historical
context, not recomputed pilot evidence.

## Current limits and next authorized step

- No official R-NBR target memory, task fit, candidate prediction, metric, Test
  access, full-Train refit, Kaggle call, or ledger mutation has occurred.
- Current fit cap is exhausted: 66 starts of 66. Ledger SHA must remain
  `6fcc4b4d427e3ff03139e0e9ee429f529f7c7bf6ad91ce0eb9a52d55e2503147` until a
  separately authorized cap amendment.
- The only next step is review of the exact four-fit budget request. A later
  six-fit continuation and one conditional full-Train fit are outside this
  request.
- After any later pilot authorization, report primary, H12_L, W4, W6,
  supported subgroups, shuffled-control gains, cached R22/legacy comparisons,
  and every negative result. Keep R06's legacy guard failures in a separate
  table. Do not auto-select continuation or a champion from a diagnostic pass.
