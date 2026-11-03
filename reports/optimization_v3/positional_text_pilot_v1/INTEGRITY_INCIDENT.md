# Integrity incident — synthetic ledger isolation

## Event

On 2026-11-03, during an unfit synthetic pilot test, the test called the
preparation runner's ledger append function without a path. The function's
default argument had captured the official shared `budget.jsonl` path when the
module was imported. As a result, one test-only
`BATCH_ALLOWANCE_AUTHORIZED` event with `batch_id=synthetic-test-only` and a
zero-fit cap was appended to the live ledger at
`2026-11-03T05:39:45.747949+00:00` UTC.

This was an unauthorized mutation of the append-only ledger. No positional
`FIT_STARTED`, `FIT_SAVED`, `FIT_FAILED`, cap-amendment, or reservation event
was appended, and no official model fit or official Train target read occurred.

## Remediation and current evidence

The accidental test event was then removed by truncating that final line. This
also violated append-only discipline; restoring the prior bytes does not erase
the fact that the ledger was both appended to and then rewritten. The SHA-256
was observed as `c0c6686d99068cececd07a41087a0f387783b2b20f7c6861629b26b34be5fd24`
after the accidental append and returned to the previously verified
`adba614b9b40b9e0bcee2d41c800cb1bf48edacc410ff4ff4fe7639392dfcbcb` after
removal. Later synthetic runs verified that same prior SHA before and after
their execution. This documents observed content identity; it does not claim
an uninterrupted append-only history.

The synthetic harness now uses a dedicated temporary-ledger reader/writer
whose paths must resolve beneath its `TemporaryDirectory`. The production
ledger reader/writer require an explicit keyword-only path, and the production
writer is monkeypatched to raise throughout synthetic execution. Tests verify
that omitted paths fail before opening, an explicit production path is rejected
in synthetic context, and a failure-injection path restores original globals
and confirms the live-ledger hash in `finally`.

## Status and gate

The shared ledger was not changed again after the restoration. No official
preflight, cap amendment, reservation, or fit has been run. Sol has been
notified of the append and removal, the restored hash, and the corrective
isolation tests. Official pilot work remains on hold until Sol reviews the
committed repair packet and explicitly releases this gate. This record is kept
with the pilot deliverables so the incident is not hidden by the restored hash.
