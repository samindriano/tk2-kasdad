# Budget request — R-NBR four-fit pilot

**Request only. No amendment or reservation has been written to the ledger.**

The current Research Next ledger is frozen at SHA-256
`6fcc4b4d427e3ff03139e0e9ee429f529f7c7bf6ad91ce0eb9a52d55e2503147`:

- effective global cap: 66;
- starts: 66; saved: 64; failed: 2; pending: 0;
- remaining capacity: 0;
- effective category caps: pilot 34, continuation 31, control replay 0,
  technical retry 1 (sum 66).

Request one new, idempotent cap amendment adding **exactly four pilot slots**:

| Scope | Current cap | Requested cap | Change |
|---|---:|---:|---:|
| Global | 66 | 70 | +4 |
| Pilot | 34 | 38 | +4 |
| Continuation | 31 | 31 | 0 |
| Control replay | 0 | 0 | 0 |
| Technical retry | 1 | 1 | 0 |

Fit IDs, if approved and separately reserved: `RNBR_PILOT_E67_OBSERVED`,
`RNBR_PILOT_E67_SHUFFLED`, `RNBR_PILOT_E75_OBSERVED`, and
`RNBR_PILOT_E75_SHUFFLED`. Each is one task-estimator fit. No retry is
requested. Reservation must be a distinct amendment/reservation ID and must
not rewrite or release prior ledger events.

This request excludes the six later continuation fits and one conditional
full-Train fit. Those seven fits remain unauthorized and must not be booked or
started automatically if the four-fit pilot is approved. The existing B0
scripts only produce a request schema; they do not amend or reserve the
ledger.

Before applying any future approved amendment, recheck the live ledger hash,
current cap/accounting, and pending starts. If its state differs from the
checkpoint above, stop and prepare a fresh exact request instead of assuming
the numbers remain valid.
