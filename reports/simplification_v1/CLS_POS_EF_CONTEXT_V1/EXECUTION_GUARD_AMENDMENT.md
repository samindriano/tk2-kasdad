# CLS_POS_EF_CONTEXT_V1 — execution guard amendment

## Purpose and scope

This amendment repairs one execution-control defect discovered after the first
reserved fit. It does not change the frozen model, input data, target boundary,
preprocessing, feature construction, objective, optimizer, thresholds, seeds,
fit IDs, or fit cap. The original `FREEZE.json`, source commit, source hashes,
reservation, and E67 REAL receipt remain intact.

The original guard treated every historical `FIT_STARTED` event as an active
fit. After E67 REAL had a matching `FIT_SAVED` terminal event, this stale check
still blocked E67 SHUFFLED before a new `FIT_STARTED` event was written. The
second command therefore did not start an estimator and consumed no fit slot.

The amended runner imports and calls the original frozen `fit_one` implementation.
It overrides only that exact stale guard message. Before allowing it through,
the adapter re-runs the frozen ledger verifier with no pending fit allowed and
independently confirms `pending = started IDs - saved/failed terminal IDs` is
empty. It does not filter, rewrite, or hide ledger events. The original driver
bytes are still verified against the committed freeze before any official fit.

The first amended-runner invocation then encountered a separate module-loader
issue before `FIT_STARTED`: the parent TP2 checkout had already loaded its
`src.models` package, so the audit-only frozen core was not visible as a package
member. The attempt is recorded in the execution-lineage sidecar with
`fit_started=false`; the official ledger confirms that no E67 SHUFFLED fit was
consumed. The next committed adapter version resolves the audit core by its
exact frozen file path, checks that file against the source hash in `FREEZE.json`,
and registers only that module under its canonical name. It does not clear or
replace the parent package tree. Its retry reconciliation records the official
ledger prefix hash, confirms no start/terminal event existed for E67 SHUFFLED,
and explicitly allows adapter-plumbing resumption while keeping optimizer retry
disabled. A synthetic preloaded-package test exercises the importlib driver
path and confirms two sequential optimizer calls against the corrected core.

## Execution lineage

- E67 REAL completed under the original frozen driver at source commit
  `21001878edbf7685c23453abb8897ecb999cfdfd`.
- E67 SHUFFLED, E75 REAL, and E75 SHUFFLED are the only fits authorized to run
  through the amended adapter. They retain the original freeze signature and
  recipe. The adapter's commit and SHA-256 are recorded per fit in the ignored
  `experiments/CLS_POS_EF_CONTEXT_V1/execution_lineage.jsonl` sidecar.
- The independent verifier reconciles all four durable fit receipts and both
  execution lineages without rewriting the original fit ledger or receipts. It
  allows a pre-fit abort only when its sidecar proves there was no official fit
  start/terminal event; any failed or interrupted started fit remains consumed
  and fails verification.
- The reservation remains exactly four starts total: one already saved and
  three unstarted at amendment creation. No new cap, retry, or fit ID is added.

This amendment was created after the E67 REAL optimizer diagnostics were
visible, so it is **not outcome-blind**. No query predictions/labels, task
metrics, Test data, or SampleSubmission were accessed when diagnosing the
guard. The change responds to a deterministic ledger-control error, not to a
model result.

## Checks before resuming

The amendment-specific tests use a temporary durable ledger and verify:

1. Two sequential fit-entry calls succeed when the first has a saved terminal.
2. An unfinished fit blocks the next fit.
3. A failed fit remains consumed but is not treated as active.
4. Duplicate start events are rejected.

Result before resuming official fits: **6 passed**. The amended runner and
verifier are committed and hash-bound by `EXECUTION_GUARD_AMENDMENT.json` before
the first amended fit starts. Do not call the original driver for the remaining
three fits; use the amended runner in frozen order.
