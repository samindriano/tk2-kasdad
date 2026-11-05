# E79 application policy R2

R2 supersedes the metadata-only R1 source attempt. R1's independent replay
failed before opening E79 because its verifier did not reproduce the runner's
inventory envelope. R2 pins the R1 receipts and reason, then replays the full
inventory independently. No E79 access or production CSV occurred in R1.

## Scope

This lane applies the already fitted local scalar `delta` to the frozen FULLREG
hinge candidate for local submission-file creation. It performs no model or
calibrator fitting. The source freeze and preflight inherit the E79 archive
path, SHA256, row contract, and ordered-ID hash from signed parent application
receipts. They must not open or hash the live E79 archive.

## Before an application root pass

Allowed actions are source and receipt hashing, metadata-only lineage checks,
source freeze creation, preflight creation, and independent source verification.
The runner must leave all of these false: E79 archive opened, live E79 hash
rechecked, test/sample read, CSV created, and Kaggle upload. A separately signed
application pass is required before reading E79 prediction members or creating
the production CSV.

## After an exact application root pass

The pass must bind the freeze, preflight, source inventory, fit state, numerical
QA, inherited E79 archive SHA256, and exact output path. It authorizes only
allowlisted E79 prediction/identity members and local CSV creation/verification.
It denies target values, test/sample access, model/calibrator fits, and upload.
The runner rechecks the archive SHA256 after approval; rejects target-like member
names; loads only `record_id`, `prediction`, `baseline`, and `missing`; checks
the inherited row, ID, and missing-route contract; and does not persist an
intermediate prediction array.

The formula is `clip(g(p) + delta * b(p), 0, 100)` for present rows, where `g`
uses the frozen hinge coefficients and `b(p) = max(0, 1 - abs(p - 50)/10)`.
For missing rows, the original raw DIRECT float64 bits must be preserved exactly.
The CSV must use the frozen row order, unique `record_id`, header
`record_id,threat_rank`, finite values in `[0, 100]`, and the parent-pinned
ordered-ID SHA256. Creation is exclusive and deterministic; an identical output
may be resumed, while a conflicting output is rejected. No upload is part of
this lane.

## Validation

Source preparation is checked by an independently implemented metadata/receipt
replay. CSV behavior is covered with synthetic fixtures only: formula and
clipping, missing float bits, row order, header, range, duplicates, tampering,
and idempotent/conflicting writes. The synthetic suite must not access E79 or
create output outside a temporary directory.
