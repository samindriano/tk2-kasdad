# E79 application policy R4

R4 is a source-only repair after the approved R3 application attempt failed in
the pin lookup before the NPZ container was opened. The R3 root pass and a
signed, root-reported failure record are preserved and bound by this freeze.
R4 changes the lookup to `expected_sha256_from_parent_freeze`; it does not waive
the archive hash check or change the candidate math, fit state, row order,
missing behavior, or authorization boundary.

## Source-only preparation

R4 source freeze, preflight, and independent verification use metadata and
source bytes only. They do not open or re-hash the live E79 archive, read
Test/Sample values, create a production CSV, or access Kaggle. The prior R3
failure record distinguishes the root-reported facts: archive bytes were hashed
before the missing-key exception, the comparison did not complete, and the NPZ
container/prediction arrays/output files were not opened or created.

## Application behavior after a new R4 pass

R4 requires a new exact application root pass that binds its own freeze,
preflight, source inventory, fit state, numerical QA, inherited E79 archive
SHA256, and R4 output path. The old R3 pass cannot authorize R4. After the R4
pass, the runner compares the live archive hash to
`expected_sha256_from_parent_freeze`, checks archive members, loads only
`record_id`, `prediction`, `baseline`, and `missing`, validates the inherited
row/ID/missing contract, and writes `fullreg_center_local_r4.csv` locally.
Targets, test/sample values, fits, and Kaggle upload remain denied.

The formula remains `clip(g(p) + delta*b(p), 0, 100)` for present rows, with the
frozen hinge coefficients and R3 delta. Missing raw DIRECT float64 bits remain
unchanged. The output remains the same 29,421 IDs in frozen order, header
`record_id,threat_rank`, and finite values in `[0,100]`.

## Full synthetic application integration

Synthetic QA creates a complete isolated producer/approval/archive fixture in a
temporary directory. It exercises signed freeze/preflight/pass validation,
archive hash comparison, allowlist checks, archive member reads, CSV creation,
independent output verification, durable receipts, readback, and a second
idempotent run. It uses 29,421 synthetic records with 1,062 synthetic missing
rows, includes harmless extra archive members, and creates no real E79 output.
