# CLS_POS_EF_CONTEXT_V1 — prelabel verifier compatibility amendment

The committed, source-frozen independent prelabel verifier has two bare helper
references that are not defined in its module globals. A builtins-aware
`symtable` audit of the full file found exactly `sha256_array` and `sha256_csr`.
Both functions already exist in the source-hash-pinned frozen producer driver.

The compatibility runner verifies the original verifier hash, repeats that
unresolved-global audit, and binds only those two names to the exact frozen
producer helper functions. It leaves the verifier source, frozen model source,
prediction bundles, optimizer receipts, readiness report, and candidate recipe
unchanged. The independent shuffle, objective, prediction, routing, and
metric-reconstruction logic remains in the frozen verifier; only its hash
helper references are supplied.

The runner executes the real `verify_prelabel_outputs` entry point and writes a
single PASS report with amendment provenance only after all checks succeed. Its
declared boundary is `query_labels_read=false` and `test_or_sample_access=false`.
The verifier may reread Train labels for the already-fitted training prefix to
reconstruct fit objectives, as its frozen contract specifies. Query-cycle labels
remain unopened until the prelabel report and prediction readiness are committed.

This is a verifier compatibility repair, not a model fit or additional experiment.
