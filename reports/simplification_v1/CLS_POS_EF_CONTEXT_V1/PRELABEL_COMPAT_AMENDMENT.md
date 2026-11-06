# CLS_POS_EF_CONTEXT_V1 — prelabel verifier compatibility amendment

The committed, source-frozen independent prelabel verifier has two bare helper
references that are not defined in its module globals. A builtins-aware
`symtable` audit of the full file found exactly `sha256_array` and `sha256_csr`.
Both functions already exist in the source-hash-pinned frozen producer driver.

A direct-call signature audit also found one callsite mismatch: the frozen
`_independent_q(ef, mu, scale)` helper is called once with a `context_state`
mapping. The compatibility runner unwraps only the existing `mu` and `scale`
keys and delegates to that same frozen function. It rejects missing keys or a
conflicting third argument. The numeric three-argument call path remains a
direct call to the frozen helper.

The compatibility runner verifies the original verifier hash, repeats that
unresolved-global audit, binds only those two names to the exact frozen
producer hash functions, and adapts the single `mu`/`scale` call shape. It leaves
the verifier source, frozen model source,
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

The first full verifier run reached query-prediction reconstruction only after
completing fit-prefix objective checks, then stopped at the mismatched callsite.
It used `write=false`, wrote no prelabel report, and did not open query labels.
The correction preserves the objective logic and fixes only the call's argument
shape.

This is a verifier compatibility repair, not a model fit or additional experiment.
