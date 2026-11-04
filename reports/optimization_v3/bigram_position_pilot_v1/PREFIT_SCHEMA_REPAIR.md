# Prefit schema successor and cache-lineage note

The initial official `run-all` stopped **before any `FIT_STARTED` event** with `ValueError: Object arrays cannot be loaded when allow_pickle=False`. No target labels were read and no optimizer fit was attempted. The single six-fit reservation remains intact.

The original freeze records these row-NPZ byte hashes: E67 `20f27346c374f4f5b0a9b434c6b1738edfe91bee17ad8b10da089f23dd627f10`; E75 `9d31cac6c28082bd325770eaf140be230ea4350d4ebba45922b31ad038dca1de`. The current files have different byte hashes because `fit_record_ids` and `score_record_ids` were reserialized from NumPy object strings to fixed-width Unicode. A scan of the available Research Next caches found no original file instance. I reconstructed each original NPZ byte sequence from the current arrays, restoring object-string IDs, and both SHA-256 hashes match the original freeze exactly. The physical pre-migration files remain unavailable; the recovered byte-identical files are stored separately from the current Unicode archive. The original freeze and its hashes remain unchanged.

Read-only comparison against the original freeze found:

- Row indexes, score cycles, and availability masks have their original logical hashes.
- Casting the current Unicode ID strings to the original object-string hash contract reproduces both frozen fit-ID and score-ID hashes exactly; IDs also align to official Train identity rows.
- All six sparse matrix files and CSR logical hashes match the original freeze exactly.
- The saved original-logit hashes, parent POS gamma, prediction/anchor bundles, and parent matrix-manifest hashes remain bound to the original freeze.
- The prior cache failure occurred before labels or fits, so there are still zero pilot `FIT_STARTED` records.

The successor preflight archives the current migrated cache **before rebuilding it**, records its actual bytes separately, rebuilds from the pinned official Train and frozen preprocessing, then independently compares all cache arrays, row/ID hashes, matrices, and parent anchors. It also reserializes the original object-ID schema and requires the exact frozen row-NPZ hashes to match before marking byte recovery successful. Any mismatch blocks successor freeze creation. The successor freeze is separately signed and bound to the one existing six-fit reservation through an idempotent ledger event; it does not add or release a slot, change the recipe, or rewrite the original freeze.

No Test/SampleSubmission data, Test labels, or `artifact_signal` values are used in successor preflight or migration verification. The fit-only label reader remains unopened until a durable `FIT_STARTED` event.
