# REAL E79 zero-fit inference continuation, version 2

## Authorization and scope

On 2026-11-04, Sol authorized a same-artifact, zero-fit continuation after the first inference attempt stopped at the R06 baseline digest guard. This amendment does not authorize another fit, a recipe change, a new candidate, or an upload.

The continuation uses only the two model states already saved under experiments/runs/reg_position_real_e79_benchmark_v1/. It preserves the original preflight, runner, verifier, trained freeze, model states, and fit receipts byte-for-byte. It adds a versioned inference reader and independent verifier for R06's original raw-byte prediction digest, followed by one inference and one independent QA pass.

## Exact authorized input bindings

- Original worktree commit: 0ca0255f85f8c8248c1aa4fa369456ed9b638180.
- Immutable zero-fit preflight SHA-256: 6f2604a1813d65d5a684df549253a0ac9f56acadc286f9dc63e29fcb88114562.
- Original trained freeze SHA-256: fc44d92aaf9285af3680082649153ac88f77c917e282a9883ceb70a1770f8027.
- Original trained freeze signature: 03977900247e92e3a715985e37a6097d5425febda4e6d7ab0fc204c26018dce5.
- The exact original worktree source hashes and both saved model and receipt hashes are pinned in scripts/reg_position_real_e79_hash_compat_v2.py.
- The source R06 manifest, prediction archive, and CSV are bound by their exact file hashes in that same compatibility module.

The adapter records its own versioned file hashes, local commit, the original preflight and freeze hashes, each saved artifact and receipt hash, and a canonical continuation signature in the inference manifest.

## Digest contract

The original R06 producer at scripts/optimization_v3_expanded_prepare_submissions.py stores prediction_audit.prediction_sha256 as sha256(final.astype("<f8").tobytes()). The old inference reader compared that field with a structured digest that also prefixes canonical shape and dtype metadata. Those digest contracts differ even when the prediction values agree.

Version 2 validates the existing manifest field using the original raw little-endian float64 byte contract. It reports that digest and the structured array digest in separate named fields. It still checks the R06 manifest, prediction archive and CSV file hashes, CSV schema and row count, prediction finiteness and range, CSV-to-archive agreement, IDs, and cycle identities against official Test only after the baseline gate passes.

The compatibility fixture proves that the two digest formats differ, that the old manifest contract accepts the raw-byte digest, and that it rejects the structured or an unrelated digest.

## Stop conditions and accounting

Any mismatch in the preflight, original source hashes, freeze, saved states, receipts, R06 source files, baseline checks, Test identity/order, formula, CSV, or independent verification stops the continuation. The adapter refuses to overwrite an existing inference bundle or CSV.

The only permitted work after this amendment is one inference from the two saved states and one independent verifier pass. Fit accounting remains two starts, two saves, zero retries, and zero added starts. Sol remains the only upload owner; this continuation does not access Kaggle.
