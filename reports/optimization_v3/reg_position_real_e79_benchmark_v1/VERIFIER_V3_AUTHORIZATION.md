# Independent verifier v3 and QA attestation

## Scope

Sol authorized verifier/schema-binding correction on 2026-11-04 after version 2 inference had completed. The original CSV, prediction bundle, signed inference manifest, trained freeze, fit ledger, model states, receipts, and version 2 adapter files remain unchanged.

Version 2 stored original-source verification results as a per-file boolean map. Its verifier expected that field to contain the pinned hash inventory. Version 3 preserves the original manifest and treats those booleans as results only. It independently loads the pinned inventory from the immutable preflight and trained freeze, requires the exact expected paths and hashes, recomputes the actual file hashes, and rejects any changed file even when every recorded boolean is true.

## Historical adapter integrity

The original inference manifest remains byte-identical and must retain canonical manifest signature 85b6df507792a6698acc3ab255ab0a1815c5f93748fb53f8cd951d22871ff663 and its original file hash. Version 3 verifies the version 2 adapter commit is the recorded full commit, exists, and is an ancestor of the version 3 commit. It reads each pinned version 2 adapter file from that historical Git commit and compares the blob hash with the original manifest's adapter hash. The current version 2 files must also continue to match those pins.

The separate version 3 QA report binds its verifier source files and local commit with SHA-256 hashes and a canonical attestation signature. It also records the original manifest signature and file hash, the fit and source inventories, and independent QA results.

## Verification

The independent verifier reconstructs the R06 baseline contract, Test/Sample/R06 identities and cycle order, REAL feature design, saved-model predictions, fixed delta formula, missing-text fallback, CSV round trip, file hashes, and duplicate label-vector scan. It uses the existing two model states without another fit or inference run. It does not read Test labels or access Kaggle.
