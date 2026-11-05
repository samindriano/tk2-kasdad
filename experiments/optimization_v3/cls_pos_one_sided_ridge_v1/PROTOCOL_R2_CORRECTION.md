# CLS positional ridge protocol R2 metadata correction

Status: source-only correction; no official fit authorized by this note.

## Revision lineage

- Parent source commit: `7d8513a5903308a80a30ce4b55a11697381ecb95`.
- Preserved predecessor: `experiments/optimization_v3/cls_pos_one_sided_ridge_v1/protocol_r1.json` (byte-preserved from the prior `protocol.json`).
- Successor: `experiments/optimization_v3/cls_pos_one_sided_ridge_v1/protocol_r2.json`.
- R1's E75 manifest digest: `67ec3f9bd376f5f4dad2afbd5e13ee5fd353d68bc044129c3f969860733e23b1`.
- R2's exact E75 manifest digest: `67ec3f9bd376f5f4dad2afbd5e13ee5fd8316126fa3894ef676fa40077166fc5`.

## Why R2 exists

The first target-free identity preflight at parent commit `7d8513a` stopped at the R1 E75 manifest-digest comparison. The mismatch was confined to the newly drafted protocol's expected digest. The official E75 matrix manifest file is unchanged and hashes to `67ec3f9bd376f5f4dad2afbd5e13ee5fd8316126fa3894ef676fa40077166fc5`. Its prior campaign freeze and prior target-free source-preflight receipt independently pin that same digest. The E75 ordered row IDs, cycle metadata, prediction bundle, all matrix files, and frozen bag artifact/receipt passed the inherited identity/cache checks. E75 retains 42,117 fit rows (cycles 47–75) and 8,491 query rows (cycles 76–79); the query prediction bundle remains `83e5beb9b10be4d395df4a2292cb464625f356e5e77a76a232b92f007c1b1306`.

## Evidence references

- Raw manifest: `experiments/runs/optimization_v3_research_next_v1/POSITIONAL_TEXT_PILOT_V1/MATRIX_MANIFEST_E75.json`; SHA-256 `67ec3f9bd376f5f4dad2afbd5e13ee5fd8316126fa3894ef676fa40077166fc5`.
- Parent POS_BAG_OFFSET freeze: `experiments/runs/optimization_v3_research_next_v1/POS_BAG_OFFSET_V1/FREEZE.json`; SHA-256 `5d754b99cf1beb789baec9707652cd9c3842ea95933c0ecc21be515f62eaef9d`; its `origins.E75.matrix_manifest_sha256` equals the raw manifest digest above.
- Parent target-free identity receipt: `reports/optimization_v3/cls_pos_frequency_ridge_v1/SOURCE_PREFLIGHT_R2.json`; SHA-256 `065f81e4d20da6559f11983c9cbf0a372e75d16537450bed7999619964383b73`; it records exact E75 Train/cache identity and `target_values_opened=false`.
- Parent R8 evaluation freeze: `experiments/optimization_v3/cls_pos_frequency_ridge_v1/EVALUATION_FREEZE_R8.json`; SHA-256 `624d5da9d75f8f78996c35e758a97e095740ba4b13c2240c81b6cd995abfe720`.
- The inherited identity verifier is `scripts/optimization_v3_cls_pos_frequency_ridge_v1.py`, `_verify_origin_identity_and_files` (lines 254–277): it binds the parent manifest digest, exact Train fit/score row metadata, cached ordered score row IDs/cycles, and all matrix byte hashes. The function passed both E67 and E75 in the failed new preflight before the new-protocol equality check stopped E75.

## Scope controls

R2 changes only the expected E75 `matrix_manifest_sha256` value. The candidate recipe, fit IDs and cap, data source, identity columns, origins, temporal windows, caches, model inputs, gates, historical verdicts, and Test/Kaggle exclusions remain byte-for-byte equal to R1's parsed scientific sections. There is no fallback that accepts either digest: code requires exact equality with the single R2 value. Synthetic QA includes a negative case for the stale R1 value and a positive case for the exact R2 value. No labels were read, no fit began, and no Test or SampleSubmission input was opened during the blocked preflight.
