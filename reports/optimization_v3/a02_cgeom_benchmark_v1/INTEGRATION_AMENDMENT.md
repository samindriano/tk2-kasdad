# A02/CGEOM integration repair amendment: six fresh component IDs

**Status:** authorized integration resume, pending pre-fit feature and synthetic QA. This amendment preserves the original candidate recipes and all historic fit events.

## Starting checkpoint and accounting

- Authorization commit: `9e8407264f47c879b609a85c100de295d36229d7`.
- Exact starting ledger: 318 events; SHA-256 `5ea031efca2a09ad0f0c9d0204288412735c051114dae9085e6f828a4cc4aca0`; cap 131; 123 active reservations; 123 starts, 117 saved, 6 failed, 0 pending.
- The original C2_B LR and its hash-fix replacement remain two distinct failed, consumed IDs. The three saved C2_B preprocessing states remain reused under their original preparation commits, artifact hashes, receipt hashes, and source hashes.
- The five remaining IDs in the original reservation remain released and are not re-reserved. Six new IDs below cover the six remaining logical components.

## Authorized cap, IDs, and limits

- Global cap: 131 → 132. Continuation cap: 48 → 49. All other category caps remain unchanged.
- Reserve exactly six fresh continuation units. Active reservations become 129 total / 49 continuation.
- Maximum total starts: 129. Maximum total consumed units for this batch: 11, including both prior failed LR IDs. If all six new fits save, final accounting is 129 starts / 123 saved / 6 failed / 0 pending.
- No further retries or replacement IDs are authorized. On a new failure, stop and release only new IDs that never started; preserve every started event and artifact.

| Logical component | Prior failed or released ID | Fresh authorized ID |
|---|---|---|
| C2_B Logistic Regression | Original LR and HASHFIX_R1 remain failed | `A02_CGEOM_BENCHMARK_V1_E79_C2B_LR_INTEGRATION_R1` |
| C1_B structured preprocessor | Original unstarted ID remains released | `A02_CGEOM_BENCHMARK_V1_E79_C1B_STRUCTURED_INTEGRATION_R1` |
| C1_B HGB fallback | Original unstarted ID remains released | `A02_CGEOM_BENCHMARK_V1_E79_C1B_HGB_INTEGRATION_R1` |
| POS_BAG_OFFSET gamma | Original unstarted ID remains released | `A02_CGEOM_BENCHMARK_V1_E79_POS_BAG_OFFSET_GAMMA_INTEGRATION_R1` |
| JOINTPOS Logistic Regression | Original unstarted ID remains released | `A02_CGEOM_BENCHMARK_V1_E79_JOINTPOS_LR_INTEGRATION_R1` |
| CGEOM aligned predictor | Original unstarted ID remains released | `A02_CGEOM_BENCHMARK_V1_E79_CGEOM_ALIGNED_PREDICTOR_INTEGRATION_R1` |

## Pre-fit and execution boundaries

- Reuse the three already saved preprocessing components exactly as recorded in `INTEGRATION_REPLACEMENT_MAPPING.json`; do not refit them.
- Run real official Train feature-only C2_B matrix and vectorizer compatibility checks before the first new `FIT_STARTED` event. Do not read Test feature rows until all trained components are frozen.
- Exercise each of the six actual `_make_fit_payload` branches with synthetic data, plus durable persistence and resume, using only a temporary ledger and artifact directory. QA must bind to the committed source and leave the shared ledger unchanged.
- Fit one authorized component at a time, using only official Train with the recipe's fit population, existing thread/RAM limits, and no retry. The cap change is not permission to fit any other ID.
- After all six fits pass receipt verification, write and commit the trained-state freeze before official Test inference. Create exactly the three already selected candidate CSVs and independent manifests/QA. Test labels remain unread; no Kaggle API, CLI, browser, or upload is part of this batch.
- Final stop status: `PREPARED_FOR_BATCH_REVIEW_NOT_SUBMITTED`.
