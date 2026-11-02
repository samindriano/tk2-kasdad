# A02 continuation — final handoff

## Decision

The frozen six-fit continuation passed all nine scientific gates. No supported subgroup crossed the configured risk-review hold threshold. The authorized conditional origin-79 full-Train fit and Test inference therefore ran. The candidate remains research evidence; this work did not submit it to Kaggle or establish a public score.

## Continuation results

| Evaluation view | Rows | C16 baseline Macro F1 | A02 observed | Shuffled control | Δ vs C16 | Δ vs shuffle |
|---|---:|---:|---:|---:|---:|---:|
| New-origin aggregate W2/W3/W5 | 21,033 | 0.835171 | 0.837187 | 0.834268 | +0.002016 | +0.002919 |
| Full six-window primary | 38,588 | 0.829944 | 0.832045 | 0.829277 | +0.002101 | +0.002767 |
| Late W5/W6 | 15,953 | 0.815790 | 0.816958 | 0.813424 | +0.001168 | +0.003534 |
| Inherited H12-L | 20,903 | 0.808961 | 0.811159 | 0.809773 | +0.002198 | +0.001386 |

New-window gains versus C16 were +0.002972 at W2, +0.004333 at W3, and −0.000823 at W5. Thus 2/3 new windows were nonnegative, and the sole negative window stayed within the frozen −0.003 harm limit. W1 is the exact C16 fallback; W4 and W6 reuse the verified pilot. H12-L also reuses pilot predictions on the same development cycles, so it is inherited evidence rather than independent confirmation.

The subgroup audit used the frozen support rule (at least 200 rows and 30 examples of each class). One historical Macro F1 harm flag exceeded 0.01; no positive-recall harm flag exceeded 0.02 or 0.03. No supported group exceeded the risk hold limits of 0.03 Macro F1 harm or 0.05 positive-recall harm.

## Final fit and output

- Fit: `A02_CONT_E79_OBSERVED_FULLTRAIN`, using all 50,608 official Train rows for the text prefix and 36,010 present-stream OOF residual-bank rows from the six frozen C16 blocks.
- C16 baseline cache and manifest hashes matched the existing frozen sparse25 artifacts. The missing-stream route returns the baseline probability exactly; present-stream predictions use the frozen A02 correction and threshold 0.50.
- Test target was not loaded. The official Test features were used only after gate passage to produce predictions. SampleSubmission was used for ordered IDs and output schema checks.
- CSV: `submissions/optimization_v3/a02_continuation_v1/a02_continuation.csv`; 29,421 rows, columns `record_id, artifact_signal`, IDs match the full official Test order with no duplicates or missing IDs, labels are integer 0/1, counts 22,719 zeros and 6,702 ones.
- CSV SHA-256: `6909d4af8a326661b81e4a947d42b42919c004852967ca99f6bd481a78c679f8`.
- Local state: `PREPARED_VALIDATED_NOT_SUBMITTED`; submission authorization is false. No Kaggle upload or public score was obtained for A02.

The independent post-fit audit replayed the persisted model on the official Test features and reproduced the saved probabilities, neighbor geometry, and corrections bit-for-bit. See `final_inference_audit.json` and `submission_manifest.json`.

## Accounting and provenance

- Frozen continuation signature: `93a2f388673e5cbb2fde60364fd0121918695ecb908c9b2c9c8b91e239de54e6`.
- Frozen protocol SHA-256: `b4438d953441e566f8107e7f40c4ef6bfa2bb504dea871d9783c88e36a9fcb96`.
- The cap amendment raised the total cap to 66 and reserved six continuation fits plus one conditional final fit. Final accounting: 66 starts, 64 saved, 2 historical failures, 0 remaining fit capacity.
- The first freeze and preflight remain preserved. A budget-adapter schema mismatch was caught before ledger mutation or any new fit. The adapter was corrected, then a successor freeze and preflight were committed without changing the protocol or gates; details are in `freeze_supersession.json`.
- Source commits: preparation `681fc5e`, adapter correction `31a8c27`, successor freeze `422e35c`, continuation result checkpoint `4e4f8ac`. Final handoff and output audit are committed separately.

## Limits and review

The continuation estimates reuse previously examined development cycles. Only the origin-specific predictions at 59, 63, and 71 are new. The results support a predictive association under the frozen recipe, not causal claims or untouched temporal validation. Test predictions do not provide a local test metric, and there is no Kaggle score for A02.

Review `SUMMARY.md`, `results.json`, `groups.csv`, `per_cycle.csv`, `submission_manifest.json`, and `final_inference_audit.json` before separately authorizing any upload.
