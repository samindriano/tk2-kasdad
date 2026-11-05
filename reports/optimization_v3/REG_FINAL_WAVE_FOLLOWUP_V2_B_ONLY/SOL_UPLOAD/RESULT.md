# Root verified REG B-only public benchmark

Kaggle **COMPLETE**, submission **58010862**, public MAE **11.04304**. One upload was made via authenticated Kaggle CLI with message `Samuel REG Global Affine Present Only Benchmark`. Prior best affine-all-rows score11.04552: observed gain **0.00248 MAE**. Private outcome unknown. No final-slot selection change.

## Exact candidate

`GLOBAL_PRESENT_ONLY_RESTORE_DIRECT_MISSING_E79`, CSV `submissions/reg_final_wave_followup_v2_b_only/global_present_only_e79.csv`.
SHA256 `85e6228e664559ba6ef772b9b2165b8aedabf9a1da4367f67f2408c7cca79a19`.
29,421unique ordered IDs; schema record_id,threat_rank; finite range[2.169196428141562,99.61371132560167]. All28,359present values remain exactly the global affine reference. Exactly1,062missing values restore original saved direct prediction, which is the frozenR06fallback on that route. No new fit, LP or inference. This is not H8recency.

Root independently verified canonical freeze/manifest/QA/adapter seals, every pinned source/input byte hash, IDs, duplicate filename history, and exact np.where(missing,direct,global) float64 reconstruction. Root QA used own replay, not builder output claims. The initial root QA probes needed two local script corrections (source dictionary attribute name and nested input-hash schema); both stopped before CSV verification/upload intent and did not mutate the candidate or pins. Successful root replay was followed by one upload intent, one accepted upload and confirmed scoring.

## Internal evidence and limits

Causal GLOBAL_PRESENT_ONLY versus same-history GLOBAL: D1 W5 MAE11.412228 vs11.416864; D2 W6 MAE9.945317 vs9.959275. Pooled unique15,953rows:10.631463 vs10.641061, gain0.009598. Missing subgroup n1033 improves13.684998->13.536771; no supported missing/tail harm>0.25. These are reused development diagnostics. Exact final affine coefficients were fitted on all20,903causalOOFrows; their same-row fit performance is not heldout evidence. The observed public gain is smaller than the pooled internal gain and does not identify public composition or guarantee private improvement.

H8 recency lost0.000556MAE pooled and remains RETIRED for this recipe. Four diagnostic LPs were completed; fifth production LP was not run. No additional parameter, threshold, weight or half-life sweep follows this result.

## Preserved technical lineage

R1 application failed before output due a source-specific digest serialization mismatch; fixed in separately frozenR2. Frozen verifier default still selectedR1; authorized runtime path override selectedR2 without relaxing any hash/input/formula checks. Full adapter command and R2freeze binding are in ../APPLICATION_OR_VERIFIER_ADAPTER_R2.json. Producer-stage manifests/QA state no upload; this later Sol receipt is authoritative for upload/scoring status.

Latest public model equation: present ->clip(2.169196428141562+0.9744451489746011*direct,0,100); missing ->direct unchanged. Existing diagram for prior11.04552 candidate remains a historical snapshot; only this final missing-route affine bypass changed.
