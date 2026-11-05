# Sol decision after completed one-sided pilot

Decision: RETIRE_EXACT_ONE_SIDED; PREPARE_ONE_GLOBAL_STRENGTH_BENCHMARK_SOURCE_ONLY.

Sol independently replayed the frozen results verifier under the shared job lock and current saved-replay resource gate: PASS_INDEPENDENT_RESULTS_RECOMPUTATION, 1,291 metric objects, maximum difference zero; prediction replay also passed for two origins/eight probability replays. Query labels were reused only behind the committed manifest. No new fit or artifact overwrite occurred in this review.

| View | n | ONE_SIDED gain vs uniform | GLOBAL_STRENGTH gain vs uniform |
|---|---:|---:|---:|
| E67 H12 | 20,903 | -0.000142020 | +0.000029247 |
| E67 W4 | 4,950 | 0.000000000 | +0.001027712 |
| E67 W5 | 7,462 | -0.000132729 | -0.000397757 |
| E67 W6 | 8,491 | -0.000233676 | -0.000210448 |
| E75 W6 | 8,491 | +0.000031027 | +0.000703139 |
| Unique E67 W4/W5 + E75 W6 | 20,903 | -0.000028988 | +0.000375109 |

ONE_SIDED fails both its mechanism and competition gates; retain that verdict. GLOBAL_STRENGTH is an informative control, now considered as an exploratory candidate, not a retroactive mechanism PASS. Its H12 gain is below the competition threshold and E67 W5/W6 are negative. Its unique-pool cycle directions are seven positive, two negative, three ties; at E75 W6 three positive and one tie. Brier improves by 0.00003725 on the unique pool but worsens slightly on E67 H12. E67/E75 W6 overlap and differ in training history, so these are not independent replications. Repeated development selection and post-result promotion of a control limit inference.

## Next bounded preparation

Luna1 may prepare complete source, synthetic end-to-end QA, and target-free preflight for exactly one future E79 gamma fit using GLOBAL_STRENGTH. No fit/Test/E79 inference/CSV is authorized until a separate root source pass. Do not rerun the four saved fits or refit existing base models, vocabularies or A02 memory. Establish whether existing immutable E79 positional matrices, prefix DF maps and uniform/C2_B artifacts suffice; inventory exact hashes/row and feature order. If a required learned transform is missing, report its exact need and cost to root instead of silently adding a preprocessing fit.

Recipe: preserve the original aligned-map formula and derive its E79 map from Train-prefix DF only, v=max(1,w_aligned), c=mean(v), then use scalar c on every positional gamma coordinate. No tuning c, smoothing, threshold, seed or blend. One possible new gamma fit, C2_B bag logits fixed, historical objective and optimizer unchanged. Global mean diagonal penalty matches the selective arm's average penalty, not effective degrees of freedom. This is regularization-strength evidence, not proof of a rare-token mechanism.

Planned deployment after separate authorization: present p=0.5*newPOS+0.5*existing pA at threshold0.50; missing use existing pA at frozen operational threshold0.53230976129477. Reuse exact current champion missing labels/probabilities. No Test targets. One potential CSV, 29,421 exact official IDs, integer artifact_signal. Preserve frozen sources and all artifacts; additive wrappers must exercise actual CLI/default data paths in synthetic QA and statically check unresolved globals. Separate execution provenance from original fit provenance. Fresh shared lock and current RAM gates, no polling/scheduler/push. Return source-ready plus exact accounting and UNKNOWNs to Sol before execution. Root later chooses a measured submission batch; no Kaggle upload by this instruction.
