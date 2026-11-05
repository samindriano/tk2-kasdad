# Sol review — fixed-p50 hinge diagnostics

Decision: PREPARE_ONE_EXPLORATORY_CSV_APPLICATION_SOURCE. This is not a qualified scientific PASS and does not authorize a final LP, Test access or upload yet.

Sol independently replayed the frozen diagnostic verifier under the shared job lock and fresh resource gate. Result: PASS_INDEPENDENT_DIAGNOSTICS_REPLAY, 15,953 unique evaluation rows, exactly two completed official LP starts and no final fit. Source, input hashes, fit states, primal objectives, row alignment, metrics, missing fallback, subgroup harm and resource receipt reconciled.

| View | n | Same-history GLOBAL present-only MAE | Hinge present-only MAE | Gain |
|---|---:|---:|---:|---:|
| D1, cycles72–75 | 7,462 | 11.412227990 | 11.404872094 | +0.007355896 |
| D2, cycles76–79 | 8,491 | 9.945316528 | 9.939831510 | +0.005485018 |
| Unique pool | 15,953 | — | — | +0.006360119 |

The >=0.02 pooled qualification threshold fails. The explicitly frozen exploratory-review rule passes: both windows improve and maximum supported subgroup harm is 0.132487165, below 0.25. Missing predictions are identical by construction. Low-target MAE improves by 0.216/0.192, while high-target MAE worsens by 0.132/0.021; both window p90 absolute errors worsen slightly. These trade-offs remain part of the decision and must not disappear from the report.

Coefficients are similar across histories: low slope 1.00155/1.00039, high slope 0.94954/0.95441. This is a small, interpretable adjustment, not evidence of a large new source of predictive information. Development labels were reused; the two windows are temporally different and D2 also changes prediction origin. No public-score uplift is promised.

## Bounded next step

Prepare one versioned final-application executor and independent verifier, then return source/freeze/synthetic QA for root review before execution. A later approval may release only the existing third slot HINGE_FINAL, fit on the same 20,903 unique historical OOF rows (not Test and not new base-model training). Reuse the fixed knot50, objective, slope bounds, support and clipping/routing policy without tuning. Do not refit D1/D2 or any GLOBAL control and do not reset historical accounting. Preserve R4 sources and receipts.

Planned application, only after separate approval: official E79 raw DIRECT predictions already pinned; present rows receive clip(g(raw_DIRECT),0,100), missing rows remain bit-identical raw_DIRECT. One CSV with record_id/threat_rank, exact official order and 29,421 unique IDs, 1,062 missing-route rows, finite values in [0,100]. No labels from Test. Bind raw DIRECT, official identity/sample format, full-OOF inputs and all source hashes in the manifest. Audit all loader/verifier/API contracts using actual highest-level synthetic execution; preflight must exercise every non-target contract used by the real path. Use new immutable output paths and durable final-slot reservation/start/save recovery. The new final authorization must explicitly bind HINGE_FINAL; the diagnostic pass does not grant it.

Sam currently prefers collecting several research results before selecting a worthwhile benchmark batch. Even after a candidate CSV exists, upload remains a separate root decision. No Kaggle or final-slot selection changes by Luna. Questions and complete-result handoff go to Sol before stopping.
