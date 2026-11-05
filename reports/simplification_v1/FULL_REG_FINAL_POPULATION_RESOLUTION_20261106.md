# Full REG LOCAL final application: population clarification

Root resolves the earlier ambiguous phrase "20,903 OOF PRESENT rows": **20,903 is the unique OOF universe; 19,528 stream-present rows are eligible for the final LOCAL calibration.** The 1,375 missing-stream rows are excluded from the fit and retain the champion fallback at application.

Root independently checked SHA256 `e75f70cccd1bb231499060b84155af876173214d984461e39e4c1364e4c5b2bf` of Luna2's `experiments/runs/reg_monotone_hinge_p50_v1/FINAL_APPLICATION_R1/fit_inputs/HINGE_FINAL.npz` and loaded only `record_ids` and `missing_mask`: 20,903 rows, 20,903 unique IDs, 19,528 present and 1,375 missing. No target member, predictions, E79/Test archive or evaluation labels were loaded. The earlier blocked receipt is preserved, not rewritten.

This clarifies the existing present-only correction, not a new recipe or retrospective gate change. Luna2 is authorized to resume source-only preparation for one final LOCAL scalar on the **full** DIRECT + fixed final hinge champion. Triangle basis uses raw DIRECT, center 50 and half-width 10; delta is restricted to [-2,0]. Fit uses the weighted lower median of residual/basis with weights equal to the positive basis. Present rows with zero basis contribute a constant objective and do not identify delta. Missing rows must remain unchanged.

Required next checkpoint: separate source/freeze, feature-only identity preflight, one-slot root-pass contract, independent verifier and full synthetic CLI/receipt/resume/tamper tests. No actual final fit, target load, E79/Test access, CSV or Kaggle upload before root source approval. No new base models or GLOBAL control fit. Preserve simplified-model checkpoints and the failed qualification verdict of the earlier full-model diagnostic.

Resolution and source-only task were sent directly to Luna2 with explicit request to return the source-ready package to root.
