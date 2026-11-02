# Measured Benchmark Milestone — Final Handoff

**Snapshot:** 2 November 2026
**Scope:** Kaggle history reconciliation, measured Classification benchmark, and zero-fit Regression midpoint audit.
**Boundary:** no Kaggle upload, no final-model/champion change, and no new Regression fit.

## A. Kaggle history reconciliation

The authenticated Kaggle UI showed 21 submissions for Samuel Indriano: 7 Classification and 14 Regression. All 21 were `Complete` at observation. Manual-upload refs and exact timestamps remain unknown except for the two initial API receipts. The full row-level registry and local manifest/CSV hashes are in `reports/submissions/reconciliation_20261102/REGISTRY.json`.

| Task | Candidate | Public score | Kaggle status |
|---|---|---:|---|
| Classification | C11 | 0.82739 | Complete |
| Classification | C17 | 0.82825 | Complete |
| Classification | C16 | 0.82844 | Complete |
| Classification | V3_CLS_SPARSE25 | 0.82908 | Complete |
| Classification | DC_CLS_CB_MISSING_V1 | 0.82701 | Complete |
| Classification | SAMUEL_CLS_Q80_ROUTE_V1 | 0.82640 | Complete |
| Classification | C4_HGB_INITIAL | 0.82641 | Complete; API ref `57808225` |
| Regression | R21 | 11.29314 | Complete |
| Regression | R11 | 11.39469 | Complete |
| Regression | R06_ORIGINAL | 11.16620 | Complete |
| Regression | R22_ORIGINAL | 11.30535 | Complete |
| Regression | V2_REG_SPARSE_LGB25 | 11.35970 | Complete |
| Regression | DC_REG_CATBOOST_V1 | 12.48607 | Complete |
| Regression | DC_REG_RIDGE_50_V1 | 11.75259 | Complete |
| Regression | DC_REG_RIDGE_25_V1 | 11.64134 | Complete |
| Regression | SAMUEL_REG_SVD128_PAIRED_V1 | 11.81149 | Complete |
| Regression | LGB_TEXT_INNER_V1 | 11.71363 | Complete |
| Regression | SVD128_INNER_V1 | 12.38411 | Complete |
| Regression | SVD_HGB_INNER_V1 | 12.34519 | Complete |
| Regression | HYBRID_RIDGE_INNER_V1 | 12.55466 | Complete |
| Regression | R1_D_INITIAL | 13.53473 | Complete; API ref `57808231` |

The `11.71363` score belongs to `LGB_TEXT_INNER_V1`, not R1_D. C16 and R22 are now confirmed Complete, superseding their stale local `PREPARED_NOT_SUBMITTED` / `NOT_SCORED` snapshots. Local hashes cannot be bound to server-upload bytes because Kaggle does not expose server-side CSV hashes. The visible `0/3` is a final-leaderboard selection counter, not upload quota; live quota is unknown. No upload occurred in this milestone. The observed competition deadline was 8 November 2026 at 23:55 WIB.

N04's old subgroup table mislabeled `harm_vs_R22` values as gains. The table/prose are corrected and an append-only correction note records the arithmetic: W1 missing-stream harm `+0.202732` and H12 missing-stream harm `+0.038067`. Underlying result JSON and aggregate MAEs did not change.

## B. Classification benchmark

### N02 continuation

Eight official temporal fits completed and saved sequentially: sketch and matched no-sketch at cutoffs 59, 63, 71, and 75. The main result is **`BENCHMARK_GATES_FAIL`**:

| View | Candidate Macro F1 | Sparse25 Macro F1 | Gain vs Sparse25 |
|---|---:|---:|---:|
| Primary full-six, n=38,588 | 0.82980280 | 0.82994355 | −0.00014074 |
| Matched no-sketch comparison | 0.82980280 | 0.82960936 | +0.00019344 vs no-sketch |
| H12 pooled, n=38,588 | 0.80365531 | 0.80296557 | +0.00068975 |
| H12 cutoff 55 | 0.79487655 | 0.79503181 | −0.00015526 |
| H12 cutoff 67 | 0.81040436 | 0.80896134 | +0.00144302 |
| Late W5+W6, n=15,953 | 0.81497757 | 0.81578974 | −0.00081217 |
| W6, n=8,491 | 0.81483141 | 0.81659919 | −0.00176778 |

The primary gain missed the frozen minimum `+0.0002`; that is the failed benchmark gate. Four of six windows were nonnegative. All 144 supported groups passed the benchmark's amended 0.03 recall-harm and 0.01 F1-harm guards, but two H12 text-load-Q4 rows still violate the original 0.02 recall guard. The original qualification verdict is unchanged. Therefore **no N02 full-Train fit or CSV was made**.

### ClassOnly exception and prepared CSV

ClassOnly had already been preselected for one full-Train benchmark CSV regardless of N02's result. Its prior temporal evidence was diagnostic-only: full-six gain `+0.000687`, late `−0.000237`, W6 `+0.001000`, pooled H12 `+0.000596`, with 64 supported group guards passing. This does not change its historical qualification status and is not a Test score.

The first ClassOnly full-Train estimator fit completed but failed bundle persistence because the runner omitted required `baseline_prediction`. That `FIT_FAILED` remains in the ledger. No fitted expert or raw expert probabilities were recoverable. After an amendment-specific synthetic test passed the production payload builder, missing-stream routing, metadata, persistence/readback/resume, and post-fit fault injection, one authorized technical retry was run with the same N01 recipe, weights, blend, and threshold. It saved successfully. CSV recovery then used that saved bundle and performed no estimator refit. A generic CSV helper initially raised on an int8/int64 readback dtype mismatch; independent value/ID checks verified exact semantic equality and passed.

Prepared artifact:

- CSV: `submissions/optimization_v3/measured_benchmark_v1/class_only.csv`
- CSV SHA-256: `59deb0ca3d0a86b16b5c646190f11c02e3fec1053ef5c9d87273f2a265dfc030`
- Rows/schema: 29,421; `record_id,artifact_signal`; exact ordered official Test IDs; integer 0/1 labels.
- Manifest: `reports/optimization_v3/measured_benchmark_v1/CLASSIFICATION/CLASS_ONLY_SUBMISSION_MANIFEST.json`
- Manifest SHA-256: `8edc26efb4336f4339bb8d969db6bff5b4e97a24b09e91d5c4e836043f187ea4`
- Persisted retry bundle receipt: `80edcf1343775231c06dac57512aa6c5887659a0dea48a14d6dea4dfc0864fae`
- Preparation status: `PREPARED_VALIDATED_NOT_SUBMITTED`; no Kaggle ref and no upload.

The estimator fit used all 50,608 official Train rows (cycles 47–79). Learned preprocessing and weights came from Train only. Test was loaded without its target column and used for inference/ID checks. Missing-stream rows use the frozen CB_MISSING path; present rows use `.75 C2_B + .25 class-only sparse expert`, threshold 0.50.

## C. Regression midpoint

The fixed zero-fit midpoint is the 50:50 R06_ORIGINAL/R22_ORIGINAL blend without reranking. Independent verification covered 25 frozen input hashes, 12 existing bundles, and 11 views; it used zero new fits and did not open Test or Kaggle.

| View | Midpoint MAE | Gain vs R06 (positive = lower MAE) |
|---|---:|---:|
| Primary, n=38,588 | 11.485671 | +0.016742 |
| Late W5+W6 | 10.067411 | +0.022616 |
| H12 pooled | 13.913031 | +0.031242 |

Four of six windows were nonnegative. The primary gain missed the frozen `+0.02` hurdle. Two low-tail subgroup guards also failed against R22: W1, n=205, harm `+0.427558`; H12 cutoff 55, n=880, harm `+0.381495`. Verdict remains **`NO_GO_MIDPOINT_SUBMISSION`**. No regression full-Train fit, CSV, or submission was prepared.

## QA and accounting

- Classification N02 independent verifier: `PASS_INDEPENDENT_ZERO_FIT_VERIFICATION`, 144 supported groups; verdict remains gate fail.
- ClassOnly retry synthetic tests: 2 passed; Python compilation passed.
- ClassOnly post-fit independent verifier: 13 checks passed; fit-state hash and Train fingerprint recomputed; routing, baseline, thresholded CSV, exact IDs, integer labels, and ledger checked.
- Regression midpoint verifier: `PASS`, zero fits, 12 bundles and 11 views.
- Fit ledger: 40 total official starts against cap 60; 39 saved, 1 failed (original persistence failure preserved); 1 technical retry consumed; no retry/interruption reset. 16 slots remain after reservations. No upload.
- Working branch is local only. No push, model promotion, Kaggle upload, or final-notebook change.

## Paths to supporting evidence

- Kaggle registry: `reports/submissions/reconciliation_20261102/REGISTRY.md` and `REGISTRY.json`
- N02 results and groups: `reports/optimization_v3/measured_benchmark_v1/CLASSIFICATION/N02_CONTINUATION_RESULTS.json` and `N02_CONTINUATION_GROUPS.csv`
- N02 independent verification: `reports/optimization_v3/measured_benchmark_v1/CLASSIFICATION/N02_CONTINUATION_VERIFICATION.json`
- ClassOnly preparation and QA: `CLASS_ONLY_RETRY_PREPARATION.json`, `CLASS_ONLY_RETRY_VERIFICATION.json`, and `CLASS_ONLY_RETRY_SYNTHETIC_PREFLIGHT.json` in the Classification report directory
- Regression midpoint protocol/result/verification: `experiments/optimization_v3/measured_benchmark_v1/reg_midpoint_protocol.json`, `reports/optimization_v3/measured_benchmark_v1/REG_MIDPOINT/result.json`, and `verification.json`
