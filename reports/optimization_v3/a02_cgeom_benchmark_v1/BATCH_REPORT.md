# A02/CGEOM three-candidate E79 preparation report

**Status: `PREPARED_FOR_BATCH_REVIEW_NOT_SUBMITTED`. No Kaggle API, CLI, browser, or upload was used.**

## Prepared CSVs

| Candidate | CSV | SHA-256 | Label count (0 / 1) | Changed labels vs A02 |
|---|---|---|---:|---:|
| CGEOM_ALIGNED | `submissions/optimization_v3/a02_cgeom_benchmark_v1/classification_cgeom_aligned_e79.csv` | `edcf3e1289498af5d37b475c23eb7bf087f357c8119e27f3a6b9ed02c91142b8` | 22711 / 6710 | 30 |
| JOINTPOS | `submissions/optimization_v3/a02_cgeom_benchmark_v1/classification_jointpos_e79.csv` | `76be1f7b31a4e252141eb464504977da63284d0248e6d378263532a5fad09fe9` | 22597 / 6824 | 890 |
| POS_BAG_OFFSET | `submissions/optimization_v3/a02_cgeom_benchmark_v1/classification_posoffset_e79.csv` | `3c7cf1aadcbf3e234264a6080b4c00416660c90a343cd73728dd0e978f8bc945` | 22531 / 6890 | 836 |

All three files have 29,421 rows, columns `record_id,artifact_signal`, unique ordered Test IDs identical to SampleSubmission, and integer labels 0/1. The Test target column was not read. Label changes and candidate disagreement counts are descriptive integrity summaries only; they were not used to select a recipe.

## Frozen recipes and historical evidence

- **CGEOM_ALIGNED** preserves the exact aligned CGEOM recipe and Sparse25/A02 fallback. Its historical verdict remains `CLOSED_GATE_FAIL`: E75 W6 was −0.000536338 Macro F1 versus A02 and −0.000245142 versus the permuted arm; all 52 subgroup checks passed. This preparation does not convert that result to a pass.
- **POS_BAG_OFFSET** preserves the real-position third-block recipe, C2_B fixed logit, original sum-loss gamma objective, C=1 and L-BFGS-B options, and C1_B missing-stream fallback. The historical E67 H12 improvement of +0.001151 remained below the +0.003 threshold; the original E75 W6 result and gate remain unchanged.
- **JOINTPOS** preserves the original real-position joint LR recipe and C1_B missing-stream fallback. Historical E67 H12 was +0.000889 versus bag and E75 W6 was −0.000246; its previous verdict is unchanged.
- No test labels or public scores were used. No candidate is declared champion or qualified based on the generated files.

## Fit and source accounting

Both prior C2_B LR failures remain failed and consumed. The six fresh integration IDs have receipts, while the three original preprocessing states were reused without refitting and retain their original source commits and hashes. The five old unstarted fit IDs remain released. The shared ledger ends at 129 starts / 123 saved / 6 failed / 0 pending; the six failures comprise four earlier historical failures and the two preserved LR failures. Global cap is 132; continuation cap is 49; other category caps are unchanged.
The six-ID amendment changes the active fit-ID mapping only; model settings, preprocessing, threshold, and candidate recipes remain frozen. The A02 E79 vectorizer, A02 residual bank, original A02 predictions, Sparse25 baseline, and three already saved preprocessing states were reused with their original source lineage. C16 cached predictions were not treated as fitted estimator states.

## Verification

- Independent output and fit-accounting verification: `PASS_THREE_CSVS_AND_SIX_ID_INTEGRATION_BUDGET_VERIFIED`.
- Original failed fit ID preserved: `A02_CGEOM_BENCHMARK_V1_E79_C2B_LR` (event SHA `666a4382c6fa42a3ab49c5208ef5954589a8952c716560543d906ff9bda2de1f`); prior hash-fix failed ID preserved: `A02_CGEOM_BENCHMARK_V1_E79_C2B_LR_HASHFIX_R1` (event SHA `be2d8c7171608464c018a99498a2baa27a4a2ef6b358e1faa2d058fcdd18f9a5`).
- A02 row-position digest algorithm: `SHA256(raw contiguous little-endian int64 bytes), via classification_a02_neighbors._array_sha256(rows, '<i8')`; stored and recomputed digest `b3412a08e9f0877cc37ef9615479de0d182a21b9806068c2d576ed06ae495bb2`.
- Independent Test prediction equation/routing audit: `PASS`; checks include offset and joint sigmoid equations, missing-stream routes, and CGEOM baseline-plus-correction reconstruction.
- A02 E79 Test prediction replay: bit-exact; probability SHA `a677df2fbec4bc5b9bab58b348ef2d75961798d33c395585d8a228d68a692447`; label SHA `7fde0af7c6c5d491e12838d5427a767c6def25829ff288285f57fecd91b8dfd9`.
- Inference-freeze source commit: `ec3a16d3a2fd89858fd0c77fd1aa7ced02a8ee8a`.
- Each fit receipt separately records its preparation commit and source hashes. The first structured state fit predates a runner-only resume-accounting correction; model recipes and learned fitting code are unchanged.
- Trained-state freeze SHA-256: `ab0307e6e1c9778ed0f091a3d6ed55e0ac8705ae31b9cb29b9f083e46b5c2b82`.
- Prediction NPZ SHA-256: `ffc55aa2d18b75438257f3cf0aa205baa3f4c2922180e2b3688b36c352c00271`.
- Test target loaded: false. Kaggle contacted: false. Upload: none.

## Pairwise label disagreement counts

- `CGEOM_ALIGNED_vs_JOINTPOS`: 892 rows.
- `CGEOM_ALIGNED_vs_POS_BAG_OFFSET`: 836 rows.
- `POS_BAG_OFFSET_vs_JOINTPOS`: 244 rows.

These are prediction agreement diagnostics only, not model evaluation. Stop for review before any upload or further experiment.
