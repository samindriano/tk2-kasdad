# REG V2 diagnostic phase closure — R3

Status: `CLOSED_DIAGNOSTICS_ONLY`  
Frozen source commit: `7798e5a3c715ab2c4a53f9f28c38d6da5ac72504`  
Branch: `codex/reg-position-failure-mechanism-v1`  
Live report-only checkpoint at execution: `5dfb83ccf04b5d19509739f102e31f9a304d7713`

## Authorization and accounting

`ROOT_SOURCE_PASS.json` authorized exactly four LPs: `D1_GLOBAL`, `D1_RECENCY_H8`, `D2_GLOBAL`, and `D2_RECENCY_H8`. Its file SHA-256 is `d5d71092e2824596cc7416a22af9de8f074b2123dca36ebf1a7b661c6b8da10f`; its receipt digest is `296a2bae4ac9b8bc94e6a33a8eeab76fe196ede4b88d760412b23b88d411c034`. It binds R3 freeze SHA-256 `70b49ff7dea59edf53622106392c1c5d943b46dcf0f4e792f5211b83453061b8`.

The append-only ledger records one `RESERVE`, one `LP_STARTED`, and one `LP_COMPLETE` for each authorized fit; there are no failures or retries. Four of five allowed LP starts were used. No final production fit, candidate CSV, Test/Sample access, or Kaggle upload occurred. The independent verifier returned `PASS_INDEPENDENT_DIAGNOSTIC_RECOMPUTATION` for D1/D2 and 15,953 unique evaluation rows.

## Target access correction

The runner's short stdout line printed `official_labels_accessed:false` because the diagnostics result omitted that field and the CLI supplied a false default. The diagnostic receipt's Train audit is authoritative for what the run read: `query_target_rows_parsed=20903`, `selected_cycle_min=68`, `selected_cycle_max=79`, and `fit_target_rows_parsed=0`. The approved diagnostic stage did read the selected official OOF target values. The 50,608-row Train file hash matches the freeze. Test/Sample files were not accessed. Preserve the raw stdout in `DIAGNOSTIC_RUN_CONSOLE_R3.txt`; do not repeat its default boolean as the access finding.

## Results

| View | Fit rows | Score rows | Global `(a,b)` | H8 `(a,b)` | Direct | Global | H8 | Global + restore missing | Prior global, descriptive |
|---|---:|---:|---|---|---:|---:|---:|---:|---:|
| D1, cycles 72–75 | 4,950 | 7,462 | (2.100484, 0.975466) | (2.145108, 0.974377) | 11.478174 | 11.416864 | 11.417621 | 11.412228 | 11.416586 |
| D2, cycles 76–79 | 12,412 | 8,491 | (2.325035, 0.976773) | (2.311691, 0.977089) | 9.986419 | 9.959275 | 9.959656 | 9.945317 | 9.945810 |
| Unique pooled | — | 15,953 | — | — | 10.684186 | 10.641061 | 10.641617 | 10.631463 | 10.633764 |

The H8 pooled MAE is 0.000556 worse than causal global; its predeclared screen requires at least 0.02 gain, so H8 is retired. Per-window H8 deltas versus global are +0.000757 (D1) and +0.000380 (D2). Global-present-only improves by 0.009598 pooled, with improvements in D1 (0.004636) and D2 (0.013959); its predeclared screen passes.

Pooled per-cycle MAE:

| Cycle | Direct | Global | H8 | Global + restore missing | Prior global, descriptive |
|---:|---:|---:|---:|---:|---:|
| 72 | 11.610437 | 11.464657 | 11.465257 | 11.467583 | 11.462291 |
| 73 | 11.712575 | 11.664069 | 11.664612 | 11.658692 | 11.664283 |
| 74 | 11.728503 | 11.651972 | 11.653351 | 11.637383 | 11.651339 |
| 75 | 10.898564 | 10.917230 | 10.917760 | 10.915533 | 10.918738 |
| 76 | 10.086920 | 10.125869 | 10.126119 | 10.113290 | 10.100846 |
| 77 | 9.801615 | 9.743950 | 9.744535 | 9.722592 | 9.729778 |
| 78 | 10.229508 | 10.112048 | 10.111841 | 10.096503 | 10.128078 |
| 79 | 9.861528 | 9.900502 | 9.901347 | 9.894845 | 9.868485 |

Pooled subgroup support and MAE:

| Group | Rows | Global | H8 | Global + restore missing |
|---|---:|---:|---:|---:|
| Present | 14,920 | 10.430311 | 10.431005 | 10.430311 |
| Missing | 1,033 | 13.684998 | 13.683576 | 13.536771 |
| Target ≤ 5 | 804 | 9.878720 | 9.887977 | 9.803056 |
| Target ≥ 95 | 808 | 6.729227 | 6.744400 | 6.733082 |

All subgroup counts exceed 200. No D1, D2, or pooled supported missing/tail risk flag exceeds 0.25. The detailed receipt retains every subgroup metric and flag. These are reused temporal development labels, not a fresh holdout; exact full-OOF global coefficients are not heldout estimates.

## Frozen receipt and data hashes

| Artifact | SHA-256 |
|---|---|
| R3 prefit freeze | `70b49ff7dea59edf53622106392c1c5d943b46dcf0f4e792f5211b83453061b8` |
| R3 identity preflight | `639c080b91121d7d1183fc0de7f6d9f8cec2e68f496f4242cddd072dd4e5f211` |
| Diagnostic receipt | `136d4a8c7a565413ec550dca1212f7cda22a2c8ca5d70dbfdd6672ceb14961d8` |
| Append-only fit ledger | `961366c4b0e8ed998aa147b624a91fa05bae9a644b04cba9c5612eeafa9fc33b` |
| D1 diagnostic rows | `6258d2580a941cd9d76ce651b00e2ab912a05c056221979c82058e78b3603f2a` |
| D2 diagnostic rows | `b0cfe1b2c07db63f0deb81217877eace6396723994954ede5cd5b2c7a58a3d64` |
| D1 global state | `e0cb43aff7c99722041404e6db6666db387aaf4a937e248c51f12ba4d5293b06` |
| D1 H8 state | `c2990283a7a5d7ea0c49c7c72c875cdc5efd1ab14f9ba9360c358863874fba73` |
| D2 global state | `01db7b8973fa0488f0f37b134c6b99e9ae7003b88dc03fd2dbcccdd17245b53f` |
| D2 H8 state | `e937ab8e37cf6e2250d229819d42d190f1731bf90965a8fc7cc7f3ba6e081e9e` |

The independent verifier's exact stdout is preserved alongside the runner output in `DIAGNOSTIC_RUN_CONSOLE_R3.txt`. R1, R2, and R3 provenance remains intact. The R3 source checkpoint is closed; the B-only application uses a separate source and freeze namespace.
