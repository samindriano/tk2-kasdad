# Regression shift diagnostics

Generated 2026-10-30T15:24:42.649695+00:00. Feature-only Test audit; no model fitting, Test target reads, Test predictions, or recipe selection.

## Zero-scope support

Train has 0/50,608 rows with any parsed scope count equal to zero; Test has 6,278/29,421 (21.34%).
All-three-zero: Train 0; Test 4,835. Test any-zero rows also below Train `scope_pressure` minimum: 6,278; pressure below that minimum overall: 6,283.

A parsed zero is distinct from a missing or parse-failed value. The course material describes scope fields but does not define zero semantics. The 6,278-row Test cohort has no labeled zero-count Train analogue; its source cause and `threat_rank` behavior are UNKNOWN. Do not impute or invent target values from this diagnostic.

Per-cycle counts below refer only to the provided Test file. They do not certify that the real-world disclosure/source universe is complete within each cycle.

| Test cycle | Provided rows | Any count zero | All three zero | Any count missing | Pressure below Train min |
|---:|---:|---:|---:|---:|---:|
| 80 | 1,060 | 0 | 0 | 245 | 0 |
| 81 | 1,618 | 15 | 10 | 348 | 15 |
| 82 | 2,474 | 446 | 340 | 562 | 446 |
| 83 | 2,823 | 796 | 614 | 661 | 797 |
| 84 | 2,763 | 746 | 581 | 640 | 746 |
| 85 | 2,338 | 469 | 352 | 563 | 469 |
| 86 | 2,616 | 455 | 336 | 597 | 456 |
| 87 | 2,472 | 289 | 221 | 593 | 289 |
| 88 | 2,242 | 241 | 184 | 519 | 241 |
| 89 | 3,051 | 682 | 518 | 705 | 682 |
| 90 | 3,317 | 1,068 | 844 | 771 | 1,071 |
| 91 | 2,647 | 1,071 | 835 | 604 | 1,071 |

## Saved R2 scope-ablation comparison

This reads the existing paired R2 result artifact only. Positive delta means removing scope/spread raised MAE on the same saved validation rows; this is conditional predictive contribution, not a causal effect and not Test-cohort error.

| View | n | Full R1-D MAE | No-scope MAE | Paired delta | Equal-cycle delta |
|---|---:|---:|---:|---:|---:|
| Primary | 38,588 | 13.349836 | 13.391396 | +0.041560 | +0.044521 |
| H12 | 38,588 | 15.577240 | 15.616123 | +0.038883 | +0.051353 |

The primary and H12 views reuse validation records, and all windows were development evidence. The ablation does not establish how either feature set behaves on the novel Test support.

## Train-only feature relationships

Pairwise correlations, EF summaries, and impact-bin support are computed from predictor columns only. `threat_rank` was not loaded. These are descriptive associations, not mechanisms or outcome results.

## Source and limitations

Course source copy: `../TK1/TP 1 KASDAD Ganjil 2026_2027.pdf`, pp. 16–17; SHA-256 `349860738eae2dc5ca2e515da819f1583b28e6a2f3160bd70a21f729947f9da0`. The tracked audit identifies a different 57-page PDF (`TP KASDAD Ganjil 2026_2027.pdf`, SHA-256 `386fa63541cc3e9b36b56ee8f8f6c479b3bbaa7753b9482d60763326f4465c0a`), so the local copy is not represented as byte-identical to that audited source. Both descriptions remain course context only; neither documents zero semantics or an authoritative TP2 target-generation formula.

Recommended controlled Stage 4 option: keep the frozen full R1-D and no-scope/spread representations as paired controls if MAIN elects a next stage. No-scope is only a robustness comparator; the current report does not authorize a feature decision, a test correction, or a new model choice.

Frozen diagnostic identity: `89cf27f528270d505c07f39f3a91b3090ed4c8d327d9cff9f64148339ce043c1`. Train/Test source hashes are recorded in the JSON artifact.
