# R02 — Wider word vocabulary pilot

**Decision: RETIRE.** The wider vocabulary improved every aggregate pilot view, but the primary gain did not meet the frozen minimum.

## Frozen recipe and execution

R02 changed the inherited sparse LightGBM regression recipe only by increasing the TF-IDF `max_features` cap from 20,000 to 60,000. It retained the fixed `0.75 × Ridge25 + 0.25 × within-cycle rank50(sparse prediction)` policy, clipped to `[0, 100]`. Rank ties use average ranks. Transforms were fit separately on each cutoff's training prefix.

Exactly two predictor fits ran: cutoff 55 used 12,020 rows and scored the next 12 cycles (17,685 rows); cutoff 67 used 29,705 rows and scored the next 12 cycles (20,903 rows). Fit times were 12.619 s and 27.232 s, totaling 39.852 s. The resulting TF-IDF vocabularies had 45,374 and 60,000 terms; the later fit reached the widened cap. The frozen signature is `6a485f4c4ec1b0d157e759a31faebab9b7765e63ac7c49cf125334767a743d87`.

## Pilot results

Positive gain means lower candidate MAE than the frozen control. Primary is W1 + W4, the pilot's 9,064-row comparison; it is not the queue's full-six primary.

| View | Rows | Candidate MAE | Control MAE | Gain |
|---|---:|---:|---:|---:|
| W1 | 4,114 | 15.834668 | 15.882112 | +0.047444 |
| W4 | 4,950 | 11.453793 | 11.455005 | +0.001212 |
| Primary (W1 + W4) | 9,064 | 13.442200 | 13.464396 | **+0.022196** |
| H12 | 38,588 | 14.037448 | 14.047533 | +0.010084 |

The primary gate requires at least +0.03 MAE gain. R02 falls short by about 0.007804. H12, W1, W4, and all supported subgroup guards pass, so the primary threshold is the sole reason for retirement.

### Subgroup diagnostics

Values are candidate-minus-control MAE harm; negative values favor R02. The subgroup numeric guard applies to stream-missing rows and target tails (`threat_rank ≤ 5` or `≥ 95`), with minimum support 100.

| Group | Primary n | Primary harm | H12 n | H12 harm |
|---|---:|---:|---:|---:|
| `entry_friction = 0` | 6,131 | −0.011042 | 26,048 | −0.003711 |
| `entry_friction = 1` | 2,933 | −0.045513 | 12,540 | −0.023322 |
| Stream present | 8,330 | −0.021438 | 36,010 | −0.009407 |
| Stream missing | 734 | −0.030803 | 2,578 | −0.019544 |
| Target ≤ 5 | 449 | +0.045743 | 1,928 | −0.025782 |
| Target ≥ 95 | 462 | −0.076443 | 1,955 | −0.049659 |

All supported tail and missing-stream harms are below the frozen +0.25 limit. No group was dropped from the reported diagnostics.

## Verification and limits

The independent verifier passed. A separate row-level check confirmed exact ordered `record_id` alignment, disjoint fit and score rows, frozen-control reconstruction, the within-cycle rank50 mapping, finite `[0,100]` predictions, and all reported MAEs and subgroup denominators. The preflight peak-memory estimate was 1.602 GiB under the 6 GiB cap; observed peak RAM was not recorded.

Two identical scikit-learn feature-name warnings appeared in the runner output during prediction. Fit receipts do not persist warning counts; all row, prediction, policy, and metric checks passed. No Test data or Kaggle results were used. These temporal development outcomes are repeatedly reused and are not an untouched holdout estimate. Because the frozen primary gate failed, R02 does not proceed to continuation.

Machine-readable results are in [`results.json`](results.json), the post-fit recipe summary is [`recipe.json`](recipe.json), and the authoritative freeze and source/input hashes are in [`freeze.json`](../../../../experiments/optimization_v3/batch1/R02/freeze.json).
