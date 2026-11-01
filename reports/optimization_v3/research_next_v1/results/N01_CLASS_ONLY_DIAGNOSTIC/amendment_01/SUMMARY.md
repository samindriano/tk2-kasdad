# Research Next V1 — N01 class-only diagnostic amendment

**Status: `COMPLETE_DIAGNOSTIC_ONLY`.** This amendment is diagnostic evidence, not a qualified continuation candidate. It preserves the original N01 pilot verdict (`RETIRE`) and does not change any frozen gate, recipe, or historical report.

## Question and scope

This branch checks whether the already-tested class-only weighting recipe remains useful across the remaining four temporal origins after the all-view subgroup audit passed. The all-view audit authorized exactly four additional class-only task fits at cutoffs 59, 63, 71, and 75. It did not authorize a class-by-volume fit or any matched-control refit.

Each fit used only official Classification Train rows with `cycle_code <= cutoff`; each scored the next four cycles from the matching frozen C16 cache. The existing N01 class-only pilot bundles at 55 and 67 were reused. The route kept the frozen 0.75 C16 + 0.25 sparse expert blend when `signal_stream` was present and exact C16 CB fallback when missing; threshold remained 0.50. Learned structured and TF-IDF preprocessing was fitted on each origin's fit prefix only.

## Results

Macro F1 gain is candidate minus the cached C16 sparse25 route; positive values favor this diagnostic branch.

| View | Rows | Gain vs C16 sparse25 | Positive-recall gain |
|---|---:|---:|---:|
| Primary, W1–W6 unique rows | 38,588 | **+0.000687** | +0.010371 |
| Late primary, W5+W6 | 15,953 | **−0.000237** | +0.008822 |
| W1 | 4,114 | +0.003430 | — |
| W2 | 6,493 | +0.000798 | — |
| W3 | 7,078 | +0.000716 | — |
| W4 | 4,950 | +0.000666 | — |
| W5 | 7,462 | −0.001584 | — |
| W6 | 8,491 | +0.001000 | — |

The full-six primary Macro F1 was 0.830630 versus 0.829944 for C16 sparse25. Five of six individual windows were nonnegative, but W5 was negative and the pooled late W5+W6 result was also negative.

H12 values are inherited from the existing pilot bundles and are not additional continuation-origin fits:

| Reused H12 origin | Rows | Gain vs C16 sparse25 |
|---|---:|---:|
| 55 | 17,685 | +0.002161 |
| 67 | 20,903 | −0.000867 |

These two H12 views overlap the primary development rows and are not independent evidence for the new origins.

## Subgroup checks

All 64 supported group rows passed the frozen descriptive harm limits; there were no unsupported groups and no guard failures. The 64 rows cover `entry_friction` (16 groups), stream availability (16), and fit-defined `text_load` quartiles (32) across primary, late, and individual-window views. Support required at least 200 rows and at least 30 examples of each class. The largest observed Macro F1 harm was 0.00554, below the 0.01 limit; positive recall did not show harm in the supported groups.

## Interpretation

The pooled primary lift is small and positive, but it does not persist in the late W5+W6 view. Because this amendment ran no matched-control refits and is explicitly diagnostic-only, these results do not qualify the branch, replace the original N01 verdict, or authorize further continuation. The result should be reviewed alongside the original N01 matched-control result: class-by-volume was `RETIRE` after a −0.000731 primary Macro F1 contrast against its class-only matched pilot.

## Verification and accounting

- Four official estimator fits, one each at 59/63/71/75; zero failed or interrupted official fits; no retries.
- Budget before the amendment: 26 of 60 official fits consumed. Afterward: 30 of 60 consumed and saved; 26 capacity remains after reservations. This amendment reserved four continuation slots and released none as unused.
- Independent verifier: `PASS_INDEPENDENT_VERIFICATION`. It reconstructed all eight reported views and all 64 subgroup rows from official Train labels, frozen C16 caches, and saved bundles; it also checked source-row alignment, route/fallback math, fit-state weight hashes, bundle receipts, and ledger accounting.
- Relevant tests: 8 passed across the N01 adapter, all-view N01/N02 audit, and this diagnostic. Synthetic end-to-end also passed, including preprocessing, routing, immutable bundle read-back, post-publish interruption recovery without refitting, metric, subgroup guard, and report persistence. Synthetic fits are outside the official fit cap.
- Figure: `window_gains.png`; it was opened and visually checked. No Test/SampleSubmission, Kaggle API, full-Train refit, or notebook was accessed or changed.

The frozen configuration is [freeze.json](../../../../../../experiments/optimization_v3/research_next_v1/frozen/N01_CLASS_ONLY_DIAGNOSTIC/amendment_01/freeze.json), signature `e3ab556f6efcc0a1247dea9f643550b229054e318859fb79f698d5af32fe8f0b`. Exact input/source hashes, fit populations, per-window results, bundle receipts, and full group tables are in `preflight.json`, `results.json`, `groups.csv`, `per_cycle.csv`, and `verification.json` in this directory.

## Research Next queue checkpoint

| Lane | Official fits | Recorded result | Checkpoint interpretation and evidence |
|---|---:|---|---|
| N00 | 0 | `CACHE_AUDIT_AND_N03_FIXED_ARM_RECONSTRUCTION_PASS` | Cache/lineage audit; zero new official fits. [Report](../../N00/amendment_05/summary.json) |
| N01 original pilot | 4 | `PILOT_COMPLETE_RETIRE` | Preserve the original class-by-volume matched-control verdict. [Report](../../N01/summary.json) |
| N01 class-only branch | 4 | `COMPLETE_DIAGNOSTIC_ONLY` | Four authorized continuation-origin fits complete; no promotion. This report. |
| N02 pilot | 4 | `PILOT_COMPLETE_ADVANCE` historically | The later all-view audit found two supported H12 high-`text_load` positive-recall guard failures; continuation was withheld without rewriting the historical pilot verdict. [N01/N02 audit](../../N01_N02_GUARD_AUDIT/amendment_01/SUMMARY.md) |
| N03 | 2 | `RETIRED_PERFORMANCE` | No continuation. [Report](../../N03/amendment_01/SUMMARY.md) |
| N04 original | 4 domain fits | `SKIPPED_SCIENTIFIC_PREFLIGHT` | Preserve the original AUC/overlap/weight-stability screen; no task estimator was fit. [Report](../../N04/amendment_02/SUMMARY.md) |
| N04 diagnostic | 4 task fits | `DIAGNOSTIC_ONLY` | Reused the saved domain fits; amendment was set after domain AUC was seen, so it is not outcome-blind. It did not change N04's verdict. [Report](../../N04_DIAGNOSTIC/amendment_01/SUMMARY.md) |
| N05 | 4 | `COMPLETE_INFORMATION` | Information-only; no point-model promotion. [Results](../../N05/amendment_01/results.json) |
| N06 | 4 | `COMPLETE_INFORMATION` | Information-only; no continuation eligibility. [Results](../../N06/amendment_01/results.json) |

The phase ledger reconciles to 30 official estimator fits saved and zero failed/interrupted starts, including the four fits in this amendment. Synthetic estimator fits are excluded from the official fit cap under the Research Next contract.

**Stop point:** the authorized Research Next queue and QA are complete. Hold for review; no new experiment, final-Train fit, Test inference, Kaggle submission, or notebook finalization was started.
