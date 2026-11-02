# A01 weighted recency classification — final review summary

**Status:** `COMPLETE_REVIEW_REQUIRED`
**Frozen scientific verdict:** `NO_GO_GATE_FAILED`
**Campaign:** `CLASSIFICATION_RECENCY_A01`
**Review boundary:** stop after this A01 checkpoint. No continuation, Test inference, Kaggle submission, or other experiment was started.

## Question and frozen recipe

A01 tested whether recency weighting of the new sparse classification expert improves temporal Macro F1. At cutoffs 55 and 67, the candidate was compared against the frozen sparse25 route and a matched control whose weights were shuffled within class. Routing, preprocessing, feature allowlist, model settings, threshold 0.50, and source boundaries remained frozen in the A01 protocol.

For each cutoff, weights were defined as `max(0.25, 2^(-(cutoff-cycle)/8))`, normalized to mean 1 within each class. The shuffled control permuted those weights within class using seed 42. The candidate route retained the frozen 75% C2_B contribution and used 25% of the new expert when the stream was present; missing streams used the frozen CB fallback. Learned preprocessing and each new expert were fit only on rows at or before its cutoff.

| Cutoff | Fit rows | Temporal score rows | Class-0 ESS ratio | Class-1 ESS ratio |
|---:|---:|---:|---:|---:|
| 55 | 12,020 | 17,685 | 0.9545 | 0.9532 |
| 67 | 29,705 | 20,903 | 0.8035 | 0.8395 |

The observed score rows are later cycles within the official Classification Train, not official Test rows. Primary combines the frozen W1 and W4 views; H12 views cover the 12 cycles following each cutoff.

## Results

All entries are Macro F1. Deltas are candidate minus the named reference, in Macro F1 units.

| View | Cycles / origins | Rows | Recency | Sparse25 | Shuffled | Δ vs Sparse25 | Δ vs shuffled |
|---|---|---:|---:|---:|---:|---:|---:|
| Primary | W1 + W4; origins 55, 67 | 9,064 | 0.823163 | 0.823972 | 0.822968 | −0.000809 | +0.000196 |
| H12 at 55 | 56–67 | 17,685 | 0.795140 | 0.795032 | 0.794981 | +0.000108 | +0.000159 |
| H12 at 67 | 68–79 | 20,903 | 0.810430 | 0.808961 | 0.809000 | +0.001468 | +0.001430 |
| H12 pooled | 56–79 | 38,588 | 0.803814 | 0.802966 | 0.802961 | +0.000849 | +0.000853 |
| W1 | 56–59 | 4,114 | 0.802065 | 0.803160 | 0.800898 | −0.001095 | +0.001167 |
| W4 | 68–71 | 4,950 | 0.840841 | 0.841418 | 0.841453 | −0.000577 | −0.000613 |

The primary nonnegative-vs-sparse25 gate failed: recency was lower by 0.000809. The primary comparison against shuffled weights was positive (+0.000196) and passed its nonnegative gate. H12 pooled gains passed their frozen minima against both shuffled weights (+0.000853; minimum +0.0002) and sparse25 (+0.000849; minimum +0.0005). The cutoff-specific H12 and W1/W4 harm guards passed. These later-window gains do not override the failed primary gate. The overall frozen result therefore remains **NO-GO**.

All **48 supported protected subgroup comparisons** passed the legacy Macro F1 and positive-recall guards; there were **zero** guard failures and **zero** risk-review holds. This does not alter the primary-gate failure.

## Fit, source, and budget audit

- Exactly four official task estimators were fit, in the frozen order: cutoff 55 recency, cutoff 55 within-class shuffled, cutoff 67 recency, cutoff 67 within-class shuffled. All four were saved with model, fitted-state, and bundle receipts. Bundles are under `experiments/runs/optimization_v3_classification_recency_a01/fit55_recency/bundle`, `fit55_shuffled_within_class/bundle`, `fit67_recency/bundle`, and `fit67_shuffled_within_class/bundle`.
- Official Classification Train: 50,608 rows; SHA-256 `875552a68a3d24b0f3e1966d6ddb5671e6f54305a1c526e2f0a76747942ed91c`. Frozen protocol SHA-256 `8b0c2c7e79e14ab55f171ec00b3535bb6ba8f786162108e2d884ef891eef8b9a`; freeze signature `205aef7f76a07d8d0c6d4d839c255fe468216c6ef0c69e6d80cab970ade7a390`; code source commit `1367a3d48c7954956ea21cb62c8c80f6103b0465`.
- Independent verification status is **PASS** across all nine checks, including freeze/protocol identity, source and input hashes, Train-only boundary, exact four-fit accounting, weight/ESS reconstruction, bundle/receipt integrity, model prediction reproduction, primary/H12/window/subgroup/gate reconstruction, and append-only budget amendment. The verifier itself performed zero official estimator fits.
- `test_read=false`, `sample_submission_read=false`, and `kaggle_access=false` in the experiment and verifier evidence. No Test labels or SampleSubmission were accessed.
- The explicit amendment `A01_PILOT_REALLOCATION_2026_11_02` moved four authorized fits from control replay to pilot: effective caps are continuation 24, control replay 2, pilot 30, technical retry 4, global 60. A01 consumed exactly four pilot fits, all saved, with zero retries and zero unused A01 fits. Ledger-wide totals after A01: 54 starts, 52 saved, 2 failures from prior ledger activity; the four historical unused N04 slots were separately released. The append-only ledger has 142 event rows and SHA-256 `4157abd03cbad744350793ea33109154e813bb726fe0c27c346e971ad4ec538c`.
- The pre-existing C16 helper hash difference was reconciled as reservation-bookkeeping metadata-only drift in `reports/optimization_v3/batch1/RECONCILIATION_01.md`; the independent source-hash check passed. This did not change the A01 recipe or verdict.

## QA artifacts

- Primary computed result: `reports/optimization_v3/classification_recency_a01/results.json`
- Independent verification: `reports/optimization_v3/classification_recency_a01/independent_verification.json`
- Protected-group details: `reports/optimization_v3/classification_recency_a01/protected_groups.csv`
- Per-cycle diagnostics: `reports/optimization_v3/classification_recency_a01/per_cycle.csv`
- Official preflight: `reports/optimization_v3/classification_recency_a01/official_preflight.json` (`OFFICIAL_PREFLIGHT_PASS`)
- Synthetic preflight tests previously completed: 8 passed. No refit was used for verification or reporting.

**Conclusion:** A01 shows encouraging pooled H12 gains, especially after cutoff 67, but does not meet its frozen primary gate and has mixed window results. Keep it as a gate-failed diagnostic; do not promote it to a qualified candidate or change the frozen verdict.
