# C07 — Category frequency/rarity pilot

**Decision: RETIRE.** The fit-only category-frequency features passed the supported subgroup guards, but their aggregate primary Macro F1 did not meet the frozen gain requirement.

## Frozen recipe and execution

C07 appended six features to the accepted sparse classifier: `log1p(fit-prefix category count)` and `fit-prefix count < 30` for each of `org_code`, `module_code`, and `pattern_code`. Missing categories used the sparse model's dedicated missing key; unseen categories received count 0. Counts used only the relevant temporal training prefix. No target rates were used. The sparse model, TF-IDF, `.75 C2_B + .25 candidate` blend for present streams, frozen CatBoost route for missing streams, and 0.50 threshold stayed fixed.

Exactly two predictor fits ran: cutoff 55 used 12,020 rows and scored the next 12 cycles (17,685 rows); cutoff 67 used 29,705 rows and scored the next 12 cycles (20,903 rows). Fit times were 7.606 s and 17.926 s, totaling 25.532 s. All six frequency features varied in each fit prefix. The preflight estimate was 1.606 GiB under the 6 GiB cap; observed peak RAM was not recorded. Frozen signature: `07852ccd497cc7aa4018a6d14a9ec788d97790ba2c8326261e46b656fdf19e5a`.

## Pilot results

Positive gain means candidate Macro F1 minus frozen-control Macro F1. The primary view is W1 + W4.

| View | Rows | Candidate Macro F1 | Control Macro F1 | Gain |
|---|---:|---:|---:|---:|
| W1 | 4,114 | 0.804062 | 0.803160 | +0.000902 |
| W4 | 4,950 | 0.840333 | 0.841418 | −0.001085 |
| Primary (W1 + W4) | 9,064 | 0.823791 | 0.823972 | **−0.000181** |
| H12 | 38,588 | 0.803182 | 0.802966 | +0.000216 |

The primary gate requires at least +0.0005. Only the primary threshold failed: H12, W1/W4 harm, independent metric/route reconstruction, and all supported primary subgroup guards passed. C07 therefore does not proceed to continuation.

### Descriptive novelty splits

Novelty categories are descriptive in this card, not additional numeric gates. On primary rows, unseen `org_code` had +0.001484 Macro F1 difference (`n=2,449`), unseen `module_code` −0.000390 (`n=4,113`), and unseen `pattern_code` had `n=78` with zero observed difference. These mixed splits do not establish a general benefit for rare or unseen categories.

## Verification and limits

The independent verifier passed. A separate row-level check confirmed exact ordered `record_id` alignment for both 12-cycle score blocks, exact reconstruction of present/missing routing, finite probabilities in `[0,1]`, threshold-0.50 Macro F1, and the W1/W4/primary/H12 metrics. The predeclared subgroup gates use support `n ≥ 200` and at least 30 examples from each class; unsupported or category-novelty groups remain descriptive.

Two identical scikit-learn feature-name warnings appeared in runner output during prediction; the fit receipts do not persist warning counts. The scored outputs and routing passed verification. The temporal development windows are reused and are not an untouched holdout. No Test, SampleSubmission, or Kaggle result was used.

Machine-readable results are in [`results.json`](results.json), the frozen recipe is [`recipe.json`](recipe.json), and the source/input hashes are in [`freeze.json`](../../../../experiments/optimization_v3/batch1/C07/freeze.json).
