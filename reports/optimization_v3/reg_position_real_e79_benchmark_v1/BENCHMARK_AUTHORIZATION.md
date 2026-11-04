# REG_POSITION_REAL_E79_EXPLORATORY_BENCHMARK_V1

## Authorization

The user-authorized continuation allows this one finite exploratory benchmark
after the zero-fit transport diagnostic. This file records the bounded scope
before any production fit starts. It does not qualify the method, change a
scientific gate, or authorize upload by this worktree.

The selected arm is fixed to REAL from REG_POSITION_L1_V1 because its E75
result improved over R06. The E67 harm and pilot gate failures remain in force.
No BAG, SHUFFLED, threshold, alpha, shrinkage, degree, seed, or parameter
alternative may be fitted or selected from diagnostic outcomes.

## Two learned units

1. One target-free word TF-IDF fit on all 50,608 official Regression Train
   texts (cycles 47–79), using the exact resolved parameters in the frozen
   standalone E67/E75 pilot vectorizer receipts.
2. One LightGBMRegressor L1 REAL residual head with COMMON_PARAMETERS plus
   PARAMETER_SETS["regularized"]: 400 estimators, learning rate 0.05, 15 leaves,
   minimum child samples 80, L2 lambda 3, random seed 42, and two jobs.

Both starts are durably written to the isolated
experiments/runs/reg_position_real_e79_benchmark_v1/fit_ledger.jsonl before
each fit call. A failed start is consumed. Retries and additional learned units
are prohibited.

## Frozen training population and features

The head uses unique present-stream Train rows in cycles 56–79. Each donor
target is threat_rank minus the final mapped R06 OOF prediction, with strictly
past four-cycle cutoffs 55, 59, 63, 67, 71, or 75. The final mapped member is
the R06 cache prediction, which is checked bit-for-bit against the existing
RNBR E79 donor bank. raw_candidate, baseline_prediction, and RNBR outputs are
excluded as training features or prediction baselines.

REAL inputs are [B, P, C]: B is L2-normalized unigram TF-IDF, P is the existing
three-third position block built with the exact pilot tokenizer, shared IDF
and row normalization, and C is final R06 OOF divided by 100. No cycle,
Classification label/probability, query target, or Test value enters head
fitting. Both learned states and source, Train, OOF, donor, feature and target
hashes are saved in an immutable training freeze before the Test feature
reader can run.

## Fixed inference policy and boundaries

After training freeze, inference uses the manifest-bound main-tree R06 ORIGINAL
Test prediction. The fixed correction is clip(0.25 * head, -5, +5), then
clip(R06 + correction, 0, 100). Missing text uses the exact R06 prediction.
No rank map is applied.

The deliverable is one CSV with record_id and threat_rank, independently
recomputed and checked for 29,421 unique ordered IDs, exact numeric round trip,
finite values in [0,100], source identity, manifest and hashes, plus a duplicate
label-vector scan. Test labels and Classification data remain unread. No Kaggle
access or upload occurs here. The isolated Sol root owns any later upload.

The pilot verdict remains RETIRE_REG_POSITION_L1_V1 and failed gates remain
failed. The trained artifact status must remain
SCIENTIFIC_GATE_FAILED_EXPLORATORY_BENCHMARK. This benchmark is exploratory
and cannot be presented as qualified or as having a public score.
