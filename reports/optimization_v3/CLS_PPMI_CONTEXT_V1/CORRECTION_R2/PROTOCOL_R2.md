# CLS_PPMI_CONTEXT_V1 scoring recovery protocol — CORRECTION_R2

This root-authorized recovery follows the failed one-read `CORRECTION_R1` pass.
It permits zero fits and exactly one further selected read of the same 20,903
frozen primary target rows and IDs. The R1 intent, consumed attempt, and failed
verification record remain immutable evidence.

## Frozen scope

- Use the committed R1 scorer and frozen E67/E75 predictions, source rows,
  thresholds, subgroup support rules, candidate probabilities, and fit outputs.
- Do not fit, refit, alter predictions, change thresholds or support criteria,
  read Test/Sample/Kaggle, or retry the R2 read after its attempt marker exists.
- Commit the R2 verifier, runner, tests, protocol, synthetic QA, R1 failure
  record, root recovery receipt, and exact selected-row intent before reading
  any R2 target values.
- The separate recovery receipt binds the root decision message, frozen source
  and prediction hashes, R1 intent/attempt/failure hashes, and corrected source
  hashes. R2 intent binds the receipt, exact selected rows/IDs, and source set.

## Anchor verification

The production anchor schema has `champion`, `candidate_real`,
`control_shuffled`, and `gain_real_vs_champion` for PRIMARY_UNIQUE, E67_H12,
and E75_W6. The three model anchors compare against their MacroF1 values. The
derived gain anchor compares against
`candidate_real_minus_champion_macro_f1`. Unknown or mismatched anchors fail
closed. Independent verification recomputes every metric, row/ID identity,
subgroup, missing route, overlap check, and gate from the one in-memory R2
target vector.

## E75 gate semantics

The original ROOT_CAMPAIGN gate is
`control_shuffled_minus_champion_macro_f1 >= 0`. The R1 correction authorization
later amended the E75 mechanism comparison to
`candidate_real_minus_control_shuffled_macro_f1 >= 0`. R2 reports these
comparisons and their gate statuses separately; it uses the original gate for
the original-campaign scientific disposition. Exploratory eligibility remains
unchanged.

## Read accounting

Report all three evaluation passes: original full-column read of 50,608 target
values, failed R1 selected read of 20,903, and R2 selected read of 20,903.
Total materialized target values are 92,414 across three target-column read
calls, covering 50,608 unique Train rows. The independent verifier uses the
R2 in-memory vector and makes zero additional target reads. Persist no target
values in results.

## Recovery lifecycle

1. Record the failed R1 read and verifier error without copying labels.
2. Run the complete synthetic score, production-schema anchor checks,
   mismatch/unknown-anchor rejection, all six view guards, pre-read commit
   guard, and signed result serialization round trip.
3. Prepare the root recovery receipt and R2 intent without target values.
4. Commit all recovery sources, synthetic QA, failed R1 evidence, receipt, and
   intent. Verify exact committed bytes and frozen source hashes.
5. Consume one selected target read. Reuse its in-memory vector for all
   independent verification. A failure after the marker ends the attempt.
6. Commit only R2 outputs, then run structural verification without labels.

The corrected metrics determine the corrected gate readout. The root retains
family disposition and the next-lane decision.
