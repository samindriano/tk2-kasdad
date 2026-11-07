# CLS_PPMI_CONTEXT_V1 correction protocol — CORRECTION_R1

This namespace corrects scoring/reporting and target-read accounting for the
already fitted pilot. It permits **zero** additional representation or
predictor fits and one additional read containing exactly the 20,903 unique
primary target values authorized by
`ROOT_CORRECTED_SCORING_R1.json`.

## Immutable inputs

- Preserve the original source freeze, predictions, prelabel receipts,
  `RESULTS.json`, `REPORT.md`, and `POSTLABEL_VERIFICATION.json` byte-for-byte.
- Require their hashes to match the root authorization before preparing the
  intent and again before the selected read.
- Read labels only for source rows and record IDs derived from committed
  `predictions_E67.npz` / `predictions_E75.npz`: E67 cycles 68–75 plus E75
  cycles 76–79. The union must be exactly 20,903 unique IDs and source rows.
- The original evaluation materialized 50,608 target values in one full-column
  read. The correction materializes 20,903 selected target values in one
  `read_csv` call. Metric verification reuses that in-memory vector; there is
  no verifier target reread. `Test.csv` and `SampleSubmission.csv` are out of
  scope.

## Scoring contract

- MacroF1 is the unweighted mean of class-0 and class-1 F1, equivalent to
  `f1_score(..., labels=[0,1], average="macro", zero_division=0)`. Retain
  positive-class F1 as a separate diagnostic.
- Use E67 W4 cycles 68–71 (4,950 rows), W5 72–75 (7,462), W6 76–79 (8,491),
  H12 68–79 (20,903), E75 W6 76–79 (8,491), and PRIMARY_UNIQUE from E67
  68–75 plus E75 76–79 (20,903 unique rows).
- Compute availability, entry-friction, and inherited fit-only text-quartile
  subgroup metrics separately in all six views. A subgroup is supported when
  it has at least 200 rows and at least 30 examples in each class. Report all
  support counts and aggregate worst supported MacroF1/positive-recall harm
  across views.
- The E75 mechanism gate compares candidate REAL directly with SHUFFLED
  (`candidate_real_minus_control_shuffled_macro_f1`). Keep primary
  candidate-minus-champion/control, E67 H12, E75 W6, and exploratory
  thresholds from the committed root campaign unchanged.
- Missing rows preserve exact champion probabilities and the frozen missing
  threshold. No candidate, threshold, weight, or fit change is permitted.

## Commit and read order

1. Add the corrected scorer, separate in-memory verifier, tests, source-only
   fit/representation audit, synthetic QA, and `CORRECTION_INTENT.json`.
2. Run the correction unit suite and synthetic lifecycle without opening the
   official target column. Commit those source/QA/intent files. The intent
   binds the root authorization, historical artifact hashes, prediction
   hashes, selected source rows/IDs, and the exact one-read plan.
3. Run the one-shot `score` command. It first writes a one-shot attempt marker,
   then makes one selected-row target read, aligns IDs, scores all views, and
   independently recomputes every metric, subgroup, gate, and root anchor from
   the same in-memory vector. Do not rerun `score` if an attempt marker exists.
4. Commit only the correction outputs. Run the structural `verify` command;
   it checks committed bytes, signatures, historical hashes, and read-count
   receipts without opening the target column.

The original report remains as history and is labeled invalid by the
correction output. Corrected metrics determine the gate status, while the root
retains the family disposition and any next-lane decision.
