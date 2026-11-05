# CLS simple present-threshold pilot

Status at source preparation: PREPARED_FOR_TARGET_FREE_PREFLIGHT. This is a
bounded calibrator study. It does not add a predictor, alter A02 probabilities,
fit a model, inspect Test, or select a Kaggle submission.

## Frozen decision

The authoritative recipe is
../tk2-simplification-audit-v1/reports/simplification_v1/CLS_SIMPLE_THRESHOLD_DECISION.md.
Its SHA256 is
e0e0ac69ce7ce0e148e8edfe2345e21e37addf55807cdcff6b2f2995a08cd0c5. Root
source commit a39721172b6d293cb68ee0d120060082d222699e must remain an ancestor
of the clean decision checkout; a later descendant HEAD is allowed.

- Keep cached canonical pA and the A02 missing decision fixed at
  pA >= 0.53230976129477.
- For present streams only, choose one global threshold in the inclusive
  interval [0.45, 0.55]; predict positive when pA >= threshold.
- Fit objective is full-row binary Macro F1, with fixed missing decisions
  included, labels [0, 1], and zero-division scores set to zero.
- Require at least 200 present rows of each class in a calibration population.
- Search every empirical decision state induced by present pA scores, using
  representable floating-point boundaries. Compare Macro F1 as exact rational
  values; resolve equal objectives by nearest threshold to 0.50, then higher
  threshold. The reported value must reproduce the stored confusion counts.
- Primary comparator is the same pA route at present threshold 0.50.
  Secondary comparator is 0.5*pP + 0.5*pA on present rows at 0.50 with the
  same fixed missing route. No positive-count quota is used.

## Frozen temporal stages

| Stage | Calibration data | Evaluation data | Rows |
|---|---|---|---:|
| D1 | E67 OOF, cycles 68–71 | E67 OOF, cycles 72–75 | 4,950 / 7,462 |
| D2 | E67 OOF, cycles 68–75 | E75 OOF, cycles 76–79 | 12,412 / 8,491 |

The two score windows are disjoint and contain 15,953 rows total. The final
candidate pool, if later considered by root, is 20,903 unique OOF rows from
E67 cycles 68–75 and E75 cycles 76–79. That final fit and any Test inference
are outside this source-ready milestone.

## Source and label boundaries

The source freeze hashes the exact official Train CSV, A02 OOF archive, root
decision, and committed producer/verifier/protocol/tests. Target-free preflight
reads only these official Train columns: record_id, cycle_code, signal_stream,
entry_friction, and text_load. It opens only OOF IDs, cycles, availability, pA,
pP, and three precomputed subgroup arrays. It never opens any *_y archive
member or parses artifact_signal values.

Preflight verifies row IDs and cycle assignments against official Train,
recomputes stream availability, entry-friction grouping, and prefix-defined
text-load quartiles, checks prediction ranges, and confirms D1/D2 support and
disjointness. Physical CSV lines must map one-to-one to parsed data records.

Official fitting requires a root-release JSON bound to the independently
verified SOURCE_READY.json. The later CLI reads artifact_signal only for IDs
selected into the current fit stage. It does not use cached OOF labels.
Scoring first writes and validates frozen candidate and comparator predictions,
then reads labels only for that stage's score IDs. A fit-start event without a
durable threshold receipt blocks retry. Frozen receipts and the append-only,
fsynced ledger make successful reruns idempotent. No code path reads Test or
SampleSubmission or contacts Kaggle.

## Result interpretation and gates

Report fit objective separately from each score-window objective. Include
full-row and present-row confusion counts, precision, recall and Macro F1;
per-cycle metrics; changed decisions; and availability, entry-friction and
prefix-defined text-load subgroups. Subgroup support is n >= 200 and at least
30 examples of each class; unsupported groups are UNKNOWN.

Qualification requires pooled Macro F1 gain at least 0.0003 over the fixed
simple threshold, nonnegative gain in each window, and no supported subgroup
harm above 0.01 Macro F1 or 0.02 positive recall. Exploratory review
requires pooled gain at least zero, each-window harm no worse than 0.001, and
the same subgroup guards. The windows are not fresh holdouts, and threshold
selection uses past OOF labels. A good result is not causal evidence and does
not alter the public POS/A02 champion: its observed public result remains
counterevidence, while no public-score tuning is permitted.

## Source-ready and later commands

After source commit and unit tests:

    python scripts/cls_simple_present_threshold_v1.py --train-csv C:\Users\Sam\Downloads\Train.csv --cache experiments\runs\optimization_v3_posoffset_a02_endgame_v1\policy_predictions.npz freeze
    python scripts/cls_simple_present_threshold_v1.py --train-csv C:\Users\Sam\Downloads\Train.csv --cache experiments\runs\optimization_v3_posoffset_a02_endgame_v1\policy_predictions.npz preflight
    python scripts/verify_cls_simple_present_threshold_v1.py

The current milestone stops after the independent source-ready receipt and
handoff to root. Do not create a root release or run fit/score until root
reviews the callback and issues that release. Later permitted commands are:

    python scripts/cls_simple_present_threshold_v1.py --train-csv C:\Users\Sam\Downloads\Train.csv --cache experiments\runs\optimization_v3_posoffset_a02_endgame_v1\policy_predictions.npz fit --stage D1 --release path\to\ROOT_RELEASE.json
    python scripts/cls_simple_present_threshold_v1.py --train-csv C:\Users\Sam\Downloads\Train.csv --cache experiments\runs\optimization_v3_posoffset_a02_endgame_v1\policy_predictions.npz score --stage D1 --release path\to\ROOT_RELEASE.json
    python scripts/cls_simple_present_threshold_v1.py summarize

Only D1 and D2 are valid pilot stages. The summarize command combines confusion
counts across the two disjoint score windows, reports qualification and
exploratory gates, and retains the complex comparator separately. There is no
final refit, Test inference, CSV generation, Kaggle upload, or automatic
champion change in this protocol.
