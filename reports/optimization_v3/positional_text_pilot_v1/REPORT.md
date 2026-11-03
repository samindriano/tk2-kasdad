# POSITIONAL_TEXT_PILOT_V1 — preregistered execution record

This file freezes the pilot question, model arms, folds, gates, data boundary,
fit accounting, and interpretation rules before any pilot fit or score-label
inspection. Executed metrics and receipts are recorded separately after the
pilot completes.

## Question and design

Does adding coarse first/middle/final-third unigram position information to the
existing C2_B hybrid classifier improve temporal Macro F1 beyond both the
faithful unordered bag model and a deterministic order-shuffled control?
This is a narrow mechanism pilot, not a new model search, full-six
qualification, or champion-selection exercise.

Three Logistic Regression arms per origin use identical expanded structured
features, the existing mixed unigram/bigram TF-IDF representation, the same
fit-only structured state, scaler, OneHotEncoder, and TF-IDF vocabulary:

1. `bag`: faithful unordered C2_B.
2. `position`: `bag` plus three position blocks over unigram entries already
   present in the shared C2_B mixed vocabulary.
3. `shuffled_position`: `bag` plus blocks built from the same tokens after the
   deterministic seed-42 text-digest shuffle.

Within each third, token counts use `(1 + ln(count)) * shared_idf`. The three
blocks are concatenated and L2-normalized once together per row. The
unordered bag/bigram and structured portion is unchanged, and no later scaling
or whole-matrix normalization is applied. The shuffle preserves token bag,
length, and exact third sizes; it changes only the positional input.

The fourth shared expert is the existing C1_B expanded structured
HistGradientBoosting recipe. At inference, rows with an empty normalized stream
route to this one C1_B expert in all three routes. Present-stream rows use the
corresponding LR model. The fixed threshold is `probability >= 0.50`.

## Frozen folds and score views

- **E67:** fit cycles 47–67; score cycles 68–79. Report W4 (68–71), W5
  (72–75), W6 (76–79), and pooled `H12_E67` (68–79).
- **E75:** fit cycles 47–75; score cycles 76–79. Report W6 as history
  sensitivity. Its W6 overlaps E67 W6 and is not an independent replication.

Only the official Classification Train feature allowlist and fit-prefix
`artifact_signal` labels are used. `record_id` is for alignment and integrity
checks only. No Test, SampleSubmission, Kaggle, Regression target, A02
residual, or external source is opened. Outer labels are opened only after
both origin predictions and their matrix, row, route, probability, and
independent prelabel verifications are durable and passing.

## Frozen gates

- E67 `H12_E67` Macro F1 gain for position is at least `+0.003` over **both**
  bag and shuffled-position.
- E75 W6 position Macro F1 is no worse than either control.
- E67 position Macro F1 harm is no greater than `0.005` versus either control
  in each of W4, W5, and W6.
- Fit-defined text-load Q4 is defined from the finite E-origin fit-prefix
  75th percentile. Where Q4 has at least 200 rows and at least 30 rows of each
  class, position's Macro F1 harm must be no greater than `0.01` and positive
  recall harm no greater than `0.02` versus each control in each reported
  aggregate/window. Unsupported Q4 groups are explicitly reported and are not
  silently treated as passing; any unsupported declared Q4 view makes the
  overall result incomplete rather than a PASS.
- All output alignment, class order, finite probabilities, exact missing
  routing, fit-only transform provenance, fit receipts, completed accounting,
  and independent prelabel/metric checks must pass.

Macro F1 is primary. Precision, positive recall, confusion matrix, ROC AUC,
average precision, log loss, and Brier score are descriptive. All per-cycle,
entry-friction, stream-availability, and text-load support is disclosed.

**PASS** advances the positional mechanism to review only. It does not qualify
the candidate for the full-six set and does not replace the champion. **FAIL**
retires this exact three-third recipe; no bins, seeds, C values, vocabularies,
or block weights are swept as rescue work.

## Budget and resource plan

Exactly 18 new fit units are reserved if the frozen live-ledger preflight
matches: 10 shared preprocessing units and 8 predictor fits across two
origins. Each origin has one C2_B structured state, scaler, OHE, TF-IDF,
one C1_B structured state, three LR estimators, and one HGB estimator. No
per-arm preprocessing refits or retries are allowed. The authorized cap
amendment changes global cap 87 to 105 and pilot cap 46 to 64; all other
category caps remain unchanged.

One job at two threads; maximum process RSS 6 GiB; at least 4 GiB available RAM
at initial preflight and at least 2 GiB available reserve throughout. The
runtime lock prevents concurrent pilot execution. Synthetic lifecycle and
resource-independent QA run in a temporary directory and ledger before and
after the official pilot.

## Provenance, result, and limits

The recipe is pinned in [`RECIPE.json`](RECIPE.json), and Sol's exact
authorization is retained in [`SOL_PILOT_AUTHORIZATION.md`](SOL_PILOT_AUTHORIZATION.md).
Source/input hashes, the clean source commit, frozen recipe, live ledger
snapshot, and resource preflight are recorded in the ignored runtime bundle.
The final committed result report is added only after execution and
independent QA; this preregistered file is not edited in response to results.

The two origins, overlapping W6 window, and one deterministic shuffle do not
provide independent replication or randomization inference. A passing pilot
supports only this controlled recipe for further review; a failed pilot does
not establish that all possible positional representations are useless.
