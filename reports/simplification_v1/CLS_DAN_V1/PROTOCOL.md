# CLS_DAN_V1 — frozen pilot protocol

## Question and comparison

Does a train-only learned mean-unigram representation with a small nonlinear
classifier improve the existing classification champion on the frozen temporal
views? The pilot is exploratory and is not a model-selection authorization for
Test or Kaggle.

The candidate is a 32-dimensional learned unigram embedding, formed as the
mean of in-vocabulary unigram occurrences with multiplicity, concatenated with
the saved C2_B structured numeric/OHE block, followed by 64-ReLU, 32-ReLU and
one logit. Dropout 0.10 follows each hidden ReLU during training only. The
matched LINEAR control learns the same embedding and uses the same structured
block and optimizer, but maps the concatenated representation directly to one
logit. This control does not isolate nonlinearity causally: capacity and
regularization paths differ, and the learned embedding itself is a
reparameterized linear projection rather than a semantic representation.

The unigram vocabulary, token preprocessing, and tokenizer come from the
hash-verified A02 prefix vectorizer. The model only uses no-space unigram terms
in their original vectorizer-index order; no vocabulary or base preprocessing
is fitted by this pilot. C2_B structured features are the saved prefix
numeric/OHE block, with text TF-IDF and the final stream-missing indicator
excluded. Stream-missing rows are not used for auxiliary fitting and route
exactly to the saved champion.

## Frozen optimizer

- Four official predictor starts only: `E67_DAN`, `E67_LINEAR`, `E75_DAN`,
  `E75_LINEAR`. A failed or interrupted start consumes its slot; no retry.
- Float64 NumPy/SciPy implementation; 20 epochs, Adam (`lr=0.001`,
  `beta1=0.9`, `beta2=0.999`, `epsilon=1e-8`), batch size 256, reset
  `default_rng(42)` for each fit, stable fit-row order, one permutation each
  epoch. DAN inverted-dropout masks are drawn from the same generator in
  minibatch order after each epoch permutation.
- Embedding initialization `N(0, 0.05)`; all dense weights use He normal
  initialization by fan-in; biases are zero. BCE-with-logits batch mean plus
  `0.5e-4 * sum(nonbias_parameter^2)` over embedding and dense weights.
- Dropout masks use inverted dropout at rate 0.10. No schedule, early stopping,
  class weighting, seed search, epoch extension, or recipe rescue. Report
  initial/final deterministic no-dropout training loss; 20 epochs do not imply
  convergence.
- Every fit stores all parameters and Adam buffers. Synthetic QA is separate
  from the four official starts.

## Data and validation boundary

Official input is Classification Train only (50,608 rows, cycles 47–79,
frozen SHA-256 in `SOURCE_INPUT_FREEZE.json`). Fit populations are E67 cycles
47–67 and E75 cycles 47–75. Queries are E67 cycles 68–79 and E75 cycles
76–79. E67 W4/W5/W6 are cycles 68–71/72–75/76–79; E67 H12 is cycles 68–79;
E75 W6 is cycles 76–79. The primary unique view uses E67 predictions for
cycles 68–75 and E75 predictions for cycles 76–79 (20,903 rows). E75 training
overlaps the E67 query at cycles 68–75, so the views are development evidence,
not independent fresh holdouts.

Reference champion on present streams is `0.5 * original POS + 0.5 * A02`
at threshold 0.50. Missing streams use the saved A02 probability at threshold
`0.53230976129477`. Candidate and LINEAR control each use a fixed
`0.75 * champion + 0.25 * auxiliary_probability` on present streams. On
missing streams their probabilities and labels must be bit-identical to the
champion. The 25% contribution is frozen before scoring.

Model fitting may read labels only for its own frozen prefix. E75 training
includes cycles 68–75, which are also part of the E67 query window; those
values are recorded as E75 fit-prefix reads and are not used to score the E75
model. Evaluation-target reads are delayed until the source/input freeze and
prediction bundle are committed. The evaluator reads the E67 query union once
for the full metric comparison and the independent verifier reuses that
in-memory array. The overlap means cycles 68–75 are seen first for E75 training
and later for E67 validation; the primary unique view uses E67 predictions on
those rows. Test, SampleSubmission, full-Train refit/inference, and Kaggle are
out of scope.

Before each official fit: acquire the shared OS advisory lock, then local lock;
require at least 1 GiB available at start and 512 MiB after-fit reserve; limit
BLAS to two threads. A lock conflict is not a failed fit and consumes no slot.

## Predeclared gates

Scientific gate requires all of:

1. Primary DAN Macro F1 gain over champion is at least `+0.0005`.
2. E67 H12 and E75 W6 DAN gains over champion are both nonnegative.
3. Primary DAN-minus-LINEAR Macro F1 is at least `+0.0002`.
4. E75 LINEAR control gain over champion is nonnegative.
5. Every supported subgroup in every required view has DAN Macro F1 harm at
   most `0.01` and positive-recall harm at most `0.02` versus champion.
6. Identity, missing-route, finite-probability, metric-verifier and frozen
   input checks pass.

Exploratory eligibility is evaluated separately using the campaign rule:
`(primary_gain > 0 and E75_gain >= -0.0005) or (E75_gain >= 0.0005 and
primary_gain >= -0.0005)`, supported subgroup harm limits `0.02` Macro F1 and
`0.03` positive recall, and exact QA/missing routing. Mechanism evidence is
reported separately as the predeclared DAN-minus-LINEAR primary contrast;
there is no additional unregistered threshold. A failed scientific gate stays
failed. No result authorizes a Test/Kaggle action.

Supported subgroup cells require at least 200 rows and at least 30 examples of
each class. Report every supported and unsupported cell for stream availability,
entry friction, and inherited fit-only text-load quartiles. Report per-cycle
signs and all negative results as well as F1, confusion matrix, precision,
recall, ROC-AUC, average precision, log loss, and Brier score.
