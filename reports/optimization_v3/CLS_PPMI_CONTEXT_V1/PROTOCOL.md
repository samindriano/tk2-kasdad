# CLS_PPMI_CONTEXT_V1 protocol

This lane builds a from-scratch word-local context representation from the
official classification Train source. It keeps the existing REG campaign
paused and writes only under this CLS_PPMI_CONTEXT_V1 namespace, its named
core/runner/verifier/test files, and the top-level experiment `.gitignore`.

## Frozen inputs and row boundaries

- Official source: `TP 2/data/raw/classification/Train.csv`, SHA-256
  `875552a68a3d24b0f3e1966d6ddb5671e6f54305a1c526e2f0a76747942ed91c`.
- Origin E67 fits on cycles 47–67 and predicts cycles 68–79.
- Origin E75 fits on cycles 47–75 and predicts cycles 76–79.
- Each origin reuses its saved, hash-verified POS matrix manifest, C2_B
  structured state, A02 text vectorizer and original POS/A02 probability cache.
  The C2_B column order is structured block, 20,000 TF-IDF columns, then the
  missing-stream flag. The structured block is 284 columns at E67 and 370 at
  E75. Runtime checks replay the frozen TF-IDF block, missing flag, row IDs,
  cycles and manifest hashes before a representation fit.
- The saved A02 vectorizer's preprocessor and tokenizer are called separately.
  The unigram map uses frozen no-space terms in their original vectorizer
  column order. Its receipt digest is canonical JSON over `(term, original
  TF-IDF column index)` pairs; embedding rows may then use a compact reindex.
  No vocabulary, IDF, numeric, category, threshold, or model preprocessing
  state is newly fit.

## Representation recipe

- Fit only on present-stream documents in each historical prefix.
- Build symmetric directed distance-1 and distance-2 context counts, unit
  weight per direction, within each document. OOV tokens remain sentinels and
  block connections across the gap; documents never join.
- REAL uses observed token order. SHUFFLED applies one deterministic
  within-document permutation with `numpy.default_rng(42)` and stable row order.
  Both arms preserve the same per-document token marginals.
- Convert positive counts using
  `PPMI_ij=max(log(M_ij*N/(rowSum_i*colSum_j)),0)`. No smoothing, shifted PMI,
  pruning, or parameter search is allowed. Counts and PPMI remain sparse.
- Each arm fits `TruncatedSVD(n_components=32, random_state=42, n_iter=5)` on
  its prefix PPMI matrix. The document vector is the mean of known token
  embeddings by occurrence; OOV tokens are excluded, and rows with no known
  tokens get zero. Append only the frozen C2_B structured block.
- Require nonzero PPMI and at least 33 active rows and columns before its HGB
  fit. Sparse memory estimates are checked without allocating a dense
  vocabulary square.

## Fixed supervised fits and policies

Exactly four predictor fits are allowed, in order: E67_REAL,
E67_SHUFFLED, E75_REAL, E75_SHUFFLED. Each uses the same
`HistGradientBoostingClassifier` parameters:

`loss=log_loss, learning_rate=0.05, max_iter=200, max_leaf_nodes=15,
min_samples_leaf=30, l2_regularization=1, max_bins=255,
early_stopping=False, random_state=42`.

Present champion probability is the fixed equal mix of original POS and A02.
Candidate and shuffled control probabilities are fixed at `0.75 * champion +
0.25 * auxiliary`. Missing-stream rows use the exact A02 champion probability
and the frozen missing threshold `0.53230976129477`; present rows use threshold
`0.50`. No weight or threshold search follows evaluation.

## Label and metric boundary

Run the resource preflight after the source freeze and owned source files are
committed. It reads only `record_id`, `cycle_code`, and `signal_stream`, binds
to the freeze, and estimates the sparse and supervised peak memory for both
origins without reading labels. Commit a passing `PREFLIGHT.json` before the
runner can reserve or start a fit. A blocked or stale preflight prevents all
official fits.

Prefix labels may be read for the four fits after the committed source freeze
and passing, committed resource preflight.
Query labels are not read for feature preparation or prediction. The runner
writes unlabeled predictions, checks IDs/cycles, availability and exact missing
routing, and commits predictions plus the prelabel receipt before the sole
query-target read. The score report includes E67 W4/W5/W6/H12, E75 W6 (cycles
76–79), and the unique 20,903-row primary view using E67 cycles 68–75 and E75
cycles 76–79. Repeated development windows are not independent holdouts.

The scientific gate requires primary F1 gain at least 0.0005, E67 H12 and E75
W6 gains nonnegative, primary gain over the shuffled control at least 0.0002,
nonnegative E75 control gain, supported subgroup F1 harm at most 0.01, positive
recall harm at most 0.02, and exact QA/missing routing. The separate exploratory
gate uses the campaign's looser preregistered gain and subgroup thresholds.
The real/shuffled comparison changes learned geometry and SVD axes as well as
token order, so any separation is evidence for the full pipeline rather than an
isolated causal order effect.

## Execution guard

Execution order: `synthetic`, `freeze`, commit the frozen source and its owned
files, `preflight`, commit `PREFLIGHT.json`, `fit`, commit the fit ledger,
summary, unlabeled prediction artifacts, and prelabel receipts, then `evaluate`
and commit the postlabel report and verification. The freeze command is bound
to the preserved base HEAD; later gates require byte-exact committed files.

The shared `%TEMP%/kasdad_single_heavy_job.lock` uses a byte-zero OS advisory
lock; file existence, timestamps, and old PID text do not indicate ownership.
The runner acquires that advisory lock before the single four-fit reservation
and leaves the marker file contents intact. Only two numerical threads are permitted. Available RAM
must be at least 1 GiB at each start and at least 512 MiB after each step.
Started failures consume their slot and are never automatically retried. No
Test.csv, SampleSubmission.csv, external corpus, pretrained embedding, install,
upload, or Kaggle access is part of this lane.
