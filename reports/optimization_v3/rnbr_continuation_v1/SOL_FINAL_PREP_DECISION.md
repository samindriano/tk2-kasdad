# Sol decision: one R-NBR origin79 fit and benchmark preparation

Decision: **GO for final CSV preparation**, not automatic champion promotion or
Kaggle upload. Sam delegated measured successor decisions and explicitly said
not to wait for further approval. This is a finite one-fit tranche.

## Evidence and counterevidence

Sol independently reconstructed the corrected primary, new-origin and late
metrics from origin-specific numeric bundles. Full-primary gain versus R06 is
0.220627 MAE (n=38,588), versus shuffle 0.229672. New-origin gain is 0.358767
(n=21,033); late gain is 0.148274 (n=15,953). Four of five learned primary
windows improve. The corrected report's frozen gates and supported subgroup
guards pass. The old evaluator overlaid E67 H12 predictions onto later primary
windows; its outputs remain superseded, not erased.

Counterevidence: W4 worsens 0.112556; inherited E67 H12 worsens 0.048271,
only narrowly within the fixed 0.05 tolerance. These repeatedly observed data
are development evidence. No public-score improvement is established. The old
R06 W1 missing-stream guard failure versus legacy remains failed. Incremental
PASS does not erase inherited risk or qualify a global champion.

## Exact authorized execution

Append an immutable amendment: global cap76->77 and continuation37->38;
pilot38, control_replay0 and technical_retry1 unchanged. Verify actual ledger
before reserving one unique origin79 observed target-memory fit. No retries,
refunds or extra base/geometry fits are authorized by this tranche.

Bank: present-stream Regression Train cycles56-79. Donor residuals use final
mapped R06 OOF forecasts: cutoffs55/59/63/67/71/75 for successive four-cycle
blocks56-59 through76-79. No Classification labels or predictions enter the
target bank. Query baseline is the existing original full-Train R06 Test
prediction artifact, verified against its original manifest.

Reuse target-free A02 fitted word-TFIDF/top32 cosine geometry. Before reuse,
verify official Regression and Classification Test IDs and the necessary shared
target-free feature values align; reorder only through explicit one-to-one IDs.
This comparison does not expand the frozen feature allowlist. If saved Test
neighbors are absent, transformation/search using the already-fitted state is
allowed; fitting a new transform is not. Missing prerequisite state must be
reported to Sol rather than silently refitted.

Keep cosine-squared weights, frozen weighted-median tie handling,
delta=clip(0.25*median,-5,5), final clip[0,100], no rerank. Missing-stream and
fewer-than-five-positive-neighbor rows use exact R06. Integrity failures are
errors, not fallback conditions. No recipe or scale tuning after Test inference.

Verified R06 provenance at decision time:

- Manifest: reports/submissions/optimization_v3_expanded_v1/R06_ORIGINAL/manifest.json
  SHA256 3f77e4ac148ada5fb79e5e2c16e7975e77726f4bfcc42a29f9234f801c697d67.
- Predictions: experiments/runs/optimization_v3_expanded_submissions_v1/R06_ORIGINAL/predictions.npz
  SHA256 ce034e82a7063ec5dd873b1126d60731dd291d3852c57936c7aa1ee449a81eba.
- CSV: submissions/optimization_v3_expanded_v1/r06_original.csv
  SHA256 4a72b13e0d6ac93e18adee23010bb15ddf15fcb2a9543dc903154d5af60ee64f.

## Verification and handoff

Freeze source/input/state hashes before bank construction. Exercise actual
safe-Unicode payload persistence/readback/resume synthetically before the fit;
never allow_pickle=True. Persist bank, raw scores and receipts before reports.
Independently check donor temporal boundaries, geometry alignment, median and
correction formula, exact missing fallback, ledger and deterministic replay.

Official Regression Test features/IDs and SampleSubmission schema are allowed
for this stage, never Test labels. CSV must have exactly29,421 rows, schema and
ordered IDs matching the official sample, unique complete IDs and finite
threat_rank in[0,100]. Manifest records all input/state/source hashes, recipe,
fit count and prediction/CSV hashes. Distinguish raw prediction precision from
CSV serialization. Do not report a Test MAE without labels.

One training job, two threads, 6GiB job ceiling and2GiB free reserve. No installs,
notebook edits, binaries/CSVs in Git, upload or push. Commit scoped preparation
code/reports only. Send results, counterevidence, QA, CSV path/hash, ledger and
commit to Sol before the executor's final reply. Stop after this handoff; Sol
will review the concrete artifact and decide manual benchmark readiness.
