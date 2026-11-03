# CGEOM aligned: one explicitly authorized public benchmark

GO_PREPARE_AND_SUBMIT_ONE_CLASSIFICATION_BENCHMARK.
Sam explicitly requested selecting a narrow gate failure and submitting it:
"COBA PILIH YG TIPIS. DAN KITA AKAN SUMIT". This authorizes the concrete
benchmark below despite failed qualification. No gate is changed into PASS.
Incoming main HEAD674b6dd5609639a6d768c405b462bf7a2b94dc06, clean.

## Selection and preserved risk

Selected: exact CGEOM_V1 ALIGNED recipe, not permuted, not a new blend.
H12 E67 gain vs A02 about+.000661,vs permuted+.000690; E67 W4/W5/W6 positive.
E75 W6 gain vs A02-.000536338,vs permuted-.000245142. All52 subgroup checks
passed. The two E75 failures remain material counterevidence and historical
CLOSED_GATE_FAIL remains unchanged. This is a post-outcome benchmark exception,
not continuation qualification, an estimate of private ranking or a score promise.
Position-offset gains are relative to C2_B, not established relative to A02;
MLP/FM have stronger negative evidence. Already-submitted B01 and C19 are not
new benchmark candidates. No threshold/shrinkage/weight/neighbor/seed search.

## Exact final recipe and data

Refit/reuse faithful C2_B Logistic Regression on ALL official Classification
Train50608,cycles47-79, with original structured feature allowlist, preprocessing,
TF-IDF parameters, all-row population, C=1/liblinear/max_iter2000/tol1e-4/seed42.
Check actual historical source confirms these settings before freeze; mismatch
goes to Sol rather than substituting settings. Reuse a verified existing E79
C2_B component if present; otherwise fresh full-Train refit is authorized.

Reuse original A02 E79 observed full-Train model:
experiments/runs/optimization_v3_a02_continuation_v1/E79/fulltrain_observed/.
Manifest reports/optimization_v3/a02_continuation_v1/submission_manifest.json;
original A02 local CSV SHA6909d4af8a326661b81e4a947d42b42919c004852967ca99f6bd481a78c679f8.
Model SHA2791eddf410c7e81b119bde2a0b1d33a694eed4de84027c61cee97f6668dcb6e.
Related B01 preparation MANIFEST.json documents full-Train Sparse25 cache,
OOF bank and bit-exact A02 reconstruction; frozen preparation status fields
are historical snapshots, not current Kaggle status. Reuse original whole-bank
rows/residuals, Sparse25 Test baseline, missing mask and A02 original Test mass.
Do not use or recompute B01 probabilities as the baseline.

E79 coefficient extraction must match CGEOM pilot: same full-Train vocabulary,
IDF, tokenizer and feature assembly, exact tokens/class order and structured
column mapping; no pilot E75 coefficients applied to E79 vocabulary. Reuse
A02 E79 fitted vectorizer for C2_B text transformation if exact historical recipe
and fit population/hash match; this is allowed cached preprocessing reuse.
New structured preprocessing learned only on all official Train is permitted.

Freeze the E79 recipe/source/input and selected coefficient state BEFORE official
Test inference. Test29421,cycles80-91, features only. Formula unchanged:
d_j=sqrt(1+abs(beta_j)/(1+abs(beta_j))); re-normalized TF-IDF and full-bank top32
positive weighted cosine, stable source-row ties; new squared weights rescaled
to sum to ORIGINAL A02 per-query M0. Same observed residuals, shrinkage4,
cap+/-.05, p=clip(Sparse25+delta,0,1),threshold>=.50. Exact original baseline
fallback on missing/zero original mass. No reranking, coefficient rescaling,
new residual centering, routing or choice using Test prediction prevalence.

## Accounting and required QA

Starting shared ledger303events,cap121,active118,starts118,saved114,failed4,pending0,
SHA5bd8aa218e17268913de9ecdac5b85ea576e611cde5566f2b3d242f371ff317a.
Verify actual state before any mutation. Authorize MAX5 new official fit units:
up to3 structured preprocessing units (same C2B_STRUCTURED,SCALER,OHE),1 C2_B
LR,1 CGEOM residual-memory predictor. Existing E79 text vectorizer reused;
zero new TF-IDF fits. Do not refit Sparse25/A02/missing expert or train permuted
control at E79. No retry; fitting one whole pipeline cannot hide subcomponent fits.
Append cap121->126 and continuation38->43 amendment; other category caps fixed.
Before starting, freeze exact required unit IDs and reserve ONLY needed slots
in continuation, at most5; cached verified units consume0 new slots. Total
starts may not exceed123. Extra global headroom is not spending permission.
On failure release only unstarted IDs; preserve all consumed fits/failures.

New benchmark namespace a02_cgeom_benchmark_v1 for scripts/reports/ignored runs.
Historical code/models/predictions/freezes/receipts remain immutable. Local
scoped commits only, no push. One job,two threads,6GiB RSS cap,4GiB available
before start and2GiB reserve; no dependency installs for this task.

Use production-schema synthetic roundtrip before fit, explicit Unicode IDs and
allow_pickle=False for NPZ; validate long IDs and expected/source identity dtype.
Explicit temporary ledgers; production writer blocked in synthetic tests.
Do not repeat known frozen-HEAD/EOL/schema blockers: separate immutable fitted
source hashes/receipts from later inference/evaluator source commits explicitly.
Synthetic lifecycle and persistence/resume must pass with no extra fit.

Validate cached source identities and independently reconstruct original A02
Test probability/label hashes before deriving CGEOM. Verify full-bank weighted
geometry, coefficient mapping, exact mass/formula/clipping, past donor cycles,
all fallback identities, Train-only learned components and immutable source binds.
Never load Test targets. Existing Train labels may only enter authorized fitting
and original OOF residual bank checks; no fresh temporal tuning/evaluation.

CSV filename classification_cgeom_aligned_e79.csv, columns EXACTLY
record_id,artifact_signal in SampleSubmission order,29421 unique official IDs,
integer labels0/1, no missing/nonfinite probabilities. Compare label changes
against original A02 for transparency, not selection. If labels identical,
stop without upload: no new scored benchmark information. Save one manifest
with recipe, source commits, input/model/artifact hashes, fit accounting, QA,
and NOT_QUALIFIED benchmark exception. No raw data/model binaries/CSV in Git.

## One upload authorization and completion

Classification slug:
tp2-classification-cyber-vulnerability-kasdd-gasal-2026-2027.
Message: Samuel CGEOM Aligned Frozen Benchmark.
Exactly ONE validated selected CSV upload is authorized. Do not upload Regression
files, permuted arm, alternatives or duplicate candidate. Check existing history
for this artifact/candidate first when authenticated access is available.
Use authenticated Kaggle CLI/API if available; browser upload is allowed when
already authenticated and tooling supports it. Do not seek credentials broadly,
expose tokens, change browser security settings or fabricate successful upload.
No need to ask Sam to approve again after validation; Sam already authorized
this one benchmark. If authentication/tooling is unavailable, stop with exact
CSV path/hash and manual upload instructions; retain PREPARED_NOT_SUBMITTED
until actual confirmation. A transport error with uncertain acceptance requires
status reconciliation before any retry, not blind resubmission.

Record actual receipt/status/score if confirmed; distinguish API/live proof from
user-reported proof, local vs server hash, internal vs public results. One score
does not trigger tuning/refit/another upload. Keep A02 as incumbent pending review.
No notebook/EDA/teammate changes. Produce one concise benchmark report and return
completion/blocker (manifest,CSV/hash,QA,fit counts,status/score,commit) to KASDAD -
TK2 - Sol,thread01a19569-16e0-7253-aa14-57a0a3160e03,host local, under Sam's
result-return authorization. Then STOP. Sol will not write concurrently.
