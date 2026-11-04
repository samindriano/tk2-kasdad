# REG_POSITION_L1_V1 — execution protocol for Luna 2

## Authority, identity and ownership

Sam requested this Regression lane and authorized implementation in the newly
created Luna 2 chat. **Luna 2 thread01a1a442-590f-7520-a09a-bcd10f292328**, host
local, executes here; **Sol thread01a19569-16e0-7253-aa14-57a0a3160e03** reviews.
Luna 1 thread01a18b41-4be9-7cb2-b8b5-d4cd45ed8e18 exclusively owns Classification
POS_CONTRAST_V1. Never send a REG execution handoff to Luna 1.

Work only in `C:/Users/Sam/OneDrive/Documents/KASDAD/tk2-regression-frontier`,
branch `codex/regression-position-research`. The main `TP 2` worktree is a
READ-ONLY source of raw data/immutable caches. Every shell call uses an explicit
workdir. Do not change main, its ledger/lock, other worktrees, old source/freezes,
notebooks, EDA or teammates' work. No merge/cherry-pick/rebase/push. Scoped local
commits are allowed. Read applicable parent AGENTS; ordinary Luna orchestra is
allowed with bounded read/test workers, one writer and one fit process in this lane.

New implementation namespace `reg_position_l1_v1`; new tracked reports live in
`reports/optimization_v3/reg_position_l1_v1/`, ignored runs in
`experiments/runs/reg_position_l1_v1/`. Keep this design namespace unchanged.
All source artifact roots are explicit CLI/environment inputs; no personal path
inside the analysis/model code. Reports store relative paths and hashes.

## Exact experiment and target boundary

Question: does anonymous token placement explain R06 OOF residuals beyond bag
text and matched shuffled position, with transferable MAE benefit?

Origins67/75. At E67, residual-training bank is the16,482 present-stream rows
cycles56–67; query cycles68–79,20,903 rows. At E75, bank28,045 present rows
cycles56–75; query76–79,8,491 rows. These banks come from the original R-NBR
bundle donor members, verified during design. Do not use R-NBR predictions as
the baseline; its `baseline_prediction` is the required query R06 vector.

For donor i, response r_i = threat_rank_i - FINAL R06 OOF_prediction_i. Each
generating cutoff must precede donor cycle; source rows/IDs must be unique and
<=origin. Reconstruct each four-cycle OOF block from original R06 caches and
require equality to the stored bank, with NO later-origin overlay. No exact R06
OOF exists before cycle56; do not invent it or import an older outcome-selected
rank recipe. Query targets never enter feature construction or head fitting.

Read source baselines:
- E67: `experiments/runs/optimization_v3_research_next_v1/RNBR_PILOT_OFFICIAL_V1/recovered_bundles/RNBR_PILOT_E67_OBSERVED/arrays.npz`.
- E75: corresponding `RNBR_PILOT_OFFICIAL_V1/bundles/RNBR_PILOT_E75_OBSERVED/arrays.npz`.
- original R06 OOF: `optimization_v3_batch1/R06/R06_fit55.npz`, `R06_fit67.npz`,
  and `optimization_v3_expanded_benchmark_v1_recovery_v2/R06/R06_fit{59,63,71,75}.npz` under main `experiments/runs/`.
- exact hash/identity readiness in the design audit JSON; R06 recipe/source in
  `experiments/optimization_v3/batch1/R06/recipe_revision_02.json` and
  `src/models/optimization_v3_batch1_r06.py`.

Classification labels, classifiers, probabilities and residuals are prohibited
inputs. Classification source archives may be opened with allow_pickle=False to
read ONLY matrix/ID/cycle metadata, never probability members. Reused TF-IDF and
positional counts are target-free and must be verified against Regression text.

## Features, arms, estimator and fixed prediction policy

Reuse standalone target-free TF-IDF objects under main
`experiments/runs/optimization_v3_research_next_v1/RNBR_IMPACT_MATCH_V2_1/text_space_replays/E{67,75}/vectorizer.joblib`.
Verify file hash, vocabulary/IDF/document/prefix identities and original receipt.
Do not load a full A02/Classification estimator container. No vectorizer.fit,
fit_transform, vocabulary update, scaler fit or preprocessing fit in the pilot.

B: transform documents with that existing word vectorizer, select unigram
columns in original feature-index order (terms containing no space), and L2
normalize the selected unigram block once. This is transform-only true bag TF-IDF.
Zero known-unigram rows remain zero. C: one scalar column R06_prediction/100,
using each donor's historical OOF score for fit and current-origin R06 for query.
No cycle, actual target bin, raw target-correlated extra columns or cross-task
labels may be added. Same rows/covariates and response in all three arms:

1. BAG: [B, C].
2. REAL: [B, P_real, C].
3. SHUFFLED: [B, P_shuffled, C].

P is the existing three-third word position suffix from POSITIONAL_TEXT_PILOT_V1
same-origin matrices; stored `fit_position`/`score_position` are AUGMENTED and
must be sliced after exact corresponding `fit_base`/`score_base` width. Original
three channels: per-third(1+log(count))*frozen unigram IDF, concatenated and
L2-normalized once. Do not use CLS contrast projection here. SHUFFLED uses the
original deterministic order permutation seed42, preserving token bag/length.
No further P normalization. Common-mode is not pure bag because of per-third log.
Join prefix matrices to donor rows by exact source row/ID; never assume the donor
bank equals all prefix rows or is sorted chronologically. Query rows likewise.
Verify all cached file/logical hashes and original identity bundle; reconstruct
64 deterministic document rows per origin/arm using the standalone vectorizer
and pure builder to check tokenizer, IDF, channel order and numerical equality.

Each head is LightGBM LGBMRegressor, objective `regression_l1`, using unchanged
`COMMON_PARAMETERS` plus `PARAMETER_SETS['regularized']` from
`src/models/regression_research_tabular.py`:400 trees, learning_rate.05,
num_leaves15, min_child_samples80, reg_lambda3, n_jobs2, deterministic true,
force_col_wise true, seed42, remaining parameters from that source. Freeze full
resolved parameters/versions before fits. Target r is in original rank units.
No early stopping/eval_set/query labels, weights, seed sweep or hyperparameter grid.

For each head, h=predicted residual; delta=clip(.25*h,-5,+5). Final
prediction=clip(R06_query+delta,0,100) on present rows. Missing/blank rows get
EXACT R06_query, zero delta. No new rank mapping: R06 already contains rank50
computed on complete cycles. Do not apply rank50 to residuals or this final
prediction, change alpha/cap, or choose raw/unshrunk head outputs as another policy.

## Fits, persistence and premetric freeze

Exactly six new predictor units, fixed order E67_BAG,E67_REAL,E67_SHUFFLED,
E75_BAG,E75_REAL,E75_SHUFFLED. No new pilot preprocessing/memory/index units.
Use a NEW isolated append-only ledger in this lane's ignored run root, cap6
pilot starts. Record the shared main-ledger snapshot as contextual evidence,
but never write it or demand that it stay unchanged while Luna 1 works.
Report isolated starts/saved/failed separately; shared totals do not include
this lane until Sol performs a later explicit reconciliation. No uncounted fits.
Preprocessing replay is not free if it calls fit; it is prohibited in this pilot.

Commit runner/model/tests/recipe and freeze all source/input/cache/model hashes
on actual execution commit before production. Small standalone adapter; import
pure helpers, do not monkeypatch old runners/budget callbacks. Run an ACTUAL
entrypoint synthetic end-to-end with temp-only ledger before official fitting:
6 reservations->start->estimator->durable model/receipt/prediction->reload->
independent metric/gate report. Guard source root against any writes. Include
Unicode IDs, unsorted cycles, wrong-origin baseline, raw-vs-final R06 mixup,
donor future/self labels, missing route, zero/OOV text, sparse signed/zero values,
tail denominators, tamper and post-save resume without refit. Use mutable explicit
test roots, never stale expected live-ledger hashes. No install/dependency changes.

Persist each fitted head, receipt and predictions before starting another fit.
Each started failure remains consumed; no automatic retry. Resume a valid saved
artifact after writer failure; never rerun estimator.fit merely to regenerate
reports. Freeze and prelabel integrity verification precede new query scoring.

One fit process in this lane,2 threads,6GiB RSS ceiling; >=6GiB free before a
production start and>=2GiB running reserve. Avoid simultaneous production fits
with Luna 1: inspect actual live training PID/command identity without printing
secrets, not mere presence of an old lockfile. Implement/tests/preflight may
proceed while its fit runs. If training slot busy, finish preparation, wait in
bounded <=30second polls for <=30minutes, then send READY_WAITING_COMPUTE to Sol
with checkpoint and stop. This is scheduling, not a failed scientific preflight;
Sol's handoff monitor must resume it when available. Never delete another lock,
kill another job, or silently count a wait as a fit failure.

## Scoring and locked decisions

All views use same query rows as unchanged R06. E67 H12=cycles68–79, with W4
68–71,W5 72–75,W6 76–79; E75 W6=76–79. Also report unique-row LATE primary
E67 W4 + E75 W6,n13,441. Overlap is explicit; do not concatenate overlapping
origins and call the larger n independent. All data are reused adaptive development.

For each arm, define gain=MAE(R06)-MAE(arm). Competition eligibility requires:
- Gain>=.03 on BOTH E67 H12 and E75 W6.
- E67 W4/W5/W6 each harm<=.10 versus R06.
- Supported groups n>=100: missing, actual target<=5/>=95, EF0/1 and prefix-fit
  impact/text_load quartiles, on all five views; MAE harm<=.25 versus R06.
  Tied quartile edges collapse bins. Unsupported UNKNOWN, never fabricated PASS.
- Missing prediction bit-exact R06; finite [0,100], input boundaries, durable
  accounting and independent formula/metrics/group verification PASS.

REAL mechanism evidence additionally requires gain versus BOTH BAG and SHUFFLED
>=.01 on BOTH E67 H12 and E75 W6. These are engineering margins, not p-values.
If REAL competition+mechanism gates pass, choose REAL. Otherwise if BAG meets
its own competition gates, return BAG as a separate bag-residual candidate with
POSITION_NOT_ESTABLISHED. This fallback is frozen now, not selected after seeing
results. SHUFFLED is never a submission candidate. If neither qualifies, retire
this bounded pilot. Preserve all negatives, including any REAL gain that is
actually explained by BAG or shuffled capacity. No automatic branch or new fit.

No fixed subgroup may be routed by observed target, query error, or the plots.
R06's inherited W1 missing harm versus legacy remains historical failed risk;
new incremental PASS does not clear it. Report raw head fit objective/loss and
descriptive rank correlation, but MAE/risk gates decide. Training loss is not
generalization evidence.

## Handoff and later score push

Finish the pilot, independent verifier, concise results and at most2 useful PNGs,
local scoped commit, then SEND all results/gates/counterevidence/exact paths,
accounting, resource notes and commit to Sol chat01a19569-16e0-7253-aa14-57a0a3160e03
hostlocal BEFORE final reply. Real blockers must be sent promptly with exact
evidence and a narrow proposed remedy, rather than left idle. Routine prefit code
fixes are authorized with new source/freeze preserving previous evidence; no
post-fit recipe/gate change. Stop after callback for Sol review.

No Test/Sample/Kaggle/full-Train work in this pilot. If eligible after independent
Sol review, the next tranche may fit one E79 target-free word vectorizer and one
selected head on strictly OOF residual bank56–79; it need not refit R06. That is
TWO additional learned units requiring a new explicit source/freeze/accounting
amendment and one reviewed CSV before any possible authorized upload. Do not
assume E75 TF-IDF is a full79 refit. No submission authorization is inferred from
this pilot file; root owns subsequent eligibility and upload decisions.
