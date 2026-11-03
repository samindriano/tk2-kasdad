# Learned-pair FM V2: one bounded repair pilot

Status: GO_IMPLEMENT_AND_EXECUTE_ONE_PILOT, authorized by Sam's latest
"oke lanjutin dah brati" after the Astra recommendation and Sol review.
This supersedes the campaign STOP only for this V2 scope. It is not permission
for another lane, full-Train refit, Test inference, submission, or push.

## Ownership and starting checkpoint

Repository: C:/Users/Sam/OneDrive/Documents/KASDAD/TP 2.
Verified incoming main HEAD: 5aa822a6862eac004c085ca82521c80df496db46, clean.
This handoff will be committed after that checkpoint. Luna owns V2 implementation,
execution, focused QA, reports and scoped local commits. Sol owns this decision
and subsequent result review. Sol will not concurrently edit Luna's files.
Use the existing chat KASDAD - TK2 - Luna; one official fit at a time.

Read first:
- reports/optimization_v3/learned_pair_pilot_v1/FAILURE_DIAGNOSTIC.md
- reports/optimization_v3/learned_pair_pilot_v1/PROTOCOL.md
- reports/optimization_v3/learned_pair_preflight_v1/REPORT.md
- src/models/learned_pair_pilot_v1.py
- scripts/optimization_v3_learned_pair_pilot_v1.py
- scripts/verify_optimization_v3_learned_pair_pilot_v1.py

Historical V1 source, freeze, optimizer state, receipts, predictions and verdict
remain unchanged. V1 stays FAIL_OPTIMIZER_INCONCLUSIVE. Do not score or resume
its failed checkpoint. V2 is four fresh fits with new initialization and IDs.

## Frozen scientific recipe

Reuse verified saved C2_B fit_base/score_base, feature order and preprocessing
from POSITIONAL_TEXT_PILOT_V1. Zero learned preprocessing fits. Main matrix X
unchanged; interaction Q is row-L2-normalized X. Rank16, C=1, w=b=0,
V=default_rng(42).normal(0,.01,(d,16)), RNG reset at each origin.
z=Xw+b+.5*sum_f((QV)^2-Q^2(V^2)). No offset, A02 residuals, class weighting,
calibration, probability blend, feature changes or threshold search.
No-pair learns only w,b using identical rows/objective/solver settings.
Fit all historical prefix rows, not only stream-present rows.
Threshold >=.50; missing/blank rows use the identical cached C1_B HGB expert
in the FM, no-pair and C2_B routes. A02 keeps its own historical predictions.

SUM logistic loss + (||w||^2+||V||^2+b^2)/(2*C); no-pair omits V penalty.
Stable signed-label logaddexp and expit; accumulate float64 data loss/gradient
in fixed1024-row chunks, apply penalty ONCE. Reuse the pure V1 mathematical
module unchanged if useful. Do not patch or monkeypatch frozen V1 runner globals
for official execution. New V2 runner/verifier/tests have separate namespaces.

The sole optimizer-setting change from final V1 is gtol=n_fit*1e-4:
- E67 n=29705 -> gtol=2.9705 (SUM scale).
- E75 n=42117 -> gtol=4.2117 (SUM scale).
L-BFGS-B,maxiter1000,maxfun2000,maxcor10,maxls50,ftol1e-9, two BLAS threads.
Both FM and no-pair use this rule. Solver success alone is insufficient:
independently recomputed ||g_SUM||inf/n_fit <=1e-4, finite parameters/logits/
loss/gradient, and final objective<=initial+1e-8 are all mandatory.
Success via ftol with excessive gradient is a terminal technical failure.
This is operational stationarity, not proof of a global optimum or prediction gain.
Report full and per-block gradient norms, actual gtol, nit/nfev/status, loss,
factor movement, pair-score summaries, elapsed time and sampled resource usage.

## Boundaries, comparisons and gates

E67 fit47-67,n29705; score68-79,n20903; W4=68-71,W5=72-75,W6=76-79.
E75 fit47-75,n42117; score76-79,n8491. E75 shares W6 labels with E67:
history sensitivity, not independent replication. Do not compute full-six here.

Require same-origin frozen C2_B routed and A02 observed predictions. Files:
experiments/runs/optimization_v3_a02_neighbors_v1/E{67,75}/observed/predictions.npz.
Existence was checked, but full preflight binding must be redone: source/receipt,
origin, fit boundary, row indices, IDs, cycles, probability range and exact score
population. Do not silently substitute another origin/current model. Missing
or corrupt A02 is a real pre-fit blocker; do not refit it. No A02 labels/probability
arrays may enter the learned predictor or parameter selection.

Separate verdicts; no post-outcome threshold amendment:
1. MECHANISM_PASS requires E67 H12 F1 gain>=.0005 vs C2_B AND>=.0002 vs no-pair;
   E75 W6 gains>=0 vs both. E67 W4/W5/W6 F1 harm<=.005 vs both; supported
   availability, fit-defined text-load and entry-friction subgroup F1 harm<=.01
   and positive-recall harm<=.02 vs both, on every declared view.
2. COMPETITION_PASS additionally requires E67 H12 AND E75 W6 F1 gain>=.0005
   vs origin-matched A02; E67 W4/W5/W6 harm<=.005 vs A02, with the same subgroup
   guards vs A02 on every declared view. Gate all rows, not a favorable subset.
Subgroup support n>=200 and each class>=30; unsupported -> disclosed UNKNOWN,
no numeric harm estimate. Use prior exact group definitions and fit-only Q4
thresholds. Comparison tolerance1e-12. Missing-route equality applies to the
shared C1_B fallback, not to A02 if its historical route differs.
Report all comparator metrics, denominators, confusion counts, per-cycle signs,
negative views and probability metrics. Better probability metrics cannot
override Macro F1 failures. Control under-optimization remains a limitation;
do not claim a capacity-matched or causal isolation of learned interactions.

## Accounting and execution lifecycle

Observed ledger SHA256:
125446878a69bdecf2d658daf259622e568147642a072d2aaf2d8e21a8f47008.
281 events, cap113, active reservations110, starts110,saved106,failed4,pending0.
Category caps pilot72,continuation38,control_replay2,technical_retry1; three
released V1 pilot slots remain historical. Verify actual state before writing.
Authorize exactly four NEW pilot reservations under LEARNED_PAIR_PILOT_V2.
Append an explicit cap113->117/pilot72->76 amendment; other caps unchanged.
Do not reuse V1 reservations or reset failures. Active reservations after
new reservation114; max114 total starts if all four run. Unallocated headroom
is not permission to spend more. Category-active snapshot must respect prior
released slots; never assume active reservations equal category caps.
Fit IDs: LEARNED_PAIR_PILOT_V2_E67_FM, _E67_NOPAIR, _E75_FM, _E75_NOPAIR.
No retry. A failed start consumes its ID; release only never-started IDs,
append-only. No ledger truncation, replacement or restoration of old bytes.

Create V2 protocol, runner, independent verifier and focused tests. Reuse existing
proven lifecycle code with explicitly new paths/source lists/IDs/state checks;
do not carry hardcoded109/113 checkpoints, old gtol assertions, optional A02,
V1 output paths, stale source hashes or old gate logic into V2 by accident.
Commit source/protocol before source-bound synthetic QA, preflight and freeze.
Synthetic optimizer calls are allowed on synthetic data only, outside repository
run roots, with explicit temporary ledger paths and production writer blocked.
Exercise numerical gradient/scaling, normalized-gate pass/fail, raw/routed swap,
A02 required/alignment failure, view isolation, subgroup gates vs all comparators,
terminal failure accounting, persistence/readback and resume without refit.
Verify live ledger bytes/hash unchanged even on injected synthetic failures.

Run source-bound synthetic lifecycle -> preflight -> freeze -> reserve -> four
sequential fresh fits -> durable predictions -> independent pre-label verifier
-> score once -> independent metric/gate verification -> reports/commit.
Load official Train fit targets only for each exact prefix while fitting.
Outer/query targets remain closed until all four technical fits and prediction
integrity checks pass. Historical labels are adaptively reused development data,
not fresh holdout. Do not open Regression labels, Test, SampleSubmission or Kaggle.
Job RSS limit6GiB, available memory at least4GiB before start and2GiB reserve.
No installs. Do not delete a lock without checking ownership/process state.

On failure stop official execution, save diagnostic and release unstarted IDs.
Routine implementation corrections and synthetic QA before the official freeze
are allowed; changes after fitting require Sol review, not silent amendments.
Use durable failure evidence; do not create success from a failed receipt.

## Deliverables, return route and stop

Own scripts/*learned_pair_pilot_v2*, tests/*learned_pair_pilot_v2*, new V2 reports,
experiments/optimization_v3/learned_pair_pilot_v2 and ignored V2 run artifacts.
Preserve all other research, notebooks, teammates' work and submission artifacts.
No binaries/CSVs/raw data in Git. Scoped local commits only; no push.
Produce one substantive RESULTS.md plus machine-readable metrics/gates and
verification/accounting receipts. Stop whether performance passes or fails.
MECHANISM_PASS alone closes this lane as academic evidence. Both passes mean
review candidate only: no automatic continuation/full-Train/Test/submission.

Send a compact completion or real-blocker message back to KASDAD - TK2 - Sol,
thread01a19569-16e0-7253-aa14-57a0a3160e03,host local, before final response.
Sam explicitly requested result-return messages in this workflow ("eh next time
kalokirim message ke luna ... suruh kirim balik hasilnya ke kamu") and has now
authorized continuing this exact handoff. Include HEAD, status, tested source,
fit IDs/counts, both verdicts, comparator metrics, counterevidence, UNKNOWNs,
ledger/hash, QA and exact report paths. A genuine blocker goes to Sol promptly;
do not self-authorize a different recipe. Sol decides any subsequent action.
After dispatch, Sol stops this turn rather than waiting while spending quota.
