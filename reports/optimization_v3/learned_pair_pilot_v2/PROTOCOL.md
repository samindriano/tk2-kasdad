# Learned-pair FM V2 — bounded diagnostic amendment

## Authority and stop boundary

This amendment is authorized by Sam through Sol after review of the V1 optimizer failure. It is diagnostic-only and does not revise the historical V1 verdict `FAIL_OPTIMIZER_INCONCLUSIVE`. It was specified after observing that V1 stopped at its iteration limit and before any V2 fit or predictive outcome; it is therefore **not outcome-blind**.

The question is: **After a bounded optimizer repair, does the fixed rank-16 learned-pair model improve classification over the same-matrix no-pair control and the frozen C2_B baseline, while meeting the preregistered robustness gates?** The handoff authorizes exactly four predictor fits at E67 and E75: FM plus matched no-pair at each origin. The recipe is otherwise unchanged. Each failed started ID is terminal; there are no retries. No Test, SampleSubmission, Kaggle, full-Train refit, model/champion change, continuation, or additional lane is authorized.

## Frozen model recipe

- Data source: official Classification Train, hash-checked; saved C2_B matrices, prediction packages, feature order, and preprocessing from `POSITIONAL_TEXT_PILOT_V1`.
- Fit populations: all rows in the saved historical prefix at cycles 47–67 (29,705 rows) and 47–75 (42,117 rows). Query labels stay closed until all four fits, durable score predictions, and the independent pre-label verifier pass.
- FM score: `Xw + b + 0.5 * sum_f((QV)^2 - Q^2(V^2))`, where `Q` is row-L2-normalized `X`; rank 16. `w=b=0`; factors use `default_rng(42).normal(0, 0.01, (d,16))`, reset at each origin. The no-pair control learns only `w,b` from the same rows and objective.
- Objective: stable binary logistic **SUM** loss plus `(||w||² + ||V||² + b²)/(2C)`, `C=1`. Accumulate float64 loss/gradient in fixed 1,024-row chunks and apply the penalty once. Reuse the unchanged V1 pure mathematical module.
- The only solver change from V1 is `gtol = n_fit * 1e-4` on the SUM objective: E67 `2.9705`; E75 `4.2117`. L-BFGS-B keeps `maxiter=1000`, `maxfun=2000`, `maxcor=10`, `maxls=50`, `ftol=1e-9`; two BLAS threads and one fit job.
- A fit passes only when SciPy reports success, parameters/logits/loss/gradient are finite, final objective is no more than initial objective + `1e-8`, and an independently recomputed mean-scale gradient infinity norm is at most `1e-4`. FTOL termination with excessive gradient is terminal failure. This is an operational stationarity gate, not proof of a global optimum or predictive gain.
- Inference uses direct learned-expert probabilities where `signal_stream` is present and the exact cached C1_B HGB fallback for missing/blank stream, with threshold 0.50. C2_B is compared through its saved availability-routed probability, not its raw expert probability. A02 retains its own historical routing.
- No offset, A02 residual input, probability blend, calibration, class weighting, threshold search, feature change, or recipe rescue is allowed.

## Exact-origin A02 comparator

Before freeze/reservation, E67 and E75 A02 observed bundles must each have their saved `predictions.npz`, `fit_receipt.json`, `fit_state.json`, `model.joblib`, and `predictions.json`. Verify the frozen A02 protocol/source/input hashes, receipt signature, model/state/output hashes, origin and fit ID, exact prefix row count/hash, no query labels passed to fit/prediction, exact score rows/IDs/cycles, and finite probabilities in `[0,1]`. The A02 prediction archive contains labels, but V2 must not read that array. A02 probabilities are only a post-fit comparator; they never enter FM/no-pair fitting, prediction, or parameter selection. Missing or misaligned A02 is a pre-fit blocker; do not substitute another origin or refit A02.

## Evaluation windows and separate verdicts

E67 is scored on W4 (68–71), W5 (72–75), W6 (76–79), and pooled H12 (68–79). E75 is scored on W6 (76–79), which overlaps the E67 W6 rows and represents history sensitivity, not independent replication. Report primary/pooled and each window, all comparator metrics, confusion counts, positive recall, probability metrics, per-cycle gains/sign counts, and all negative views.

`MECHANISM_PASS` requires all of:

- E67 H12 Macro F1 gain at least `+0.0005` versus routed C2_B and at least `+0.0002` versus no-pair.
- E75 W6 Macro F1 gain nonnegative versus both routed C2_B and no-pair.
- E67 W4/W5/W6 Macro F1 harm no worse than `-0.005` versus both.
- For every supported availability, fit-defined text-load, and entry-friction subgroup, Macro F1 harm no worse than `-0.01` and positive-recall harm no worse than `-0.02` versus both.

`COMPETITION_PASS` additionally requires:

- E67 H12 and E75 W6 Macro F1 gains of at least `+0.0005` versus origin-matched A02 observed predictions.
- E67 W4/W5/W6 Macro F1 harm no worse than `-0.005` versus A02.
- The same supported subgroup Macro F1 and positive-recall guards versus A02 on every declared view.

Subgroup support is `n >= 200`, with at least 30 examples of each class. Unsupported groups are recorded as `UNKNOWN_UNSUPPORTED`, with denominator and class counts; no harm estimate is made. They do not receive fabricated numeric estimates. Comparisons use `1e-12` tolerance. Report all negative results. Better ROC AUC, average precision, log loss, or Brier score cannot override a Macro F1/recall gate failure. Two origins cannot establish a general relation between AUC and task gain. Control under-optimization limits causal or capacity-matched interpretation.

Both verdicts are diagnostic evidence only. Even `MECHANISM_PASS` or `COMPETITION_PASS` does not qualify a candidate, replace a champion, or authorize another fit, continuation, refit, inference, or upload.

## Budget and lifecycle

The pre-amendment live ledger checkpoint must be re-read and match exactly: global cap 113; caps continuation 38, control replay 2, pilot 72, technical retry 1; active reservations 38, 2, 69, 1 (total 110); 110 starts, 106 saved, 4 failed, 0 pending. Preserve all historic release events. Append one amendment from global 113 to 117 and pilot 72 to 76, then reserve these exact IDs:

1. `LEARNED_PAIR_PILOT_V2_E67_FM`
2. `LEARNED_PAIR_PILOT_V2_E67_NOPAIR`
3. `LEARNED_PAIR_PILOT_V2_E75_FM`
4. `LEARNED_PAIR_PILOT_V2_E75_NOPAIR`

After reservation, maximum starts are 114 and active reservations 114. No other headroom may be spent. V2 accounting is append-only and idempotent; the old V1 reservation and its three released IDs are never reused or counted twice. A failed started ID is consumed. On failure, stop, save a diagnostic, release only unstarted V2 IDs, and do not score targets.

Execution order is committed source/protocol/tests → source-bound synthetic lifecycle QA → read-only official preflight → freeze → reserve → four sequential fits → durable predictions → independent pre-label verification → one score pass → independent metrics/gate review → report/commit. Use one process at a time, at most two threads, process RSS under 6 GiB, at least 4 GiB free before a fit and 2 GiB reserve. No installs. Never delete a lock without verifying process ownership.

## Reproducibility commands

Run from the repository root. The synthetic root must be a new empty folder outside the repository.

```powershell
python -m pytest -q tests/test_learned_pair_pilot_v2.py
python scripts/optimization_v3_learned_pair_pilot_v2.py synthetic --temp-root "$env:TEMP\kasdad-learned-pair-v2-synthetic"
python scripts/optimization_v3_learned_pair_pilot_v2.py preflight
python scripts/optimization_v3_learned_pair_pilot_v2.py freeze
python scripts/optimization_v3_learned_pair_pilot_v2.py reserve
```

Then invoke `fit-next` once per fit, waiting for each process to finish. Only if all four receipts are successful, run `predictions`, `verify-prelabel`, then `score`. Keep raw Train, checkpoints, model binaries, and CSVs out of Git. Send a result or genuine blocker to the Sol chat and stop for review after this amendment; do not enter the original continuation queue or start a new experiment.
