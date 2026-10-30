# Samuel Regression Round 2 — protocol and reproduction

This milestone investigates **feature contribution, training-history sensitivity and temporal errors**. It is adaptive development research after Round 1. It does not choose a final model. Classification checkpoints remain read-only. The separately authorized initial-submission track uses frozen Round 1 regression and Round 2 classification recipes, never new regression results or leaderboard feedback for research selection.

## Frozen design

Machine-readable contract: `experiments/regression_round2/protocol.json`; identity receipt: `freeze.json`. The freeze binds the protocol, model/metrics/runner code, original Regression Round 1 identity, raw input hashes, Python and package versions. Existing tracked checkpoints except the README front door are protected. Text identity uses UTF-8 LF; CSV/binary identity uses exact bytes. The base HEAD is historical context, not a requirement to remain at the same HEAD after committing results.

The narrow `.gitattributes` rules preserve new Regression Round 2 JSON artifacts byte-for-byte, including their recorded line endings. This matters because the adaptive selection binds the exact base-results file hash and verification binds the exact completed-results file hash. Git must not convert those saved receipt/result bytes during checkout. Historical checkpoint attributes are unchanged; source-code identity still normalizes UTF-8 LF.

Primary windows are unchanged: fit 47–55→validate 56–59; 47–59→60–63; 47–63→64–67; 47–67→68–71; 47–71→72–75; 47–75→76–79. Secondary views are 47–55→56–67 and 47–67→68–79. Both families contain the same 38,588 validation records. **H12 is not independent replication.** All Train has already been explored and the experiment questions were informed by Round 1 results.

The 15 base recipes comprise:

- Exact R1-C and R1-D controls, delegated to their original implementations.
- Five expanded-feature removals: cycle; three categorical codes; scope/spread (`scope_pressure`, `scope_trace`, `org_spread`, `module_spread`); telemetry; numeric `text_load`. Removing a source also removes all its derived flags. Numeric text volume is kept separate from the scope/spread group; `signal_stream` remains excluded.
- R1-C and R1-D feature sets under recent 8 cycles, recent 12 cycles, and two matched-size samples, seeds 42 and 31415.

Recent histories retain the last *observed complete cycles*, strictly before validation. Matched samples select exactly the recent-8 row count uniformly without replacement from expanding past rows; the RNG uses `SeedSequence([seed, fit_end])`, and sampled positions are sorted back to official source order. Sampling never reads targets. Matched samples can split cycles. Equal sample size partially controls observation count, while era, category support and temporal dependence remain different. Two samples are sensitivity checks, not independent trials or confidence intervals.

All estimators retain absolute-error HGB, learning rate .05, 200 iterations, 15 leaves, minimum leaf 30, L2 1, max bins 255, seed 42, early stopping disabled, thread limit 2. Preprocessing preserves Round 1 parsers, deterministic flags, train-only medians, train-only rare-category grouping (minimum frequency 30, cap 254), native categorical masks and unknown/missing code −1. No target transform, target reranking, calibration, text estimator, ensemble or hyperparameter search is introduced.

After base results, MAIN may select at most two ablated feature sets and four interactions with recent-8 and matched-8/seed-42 histories. The allowed sets/histories and budget are frozen in the base protocol. The numerical rationale and exact choices are saved once in `interaction_selection.json`, bound to the base-results file hash, **before any interaction fit**. This second stage is explicitly adaptive; it is not confirmation on fresh data.

The frozen choice was `no_codes` and `no_cycle`, each with recent-8 and matched-8/seed-42. A separate `interaction_rationale_precision_note.json` corrects excess precision in several transcribed prose numbers (differences below 0.000006 MAE), preserving the original selection unchanged. The machine-saved metrics are authoritative; no interaction outcome was consulted for that correction.

## Metrics and interpretation

Primary metric is raw/unclipped row-pooled MAE. Also report equal-cycle MAE, every window/cycle, RMSE, signed error (prediction minus actual), EF0/EF1, fixed thirds of actual target, tails actual ≤5 and ≥95, exact impact groups, prediction quantiles/range and over/underprediction. Clipped [0,100] metrics are separate. Paired comparisons require identical sorted validation positions. Feature removals measure predictive contributions conditional on this model/refit, not causal importance.

Category novelty is calculated relative to each selected fit subset. Unknown and missing groups can overlap across columns, so their counts must not be added. The test-feature report describes supplied official Test features only after the experiment selection is complete; no Test labels, test predictions or transductive model selection occur in the research runner. Availability of ten added supplied-schema features remains timing-provisional as documented in `ROUND1_FEATURE_ELIGIBILITY.md`.

## Environment and commands

Use Python 3.13.5 and the repository's pinned `requirements.txt`. Raw files remain ignored under `data/raw/regression/`; alternatively set `TP2_DATA_DIR` to a root containing `regression/`. No raw data or prediction vectors are committed.

```powershell
python scripts/run_regression_round2.py --freeze-only
python scripts/run_regression_round2.py --stage base
```

The completed canonical base result is immutable: a later base invocation preserves it. Exact adaptive choices and historical base metrics are checked in. A fresh checkout can reconstruct missing vectors with the all stage while reusing that historical selection, without making a new outcome-driven choice. Runtime/timestamps in a freshly recomputed base JSON would change its byte hash; do not overwrite the historical base result or selection receipt. Use a separate cache/report namespace for a repeat. Existing verified local caches reconstruct tables without refitting.

```powershell
python scripts/run_regression_round2.py --stage all
python scripts/verify_regression_round2.py
python scripts/report_regression_round2.py
python scripts/build_regression_round2_notebook.py
```

Full vectors are local ignored files in `experiments/runs/regression_round2_v1/`, not model binaries. A single fit is reused for windows sharing the same cutoff/features/source subset, including the early recent-12/expanding aliases. Cache keys bind the freeze and selected fit/validation positions. One-sided cache pairs or altered identity/audit metadata fail closed; preserve and investigate them instead of deleting history.

For an actual representative refit, use a distinct cache/report namespace:

```powershell
python scripts/run_regression_round2.py --stage all --only-fit-end 67 --run-id regression_round2_repeat --report-dir "$env:TEMP/tp2-regression-round2-repeat" --force
python scripts/verify_regression_round2.py --repeat-report "$env:TEMP/tp2-regression-round2-repeat/results.json"
```

The receipt only claims reproducibility for the cutoffs and arrays actually compared. Bit identity across arbitrary machines is not promised. The verifier independently recomputes error metrics from saved vectors, reconciles fit/validation positions and preprocessing, checks original control vectors, and binds any repeat PASS to current result/vector identities.

On a fresh checkout, ignored Round 1 vectors need not be reconstructed: control equality is checked against the immutable committed Round 1 prediction hashes and protocol-derived row positions. When old local vectors exist, the verifier additionally compares their arrays directly. Both paths reject changed predictions or source ordering; no historical report writer is needed.

Scoped validation avoids invoking historical report writers:

```powershell
python -m unittest tests.test_regression_round2_models tests.test_regression_round2_metrics tests.test_regression_round2_evaluation tests.test_regression_round2_results tests.test_regression_round2_reporting tests.test_round1_models tests.test_round1_evaluation -v
```

## Outputs and stop point

`notebooks/samuel/07_samuel_regression_round2.ipynb` is the compact executed narrative. `reports/regression_round2/` contains all-candidate tables, saved metric summaries, fit audits, residual/history/test-feature diagnostics, figures, findings and verification. Detailed computation lives in reusable scripts/modules. Inspect figures at reading size as well as executing the notebook.

Stop after Regression Round 2 review. The isolated initial-submission authorization does not permit more Kaggle submissions, leaderboard tuning, automatic Round 3, or a final model decision.
