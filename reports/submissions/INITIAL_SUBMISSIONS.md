# TP2 Initial Kaggle Submissions

The user authorized one benchmark submission for each TP2 competition using the best temporary candidate from the completed model-development rounds.

## Classification

- Recipe: C4_HGB, threshold 0.50; nonblank `signal_stream` routes to the complete C2_B hybrid logistic model; missing/blank stream routes to C1_B structured HistGradientBoosting.
- Submission file: `submissions/classification_initial.csv` (local and gitignored).
- Manifest: `reports/submissions/classification_initial_manifest.json`.
- Competition: `tp2-classification-cyber-vulnerability-kasdd-gasal-2026-2027`.
- Kaggle message: `Samuel C4-HGB Routing Initial Benchmark`.

## Regression

- Recipe: R1_D expanded structured HistGradientBoostingRegressor, absolute-error objective, unchanged Round 1 feature allowlist, preprocessing, and fixed hyperparameters; full official Train refit, then clip predictions to [0,100].
- Submission file: `submissions/regression_initial.csv` (local and gitignored).
- Manifest: `reports/submissions/regression_initial_manifest.json`.
- Competition: `tp2-regression-cyber-vulnerability-kasdd-gasal-2026-2027`.
- Kaggle message: `Samuel R1-D Expanded HGB Initial Benchmark`.

Both output files must contain exactly 29,421 rows, preserve the official SampleSubmission columns and row order, and match all Test record IDs. The preparation script independently checks hashes against the committed Round 1/Round 2 data identities, fits only labeled Train rows, and round-trips the emitted CSV before recording its SHA-256.

## Kaggle status

Both one-time submissions were sent through Kaggle CLI 2.2.5 after `kaggle quota` confirmed authenticated access. Kaggle submission history was empty before uploading, so neither request duplicated an earlier benchmark. Current references and score status are recorded separately in `reports/submissions/KAGGLE_RESULTS.md` and the task manifests.

The Kaggle CLI was installed only in a temporary virtual environment. No Kaggle credential values were read into tool output or copied into the repository. To inspect later, run `kaggle competitions submissions <competition-slug> --format json`. Do not repeat either upload; only the two authorized submissions were made.

These are initial public-test benchmarks, not final-model decisions. Their scores, if later available, should be recorded separately from temporal development validation and must not be used to tune this milestone.
