# Autonomous Regression Research — reproduction and boundaries

The user approved Regression Rounds1–2 and the initial Kaggle track, then authorized deeper regression research and at most **six additional Regression submissions**. Classification and all historical checkpoint artifacts remain unchanged. This phase used four additional submissions and stopped after a strong provisional candidate beat the user-supplied baseline and stretch target. It does not make a permanent final-model decision.

## Evidence and design

Stage-specific receipts live in `experiments/regression_research/`. Every stage freezes its declared protocol, training/analysis source hashes, raw regression input hashes, dependency identity and tracked checkpoint hashes before its first fit/evaluation. Text/code hashes normalize UTF-8 LF; raw input hashes are byte-exact. New research JSON artifacts preserve exact bytes through narrow `.gitattributes` rules. Base Git HEADs are historical context, not a requirement to retain an old HEAD after committing results.

All new models retain the six expanding primary windows and two H12 views from Round1. Fit cutoffs55/59/63/67/71/75 use only past rows; the inner partition is47..E−4 fitted and E−3..E scored, followed by a full47..E outer fit. Vocabulary/IDF/SVD/imputers/encoders are fitted independently inside each such fit partition. No outer-validation labels enter fitting or parameter/mapping selection. Both families have38,588 records; H12 reuses primary records and is not independent replication. Repeated prior EDA/model rounds mean the whole phase is adaptive development.

Raw/unclipped scores and mapped/clipped policies remain separately labelled. Within each supplied cycle, percentile prediction is100×average_rank(model_score)/N with average ties. A convex blend uses `(1−a)*raw + a*percentile`, clipped to[0,100]. The frozen grid is0/.25/.50/.75/1.00. Learned weights use inner MAE only. Incomplete-batch stress uses target-blind deterministic subsets; real source-universe completeness remains unknown. No claim is made that a target-on-target rank identity proves the official target generator.

The original12 structured inputs and their parsers retain their prior eligibility. **signal_stream is explicitly eligible for this user-authorized regression research**, with train-only vocabulary and operational timing still provisional. ID, target/other-task labels, audit_lane, checksum_hint and pulse fields stay excluded. No TP1 label join, target reranking, percentile-label synthesis or classification model is introduced.

Stage1 reuses four saved R2 control vectors without fits. Stage23 evaluates fixed text Ridge, hybrid Ridge and native-structured HGB+SVD32, plus one LightGBM family with three small predeclared complexities. Stage4 evaluates SVD128, no-scope SVD32 and a same-family structured Ridge control. Stage5 reuses the SVD32 representation and two earlier LightGBM settings; it is the model×information interaction suggested by the earlier results. Parameter and mapping are jointly selected on inner data. There is no new parameter grid, second tabular family, transformer, stacking or large ensemble after the breakthrough.

## Environment and local inputs

Use Python3.13.5 with `requirements-research.txt`: original versions remain intact, adding LightGBM4.6.0. Thread limit2 and seed42 are fixed. Numeric/categorical parsing is shared read-only with Rounds1–2. LightGBM uses deterministic CPU mode/force_col_wise and treats negative native-category sentinel−1 as missing; its warning about conversion to NaN is expected, not an undisclosed parse failure.

Raw files stay ignored under `data/raw/regression/`. `TP2_DATA_DIR` can point to another root containing `regression/`. Train/Test/Sample byte hashes are pinned in the historical receipts; no secret or raw row is committed. Caches under `experiments/runs/regression_research_v1/` contain validation arrays only and are ignored. No model weights are committed.

```powershell
python -m pip install -r requirements-research.txt
python scripts/reproduce_regression_research.py --stage models
# Optional Stage1 replay requires the historical R2 control cache:
python scripts/reproduce_regression_research.py --stage stage1
python scripts/verify_regression_research.py --stage all
python scripts/report_regression_research.py
python scripts/build_regression_research_notebook.py
```

The ranking runner expects the four R2 control caches. On a fresh checkout, reconstruct those frozen R2 vectors by following `REGRESSION_ROUND2.md` before Stage1, or run the later modeling stages independently. The reproduction wrapper snapshots only the named historical stage reports, runs the original frozen scripts, saves new measured reports/logs under an ignored reproduction directory, then restores original report bytes even on failure. This prevents newly measured runtimes from breaking the next stage's historical checkpoint hashes. Modeling stages rebuild missing ignored vectors without changing recipes. Use an isolated checkout for a full fresh reproduction. The feature-shift and sensitivity scripts remain separately frozen historical diagnostics; rerunning the latter performs two real refits, so it is not needed for normal presentation. Notebook08 is deliberately a presentation/read-only computation surface: run reconstruction and the verifier first, then Restart & Run All. It does not submit or retrain inside the notebook.

The independent verifier uses saved arrays and `math.fsum`/pandas average ranks, recomputes fit fingerprints and inner argmins, then checks per-window/aggregate/cycle/EF/tail/stream metrics and Spearman. Detailed exact-impact metrics are retained in results; the independent verifier checks their support counts rather than independently recomputing every impact-group distribution. Some fit histories recur for inner versus outer evaluation;90 cached fit/predict records were measured, without a redundant whole-campaign replay. Two additional refits serve the unscored representation-stress probe and match existing baseline vectors exactly. Cross-hardware bit identity is not promised.

## Scoped tests and plots

```powershell
python -m unittest tests.test_regression_research_ranking tests.test_regression_research_text tests.test_regression_research_tabular tests.test_regression_research_selection tests.test_regression_research_refinement tests.test_regression_research_lgb_text tests.test_regression_research_submission tests.test_regression_research_verification tests.test_regression_research_reproduction tests.test_round1_models tests.test_round1_evaluation tests.test_initial_submission -v
```

The final completed suite has65 passing tests and no failures/errors/skips (5.415seconds), including two report-restoration tests for success and failure. They use synthetic fixtures or small software checks, not new model-selection trials. The notebook builder uses nbformat/nbclient, executes top-to-bottom and retains figure outputs. Seven figures are generated from saved metrics/vectors and the confirmed submission ledger, with raw versus mapped policies identified. Inspect plots at reading size; execution alone is not visual QA.

## Kaggle gate and manifests

`reports/regression_research/KAGGLE_LOG.md` is the readable record; `kaggle_log.json` is the hash-chained at-most-once ledger. Approved model manifests are in its `submissions/` subdirectory; CSVs remain ignored under `submissions/regression_research/`. `best_candidate.json` points to the exact best provisional model manifest, CSV hash and Kaggle ref. Historical initial manifests are untouched.

`prepare_regression_research_candidate.py` prepares the approved Stage23/4 recipes; `prepare_regression_lgb_text_candidate.py` prepares the Stage5 recipe. They choose the full-Train configuration using47–75→76–79 inner data, fit all50,608 Train rows, predict29,421 Test rows, verify IDs/header/range/finite values and preserve accepted CSV bytes. Candidate preparations require an immutable MAIN-approved decision. Do not recreate their existing manifests in place.

The submitter defaults to local audit and requires explicit `--execute` to upload. It checks existing Kaggle history and current daily allowance, limits this phase to six additional candidates, reserves before upload and never retries an ambiguous response automatically. Credentials stay in the pre-existing local Kaggle configuration; the official CLI is in a temporary environment. Do not display or copy credential contents.

```powershell
# Read-only status refresh. This never uploads.
python scripts/submit_regression_research.py --refresh
```

No more uploads are needed in this completed phase. The four refs are57809775,57809802,57810028 and57810150. `lgb_text_inner_v1` is COMPLETE at public MAE11.71363; private score is unknown. Public feedback is an external limited benchmark, not proof of future generalization. Two remaining authorized slots were intentionally not used.

## Review and stop

Read the research `FINDINGS.md`, all-candidate `COMPARISON_TABLES.md`, notebook08 and `best_candidate.json`. The original formal EDA question and statistical conclusions remain unchanged; the new report contains the modeling appendix. Review the candidate's early-H12 counterexample, missing-stream bias, target tails and cycle-batch assumption before final packaging. No automatic extra research, Classification optimization, permanent final model or further Kaggle submission follows.
