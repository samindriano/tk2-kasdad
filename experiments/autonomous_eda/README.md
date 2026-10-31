# Autonomous EDA-driven research — reproduction and artifact map

Start `ef475dd`; source/selection checkpoint `1c48e69`; final results are under `reports/autonomous_eda/`. The user allowed max2 new Kaggle submissions per task; exactly1 per task was used. All distinct recipes were selected before any new public score. No notebook, official EDA question or official model was edited.

Use the repository Python environment and original raw Train/Test/SampleSubmission files through `src.utils.paths.task_files`. `TP2_DATA_DIR` may point to a parent containing `classification/` and `regression/`; default is `data/raw/`. Requirements reference Python3.13.5, NumPy2.4.2, pandas2.3.3, scipy1.18.0, sklearn1.8.0, threadpoolctl3.6.0. Regression additionally needs **LightGBM4.6.0**, explicitly recorded in freezes/manifests. Install pytest only if using the test command below. Secrets are never arguments or repository files.

```powershell
python -m pytest tests/test_classification_models.py tests/test_classification_round2_models.py tests/test_autonomous_classification.py tests/test_autonomous_volume_route.py tests/test_autonomous_regression.py tests/test_autonomous_submission.py -q
python scripts/verify_autonomous_classification.py
python scripts/verify_autonomous_volume_route.py
python scripts/verify_autonomous_regression.py --verify-inner
python scripts/summarize_autonomous_research.py
python scripts/verify_autonomous_checkpoint.py
```

Verification uses ignored **local saved vectors** whose row/prediction hashes must match tracked frozen audits. Historical dependency paths: `experiments/runs/classification_round1_v1/` (original C1_B/C2_B vectors), `experiments/runs/regression_research_v1/` (Stage5 controls), then isolated new namespaces `autonomous_cls_v1/`, `autonomous_cls_volume_v1/`, `autonomous_reg_v1/` and `autonomous_submission_v1/`. Verifiers fail clearly if these are absent; they do not silently replace actual pipeline execution with cached aggregate JSON.

To reconstruct temporal fits from raw Train, use `research_autonomous_classification.py`, then `research_autonomous_volume_route.py` (no new fits, verified route composition). Regression original fit identity and linked report-only amendments are retained; its six source model fits can be rebuilt with the frozen adapter and declared choices, then checked against the recorded raw hashes. Current report-only command is `research_autonomous_regression.py --report-only`; `--run-inner-refinement` implements the separately frozen12-fit past-only stage and reuses those outers. In a fresh checkout without caches, the initial regression command may stop on the intentional original-code/final-report amendment guard; do not weaken the hash checks to bypass it. Reconstruct fits in a distinct namespace from the frozen adapter, exact source receipt and declared boundaries, then verify hashes. This is a research artifact dependency, not a claim that an academic notebook is standalone executable.

Full-Train APIs are independently executable from raw inputs with no model-cache dependency:

- Classification: `src.models.autonomous_volume_route.build_volume_route_classifier()` → `fit(X_train, y)`, `predict_proba(X_test)`, fixed threshold0.50. Three experts fit internally, Q80 from fit inputs only.
- Regression: `src.models.autonomous_regression.build_autonomous_regression('base')` → `fit`, `predict`, then original `mapped_scores(pred, test_cycles,0.5)` and clipping. The paired recipe retains original full-Train parameter/rank policy; optional SVD128 past-only selection47–75→76–79 happened to choose the same policy.

`prepare_autonomous_candidates.py <candidate_id>` executes the selected recipe, checks official input SHA, refits, produces ignored CSV/vector and a **new exclusive manifest**; it refuses to overwrite existing outputs. The accepted baseline classification CSV is required for exact unchanged-row comparison. If absent, restore submission57808225 through authenticated Kaggle download and demand SHA `97156470466ec8e91f1454bcdbef464f9e43f928ede67e2a9dadada1fb6d090f`. It was restored this way in the executed run. Do not fabricate it.

Kaggle standard SDK lives in the already authenticated CLI environment used in previous milestones. `autonomous_kaggle.py preflight <task>` checks live team/rules/slot; upload and poll subcommands are separate. **Do not rerun upload to reproduce this milestone:** both candidate IDs already have accepted COMPLETE ledgers. The tool reserves before API mutation and prevents duplicate/ambiguous retries. Existing CSVs/manifests/ledgers remain unchanged. The tests use synthetic local fixtures and mock API calls only; actual predictions always come from official data.

Main narrative: `reports/autonomous_eda/FINDINGS.md`. Fine detail: Classification subset fits `classification/`, adaptive limited Q80 route `classification_volume/`, paired and inner SVD128 `regression/`, full-Train manifests and official API receipts `kaggle/`. `QA.json` binds final sources, official data, CSV checksums, exact COMPLETE references and historical preservation. The final model recommendation is **retain both originals**; nothing was promoted or packaged for coursework submission.
