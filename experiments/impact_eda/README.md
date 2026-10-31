# Impact-driven EDA research — reproducibility

Start: commit `8280ac3`. Scope: raw Classification availability analysis and one Regression coverage scalar. Final notebooks/models/submissions are protected; no new Kaggle access. `protocol.json` records raw Train hashes, source/reference hashes, six frozen per-fold parameter/rank choices, practical improvement gate, versions and 459 protected files. `freeze.json` was created before coverage-model fits. No outcomes changed the feature/gate.

From repository root in the reference Python environment (`requirements.txt`; exact versions in protocol):

```powershell
python -m unittest tests.test_impact_oov tests.test_regression_research_lgb_text -v
python scripts/research_impact_availability.py
python -u scripts/run_impact_oov_ablation.py
python scripts/research_impact_oov_eda.py
python scripts/verify_impact_eda.py
python scripts/summarize_impact_eda.py
```

Raw data resolves through `src.utils.paths.task_files`: repository `data/raw/{classification,regression}/Train.csv`, or `TP2_DATA_DIR` pointing to a parent containing both task directories. Neither Test labels nor public score is loaded by the scripts. Raw data and per-row prediction caches remain ignored.

The ablation runner needs only raw Regression Train plus tracked historical audit reports. It reads hash-verified control predictions if the old ignored cache exists; otherwise it refits the original selected controls into the new isolated `experiments/runs/impact_eda_oov_v1/` namespace and requires exact historical hashes. Each new coverage fit has its own resumable vector/audit pair. Same reference dependencies, thread limit 2 and seed 42 are required for control bit identity; this is not a cross-hardware bit-identity guarantee.

The retrospective OOV EDA currently reads the verified Stage5 control cache under `experiments/runs/regression_research_v1/` (raw prediction+audit and saved mapped vectors). The routing reconciliation in the verifier reads frozen Classification R1 C2_B/C1_B vectors under `experiments/runs/classification_round1_v1/`. These ignored historical caches are **explicit dependencies** for those two historical-reconciliation steps; they are not included in Git or silently replaced by cached aggregate JSON. To rebuild absent vectors, use the original frozen wrappers on their declared fold-fit rows in a separate cache namespace, then require original audit hashes before analysis. Do not overwrite historical reports or disable the checks. The independent raw availability EDA and new Regression ablation do not require those historical caches to run. All measured outputs in this milestone used locally available caches whose source/row/probability hashes were checked.

The narrative temporal split uses cycles56–67 for discovery and68–79 for sensitivity, but every cycle was already used in earlier development. Vocabulary is learned separately for each fit history; it does not use either target. All six original parameter/rank choices are retained, rather than adapting rank weights to a newly fitted feature.

Main output: `reports/impact_eda/DECISION.md`. Raw details: `availability/` and `oov/`. New feature results: `ablation/`. `QA.json` records the independent source/metric/preservation checks. One helper adds the coverage column without modifying the original LightGBM adapter; missingness sentinel0 retains the original explicit missing flag. A fraction outside TF-IDF vocabulary includes filtering/truncation and is not a semantic-novelty score.
