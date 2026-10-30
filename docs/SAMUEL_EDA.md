# EDA Samuel — status and method

**Status: blocked pending official regression files.** The current `data/raw/regression/` folder has no files. The available TP2 classification data includes `impact_load` and `cycle_code`, but not `threat_rank`; it is not a valid substitute for this regression question.

The EDA runner looks for `Train.csv`, `Test.csv`, and `SampleSubmission.csv` under `data/raw/regression/`. Alternatively, set `TP2_DATA_DIR` to a directory containing `regression/`. File discovery ignores filename case but stops on ambiguous duplicate names. The target is accepted only when `threat_rank` is the sole train-only column and the sample submission has exactly `record_id` and `threat_rank`. A missing or ambiguous target remains blocked. This schema confirms the target column name, not its numeric scale or domain meaning; those remain UNKNOWN until an official definition is available.

Run from any working directory after installing `requirements.txt`:

```powershell
python "<TP2 project root>/scripts/samuel_eda.py"
python "<TP2 project root>/scripts/build_samuel_notebook.py"
```

The CLI writes `reports/eda/samuel_regression_eda.json`; when inputs are valid it also writes PNG figures beside the report. The notebook is `notebooks/samuel/01_impact_threat_cycle.ipynb` and executes from the repository root or a nested folder. In the current blocked state, it records the missing official files without emitting fabricated numbers or plots. Reports store relative filenames or generic `TP2_DATA_DIR/...` labels, not absolute user paths.

Once the regression data is available, the analysis checks column presence, parse failures, non-finite values, ranges, quantiles, and univariate IQR outliers. It calculates Pearson and Spearman associations on complete finite pairs, summarizes target spread within common equal-width `impact_load` bins, and compares distributions and correlations by `cycle_code` and by three contiguous groups of its sorted observed values. The period split is chosen without consulting `threat_rank`; the impact-bin edges are held common across periods. Test `impact_load` and `cycle_code` availability is checked, while test target values are never used.

The cycle chronology is an explicit working assumption until its domain meaning is confirmed. Descriptive differences do not prove temporal drift or a feature effect; correlations do not establish causality or statistical significance. Outlier flags use 1.5 × IQR and do not prescribe deletion. This work performs no predictive modeling, tuning, or submission generation.
