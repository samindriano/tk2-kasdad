# CLS_PPMI_CONTEXT_V1 corrected scoring — CORRECTION_R2

Corrected original-campaign scientific gate: **FAIL**.
Exploratory eligibility (unchanged): **NOT_ELIGIBLE**.

The original R1 result/report remain preserved as history. R1 consumed its one selected read but failed during in-memory root-anchor verification; R2 is the separately authorized recovery pass.

## Corrected MacroF1 views

| View | Rows | Champion | Real candidate | Shuffled control | Real − champion | Control − champion | Real − control |
|---|---:|---:|---:|---:|---:|---:|---:|
| E67_W4 | 4950 | 0.844337416 | 0.845949846 | 0.842425731 | +0.001612430 | -0.001911685 | +0.003524115 |
| E67_W5 | 7462 | 0.806246025 | 0.804230271 | 0.804143053 | -0.002015753 | -0.002102972 | +0.000087219 |
| E67_W6 | 8491 | 0.792506512 | 0.794503818 | 0.785837743 | +0.001997306 | -0.006668769 | +0.008666075 |
| E67_H12 | 20903 | 0.811327491 | 0.811899674 | 0.807557515 | +0.000572183 | -0.003769977 | +0.004342159 |
| E75_W6 | 8491 | 0.818884188 | 0.816245949 | 0.816381805 | -0.002638238 | -0.002502382 | -0.000135856 |
| PRIMARY_UNIQUE | 20903 | 0.821940768 | 0.820673218 | 0.819785000 | -0.001267549 | -0.002155768 | +0.000888218 |

## E75 gate semantics

The original ROOT_CAMPAIGN E75 gate is control-shuffled minus champion >= 0. R1 introduced a later amended comparison, real candidate minus shuffled control >= 0. Both are shown separately; R2 preserves the original gate as the scientific decision contract.
- Original control-shuffled minus champion: -0.002502382 — **FAIL**.
- R1-amended real candidate minus shuffled control: -0.000135856 — **FAIL**.
- Original corrected scientific gate: **FAIL**.
- R1-amended scientific gate: **FAIL**.

## All-view guards

- Worst supported subgroup MacroF1 harm: 0.010934904.
- Worst supported subgroup positive-recall harm: 0.033082707.
- Supported groups by view: `{"E67_H12": 8, "E67_W4": 8, "E67_W5": 8, "E67_W6": 8, "E75_W6": 8, "PRIMARY_UNIQUE": 8}`.

## Read accounting

- Original evaluation: 50,608 target values in one read.
- Failed R1 correction: 20,903 selected values in one read; no result was persisted.
- R2 recovery: 20,903 selected values in one read; independent verification reused this vector.
- Cumulative: 3 target-column read calls and 92,414 materialized target values; 50,608 unique Train rows; 0 additional verifier reads.
- No fit, prediction, policy, threshold, or support change. Test/Kaggle were not opened.

## Decision boundary

This report corrects scoring and gate accounting only. The historical original output remains invalid for the intended MacroF1/windows/subgroup contract, and the root retains family disposition and next-lane ownership.
