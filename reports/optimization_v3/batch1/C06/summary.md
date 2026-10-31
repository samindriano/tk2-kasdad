# C06 — Learnable Ordered Category Novelty

## Decision

**RETIRE after a valid two-fit temporal pilot.** The candidate failed the preregistered primary-gain threshold. No Test, SampleSubmission, Kaggle data, or score was used.

## Frozen recipe and population

- Six appended binary features: prior-cycle novelty and missingness for `org_code`, `module_code`, and `pattern_code`.
- For a training row in cycle *c*, novelty compares only against categories from cycles `< c`; the vocabulary updates after all rows in cycle *c* are encoded. Score rows compare against the complete outer-fit vocabulary.
- The frozen sparse LightGBM parameters, TF-IDF settings, structured preprocessing, C2_B 0.75 / sparse expert 0.25 present-stream blend, CatBoost missing-stream branch, and 0.50 threshold were preserved.
- Official Train: 50,608 rows, SHA-256 `875552a68a3d24b0f3e1966d6ddb5671e6f54305a1c526e2f0a76747942ed91c`. Fit populations were 12,020 rows at cutoff 55 and 29,705 at cutoff 67; scored populations were 17,685 and 20,903 rows. Pilot primary compares 9,064 rows (W1+W4); H12 compares 38,588 rows.
- All six feature flags passed the predeclared minimum of 30 zeros and 30 ones at both cutoffs. Full support counts and per-cycle vocabulary audits are in the frozen recipe identity at `experiments/optimization_v3/batch1/C06/freeze_amendment_01.json`.
- Active freeze signature: `2f50b81a0b000f7e8bc591cbc9796092fba0227f888c83575b7abfcb44db308f`. It binds 21 input hashes and 21 source hashes. The code/test source commit is `ee25c0b`; the active freeze was committed in `de88cda`.

## Results

| View | Rows | Baseline Macro F1 | Candidate Macro F1 | Gain |
|---|---:|---:|---:|---:|
| W1 | 4,114 | 0.803160 | 0.802164 | -0.000996 |
| W4 | 4,950 | 0.841418 | 0.841183 | -0.000235 |
| Primary (W1+W4) | 9,064 | 0.823972 | 0.823397 | -0.000576 |
| H12 | 38,588 | 0.802966 | 0.803058 | +0.000092 |

Primary required at least +0.000500; observed gain was -0.000576. H12 harm, individual-view harm, and supported primary subgroup guards passed. No supported subgroup failures were found. The pilot therefore retires under its frozen gate; it does not justify continuation or model selection.

## Execution and QA

- Predictor fits: **2 started, 2 saved, 0 failed**; cutoff 55 took 6.19 seconds and cutoff 67 took 16.18 seconds (22.37 seconds total). Per-fit sidecars and event receipts are preserved under `experiments/runs/optimization_v3_batch1/C06/` and `reports/optimization_v3/batch1/C06/attempts.jsonl`.
- Preflight estimated peak additional RAM at 1.61 GiB; 7.83 GiB was available at pilot start. Peak RSS was not sampled, so the estimate is not an observed peak.
- Generic independent metric/row/route verification and separate C06 fit-state, source-boundary, and novelty-support reconstruction both passed. Results are recorded in `results.json`, `verification.json`, and `gate.json`.
- Six C06 tests passed, including a two-cutoff synthetic `run_pilot` through independent report aggregation, same-cycle novelty semantics, score-only unseen categories, minimum support, fixed parameters and routing, tied quantile values, empty groups, and string record IDs.
- scikit-learn emitted two feature-name warnings during sparse model inference; both fits completed, both prediction bundles verified, and no convergence or resource failure was recorded.
- Temporal Train windows are development evidence reused across research; they are not an untouched holdout.
