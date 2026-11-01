# C16 and R22_ORIGINAL submission status

As of 2026-11-01, both authorized full-Train submissions are prepared and locally validated. Neither was uploaded. No public score or Kaggle submission reference exists for either new CSV.

## Internal validation

### Classification — C16 experimental benchmark

- Continuation verdict remains **gate-failed**. C16 failed the primary gain gate, late W5+W6 gate, and W2 text-load Q4 subgroup guard. This upload authorization does not qualify C16.
- Existing internal validation: primary Macro F1 0.830248 versus 0.829944 baseline (gain +0.000304); H12 0.805802 versus 0.802966 (gain +0.002837); W5+W6 gain −0.001078. W2 Q4 has n=1,273, Macro F1 change −0.011726 and positive-recall change −0.019672.
- Full-Train fit used all 50,608 official Classification Train rows, cycles 47–79, two threads, and the frozen C16 LightGBM/TF-IDF recipe. Present-stream routing is 0.75 C2_B + 0.25 C16, with exact CB_MISSING fallback and threshold 0.50.
- C2_B and CB_MISSING predictions were reused only after verifying their 50,608-row fit fingerprints, recipe/state, historical source hashes, cache hashes, and ordered Test IDs. C16 itself was newly fit.
- Output: `submissions/c16_r22/classification_c16.csv` — 29,421 rows; integer labels 0/1; label counts class 0: 22,716 and class 1: 6,705; CSV SHA-256 `6cae193f6e1d301cb9ee8556a1eac649b3940e86280180373a0f31235fd3db60`.
- Fitted-state SHA-256: `3c7633ca37165e0a27839235248358fbc82e22b41ee675b40a152814b3cf6127`.

### Regression — R22_ORIGINAL

- Continuation selected and qualified **R22_ORIGINAL**. No PCT_PRESENT, R22+R06, R16, or other recipe was used.
- Existing internal validation: primary MAE 11.519691 versus 11.559390 champion (gain 0.039699); H12 MAE 13.922659 versus 14.047533 (gain 0.124873); W5+W6 gain 0.021379; five of six windows are nonnegative.
- Full-Train XGBoost pseudo-Huber fit used all 50,608 official Regression Train rows, cycles 47–79, two CPU threads, and the frozen structured + TF-IDF/SVD32 recipe. The output combines 75% of the complete three-component champion (LGB SVD32, Hybrid Ridge, sparse LGB-L1) with 25% of R22's rank50 component; cycle ranks use average ties over each complete Test cycle, followed by clipping to [0,100].
- Output: `submissions/c16_r22/regression_r22_original.csv` — 29,421 rows; finite values in [0,100]; CSV SHA-256 `0bd99ee1524a87a7a38fa6e4178ed2da0197759179297bf05b4ac67bfe4bbeab`.
- Fitted-state SHA-256: `0ffdba818f16973abd40903d033331ee92753b9c274e8e7a8b73e27b8902e046`.

Internal metrics above are temporal Train validation results. They are not Kaggle public scores.

## Kaggle public status

- C16: **not submitted**, no submission reference, score, or scoring status.
- R22_ORIGINAL: **not submitted**, no submission reference, score, or scoring status.
- Existing comparator scores are not scores for these candidates: Classification ref `57837821` scored 0.82908; Regression ref `57835122` scored MAE 11.3597.
- Local status ledgers were scanned (10 ledger files); neither authorized message was already present locally. This does not establish the complete live shared ledger or current quota.
- Last authenticated preflight snapshot is from 2026-10-31 12:23 UTC. It recorded team `kasdud lovers`, deadline `2026-11-08 16:55:00` API time, rules hash `fe265a808f19492ff8103a627adcd5f91ba0928796f004b464b303c9fd09cffe`, and historical `num_allowed_now` of 7 for Classification and 5 for Regression. These values are stale and were not used to authorize an upload.

## Blocker and audit trail

The authenticated Kaggle CLI/API could not be used, so current rules, team, deadline, live shared quota, and complete live ledger/idempotency state could not be checked. The configured temporary CLI executable lacked its virtual-environment `Scripts/python.exe`; prior read-only CLI probes returned exit code 1 without output. The global Python environment had no Kaggle module, and the standard Kaggle token file and `KAGGLE_*` environment entries were absent. No credentials were searched for elsewhere, no Chrome control was used, and no upload was attempted.

Upload remains blocked until an authenticated Kaggle API/CLI runtime is available and all required live checks pass. The authorization remains limited to one upload per competition; the prepared CSVs have not consumed either authorization.

## Provenance and QA

- Preparation source commit: `1d3c6bf46fe7b2eff5ef9ead9cc9c2f4156e365f` (based directly on authorized start `6ffbb80c19277279ce022536dab26fd69c539f21`). The config and source hashes were frozen before either fit. Execution-freeze SHA-256: `8eb7b0ee930e4c62b46cc2ecdb7b3ef67d45d7250325e5aa1eaf581ccb7346f0`; contract SHA-256: `97396a821c96b33e75ce9bcd0e82929e31f2663a011666c2c74b0a9c8ca241ad`.
- Official Train counts: 50,608 each. Official Test and SampleSubmission counts: 29,421 each. Test IDs/order match SampleSubmission and the frozen ordered-ID hash.
- Official input hashes and full component manifests are recorded in [`SUBMISSION_STATUS.json`](SUBMISSION_STATUS.json) and [`CONTRACT.json`](CONTRACT.json).
- Focused preparation tests: 10 passed. Independent CSV, ID, range, threshold, exact fallback, and R22 blend reconstruction checks passed.
- CSVs and prediction-array artifacts remain Git-ignored. No push was performed.
