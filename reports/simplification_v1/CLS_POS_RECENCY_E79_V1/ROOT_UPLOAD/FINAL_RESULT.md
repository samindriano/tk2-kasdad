# Root final result — CLS positional recency benchmark

Kaggle confirmed **COMPLETE** for submission **58068197** on 7 November 2026 WIB, public Macro F1 **0.82987**. Filename `classification_cls_pos_recency_e79_v1.csv`; message `Samuel CLS POS Recency H8 Frozen Exploratory Benchmark`. Previous/current best confirmed public remains **0.83225** (ref58001282). Difference -0.00238 Macro F1, not percentage points. Private performance remains unknown.

## Decision

**KEEP_CHAMPION_CLOSE_THIS_RECENCY_RECIPE.** The predeclared exploratory gate permitted this one benchmark, despite the scientific gate failure. Its local positive result did not reproduce on this observed public set. Do not change the old verdict to PASS or search half-life, floor, blend, class normalization or thresholds after seeing public feedback. This result alone does not prove that all recency mechanisms fail, explain the public distribution, or justify a confidence interval without public labels.

The recipe remains original fullTrain C2B bag offset plus weighted positional gamma, blended50/50 original A02 when present, exact operational missing fallback. At local unique20903rows, F1gain+0.000354522; E67H12+0.000980991; E75W6+0.001267681; unique RECENCY-minus-SHUFFLED+0.000392590. E67W5 was negative and only7/12cycle improved. E67/E75 views overlap and all development data have been reused. Local and public figures are separate evidence.

## Root checks and transaction

- Independent application verifier rerun PASS; focused E79/recovery tests10passed.
- Root recomputed sigmoid(X_POSITION@gamma+z_BAG),50/50A02route and both thresholds directly from immutable matrices/arrays; exact equality with saved outputs and CSV.
- 29,421 ordered unique official IDs; exact SampleSubmission header; binary artifact_signal. Positive predictions6,843, originalchampion6759. Total flips200 (0->1:142;1->0:58). These Test label changes do not identify error types without targets.
- All1062missing probabilities bit-exact to originalA02 and labels unchanged from champion.
- Local duplicate scan25complete vectors found no identical candidate; live pre-upload Kaggle history had no filename/message match and confirmed priorchampion0.83225.
- Durable exclusive UPLOAD_INTENT preceded one official CLI upload. Command returncode0 included ref58068197; separate API history then confirmed COMPLETE/score. No retry, final-slot changes, or additional submission.
- Root performed zero new fits or target-label reads. One E79 optimizer was already saved by Luna; all source/preprocessing/failure artifacts remain unchanged.

CSV SHA256 `6e8f6b08e769135717de4ce1f821af8ec696b847849b516d0f8749da4b59da55`. Root review source checkpoint2e4dd42; review commit e690d45; upload intent binds that commit. Manifest recipe/input/source hashes remain in immutable APPLICATION_MANIFEST.json. Its LOCAL_CSV_PREPARED_NOT_UPLOADED status describes construction time; subsequent upload truth is recorded separately here, UPLOAD_INTENT.json, UPLOAD_RESPONSE.json and KAGGLE_STATUS_01.json.

Kaggle submission check: https://www.kaggle.com/competitions/tp2-classification-cyber-vulnerability-kasdd-gasal-2026-2027/submissions . No final leaderboard candidate selection was changed. REG remains paused; no new lane or fit is triggered by this public result.
