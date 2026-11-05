# Root diagnostic decision — B_ONLY, no fifth LP

Root reran `python scripts/verify_reg_final_wave_followup_v2.py --verify-diagnostics`: PASS_INDEPENDENT_DIAGNOSTIC_RECOMPUTATION, unique15953 rows, D1/D2 metrics verified. Source7798e5a, R3freeze70b49ff7dea59edf53622106392c1c5d943b46dcf0f4e792f5211b83453061b8; diagnostics file136d4a8c7a565413ec550dca1212f7cda22a2c8ca5d70dbfdd6672ceb14961d8. Four starts/four successes, zero failures. No fifth LP or CSV is authorized through the old paired production command.

| View | Causal GLOBAL MAE | RECENCY_H8 MAE | GLOBAL_PRESENT_ONLY MAE |
|---|---:|---:|---:|
| D1 W5,7462rows |11.416864|11.417621|11.412228|
| D2 W6,8491rows |9.959275|9.959656|9.945317|
| Unique pooled,15953rows |10.641061|10.641617|10.631463|

H8 loses about0.000556 pooled and misses predeclared0.02 gain: RETIRE_THIS_H8_RECIPE. No fifth production LP, no H8CSV or half-life rescue. This near-null result does not establish zero temporal drift generally.

Present-only gains0.009598 pooled with gains in bothwindows, supported missing MAE13.684998->13.536771(n1033), no>0.25 subgroup risk flag. Gain is small; private/public improvement unknown. ADVANCE_ONE_B_ONLY_ARTIFACT. Fixed policy uses EXISTING full-support global affine predictions on present rows and restores ORIGINAL saved direct-model output on missing rows. No coefficient refit or new learner. Two causal calibration windows reuse development labels; final existing-global coefficients are selected on all20903 OOF rows, so do not claim heldout performance for the exact final coefficient values.

Material reporting correction: diagnostics short CLI prints official_labels_accessed:false via a default because the result omits the field. Actual Train audit parsed20903 official OOF targets for approved diagnostics. Preserve raw console and diagnostic audit, document discrepancy; do not change frozen source now or propagate false claim. No Test-label/Sample access occurred.

## B-only additive application milestone

Luna2 may implement/test/freeze and build exactlyONE zero-fit B-only CSV, with a new source/application namespace and independent replay QA. Do not call the existing paired --run-production path, create ROOT_PRODUCTION_PASS for it, or modify R3 runner/verifier/protocol to bypass its required H8 fit. The four-fit diagnostic phase is closed. Preserve all R1/R2/R3 provenance and terminal accounting; commit named diagnostic artifacts coherently before opening the separate B-only application source. Later new-stage ancestry is distinct from the closed R3 execution checkpoint; do not rewrite old freezes.

Use saved global CSV (SHA4abc56101e1863520771c125ad9d1f61e9f1530ee43575f2ae0cbd49331d4b56), signed global state/manifest, original direct CSV (SHA49ae7f668aea99e433c2bec72ed5ff51126fb12d888c3185ec048730d767ea6f), signed direct archive/manifest and exact1062-row missing mask. Existing affine parameters remain2.169196428141562,0.9744451489746011. Exactly29421unique ordered IDs, record_id/threat_rank schema, finite0..100, present bit-exact global, missing bit-exact direct. Hash-freeze source/input mapping before selective saved E79 array/CSV loading. No raw Test/Sample/new inference or Train-target access needed. No candidate metric computation on Test or public-score-guided choices. Independent QA must separately rebuild row-aligned prediction formula and check no duplicate existing candidate vector.

Callback Sol with one CSV/hash/manifest/sourcecommit/QA/changecounts/range and frozen current-best comparison. No Kaggle upload by Luna. Root will independently review then decide one measured upload under Sam's standing authorization. No further fits/research/new variants, no user questions/schedules/polling. CLS evaluation can use heavy lock now; B-only metadata/source prep is light, any matrix/resource-heavy work must still honor shared lock.
