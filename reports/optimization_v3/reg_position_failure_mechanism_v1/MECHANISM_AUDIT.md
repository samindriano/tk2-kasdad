# REG_POSITION_FAILURE_MECHANISM_V1

**Status: PASS — zero-fit descriptive audit.** This is adaptive repeated-development evidence, not untouched validation. E67_H12 contains the E75_W6 records. The REG_POSITION_DIRECT_L1_V1 FAIL verdict is preserved. This report diagnoses saved outputs; it does not change selection, score a new policy, or establish causality.

Created UTC: 2026-11-04T10:50:58.522104+00:00
Frozen direct signature: `80614c292413e00fd4234bb1d4da4f3b40f2aea019bb6d510dab035e356b4003`; continuation `58d74e80caa1236d105389d52b5875fe05c7dd9d62b7aadd1011fe718f39bf57`; repair `01ca59f661873fb8cba68dac1bcdcf61dfed1309bd20fc19488c596ce8fe572c`.
Query boundary: 20,903 already-scored labels, cycles 68-79; zero fit-prefix target rows parsed. Test, Sample and Kaggle were not accessed.

## Frozen score verification

Recomputed `10` origin/view score blocks from saved prediction bundles. Every saved R06, arm MAE, and arm gain reproduced within 1e-10.

## Direct versus residual target scales

`arm_window_metrics.csv` reports direct auxiliary MAE against `threat_rank`. Residual auxiliary MAE uses the correct origin-specific target `threat_rank - that origin's saved R06`; those residual losses are not compared across origins or against direct target-scale losses. Final policy MAE is always evaluated against the same official threat_rank labels.

The table includes each arm's within-cycle Spearman (summarized by unweighted cycle mean/median in `within_cycle_metrics.csv`), final policy MAE, R06 gain and REAL-vs-control differences. H12 and W4/W5/W6 use the frozen cycle windows.

## Rank50 and matched W6 decomposition

`direct_rank50_decomposition.csv` recomputes the saved full-cycle map before the missing fallback. Rank shift is exactly `0.5 * (percentile - raw)`; it reports signed mean/median/p10/p90, raw/mapped auxiliary MAE, rank-shift direction versus the R06 residual, and the actual frozen policy correction direction versus that residual. Present and missing rows are separate; missing policy outputs remain exact R06.

`matched_w6_origin_changes.csv` separates current R06 improvement from auxiliary quality and final policy gain on the same 8,491 IDs/cycles/labels. Direct raw auxiliary scores use threat_rank. Residual heads use origin-specific residual targets and compare only to that origin's zero-residual reference. Per-cycle signs and same-cycle R06 catch-up are in `matched_w6_by_cycle.csv`.

### What the saved outputs support

On matched W6, R06 MAE improves by 1.731837 from E67 to E75. Direct REAL raw-head MAE improves by 1.559399, so its raw-head deficit versus R06 widens by 0.172438. This is evidence of baseline catch-up relative to direct auxiliary quality; the absolute head loss itself improves.
The existing rank50 map lowers direct REAL auxiliary MAE by 0.299736 at E67 W6 and 0.093777 at E75 W6. Even after mapping, auxiliary MAE remains worse than R06 by 0.295432 and 0.673828. E75's saved final policy harms R06 for all three arms.
REAL beats BAG/SHUFFLED by 0.029528/0.023266 at E67 W6, then trails them by 0.008499/0.013145 at E75 W6. That sign change is descriptive; every E75 direct arm still harms R06, and this audit cannot attribute the change causally to position.

The earlier residual REAL head has within-origin gain versus zero-residual of -0.702560 at E67 and 0.049093 at E75, with final policy gains 0.021282 and 0.151424. Residual heads target different origin-specific residuals; their absolute losses are not compared across origins or to direct MAE.

## Fixed groups

`fixed_group_metrics.csv` uses disjoint fixed target groups (<=5, >5 and <95, >=95) and the already-frozen V4 fit-prefix text_load quartiles. Cells with n<100 are UNKNOWN; no bins were tuned or refit.

## Historical novelty and next lane

The broad rank50+blend and residual-position transport ideas are not novel here. Direct-target rank50 decomposition is a new diagnostic of the frozen failed pilot, not evidence for another candidate.

The frozen direct pilot fails on E75 W6 against R06, BAG and SHUFFLED; prior executed rank50 blends and residual-position transport diagnostics cover the nearest candidate families. This post-hoc decomposition cannot isolate a causal positional-information effect. No new fit or recipe lane is justified by this audit.

**Recommendation: NONE_JUSTIFIED.** Keep the frozen candidate FAIL. Any new model fit, preprocessing fit, rank/weight variation, E79 run, or submission would need separate authorization.

## Scope and reproducibility

No fitted model was loaded. No fit/predict call, threshold/weight grid, Test/Sample/Kaggle access, submission CSV/export, submission, ledger mutation, or source/cache write occurred. Six diagnostic CSV tables were written in this report directory; zero plots were created.

Reproduce with the three explicit paths recorded in AUDIT.json and run `python scripts/reg_position_failure_mechanism_v1.py --source-root ... --saved-artifact-root ... --repo-root ...`. The output hashes and source signatures are in INPUT_HASHES.json and AUDIT.json.

Input files and saved prediction/receipt bundles are SHA-256 pinned in INPUT_HASHES.json. Output CSV hashes are pinned in AUDIT.json.
