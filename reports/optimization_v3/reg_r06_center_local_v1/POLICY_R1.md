# R06 center-local transfer — source-only protocol R1

## Purpose and decision boundary

This protocol prepares a bounded transfer of the reviewed fixed triangle correction from the complex regression route to the SIMPLE R06 route. The transported comparator is the immutable same-cutoff R4 p50 hinge applied to raw R06 predictions. The failed native R2C states are excluded from the candidate and comparator. This protocol does not authorize official fits or score-target access; those require a new signed root source pass after source review.

The original full-DIRECT R4 hinge replay is a separate diagnostic. The earlier complex `center_local` result is historical context only. Neither changes the primary comparator or the gates below.

## Frozen recipe

- Base: raw R06 SIMPLE prediction `p` plus the saved same-cutoff R4 p50 hinge coefficients `(a, s_low, s_high)`; present predictions use `clip(a + s_low*min(p,50) + s_high*max(p-50,0), 0, 100)`.
- Triangle: `b = max(0, 1 - abs(p - 50)/10)` with center 50 and half-width 10.
- LOCAL: add `delta*b`, where `delta` is the exact lower weighted median of `(y - un-clipped base)/b` over present fit rows with `b>0`, weighted by `b`; constrain `delta` to `[-2,0]`.
- GLOBAL_CONTROL: add one constant `delta` from the lower empirical median of `y - un-clipped base` over the same present fit rows; constrain `delta` to `[-2,0]`.
- Both objectives use unweighted sum absolute error without clipping during fitting. Application clips present rows to `[0,100]`. Missing rows for transported R4-on-R06, LOCAL, GLOBAL_CONTROL, and raw R06 remain bit-identical to raw R06.
- The R4 base must remain nondecreasing for all `delta` in `[-2,0]`. No width, center, or bound tuning is allowed.

## Populations and fit budget

| Stage | Fit population | Present fit rows | Excluded missing rows | Score population |
| --- | --- | ---: | ---: | --- |
| D1 | E67 cycles 68–71 | 4,608 of 4,950 | 342 | E67 cycles 72–75 (7,462 rows) |
| D2 | E67 cycles 68–75 | 11,563 of 12,412 | 849 | E75 cycles 76–79 (8,491 rows) |

Only present fit rows contribute to LOCAL and GLOBAL_CONTROL. The unique pooled score denominator is 15,953. Cycle 74 remains a mandatory D1 harm report. A possible full-history final calibrator is a single conditional slot under the new cap of five; it is not authorized by source preparation and is not executable until a separate reviewed source pass defines its exact base/state mapping. No prior ledger or slot counter is reset. Base-model fits and preprocessing fits are both zero.

## Metrics and gates

Every score window reports overall, present-only, raw-R06 prediction bins, cycles, and target tails `y<=5` and `y>=95`. Each supported group reports n, MAE, mean error, and P90 absolute error. Raw R06 edges are `[0,20,40,60,80,100]`; group support is 200, and cycle support is 50. Unsupported groups remain UNKNOWN. The D1 cycle-74 harm is mandatory and must be supported.

The gates stay separate:

1. **Mechanism:** both LOCAL deltas are negative, and LOCAL improves by more than `1e-12` over both transported R4-on-R06 and GLOBAL_CONTROL in both windows.
2. **Candidate qualification:** pooled all-row LOCAL gain against transported R4-on-R06 is at least `0.01`; each window gain is nonnegative; maximum supported all-row subgroup harm is strictly below `0.25`.
3. **Exploratory review:** pooled gain is nonnegative; harm in each window is at most `0.02`; maximum supported all-row subgroup harm is strictly below `0.25`.

A scientific gate failure does not automatically impose global root benchmark policy. Root retains that decision.

## Data and execution boundary

The source freeze pins the OOF identity bundle, R06 caches and sidecars, R4 predecessor receipts/state/score rows, resource policy, source code, and protocol. Non-target preflight may read identity, missingness, cycles, raw R06 predictions, and immutable R4 states/replays. It must not load the OOF `targets` member. Fit targets can only be read after root authorization. Score targets can only be read after all authorized fits complete and prediction files are frozen. Test/sample, E79, CSV creation, and Kaggle upload remain out of scope.

Expected source-ready artifacts are `experiments/reg_r06_center_local_v1/PREFIT_FREEZE_R1.json`, `experiments/reg_r06_center_local_v1/fit_identity_R1/`, `experiments/reg_r06_center_local_v1/score_identity_R1/`, `experiments/reg_r06_center_local_v1/FIT_POPULATION_IDS_R1.json`, `reports/optimization_v3/reg_r06_center_local_v1/IDENTITY_PREFLIGHT_R1.json`, and `reports/optimization_v3/reg_r06_center_local_v1/SOURCE_READY_VERIFICATION_R1.json`.
