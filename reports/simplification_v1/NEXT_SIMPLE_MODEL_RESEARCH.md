# Upgrade the simpler models — root decision

Sam explicitly asked to continue research after each result without waiting for another routine instruction. Root remains responsible for independent review and next decision; worker milestones do not grant unlimited new fits or uploads. Preserve champions and all prior research checkpoints. No schedules or active polling.

## Confirmed simplification trade-off

Root independently checked authenticated Kaggle CLI history. Classification NO_POS ref58024217 is COMPLETE0.82951 versus champion0.83225. Predictor blocks4→3 means **25% fewer fitted predictor blocks**, while public F1 loss0.00274 is0.274 percentage points or about0.3292% relative to champion F1. Regression NO_POS ref58024398 is COMPLETE11.03430 versus full hinge ref58019539 at11.01510; predictor blocks5→4 means **20% fewer predictors**, with MAE increase0.01920 or about0.1743%. Regression NO_CHAR ref58024430 is COMPLETE11.09980. No measured latency/RAM percentage improvement is established. Shared preprocessing and A02 donor memory are separate complexity dimensions.

The smaller models retain much of the score but neither is the best observed public model. Local NO_POS gains remain unchanged and do not override public counterevidence. Private scores are unknown. No final submission selection changes.

## REG — native calibration for four-leaf R06

Luna2 prepares reg_r06_native_hinge_v1. The previous simpler submission transported hinge coefficients learned on full DIRECT. Test whether fitting the SAME three-parameter p50 hinge directly to R06 predictions removes that mismatch. No new base model, feature transform, rank policy, knot or grid.

Fit D1 on E67 R06 cycles68–71, score72–75; fit D2 on E67 R06 cycles68–75, score E75 R06 cycles76–79. Same cached labels/orderedIDs and same all-row calibration population as the original hinge; present-only application, rawR06 missing fallback. Compare against R06 with old full-model hinge, original full DIRECT+hinge, and rawR06. Report primary gain against transported hinge and the full-model gap separately. Two diagnostic calibrator fits plus separately approved conditional one-final slot on20,903 unique R06 OOF rows; no old-budget reset. Same summed absolute-error objective, slopes[0,2], knot50, clipping only at application.

Qualification: pooled gain>=0.005 over transported hinge, both window gains>=0, maximum supported availability/tail harm<=0.25. Exploratory review: pooled gain>=0, each window harm<=0.02, group harm<=0.25. Support n>=200 for groups, n>=50 for cycle diagnostics. Keep individual windows, p90 and missing/tail evidence visible. No fresh-holdout claim. Source, independent verifier, actual default-CLI/receipt E2E tests and real non-target preflight require root source approval before official fits. Root then reviews results, conditionally releases final application and at most one measured upload without asking Sam again. No guarantee of improvement.

## CLS — ranking versus decision allocation

Luna1 executes zero-fit CLS_SIMPLE_RANK_ALLOCATION_DIAGNOSTIC_V1, comparing frozen A02 pA with complex0.5pP+0.5pA. Missing predictions and operational threshold remain identical. For each origin-cycle, fix K to the complex model's present predicted-positive count (no labels), then rank A02 into that count with deterministic record-ID ties. Also reverse the comparison at the simple model's original count. Origin-level count matching is a declared sensitivity. Never select optimalK or derive a deployment quota from these diagnostics.

Report present AUC/AP, confusion/TP/FP/F1, operational full-row and present-only metrics, E67 W4/W5/W6/H12, E75 W6 and unique20,903-row pool. Freeze input/source/rules first; no raw Train labels, fit, Test/E79, candidate CSV or upload. Existing A02-versus-Sparse25 allocation research is not the same comparison; check a bounded four-document novelty scope. Independent confusion/ID/tie/overlap QA. One memo plus machine summary and at most two plots. Use evidence to recommend at most one cheap calibration follow-up or explain why POS information cannot be recovered by thresholding; both temporal reuse and public-selection bias stay explicit.

## Continuation rules

The previous global-strength E79 replay and complex-model center-local final application remain held at their checkpoints, not discarded. Current priority is these simpler-model directions. Workers return complete evidence to root before final. Root reviews, makes one next bounded decision, and dispatches without human approval repetition when it stays within Sam's authorized scope. A true technical/resource/auth blocker is preserved and reported, not bypassed or polled indefinitely. Existing lightweight parallel policy permits one small job per lane with shared aggregate reserve; heavy reconstruction/fits retain their specific resource gates.
