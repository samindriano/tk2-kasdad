# Center-local residual correction R1 — source preparation

## Authorization boundary

This policy prepares one prospective diagnostic pilot. It does not authorize a
fit or score. Actual calibrator fits and score-target reads require a separate,
valid `ROOT_SOURCE_PASS_R1.json` after source review. The preparation preflight
may hash the cached R4 bundles and read only non-target identity/prediction
arrays. It never materializes their `targets` arrays.

The prior R1 residual memo's recommendation of no follow-up remains intact. The
root decision to prepare this candidate is a documented departure based on the
repeated `[40,60)` signal. This center and support were selected after looking at
those diagnostics; any later result is adaptive, not a fresh holdout. Broader
novelty remains unknown beyond the six frozen executed summaries.

The later root callback must be an immutable, canonically signed
`ROOT_SOURCE_PASS_R1.json` with verdict `PASS`, experiment id
`REG_CENTER_LOCAL_CORRECTION_V1_R1`, pass kind `CENTER_LOCAL_DIAGNOSTIC`, current
`source_commit`, and exact `prefit_freeze_sha256` and
`identity_preflight_sha256`. It must authorize exactly the four ordered
calibrator ids under cap four, set `authorized_calibrator_fits=4`,
`base_model_fits=0`, `preprocessing_fits=0`, `official_lp_starts=0`, and
`prior_three_slot_budget_reset=false`; it must set
`score_target_access_authorized=true`, `test_or_sample_authorized=false`, and
`kaggle_upload_authorized=false`. Any missing, changed, or extra authorization
does not open the fit or score boundary. Its exact key set is `verdict`,
`experiment_id`, `pass_kind`, `source_commit`, `prefit_freeze_sha256`,
`identity_preflight_sha256`, `authorized_fit_ids`, `calibrator_fit_cap`,
`authorized_calibrator_fits`, `base_model_fits`, `preprocessing_fits`,
`official_lp_starts`, `prior_three_slot_budget_reset`,
`score_target_access_authorized`, `test_or_sample_authorized`,
`kaggle_upload_authorized`, and `signature`.

## Fixed recipe

For each saved same-history fixed-p50 hinge base `g(p) = a + sL*min(p,50) +
sH*max(p-50,0)`, define `b(p) = max(0, 1 - abs(p-50)/10)`. Fit `LOCAL` by
minimizing the unweighted sum of absolute residuals on present fit rows,
`sum(abs(y - (g(p) + delta*b(p))))`, with `delta` constrained to `[-2,0]`.
Use the exact lower weighted median of `(y-g(p))/b(p)` with weights `b(p)` for
`b(p)>0`; rows with `b=0` add a constant to the objective. Resolve exact-half
ties to the lower minimizer endpoint, report the full bounded minimizer
interval, and do not substitute an unweighted median.

Fit `GLOBAL_CONTROL` on the same present fit rows by minimizing
`sum(abs(y - (g(p) + delta_global)))`, with `delta_global` in `[-2,0]`. Select
the lower empirical median, and report its bounded minimizer interval. This is
a same-base constant residual shift; it is not the pre-existing affine GLOBAL
model and does not match the LOCAL arm's affected-row count or effective
complexity.

Do not clip either objective. At application, clip present predictions to
`[0,100]`; missing predictions remain raw DIRECT float64 bits. The LOCAL
function is continuous. Verify its minimum derivative from `sL + delta/10` on
`(40,50)`, `sH - delta/10` on `(50,60)`, and saved slopes outside. Do not adjust
the recipe if monotonicity fails. Confirm the LOCAL change is exactly zero
outside `[40,60]` before clipping.

## Fixed populations, fits, and gates

Use saved R4 D1 and D2 bundles only. D1 fits E67/cycles 68–71 (4,950 total,
4,608 expected present) and scores E67/cycles 72–75 (7,462). D2 fits
E67/cycles 68–75 (12,412 total, 11,563 expected present) and scores
E75/cycles 76–79 (8,491). The two score windows must contain 15,953 unique
record IDs. Do not pool away the separate window results.

The only four possible calibrator fit IDs, in order, are `D1_LOCAL`,
`D1_GLOBAL_CONTROL`, `D2_LOCAL`, `D2_GLOBAL_CONTROL`. This is a new cap-4
diagnostic ledger; the prior three-slot model-fit budget stays complete and is
not reset. Base/preprocessing fits are zero. There is no automatic retry or
rescue. Require at least 200 present D1/D2 fit rows with `b(p)>0` for LOCAL;
otherwise mark unsupported and stop without forcing the fit.

Freeze all source and cached-input byte hashes before fitting. During a fit,
verify the non-target row identity again before reading fit targets. Persist
score predictions without target values and freeze their hashes before the
separate score command is allowed to load score targets. Report fit and score
full-row and present-only MAE, mean error, p90 absolute error, fixed raw-DIRECT
bin/cycle/tail groups, support, gains, target-region gain, monotonicity, and
outside-region exact-zero effects. Bin and target-tail support minimum is 200;
cycle support minimum is 50. Unsupported results are `UNKNOWN`, never zero.

Mechanism gate: LOCAL must beat both unchanged same-history HINGE and
GLOBAL_CONTROL by more than `1e-12` in each score window, and both LOCAL fit
deltas must be negative. Candidate gate: pooled all-row score MAE gain over
HINGE must be at least `0.01`, both window gains nonnegative, and maximum
supported all-row subgroup harm versus HINGE at most `0.25`. Exploratory root
review is allowed only if pooled gain is nonnegative, each window's harm versus
HINGE is at most `0.02`, and subgroup harm is at most `0.25`. Report all gates;
do not alter them. A positive result remains adaptive evidence, not a fresh
holdout.

## Frozen novelty scan

The bounded scan covers six executed protocol/decision/result summaries only:

1. `regression_residual/FINDINGS.md`
2. `regression_position_frontier_v1/FINDINGS.md`
3. `reg_position_transport_diagnostic_v1/TRANSPORT_DIAGNOSTIC.md`
4. R4 `SOL_DIAGNOSTIC_DECISION_R4.md`
5. R4 `PROTOCOL_R4.md`
6. R3 `PROTOCOL_R3.md`

No exact triangle-at-50/width-10 correction or matched same-row constant-offset
HINGE control appeared in these six documents. The earlier shallow residual
head was harmful; the position-aware pilot was mixed; the transport diagnostic
did not evaluate a new policy. Exact novelty elsewhere in the repository is
`UNKNOWN`.

## Resource and access controls

Use the current `small_calibration_lp` RAM policy for the four calibrator fits,
one shared OS lock, one job, and at most two threads. The non-target identity
preflight uses the saved-prediction evaluation policy. Lock-file existence is
not evidence that the OS lock is held. No Test, SampleSubmission, E79 inference,
CSV, or Kaggle access is in scope.
