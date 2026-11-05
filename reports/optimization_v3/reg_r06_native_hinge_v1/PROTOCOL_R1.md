# R06-native p50 hinge pilot (R1)

## Purpose and scope

Test whether the existing three-parameter p50 hinge transports poorly from the
full DIRECT prediction vector to simpler R06 predictions. The pilot reuses the
exact saved R06 OOF predictions and the cached 20,903 OOF labels/IDs. It adds no
base learner or preprocessing fits.

The prediction source is R06 fit67 for cycles 68–75 and R06 fit75 for cycles
76–79. R06 fit71 is not used. D1 fits E67 cycles 68–71 and scores E67 cycles
72–75. D2 fits E67 cycles 68–75 and scores E75 cycles 76–79. The conditional
final fit uses all 20,903 unique rows, including missing rows.

## Frozen model and application

Fit the three-parameter p50 hinge:

    intercept + slope_low * min(p, 50) + slope_high * max(p - 50, 0)

Use the existing R4 HiGHS solver, with slopes bounded to [0, 2], an unweighted
sum-absolute-error objective, two threads, and the existing 180-second solver
limit. Keep the p50 knot fixed. Do not search knots, grids, weights, or seeds.

Fit all rows, including those marked missing. At application, apply the
calibrator only to present rows, clip to [0, 100], and preserve missing-row
predictions bit-for-bit from raw R06.

## Comparators

1. Primary: same-history R06 with the saved old DIRECT-trained HINGE_FINAL
   coefficients.
2. Separate full-model comparison: the original DIRECT predictions with those
   same saved HINGE_FINAL coefficients.
3. Raw R06.

The primary gain is candidate MAE improvement over comparator 1. Comparator 2
is reported separately and does not determine the primary gate. The old
DIRECT-trained hinge was fit on the full 20,903-row OOF bundle, so its comparison
on D1/D2 score rows reuses labels. The result is exploratory and makes no fresh
holdout claim.

## Gates and final slot

Qualification requires pooled gain versus the primary comparator of at least
0.005, nonnegative gain in each diagnostic window, and supported
availability/tail group harm below 0.25. The exploratory gate requires
nonnegative pooled gain versus raw R06, harm below 0.02 in each window, and
supported group harm below 0.25. Groups are present/missing availability,
target <= 5, and target >= 95; a group is supported at n >= 100. Any
unsupported required group is UNKNOWN and prevents a gate pass.

The new pilot has its own append-only three-slot ledger: D1, D2, then FINAL.
FINAL is available only if both diagnostic gates pass. The previous R4 ledger
already has three starts and remains unchanged. The final fit is a calibration
fit on the complete OOF bundle, not a validation result.

## Source and execution gates

The freeze, preflight, and synthetic QA commands are the only commands needed
for the source-ready handoff. They do not load target values, read Test labels,
create a CSV, access Kaggle, or start an official LP. The preflight checks IDs,
cycles, physical rows, source origins, cache hashes, missing fallback identity,
resource floors, and the shared heavy-job lock.

The root reviewer creates
experiments/reg_r06_native_hinge_v1/ROOT_SOURCE_PASS_R1.json only after
reviewing the exact freeze, preflight, source hashes, and synthetic receipt.
The executor requires that pass before loading the cached target member or
starting a fit. The local runner never creates or edits the root pass.

Commands:

    python -m scripts.reg_r06_native_hinge_v1 freeze
    python -m scripts.reg_r06_native_hinge_v1 preflight
    python -m scripts.reg_r06_native_hinge_v1 synthetic-qa
    python -m scripts.reg_r06_native_hinge_v1 verify-source-ready

After an explicit root source pass, the later executor commands are:

    python -m scripts.reg_r06_native_hinge_v1 run-diagnostics
    python -m scripts.reg_r06_native_hinge_v1 run-final
