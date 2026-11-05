# R06-native p50 hinge pilot (R2C)

## Scope and frozen rows

R2C is the corrected source generation for the two-window calibration
diagnostic. Earlier R2 and R2B freezes/receipts remain preserved as superseded
evidence. R2C uses only saved R06 OOF predictions: fit67 for cycles 68–75 and
fit75 for cycles 76–79; fit71 is excluded. D1 scores 7,462 rows in cycles
72–75, D2 scores 8,491 rows in cycles 76–79, for 20,903 unique OOF rows. No
base or preprocessing fit is added.

Fit the fixed three-parameter p50 hinge with the saved R4 HiGHS solver, slopes
in [0, 2], unweighted sum-absolute-error, two threads, and the existing
180-second limit. Fit on present and missing rows. Apply to present rows, clip
to [0, 100], and preserve missing rows bit-for-bit from raw R06. Do not search
knots, grids, weights, or seeds.

## Past-only comparators

D1 uses saved R4 `HINGE_D1` coefficients fit on cycles 68–71; D2 uses saved R4
`HINGE_D2` coefficients fit on cycles 68–75. Each same-cutoff state supplies
both the same-history R06 comparator and the original DIRECT comparator. R2C
pins and verifies the R4 root pass, diagnostic replay, fit ledger, saved states,
and fit-input hashes.

The full-data `HINGE_FINAL` state is descriptive/deployment context only. Its
coefficients never enter D1/D2 diagnostic predictions or gates. Raw R06 is
reported as a diagnostic comparator and does not determine the primary gate.

## Metrics and gates

Report MAE, p90 absolute error, and mean prediction error for full rows,
present-only rows, every scored cycle, and availability/tail groups. Report the
candidate, same-cutoff R4 hinge on same-history R06, same-cutoff R4 hinge on
DIRECT, and raw R06. A group is supported at `n >= 200`; a cycle at `n >= 50`.
Unsupported groups/cycles are disclosed as `UNKNOWN` and block scientific gate
passes.

Qualification requires all groups and cycles supported, pooled gain versus
same-cutoff primary at least 0.005 MAE, gain in each window at least 0, and
maximum supported availability/tail harm at most 0.25 MAE.

Exploratory review uses the same cutoff primary: pooled gain at least 0, harm
in each window at most 0.02 MAE, and supported group harm at most 0.25 MAE.
All boundaries are inclusive. Unsupported required groups or cycles block both
gates. Raw R06 comparison remains diagnostic.

`root_review_eligible` is qualification OR exploratory. It requests root review
only. R2C source approval authorizes D1/D2 only and leaves final-fit
authorization false.

## Separate final-fit approval

The FINAL slot is full-OOF calibration, not holdout validation. It requires a
separate immutable
`experiments/reg_r06_native_hinge_v1/ROOT_FINAL_FIT_APPROVAL_R2C.json` after root
review. The canonical signature must bind this experiment, `R06_HINGE_FINAL`,
the freeze SHA, identity-preflight SHA, R2C source-pass SHA, diagnostic-report
SHA, source commit, exact source hashes, exact input hashes,
`authorized_fit_ids: ["R06_HINGE_FINAL"]`, and `final_fit_authorized: true`. The
fit state, ledger, and receipt record the approval and diagnostic hashes. No
final fit starts automatically when a gate passes.

## Source and execution gates

The executor accepts the exact R2C source pass, which binds the source commit
as an ancestor of the current checkout plus exact source and input hashes.
Before targets are read, any saved R2C diagnostic slot must match its D1/D2
cutoff identity and append-only ledger; FINAL cannot substitute for either
diagnostic.

Freeze, non-target preflight, synthetic QA, and source-ready verification do
not load target values, read Test labels, create a CSV, access Kaggle, or start an
official LP. Non-target checks cover row identity, R06 origins and hashes,
missing fallback identity, R4 comparator lineage, resource floors, and the
shared heavy-job lock. The `verify-source-ready` command dispatches to the R2C
verifier.

    python -m scripts.reg_r06_native_hinge_v2 freeze
    python -m scripts.reg_r06_native_hinge_v2 preflight
    python -m scripts.reg_r06_native_hinge_v2 synthetic-qa
    python -m scripts.reg_r06_native_hinge_v2 verify-source-ready

After a signed R2C source pass, D1/D2 may start:

    python -m scripts.reg_r06_native_hinge_v2 run-diagnostics

FINAL remains closed until separate root final-fit approval is present:

    python -m scripts.reg_r06_native_hinge_v2 run-final
