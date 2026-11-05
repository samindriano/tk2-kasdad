# R06-native p50 hinge pilot (R2)

## Scope and frozen rows

This is a two-window calibration diagnostic using the already saved R06 OOF
predictions. R06 fit67 supplies cycles 68–75 and fit75 supplies cycles 76–79;
fit71 is excluded. No base model or preprocessing fit is added. The diagnostic
rows are 20,903 unique OOF rows, with D1 scoring 7,462 rows in cycles 72–75 and
D2 scoring 8,491 rows in cycles 76–79.

Each candidate is the fixed three-parameter p50 hinge

    intercept + slope_low * min(p, 50) + slope_high * max(p - 50, 0)

fit by the saved R4 HiGHS solver with slopes in [0, 2], an unweighted
sum-absolute-error objective, two threads, and the existing 180-second limit.
Fit rows include both present and missing observations. Apply the calibrator to
present rows, clip to [0, 100], and preserve missing-row predictions bit-for-bit
from raw R06. Do not search knots, grids, weights, or seeds.

## Past-only comparators

For D1, use the saved R4 `HINGE_D1` coefficients, fit on cycles 68–71, for both
the same-history R06 comparator and the original DIRECT comparator. For D2,
use saved R4 `HINGE_D2` coefficients, fit on cycles 68–75, for those same two
comparators. The R4 root pass, diagnostic replay, fit ledger, saved states, and
fit-input hashes are pinned and checked before the R2 freeze is created.

`HINGE_FINAL`, trained on all 20,903 rows, is descriptive/deployment context
only. Its coefficients never enter a D1/D2 diagnostic prediction or gate. Raw
R06 is also reported as a diagnostic comparator; it is not the primary gate.

## Metrics and support

Report MAE, p90 absolute error, and mean prediction error for full rows,
present-only rows, each scored cycle, and availability/tail groups. Compare the
new candidate with the same-cutoff R4 hinge on same-history R06. Also report the
matching DIRECT comparator and raw R06 in each summary. Group support is
`n >= 200`; cycle support is `n >= 50`. Unsupported groups/cycles are marked
`UNKNOWN`, disclosed, and block scientific gate passes.

## Diagnostic gates and root review

Qualification passes only when all groups/cycles are supported, pooled gain
versus the same-cutoff primary is at least 0.005 MAE, each window gain is at
least 0, and the maximum supported availability/tail harm is at most 0.25 MAE.

The exploratory gate is also against the same-cutoff primary: pooled gain is at
least 0, harm in each window is at most 0.02 MAE, and supported group harm is
at most 0.25 MAE. Every boundary is inclusive. Unsupported required groups or
cycles block both gates.

`root_review_eligible` is qualification **or** exploratory. Eligibility only
requests root review; it does not start the final fit. The R2 source pass may
authorize D1 and D2 only and must explicitly leave final-fit authorization
false.

## Separate final-fit approval

The FINAL slot is a full-OOF calibration fit, not a holdout validation result.
It remains closed unless a separate immutable
`experiments/reg_r06_native_hinge_v1/ROOT_FINAL_FIT_APPROVAL_R2.json` is
provided after review. The approval must have a valid canonical signature and
bind the R2 experiment, `R06_HINGE_FINAL`, prefit-freeze SHA, identity-preflight
SHA, R2 source-pass SHA, signed diagnostic-report SHA, source commit, exact
source hashes, exact input hashes, `authorized_fit_ids: ["R06_HINGE_FINAL"]`,
and `final_fit_authorized: true`. The final fit state, ledger, and receipt record
the approval and diagnostic hashes. No fit starts automatically after either
diagnostic gate passes.

## Source and execution gates

The executor accepts only the exact R2 root source pass, which must bind the
source commit as an ancestor of the current checkout plus the exact source and
input hashes. This allows later receipt commits without weakening source
identity. Before any target access, saved R2 diagnostic slots must match their
own D1/D2 cutoff identities and append-only ledger entries; a FINAL state
cannot substitute for either slot.

Freeze, non-target preflight, synthetic QA, and source-ready verification do not
load target values, read Test labels, create a CSV, access Kaggle, or start an
official LP. The preflight checks row IDs, cycles, physical rows, R06 origins,
cache hashes, missing fallback identity, R4 comparator lineage, resource floors,
and the shared heavy-job lock.

Commands:

    python -m scripts.reg_r06_native_hinge_v2 freeze
    python -m scripts.reg_r06_native_hinge_v2 preflight
    python -m scripts.reg_r06_native_hinge_v2 synthetic-qa
    python -m scripts.reg_r06_native_hinge_v2 verify-source-ready

Only after the root creates the signed R2 source pass may D1/D2 start:

    python -m scripts.reg_r06_native_hinge_v2 run-diagnostics

Only after root review and the separate final-fit approval may FINAL start:

    python -m scripts.reg_r06_native_hinge_v2 run-final
