# FULLREG center-local calibration R3

## Scope and boundary

R3 repairs the source verifier's immutable-receipt readback path and supersedes
R2 source preparation. R2's freeze, preflight, and verification remain byte-
preserved and are pinned as historical inputs. The original
`SOURCE_PREP_BLOCKED_R1.json` also remains unchanged. The resolved OOF universe
has 20,903 unique rows. The fit-eligible population is exactly the 19,528 rows
with `missing_mask == false`; the other 1,375 rows are excluded from fitting
and must retain raw DIRECT float64 values at application.

This is one final local scalar calibration using fixed FULLREG `HINGE_FINAL`
coefficients. It adds no base model and performs no preprocessing fit. The old
R06 simplified model, global calibration, global control, and additional
candidate search are outside this lane.

The R3 source freeze, target-blind identity preflight, independent source
verification, root-pass contract, verifier, and synthetic CLI test do not load
the `targets` member, run a real fit, read E79/Test/SampleSubmission, create a
CSV, or access Kaggle.

## Fixed calibration

For raw DIRECT value `p`, the frozen base is

`g(p) = a + s_low * min(p, 50) + s_high * max(p - 50, 0)`

with `a = 1.7288079036674198`, `s_low = 0.9933285599956824`, and
`s_high = 0.957778072288735`. The local basis is

`b(p) = max(0, 1 - abs(p - 50) / 10)`.

On present rows with `b(p) > 0`, choose `delta` as the exact lower weighted
median of `(y - g(p)) / b(p)`, with weights `b(p)`, bounded to `[-2, 0]`.
Exact binary weights determine ties. Record the full bounded minimizer
interval. Present rows with `b(p) == 0` have no choice-dependent objective and
do not affect `delta`. Missing rows do not enter fitting.

At any separately authorized application, present predictions are
`clip(g(p) + delta*b(p), 0, 100)`. Missing predictions remain raw DIRECT
float64 bits unchanged. R3's fit pass does not authorize application.

## Frozen population and causality

The only OOF source is
`experiments/runs/reg_monotone_hinge_p50_v1/FINAL_APPLICATION_R1/fit_inputs/HINGE_FINAL.npz`,
bound by its SHA256 in `PREFIT_FREEZE_R3.json`. The non-target preflight loads
only physical rows, record IDs, prediction origins, original bundle indices,
cycles, DIRECT predictions, and the missing mask. The `targets` member exists
but is not read during preparation.

The final application report, producer freeze/pass/preflight, independent
verification/receipt, and final HINGE fit-state hashes establish producer
ancestry. The producer source commit must be an ancestor of this checkout.
Current HEAD equality is not used; R3 binds source and input bytes directly.
R3 also pins the superseded R2 receipts so its history remains reviewable.

Preflight requires 20,903 unique record IDs and physical rows, 20,903 unique
`(prediction_origin, original_bundle_index)` pairs, and strictly earlier
prediction origins than target cycles. Origin 67 contributes 12,412 rows for
cycles 68–75; origin 75 contributes 8,491 rows for cycles 76–79. Of those,
11,563 and 7,965 rows respectively are present. R2's observed support count
was 4,125 present rows with `b > 0`; R3 must independently reproduce it.

## Root-pass boundary

Only the exact signed pass in `ROOT_SOURCE_PASS_CONTRACT_R3.md` opens the single
`LOCAL_CENTER_FINAL` target-access/fit slot. It binds the R3 freeze, identity
preflight, source inventory, and producer report. It authorizes one
closed-form scalar fit, zero base/preprocessing fits, zero LP starts, and no
E79/Test/SampleSubmission, CSV, or Kaggle access. Missing, changed, extra, or
malformed authorization fails closed.

The runner reserves its one slot in an append-only ledger before target access.
A completed result resumes by validating its state, receipt, and ledger without
reading targets again. An ambiguous partial reservation blocks retry. Synthetic
QA exercises CLI freeze/preflight/source verification, one synthetic fit,
durable receipt/readback/resume, and state/receipt tamper rejection in a
temporary fixture. It never touches the real target member.

## Commands

From the repository root, source preparation uses:

```powershell
python scripts/reg_center_local_application_v1.py --freeze
python scripts/reg_center_local_application_v1.py --preflight
python scripts/reg_center_local_application_v1.py --verify-source
python scripts/reg_center_local_application_v1.py --synthetic-qa
```

The fit commands remain gated:

```powershell
python scripts/reg_center_local_application_v1.py --fit
python scripts/reg_center_local_application_v1.py --verify-fit
```

Do not run `--fit` before root creates and reviews the exact pass. Any later
E79/Test application, CSV, or upload requires a separate root decision.
