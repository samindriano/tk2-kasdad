# FULLREG center-local calibration R2

## Scope and boundary

R2 supersedes the population ambiguity for new preparation only. The original
`SOURCE_PREP_BLOCKED_R1.json` remains unchanged and is pinned as historical
evidence. Its status is not rewritten. The resolved OOF universe has 20,903
unique rows. The fit-eligible population is exactly the 19,528 rows with
`missing_mask == false`; the other 1,375 rows are excluded from fitting and
must retain the raw DIRECT float64 value at application.

This is one final local scalar calibration using the fixed FULLREG
`HINGE_FINAL` coefficients. It adds no base model and performs no preprocessing
fit. The old R06 simplified model, global calibration, global control, and
additional candidate search are outside this lane.

The source freeze, target-blind identity preflight, independent source
verification, root-pass contract, verifier, and synthetic CLI test are prepared
before any target access. They do not load the `targets` member, run a fit, read
E79/Test/SampleSubmission, create a CSV, or access Kaggle.

## Fixed calibration

For raw DIRECT value `p`, the frozen base is

`g(p) = a + s_low * min(p, 50) + s_high * max(p - 50, 0)`

with `a = 1.7288079036674198`, `s_low = 0.9933285599956824`, and
`s_high = 0.957778072288735`. The fixed local basis is

`b(p) = max(0, 1 - abs(p - 50) / 10)`.

On present rows with `b(p) > 0`, choose `delta` as the exact lower weighted
median of `(y - g(p)) / b(p)`, with weights `b(p)`, bounded to `[-2, 0]`.
Exact binary weights determine ties. The full bounded minimizer interval is
recorded. Present rows with `b(p) == 0` have no choice-dependent objective and
do not affect `delta`. Missing rows do not enter fitting.

At any separately authorized application, present predictions are
`clip(g(p) + delta*b(p), 0, 100)`. Missing predictions remain raw DIRECT
float64 bits unchanged. Application itself is not authorized by R2's fit pass.

## Frozen population, provenance, and causality

The only OOF source is
`experiments/runs/reg_monotone_hinge_p50_v1/FINAL_APPLICATION_R1/fit_inputs/HINGE_FINAL.npz`,
with its pinned SHA256 in `PREFIT_FREEZE_R2.json`. The non-target preflight
loads only physical rows, record IDs, prediction origins, original bundle
indices, cycles, DIRECT predictions, and the missing mask. The `targets` member
is present in the archive but is not read during preparation.

The pinned final application report, producer freeze/pass/preflight, independent
verification/receipt, and final HINGE fit-state hashes establish the existing
FULLREG HINGE producer chain. Its source commit must be an ancestor of this
checkout. Current HEAD equality is not used as an identity check; R2 binds the
source files and input bytes directly.

Preflight requires 20,903 unique record IDs and physical rows, 20,903 unique
`(prediction_origin, original_bundle_index)` pairs, and strictly earlier
prediction origins than target cycles. Origin 67 contributes 12,412 rows for
cycles 68–75; origin 75 contributes 8,491 rows for cycles 76–79. Of those,
11,563 and 7,965 rows respectively are present. Counts, per-cycle coverage,
identity hashes, and present positive-basis support are recorded in the
preflight receipt.

## Root-pass contract

Only the exact signed pass specified in `ROOT_SOURCE_PASS_CONTRACT_R2.md` opens
the single `LOCAL_CENTER_FINAL` target-access/fit slot. It must bind the exact
R2 freeze, identity preflight, source inventory, and producer report. It
authorizes one target-array read for this fit, zero base/preprocessing fits,
zero LP starts, and no E79/Test/SampleSubmission, CSV, or Kaggle access.
Missing, changed, extra, or malformed authorization fails closed.

The fit runner reserves its one slot in an append-only ledger before accessing
targets. A complete durable result resumes by validating its state, receipt,
and ledger without reading targets again. An ambiguous partial reservation
blocks retry. Synthetic QA exercises the full CLI/reservation/receipt/readback/
resume/tamper path in an isolated temporary fixture; it never touches the real
OOF target member.

## Commands

From the repository root, source preparation uses:

```powershell
python scripts/reg_center_local_application_v1.py --freeze
python scripts/reg_center_local_application_v1.py --preflight
python scripts/reg_center_local_application_v1.py --verify-source
python scripts/reg_center_local_application_v1.py --synthetic-qa
```

The post-review fit command is deliberately separate:

```powershell
python scripts/reg_center_local_application_v1.py --fit
python scripts/reg_center_local_application_v1.py --verify-fit
```

Do not run `--fit` until the root creates and reviews the exact pass. A
separate root decision is required before any E79/Test application, CSV, or
upload.
