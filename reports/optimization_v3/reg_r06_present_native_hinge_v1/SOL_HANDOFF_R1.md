# Sol handoff — present-only R06 native hinge source review

## Decision requested

Review this versioned source-only package for one present-only fit-population
ablation. If source and input lineage pass, create a signed
`experiments/reg_r06_present_native_hinge_v1/ROOT_SOURCE_PASS_R1.json` that
binds the exact freeze, non-target preflight, source hashes, input hashes, and
source commit. Authorize only
`R06_PRESENT_HINGE_D1` and `R06_PRESENT_HINGE_D2`; keep
`final_fit_authorized=false`. No official new fit has started.

## Why this one variable

The prior all-row R2C native hinge completed two valid diagnostic fits but
failed qualification and exploratory review: pooled gain against transported
R4 was -0.000364 MAE, D1 -0.002929, D2 +0.001891, and supported low-target D1
harm 0.265530 exceeded 0.25. Root independently replayed that result. It is a
negative mechanism result; no FINAL was authorized.

The present-only fit-population ablation tests one plausible but unproven
explanation: fitting only present rows may change the hinge calibration.
Missing rows may instead have regularized the all-row fit. Historical labels
and outcomes have been reused, and this follow-up was chosen after the negative
result. Do not claim causality or a new holdout result.

Four prior calibration/decision summaries were checked, with no exact duplicate
of this three-parameter present-only native hinge fit found in those four.
The related center-local present-only correction used a different
one-parameter correction. Scope and hashes of the four documents are recorded
in `PROTOCOL_R1.md`; this is not an exhaustive repository-wide novelty claim.

## Frozen comparison design

- Fit the existing p50 hinge with the unchanged sum-absolute-error HiGHS
  solver and slope bounds [0, 2].
- D1 fits E67 cycles 68–71 on present rows only (expected 4,608 rows; exclude
  342 missing) and scores the same all-row E67 cycles 72–75 IDs (7,462 rows).
- D2 fits E67 cycles 68–75 on present rows only (expected 11,563 rows; exclude
  849 missing) and scores the same all-row E75 cycles 76–79 IDs (8,491 rows).
- Preserve exact row-level present-fit IDs and missing exclusions in
  `FIT_POPULATION_IDS_R1.json`; preflight is target-free. Missing score
  predictions remain bit-exact raw R06.
- Mechanism control: saved all-row native R2C hinge on the exact score IDs.
- Primary comparison: same-cutoff R4 hinge transported onto R06 on those IDs.
  Matching R4-on-DIRECT and raw R06 stay separate diagnostics.
- Mechanism gain must be strictly positive in both windows. Qualification and
  exploratory gates and support rules are frozen in `PROTOCOL_R1.md`.
- Full-OOF FINAL coefficients do not enter temporal comparisons.

## Budget and boundaries

This is a new isolated ledger with two diagnostic slots and one conditional
present-only FINAL slot. The two prior native R2C LP starts remain recorded as
consumed; neither existing ledger is reset or rewritten. The old all-row
FINAL slot is not reused. A separate root approval bound to the exact
diagnostics is required before any new FINAL fit.

The source-ready runner and independent verifier must establish exact source
and input hashes, R4 and R2C comparator lineage, present-fit/excluded-missing
IDs and counts, synthetic actual-CLI fit/receipt/metric E2E, resource preflight,
and zero official starts before this review. No target value, Test label,
E79 route, CSV, or Kaggle access is allowed at this stage.

Review artifacts:

- `experiments/reg_r06_present_native_hinge_v1/PREFIT_FREEZE_R1.json`
- `reports/optimization_v3/reg_r06_present_native_hinge_v1/IDENTITY_PREFLIGHT_R1.json`
- `reports/optimization_v3/reg_r06_present_native_hinge_v1/FIT_POPULATION_IDS_R1.json`
- `reports/optimization_v3/reg_r06_present_native_hinge_v1/SYNTHETIC_QA_R1.json`
- `reports/optimization_v3/reg_r06_present_native_hinge_v1/SOURCE_VERIFICATION_R1.json`
- `reports/optimization_v3/reg_r06_present_native_hinge_v1/PROTOCOL_R1.md`

Only after the independent source-ready verifier passes should root review
consider a source pass. This handoff requests no FINAL authorization.
