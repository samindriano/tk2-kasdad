# R1 source generation — superseded before root review

R1 is preserved as an immutable failed source generation. Do not use it for a
root source pass or official fit.

- Freeze: `experiments/reg_r06_present_native_hinge_v1/PREFIT_FREEZE_R1.json`
- Freeze SHA-256: `1ff835a09f00b6bcad455a4e3a0618000f886d4ee745a605132d7e7392a2ef35`
- Synthetic QA receipt SHA-256: `702e1cc23e490eb157d8f37e6ec79462e456289ad55cb8478756ee0ed05342cb`
- Frozen source commit: `dc2365f1c0e8ba4465dd48a0bfe3505c19ad3a28`

The real non-target preflight reconstructed the expected D1/D2 fit populations
correctly: 4,608 present D1 rows (342 missing excluded) and 11,563 present D2
rows (849 missing excluded). It then rejected the incorrect conditional
full-history present-row count: R1 expected 19,870, while the identity bundle
contains 20,903 rows with 1,375 missing, so the correct count is 19,528.

The failure occurred before any target value was loaded, preflight receipt was
written, root source pass was created, or official LP start was reserved. The
source code selected only identity, cycle, prediction-origin, missing-mask,
and prediction members. R1's two synthetic LP solves are synthetic QA, not
official fit starts. No Test label, E79 route, CSV, or Kaggle access occurred.

R2 uses distinct freeze, preflight, fit-ID roster, verification, and ledger
paths and records the corrected conditional FINAL row count. R1's files remain
unchanged for auditability.
