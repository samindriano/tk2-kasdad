# Sol handoff: R2C D1/D2 result and independent replay

## Execution record

The signed R2C source pass authorized only `R06_HINGE_D1` and
`R06_HINGE_D2`. Exactly two official LP starts occurred, one per slot; both
completed successfully. No FINAL slot, submission, Test-label access, CSV, or
Kaggle action occurred. No solver retry, parameter search, or base/preprocessing
fit was run.

| Artifact | SHA-256 |
|---|---|
| Source pass | `044484845d08fe672c377a03079498a8e9646227d0adae3ebeea0981b2b4cf91` |
| Diagnostics report | `2e50ec60ea275dfd16ed3fd98dc9f05cc15c44eabbcf9edef31bc8e27940b539` |
| Independent verification receipt | `9707702995b9e3bb3ad651122841151488f4880c31a6c859876ef41bdf4c82f6` |
| Independent audit script | `d8bb3cd7af1797902ecc2c7c31e58c4979a10e7c793f7965b811c00a88b28fe1` |
| Append-only LP ledger | `fada741a13e459cb91f0230d041c58cacae300c944fd58300d9918e66a1a30c1` |

The independent audit replayed frozen source/input hashes and authorization,
R06 row IDs/cycles/origins/physical positions, missing-row bit identity, each
fit-input array and hash, solver objective/slacks and constraints, resource
snapshots, all full/present/group/cycle metrics, gates, and the exact six-event
ledger. It passed with status `PASS_INDEPENDENT_DIAGNOSTICS_REPLAY` and
confirmed two starts, two completions, and no FINAL start. This PASS is an
integrity/replay result; the scientific gates below both fail.

## Fit states

| Slot | Fit rows | Intercept | Slope low | Slope high | Objective | State SHA-256 |
|---|---:|---:|---:|---:|---:|---|
| D1 | 4,950 | 2.0135333074 | 0.9902154011 | 0.9496235769 | 56,087.3517723 | `abd8af36222a26fd91b70f41caf819eb1d0d15ed72fdcbd626427690550a4f82` |
| D2 | 12,412 | 2.1080438857 | 0.9905686272 | 0.9595404950 | 141,181.7724022 | `49310b6318eead2fed80205e8a326ed31d483fba5940d4d4c6a49f3301cdbd5f` |

Both HiGHS solves returned status 0/optimal, knot 50, slopes in [0,2], and
thread count 2. The replayed sum-absolute objectives matched solver metadata
and saved slacks; maximum primal residual-minus-slack was below `2.3e-14`.

## Diagnostic results

Primary is the same-cutoff R4 hinge applied to same-history R06. Raw R06 and the
matching same-cutoff R4 hinge on DIRECT are diagnostic comparisons.

| Window | Rows | Candidate MAE | Primary MAE | Gain vs primary | Gain vs raw R06 | Candidate p90 abs error | Candidate mean error |
|---|---:|---:|---:|---:|---:|---:|---:|
| D1 | 7,462 | 11.402320 | 11.399390 | -0.002929 | +0.075128 | 24.633640 | -0.659473 |
| D2 | 8,491 | 9.876721 | 9.878612 | +0.001891 | +0.044892 | 22.843228 | +0.635972 |
| Pooled | 15,953 | 10.590318 | 10.589955 | -0.000364 | +0.059035 | 23.679499 | +0.030029 |

Pooled candidate MAE is 10.625103 against the matching DIRECT comparator
(gain `+0.034784`). Present-only MAE is 10.386318 versus 10.385929 for the
primary. There are 14,920 present and 1,033 missing scored rows; all missing
rows retain raw R06 predictions exactly. All required groups and cycles meet
support (`group n >= 200`, `cycle n >= 50`).

The maximum supported group harm is `0.265530` MAE in D1 for `target <= 5`,
above the `0.25` cap. D1 has a small window harm of `0.002929`, within the
exploratory `0.02` window limit; D2 gains `0.001891`. The pooled gain versus
primary is `-0.000364`, below zero. Therefore qualification=false,
exploratory=false, and root_review_eligible=false. No final fit is authorized
or started.

## Artifact paths

- Report: `reports/optimization_v3/reg_r06_native_hinge_v1/DIAGNOSTICS_R2C.json`
- Replay receipt: `reports/optimization_v3/reg_r06_native_hinge_v1/DIAGNOSTICS_VERIFICATION_R2C.json`
- Independent read-only audit: `reports/optimization_v3/reg_r06_native_hinge_v1/INDEPENDENT_DIAGNOSTIC_AUDIT_R2C.py`
- Ledger and fit states: `experiments/runs/reg_r06_native_hinge_v1/R2C/`

No further fit, finalization, application, CSV, or upload is implied by these
diagnostics. The next decision is root review of this independently replayed
negative gate result under Sam's steering.
