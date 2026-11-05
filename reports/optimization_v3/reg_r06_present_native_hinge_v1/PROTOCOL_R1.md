# R06 present-only native hinge population ablation — R1 protocol

## Question and scope

This is one prospective ablation of the calibration fit population after the
all-row native R2C calibration failed its frozen scientific gates. It tests
whether fitting the same fixed-knot hinge on present rows changes the result.
It does not establish that missing rows caused the previous result. Missing-row
training could have regularized the all-row fit and may have helped it. The
choice follows reused development results, so this is adaptive research, not a
fresh holdout or an outcome-blind discovery.

Keep the existing R06 base predictions, cached OOF labels, R4 calibration,
hinge equation, knot, slope bounds, unweighted L1 objective, and application
policy fixed. Do not add base predictors, preprocessing, row weights, knots,
searches, or post-fit gate changes.

## Bounded duplicate check

Four prior calibration/decision summaries were checked; no exact prior
three-parameter present-only native R06 hinge fit was found in those documents:

| Prior summary | SHA-256 | Evidence considered |
|---|---|---|
| `reports/optimization_v3/reg_r06_native_hinge_v1/SOL_HANDOFF_R2C.md` | `e078e622a7524bf44868ca47645c55f4f44f842074d59df03c830dfd66c0bc15` | R06 cache/history and same-cutoff comparator contract, lines 22–48 |
| `reports/optimization_v3/reg_r06_native_hinge_v1/SOL_HANDOFF_R1.md` | `78d0e2bbef1e48813d8bf9c7e3fe23cf5f183351ebe3446d379a1977ccf01ed2` | Earlier all-row R06 D1/D2 windows and fit budget, lines 19–25 |
| `reports/optimization_v3/reg_monotone_hinge_p50_v1/SOL_DIAGNOSTIC_DECISION_R4.md` | `538007fd2449b472e1fad657c916760710b1c6dd148287f051959162318ff7ba` | R4 temporal diagnostics and reported trade-offs, lines 5–15 |
| `reports/optimization_v3/reg_center_local_correction_v1/SOL_HANDOFF.md` | `d33b01ddf4d556cc0be9ae9b647950c5dc2571669f643893d64273a8b829a82a` | Present-only one-parameter local residual correction/control, lines 9–23 |

The present-only local residual calibrator is a related counterexample but has
a different one-parameter correction form. The exact-duplicate verdict is
limited to these four summaries; it does not claim exhaustive repository-wide
novelty.

## Frozen populations and prediction controls

Use the same R06 cache and IDs as the all-row R2C diagnostic:

| Slot | Fit cycles | Fit rows | Missing rows excluded | Scored cycles | Score rows |
|---|---|---:|---:|---|---:|
| `R06_PRESENT_HINGE_D1` | E67 / 68–71 | 4,608 present | 342 | E67 / 72–75 | 7,462 |
| `R06_PRESENT_HINGE_D2` | E67 / 68–75 | 11,563 present | 849 | E75 / 76–79 | 8,491 |

The source-stage preflight must recompute these counts from cached non-target
identity. It writes a signed `FIT_POPULATION_IDS_R1.json` containing every
present fit row's record ID, physical row, cycle, and prediction origin, plus
the explicitly excluded missing-row IDs. Score-window record-ID hashes bind
the same evaluation rows. An independent verifier reconstructs those rows
from the caches and compares them before any source-pass review.

Fit the unchanged form

`intercept + slope_low * min(p, 50) + slope_high * max(p - 50, 0)`

by the existing R4 HiGHS sum-absolute-error solver, with both slopes in [0, 2],
2 threads and the existing time limit. Only present rows enter D1, D2, or the
conditional full-history fit. At application, clip predictions for present
score rows to [0, 100]. Missing score rows keep same-row raw R06 values bit for
bit.

Compare every score ID against both controls:

1. **Mechanism control:** saved all-row native R2C D1/D2 hinge coefficients
   applied to the same R06 score rows.
2. **Primary competition comparator:** saved same-cutoff R4 D1/D2 hinge
   coefficients transported onto the same-history R06 score rows.

Report matching R4-on-DIRECT and raw R06 separately as diagnostics. They do not
replace either gate comparator. Never use full-OOF FINAL coefficients in
temporal comparisons.

Report full-row and present-only MAE, p90 absolute error, and mean error; fixed
availability and target-tail groups; and every scored cycle. Group support is
at least 200 rows; cycle support is at least 50. Unsupported groups/cycles are
UNKNOWN and block scientific PASS. Boundaries below are inclusive except the
mechanism requirement, which is strictly positive.

## Frozen gates

- **Mechanism:** present-only candidate gain over the all-row native R2C
  control must be strictly greater than zero in both D1 and D2.
- **Qualification:** pooled gain over transported R4 is at least 0.005 MAE;
  both window gains are nonnegative; maximum supported group harm against the
  transported R4 primary is at most 0.25 MAE.
- **Exploratory review:** pooled gain over transported R4 is nonnegative; each
  window harm is at most 0.02 MAE; maximum supported group harm is at most
  0.25 MAE.
- Root-review eligibility requires the mechanism gate and either scientific
  gate. UNKNOWN blocks qualification and exploratory review.

These rules are frozen in the source before official fits. Do not change them
after results are visible.

## Source, target, and budget boundaries

`freeze`, `preflight`, `synthetic-qa`, and `verify-source-ready` must finish
before a root source pass. Preflight selects identity, origin, cycle, missing,
and prediction members from the cached OOF bundle; the presence of its target
member is inventoried without loading target values. The CLI synthetic E2E
uses only generated synthetic values. The independent verifier checks source
hashes, frozen input hashes, R4 and R2C lineage, exact present/excluded fit
IDs, and independently recomputes synthetic score metrics.

The new ledger is isolated under
`experiments/runs/reg_r06_present_native_hinge_v1/R1/`. It has a total cap of
three new starts: D1, D2, and one conditional present-only FINAL. Record the
two completed native R2C diagnostic LPs as already consumed. Preserve both old
ledgers and all old fit states; do not reset or repurpose the unused old
all-row FINAL slot.

The root source pass may authorize only D1 and D2 and must explicitly keep
`final_fit_authorized=false`. A separate immutable root approval bound to the
diagnostics is required before the conditional FINAL fit. It is calibration,
not holdout validation. CSV construction or a benchmark/upload decision is a
separate later root gate and is outside this source package.

No Test labels, E79 route, CSV, or Kaggle access occurs during preparation or
the diagnostic/conditional-fit source package. No base or preprocessing fit
is permitted. The old R4 ledger, old all-row R2C ledger, existing experiment
directories, and unrelated worktree files remain immutable.

## Commands and expected source-ready state

```powershell
python -m scripts.reg_r06_present_native_hinge_v1 freeze
python -m scripts.reg_r06_present_native_hinge_v1 preflight
python -m scripts.reg_r06_present_native_hinge_v1 synthetic-qa
python -m scripts.reg_r06_present_native_hinge_v1 verify-source-ready
```

The independent source verifier must return
`PASS_SOURCE_READY_PENDING_ROOT_RELEASE_R1`, with zero official LP starts and
no target values loaded. Only after root review may the D1/D2 runner be
authorized. The present-only FINAL remains closed until its own root approval.
