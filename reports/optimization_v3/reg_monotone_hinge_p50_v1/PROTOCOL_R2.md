# Fixed-knot monotone hinge calibration — R2 source protocol

## Purpose and authority boundary

R2 prepares a reproducible, executable two-window diagnostic for one calibration
of the already saved raw DIRECT predictions. This is a successor implementation
and identity freeze; R1 remains the immutable no-start predecessor. R3's old
`ROOT_SOURCE_PASS.json` applies only to its own fit IDs. It does not authorize
R2 label access or LP starts.

The R2 source pass must bind the committed runner, independent verifier, tests,
this protocol, the R2 identity freeze, every inherited source/input hash, the
revised user resource policy, the exact diagnostic fit IDs, and the cap of three
possible starts. Its deterministic `signature` is the SHA-256 of canonical JSON
for the approval body with the `signature` field omitted. Until that exact pass
exists and validates, `--run-diagnostics` and `--verify-diagnostics` stop before
Train target loading, reservation, or an LP start.

The current source-ready scope authorizes only preparation of `HINGE_D1` and
`HINGE_D2`. `HINGE_FINAL` is listed as a separate conditional third slot. This
R2 command has no final-fit action. Any final fit, production CSV, Test/Sample
access, Kaggle action, or upload requires a later explicit root-controlled stage
and is outside this protocol.

## Fixed estimator

For raw DIRECT prediction `p`, fit

```text
g(p) = a + sL * min(p, 50) + sH * max(p - 50, 0)
0 <= sL <= 2; 0 <= sH <= 2; a is free
```

The fit minimizes the **unweighted sum of absolute errors**, including OOF rows
where DIRECT is marked missing. It uses one sparse HiGHS linear-program start,
two solver threads, and a fixed 180-second limit. The implementation has no
weights argument, retry, random seed, knot search, or slope grid. Predictions
must be finite and in `[0, 100]`. Clipping applies only when constructing
application predictions. The slopes' upper bound is a fixed design choice.

Before each diagnostic LP, support must include at least 1,000 fit rows and at
least 200 DIRECT values strictly below 50 and 200 at or above 50. The code stops
before reservation if support fails. Score-side support and denominators are
reported. No preprocessing or base-model fits are added.

## Frozen rows and causal windows

R2 reuses the byte-pinned R3 direct bundles and the established physical-row
stitch. The selective NPZ loader reads only direct prediction identity and
missing-mask fields; it does not load cached R3 target arrays. Before labels are
read, the assembled rows, IDs, prediction origins, bundle indices, cycles,
DIRECT values, and missing flags must match the R1/R2 frozen identity hashes.

| Fit ID | Calibration rows | Score rows | Expected fit / score rows |
| --- | --- | --- | ---: |
| `HINGE_D1` | E67 cycles 68–71 | E67 cycles 72–75 | 4,950 / 7,462 |
| `HINGE_D2` | E67 cycles 68–75 | E75 cycles 76–79 | 12,412 / 8,491 |

The two score windows contain 15,953 unique pooled rows. Reuse exactly the
saved `D1_GLOBAL` and `D2_GLOBAL` states as causal present-only comparators.
Also report raw DIRECT and causal GLOBAL on all rows for context. Do not use
full-support GLOBAL coefficients as validation. D2 changes prediction origin
from E67 to E75 and remains an acknowledged limitation. All Train outcomes
have been reused in prior work; these windows are adaptive diagnostics, not a
fresh holdout.

After a valid R2 source pass, the selected Train targets are parsed only for the
frozen 20,903 OOF physical rows and exact IDs/cycles. The audit must report zero
fit-target rows parsed and match the frozen Train file hash. The independent
verifier repeats this same selected-row reconstruction and compares each fit
and score artifact row-by-row, including targets, to the frozen identity and
selected Train values.

## Ledger, recovery, and resource rules

The persistent R2 ledger allows at most three starts across `HINGE_D1`,
`HINGE_D2`, and the conditional `HINGE_FINAL`, in that order. The current
diagnostic source pass can authorize only the first two IDs. Each start records
`RESERVE` and `LP_STARTED` durably before invoking the solver; then the fit
state is atomically published and a terminal event appended. A matching valid
state published before a crash is recovered without calling the solver again.
An `LP_STARTED` without a valid published state is ambiguous and permanently
blocks that fit ID. Failures consume their start. No ledger reset, retry, or
replacement fit is permitted.

The shared lock is `%TEMP%/kasdad_single_heavy_job.lock`, with one heavy job and
two threads. The bound input is
`reports/optimization_v3/RESOURCE_POLICY_USER_REVISED_20261105.json` for the
small calibration LP:

- Initial available memory is at least 2 GiB.
- The incremental estimate is 512 MiB; projected free memory must remain at
  least 1 GiB.
- Actual available memory at later stage boundaries must be at least 1 GiB.
- Capture process RSS and OS peak working set when supported. Mark each metric
  as measured or unavailable; the 512 MiB estimate is not a measured peak.

The independent verifier checks the report's resource receipt path and hash,
freeze/source-pass/policy bindings, every threshold, and the resource records
on both ledger pre-fit events and the published post-solver state. It also
checks the RSS and OS peak measurement flags for internally consistent values.

## Reporting and decision gate

For each window, report coefficient values and bounds, raw DIRECT, causal GLOBAL
all-row and present-only metrics, hinge present-only metrics, exact denominators,
MAE, mean error, p90 absolute error, per-cycle metrics, supported missing/
present and target `<=5`/`>=95` subgroups, subgroup harm, and application-side
clipping count. A subgroup under 200 rows is marked `UNKNOWN_UNSUPPORTED`.
Missing application rows must preserve raw DIRECT float64 bits exactly.

The independent verifier recomputes solver objective and constraints from saved
fit arrays, score formulas from saved coefficients and frozen inputs, per-window
and pooled metrics, subgroup harms, the decision gate, and the 15,953-row unique
pool. It verifies both the input bundles and output row bundles by hash. It
does not accept producer-supplied subgroup-harm values without recomputation.

- `QUALIFIED`: pooled gain versus causal GLOBAL present-only is at least 0.02,
  each window gain is nonnegative, and maximum supported subgroup harm is at
  most 0.25.
- `EXPLORATORY_ROOT_REVIEW`: if not qualified, pooled gain is nonnegative, or a
  window gain reaches 0.02 while pooled harm is at most 0.01; every window harm
  is at most 0.03; and supported subgroup harm is at most 0.25.
- Otherwise `RETIRE` this exact recipe. A failed gate does not trigger a nearby
  knot, slope grid, or other automatic rescue.

No R2 command creates an official candidate CSV. The future CSV contract, if a
separate stage is authorized, is exactly `record_id,threat_rank`, 29,421 ordered
IDs, finite `[0,100]` values, and exactly 1,062 bit-exact raw DIRECT fallbacks.
The verifier rejects `prediction` and `artifact_signal` headers.

## Reproduction commands and expected boundaries

From the repository root:

```powershell
python -m unittest tests.test_reg_monotone_hinge_p50_v2 -v
python scripts/reg_monotone_hinge_p50_v2.py --synthetic-qa
python scripts/reg_monotone_hinge_p50_v2.py --freeze-identity
```

The tests and QA use synthetic arrays/files only. The identity freeze reads and
hashes pinned metadata and artifacts by bytes; it does not parse Train labels,
read Test/Sample, create a CSV, or start an official LP. Only after a root review
adds a matching `ROOT_SOURCE_PASS_R2.json` may this source revision run:

```powershell
python scripts/reg_monotone_hinge_p50_v2.py --run-diagnostics
python scripts/reg_monotone_hinge_p50_v2.py --verify-diagnostics
```

Each diagnostic LP has a 180-second solver limit. Actual official runtime and
compute cost remain unknown until an authorized source pass and successful
resource/support gates permit those starts.
