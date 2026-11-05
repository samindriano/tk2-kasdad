# Fixed-knot monotone hinge calibration — R3 source and identity preflight

## R2 predecessor and authority boundary

R3 is an additive, versioned successor to the sealed R2 source. R2's freeze,
source pass, code, and failed zero-start execution remain immutable. That attempt
stopped at the direct receipt API boundary before NPZ arrays, Train targets,
reservation, or LP start. R2's source pass does not authorize R3 execution.

R3 fixes the direct receipt verification call to the producer's actual signature:
`verify_signed(value, field, label)`. Before R3 may receive a root source pass,
it runs a **real identity-only integration preflight** against the pinned E67 and
E75 direct prediction bundles, the frozen direct row manifests, and saved D1/D2
GLOBAL state files. This is an identity and input integration check; it does not
fit, score, evaluate labels, or create a ledger.

The R3 root source pass must bind the committed R3 source hashes, R3 identity
freeze SHA-256, complete input hash map, the successful identity-preflight
SHA-256 and status, exactly `HINGE_D1`/`HINGE_D2`, and the existing cap of three
possible starts. The deterministic `signature` is SHA-256 of canonical JSON for
the approval body with `signature` omitted. The R2 source pass is explicitly
insufficient.

## Identity-only integration preflight

Run after the R3 source commit and R3 identity freeze, before creating the R3
root source pass:

```powershell
python scripts/reg_monotone_hinge_p50_v3.py --identity-preflight
```

The preflight uses the shared OS lock `%TEMP%/kasdad_single_heavy_job.lock`, one
heavy job, two threads, a fresh 2 GiB initial available-memory floor, a 512 MiB
incremental estimate with at least 1 GiB projected free memory, and at least 1
GiB actual free memory at each later boundary. It records RSS and OS peak working
set accurately where available. A failed memory or lock gate stops before
loading the prediction arrays.

Only these direct NPZ members may be loaded, for E67 and E75:

```text
origin, arm, fit_rows, query_rows, query_record_ids, query_cycles,
baseline, raw_auxiliary, mapped_auxiliary, prediction, missing_mask
```

Non-allowlisted members are recorded by name only and never loaded; a
target-like extra member name blocks the preflight. It validates the real signed
direct receipt with the three-argument producer helper and compares all five producer
array digests: `query_cycles_sha256`, `query_ids_sha256`,
`raw_auxiliary_sha256`, `mapped_auxiliary_sha256`, and `prediction_sha256`.
`sha_pilot_array(value, *, dtype=None)` hashes canonical shape and dtype plus
contiguous C-order numeric bytes; object/string arrays use canonical JSON of
their string values. `verify_prefixes(e67, e75, direct_freeze)` validates exact
temporal prefix and baseline identity; `stitch_oof_rows(e67, e75)` validates
physical-row uniqueness, E67/E75 late-cycle correspondence, IDs, and cycle
scope. The assembled 20,903-row identity must reproduce the frozen row, ID,
origin, original bundle index, cycle, direct prediction, and missing-mask hashes
and all stage counts/support.

The preflight also hashes and parses the saved `D1_GLOBAL` and `D2_GLOBAL` JSON
states by their frozen input identities. It reports their coefficient values
and state hashes. It does not access Train values, any cached R3 diagnostic-row
NPZ targets, Test/Sample, target-derived metrics, or labels. It does not call the
target loader or solver, reserve an LP, create a ledger, or score. Train's raw
file hash remains bound by the inherited freeze; identity preflight does not
parse it. Only the explicit R3 preflight input allowlist is rehashed during this
stage; Train and target-bearing cached row bundles are excluded.

The immutable result is
`reports/optimization_v3/reg_monotone_hinge_p50_v1/IDENTITY_PREFLIGHT_R3.json`.
It records direct receipt/archive hashes, the exact allowlisted loaded members,
producer array-hash checks, row fingerprints/counts, GLOBAL state hashes,
resource snapshots, and explicit zero-target/zero-score/zero-LP boundaries.
Root review of this report must happen before a new R3 source pass is issued.

## Fixed estimator and diagnostic windows

The scientific recipe is unchanged:

```text
g(p) = a + sL * min(p, 50) + sH * max(p - 50, 0)
0 <= sL <= 2; 0 <= sH <= 2; a is free
```

Fit the unweighted sum of absolute errors with one sparse HiGHS LP start, two
threads, and a fixed 180-second limit. There is no retry, weight argument,
random seed, knot search, slope grid, preprocessing fit, or new base fit.
Clipping occurs only for application predictions; missing outputs preserve raw
DIRECT float64 bits. Fit support is at least 1,000 rows and at least 200 values
strictly below 50 and 200 at or above 50.

| Fit ID | Calibration rows | Score rows | Expected fit / score rows |
| --- | --- | --- | ---: |
| `HINGE_D1` | E67 cycles 68–71 | E67 cycles 72–75 | 4,950 / 7,462 |
| `HINGE_D2` | E67 cycles 68–75 | E75 cycles 76–79 | 12,412 / 8,491 |

Pool the two windows as exactly 15,953 unique rows. Reuse the saved `D1_GLOBAL`
and `D2_GLOBAL` states as causal present-only comparators. These Train outcomes
have been reused in prior work and are adaptive diagnostics, not a fresh
holdout. The qualification/exploratory gates, subgroup limits, score reporting,
and conditional final-fit accounting remain exactly as R2.

## Source preparation and permitted execution

From the repository root:

```powershell
python -m unittest tests.test_reg_monotone_hinge_p50_v3 -v
python scripts/reg_monotone_hinge_p50_v3.py --synthetic-qa
python scripts/reg_monotone_hinge_p50_v3.py --freeze-identity
python scripts/reg_monotone_hinge_p50_v3.py --identity-preflight
```

Tests and synthetic QA use synthetic data only. The freeze binds committed
source and predecessor/input hashes but does not parse labels. The real
identity-preflight command reads only the allowlisted direct metadata and
prediction/ID arrays and saved GLOBAL coefficients described above. Actual
diagnostic execution remains gated on a separately reviewed R3 source pass:

```powershell
python scripts/reg_monotone_hinge_p50_v3.py --run-diagnostics
python scripts/reg_monotone_hinge_p50_v3.py --verify-diagnostics
```

Only `HINGE_D1` and `HINGE_D2` are in scope for that pass. `HINGE_FINAL`, CSV
creation, Test/Sample access, and Kaggle are not authorized by this protocol.
