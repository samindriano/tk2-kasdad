# Fixed-knot monotone hinge calibration — R4 source and identity preflight

## R4 scope and predecessor boundary

R4 is an additive, versioned successor to the sealed R3 source. Preserve all
R3 source, freeze, pass, preflight, resource-blocker, and failed-attempt
artifacts unchanged. R3's prestart resource blocker records no diagnostic
command, labels, or LP start. Its later attempt failed with `KeyError('physical_rows')`
at the pretarget identity check, before the selected Train-row reader, labels,
ledger entry, or LP start. That failure is not retried or edited in place.

R4 binds the exact R3 artifact bytes: identity freeze
`9ddf50d2796b9b19e4e4ca61767dea80cbc81161daa50a8536f3f5356aceedb5`, root
source pass `340043c85cec86413f0fe9530f5017ca40a1b0af2008aa22b003bb77d62a9c8e`,
identity preflight
`0343e5e91aa934f9bd0270b1ebd110b7b2148a850179d0f8d1b7f01f2e9e8c0f`,
prestart resource blocker
`f26f827555688d9cd0d9fb220d78459d13cb181261e31bd23c6e3d6a7f6dd94b`, and
pretarget failure
`2308b66db3a6bee5a73af40acbe23fbd55840d8d3051177a11d41e257f000eb2`.
The R3 pass is not authorization for R4. R4's own root source pass remains a
separate review gate.

## Identity schema repair

The direct identity loader returns the compact fields `rows`, `ids`, `origins`,
`indices`, `cycles`, `direct`, `missing`, and recomputed `row_counts`. R4's
adapter rejects missing or unknown keys, non-vector or misaligned arrays,
duplicate physical rows or record IDs, and counts that differ from the origin
and cycle arrays. It maps this schema explicitly to the frozen verifier fields
`physical_rows`, `record_ids`, `prediction_origins`,
`original_bundle_indices`, `cycles`, `direct_predictions`, `missing_mask`, and
`row_counts`. The verifier recomputes all seven frozen array hashes and the
three row counts; optional returned hash metadata must match those recomputed
values.

`verify_oof_identity` is the single identity validator at both boundaries:
identity-only preflight calls it after loading direct identities, and
`load_targets_after_source_pass` calls it before the selected Train-row reader.
The preflight receipt records that exact validator passed and that it recomputed
row counts. Helper API calls retain their producer signatures; in particular,
`read_scored_query` receives `v2.SOURCE` as its source root.

## Synthetic end-to-end acceptance gate

Before requesting an R4 source review, run the actual top-level
`run_diagnostics()` against synthetic arrays and then run its actual independent
verifier. Mock only external source/data inputs, the shared lock, and resource
probes. Keep the orchestration, identity validator and target boundary,
optimizer, state writes, ledger, report persistence, and independent replay
active. The synthetic target reader asserts exact ordered physical rows,
record IDs, and cycles for each call. Acceptance requires both D1 and D2 state
files, exactly two ledgered LP starts, both diagnostic receipts, successful
independent verification, and zero official data or labels read. Negative tests
cover reordered rows, changed IDs, missing compact keys, the incompatible
long-key-only schema, and inconsistent row counts.

The synthetic end-to-end test does not create an R4 root pass or authorize any
official diagnostic. Tests and synthetic QA must use synthetic data only.

## Identity-only integration preflight

After the R4 source commit and identity freeze, and after coordinating use of
the shared heavy-job slot with Luna1, run:

```powershell
python scripts/reg_monotone_hinge_p50_v4.py --identity-preflight
```

This stage verifies direct E67/E75 row identity and saved D1/D2 GLOBAL state
files under the existing shared-lock and resource policy. It reads only the
allowlisted direct metadata and prediction/ID arrays, plus frozen GLOBAL
coefficient states. A target-like extra NPZ member blocks the preflight. It
does not load target arrays, parse Train labels, call the target reader or
solver, create a ledger, score, or access Test/Sample. It does not create
`ROOT_SOURCE_PASS_R4.json`.

The immutable output is
`reports/optimization_v3/reg_monotone_hinge_p50_v1/IDENTITY_PREFLIGHT_R4.json`.
It binds source and input hashes, direct receipt/archive evidence, allowlisted
members, row fingerprints and recomputed counts, GLOBAL state hashes, resource
snapshots, and explicit zero-target/zero-score/zero-LP boundaries. A fresh
separate root review must issue an R4 source pass before any official
diagnostic command can run.

## Fixed estimator and diagnostic windows

The scientific recipe is inherited unchanged:

```text
g(p) = a + sL * min(p, 50) + sH * max(p - 50, 0)
0 <= sL <= 2; 0 <= sH <= 2; a is free
```

Fit unweighted absolute error with one sparse HiGHS LP start per diagnostic,
two threads, and a fixed 180-second limit. There is no retry, weight argument,
random seed, knot search, slope grid, preprocessing fit, or new base fit.
Clipping occurs only for application predictions; missing outputs preserve raw
DIRECT float64 bits. Fit support remains at least 1,000 rows and 200 values on
each side of 50.

| Fit ID | Calibration rows | Score rows | Expected fit / score rows |
| --- | --- | --- | ---: |
| `HINGE_D1` | E67 cycles 68–71 | E67 cycles 72–75 | 4,950 / 7,462 |
| `HINGE_D2` | E67 cycles 68–75 | E75 cycles 76–79 | 12,412 / 8,491 |

The two score windows contain exactly 15,953 unique pooled rows. Reuse saved
`D1_GLOBAL` and `D2_GLOBAL` states as causal present-only comparators. These
Train outcomes have been reused in prior work and are adaptive diagnostics, not
a fresh holdout. Qualification and exploratory gates, subgroup limits, score
reporting, and conditional final-fit accounting remain unchanged from R3.

## Source preparation and current authorization

From the repository root:

```powershell
python -m unittest tests.test_reg_monotone_hinge_p50_v4 -v
python scripts/reg_monotone_hinge_p50_v4.py --synthetic-qa
python scripts/reg_monotone_hinge_p50_v4.py --freeze-identity
python scripts/reg_monotone_hinge_p50_v4.py --identity-preflight
```

Freeze and preflight bind committed R4 source, exact R3 predecessor artifacts,
and the inherited direct inputs. The real identity preflight is an identity
check only. Do not run `--run-diagnostics` or any LP on official data until a
separate, reviewed `ROOT_SOURCE_PASS_R4.json` is issued. `HINGE_FINAL`, CSV
creation, Test/Sample access, and Kaggle remain outside this authorization.
