# E79 gamma artifact recovery audit v1

## Scope and immutable history

This is a separate, zero-optimizer numerical audit of the gamma vector saved by
the one authorized `E79_GLOBAL_STRENGTH` full-Train attempt. The original
attempt remains `FAIL_E79_GAMMA_FIT_NO_RETRY` with `KeyError('checks')`, and its
consumed fit count remains one. This audit cannot turn that historical fit
receipt into a PASS and does not recover optimizer metadata such as `nit`,
`nfev`, message, loss, or gradient; those remain unknown because they were not
persisted.

The audit only considers the exact existing gamma artifact and the exact frozen
E79 Train matrices, offset, scalar penalty, and official Train target. It does
not choose a model, calculate predictive performance, make predictions, read
Test or SampleSubmission, write a CSV, contact a network, or upload to Kaggle.
The output is in the distinct `gamma_v1/recovery_v1/` namespace. A recovery
failure is terminal in that namespace; no optimizer rescue or retry is allowed.

## Source-control-flow evidence

The committed original source places the artifact write after `_fit_arrays`
returns. `_fit_arrays` checks that the in-memory optimizer metadata reports
`success=True` before it returns the vector. The subsequent receipt construction
looks up `built["matrices"]["checks"]`, which is the lookup named in the saved
`KeyError('checks')` failure. This ordering supports the inference that the
saved vector came through the inherited success gate before the receipt failed.
It is source-control-flow inference only. It does not recover or replace the
missing optimizer metadata, nor prove more than the pinned code path.

## Frozen numerical certificate

For the exact saved arrays, E79 minimizes the summed binary logistic loss plus
`0.5 * c * ||gamma||_2^2`, with `c = 1.1859646641031965`. The Hessian is bounded
below by `c I`, since the logistic Hessian contribution is positive
semidefinite. Therefore the absolute excess-objective bound is

```text
B = ||gradient(gamma)||_2^2 / (2*c)
f(gamma) - min_x f(x) <= B
```

The acceptance rule was fixed before target access:

```text
finite arrays, losses, and gradients
loss(gamma) <= loss(0) + 1e-8
independent stable objective/gradient implementation agrees with frozen helper
B <= max(1e-4, 1e-8 * abs(loss(gamma)))
```

The absolute bound is the sole certificate threshold. `B / abs(loss(gamma))` is
reported as a normalized bound against the attained loss, not as a second gate.
When `loss(gamma) > B`, `B / (loss(gamma)-B)` is also reported as a conservative
relative bound against the unknown optimum. No threshold may be adjusted after
the gradient is seen.

## Execution and recovery namespace

Before any target-only read, the runner verifies the committed recovery source
freeze; the original source freeze/pass; original attempt, artifact, and failure
receipt hashes; official Train hash; exact full-Train saved-state, DF-map, ID,
matrix, and offset parity; the scalar `cI` penalty; required project locks; and
the 2 GiB initial RAM gate plus 1 GiB post-build reserve. It then writes one
separate attempt marker and reads only the official Train `artifact_signal`
column. Runtime guards deny the known SciPy optimizer and E79 fit entry points.

The audit computes loss and gradient at the existing gamma and at zero with the
unchanged frozen helper, then cross-checks them using a separately written
stable `logaddexp`/`expit` implementation. It writes one terminal recovery
receipt with input, target, matrix, offset, DF-map, and gamma hashes, memory
observations, certificate, and zero-fit/access counts. The independent verifier
checks the committed freeze, original failure/artifact identity, receipt
self-hash, all operation denials, and the persisted numerical certificate. It
does not reload target values or replay the calculation; running it twice must
produce the same verification digest.

## Synthetic QA

The synthetic path uses only tiny generated matrices and a caller-provided
temporary directory. It exercises a valid exact-optimum recovery, independent
receipt readback, stable repeated verification, tamper rejection, a mismatched
gamma hash, a deliberately failed numerical certificate, and one-shot resume
behavior. All optimizer and fit entry points are guarded. It does not open
repository data or write official artifacts.

Commands from repository root after the source freeze commit:

```powershell
python -m scripts.audit_optimization_v3_cls_pos_global_strength_e79_gamma_recovery_v1 --verify-source
python -m pytest -q tests/test_optimization_v3_cls_pos_global_strength_e79_gamma_recovery_v1.py
$qa = Join-Path $env:TEMP 'kasdad-e79-gamma-recovery-synthetic-v1'
python -m scripts.audit_optimization_v3_cls_pos_global_strength_e79_gamma_recovery_v1 --synthetic-e2e --output-dir $qa
python -m scripts.verify_optimization_v3_cls_pos_global_strength_e79_gamma_recovery_v1 --verify-synthetic --output-dir $qa
python -m scripts.audit_optimization_v3_cls_pos_global_strength_e79_gamma_recovery_v1 --audit-official
python -m scripts.verify_optimization_v3_cls_pos_global_strength_e79_gamma_recovery_v1 --verify-official
```

The official audit is one-shot after `ATTEMPT.started.json` is written. On
numerical or runtime failure, preserve its terminal receipt and stop. No Test
application pass or submission is authorized by this recovery protocol.
