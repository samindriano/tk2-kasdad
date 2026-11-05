# Classification NO_POS E79 benchmark

## Frozen decision

This is one controlled deployment benchmark of the already-audited `NO_POS`
candidate. It removes positional gamma / token-third effects while retaining
the A02 probability route and its donor-memory correction. The reported local
retention result remains the historical audit result; this application does not
recompute or upgrade that verdict.

For each official Test row, read only `record_id` and `signal_stream` from Test.
Use the existing, hash-bound E79 `a02_probability` (`pA`) vector:

- Stream present: predict 1 when `pA >= 0.50`.
- Stream missing after whitespace trimming: predict 1 when `pA >= 0.53230976129477`.
- Both comparisons use `>=`. The missing route is the frozen A02/CatBoost
  fallback; no POS output is used.

No classifier, neighbor, threshold, preprocessor, or feature state is fitted.
No new model inference is run. The committed `PRE_ACCESS_FREEZE.json` binds
the current audit decision, existing E79 probability bundle, official Test,
SampleSubmission, and source code before those E79/Test inputs are opened.

## Audit evidence and limits

The read-only simplification audit selects `NO_POS` as its only Classification
candidate passing the declared local retention guard. It reports Macro F1 gains
of `+0.0006066` at E67 H12, `+0.0004897` at E75 W6, and `+0.0003276` on the
unique pool. The maximum supported subgroup F1 harm is `0.0043974`; maximum
positive-recall harm is `0.0195489`, close to the `0.02` bound. These are
reused, adaptive temporal views, not fresh independent holdouts.

The audit's public counterevidence remains visible: historical A02 public Macro
F1 was reported as `0.82930`; the later MIX50/A02 operational champion reached
`0.83225` (submission `58001282`).
Therefore a better local simplification result does not establish a better
public score or justify changing the champion. The single authorized Kaggle
submission is only a benchmark, not an automatic model-selection decision.

## QA, output, and stop conditions

The builder must verify all frozen hashes, upstream E79 receipts, 29,421 unique
ordered IDs across the probability bundle, official Test, and SampleSubmission,
the availability mask, exact columns, integer 0/1 labels, and output replay.
It scans local same-schema submissions for exact label-vector duplicates. A
hash mismatch, absent input, unsupported archive schema, duplicate already
submitted vector, or failed live Kaggle rules/auth/quota/idempotency check blocks
upload. The CSV is exclusive-write; a different existing file is never replaced.

After CSV QA, query authenticated Kaggle CLI submission history and live
competition information. Preserve an upload-intent receipt before the one
authorized upload. Never retry an ambiguous upload; confirm server status and
score before reporting success. Keep the CSV and model artifacts out of Git.

## Resource policy

The audit's historical saved-replay gate was 1.5 GiB initial free RAM. On
2026-11-05 Sam authorized two parallel **light cached-vector/CSV** jobs (one CLS,
one REG), each estimated at or below 512 MiB, with at least 2.5 GiB available
before start and at least 1 GiB reserve at boundaries. This task uses a
per-candidate idempotency lock. It does not fit or reconstruct large feature
matrices and does not acquire/bypass the shared heavy-fit lock.

Run the synthetic source check with:

```powershell
python -m scripts.cls_simplification_benchmark_v1 --self-test
python -m pytest tests/test_cls_simplification_benchmark_v1.py -q
```

After the pre-access freeze commit and a passing resource gate, build the one
candidate with:

```powershell
python -m scripts.cls_simplification_benchmark_v1 --build
```
