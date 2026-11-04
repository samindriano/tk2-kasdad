# POS_MOMENTS_V1 — historical prelabel verification blocker

**Historical status at the time of this report: BLOCKED before outcome metrics.**
This records the original prelabel failures. Those failures were subsequently
resolved by the separately authorized zero-fit safe-export and scalar-replay
amendments described at the end of this report. The original failure receipts
remain preserved.

## Frozen experiment and accounting

- Freeze signature: `ea1b2ee797356ab5dd889f599ea60bfb63a2c5f19ddf52231364b818bb7f4ea3`
- Frozen source commit: `6a2baeb92406c38ca50b1b7a4e85a938e03cf1fa`
- Local checkpoint before the verification amendment: `08087d60d94f47e71b3de412d987059ec3b4cfb7`
- Official ledger SHA-256: `e913ea21aaf8357916e051b0e22ab12bbf4b5bec0a4c337117753f5aef77bcb0`
- Ledger: 356 events; cap 142; 137 active reservations; 137 fit starts; 131 saved;
  6 historical failures; 0 pending. POS_MOMENTS_V1 added exactly four starts and
  four saves, with no failure or retry.

## Durable fits

All four L-BFGS-B optimizers reported success. Gamma, receipt, and artifact hashes
are recorded here; the binary bundles and receipts remain in the ignored run tree.

| Fit | Iterations | Gamma SHA-256 | Receipt SHA-256 | Artifact SHA-256 |
|---|---:|---|---|---|
| `POS_MOMENTS_V1_E67_REAL` | 41 | `b1796eb499553ad23e93cce3e68d7fc244948b042736f16808d6bf62e30f73cd` | `15d6b816b52dc53b5178c1de5ba0fca3751c311c0b4ecaa4bed7370d4017d4da` | `d9661ee236c906fbb32df94d205a49d0033dcdb909c3ce9a7899a508c6505ac7` |
| `POS_MOMENTS_V1_E67_SHUFFLED` | 37 | `0fbe4fa9d0c881c55aa8430ac560c93d505ad624cf59dc296418352b44a4020c` | `4812c25eadbdb1d901f1d65560da645f0140b28f20f684e0977b7e842f6ccbdc` | `d90c92237dc5d3a2c2862c1d9f3efa2097b9027da17b83f5c84e0dc735aa917d` |
| `POS_MOMENTS_V1_E75_REAL` | 51 | `4b2c1e0974b07af3de0a3c5c745d2e9c32d506d58d79c02179a55f988f8a501a` | `854c7f55110742d9663216f0728eb602c6538c1252b480b1e2ad8319efb4cf9e` | `acf048b110c1ba66c2fc3f70727d2ccc17267609d8e388f29411fcb06198501a` |
| `POS_MOMENTS_V1_E75_SHUFFLED` | 46 | `77c603aa6db16b86632df77184a897b9c40c72dd4d6ae4e26781c26a65be1953` | `3be5d2058ec0f6a69bbbc62343dfb1390123e028260c9dbe0bbe43cff75f056b` | `df2610652976d4ab0605eb1b590b13f5fa22b3b87370aeffcb55424df9727253` |

Prediction bundles are present and hash-bound in `PREDICTIONS_READY.json`:

- E67, 20,903 rows: `e825b87ce10cad00484737870ae8f30a0cda7b2cdf332a4189cdda2bd8260c13`
- E75, 8,491 rows: `6192e2ddee664236c63c714fe7b04d62018d8ca922d36ba8715ece9b49681d9c`

## Verification failures

1. The frozen `verify-prelabel` entrypoint first failed with
   `KeyError: 'predictions'`. Its unchanged verifier expects a mapping, while the
   historical `POSITIONAL_TEXT_PILOT_V1/PREDICTIONS_READY.json` stores the same
   per-origin lineage in an `origin_predictions` list. The historical receipt was
   not edited.
2. With an in-memory-only, origin-keyed adapter, both historical parent prediction
   files and matrix manifest paths/hashes matched the prefit freeze. The unchanged
   verifier then stopped on the E67 output bundle: `score_record_ids.npy` has NumPy
   dtype `|O` and shape `(20903,)`. Its fail-closed `allow_pickle=False` loader
   correctly rejected that object array. The `.npy` header was inspected without
   unpickling the array. No bundle was rewritten.

The frozen verifier source SHA-256 remains
`eb7d8ec4cc49c3f6663ba284bb67fa4a048456bb007b1e4d1e677cf5fc9da7c2`.
The separate adapter source SHA-256 is
`4135c8427ed8505f41c6521db916d02f7d5a3faf8c4df5e7ff9a2c109c5e7138`.
The original failed attempt and adapter failure are also preserved in
`PRELABEL_SCHEMA_ADAPTER_FAILURE_V1.json`.

## Boundaries and next decision

- Both verifier attempts failed before any query-label access; the verifier reads
  only Train feature columns in this stage.
- Test and SampleSubmission were not accessed.
- No outcome metrics or gates were opened; do not interpret optimizer loss as
  validation performance.
- No E79, additional fit, Test inference, Kaggle action, or submission occurred.
- The first postfit test run passed 16 tests for the parent schema adapter and
  synthetic lifecycle. After the zero-fit export amendment was added, 23 tests
  passed, including safe Unicode export/readback, copied-member identity,
  wrong-row/hash, duplicate/missing ID, tamper rejection, and the synthetic
  lifecycle suite.

Sol authorized a narrow zero-fit serialization amendment. Signed safe exports
are recorded in `SAFE_EXPORT_MANIFEST_V1.json`; both origins were validated
against Train identity rows and parent outputs. The frozen verifier's
vectorized reconstruction then differed from the cached CSR values only at
floating-point summation precision, so a separate scalar-order replay was
implemented and tested without changing the frozen formula or verifier.

## Resolution and final outcome

- The signed safe-export prelabel verification passed using the literal scalar
  accumulation order, with source/Train hashes, parent lineage, named arrays,
  IDs/cycles, fit receipts, independent matrix reconstruction, fixed routing,
  and MIX50 all verified. The prelabel stage read no query labels and accessed
  no Test or SampleSubmission data.
- The authorized scoring pass then read only the frozen Train query labels.
  Independent QA passed integrity/recomputation checks (`QA.json`), while the
  scientific verdict remained `PILOT_GATE_FAILED`: 45 of 47 gates passed and
  the two primary real-gain requirements failed. The full results and gate
  deltas are in `RESULTS.md` and `RESULTS.json`.
- Final accounting records exactly four saved pilot fits, zero failures or
  retries for this card, no E79 reservation/start, and no Test/Sample access.
  No champion change, full-Train refit, upload, or Kaggle submission occurred.
- The frozen verifier, freeze, gamma artifacts, receipts, original prediction
  archives, and prediction values remain unchanged. The schema/dtype readers,
  scalar replay, receipts, and synthetic checks are separate amendments; the
  two earlier prelabel failures remain preserved in their original receipts.
