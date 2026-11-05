# R3 saved-state replay on an attested-source descendant

## Purpose

The frozen R3 replay script requires its invocation `HEAD^` to equal the
authorization commit. The present checkout has later commits, while the
attested R3 source and all of its pinned current-source paths remain exact.
This narrow continuation preserves the original R3 runner, verifier,
attestation, preprocessing, matrix recipe, and gates. It accepts the R3
producer commit only when that exact commit is an ancestor of the invocation
checkout, and independently checks every source path named in the attestation
against both the current worktree and current committed tree.

The candidate producer commit is
`f6eb5938f9884a3b154ebd4bb6a5daa66ee2239d`; its parent remains
`71192b65c4ba0b1e31fd9273476d00b28f96b343`. The current invocation HEAD is
recorded separately. A later unrelated commit is allowed only when the exact
attested source paths, attestation body, and immutable input hashes all pass.

## Scope and controls

- Reuse the original R3 `_execute_feature_only_replay` once, unchanged.
- Reuse the original R3 attestation and independent receipt verifier.
- Run on official Classification Train feature columns only.
- Do not read `artifact_signal`, `threat_rank`, Test, or SampleSubmission.
- Do not fit an estimator, preprocessing state, vocabulary, IDF, DF map, or
  gamma; do not predict, infer, create a CSV, or contact Kaggle.
- Keep the R3 1 GiB incremental estimate, 768 MiB free reserve, two-thread
  limit, shared OS lock, and CGeom local owner-file lock.
- If resource or lock preflight fails, do not write an attempt marker or
  receipt. Once an attempt marker is written, any failure is durable and must
  not be retried under this authorization.

## Execution order

1. Verify R3 attestation, all 30 immutable input hashes, source paths, source
   ancestry, current committed-path equality, and this continuation freeze.
2. Run both source test suites. R3's 11-test suite covers its synthetic
   end-to-end persistence/verifier path; continuation tests cover descendant
   ancestry, exact-path parity, resource blocking, and no-retry persistence.
3. Build and commit `DESCENDANT_REPLAY_FREEZE.json` after the protocol, adapter,
   tests, and narrow line-ending rules are committed. The freeze binds their hashes, the R3 attestation
   and internal digest, attested source paths, immutable input hashes, producer
   ancestry, and prepared source HEAD.
4. Run read-only lock/RAM preflight. A failed resource or lock gate writes
   neither the one-shot attempt marker nor a replay receipt.
5. Execute one saved-state feature replay and persist its one-shot receipt at
   `experiments/runs/optimization_v3_cls_pos_global_strength_e79_v3/descendant_replay_v1/`.
6. Independently verify the receipt with the original R3 verifier and preserve
   the result. If base and positional matrix parity passes, a separate frozen
   DF-map milestone may proceed under its own source/input freeze and tests.

## Reproduction commands

Run from the repository root, after committing the protocol, adapter, and tests:

```powershell
python -m pytest -q tests/test_cls_pos_global_strength_e79_v3.py tests/test_optimization_v3_cls_pos_global_strength_e79_r3_descendant_replay_v1.py
python -m scripts.optimization_v3_cls_pos_global_strength_e79_r3_descendant_replay_v1 --build-freeze
```

Commit only the narrow `.gitattributes` line-ending rules, new adapter, its
tests, this continuation protocol, and generated freeze. Then run:

```powershell
python -m scripts.optimization_v3_cls_pos_global_strength_e79_r3_descendant_replay_v1 --preflight
python -m scripts.optimization_v3_cls_pos_global_strength_e79_r3_descendant_replay_v1 --feature-replay
python -m scripts.optimization_v3_cls_pos_global_strength_e79_r3_descendant_replay_v1 --verify-replay
```

The feature-replay command is one-shot. Do not rerun it after an attempt marker
exists, including after interruption or a durable failure receipt.

The continuation does not alter historical R1/R2/R3 attempts, their verdicts,
the R3 source attestation, the original recipes, or the champion. It does not
authorize a gamma fit, Test inference, CSV, Kaggle upload, or any additional
model/preprocessing work. Root review is required before any supervised gamma
fit.
