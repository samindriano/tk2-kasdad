# CLS_POS_POISSON_BAG2_E79_V1 protocol

## Purpose and status

This is a source-only implementation for one explicitly bounded E79 benchmark
continuation of the existing positional Poisson BAG2 pilot. The pilot remains
historical exploratory evidence and is not overwritten or requalified here.
The continuation is fixed to exactly two official full-Train gamma fits and one
Test application after a separate committed root pass. Source preparation and
synthetic QA do not perform either official action.

## Fixed fit recipe

- Official fit population: Classification Train, 50,608 ordered rows, cycles
  47–79, target `artifact_signal`.
- Feature lineage: the frozen C2_B/base and positional matrices plus C2_B offset
  from the pinned official `optimization_v3_cls_pos_global_strength_e79_gamma_v1.py`
  builder. Both matrices, ordered Train IDs, raw Train, old source dependencies,
  and old immutable inputs must match the hashes in the prefit freeze.
- Reuse the full-Train CGeom E79 `POS_BAG_OFFSET` gamma package member
  `payload["gamma"]` (artifact SHA-256
  `d741d016f990d8742208edc919f5c58305b7579980d059b202609df4e5b3e650`, logical
  array SHA-256 `31f8bde49ca3fa85f866e7fb685d44825bd77e5cc2d11ae848494aca687ac0b3`).
  The recovered global-strength gamma is not admissible.
- Fit only two new positional gamma vectors, with Poisson(1) row counts seeded
  17 and 137 and weights normalized to sum to 50,608. Use the inherited
  zero-initialized L-BFGS-B routine, maxiter 2,000, ftol 1e-12, gtol 1e-6,
  maxls 50, and the uniform C=1 penalty. No preprocessing or original head is
  fit again. There is no retry or seed replacement.
- Fixed full-champion probability is the present-stream 50/50 blend of the
  original positional probability and A02 pA, with A02 pA alone for missing
  stream. Candidate positional probability is the mean of original, seed-17,
  and seed-137 probabilities; candidate present-stream probability is the same
  50/50 blend with A02 pA. Missing-stream candidate probability is exactly A02
  pA. Thresholds remain 0.50 present and 0.53230976129477 missing, using `>=`.

## Source and data boundaries

The source freeze binds the audit source commit and six candidate source files,
official repository HEAD, upstream source freeze and file hashes, original
full-champion gamma package, official Train byte hash, and the historical
metadata identities for Test and the A02 pA artifact chain. Source preparation
may verify hashes and load the already-fitted original gamma package. It does
not load Train targets, open Test, open pA predictions, read SampleSubmission,
or contact Kaggle.

After a committed root pass, `reserve` and `fit-all` may use official Train
features and then read `artifact_signal` only after exact matrix and ordered-ID
parity passes. Each attempt marker is persisted before a fit starts; a failed
or partial attempt consumes that fit ID and is never retried. Artifacts and
receipts are independently read back. Test application has its own one-shot
marker, verifies the full-champion decisions against the pinned prior CSV on
the exact official Test IDs, and writes the candidate CSV only after exact
baseline parity. The producer does not load Test labels or SampleSubmission and
does not contact Kaggle.

## Resource and persistence controls

At most two numerical threads are used. The fit path must acquire both the
shared research lock and the official CGeom lock. It requires at least 2 GiB
available RAM before feature reconstruction, 1 GiB after reconstruction and
before each fit, and 512 MiB after a fit. A low-memory or lock failure occurs
before any new fit attempt marker. Once a fit marker exists, no retry is allowed.

The runtime namespace is `experiments/CLS_POS_POISSON_BAG2_E79_V1/`; it is
ignored by Git. The official submission CSV, if later created by the approved
application, is also outside version control. Tracked source, freeze, receipts,
and report metadata remain auditable.

## Commands

Run from the audit worktree with the official TP2 checkout path explicitly
provided. Use the exact commands below only after the relevant gate is satisfied.

```powershell
python -O scripts/cls_pos_poisson_bag2_e79_v1.py --command synthetic --output-dir "$env:TEMP\CLS_POS_POISSON_BAG2_E79_V1_SYNTHETIC"
python -O scripts/cls_pos_poisson_bag2_e79_v1.py --command verify-freeze --repository "C:\Users\Sam\OneDrive\Documents\KASDAD\TP 2"
python -O scripts/cls_pos_poisson_bag2_e79_v1.py --command reserve --repository "C:\Users\Sam\OneDrive\Documents\KASDAD\TP 2"
python -O scripts/cls_pos_poisson_bag2_e79_v1.py --command fit-all --repository "C:\Users\Sam\OneDrive\Documents\KASDAD\TP 2"
python -O scripts/cls_pos_poisson_bag2_e79_v1.py --command apply-test --repository "C:\Users\Sam\OneDrive\Documents\KASDAD\TP 2"
```

The `freeze` command is source-preparation only and writes the one-shot
`PREFIT_FREEZE.json`; it must run only after candidate source files have been
committed and the audit worktree is clean. `reserve`, `fit-all`, and
`apply-test` fail closed unless the source freeze is valid and a committed
`ROOT_PASS.json` authorizes the same freeze, exact two fit IDs, and one
application. Run `verify-freeze` before root review and before every later
official command. Do not invoke `fit-all` or `apply-test` as part of source QA.

