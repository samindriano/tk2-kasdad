# CLS_POS_RECENCY_V1 execution protocol

This is the one fixed four-fit pilot authorized by `ROOT_PLAN.md`. It changes
only the positional gamma by replacing the saved uniform gamma with a new gamma
trained from zero over the saved positional matrix and `z_BAG`. It retains the
original A02 blend, routing, and thresholds. No Test, SampleSubmission, Kaggle,
or additional model/preprocessing fit is part of this protocol.

## Frozen recipe and windows

- Fit IDs, in order: `E67_RECENCY`, `E67_SHUFFLED_WITHIN_CLASS`,
  `E75_RECENCY`, `E75_SHUFFLED_WITHIN_CLASS`.
- Prefix-present rows only; cycles 47–67 for E67 and 47–75 for E75.
- Base weight: `max(0.25, 2**(-(origin-cycle)/8))`; normalize to preserve each
  target class's original row mass.
- Control: permute weight values within target class, seed 42 reset per origin,
  class 0 followed by class 1 in stable input order.
- Objective: summed weighted binary logistic loss on `z_BAG + Xthirds @ gamma`
  plus `0.5 * ||gamma||²`; L-BFGS-B, float64, fixed options in the source.
- Present route: replace gamma; blend positional and saved A02 probabilities
  50/50; threshold 0.50. Missing route: saved A02 probability bit-exactly and
  threshold `0.53230976129477`.
- Query windows: E67 cycles 68–79 (20,903 rows); E75 cycles 76–79 (8,491 rows),
  which must be the exact E67 subset. Primary is the unique E67 68–75 plus E75
  76–79 pool (20,903 rows).

## Ordered local commands

Run from the repository root, with the official KASDAD root as the default
`--official-root` (`..` from this repository):

```powershell
py -m scripts.cls_pos_recency_v1 prefit
git add reports/simplification_v1/CLS_POS_RECENCY_V1/PREFIT_QA.json reports/simplification_v1/CLS_POS_RECENCY_V1/FREEZE.json
git commit -m "Freeze CLS positional recency pilot inputs"
py -m scripts.cls_pos_recency_v1 reserve
git add reports/simplification_v1/CLS_POS_RECENCY_V1/RESERVATION.json
git commit -m "Reserve four CLS positional recency fits"
py -m scripts.cls_pos_recency_v1 fit-all
py -m scripts.cls_pos_recency_v1 predict
py -m scripts.cls_pos_recency_v1 prelabel
git add reports/simplification_v1/CLS_POS_RECENCY_V1/PREDICTIONS_READY.json reports/simplification_v1/CLS_POS_RECENCY_V1/PRELABEL_VERIFICATION.json
git commit -m "Verify target-free CLS recency predictions"
py -m scripts.cls_pos_recency_v1 score
py -m scripts.cls_pos_recency_v1 verify-score
```

Do not run `fit-all` until the source, protocol, and tests have been committed;
the runner checks this. Every optimizer start consumes its named slot, failures
are terminal, and saved units may only be read back. Fit execution requires at
least 1 GiB available before start and retains at least 512 MiB afterward, under
the shared OS lock and a two-thread limit. Scoring starts at 512 MiB and records
whether 256 MiB remains after metric construction.

The prefit check reads one prefix-only target vector for the ESS/class-mass
preflight. Each of four training fits and its independent objective verifier
reads its own prefix-present targets. Query labels are not opened until both
prediction bundles and independent prelabel verification are durable and
committed; scoring opens the unique 20,903-row Train query once. The ledger and
reports must reconcile every read. Query verification does not reopen labels.

## Required local QA

Before official optimizer starts, run:

```powershell
py -m py_compile src/models/cls_pos_recency_v1.py scripts/cls_pos_recency_v1.py scripts/verify_cls_pos_recency_v1.py
py -m scripts.cls_pos_recency_v1 --help
py -m pytest tests/test_cls_pos_recency_v1.py tests/test_cls_pos_recency_v1_synthetic.py -q
```

The synthetic integration test uses only generated data in a temporary folder.
It checks both prediction origins, unique/non-overlap rows, availability and
fit-prefix text-load groups with supported and unsupported denominators,
bit-exact missing fallback, one-pass synthetic scoring, independent metric and
gate reconstruction, receipt tampering, non-retryable consumed starts, saved
unit readback without optimization, and missing/duplicate source IDs. Its
outputs do not count against the four official fits.

## Stop and interpretation

Stop after the single score pass and independent score verification. Report the
primary, E67 W4/W5/W6/H12, E75 W6, per-cycle, subgroup, recency-minus-shuffled,
resource, and read-accounting results, including negative findings. A scientific
gate failure remains a failure. Exploratory eligibility is diagnostic only and
does not authorize another fit or Test inference. If exploratory eligible,
inventory E79 inputs only for root review; otherwise close this recipe.
