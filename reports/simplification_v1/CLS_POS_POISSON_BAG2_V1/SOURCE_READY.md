# CLS_POS_POISSON_BAG2_V1 — Source-ready handoff

**Current stage:** source and synthetic QA only. No official fit has started.
No official query labels, Test rows, SampleSubmission, CSV, or Kaggle service
have been accessed by this candidate runner.

## Source-only checks already run

Recorded in the current source-preparation pass: `py_compile` passed for all
four candidate Python files, and the focused suite passed **12 tests**. One test
invokes the synthetic CLI under `python -O` twice against the same output
directory and checks byte-stable JSON output on resume. The suite does not
perform official-data fits or inspect official score values.

The post-fit path also has a separate verifier that will re-read the approved
Train labels after prediction readback, reconstruct each temporal window,
cycle, subgroup denominator, Macro F1/recall contrast, qualification gate,
query-label ledger event, and lineage hash. A persisted score result is
recomputed on resume; a changed result or approval hash is rejected.

From the audit worktree root:

```powershell
python -m py_compile `
  src/models/cls_pos_poisson_bag2_v1.py `
  scripts/cls_pos_poisson_bag2_v1.py `
  scripts/verify_cls_pos_poisson_bag2_v1.py `
  tests/test_cls_pos_poisson_bag2_v1.py

python -m pytest -q tests/test_cls_pos_poisson_bag2_v1.py
```

The synthetic lifecycle is deliberately isolated from official repository
inputs and outputs. It creates four small gamma fits, writes artifacts and
receipts, runs prediction and independent readback, reopens the same outputs to
check deterministic resume, and verifies tampering fails closed. It also
includes an NPZ with an object-valued `labels` member and confirms the A02
loader returns only the explicit non-label allowlist.

## Required source and freeze sequence

Run once before the input freeze commit:

```powershell
python -O scripts/cls_pos_poisson_bag2_v1.py `
  --command synthetic `
  --output-dir "$env:TEMP\CLS_POS_POISSON_BAG2_V1_SYNTHETIC"
```

Commit the reviewed, named source files first. Then, with a clean worktree,
create the metadata-only prefit freeze:

```powershell
python -O scripts/cls_pos_poisson_bag2_v1.py `
  --command freeze `
  --repository "C:\Users\Sam\OneDrive\Documents\KASDAD\TP 2"
```

Review and commit only
`reports/simplification_v1/CLS_POS_POISSON_BAG2_V1/PREFIT_FREEZE.json` after
checking its source commit, official repository HEAD, official Train hash,
input file hashes, external source hashes, and `data_boundary` fields. The
freeze command reads JSON metadata and hashes the complete pinned files; it
does not open Train labels, load matrix contents, index the A02 `labels`
member, or access Test/SampleSubmission.

## Root prefit approval required before official fit

Sol must review the committed source and freeze, then create and commit
`reports/simplification_v1/CLS_POS_POISSON_BAG2_V1/ROOT_PREFIT_PASS.json`.
The runner requires exactly these fields and values, with `validation_label_scoring`
set to `true` only if matched official-Train validation scoring is approved:

```json
{
  "status": "PASS_ROOT_PREFIT_APPROVED",
  "candidate_id": "CLS_POS_POISSON_BAG2_V1",
  "freeze_signature": "<exact PREFIT_FREEZE.json signature>",
  "audit_source_commit": "<exact source commit recorded in freeze>",
  "fit_ids": [
    "CLS_POS_POISSON_BAG2_V1_E67_P17",
    "CLS_POS_POISSON_BAG2_V1_E67_P137",
    "CLS_POS_POISSON_BAG2_V1_E75_P17",
    "CLS_POS_POISSON_BAG2_V1_E75_P137"
  ],
  "predictor_fit_cap": 4,
  "validation_label_scoring": true,
  "test_or_sample_access": false,
  "kaggle_upload": false
}
```

The approval file must already be committed at `HEAD`; an uncommitted or
rebound approval fails closed. No approval file is included in this source
handoff.

## Exact continuation commands after root approval

All commands take the explicit official TP2 checkout. Run **one fit at a
time**, in the listed order, and stop permanently on any fit failure. There are
no retries and no substitute fits.

```powershell
$repo = "C:\Users\Sam\OneDrive\Documents\KASDAD\TP 2"
python -O scripts/cls_pos_poisson_bag2_v1.py --command verify-freeze --repository $repo
python -O scripts/cls_pos_poisson_bag2_v1.py --command reserve --repository $repo
python -O scripts/cls_pos_poisson_bag2_v1.py --command fit --repository $repo --fit-id CLS_POS_POISSON_BAG2_V1_E67_P17
python -O scripts/cls_pos_poisson_bag2_v1.py --command fit --repository $repo --fit-id CLS_POS_POISSON_BAG2_V1_E67_P137
python -O scripts/cls_pos_poisson_bag2_v1.py --command fit --repository $repo --fit-id CLS_POS_POISSON_BAG2_V1_E75_P17
python -O scripts/cls_pos_poisson_bag2_v1.py --command fit --repository $repo --fit-id CLS_POS_POISSON_BAG2_V1_E75_P137
python -O scripts/cls_pos_poisson_bag2_v1.py --command predictions --repository $repo
python -O scripts/cls_pos_poisson_bag2_v1.py --command score --repository $repo
```

The four fits read only the fixed E67/E75 fit-prefix labels after starting a
single ledger event. Prediction bundles are persisted and independently
verified before the `score` command can read matched temporal Train labels.
`score` also requires `validation_label_scoring: true` in the committed root
approval. Prediction and score commands do not infer or inspect official Test
or submission files. This lane authorizes no Test inference or Kaggle upload.

Fit start requires at least 1 GiB available RAM and post-fit requires at least
512 MiB available. The runner takes the shared `kasdad_single_heavy_job.lock`,
sets a two-thread limit, and records resource observations. A visible lock file
can persist after the OS advisory lock is released; its existence alone is not
evidence that another job is running.
