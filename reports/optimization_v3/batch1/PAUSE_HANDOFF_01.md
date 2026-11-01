# V3 Batch 1 — pause handoff

**Campaign status: `PAUSED_FOR_REVIEW` — not complete.** This checkpoint follows screening commit `5e885db338cac1ba6dc658b6b36f006a13f023d4`. No continuation fit, full-Train refit, Test inference, or Kaggle submission was started after it.

## Pilot screening and QA

All 45 IDs are reconciled in [`batch_ledger.json`](batch_ledger.json) and [`BATCH_FINDINGS.md`](BATCH_FINDINGS.md): 7 ADVANCE, 36 RETIRED on valid predeclared gate evidence, and 2 prerequisite skips. QA outcomes are 28 `PASS`, 16 `INDEPENDENT_QA_PASS`, and 1 `PASS_WITH_CONTROL_SCOPE_RECONCILIATION` (C09). The reconciliation note and card-level reasons, failed gate checks, subgroup failures, source/input hashes, artifact hashes, and verifiers are in the registry.

- R10: skipped with zero fits because R09 had negative primary and H12 gains and supported-tail worsening at both origins.
- R13: skipped with zero fits because its R12 prerequisite lacked the declared per-origin within-cycle Spearman evidence; the aggregate primary/H12 gates had passed.
- The 36 RETIRED recipes have valid saved Train evidence and passing QA; card-level blockers are their predeclared gate failures, not pending technical rework.

## Continuation screen

The selected STANDARD candidates are **C16** (`sparse_capacity`), **C11** (`character_boost`), **R22** (`xgb_loss`), and **R06** (`character_boost`). Selection used same-profile pilot gains on the same denominators (primary 9,064; H12 38,588; W1 4,114; W4 4,950), with family diversity within the two-candidate-per-task cap.

| Card | Task | Primary gain | H12 gain | W1 / W4 gain |
|---|---|---:|---:|---:|
| C16 | Classification, `sparse_capacity` | +0.001971 Macro F1 | +0.002837 | +0.003612 / +0.000597 |
| C11 | Classification, `character_boost` | +0.000661 Macro F1 | −0.000310 | +0.000369 / +0.000880 |
| R22 | Regression, `xgb_loss` | +0.098176 MAE | +0.124873 | +0.138456 / +0.064698 |
| R06 | Regression, `character_boost` | +0.061726 MAE | +0.103260 | +0.065279 / +0.058773 |

C17 (same family as C16), R21 (same family as R22), and R11 (lower primary rank, 18 predictor fits per pilot) remain ADVANCE but were not selected under the slot cap and ranking rule. No LATE candidate passed the continuation gate. The selection is a screening recommendation awaiting review, not a final model decision.

**Continuation status:** not started. None of cutoffs 59, 63, 71, or 75 has been fitted for these candidates. The already saved pilot outputs use cutoffs 55 and 67. The results already reviewed are the saved pilot outcomes for all 45 cards and the selected candidates' pilot gains/gates; these are earlier pilot evidence, not continuation evaluation. Existing full-six control references inside frozen artifacts are control provenance, not newly run candidate continuation results.

**Technical blocker before any resume:** the current C16/C11/R22/R06 runners are pilot-specific. Do not call them for continuation. After review authorizes resumption, prepare a separate continuation wrapper with a new frozen identity and output namespace, reserve the full 16 predictor fits (4 new origins × 1 fit × 4 candidates), and independently verify all six primary windows, H12, subgroup support/gates, row alignment, and source boundaries. The existing continuation audit confirmed the per-card cards declare those four cutoffs, but did not verify that current static memory estimates cover them.

## Budget, attempts, and saved bundles

- Cap: 180 candidate predictor fits. Existing deduplicated reservations: 94; verified predictor fits saved: 94; available budget: 86. The 16 planned continuation fits have **not** been reserved. Three legacy reservations (C06, R12, C01) were imported append-only; the ignored runtime ledger is `experiments/runs/optimization_v3_batch1/batch_ledger.jsonl`.
- Two `FIT_FAILED` events occurred: R21 cutoff 55 (control `baseline_sparse_raw` outside [0,100]) and R04 cutoff 55 (control bridge lacked `control_rows`). Each later saved cutoff 55 under an amended freeze and saved cutoff 67. These failures remain in the attempt history; no failed fit is represented as a successful result.
- C11 cutoff 67 was interrupted once at 0.49 GiB system free RAM (below the frozen 2 GiB reserve; process working set 2.22 GiB). Its cutoff 55 bundle was preserved and verified on resume; cutoff 67 was subsequently saved and verified. There was one `RESUMED_VERIFIED` event.
- C15 had one runner dispatch failure before fit (`attempted_predictor_fits=0`); the later corrected pilot result is the saved result. R11 reports 18 predictor fits, not two: its two saved origin events each include nine ordinal heads.
- Pilot NPZ and JSON sidecars remain in `experiments/runs/optimization_v3_batch1/{C16,C11,R22,R06}/`. The selected pilot bundles are `C16_fit55/67`, `C11_fit55/67`, `R22_fit55/67`, and `R06_fit55/67` (each with `.npz` and `.json`). Do not overwrite them. Reports, gates and QA remain in `reports/optimization_v3/batch1/{ID}/`.
- Frozen pilot identities: `experiments/optimization_v3/batch1/C16/freeze.json`, `.../C11/freeze.json`, `.../R22/freeze.json`, and active R06 `.../R06/freeze_revision_02.json`. Retain these; continuation would require a new, separate frozen package.

## Safe resume state

At pause audit, branch was `main`, HEAD was the screening checkpoint above, working tree was clean before this handoff change, and no Python training process was running. The latest live RAM snapshot was 8.57 GiB free of 31.29 GiB; this is informational and must not substitute for a fresh preflight later. A one-byte `experiments/runs/optimization_v3_batch1/.training.lock` file is present, while no Python process was found. Preserve it; inspect lock ownership/state before any resume, and do not delete it merely to force a run.

After review and explicit resumption: confirm current HEAD and this pause state; preserve all pilot/frozen files; inspect the lock; implement and verify the continuation-only path and full-six verifier; recheck RAM/dependencies; reserve all continuation fits before the first fit; then run one two-thread job at a time. Do not access Test or Kaggle during continuation. No goal tracker was created, so there is no separate goal state to pause.
