# A02/CGEOM three-candidate E79 benchmark preparation

**Status:** preparation authorized; stop after three locally validated CSVs. No Kaggle upload is part of this batch.

## Scope and starting state

- Authorized starting HEAD: `f3baa4014378a9fcb1a58e898d1b4c9a87679728` on `main`; working tree was clean.
- Namespace: `a02_cgeom_benchmark_v1`.
- Input population: all 50,608 official Classification Train rows, cycles 47–79. Test may be read only after all three recipes and every Train learned state are frozen and hashed. Test targets are never loaded.
- Maximum official fit units: **9**. No retries, extra controls, temporal evaluation, Kaggle calls, or uploads.
- New shared cap amendment: global cap `121 -> 130`; continuation cap `38 -> 47`; pilot `80`, control replay `2`, and technical retry `1` stay fixed. At the checkpoint there were 118 starts, 114 saved, 4 failed, 0 pending, and 3 units of global headroom. This batch reserves exactly 9 continuation units; it does not spend the remaining three units.

## Reconciled fit units

| Unit | Purpose | Reuse / fit population |
|---|---|---|
| C2_B structured state | Expanded structured preprocessing | Fit on official Train features only |
| C2_B scaler | Numeric standardization | Fit on official Train features only |
| C2_B OHE | One-hot categories | Fit on official Train features only |
| C2_B LR | Shared C2_B bag baseline | All Train labels; C=1, liblinear, max_iter=2000, tol=1e-4, seed 42 |
| C1_B structured state | Native categorical preprocessing | Fit on official Train features only |
| C1_B HGB | Shared missing-stream fallback for POS_BAG_OFFSET and JOINTPOS | All Train labels; original C1_B settings |
| POS_BAG_OFFSET gamma | Fixed C2_B bag logit offset plus real-position third blocks | All Train labels; original sum logistic loss, C=1, L-BFGS-B settings |
| JOINTPOS LR | Joint C2_B bag plus real-position thirds | All Train labels; original C2_B LR settings |
| CGEOM aligned predictor | Coefficient-weighted A02 donor geometry | Existing A02 residual bank and TF-IDF state; one indexed predictor unit |

The C2_B TF-IDF vectorizer is reused from the verified full-Train A02 E79 fitted state. No vectorizer fit is permitted. Previous C16 outputs contain only component predictions and do not contain refittable C2_B or C1_B estimator states. Temporal C1_B/C2_B states are not full-Train substitutes.

## Frozen candidate recipes

1. `classification_cgeom_aligned_e79.csv`: exact `CGEOM_V1` aligned recipe, original Sparse25/A02 baseline, complete A02 residual bank, original top-32 weighted cosine, per-query mass matching, residual correction clipping and original A02 fallback. The historical `CLOSED_GATE_FAIL` remains unchanged; this is batch preparation evidence only.
2. `classification_posoffset_e79.csv`: exact `POS_BAG_OFFSET_V1` real-position recipe; shared full-Train C2_B structured state, scaler, OHE, A02 E79 vectorizer, and C2_B LR; third-position blocks from the frozen positional builder; only gamma is newly optimized with the original SUM logistic loss plus C=1 penalty and original L-BFGS-B options; C1_B HGB fallback for stream-missing rows.
3. `classification_jointpos_e79.csv`: exact real-position joint LR recipe; C2_B structured features, shared A02 E79 TF-IDF and real-position third blocks, original C2_B LR settings fit over all Train rows; same C1_B HGB missing fallback.

All three use the existing 0.50 threshold. Each keeps its original route and preprocessing. No recipe, threshold, weight, seed, binning, or blend search is allowed. Test-derived label counts and pairwise CSV deltas are descriptive integrity records only; they cannot choose a recipe or change any historic verdict.

## Mandatory order and stop

1. Commit this amendment, the exact recipe manifest, runner, and synthetic tests before official fitting.
2. Run the synthetic end-to-end and persistence/resume checks with a temporary ledger; prove they cannot write to the shared ledger.
3. Recheck actual shared budget state and resource limits. Append the prospective cap amendment and one idempotent 9-unit reservation only after synthetic QA passes.
4. Fit one unit at a time, 2 threads maximum, at least 4 GiB free before first unit, at least 2 GiB reserve throughout, and at most 6 GiB process RSS. Preserve every receipt. Never retry a started failed unit.
5. Persist/hash all Train-fitted states and commit the E79 inference freeze before reading Test feature rows.
6. Generate and verify the three CSVs in SampleSubmission order. Do not load Test labels, score Test, or contact Kaggle.
7. Save a single batch manifest/report and make one local scoped commit. Leave all frozen evidence and historical files intact; do not push.

The result status is `PREPARED_FOR_BATCH_REVIEW_NOT_SUBMITTED`. Stop for review after the CSVs, manifest, QA, and local commit are complete, or report a concrete blocker if the batch cannot be completed safely.
