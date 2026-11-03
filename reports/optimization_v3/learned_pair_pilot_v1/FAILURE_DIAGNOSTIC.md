# Learned-pair pilot — failure diagnostic

**Status: `FAIL_OPTIMIZER_INCONCLUSIVE`.** The pilot stopped after its first official predictor did not meet the frozen optimizer convergence rule. No query/outer target was read, no prediction or predictive gate was evaluated, and no champion decision follows from this run.

## Frozen scope and input integrity

- Frozen source commit: `5758c1402daaada936a2a8196ddab7f3c15358eb`.
- Freeze signature: `d92c6cfe57b2562a629e562efdd38edac5951bc93e0f51c85061740b75561c1a`.
- Official Classification Train: 50,608 rows; SHA-256 `875552a68a3d24b0f3e1966d6ddb5671e6f54305a1c526e2f0a76747942ed91c`.
- Recipe SHA-256: `4e48277d9f80b633689120a4d96cc564a337831d5ea8e5ae1cb4f0c1d301916f`.
- Sol authorization SHA-256: `f09818090e0ff9df871e722aadbb7accf433f7e83b72208fd7d9f649a11e4bb7`.
- The final read-only preflight passed for the saved E67/E75 matrices, row IDs/cycles, fit-population lineage, HGB fallback, cached references, source hashes, resource guard, and 109-fit ledger checkpoint.

Preflight also clarified the cached baseline binding. The saved `prob_bag` is the raw C2_B expert output; `prob_route_bag` is the historical availability-routed baseline. Both origins verified exactly as raw C2_B on present-stream rows and the saved C1_B HGB fallback on missing/blank rows. All gates use the routed baseline. The code and synthetic check reject a raw/routed swap; no historical prediction was rewritten.

## Fit outcome

Only `LEARNED_PAIR_PILOT_V1_E67_FM` started. It used the authorized 29,705-row E67 fit population, saved C2_B `fit_base`, rank 16, factor seed 42, SUM logistic objective, `C=1`, and frozen L-BFGS-B settings. The optimizer stopped at the iteration limit:

| Quantity | Result |
|---|---:|
| Status | `STOP: TOTAL NO. OF ITERATIONS REACHED LIMIT` |
| `nit` / `nfev` | 1,000 / 1,044 |
| Initial objective (SUM) | 20,606.106149 |
| Final objective (SUM) | 6,677.625305 |
| Final gradient infinity norm (SUM) | 1.345563 |
| Final gradient infinity norm (mean scale) | 0.00004530 |
| Factor movement norm | 74.091632 |
| Elapsed time | 367.8 seconds |
| Peak process RSS sample | 357,326,848 bytes |
| Minimum available RAM sample | 12,607,963,136 bytes |

The mean-scale gradient is below `gtol`, but the frozen optimizer minimizes the SUM objective and requires SciPy success. SciPy returned `success=false`, status code 1, so this fit is a terminal nonconvergence. Its objective reduction does not qualify it as converged and is not predictive evidence.

An independent, non-fitting calculation on the exact E67 fit-prefix reproduced all 29,705 saved fit logits exactly, final objective `6,677.625305454964`, and SUM gradient infinity norm `1.3455633344948739`. It verified the saved failure receipt and parameter artifact; it did not score outer rows.

## Budget and stop accounting

| Checkpoint | Global cap | Active reserved | Started | Saved | Failed | Pending |
|---|---:|---:|---:|---:|---:|---:|
| Before amendment | 109 | 109 | 109 | 106 | 3 | 0 |
| After four-ID reservation | 113 | 113 | 109 | 106 | 3 | 0 |
| Terminal failure checkpoint | 113 | 110 | 110 | 106 | 4 | 0 |

The three unstarted IDs—E67 no-pair, E75 FM, and E75 no-pair—were released by one append-only `RESERVATION_UNUSED_RELEASED` event. The started E67 FM ID was consumed once and not retried. The final ledger contains 281 events and has SHA-256 `125446878a69bdecf2d658daf259622e568147642a072d2aaf2d8e21a8f47008`. The pre-amendment ledger SHA-256 was `e6bf5051d4f45c3654ae6ce21f31437f30662db03e9c26e63f1051b2c7b4dc2b`.

## Interpretation and boundary

This outcome is inconclusive about the predictive value of learned pair interactions: the planned matched control and second origin did not run, and the outer metrics remain unevaluated. The model is not qualified, no champion changed, and no continuation or upload is authorized. A future optimizer change or additional fit needs a separate review and authorization; this checkpoint itself starts nothing further.

Machine-readable receipt: `FAILURE_DIAGNOSTIC.json`. Frozen ledger-bound completion receipt: `experiments/runs/optimization_v3_research_next_v1/LEARNED_PAIR_PILOT_V1/RESULTS.json` (SHA-256 `5feab4a3f039e16c3657fda40883a314d321a1696bd89b84f1a880da690c1b05`).
