# Learned pair pilot V1 — frozen execution protocol

## Authorization and scope

This is the one four-predictor Classification mechanism pilot authorized by Sam through Sol on 2026-11-03. It increases the Research Next budget from 109 to 113 starts and pilot reservations from 68 to 72. Other categories remain continuation 38, control replay 2, and technical retry 1. The authorized fit IDs are `LEARNED_PAIR_PILOT_V1_E67_FM`, `LEARNED_PAIR_PILOT_V1_E67_NOPAIR`, `LEARNED_PAIR_PILOT_V1_E75_FM`, and `LEARNED_PAIR_PILOT_V1_E75_NOPAIR`. No preprocessing fit, retry, quota reset, or other predictor fit is authorized.

The result is adaptive development evidence for mechanism review. A gate pass does not select a champion, authorize full-six qualification, continuation, full-Train refit, Test inference, or upload.

## Data and recipe

- Source: official Classification Train only, 50,608 records, cycles 47–79, source SHA-256 `875552a68a3d24b0f3e1966d6ddb5671e6f54305a1c526e2f0a76747942ed91c`.
- Origins: saved C2_B `fit_base` and `score_base` matrices for E67 and E75 from the verified `POSITIONAL_TEXT_PILOT_V1` lineage. The fit populations and matrix hashes are recorded in `FREEZE.json` after commit. No vocabulary, IDF, scaling, OHE, category, or other preprocessing state is refit.
- Score: `Xw + b + 0.5 * sum_f((QV)^2 - Q^2(V^2))`, where `Q` is the row-L2-normalized copy of unchanged `X`. Rank 16.
- Objective: SUM binary logistic loss using stable signed-label `logaddexp`, plus `(||w||^2 + ||V||_F^2 + b^2)/(2C)`, `C=1`. Accumulate raw data loss/gradient over 1,024-row chunks, including the uneven final chunk, then apply L2 once.
- Initialization: `w=0`, `b=0`; candidate factors use `default_rng(42).normal(0, 0.01, (d,16))`, reset for each origin. The no-pair control has only `w,b` with the same zero initialization.
- Solver: SciPy L-BFGS-B, float64, `maxiter=1000`, `maxfun=2000`, `maxcor=10`, `maxls=50`, `ftol=1e-9`, `gtol=1e-4`; two BLAS threads and one job. No optimizer, seed, penalty, or solver rescue.
- Routing: present stream uses the direct learned expert probability; missing/blank stream uses the exact saved C1_B HGB fallback. Threshold 0.50. No offset, A02 residual, probability blend, calibration, or changed row boundary.

## Comparisons and views

The frozen C2_B bag route is the primary cached baseline. Its source package retains both `prob_bag` (raw expert output) and `prob_route_bag` (the availability-routed baseline). Verify the latter exactly equals raw C2_B on present-stream rows and the saved C1_B HGB fallback on missing/blank rows; gate comparisons use this routed baseline, and the raw array is retained separately for lineage. The learned FM and no-pair experts use the same exact HGB fallback on missing/blank rows. Cached C16 Sparse25 and A02 observed probabilities are aligned by exact score row, record ID, and cycle and reported as separately labeled references only; they are never fit inputs and do not define gates. Their bundle hashes are frozen with the preflight. A missing optional reference is recorded as UNKNOWN.

E67 contains W4, W5, W6, and H12_E67. E75 contains W6. E75 W6 overlaps the E67 W6 rows and is history sensitivity, not independent replication. Full-six primary is not computed.

Gate thresholds are frozen as follows:

- E67 H12_E67: FM Macro F1 gain at least 0.0005 vs cached C2_B bag and at least 0.0002 vs no-pair.
- E75 W6: FM gain nonnegative vs both.
- E67 W4/W5/W6: FM harm no worse than 0.005 vs either comparator.
- Every supported availability, fit-defined text-load, and entry-friction subgroup: Macro F1 harm no worse than 0.01 and positive-recall harm no worse than 0.02 vs either comparator, with support `n >= 200` and at least 30 examples of each class. Unsupported groups are reported explicitly and do not receive a harm estimate.
- All four fits converge; matrix, row, receipt, budget, exact missing fallback, and independent score/gradient checks pass.

An unsupported subgroup is recorded as `UNKNOWN_UNSUPPORTED` with its observed
denominator and class counts; its harm gate is not estimated and it does not fail
the overall numeric gate. Every evaluable numeric gate must still pass. Threshold
comparisons use a `1e-12` floating-point tolerance so values exactly on a frozen
boundary are handled consistently.

## Optimizer and failure handling

Success requires SciPy success, a finite final parameter/loss/gradient state, and final objective no greater than initial objective plus `1e-8`. Each receipt reports `nit`, `nfev`, status/message, initial/final SUM loss, SUM and mean gradient infinity norms, factor movement/norm, pair-score contribution, and process/system RAM samples. RAM samples are not an OS peak RSS counter.

A nonconverged or interrupted fit consumes its started ID and is terminal. The durable failure receipt is retained, unstarted IDs are released append-only, and no later fit or retry is run. Such a result is inconclusive about the general usefulness of learned interactions.

## Query-label barrier and reproducibility

Fit-prefix Classification labels are accessed only for the exact frozen fit rows. Query/outer `artifact_signal` values stay closed until all four fit receipts and raw score outputs are durable, row/source hashes pass, and the separate verifier reproduces raw scores, missing routes, fit logits, SUM objective, and gradients without fitting. `QUERY_LABEL_ACCESS.json` is an immutable once-created authorization/intention receipt; `QUERY_LABEL_READ.json` confirms the first completed query-label read, including its input hash, row count, columns, and access-receipt hash. Both preserve their initial timestamps on resume. No Regression target, Test, SampleSubmission, or Kaggle path is opened.

Before official fitting, run the temporary end-to-end harness. It uses an explicit new directory outside the repository, performs an actual synthetic optimizer call, writes synthetic parameters/predictions/receipt/ledger, exercises the production resume helper without refitting, runs the separate verifier, rejects a tampered matrix, checks production ledger guards, and confirms the live budget hash is unchanged in `finally`.

From the repository root:

```powershell
python -m pytest -q tests/test_learned_pair_pilot_v1.py
python scripts/optimization_v3_learned_pair_pilot_v1.py synthetic --temp-root "$env:TEMP\kasdad-learned-pair-synthetic"
python scripts/optimization_v3_learned_pair_pilot_v1.py preflight
python scripts/optimization_v3_learned_pair_pilot_v1.py freeze
python scripts/optimization_v3_learned_pair_pilot_v1.py reserve
```

Then run `fit-next` once per fit, in the exact order above. Do not start another invocation while the prior process is running. After all four `FIT_SAVED` receipts, run `predictions`, `verify-prelabel`, and `score` in that order. The execution freeze pins the clean committed source HEAD, input, matrices, preprocessing state, cached fallbacks/references, solver, gates, and pre-reservation budget hash.
