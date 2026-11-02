# R-NBR readiness — Regression neighbor transfer

## Decision

**GO for a frozen diagnostic protocol; NO-GO for fits under the current allowance.** The first-four-cycle R06 OOF predictions and reusable, target-independent neighbor geometry are now verified across all usable origins. The Research Next ledger is already at **66/66 starts** (64 saved, 2 failed, 0 pending), so the proposed 10 Regression target-memory fits cannot start without a separate budget decision. No fit, target join, candidate prediction, metric replay, Test/Sample access, or Kaggle call was made here.

The R06 candidate itself remains an experimental gate exception. Its existing guard failure is the supported W1 missing-stream group (harm 0.395556 versus the 0.25 limit). The proposed text-neighbor correction leaves missing-stream predictions unchanged, so even a positive pooled MAE result cannot repair that inherited group failure or qualify R06 under its original gate. Treat a future run as **diagnostic-only** unless a separately reviewed protocol addresses that gate without weakening it.

## Data and source boundaries

- Official Regression Train: 50,608 rows, SHA-256 `756523ff8156a9cd776e52249f7e2a8f23dcec432b8d10570e55ff6db0e0f082`.
- Official Classification Train: SHA-256 `875552a68a3d24b0f3e1966d6ddb5671e6f54305a1c526e2f0a76747942ed91c`.
- The 18 shared fields excluding `artifact_signal` and `threat_rank` match row-for-row across all 50,608 rows. The audit loaded no target column from either Train. It did not open Test or SampleSubmission.
- R06 recipe `experiments/optimization_v3/batch1/R06/recipe_revision_02.json` has SHA-256 `6814978b3228fb741429dbd6b8af6d2c6caa8044eff0e079bbeb50066643b788`, source commit `be5e86750ec725e45698054a91fad299ddc24eb4`, and the current source file matches the source-commit blob (`37869d6125d48fc4afb2a3c439cbb19d5586f35f1cd3a501b762e42530fcd970`). It is the fixed 13-feature char TF-IDF + structured/OHE + missing-indicator LightGBM `regression_l1` recipe with the original per-cycle rank50 policy.
- Every R06 cache/sidecar pair passes NPZ SHA, canonical identity, sidecar integrity, and external freeze-signature checks. Score rows and IDs are unique and match Regression Train at their source positions. Prediction-array headers align with the saved rows. No cached prediction values were opened or rescored.

## OOF windows and donor support

| Fit cutoff | Saved query cycles | First four cycles used by this design | Rows by cycle | Present past-only donors by cutoff |
|---:|---:|---|---|---:|
| 55 | 56–67 | 56–59 | 564, 962, 1,205, 1,383 | 0 |
| 59 | 60–63 | 60–63 | 1,478, 1,632, 1,500, 1,883 | 3,722 |
| 63 | 64–67 | 64–67 | 1,807, 1,567, 1,751, 1,953 | 9,846 |
| 67 | 68–79 | 68–71 | 577, 1,079, 1,605, 1,689 | 16,482 |
| 71 | 72–75 | 72–75 | 1,811, 1,880, 1,814, 1,957 | 21,090 |
| 75 | 76–79 | 76–79 | 1,822, 2,335, 2,146, 2,188 | 28,045 |

The donor count is the number of present-stream rows in cycles 56 through the query cutoff. Each donor's R06 prediction comes from a strictly earlier scoring cutoff: 56–59 from fit55, 60–63 from fit59, 64–67 from fit63, 68–71 from fit67, 72–75 from fit71, and 76–79 from fit75. Cutoff55 has no earlier R06 OOF residual bank; the proposed method therefore falls back to the unchanged R06 prediction for its 56–59 query window.

## Saved geometry is reusable

R06's own six NPZ bundles contain predictions, IDs, rows, cycles, and audit/control vectors, but no neighbor geometry. However, the classification A02 experiment has persisted its top-32 text-neighbor row positions, IDs, and cosine similarities for origins 59, 63, 67, 71, and 75. These are sufficient for the Regression donor windows:

- all five origin-specific observed/shuffled geometry arrays match exactly;
- their query IDs and cycles match Regression Train and the corresponding R06 cache rows;
- every selected donor is present-stream, lies in cycles 56 through the cutoff, and has an exact strictly-past R06 OOF prediction;
- each donor ID agrees with the stored row position; cosine weights are finite and ordered;
- freeze, model, state, prediction bundle, and fit-receipt hashes verify.

The current script loaded only A02 row IDs, row positions, cycles, neighbor counts, neighbor IDs, and cosine arrays. It did **not** load the NPZ `labels`, class predictions, corrections, or residual arrays; it did not unpickle A02 models. Classification and Regression feature identity is used only to justify this target-free geometry reuse. No Classification target is joined to Regression.

The A02 geometry uses the established word TF-IDF recipe (`ngram_range=(1,2)`, `min_df=3`, `max_df=0.995`, `max_features=20,000`, sublinear TF, lowercase, Unicode accent stripping, L2 norm) fit on the target-free prefix; it returns at most 32 positive-cosine neighbors. The separate geometry arrays are reused only as row/similarity evidence. They do not reuse A02 class residuals or predictions.

## Why a future diagnostic remains uncertain

R06's existing full-six result is mixed: primary MAE 11.502413 versus champion 11.559390, H12 MAE 13.944273 versus 14.047533, but W6 is worse than champion by 0.013929 and R06 fails the missing-stream W1 guard. This is evidence that R06 can help aggregate MAE while retaining a material protected-group weakness.

There is direct counterevidence against assuming residual transfer will work. The prior shallow Regression residual head worsened W4 by 0.158966, H12 late by 0.149310, pooled W1+W4 by 0.086814, and pooled H12 by 0.080881. Reused OOF diagnostics also show mean signed residual changing from +1.333 in H12_E to −1.296 in H12_L; low targets are overpredicted and high targets underpredicted. These are repeated development observations, not independent validation.

## Proposed future comparison

See [R-NBR proposed card](R-NBR_PROPOSED_CARD.md). In brief: use the frozen R06 final prediction as the base; estimate a local **weighted median** of strictly past R06 OOF residuals from the saved top-32 neighbors; compare with the same geometry under a within-cycle × R06-prediction-decile residual shuffle. Missing or unsupported queries retain R06 exactly. Weighted median matches the MAE objective; weighted mean transport instead targets a conditional mean and is not the candidate in this card.

## Accounting and stop point

The live Research Next ledger SHA-256 is `6fcc4b4d427e3ff03139e0e9ee429f529f7c7bf6ad91ce0eb9a52d55e2503147`: cap 66, starts 66, saves 64, failures 2, pending 0, remaining 0. A complete corrected-versus-shuffled comparison at cutoffs 59, 63, 67, 71, and 75 requires **10 new task fits** (one Regression target-memory predictor per origin and arm). The cached geometry adds no new fit, but constructing each Regression target memory counts as a fit under the A02 convention even without gradient optimization. No reservation or fit was attempted.

The separate positional-text reserve is design-only; its scoped code/report search found no explicit absolute-position experiment. See [positional text design](POSITIONAL_TEXT_DESIGN.md).

Reproduce the read-only inventory with:

```powershell
python scripts/optimization_v3_rnbr_preflight.py
```

It writes [INVENTORY.json](INVENTORY.json) and refuses mismatched input, source, freeze, cache, geometry, alignment, or ledger evidence.
