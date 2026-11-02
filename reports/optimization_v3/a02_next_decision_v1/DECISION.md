# Sol next decision: test residual centering, not reliability gating

**Decision:** retain A02, reject a confidence-compatible-neighbor training card
for now, and select one zero-fit target-centering ablation (B01). This is a
research-only task; no fit, Test access, submission, or cap increase is approved.

## What the new controls show

K is derived solely from frozen predictions in each eligible (origin, cycle)
group. Missing streams and W1 fallback remain fixed. A02 and the comparator
therefore have exactly the same predicted-positive count in each group.

| View | Rows | Original A02 gain | Rank contrast at A02 count | Rank contrast at Sparse25 count |
|---|---:|---:|---:|---:|
| Full primary | 38,588 | +0.002101 | +0.001473 | +0.001647 |
| New origins | 21,033 | +0.002016 | +0.001726 | +0.002036 |
| Late | 15,953 | +0.001168 | +0.000601 | +0.000302 |
| Inherited H12-L | 20,903 | +0.002198 | +0.001753 | +0.002088 |

At A02's count, its full-primary decisions include 26 more true positives than
Sparse25's ranked decisions. The contrast is positive at four of five learned
origins; E71 remains negative. Origin-level matching, declared as sensitivity,
also supports a positive full-primary contrast (+0.001699). Late evidence is
weaker: under origin matching at Sparse25's count it is slightly negative
(-0.000151). There is no robust claim of universal late ranking superiority.

The two algebraic paths reproduce the original total gain exactly, but their
components depend on the chosen reference count and grouping. Do not interpret
the ratios as a unique or causal percentage of improvement. These controls do
not determine an optimal deployment count or authorize top-K submission.

## A rejected idea

The next descriptor check assessed whether useful flips have neighbors with
historical baseline probabilities closer to the query baseline. On pooled
first-four-window flips, useful cases instead have a *larger* median weighted
absolute probability gap at four of five origins. Within fixed boundary bands
the signs are mixed and counts often small. This does not support preferring
confidence-compatible neighbors; nor does it justify reversing the weight to
prefer incompatible neighbors. No weighting policy was tested or selected.

Together with the prior unsupported reliability selector, this is a reason to
avoid another fitted reliability gate. It is not proof all such models fail.

## Selected next research question

A02 transfers `y_bank - p_bank_OOF` through text similarity. It remains unclear
whether subtracting each historical model's probability is essential, versus
using a shrunk local label rate. B01 changes that one reference term and retains
the same neighbors, squared-cosine geometry, shrinkage, correction cap, fallback,
and decision threshold. It is not a confidence/threshold/hyperparameter grid.

This is a mechanistic ablation, not a promise of score improvement. It can be
computed from frozen states without fitting a new index, predictor, or
preprocessor. It must compare against the stronger current A02, not only
Sparse25. If it fails the fixed diagnostic criteria, close the ablation without
an inverse-weight or smoothing rescue.

## Verification and scope

The matched-count code verified official Train SHA, exact row/cycle/target joins,
binary/count identities, unchanged fallback, and unchanged fit ledger. Its
confusion-count implementation passed three tests including a sklearn reference,
ties/K=0/K=n, and target-free stratified allocation. The confidence-gap check read
model receipts/hashes and past bank cycles; it never called fit or predict and
created no new candidate probabilities. A first ad-hoc descriptor helper was
stopped because repeatedly indexing a compressed archive caused unnecessary
decompression; the persisted module loads arrays once and completed normally.
No fitted state or budget was affected.

Reproduce from repository root:

```powershell
python -B -m pytest -q tests/test_optimization_v3_a02_next_decision.py
python -B -m scripts.optimization_v3_a02_next_decision
python -B -m scripts.optimization_v3_a02_confidence_gap
```

Matched-count definitions were fixed before their results. The confidence-gap
question was added after inspecting those results and before computing the gap
contrasts. All development labels have been used repeatedly. H12-L overlaps
primary and is not independent evidence. The independent Luna read-only review
found no equivalent matched-count A02/Sparse25 comparison in its six targeted
files, but that is not an exhaustive historical proof of novelty.

Budget remains 66/66 starts consumed. No Test/SampleSubmission/Kaggle access,
new model fit, existing prediction edit, or submission occurred.
