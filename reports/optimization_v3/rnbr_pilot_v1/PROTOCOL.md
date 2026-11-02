# R-NBR Late Two-Origin Pilot — B0 Protocol

## Status and question

This is a preparation protocol for a **four-fit diagnostic pilot**. No official
task fit, official target-memory construction, candidate prediction, or new
metric evaluation has run under this protocol. The current verdict is
`READY_FOR_FOUR_FIT_BUDGET_REVIEW`, conditional on the synthetic QA and
target-free preflight recorded alongside this file. It is not a qualification
or champion-selection protocol.

Question: **Do R06 out-of-fold regression residuals from structurally similar
past reports provide a small, robust correction to the frozen R06 prediction
on late cycles, beyond the same residuals after within-cell shuffling?**

This pilot is separate from N04 and its AUC amendment. It does not alter the
historical `rnbr_readiness_v1` card or any Classification A02 evidence.

## Frozen candidate and support

For each query row, use the already saved, target-free A02 top-neighbor geometry:
at most 32 positive-cosine neighbors, ordered by descending cosine and then
source row. Similarity weights are cosine squared. The Regression donor bank
contains only `signal_stream`-present rows from cycles 56 through the origin.
Each donor residual is `threat_rank - R06 OOF prediction`, where the R06 OOF
source cutoff is fixed by donor cycle and strictly precedes that cycle.

For a present query with at least five saved neighbors:

```text
median = weighted_median(donor residuals, weights = cosine ** 2)
delta = clip(0.25 * median, -5, +5)
candidate = clip(frozen R06 query prediction + delta, 0, 100)
```

The weighted median uses stable ordering by residual then source row. If
cumulative weight is exactly half between two distinct residuals, it returns
their midpoint. Missing-stream rows, fewer than five neighbors, and zero
neighbors keep the R06 prediction exactly unchanged. A row/ID/cycle/hash,
lineage, or cache integrity failure aborts; it never falls back to R06.
Predictions are not reranked.

| Origin | Present donor bank | R06 OOF source cutoff by donor cycle | Declared query horizon |
|---|---|---|---|
| E67 | cycles 56–67 | 55 for 56–59; 59 for 60–63; 63 for 64–67 | cycles 68–79 (W4, W5, W6) |
| E75 | cycles 56–75 | 55 for 56–59; 59 for 60–63; 63 for 64–67; 67 for 68–71; 71 for 72–75 | W6, cycles 76–79 |

These are the two observed-memory fits. For each origin, the control permutes
the donor residuals within exact `cycle_code × fixed R06 prediction decile`,
using `min(9, floor(clip(prediction,0,100)/10))`, `default_rng(42)` reset per
origin, ascending cycle then decile, and stable ascending source-row order.
This preserves each cell's residual multiset and breaks its local alignment to
the saved neighbor geometry. It uses no target-derived bin edges.

The task-estimator budget is exactly four: E67 observed, E67 shuffled, E75
observed, and E75 shuffled. The E67 observed/shuffled memories predict its
entire saved horizon, cycles 68–79. E75 only predicts cycles 76–79. Scoring all
12 E67 cycles does not require another fit. A residual-memory construction
counts as a task fit. Saved R06/A02 artifacts are reused only after the hash
and identity checks in the preflight; no new text transform or
neighbor-geometry fit is planned.

## Evaluation views and gates

The primary view is W4 + W6. The target-free preflight confirms `n=13,441`:
W4 uses E67 cycles 68–71 and has 4,950 rows (4,608 present, 342 missing); W6
uses E75 cycles 76–79 and has 8,491 rows (7,965 present, 526 missing). Missing
rows remain in evaluation and retain their R06 predictions. In these primary
windows, all 12,573 present query rows have 32 positive-cosine neighbors; zero
present rows are below the five-neighbor minimum. This is a target-free support
result, not evidence that the residual correction improves MAE.
Report R06, observed-memory, shuffled-memory, R22_ORIGINAL, and legacy MAE on
identical rows, plus gains and all negative results. Also report W4 and W6
separately. Compare against the cached R22/legacy values only after verifying
their row alignment and hashes. Do not select a recipe from Kaggle public
scores.

H12_L is the full E67 long-horizon view (`n=20,903`, confirmed by preflight):
all E67 queries in cycles 68–79, with E67 candidate, E67 shuffled-control, and
the E67 R06 baseline on those exact rows. W5 is predicted with the E67 memory
and may receive a correction. E75 predictions, labels, baseline, or donor bank
do not enter H12_L. H12_L overlaps the primary W4 rows and is not an
independent sample. H12_E is not a new fit comparison for this pilot.

The predeclared **incremental diagnostic** thresholds compare the candidate
with R06 and the shuffled control:

- Primary MAE gain versus R06 is at least 0.03.
- Primary MAE gain versus shuffled control is at least 0.01.
- H12_L harm versus R06 is at most 0.05.
- Harm versus R06 is below 0.25 in each of W4 and W6.
- For supported groups (`n >= 100`), harm versus R06 is at most 0.25 for
  missing-stream, `threat_rank <= 5`, and `threat_rank >= 95`.
- Missing-stream incremental harm must be exactly zero because those rows are
  unchanged. Unsupported groups remain reported as unsupported.

These gates ask whether this late diagnostic adds evidence against the cached
R06 predictions. They cannot qualify the original R06 recipe or clear its
historical legacy subgroup failures. All historical R06-versus-legacy guard
failures must be reported separately. A pass is diagnostic-only and cannot
enter a continuation or replace a candidate without a later review.

## Boundaries and failure behavior

- Donor targets are Regression `threat_rank` only. Classification labels and
  A02 prediction/correction arrays are not donor data. Target-free equality of
  the shared 18 Train columns is provenance evidence only; it does not expand
  the R06 feature allowlist.
- Query labels are accessed only after both observed and shuffled predictions
  are frozen and independently verified.
- No Test or SampleSubmission data, Kaggle calls, full-Train refit, or public
  leaderboard selection.
- One model-fitting/training job at a time; two threads, 6 GiB process ceiling
  with 2 GiB reserved. No all-pairs matrix or new TF-IDF fit.
- A missing or corrupt bundle fails closed. Resume reuses a verified bundle;
  it never silently rebuilds a mismatched fit.
- The synthetic lifecycle test exercises save, interruption, resume,
  independent recomputation, and corruption rejection without official rows,
  targets, predictions, or ledger events.
- Synthetic and official view-provenance checks assert E67 cycles 68–79 for
  H12_L, and independently assert E67 W4 + E75 W6 for primary.

## Evidence already available (historical, not pilot results)

The verified full-six R06 report records primary MAE 11.502413 (n=38,588),
H12 MAE 13.944273, W6 harm versus legacy 0.013929, and a supported W1
missing-stream harm versus legacy of 0.395556 (n=392), exceeding the historical
0.25 group guard. R06 therefore remains historically unqualified. These are
unchanged archival results, not outcomes from this pilot.

The cached R22 report records primary MAE 11.519691 and H12 MAE 13.922659
against legacy MAE 11.559390 and 14.047533, respectively. These historical
figures are comparison context only. No R-NBR result exists yet.

## Reproduction

From the repository root:

```powershell
python scripts/optimization_v3_rnbr_pilot_preflight.py
python -m pytest -q tests/test_optimization_v3_rnbr_pilot.py
python scripts/optimization_v3_rnbr_pilot_synthetic.py --run-dir $env:TEMP\rnbr-pilot-b0
```

The first command checks target-free official Train alignment and saved-cache
provenance only. The other commands use synthetic fixtures and a temporary
directory. No command in this protocol starts an official fit.
