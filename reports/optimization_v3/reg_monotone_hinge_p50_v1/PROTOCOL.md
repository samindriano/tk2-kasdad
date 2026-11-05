# REG fixed-knot p50 monotone hinge

## Scope and stage gate

This lane tests one continuous two-segment calibration of the existing raw DIRECT
prediction. The knot is fixed in advance at 50. The source and synthetic QA stage
does not open Train.csv, materialize labels, open Test.csv or SampleSubmission,
run an official LP, write a candidate CSV, or upload to Kaggle. The previous R3
`ROOT_SOURCE_PASS.json` applies to its old fit IDs only and does not authorize this
lane. Official label access and the first diagnostic LP require a new source pass
bound to this namespace, committed sources, frozen inputs, and this exact protocol.

## Estimator

\[
g(p)=a+s_L\min(p,50)+s_H\max(p-50,0),
\qquad 0\leq s_L\leq2,\quad0\leq s_H\leq2.
\]

Fit the three free parameters by minimizing the **unweighted sum of absolute
errors** over the same OOF population as the causal GLOBAL control, including
missing rows. `p` and `y` must be finite and in `[0,100]`. The free intercept is
unbounded. Use one sparse HiGHS LP start, two threads, and a fixed 180 second time
limit. No retry, weights, seed, knot search, or slope grid is allowed. Clipping to
`[0,100]` happens only when producing application predictions. The bound of 2 is
a disclosed design choice, not a discovered optimum.

## Causal stages and fit accounting

| Fit ID | Fit source rows | Score rows | Authorized rows |
| --- | --- | --- | ---: |
| `HINGE_D1` | E67, cycles 68–71 | E67, cycles 72–75 | 4,950 / 7,462 |
| `HINGE_D2` | E67, cycles 68–75 | E75, cycles 76–79 | 12,412 / 8,491 |
| `HINGE_FINAL` | all unique OOF E67/E75 rows, cycles 68–79 | none | 20,903 |

The pooled diagnostic evaluation contains 15,953 unique rows across D1 and D2.
Use the frozen R3 row-index stitch and exact record IDs, prediction origins,
bundle indices, and cycles. Reuse the saved `D1_GLOBAL` and `D2_GLOBAL` states as
the causal comparators, plus raw DIRECT context. Do not use the full-support
GLOBAL coefficients as independent validation; they are descriptive only. The
D2 prediction-origin transition from E67 to E75 remains a limitation.

Two diagnostic LP starts are planned. `HINGE_FINAL` is conditional on Sol review
of both diagnostic results and is the third and final possible calibrator start.
Each fit ID has one start and one immutable publication. A failed start consumes
its slot. A published state is identity-checked and recovered after interruption;
it is never refit to repair a later reporting failure. New base-model fits and
learned preprocessing fits are both zero.

Before any LP start, require at least 1,000 rows and 200 rows on each side of 50;
the final fit requires at least 5,000 rows and 200 on each side. Report score-side
support. Unsupported stages stop before an LP start.

## Application and reporting boundary

After separate authorization of final output, transform present E79 rows as
`clip(g(raw_DIRECT),0,100)`. Keep every missing E79 value bit-exact raw DIRECT.
The CSV must contain the 29,421 frozen official IDs in their exact order, all
finite predictions in `[0,100]`, and exactly 1,062 missing fallbacks. This source
stage creates no official candidate CSV.

Diagnostic reporting must include all-row and present-only MAE and denominators,
coefficients, bound activity, error mean, clipping, per-cycle signs, and supported
subgroups for missing/present and `y<=5`/`y>=95`; a subgroup below 200 rows is
`UNKNOWN`. Compare with the causal GLOBAL present-only control. Reconstruct fit
objectives, constraints, metrics, prediction hashes, and CSV routes independently
before promotion. Training-loss improvement alone does not establish
generalization.

Promotion requires pooled gain at least 0.02 versus causal GLOBAL present-only,
nonnegative gains in both windows, and supported subgroup harm no greater than
0.25. Otherwise only the defined exploratory root-review gate may continue:
pooled gain at least zero, or one window gain at least 0.02 with pooled harm at
most 0.01; each window harm at most 0.03; and supported subgroup harm at most
0.25. No automatic nearby-knot or slope rescue follows a failed lane. Public
scores do not select parameters.

## Reproduction and resource accounting

From the repository root, run:

```powershell
python -m unittest tests.test_reg_monotone_hinge_p50_v1 -v
python scripts/reg_monotone_hinge_p50_v1.py --synthetic-qa
python scripts/reg_monotone_hinge_p50_v1.py --freeze-identity
```

The synthetic suite uses one small known-function LP and temporary synthetic
files only. The identity freeze hashes the committed source, R3 source/input
receipts, causal GLOBAL states, and diagnostic row bundles by bytes; it does not
load NPZ members or parse Train.csv. Each official LP is capped at 180 seconds
with two threads. Actual official compute remains unknown until the source pass
and support checks permit a fit.
