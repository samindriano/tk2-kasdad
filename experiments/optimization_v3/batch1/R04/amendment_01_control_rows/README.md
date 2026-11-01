# R04 amendment 01 — control row identity

This revision supersedes the parent R04 `recipe.json` and `freeze.json` for any new pilot. Keep those original files unchanged as the first frozen protocol attempt.

The first official `--run-frozen-pilot` attempt reserved the declared two predictor fits and started cutoff 55. It failed in the control bridge before the R04 estimator's `fit` method because `control_rows` was not passed to the adapter. The run ledger records one `FIT_FAILED`; no candidate model was fitted, no prediction bundle was saved, and no candidate score was produced. The original R04 reservation remains in the shared budget ledger and is reused; do not add a second two-fit reservation.

The amendment adds the exact row positions from the frozen control bundle as `control_rows`, with a unit test for the bridge and the existing end-to-end synthetic test. Re-run preflight, syntax/tests, and freeze this revision before another official fit. Preserve the original attempt in the ignored run ledger.
