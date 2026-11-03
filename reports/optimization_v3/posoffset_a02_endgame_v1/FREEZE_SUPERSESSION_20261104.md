# Pre-metric freeze supersession — 2026-11-04

The first implementation source and freeze checkpoints are preserved at
commits `761453dc47e61586de9c74ec2445dfe1bb25b512` and
`9a6bc5c646174510c2913eb354d86788fc07d25d`. The first attempt to run the
post-freeze A02 source audit stopped before writing an audit receipt or opening
any policy metrics. Its bank verifier incorrectly required all historical OOF
rows to be present-stream rows. The frozen A02 protocol retains the complete
OOF row universe for alignment, then filters missing-stream rows before fitting
the lexical residual model.

The verifier has been corrected to validate both counts and identities:

- E67 OOF bank: 17,685 aligned OOF rows; 16,482 present-stream donor rows.
- E75 OOF bank: 30,097 aligned OOF rows; 28,045 present-stream donor rows.

This was a verifier implementation defect, not a source artifact discrepancy.
No model, prediction, recipe, gate, input, fit ledger, or historical verdict
changed. `PREFLIGHT_SUCCESSOR_01.json` and `FREEZE_SUCCESSOR_01.json` form a new
source-bound chain. The first `PREFLIGHT.json` and `FREEZE.json` remain intact
and are not used by the successor audit. Policy metrics remain unopened until
the successor freeze is committed and its complete original-artifact audit
passes.
