# R2 source freeze superseded

The immutable R2 freeze at
`experiments/reg_r06_native_hinge_v1/PREFIT_FREEZE_R2.json` is preserved with
SHA-256 `ca9d8b34eb0c09ee3013629195d79a08da3b825e849ce14176ddd4cac74aff2a`.
It is superseded because the pinned runner referenced an undefined
`coeffs` name while constructing non-target identity metadata, so the R2
preflight did not complete. The exception occurred after non-target cached row
identity was loaded, but before target values were loaded, any R2 preflight
receipt was written, or any official LP started. R2 must not receive a source
pass or be used for a diagnostic fit.

R2B is the corrected generation. It uses separate freeze, preflight, receipt,
run, source-pass, and final-approval paths; the failed R2 freeze remains
unchanged.
