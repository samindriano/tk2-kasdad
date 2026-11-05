# R2B source package superseded

The R2B freeze
`experiments/reg_r06_native_hinge_v1/PREFIT_FREEZE_R2B.json` (SHA-256
`0dc45a5edb2de96169056d83f064be842bd42f7612b29d3560402caafea11cd1`), its
non-target preflight, synthetic QA, and direct verifier receipt are preserved.
The direct R2B verifier passed with zero fit starts. However, the documented
`verify-source-ready` command in the frozen R2B runner imported the R1 verifier
and returned the pre-existing R1 verification receipt instead of dispatching
to R2B. R2B must not receive a root source pass or be used for a diagnostic
fit.

R2C fixes and tests that CLI dispatch, uses separate immutable artifact paths,
and is the current source package. No R2B target values or official LP starts
were recorded.
