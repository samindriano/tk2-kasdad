# POS_CONTRAST REAL E79 — exploratory benchmark

**Status:** one full-Train theta fit completed; local Test-feature CSV passed independent structural and policy QA. This candidate remains exploratory. The historical `POS_CONTRAST_V1` pilot verdict remains `PILOT_GATE_FAILED`.

## Frozen recipe

The one new estimator is the REAL three-position contrast vector `theta`, fit on all 50,608 official Classification Train rows. Its fixed offset is the original C2_B intercept plus the original C2_B bag matrix multiplied by the saved coefficients. The contrast is `Z1=(first-last)/sqrt(2)` and `Z2=(first-2*middle+last)/sqrt(6)`, appended to the unchanged base and left unnormalized after projection. The optimizer is the original sum-logistic-loss plus L2/2 L-BFGS-B recipe, initialized at zero, with `maxiter=2000`, `ftol=1e-12`, `gtol=1e-6`, and `maxls=50`.

Present-stream probability is `0.5 * sigmoid(C2_B logit + Z @ theta) + 0.5 * original A02 probability`. Missing-stream probability is the exact saved full-Train C1_B HGB probability. Labels use `probability >= 0.50`.

## Evidence

- Train rows/cycles: 50608 / 47–79; Train SHA-256 `875552a68a3d24b0f3e1966d6ddb5671e6f54305a1c526e2f0a76747942ed91c`.
- New official fits: 1; prerequisite refits: 0; optimizer: `CONVERGENCE: RELATIVE REDUCTION OF F <= FACTR*EPSMCH` after 37 iterations.
- Test CSV: `submissions/optimization_v3/contrast_real_e79_exploratory_v1/classification_pos_contrast_real_e79_exploratory.csv`; SHA-256 `36b25918b2029f36ee5744ef3da3d6a23e6c1026ebafa568d6ecd25ae77a0033`; 29,421 rows in official SampleSubmission order.
- Independent QA: `PASS_INDEPENDENT_CSV_AND_POLICY_QA`; gamma-zero replay `PASS_BIT_EXACT`; exact missing fallback `True`.
- This lane did not evaluate Test labels or upload to Kaggle. No predictive metric or leaderboard result is claimed here.

## Limits

The frozen historical pilot failed its temporal gates. This full-Train fit is an exploratory benchmark on a different population and fixed E79 route; it does not retroactively pass those gates, qualify as a champion, or provide a Test metric. Kaggle upload remains the separate owner-controlled step.
