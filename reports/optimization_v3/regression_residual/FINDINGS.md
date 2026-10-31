# Regression residual pilot — retire

The fixed shallow correction head was trained once at cutoff67, on actual incumbent OOF predictions and labels from cycles56–67. The oldest cutoff55 remains an unchanged fallback because no exact current-champion OOF history exists before56 without importing outcome-selected historical rank weights.

| Development view | Incumbent MAE | Corrected MAE | Decision |
|---|---:|---:|---|
| W4 | 11.455005 | 11.613971 | Worse |
| H12 late | 11.611686 | 11.760996 | Worse |
| W1+W4 pilot pooled | 13.464396 | 13.551210 | Worse |
| H12 pooled | 14.047533 | 14.128414 | Worse |

No further candidate fits, Test inference or upload are justified. The observed global and tail biases do not imply that a correction estimated on earlier OOF residuals transfers to later cycles.

The original verifier source is preserved. Use the amendment-bound **V5** verifier: `python scripts/optimization_v3_verify_reg_residual_v5.py`. Its receipt verifies official Train/source/input hashes, exact OOF block alignment, fit/data/score/target fingerprints, unchanged fallback vectors, correction bounds, independent metrics and the failed continuation gate. Earlier verifier versions remain as failure evidence and are not the active reproduction path. The amendments correct verification errors only; no model outputs or candidate parameters changed and V5 performed zero refits.

These known Train cycles are adaptive development evidence. Full numerical detail remains in `pilot.json` and `verification.json`.
