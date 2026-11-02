# N04_DIAGNOSTIC sign-label correction

This note corrects Markdown labels and interpretation only. The frozen source files, raw JSON metrics, fit receipts, verifier, ledger, and historical verdict are preserved unchanged. N04 remains `SKIPPED_SCIENTIFIC_PREFLIGHT`; N04_DIAGNOSTIC remains `DIAGNOSTIC_ONLY`.

The subgroup table in `SUMMARY.md` previously labeled values as “Gain vs R22”, although it displayed `harm_vs_R22`. The correct formula is:

- `harm_vs_R22 = candidate MAE - R22 MAE`; positive means worse than R22.
- `gain_vs_R22 = R22 MAE - candidate MAE`; positive means better than R22.
- `gain_vs_shuffle = shuffled-weight MAE - candidate MAE`; positive means better than the shuffled control.
- `harm_vs_legacy = candidate MAE - legacy MAE`; positive means worse than the legacy model.

Correct examples from the retained JSON:

- Pooled H12 missing-stream candidate MAE `14.389503` versus R22 `14.351436`: harm `+0.038067`, not a gain.
- Pooled H12 target `<=5` candidate MAE `14.763003`: harm vs R22 `+0.029075`, equivalent gain `-0.029075`.
- Pooled H12 target `>=95`: harm vs R22 `-0.005277`, equivalent gain `+0.005277`.

No measurements or conclusions beyond the sign labels are changed.
