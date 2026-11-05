# CLS POS Frequency Ridge — fixed-policy evaluation R1

## Disclosure

This is a repeated development-data evaluation, not a fresh or untouched holdout. Before this evaluator was frozen, a prior inventory materialized E67_y and E75_y to print only names, shapes, and dtypes. The access-order deviation is recorded in EVALUATION_ACCESS_INCIDENT_01.json; pre-exposure cannot be undone. No metric or fit was made from that inventory. These results must not be described as outcome-blind.

## Frozen comparison

The component comparison uses ALIGNED, PERMUTED, and original uniform POS probabilities with the same cached HGB missing-stream fallback. The competition comparison uses 0.5 * new POS + 0.5 * same-origin pA on present-stream rows and pA on missing-stream rows. Primary missing classification threshold is 0.50. The 0.53230976129477 missing threshold is reported only as a post-selected operational description.

## Competition Macro F1 gains

| View | Gain vs original MIX50 with pA missing |
|---|---:|
| E67_H12 | -0.001540 |
| E75_W6 | -0.002677 |
| UNIQUE_E67_W4_W5_PLUS_E75_W6 | -0.002105 |

## Decision gates

- Mechanism gate: **FAIL**.
- Competition gate: **FAIL**.
- The unique pool is E67 W4 + E67 W5 + E75 W6. E67 W6 and E75 W6 overlap and are history sensitivity, not independent replications.
- Unsupported subgroup cells remain UNKNOWN; the JSON has denominators and supported-group effects.
- No model fit, Test read, or Kaggle action was performed.

Full confusion matrices, precision/recall, calibration metrics, present-only results, per-cycle directions, subgroup supports, and gate checks are in EVALUATION_RESULTS_R1.json.
