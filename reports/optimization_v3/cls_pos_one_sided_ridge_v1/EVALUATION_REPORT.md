# CLS one-sided positional ridge — fixed-policy evaluation

## Disclosure

This is a repeated development-data evaluation. The origins and query cycles were used in earlier model development, including the mean-one ALIGNED test whose gate failures motivated this adaptive follow-up. Results are not independent confirmation and do not support causal interpretation. The historical access incident remains recorded in the inherited research report.

## Frozen comparison

The one-sided map is exactly `v_j = max(1, w_j)` from the saved prefix-only ALIGNED map, with no renormalization. The GLOBAL_STRENGTH control uses the single scalar `mean(v)` repeated for every coordinate. Saved UNIFORM coefficients and predictions are reused. Both new arms use the same frozen C2_B offset and exact missing-stream HGB fallback.

## Main results

| View | One-sided component Macro F1 gain vs uniform | One-sided vs global component Macro F1 | Competition gain vs same-policy uniform reference |
|---|---:|---:|---:|
| E67_H12 | -0.000044 | -0.000100 | -0.000142 |
| E75_W6 | +0.000264 | +0.000142 | +0.000031 |
| UNIQUE_E67_W4_W5_PLUS_E75_W6 | +0.000081 | -0.000198 | -0.000029 |

- Mechanism gate: **FAIL**.
- Competition gate: **FAIL**.
- The primary competition reference uses pA on missing-stream rows. The archived A02 MIX50 missing fallback used pP and is not the primary comparator here.
- E67 H12 contains E67 W6. E67 W6 and E75 W6 share cycles 76–79; they are history sensitivity, not independent replication.
- The unique pool is E67 W4 + E67 W5 + E75 W6. Unsupported subgroup cells remain UNKNOWN.
- The previous mean-one ALIGNED recipe's negative gates remain counterevidence. This new result, whatever its value, does not update the old verdict.
- No Test data, Kaggle score, or submission was used.

Full and present-only component/competition metrics, confusion matrices, precision/recall, calibration metrics, per-cycle signs, subgroup denominators, all negative findings, and UNKNOWN cells are in EVALUATION_RESULTS.json.
