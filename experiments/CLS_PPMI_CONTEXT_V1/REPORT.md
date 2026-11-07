# CLS_PPMI_CONTEXT_V1 pilot results

Status: **NO_ADVANCE**

The pilot used four prefix-specific PPMI/SVD representations and four fixed HGB predictor fits. Evaluation labels were opened only after the immutable prelabel commit.

## Main windows

| View | n | Champion F1 | Real candidate F1 | Shuffled control F1 | Candidate gain | Candidate minus control |
|---|---:|---:|---:|---:|---:|---:|
| E67_W4 cycles 68-71 | 4950 | 0.804871 | 0.805897 | 0.801743 | +0.001025 | +0.004154 |
| E67_W5 cycles 68-72 | 6761 | 0.793535 | 0.794621 | 0.791616 | +0.001086 | +0.003005 |
| E67_W6 cycles 68-73 | 8641 | 0.780163 | 0.779072 | 0.776960 | -0.001091 | +0.002112 |
| E67_H12 cycles 68-79 | 20903 | 0.744227 | 0.743437 | 0.737994 | -0.000790 | +0.005444 |
| E75_W6 cycles 76-79 | 8491 | 0.739983 | 0.734703 | 0.735288 | -0.005281 | -0.000585 |
| PRIMARY_UNIQUE cycles 68-79 | 20903 | 0.756290 | 0.753277 | 0.752313 | -0.003013 | +0.000963 |

## Gates

- Scientific gate: **FAIL**.
- Exploratory gate: **NOT_ELIGIBLE**.
- Mechanism comparison: **NOT_SEPARATED_FROM_SHUFFLED_CONTROL**.
- Worst supported subgroup F1 harm: 0.011153368555739673.
- Worst supported subgroup positive recall harm: 0.026368159203980168.

## Readout

The fitted real-order PPMI/SVD+HGB policy was compared with the full frozen champion and a same-population, same-estimator within-document shuffled-context control.

The real-versus-shuffled comparison changes the learned context geometry and SVD axes as well as local order. It is evidence about the full representation pipeline, not an isolated causal order effect.

## Boundary and next step

- Official source SHA-256: `875552a68a3d24b0f3e1966d6ddb5671e6f54305a1c526e2f0a76747942ed91c`.
- Query rows scored: 20903 unique rows; target column read once after the committed prelabel.
- Recommendation: Close this PPMI-context family; root should select one distinct, source-only novelty scan from the untested CLS mechanisms before any new fit.
