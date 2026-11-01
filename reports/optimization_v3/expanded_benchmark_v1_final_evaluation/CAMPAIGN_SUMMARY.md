# V3 expanded benchmark recovery evaluation

Parent freeze: `c6a751f2ea3b23ac8d2b8d47c8a71f2d87283ff66f749cb6cd819e2c1ac9d78f`
Recovery freeze: `6cdecf6e0729b625981f23c7b7be4152a12806011a0a1852dd978eaec30d21ae`

| Candidate | Gate status | Primary | H12 | W5+W6 | Verification |
|---|---|---:|---:|---:|---|
| C17 | EXPERIMENTAL_GATE_EXCEPTION | -0.000545 | 0.001411 | -0.002063 | PASS |
| C11 | EXPERIMENTAL_GATE_EXCEPTION | -0.000479 | -0.000310 | -0.000622 | PASS |
| R21 | EXPERIMENTAL_GATE_EXCEPTION | -0.006106 | 0.044280 | -0.026890 | PASS |
| R06 | EXPERIMENTAL_GATE_EXCEPTION | 0.056977 | 0.103260 | 0.023589 | PASS |

R11 is blocked before metric evaluation; its performance gate is unknown. Performance failures on the other candidates remain eligible for the authorized experimental submission exception. No Test rows, public scores, or new estimator fits were used in this evaluation.
