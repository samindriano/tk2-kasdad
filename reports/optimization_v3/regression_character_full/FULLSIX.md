# V3 Regression Character Ridge — Full Six Windows

This continues only the frozen 25% character-Ridge rank50 component blended with 75% of the current three-component champion. The current champion is reconstructed from the official V2 row-aligned caches by the V3 diagnostic `_build_view` helper. Four new fits use cutoffs 59, 63, 71, and 75; cutoff 55 and 67 predictions are reused from the independently verified pilot.

No parameters, rank rule, blend weight, or clipping rule changed. No official Test data or Kaggle service was read. These temporal windows have already been used during adaptive development, so results are comparative development evidence, not an untouched holdout estimate.

## Six-window primary results

| Window | n | Champion MAE | Character blend MAE | MAE gain | Equal-cycle gain | Bias shift | Candidate p95 AE |
|---|---:|---:|---:|---:|---:|---:|---:|
| W1 | 4,114 | 15.8821 | 15.6854 | +0.1967 | +0.1894 | -0.169 | 40.150 |
| W2 | 6,493 | 11.9085 | 12.2320 | -0.3235 | -0.3252 | -0.111 | 34.861 |
| W3 | 7,078 | 12.0582 | 12.4508 | -0.3925 | -0.3850 | -0.047 | 34.723 |
| W4 | 4,950 | 11.4550 | 11.4377 | +0.0173 | -0.0256 | -0.185 | 31.891 |
| W5 | 7,462 | 10.3479 | 10.5111 | -0.1632 | -0.1639 | -0.147 | 29.736 |
| W6 | 8,491 | 9.9077 | 10.0148 | -0.1071 | -0.1048 | -0.081 | 28.153 |

## Pooled and H12 views

| View | n | Champion MAE | Character blend MAE | MAE gain | Equal-cycle gain | Bias shift |
|---|---:|---:|---:|---:|---:|---:|
| Primary six-window pooled | 38,588 | 11.5594 | 11.7178 | -0.1584 | -0.1359 | -0.115 |
| H12 fixed-origin pooled | 38,588 | 14.0475 | 13.9175 | +0.1301 | +0.1236 | -0.207 |
| H12 early (56–67) | 17,685 | 16.9266 | 16.7050 | +0.2217 | +0.2153 | -0.131 |
| H12 late (68–79) | 20,903 | 11.6117 | 11.5591 | +0.0526 | +0.0320 | -0.272 |

## Guardrails and subgroup coverage

- Primary MAE gain: **-0.1584**.
- H12 pooled MAE gain: **+0.1301**.
- Supported subgroup harm limit: **0.25 MAE** across **365** comparisons with n≥100; max harm was **+0.8371**; violations: **57**.
- Subgroups include fit-defined impact_load, scope_pressure, text_load, and scope_trace quantiles; EF0/EF1; stream present/missing; target tails; and raw category novelty. Target tails are descriptive after scoring only.
- Full per-cycle, group, bias, and support tables are in `fullsix.json`.

## Fit-defined character volume

| Cutoff | Fit rows | Validation rows | Char vocabulary | Fit source |
|---:|---:|---:|---:|---|
| 55 | 12,020 | 17,685 | 30,000 | root_pilot_reused |
| 59 | 16,134 | 6,493 | 30,000 | new_fit |
| 63 | 22,627 | 7,078 | 30,000 | new_fit |
| 67 | 29,705 | 20,903 | 30,000 | root_pilot_reused |
| 71 | 34,655 | 7,462 | 30,000 | new_fit |
| 75 | 42,117 | 8,491 | 30,000 | new_fit |

Reproduce the four new fits with `python scripts/optimization_v3_character_fullsix.py --run` after reviewing the frozen identity. Independently verify saved predictions with `python scripts/optimization_v3_verify_character_fullsix.py`.
