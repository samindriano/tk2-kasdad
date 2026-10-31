# Stage 1 — Paired LightGBM / Hybrid Ridge blend

## Question and design
Do the frozen Hybrid Ridge predictions add complementary information to the LightGBM SVD32 champion? Ridge share is α. Component rank mappings use each model's own existing past-only selected policy; the ensemble uses the predeclared fixed α values 0.25/0.50 or selects α on a strictly earlier inner window.

- Experiment freeze: `6e6e4ead07c62868de0f52017c917581c207b81fd1393d2975d5c10d2f3b40dc`; source commit `14cd13135da8e627c43828e54080ced12c17a62f`.
- Chronology: protocol/initial runner existed before scoring; the checksum receipt was consolidated after the first exploratory result. This is not described as a full pre-score cryptographic freeze.
- Official Train: 50,608 rows, cycles 47–79.
- Evaluation support: 38,588 primary rows and 38,588 H12 rows. H12 overlaps primary.
- Inner α selection at each cutoff uses mapped component predictions for cycles E−3..E generated from fits ending at E−4; outer evaluation uses fits ending at E and validation strictly after E.
- Prediction caches were reused. Component code, Train SHA, model-state and prediction hashes, row positions, past-only boundaries and rank mapping were checked. No Test access or new model fit.
- All validation cycles were used in earlier adaptive research; results are development evidence, not an untouched estimate.

## Results

| Candidate | Primary MAE | Equal-cycle MAE | H12 pooled | H12_E | H12_L | Better primary windows vs LGB |
|---|---:|---:|---:|---:|---:|---:|
| lgb_text_inner_v1 | 11.931600 | 12.283548 | 14.704041 | 17.473257 | 12.361143 | 0/6 |
| ridge_25 | 11.800756 | 12.128720 | 14.252094 | 17.043582 | 11.890354 | 4/6 |
| ridge_50 | 11.875517 | 12.175750 | 13.969345 | 16.711342 | 11.649476 | 4/6 |
| hybrid_ridge_inner_v1 | 12.551305 | 12.790073 | 13.938115 | 16.395455 | 11.859081 | 2/6 |
| inner_selected | 12.042797 | 12.360919 | 14.354853 | 16.711342 | 12.361143 | 3/6 |

## Trade-off detail

| Window | LightGBM | Ridge 25% | Δ vs LGB | Ridge 50% | Δ vs LGB |
|---|---:|---:|---:|---:|---:|
| W1 | 16.405781 | 16.011308 | -0.394473 | 15.709446 | -0.696335 |
| W2 | 12.042870 | 12.151934 | +0.109064 | 12.444067 | +0.401197 |
| W3 | 12.354210 | 12.409597 | +0.055387 | 12.713471 | +0.359261 |
| W4 | 11.999389 | 11.666102 | -0.333288 | 11.536954 | -0.462435 |
| W5 | 10.774911 | 10.595422 | -0.179489 | 10.663344 | -0.111568 |
| W6 | 10.303425 | 10.122384 | -0.181041 | 10.147301 | -0.156124 |
| H12_E | 17.473257 | 17.043582 | -0.429675 | 16.711342 | -0.761915 |
| H12_L | 12.361143 | 11.890354 | -0.470790 | 11.649476 | -0.711667 |

Subgroup MAE on pooled primary support:

| Subgroup | n | LightGBM | Ridge 25% | Δ | Ridge 50% | Δ |
|---|---:|---:|---:|---:|---:|---:|
| EF0 | 26,048 | 12.979047 | 12.871217 | -0.107830 | 12.989562 | +0.010515 |
| EF1 | 12,540 | 9.755852 | 9.577202 | -0.178650 | 9.561433 | -0.194419 |
| tail_low_le5 | 1,928 | 12.235391 | 11.821239 | -0.414152 | 11.445612 | -0.789779 |
| tail_high_ge95 | 1,955 | 9.495441 | 9.379233 | -0.116208 | 9.325531 | -0.169910 |
| stream_missing | 2,578 | 13.041813 | 13.130718 | +0.088905 | 13.388225 | +0.346412 |
| stream_present | 36,010 | 11.852119 | 11.705542 | -0.146576 | 11.767221 | -0.084898 |

Inner-selected Ridge shares by cutoff 55/59/63/67/71/75: 0.5, 1.0, 0.0, 0.0, 0.5, 0.25. This adaptive policy is included as a sensitivity check and is less stable than the fixed 25% blend.


## Error complementarity

On the 38,588 paired primary rows, signed-error correlation was 0.9133. Ridge had lower absolute error on 53.6% of rows; LightGBM on 46.1%; ties 0.3%. This shows potential for complementarity but does not establish that a blend generalizes.

## Interpretation limits

Inspect per-window/per-cycle paired results and EF, tail, and stream subgroups in `stage1_paired_blend.json`. Do not promote based only on pooled MAE. Public Kaggle scores were not used in selection. Stage 2 model-family work awaits MAIN's decision.

Figures: `stage1_pooled_mae.png` compares pooled primary/H12; `stage1_window_delta.png` compares each temporal window with LightGBM. Negative delta favors blending.
