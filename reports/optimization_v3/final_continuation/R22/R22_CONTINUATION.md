# R22 continuation result

- Status: `RESULT_VERIFIED`
- Freeze signature: `2ab551f7d29c920cda46c61b4a5793a6a0256bf582ee3f4619eff1640bd3a953`
- Selected policy: **R22_ORIGINAL**
- Decision: PCT_PRESENT did not satisfy every predeclared selection condition; ORIGINAL qualifies.
- New predictor fits: 4 (cutoffs 59, 63, 71, 75); pilot 55/67 reused.
- Evaluation labels are previously reused temporal Train development data; these are not untouched holdouts.
- Primary and H12 have the same 38,588 Train row identities scored from different fit origins; they are not independent samples.

## Metric summary

MAE gains are champion MAE minus candidate MAE; positive values mean improvement. Gates use unrounded values.

| Policy | View | n | Candidate MAE | Champion MAE | Gain |
|---|---:|---:|---:|---:|---:|
| R22_ORIGINAL | primary | 38588 | 11.519691 | 11.559390 | 0.039699 |
| R22_ORIGINAL | H12 | 38588 | 13.922659 | 14.047533 | 0.124873 |
| R22_ORIGINAL | W1 | 4114 | 15.743656 | 15.882112 | 0.138456 |
| R22_ORIGINAL | W2 | 6493 | 11.833929 | 11.908486 | 0.074557 |
| R22_ORIGINAL | W3 | 7078 | 12.084101 | 12.058230 | -0.025871 |
| R22_ORIGINAL | W4 | 4950 | 11.390307 | 11.455005 | 0.064698 |
| R22_ORIGINAL | W5 | 7462 | 10.333223 | 10.347945 | 0.014722 |
| R22_ORIGINAL | W6 | 8491 | 9.880454 | 9.907683 | 0.027230 |
| R22_ORIGINAL | W5+W6 | 15953 | 10.092236 | 10.113615 | 0.021379 |
| R22_PCT_PRESENT | primary | 38588 | 11.512574 | 11.559390 | 0.046816 |
| R22_PCT_PRESENT | H12 | 38588 | 13.914622 | 14.047533 | 0.132910 |
| R22_PCT_PRESENT | W1 | 4114 | 15.749099 | 15.882112 | 0.133013 |
| R22_PCT_PRESENT | W2 | 6493 | 11.835840 | 11.908486 | 0.072646 |
| R22_PCT_PRESENT | W3 | 7078 | 12.066757 | 12.058230 | -0.008527 |
| R22_PCT_PRESENT | W4 | 4950 | 11.412922 | 11.455005 | 0.042083 |
| R22_PCT_PRESENT | W5 | 7462 | 10.294670 | 10.347945 | 0.053275 |
| R22_PCT_PRESENT | W6 | 8491 | 9.879165 | 9.907683 | 0.028518 |
| R22_PCT_PRESENT | W5+W6 | 15953 | 10.073517 | 10.113615 | 0.040098 |

## Qualification and group guards

### R22_ORIGINAL: QUALIFIED
- PASS — primary_gain_at_least_0_03
- PASS — H12_harm_at_most_0_05
- PASS — at_least_four_of_six_windows_nonnegative
- PASS — W5_W6_gain_positive
- PASS — all_supported_group_harm_at_most_0_25

| View | Group | n | Harm (candidate − champion MAE) | Status |
|---|---|---:|---:|---|
| primary | signal_stream_missing | 2578 | 0.053031 | PASS |
| primary | threat_rank_le_5 | 1928 | -0.287247 | PASS |
| primary | threat_rank_ge_95 | 1955 | -0.006376 | PASS |
| H12 | signal_stream_missing | 2578 | 0.026304 | PASS |
| H12 | threat_rank_le_5 | 1928 | -0.678663 | PASS |
| H12 | threat_rank_ge_95 | 1955 | 0.007586 | PASS |
| W1 | signal_stream_missing | 392 | -0.087142 | PASS |
| W1 | threat_rank_le_5 | 205 | -0.922117 | PASS |
| W1 | threat_rank_ge_95 | 213 | -0.314295 | PASS |
| W2 | signal_stream_missing | 369 | 0.079779 | PASS |
| W2 | threat_rank_le_5 | 322 | -0.512436 | PASS |
| W2 | threat_rank_ge_95 | 325 | -0.074333 | PASS |
| W3 | signal_stream_missing | 442 | 0.163251 | PASS |
| W3 | threat_rank_le_5 | 353 | -0.167001 | PASS |
| W3 | threat_rank_ge_95 | 360 | 0.006875 | PASS |
| W4 | signal_stream_missing | 342 | -0.050552 | PASS |
| W4 | threat_rank_le_5 | 244 | -0.270879 | PASS |
| W4 | threat_rank_ge_95 | 249 | 0.171021 | PASS |
| W5 | signal_stream_missing | 507 | 0.058494 | PASS |
| W5 | threat_rank_le_5 | 362 | -0.217922 | PASS |
| W5 | threat_rank_ge_95 | 374 | 0.053251 | PASS |
| W6 | signal_stream_missing | 526 | 0.108195 | PASS |
| W6 | threat_rank_le_5 | 442 | 0.009410 | PASS |
| W6 | threat_rank_ge_95 | 434 | 0.031479 | PASS |

### R22_PCT_PRESENT: QUALIFIED
- PASS — primary_gain_at_least_0_03
- PASS — H12_harm_at_most_0_05
- PASS — at_least_four_of_six_windows_nonnegative
- PASS — W5_W6_gain_positive
- PASS — all_supported_group_harm_at_most_0_25

| View | Group | n | Harm (candidate − champion MAE) | Status |
|---|---|---:|---:|---|
| primary | signal_stream_missing | 2578 | 0.000000 | PASS |
| primary | threat_rank_le_5 | 1928 | -0.678737 | PASS |
| primary | threat_rank_ge_95 | 1955 | -0.425191 | PASS |
| H12 | signal_stream_missing | 2578 | 0.000000 | PASS |
| H12 | threat_rank_le_5 | 1928 | -1.123385 | PASS |
| H12 | threat_rank_ge_95 | 1955 | -0.453143 | PASS |
| W1 | signal_stream_missing | 392 | 0.000000 | PASS |
| W1 | threat_rank_le_5 | 205 | -1.850911 | PASS |
| W1 | threat_rank_ge_95 | 213 | -0.559259 | PASS |
| W2 | signal_stream_missing | 369 | 0.000000 | PASS |
| W2 | threat_rank_le_5 | 322 | -1.018458 | PASS |
| W2 | threat_rank_ge_95 | 325 | -0.684226 | PASS |
| W3 | signal_stream_missing | 442 | 0.000000 | PASS |
| W3 | threat_rank_le_5 | 353 | -0.537021 | PASS |
| W3 | threat_rank_ge_95 | 360 | -0.615445 | PASS |
| W4 | signal_stream_missing | 342 | 0.000000 | PASS |
| W4 | threat_rank_le_5 | 244 | -0.533284 | PASS |
| W4 | threat_rank_ge_95 | 249 | -0.177233 | PASS |
| W5 | signal_stream_missing | 507 | 0.000000 | PASS |
| W5 | threat_rank_le_5 | 362 | -0.432427 | PASS |
| W5 | threat_rank_ge_95 | 374 | -0.360823 | PASS |
| W6 | signal_stream_missing | 526 | 0.000000 | PASS |
| W6 | threat_rank_le_5 | 442 | -0.282796 | PASS |
| W6 | threat_rank_ge_95 | 434 | -0.205332 | PASS |

## Fit receipts

| Cutoff | Fit rows | Score rows | Seconds | RSS after fit (bytes) | Bundle SHA-256 |
|---:|---:|---:|---:|---:|---|
| 59 | 16134 | 6493 | 7.675 | 457519104 | `3682bdb816f510adf3cb53bef81ba9545111e872ca277ced1f6da9a52826f600` |
| 63 | 22627 | 7078 | 8.317 | 471416832 | `fe3374619abcb514fa6d99b5187603897d33a70549bbe2c03e882456c8506b83` |
| 71 | 34655 | 7462 | 12.339 | 484753408 | `5b5e6684a5afaaabf70afe6b253eb4f732bcdbcc7bdeb311a9f31561aa87d69b` |
| 75 | 42117 | 8491 | 15.924 | 491630592 | `91981eb993becda4d6f2cad33fabdb8348fdbc8891c04af90e13e6b58e492a36` |

## Diagnostics and limits

H12 early/late and per-cycle Spearman/MAE results are stored in `result.json` as descriptive diagnostics only; they do not change the predeclared gates.
The percentile policy ranks raw predictions across every record in each complete cycle, including missing-stream records; only the output blend falls back to the exact champion for missing streams.
No Test, SampleSubmission, or Kaggle data were accessed. No full-Train refit or submission is part of this continuation.
