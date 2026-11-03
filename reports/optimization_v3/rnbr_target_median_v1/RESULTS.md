# RNBR Target Median V1 — Results

**Verdict:** `FAIL_RETIRE_FUNCTIONAL` (diagnostic-only; continuation qualified: no).

## Primary pooled W4 + W6

| Arm | MAE | Gain vs R06 |
|---|---:|---:|
| r06 | 10.4647 | — |
| observed | 10.5419 | -0.0772 |
| target_shuffle | 10.7483 | -0.2836 |
| original_rnbr | 10.3927 | +0.0720 |

Observed vs matched target-shuffle gain: +0.2064 MAE.
Observed vs original RNBR gain: -0.1492 MAE.

## Fixed views

| View | n | R06 MAE | Observed MAE | Shuffle MAE | Original RNBR MAE | Gain vs R06 |
|---|---:|---:|---:|---:|---:|---:|
| W4 | 4950 | 11.3962 | 11.5003 | 11.6684 | 11.5088 | -0.1041 |
| H12_E67 | 20903 | 11.5297 | 11.6241 | 11.7464 | 11.5780 | -0.0944 |
| W6 | 8491 | 9.9216 | 9.9831 | 10.2119 | 9.7420 | -0.0615 |

## Gates

| Gate | Result |
|---|---|
| W4_gain_vs_R06_nonnegative | FAIL |
| W6_gain_vs_R06_nonnegative | FAIL |
| pooled_W4_W6_gain_vs_R06_at_least_0_03 | FAIL |
| H12_E67_gain_vs_R06_nonnegative | FAIL |
| pooled_gain_vs_target_shuffle_at_least_0_01 | PASS |
| H12_gain_vs_target_shuffle_nonnegative | PASS |
| pooled_gain_vs_original_RNBR_at_least_0_03 | FAIL |
| H12_gain_vs_original_RNBR_nonnegative | FAIL |
| POOLED_W4_W6_signal_stream_missing_harm_at_most_0_25 | PASS |
| POOLED_W4_W6_missing_exact_R06 | PASS |
| POOLED_W4_W6_threat_rank_le_5_harm_at_most_0_25 | FAIL |
| POOLED_W4_W6_threat_rank_ge_95_harm_at_most_0_25 | FAIL |
| H12_E67_signal_stream_missing_harm_at_most_0_25 | PASS |
| H12_E67_missing_exact_R06 | PASS |
| H12_E67_threat_rank_le_5_harm_at_most_0_25 | FAIL |
| H12_E67_threat_rank_ge_95_harm_at_most_0_25 | FAIL |

## Protected subgroup guards

### pooled_W4_W6

| Group | n | MAE harm vs R06 | Guard |
|---|---:|---:|---|
| signal_stream_missing | 868 | +0.0000 | PASS |
| threat_rank_le_5 | 686 | +0.8077 | FAIL |
| threat_rank_ge_95 | 683 | +2.3118 | FAIL |
### H12_E67

| Group | n | MAE harm vs R06 | Guard |
|---|---:|---:|---|
| signal_stream_missing | 1375 | +0.0000 | PASS |
| threat_rank_le_5 | 1048 | +0.8665 | FAIL |
| threat_rank_ge_95 | 1057 | +2.4843 | FAIL |

## Audit boundary

Independent verification: `PASS_INDEPENDENT_INTEGRITY_AND_FORMULA`; four memory fits: 4; ledger starts/saves/pending: 87/84/0.
Query Train targets were read only after all four memory bundles passed independent receipt, lineage, shuffle, and formula checks. Test, SampleSubmission, Kaggle, and Classification labels/probabilities were not accessed.

This outcome does not change the original RNBR verdict, and it does not authorize continuation or a Kaggle submission.
