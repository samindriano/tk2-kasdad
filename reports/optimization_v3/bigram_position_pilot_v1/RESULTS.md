# CLS_BIGRAM_POSITION_PILOT_V1 — hasil pilot

**Status gate:** FAIL (engineering criteria; diagnostic-only)

Pertanyaan: apakah posisi bigram yang dikenal memberi tambahan prediksi dibandingkan bag bigram dan shuffled-position, di atas logit C2B + POS unigram yang dibekukan?

## Hasil per temporal view

Macro F1 ditampilkan untuk MIX50 incumbent, BAG, REAL, dan SHUFFLED. Semua precision/recall/confusion/ROC-AUC dan denominator tersedia di `RESULTS.json`.

| Origin | View | n | MIX50 | BAG | REAL | SHUFFLED | REAL − MIX50 | REAL − BAG | REAL − SHUFFLED |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| E67 | W4 | 4950 | 0.846365 | 0.845271 | 0.845404 | 0.844274 | -0.000961 | +0.000133 | +0.001130 |
| E67 | W5 | 7462 | 0.806465 | 0.805976 | 0.808451 | 0.805360 | +0.001986 | +0.002475 | +0.003091 |
| E67 | W6 | 8491 | 0.792531 | 0.789332 | 0.791876 | 0.790121 | -0.000655 | +0.002544 | +0.001755 |
| E67 | H12_E67 | 20903 | 0.811937 | 0.810225 | 0.812143 | 0.810053 | +0.000206 | +0.001918 | +0.002090 |
| E75 | W6 | 8491 | 0.816099 | 0.815826 | 0.814617 | 0.815011 | -0.001482 | -0.001209 | -0.000394 |

## Keputusan gate

| Gate | Nilai | Batas | Hasil |
|---|---:|---:|---|
| E67_W4_REAL_harm_vs_MIX50_max_0.003 | -0.000961 | -0.003000 | PASS |
| E67_W4_availability_present_F1_harm_max_0.01 | +0.001037 | +0.010000 | PASS |
| E67_W4_availability_present_positive_recall_harm_max_0.02 | -0.000584 | +0.020000 | PASS |
| E67_W4_availability_missing_F1_harm_max_0.01 | +0.000000 | +0.010000 | PASS |
| E67_W4_availability_missing_positive_recall_harm_max_0.02 | +0.000000 | +0.020000 | PASS |
| E67_W4_entry_friction_0_F1_harm_max_0.01 | -0.001288 | +0.010000 | PASS |
| E67_W4_entry_friction_0_positive_recall_harm_max_0.02 | -0.001657 | +0.020000 | PASS |
| E67_W4_entry_friction_1_F1_harm_max_0.01 | +0.005735 | +0.010000 | PASS |
| E67_W4_entry_friction_1_positive_recall_harm_max_0.02 | +0.001548 | +0.020000 | PASS |
| E67_W4_text_load_Q1_F1_harm_max_0.01 | +0.000000 | +0.010000 | PASS |
| E67_W4_text_load_Q1_positive_recall_harm_max_0.02 | +0.000000 | +0.020000 | PASS |
| E67_W4_text_load_Q2_F1_harm_max_0.01 | +0.012117 | +0.010000 | FAIL |
| E67_W4_text_load_Q2_positive_recall_harm_max_0.02 | +0.011834 | +0.020000 | PASS |
| E67_W4_text_load_Q3_F1_harm_max_0.01 | +0.001946 | +0.010000 | PASS |
| E67_W4_text_load_Q3_positive_recall_harm_max_0.02 | +0.000000 | +0.020000 | PASS |
| E67_W4_text_load_Q4_F1_harm_max_0.01 | -0.004955 | +0.010000 | PASS |
| E67_W4_text_load_Q4_positive_recall_harm_max_0.02 | -0.007353 | +0.020000 | PASS |
| E67_W5_REAL_harm_vs_MIX50_max_0.003 | +0.001986 | -0.003000 | PASS |
| E67_W5_availability_present_F1_harm_max_0.01 | -0.002157 | +0.010000 | PASS |
| E67_W5_availability_present_positive_recall_harm_max_0.02 | -0.004095 | +0.020000 | PASS |
| E67_W5_availability_missing_F1_harm_max_0.01 | +0.000000 | +0.010000 | PASS |
| E67_W5_availability_missing_positive_recall_harm_max_0.02 | +0.000000 | +0.020000 | PASS |
| E67_W5_entry_friction_0_F1_harm_max_0.01 | -0.002096 | +0.010000 | PASS |
| E67_W5_entry_friction_0_positive_recall_harm_max_0.02 | -0.002677 | +0.020000 | PASS |
| E67_W5_entry_friction_1_F1_harm_max_0.01 | -0.001709 | +0.010000 | PASS |
| E67_W5_entry_friction_1_positive_recall_harm_max_0.02 | -0.005682 | +0.020000 | PASS |
| E67_W5_text_load_Q1_F1_harm_max_0.01 | -0.002562 | +0.010000 | PASS |
| E67_W5_text_load_Q1_positive_recall_harm_max_0.02 | -0.010811 | +0.020000 | PASS |
| E67_W5_text_load_Q2_F1_harm_max_0.01 | -0.003789 | +0.010000 | PASS |
| E67_W5_text_load_Q2_positive_recall_harm_max_0.02 | -0.003419 | +0.020000 | PASS |
| E67_W5_text_load_Q3_F1_harm_max_0.01 | -0.003807 | +0.010000 | PASS |
| E67_W5_text_load_Q3_positive_recall_harm_max_0.02 | -0.007812 | +0.020000 | PASS |
| E67_W5_text_load_Q4_F1_harm_max_0.01 | +0.001828 | +0.010000 | PASS |
| E67_W5_text_load_Q4_positive_recall_harm_max_0.02 | +0.006015 | +0.020000 | PASS |
| E67_W6_REAL_harm_vs_MIX50_max_0.003 | -0.000655 | -0.003000 | PASS |
| E67_W6_availability_present_F1_harm_max_0.01 | +0.000703 | +0.010000 | PASS |
| E67_W6_availability_present_positive_recall_harm_max_0.02 | -0.000429 | +0.020000 | PASS |
| E67_W6_availability_missing_F1_harm_max_0.01 | +0.000000 | +0.010000 | PASS |
| E67_W6_availability_missing_positive_recall_harm_max_0.02 | +0.000000 | +0.020000 | PASS |
| E67_W6_entry_friction_0_F1_harm_max_0.01 | +0.000747 | +0.010000 | PASS |
| E67_W6_entry_friction_0_positive_recall_harm_max_0.02 | +0.001406 | +0.020000 | PASS |
| E67_W6_entry_friction_1_F1_harm_max_0.01 | +0.000841 | +0.010000 | PASS |
| E67_W6_entry_friction_1_positive_recall_harm_max_0.02 | -0.002783 | +0.020000 | PASS |
| E67_W6_text_load_Q1_F1_harm_max_0.01 | -0.002586 | +0.010000 | PASS |
| E67_W6_text_load_Q1_positive_recall_harm_max_0.02 | -0.006345 | +0.020000 | PASS |
| E67_W6_text_load_Q2_F1_harm_max_0.01 | +0.003405 | +0.010000 | PASS |
| E67_W6_text_load_Q2_positive_recall_harm_max_0.02 | +0.001712 | +0.020000 | PASS |
| E67_W6_text_load_Q3_F1_harm_max_0.01 | -0.001457 | +0.010000 | PASS |
| E67_W6_text_load_Q3_positive_recall_harm_max_0.02 | -0.002299 | +0.020000 | PASS |
| E67_W6_text_load_Q4_F1_harm_max_0.01 | +0.003631 | +0.010000 | PASS |
| E67_W6_text_load_Q4_positive_recall_harm_max_0.02 | +0.005772 | +0.020000 | PASS |
| E67_H12_E67_REAL_gain_vs_mix50 | +0.000206 | +0.000500 | FAIL |
| E67_H12_E67_REAL_gain_vs_bag | +0.001918 | +0.000000 | PASS |
| E67_H12_E67_REAL_gain_vs_shuffled | +0.002090 | +0.000000 | PASS |
| E67_H12_E67_availability_present_F1_harm_max_0.01 | -0.000225 | +0.010000 | PASS |
| E67_H12_E67_availability_present_positive_recall_harm_max_0.02 | -0.001763 | +0.020000 | PASS |
| E67_H12_E67_availability_missing_F1_harm_max_0.01 | +0.000000 | +0.010000 | PASS |
| E67_H12_E67_availability_missing_positive_recall_harm_max_0.02 | +0.000000 | +0.020000 | PASS |
| E67_H12_E67_entry_friction_0_F1_harm_max_0.01 | -0.000829 | +0.010000 | PASS |
| E67_H12_E67_entry_friction_0_positive_recall_harm_max_0.02 | -0.000970 | +0.020000 | PASS |
| E67_H12_E67_entry_friction_1_F1_harm_max_0.01 | +0.001123 | +0.010000 | PASS |
| E67_H12_E67_entry_friction_1_positive_recall_harm_max_0.02 | -0.002688 | +0.020000 | PASS |
| E67_H12_E67_text_load_Q1_F1_harm_max_0.01 | -0.001942 | +0.010000 | PASS |
| E67_H12_E67_text_load_Q1_positive_recall_harm_max_0.02 | -0.006369 | +0.020000 | PASS |
| E67_H12_E67_text_load_Q2_F1_harm_max_0.01 | +0.002652 | +0.010000 | PASS |
| E67_H12_E67_text_load_Q2_positive_recall_harm_max_0.02 | +0.001991 | +0.020000 | PASS |
| E67_H12_E67_text_load_Q3_F1_harm_max_0.01 | -0.001390 | +0.010000 | PASS |
| E67_H12_E67_text_load_Q3_positive_recall_harm_max_0.02 | -0.003506 | +0.020000 | PASS |
| E67_H12_E67_text_load_Q4_F1_harm_max_0.01 | +0.000181 | +0.010000 | PASS |
| E67_H12_E67_text_load_Q4_positive_recall_harm_max_0.02 | +0.001472 | +0.020000 | PASS |
| E75_W6_REAL_gain_vs_mix50 | -0.001482 | +0.000500 | FAIL |
| E75_W6_REAL_gain_vs_bag | -0.001209 | +0.000000 | FAIL |
| E75_W6_REAL_gain_vs_shuffled | -0.000394 | +0.000000 | FAIL |
| E75_W6_availability_present_F1_harm_max_0.01 | +0.001585 | +0.010000 | PASS |
| E75_W6_availability_present_positive_recall_harm_max_0.02 | +0.001288 | +0.020000 | PASS |
| E75_W6_availability_missing_F1_harm_max_0.01 | +0.000000 | +0.010000 | PASS |
| E75_W6_availability_missing_positive_recall_harm_max_0.02 | +0.000000 | +0.020000 | PASS |
| E75_W6_entry_friction_0_F1_harm_max_0.01 | +0.000785 | +0.010000 | PASS |
| E75_W6_entry_friction_0_positive_recall_harm_max_0.02 | +0.000000 | +0.020000 | PASS |
| E75_W6_entry_friction_1_F1_harm_max_0.01 | +0.002644 | +0.010000 | PASS |
| E75_W6_entry_friction_1_positive_recall_harm_max_0.02 | +0.002783 | +0.020000 | PASS |
| E75_W6_text_load_Q1_F1_harm_max_0.01 | +0.005769 | +0.010000 | PASS |
| E75_W6_text_load_Q1_positive_recall_harm_max_0.02 | +0.011524 | +0.020000 | PASS |
| E75_W6_text_load_Q2_F1_harm_max_0.01 | +0.004129 | +0.010000 | PASS |
| E75_W6_text_load_Q2_positive_recall_harm_max_0.02 | +0.003273 | +0.020000 | PASS |
| E75_W6_text_load_Q3_F1_harm_max_0.01 | -0.003559 | +0.010000 | PASS |
| E75_W6_text_load_Q3_positive_recall_harm_max_0.02 | -0.011287 | +0.020000 | PASS |
| E75_W6_text_load_Q4_F1_harm_max_0.01 | -0.003085 | +0.010000 | PASS |
| E75_W6_text_load_Q4_positive_recall_harm_max_0.02 | -0.004511 | +0.020000 | PASS |

## Interpretasi dan batas

REAL dibandingkan dengan BAG dan SHUFFLED untuk membedakan nilai posisi dari frekuensi pasangan dan dari urutan acak yang mempertahankan bag serta ukuran bin. Hasil E67/E75 adalah development evidence dengan jendela yang tumpang tindih. Gate bukan uji signifikansi.

Semua subgroup yang tidak memenuhi dukungan n≥200 dan masing-masing kelas≥30 dilaporkan UNKNOWN. Tidak ada klaim kausal atau semantik. Tidak ada Test, SampleSubmission, atau Kaggle access.

## Visual

![primary_comparison](reports/optimization_v3/bigram_position_pilot_v1/primary_comparison.png)

![temporal_deltas](reports/optimization_v3/bigram_position_pilot_v1/temporal_deltas.png)
