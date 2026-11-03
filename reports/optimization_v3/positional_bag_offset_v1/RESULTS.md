# POS_BAG_OFFSET_V1 â€” hasil diagnostic

**Status:** `FAIL_DIAGNOSTIC_ONLY`. Kandidat tetap diagnostic-only; tidak ada perubahan champion atau izin upload.

## Pertanyaan dan batas interpretasi

Eksperimen ini menilai apakah kanal posisi memberi tambahan prediktif ketika koefisien dan intercept C2_B bag dibekukan sebagai offset logit. Gamma saja dipelajari memakai objective logistic SUM dengan penalti L2 dan C=1.
Offset bag berasal dari prediksi in-sample pada prefix. Karena itu ini sequential additive fitting, bukan cross-fitted residual training. Amendment ini dipilih adaptif setelah melihat diagnostic sebelumnya; hasil temporal juga tumpang tindih.

## Metrik utama

Macro F1 memakai threshold 0,50. `bag` adalah C2_B dengan fallback HGB tersimpan; `offset_real` dan `offset_shuffled` memakai gamma masing-masing; `joint_position_reference` hanya referensi deskriptif dari pilot lama.

| Origin | View | Bag | Offset real | Offset shuffled | Joint-position ref. | Real âˆ’ bag | Real âˆ’ shuffled |
|---|---|---:|---:|---:|---:|---:|---:|
| E67 | W4 | 0.842554 | 0.840145 | 0.836163 | 0.838262 | -0.002409 | +0.003982 |
| E67 | W5 | 0.799093 | 0.798799 | 0.798227 | 0.799238 | -0.000294 | +0.000572 |
| E67 | W6 | 0.784403 | 0.788935 | 0.778195 | 0.789240 | +0.004532 | +0.010741 |
| E67 | H12_E67 | 0.805106 | 0.806258 | 0.800608 | 0.805996 | +0.001151 | +0.005650 |
| E75 | W6 | 0.810384 | 0.811443 | 0.808951 | 0.810138 | +0.001059 | +0.002491 |

## Gate checks

| Gate | Nilai | Ambang | Hasil |
|---|---:|---:|---|
| E67 H12_E67_gain_vs_bag_min_0.003 | 0.001151 | 0.003 | FAIL |
| E67 H12_E67_gain_vs_offset_shuffled_min_0.003 | 0.005650 | 0.003 | PASS |
| E67 W4_harm_vs_bag_max_0.005 | -0.002409 | -0.005 | PASS |
| E67 W4_harm_vs_offset_shuffled_max_0.005 | 0.003982 | -0.005 | PASS |
| E67 W5_harm_vs_bag_max_0.005 | -0.000294 | -0.005 | PASS |
| E67 W5_harm_vs_offset_shuffled_max_0.005 | 0.000572 | -0.005 | PASS |
| E67 W6_harm_vs_bag_max_0.005 | 0.004532 | -0.005 | PASS |
| E67 W6_harm_vs_offset_shuffled_max_0.005 | 0.010741 | -0.005 | PASS |
| E67 W4_Q4_support | {"n": 1765, "negative": 1085, "positive": 680} | {'n': 200, 'each_class': 30} | PASS |
| E67 W4_Q4_macro_f1_harm_vs_bag_max_0.01 | 0.003839 | -0.01 | PASS |
| E67 W4_Q4_positive_recall_harm_vs_bag_max_0.02 | 0.014706 | -0.02 | PASS |
| E67 W4_Q4_macro_f1_harm_vs_offset_shuffled_max_0.01 | 0.009332 | -0.01 | PASS |
| E67 W4_Q4_positive_recall_harm_vs_offset_shuffled_max_0.02 | 0.014706 | -0.02 | PASS |
| E67 W5_Q4_support | {"n": 2018, "negative": 1353, "positive": 665} | {'n': 200, 'each_class': 30} | PASS |
| E67 W5_Q4_macro_f1_harm_vs_bag_max_0.01 | -0.005293 | -0.01 | PASS |
| E67 W5_Q4_positive_recall_harm_vs_bag_max_0.02 | 0.006015 | -0.02 | PASS |
| E67 W5_Q4_macro_f1_harm_vs_offset_shuffled_max_0.01 | -0.004468 | -0.01 | PASS |
| E67 W5_Q4_positive_recall_harm_vs_offset_shuffled_max_0.02 | -0.001504 | -0.02 | PASS |
| E67 W6_Q4_support | {"n": 2080, "negative": 1387, "positive": 693} | {'n': 200, 'each_class': 30} | PASS |
| E67 W6_Q4_macro_f1_harm_vs_bag_max_0.01 | -0.006566 | -0.01 | PASS |
| E67 W6_Q4_positive_recall_harm_vs_bag_max_0.02 | -0.001443 | -0.02 | PASS |
| E67 W6_Q4_macro_f1_harm_vs_offset_shuffled_max_0.01 | 0.002316 | -0.01 | PASS |
| E67 W6_Q4_positive_recall_harm_vs_offset_shuffled_max_0.02 | -0.008658 | -0.02 | PASS |
| E67 H12_E67_Q4_support | {"n": 5863, "negative": 3825, "positive": 2038} | {'n': 200, 'each_class': 30} | PASS |
| E67 H12_E67_Q4_macro_f1_harm_vs_bag_max_0.01 | -0.002824 | -0.01 | PASS |
| E67 H12_E67_Q4_positive_recall_harm_vs_bag_max_0.02 | 0.006379 | -0.02 | PASS |
| E67 H12_E67_Q4_macro_f1_harm_vs_offset_shuffled_max_0.01 | 0.002459 | -0.01 | PASS |
| E67 H12_E67_Q4_positive_recall_harm_vs_offset_shuffled_max_0.02 | 0.001472 | -0.02 | PASS |
| E75 W6_gain_vs_bag_min_0 | 0.001059 | 0.0 | PASS |
| E75 W6_gain_vs_offset_shuffled_min_0 | 0.002491 | 0.0 | PASS |
| E75 W6_Q4_support | {"n": 1993, "negative": 1328, "positive": 665} | {'n': 200, 'each_class': 30} | PASS |
| E75 W6_Q4_macro_f1_harm_vs_bag_max_0.01 | -0.005332 | -0.01 | PASS |
| E75 W6_Q4_positive_recall_harm_vs_bag_max_0.02 | 0.003008 | -0.02 | PASS |
| E75 W6_Q4_macro_f1_harm_vs_offset_shuffled_max_0.01 | -0.006132 | -0.01 | PASS |
| E75 W6_Q4_positive_recall_harm_vs_offset_shuffled_max_0.02 | -0.012030 | -0.02 | PASS |

## Siklus dan subgroup

Detail metrik tiap siklus, stream availability, entry_friction, dan fit-defined text_load Q4 tersimpan di `RESULTS.json`. Angka negatif dipertahankan; gate Q4 mensyaratkan nâ‰¥200 dan minimal 30 observasi pada tiap kelas.

- E67: tanda positif per siklus realâˆ’bag 5/12; realâˆ’shuffled 9/12.
- E75: tanda positif per siklus realâˆ’bag 3/4; realâˆ’shuffled 3/4.

## Rekonsiliasi eksekusi dan provenance

- Source commit: `8e8a76cea6d22828cb39909ac904cdd05ef54a47`; freeze: `ba58b1e446a6be8546789c5a56596fbdbf075efc0ee19fcc1f41e02af7b419d7`.
- Input SHA-256: `875552a68a3d24b0f3e1966d6ddb5671e6f54305a1c526e2f0a76747942ed91c`. Empat fit resmi, tanpa retry; semua empat optimizer sukses.

| Fit | Optimizer | Iterasi | Loss γ=0 | Loss akhir | Receipt SHA-256 |
|---|---|---:|---:|---:|---|
| E67_REAL | PASS | 38 | 7824.738 | 7111.615 | `e95919b0bcf5` |
| E67_SHUFFLED | PASS | 30 | 7824.738 | 7212.348 | `5ae7d257bf66` |
| E75_REAL | PASS | 46 | 11297.983 | 10380.574 | `1cf8e8507036` |
| E75_SHUFFLED | PASS | 34 | 11297.983 | 10506.770 | `8a747006b8e6` |

- Ledger saat evaluasi: 109 starts, 106 saved, 3 failure historis, 0 pending; cap 109.
- Alokasi kategori tetap: pilot 68, continuation 38, control replay 2, technical retry 1. Empat slot amendment masuk dalam cap pilot yang telah diamendemen.
- Sesudah event completion, satu replay `score` mengembalikan hasil tersimpan; ledger tidak berubah dan tidak membuka ulang label. Hash ledger aktual setelah completion: `e6bf5051d4f45c3654ae6ce21f31437f30662db03e9c26e63f1051b2c7b4dc2b`; terdapat satu event `CARD_COMPLETED`.

## Konsistensi arah per siklus

- E67: real−bag positif 5/12 siklus; tidak positif pada siklus 68, 69, 70, 72, 75, 76, 77. Real−shuffled positif 9/12; tidak positif pada siklus 68, 70, 72.
- E75: real−bag positif 3/4 siklus; tidak positif pada siklus 77. Real−shuffled positif 3/4; tidak positif pada siklus 77.

## Ringkasan subgroup yang dilindungi

Semua perbandingan terperinci beserta denominator, Macro F1, recall positif, dan metrik probabilistik tersimpan di `RESULTS.json`. Stream missing/blank mengikuti fallback C1_B yang sama persis dengan arm bag pada semua view (E67: n=342–1.375 per view; E75 W6: n=526), sehingga tidak ada perubahan prediksi di slice fallback.
Gate dukungan dan harm untuk text_load Q4 lulus pada semua view yang ditetapkan. Namun nilai Q4 E75 W6 tetap menunjukkan Macro F1 real−bag −0,005332; ini masih di dalam batas harm −0,01, bukan peningkatan Q4.

## Kesimpulan

Satu gate gabungan gagal: E67 H12 real−bag +0,001151, di bawah minimum +0,003. Perbaikan pada beberapa window dan terhadap shuffled control tidak konsisten antar-siklus maupun Q4. Dengan pemilihan adaptif setelah diagnostic sebelumnya dan window E67/E75 yang overlap, hasil hanya mendukung review diagnostic. Kandidat tidak qualified, tidak mengganti champion, dan tidak memberi otorisasi Kaggle.
