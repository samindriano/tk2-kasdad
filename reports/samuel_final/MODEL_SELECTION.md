# Samuel — pemilihan model untuk final notebook review

31 Oktober 2026. Kandidat dipertahankan sesuai riset dan keputusan pengguna. Tidak ada perubahan recipe, model search besar, atau upload Kaggle pada milestone ini.

| Task | Kandidat | Primary | H12 pooled | Public Kaggle |
|---|---|---:|---:|---:|
| Classification | C4_HGB availability routing; threshold 0,50 | Macro F1 0,825481 | 0,796810 | 0,82641; ref 57808225 |
| Regression | LightGBM regression_l 1 + structured + TF-IDF/SVD32 + inner rank blend | MAE 11,931600 | 14,704041 | 11,71363; ref 57810150 |

Classification memakai C2_B hybrid LR saat stream tersedia dan C1_B structured HGB saat NaN/blank setelah strip. Missing-stream primary F1 meningkat 0,710427→0,781712 sementara present tetap 0,828555. C4_HGB fixed-0,50 adalah pilihan yang disubmit; bukan threshold policy universal. C 4_LR dengan inner equal-cycle memberi H12 pooled 0,802098, counterexample yang tetap dicatat.

Regression: structured Ridge primary 14,178654/H1215,205818 menjadi hybrid 12,551305/13,938115. LightGBM+SVD32 terbaik primary, equal-cycle 12,283548, dan membaik 6/6 windows atas R1-D. Tetapi **hybrid Ridge H1213,938115 lebih baik daripada LightGBM 14,704041**; LightGBM H12_E/L 17,473257/12,361143. Pemilihan ini mengutamakan primary tanpa mengklaim dominasi seluruh horizon.

Untuk full-Train submission, inner 47–75→76–79 memilih base 31 leaves/minleaf 40/L 2=1 dan rank weight 0,50. Historical primary menilai kebijakan inner selection per outer cutoff, bukan fixed 0,50 pada seluruh fold. Rank blend memakai prediksi per supplied cycle dan clipping; tidak membuktikan formula target resmi. Kelengkapan universe cycle belum terverifikasi.

[EDA A11](../../notebooks/samuel/11_samuel_formal_eda_a.ipynb) tentang TTR dan artifact association tetap utuh. Ia tidak membuktikan penyebab gain F1 atau manfaat teks Regression; evidence modeling berasal dari eksperimen terpisah. EDA temporal 03 menjadi arsip.

Semua cycle Train sudah dipakai development; H12 memakai ulang primary records. Stage 5 bersifat adaptif setelah sebagian public feedback. Tidak ada untouched holdout atau private-score claim. Novel zero-scope Test belum mempunyai labeled analogue.

[Classification R 2](../classification_round2/FINDINGS.md), [Regression findings](../regression_research/FINDINGS.md), dan accepted manifests [Classification](../submissions/classification_initial_manifest.json)/[Regression](../regression_research/submissions/lgb_text_inner_v1.json) tetap authoritative. [VALIDATION.json](VALIDATION.json) mencatat training nyata W6 dan membedakannya dari angka historis/public. Kedua notebook berhenti pada final review pengguna, bukan persetujuan upload tugas.
