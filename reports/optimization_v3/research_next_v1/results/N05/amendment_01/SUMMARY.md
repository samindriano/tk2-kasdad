# N05 — Conditional uncertainty from quantile intervals

## Pertanyaan dan ruang lingkup

Apakah lebar interval kondisional dari dua head kuantil membantu menggambarkan kesulitan prediksi dibanding interval residual konstan yang dibangun dari prediksi R22 pada baris OOF masa lalu?

Ini adalah lane informasi. Hasilnya tidak mengubah prediksi titik R22, tidak menjadi kandidat continuation, dan tidak memilih threshold atau model dari Test/Kaggle.

## Pelaksanaan

- Dua head LightGBM quantile (`alpha=0.1` dan `0.9`) dilatih pada desain structured/OHE + TF-IDF/SVD32 R22 dengan preprocessing yang di-fit sekali per origin.
- Fit prefix resmi berisi 29.705 baris pada cutoff 67 dan 42.117 pada cutoff 75. Keempat estimator fit dijalankan serial, masing-masing tersimpan segera.
- Interval pembanding memakai residual bertanda `threat_rank - R22_ORIGINAL` dari primary OOF unik yang seluruh cycle-nya tidak melewati cutoff: 17.685 baris untuk cutoff 67 dan 30.097 untuk cutoff 75.
- Endpoint raw diperiksa untuk crossing, dipotong ke `[0,100]`, lalu diurutkan. Target tail hanya menjadi subgroup evaluasi setelah prediksi.
- Tampilan primary LATE adalah origin 67 pada cycle 68–71 dan origin 75 pada cycle 76–79. Prediksi H12 dari origin 67 dilaporkan terpisah sebagai diagnostik yang overlap dengan primary.

## Temuan

| View | n | Coverage kuantil | Coverage residual | Lebar kuantil | Lebar residual |
|---|---:|---:|---:|---:|---:|
| Primary origin 67, cycle 68–71 | 4.950 | 72,71% | 85,01% | 36,46 | 39,79 |
| Primary origin 75, cycle 76–79 | 8.491 | 74,57% | 86,20% | 31,78 | 36,98 |
| Pooled primary LATE, baris unik | 13.441 | 73,89% | 85,76% | 33,51 | 38,02 |
| H12 LATE dari origin 67 | 20.903 | 73,01% | 85,44% | 36,89 | 39,79 |

Lebar kondisional mengandung sinyal deskriptif tentang error titik R22. Pada pooled primary, korelasi Spearman lebar interval dengan `|y - R22_ORIGINAL|` adalah **0,367**, dibanding **0,301** pada kontrol residual setelah clipping. Rata-rata error R22 per kuintil lebar kondisional meningkat dari **6,13** pada kuintil terendah menjadi **15,68** pada tertinggi. Coverage kondisional naik dari **63%** ke **81%**, tetapi tidak monoton pada semua kuintil.

Interval kondisional lebih sempit rata-rata **4,51 poin**, namun coverage-nya **6,11 pp di bawah nominal 80%** dan **11,87 pp di bawah kontrol residual**. Undercoverage tampak pada kedua primary window dan H12. Pada subgroup evaluasi target `<=5` (n=686), coverage kondisional 42,27% versus 88,63% pada kontrol; pada target `>=95` (n=683), 45,53% versus 94,00%. Subgroup tersebut memakai target aktual hanya untuk mengelompokkan hasil evaluasi, bukan sebagai fitur.

Jadi lebar kuantil membawa informasi relatif tentang error, tetapi interval yang dihasilkan belum terkalibrasi untuk nominal 80% dan kurang mencakup observasi dibanding kontrol yang lebih lebar. Dua origin saja juga belum cukup untuk menyimpulkan hubungan umum atau menjamin perilaku pada data mendatang.

## QA, integritas, dan accounting

- 6 unit tests N05 lulus; synthetic end-to-end memeriksa alpha 0,1/0,9, crossing, empty subgroup, larangan promosi point model, serta persist/read/resume bundle.
- Keempat bundle cocok dengan row/ID/cache R22, fit prefix, target hash fit, control residual, dan ledger. Independent verification mengulang rekonstruksi cache dan metrik primary/H12; **32 frozen source/input hashes** lolos.
- Runner's first verifier berhenti setelah fit karena membandingkan crossing rate full H12 origin 67 dengan angka primary-only. Empat fit dan report telah tersimpan; tidak ada fit ulang. Kode/verdict frozen tetap dipertahankan, dan verifier pasca-fit terpisah menghitung crossing pada mask primary yang benar lalu mengonfirmasi hasil.
- N05 memakai 4 fit resmi dari cap fase 60. Ledger fase saat laporan: 18 start tersimpan kumulatif, 0 failed/interrupted, 4 slot tak terpakai dari N04 asli tetap dirilis; total reservasi 22, active reserved 18.
- Official Test, SampleSubmission, Kaggle, full-Train refit, dan notebook tidak digunakan.

## Artefak

- Detail angka seluruh origin/cycle/subgroup: `results.json`.
- Receipt dan rekalkulasi independen: `verification.json`.
- Visual utama yang ditinjau: `interval_diagnostic_final.png`.
- Grafik runner awal tetap disimpan sebagai `interval_diagnostic.png`; grafik final memisahkan dua origin dengan legenda agar pembacaannya jelas.
- Plot final dapat dibuat ulang dari `results.json` dengan `scripts/optimization_v3_research_next_n05_plot.py`; tidak ada refit.

**Verdict N05:** `COMPLETE_INFORMATION`. Diagnostic mendukung asosiasi lebar dengan tingkat error, tetapi menunjukkan undercoverage nyata dan tidak memiliki kewenangan untuk promosi model.
