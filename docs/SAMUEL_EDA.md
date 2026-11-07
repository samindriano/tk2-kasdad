# EDA resmi Samuel — availability `signal_stream`

**Keputusan pengguna, 31 Oktober 2026:** pertanyaan resmi EDA individual adalah:

> Apakah persentase laporan berlabel `artifact_signal = 1` berbeda antara `signal_stream` kosong dan terisi, setelah mempertimbangkan waktu dan volume laporan?

Waktu memakai `cycle_code` dan volume laporan memakai `text_load`. Ini penyederhanaan bahasa dari pertanyaan yang sama; variabel, metode, dan hasil analisis tetap.

Notebook final: [12_samuel_formal_eda_availability.ipynb](../notebooks/samuel/12_samuel_formal_eda_availability.ipynb). Notebook menggunakan official Classification Train dan menghitung hasil dari raw data, bukan laporan cache. Ringkasan keputusan ada di [FINDINGS.md](../reports/academic_availability/FINDINGS.md); metodologi dan eksekusi tercatat pada [QA.json](../reports/academic_availability/QA.json).

Hasil utama: 50.608 baris; stream missing pada 3.404 (6,73%) dan present pada 47.204. Artifact prevalence marginal ialah 38,13% (missing) dan 34,47% (present), gap **+3,66 pp**. Setelah standardisasi exact cycle gap **+3,96 pp** dengan 100% coverage. Setelah standardisasi periode × kuantil global `text_load`, gap berubah arah menjadi **−2,30 pp** untuk lima bin (100% coverage) dan **−2,15 pp** untuk sepuluh bin (94,33%; 28/30 sel didukung; minimum 20 observasi per status per sel).

Periode gap tidak seragam: −1,84 pp pada 47–57, +10,01 pp pada 58–68, dan +3,18 pp pada 69–79. Median `text_load` adalah 21,81 saat stream missing dan 31,78 saat present. Batas nilai yang sama-sama teramati adalah 20,65–77,52; bin kasar tidak menjamin overlap tepat.

Ini asosiasi deskriptif pada Train, bukan efek kausal. Standardisasi hanya menyamakan komposisi cycle/periode-volume yang didefinisikan, tidak menghilangkan seluruh confounding, dan tidak membuktikan pola pada official Test. EDA raw ini dikembangkan setelah modeling/error diagnostics; ia memberi konteks availability, bukan penyebab keputusan routing. TTR EDA terdahulu tetap diarsipkan pada notebook11; EDA temporal lama tetap diarsipkan pada [notebook03](../notebooks/samuel/03_samuel_formal_eda.ipynb). Keduanya tidak ditimpa atau dihapus.

Pertanyaan Rousan, Abhi, dan Rizky tidak berubah.

## Arsip — catatan sebelum keputusan EDA availability

Bagian berikut merekam pertanyaan dan riset historis. Ia bukan status EDA resmi saat ini.

### EDA Samuel — formal milestone dan titik review

**Status pada milestone saat itu:** pertanyaan temporal utama tetap dan EDA formal selesai. Modeling Regression Round 1 yang diotorisasi pengguna juga selesai; hasil menunggu review arah berikutnya. Notebook EDA `notebooks/samuel/03_samuel_formal_eda.ipynb` dan hasil `reports/eda/SAMUEL_FORMAL_EDA.md` dipertahankan. Notebook modeling baru `notebooks/samuel/04_samuel_regression_round1.ipynb`; hasil di `reports/round1/ROUND1_FINDINGS.md` dan QA di `reports/round1/VALIDATION.md`. Dokumen rancangan lama dipertahankan sebagai riwayat sebelum modeling.

Jalankan `python scripts/build_formal_eda_notebook.py` atau Restart & Run All notebook utama. Alternatif runner deskriptif `python scripts/run_formal_eda.py`. `TP2_DATA_DIR`/`--data-dir` mengubah root raw yang berisi regression/. Formal EDA hanya membutuhkan data TP2; tidak membaca ulang model atau raw TP1.

Pertanyaan: **Apakah hubungan antara impact_load dan threat_rank mengalami perubahan antarperiode cycle_code, meskipun distribusi threat_rank relatif stabil?**

Evidence pendukung lengkap: `temporal_robustness.json` (33 cycle, periodisasi 2/3/4, equal-cycle, leave-one-out, resampling eksploratif, rankgeometry), `composition_robustness.json` (support, EF mixture, conditionalquantiles, count-only standardization, mean decomposition), `formal_eda_summary.json` (hash/environment, missingness strata dan tailcoverage support).

Prinsip interpretasi: asosiasi deskriptif; kuantil adalah spread observasi, bootstrap cycle bukan serial-dependence-robust CI, mean decomposition bukan kausal atau decomposition Pearson. Common support mencakup 96,09% baris dan mengecualikan low-tail lebih banyak secara proporsional. Geometri rank memakai target terhadap target itu sendiri dan tidak menjadi fitur atau dasar meranking ulang target pada fold.

Notebook 01 dan 02 tetap dipertahankan sebagai diagnostik/discovery historis. Perbandingan TP1 tersedia di laporan discovery dan receipt hash report lama; tidak ada TP1 refit. Delapan recipe Round 1 telah dijalankan pada delapan window tanpa mengubah pertanyaan EDA atau melatih classification. Belum ada final model, Round 2 atau submission.
