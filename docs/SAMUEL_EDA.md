# EDA resmi Samuel — keragaman token dan artifact

**Keputusan pengguna, 31 Oktober 2026:** kandidat A resmi menjadi pertanyaan EDA individual Samuel untuk TP2.

> Pada laporan dengan jumlah token yang sama, apakah keragaman token dalam "signal_stream" berkaitan dengan proporsi "artifact_signal = 1" yang teramati?

Notebook EDA: [11_samuel_formal_eda_a.ipynb](../notebooks/samuel/11_samuel_formal_eda_a.ipynb). [Ringkasan, compliance dan reproduksi](../reports/eda_a_formal/SUMMARY.md) merangkum hasil. Notebook ini menghitung ulang dari official Classification Train; detail pendukung tetap pada [discovery](../reports/eda_candidates/DISCOVERY_FINDINGS.md) dan [targeted robustness](../reports/eda_a_robustness/FINDINGS.md).

TTR adalah proporsi token unik setelah regex `[A-Za-z0-9]+` dan lowercase. Pertanyaan meneliti asosiasi, bukan kualitas/semantik teks atau sebab-akibat. Pertanyaan Rousan, Abhi, dan Rizky tidak berubah. Dokumen alokasi bersama tidak diedit; halaman khusus Samuel ini mencatat keputusan terbaru.

EDA temporal lama dipertahankan sebagai **historical research/archive**: [notebook03](../notebooks/samuel/03_samuel_formal_eda.ipynb) dan [laporan formal lama](../reports/eda/SAMUEL_FORMAL_EDA.md). Notebook final modeling dan submission Kaggle berada di luar milestone ini. Penyajian paket final pengumpulan dikerjakan pada tahap akhir.

## Arsip — catatan sebelum keputusan EDA A

Teks berikut merekam milestone temporal terdahulu, bukan status EDA resmi saat ini. Perintah di bagian arsip berlaku untuk penelitian lama; gunakan instruksi EDA A pada tautan di atas untuk pekerjaan terbaru.

### EDA Samuel — formal milestone dan titik review

**Status sekarang:** pertanyaan utama tetap dan EDA formal selesai. Modeling Regression Round 1 yang diotorisasi pengguna juga selesai; hasil menunggu review arah berikutnya. Notebook EDA `notebooks/samuel/03_samuel_formal_eda.ipynb` dan hasil `reports/eda/SAMUEL_FORMAL_EDA.md` dipertahankan. Notebook modeling baru `notebooks/samuel/04_samuel_regression_round1.ipynb`; hasil di `reports/round1/ROUND1_FINDINGS.md` dan QA di `reports/round1/VALIDATION.md`. Dokumen rancangan lama dipertahankan sebagai riwayat sebelum modeling.

Jalankan `python scripts/build_formal_eda_notebook.py` atau Restart & Run All notebook utama. Alternatif runner deskriptif `python scripts/run_formal_eda.py`. `TP2_DATA_DIR`/`--data-dir` mengubah root raw yang berisi regression/. Formal EDA hanya membutuhkan data TP2; tidak membaca ulang model atau raw TP1.

Pertanyaan: **Apakah hubungan antara impact_load dan threat_rank mengalami perubahan antarperiode cycle_code, meskipun distribusi threat_rank relatif stabil?**

Evidence pendukung lengkap: `temporal_robustness.json` (33 cycle, periodisasi 2/3/4, equal-cycle, leave-one-out, resampling eksploratif, rankgeometry), `composition_robustness.json` (support, EF mixture, conditionalquantiles, count-only standardization, mean decomposition), `formal_eda_summary.json` (hash/environment, missingness strata dan tailcoverage support).

Prinsip interpretasi: asosiasi deskriptif; kuantil adalah spread observasi, bootstrap cycle bukan serial-dependence-robust CI, mean decomposition bukan kausal atau decomposition Pearson. Common support mencakup 96,09% baris dan mengecualikan low-tail lebih banyak secara proporsional. Geometri rank memakai target terhadap target itu sendiri dan tidak menjadi fitur atau dasar meranking ulang target pada fold.

Notebook 01 dan 02 tetap dipertahankan sebagai diagnostik/discovery historis. Perbandingan TP1 tersedia di laporan discovery dan receipt hash report lama; tidak ada TP1 refit. Delapan recipe Round 1 telah dijalankan pada delapan window tanpa mengubah pertanyaan EDA atau melatih classification. Belum ada final model, Round 2 atau submission.
