# EDA Samuel — formal milestone dan titik review

**Status sekarang:** pertanyaan utama sudah dipilih pengguna dan EDA formal selesai. Notebook demo utama `notebooks/samuel/03_samuel_formal_eda.ipynb`; hasil dan batas interpretasi di `reports/eda/SAMUEL_FORMAL_EDA.md`. Modeling tetap menunggu review atas `docs/REGRESSION_EXPERIMENT_REVIEW.md`.

Jalankan `python scripts/build_formal_eda_notebook.py` atau Restart & Run All notebook utama. Alternatif runner deskriptif `python scripts/run_formal_eda.py`. `TP2_DATA_DIR`/`--data-dir` mengubah root raw yang berisi regression/. Formal EDA hanya membutuhkan data TP2; tidak membaca ulang model atau raw TP1.

Pertanyaan: **Apakah hubungan antara impact_load dan threat_rank mengalami perubahan antarperiode cycle_code, meskipun distribusi threat_rank relatif stabil?**

Evidence pendukung lengkap: `temporal_robustness.json` (33 cycle, periodisasi 2/3/4, equal-cycle, leave-one-out, resampling eksploratif, rankgeometry), `composition_robustness.json` (support, EF mixture, conditionalquantiles, count-only standardization, mean decomposition), `formal_eda_summary.json` (hash/environment, missingness strata dan tailcoverage support).

Prinsip interpretasi: asosiasi deskriptif; kuantil adalah spread observasi, bootstrap cycle bukan serial-dependence-robust CI, mean decomposition bukan kausal atau decomposition Pearson. Common support mencakup 96,09% baris dan mengecualikan low-tail lebih banyak secara proporsional. Geometri rank memakai target terhadap target itu sendiri dan tidak menjadi fitur atau dasar meranking ulang target pada fold.

Notebook 01 dan 02 tetap dipertahankan sebagai diagnostik/discovery historis. Perbandingan TP1 tersedia di laporan discovery dan receipt hash report lama; tidak ada TP1 refit pada milestone ini. Seluruh model/eksperimen pada rancangan masih belum dijalankan.
