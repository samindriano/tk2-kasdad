# Validasi milestone EDA formal Samuel — 30 Oktober 2026

**Review integration: APPROVE untuk scope EDA dan rancangan eksperimen.** Pertanyaan utama dipilih pengguna dan tidak diubah. Tidak ada approval model/fitur final atau validation protocol; milestone berhenti pada titik review pengguna.

## Eksekusi dan evidence

- Runner formal dan notebook `03_samuel_formal_eda.ipynb` dijalankan pada seluruh 50.608 baris train resmi, seluruh 33 cycle. Notebook memiliki 26 cell (9 code), tanpa error, tiga embedded figures, valid nbformat. Supporting figures/JSON dihitung ulang dari source melalui helper yang dapat diperiksa.
- Empat belas unit checks lulus. Check baru merekonsiliasi semua 33 Pearson dengan numpy, two-sided uniform ECDF distance dengan scipy, pemilihan common support hanya dari counts, weighted correlation dari **cell moments** terpisah dari implementasi row-weight, serta symmetric mean decomposition dari actual source rows.
- Check sebelumnya memverifikasi CRC/SHA archive dan CSV, target schema, sample IDs, cycle grouping, source immutability dan rank identity memakai scipy rankdata. Raw tetap ignored; tidak ada data/model/prediction artifacts yang ditambahkan ke Git.
- Review independen read-only mereproduksi 33 common cells (48.630 baris), weighted correlation 0,5243955/0,6617331/0,6836578, mean decomposition+0,16755014=+1,05035856−0,88280842 dan rank-implied mean residual maksimum 2,8316 e−6. Tidak ditemukan blocker faktual atau runtime.
- Python compilation dan pemeriksaan whitespace Git dijalankan. Export notebook ke HTML berhasil; warning Windows zmq event-loop tidak menggagalkan eksekusi.

## QA visual dan batasnya

Grafik diperiksa langsung pada PNG: all-cycle relationship, period sensitivity, common-impact distributions, EF composition, mean decomposition dan dua selected conditional groups. Label skor exact-impact dipertahankan empat desimal; target scales pada comparative panels disamakan. Denominator cycle dipindah ke panel bar terpisah untuk menghindari overlap label. Notebook utama memakai dua ilustrasi conditional groups agar alur demo tidak menjadi delapan panel investigasi.

P10/P90 dan IQR adalah spread observasi. Resampling cycle diberi label exploratory sensitivity, bukan confidence interval dan tidak mengatasi serial dependence. Mean decomposition memakai retained common-support denominator dan tidak disebut causal/Pearson decomposition.

Inspeksi visual ini mencakup PNG dan struktur/output notebook. HTML export penuh belum dibuka dalam browser karena kebijakan file-URL yang sebelumnya terblokir; tidak dilakukan bypass. Keberhasilan eksekusi tidak disebut sebagai bukti review visual seluruh halaman HTML.

## Reproducibility dan scope

`python scripts/build_formal_eda_notebook.py` menjalankan milestone dari awal; `python scripts/run_formal_eda.py` menjalankan perhitungan tanpa notebook. Path default relatif root kode; `TP2_DATA_DIR` atau `--data-dir` untuk raw di lokasi lain. Formal EDA tidak membutuhkan TP1 checkout. Source train SHA tercatat di summary dan input manifest.

Raw input dan TP1 dipertahankan; notebook/discovery/eksperimen historis tidak dihapus. Pekerjaan anggota lain tidak diubah. Rancangan eksperimen hanya mendokumentasikan hipotesis, metode evaluasi, risiko dan urutan diskusi. Tidak ada estimator fit/predict, training, tuning, ensemble, model final atau Kaggle submission.

UNKNOWN yang masih membatasi keputusan: definisi resmi threat_rank, absolute calendar/serial dependence cycle, availability timing fitur, dan performa prediksi pada cycle baru. Seluruh label train telah dipakai EDA; validation berikutnya tidak disebut untouched holdout.
