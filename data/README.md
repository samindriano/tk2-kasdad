# Dataset lokal dan lineage

Simpan unduhan resmi apa adanya:

```text
data/raw/classification/Train.csv
data/raw/classification/Test.csv
data/raw/classification/SampleSubmission.csv
data/raw/regression/Train.csv
data/raw/regression/Test.csv
data/raw/regression/SampleSubmission.csv
```

Pada 30 Oktober 2026 kedua task tersedia: masing-masing train 50.608 × 19, test 29.421 × 18, sample 29.421 × 2. Target classification `artifact_signal` dan regression `threat_rank` terverifikasi lewat skema train/test/sample; 17 prediktor selain record_id. Arsip regression dipindahkan dari Downloads dan diekstrak byte-identical; receipt CRC/SHA-256 di `reports/audits/regression_acquisition.json`.

Provenance sebelumnya dicatat sebagai unduhan resmi Kaggle oleh dokumen 27 Oktober. Audit sekarang hanya memverifikasi byte lokal melalui SHA-256, bukan tanggal publikasi atau autentisitas upstream. Simpan informasi asal unduhan dan data dictionary ketika tersedia. Semua hash input, ukuran byte, dtype, missingness, unique count, distribusi dan perbandingan split tersimpan di `reports/audits/*_audit.json`.

`TP2_DATA_DIR` mengubah root default data; isinya tetap harus memiliki subfolder `classification/` dan `regression/`. CLI `--data-dir` mengalahkan environment. Jangan gunakan raw TP1 sebagai fallback. Jangan upload raw ke GitHub; `.gitignore` hanya mengizinkan `.gitkeep` di raw. Hasil transformasi terpisah di `data/processed/`; data/derived dipertahankan sebagai folder legacy kosong.

Sample submission hanya digunakan untuk mengonfirmasi header prediksi, jumlah baris, dan urutan ID; nilai placeholdernya bukan label test. Jangan melatih model atau mengukur target shift dari sample. Konfirmasi resmi yang diberikan pengguna pada Round1: `threat_rank` adalah indeks prioritas respons kerentanan berskala0–100 dan dievaluasi dengan MAE; `artifact_signal` indikator artefak eksploit publik0/1 dengan MacroF1. Formula percentile/rank yang membentuk label tetap belum dikonfirmasi.

Materi lokal halaman 16 menjelaskan record_id sebagai ID observasi kerentanan dan cycle_code sebagai urutan siklus disclosure (besar = lebih baru); kalender absolut, lineage `audit_lane`/`checksum_hint`, serta timestamp/provenance rinci fitur tambahan lain tetap UNKNOWN. Pengguna mengonfirmasi impact_load dan entry_friction merupakan predictor resmi yang boleh digunakan pada TP2. Lihat docs/COMPETITION_INFO.md dan eligibility Round1.
