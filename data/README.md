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

Pada 30 Oktober 2026 hanya tiga file classification tersedia. Train 50.608 × 19, test 29.421 × 18, sample 29.421 × 2. Target terverifikasi: `artifact_signal`; 17 prediktor selain `record_id`. Target regression `threat_rank` belum terverifikasi karena semua file regression belum tersedia. Nama target pada dokumen perencanaan bukan pengganti bukti skema.

Provenance sebelumnya dicatat sebagai unduhan resmi Kaggle oleh dokumen 27 Oktober. Audit sekarang hanya memverifikasi byte lokal melalui SHA-256, bukan tanggal publikasi atau autentisitas upstream. Simpan informasi asal unduhan dan data dictionary ketika tersedia. Semua hash input, ukuran byte, dtype, missingness, unique count, distribusi dan perbandingan split tersimpan di `reports/audits/*_audit.json`.

`TP2_DATA_DIR` mengubah root default data; isinya tetap harus memiliki subfolder `classification/` dan `regression/`. CLI `--data-dir` mengalahkan environment. Jangan gunakan raw TP1 sebagai fallback. Jangan upload raw ke GitHub; `.gitignore` hanya mengizinkan `.gitkeep` di raw. Hasil transformasi terpisah di `data/processed/`; data/derived dipertahankan sebagai folder legacy kosong.

Sample submission hanya digunakan untuk mengonfirmasi header prediksi, jumlah baris, dan urutan ID; nilai placeholdernya bukan label test. Jangan melatih model atau mengukur target shift dari sample. Jangan menyimpulkan `threat_rank` adalah ranking ordinal, indeks kontinu, atau berskala 0–100 sebelum definisi resmi dan data diperiksa.

Semantik satu record, kalender cycle, lineage `audit_lane`/`checksum_hint`, dan saat ketersediaan `impact_load`/`entry_friction` tetap UNKNOWN. Minta penjelasan asisten sebelum menganggap fitur tersebut aman pada waktu prediksi.
