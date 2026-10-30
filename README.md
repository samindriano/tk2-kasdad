# KASDAD TP2 — A06 KasdudLovers

Cyber Vulnerability Lifecycle · Semester Ganjil 2026/2027 · Samuel, Rousan, Abhi, Rizky.

Repository independen dan private: [samindriano/tk2-kasdad](https://github.com/samindriano/tk2-kasdad).
Persiapan dan audit dijalankan pada **30 Oktober 2026**. Tidak ada training, tuning, atau submission TP2 pada tahap ini.

## Status dan temuan awal

| Bagian | Status | Evidence |
|---|---|---|
| Classification | Audit aktual selesai; bukan sertifikasi bebas leakage | Train 50.608 × 19, test 29.421 × 18; target `artifact_signal` terverifikasi lewat train/test/sample |
| Regression | **BLOCKED_MISSING_DATA** | Folder raw regression kosong; `threat_rank` masih dugaan yang belum diverifikasi dari file |
| EDA Samuel | Notebook dieksekusi dengan status blocked | Ketiga file regression resmi belum ada; belum ada korelasi atau grafik relasi target |

- Classification: kelas 0 = 33.038; kelas 1 = 17.570 (34,72%). Tidak ada missing target, duplikasi baris/ID, atau ID overlap antar-split.
- Train cycle 47–79, test 80–91. Makna kalender dan mekanisme `cycle_code` belum dikonfirmasi; perbedaan periode tidak membuktikan causal drift.
- Missing `org_code`: 5,31% train vs 23,27% test; `module_code`: 7,99% vs 25,14%. Tangani missing dan kategori baru secara eksplisit.
- `impact_load` dan `entry_friction` tersedia sebagai fitur classification resmi. Kesamaan dengan target TP1 tidak otomatis membuktikan leakage; provenance dan waktu ketersediaan fitur tetap perlu dikonfirmasi.
- `scope_pressure` memiliki perbedaan distribusi train/test terbesar di antara fitur numerik non-ID/cycle (KS D = 0,2138). Test dipakai untuk audit skema/distribusi, bukan memilih model.

Lihat [ringkasan audit](reports/audits/FINDINGS.md), [audit classification](reports/audits/classification_audit.md), [audit regression](reports/audits/regression_audit.md), dan [status EDA Samuel](docs/SAMUEL_EDA.md). JSON pendamping menyimpan profil setiap kolom, hash input, dan versi runtime.

## Menjalankan ulang

Python yang dipakai: 3.13.5. Dari root repository:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python scripts/audit_data.py
python scripts/samuel_eda.py
python scripts/build_audit_notebook.py
python scripts/build_samuel_notebook.py
python -m unittest discover -s tests -v
```

Builder notebook menjalankan cell dari awal sampai akhir dan menyimpan output. Notebook juga dapat dibuka langsung di VS Code/Jupyter dengan kernel environment di atas, lalu **Restart & Run All**. Buka dari root repo atau subfolder notebook. Tidak perlu mengubah source untuk pindah komputer.

Data default: `data/raw/{classification,regression}/`. Untuk lokasi lain:

```powershell
$env:TP2_DATA_DIR = 'D:\course-data\tp2'
python scripts/audit_data.py
```

`TP2_DATA_DIR` adalah root yang berisi dua folder task, bukan path CSV. Script CLI juga menerima `--data-dir`; audit menerima `--tp1-data-dir` untuk perbandingan opsional read-only dengan raw TP1. Path CLI relatif dievaluasi terhadap working directory; default output selalu relatif terhadap root kode. Audit dengan input hilang tetap menghasilkan laporan **blocked**, bukan hasil sukses analitis.

## Dataset

Tambahkan `Train.csv`, `Test.csv`, dan `SampleSubmission.csv` **resmi** untuk masing-masing task. Nama file dideteksi tanpa membedakan huruf besar/kecil; nama ambigu ditolak. Jangan mengganti regression dengan classification atau menyusun target dari TP1. File raw hanya dibaca dan tidak masuk Git. Provenance lokal sebelumnya menyebut unduhan Kaggle; sesi ini memverifikasi byte/hash lokal, tidak mengautentikasi ulang sumber unduhan.

Panduan rinci: [data/README.md](data/README.md). Definisi domain `threat_rank`, rentang resminya, unit fitur, dan timing pembentukan kolom memerlukan data dictionary/materi resmi. Catatan lama mengenai target/rentang/metric belum dianggap terverifikasi hanya karena ada dalam dokumen.

## Pembagian EDA

| Anggota | Task | Pertanyaan yang disepakati |
|---|---|---|
| Samuel | Regression | Seberapa kuat hubungan `impact_load` dan `threat_rank`, serta apakah konsisten pada berbagai periode `cycle_code`? |
| Rousan | Classification | Apakah proporsi `artifact_signal` berbeda berdasarkan `entry_friction`, dan bagaimana hubungan berubah pada berbagai tingkat `impact_load`? |
| Abhi | Regression | Apakah `scope_pressure` memberi informasi tambahan mengenai `threat_rank` setelah mempertimbangkan `impact_load`? |
| Rizky | Classification | Bagaimana hubungan `telemetry_flag` dengan `artifact_signal`, dan apakah berbeda pada berbagai tingkat `module_spread`? |

Hanya EDA Samuel disiapkan pada tahap ini. Folder anggota lain adalah placeholder. [Pembagian kerja terbaru](docs/WORK_ALLOCATION.md) mengontrol ruang lingkup sekarang; catatan 27 Oktober dipertahankan sebagai riwayat.

## Struktur

```text
data/                 raw lokal, processed, derived, panduan data
notebooks/            samuel, rousan, abhi, rizky; audit sebagai companion Samuel
src/data/             audit dan EDA deskriptif
src/features/         placeholder
src/models/           placeholder; belum ada training
src/utils/            konfigurasi path portable
scripts/              entry point audit dan builder notebook
reports/audits/       laporan Markdown/JSON, hash dan profil input
reports/eda/          status/hasil EDA Samuel
experiments/          aturan pencatatan dan template; belum ada run model
submissions/          placeholder, output diabaikan Git
deliverables/individual-final/  placeholder
tests/                pemeriksaan path, skema, dan konsistensi audit
docs/                 pembagian kerja, referensi TP1, keterbatasan
```

Folder `data/derived`, `notebooks/classification`, `notebooks/regression`, dokumen 27 Oktober, dan `reports/classification_data_audit.json` sudah ada sebelum inisialisasi Git. Dipertahankan; audit saat ini berada di `reports/audits/`. Tidak perlu mengedit raw atau menyalin pipeline TP1.

## Reproducibility dan batas kerja

- Simpan source data utuh; transformasi ke `data/processed/`, artifact besar ke folder ignored. Jangan commit credential, `.env`, dataset, atau model.
- Gunakan hash input, versi environment, preprocessing, seed, split/cycle, metric dan identitas commit untuk eksperimen berikutnya. Template: [experiments/README.md](experiments/README.md).
- Pelajari imputasi/encoding/vektor teks hanya dari train fold. Jangan menggabungkan label tambahan TP1 atau memilih konfigurasi berdasarkan test.
- Belum ada validation split/metric TP2 yang dibekukan; konfirmasi rubric dan makna cycle dahulu. Jangan menganggap resep TP1 pasti cocok dengan target TP2.
- [Referensi TP1](docs/TP1_REFERENCE.md) dibaca tanpa perubahan. Ada perbedaan antara ringkasan resep lama dan artifact final yang ditemukan; perbedaannya didokumentasikan.

Langkah berikutnya: tambahkan file regression resmi, jalankan ulang audit dan notebook Samuel, tinjau definisi target serta availability fitur, baru sepakati rancangan validasi sebelum modeling.
