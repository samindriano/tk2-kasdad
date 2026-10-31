# KASDAD TP2 — A06 KasdudLovers

Cyber Vulnerability Lifecycle · Semester Ganjil 2026/2027 · Samuel, Rousan, Abhi, Rizky.

Repository independen dan private: [samindriano/tk2-kasdad](https://github.com/samindriano/tk2-kasdad).
**Status akademis terbaru:** EDA resmi Samuel adalah availability `signal_stream` dan volume narasi `text_load`, sesuai keputusan pengguna 31 Oktober 2026. Notebook formal: [EDA12](notebooks/samuel/12_samuel_formal_eda_availability.ipynb); approach individual: [Classification10](notebooks/samuel/10_samuel_final_classification.ipynb) dan [Regression09](notebooks/samuel/09_samuel_final_regression.ipynb). EDA temporal03 dan TTR11 tetap arsip. Model akademis tetap C4_HGB (public F1 0,82641) dan LightGBM/SVD32 (public MAE 11,71363). Riset Kaggle lanjutan terpisah dari notebook resmi; hasil historis dipertahankan.

## Status dan temuan awal

| Bagian | Status | Evidence |
|---|---|---|
| Classification | Audit aktual selesai; bukan sertifikasi bebas leakage | Train 50.608 × 19, test 29.421 × 18; target `artifact_signal` terverifikasi lewat train/test/sample |
| Regression | Audit aktual selesai | Train 50.608 × 19, test 29.421 × 18; `threat_rank` terverifikasi lewat train/test/sample |
| EDA Samuel | Availability + volume disetujui pengguna; notebook formal12 | Prevalence marginal, periode/cycle, distribusi volume dan standardisasi common-support; temporal03/TTR11 menjadi arsip |
| Regression Samuel Round 1 | Selesai; disetujui pengguna untuk progression | 64 fit + 64 refit terpisah; seluruh prediction vectors bit-identical; R1-D memimpin primary, tetapi R1-C sedikit lebih baik pada pooled H12 |
| Classification Samuel Round 1 | Selesai; disetujui sebagai checkpoint | 7 recipe, 56 evaluation jobs; 78 fits +13 representative refits. Hybrid LR fixed F1 0,8206, dengan missing-stream dan threshold trade-offs |
| Classification Samuel Round 2 | Selesai; berhenti untuk review | 10 recipe, 4 policies, 80 window jobs, 72 fits +12 representative refits. Fixed HGB fallback route F1 0,8255 primary /0,7968 H12 |
| Regression Samuel Round 2 | Selesai; berhenti untuk review | 19 recipe, 152 window evaluations, 112 unique fits +19 representative refits. Full recent-8 MAE 13,2172 primary; expanding no-codes MAE 15,0535 H12, dengan trade-off early/late dan high-tail |
| Initial Kaggle benchmarks | COMPLETE; satu upload per kompetisi | Classification C4_HGB fixed public Macro F1 0,82641; Regression R1-D asli clipped public MAE 13,53473. Tidak dipakai memilih recipe Round 2 |
| Autonomous Regression research | Selesai; strong provisional candidate | LightGBM MAE + TF-IDF/SVD32 + inner-selected rank blend: primary 11,9316 / H12 14,7040; public 11,71363. Empat dari enam additional submissions dipakai |

Riwayat autonomous Regression research: [notebook autonomous Regression research](notebooks/samuel/08_samuel_regression_research.ipynb), [hasil, kegagalan dan rekomendasi](reports/regression_research/FINDINGS.md), [seluruh perbandingan](reports/regression_research/COMPARISON_TABLES.md), [Kaggle log dan refs](reports/regression_research/KAGGLE_LOG.md), [best provisional manifest pointer](reports/regression_research/best_candidate.json), serta [protocol/reproduksi](docs/REGRESSION_RESEARCH.md). Text menambah informasi; pure percentile mapping tidak otomatis membantu. Hybrid Ridge tetap leader pooled H12 (13,9381), sedangkan LightGBM+text terbaik primary/public. H12 early risk dan adaptive validation bias tetap berlaku.

Checkpoint Regression Round 2: [notebook](notebooks/samuel/07_samuel_regression_round2.ipynb), [findings dan rekomendasi](reports/regression_round2/FINDINGS.md), [seluruh kandidat/window/cycle](reports/regression_round2/COMPARISON_TABLES.md), [temporal generalization risk](reports/regression_round2/TEMPORAL_GENERALIZATION.md), [protocol/reproduksi](docs/REGRESSION_ROUND2.md), dan [verification receipt](reports/regression_round2/verification.json). Recency membantu sedikit pada primary tetapi tidak H12; kontribusi kategori/cycle tergantung horizon/history. H12 memakai ulang records primary dan bukan independent replication.

Track submission terpisah: [status dan public scores](reports/submissions/KAGGLE_RESULTS.md), [recipe dan panduan](reports/submissions/INITIAL_SUBMISSIONS.md), [manifest classification](reports/submissions/classification_initial_manifest.json), serta [manifest regression](reports/submissions/regression_initial_manifest.json). Model berasal dari checkpoint sebelum Regression Round 2. CSV lokal tetap ignored; jangan mengulang upload tanpa izin baru.

Checkpoint Classification Round 2: [notebook](notebooks/samuel/06_samuel_classification_round2.ipynb), [findings dan rekomendasi](reports/classification_round2/FINDINGS.md), [tabel detail](reports/classification_round2/comparison_tables.md), [protocol/reproduksi](docs/CLASSIFICATION_ROUND2.md), serta [verification receipt](reports/classification_round2/verification.json). Fixed routing memperbaiki missing-stream; dua text weights dan alternatif threshold belum memberi perbaikan universal. Seluruh hasil adalah adaptive development evidence dari validation yang dipakai ulang.

Checkpoint Classification Round 1: [notebook](notebooks/samuel/05_samuel_classification_round1.ipynb), [hasil dan rekomendasi](reports/classification_round1/FINDINGS.md), [tabel detail](reports/classification_round1/comparison_tables.md), serta [protocol dan reproduksi](docs/CLASSIFICATION_ROUND1.md). Structured/text/hybrid dibandingkan dengan threshold 0,50 dan inner temporal selection. Semua source dan hasil kedua task Round 1 dipertahankan.

Mulai dari [notebook Round 1](notebooks/samuel/04_samuel_regression_round1.ipynb), [hasil dan rekomendasi](reports/round1/ROUND1_FINDINGS.md), [seluruh tabel](reports/round1/COMPARISON_TABLES.md), [QA](reports/round1/VALIDATION.md), dan [reproduksi](docs/ROUND1_REPRODUCIBILITY.md). Primary MAE R1-D **13,3498** versus R1-C **15,1490**; pada H12_E R1-D memburuk **17,7956** versus **16,4321**. Kandidat terbaik masih sementara.

- Classification: kelas 0 = 33.038; kelas 1 = 17.570 (34,72%). Tidak ada missing target, duplikasi baris/ID, atau ID overlap antar-split.
- Train cycle 47–79, test 80–91. Makna kalender dan mekanisme `cycle_code` belum dikonfirmasi; perbedaan periode tidak membuktikan causal drift.
- Missing `org_code`: 5,31% train vs 23,27% test; `module_code`: 7,99% vs 25,14%. Tangani missing dan kategori baru secara eksplisit.
- Pengguna telah mengonfirmasi `impact_load` dan `entry_friction` sebagai predictor resmi kedua task. Target regression adalah indeks prioritas respons 0–100 dengan MAE; classification adalah indikator artefak eksploit publik 0/1 dengan Macro F1. Provenance/timing rinci fitur lain tetap uncertainty; lihat [otoritas informasi kompetisi](docs/COMPETITION_INFO.md).
- `scope_pressure` memiliki perbedaan distribusi train/test besar (KS D = 0,2138). Round 2 juga menemukan 6.278 Test rows dengan zero scope/spread count dan pressure di bawah minimum Train, tanpa padanan zero-count pada Train. Ini descriptive support risk; Test tidak dipakai memilih recipe.

Lihat [EDA formal Samuel](reports/eda/SAMUEL_FORMAL_EDA.md), [rancangan eksperimen untuk review](docs/REGRESSION_EXPERIMENT_REVIEW.md), [riwayat discovery](reports/eda/REGRESSION_DISCOVERY.md), [riwayat audit awal](reports/audits/FINDINGS.md), [audit classification](reports/audits/classification_audit.md), [audit regression](reports/audits/regression_audit.md), dan [status EDA Samuel](docs/SAMUEL_EDA.md). JSON pendamping menyimpan profil setiap kolom, hash input, dan versi runtime.

## Menjalankan ulang

Python yang dipakai: 3.13.5. Dari root repository:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements-research.txt
python scripts/reproduce_regression_research.py --stage models
python scripts/verify_regression_research.py --stage all
python scripts/build_regression_research_notebook.py
```

Research memakai namespace sendiri dan menjaga historical report bytes melalui reproduction wrapper. Stage1 ranking memerlukan cache kontrol Regression Round 2; panduan research menjelaskan fresh-checkout prerequisites dan scoped tests. Tidak ada upload di notebook/reproduction wrapper. Reproduksi checkpoint Regression Round 1 secara terpisah:

```powershell
python scripts/run_round1.py
python scripts/build_round1_notebook.py
python -m unittest discover -s tests -v
```

Runner memverifikasi receipt freeze yang sudah ada sebelum fit/resume. Notebook memakai validation cache yang diverifikasi atau menjalankan ulang recipe tetap jika cache belum tersedia. Detail repeat run, input hash dan batas environment ada di [panduan reproduksi Round 1](docs/ROUND1_REPRODUCIBILITY.md).

Untuk menjalankan ulang audit/discovery historis secara terpisah:

```powershell
python scripts/audit_data.py
python scripts/samuel_eda.py
python scripts/build_audit_notebook.py
python scripts/build_samuel_notebook.py
python scripts/discover_regression.py --tp1-data-dir "../TK1/tk1-eda-3/data/raw"
python scripts/build_discovery_notebook.py
python scripts/build_formal_eda_notebook.py
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

Panduan rinci: [data/README.md](data/README.md). Makna/rentang target, metric dan izin impact/EF telah dikonfirmasi pengguna dari halaman resmi; lihat [COMPETITION_INFO.md](docs/COMPETITION_INFO.md). Formula pembentukan target, unit rinci dan timing fitur lain masih memerlukan evidence tambahan. Catatan UNKNOWN dalam laporan lama mencerminkan waktu audit tersebut.

## Pembagian EDA

| Anggota | Task | Pertanyaan yang disepakati |
|---|---|---|
| Samuel | Classification | Apakah proporsi `artifact_signal = 1` berbeda menurut ketersediaan `signal_stream`, dan bagaimana perbedaannya berubah setelah mempertimbangkan periode `cycle_code` serta volume narasi (`text_load`)? |
| Rousan | Classification | Apakah proporsi `artifact_signal` berbeda berdasarkan `entry_friction`, dan bagaimana hubungan berubah pada berbagai tingkat `impact_load`? |
| Abhi | Regression | Apakah `scope_pressure` memberi informasi tambahan mengenai `threat_rank` setelah mempertimbangkan `impact_load`? |
| Rizky | Classification | Bagaimana hubungan `telemetry_flag` dengan `artifact_signal`, dan apakah berbeda pada berbagai tingkat `module_spread`? |

Pertanyaan resmi Samuel kini berfokus pada Classification; notebook Regression tetap memuat pendekatan modeling individual tanpa membuat EDA Regression baru. Folder serta pertanyaan anggota lain tidak diubah. [Pembagian kerja](docs/WORK_ALLOCATION.md) dan [status EDA Samuel](docs/SAMUEL_EDA.md) mencatat keputusan terbaru; catatan terdahulu dipertahankan sebagai riwayat.

## Struktur

```text
data/                 raw lokal, processed, derived, panduan data
notebooks/            samuel, rousan, abhi, rizky; audit sebagai companion Samuel
src/data/             audit dan EDA deskriptif
src/features/         placeholder
src/models/           recipe, evaluation, diagnostics dan reporting kedua task; Classification Round 2 terpisah
src/utils/            konfigurasi path portable
scripts/              entry point audit, training/evaluation dan builder notebook
reports/audits/       laporan Markdown/JSON, hash dan profil input
reports/eda/          status/hasil EDA Samuel
reports/round1/       aggregate metrics, fit audits, figures, findings dan QA
reports/classification_round1/  classification review, metrics, fit audits, figures dan findings
reports/classification_round2/  ablation/routing/weight/policy comparisons dan independent receipts
experiments/round1/   protocol dan receipt freeze sebelum fit
experiments/classification_round1/  protocol dan freeze classification
experiments/classification_round2/  protocol, source/reference identity dan checkpoint protection
experiments/runs/     ignored validation vectors untuk reproduksi lokal
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
- [Protokol Round 1](docs/ROUND1_PROTOCOL.md) dan [feature eligibility](docs/ROUND1_FEATURE_ELIGIBILITY.md) dibekukan sebelum eksperimen. Enam primary windows adalah development validation karena seluruh train sudah dipakai EDA. Recipe TP1 tidak dipindahkan otomatis.
- [Referensi TP1](docs/TP1_REFERENCE.md) dibaca tanpa perubahan. Ada perbedaan antara ringkasan resep lama dan artifact final yang ditemukan; perbedaannya didokumentasikan.

Catatan titik review historis Round 2: prioritas saat itu adalah memilih arah penelitian kedua task. Checkpoint lanjutan sudah tersedia; lihat status akademis terbaru di awal halaman. Kenaikan skor agregat tetap dibaca bersama horizon panjang, reused-validation selection bias, threshold policy serta missing-stream/EF subgroups.

## Arsip discovery Regression

Arsip regression dipindahkan dari Downloads ke raw lokal; CRC dan SHA-256 diverifikasi. Tidak ada overwrite data lama. Kedua task memakai file test yang byte-identical; train regression memakai 18 kolom input yang sama (termasuk ID), dengan label threat_rank.

Threat_rank memiliki 34.333 nilai unik, range teramati 0,0428–100, median 50,0282, top-10 nilai hanya 1,08% baris; TP1 impact_load memiliki 80 nilai dan top-10 mencakup 79,10%. Pearson impact–threat 0,6370 dan Spearman 0,6211. Pearson per periode 47–57 / 58–68 / 69–79 adalah 0,4994 / 0,6722 / 0,6964, sementara marginal target hampir tetap. Ini deskriptif dan tidak membuktikan causal drift atau error model TP2.

Notebook discovery: `notebooks/samuel/02_regression_discovery.ipynb`. Perbandingan TP1 opsional memerlukan root raw TP1 via `--tp1-data-dir` atau environment `TP1_DATA_DIR`; Round 1 hanya membutuhkan raw TP2. Definisi cycle terurut tercantum dalam materi lokal TP1 halaman 16; makna target TP2 kini dikonfirmasi pengguna dan formula rinci tetap belum diketahui.

## Arsip EDA temporal dan titik review terdahulu

Notebook demo pada milestone lama: `notebooks/samuel/03_samuel_formal_eda.ipynb`; jalankan Restart & Run All atau `python scripts/build_formal_eda_notebook.py`. Runner tanpa notebook: `python scripts/run_formal_eda.py`; path default relatif root kode, dengan override `TP2_DATA_DIR` atau `--data-dir`. Pendukung: `temporal_robustness.json`, `composition_robustness.json`, `formal_eda_summary.json` dan lima grafik diagnostik. Notebook lama dipertahankan sebagai sejarah discovery.

Hubungan tetap berbeda setelah mix exact impact×EF dibuat sama: Pearson 0,5244 → 0,6617 → 0,6837 pada 33 common cells, coverage96,09%. EF0 datar setelah blok tengah, EF1 masih menguat. Dua kelompok impact dan EF tetap menunjukkan conditional shifts berlawanan. Mean target percycle hampir persis50+50/n, conditional pada geometri average percentile yang teramati; kestabilan marginal tidak membuktikan kestabilan hubungan fitur atau mekanisme sumber.

Usulan R0/R1 dari EDA telah dijalankan pada Round 1 dengan otorisasi pengguna. Semua cycle train telah dipakai EDA sehingga hasil modeling tetap development validation. Lihat `reports/audits/FORMAL_EDA_VALIDATION.md` untuk QA historis EDA dan `reports/round1/VALIDATION.md` untuk QA modeling.
