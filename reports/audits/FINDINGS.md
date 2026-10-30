# Ringkasan audit TP2 — 30 Oktober 2026

**Classification: profil aktual tersedia; regression dan EDA relasi Samuel: BLOCKED karena semua file regression resmi belum ada.** Tidak ada training, tuning atau Kaggle submission.

## Sumber, grain, dan coverage

File lokal diperiksa dari `data/raw/classification/`. Nama, ukuran byte dan SHA-256 di `classification_audit.json`; hash cocok dengan catatan audit lama. Provenance lama menyebut Kaggle, tetapi sesi ini tidak memverifikasi ulang unduhan upstream. Grain teramati adalah satu baris per `record_id` unik; arti domain persisnya belum dikonfirmasi.

| Split | Rows | Columns | Target |
|---|---:|---:|---|
| Train | 50.608 | 19 | artifact_signal |
| Test | 29.421 | 18 | tidak tersedia |
| Sample | 29.421 | 2 | placeholder, bukan label test |

Target classification terverifikasi melalui satu kolom train-only yang sama dengan header prediksi sample. Ada 17 predictor selain ID. Semua dtype, missing, distinct, kuantil numerik, IQR, kategori berkardinalitas rendah dan hitungan cycle dicatat di JSON. Tidak ada duplicate row, duplicate ID, missing ID, ID overlap, duplicate fitur tanpa ID/target, maupun shared distinct feature row antar-split. Urutan sample ID sama dengan test. Numerik tidak mengandung infinity.

## Temuan dan implikasi

| Temuan | Evidence | Risiko / tindakan |
|---|---|---|
| Imbalance moderat | 0: 33.038 (65,28%); 1: 17.570 (34,72%); target tidak missing | Pertimbangkan metric per kelas sesuai rubric; jangan menyimpulkan perlu resampling sebelum evaluasi |
| Missing bergeser | org_code 5,31% → 23,27%; module_code 7,99% → 25,14%; pattern_code 16,17% → 8,61% | HIGH, confidence tinggi: evaluasi robustness missing/unknown category; fit preprocessing hanya pada fold train |
| Kategori baru di test | 2.047 org_code dan 3.744 module_code baru; mengenai 20,96% dan 36,77% dari baris test yang nonmissing di kolom terkait | HIGH, confidence tinggi: encoder harus menerima unknown; angka kategori unik berbeda dari proporsi baris |
| Cakupan cycle terpisah | train 47–79, test 80–91; KS cycle = 1 karena support disjoint | HIGH untuk risiko validasi: pertimbangkan forward validation setelah urutan waktu dikonfirmasi; tidak membuktikan drift mekanistik |
| Distribusi numerik berbeda | scope_pressure KS D=0,2138; median 70,08 → 69,54 tetapi mean 128,38 → 95,69; 6.284 test di luar range train | MEDIUM/HIGH, confidence tinggi pada perbedaan deskriptif: periksa tail/validitas domain, jangan tuning memakai test |
| Outlier deskriptif | scope_pressure train 3.593 flag IQR; test 10.921 memakai fence masing-masing split. Range train 68,4148–756,329; test −0,8996–779,5182 | Perbedaan flag tidak menggunakan threshold bersama; bukan bukti error. Konfirmasi makna nilai negatif; jangan otomatis menghapus/memotong |
| impact_load tersedia | train/test tidak missing; range train 22,6469–157,9725 dan test 4,3402–157,9725; KS D=0,0943 | Terverifikasi hanya pada classification; ketersediaan regression UNKNOWN |
| Format heterogen | telemetry: inactive/false/N/0 dan active/true/Y/1; kolom spread/scope_trace dibaca object dengan bentuk campuran | MEDIUM: pertahankan raw, dokumentasikan parser dan perlakuan tanda perkiraan; belum dilakukan cleaning |

Proporsi string teks utuh baru di test adalah 100% pada `signal_stream` nonmissing. Ini perbandingan nilai teks lengkap, **bukan** bukti token vocabulary drift atau kerusakan data.

## Leakage dan batas otoritas

Asal dan waktu pembentukan `audit_lane`, `checksum_hint`, `impact_load` serta `entry_friction` belum diketahui. Audit asosiasi marginal tidak menemukan sinyal kuat pada checksum (Pearson dengan target ≈0,00172); ini tidak membuktikan bebas leakage. Belum ada bukti apakah fitur terbentuk sebelum target, dari target, atau tersedia pada waktu prediksi. **HIGH / confidence tinggi pada gap evidence:** minta data dictionary/timing dari asisten sebelum menetapkan feature allowlist.

Rekonsiliasi read-only dengan TP1 berdasarkan ID menunjukkan semua kolom bersama classification sama: 50.608 ID train dan 29.421 ID test, seluruh mismatch count = 0. TP1 classification menargetkan entry_friction; TP2 menjadikannya fitur dan menambah impact_load serta target artifact_signal. Jangan melakukan join label tambahan lintas-task. Perubahan ini membuat recipe TP1 bukan pilihan otomatis TP2.

Target test tidak tersedia, sehingga target shift train/test tidak bisa diuji. Calendar cycle, semantik record, unit/range resmi dan lineage fitur tetap UNKNOWN. Tidak ada klaim sebab-akibat, statistical significance, atau sertifikasi readiness modeling dari profil ini.

## Regression dan Samuel

Tidak ditemukan CSV regression di seluruh inventaris file TP2 lokal. Dibutuhkan tiga file resmi di `data/raw/regression/`: **Train.csv, Test.csv, SampleSubmission.csv**. Target `threat_rank`, definisi ordinal/kontinu/ranking dan rentang resminya belum terverifikasi. Data classification tidak digunakan sebagai pengganti.

Notebook Samuel telah dieksekusi dengan status blocked. Kode analisis disiapkan untuk Pearson/Spearman, hexbin, variasi target dalam bin impact yang sama, tiga periode cycle terurut, korelasi per cycle/periode, dan flag outlier. Jalur analisis regression lengkap belum dapat dijalankan atau diperiksa visualnya tanpa file resmi. Tidak ada angka relasi atau grafik yang direkayasa; belum ada dasar untuk mengganti pertanyaan EDA atau mengusulkan alternatif yang lebih kuat.

## Reproduksi dan langkah berikutnya

`python scripts/audit_data.py` menghasilkan profil task yang tersedia dan laporan blocked untuk task yang tidak ada. Untuk mereproduksi perbandingan TP1 opsional, tambahkan `--tp1-data-dir` dengan root raw TP1 resmi milik pengguna. Notebook audit adalah companion inspectable untuk perhitungan dan visualisasi kualitas data.

Tambahkan file regression resmi dan data dictionary, jalankan ulang audit dan notebook Samuel, lalu tinjau target/schema/timing fitur. Setelah itu sepakati metric rubric dan validation strategy sebelum training. Dokumen 27 Oktober serta audit JSON lama dipertahankan sebagai riwayat, bukan otoritas readiness saat ini.
