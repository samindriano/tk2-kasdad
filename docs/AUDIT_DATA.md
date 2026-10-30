# Audit kesiapan data — 27 Oktober 2026

## Klasifikasi: siap memulai EDA dan merancang pipeline; preprocessing masih diperlukan

Sumber adalah file resmi klasifikasi TP 2 yang diunduh dari Kaggle. Detail dan SHA-256 tersedia dalam reports/classification_data_audit.json. File mentah tidak diubah.

- Train 50.608 × 19, test 29.421 × 18, sample 29.421 × 2. Skema fitur train dan test cocok dan urutan ID sample sama dengan test.
- Tidak ada duplikasi baris atau record_id dalam masing-masing file; tidak ada ID yang sama antara train dan test. Target tidak kosong: kelas 0 sebanyak 33.038 dan kelas 1 sebanyak 17.570. Tidak ditemukan infinity pada kolom numerik terbaca.
- Train cycle 47–79, test 80–91. Rancang validasi berdasarkan waktu untuk mencerminkan kondisi test.
- Missing train: org_code 2.686, module_code 4.044, pattern_code 8.183, signal_stream 3.404, scope_trace 4.232, org_spread 4.358, module_spread 4.159 dan telemetry_flag 5.703. Test juga memiliki missing.
- scope_trace dan spread memiliki format seperti 1, 1.0, 001, ~1 dan 1 nodes; telemetry_flag memakai inactive, false, N, 0, active, true, Y dan 1. Pertimbangkan makna perkiraan ~ sebelum menyamakan nilai.
- impact_load dan entry_friction tidak missing di train maupun test. Semua 16 kolom dasar train/test cocok dengan raw TP 1 setelah disejajarkan menurut record_id; kedua fitur TP 1 pada train cocok dengan target train TP 1. Ini bukan bukti bebas leakage.
- Perlu pemeriksaan domain, parsing, missingness, kategori baru dan validasi pipeline. Jangan menghapus baris hanya karena missing. Jangan memakai test untuk memilih model.

## Regresi: belum dapat dinilai

Kaggle mendeskripsikan target threat_rank dan jumlah baris, tetapi daftar file tidak tampil pada halaman Data setelah bergabung dan dimuat ulang. File regresi belum diunduh; kualitas, skema dan kesiapan aktualnya belum diverifikasi. Jangan menggantinya dengan data klasifikasi atau mengasumsikan target tersedia dalam TP 1.

Dapatkan file resmi regresi dan ulangi audit. Simpan raw apa adanya; fit transformasi yang dipelajari pada bagian train tiap fold. EDA dan desain validasi dapat dimulai, tetapi kesiapan kedua task belum lengkap.
