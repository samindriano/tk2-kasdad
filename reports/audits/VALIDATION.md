# Verifikasi setup — 30 Oktober 2026

> Riwayat validasi saat regression belum tersedia. Validasi lanjutan ada di `REGRESSION_VALIDATION.md`.

- Audit CLI dijalankan terhadap tiga CSV classification aktual. Independent read-only audit mengonfirmasi shape, target, class counts, duplicate/ID checks dan missingness utama.
- `reports/audits/*_audit.json` dihasilkan kode, bukan hasil yang diisi manual. Reconciliation TP1 memakai join ID unik dan pengecekan nilai/NA semua kolom bersama; tidak memodifikasi TP1.
- Notebook audit dan notebook Samuel dieksekusi top-to-bottom dengan nbclient, lalu divalidasi nbformat. Audit notebook memiliki dua output figure; notebook Samuel menampilkan blocker tanpa relasi target atau figure palsu.
- Figure missingness serta target/cycle audit diperiksa visual pada ukuran baca normal: label terbaca, angka cocok dengan agregat, tidak ada label terpotong. Export HTML kedua notebook berhasil. **Preview HTML penuh belum diperiksa secara visual:** browser menolak URL file lokal menurut kebijakan protokolnya. Tidak dilakukan bypass; verifikasi visual terbatas pada figure PNG yang diekstrak dari output notebook dan pemeriksaan struktur/output notebook.
- Python compilation dan unit checks mencakup input hilang, override path, ambiguitas nama, serialisasi profile tanpa observasi, target schema yang mewajibkan ID sample, serta hash/shape/class counts terhadap data aktual. Tidak digunakan fabricated observations.
- Lima unit checks lulus. Kedua CLI juga dijalankan dari direktori temporary di luar repository dengan data/output override; profil classification dan status blocked regression tetap benar.
- Jalur regression lengkap, grafik relasi Samuel, dan interpretasi numeriknya **belum terverifikasi** karena tidak ada ketiga input resmi. Smoke check helper dari kolom classification aktual hanya memeriksa software dan menolak `artifact_signal` sebagai target regression; tidak dianggap evidence regression.
- Warning Windows dari zmq terkait event loop muncul saat eksekusi notebook; eksekusi selesai. Bukan kegagalan analisis.

Pemeriksaan akhir memastikan raw CSV diabaikan Git, source hash tetap cocok, dan hanya file setup, kode, notebook, dokumentasi, placeholder serta laporan agregat yang masuk commit. Status COMPLETE dalam audit berarti profiling selesai, bukan admission otomatis untuk modeling atau sertifikasi bebas leakage.
