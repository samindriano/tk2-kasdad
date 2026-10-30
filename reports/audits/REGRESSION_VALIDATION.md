# Validasi audit regression dan discovery — 30 Oktober 2026

## Scope dan sumber

Data resmi yang diberikan pengguna: archive regression di Downloads, dipindahkan ke ignored raw TP2. CRC archive lolos; SHA-256 compressed dan tiga CSV disimpan di `regression_acquisition.json`. Unit checks merekonsiliasi byte extracted CSV dengan member archive dan hash laporan. Raw classification yang sudah ada tetap cocok dengan hash audit awal. PDF lokal hanya dibaca untuk definisi kolom bersama; sumber tidak diunggah.

TP1 dibaca dari canonical checkout dan delivery individual. HEAD serta hash laporan historis di `reports/eda/tp1_regression_history.json`; tidak ada model TP1 yang dijalankan ulang. Status TP1 dipertahankan termasuk file untracked yang sudah ada. Semua error/MAE TP1 dalam discovery adalah kutipan/parafrase saved development evidence, bukan pengukuran sesi baru.

## Pemeriksaan yang dijalankan

- Audit classification dan regression berhasil dijalankan; keduanya COMPLETE untuk profiling, bukan sertifikasi bebas leakage. Target regression, sample alignment, ID disjointness, missingness, duplicate vectors dan nonfinite diperiksa.
- Fitur classification/regression direkonsiliasi read-only menurut 50.608 ID: 17 kolom fitur bersama cocok seluruhnya, termasuk NA; kedua file test memiliki SHA-256 identik.
- `samuel_eda.py` dan discovery berjalan pada seluruh 50.608 train regression. Tidak ada sampling untuk angka utama. Output hanya agregat/figures, tanpa raw rows, record ID atau text stream.
- Review independen mengonfirmasi shapes, target, correlations, tiga blok cycle, tail counts, repeated-impact medians, konsentrasi target TP1 dan exact rank geometry. Tidak ditemukan mismatch faktual pada lima kandidat dan rekomendasi yang tetap belum final.
- Sembilan unit checks lulus, termasuk perhitungan ulang percentile dengan **scipy rankdata** secara terpisah dari implementasi pandas, perbandingan correlation memakai numpy, keutuhan cycle per blok, SHA/CRC dan sample schema. Tidak ada fabricated observations.
- Notebook audit dan diagnostik Samuel dieksekusi top-to-bottom dengan nbclient dan valid nbformat; discovery memiliki notebook sendiri. Seluruh output tersimpan. Python compilation dan pemeriksaan whitespace Git dijalankan sebelum commit.
- Notebook akhir: audit 0 error/2 figures; diagnostik Samuel 0 error/3 figures; discovery 0 error/5 figures. Export HTML discovery berhasil, sementara batas inspeksi visual penuh tetap seperti di bawah.
- Portabilitas TP1 opsional diperiksa pada data TP2 aktual: path TP1 yang tidak ada menghasilkan status NOT_FOUND dan tetap menghitung 50.608 baris, repeated-impact summaries serta empat figure TP2. Notebook juga dieksekusi dengan TP1_DATA_DIR tidak tersedia tanpa error. Artifact akhir dikembalikan ke referensi TP1 nyata (INCLUDED, lima figures). CLI tanpa referensi menandai NOT_PROVIDED; tidak ada fallback label/model.

## Visual dan interpretasi

Lima figure discovery serta tiga figure diagnostik diperiksa langsung pada PNG: axis, nilai skor, quantile spread, label dan legend terbaca. Kronologi panel diagnostik diperbaiki menjadi cycle47–57,58–68,69–79; period labels pada figure discovery dipetakan ke range cycle. P10/P90 dan IQR adalah sebaran observasi, bukan confidence interval. Tidak ada klaim uncertainty/signifikansi dari gambar.

Verifikasi visual dilakukan pada PNG dan pemeriksaan struktur/output notebook. Preview HTML keseluruhan belum diperiksa visual karena browser lokal sebelumnya menolak protokol file; tidak dilakukan bypass. Keberhasilan eksekusi notebook tidak disebut sebagai bukti visual penuh.

Interpretasi rank geometry secara eksplisit membandingkan target dengan rank target itu sendiri, bukan fitur prediktif atau bukti lineage. Ketidakpastian definisi resmi threat_rank, kalender cycle, timing dan provenance fitur tetap dicatat. Tidak ada klaim over/underprediction TP2, causal drift, atau pertanyaan final yang sudah disetujui.

Tidak ada training, hyperparameter tuning, model artifact, submission atau upload raw. Commit hanya memuat kode deskriptif, laporan agregat, notebook dengan figures, dokumentasi dan checks.

**Review integration: APPROVE** untuk scope audit/discovery ini, berdasarkan rekonsiliasi angka independen, sembilan checks yang lulus, eksekusi notebook dan QA figure. Tidak berarti approval pertanyaan final, model, validation protocol atau readiness bebas leakage.
