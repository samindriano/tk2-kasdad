# Pembagian kerja TP 2 — usulan final

Tanggal: 27 Oktober 2026. Kelompok A06 kasdud lovers; dataset Cyber Vulnerability Lifecycle.

Pembagian ini adalah rancangan kerja untuk disepakati anggota, bukan bukti pekerjaan telah diselesaikan. Setiap anggota merumuskan dan memahami pertanyaan EDA-nya sendiri, serta mengerjakan satu pipeline klasifikasi dan satu regresi end-to-end.

## EDA

| Anggota | Fokus dan rumusan awal | Batas |
|---|---|---|
| Samuel | Bagaimana hubungan impact_load dan threat_rank? Pada tingkat dampak teknis yang serupa, seberapa lebar rentang prioritas respons? | Fokus bentuk hubungan dan variasi prioritas; bukan mengevaluasi tambahan signal_stream. Memerlukan data regresi resmi. |
| Rousan | Bagaimana proporsi artifact_signal berbeda menurut entry_friction, dan apakah perbedaannya terlihat pada kelompok impact_load yang berbeda? | Fokus hubungan akses dan indikasi artefak; bukan analisis pola missing. |
| Abhi | Bagaimana scope_pressure berkaitan dengan threat_rank pada kelompok impact_load yang serupa? | Fokus cakupan terhadap prioritas, dengan impact_load sebagai konteks. Tidak menjawab pertanyaan Samuel tentang bentuk hubungan dampak-prioritas. Memerlukan data regresi resmi. |
| Rizky | Bagaimana proporsi artifact_signal berbeda menurut telemetry_flag dan module_spread? | Fokus telemetry dan sebaran modul; tampilkan missing sebagai kelompok tersendiri dan jelaskan normalisasi format. Bukan studi perubahan cycle atau validasi acak vs temporal. |

Pertanyaan akhir harus disusun sendiri oleh masing-masing anggota. Gunakan train berlabel saja untuk analisis hubungan target. Cantumkan ukuran kelompok, ketidakpastian yang relevan dan keterbatasan; jangan menyatakan hubungan sebagai sebab-akibat. Sampling dan batas kelompok perlu dinyatakan sebelum melihat hasil; hindari kesimpulan dari sel yang kecil.

## Arah pendekatan individu — berlaku untuk kedua task

- Samuel: pendekatan teks dan terstruktur, dengan pembandingan terstruktur saja vs penambahan representasi signal_stream pada split yang sama. Ini pengembangan pipeline TP 1; EDA utama berbeda.
- Rousan: pipeline model pohon dan pendekatan representasi missingness/kategori eksplisit; bandingkan dengan perlakuan sederhana. Ini pilihan preprocessing untuk kedua task, bukan pengulangan pertanyaan EDA lama.
- Abhi: pipeline terstruktur yang mengeksplorasi fitur cakupan dan interaksinya dengan dampak; bandingkan dengan fitur dasar. Untuk klasifikasi, keputusan threshold dikalibrasi menggunakan data masa lalu dan dilaporkan terpisah.
- Rizky: pipeline yang mempelajari sinyal telemetry dan spread dengan parsing terdokumentasi, lalu bandingkan dengan baseline tanpa fitur tambahan tersebut. Analisis per kelompok mendukung interpretasi tanpa mengklaim sebab-akibat.

Pilihan model, fitur, dan eksperimen harus benar-benar dikerjakan sendiri, dicatat, dan dijelaskan oleh pemilik; ini bukan jaminan kenaikan skor. Perbedaan bermakna perlu ditinjau setelah rancangan konkret tersedia.

## Aturan bersama dan urutan

1. Dapatkan dan audit Train.csv, Test.csv, dan SampleSubmission.csv regresi. Saat rencana dibuat, file regresi resmi belum tersedia lokal, jadi dua EDA regresi dan skema belum terverifikasi.
2. Sepakati pembagian waktu dan metrik bersama untuk perbandingan utama: macro F1 untuk classification, MAE untuk regression. Train mencakup cycle 47–79 dan test classification 80–91; jangan pakai test untuk memilih model.
3. Fit imputasi, encoding, vektor teks, seleksi fitur, dan kalibrasi hanya pada data train yang sesuai di setiap split. Simpan data raw utuh.
4. Masing-masing: EDA dengan visualisasi dan interpretasi; preprocessing, model, evaluasi, alasan pengembangan TP 1, dua notebook. Ulas pendekatan bersama untuk memastikan berbeda secara bermakna.
5. Satu perwakilan menyiapkan dua ZIP sesuai PPT setelah masing-masing notebook berhasil Restart & Run All. Submission Kaggle hanya melalui tim.

Kaitan TP 1: Rizky mengerjakan random vs validasi waktu; Rousan missingness dan cycle; Samuel nilai tambah signal_stream; Abhi sensitivitas threshold. TP 2 dapat memakai pelajaran tersebut, tetapi pertanyaan utama EDA di atas tidak sekadar mengganti label pada pertanyaan sebelumnya.

Tenggat yang tertera di Kaggle adalah 8 November 2026 23.55 WIB. PPT menyebut Jumat untuk tanggal itu, padahal 8 November 2026 adalah Minggu; cek perubahan resmi.
