# Rancangan eksperimen regression — menunggu review pengguna

30 Oktober 2026 · Tidak ada eksperimen model pada dokumen ini yang sudah dijalankan. Pertanyaan EDA utama sudah dipilih pengguna; model final, fitur final, validation protocol dan eksperimen tetap belum diputuskan.

## Evidence yang memotivasi rancangan

EDA formal menemukan hubungan impact–threat yang lebih kuat pada periode akhir, termasuk setelah distribusi exact impact × entry_friction dibuat sama. Namun EF0 berhenti menguat setelah blok tengah, EF1 masih menguat, dan conditional shifts berbeda arah. Target mempunyai geometri average-percentile per-cycle; marginal stabil tidak menghilangkan risiko perubahan hubungan kondisional. Laporan dan angka: `reports/eda/SAMUEL_FORMAL_EDA.md`.

## Rancangan evaluasi untuk disepakati dahulu

**Usulan utama:** expanding-window dengan cycle utuh. Enam window empat cycle: fit 47–55 → val 56–59; fit 47–59 → val 60–63; fit 47–63 → val 64–67; fit 47–67 → val 68–71; fit 47–71 → val 72–75; fit 47–75 → val 76–79. Validation blocks ini tidak overlap satu sama lain; fit sets bertumbuh dan saling overlap. Ukuran window bukan hasil tuning. Laporkan tiap cycle dan equal-cycle mean, serta row-pooled metric karena ukuran cycle tidak sama.

**Sensitivitas horizon:** bandingkan val 12 cycle (fit 47–55 → val 56–67 dan fit 47–67 → val 68–79), karena test resmi mencakup 12 cycle. Ini bukan holdout baru: semua label train sudah dipakai EDA. Jika dipakai bersama window empat cycle, validation rows akan overlap; jangan menghitungnya sebagai bukti independen atau memilih konfigurasi dari banyak horizon tanpa mencatat reuse.

**Random split:** satu pembanding diagnostik untuk melihat perbedaan estimasi ketika periode lama/baru tercampur. Jangan menjadikannya dasar tunggal pemilihan model untuk test 80–91. Jangan mencampur preprocessing, target aggregate atau tuning antara validation dan fit.

**Metric yang diusulkan:** MAE jika sesuai rubric resmi TP2; konfirmasi sebelum training. Laporkan juga mean signed error (pred−truth), RMSE diagnostik, Spearman dalam tiap cycle, dan MAE/bias pada y≤5,5<y<95,y≥95. Semua segmentasi target hanya untuk evaluasi, bukan fitur. Denominator tiap segmen harus dicatat; tercile TP1 bukan tail 5% TP2. Untuk quantile experiment, tambahkan pinball loss dan coverage P10–P90 pada validation.

Bekukan input hashes, fold definitions, metric, fitur allowlist dan recipe pertama setelah review. Semua imputasi, parsing yang dipelajari, encoding, TF-IDF, support target, threshold dan calibration dipelajari pada fit fold saja. Tidak ada target test. Jangan mengubah label resmi atau merangking ulang label pada fold.

## Eksperimen yang diusulkan

| Prioritas / ID | Hipotesis dan metode | Perbandingan / apa yang dipelajari | Risiko dan syarat |
|---|---|---|---|
| **P0 / R0 — baseline interpretabel** | Global median train, lalu median per impact dan impact×EF dengan fallback train-global untuk group kecil/baru. Seluruh tabel dipelajari di fit fold. | Evaluasi seberapa jauh lookup kondisional sederhana memanfaatkan support diskret impact dan heterogenitas regime; bandingkan semua recipe pada fold sama. | Kelompok langka/unseen membutuhkan fallback yang dibekukan; jangan membuat table dari gabungan fit+validation. Median target utama selaras MAE tetapi belum dibuktikan unggul di TP2. |
| **P1 / R1 — respons nonlinear terstruktur** | Satu model pohon regresi/boosting dengan loss absolute/quantile 0.5, konfigurasi kecil tetap; mulai hanya impact,EF. Bandingkan dengan bentuk linear sederhana. | Apakah nonlinear shape dan interaksi impact–EF memberi manfaat di cycle baru di luar lookup/linear baseline? | Eta² EDA adalah association in-sample, bukan alasan menjanjikan skor. Jangan memaksakan monotonic impact global karena mixture/reversal. Belum memilih library/model final. |
| **P1 / R2 — waktu dan training history** | Pada keluarga model yang sama: tanpa cycle vs cycle numerik terurut; all-past vs recent-cycle training sebagai ablation terpisah. | Bedakan manfaat time information dari perubahan training population; terutama cek EF0/EF1 dan late cycles. | Cycle numerik di test melampaui train; pohon tidak mengekstrapolasi tren linear dengan sendirinya. Jangan memilih breakpoint 56 atau recency window dari EDA lalu mengklaimnya blind; pilihan window perlu dikunci/review dan tuning jika ada hanya di inner past folds. |
| **P2 / R3 — median dan conditional spread** | Quantile 0.1/0.5/0.9 pada label numerik asli, atau discrete bins sederhana yang ditentukan dari train sebagai eksperimen alternatif. | Uji pelajaran TP1 bahwa distribusi/ambiguity bermanfaat: apakah conditional spread dan tail risk terbaca, dan apakah median valid lebih baik? | TP1 memiliki 80 target values, TP2 memiliki 34.333; multiclass satu kelas per nilai TP2 tidak diusulkan. Quantile crossing, sparse tails dan rounding perlu diperiksa. Tidak menganggap interval sebagai confidence of mean. |
| **P2 / R4 — target relatif per-cycle** | Pertama evaluasi raw bounded-score predictions dengan MAE dan within-cycle rank metrics. Batch percentile calibration dari **prediksi**, jika seluruh batch cycle tersedia, hanya sebagai opsi terpisah setelah review. | Bedakan kemampuan mengurutkan record dari kemampuan menempatkannya pada skala label 0–100. | Re-ranking prediksi memerlukan batch lengkap dan asumsi target/batch yang belum resmi; dapat memperburuk MAE atau ties. Jangan memakai label validation/test untuk meranking/calibration; jangan otomatis melakukan transductive step atau memperlakukan diagnostic target-on-target sebagai feature. |
| **P2 / R5 — nilai tambah text dan missingness** | Setelah baseline terstruktur terkunci: ablation +signal_stream representation dan missing indicators, ditambah evaluasi present/missing. | Apakah konteks text memperkecil ambiguity pada impact×EF yang sama dan bertahan pada forward validation? | Parser/vocabulary hanya fit fold; kategori/missingness test berbeda. Jalur Ridge fallback TP1 adalah pengalaman historis, bukan requirement TP2. Hindari mengulang stack/ensemble kompleks sebelum ablation sederhana menjawab hipotesis. |

R0 dan R1 layak menjadi putaran pertama yang kecil setelah review. R2 dipakai untuk menguji generalisasi waktu; R3/R4/R5 menunggu hasil putaran pertama dan konfirmasi sumber. Prioritas adalah rekomendasi diskusi, bukan pemilihan approach final.

## Hubungan TP1 dan hal yang tidak ditransfer

REG-Lite B mengestimasi distribusi pada support impact yang terkonsentrasi lalu mengambil conditional median; S6 signed-bias gate tetap gagal walau ada keuntungan MAE. Karena itu evaluasi mean signed error dan target-region tetap diperlukan di TP2. Phase 16 B2 menunjukkan bahwa regime evidence bisa membantu disambiguasi, tetapi ia memakai pipeline dan source labels TP1 berbeda. Pada TP2 entry_friction sudah menjadi fitur resmi; jangan menggabungkan label/OOF TP1 atau menyalin gate lama.

**Belum ada prediction bias TP2 yang terukur.** EDA hanya menunjukkan risiko conditional relationship mismatch dan support/measurement assumptions. Lebih kuatnya korelasi di train akhir tidak menjamin test lebih mudah, recent-window lebih baik, atau forward MAE tertentu.

## Keputusan yang diminta sebelum implementasi berikutnya

1. Setujui atau ubah validation windows, metric utama sesuai rubric, dan pelaporan equal-cycle vs row-pooled.
2. Pilih scope putaran pertama (rekomendasi R0 + R1), termasuk fitur yang ketersediaan waktunya sudah diyakini.
3. Konfirmasi definisi resmi threat_rank dan apakah scoring dilakukan sebagai batch cycle penuh; ini menentukan apakah R4 patut dicoba.
4. Setujui batas jumlah recipe awal dan kriteria melanjutkan R2/R3/R5. Jangan menetapkan model final sekarang.

Milestone ini berhenti di titik review tersebut. Tidak ada pipeline classification tambahan yang dimulai karena tidak diperlukan untuk menjawab pertanyaan Samuel.
