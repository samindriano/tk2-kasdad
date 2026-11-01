# Disposisi riset seluruh 45 kartu

Snapshot `49b3fdf`, 1 November 2026. Angka gain positif berarti membaik. Kolom P/H memakai view masing-masing kartu: STANDARD W1+W4/H12; LATE W4+W6/H12_L; CACHE_SIX primary enam-origin/H12. Jangan meranking antarprofile. Semua verdict historis dipertahankan.

Sumber per baris: [registry](../batch1/batch_ledger.json) menghubungkan exact result variant, freeze, gate dan QA; [queue](../../../experiments/optimization_v3/research_backlog/queue.json) mengunci recipe. [Diagnostics](diagnostics.json) menghitung ulang 35 STANDARD; [review utama](REVIEW.md) menjelaskan sisa coverage dan batas audit.

## Classification — 20 kartu

| ID | Gain P / H | Apa yang benar-benar diuji dan dipelajari | Disposisi review / rescue |
|---|---:|---|---|
| C01 | −0,002174 / −0,002175 | CACHE_SIX: mengganti missing branch dengan structured HGB; missing W1 F1 −0,0791, recall −0,145. Bukan semua routing buruk. | RETIRED kuat; pertahankan CatBoost fallback, jangan rescue route ini. |
| C02 | −0,000534 / +0,000089 | Fit hybrid LR hanya stream-present tidak memberi primary gain. Mengeluarkan missing dari fit tidak otomatis membantu. | RETIRED recipe; tidak perlu mengulang fit-population grid. |
| C03 | −0,000319 / −0,000471 | Membagi kuota unigram/bigram tidak membantu recipe LR ini. | RETIRED; tidak membatalkan semua n-gram, tetapi tak ada alasan rescue khusus. |
| C04 | −0,000212 / +0,000810 | Vocabulary 60k memberi H12 sedikit lebih baik, primary tidak. | RETIRED; widening saja bukan bottleneck yang terbukti. |
| C05 | −0,001260 / −0,000284 | min_df lebih rendah pada vocabulary lebar memperburuk kedua aggregate. | RETIRED; jangan menambah token langka tanpa diagnosis. |
| C06 | −0,000576 / +0,000092 | Novelty kategori dihitung strictly prior-cycle; chronology benar, informasi belum membantu. | RETIRED recipe; bukan kegagalan teknis atau bukti kategori tak relevan. |
| C07 | −0,000181 / +0,000216 | Frekuensi/rarity kategori memberi perubahan kecil berlawanan antarview. | RETIRED; fitur deskriptif belum memberikan gain material. |
| C08 | +0,000128 / −0,000911 | Target rates kronologis tersmoothing tidak mencapai gain dan mendekati batas harm H12. | RETIRED; jangan longgarkan smoothing/threshold dari label outer yang sama. |
| C09 | +0,000110 / −0,000168 | Interaksi TF-IDF×EF pada LR; W1 negatif, W4 positif. Kontrol awal salah scope, perbandingan final row-matched telah direkonsiliasi. | RETIRED valid dengan catatan QA khusus; bukan bukti interaksi kausal. |
| C10 | +0,000343 / +0,000878 | Blok trigram memberi sinyal kecil, belum mencapai +0,0005. | RETIRED; near-miss bukan alasan menurunkan gate. Reserve hanya jika diagnosis independen muncul. |
| C11 | +0,000661 / −0,000310 | Word+char sparse boosting lolos tipis; late F1 −0,000947, ranking AUC/AP juga sedikit turun. | ADVANCE eksploratif family kedua; tidak ditune ulang. Nilainya untuk membuktikan/menolak complementarity. |
| C12 | +0,001177 / −0,001155 | Char-only memenuhi primary tetapi melampaui H12 harm; late −0,003102. | RETIRED; kata lengkap tampak penting dalam perbandingan ini. Tidak rescue lewat threshold post hoc. |
| C13 | −0,002459 / −0,002541 | Structured+SVD32 HGB gagal. Hilangnya informasi atau learner mungkin berperan; tidak teridentifikasi terpisah. | RETIRED; tidak menyimpulkan SVD selalu buruk karena learner/task lain berbeda. |
| C14 | −0,006724 / −0,002919 | Random partitions+LR merugikan primary, H12, window dan subgroup. | RETIRED kuat; tidak ada rescue murah yang didukung. |
| C15 | −0,001339 / +0,003281 (LATE) | Dua-score prequential metahead membantu long horizon tetapi merugikan W4/W6 dan positive recall text-load tinggi. | RETIRED; gain H12_L tidak boleh menutupi kegagalan policy atau dibandingkan dengan STANDARD. |
| C16 | +0,001971 / +0,002837 | Kapasitas/regularisasi sparse model membantu; late masih +0,000830, jauh lebih kecil dari early. Banyak parameter berubah bersama. | ADVANCE utama; lanjut recipe asli, jangan atribusikan gain pada satu parameter. |
| C17 | +0,001541 / +0,001411 | Family sama, delta berkorelasi 0,790 dengan C16; late hampir nol. | ADVANCE reserve, tidak continuation otomatis karena redundant dan kalah prioritas pooled. |
| C18 | −0,000045 / −0,000496 | Random-threshold sparse boosting tidak membantu. Gate primary subgroup pass, detail W1 Q3 gagal F1 −0,010471. | RETIRED tetap benar; scope guard perlu diungkapkan, bukan klaim seluruh-view lolos. |
| C19 | +0,000375 / +0,004152 | Dense XGB auxiliary blend memberi late +0,005160 dan ranking lebih baik, tetapi primary kurang dan Q4 recall gagal. | RETIRED dengan hipotesis horizon menarik; archive, bukan rescue cepat menggunakan label yang sama. |
| C20 | −0,001573 / −0,004157 | CACHE_SIX q80 volume routing: tidak satu pun enam-window nonnegatif. | RETIRED kuat; routing berdasarkan volume bukan perbaikan yang terbukti. |

## Regression — 25 kartu

| ID | Gain P / H | Apa yang benar-benar diuji dan dipelajari | Disposisi review / rescue |
|---|---:|---|---|
| R01 | +0,034940 / +0,076438 | LinearSVR L1 membantu aggregate, tetapi W4 target≥95 harm +0,4600; late aggregate −0,00531. | RETIRED karena risiko, bukan tanpa signal. Missing routing tidak menyelesaikan tail-high; tidak diprioritaskan. |
| R02 | +0,022196 / +0,010084 | Vocabulary sparse lebih besar memberi gain kecil di bawah materiality gate. | RETIRED; widening lanjutan bernilai rendah. |
| R03 | +0,013836 / +0,013678 | Split unigram/bigram membantu sedikit, belum +0,03. | RETIRED; menunjukkan alokasi ini bukan perbaikan besar, bukan semua bigram tak berguna. |
| R04 | +0,021887 / +0,014318 | Hashing tidak cukup membantu. Kegagalan bridge prefit disimpan, retry menghasilkan evidence valid. | RETIRED performa; jangan salah menggolongkan technical attempt sebagai penyebab verdict final. |
| R05 | +0,014812 / +0,016808 | Binary token presence tanpa IDF memberi perbaikan kecil. | RETIRED; tidak perlu presence/count grid. |
| R06 | +0,061726 / +0,103260 | Char sparse L1 LightGBM: 24/24 cycle positif, late +0,081977; missing H12 harm +0,1542. | ADVANCE prioritas; jangan klaim semantik substring. Route missing reserve, bukan modifikasi recipe utama. |
| R07 | −0,084132 / −0,085097 | NMF32 replacement gagal kedua aggregate. | RETIRED; tidak membuktikan semua latent representation gagal. |
| R08 | −0,097574 / −0,121789 | Text scalar tambahan dalam recipe ini merugikan aggregate dan tail. | RETIRED; fitur EDA relevan secara asosiasi belum tentu menambah prediksi. Jangan menghapus EDA. |
| R09 | −0,036425 / −0,081655 | LightGBM Huber merugikan aggregate dan tail. | RETIRED; counterexample R22 berarti robust-loss family tidak bisa ditutup universal. |
| R10 | Tidak diuji | Fair conditional pada R09; syarat tidak terpenuhi, nol fit. | SKIPPED, bukan measured failure. Tidak override condition demi melengkapi tabel. |
| R11 | +0,031602 / +0,008817 | Sembilan ordinal heads+median: tails/missing membantu, late aggregate +0,00433; 18 predictor fits. | ADVANCE reserve; biaya dan gain kalah dibanding R22/R06. Bukan bukti pendekatan TP1 otomatis unggul TP2. |
| R12 | +0,245801 / +0,284490 | LambdaRank whole-cycle percentile membantu aggregate/tails; missing W1/primary/H12 gagal guard. | RETIRED asli dipertahankan. **Prioritas rescue tunggal:** present-only setelah full-cycle mapping, dengan champion fallback dan amendment baru. |
| R13 | Tidak diuji | CatBoost ranker butuh R12 Spearman naik kedua origin dan MAE aggregate gagal. Syarat majemuk tidak terbukti/terpenuhi. | SKIPPED substantif, bukan sekadar kolom Spearman hilang. Jangan otomatis aktifkan. |
| R14 | +0,043122 / +0,026910 | EF1 specialist membantu aggregate, tetapi H12_L high-tail harm +0,555867. | RETIRED; routing missing tidak menyelesaikan high-tail. Target-conditioned routing dilarang sebagai recipe prediksi. |
| R15 | +0,007600 / +0,004388 (LATE) | Structured missing expert 10% blend tidak memberi gain cukup. | RETIRED; tidak membatalkan fallback ke incumbent, karena itu policy berbeda tanpa specialist baru. |
| R16 | +0,015729 / +0,017173 | Half-life8 weighting primary kurang; late H12 +0,099974 vs early negatif. | RETIRED asli; reserve mekanisme recency/history, bukan prioritas di atas kandidat konsisten. |
| R17 | +0,032732 / +0,031992 (LATE) | Affine OOF L1 calibration membantu aggregate tetapi low-tail gagal. | RETIRED risk tradeoff; jangan longgarkan guard. Monotone calibration bukan perbaikan ranking. |
| R18 | +0,014361 / +0,027931 (LATE) | Past4-cycle median residual offset: gain kecil, tail-low merugi. Nol predictor fit by design. | RETIRED; offset universal memiliki konflik mekanis dengan sebagian ekstrem. |
| R19 | +0,012735 / +0,017895 (LATE) | Past8-cycle EF-specific median offset tetap tidak cukup dan tail guard gagal. | RETIRED; melokalkan offset per EF belum menyelesaikan konflik tail. |
| R20 | −0,105678 / −0,115513 | Fit-only supervised token selection128 merugikan aggregate dan tail. | RETIRED; tidak membuktikan text identity tidak informatif, hanya selection/compression recipe ini gagal. |
| R21 | +0,030828 / +0,044280 | XGB absolute auxiliary lolos tipis, missing merugi, late +0,018634. | ADVANCE reserve, kalah R22 family sama. Present-only masuk pelajaran umum, bukan rescue pertama. |
| R22 | +0,098176 / +0,124873 | XGB pseudo-Huber slope5 auxiliary: late +0,100466, 23/24 cycle positif. | ADVANCE utama; loss recipe mengungguli R21 pada pilot, bukan bukti semua smooth losses unggul. |
| R23 | +0,092085 / +0,111498 | ExtraTrees L1 memberi signal namun missing/tail-high gagal. Routing missing menyelesaikan pilot guard secara post hoc, late gain hanya +0,022774. | RETIRED asli; rescue reserve setelah R12, bukan menggusur R06 yang late-nya lebih kuat. |
| R24 | +0,003289 / +0,002348 | Interaksi TF-IDF×EF pada Ridge nyaris netral. | RETIRED; interaksi EF linear ini belum memberi informasi material. |
| R25 | +0,017347 / +0,020332 | Unnormalized counts memberi gain kecil di bawah gate. | RETIRED; multiplicity-aware, jangan turunkan ambang demi menghidupkan near-miss. |

## Cara membaca disposisi

"RETIRED" hanya berlaku pada recipe/protokol yang diuji. "Reserve" berarti pelajaran/hipotesis disimpan, bukan tugas training otomatis. "Rescue" memerlukan perubahan dengan alasan spesifik, ID baru dan prospective evaluation; hasil lama tidak boleh diganti. Klaim source QA lulus adalah status historis yang diperiksa, bukan garansi semua interpretasi laporan benar.
