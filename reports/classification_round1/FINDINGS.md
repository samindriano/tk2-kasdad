# Samuel Classification Round 1 — hasil dan review point

30 Oktober 2026 · **Selesai untuk review; belum ada final model, Round 2 atau submission.**

## Kandidat sementara dan hasil seluruh recipe

**C2-B, structured + ordinary TF-IDF Logistic Regression pada threshold 0,50, memimpin primary pooled Macro F1: 0,8206.** Inner-selected threshold menghasilkan 0,8188 pada primary, tetapi sedikit lebih baik pada pooled H12. Karena itu pemilihan threshold policy masih terbuka; tidak ada final model atau klaim skor Kaggle.

| Recipe | Metode | Primary fixed F1 | Primary selected F1 | Fixed equal-cycle F1 | Pooled AP | H12 fixed F1 | H12 selected F1 |
|---|---|---:|---:|---:|---:|---:|---:|
| C0_A | Majority | 0,3894 | 0,3894 | 0,3868 | 0,3420 | 0,3894 | 0,3894 |
| C0_B | EF rate lookup | 0,3894 | 0,5469 | 0,3868 | 0,3800 | 0,3894 | 0,3894 |
| C0_C | LR impact + EF | 0,3894 | 0,5549 | 0,3868 | 0,4107 | 0,3894 | 0,5511 |
| C1_A | LR structured | 0,7145 | 0,7112 | 0,7121 | 0,6734 | 0,6768 | 0,6893 |
| C1_B | HGB structured | 0,7310 | 0,7333 | 0,7262 | 0,7086 | 0,6891 | 0,6963 |
| C2_A | TF-IDF + LR | 0,8014 | 0,8057 | 0,7916 | 0,8181 | 0,7819 | 0,7824 |
| C2_B | Structured + TF-IDF LR | 0,8206 | 0,8188 | 0,8124 | 0,8276 | 0,7886 | 0,7924 |

AP adalah average precision dari probabilities, sama untuk kedua threshold policies. Pooled AP memuat perbedaan probability antarhistory; lihat per-cycle metrics untuk konteks. Majority selected adalah alias control, tanpa threshold search. Semua C0 pada fixed 0,50 mempunyai prediksi identik: seluruhnya class 0, termasuk nol true positives.

Enam primary windows mencakup 38.588 unique development rows, cycle 56–79. Pooled Macro F1 dihitung ulang dari prediksi gabungan; equal-cycle F1 memberi bobot sama pada setiap cycle. H12 memakai kembali record primary sehingga bukan evidence independen tambahan. Seluruh Train telah tersedia dalam project exploration; ini bukan untouched holdout.

Seluruh tujuh model × enam window, kedua policies, H12_E/H12_L, precision/recall/F1 kedua kelas, confusion matrices, subgroup dan reliability bins berada di [comparison_tables.md](comparison_tables.md) serta `results.json`. Recipe, folds, parser, threshold grid dan reproduksi berada di `docs/CLASSIFICATION_ROUND1.md` dan frozen JSON; tidak diulang di sini.

## Apa yang dipelajari dari target dan fitur

Positive prevalence keseluruhan 34,72%, tetapi berubah non-monotonik: **30,64% → 41,01% → 32,25%** untuk cycle 47–57, 58–68, 69–79. EF1 lebih sering positif daripada EF0 (40,93% vs 31,87% pada full Train), tetapi keduanya di bawah 50%. Empirical impact thirds juga non-monotonik (33,10%, 29,14%, 43,34%). Ini asosiasi, bukan bukti kausal atau alasan memaksakan monotonic constraints.

Dua fitur resmi mempunyai ranking signal terbatas untuk target baru. C0-C pada threshold 0,50 tidak memprediksi positif; inner-selected threshold memperbaiki pooled F1 menjadi 0,5549, tetapi masih jauh di bawah structured/text models. Ini berbeda dari kuatnya conditional impact relationship pada regression. Fitur yang sah dan berguna pada satu target tidak otomatis cukup untuk target lain.

Expansion ke structured inputs meningkatkan fixed F1 dari 0,3894 menjadi 0,7145 pada LR. HGB menaikkannya lagi menjadi 0,7310, menang 4/6 windows dan 14/24 cycles terhadap LR pada fixed policy. Pada selected policy HGB menang 5/6 windows dan 18/24 cycles. Nonlinearity membantu secara terbatas; manfaatnya jauh lebih kecil daripada penambahan text.

Ordinary TF-IDF sudah memberikan nilai tambah besar. Controlled LR structured → hybrid meningkatkan fixed F1 **0,7145 → 0,8206**, menang 6/6 windows dan 23/24 cycles. Text-only → hybrid meningkat **0,8014 → 0,8206**, menang 6/6 windows dan 22/24 cycles. Class-1 F1 text-only 0,7337 menjadi 0,7692 pada hybrid; class-0 F1 0,8691 menjadi 0,8721. Recall class 1 meningkat 0,6675 → 0,7570, dengan precision turun 0,8145 → 0,7817.

Perbandingan ini menambahkan text block **beserta missing-stream indicator**; belum memisahkan kontribusi indicator, representasi text, perubahan coefficients, atau block scaling. Tidak ada NB-weighting atau stacking pada Round 1.

## Trade-off terbesar: missing stream

| Model, fixed 0,50 | Stream present F1 (n=36.010) | Stream missing F1 (n=2.578) | EF0 F1 | EF1 F1 |
|---|---:|---:|---:|---:|
| C1_A structured LR | 0,7106 | 0,7596 | 0,7116 | 0,7139 |
| C1_B structured HGB | 0,7265 | **0,7817** | 0,7334 | 0,7177 |
| C2_A text-only | 0,8239 | 0,3647 | 0,8079 | 0,7817 |
| C2_B hybrid | **0,8286** | 0,7104 | **0,8290** | **0,7982** |

Text-only memprediksi class 0 pada seluruh missing-stream rows dan melewatkan **1.098 positives** pada subset tersebut. Hybrid menyelamatkan sebagian besar kasus itu, tetapi tetap lebih buruk dari structured-only pada Macro F1 missing-stream.

Counterexample ini juga terlihat dari error counts: LR structured → hybrid mengurangi present-stream errors **9.862 → 5.609**, sementara missing-stream errors bertambah **617 → 743**. Total errors turun **10.479 → 6.352**. Error counts dapat dijumlahkan; Macro F1 sendiri tidak didekomposisi sebagai weighted subgroup mean.

Fenomena ini serupa dengan pelajaran Phase 11 TP1: zero text vector tidak menjamin output missing-stream tetap sama ketika model gabungan di-fit ulang. Namun, Phase 11 memakai NB weighting dan target berbeda. Pada TP2, missing indicator, regularization/block scale dan coefficients/intercept gabungan merupakan penjelasan alternatif yang belum dipisahkan. Temuan ini mendukung controlled ablation, bukan langsung mengadopsi recipe TP1 atau menolak text secara keseluruhan.

Hybrid meningkatkan F1 kedua EF secara pooled. Keuntungan lebih besar pada EF0, tetapi EF1 juga mendapat manfaat. Prevalence antar-subgroup berbeda; skor antar-subgroup tidak membuktikan penyebab kesulitan. Unknown/missing category groups dapat overlap dan tidak dijumlahkan.

## Threshold: bebas leakage tidak berarti selalu menguntungkan

Seluruh 36 threshold pilihan dihitung hanya dari inner temporal holdout. Semua history memenuhi coverage; tidak ada fallback yang diperlukan. Contoh hybrid thresholds untuk fit ends 55/59/63/67/71/75: **0,47 / 0,34 / 0,47 / 0,53 / 0,51 / 0,46**. Model kemudian di-fit baru pada outer history lengkap sebelum evaluasi.

Hybrid primary fixed → selected mengubah:

- Class-1 precision **0,7817 → 0,7579**, recall **0,7570 → 0,7849**.
- False negatives **3.397 → 3.007**, false positives **2.955 → 3.505**.
- Class-0 F1 **0,8721 → 0,8663**, class-1 F1 **0,7692 → 0,7712**.
- Macro F1 **0,8206 → 0,8188**; positive prediction rate **35,09% → 37,52%** versus actual 36,23%.

Trade-off ini menunjukkan threshold yang optimal di inner history belum tentu mentransfer ke outer cycle. Probability shift, prevalence, jumlah data dan ranking dapat berubah bersama; belum ada calibration fitting untuk mengatasi hal itu.

Baseline EF memberi peringatan tambahan tentang aggregate metric. Selected pooled F1 **0,5469**, tetapi equal-cycle hanya **0,4533**. W1 dan W4 memprediksi seluruh rows class 0; W2 memprediksi seluruh rows class 1. Pooled confusion mencampurkan keputusan yang berbeda antarwindow sehingga terlihat lebih baik daripada kebanyakan local windows. Pada kedua H12, selected EF kembali memprediksi seluruh rows class 0. Jangan mempromosikan baseline ini berdasarkan pooled score saja.

## Temporal generalization, H12 dan probability behavior

| Model | W1 fixed | W2 fixed | W3 fixed | W4 fixed | W5 fixed | W6 fixed |
|---|---:|---:|---:|---:|---:|---:|
| C1_A | 0,7200 | 0,7764 | 0,7285 | 0,7176 | 0,6605 | 0,6755 |
| C1_B | 0,7244 | 0,7662 | 0,7228 | 0,7241 | 0,7212 | 0,7037 |
| C2_A | 0,7493 | 0,8193 | 0,8037 | 0,8178 | 0,7937 | 0,7974 |
| C2_B | 0,7767 | 0,8471 | 0,8257 | 0,8409 | 0,8040 | 0,8056 |

Text/hybrid lebih kuat pada cycle akhir daripada structured-only. C1_A/C1_B melemah dibanding puncak W2, sedangkan C2_A/C2_B mempertahankan skor sekitar 0,79–0,81 di W5/W6. Ini berbeda dari Regression Round 1, ketika expanded model terutama memiliki counterexample H12 awal dan membaik di primary periode akhir. Pelajaran bersama adalah horizon sensitivity, bukan kesamaan pola penurunan kedua target.

Hybrid H12_E fixed/selected = **0,7759/0,7827** dan H12_L = **0,7994/0,8001**; text-only masing-masing **0,7669/0,7693** dan **0,7940/0,7933**. Hybrid tetap mengungguli text-only pada kedua H12, tetapi pooled gain fixed menyusut menjadi sekitar 0,0067. H12 selected hybrid 0,7924 adalah leader secondary pada putaran ini, tanpa mengubah pilihan primary menjadi model final.

Novelty kategori relevan, tetapi bukan penjelasan tunggal. Pada W6 subset semua kode dikenal/nonmissing, fixed F1 HGB **0,7182** versus hybrid **0,8126**; pada any-unseen, **0,6705** versus **0,7905**. Perbedaan dalam known subset tetap besar. Ini tidak mengisolasi kontribusi kategori, cycle atau fitur numeric.

Paired-history diagnostic memakai record validation identik dan fixed threshold yang sama, tanpa fit tambahan. Pada cycle 64–67, hybrid fit sampai 55 menghasilkan F1 **0,7539**, sedangkan fit sampai 63 menghasilkan **0,8257**; text-only **0,7522 → 0,8037**, HGB **0,6710 → 0,7228**. Pada cycle 76–79, hybrid fit sampai 67 menghasilkan **0,7777**, versus fit sampai 75 **0,8056**. Updating history membantu pada pairs tersebut, tetapi era dan sample size berubah bersama; bukan bukti kausal recency atau satu fitur tertentu.

Brier score pooled LR structured/HGB/text/hybrid adalah **0,1690 / 0,1593 / 0,1259 / 0,1187**. Mean probability hybrid 0,3707 versus actual prevalence 0,3623. Reliability bins dan per-cycle prediction rates menunjukkan probability behavior yang perlu dibaca bersama; kedekatan rata-rata tidak membuktikan calibration baik pada semua cycle/subgroup. Tidak ada calibration model atau probability correction yang dilatih.

## Rekomendasi Round 2 — belum dijalankan

1. **Prioritas pertama: missing-stream dan block ablation.** Tambahkan missing indicator ke structured control, bandingkan joint text/structured block weights yang ditetapkan sebelumnya, lalu pertimbangkan stream-specific fitting atau fallback terkontrol. Expected information gain: memisahkan sumber penurunan missing-stream dari gain present-stream. Risiko: subgroup lebih kecil dan pemilihan policy berlebihan. Belum ada gate/ensemble yang dijalankan.
2. **Threshold transfer/stability.** Inner windows berganda atau history policy yang lebih representatif, dengan outer labels tetap terlarang untuk selection. Pisahkan ranking/AP, calibration/Brier dan positive allocation. Expected information gain: menjelaskan kapan recall gain menghasilkan terlalu banyak FP. Hindari mengoptimalkan satu outer window.
3. **Attribution structured features dan horizon.** Controlled ablation cycle/categorical/numeric blocks pada family/folds tetap; pertimbangkan expanding versus recent history yang dibekukan. Expected information gain: membedakan novelty, time dependence dan informasi yang membantu stream kosong. Sample size/era harus dikontrol semampunya; timing operasional tetap caveat.
4. **Text representation terbatas setelah kontrol di atas.** Ordinary TF-IDF sebagai faithful control; NB weighting atau alternative n-grams sebagai satu hipotesis perbandingan, dengan missing-stream guardrails dan H12. Expected information gain: apakah ranking dapat ditingkatkan tanpa mengorbankan subgroup. Phase 11 TP1 tidak menjamin promosi untuk target baru.

Tidak perlu menambah banyak algoritma atau langsung membuat stacking. Rekomendasi awal adalah memperdalam trade-off hybrid/missing-stream sebelum mengejar skor yang lebih tinggi.

## QA dan batas penggunaan

Protocol/seed/config/source/input/runtime dibekukan sebelum actual fit; original run menyelesaikan **78 model fits** (42 outer +36 inner) untuk 56 evaluation jobs, sekitar 310,03 detik pada environment lokal. Representative refit history 67 melakukan **13 fresh fits** terpisah: seluruh 7 outer probability vectors dan 6 inner vectors bit-identical, audit states serta threshold records sama. Ini bukan klaim full repeat seluruh histories atau jaminan lintas hardware.

Independent sklearn reconciliation memeriksa **112 policy-window metrics, 672 per-cycle metrics, 28 family/policy/model aggregates dan 36 threshold choices** terhadap saved vectors/original Train; max numeric difference 0. Semua Macro F1, confusion ordering, class precision/recall/F1 dan AP yang diperiksa cocok. Receipt: `verification.json`. Raw byte hashes tetap cocok dan 35 protected Regression Round 1 files tetap sama.

Notebook menjalankan frozen cache verification dan reporting; cache yang belum tersedia dibangun ulang memakai recipe tetap. Semua figures/tables berasal dari output aktual. Full-page HTML visual preview tidak tersedia dalam jalur browser lokal; pemeriksaan visual dilakukan pada enam PNG dan outputs/struktur notebook, tanpa klaim GUI preview penuh. Export HTML berhasil; nbconvert memberi warning alt-text untuk 6 image dan Jupyter memakai Windows selector fallback, tanpa cell error pada run akhir. QA akhir: `python -m unittest discover -s tests -v` lulus 44/44 tests dalam 40,581 detik, tanpa error/failure/skip. Notebook valid dan berhasil Run All: 19 cells, 8 code cells berurutan, 0 cell errors, 6 embedded figures. MAIN memeriksa keenam PNG, labels, denominators dan nilai tabel terhadap results; independent review juga menyetujui actual vectors/threshold boundaries. Percobaan build notebook awal menemukan format tabel EF yang tidak sesuai dan telah diperbaiki di presentation layer; recipe/freeze/hasil model tidak diubah.

Fitur tambahan/text bersifat supplied-schema eligible dengan uncertainty timestamp/provenance operasional. Official Test tidak diprediksi; sample bukan label test. Tidak ada raw dataset/text/IDs, credentials atau saved model binaries yang masuk Git. TP1 dan pekerjaan anggota lain tidak diubah.

**STOP untuk review pengguna.** Pilih arah lanjutan bersama; belum ada Classification/Regression Round 2, final model, official-test prediction atau Kaggle submission.
