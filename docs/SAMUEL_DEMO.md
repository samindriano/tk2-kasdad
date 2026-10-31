# Samuel — ringkasan demo TP 2

Buka [Classification 10](../notebooks/samuel/10_samuel_final_classification.ipynb), [Regression 09](../notebooks/samuel/09_samuel_final_regression.ipynb), dan [EDA availability12](../notebooks/samuel/12_samuel_formal_eda_availability.ipynb). Bedakan metrik fold W6 yang dihitung ulang, hasil primary/H12 historis, dan Kaggle public.

## EDA availability

**Apa pertanyaannya dan mengapa dipilih?** Apakah proporsi `artifact_signal = 1` berbeda menurut ketersediaan `signal_stream`, dan bagaimana perbedaan itu berubah setelah mempertimbangkan periode `cycle_code` serta volume narasi (`text_load`)? Pertanyaan ini memeriksa apakah ketiadaan informasi teks berkaitan dengan komposisi label dan apakah pola marginal bertahan setelah dua aspek struktur data diperhitungkan.

**Apa hasil marginalnya?** Official Classification Train berisi 50.608 baris: 3.404 stream missing dan 47.204 present. Prevalensi artifact ialah 38,13% (1.298/3.404) saat missing versus 34,47% (16.272/47.204) saat present. Gap missing − present adalah +3,66 poin persentase (pp).

**Apa yang berubah setelah cycle dipertimbangkan?** Standardisasi pada exact `cycle_code`, minimum n=20 per status setiap cycle, mempertahankan semua 50.608 baris dan menghasilkan gap +3,96 pp. Jadi standardisasi waktu saja tidak membalik arah. Namun hasil tidak universal: gap positif di 21/33 cycle, negatif di 12; gap per periode 47–57 −1,84 pp, 58–68 +10,01 pp, dan 69–79 +3,18 pp.

**Mengapa arah berubah pada analisis periode × volume?** `text_load` lebih rendah saat stream missing (median 21,81) daripada present (31,78). Jika kedua status dibandingkan dengan bobot komposisi periode dan global quantile-bin `text_load` yang sama, gap menjadi −2,30 pp pada 5 bin (15/15 sel, coverage 100%) dan −2,15 pp pada 10 bin (28/30 sel, coverage 94,33%). Perbedaan komposisi dapat menjelaskan mengapa ringkasan marginal dan tersandar berubah arah; ini asosiasi deskriptif, bukan efek kausal `text_load` atau missingness.

**Apa arti common support?** Setiap cell harus memiliki sedikitnya 20 baris present dan 20 missing. Untuk 10 bin, denominator didukung adalah 3.372/3.404 missing dan 44.368/47.204 present; 906 baris missing berada di luar rentang `text_load` yang juga diamati pada present (20,65–77,52), sementara 65 present berada di luar rentang yang juga diamati pada missing. Binning adalah pendekatan kasar; dukungan cell tidak membuktikan overlap tepat pada tiap nilai volume.

**Apakah hasil membuktikan sebab-akibat atau pola Test?** Tidak. Ini Training data yang dipakai dalam development; adjustment hanya mencakup strata yang ditetapkan, tidak menghilangkan seluruh confounding, dan tidak disertai inferensi untuk efek kausal. Stream kosong tidak memiliki isi teks. EDA tidak mengidentifikasi makna token anonim, kualitas narasi, atau prevalensi pada official Test.

**Mengapa TTR tidak lagi menjadi EDA resmi?** Pengguna secara eksplisit memilih availability sebagai pertanyaan baru. EDA keragaman token yang lama tetap dipertahankan di [notebook11](../notebooks/samuel/11_samuel_formal_eda_a.ipynb) sebagai arsip; ia tidak dihapus atau dinyatakan salah.

**Bagaimana kaitannya dengan model?** Routing C4_HGB sebelumnya ditemukan lewat modeling/error analysis: pada missing-stream subgroup, hybrid C2_B Macro F1 0,710427 dan structured HGB C1_B 0,781712. Routing kemudian menggabungkan dua cabang. Hasil historis primary berubah dari 0,820633 (hybrid) ke 0,825481 (routing), H12 dari 0,788581 ke 0,796810. EDA raw availability dikembangkan setelah diagnostics tersebut; hasil ini memberi konteks komposisi, bukan penyebab atau bukti tunggal keuntungan routing. Kaggle public Macro F1 0,82641 adalah evaluasi terpisah.

## Classification

**Mengapa Macro F1?** Ia merata-ratakan F1 kedua kelas dengan bobot sama. Karena label 0 lebih banyak, accuracy saja dapat menutupi kelas 1; precision dan recall tetap dilaporkan.

**Mengapa TF-IDF dan hybrid?** Structured LR primary 0,7145 menjadi hybrid 0,8206; text-only 0,8014. Penambahan blok teks beserta indikator missing-stream memberi evidence prediktif. Ini terpisah dari EDA TTR, bukan efek kausal TTR terhadap F1.

**Mengapa routing?** Hybrid kuat pada stream tersedia, tetapi structured HGB lebih baik saat teks kosong. C4_HGB menggunakan C2_B hybrid LR untuk present, C1_B HGB untuk NaN/blank setelah strip, lalu threshold 0,50. Gate melihat availability saja, bukan label atau masa depan.

**Apa hasil dan trade-off?** Primary 0,825481, H12 pooled 0,796810, public 0,82641 (ref 57808225). Routing terutama mengurangi false positives dibanding hybrid; false negatives masih ada. Threshold inner tidak selalu transfer, jadi tuning tidak otomatis memperbaiki Macro F1.

**Apa yang dilatih di notebook?** Kedua komponen fit pada 47–75 dan dievaluasi pada 76–79. Semua imputasi, encoding, scaling, vocabulary/IDF dipelajari hanya dari fit. Probabilities cocok dengan hash W6 historis. Macro F1 C4_HGB W6 adalah 0,810384; ini bukan holdout baru.

## Regression

**Mengapa MAE dan median?** MAE mengukur rata-rata absolute error dalam unit target. Conditional median meminimalkan expected absolute loss. Baseline median membantu memahami sinyal awal sebelum model lebih kompleks.

**Apa perkembangan utama?** Median → OLS/HGB squared error → HGB absolute error → expanded features. H12 dan feature/history ablation menunjukkan trade-off waktu. Text melengkapi struktur; HGB+SVD dan LightGBM+SVD kemudian dibandingkan pada protokol yang sama.

**Apa SVD32 dan LightGBM?** SVD memampatkan TF-IDF sparse menjadi 32 koordinat padat untuk pohon; bukan 32 topik yang maknanya sudah diketahui. Kandidat adalah satu LightGBM regression_l 1, bukan stacking. Recipe full-Train yang disubmit menggunakan 400 trees, learning rate 0,05, base 31 leaves/minleaf 40/L 2=1.

**Apa rank blend dan apakah leakage?** Prediksi diurutkan per supplied cycle, diubah menjadi 100×average_rank/n, diblend dengan raw, lalu di-clip[0,100]. Rank memakai prediksi, bukan label inference. Konfigurasi/bobot dipilih lewat inner temporal labels; full-Train recipe memilih base/0,50 dari 47–75→76–79. Mapping bersifat batch-dependent dan tidak memperbaiki ranking inversions; kelengkapan sumber cycle belum diketahui.

**Mengapa LightGBM bila Ridge menang H12?** LightGBM primary 11,931600, equal-cycle 12,283548, H1214,704041; hybrid Ridge H1213,938115 lebih baik. LightGBM membaik pada 6/6 primary windows atas R1-D, tetapi bukan pemenang semua horizon. H12_E 17,473257 masih menjadi risiko. Public 11,71363 (ref 57810150) tidak menghapus counterevidence tersebut.

**Apa yang dilatih ulang dan bagaimana tails?** Dua inner fits memilih parameter/bobot sebelum outer fit W6; dua baseline median juga dijalankan. W6 MAE 10,303425 berbeda dari primary enam windows. Bias low/high-tail primary tetap+12,022/−9,080, meskipun bias global dekat nol. Missing-stream dan novel zero-scope Test juga membatasi generalisasi.

## Pertanyaan lanjutan

**Apakah H12 independen?** Tidak. Ia memakai ulang 38.588 primary validation records dengan history/horizon berbeda.

**Mengapa validasi temporal?** Test 80–91 lebih baru daripada Train 47–79. Expanding past→future mengikuti struktur tersebut. Semua cycle Train sudah dipakai discovery/development, sehingga ini bukan untouched holdout.

**Apakah public menjamin private?** Tidak. Public hanya subset Test. Primary, W6, H12, dan public mempunyai populasi/fit yang berbeda.

**Apakah siap dikumpulkan?** Siap untuk review notebook, belum persetujuan pengumpulan. Periksa roster, format/source dependencies dalam paket, kontribusi anggota lain, baseline Classification, final leaderboard, dan jadwal demo. Tidak ada ZIP atau upload baru pada milestone ini.
