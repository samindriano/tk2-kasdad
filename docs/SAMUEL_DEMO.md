# Samuel — ringkasan demo TP 2

Buka [Classification 10](../notebooks/samuel/10_samuel_final_classification.ipynb), [Regression 09](../notebooks/samuel/09_samuel_final_regression.ipynb), dan [EDA A11](../notebooks/samuel/11_samuel_formal_eda_a.ipynb). Bedakan hasil W6 yang dihitung ulang, primary/H12 historis, dan Kaggle public.

## EDA A

**Apa pertanyaannya?** Pada laporan dengan jumlah token yang sama, apakah keragaman token berkaitan dengan proporsi artifact_signal=1?

**Apa TTR dan mengapa panjang dibuat sama?** TTR adalah token unik/token total setelah regex `[A-Za-z 0-9]+` dan lowercase. TTR dipengaruhi panjang, sehingga perbandingan dilakukan dalam setiap exact token count. TTR mengukur repetisi, bukan kualitas atau makna token anonim.

**Bagaimana grup dibuat?** Average percentile rank TTR≤0,25 versus≥0,75; ties tidak dipecah. Minimum 120 baris per panjang dan 20 tiap tail. Kedua grup menggunakan bobot panjang yang sama.

**Apa hasilnya?** Weighted prevalence 24,19% versus 38,39%, gap 14,20 pp;64/71 panjang searah. Support 37.681 berbeda dari 9.011 rendah dan 9.354 tinggi yang benar-benar dibandingkan. Asosiasi tidak kausal, tidak berlaku otomatis pada stream kosong/Test, dan tidak membuktikan TTR meningkatkan F1.

**Apakah hanya perbedaan waktu?** Gap tiga periode tetap positif 5,65/15,19/13,82 pp. Ringkasan exact length×cycle 7,99 pp per+0,10 TTR mempunyai estimand linear berbeda dari gap quartile. Ada counterexample dan confounding lain.

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
