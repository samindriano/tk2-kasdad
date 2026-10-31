# Temuan EDA formal Samuel — availability signal_stream

Pertanyaan resmi: **Apakah proporsi artifact_signal = 1 berbeda menurut ketersediaan signal_stream, dan bagaimana perbedaannya berubah setelah mempertimbangkan periode cycle_code serta volume narasi (text_load)?**

Sumber: official Classification Train saja (50.608 baris, 19 kolom; cycle 47–79). Availability memakai aturan pipeline Classification: null, string kosong, atau whitespace-only adalah missing. Tidak ada Test atau label Test yang dibuka.

## Bukti utama

| Analisis | Present | Missing | Gap missing − present | Support |
|---|---:|---:|---:|---|
| Marginal | 34,47% (16.272/47.204) | 38,13% (1.298/3.404) | +3,66 pp | Semua 50.608 baris |
| Standardisasi exact cycle | 34,47% | 38,43% | +3,96 pp | 33/33 cycle; semua baris |
| Periode × 5 kuantil global text_load | 35,17% | 32,87% | −2,30 pp | 15/15 sel; semua baris |
| Periode × 10 kuantil global text_load | 35,52% | 33,36% | −2,15 pp | 28/30 sel; 47.740 baris (94,33%) |

Baris present dan missing pada baris standardisasi adalah prevalence tersandar, bukan pooled raw rate. Pada setiap sel yang memenuhi minimum n=20 untuk kedua status, bobotnya adalah ukuran cell gabungan dibagi seluruh baris didukung; bobot yang sama digunakan untuk menghitung kedua prevalence. Cakupan q10 adalah 44.368/47.204 present (93,99%) dan 3.372/3.404 missing (99,06%).

Periode memiliki gap marginal −1,84 pp (47–57), +10,01 pp (58–68), dan +3,18 pp (69–79). Pada exact cycle, 21 dari 33 cycle menunjukkan gap positif dan 12 negatif; rentang −17,04 sampai +23,73 pp. Median text_load adalah 31,78 untuk present dan 21,81 untuk missing. Rentang text_load yang sama-sama teramati adalah 20,65–77,52; terdapat 906 baris missing dan 65 present di luarnya.

## Interpretasi

Gap positif marginal sedikit meningkat ketika hanya cycle disamakan, tetapi berubah negatif setelah distribusi periode dan bin volume disamakan. Ini menunjukkan ringkasan asosiasi peka terhadap komposisi waktu dan volume. Karena volume dibagi menjadi quantile bins global yang kasar, dukungan pada bin bukan bukti overlap tepat pada seluruh nilai covariate. Standardisasi juga hanya mengendalikan strata yang ditetapkan. Hasil tidak membuktikan bahwa missingness atau text_load menyebabkan perubahan target; tidak menghilangkan seluruh confounding; dan tidak memastikan pola pada official Test.

Data missing berarti tidak ada isi stream untuk dinilai. Analisis tidak menafsirkan makna token anonim atau kualitas teks. Interval Wilson pada prevalence marginal adalah per kelompok secara terpisah; tidak ada interval adjusted atau uji signifikansi untuk contrast tersandar.

## Hubungan dengan Classification

Model-routing evidence berasal dari modeling/error analysis yang dikembangkan lebih dahulu. Pada subgroup stream missing, hybrid C2_B memiliki Macro F1 0,710427 dan structured HGB C1_B 0,781712; C4_HGB kemudian memakai routing availability. Raw EDA ini dibangun setelah diagnosis tersebut: ia memberi konteks komposisi label, bukan penyebab routing atau bukti tunggal peningkatan model. EDA tidak bertujuan menghasilkan model atau Kaggle score baru.

Empat grafik dan tabel lengkap yang dihitung saat run terdapat pada [notebook formal](../../notebooks/samuel/12_samuel_formal_eda_availability.ipynb). Fresh-kernel execution dan source-copy receipt ada di [QA.json](QA.json).

