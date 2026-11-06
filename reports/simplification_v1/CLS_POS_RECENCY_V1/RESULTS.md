# CLS_POS_RECENCY_V1 — hasil pilot dan gate QA

## Status dan keputusan

**Status akhir: SCORED_DIAGNOSTIC_ONLY; exploratory eligible = YA; scientific qualified = TIDAK.** Ini pilot diagnostik yang selesai dievaluasi pada temporal rows official Train, bukan kandidat champion dan bukan klaim mekanisme temporal yang terbukti.

Kriteria scientific yang tidak terpenuhi adalah gain Macro F1 pada unique primary: **+0.000355**, di bawah ambang **+0.000500**. Gerbang lain lulus: E67 H12 +0.000981; E75 W6 +0.001268; RECENCY minus SHUFFLED pada unique primary +0.000393 (minimum +0.000200) dan E75 W6 +0.001666; subgroup guards, missing route, independent metric QA, serta resource reserve lulus. Exploratory gate yang terpisah lulus sesuai definisi beku. Tidak ada gate yang diubah.

| Batas | Nilai |
|---|---|
| Source code commit yang dibekukan | 9bb0716ccb63932f4f67c560b2731b28d0ba3958 |
| Freeze signature | ed69075f56c9e6db5313cd55405a439896d07f811e39ef850fc17e50e3e01cb8 |
| Official Classification Train (50,608 rows), SHA-256 | 875552a68a3d24b0f3e1966d6ddb5671e6f54305a1c526e2f0a76747942ed91c |
| Score signature / SHA-256 | dde8767b8ef2678f504410edfa4bc26eb462bca6f86f1adc6146264675973046 / 86d9eebf1122993c38619ce9da5d3d32632ae2502f7c057236a3ce86829a8ac2 |
| Metric verification SHA-256 | a19ebbdf3dd362dce1e4f3ab190ea2dca7d61e0511a62c5bf547a33ad788c7ce |
| Independent gate verification SHA-256 | a45daad711e0957fe2753de1594787830b45bbaf3e4c9b3fa2324161bedecd5f |

## Recipe yang dijalankan

Pada E67 dan E75, gamma positional di-fit dari prefix stream-present saja pada frozen z_BAG + X_POS. Bobot recency mengikuti half-life 8 cycle dengan floor 0.25, lalu dinormalisasi terpisah per kelas agar massa kelas tetap sama. Kontrol mengacak multiset bobot di dalam kelas dengan seed 42; tidak ada parameter yang dicari setelah hasil terlihat. Champion dan routing missing-stream tetap comparator yang dibekukan.

## Kinerja utama

Unique primary menggabungkan E67 cycle 68–75 dengan E75 cycle 76–79 agar setiap row hanya muncul sekali. E67 H12 memakai E67 untuk cycle 68–79; E75 W6 memakai cycle 76–79 dan merupakan subset E67 W6. Jadi jendela yang dilaporkan saling tumpang tindih dan bukan replikasi independen.

Untuk kolom LogLoss improvement dan Brier improvement, angka positif berarti loss RECENCY lebih rendah dari champion; untuk metrik lain delta dihitung RECENCY − champion. Confusion matrix ditulis sebagai [[TN, FP], [FN, TP]].

| View | n (neg/pos) | Macro F1 | Prec+ | Recall+ | ROC AUC | AP | LogLoss improvement | Brier improvement | Confusion matrix |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| UNIQUE_PRIMARY UNIQUE_PRIMARY | 20,903 (14,176/6,727) | 0.822295 | 0.773702 | 0.740003 | 0.905076 | 0.798496 | 0.000724 | 0.000267 | [[12720, 1456], [1749, 4978]] |
| E67 H12_E67 | 20,903 (14,176/6,727) | 0.812308 | 0.748948 | 0.740598 | 0.898542 | 0.792038 | 0.001430 | 0.000481 | [[12506, 1670], [1745, 4982]] |
| E67 W4 | 4,950 (3,097/1,853) | 0.846191 | 0.813048 | 0.800324 | 0.913663 | 0.825270 | 0.001007 | 0.000299 | [[2756, 341], [370, 1483]] |
| E67 W5 | 7,462 (5,088/2,374) | 0.804686 | 0.742523 | 0.721567 | 0.892201 | 0.781953 | 0.001333 | 0.000456 | [[4494, 594], [661, 1713]] |
| E67 W6 | 8,491 (5,991/2,500) | 0.795134 | 0.708449 | 0.714400 | 0.891505 | 0.772310 | 0.001761 | 0.000608 | [[5256, 735], [714, 1786]] |
| E75 W6 | 8,491 (5,991/2,500) | 0.820152 | 0.773773 | 0.712800 | 0.908574 | 0.790296 | 0.000024 | 0.000081 | [[5470, 521], [718, 1782]] |

### RECENCY vs champion: metric deltas

| View | ΔF1 | ΔPrec+ | ΔRecall+ | ΔROC AUC | ΔAP | LogLoss improvement | Brier improvement |
|---|---:|---:|---:|---:|---:|---:|---:|
| UNIQUE_PRIMARY UNIQUE_PRIMARY | +0.000355 | +0.002331 | -0.001784 | +0.000307 | +0.000930 | +0.000724 | +0.000267 |
| E67 H12_E67 | +0.000981 | +0.005548 | -0.004460 | +0.000310 | +0.000961 | +0.001430 | +0.000481 |
| E67 W4 | +0.001854 | +0.005774 | -0.002159 | +0.000356 | +0.000684 | +0.001007 | +0.000299 |
| E67 W5 | -0.001560 | +0.001623 | -0.007161 | +0.000327 | +0.001169 | +0.001333 | +0.000456 |
| E67 W6 | +0.002628 | +0.008644 | -0.003600 | +0.000221 | +0.000815 | +0.001761 | +0.000608 |
| E75 W6 | +0.001268 | +0.000213 | +0.003600 | -0.000006 | +0.000027 | +0.000024 | +0.000081 |

### RECENCY vs shuffled control: metric deltas

| View | ΔF1 | ΔPrec+ | ΔRecall+ | ΔROC AUC | ΔAP | LogLoss improvement | Brier improvement |
|---|---:|---:|---:|---:|---:|---:|---:|
| UNIQUE_PRIMARY UNIQUE_PRIMARY | +0.000393 | +0.002904 | -0.002378 | +0.000295 | +0.000362 | +0.000675 | +0.000258 |
| E67 H12_E67 | +0.001476 | +0.007227 | -0.005203 | +0.000493 | +0.000792 | +0.001823 | +0.000634 |
| E67 W4 | +0.001485 | +0.006997 | -0.004857 | +0.000078 | -0.000339 | +0.000458 | +0.000040 |
| E67 W5 | -0.001515 | +0.003281 | -0.009267 | +0.000384 | +0.000784 | +0.001480 | +0.000539 |
| E67 W6 | +0.004151 | +0.010865 | -0.001600 | +0.000956 | +0.002069 | +0.002919 | +0.001063 |
| E75 W6 | +0.001666 | -0.000887 | +0.006000 | +0.000044 | -0.000272 | +0.000093 | +0.000139 |
## Konsistensi antar-siklus

Delta per siklus dihitung terhadap champion pada baris yang sama; arah positif/negatif menunjukkan tanda delta Macro F1 RECENCY − champion. Perbedaan arah menunjukkan efeknya tidak seragam antar-cycle.

| Cycle | n | ΔMacro F1 vs champion | ΔRecall+ vs champion | ΔMacro F1 vs shuffled |
|---:|---:|---:|---:|---:|
| 68 | 577 | -0.001818 | -0.011696 | -0.001818 |
| 69 | 1,079 | +0.000811 | -0.002268 | +0.001622 |
| 70 | 1,605 | +0.001750 | -0.001637 | +0.001483 |
| 71 | 1,689 | +0.003496 | +0.000000 | +0.002234 |
| 72 | 1,811 | -0.004621 | -0.009434 | -0.007330 |
| 73 | 1,880 | -0.002134 | -0.009646 | -0.005106 |
| 74 | 1,814 | -0.000249 | -0.009009 | +0.004047 |
| 75 | 1,957 | +0.001001 | +0.000000 | +0.002763 |
| 76 | 1,822 | +0.004281 | +0.009615 | +0.009666 |
| 77 | 2,335 | +0.000634 | +0.001502 | +0.000556 |
| 78 | 2,146 | -0.002877 | -0.001862 | -0.002409 |
| 79 | 2,188 | +0.002764 | +0.004458 | -0.000610 |

Arah gain terhadap champion positif pada **7/12** cycle, negatif pada **5/12**, dan nol pada 0. Tidak ada dasar untuk menyatakan perbaikan merata sepanjang waktu.

## Subgroup dan guard

Tiap subgroup memenuhi syarat dukungan n ≥ 200 dan sekurangnya 30 contoh per kelas. Seluruh enam view menyertakan 8 kelompok (availability, entry_friction, text-load), sehingga **48/48** perbandingan subgroup didukung. Denominator ditulis n (kelas 0/kelas 1). Delta negatif berarti penurunan RECENCY.

### Unique primary

| Kelompok | n (0/1) | ΔF1 vs champion | ΔRecall vs champion | ΔF1 vs shuffled | ΔRecall vs shuffled |
|---|---:|---:|---:|---:|---:|
| availability=False (didukung) | 1,375 (888/487) | +0.000000 | +0.000000 | +0.000000 | +0.000000 |
| availability=True (didukung) | 19,528 (13,288/6,240) | +0.000383 | -0.001923 | +0.000424 | -0.002564 |
| entry_friction=0 (didukung) | 14,344 (10,221/4,123) | -0.000431 | -0.003396 | -0.000310 | -0.004366 |
| entry_friction=1 (didukung) | 6,559 (3,955/2,604) | +0.001801 | +0.000768 | +0.001655 | +0.000768 |
| fit_prefix_text_load=0 (didukung) | 5,452 (3,418/2,034) | -0.000446 | -0.001475 | -0.000832 | -0.001967 |
| fit_prefix_text_load=1 (didukung) | 4,944 (3,410/1,534) | +0.000355 | -0.000652 | +0.000445 | -0.001956 |
| fit_prefix_text_load=2 (didukung) | 4,731 (3,582/1,149) | +0.000020 | -0.001741 | -0.000201 | -0.003481 |
| fit_prefix_text_load=3 (didukung) | 5,776 (3,766/2,010) | +0.001323 | -0.002985 | +0.001887 | -0.002488 |

### E67 H12

| Kelompok | n (0/1) | ΔF1 vs champion | ΔRecall vs champion | ΔF1 vs shuffled | ΔRecall vs shuffled |
|---|---:|---:|---:|---:|---:|
| availability=False (didukung) | 1,375 (888/487) | +0.000000 | +0.000000 | +0.000000 | +0.000000 |
| availability=True (didukung) | 19,528 (13,288/6,240) | +0.001057 | -0.004808 | +0.001591 | -0.005609 |
| entry_friction=0 (didukung) | 14,344 (10,221/4,123) | +0.000404 | -0.005093 | +0.000665 | -0.006064 |
| entry_friction=1 (didukung) | 6,559 (3,955/2,604) | +0.002217 | -0.003456 | +0.003203 | -0.003840 |
| fit_prefix_text_load=0 (didukung) | 5,484 (3,443/2,041) | +0.000974 | -0.001470 | +0.000181 | -0.002940 |
| fit_prefix_text_load=1 (didukung) | 4,841 (3,334/1,507) | +0.001818 | -0.002654 | +0.004164 | -0.003981 |
| fit_prefix_text_load=2 (didukung) | 4,715 (3,574/1,141) | +0.000053 | -0.007011 | +0.000319 | -0.010517 |
| fit_prefix_text_load=3 (didukung) | 5,863 (3,825/2,038) | +0.000724 | -0.007360 | +0.001133 | -0.005397 |

### E67 W4

| Kelompok | n (0/1) | ΔF1 vs champion | ΔRecall vs champion | ΔF1 vs shuffled | ΔRecall vs shuffled |
|---|---:|---:|---:|---:|---:|
| availability=False (didukung) | 342 (202/140) | +0.000000 | +0.000000 | +0.000000 | +0.000000 |
| availability=True (didukung) | 4,608 (2,895/1,713) | +0.002005 | -0.002335 | +0.001613 | -0.005254 |
| entry_friction=0 (didukung) | 3,418 (2,211/1,207) | +0.000916 | -0.004143 | +0.002045 | -0.004971 |
| entry_friction=1 (didukung) | 1,532 (886/646) | +0.003847 | +0.001548 | +0.000415 | -0.004644 |
| fit_prefix_text_load=0 (didukung) | 1,092 (579/513) | +0.000902 | -0.001949 | -0.001842 | -0.003899 |
| fit_prefix_text_load=1 (didukung) | 963 (625/338) | +0.004334 | +0.002959 | +0.003778 | -0.005917 |
| fit_prefix_text_load=2 (didukung) | 1,130 (808/322) | -0.001248 | -0.003106 | -0.000549 | -0.006211 |
| fit_prefix_text_load=3 (didukung) | 1,765 (1,085/680) | +0.002982 | -0.004412 | +0.003534 | -0.004412 |

### E67 W5

| Kelompok | n (0/1) | ΔF1 vs champion | ΔRecall vs champion | ΔF1 vs shuffled | ΔRecall vs shuffled |
|---|---:|---:|---:|---:|---:|
| availability=False (didukung) | 507 (331/176) | +0.000000 | +0.000000 | +0.000000 | +0.000000 |
| availability=True (didukung) | 6,955 (4,757/2,198) | -0.001694 | -0.007734 | -0.001648 | -0.010009 |
| entry_friction=0 (didukung) | 5,188 (3,694/1,494) | -0.002830 | -0.009371 | -0.004608 | -0.014726 |
| entry_friction=1 (didukung) | 2,274 (1,394/880) | +0.000945 | -0.003409 | +0.004546 | +0.000000 |
| fit_prefix_text_load=0 (didukung) | 2,116 (1,376/740) | -0.002668 | -0.005405 | -0.002414 | -0.008108 |
| fit_prefix_text_load=1 (didukung) | 1,696 (1,111/585) | -0.003070 | -0.008547 | -0.001472 | -0.010256 |
| fit_prefix_text_load=2 (didukung) | 1,632 (1,248/384) | -0.001675 | -0.007812 | -0.001990 | -0.010417 |
| fit_prefix_text_load=3 (didukung) | 2,018 (1,353/665) | +0.000842 | -0.007519 | -0.000335 | -0.009023 |

### E67 W6

| Kelompok | n (0/1) | ΔF1 vs champion | ΔRecall vs champion | ΔF1 vs shuffled | ΔRecall vs shuffled |
|---|---:|---:|---:|---:|---:|
| availability=False (didukung) | 526 (355/171) | +0.000000 | +0.000000 | +0.000000 | +0.000000 |
| availability=True (didukung) | 7,965 (5,636/2,329) | +0.002818 | -0.003864 | +0.004454 | -0.001717 |
| entry_friction=0 (didukung) | 5,738 (4,316/1,422) | +0.003149 | -0.001406 | +0.004930 | +0.002110 |
| entry_friction=1 (didukung) | 2,753 (1,675/1,078) | +0.002287 | -0.006494 | +0.003665 | -0.006494 |
| fit_prefix_text_load=0 (didukung) | 2,276 (1,488/788) | +0.004510 | +0.002538 | +0.003644 | +0.002538 |
| fit_prefix_text_load=1 (didukung) | 2,182 (1,598/584) | +0.004619 | +0.000000 | +0.009131 | +0.003425 |
| fit_prefix_text_load=2 (didukung) | 1,953 (1,518/435) | +0.001857 | -0.009195 | +0.002236 | -0.013793 |
| fit_prefix_text_load=3 (didukung) | 2,080 (1,387/693) | -0.001620 | -0.010101 | +0.000439 | -0.002886 |

### E75 W6

| Kelompok | n (0/1) | ΔF1 vs champion | ΔRecall vs champion | ΔF1 vs shuffled | ΔRecall vs shuffled |
|---|---:|---:|---:|---:|---:|
| availability=False (didukung) | 526 (355/171) | +0.000000 | +0.000000 | +0.000000 | +0.000000 |
| availability=True (didukung) | 7,965 (5,636/2,329) | +0.001352 | +0.003864 | +0.001776 | +0.006441 |
| entry_friction=0 (didukung) | 5,738 (4,316/1,422) | +0.001260 | +0.003516 | +0.002720 | +0.007032 |
| entry_friction=1 (didukung) | 2,753 (1,675/1,078) | +0.001316 | +0.003711 | -0.000015 | +0.004638 |
| fit_prefix_text_load=0 (didukung) | 2,244 (1,463/781) | +0.001073 | +0.002561 | +0.001248 | +0.005122 |
| fit_prefix_text_load=1 (didukung) | 2,285 (1,674/611) | +0.001457 | +0.004910 | +0.000851 | +0.008183 |
| fit_prefix_text_load=2 (didukung) | 1,969 (1,526/443) | +0.002485 | +0.004515 | +0.001902 | +0.004515 |
| fit_prefix_text_load=3 (didukung) | 1,993 (1,328/665) | +0.000442 | +0.003008 | +0.002823 | +0.006015 |

Worst supported subgroup harm vs champion: Macro F1 -0.003070 at E67/W5/fit_prefix_text_load=1; positive recall -0.010101 at E67/W6/fit_prefix_text_load=3.

Worst supported subgroup harm vs shuffled: Macro F1 -0.004608 at E67/W5/entry_friction=0; positive recall -0.014726 at E67/W5/entry_friction=0.

Gate yang dipakai tetap gate beku: scientific subgroup harm maksimum 0.010 F1 / 0.020 recall; exploratory harm maksimum 0.020 F1 / 0.030 recall. Verifier merekonstruksi seluruh gate sebagai scientific false, exploratory true. Missing-stream fallback identik dengan champion, sehingga subgroup availability=False menghasilkan delta nol untuk route itu.

## Fit, data boundary, dan accounting

| Fit | Prefix-present n | ESS ratio class 0 / class 1 | Hasil |
|---|---:|---:|---|
| E67 RECENCY | 27,676 | 0.8048 / 0.8384 | PASS_INDEPENDENT_FIT_RECONSTRUCTION |
| E67 SHUFFLED_WITHIN_CLASS | 27,676 | 0.8048 / 0.8384 | PASS_INDEPENDENT_FIT_RECONSTRUCTION |
| E75 RECENCY | 39,239 | 0.7744 / 0.7854 | PASS_INDEPENDENT_FIT_RECONSTRUCTION |
| E75 SHUFFLED_WITHIN_CLASS | 39,239 | 0.7744 / 0.7854 | PASS_INDEPENDENT_FIT_RECONSTRUCTION |

Reservation tepat 4 predictor fits; keempatnya dimulai, tersimpan, dan lolos independent fit reconstruction. Retry/interruption optimizer: 0; preprocessing fits: 0. Satu prefit replay teknis diizinkan karena laporan prefit pertama kehilangan field prefix_label_target_sha256 saat menyusun JSON dan berhenti sebelum freeze/optimizer. Replay membaca prefix E75 yang sama 39.239 baris, hash target sama, dan dicatat eksplisit. E75 prefix beririsan temporal dengan query E67; labels tersebut hanya dipakai untuk E75 fitting/class-mass preflight, bukan dinilai sebagai holdout baru.

| Tahap pembacaan target resmi Train | Jumlah pembacaan | Baris per pembacaan | Total row-reads (pembacaan berulang dihitung) |
|---|---:|---:|---:|
| Prefit E75 (gagal sebelum fit + replay resmi) | 2 | 39,239 | 78,478 |
| 4 official fits (2 E67, 2 E75) | 4 | 27,676 / 39,239 sesuai origin | 133,830 |
| Independent reconstruction 4 fit receipts | 4 | 27,676 / 39,239 sesuai origin | 133,830 |
| One-pass temporal score, unique E67 query rows | 1 | 20,903 | 20,903 |
| **Total** | **11** | — | **367,041** |

Query target hanya dibuka satu kali untuk scoring; metric/gate verifier tidak membuka ulang label. Predictions dan prelabel QA sebelumnya target-free; prelabel mencatat 20.903 E67 rows dan 8.491 E75 rows, E75 tepat subset E67 W6. Tidak ada Test, SampleSubmission, Kaggle, atau public leaderboard access. Tidak ada public confidence interval.

## QA dan reproducibility evidence

- Metric verifier: PASS_INDEPENDENT_METRIC_RECALC, 186 comparison calculations; query labels tidak dibaca ulang.
- Score/gate verifier independen: PASS_INDEPENDENT_METRIC_AND_GATE_VERIFICATION; score SHA dan metric-report SHA cocok dengan artefak yang diverifikasi.
- Prediction readiness: seluruh empat fit receipt durable dan independently verified; missing fallback exact; target-free prediction bundles E67/E75 masing-masing 20.903/8.491 baris.
- Prefit source QA: PASS_PREFIT_SOURCE_QA; dua prefix label reads identik dan tercatat; query/Test/Sample tidak dibuka.
- Prelabel: PASS_TARGET_FREE_PRELABEL; optimizer refits 0, query label reads 0, preprocessing fits 0.
- Resource reserve check setelah score: PASS.
- Source-level tests and synthetic lifecycle suite from implementation phase: 12 passed; this reporting-only step made no code or recipe changes and did not rerun tests.

## E79 static source inventory

Hasil exploratory-eligible memicu batas root-plan untuk E79: hanya inventory statis, tanpa fit, inference, label read, CSV, atau upload. Source saat ini mendaftarkan hanya origin 67 dan 75, dan validator membatasi score rows pada origin+1 hingga 79. Dengan origin 79, rentang score dalam batas Train kosong; evaluasi ke depan akan melewati cycle 79 ke Test dan tidak termasuk pilot ini. Karena itu E79 belum siap dijalankan dalam freeze yang ada dan memerlukan review/otorisasi root terpisah sebelum perubahan scope. Tidak ada Test/Sample dibuka untuk kesimpulan ini.

## Kesimpulan dan batas interpretasi

RECENCY menunjukkan peningkatan terhadap champion pada unique primary dan mengalahkan shuffled control menurut ambang eksploratori, tetapi gain primary terlalu kecil untuk scientific gate. Tanda gain berbeda pada 12 cycle; hasilnya mendukung paling jauh benchmark diagnostik terbatas. Kontrol shuffle mempertahankan multiset bobot per kelas dan ESS, tetapi tidak mengisolasi seluruh confounding atau menyamakan effective degrees of freedom. E67/E75 window overlap, dan E75 prefix label overlap dengan periode query E67, sehingga hasil ini bukan holdout yang sepenuhnya segar. Root tetap memutuskan apakah ada langkah lanjutan; tidak ada champion, recipe, gate, atau submission yang diubah.
