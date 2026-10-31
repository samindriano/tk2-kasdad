# Rencana riset berikutnya — V3 belum selesai

1 November 2026. **Milestone ini hanya riset dan handoff. Tidak ada candidate fit atau submission baru.** Empat fit supervised-token yang sudah berjalan sebelumnya kini tersimpan dan terverifikasi di checkpoint `cb27597`; kedua task gagal gate performa. Kampanye V3 belum diberi kesimpulan akhir.

## Paket yang dapat dipakai Luna

Ada **45 recipe kandidat/policy: 20 Classification dan 25 Regression**. Setiap ID mempunyai kartu operasional, bukan hanya nama algoritme. Ini 45 usulan, bukan 45 hasil baru atau bukti independen. Alternatif bersyarat boleh dilewati bila dasar atau dukungannya tidak terpenuhi.

- [Katalog 45 eksperimen](CATALOG.md): pilih ID dan buka kartunya.
- [Panduan dan prompt Luna](LUNA_HANDOFF.md): satu lane end-to-end atau seluruh antrean dalam satu chat Luna, tanpa review Sol per eksperimen; orchestra diperbolehkan.
- [Queue canonical](../../../experiments/optimization_v3/research_backlog/queue.json): recipe, policy, folds, gates, fit counts, conditions dan status `PLANNED_NOT_RUN`.
- [Feasibility](feasibility.json): inspeksi vocabulary dan dependency tanpa target/fitting.

**Implementasi kandidat belum dibuat.** Luna harus membangun adapter dan shared harness B0, lalu menguji dan membekukannya sebelum training. Dokumen `*_draft.md` menyimpan audit worker; queue, kartu dan handbook mengendalikan eksekusi jika ada perbedaan.

## Kontrol yang dibekukan sepanjang batch

| Task | Recipe | Primary | H12 | Accepted public / ref |
|---|---|---:|---:|---|
| Classification | Present: 75% C2_B + 25% sparse LGB; missing: CatBoost; threshold 0,50 | 0,829944 | 0,802966 | 0,82908 / 57837821 |
| Regression | 56,25% mapped LGB SVD32 + 18,75% mapped Ridge + 25% mapped sparse LGB | 11,559390 | 14,047533 | 11,35970 / 57835122 |

Public di atas berasal dari receipt accepted historis. Ini bukan refresh rank, kuota atau private score terbaru. Kandidat yang membaik di tengah batch belum mengganti kontrol untuk kandidat lain.

## Arah dengan alasan terkuat

1. **Vocabulary dan alokasi identitas token — C03/C04, R02/R03.** Audit label-blind menemukan 93.338 unigram pada 47.204 dokumen non-null. Setelah syarat document frequency ≥3, masih ada **14.985 unigram + 66.701 bigram**, dibanding cap gabungan 20 ribu. Receipt fit55/fit75 mengonfirmasi cap tersebut benar-benar terisi. Kolom teks kedua Train identik tanpa membaca target; pada min_df2 milik Regression terdapat 21.641 unigram + 111.652 bigram yang memenuhi dukungan. Retensi tambahan belum tentu berguna; setiap model tetap belajar vocabulary hanya dari fit fold, tanpa memakai vocabulary audit global.
2. **Solver linear sparse — R01.** SGD L1 sebelumnya selalu mencapai batas 2.000 iterasi. LinearSVR menguji jalur optimasi lain. C=1 adalah pilihan tetap, bukan penalty yang diklaim identik dengan SGD; konvergensi tetap diperiksa. [LinearSVR 1.8](https://scikit-learn.org/1.8/modules/generated/sklearn.svm.LinearSVR.html)
3. **Interaksi teks–metadata — C09/R24.** Pertahankan efek utama teks lalu tambahkan koefisien delta untuk `entry_friction == 1`. Ini berbeda dari pengali bobot teks global yang telah gagal.
4. **Populasi dan generalisasi kategori — C02/C06/C07/C08.** Isolasi present-only fit, novelty kronologis, frekuensi kategori dan rate dari cycle sebelumnya. Unknown flag dari vocabulary fit penuh akan selalu nol pada training; desain itu dibuang.
5. **Bias learner dan representasi — C13/C14/C18/C19, R21/R23.** SVD+HGB, partisi acak, random-threshold boosting, XGBoost dan ExtraTrees memberi pembanding yang berbeda. Dependency utama sudah tersedia; ini belum merupakan evidence gain. [RandomTreesEmbedding](https://scikit-learn.org/1.8/modules/generated/sklearn.ensemble.RandomTreesEmbedding.html), [XGBoost 3.2](https://xgboost.readthedocs.io/en/release_3.2.0/parameter.html)
6. **Waktu, calibration dan ranking — R16/R17/R18/R19/R12.** Gunakan bobot history halus atau residual OOF masa lalu. Ranker mempelajari urutan per cycle lalu menghasilkan percentile seluruh cycle; objective ranking tidak otomatis mengoptimalkan MAE. [LightGBM 4.6](https://lightgbm.readthedocs.io/en/v4.6.0/Parameters.html)
7. **Cadangan bersyarat.** Character boosting pada token hash-like, NMF, sembilan ordinal heads, specialist kecil dan supervised feature selection berada di belakang opsi yang lebih murah atau lebih kuat alasannya.

## Satu chat Luna: dari awal sampai akhir

Mode utama **ONE_CHAT_END_TO_END** memproses seluruh 45 kartu yang memenuhi prasyarat. **24 ID pertama adalah prioritas awal, bukan stop point**; prefix tersebut memerlukan 44 predictor fit bila seluruh prasyarat lolos. Seluruh pilot mempunyai 98 predictor fits jika semua condition lolos. Luna melanjutkan sendiri sampai screening, continuation, QA, qualified refit/submission, dan laporan konsolidasi selesai atau ada batas nyata.

Urutan awal: **C03 → R01 → C02 → R02 → C07 → R18 → C09 → R03 → C04 → R16 → C18 → R21**. Seluruh urutan ada di katalog. Luna menerapkan fixed gates, mencatat hasil, lalu lanjut tanpa menunggu pengguna atau Sol di tiap ID. Mode **SINGLE_LANE** juga mencakup seluruh alur lane terpilih, bukan berhenti setelah pilot.

Pemilihan continuation dipisahkan menurut view: maksimal **dua STANDARD** per task berdasarkan W1+W4/H12, dan secara terpisah maksimal **satu LATE** berdasarkan W4+W6/H12_L. Jangan mencampur raw pilot gains dari window berbeda. Kandidat CACHE_SIX sudah memakai full-six. Seluruh finalist baru dibandingkan bersama setelah mempunyai evaluasi enam window yang sama, dengan fallback/support rule yang sudah dibekukan.

Budget penelitian default 180 successful candidate predictor fits dan 24 control/verification refits, cukup untuk pilot serta continuation yang dideklarasikan; bukan target yang harus dihabiskan. Batas dua konfigurasi berlaku **per task dan family**. Admission atomik per kartu: R11 membutuhkan 18 heads dan tidak dimulai bila sisa budget kurang. Transform/inner fits, attempts dan retry juga dicatat. Tidak ada hard stop enam jam secara default; checkpoint sekitar delapan jam lalu lanjut bila sehat. Batas waktu keras hanya bila pengguna memilihnya atau resource benar-benar membatasi.

Satu Luna MAIN boleh memakai worker Luna XHigh mengikuti orchestra yang berlaku: kepemilikan file terpisah, MAIN memegang integrasi/Git/Kaggle, worker tidak menulis scope yang sama. **Satu training job dua thread** tetap berlaku; pekerjaan implementasi dan review yang independen boleh paralel. Waktu sparse/character lama sekitar 10–57 detik adalah comparator, bukan janji model baru. RAM saat audit sekitar 31 GiB total dan 7,9 GiB bebas; cek ulang. Torch belum terpasang dan GPU belum diverifikasi, sehingga neural/pretrained bukan default.

## Koreksi metode yang sudah dimasukkan

- Case-preserving text tidak mempunyai dasar pada scan ini: tidak ada huruf besar. Tidak ada klaim makna semantik karakter token anonim.
- EF berarti `entry_friction`; tidak diganti `telemetry_flag`. Specialist memakai raw prediction dalam skala target, bukan percentile pada subset EF/missing yang dianggap uniform 0–100.
- Temperature-only pada probability akhir tidak mengubah label di threshold 0,50; tidak dihitung sebagai eksperimen peningkatan Macro F1.
- LightGBM `linear_tree=True` tidak mendukung L1. Kombinasi tersebut tidak masuk antrean. Poisson/Gamma/Tweedie juga tidak dijadikan default untuk target rank terbatas ini. [Batasan LightGBM](https://lightgbm.readthedocs.io/en/v4.6.0/Parameters.html#linear_tree)
- Tidak mengulang identik token-summary lama, residual A lama, CDF median lama, character Ridge lama atau public weight/SVD sweeps. Perubahan learner×representasi diberi label dan confidence yang sesuai.
- Kartu canonical R18/R19 memakai **median** residual agar selaras MAE. R15 memakai minimal 1.000 missing fit rows dan pilot late. Ini mengatasi perbedaan dengan draft awal.
- Model ranker menghasilkan skor tanpa satuan target: konversi dulu ke percentile seluruh cycle; jangan mencampur raw score ranker dengan angka target 0–100 melalui rank50.

## Validasi dan batas interpretasi

Semua outcome Train telah dipakai dalam beberapa putaran. Memperbanyak recipe dapat mengoptimalkan noise validation; enam window dan H12 tidak menjadi untouched holdout baru. Fixed recipe, past-only preprocessing, pelaporan seluruh trial dan guardrails membatasi kebebasan adaptif, tetapi tidak menghilangkan selection bias. Bootstrap pada cycle yang sama tidak menciptakan data independen. [Cawley–Talbot, 2010](https://jmlr.org/papers/v11/cawley10a.html)

Gate di queue adalah usulan aturan riset, bukan requirement tugas resmi:

- Classification: gain primary ≥0,0005; harm H12 ≤0,001; minimal 4/6 window nonnegatif; guard kelas dan subgroup.
- Regression: gain MAE ≥0,03; harm H12 ≤0,05; minimal 4/6 window nonnegatif; harm tail/missing ≤0,25 pada support ≥100.
- Pilot LATE menggunakan **W4+W6/H12_L**, early fallback eksplisit. Jangan melabelinya W1+W4 atau full-six, dan jangan mempromosikan ke eksternal dari late-only pilot.

Setiap kartu menjelaskan perubahan, policy, data hash, folds, fit count, support, source reuse, output dan prompt satu lane. Lihat [CATALOG.md](CATALOG.md). Finalist yang lolos full-six boleh masuk full-Train refit dan Test inference/submission sesuai otorisasi V3 yang lama, fresh live gates dan global budget yang tidak direset. Tidak ada pause approval per milestone. Bila seluruh recipe gagal, mempertahankan champion dan menyelesaikan laporan merupakan akhir lane yang sah.

## Stop point

Checkpoint `cb27597` menyimpan pilot supervised-token dengan QA PASS dan gate performa FAIL, tanpa refit saat report diperbaiki. EDA resmi, notebook akademis dan pekerjaan anggota lain tetap utuh. **V3 belum selesai.** Milestone sekarang berhenti setelah paket riset dan panduan Luna direview, commit dan push. Tidak ada queue training, automation malam, Test inference atau Kaggle upload yang dimulai sekarang.
