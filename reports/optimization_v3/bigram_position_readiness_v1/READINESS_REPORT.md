# CLS_BIGRAM_POSITION_READINESS_V1

## Keputusan readiness

**Status: SUPPORT_READY_FOR_REVIEW.** Dukungan fitur dan common support cukup untuk mengajukan satu pilot temporal yang membandingkan bigram bag, bigram berposisi, dan kontrol shuffled. Pemeriksaan ini tidak menguji apakah fitur tersebut meningkatkan Macro F1. Tidak ada classifier yang di-fit atau dinilai.

Pertanyaan pilot yang diusulkan:

> Pada laporan yang sudah memiliki logit dasar C2B bag + POS unigram yang sama, apakah posisi kemunculan bigram yang dikenal dalam tiga bagian urutan token menambah informasi prediktif dibanding bag bigram dan kontrol shuffled?

## Batas dan reproducibility

- Sumber: official Classification Train, 50.608 baris, cycle 47–79 lengkap. SHA-256: `875552a68a3d24b0f3e1966d6ddb5671e6f54305a1c526e2f0a76747942ed91c`.
- Pembacaan CSV dibatasi ke `cycle_code` dan `signal_stream`. `artifact_signal`, `threat_rank`, `record_id`, Test, dan SampleSubmission tidak diparse atau dibuka. Hash di atas mengidentifikasi berkas sumber secara keseluruhan.
- Vocabulary dan IDF dibaca dari artefak frozen E67/E75 yang hash-nya cocok dengan receipt, freeze signature, Train input hash, dan cutoff masing-masing. Tidak ada vectorizer baru yang di-fit.
- Kode hanya membuat descriptor sparse dan ringkasan dukungan. Tidak ada task fit, candidate scoring, target access, Test access, atau submission.
- Ledger budget bersama tidak berubah: SHA-256 sebelum dan sesudah `C62D03ED9DEF54EBBDA891B0931C0E235B4BAA444A9F5322BBBA3A517FF88EFA`.
- Dokumen dan istilah token aktual dari `signal_stream` tidak disimpan pada artefak Git. Kata pada tes adalah data sintetis; JSON hanya berisi hitungan agregat, metadata, dan hash.

Token berasal dari preprocessor dan tokenizer vectorizer frozen: Unicode accent normalization, lowercase, dan token pattern `(?u)\b\w\w+\b`. Untuk `n_token` token, pasangan berurutan berjumlah `n_pair = n_token - 1`. Posisi pasangan ke-`j` memakai `floor(3*j/n_pair)`; pasangan OOV tetap berada dalam urutan dan tetap menentukan batas bin. Hanya pasangan yang tepat ada di vocabulary prefix yang menjadi fitur. Nilai menggunakan `(1 + log(count)) × IDF` dari kolom bigram frozen, lalu seluruh blok dinormalisasi L2 satu kali per baris. Kontrol null memakai `deterministic_order_shuffle` pada urutan string bigram, seed 42, sehingga bag pasangan dan ukuran ketiga bin dipertahankan.

## Dukungan aktual

| Prefix | Split | Baris | Stream ada | Dokumen dengan bigram dikenal | Pasangan dikenal / seluruh pasangan | Tipe vocabulary terlihat | Descriptor berubah saat shuffle |
|---|---|---:|---:|---:|---:|---:|---:|
| E67 | Fit, cycle 47–67 | 29.705 | 27.676 | 27.672 | 719.075 / 1.118.843 = 64,27% | 14.325 / 14.325 | 27.650 / 27.672 = 99,92% |
| E67 | Query, cycle 68–79 | 20.903 | 19.528 | 19.525 | 480.704 / 797.062 = 60,31% | 12.378 / 14.325 = 86,41% | 19.506 / 19.525 = 99,90% |
| E75 | Fit, cycle 47–75 | 42.117 | 39.239 | 39.234 | 1.029.351 / 1.606.679 = 64,07% | 14.277 / 14.277 | 39.208 / 39.234 = 99,93% |
| E75 | Query, cycle 76–79 | 8.491 | 7.965 | 7.963 | 187.461 / 309.226 = 60,62% | 11.063 / 14.277 = 77,49% | 7.952 / 7.963 = 99,86% |

Denominator dokumen untuk persentase stream-present adalah semua baris pada split dengan `signal_stream` non-null dan bukan whitespace saja. Stream kosong berjumlah 1.375 pada query E67 dan 526 pada query E75; stream itu tidak memiliki pasangan token.

Setiap query cycle E67 (68–79) memiliki 525–2.162 dokumen dengan bigram dikenal. Cakupan occurrence per cycle berkisar 57,59%–63,43%; tipe bigram dikenal yang terlihat per cycle 4.192–7.132. Setiap query cycle E75 (76–79) memiliki 1.704–2.162 dokumen dengan bigram dikenal. Cakupan occurrence per cycle berkisar 59,24%–62,60%; tipe bigram dikenal yang terlihat per cycle 6.592–7.459. Median/p90 jumlah occurrence dikenal per dokumen yang memiliki fitur adalah 20/49 pada query E67 dan 19/49 pada query E75.

Vectorizer E67 memiliki 20.000 kolom total, 14.325 di antaranya bigram; E75 memiliki 20.000 kolom total dan 14.277 bigram. Cakupan tipe yang terlihat turun dari 86,41% ke 77,49% di query yang lebih akhir, tetapi tetap luas. Pengacakan mengubah descriptor posisi pada hampir semua dokumen yang punya bigram, sementara bag dan bin size tetap sama. Artinya kontrol dapat membedakan informasi urutan dari identitas/frekuensi pasangan.

### Layar dukungan operasional

Sebelum profiling, readiness ditetapkan sebagai: sedikitnya 2.000 dokumen berfitur di prefix; sedikitnya 1.000 dokumen berfitur pada query; sedikitnya 100 dokumen berfitur pada setiap query cycle; coverage occurrence query sedikitnya 20%; dan descriptor berubah pada sedikitnya 30% dokumen berfitur setelah shuffle. E67 dan E75 memenuhi kelima syarat. Ini pemeriksaan kelayakan fitur, bukan jaminan power statistik atau gate performa.

Blok CSR untuk gabungan fit dan query membutuhkan sekitar 13,2–13,8 MiB per arm pada dua origin; blok shuffled sekitar 13,6–13,8 MiB. Angka tersebut adalah ukuran array CSR aktual dan tidak memasukkan solver workspace, proses Python, atau source text. Tidak ada pengukuran peak RAM melalui task fit.

## Hubungan dengan eksperimen sebelumnya

| Eksperimen | Temuan yang relevan | Makna untuk proposal ini |
|---|---|---|
| C03, bigram bag terpisah | Primary Δ Macro F1 −0,000319; H12 −0,000471; W1 −0,001562. W4 +0,000703 tidak mengubah verdict **RETIRE**. | Frekuensi bigram sendiri belum lolos gate. Karena itu pilot harus punya BAG control dan menguji nilai posisi secara terpisah. |
| C10, trigram context | Primary +0,000343, di bawah syarat +0,0005; W1 −0,001512; H12 +0,000878. Verdict **RETIRE**. | Konteks n-gram memberi sinyal campuran dan tidak cukup melewati gate. Posisi bigram adalah hipotesis baru, tetapi ekspektasi manfaat harus konservatif. |
| C11, expanded benchmark | Primary Δ −0,000479; H12 −0,000310; gabungan late W5/W6 −0,000622; `qualified=false`. | Kandidat text/feature expansion lain belum memberi hasil yang memenuhi qualification. Readiness support bigram-position tidak membatalkan hasil ini. |

Ketiga hasil itu memakai temporal development outcomes yang sudah pernah dipakai. Query E75 (76–79) juga termasuk dalam query E67 (68–79); kedua origin bukan replikasi independen. Tidak ada klaim untuk holdout yang belum tersentuh.

## Usulan pilot terkontrol

Gunakan logit dasar frozen yang sama pada setiap origin: `z_original = C2B bag logit + ORIGINAL unigram POS gamma`. Untuk tiap origin, fit hanya satu tambahan gamma per arm: **BAG**, **REAL position**, dan **SHUFFLED position**. Total biaya adalah **6 task estimator fits** (3 arm × 2 origin); tidak perlu refit vectorizer, C2B bag, POS unigram, atau A02.

Untuk arm, `q = sigmoid(z_original + P @ gamma)`. Pada stream-present rows, keluaran akhir tetap `0,5 × q + 0,5 × p_A02`, threshold 0,50. Missing stream tetap memakai frozen `pP`. Jika satu baris tidak memiliki fitur bigram dikenal, `P` kosong dan kontribusi gamma nol; baris itu kembali persis ke route incumbent `0,5 × sigmoid(z_original) + 0,5 × p_A02`. Jangan mengisi fitur nol dengan nilai buatan atau mengubah fallback.

Detail eksekusi dan kondisi preflight ada di [PROPOSED_CARD.md](PROPOSED_CARD.md). Fit belum dimulai dan tidak ada slot budget yang direservasi.

## Keterbatasan dan hal yang belum diverifikasi

- Bukti di laporan ini hanya memastikan ketersediaan dan variasi fitur. Belum diketahui apakah posisi bigram menambah Macro F1 atau subgroup safety.
- Untuk fit berikutnya, verifikasi ulang hash dan row alignment artefak `z_original`, `p_A02`, serta `pP`; ketiganya tidak dibuka pada audit target-free ini.
- Sebelum fit pertama, cocokkan objective, regularisasi, convergence, dan resume protocol dengan freeze asli POS_BAG_OFFSET. Audit ini tidak membuka atau mengubah recipe tersebut; bila lineage tepat tidak bisa dibuktikan, fit harus berhenti.
- Layar support adalah ambang operasional, bukan perhitungan power. Hubungan dapat berubah menurut cycle dan label distribution.
- Pemeriksaan historis dibatasi pada rujukan C03, C10, C11, dan sumber unigram-position yang disebutkan untuk milestone ini; tidak dilakukan pencarian arsip eksperimen yang luas.
- Posisi bigram bukan klaim kausal, semantik, atau kualitas teks.

## Jejak file

- Statistik terhitung: [SUPPORT_PROFILE.json](SUPPORT_PROFILE.json)
- Builder murni: [optimization_v3_bigram_position_readiness_v1.py](../../../src/models/optimization_v3_bigram_position_readiness_v1.py)
- Reproducer: [optimization_v3_bigram_position_readiness_v1.py](../../../scripts/optimization_v3_bigram_position_readiness_v1.py)
- Tes: [test_optimization_v3_bigram_position_readiness_v1.py](../../../tests/test_optimization_v3_bigram_position_readiness_v1.py)
- Rujukan: `reports/optimization_v3/batch1/C03/RESULTS.md`; `reports/optimization_v3/batch1/C10/summary.md`; `reports/optimization_v3/expanded_benchmark_v1_final_evaluation/C11/results.json`; `src/models/optimization_v3_positional_text_pilot_v1.py`.
