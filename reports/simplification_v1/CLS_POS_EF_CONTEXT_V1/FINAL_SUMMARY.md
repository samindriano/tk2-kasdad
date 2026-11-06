# CLS POS x entry_friction: hasil pilot

**Keputusan: tutup cabang ini; pertahankan champion.** Hasil negatif berasal dari perbandingan utama, bukan hanya kebijakan subgroup. Tidak ada E79 refit, CSV baru, atau submission.

Hipotesis: koreksi posisi token yang bergantung pada EF menambah informasi di atas logit POS dan A02 champion. Empat gamma fits (REAL dan EF-shuffled pada E67/E75) telah selesai. Offset, blend 50/50, fallback missing, serta threshold dibekukan; tidak ada tuning sesudah evaluasi.

| View | n | F1 champion | F1 REAL | REAL - champion | REAL - shuffled |
|---|---:|---:|---:|---:|---:|
| UNIQUE_PRIMARY/UNIQUE_PRIMARY | 20,903 | 0.821941 | 0.820050 | -0.001890 | -0.000712 |
| E67/H12_E67 | 20,903 | 0.811327 | 0.809022 | -0.002306 | -0.002055 |
| E67/W4 | 4,950 | 0.844337 | 0.843843 | -0.000494 | -0.000664 |
| E67/W5 | 7,462 | 0.806246 | 0.805316 | -0.000930 | +0.000000 |
| E67/W6 | 8,491 | 0.792507 | 0.787647 | -0.004859 | -0.004930 |
| E75/W6 | 8,491 | 0.818884 | 0.815039 | -0.003845 | -0.001409 |

Primary memakai 20.903 baris unik: E67 cycle 68-75 dan E75 cycle 76-79; E67 H12 tumpang tindih sehingga bukan bukti independen. REAL kalah pada semua view agregat dan hanya positif pada 3/12 cycle (satu cycle nol). Kontrol shuffled pun kalah pada primary; REAL lebih buruk lagi. Data ini tidak mendukung tambahan interaction tersebut pada recipe yang diuji. Ini tidak membuktikan EF atau semua interaction tidak informatif.

Satu guard legacy gagal pada E67/W6, text-load group 1 (F1 -0,010442; recall -0,015411). Guard exploratory yang lebih longgar lulus, tetapi gate gain tetap gagal. Scientific dan exploratory eligibility sama-sama false. Missing fallback tetap persis.

## Pemulihan teknis dan QA

Scorer historis menggunakan `key` sebelum assignment dalam loop subgroup. Percobaan awal telah membaca label temporal lalu berhenti tanpa score durable. Frozen source tetap disimpan. Recovery terpisah memindahkan assignment dan mencatat satu transisi yang diotorisasi di ROOT_SCORE_RECOVERY_R1.json. Prediksi, input, baris, formula metrik, dan gate tidak berubah.

Ledger menyimpan **dua pembacaan label aktual**: satu percobaan awal gagal dan satu recovery atas baris/ID identik. Ini satu evaluasi recipe tetap; bukan dua kandidat. Prelabel valid telah committed sebelum keduanya. Verifier recovery melaporkan kedua event, tidak menghapus riwayat. Verifikasi metrik/gate setelah score tidak membuka label lagi.

- Empat fit started dan saved; nol fit ulang.
- Synthetic full-score test mencakup 20.903 baris buatan, seluruh view, subgroup supported/unsupported, metric IDs, ledger recovery, serta gate reconstruction.
- 19 tests passed untuk core, execution amendment, prelabel compatibility, dan recovery.
- 78 perbandingan metrik direkalkulasi oleh implementasi independen: PASS.
- Recovery ledger dan gate verifier: PASS.
- Tidak ada akses Test, SampleSubmission, atau Kaggle.

Reproduksi hasil tersimpan: baca SCORE.json, METRIC_VERIFICATION.json, SCORE_RECOVERY_VERIFICATION.json, dan ROOT_DECISION.json. Recovery bersifat one-shot: jangan menjalankannya lagi. QA sintetis dapat diulang dengan `python -m pytest -q tests/test_cls_pos_ef_context_v1_score_recovery.py tests/test_cls_pos_ef_context_v1.py tests/test_cls_pos_ef_context_v1_prelabel_compat.py tests/test_cls_pos_ef_context_v1_execution_amendment.py`.

## Implikasi arah

Pertahankan champion publik 0,83225 yang sudah tercatat sebelumnya; skor itu tidak diperiksa ulang ke Kaggle pada milestone ini. Jangan mencari bobot shrink interaction dari label yang baru dibuka. Penelitian CLS selanjutnya perlu hipotesis dan novelty check tersendiri; hasil ini bukan alasan mengulang keluarga interaction yang sama tanpa kontrol baru. REG dipause atas instruksi pengguna, dengan source/preprocessing disimpan di checkpoint Luna2 be345d6 dan nol official supervised fit baru.
