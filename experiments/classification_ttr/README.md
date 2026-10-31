# Controlled TTR ablation — Samuel

Pertanyaan: apakah satu statistik keragaman token menambah performa pada recipe Classification yang telah dibekukan? Ini eksperimen pengembangan terbatas, bukan EDA baru atau pembaruan model resmi.

## Desain yang ditetapkan sebelum fit

| ID | Recipe dasar | Tambahan |
|---|---|---|
| S0 | C1_A structured Logistic Regression | Tidak ada |
| S_AV | C1_A | Indikator stream missing, kontrol diagnostik |
| S_LEN | C1_A | Jumlah token regex |
| S_TTR | C1_A | TTR |
| S_BOTH | C1_A | TTR + jumlah token |
| H0 | C2_B structured + TF-IDF Logistic Regression | Tidak ada |
| H_LEN | C2_B | Jumlah token regex |
| H_TTR | C2_B | TTR |
| H_BOTH | C2_B | TTR + jumlah token |
| C4_HGB | H0 untuk present; C1_B HGB untuk missing | Kontrol resmi |
| R_LEN / R_TTR / R_BOTH | H_LEN / H_TTR / H_BOTH untuk present | Fallback C1_B identik dengan C4 |

Tidak ada tuning hyperparameter, threshold, transformasi nonlinear TTR, atau entropy. Tujuh tambahan LR × enam cutoff = 42 fit baru. Tiga komponen kontrol menggunakan prediksi historis yang diverifikasi terhadap hash audit; jika cache historis tidak tersedia, runner melakukan refit dan menuntut hash prediksi identik.

Regex mengikuti EDA A: `[A-Za-z0-9]+` sesudah lowercase. TTR = jumlah tipe unik / jumlah token. Missing adalah NaN atau string kosong/whitespace sesuai wrapper lama. TTR pada missing atau teks tanpa token adalah undefined; count pada missing juga undefined, sementara teks nonblank tanpa token mempunyai count 0. Undefined values diimputasi median fit saja, lalu seluruh tambahan diskalakan dengan StandardScaler fit saja. Tidak ada indikator baru selain kontrol eksplisit S_AV. Blok tambahan ditempel sesudah desain dasar yang tidak berubah; urutan kolom diterapkan konsisten saat prediksi.

`repeat_ratio = (n - n_unique) / n` sama dengan `1 - TTR`, sehingga tidak ditambahkan. Definisi lain berupa proporsi seluruh kemunculan token yang tipenya berulang tidak identik dengan TTR; definisi tersebut tidak diuji. Tokenizer TF-IDF tetap tokenizer recipe lama, bukan diganti dengan regex EDA.

`text_load` sudah berada di structured baseline; count-only controls membantu memisahkan informasi panjang dari TTR. Perbandingan H_BOTH–H_LEN lebih tepat untuk kontribusi TTR setelah exact count tersedia daripada H_BOTH–H_TTR saja. Eksperimen ini menguji tambahan scalar linear pada LR, bukan seluruh kemungkinan pemanfaatan keragaman.

## Batas data dan keputusan

Hanya official Classification Train (50.608 baris, cycle 47–79). SHA-256, fitur, enam primary windows, dua H12 windows, cutoff 0,50, seed 42, versi dependency, dan gate tertulis dalam `protocol.json`. Setiap keluarga validasi berisi 38.588 baris unik. H12 memakai ulang observasi primary dengan training history berbeda; keduanya bukan replikasi independen. EDA dan pemilihan hipotesis telah melihat Train sehingga ini tetap development evidence.

Gate praktis yang dipilih sebelum hasil: gain primary ≥0,002; equal-cycle ≥0,0015; H12 ≥0,002; minimal 5/6 primary windows positif; tidak ada window atau F1 kelas yang turun lebih dari 0,002; present-stream tidak menurun pada kedua keluarga; missing-route probability harus identik. Ini batas keputusan operasional, bukan ambang signifikansi statistik atau ketentuan tugas. Hanya R_TTR/R_BOTH dapat dipertimbangkan untuk satu upload bersyarat; public leaderboard tidak digunakan memilih kandidat.

Freeze awal mengikat protocol dan kode fit. Amendment pelaporan dicatat sebelum hasil ablation diperiksa: normalisasi metadata threshold saat membandingkan aggregate dengan laporan lama (`None` + range vs `0.5`), kemudian toleransi <1e−15 hanya untuk pembulatan reduksi mean probability/Brier pada kontrol. Semua metrik lainnya dan hash probabilitas kontrol tetap diperiksa persis. Tidak ada perubahan recipe, fitur, gate, prediksi, atau fit identity. Runner pertama memakai kode awal yang sudah dimuat; dua assertion pelaporan tersebut menghentikan pembuatan laporan, lalu versi terkoreksi memakai checkpoint fit terverifikasi. Versi saat ini dapat dijalankan langsung. Detail checksum dan waktunya ada di `freeze.json`.

## Reproduksi dari raw data

Dari root repository, pasang `requirements.txt` (versi lingkungan referensi tercatat di protocol). Letakkan official Train pada struktur `data/raw/classification/` yang dijelaskan README repository, atau atur `TP2_DATA_DIR` ke direktori induk task yang didukung `src/utils/paths.py`. Dataset, cache prediksi, dan model binaries tidak masuk Git.

```powershell
python -m unittest tests.test_classification_models tests.test_classification_round2_models tests.test_classification_ttr -v
python -u scripts/run_classification_ttr_ablation.py
python scripts/verify_classification_ttr.py
python scripts/summarize_classification_ttr.py
```

Runner dapat melanjutkan checkpoint per fit, serta menolak input/kode/cache dengan identitas tidak cocok. Tanpa cache, semua komponen kontrol dan variant akan dilatih dari raw Train; laporan audit historis yang diperlukan ada di Git. Hash identik kontrol mensyaratkan lingkungan referensi yang sama. Tidak diperlukan ignored personal cache untuk memulai. Runner tidak membaca official Test, membuat official inference CSV, atau mengirim submission.

Hasil utama: `reports/classification_ttr/FINDINGS.md`; detail mesin: `results.json`; pemeriksaan independen: `QA.json`; tabel primary/H12, window, kelas, subgroup, dan cycle disimpan di folder laporan yang sama.
