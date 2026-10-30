# Protokol Samuel Regression Round 1 — dibekukan sebelum training

Kontrak machine-readable: `experiments/round1/protocol.json`. Sebelum fit pertama, runner `--freeze-only` merekam SHA protokol, input, kode training/evaluation, seed, package versions, base Git HEAD dan waktu UTC ke receipt. Training menolak protocol/code/input yang tidak cocok dengan receipt. Bug fix setelah freeze harus dicatat sebagai engineering amendment dan dibuatkan receipt baru sebelum rerun; recipe/split tidak diubah berdasarkan hasil validation.

Protocol hash memakai canonical JSON; kode training/evaluation serta helper audit/path di-hash sebagai UTF-8 dengan LF agar checkout CRLF tidak mengubah identitas semantik. CSV tetap memakai byte-exact SHA-256. Base Git HEAD adalah konteks historis persiapan, bukan syarat HEAD harus tetap sama setelah commit hasil; code/input/protocol/runtime identity tetap diverifikasi. Software unit smoke bukan experiment score dan tidak dipakai memilih recipe.

## Validation dan estimand

Primary enam expanding windows W1–W6 sesuai instruksi pengguna:47–55→56–59,47–59→60–63,47–63→64–67,47–67→68–71,47–71→72–75,47–75→76–79. Cycle utuh, fit strictly earlier, validation blocks primary disjoint. Secondary H12_E47–55→56–67 dan H12_L47–67→68–79. Secondary memakai ulang validation records primary dengan history fit berbeda; jangan menggabungkan primary+secondary sebagai tambahan observasi independen.

Primary adalah **raw, unclipped MAE**. Agregasi row-pooled menggabungkan semua validation primary sekali per model; equal-cycle mean memberi bobot sama pada24cycle56–79. Laporkan setiap window/cycle, RMSE dan mean signed error(pred−truth). Enam window adalah development validation karena seluruh train sudah dipakai EDA; bukan untouched holdout. Random split tidak dijalankan pada putaran terbatas ini.

Target-region utama memakai batas tetap0,100/3,200/3,100 (low/middle/high; lower-inclusive, upper-exclusive kecuali high mencakup100). Tail≤5 dan≥95 adalah diagnostic tambahan, bukan tercile TP1. Laporkan juga EF0/EF1, common impact group errors, missingness, prediction quantiles, over/under rates dengan equality tolerance1e−10, dan prediksi di luar0–100. Clipped[0,100] dievaluasi terpisah tanpa mengganti angka primary.

## Recipe tetap

R0-A global median; R0-B exactimpact median dengan≥30fit rows elseglobal; R0-C jointimpact×EF≥30 →impact≥30 →EF≥30 →global. Semua lookup/fallback dipelajari hanya dari fit. Threshold30 dipilih sebelum melihat hasil model validation, bukan tuning. Simpan hitungan fallback pada prediction.

R1-A adalah standardized OLS dua fitur, objective squared error sebagai baseline linear. R1-B HistGradientBoosting squared error, R1-C keluarga sama absolute error (median-oriented). Parameter tetap: learning rate0,05;200iterations;15leaves;minleaf30;L2=1;255bins;seed42;early_stopping=False. Tidak ada internal random validation yang mengambil label validation resmi. Dua fitur lengkap sehingga tidak ada target-dependent cleaning.

Satu diagnostic **R1-B-ADD** memakai parameter R1-B yang sama dengan `interaction_cst=no_interactions`. Tujuannya mengisolasi nilai tambah interaksi pada keluarga yang sama; ini satu perbandingan terarah yang ditetapkan sebelum training, bukan search.

R1-D memakai absolute error dan parameter R1-C yang sama, hanya input terstruktur diperluas. Eligibility review di `ROUND1_FEATURE_ELIGIBILITY.md`. Tidak ada signal_stream, ID, audit_lane, checksum_hint, pulse atau label task lain. Fitur awal impact dan EF resmi diperbolehkan pengguna. Ketersediaan operasional rinci kandidat lain tetap caveat, bukan klaim terverifikasi.

## Reproducibility dan boundaries

Satu proses training, threadpool limit2 untuk BLAS/OpenMP; seed42. Tidak memakai early stopping atau hyperparameter search. Numeric imputation/scale/category mapping hanya fit fold. Mixed parser deterministik tidak membaca target; approximate/missing/parse-failure dipisahkan. Native categorical encoding belajar rare grouping dari fit dan menangani unknown tanpa target encoding.

Runner menyimpan metrics nyata, waktu, fit/validation row counts, fitur dan parameter aktual, preprocessing-state hash, input hash dan environment. Local ignored `experiments/runs/round1_v1/` menyimpan validation prediction arrays/row positions untuk audit; aggregate JSON, laporan dan figures masuk Git. Tidak menyimpan trained model besar atau raw dataset di Git.

Tidak ada official-test predict, final all-train refit, submission, ensemble, classification, Round2 atau final-model selection. Candidate terbaik hanya sementara berdasarkan evidence seluruh window dan subgroup; berhenti untuk review setelah Round1.
