# Reproduksi Samuel Regression Round 1

Round 1 adalah development experiment yang selesai, bukan model final. Delapan recipe tetap × delapan window menghasilkan 64 fit. Satu pengulangan terpisah menghasilkan 64 fit baru; seluruh validation prediction vectors bit-identical, fitted audit-state hashes identik dan selisih MAE/RMSE/signed error nol. Receipt ada di `reports/round1/reproduction.json`.

## Environment dan input

Gunakan Python **3.13.5** dan versi pada `requirements.txt`. Identity training membekukan numpy 2.4.2, pandas 2.3.3, scipy 1.18.0, scikit-learn 1.8.0 dan threadpoolctl 3.6.0. Seed 42; threadpool BLAS/OpenMP dibatasi 2 pada setiap fit/predict. HGB tidak memakai early stopping atau search.

Raw resmi tetap lokal dan ignored:

```text
data/raw/regression/Train.csv
data/raw/regression/Test.csv
data/raw/regression/SampleSubmission.csv
```

Nama CSV tidak peka kapitalisasi; nama ambigu ditolak. Untuk root lain, set `$env:TP2_DATA_DIR = 'D:\course-data\tp2'`; root tersebut harus berisi folder `regression/`. Runner juga menerima `--data-dir`. Gunakan environment override untuk notebook, reporting, behavior diagnostics dan tests yang dijalankan bersama. Path default berbasis lokasi repository; notebook dibuka dari root/subfolder repository.

| Input | SHA-256 byte-exact |
|---|---|
| Train | `756523ff8156a9cd776e52249f7e2a8f23dcec432b8d10570e55ff6db0e0f082` |
| Test | `4e6c381d582ed24f92d17f620f4f0ad25e2e0def29eb57a91a5482867f6afe3c` |
| SampleSubmission | `63a51f4b7dfbb0eec9800099407abd8be14d25ec18a533cd39dc854ffc190a32` |

Test/Sample dibaca untuk identity checks saja. Tidak ada official-test prediction atau submission. Hanya label Train dipakai untuk fit/development evaluation, dengan cycle fit strictly earlier dari validation.

## Receipt yang harus dipertahankan

`experiments/round1/protocol.json` menyimpan folds, recipe, fitur, loss, parameter, seed, support threshold dan aturan evaluasi. `freeze.json` mencatat identity sebelum fit pertama pada **2026-10-30 08:39:47.852149 UTC**.

- Freeze signature: `7a6bfcb7900fa86f2df7e4ade61e9aac3ec617275a556fd3a513a5cdaf5ead02`.
- Canonical protocol digest: `094e1687bb71ebf892e06980e8a36ad88e97f82467da27409f8dd04c767e8322`.
- Base preparation HEAD: `c4e469d612db2dd09caa4c8f89e0f92722fc8d88`; ini konteks historis, sehingga commit hasil tidak harus mempunyai HEAD yang sama.

Code identity mencakup `round1_models.py`, `round1_evaluation.py`, `run_round1.py`, helper `audit.py` dan `paths.py`. Kode di-hash sebagai UTF-8 dengan LF agar checkout CRLF tetap kompatibel; CSV di-hash sebagai bytes dan protocol memakai canonical JSON. Runtime juga harus cocok. Reporting/notebook tidak mengubah recipe training.

Jika input, runtime, protocol atau training code berbeda, runner menolak fit dan cache reuse. `--freeze-only` memakai ulang receipt yang cocok dan menolak perbedaan; jangan menghapus atau menimpa receipt untuk melewati pemeriksaan. Perubahan engineering di masa depan memerlukan amendment dan receipt baru sebelum eksperimen; hasil Round 1 tetap dipertahankan.

## Jalankan dari awal

Dari root repository, setelah environment aktif:

```powershell
python -m pip install -r requirements.txt
python scripts/run_round1.py --freeze-only
python scripts/run_round1.py
python scripts/report_round1.py
python scripts/analyze_round1_behavior.py
python scripts/build_round1_notebook.py
python -m unittest discover -s tests -v
```

Runner pertama melatih 64 job jika cache belum ada. Run berikutnya memverifikasi recipe key, validation row positions dan prediction hashes sebelum memakai cache. Tidak ada final fit ke semua Train. `--force` meminta refit seluruh job; gunakan run ID dan report directory terpisah untuk menjaga hasil asli.

Notebook `notebooks/samuel/04_samuel_regression_round1.ipynb` dapat dijalankan dengan **Restart & Run All** pada kernel environment tersebut. Builder menjalankan semua cell melalui nbclient dan menyimpan outputs. Notebook memverifikasi freeze dan vectors; jika cache belum tersedia, ia menjalankan ulang recipe tetap. Angka tabel berasal dari results, grafik dibuat dari vectors yang diverifikasi; tidak ada angka eksperimen yang ditulis sebagai pengganti pengukuran.

## Pengulangan penuh dan audit numerik

Perintah yang benar-benar dijalankan pada milestone ini:

```powershell
python scripts/run_round1.py --run-id round1_repeat --report-dir "$env:TEMP/tp2-round1-repeat" --force
python scripts/verify_round1_results.py --repeat-report "$env:TEMP/tp2-round1-repeat/results.json"
```

Run asli: 64 fresh fits, 62,41 detik. Repeat: 64 fresh fits, 53,59 detik. Ini wall time runner lokal, bukan janji kecepatan komputer lain. Verifier mengecek freeze, validation vectors terhadap actual Train dan membandingkan hashes serta metrics repeat. Recompute MAE/bias dengan `math.fsum` berbeda maksimum 3,55e−15 dari agregasi numpy. Pengulangan bit-identical telah terbukti pada environment lokal ini; tidak diklaim universal lintas hardware/BLAS.

`python scripts/verify_round1_results.py` tanpa repeat path memverifikasi metrics vectors dan membuat ulang behavior diagnostics, tanpa fit baru. `reproduction.json` adalah receipt dari repeat penuh yang benar-benar dijalankan; tidak dibuat ulang sebagai PASS tanpa repeat evidence.

## Output dan review

| Lokasi | Isi |
|---|---|
| `reports/round1/results.json` | Semua metrics per job/cycle/EF/region/tail/impact, aggregate primary/H12, seed/config/input identity |
| `reports/round1/fit_audits/*.json` | Fit-only preprocessing state, fit/validation sizes, parameter aktual, fit start time/duration, hashes dan fallback/unknown diagnostics |
| `reports/round1/behavior_diagnostics.json` | Recompute independen, perbandingan history pada record sama, category novelty, difficult impact groups |
| `reports/round1/COMPARISON_TABLES.md` | Tabel lengkap yang dihasilkan dari output aktual |
| `reports/round1/figures/` | Empat grafik evaluasi |
| `reports/round1/ROUND1_FINDINGS.md` | Interpretasi, limitations dan prioritas Round 2 untuk review |
| `experiments/runs/round1_v1/` | Ignored local `.npz` validation vectors/row positions dan metadata; bukan model weights atau prediksi Test |
| `experiments/runs/round1_repeat/` | Cache repeat terpisah, juga ignored |

Raw data, credentials, validation row arrays dan trained models tidak masuk Git. Yang disimpan di repository adalah kode, protocol/receipt, aggregate metrics/audit states, notebook, figure dan laporan. Tidak ada TP1 refit/modification atau classification training.

Berhenti setelah Round 1. R1-D adalah leader primary sementara; R1-C sedikit lebih baik pada pooled H12. Review hasil sebelum memutuskan Round 2, milestone classification atau final model.
