# Reproduksi notebook modeling Samuel

Notebook [Regression09](../notebooks/samuel/09_samuel_final_regression.ipynb) dan [Classification10](../notebooks/samuel/10_samuel_final_classification.ipynb) membutuhkan repository source, committed protocol/evidence JSON, dependencies, dan raw Train resmi. Keduanya tidak diklaim self-contained sebagai satu file tanpa repository. [EDA11](../notebooks/samuel/11_samuel_formal_eda_a.ipynb) tetap utuh.

## Setup dan Run All

Lingkungan referensi Python3.13.5. `requirements-research.txt` mencakup pinned core dan LightGBM4.6.0; jangan mengubah versi atau recipe hanya untuk meloloskan assertion.

```powershell
python -m pip install -r requirements-research.txt
python -m jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=1800 notebooks/samuel/09_samuel_final_regression.ipynb
python -m jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=1800 notebooks/samuel/10_samuel_final_classification.ipynb
```

Default input: `data/raw/regression/Train.csv` dan `data/raw/classification/Train.csv`. Alternatif: set `TP2_DATA_DIR` ke folder induk yang berisi subfolder `regression/` dan `classification/`. Resolver menerima nama CSV case-insensitive, tetapi menolak ambiguitas. Hash kedua Train diverifikasi; data tidak diedit atau masuk Git.

Classification menjalankan dua fit komponen C2_B dan C1_B pada cycle47–75, memprediksi76–79, kemudian melakukan routing/cutoff0,50 dan menghitung metrics. Regression menjalankan dua inner fits47–71→72–75 untuk memilih konfigurasi/bobot dari grid lama, satu outer fit47–75→76–79, serta dua baseline median. Ini demonstrasi protokol yang sudah dibekukan, bukan pencarian model baru.

Imputasi, encoder, scaler, TF-IDF/IDF, SVD, dan estimator belajar hanya pada fit fold. Tabel primary/H12 berasal dari committed reports dengan provenance; public Kaggle dari accepted manifests. Tidak ada refresh API atau leaderboard tuning.

## Pemeriksaan tanpa cache

```powershell
python scripts/validate_samuel_final_notebooks.py --execute
```

Validator membuat salinan source/evidence dan **dua Train saja**, memakai fresh kernels. Tidak ada ignored `experiments/runs`, Test, SampleSubmission, model binaries, atau credentials. Ini file isolation pada dependencies terpasang, bukan fresh dependency installation. Opsi `--check-preservation` memeriksa bytes423 historical tracked files, termasuk EDA11, pada checkout referensi; perbedaan line endings di platform lain dapat memengaruhi byte checks tanpa berarti numerical failure.

Core metrics dan probability/prediction hashes direkonsiliasi dengan W6 historis. Jika berbeda, selidiki input/order/software; jangan melonggarkan assertion tanpa review. Unit tests yang dijalankan serta visual-review status ada di [VALIDATION.json](../reports/samuel_final/VALIDATION.json).

## Refit dan inference opsional

Default `RUN_FULL_TRAIN=False` dan `RUN_FULL_TRAIN_INFERENCE=False`. Mengaktifkannya memerlukan official Test dan SampleSubmission untuk task terkait. Sel mempelajari ulang pipeline pada semua Train, mengecek schema/ID/range, dan menulis CSV lokal di ignored `artifacts/samuel_final/` dengan guard terhadap penggantian output yang berbeda. Tidak ada upload atau overwrite accepted manifests/CSV.

**Tahap opsional tidak dijalankan pada eksekusi core milestone ini.** Accepted manifests tetap menjadi bukti refit/submission sebelumnya. Notebook menunjukkan kode tahap tersebut, tetapi tidak mengklaim output optional baru. Tidak perlu menjalankannya untuk demo evaluasi temporal.

Packaging akhir perlu membawa source pendukung (`src/`, helper `scripts/`, selected committed protocols/reports, requirements) atau mengikuti format lain yang disepakati asdos agar notebook tetap runnable. User belum mengotorisasi penguncian ZIP atau pengumpulan tugas. Nama export resmi tercantum pada [compliance checklist](SAMUEL_FINAL_COMPLIANCE.md); milestone berhenti pada review notebook.
