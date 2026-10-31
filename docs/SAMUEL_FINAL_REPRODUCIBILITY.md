# Reproducibility notebook Samuel — TP2

Notebook formal EDA [availability12](../notebooks/samuel/12_samuel_formal_eda_availability.ipynb), [Classification10](../notebooks/samuel/10_samuel_final_classification.ipynb), dan [Regression09](../notebooks/samuel/09_samuel_final_regression.ipynb) memerlukan repository source, protokol/evidence yang tercatat di repository, dependencies, serta official Train. Notebook tidak dimaksudkan sebagai file terisolasi tanpa source code pendukung. Notebook TTR lama [11](../notebooks/samuel/11_samuel_formal_eda_a.ipynb) dipertahankan sebagai arsip dan tidak diedit.

## Data dan environment

Lingkungan validasi menggunakan Python 3.13 dan dependencies repository. Setup di PowerShell:

    python -m pip install -r requirements-research.txt

Sediakan hanya official Classification Train dan Regression Train untuk eksekusi inti:

    data/raw/classification/Train.csv
    data/raw/regression/Train.csv

Alternatifnya, atur TP2_DATA_DIR ke direktori induk yang memiliki subfolder classification dan regression. Resolver nama CSV tidak membedakan kapitalisasi dan menolak nama ambigu. File Test, SampleSubmission, submission, dan credentials tidak diperlukan untuk evaluasi temporal inti. Input raw tidak diedit dan tidak masuk Git.

## Run All

Dari root repository, jalankan masing-masing perintah dalam kernel baru:

    python -m jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=1800 notebooks/samuel/12_samuel_formal_eda_availability.ipynb
    python -m jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=1800 notebooks/samuel/10_samuel_final_classification.ipynb
    python -m jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=1800 notebooks/samuel/09_samuel_final_regression.ipynb

Untuk pemeriksaan portable dengan source tracked dan dua official Train saja, jalankan:

    python scripts/qa_samuel_availability_final.py

QA membuat salinan sementara dari source tracked pada commit asal dan perubahan akademis Samuel yang diotorisasi. Salinan hanya memperoleh dua file Train; Test, ignored experiment caches, model binaries, submissions, serta credentials tidak disalin. Setiap notebook dijalankan melalui fresh kernel. Dependency yang sudah terpasang dipakai tanpa instalasi ulang. Receipt dan hash input disimpan di [reports/academic_availability/QA.json](../reports/academic_availability/QA.json); empat grafik EDA disimpan di folder figures untuk visual review.

## Apa yang dihitung

EDA12 membaca official Classification Train. Ia membangun label availability dengan fungsi yang sama seperti pipeline Classification, memvalidasi target biner dan ID, lalu menghitung prevalensi marginal, periode/cycle, distribusi text_load, dan kontras standardisasi dengan minimum 20 baris per availability per cell.

Classification10 menjalankan dua komponen C2_B dan C1_B pada cycle 47–75, memprediksi 76–79, lalu mengevaluasi routing C4_HGB pada threshold 0,50. Regression09 menjalankan inner temporal selection pada 47–71 ke 72–75, satu outer fit 47–75 ke 76–79, dan baseline median. Itu menunjukkan pipeline model yang sebenarnya pada satu representative window, bukan mengulang semua ratusan fit historis.

Imputasi, encoding, scaling, TF-IDF/IDF, SVD, dan estimator dipelajari hanya dari data fit. Tabel primary/H12 berasal dari laporan eksperimen terverifikasi dengan provenance; skor Kaggle public berasal dari manifests sebelumnya dan merupakan evaluasi terpisah. Tidak ada upload baru atau leaderboard tuning dalam QA ini.

## Inference opsional

Refit seluruh Train dan prediksi official Test dinonaktifkan secara default. Mengaktifkan inference memerlukan Test dan SampleSubmission; proses opsional memeriksa schema, IDs, serta range, dan menyimpan CSV lokal. QA ini tidak menjalankan refit inference, tidak membuat submission, dan tidak mengunggah ke Kaggle.

Untuk pengumpulan akhir, notebook memerlukan source pendukung, dependencies, dan protokol/evidence yang dirujuk atau paket yang disetujui asdos. Checklist nama export resmi ada di [SAMUEL_FINAL_COMPLIANCE.md](SAMUEL_FINAL_COMPLIANCE.md). Arsip TTR EDA dan EDA temporal lama tetap dipertahankan.
