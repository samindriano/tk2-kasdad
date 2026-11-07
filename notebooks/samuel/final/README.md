# Notebook individual Samuel — final review EDA

Identitas mengikuti TP1: **A06 — KasdudLovers — Samuel Indriano — 2406400524**.

- `TP2_Classification_A06_KasdudLovers_SamuelIndriano_2406400524.ipynb`
- `TP2_Regression_A06_KasdudLovers_SamuelIndriano_2406400524.ipynb`

Keduanya memuat pertanyaan, perhitungan dari raw Train, empat visual EDA, interpretasi, dan kesimpulan yang sama. Pertanyaan: **Apakah persentase laporan berlabel artifact_signal = 1 berbeda antara signal_stream kosong dan terisi, setelah mempertimbangkan waktu dan volume laporan?**

EDA memakai Classification Train. Pencantumannya pada Regression tidak mengubah target EDA menjadi threat_rank. Tidak ada label Classification yang dimasukkan ke pipeline Regression. Bagian modeling mempertahankan approach individual akademis sebelumnya (C4_HGB dan LightGBM/SVD dengan inner-selected rank blend); ini bukan dokumentasi ensemble champion kelompok terbaru.

## Menjalankan

1. Buka notebook dari checkout repository TP2 ini. Notebook membutuhkan source `src/`, laporan riset terverifikasi dan freeze JSON yang sudah dilacak Git; bukan file ipynb mandiri.
2. Pasang dependencies `python -m pip install -r requirements-research.txt` menggunakan environment Python yang sesuai dengan pins repository.
3. Sediakan official `data/raw/classification/Train.csv`. Notebook Regression juga memerlukan `data/raw/regression/Train.csv`. Alternatif: set `TP2_DATA_DIR` ke folder induk yang memiliki kedua subfolder task. Raw data tidak disertakan dalam Git.
4. Restart Kernel & Run All. Refit penuh/Test inference tetap `False`; tidak ada upload Kaggle. Eksekusi modeling temporal yang sudah ada adalah reproduksi approach individual, bukan riset baru.
5. Output receipt review ditulis terpisah ke `artifacts/samuel_eda_final_review/`, sehingga receipt akademis sebelumnya tetap terjaga.

Sumber EDA: `../12_samuel_formal_eda_availability.ipynb`; sumber modeling: `../10_samuel_final_classification.ipynb` dan `../09_samuel_final_regression.ipynb`. Builder: `scripts/finalize_samuel_individual_eda.py` (menulis ulang salinan final tanpa outputs; jalankan Run All setelah rebuild). Notebook sumber/historis tidak ditimpa.

## Pemeriksaan ketentuan tugas

PDF resmi yang ditunjuk pengguna `TP KASDAD Ganjil 2026_2027 (1).pdf` (57 halaman), SHA256 `386fa63541cc3e9b36b56ee8f8f6c479b3bbaa7753b9482d60763326f4465c0a`, dibaca ulang 7 November 2026.

- Hlm. 2: minimal satu pertanyaan EDA individual beserta eksplorasi, visualisasi, interpretasi — ada di kedua salinan.
- Hlm. 7: satu notebook per task dan template filename — nama mengikuti format tersebut.
- Hlm. 9: dapat dijalankan dari awal sampai akhir — hasil eksekusi tercatat di `reports/samuel_eda_final_review/QA.json`.
- Penggunaan EDA yang sama pada kedua notebook adalah penempatan satu kontribusi individual; PDF tidak menyatakan harus ada dua pertanyaan EDA berbeda.

Angka inti: kosong 38,13% versus terisi 34,47%; selisih +3,66 poin persentase. Setelah komposisi cycle disamakan +3,96 pp; setelah periode dan kelompok volume disamakan −2,30 pp (5 kelompok) dan −2,15 pp (10 kelompok; coverage 94,33%). Minimum 20 baris per status per kelompok. Ini asosiasi deskriptif dan bergantung pada komposisi, bukan bukti sebab-akibat atau jaminan performa model.

Belum dibuat ZIP, belum dikumpulkan, dan tidak ada submission Kaggle. Review Samuel tetap diperlukan untuk memahami jawaban demo dan memastikan paket kelompok membawa dependencies/source yang dibutuhkan.

Versi presentasi: narasi diringkas; detail administrasi/reproduksi berada di README ini. Modeling dan perhitungan grafik tetap sama.
