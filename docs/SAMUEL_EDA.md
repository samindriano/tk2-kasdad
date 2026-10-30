# EDA Samuel — discovery, pertanyaan belum final

Regression resmi sudah tersedia dan audit aktual selesai. `threat_rank` dikonfirmasi lewat train/test/sample, bukan diasumsikan dari rencana. Tidak ada modeling, tuning atau submission.

- `notebooks/samuel/01_impact_threat_cycle.ipynb`: diagnostik relasi awal dan kuantil periode; pertanyaan lama menjadi salah satu lensa, bukan pertanyaan final.
- `notebooks/samuel/02_regression_discovery.ipynb`: distribusi target TP1/TP2, repeated impact, hubungan percycle, pola percentile percycle dan tails.
- `reports/eda/REGRESSION_DISCOVERY.md`: audit regression, riwayat REG-Lite B/Phase16 B2, lima kandidat pertanyaan dan rekomendasi utama yang belum dikunci.
- `reports/eda/discovery_summary.json` dan `samuel_regression_eda.json`: evidence agregat yang dapat dihitung ulang; source hashes tersimpan.

## Jalankan dari root repository

```powershell
python scripts/audit_data.py --tp1-data-dir "../TK1/tk1-eda-3/data/raw"
python scripts/samuel_eda.py
python scripts/discover_regression.py --tp1-data-dir "../TK1/tk1-eda-3/data/raw"
python scripts/build_samuel_notebook.py
python scripts/build_discovery_notebook.py
```

Default raw TP2 relatif terhadap root kode; `TP2_DATA_DIR` dapat mengubah root yang berisi folder regression/classification. Untuk notebook discovery, `TP1_DATA_DIR` adalah root raw TP1 (berisi regression/) jika checkout referensi tidak tersedia di lokasi tetangga. TP1 opsional: TP2 tetap dianalisis tanpa perbandingan TP1. Tidak ada data fallback atau label tambahan yang diambil dari TP1.

Pearson merangkum asosiasi linear dan Spearman asosiasi urutan pada pasangan finite. Tiga periode membagi 33 cycle unik menjadi tiga blok 11 cycle, tanpa memotong cycle atau melihat target untuk batas periode. Bin impact memakai batas bersama; perbandingan exact-impact memakai nilai yang sama persis. P10/P90 atau IQR adalah sebaran observasi, bukan confidence interval. Batas target <=5 dan >=95 adalah definisi deskriptif eksploratif, bukan preregistration atau threshold model.

Materi TP1 halaman 16 menyatakan cycle lebih besar lebih baru; receipt di reports/audits/source_definitions.json. Semantik khusus threat_rank tetap belum terkonfirmasi. Identitas average percentile dari target itu sendiri adalah audit geometri skala, bukan predictor, bukti lineage generator, atau izin merangking ulang target pada validation/test.

Rekomendasi utama meneliti perubahan conditional threat_rank pada impact_load yang sama di balik marginal target yang hampir stabil. Prinsip distribution/tail REG-Lite B relevan untuk EDA, sedangkan recipe multiclass atas skor diskret TP1 tidak otomatis cocok dengan target TP2 yang jauh lebih padat. Tidak ada klaim error over/underprediction TP2 karena belum ada prediksinya.
