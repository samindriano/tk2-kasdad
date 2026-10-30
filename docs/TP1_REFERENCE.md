# Referensi TP1 — ditinjau read-only

Referensi pengguna: https://github.com/samindriano/tk1-eda-3. GitHub metadata sesi ini mengembalikan nama canonical `samindriano/tk1-kasdud` (private); remote checkout lokal masih memakai URL lama. Repo TP2 berdiri sendiri dan TP1 tidak diubah.

Checkout yang dibaca: `TK1/tk1-eda-3`, branch `main`, HEAD `b065c884728fcb428657c78806910213a8786ca3`. Ada file untracked milik pekerjaan lama; tidak disentuh. Tinjauan ini tidak menjalankan ulang model atau membuktikan reproduksi metrik TP1.

## Struktur dan pengalaman yang dapat dipakai

- `src/tp1/audit.py:26–34` dan `src/tp1/io.py:13–19` menetapkan target eksplisit: classification `entry_friction`, regression `impact_load`. Loader memeriksa skema, ID dan finite target (`io.py:42–75`).
- Raw TP1 kedua task: train 50.608 × 17, test 29.421 × 16. Ada 16 predictor termasuk ID. TP2 classification menambah `impact_load` dan mengganti target menjadi `artifact_signal`; `entry_friction` tetap muncul sebagai predictor.
- Audit TP2 merekonsiliasi record berdasarkan ID ke raw TP1 secara read-only. Semua 50.608/29.421 ID dan nilai kolom bersama classification cocok. Detail mismatch dan hash input TP1 di JSON audit, opsi `--tp1-data-dir` untuk mengulang.
- Pelajaran struktur: kode pemuatan/audit terpisah dari notebook; environment override untuk path; raw dan output besar ignored; laporan agregat, hash, split/cycle dan recipe tercatat. TP2 memakai pola ini tanpa menyalin model atau menambah label lintas-task.

## Perbedaan ringkasan resep dengan final yang ditemukan

Pengguna menyebut final classification Samuel NB-weighted TF-IDF + Logistic Regression, dan regression LightGBM multiclass + conditional median dengan Ridge fallback. Repository berisi sejarah yang lebih panjang:

- `docs/phase11_completion.md` menyatakan percobaan NB-ratio-weighted balanced LR **tidak dipromosikan**: kenaikan rata-rata Macro F1 +0,003448 di bawah gate +0,006; subgroup missing stream turun −0,011954. Ini pembacaan laporan lama, bukan pengukuran baru.
- `docs/FINAL_CONTEXT.md:16–20` dan `deliverables/README.md:5–20` menunjuk final individual Phase14 classification dan Phase16 B2 regression.
- `docs/phase16_report.md:46–58` menjelaskan multiclass impact distribution 300 round, median decoder, soft gate entry_friction saat stream tersedia, serta fallback saat stream hilang. Ini tidak identik dengan ringkasan satu resep Ridge sederhana.

Karena itu resep awal diperlakukan sebagai konteks historis, bukan kontrak final TP2. Tidak ada hasil TP1 yang membenarkan pemilihan model TP2 sebelum target, metric, leakage dan validasi TP2 ditetapkan.
