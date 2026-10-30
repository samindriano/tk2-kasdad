# Samuel Regression Round 1 — validation receipt

30 Oktober 2026 · **PASS untuk milestone Round 1; berhenti untuk review, bukan final-model approval.**

## Pemeriksaan yang benar-benar dijalankan

| Pemeriksaan | Hasil aktual | Evidence |
|---|---|---|
| Protocol freeze sebelum fit | PASS | Freeze `2026-10-30T08:39:47.852149+00:00`; fit pertama `2026-10-30T08:39:51.974429+00:00`; protocol/seed/parameter tetap |
| Seluruh eksperimen | PASS | 8 recipe × (6 primary + 2 secondary) = 64 fresh fits; status `ROUND1_COMPLETE_REVIEW_NO_FINAL_MODEL` pada `results.json` |
| Pengulangan penuh terpisah | PASS | 64 fresh fits tambahan; 64/64 prediction vectors bit-identical, fitted audit-state hashes identik, max metric difference 0 pada `reproduction.json` |
| MAE dan signed error dari vectors | PASS | Semua 64 job dihitung ulang dengan Python `math.fsum`; selisih maksimum 3,552713678800501e−15 dari numpy pada `behavior_diagnostics.json` |
| Review independen kode dan hasil | PASS | Review sebelum fit memeriksa folds, loss, preprocessing dan boundary; review sesudah fit menghitung ulang actual/predicted vectors asli/repeat, raw/clipped MAE, RMSE dan bias serta aggregate/equal-cycle metrics |
| Primary row/cycle accounting | PASS | 38.588 unique validation rows per model; 24 cycle 56–79; keenam primary validation blocks disjoint; tidak ada cycle terpecah atau fit cycle yang mencapai validation |
| Secondary accounting | PASS dengan batas interpretasi | Dua H12 memakai kembali record primary; dilaporkan terpisah, tidak dipool sebagai bukti independen tambahan |
| Fit-only preprocessing | PASS | Model menerima input allowlist dan y fit saja; numeric medians/scaler/rare-category mapping belajar dari fit fold; unknown/missing eksplisit; tidak ada target encoding atau validation labels untuk fit |
| Fitted audit state tidak berubah saat predict | PASS | Runner membandingkan hashes state sebelum/sesudah prediction pada seluruh job |
| Category unknown counts | PASS | Hitungan independen vocabulary-fit cocok dengan model diagnostics pada seluruh 8 window |
| Same-fit overlap consistency | PASS | W1/H12_E dan W4/H12_L mempunyai prediksi identik pada rows overlap untuk semua recipe |
| Unit/integration checks | PASS | `python -m unittest discover -s tests -v`: **27 tests**, 9,681 detik, tanpa failure/error/skip; mencakup 14 checks audit/EDA lama dan 13 checks Round 1 |
| Notebook Run All | PASS | nbclient mengeksekusi 19 cell, termasuk 7 code cells berurutan 1–7; 0 cell error; nbformat valid; 4 embedded PNGs |
| Source syntax | PASS | `python -m compileall -q` pada lima entry points Round 1 dan empat module Round 1 |
| Raw identity | PASS | Byte hashes Train/Test/Sample cocok dengan receipt sebelum dan sesudah eksperimen; raw tidak ditulis; tests audit lama juga lulus |
| TP1 dan pekerjaan lain | PASS pada scope yang dipakai | Canonical TP1 HEAD `b065c884728fcb428657c78806910213a8786ca3`, dua untracked paths lama tetap; executed-reference worktree HEAD `d26501a05fe9bf5bec265441de7f8aad21d3ecab` tetap clean. Tidak ada penulisan TP1 atau notebook anggota lain |

## Visual dan tabel

MAIN memeriksa langsung keempat PNG: window heatmap, per-cycle MAE, tail MAE/bias, serta actual/raw-prediction histograms. Labels, legends, jumlah baris, tanda bias dan angka sesuai hasil. Bar MAE tail memakai baseline nol; rentang histogram mencakup prediksi di luar 0–100; batas resmi ditandai. Tabel notebook dan `COMPARISON_TABLES.md` dibuat dari metrics aktual yang telah direkonsiliasi. Full per-cycle/subgroup/impact metrics tersedia pada `results.json`.

Notebook diekspor ke HTML memakai nbconvert tanpa cell execution tambahan. Full-page HTML tidak diperiksa secara visual karena jalur preview browser lokal dibatasi; QA visual dilakukan pada PNG asli dan struktur/outputs notebook. Exporter memberi warning alt-text pada 4 image; output PNG dan notebook tetap valid. Windows Jupyter juga memberi warning Proactor/selector fallback saat execute, tanpa cell error. Tidak ada klaim visual QA browser penuh.

## Batas ilmiah dan penggunaan

- Seluruh Train sudah dipakai EDA; metrics adalah development validation, bukan untouched-holdout atau skor Kaggle.
- Informasi resmi yang dikonfirmasi pengguna menggantikan UNKNOWN lama untuk target/range/metric dan izin impact/EF. Formula target dan timestamp rinci fitur tambahan tetap belum diketahui.
- R1-D adalah controlled supplied-schema comparison dengan caveat timing/provenance; tidak disebut timestamp-certified deployment model.
- R1-D unggul pada primary pooled, tetapi kalah pada H12_E dan sedikit kalah pada secondary pooled. Tidak ada final-model approval.
- Paired-history diagnostics menggunakan vectors yang sudah ada, tanpa fit tambahan. Sample size dan era berubah bersamaan; tidak membuktikan penyebab kausal atau fitur penyebab gain.
- Tidak ada official-test prediction, all-Train final refit, submission, classification training, ensemble, extensive tuning atau Round 2.

Protocol dan commands: `docs/ROUND1_PROTOCOL.md`, `docs/ROUND1_REPRODUCIBILITY.md`. Hasil dan prioritas review: `ROUND1_FINDINGS.md`. Local validation vectors tetap ignored; aggregate metrics, audit states, code, figures dan notebook menjadi bukti repository.
