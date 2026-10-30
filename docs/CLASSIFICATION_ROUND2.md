# Samuel Classification Round 2 — protocol dan reproducibility

Scope: hybrid robustness, missing-stream ablation, availability routing, block balance dan threshold transfer. Target tetap `artifact_signal`, metric utama Macro F1. Seluruh artifacts/source Classification Round 1, Regression Round 1 dan TP1 dipertahankan. Hanya README utama boleh diperbarui sebagai pintu masuk milestone baru.

## Kontrak sebelum eksperimen

`experiments/classification_round2/protocol.json` menetapkan recipe dan keputusan berikut **sebelum skor Round 2 dihitung**. `freeze.json` mengikat source baru, runtime, input, Round 1 freeze, references hashes seluruh control vectors dan checkpoint files. Base HEAD adalah `6ca43f5c3ac7eb6d3fe5efb1dd9777fd68528214`, konteks persiapan yang tidak harus tetap sama setelah commit hasil.

Primary enam expanding windows sama dengan Round 1: 47–55→56–59, 47–59→60–63, 47–63→64–67, 47–67→68–71, 47–71→72–75, 47–75→76–79. Secondary H12_E 47–55→56–67 dan H12_L 47–67→68–79. W1/H12_E dan W4/H12_L berbagi fit/threshold. Primary mencakup 38.588 unique rows/cycle 56–79 per recipe; H12 memakai ulang record tersebut.

**Selection bias:** Round 2 dirancang dari hasil Round 1 pada cycle yang sama. Freeze baru membatasi keputusan tambahan setelah rancangan, tetapi tidak mengembalikan status untouched holdout. Semua hasil adalah adaptive development evidence; H12 tidak independen, dan pemenang development tidak menjamin test performance. Tidak ada leaderboard optimization atau official-test prediction.

## Recipe tetap

| ID | Eksperimen | Perubahan terhadap kontrol |
|---|---|---|
| C1_A | Structured LR control | Unmodified Round 1 wrapper |
| C1_B | Structured HGB control | Unmodified Round 1 wrapper |
| C2_A | Text-only LR control | Unmodified Round 1 wrapper |
| C2_B | Hybrid control; C3-A/C3-D reference | Unmodified Round 1 wrapper, termasuk missing indicator |
| C3_B | Structured LR + missing indicator | C1_A + satu feature availability; konten text tidak menjadi input estimasi |
| C3_C | Hybrid tanpa missing indicator | C2_B − indicator; text/structured parsing tetap |
| C4_LR | Availability routing | Present→C2_B; missing→C1_A |
| C4_HGB | Availability routing | Present→C2_B; missing→C1_B |
| C5_LOW | Text block 0,5× | C2_B dengan TF-IDF block dikalikan 0,5 pada fit/predict |
| C5_HIGH | Text block 2× | C2_B dengan TF-IDF block dikalikan 2 pada fit/predict |

LR C=1, liblinear, max_iter=2000, tol=1 e−4, class_weight=None, seed 42. HGB dan TF-IDF memakai parameter serta helper Round 1 tanpa perubahan. Structured numeric imputer/scaler/OHE dan TF-IDF vocabulary/IDF hanya fit rows. Per-history fingerprints memastikan learned structured transform enam LR terstruktur identik dan learned text transform lima text-bearing models identik. Nilai text weight bukan hasil grid search; dua arah perubahan ditetapkan untuk menguji robustness secara terbatas.

Pada C tetap, text scaling mengubah regularisasi efektif. Jika original-basis coefficient adalah β dan feature weight w, fitted coefficient menjadi β/w untuk fungsi yang sama sehingga kontribusi L2 penalty menjadi proporsional `β²/w²`. Hasil C5 bukan pure causal feature importance atau perubahan data text.

Gate routing deterministik: missing jika NaN atau `strip(signal_stream)` kosong, tepat seperti Round 1. Tidak ada learned gate, pemilihan cabang berdasarkan target, model missing terbaik per outer window, atau stream-specific threshold. Inner branch predictions dibentuk pada rows/history yang sama sebelum satu threshold dipilih untuk probabilitas route gabungan.

## Empat threshold policies

Semua sepuluh recipe memakai empat policies yang dilaporkan terpisah; **fixed 0,50 adalah kontrol utama**.

1. `fixed_0.5`: cutoff tetap 0,50.
2. `inner_pooled`: policy Round 1; fit sampai E menggunakan inner fit 47..E−4 dan validation E−3..E; maximize pooled Macro F1 pada grid 0,10..0,90 step 0,01.
3. `inner_shrink50`: `0.5 + 0.5*(t_inner_pooled−0.5)` tanpa pembulatan tambahan atau optimization baru.
4. `inner_equal_cycle`: maximize mean Macro F1 empat inner cycle pada grid/tie rule yang sama.

Tie dalam 1 e−12 memilih cutoff paling dekat 0,50, lalu lebih rendah. Fallback 0,50 jika inner fit <3 cycle/<500 rows/tidak memiliki dua kelas, atau inner validation <200 rows/tidak memiliki dua kelas/<4 cycle. Tidak ada label outer untuk fitting, calibration, feature choice, threshold atau gate. Semua model refit pada full outer history setelah threshold dipilih; control inner predictions Round 1 yang valid dipakai ulang.

Macro F1 selalu memakai labels `[0,1]`, zero-division=0. Confusion rows=actual, columns=predicted. Pooled metric dihitung ulang dari gabungan prediksi, bukan mean subgroup/window F1; equal-cycle dilaporkan terpisah. AP adalah sklearn average precision, bukan trapezoidal PR-AUC. Single-class actual subsets ditandai; AP/balanced accuracy null bila kedua kelas tidak ada. Brier/reliability hanya diagnostic, tanpa calibration fitting.

## Koefisien, probabilitas dan interpretasi

Fresh controls memungkinkan ekstraksi coefficient/intercept yang tidak tersimpan penuh pada Round 1. Summary Git hanya menyimpan coefficients fitur numeric dengan nama aman, block dimensions/norms, intercept, missing coefficient dan preprocessing hashes; tidak ada token vocabulary, category values atau full model weights.

Per-row additive logit components (intercept, numeric, categorical, text, missing indicator) disimpan pada ignored cache. Sum direkonsiliasi terhadap probability menggunakan sigmoid; text contribution harus tepat nol pada missing stream. Ringkasan per present/missing dihitung dari komponen tersebut. Untuk fit end 55 dan 67, group-level prediction range mencakup H12; laporan membedakan range ini dari window 4 cycle.

C1_A→C3_B menguji indicator tanpa text. C2_B→C3_C menguji net refit effect penghapusan indicator. C3_B→C2_B mempertahankan indicator dan structured transform yang sama sehingga perubahan missing logits dapat ditelusuri ke coefficients/intercept yang dipelajari bersama text. Ini tidak mengisolasi penyebab kausal per token, perubahan optimization, atau generalisasi test. Logit component attribution bersifat accounting, bukan intervensi kausal.

## Urutan dan compute

Sesudah freeze, dua routes fixed 0,50 lebih dulu dievaluasi dari saved Round 1 out-of-fold probabilities, tanpa Round 2 fits. Kemudian 24 control outer refits, 24 variant outer fits dan 24 variant inner fits menghasilkan **72 actual fits** pada enam histories. Routes tidak melatih estimator. Fresh controls harus bit-identical pada probabilities dan fitted audit-state hashes terhadap Round 1; fresh routed vectors juga dibandingkan eksplisit terhadap 24 diagnostic window vectors (16 route +8 hybrid control).

Representative reproduction yang ditetapkan sebelumnya memakai history end 67: delapan outer danempat variant inner refits, total 12. Inner control vectors tetap dipakai ulang. Ini bukan full duplicate training semua histories.

## Menjalankan ulang

Python 3.13.5, dependencies pada `requirements.txt`, seed 42 dan threadpool limit 2. Raw default `data/raw/classification/{Train,Test,SampleSubmission}.csv`; `TP2_DATA_DIR`/`--data-dir` menunjuk root berisi `classification/`. Test/Sample hanya identity checks.

```powershell
python scripts/run_classification_round2.py --freeze-only
python scripts/run_classification_round2.py --diagnostic-only
python scripts/run_classification_round2.py
python scripts/analyze_classification_round2.py
python scripts/verify_classification_round2.py
python scripts/report_classification_round2.py
python scripts/build_classification_round2_notebook.py
```

Runner memverifikasi cache key, rows, probabilities dan logit components. Run IDs wajib memakai prefix `classification_round2_`; namespace dependency reserved. Report destinations dibatasi pada `reports/classification_round2/`/subfolder atau dedicated system-temp subfolder. Run parsial wajib memakai ID dan output terpisah agar tidak menimpa canonical results. Tidak ada penulisan ke Round 1.

```powershell
python scripts/run_classification_round2.py --run-id classification_round2_repeat --only-fit-end 67 --report-dir "$env:TEMP/tp2-cls2-repeat" --force
python scripts/verify_classification_round2.py --repeat-report "$env:TEMP/tp2-cls2-repeat/results.json"
```

Saved Round 1 vectors biasanya tersedia di local ignored `experiments/runs/classification_round1_v1/`. Pada checkout baru yang tidak memilikinya, tambahkan `--bootstrap-dependencies`: hanya empat controls yang diperlukan direkonstruksi dalam `classification_round2_dependencies`, **tanpa** menulis Round 1, dan wajib cocok bit-identical dengan original vector/state hashes di tracked fit audits. Reconstruction fits dihitung terpisah. Jika tidak cocok, stop dengan mismatch; jangan menimpa freeze atau memakai tolerance sebagai pengganti. Reproduksi lintas hardware tidak diasumsikan bit-identical.

Notebook `notebooks/samuel/06_samuel_classification_round2.ipynb` menjalankan verification/reporting dan memakai cache atau frozen runner jika cache belum tersedia. Semua tabel/figures berasal dari output aktual. Detailed machine JSON memakai format compact untuk membatasi whitespace; tabel Markdown/notebook menjadi tampilan review. Core/input/runtime mismatch memerlukan engineering amendment dan receipt baru sebelum refit; checkpoint lama tetap utuh.

Unit/integration scope Round 2 dan helper Round 1 dapat dijalankan dengan:

```powershell
python -m unittest tests.test_classification_round2_models tests.test_classification_round2_metrics tests.test_classification_round2_evaluation tests.test_classification_round2_results tests.test_classification_models tests.test_classification_evaluation -v
```

Hasil aktual dicatat di `reports/classification_round2/FINDINGS.md`. Tests ini tidak menjalankan ulang builder/report/verifier Round 1 yang menulis checkpoint tersebut.

Berhenti untuk review setelah Classification Round 2. Tidak ada final model, ensemble kompleks, calibration search, Round 3, Regression Round 2, official-test predictions atau Kaggle submission.
