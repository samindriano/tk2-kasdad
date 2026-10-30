# Samuel Classification Round 1 — protocol dan reproduksi

Milestone ini mempelajari `artifact_signal`, dengan Macro F1 sebagai metric utama, menggunakan official classification Train. Kelas 1 berarti indikasi artefak eksploit publik teramati; kelas 0 berarti indikasi tidak teramati, bukan bukti bahwa artefak pasti tidak ada. Pengguna mengonfirmasi metric, cycle Train 47–79/Test 80–91 serta izin `impact_load` dan `entry_friction`. Angka leaderboard yang disampaikan pengguna (0,78941 dan 0,82782) hanya konteks historis; bukan target validasi, tidak diverifikasi ulang, dan tidak comparable langsung dengan development folds.

Pertanyaan EDA formal Samuel tetap Regression. Analisis classification di sini adalah modeling support. Regression Round 1 dan TP1 dipertahankan tanpa perubahan; tidak ada target join TP1 atau test prediction.

## Dasar keputusan sebelum training

Audit classification terdahulu tetap menjadi sumber schema/quality. `reports/classification_round1/data_review.json` menambah ringkasan prevalence per cycle/period, EF/impact, teks kosong/panjang, missingness serta novelty kategori. Seluruh 33 cycle diringkas, tanpa mengulang audit umum atau mengekspor raw text/IDs.

Pelajaran historis TP1 yang dibaca read-only:

- `docs/phase11_results.md:5–43` dan `phase11_plan.md:30–34`: NB-weighted LR menaikkan mean Macro F1 0,003448, tetapi tidak memenuhi gate 0,006 dan missing-stream R4 memburuk 0,011954. Kandidat tidak dipromosikan. Angka historis, tidak diukur ulang di TP2.
- `phase11_results.md:43`: vector teks kosong tidak menjamin output missing-stream tetap sama karena coefficients/intercept model gabungan ikut berubah.
- `phase14_report.md:133–160,189–197` dan `phase14_plan.md:72–77`: meta-feature model berikutnya dipromosikan dengan gain sekitar 0,0063, tetapi uncertainty antarwindow tetap besar. Positive-count transfer memisahkan ranking dari perubahan allocation. Meta-feature/cross-task label TP1 tidak dipindahkan ke TP2.
- `scripts/phase7_classification.py:10–35`: threshold policies memakai history sebelumnya. Untuk TP2 digunakan inner temporal holdout yang eksplisit; bukan cutoff TP1 0,60/0,50.

Lokasi referensi: canonical `TK1/tk1-eda-3` HEAD `b065c884728fcb428657c78806910213a8786ca3`; executed reference `tk1-eda-3-individual-executed-20261009` HEAD `d26501a05fe9bf5bec265441de7f8aad21d3ecab`. Review tidak menjalankan ulang TP1.

## Validation dan threshold

Machine-readable contract: `experiments/classification_round1/protocol.json`, dibekukan bersama source/input/runtime identity di `freeze.json` **sebelum actual experiment fit**.

| Window | Fit | Validation |
|---|---|---|
| W1 | 47–55 | 56–59 |
| W2 | 47–59 | 60–63 |
| W3 | 47–63 | 64–67 |
| W4 | 47–67 | 68–71 |
| W5 | 47–71 | 72–75 |
| W6 | 47–75 | 76–79 |
| H12_E | 47–55 | 56–67 |
| H12_L | 47–67 | 68–79 |

Semua cycle utuh. Primary validation blocks disjoint dan mencakup 38.588 rows/cycle 56–79 sekali per model. H12 memakai kembali record primary dengan history/horizon berbeda; dilaporkan terpisah. Ini development validation, bukan untouched holdout. Random split tidak dijalankan.

Setiap classifier dievaluasi pada fixed threshold **0,50**. Enam recipe selain majority juga mempunyai policy inner-selected. Untuk outer fit yang berakhir di E: inner fit 47–(E−4), inner validation (E−3)–E; threshold dipilih dari 0,10–0,90 dengan step 0,01 berdasarkan inner pooled Macro F1 saja. Tie dalam 1e−12 memilih yang paling dekat 0,50, lalu threshold lebih rendah. Model kemudian di-fit baru pada seluruh outer history dan threshold dikunci untuk outer validation.

Fallback 0,50 ditentukan sebelum training jika inner fit <3 cycle/<500 rows/tidak memiliki kedua kelas, atau inner validation <200 rows/tidak memiliki kedua kelas. Majority tetap memakai threshold 0,50; policy selected-nya alias control tanpa pencarian. Inner scores bersifat selection scores, tidak dilaporkan sebagai unbiased outer performance. Tidak ada outer label untuk fitting, calibration, feature selection, imputation, vocabulary atau threshold.

Pooled Macro F1 dihitung dari seluruh prediksi gabungan, sedangkan equal-cycle adalah mean Macro F1 per cycle. Label ordering selalu `[0,1]`; undefined class metrics memakai 0. Single-class subsets ditandai; AP/balanced accuracy null jika kedua actual classes tidak tersedia. PR-AUC dilaporkan sebagai sklearn **average precision**, bukan trapezoidal integration. Confusion rows=actual, columns=predicted. Reliability 10 bin tetap dan Brier score hanya diagnostic, tanpa calibration fitting.

## Tujuh recipe tetap

| ID | Metode | Input |
|---|---|---|
| C0_A | Fit majority via positive prevalence, threshold 0,50 | Training target aggregates saja |
| C0_B | Exact EF positive rate, support ≥30, fallback global fit rate | entry_friction |
| C0_C | Standardized Logistic Regression | impact_load + entry_friction |
| C1_A | Logistic Regression structured | Expanded 12 fields |
| C1_B | HistGradientBoostingClassifier | Expanded 12 fields |
| C2_A | Ordinary TF-IDF + Logistic Regression | signal_stream + missing-stream indicator |
| C2_B | Structured + ordinary TF-IDF + Logistic Regression | Expanded fields + text + missing-stream indicator |

LR: C=1, liblinear, L2 default, max_iter=2000, tol=1e−4, class_weight=None, seed=42. No balancing variant; threshold comparison dibuat terpisah. HGB: log-loss, learning_rate=0,05, 200 iterations, 15 leaves, minleaf=30, L2=1, max_bins=255, early_stopping=False, seed=42. Thread limit 2.

Expanded allowlist: impact_load, entry_friction, cycle_code, scope_pressure, text_load, scope_trace, org_spread, module_spread, telemetry_flag, org_code, module_code, pattern_code. Parser/imputer dari Regression Round 1 diimpor read-only. Linear memakai numeric StandardScaler dan OneHotEncoder fit-only (min_frequency=30, max_categories=254, unknown infrequent-if-exist); HGB memakai native categorical encoding fit-only dengan unknown/missing −1. Imputer medians/category states dicatat.

Fitur tambahan dan text ada dalam supplied scoring schema serta materi course, tetapi timestamp/provenance operasional rinci tetap provisional. Izin impact/EF resmi terkonfirmasi pengguna. Tidak ada `record_id`, target lain, `audit_lane`, `checksum_hint`, `pulse_a` atau `pulse_b`; fitur itu tidak dinyatakan terbukti leakage. Tidak ada join label TP1.

TF-IDF: word unigram+bigram, max_features=20.000, min_df=3, max_df=0,995, sublinear_tf=True, lowercase=True, strip_accents=unicode, default token pattern. Vocabulary/IDF hanya fit rows; text kosong menjadi zero vector dengan indicator eksplisit. Korpus tanpa vocabulary setelah filtering memakai zero text columns + indicator, dicatat sebagai fallback. Hybrid memberi weight 1 pada masing-masing blok. Tidak ada NB-weighting, alternative n-gram search atau ensemble.

Subgroups mencakup EF, stream missing/present dan kombinasinya, fit-derived impact thirds, missing org/module/pattern, serta known/unseen categories. Impact terciles memakai quantiles fitur pada outer fit, bukan target atau validation; cutpoints dicatat. Groups unknown/missing bisa overlap. Same-record history comparison boleh memakai frozen probabilities yang sudah ada, tanpa fit tambahan atau perubahan recipe.

## Reproduksi dan output

Python 3.13.5 dan `requirements.txt` yang sudah ada mencakup dependencies. Raw default `data/raw/classification/{Train,Test,SampleSubmission}.csv`; override `TP2_DATA_DIR` menunjuk root yang berisi `classification/`. CLI runner juga menerima `--data-dir`. Test/Sample hanya diperiksa identity; tidak diprediksi.

```powershell
python scripts/review_classification.py
python scripts/run_classification_round1.py --freeze-only
python scripts/run_classification_round1.py
python scripts/report_classification_round1.py
python scripts/verify_classification_round1.py
python scripts/build_classification_round1_notebook.py
python -m unittest discover -s tests -v
```

Same-fit reuse W1/H12_E dan W4/H12_L menghasilkan **42 distinct outer fits +36 inner fits**, untuk 56 recipe×window evaluation jobs. Cached vectors/row positions dan metadata berada di ignored `experiments/runs/classification_round1_v1/`; tidak ada saved model weights. Resume memverifikasi key dan probability hashes. Source/input/runtime berbeda ditolak; preserve freeze dan catat amendment sebelum perubahan engineering/refit.

Representative refit yang ditetapkan sebelum eksperimen: history end 67, mencakup seluruh tujuh recipe dan inner thresholds dalam namespace terpisah. Ini memeriksa reproducibility semua keluarga tanpa full duplicate training yang tidak menambah informasi.

```powershell
python scripts/run_classification_round1.py --run-id classification_round1_repeat --only-fit-end 67 --report-dir "$env:TEMP/tp2-cls1-repeat" --force
python scripts/verify_classification_round1.py --repeat-report "$env:TEMP/tp2-cls1-repeat/results.json"
```

Partial runs wajib menggunakan run ID dan report directory terpisah; runner menolak overwrite canonical report. Source hashes memakai UTF-8 LF, CSV/PNG exact bytes; base Git HEAD adalah konteks persiapan, bukan syarat HEAD tetap sama sesudah commit. Reproducibility lintas hardware tidak diasumsikan bit-identical.

Output utama: `reports/classification_round1/results.json` (seluruh window/cycle/class/subgroups), `fit_audits/` (inner/outer fit state, threshold, hashes, sizes, waktu), `data_review.json`, `verification.json`, `figures/`, dan `FINDINGS.md`. Notebook demo Samuel: `notebooks/samuel/05_samuel_classification_round1.ipynb`; Run All memverifikasi cache atau menjalankan kembali recipe tetap jika cache belum ada. Reporting dan verification tidak melatih model.

Berhenti setelah Classification Round 1. Tidak ada classification/regression Round 2, model final, official-test prediction, leaderboard optimization atau Kaggle submission.
