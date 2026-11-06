# REG_MISSING_AFFINE_L1_V1 — hasil pilot R1

**Verdict: tidak eligible; kandidat tidak dipromosikan.** Independent metric QA lulus (`PASS`), tetapi scientific qualification dan exploratory benchmark eligibility keduanya `false`. Semua hasil adalah development evidence, bukan validasi independen atau fresh holdout.

## Desain dan eksekusi

- Kandidat: missing-only `clip(a + b × R06, 0, 100)`, `a` bebas dan `b ≥ 0`; baris present menyalin champion hinge tersimpan bit-identik. Parent tetap 11.01510 five-learner DIRECT + P50 hinge.
- D1 fit: E67 cycles 68–71; score E67 cycles 72–75. D2 fit: E67 cycles 68–75; score E75 cycles 76–79. D2 fit mencakup D1 score cycles 72–75. Score ID D1/D2 berbeda, tetapi bukan validasi independen dan memakai ulang development rows.
- Follow-up adaptif sesudah missing-median yang gagal; opposite offset signs sebelumnya memberi peringatan nonstationarity. Slope dapat overfit. Tidak ada rescue, sweep, penalty, cap, atau control refit.
- Source freeze R1 tetap utuh. Fit preflight R1 berhenti sebelum target-reader, ledger, dan LP karena runtime role arrays tidak membawa field support; tidak ada label atau fit/application unit yang terpakai. Source freeze R2 memperbaiki guard dengan menghitung ulang gate dari baseline beku; independen source QA R2 PASS sebelum fit.
- Dua LP affine + dua aplikasi selesai tepat sekali di bawah lock heavy-job bersama, RAM available saat fit 12.60 GB (minimum 768 MiB), dua thread. Selection labels dipakai hanya untuk fit; score labels belum dibaca pada prelabel QA. Satu candidate score pass selesai; independent metric QA melakukan replay terpisah. Tidak ada Test/Sample, upload, atau final-source/final application.
- Support: D1 342 missing fit rows, 342 baseline unik, IQR 44.090159; D2 849 rows, 849 unik, IQR 43.454799. Score missing: D1 507, D2 526; pooled unique score rows 15,953. Semua gate support lulus.

## Hasil fit

| Fold | LP status | Iterasi | Intercept `a` | Slope `b` | Objective L1 | Primal feasibility |
|---|---:|---:|---:|---:|---:|---|
| D1 | 0 | 345 | 14.068925849 | 0.789493836 | 4586.535316 | PASS (max violation 0.0) |
| D2 | 0 | 850 | 3.691369326 | 0.926318386 | 11716.025942 | PASS (max violation 0.0) |

## MAE vs zero correction (champion) dan cached prior median control

Gain didefinisikan sebagai MAE kontrol dikurangi MAE kandidat; angka negatif berarti kandidat lebih buruk. Baris present tetap identik, sehingga perubahannya terbatas pada missing rows.

| View | Rows | Candidate overall | Champion | Median control | Gain overall vs champion | Gain overall vs median | Missing n | Candidate missing | Champion | Median control | Gain missing vs champion | Gain missing vs median |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| D1 | 7462 | 11.440983 | 11.404872 | 11.422554 | -0.036111 | -0.018429 | 507 | 14.511532 | 13.980048 | 14.240289 | -0.531484 | -0.271243 |
| D2 | 8491 | 9.944353 | 9.939832 | 9.929211 | -0.004522 | -0.015142 | 526 | 13.182494 | 13.109505 | 12.938065 | -0.072989 | -0.244429 |
| Pooled | 15953 | 10.644400 | 10.625103 | 10.627721 | -0.019298 | -0.016680 | 1033 | 13.834791 | 13.536771 | 13.577201 | -0.298020 | -0.257589 |

## Arah per siklus

Gain memakai definisi yang sama: champion minus kandidat. Angka negatif berarti kandidat memperburuk MAE. Missing directions dibanding champion negatif pada 7 dari 8 siklus; cycle 79 satu-satunya missing gain positif dan overall gain kecil.

| Cycle | Rows | Missing rows | Gain overall vs champion | Gain missing vs champion | Gain missing vs median |
|---:|---:|---:|---:|---:|---:|
| 72 | 1811 | 124 | -0.062264 | -0.909351 | -0.989686 |
| 73 | 1880 | 142 | -0.023519 | -0.311381 | -0.048242 |
| 74 | 1814 | 135 | -0.047178 | -0.633929 | -0.023703 |
| 75 | 1957 | 106 | -0.013749 | -0.253833 | -0.044801 |
| 76 | 1822 | 118 | -0.013138 | -0.202860 | -0.312022 |
| 77 | 2335 | 172 | -0.002009 | -0.027268 | -0.297159 |
| 78 | 2146 | 131 | -0.006945 | -0.113769 | -0.275009 |
| 79 | 2188 | 105 | +0.002349 | +0.048944 | -0.043938 |

## Harm pada fixed groups dan tails

Harm didefinisikan sebagai MAE kandidat dikurangi MAE champion; positif berarti memburuk. Batas guard adalah `≤ +0.25` untuk setiap view, dengan dukungan minimum 100 rows. Ke-14 grup terukur pada masing-masing D1, D2, dan pooled; tidak ada UNKNOWN.

| Group | D1 harm (n) | D2 harm (n) | Pooled harm (n) |
|---|---:|---:|---:|
| EF0 | +0.035649 (n=5188) | +0.003622 (n=5738) | +0.018829 (n=10926) |
| EF1 | +0.037166 (n=2274) | +0.006397 (n=2753) | +0.020316 (n=5027) |
| impact_load_Q1 | -0.015027 (n=1716) | -0.004593 (n=1957) | -0.009467 (n=3673) |
| impact_load_Q2 | +0.057851 (n=2181) | +0.013494 (n=2143) | +0.035867 (n=4324) |
| impact_load_Q3 | +0.003460 (n=1588) | -0.006212 (n=1770) | -0.001638 (n=3358) |
| impact_load_Q4 | +0.082742 (n=1977) | +0.011239 (n=2621) | +0.041983 (n=4598) |
| missing | +0.531484 (n=507) | +0.072989 (n=526) | +0.298020 (n=1033) |
| present | +0.000000 (n=6955) | +0.000000 (n=7965) | +0.000000 (n=14920) |
| target_ge95 | +0.198018 (n=374) | +0.107821 (n=434) | +0.149571 (n=808) |
| target_lt5 | +0.426072 (n=362) | +0.084665 (n=442) | +0.238383 (n=804) |
| text_load_Q1 | +0.070817 (n=2116) | +0.012745 (n=2244) | +0.040929 (n=4360) |
| text_load_Q2 | -0.000842 (n=1696) | +0.003814 (n=2285) | +0.001831 (n=3981) |
| text_load_Q3 | +0.051891 (n=1632) | -0.001605 (n=1969) | +0.022640 (n=3601) |
| text_load_Q4 | +0.018015 (n=2018) | +0.002126 (n=1993) | +0.010120 (n=4011) |

Maximum per view: D1 `missing` n=507, harm +0.531484; D2 `target_ge95` n=434, harm +0.107821; pooled `missing` n=1033, harm +0.298020. All-view maximum is `D1/missing` n=507, harm **+0.531484**; guard FAIL.

## Qualification

| Check | Result | Evidence |
|---|---|---|
| Both fit support gates | PASS | D1 342 rows; D2 849 rows |
| Both affine LP fits | PASS | HiGHS status 0; primal feasibility PASS |
| Present route bit identity | PASS | Independent metric replay |
| Independent source QA R2 | PASS | Score labels not read |
| Independent prelabel QA | PASS | Score labels not read; fit/apply readback passed |
| Independent metric replay | PASS | 15,953 unique pooled rows |
| Scientific qualification | **FAIL** | Missing gain not positive in both folds; pooled overall gain -0.019298; pooled missing gain vs median -0.257589; all-view guard FAIL |
| Exploratory eligibility | **FAIL** | Pooled overall gain is not positive; D1 missing harm exceeds +0.10; all-view guard FAIL |

No post-result coefficient, window, constraint, or control rescue was run. Since neither gate passes, no final target-free source was prepared and no final coefficients, final fit, or application outside the authorized pilot were created.

## Receipts

- Source R2: `source_signature=1f75901ac050928afddd10a435520cc465995a7ccf55d2993fc9502d43bd59e0`; independent QA `qa_signature=0124e0749cfdc25c594d1a05d2887e0763a3b1f429a0c7cece735c020795b646`.
- Prelabel QA: `qa_signature=e8d9e6cd497e88b744c082bdf100eff06b5e75d0ea9d530fdbccaa7d95726523`.
- Fit/application manifest SHA-256: `a3cfac04bd1e2e2c4a95c074066d9beab2f6af039476a191aca292ee711404cf`.
- Score: `score_signature=02de0f016a6daed5cbae4b142ab0bbaece4ea4bec7b665e266d5b126b849872d`.
- Independent metric QA: `qa_signature=8bab5e372261bab2d4c1b01e386e32e95bcaf5b4bd655e37c838aa5741a776fd`; QA status `PASS`.
- Full signed per-cycle and group metrics remain in `SCORE_R1.json`; independent checks remain in `METRIC_QA_R1.json`. Fit states, prediction receipts, and operation ledger remain under `experiments/runs/REG_MISSING_AFFINE_L1_V1/` and are not raw targets.
