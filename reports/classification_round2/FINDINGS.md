# Samuel Classification Round 2 — robustness dan review point

30 Oktober 2026 · **Classification Round 2 selesai untuk review pengguna.** Belum ada final model, Round 3 atau submission.

## Kesimpulan sementara

**C4_HGB fixed threshold 0,50 adalah leader primary dan H12 pada kebijakan kontrol yang sama.** Present stream memakai C2_B original, missing stream memakai structured HGB C1_B. Macro F1 meningkat **0,820633→0,825481** pada primary dan **0,788581→0,796810** pada pooled H12. Gain masing-masing **+0,004848 / +0,008229**; bukan lompatan besar atau bukti independent holdout.

Skor pooled H12 tertinggi di seluruh kombinasi yang sudah ditetapkan adalah **C4_LR + inner_equal_cycle:0,802098**, dibanding 0,792544 untuk C2_B dengan policy yang sama. Ini berbeda dari memilih fixed-policy leader. C4_LR/equal-cycle mendapat sebagian besar gain tambahan dari H12_E; H12_L hanya berubah sedikit. Tidak ada satu threshold policy yang menang konsisten pada semua recipe/horizon.

Round 2 dirancang dari outcome Round 1 pada cycle yang sama. Freeze baru mencegah penggantian konfigurasi setelah skor baru terlihat, tetapi **tidak menghilangkan adaptive selection bias**. H12 memakai kembali records primary. Semua hasil tetap development evidence, bukan untouched validation, estimasi skor Kaggle, atau keputusan model final.

## Seluruh kandidat — fixed 0,50

| Recipe | Primary F1 | Equal-cycle F1 | H12 F1 | Missing primary F1 | Present primary F1 | Delta primary vs C2_B |
|---|---|---|---|---|---|---|
| C1_A | 0,714525 | 0,712132 | 0,676786 | 0,759610 | 0,710577 | -0,106108 |
| C1_B | 0,731019 | 0,726170 | 0,689059 | 0,781712 | 0,726465 | -0,089614 |
| C2_A | 0,801388 | 0,791598 | 0,781924 | 0,364712 | 0,823880 | -0,019245 |
| C2_B | 0,820633 | 0,812379 | 0,788581 | 0,710427 | 0,828555 | 0,000000 |
| C3_B | 0,714474 | 0,712120 | 0,677067 | 0,757085 | 0,710659 | -0,106159 |
| C3_C | 0,819430 | 0,810716 | 0,787020 | 0,700989 | 0,828254 | -0,001203 |
| C5_LOW | 0,809173 | 0,801445 | 0,775505 | 0,722460 | 0,815073 | -0,011460 |
| C5_HIGH | 0,819084 | 0,811606 | 0,789449 | 0,724486 | 0,825831 | -0,001549 |
| C4_LR | 0,824208 | 0,817275 | 0,794513 | 0,759610 | 0,828555 | 0,003575 |
| C4_HGB | 0,825481 | 0,818278 | 0,796810 | 0,781712 | 0,828555 | 0,004848 |

Kontrol C1_A/C1_B/C2_A/C2_B adalah unmodified Round 1 wrappers. C3_B dibandingkan terhadap C1_A untuk efek indicator; C3_C/C5 dibandingkan terhadap C2_B. C3-A/C3-D memakai C2_B sebagai referensi yang sama, tidak dihitung sebagai model tambahan. Seluruh per-window, per-cycle, per-class, AP/Brier, confusion dan 20 subgroups tersedia pada `comparison_tables.md` serta `results.json`.

## C3: indicator bukan penjelasan utama

Menambah indicator ke structured LR hampir tidak mengubah hasil: primary F1 **0,714525→0,714474**, missing F1 **0,759610→0,757085**. C3_B menang 3/6 windows dan 9/24 cycles terhadap C1_A; tidak memberi gain konsisten. Asosiasi marginal missingness dengan label tidak otomatis berarti indicator mempunyai informasi conditional besar setelah structured inputs.

Menghapus indicator dari hybrid juga tidak memperbaiki masalah. C3_C primary F1 **0,819430**, missing F1 **0,700989**, keduanya di bawah C2_B **0,820633 /0,710427**. C3_C memburuk pada kedua H12. Pada hybrid yang diuji, indicator justru membantu mengompensasi text absence; menghapusnya bukan solusi.

Controlled comparison **C3_B→C2_B** mempertahankan indicator, fit rows, LR C dan learned structured preprocessing. Pada history 67, validation 68–79 terdapat 1.375 missing rows:

| Komponen mean logit delta C2_B−C3_B | Nilai |
|---|---:|
| Intercept | +0,007244 |
| Numeric contribution | −0,286748 |
| Categorical contribution | −0,503071 |
| Text contribution | **0 tepat** |
| Missing-indicator contribution | +0,791985 |
| Total | +0,009410 |

Mean logit hampir saling membatalkan, tetapi **mean probability berubah−0,023149**, mean absolute probability change **0,067330**, dan 73 missing labels berubah pada cutoff 0,50. Numeric coefficient difference L2 **0,380784** pada joint-fitting comparison, versus hanya **0,001961** saat indicator ditambahkan ke structured LR pada history yang sama.

Ini menunjukkan bahwa zero text contribution tidak menjaga model structured/intercept tetap sama. Perubahan coefficients/intercept akibat joint fitting merupakan penjelasan mekanis yang didukung controlled ablation; indicator saja bukan sumber utama deficit. Accounting ini bukan causal attribution per token atau sertifikasi penyebab mengapa HGB menang. Family nonlinearity, regularization dan conditional ranking masih dapat berperan. Sigmoid nonlinear: sigmoid(mean logit) tidak sama dengan mean probability. Intercept/norm across-history juga tidak dibandingkan sebagai besaran causal karena scaler/vocabulary berubah.

## C4: routing membantu, dengan trade-off

Saved OOF diagnostic dilakukan setelah freeze dan sebelum fit pertama, tanpa model baru. Setelah itu seluruh 24 fresh control outer vectors/fitted audit states cocok bit-identical dengan Round 1; seluruh 24 diagnostic window vectors (16 route+8 hybrid control) juga identik dengan fresh outputs. Gate hanya availability NaN/blank-after-strip, tanpa target atau pemilihan branch perwindow.

| Recipe fixed 0,50 | Primary F1 | H12 F1 | Missing primary F1 | Missing H12 F1 | Present primary F1 |
|---|---|---|---|---|---|
| C2_B | 0,820633 | 0,788581 | 0,710427 | 0,630459 | 0,828555 |
| C4_LR | 0,824208 | 0,794513 | 0,759610 | 0,708544 | 0,828555 |
| C4_HGB | 0,825481 | 0,796810 | 0,781712 | 0,745006 | 0,828555 |

Pada fixed 0,50, present probabilities dan predictions tepat sama dengan hybrid. Missing predictions tepat sama dengan structured branch. C4_LR memperbaiki 6/6 primary windows dan 17/24 cycles; C4_HGB memperbaiki 5/6 windows dan 19/24 cycles. Pengecualian C4_HGB adalah W2, turun hanya **0,000057**. Kedua route membaik pada kedua H12; C4_HGB memperbaiki 22/24 H12 cycles.

C4_HGB versus C2_B primary mengurangi FP **2.955→2.753**, tetapi FN sedikit naik **3.397→3.406**. Class 1 precision **0,781742→0,793442**, recall **0,757027→0,756384**. C4_LR memberi FP **2.895** dan FN **3.331**, sehingga recall lebih tinggi dengan pooled gain lebih kecil.

EF0/EF1 juga membaik secara pooled: C2_B **0,829032/0,798249**, C4_HGB **0,834979/0,800971**, C4_LR **0,833019/0,801092**. Prevalence antar-subgroup berbeda; ini tidak menetapkan causal effect EF. Unknown/missing groups overlap dan tidak dijumlahkan.

**Missing-stream performance membaik melalui fallback**, sementara single-hybrid deficit belum diselesaikan oleh ablation/weights. Routing bukan bukti bahwa setiap window/subgroup pasti membaik. Pooled F1 dihitung dari gabungan prediksi, tidak dirata-ratakan dari F1 present/missing.

## C5: block weights tidak memberi perbaikan konsisten

Text weight 0,5× memperbaiki missing primary F1 menjadi **0,722460**, tetapi present F1 turun **0,828555→0,815073**. Primary pooled **0,809173** dan H12 **0,775505** memburuk; kalah pada seluruh 6 primary dan kedua H12 windows. Dampak pada present population yang jauh lebih besar mengalahkan perbaikan missing subgroup.

Weight 2× memberi missing primary F1 **0,724486**, tetapi primary pooled **0,819084** masih di bawah hybridoriginal. H12 pooled sedikit lebih baik **0,789449**, dengan H12_E membaik **+0,007054** dan H12_L memburuk **−0,004555**. Hanya 2/6 primary windows membaik. Tidak ada dasar memperluas weight grid dari hasil ini.

Seluruh C tetap 1. Dalam original TF-IDF basis, penalty efektif relatif berubah menjadi 4× untuk weight 0,5 dan 0,25× untuk weight 2. Norm/effective-coefficient summaries dicatat pada fit audits. Eksperimen ini menunjukkan trade-off representation/regularization, bukan pure causal feature importance atau bukti pasti overfitting.

## C6: transfer cutoff, probability shift dan allocation

| Recipe | Primary fixed | Primary pooled | Primary shrink | Primary equal | H12 fixed | H12 pooled | H12 shrink | H12 equal |
|---|---|---|---|---|---|---|---|---|
| C2_B | 0,820633 | 0,818757 | 0,820245 | 0,818988 | 0,788581 | 0,792403 | 0,790277 | 0,792544 |
| C4_LR | 0,824208 | 0,823063 | 0,824344 | 0,824847 | 0,794513 | 0,798507 | 0,796471 | 0,802098 |
| C4_HGB | 0,825481 | 0,823607 | 0,825361 | 0,823607 | 0,796810 | 0,799839 | 0,798462 | 0,799839 |
| C5_LOW | 0,809173 | 0,809185 | 0,810456 | 0,807797 | 0,775505 | 0,785550 | 0,781690 | 0,785550 |
| C5_HIGH | 0,819084 | 0,819685 | 0,818083 | 0,819685 | 0,789449 | 0,793340 | 0,790959 | 0,793340 |

Existing inner-pooled, half-shrink menuju 0,50 dan equal-cycle-inner semuanya memakai past-only inner predictions. Routes memilih threshold dari probabilitas inner route gabungan. Tidak ada outer label untuk selection, calibration atau gate. Probabilities/AP/Brier untuk recipe/window yang sama tidak berubah antarpolicies; perbedaan F1 berasal dari positive allocation.

Contoh penting adalah hybrid W2. Inner prevalence **42,22%** naik menjadi outer **45,42%**, sementara mean probability berubah **0,340839→0,445943** setelah model refit/history/sample berubah. Jadi kegagalan transfer ini tidak dapat dijelaskan hanya sebagai penurunan prevalence:

| Policy hybrid W2 | Threshold | Predicted positive rate | Macro F1 | Precision 1 | Recall 1 |
|---|---:|---:|---:|---:|---:|
| Fixed | 0,50 | 44,73% | 0,847121 | 0,838499 | 0,825704 |
| Inner pooled | 0,34 | 54,55% | 0,823348 | 0,754376 | 0,906070 |
| Half shrink | 0,42 | 49,44% | 0,839557 | 0,797508 | 0,868091 |

Outer AP **0,899997** sama untuk ketiga policies. Cutoff lama mengalokasikan terlalu banyak positives; shrink mengurangi kerusakan tetapi belum mengalahkan fixed. Ini bukan ranking model baru atau perbaikan calibration.

Equal-cycle policy memberi C4_LR pooled H12 **0,802098**, tetapi H12_E/L masing-masing **0,801135/0,801157**. Terhadap C2_B dengan equal-cycle policy yang sama, gain **+0,018447/+0,000725**. Pooled F1 bukan mean dua window tersebut. Hasil ini menarik sebagai sensitivity, bukan dasar cutoff final atau alasan memilih dari outer labels lalu menyebutnya unbiased.

Fixed 0,50 tetap kontrol terkuat untuk hybrid original dan HGB route pada primary. Half-shrink membantu beberapa structured controls, tetapi tidak universal. Equal-cycle lebih relevan pada beberapa long-horizon views; empat inner cycles tetap estimasi yang bising.

## Hubungan dengan TP1 dan rekomendasi berikutnya

Phase 11 TP1 menunjukkan zero text vector tidak membekukan structured coefficients dan bahwa average gain bisa menyembunyikan missing-stream harm. Round 2 memberi controlled evidence lebih langsung untuk target baru ini. Phase 14 positive-count transfer memisahkan ranking dari allocation; C6 juga menunjukkan bahwa cutoff dapat mengubah F1 tanpa mengubah AP. Ini pelajaran metodologis, bukan izin menyalin NB-weighting, cross-task meta labels atau stacking TP1.

Rekomendasi **Classification Round 3, belum dijalankan**:

1. **Prioritas robustness route dan aturan penerimaan.** Pertahankan C2_B, C4_LR danC4_HGB sebagai frozen controls. Tetapkan target konsistensi window/H12, missing/EF guardrails dan trade-off recall/FP sebelum eksperimen berikutnya. Semua data development telah digunakan; evidence baru harus diperlakukan sesuai reuse tersebut.
2. **Satu hipotesis transfer allocation yang lebih terarah jika disetujui.** Past-only quantile/count-transfer atau probability-scale diagnostic yang dibekukan, tanpa oracle outer prevalence. Expected information gain: membedakan score-scale drift dari perubahan ranking. Risiko: prevalence forecast meleset; bukan kebijakan yang sudah terbukti di TP2.
3. **Conditional ranking pada missing rows.** Bandingkan kesalahan LR/HGB pada records sama dan ablate structured feature groups secara terbatas; bukan menambah banyak algoritma. Ini lebih informatif daripada meneruskan text-weight grid yang lemah.
4. **Text representation hanya setelah kontrol di atas.** NB weighting/alternative n-grams tetap kandidat terpisah dengan missing-stream guardrail, bukan lanjutan otomatis. Potensi score gain harus dibaca bersama tambahan adaptivity dan interpretability cost.

Untuk **Regression**, prioritas berikutnya masih R1-D feature/history attribution dan H12 robustness: hasil primary bagus belum mengatasi counterexample horizon awal. Classification sudah memiliki kandidat robustness sederhana, sehingga Regression Round 2 dapat memberi information gain lintas-task lebih besar daripada mengejar perubahan kecil classification. Ini rekomendasi diskusi, bukan pekerjaan yang sudah dijalankan.

## Reproducibility dan QA

Protocol dibekukan sebelum diagnosis dan model fit. Original run menyelesaikan **72 fresh fits dalam 318,16 detik**, tanpa dependency reconstruction. Representative history 67 melakukan 12 fresh fits;8 outer vectors,4 variant inner vectors,4 control inner reuses dan 35 logit component arrays identik bit demi bit; seluruh fitted audit states/threshold records cocok. Tidak ada full redundant repeat atau klaim identik lintas hardware.

Independent verifier memeriksa 80 jobs×4 policies=**320 metric records**,640 per-class records,1.920 per-cycle records,80 pooled/equal-cycle aggregates,6.400 job-subgroups dan 1.600 aggregate-subgroups. Seluruh 240 unique history-policy cut records direkonsiliasi dari inner predictions.42 linear component groups merekonstruksi probability dengan maxerror **1,22×10⁻¹⁵**, text component missing tepat nol, dan learned transforms cocok dalam setiap history.

Freeze memverifikasi **161 protected checkpoint files**; kontrol Round 1 dan source/report Regression tetap sama. Old cache/protocol/results hanya dibaca. Raw identity sesuai frozen hashes; tidak ada official-test labels/predictions, raw IDs/text/vocabularies, credentials atau model binaries masuk Git. Component arrays/probabilities lokal tetap ignored.

QA akhir: **39 scoped tests lulus dalam 75,316 detik, tanpa failures/errors/skips**, melalui command enam module tests pada panduan reproduksi. Suite mencakup model adapters, independent metric/threshold reconciliation, old-checkpoint preservation, scope binding repeat receipt dan compatibility helpers Round1. Notebook valid dan Run All: **17 cells,8 code cells berurutan,0 cell errors,6 embedded PNGs**. MAIN memeriksa keenam figures dan tabel terhadap output aktual; cycle figure memuat cycle79 dan legends terpisah, subgroup figures mencantumkan denominator. Full-page HTML diekspor tetapi tidak diperiksa lewat GUI browser lokal; QA visual dilakukan pada PNG asli dan outputs/struktur notebook. nbconvert memberi warning alt-text enam image dan Jupyter memakai Windows selector fallback, tanpa execution error pada run akhir. Syntax compile juga lulus. Sumber: `verification.json`, `routing_diagnostic.json`, `behavior_diagnostics.json`, `fit_audits/`, `results.json`; seluruh commands di `docs/CLASSIFICATION_ROUND2.md`.

**STOP setelah Round 2 untuk review pengguna.** Tidak ada final model, Classification Round 3, Regression Round 2, ensemble kompleks atau Kaggle submission.
