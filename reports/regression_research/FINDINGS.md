# Samuel — Autonomous Regression Research Findings

30 Oktober 2026 · **Target minimum dan stretch goal tercapai.** Phase selesai untuk review; belum ada keputusan model final permanen. Pertanyaan dan kesimpulan EDA formal dipertahankan.

## Hasil terbaik dan keputusan review

**Strong provisional candidate: LightGBM MAE + structured native + TF-IDF/SVD32 + inner-selected rank blend (`lgb_text_inner_v1`).** Kaggle mengonfirmasi ref **57810150**, status **COMPLETE**, public MAE **11.71363**. Ini lebih rendah dari baseline 12.23275 yang diberikan pengguna sebesar **0.51912 (4.2437%)**, dan memenuhi stretch goal <12.0. Dibanding initial R1-D 13.53473, penurunan MAE 1.82110.

Kandidat yang sama mempunyai internal primary MAE **11.931600**, membaik pada **6/6 primary windows** dan kedua H12 dibanding R1-D. Tetapi **best pooled H12 tetap hybrid Ridge+text: 13.938115**, bukan LightGBM+text 14.704041. Karena H12_E kandidat terbaik primary masih 17.473257 versus H12_L 12.361143, public score tidak menghapus risiko era/history/horizon.

Simpan kandidat sebagai **strong provisional**, layak masuk final-model review. Tidak ada model final ditetapkan, tidak ada Classification optimization, CatBoost/second family, transformer, stacking, atau ensemble. Empat dari enam additional Regression submissions dipakai; dua slot sengaja disimpan. Setelah breakthrough, tidak ada eksperimen atau upload baru untuk mengejar skor kecil.

Best-model manifest lengkap ada di `submissions/lgb_text_inner_v1.json`; `best_candidate.json` mengikat manifest/CSV hash, Kaggle ref, internal metrics dan batas interpretasi. Log lengkap ada di `KAGGLE_LOG.md` dan ledger hash-chained `kaggle_log.json`. Initial submission manifests tetap utuh.

## Perbandingan internal utama

Raw controls ditampilkan sebagai raw. Semua recipe baru di bawah menggunakan kebijakan mapping yang dipilih hanya dari inner temporal labels. Fixed rank-policy views lengkap tetap ada di COMPARISON_TABLES.md.

| Recipe | Policy | Primary MAE | Equal-cycle MAE | H12 MAE | H12_E | H12_L |
| --- | --- | --- | --- | --- | --- | --- |
| R1_C | raw | 15.148986 | 15.254088 | 15.477893 | 16.432126 | 14.670564 |
| R1_D | raw | 13.349836 | 13.589657 | 15.577240 | 17.795604 | 13.700391 |
| B_D_RECENT8 | raw | 13.217154 | 13.507105 | 15.794561 | 17.887206 | 14.024077 |
| A_NO_CODES | raw | 14.078601 | 14.190019 | 15.053467 | 16.132087 | 14.140901 |
| text_ridge | inner | 18.274839 | 18.465759 | 19.474794 | 21.440805 | 17.811449 |
| hybrid_ridge | inner | 12.551305 | 12.790073 | 13.938115 | 16.395455 | 11.859081 |
| svd_hgb | inner | 12.366181 | 12.677743 | 14.714120 | 17.272561 | 12.549550 |
| lgb_inner | inner | 13.064889 | 13.322505 | 15.604153 | 17.886588 | 13.673097 |
| svd128 | inner | 12.299353 | 12.618070 | 14.648435 | 17.188415 | 12.499482 |
| svd_noscope32 | inner | 12.352186 | 12.673786 | 14.739826 | 17.337365 | 12.542176 |
| structured_ridge | inner | 14.178654 | 14.332997 | 15.205818 | 17.229242 | 13.493898 |
| lgb_text | inner | 11.931600 | 12.283548 | 14.704041 | 17.473257 | 12.361143 |

## Apa yang membantu

**Text adalah sumber gain terbesar.** Text-only Ridge lemah (inner primary 18.274839/H12 19.474794), sehingga text tidak menggantikan structured predictors. Namun same-family structured-only Ridge memberi14.178654/15.205818 versus hybrid 12.551305/13.938115. Kontrol mempertahankan scaler/OHE/imputer/alpha 10 dan mengganti semua stream dengan constant empty string; selisih mencakup content plus availability, bukan token coefficients saja. Karena pembandingnya sekeluarga, gain hybrid tidak dapat dijelaskan hanya oleh pergantian family dari HGB ke Ridge.

Pada exact impact **85.9730** (n=2,819 primary), R1-D MAE 16.763560 turun menjadi hybrid 13.839382 dan LightGBM+text 14.008717. Pada impact154.0178 (n=6,000),9.348884→8.320662/8.154837. H12 pada85.9730 juga membaik18.430467→14.876795/16.129421. Seluruh 79 exact-impact groups validation disimpan dalam metrics; contoh ini bukan causal adjustment atau pembuktian text mechanism, tetapi menunjukkan predictive information di dalam level impact yang sama.

**Model×information lebih penting daripada estimator saja.** Structured LightGBM inner primary 13.064889/H12 15.604153 hanya memberi gain kecil atas R1-D dan tidak mengatasi H12. Menggunakan representasi SVD32 yang sama dengan HGB, LightGBM menjadi11.931600/14.704041. Parameternya hanya dua konfigurasi yang sudah ditentukan pada eksperimen LightGBM sebelumnya; tidak ada model zoo atau pencarian ratusan kombinasi. HGB+SVD32 sendiri sudah memberi12.366181/14.714120, sehingga text dan estimator memberi kontribusi yang berbeda.

**Rank blend membantu sebagian representasi, bukan semua.** Pada kontrol structured, full prediction-only rank justru merugikan. R1-D raw 13.349836 menjadi13.842403 primary dan15.577240→16.107358 H12. Blend fixed25% memberi gain kecil13.297560/15.548290. Pada hybrid Ridge, raw 13.242685/14.706501 menjadi inner12.551305/13.938115; mapping membantu score spacing yang lebih terkompresi. Pada LightGBM+text, raw 12.047761/14.786377 menjadi11.931600/14.704041: gain mapping lebih kecil daripada penambahan informasi text.

Mapping tidak menggunakan target validation/Test: di setiap supplied cycle,100×average_rank(prediction)/N dengan average ties, kemudian convex blend dan clip[0,100]. Ia tidak memperbaiki inversi urutan model. Aturan mapping dan grid0/.25/.5/.75/1.0 dibekukan; alpha setiap outer cutoff dipilih dari inner past-only data, bukan MAE outer. Tidak ada isotonic/target reranking/percentile label synthesis yang dijalankan.

## Apa yang tidak berhasil atau belum konsisten

- **Pure percentile mapping** kalah pada semua empat structured controls secara pooled primary dan H12. Marginal rank geometry target tidak cukup untuk menjamin percentile prediction optimal di bawah MAE.
- **Text-only Ridge** tidak cukup; structured/text information saling melengkapi. Tidak ada alasan mempromosikan text-only atau menambah tuning baseline itu setelah gap besar.
- **SVD128** memperbaiki internal primary 12.366181→12.299353 dan H12 14.714120→14.648435, tetapi public 12.38411 lebih buruk daripada SVD32public12.34519. Ini counterexample transfer internal→public, bukan alasan menyesuaikan jumlah komponen lagi dari leaderboard.
- **No-scope SVD32** memberi primary 12.352186 tetapi H12 14.739826, hanya perubahan kecil dan berbeda arah dari full32. Ia tidak dipromosikan semata-mata karena Test memiliki novel zero cohort.
- **Structured LightGBM** bukan breakthrough. Hal ini tidak berarti family tersebut tidak berguna: interaksinya dengan text memberi hasil yang lebih kuat.

Stage1–4 dipilih dan dibekukan sebelum public feedback terkait. Stage5 disusun setelah dua public score pertama tersedia dan ditandai adaptif; pilihan konfigurasinya berasal dari contrast internal model×text, bukan fitting ke skor public. Empat candidate approvals merekam outcome yang sudah diketahui saat keputusan dibuat. Tidak ada klaim unbiased selection setelah banyak development rounds.

## Recipe kandidat terbaik

Predictors:12 structured fields eligible dari R1-D ditambah signal_stream. Numeric/mixed parsing, approximate/suffix/parsefail/missing flags, median imputers, rare-category grouping dan native categorical positions belajar pada fit partition saja. TF-IDF word 1–2, min_df=2, max_features=20,000, sublinearTF, float64; randomized TruncatedSVD32, n_iter=7, seed 42; satu missing-stream flag. Tidak ada ID, target task lain, audit/checksum/pulse fields atau TP1 label join.

LightGBM 4.6.0 `regression_l1`,400 trees,learning_rate0.05, seed 42,n_jobs 2, deterministic CPU/force_col_wise. Inner temporal search terbatas pada base 31 leaves/minleaf 40/L2=1 dan regularized15/80/3. Negative unknown/missing categorical sentinel −1 diperlakukan LightGBM sebagai missing; explicit flags tetap ada. Library warning konversi kategori negatif ke NaN adalah perilaku yang diharapkan.

Untuk full-Train submission, inner47–75→76–79 memilih **base** dan **rank weight 0.50**. Lalu seluruh 50,608 Train rows mempelajari ulang preprocessing/model, dan 29,421 Test rows diprediksi bersama sesuai provided cycle. Final output `clip(0.5×raw +0.5×cycle_percentile,0,100)`. Ini pilihan dari Train inner data, bukan skor Kaggle. Internal outer-fold score11.931600 menilai kebijakan inner selection, yang memilih alpha berbeda pada beberapa cutoff; jangan menyebutnya skor kebijakan fixed0.50 pada semua outer folds.

## Temporal, subgroup dan tail risks

LightGBM+text dibanding R1-D membaik pada setiap primary window: deltas W1–W6 **−0.477634/−0.901714/−1.639615/−1.062587/−1.913570/−1.856439**. H12_E/L membaik **−0.322347/−1.339248**, tetapi early H12 tetap **17.473257**, lebih buruk daripada hybrid Ridge 16.3955. H12 pooled14.704041 tidak sama dengan mean tak berbobot kedua window. H12 memakai ulang seluruh 38,588 primary records dan bukan independent replication.

Primary EF0/EF1 MAE kandidat **12.979047/9.755852**, versus R1-D14.573469/10.808115. Low/high-tail MAE **12.235391/9.495441**, versus18.221372/11.514815. Bias conditional tetap berlawanan: low-tail mean error **+12.022070**, high-tail **−9.079559**, meskipun global−0.418158. Rank mapping memperbaiki tail compression namun tidak menghilangkan prediction uncertainty atau membuktikan quantile calibration.

Primary stream-present/missing MAE **11.852119/13.041813** (n36,010/2,578). Missing rows masih sulit dan biasnya+2.412151. Inner mapping sedikit memperburuk missing MAE dibanding raw 12.972384, walaupun pooled/present membaik. Di H12, missing MAE 14.516775 lebih rendah daripada present14.717447; subgroup ordering tidak universal. Tidak ada tambahan routing/ensemble untuk mengejar gain kecil pada fase ini.

Semua validation cycles telah dipakai EDA dan beberapa rounds; banyak perbandingan adaptif tetap menjadi risiko. Inner selection mencegah outer-label fitting, tetapi tidak mengembalikan untouched holdout. Private score tidak tersedia. Proporsi sekitar30%public/70%private berasal dari instruksi pengguna, bukan verifikasi aturan kompetisi yang terpisah.

## Novel zero-scope Test cohort

Sebanyak **6,278/29,421 (21.34%)** Test rows mempunyai count nol pada sedikitnya satu scope/spread field; Train tidak memiliki count nol. Semua 6,278 juga mempunyai pressure di bawah minimum Train; 4,835 memiliki ketiga count nol. Parser menerima nol sebagai angka, bukan parse failure. Makna genuine zero vs missing representation vs source-process change **UNKNOWN**. Tidak ada target Test atau labeled Train analogue untuk menilai error cohort ini.

Feature-only Train correlations menunjukkan scope_pressure dekat dengan scope_trace (Spearman sekitar 0.917), sedangkan asosiasinya dengan impact_load hanya sekitar 0.025. Itu hubungan fitur deskriptif, bukan causal importance. Scope group pada R2 memberi kontribusi incremental kecil; no-scope text ablation juga tidak konsisten membantu.

Unscored stress mem-fit dua model pada47–67, lalu mengubah hanya copy input68–79: tiga count menjadi0 dan pressure di bawah minimum fit. Full HGB+SVD32 mean |prediction delta| **0.052585**, p99 **0.984699**, maksimum **3.217934**; sekitar 0.995% berubah lebih dari1poin. No-scope predictions tepat identik, seperti yang diharapkan dari feature exclusion. Baseline refits keduanya cocok persis dengan saved vectors. **Tidak ada MAE perturbed**, karena label record asli belum tentu sesuai input counterfactual. Probe ini tidak membuktikan akurasi pada actual Test cohort dan tidak diuji langsung pada model LightGBM terbaik.

PDF source audit historis mencatat57pages/hash386fa6..., sedangkan sibling PDF lokal yang tersedia44pages/hash349860...; kedua identitas dicatat di shift diagnostics dan tidak dianggap file yang sama. Tidak ada dokumentasi yang memverifikasi zero semantics atau target generator. Tidak ada zero-to-median imputation, pseudo-label, Test-specific correction atau transductive choice berdasarkan fabricated target.

## Hubungan dengan EDA Samuel dan TP1

Pertanyaan formal tetap: **“Apakah hubungan antara impact_load dan threat_rank mengalami perubahan antarperiode cycle_code, meskipun distribusi threat_rank relatif stabil?”** Bukti modeling menambah appendix: hubungan target kondisional dan staleness/history risk penting, sementara text menambah informasi di dalam exact-impact groups. Ia tidak mengganti kesimpulan statistik EDA atau mengidentifikasi structural break.

Pelajaran REG-Lite B yang bertahan adalah median-oriented MAE, conditional error dan tail/regime checks. Target TP2 jauh lebih padat daripada 80 nilai dukungan TP1, sehingga multiclass satu kelas per nilai TP2 tidak disalin. Keberhasilan text pada Classification dipakai sebagai hipotesis, kemudian diuji terhadap target Regression secara tersendiri; tidak ada Classification pipeline yang diubah atau dilatih ulang.

## Validation, reproducibility dan batas klaim

Independent final verifier PASS untuk Stage1,23,4,5: **384 outer score jobs**,**96 aggregate cells**, past-only fit fingerprints, inner argmins/mapping, per-cycle/EF/tail/stream MAE/bias/RMSE/Spearman dan position/hash checks. Stage1 juga memverifikasi 144 incomplete-batch stress records. Detail exact-impact metrics disimpan; independent verifier menguji support counts, bukan mengulang setiap distribution metric impact. Batas ini dinyatakan dalam receipts.

Terdapat **90 cached fit/predict records** (48 Stage23 + 27 Stage4 + 15 Stage5). Sebagian history sama dievaluasi untuk inner versus outer horizon; ini bukan 90 independent trials. Tidak ada full redundant campaign repeat. Dua additional refits diperlukan untuk unscored scope stress dan cocok dengan baseline caches. Input/metadata/model-state hashes serta source/environment identity direkam. LightGBM 4.6.0 tersedia sebelumnya dan tidak ada upgrade dependency inti.

Final scoped suite **65 tests PASS dalam 5.415 detik tanpa failures/errors/skips**, termasuk dua tes report restoration pada reproduction success/failure. Notebook08 berhasil Restart & Run All dengan 8 code cells berurutan, 0 errors dan 7 embedded figures. MAIN memeriksa ketujuh PNG pada ukuran baca. Full-page HTML preview diekspor tetapi belum diperiksa melalui GUI browser; QA visual berasal dari PNG dan output/struktur notebook. Jupyter memakai Windows selector fallback dan nbconvert dapat memberi warning alt-text, tanpa execution error. Receipt completion merekam angka akhir.

Frozen source/checkpoints lama, formal EDA, Rounds1–2, initial manifests dan pekerjaan anggota lain dipertahankan. CSV/prediction vectors/raw data/model binaries/credentials tidak masuk Git. Perubahan README hanya front door. Package reproduksi menjaga historical report bytes saat mengukur runtime baru agar future-stage checkpoint pins tidak rusak; hasil reproduction tersimpan ignored, bukan menggantikan original campaign evidence.

## Next step dan stop point

**Layak masuk final-model review**, dengan LightGBM+SVD32 sebagai strong provisional candidate dan hybrid Ridge sebagai pembanding H12. Tidak perlu menambah family/model atau menghabiskan dua submission slots yang tersisa. Jika review mengotorisasi satu studi berikutnya, prioritasnya **horizon/history robustness untuk text model yang sudah kuat**, dengan fixed H12/EF/missing/tail guardrails; bukan retuning jumlah komponen atau mapping dari public scores. Batch availability untuk inference dan arti novel scope support masih perlu source clarification.

Tidak ada model final permanen, additional Classification work, complex ensemble atau upload berikutnya setelah fase ini. Target baseline dan stretch tercapai; private generalization tetap terbuka.
