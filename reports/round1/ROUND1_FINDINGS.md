# Samuel Regression Round 1 — hasil dan titik review

30 Oktober 2026 · **Round 1 selesai; belum ada final model, Round 2, classification, ensemble atau submission.**

## Ringkasan keputusan

**R1-D adalah kandidat sementara terbaik pada primary validation**, tetapi belum menjadi pemenang yang stabil di seluruh horizon. HGB absolute-error dengan expanded structured inputs memberi primary MAE 13,3498, dibanding 15,1490 untuk dua fitur. Pada H12_E hasilnya memburuk 17,7956 vs 16,4321; pooled secondary justru sedikit lebih baik pada R1-C. Temuan ini menjadi alasan menguji robustness/history pada Round 2 setelah review, bukan otomatis memilih model final.

Informasi resmi yang dikonfirmasi pengguna: threat_rank adalah indeks prioritas respons 0–100, MAE lower-is-better, dan impact_load/entry_friction boleh menjadi predictor. Detail otoritas dan residual uncertainty di `docs/COMPETITION_INFO.md` serta `ROUND1_FEATURE_ELIGIBILITY.md`.

## Protokol dan validasi

Protokol delapan recipe dibekukan sebelum fit eksperimen pertama: enam expanding windows 4 cycle dan dua horizon 12 cycle sesuai instruksi. Threshold lookup 30, seed 42, HGB 200 iterations / 15 leaves / learning_rate = 0,05 / minleaf 30 / L2=1, early_stopping=False, thread limit 2. Tidak ada hyperparameter search. Raw unclipped MAE adalah primary; clipping terpisah.

Setiap family primary dan secondary mencakup 38.588 unique validation rows dari 24 cycle 56–79. Primary valblocks disjoint; H12 memakai ulang record yang sama dalam view history/horizon lain dan bukan bukti independen. Seluruh train sudah dipakai EDA, jadi development validation, bukan untouched holdout. Label resmi tidak diranking ulang.

## Semua recipe — primary aggregate

| Recipe | Metode | Pooled MAE | Equal-cycle MAE | RMSE | Mean signed error |
|---|---|---:|---:|---:|---:|
|R0_A|Global median|25.0000|25.0000|28.8671|-0.0069|
|R0_B|Impact median|15.5616|15.7032|19.5610|+0.0468|
|R0_C|Impact×EF median + hierarchy|15.3114|15.4389|19.2570|-0.1382|
|R1_A|Standardized OLS|15.6226|15.7343|19.1821|+0.6254|
|R1_B|HGB squared error|15.5710|15.6962|19.0908|+0.5482|
|R1_B_ADD|HGB squared, no interactions|15.6015|15.7260|19.1328|+0.5266|
|R1_C|HGB absolute error, 2 features|15.1490|15.2541|19.0575|-0.2857|
|R1_D|HGB absolute error, expanded|13.3498|13.5897|17.3576|-0.0371|

## Hasil setiap window utama

|Recipe|W1|W2|W3|W4|W5|W6|
|---|---:|---:|---:|---:|---:|---:|
|R0_A|25.0000|24.9999|25.0000|25.0000|24.9999|25.0000|
|R0_B|16.9696|15.7896|16.7414|14.4496|14.6167|15.2001|
|R0_C|16.7349|15.3696|16.5437|14.2514|14.3991|14.9697|
|R1_A|16.6113|15.7118|16.7795|14.7717|14.8428|15.2926|
|R1_B|16.7900|15.7824|16.7189|14.6648|14.6621|15.1889|
|R1_B_ADD|16.7823|15.8104|16.7254|14.6920|14.6937|15.2607|
|R1_C|16.3860|15.1531|16.3275|14.1140|14.2760|14.9347|
|R1_D|16.8834|12.9446|13.9938|13.0620|12.6885|12.1599|

W1 fit 47–55→val 56–59;W2 fit 47–59→60–63;W3 fit 47–63→64–67;W4 fit 47–67→68–71;W5 fit 47–71→72–75;W6 fit 47–75→76–79. Fit/validation cycle utuh.

## Apa yang dipelajari dari baseline dan objective

- R0-A sekitar 25 MAE di semua window, konsisten dengan target marginal hampir uniform dan median fit sekitar 50. Stabilitas baseline ini bukan bukti conditional relationship stabil.
- R0-B mengurangi pooled MAE menjadi 15,5616; R0-C menjadi 15,3114 dan mengungguli R0-B pada 6/6 window serta 24/24 cycle. Impact dan EF memberi informasi nyata pada development validation; hierarchical fallback menangani support langka tanpa memakai label validation.
- OLS15,6226 lebih buruk daripada lookup R0-C meskipun RMSE sedikit lebih baik. Model nonlinear squared-error15,5710 hanya sedikit lebih baik daripada OLS dan masih lebih buruk dari R0-C. Jadi kompleksitas saja tidak cukup.
- R1-C absolute error mengungguli R1-B sebesar 0,4220MAE pada 6/6 window dan 24/24 cycle. R1-C juga mengungguli R0-C pada 6/6 window/24 cycle, tetapi selisih pooled hanya 0,1624. Loss yang sesuai MAE merupakan temuan lebih konsisten daripada tambahan kompleksitas.
- R1-B vs additive-no-interactions membaik hanya 0,0305MAE, dengan 5/6 window dan 15/24 cycle membaik. Ini evidence kecil untuk interaksi pada **konfigurasi squared-error ini**, bukan pernyataan bahwa interaksi tidak berguna pada semua model.

## Feature expansion: keuntungan nyata dengan horizon risk

R1-D dibanding R1-C menurunkan primary MAE 1,7991 poin (11,88%), membaik5/6window dan 21/24 cycle. Kedua EF mendapat manfaat secara pooled: EF0 MAE 16,8175→14,5735 (n = 26.048); EF1 11,6832→10,8081 (n = 12.540). Keuntungan lebih besar pada EF0, tetapi tidak eksklusif pada EF0. W1 memburuk +0,4975MAE.

|Model|H12_E fit 47–55→val 56–67|H12_L fit 47–67→val 68–79|Pooled kedua H12|
|---|---:|---:|---:|
|R0_A|25.0000|24.9999|25.0000|
|R0_B|17.0462|15.0050|15.9405|
|R0_C|16.7388|14.7735|15.6742|
|R1_A|16.6445|15.1500|15.8350|
|R1_B|16.9045|15.0481|15.8989|
|R1_B_ADD|16.9021|15.0857|15.9182|
|R1_C|16.4321|14.6706|15.4779|
|R1_D|17.7956|13.7004|15.5772|

Feature expansion memperbaiki H12_L tetapi merusak H12_E +1,3635MAE. Pooled secondary R1-D 15,5772 vs R1-C 15,4779. Karena early dan late windows memakai training era serta sample size berbeda, ini tidak menetapkan fitur mana penyebabnya atau menjamin pola pada test 80–91.

### Investigasi tambahan tanpa recipe baru: manfaat history yang diperbarui

Prediksi yang sudah ada dibandingkan pada **record validation yang sama**, dengan parameter model tetap namun training history lebih baru. Pada cycle 60–63, R1-D fit 47–55 MAE 17,1342, sedangkan fit 47–59 menjadi 12,9446. Pada 64–67,18,9325→13,9938. R1-C pada kelompok yang sama berubah15,7341→15,1531 dan17,0993→16,3275. Pada 76–79, R1-D fit 47–67 menjadi14,1639 vsfit 47–75 sebesar12,1599.

Ini mendukung hipotesis bahwa expanded model lebih sensitif terhadap history/staleness dan mendapat manfaat lebih besar dari data masa lalu terbaru. Training size dan era berubah bersamaan, jadi belum membedakan efek recency, cycle feature, category novelty atau sample size. Tidak ada model tambahan yang dilatih untuk diagnostic ini; semua berasal dari 64 frozen jobs.

Novelty kategori bukan penjelasan tunggal W1: pada subset yang semua kodenya known dan nonmissing (n = 1.323), expansion juga memburuk 16,9719→18,0991. Kelompok unknown/missing bersifat overlap dan bukan causal adjustment. Perlu controlled ablation pada Round 2 untuk menentukan kolom yang membantu/merusak; jangan mengatribusikan gain set-level ke satu fitur.

## Bias, tails, dan prediction distribution

Bias global R1-D hanya−0,0371, tetapi ini menutupi error conditional yang berbeda arah. Pada low-tail actual ≤ 5 (n = 1.928), MAE 18,2214 dan bias +18,1913; pada high-tail ≥ 95 (n = 1.955), MAE 11,5148 dan bias −11,2520. Dibanding R1-C, kedua tail MAE membaik (20,7920→18,2214 dan 16,7240→11,5148), tetapi overprediction target rendah serta underprediction target tinggi tetap nyata pada validation.

R1-C prediksi primary hanya 1,1210–87,9910, jauh lebih sempit daripada label aktual hampir 0–100. R1-D memperluas rentang menjadi −3,3077–103,5449, P1/P99 sekitar 7,94/96,95. Expansion mengurangi kompresi ke tengah tetapi tidak menghilangkan tail bias. Ini pola prediksi development, bukan bukti mekanisme kausal atau ekspektasi error test.

R1-D menghasilkan 10 prediksi<0 dan 62 > 100 dari 38.588. Clipping terpisah mengubah MAE 13,349836→13,347935 (gain 0,001901). Semua angka perbandingan utama tetap raw. Tidak ada clipping yang diam-diam dicampurkan ke kandidat utama.

Error impact group turut bervariasi. Di antara grup primary R1-D dengan n ≥ 100, impact 98,0411(n = 180) MAE 18,0227 dan 85,9730(n = 2.819) MAE 16,7636 adalah contoh sulit; full table tersimpan untuk menghindari cherry-picking. Threshold kelompok ini hanya diagnostic setelah fitting.

## TP1 → TP2

Pelajaran conditional median dan evaluasi tails dari REG-Lite B tetap relevan: loss absolute memberi keuntungan stabil, bias global kecil tidak meniadakan bias kelompok. Phase16 B2 memotivasi pemeriksaan regime, tetapi tidak ada gate/label/model TP1 yang disalin. TP1 raw target support 80 nilai dan regresi TP2 lebih padat; tidak ada multiclass satu kelas per 34.333 nilai TP2. Low/high tercile TP1 tidak disamakan dengan tails ≤ 5 / ≥ 95.

## Round 2: rekomendasi berdasarkan information gain, belum dijalankan

1. **Prioritas pertama: attribution feature expansion dan staleness.** Controlled ablation cycle-only contribution, tanpa code kategori, dan kelompok numeric/scope yang jelas dengan family/loss/folds sama. Hipotesis: sebagian manfaat expanded features sensitif pada training era dan tidak mentransfer jauh. Ini lebih informatif daripada menambah algoritma/ensemble baru.
2. **History sensitivity.** Bandingkan expanding vs recent windows yang dibekukan sebelumnya, termasuk horizon 12 cycle. Gunakan paired records dan laporkan trade-off EF/tails; jangan memilih recency dari satu score atau menganggap cycle feature bisa mengekstrapolasi tanggal baru.
3. **Conditional spread dan rank handling.** Pertimbangkan quantile outputs serta raw-score versus within-cycle ranking dari **prediksi** saja, jika batch-cycle lengkap memang tersedia dan pengguna menyetujui. Ranking dapat memperbaiki kompresi atau memperburuk MAE ketika urutan lemah; target-generation formula tetap belum diketahui.
4. **Text ablation setelah review.** Signal_stream dapat membawa informasi di dalam kelompok impact×EF, tetapi memerlukan train-only vocabulary dan baseline yang comparable. Belum dijalankan. Classification dapat menjadi milestone terpisah jika prioritas bergeser ke coverage dua task.

Tidak menyarankan model final dari putaran ini. R1-D adalah leader primary sementara; R1-C adalah pembanding lebih sederhana yang sedikit lebih baik pada secondary pooled. Pilihan apakah memperdalam regression atau mulai classification tetap di tangan pengguna.

## QA dan reproducibility

64 fit pertama selesai, lalu 64 fit ulang penuh dengan recipe, seed, input dan kode yang sama. Semua 64 prediction arrays bit-identical; seluruh fitted audit-state hashes identik; selisih MAE/RMSE/bias nol. Recompute independen memakai math.fsum dari validation arrays berbeda maksimum 3,55e−15 terhadap metrics. Unknown category counts cocok pada seluruh 8 window.

Protokol/freeze di `experiments/round1/`; source/run commands di `docs/ROUND1_REPRODUCIBILITY.md`; hasil aggregate lengkap di `results.json`; per-cycle/subgroup/table di `COMPARISON_TABLES.md`; errorhistory/novelty di `behavior_diagnostics.json`; reproduction receipt di `reproduction.json`. Local ignored arrays dipertahankan untuk audit, bukan submission/model final.

Notebook demo `notebooks/samuel/04_samuel_regression_round1.ipynb`. Tidak ada test prediction, Kaggle submission, classification model, extensive tuning atau ensemble. Raw/TP1/pekerjaan anggota lain dipertahankan.
