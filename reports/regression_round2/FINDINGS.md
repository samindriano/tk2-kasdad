# Samuel Regression Round 2 — feature, history and temporal review

30 Oktober 2026 · **19 recipe selesai dievaluasi; kandidat masih sementara.** Tidak ada Regression Round 3 atau pemilihan model final. Initial Kaggle submission berjalan melalui track terpisah memakai R1-D asli; skor publik tidak digunakan untuk memilih recipe, fitur atau history Round 2.

## Keputusan sementara

**Primary leader: full expanded R1-D dengan recent-8 history**, MAE **13,217154**, versus expanding R1-D **13,349836** dan R1-C **15,148986**. Keuntungan atas R1-D hanya 0,132682, dengan 4/6 windows dan 13/24 cycles membaik. Equal-cycle MAE juga membaik, 13,589657→13,507105. Ini kandidat primary, bukan pemenang universal.

**Pooled H12 leader: expanding tanpa kode kategorikal**, MAE **15,053467**, dibanding R1-D **15,577240** dan R1-C **15,477893**. Namun gain terkonsentrasi pada H12_E: **17,795604→16,132087**; H12_L justru memburuk **13,700391→14,140901**. High-tail H12 juga memburuk **13,146680→14,661021**. Tidak ada recipe yang menyelesaikan seluruh trade-off horizon dan tails.

R1-D expanding masih layak sebagai **kontrol dan kandidat utama yang kuat**: primary-nya dekat dengan leader, H12 lebih baik daripada recent-8, dan high-tail lebih baik daripada no-codes. R1-C tetap pembanding sederhana, bukan diganti atau dihapus. Belum ada dasar menetapkan fitur/history final untuk cycle 80–91.

## Design, provenance dan batas evidence

Sama dengan Round 1: enam 4-cycle primary windows dan dua H12 views. Masing-masing family mencakup **38.588 unique rows, cycle 56–79**. H12 memakai kembali seluruh records primary; tidak menjadi independent replication. Round 2 disusun dari outcome Round 1, sehingga freeze tidak menghapus adaptive selection bias. Seluruh hasil adalah development validation, bukan estimasi unbiased skor official test.

Absolute-error HGB tetap: 200 iterations, 15 leaves, learning rate 0,05, minimum leaf 30, L2 1, max bins 255, seed 42, early stopping=False, thread limit 2. Exact R1-C/R1-D controls menggunakan wrapper lama. Numeric imputer, category vocabulary/rare grouping dan seluruh preprocessing belajar hanya dari selected fit rows. Native categorical handling tidak memperlakukan kode sebagai besaran numerik. Denylist Round 1 tidak berubah; tambahan fitur masih supplied-schema eligible dengan operational timing provisional.

15 recipe dasar dibekukan sebelum fit pertama. Empat interaction recipe dipilih dari budget yang telah ditetapkan, lalu dibekukan terpisah sebelum fitting: no_codes/no_cycle × recent8/matched8_seed42. Alasan numerik, counterexample dan sifat adaptifnya ada di `experiments/regression_round2/interaction_selection.json`. Catatan koreksi presisi terpisah memperbaiki beberapa angka prose yang ditulis terlalu presisi dari tampilan pembulatan (selisih <0,000006 MAE); receipt/recipe asli dipertahankan dan machine results menjadi otoritas.

## Seluruh kandidat

Raw/unclipped MAE, lower is better. Recent/matched memakai cutoff validation yang sama. Nilai H12 adalah row-pooled, bukan mean tak berbobot dari dua window.

| Recipe | Primary MAE | Equal-cycle MAE | H12 MAE | H12_E | H12_L |
| --- | --- | --- | --- | --- | --- |
| R1_C | 15.148986 | 15.254088 | 15.477893 | 16.432126 | 14.670564 |
| R1_D | 13.349836 | 13.589657 | 15.577240 | 17.795604 | 13.700391 |
| A_NO_CYCLE | 14.214105 | 14.366927 | 15.582091 | 17.495574 | 13.963187 |
| A_NO_CODES | 14.078601 | 14.190019 | 15.053467 | 16.132087 | 14.140901 |
| A_NO_SCOPE_SPREAD | 13.391396 | 13.634178 | 15.616123 | 17.919023 | 13.667752 |
| A_NO_TELEMETRY | 13.452036 | 13.681743 | 15.688186 | 17.825150 | 13.880205 |
| A_NO_TEXT_LOAD | 13.420151 | 13.671793 | 15.586731 | 17.865197 | 13.659033 |
| B_C_RECENT8 | 15.097192 | 15.228874 | 15.629188 | 16.435281 | 14.947192 |
| B_C_RECENT12 | 15.136217 | 15.249599 | 15.562554 | 16.432126 | 14.826852 |
| B_C_MATCHED8_S42 | 15.169310 | 15.276058 | 15.532503 | 16.436255 | 14.767882 |
| B_C_MATCHED8_S31415 | 15.174464 | 15.285239 | 15.534291 | 16.489867 | 14.725824 |
| B_D_RECENT8 | 13.217154 | 13.507105 | 15.794561 | 17.887206 | 14.024077 |
| B_D_RECENT12 | 13.249852 | 13.513589 | 15.700846 | 17.795604 | 13.928574 |
| B_D_MATCHED8_S42 | 13.501785 | 13.743191 | 15.717649 | 17.912219 | 13.860931 |
| B_D_MATCHED8_S31415 | 13.528526 | 13.749043 | 15.559120 | 17.842859 | 13.626961 |
| C_NO_CODES_RECENT8 | 14.023118 | 14.176941 | 15.275992 | 16.350551 | 14.366861 |
| C_NO_CODES_MATCHED8_S42 | 14.217541 | 14.310151 | 15.164060 | 16.209452 | 14.279604 |
| C_NO_CYCLE_RECENT8 | 13.483128 | 13.740731 | 15.579529 | 17.420885 | 14.021648 |
| C_NO_CYCLE_MATCHED8_S42 | 14.285905 | 14.453660 | 15.700965 | 17.597293 | 14.096576 |

## R2-A — kontribusi kelompok fitur

| Removal dari full expanding | Delta primary MAE | Primary windows membaik | Delta H12 MAE | Interpretasi |
|---|---:|---:|---:|---|
| cycle_code | +0,864268 | 1/6 | +0,004851 | Kontribusi primary terbesar; tidak memberi pooled H12 gain yang sama |
| org/module/pattern codes | +0,728764 | 1/6 | −0,523773 | Membantu primary, tetapi menimbulkan trade-off awal/akhir H12 |
| scope/spread | +0,041560 | 3/6 | +0,038883 | Gain incremental kecil pada konfigurasi ini |
| telemetry | +0,102199 | 2/6 | +0,110946 | Gain kecil; removal membantu low-tail tetapi merugikan overall/high-tail |
| numeric text_load | +0,070314 | 0/6 | +0,009491 | Gain primary kecil namun konsisten pada enam windows |

Scope/spread terdiri dari scope_pressure, scope_trace, org_spread dan module_spread; numeric text_load diuji terpisah dari text signal_stream yang tetap excluded. Menghapus source column juga menghapus seluruh parse/missing/approx/suffix flags terkait. Ini predictive contribution conditional pada model, loss, preprocessing dan refit yang sama; bukan causal importance atau bukti sebuah kelompok tidak berguna universal.

Kategori/cycle memang memberi nilai pada primary, tetapi manfaatnya bergantung era/horizon. No-codes tetap mengalahkan two-feature R1-C pada seluruh enam primary windows, walaupun kalah terhadap R1-D pada lima. Dengan demikian ekspansi non-code masih membawa informasi; kesimpulannya bukan bahwa seluruh manfaat expansion berasal dari kode.

## R2-B — recency versus jumlah observasi

Recent-8 mengambil delapan observed cycles terakhir secara utuh; recent-12 mengambil dua belas. Matched-8 memakai jumlah baris tepat sama dengan recent-8, disampel uniform tanpa replacement dari seluruh expanding past, seed 42/31415 dan cutoff. Sampling tidak melihat target. Matched sample dapat memotong cycle; komposisi kategori, support dan dependence tetap berbeda, jadi ini bukan efek kausal recency.

Contoh cutoff 67: expanding **29.705** rows (47–67), recent-8 **13.571** rows (60–67), recent-12 **17.685** rows (56–67), kedua matched sample **13.571** rows dari 47–67. Pada cutoff 75: expanding 42.117 versus recent/matched-8 12.412. Early recent-12 identik dengan expanding karena hanya sembilan cycle tersedia; runner mengalias fit itu dan tidak menghitungnya sebagai eksperimen independen.

Full D recent-8 primary **13,217154** lebih baik daripada kedua matched samples **13,501785/13,528526**, dengan ukuran sama. Ini mendukung nilai period selection pada primary melampaui pengurangan jumlah data saja. Tetapi recent-8 H12 **15,794561** lebih buruk dari expanding **15,577240** dan kedua matched H12 **15,717649/15,559120**. Recent-12 juga tidak meningkatkan H12; early H12 sama dengan expanding, late memburuk.

R1-C recent-8 primary hanya membaik **0,051794** dan H12 memburuk **0,151295**; D lebih sensitif pada pemilihan periode. Perbedaan dua matched seeds untuk D kecil pada primary (**0,026741**) tetapi lebih besar pada H12 (**0,158529**). Dua seeds bukan confidence interval. Evidence tidak mendukung slogan data baru selalu lebih baik atau data lebih banyak selalu lebih baik: recency membantu sedikit pada short horizon, expanding history lebih tahan pada beberapa long-horizon views.

## R2-C — interaksi fitur × history

Efek penghapusan kategori hampir sama di seluruh history: delta primary +0,728764 expanding, +0,805964 recent-8 dan +0,715757 matched-8; delta H12 −0,523773/−0,518569/−0,553589. Keuntungan early-H12 dan kerugian late/high-tail berulang. Dugaan bahwa recent history akan menghilangkan trade-off kategori **tidak didukung** oleh empat pemeriksaan terarah ini.

Nilai incremental cycle_code pada primary lebih tergantung history: removal penalty **+0,864268** expanding menjadi **+0,265974** recent-8, tetapi tetap **+0,784120** matched-8. Difference-in-differences recent adalah **−0,598294**, matched hanya −0,080149. Recent-history composition memoderasi kontribusi cycle lebih besar daripada sekadar jumlah baris yang diperkecil; mekanisme kausal belum diketahui.

No-cycle recent-8 primary **13,483128** masih kalah dari full recent-8 **13,217154**. H12-nya **15,579529** memperbaiki full recent-8 sebesar 0,215032 tetapi hampir sama dengan full expanding **15,577240**. Ini alternatif diagnostik yang menarik untuk tails/EF, bukan model yang mendominasi kontrol. Tidak ada add-back/grid/interaction lain setelah hasil ini.

## R2-D — residual, tails dan conditional spread

Signed error = prediction minus actual. Full D primary bias global hanya **−0,037145**, tetapi low-tail actual≤5 (n=1.928) bias **+18,191251** dan high-tail actual≥95 (n=1.955) **−11,252023**. Recent-8 memperbaiki tail MAE menjadi **17,145575/11,404844**, tetapi pooled signed error makin negatif **−0,450290** dan RMSE tidak ikut membaik (17,357634→17,366347). MAE, RMSE dan bias menjawab pertanyaan berbeda.

No-codes primary low-tail hampir sama dengan D (18,188690 versus 18,221372), tetapi high-tail memburuk (13,194683 versus 11,514815). Pada H12, low-tail membaik 22,349143→21,390278 sementara high-tail memburuk 13,146680→14,661021. Jadi pooled H12 gain tidak boleh disebut perbaikan target ekstrem secara umum. Menghapus telemetry juga memperbaiki primary low-tail (17,273038), walaupun merugikan pooled MAE dan high-tail: contoh konkret fitur bermanfaat overall dapat merugikan segmen tertentu.

Efek EF: D expanding primary EF0/EF1 **14,573469/10,808115**; full recent-8 **14,508001/10,535816**. No-cycle recent-8 **14,849333/10,645256** mengorbankan EF0 sambil membantu EF1 terhadap expanding. No-codes **15,496776/11,132778** merugikan keduanya pada primary; early-H12 gain-nya terutama EF0. Semua EF comparisons bersifat observasional, bukan treatment effect.

Prediction range primary: two-feature C **1,120951–87,990976**, full D **−3,307724–103,544901**, recent-8 **−1,999004–104,079551**. Expansion mengurangi kompresi global tetapi tidak menghapus conditional tail bias. Clipping terpisah hanya menurunkan primary MAE D sebesar **0,001901**, recent-8 **0,000918**; seluruh tabel utama tetap raw.

Semua exact-impact groups disimpan, bukan hanya contoh sulit. Di antara n≥100, D pada impact **98,0411** (n180) MAE 18,022678; impact **85,9730** (n2.819) MAE 16,763560. Untuk 85,9730, actual IQR **34,617** versus prediction IQR **8,087**. Ini tidak otomatis membenarkan memperlebar prediksi: conditional median tidak harus meniru conditional outcome variance, dan heterogeneity yang tidak terukur/ranking error bisa menyebabkan spread besar. Jangan langsung memakai target transform atau percentile calibration.

Postfit ranking diagnostic dari saved vectors: primary pooled Spearman full D **0,798622** dan recent-8 **0,798669** hampir sama; equal-cycle Spearman **0,792083/0,792025**. Jadi gain MAE recent-8 bukan lompatan ranking quality yang jelas. Pada H12, no-codes meningkatkan pooled Spearman **0,731517→0,748607**, sejalan dengan MAE yang membaik. Ini mendukung pemisahan ranking dari location/spacing prediksi sebelum target-rank calibration; tidak ada transformasi prediksi yang dijalankan. Per-cycle dan exact-impact correlations tetap diagnostic, bukan causal mechanism.

## Temuan tambahan — category status sebagai data cohort

D primary all-codes-known/nonmissing MAE **12,801328** (n14.452) versus any-unknown **13,972788** (n18.645), dengan signed error **−2,585950/+2,606406**. Pada H12, unknown cohort justru mempunyai MAE lebih rendah 15,364813 versus known 15,863667. Jadi novelty bukan penalti yang seragam antarhorizon.

Counterexample penting: model no-codes yang sama sekali tidak memakai kategori masih menunjukkan primary known/unknown MAE **13,713854/14,578270**, bias **−2,801350/+3,213900**. Status kategori merupakan proxy komposisi/cohort yang berkorelasi dengan kesulitan data, bukan bukti bahwa unknown encoder sendiri menyebabkan error. Unknown dan missing groups overlap; jangan menjumlahkan denominators-nya. Seluruh cohort dibentuk relatif terhadap selected fit subset dan dibandingkan pada aligned validation records.

## Generalization risk dan rekomendasi Round 3

Laporan terpisah `TEMPORAL_GENERALIZATION.md` membedakan feature shift yang teramati dari dugaan future error. Test predictors dibaca setelah seluruh pilihan eksperimen selesai. Tidak ada Test labels/predictions dalam research track atau model selection berdasarkan distribusi Test.

Prioritas rekomendasi, belum dijalankan:

1. **Conditional ranking dan uncertainty/spread lebih dahulu sebagai diagnostic terarah.** Bekukan C, full D expanding/recent-8 dan no-codes sebagai controls. Ukur apakah kesalahan terutama urutan relatif di dalam cycle/impact atau bentuk conditional median. Bila lanjut ke quantile model, gunakan pinball loss dengan quantiles tetap dan calibration fit-only. Jangan menyamakan distribusi rank target dengan izin otomatis meranking ulang prediksi.
2. **Target-rank handling hanya sebagai eksperimen terpisah**, jika batch-cycle lengkap tersedia. Prediction-only within-cycle mapping dapat memperbaiki rentang sekaligus merusak MAE ketika ordering lemah. Bekukan acceptance pada primary/H12/EF/tails sebelum score; jangan memilih dari public leaderboard.
3. **Text features tetap kandidat information-gain berikutnya**, untuk membedakan records pada exact impact yang sama. Keberhasilan text pada classification tidak membuktikan manfaat regression. Gunakan train-only vocabulary, missing-stream guardrails dan pembanding structured yang sama; tidak perlu menyalin multiclass TP1 atau menggabungkan target kedua task.
4. **Jangan menambah history lengths/feature combinations acak.** Hasil saat ini lebih mendukung menjaga beberapa frozen controls dan menguji satu pertanyaan substantif daripada mengejar gain primary 0,13 sambil mengabaikan H12. Verifikasi semantik novel zero-count support sebelum mengaitkannya dengan mekanisme proses sumber.

Pelajaran REG-Lite B yang tetap tepat adalah conditional median untuk MAE, evaluasi tail dan regime heterogeneity. Target TP2 jauh lebih padat dan mempunyai within-cycle rank geometry; tidak ada pembenaran menyalin LightGBM multiclass satu kelas per skor. Missing/known category patterns dan history interaction di TP2 adalah evidence baru, bukan kelanjutan mekanis recipe TP1.

## Reproducibility dan batas pekerjaan

Base: **88 fresh fits**, 2 aliases, 418,29 detik. Interaction/all invocation: **24 fresh fits**, 90 cache/alias reuses, 336,76 detik. Total **112 unique fits** dari 114 recipe-history entries. H12 berbagi fitted model dengan primary yang mempunyai cutoff sama. Representative cutoff 67: **19 fresh fits**, 38 window evaluations, namespace terpisah; detail bit identity dan metric reconciliation ada di `verification.json`.

196 existing tracked checkpoint files dilindungi freeze; hanya README front door boleh diperbarui. Raw hashes diverifikasi, tidak ada raw datasets, per-row predictions, model binaries atau credentials masuk Git. TP1 tidak ditulis. `scope_preservation.json` secara jujur mencatat current status dua reference repos, termasuk unrelated untracked files yang dipertahankan; itu bukan byte-by-byte external before/after certificate.

QA numerik akhir **PASS**: 114 cache/audit groups, 152 jobs, 38 aggregates, 76 paired comparisons dan 16 exact Round1 controls (termasuk H12). Representative repeat 19 fresh fits/38 evaluations mereproduksi seluruh prediction vectors, fitted audit-state hashes dan complexity/tree hashes yang diperiksa secara bit-identical. Receipt dibind ke results hash serta prediction/position/audit manifests; tidak ada refit oleh verifier.

**38 scoped regression/compatibility tests lulus dalam 4,446 detik tanpa failure/error/skip.** Notebook Restart & Run All berhasil: 16 cells, 8 code cells dengan execution counts 1–8, 0 errors, 6 embedded PNGs. Enam figures diperiksa langsung pada ukuran baca; label fixed bins/cycle79/tail denominators serta garis/marker dibedakan. HTML preview diekspor; full-page HTML belum diperiksa melalui GUI browser. Visual QA berbasis PNG asli dan outputs/struktur notebook. Jupyter memakai Windows selector fallback dan nbconvert memberi warning alt-text pada 6 images, tanpa execution error. Dua tes tambahan memeriksa reference-hash fallback saat ignored Round1 vectors tidak tersedia; seluruh 16 kontrol aktual juga lulus melalui jalur tersebut. Submission Track menjalani 7 tes terpisah; scope dan hasil Kaggle ada di laporan submission. Semua angka ilmiah berasal dari results/vectors; Kaggle scores tidak mengubah keputusan penelitian ini.

**STOP setelah Regression Round 2 untuk review pengguna.** Belum ada model final, Regression Round3 atau Classification Round3. Authorization submission terbatas pada dua initial recipes lama di track terpisah.
