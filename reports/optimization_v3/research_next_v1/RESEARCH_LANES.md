# Research sesudah 45 recipe — shortlist mekanisme baru

> **Catatan riset awal, bukan panduan eksekusi terbaru.** Gunakan [LUNA_HANDOFF.md](LUNA_HANDOFF.md), [queue canonical](../../../experiments/optimization_v3/research_next_v1/queue.json), dan [prompt](START_LUNA.md) untuk menjalankan fase ini. Queue operasional membatasi N00–N06, memperjelas biaya worst-case26 pilot fits, menunda N07/N08, dan menetapkan N05 pada origin67/75 dengan empat quantile heads baru. R11 survival probabilities tidak tersimpan dalam bundle yang tersedia. Recipe operasional mengatasi setiap perkiraan/ambiguitas di catatan awal ini.

1 November 2026. **PLANNED_NOT_RUN; planning only, bukan freeze atau perintah training/submission.** Review dimulai pada `92b765a`; agent kampanye lain masih aktif dan HEAD terus bergerak (snapshot lanjut tercatat di [feasibility.json](feasibility.json)). Tidak ada file eksperimen aktif, ledger, source model, notebook, atau Git index yang diubah untuk menyusun dokumen ini.

## Keputusan riset

Masih ada ruang untuk lane bernilai. Urutan awal yang disarankan: **N00 diagnosis dari continuation terbaru → N01 group/class weighting Classification → N02 interaksi TensorSketch → N03 risk-aware Regression routing**. N04 domain weighting bersyarat; N05–N08 merupakan pendalaman mekanisme atau pendekatan lebih spekulatif. Ini bukan daftar yang harus dihabiskan secara otomatis.

Kampanye lama terutama menguji representasi, learner dan fixed blend. Lane baru harus menjawab mekanisme yang belum teruji, bukan membuka ulang vocabulary/rank/threshold grid. Semua hipotesis di sini adaptif terhadap data development yang sama; tidak ada untouched Train holdout baru.

| ID | Lane | Prioritas | Perkiraan predictor fit awal, termasuk pembanding |
|---|---|---|---:|
| N00 | Atlas error sesudah expanded continuation | Wajib sebelum fit | 0 jika bundle+QA lengkap |
| N01 | Classification: bobot kelas × volume | Tinggi | 4 |
| N02 | Classification: interaksi degree-2 TensorSketch | Tinggi | 2–4 |
| N03 | Regression: router belajar risiko kombinasi dari OOF | Tinggi, setelah N00 | 2 kepala pilot LATE; sampai 5 kepala untuk enam-origin |
| N04 | Bobot berdasar covariate shift, bukan umur saja | Bersyarat | 8 per task untuk pilot dua origin |
| N05 | Lebar conditional quantile sebagai informasi ketidakpastian | Bersyarat/information lane | 0 jika ordinal arrays cukup; kalau tidak 4 quantile fits |
| N06 | Character surface versus token identity | Mechanism diagnostic | 4 char fits untuk dua bijeksi × dua origin |
| N07 | Learned factorization-machine interactions | Spekulatif, feasibility dahulu | 2–4 per task, backend/optimizer harus dikunci dahulu |
| N08 | Deep averaging network kecil dari nol | Spekulatif, feasibility dahulu | 2 kandidat +2 matched linear-head controls per task |

Biaya bukan janji waktu. Transform, indexing, domain classifier, auxiliary heads, negative controls dan retry juga dicatat. Cost full-six/refit belum termasuk kecuali disebut. Jangan menghitung neural/FΜ sebagai siap jalan sebelum dependency dan implementasinya tervalidasi.

## N00 — perbarui atlas error tanpa fit

**Pertanyaan:** setelah hasil R06/R21/C11/C17 terbaru lolos QA, apakah ada kekurangan yang tetap muncul pada origin berbeda, dan siapa yang mempunyai error saling melengkapi?

- Baca hanya bundle final yang diverifikasi, termasuk recovery v2; jangan memperlakukan R11 yang gagal disimpan sebagai model gagal performa.
- Join row IDs dan origin secara eksplisit; gunakan primary enam-origin dan H12 terpisah. Tampilkan cycle, stream availability, EF, serta empat kuantil impact dari fit prefix; minimum subgroup n100 Regression, n200 dan30/kelas Classification.
- Laporkan gain, harm, forecast horizon, disagreement dan arah antarorigin; jangan memilih hanya subgroup yang menang.
- Oracle minimum error per baris boleh dipakai sebagai headroom DESKRIPTIF, bukan prediksi router atau achievable upper bound generalisasi.
- Bila tidak ada pola feature-available yang stabil, jangan memaksakan routing baru. Hasil ini boleh mengubah prioritas lane SEBELUM recipe baru dibekukan.

Ini memberi nilai dari fit yang sudah dibayar. Tidak mengizinkan membuka Test atau menggunakan public leaderboard untuk memilih lane.

## N01 — group/class weighting Classification

**Hipotesis:** kelemahan positive recall pada volume tinggi sebagian merupakan alokasi training loss, bukan kurangnya vocabulary atau threshold yang salah.

Bukti: C16 continuation gagal guard text-load Q4 W2; C15 juga mempunyai harm recall Q4. Menambah model/score sebelumnya belum otomatis melindungi kelompok tersebut. Ini alasan terarah, bukan bukti weighting pasti membantu.

**Minimal recipe:** sparse LightGBM expert saat ini, features dan parameter persis; tambah hanya sample weights. Hitung text_load quartiles dari fit rows. Untuk G bin yang benar-benar terbentuk, hitung cell kelas×bin. Gunakan `sqrt(n_fit / (2*G*n_cell))`, clip[0.5,2], lalu normalize mean1. Catat efek normalisasi pada batas akhir. Tidak memakai outer labels untuk bins, bobot, cutoff atau rute. Sel n<100 adalah blocker support, bukan alasan mengganti formula sesudah melihat skor.

**Pembanding penting:** (a) unweighted expert yang identik dan sudah diverifikasi; (b) global class-only weights yang mempunyai total massa bobot per kelas sama dengan candidate, tetapi tidak membedakan volume dalam kelas. Bobot control per kelas = mean candidate weight di kelas tersebut. Ini membedakan manfaat volume-specific dari sekadar class rebalancing.

Fit kandidat dan class-only control masing-masing pada55/67 =4 fits. Route tetap .75 frozen C2_B+.25 new sparse expert pada present; CatBoost missing tetap; threshold.50. Bila baseline accepted saat batch baru berbeda, jangan transplant recipe diam-diam—bekukan amendment dan kontrol sebelum fit.

**Berhenti:** tidak memperbaiki champion secara material menurut gate yang dibekukan, tidak mengungguli matched control secara konsisten, atau guard kelas/grup lain memburuk. Jangan mengubah threshold untuk menyelamatkan weighting.

Berbeda dari C06–C08 (fitur kategori), C15 (score head), dan R16 (time weighting). Fokusnya objective allocation, bukan fitur baru.

## N02 — TensorSketch: interaksi pasangan yang lebih luas

**Hipotesis:** main-effect LR dan tree yang memecah satu fitur per split belum mengekstrak interaksi distributed token×structured secara efisien.

C09 hanya menguji interaksi teks×EF; C14 random structured partitions gagal. Keduanya counterexample yang menurunkan ekspektasi, tetapi bukan uji polynomial feature map degree2.

**Minimal recipe:** pertahankan main-effect matrix C2_B (structured/OHE+TF-IDF) persis. Buat L2-normalized COPY gabungan khusus input sketch; jangan mengubah main-effect block. Tambah satu `PolynomialCountSketch(degree=2,n_components=1024,gamma=1,coef0=0,random_state=42)`; LR C1/liblinear/max_iter2000/seed42. Pipeline fit-only; sketch output dense1024 saja, bukan TF-IDF penuh yang didensify.

**Policy tetap:** pada present `.75*current_frozen_classification_route +.25*new_expert`; missing memakai fallback lama persis; threshold.50. Ini single fixed policy, bukan pilihan antara replace/aux sesudah skor.

**Matched control:** expert LR sama tanpa sketch, policy blend sama. Reuse C2_B hanya jika fitted recipe/feature matrix/score equivalence terbukti; kalau tidak,2 fit control tambahan. Candidate2 fits pada55/67; control0–2. Dibandingkan dengan champion dan no-interaction control, bukan hanya baseline lemah.

**Berhenti:** sketch tidak memberi incremental gain, solver/numerics tidak memadai, atau guard subgroup gagal. Tidak ada grid dimensi/seed/degree. Uji bentuk, determinism dan RAM sebelum fit. Seluruh1024 float64 kolom pada29.705 fit rows sekitar232 MiB, belum termasuk score rows, copies, solver dan transform temporaries; itu bukan peak RSS.

Ini uji joint second-order representation: mencakup text×text dan structured×structured juga, bukan identifikasi kausal khusus cross-modal. [Dokumentasi sklearn1.8](https://scikit-learn.org/1.8/modules/generated/sklearn.kernel_approximation.PolynomialCountSketch.html) menjelaskan pendekatan polynomial kernel melalui TensorSketch.

## N03 — belajar kapan kombinasi merugikan, bukan sweep bobot

**Hipotesis:** input yang tersedia saat inference dapat memprediksi selisih loss antara R22 dan kombinasi R22+R06, sehingga manfaat komplementaritas dapat dipakai tanpa menambah harm high-tail.

Bukti: fixed combination yang telah direview memberi pilot H12 gain sekitar0.221/late0.168 terhadap legacy champion, tetapi high-tail W4 harm0.305>0.25. Besaran sama-sama50% komponen baru dari satu learner lebih lemah daripada kombinasi pada late; ada indikasi complementarity, tetapi combo asli tetap gagal guard. Ini BUKAN izin rerun blend tersebut untuk berharap W4 hilang.

**Dua arm yang tetap:** (A) R22_ORIGINAL; (B) pada present `clip(.5*legacy_champion+.25*rank50(raw_R22)+.25*rank50(raw_R06),0,100)`. Pada missing kedua arm menggunakan A persis. Rank tetap seluruhcycle sebelum mask. Tidak menambah bobot atau arm lain.

**Router:** bentuk training target OOF `abs(y-B)-abs(y-A)` hanya dari empat-cycle prediction blocks yang origin fit-nya berakhir sebelum cycle target. Fit HGB `loss=squared_error,max_depth=2,max_iter=100,min_samples_leaf=200,l2_regularization=10,random_state=42,early_stopping=False` untuk expected loss difference. Inputs dibatasi skor kedua arm, disagreement, parsed impact/EF/text_load/availability; tidak observed target, error atau actual-tail flag ketika inference. Route ke B hanya jika predicted difference <−0.03, selain itu A. Nilai tersebut harus dibekukan sebelum fit router dan bukan hasil threshold sweep.

Minimal2.000 present OOF rows; bila tidak cukup, fallback A yang ditentukan sejak awal. Earliest outer55 tidak mempunyai OOF yang sah untuk exact arms: fallback A, bukan fabricated early residuals. Training cutoff67 hanya memakai target OOF≤67; cutoff75 hanya≤75.

**Biaya:** memakai verified base caches, bukan fit ulang R22/R06. Dua head pilot pada67/75 memakai profile LATE; jangan meranking raw gain-nya terhadap STANDARD. Bila lanjut, tambah59/63/71 untuk melengkapi lima heads;55 fallback. Controls: A, B dan constant-arm choice berdasarkan PAST mean loss saja. Bandingkan conditional router dengan constant decision untuk menguji manfaat conditioning.

**Stop:** tidak mengalahkan A/control constant, coverage kecil, atau tail/window guard gagal. Audit terhadap current R22 DAN legacy protected baseline: jangan merebase guard agar harm0.305 lama tersembunyi. Ini adaptasi pemilihan arm, berbeda dari R17 affine/R18–19 offset yang mengubah nilai prediksi. Masih berisiko overfit dan tidak otomatis lebih baik dari residual head lama.

OOF wajib nyata; random KFold default atau in-sample stacking tidak boleh dipakai. [Dokumentasi stacking](https://scikit-learn.org/1.8/modules/generated/sklearn.ensemble.StackingRegressor.html) memperingatkan risiko memakai skor model yang dilatih pada data meta-head yang sama.

## N04 — covariate-shift weighting tanpa Test

**Hipotesis:** contoh lama yang secara fitur menyerupai periode terbaru lebih berguna daripada bobot berdasarkan umur saja.

Untuk setiap outer cutoff, seluruh data tetap≤cutoff. Empat cycle terakhir DALAM FIT PREFIX menjadi pseudo-target domain, sisanya source. Domain classifier tidak boleh melihat X validation/Test, target task, record_id atau cycle_code eksplisit. Tidak menggunakan timestamp proxy yang sengaja merekonstruksi label domain.

Gunakan structured+word-TFIDF fit-only; domain LR C0.1/class_weight balanced/max_iter2000/seed42. Dua-fold grouping by cycle, pastikan tiap fold mempunyai kedua domain, untuk out-of-fold domain probabilities. Setelah prior correction, clipped odds[.25,4] memberi weights source; recent rows weight1; normalize. Target task learner pertama adalah exact R22 dengan sample weights, tanpa mengganti features/mapping/policy.

**Controls:** unweighted cached R22; weighted model dengan weights dipermutasi di DALAM cycle menggunakan seed42. Permutasi menjaga alokasi waktu tetapi menghilangkan penyesuaian per fitur. Dua domain fits+weighted task fit+shuffled-weight task fit per origin; pilot55/67 total8 predictor fits. Tidak menyebut domain fits gratis.

**Preflight:** held-out domain AUC dan histogram/overlap; ESS=(sumw)^2/sum(w^2) harus≥50% fit n. Bila domain tak terbedakan atau hampir terpisah sempurna, jangan memaksakan weighting. Threshold diagnostic yang dipakai untuk skip harus dibekukan sebelum task scores; tidak menganggap AUC tinggi selalu baik.

**Stop:** gagal melampaui unweighted/shuffled controls, effective support jatuh, atau harms. Lebih dulu Regression; Classification bukan otomatis ikut bila yang pertama gagal. [Teori covariate shift](https://jmlr.org/papers/v8/sugiyama07a.html) memerlukan asumsi perubahan P(X) dengan hubungan target bersyarat yang cukup stabil; regime change dapat melanggarnya. Jangan menyatakan estimasi ini mencerminkan distribusi Test yang belum dibaca.

## N05 — uncertainty dari conditional quantiles

**Pertanyaan awal adalah informasi, bukan skor:** apakah lebar interval prediksi membantu mengenali kondisi sulit di luar impact/EF dan disagreement komponen?

Periksa dulu apakah saved R11 pilot menyimpan cumulative probabilities yang dapat dipakai untuk interval diagnostic; jangan menganggap bundle fit yang gagal ada. Jika tidak ada, dua LightGBM quantile heads alpha0.1/0.9 dengan structured+SVD32 yang sama, parameter ditetapkan sebelum fit, dua origin =4 predictor fits. Crossing ditangani dengan sorting pasangan; laporkan crossing rate, interval coverage/width percycle dan subgroup.

Bandingkan dengan interval lebar konstan dari PAST residual quantiles; bukan interval oracle dari validation. Jangan menganggap mengestimasi interval otomatis memperbaiki MAE point prediction. Baru boleh menambah width ke N03 dalam eksperimen TERPISAH jika diagnostic memberi alasan, dengan kepala tambahan dihitung. Quantile alpha0.5 saja mengulang target functional L1 dan bukan lane baru.

## N06 — apakah char model memakai pola permukaan token anonim?

Re-encode token melalui deterministic bijection independen dari target, preserving token identity recurrence, batas, urutan, multiplicity dan panjang bila memungkinkan. Refit char R06 pada dua mapping seeds tetap dan dua origin =4 fits, original R06 sebagai control. Jangan sekadar mengganti token pada validation tanpa retrain: itu hanya menciptakan OOV shift.

Unseen token rule harus sama dan tidak mempelajari lexicon dari future/Test. Collision check wajib. Jika memakai word-TFIDF invariance control, pertahankan vocabulary IDs dan tie-breaking max_features; rename dapat mengubah lexical tie selection sehingga baseline tidak otomatis identik.

Lane ini menjelaskan risiko generalisasi encoding; tidak menjanjikan kandidat dengan score lebih tinggi atau semantik token. Bila char gain hilang setelah rename tetapi word control invariant, original surface memang berperan; dua seeds belum membuktikan mekanisme pembentukan dataset.

## N07 — sparse learned factorization machine

Rank16 second-order FM pada token+structured sparse design, satu seed dan konfigurasi regularisasi. Bandingkan dengan exact linear main-effect control; no pretrained embedding. Ini alternatif learned pair interactions terhadap random TensorSketch, bukan tambahan vocabulary.

Dependency/optimizer feasibility belum lulus. Torch tidak terpasang pada snapshot; jangan memasang dependency atau membuat implementasi custom besar di tengah kampanye aktif. Sebelum executable card dibuat, tentukan backend yang benar-benar mendukung objective logistic untuk Class atau L1 untuk Reg, gradient test, normalization, epochs, dan anggaran matched controls. Jangan menyebut regression squared-loss FM sebagai pengujian L1. Author implementation/reference: [libFM](https://www.libfm.org/).

Jika N02 gagal, N07 tidak otomatis sia-sia (learned vs random interactions), tetapi prioritasnya turun kecuali ada bukti underfit/fit quality yang spesifik.

## N08 — small deep averaging network dari nol

Ini eksperimen learner yang berbeda: unigram vocabulary20k fit-only; embedding32, mean pooling, structured features, MLP64→32 ReLU/dropout0.1, fixed20epochs/Adam1e-3/batch256/seed42; task loss BCE atau MAE pada y/100. Tidak pretrained/LLM semantic embedding. Label task lain tidak dipakai. Inner-only tuning jika kelak dipilih harus dihitung; rancangan awal tidak menggunakan outer early stopping.

Control mean-pooling dengan linear head dan preprocessing sama untuk membedakan nonlinear head dari sekadar representasi bottleneck. Dua candidate+dua matched control fits per task. CPU/RAM/optimizer stability harus diukur dahulu; waktu belum diketahui. Dengan data50k dan token anonim, overfit serta OOV merupakan risiko besar. Ini high-risk lane sesudah pilihan murah, bukan transformer otomatis.

Dasar arsitektur: [Deep Unordered Composition, ACL2015](https://aclanthology.org/P15-1162/). Paper mendukung family model, bukan bukti bahwa semantic text result akan berpindah ke token anonim TP2.

## Yang tidak diprioritaskan lagi

- Vocabulary/min_df/SVD dimension/threshold/weight sweeps, C09 EF interaction yang sama, global/EF offsets, old residual HeadA, atau nama model baru dengan policy lama tanpa hipotesis.
- Exact text duplicate lookup: profile Train present47.204, unique normalized strings47.202, hanya1 row mempunyai stream identik pada cycle sebelumnya. Ini melemahkan exact lookup, BUKAN membuktikan semua nearest-neighbor similarity tidak berguna. Approximate retrieval membutuhkan support/similarity feasibility terpisah sebelum menjadi lane.
- Rank/CDF broadening hanya untuk memulihkan tail: matched mapping-only control sebelumnya bisa memperbaiki tail sambil merugikan agregat. Jangan menganggap tail gain sebagai bukti learner baru.
- Threshold-only temperature scaling pada probability akhir di0.50 tidak mengubah label. Calibration/stacking biasa juga sudah banyak diuji V2/V3.

## Cara melanjutkan dengan Luna tanpa mengulang masalah batch lama

1. Tuntaskan QA/receipt kampanye aktif dahulu. Jangan mengambil slot training atau mengubah file/index milik agent lain.
2. Bekukan baseline baru untuk SELURUH batch berikutnya berdasarkan internal evidence dan review; public score hanya dicatat. R22 adalah qualified Reg pada snapshot, sparse25 adalah Class fallback terakhir yang terverifikasi. Expanded outcomes yang belum selesai dapat mengubah baseline sebelum batch dimulai, bukan di tengah batch.
3. Jalankan N00, lalu pilih2–3 lane murah dahulu. Usulan tranche awal N01(4)+N02(2–4)+N03(2)=8–10 predictor pilot fits, ditambah kontrol yang ternyata harus direfit. Ini usulan budget BARU, bukan izin reset180. Domain/FM/DAN tidak otomatis ditambahkan.
4. Budget aktif terakhir yang dibaca175/180 reserved; nilai bisa bergerak. Proyeksi log177/180 adalah snapshot berbeda. Setelah pekerjaan aktif direkonsiliasi, minta/rekam amendment cumulative cap atau budget fase terpisah secara eksplisit. Tetap catat175 dan retries historis; tidak menghapus atau mengganti nama namespace agar counter nol.
5. Smoke test persistence harus mencapai saved bundle+sidecar+reload+independent metric dan report untuk SETIAP adapter. Masukkan dict diagnostics, string IDs, empty groups, rank ties dan resume. Jangan hanya compile/test estimator. Jika penyimpanan pertama gagal, jangan terus melatih origin lain lewat jalur gagal yang sama; perbaiki contract lalu verifikasi sebelum retry. Jika model masih in-memory, simpan valid output lewat versioned amendment, jangan refit otomatis.
6. Freeze exact inputs/source/recipe, gates, subgroup views dan budget sebelum task score baru. Gunakan scope guard primary/H12/each primary window konsisten seperti final_continuation_plan; threshold/gain lama hanya titik awal yang harus dinyatakan sebelum eksekusi. N03 profile LATE dipisahkan, bukan dibandingkan raw dengan STANDARD.
7. Satu training job dua thread, jobcap6GiB/reserve2GiB, actual RAM preflight perorigin. Worker boleh implementasi/review scopes berbeda. Pilot lolos baru full-six, maksimal satu recipe terbaik permechanism; kalah bukan alasan grid rescue tanpa batas.
8. Tidak ada Test/Kaggle pada tahap research ini. Submission membutuhkan milestone tersendiri dengan recipe/gate/exception yang eksplisit, bukan otomatis meneruskan izin tujuh benchmark sebelumnya ke semua lane baru.

## Referensi evidence proyek

- [45-card review](../postscreen_review/CARD_REVIEW.md), [review mekanisme](../postscreen_review/REVIEW.md).
- [Guard audit dan keputusan continuation](../final_continuation_plan/DECISION.md), [hasil continuation awal](../final_continuation/CAMPAIGN_SUMMARY.md).
- [V2 calibration/stacking dan representation failures](../../optimization_v2/FINDINGS.md).
- [History/feature interactions dan impact spread](../../regression_round2/FINDINGS.md).
- [Impact/availability/OOV research](../../impact_eda/DECISION.md).

Semua lane berstatus **belum dilatih**. Dokumen ini tidak mengklaim ada expected Kaggle gain atau bahwa sisa error seluruhnya dapat dipelajari. Semakin banyak eksperimen pada Train yang sama, semakin kuat kebutuhan matched controls, counterexamples, dan pembatasan adaptasi.
