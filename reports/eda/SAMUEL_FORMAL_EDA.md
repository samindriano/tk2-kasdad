# EDA formal Samuel — hubungan impact–threat antarcycle

30 Oktober 2026 · Pertanyaan dipilih pengguna: **“Apakah hubungan antara impact_load dan threat_rank mengalami perubahan antarperiode cycle_code, meskipun distribusi threat_rank relatif stabil?”**

**Jawaban:** ya, terdapat perbedaan asosiasi dan distribusi kondisional yang teramati pada train. Pola awal bertahan pada beberapa pembagian periode, pembobotan cycle dan standardisasi komposisi impact×entry_friction. Namun hubungan tidak meningkat monotonik, perubahan tidak satu arah pada semua kelompok, dan bukti ini belum menetapkan structural break, causal drift atau prediction bias TP2.

## Data dan batas analisis

Seluruh 50.608 train regression digunakan; 33 cycle 47–79. Impact,EF,cycle,target lengkap dan finite. Test 80–91 tidak mempunyai label; hanya metadata/schema/ID yang diperiksa. SHA train `756523ff8156a9cd776e52249f7e2a8f23dcec432b8d10570e55ff6db0e0f082`. Raw dipertahankan. Satu record adalah observasi kerentanan menurut materi; satu cycle tetap utuh untuk perbandingan. Materi menjelaskan cycle lebih besar lebih baru; kalender absolut dan lineage target masih tidak diketahui.

EDA menggunakan exact impact values, Pearson/Spearman, kuantil dan mean decomposition deskriptif. Tidak ada estimator, prediction, training, tuning atau submission. Threshold support dan ilustrasi dipilih dalam analisis eksploratif, bukan praregistrasi. Angka pada laporan ini merujuk hash tersebut; perubahan dataset memerlukan review narasi.

## 1. Seluruh cycle: pola menguat, tetapi berfluktuasi

Pearson per-cycle berkisar **0,4472–0,7557**; Spearman **0,4148–0,7428**. Seluruh 33 cycle dan ukuran sampelnya ada di `temporal_robustness.json` dan figure `temporal_per_cycle.png`. Contoh fluktuasi: cycle 55→56, Pearson 0,4472→0,6511 (n 1.608→564), kemudian 0,6205 di 57 (n 962). Penurunan 73→74 dan kenaikan 74→75 menjadi counterexample terhadap tren monotonik. Cycle 56 menarik untuk dicatat, tetapi bukan breakpoint yang sudah dibuktikan atau alasan otomatis membuang cycle sebelumnya.

Periodisasi utama membagi 33 nilai cycle unik terurut menjadi 3 blok 11 cycle, tanpa memotong cycle:

| Cycle | n | Pooled Pearson | Spearman | Equal-cycle mean Pearson |
|---|---:|---:|---:|---:|
|47–57|13.546|0,4994|0,4773|0,5130|
|58–68|16.736|0,6722|0,6631|0,6733|
|69–79|20.326|0,6964|0,6798|0,7022|

First-to-last delta pooledPearson **+0,1970** dan equal-cycle **+0,1892**. Ini bukan hanya efek cycle akhir yang lebih besar. Membagi menjadi 2 blok memberi 0,5794→0,6827; 4 blok memberi 0,4830→0,6845→0,6827→0,6827. Jadi cerita yang lebih akurat adalah hubungan awal lebih lemah dengan fluktuasi sesudahnya, bukan peningkatan mulus sepanjang waktu.

Mengeluarkan satu cycle dari endpoint contrast mempertahankan delta **+0,1885 hingga+0,2063**. Resampling 2.000 cycle-cluster, seed 42, memberi rentang persentil 2,5–97,5% **0,1510–0,2365**. Ini **sensitivitas eksploratif**, bukan confidence interval: cycle dianggap exchangeable di dalam blok, ketergantungan serial tidak ditangani, dan interval bukan perkiraan cycle 80–91.

Nonlinearity diagnostic: eta² untuk kelompok exactimpact **0,5026**, dibanding Pearson² **0,4058**. Ini variance decomposition in-sample, bukan predictiveR² atau bukti causal nonlinear effect. Bentuk kelompok dan EF mixture dapat menjelaskan selisih; pendekatan linear global belum tentu cukup, tetapi model nonlinear tetap harus diuji nantinya.

## 2. Komposisi EF menjelaskan sebagian pola, tidak seluruhnya

| Kelompok / pembobotan |47–57|58–68|69–79|
|---|---:|---:|---:|
|Pooled Pearson|0,4994|0,6722|0,6964|
|EF=0|0,3663|0,5397|0,5385|
|EF=1|0,3855|0,4966|0,6251|
|Distribusi impact×EF dibuat sama|0,5244|0,6617|0,6837|

Temuan tambahan: EF=0 hampir datar setelah blok tengah, sedangkan EF=1 masih menguat. Kesimpulan pooled tidak mewakili lintasan semua regime.

Untuk standardisasi, pilih exactimpact×EF cell dengan **≥30 baris di setiap blok**, hanya berdasarkan hitungan fitur. Ada 33 cell yang mencakup 48.630 baris (**96,09%**). Setiap cell mendapat share tetap dari pooled counts tiga blok, lalu share dibagi rata ke baris cell pada masing-masing blok. Tidak ada target dalam pemilihan bobot. Effective n 12.557/15.796/19.315 dekat dengan retained n 12.969/16.015/19.646; tidak ada target-based weighting atau klaim adjustment kausal.

Perbedaan masih terlihat setelah komposisi fitur dibuat sama. Jadi **perubahan share impact×EF saja tidak cukup** menjelaskan keseluruhan perbedaan pada populasi common support. Fitur lain, kualitas observasi dan skala rank masih dapat berperan.

Support bukan seluruh populasi:1.978 baris dikeluarkan;182 dari 2.527 low-tail labels dan 61 dari 2.560 high-tail labels berada di luar common support. Karena itu coverage low-tail 92,80% dan high-tail 97,62%; generalisasi pada kelompok langka tetap terbatas. Impact-only support ≥30 mencakup 32/80 level dan 49.061 baris (96,94%); sensitivitas threshold 20/50/100 masing-masing mencakup 49.563/47.859/47.152 baris. Daftar lengkap level/cell dan coverage ada di JSON.

**Positivity:** hanya 3 dari 32 common impact levels memiliki kedua EF dengan≥30 baris pada setiap blok. Jangan menganggap ini matched causal comparison EF atau mengestimasi “efek EF” di semua impact.

## 3. Conditional shifts berbeda arah dan bentuk

Contoh dipilih setelah melihat discovery untuk menjelaskan pola; seluruhcommoncell tetap tersimpan, sehingga dua ilustrasi tidak dianggap mewakili semua kelompok.

| Exact impact / EF | Blok | n | Median | P10 | P90 | IQR |
|---|---|---:|---:|---:|---:|---:|
|154,0178 /1|47–57|1.498|83,486|54,495|98,197|26,247|
|154,0178 /1|58–68|2.698|86,256|59,337|98,225|17,543|
|154,0178 /1|69–79|3.108|89,964|71,805|98,347|14,080|
|85,9730 /0|47–57|1.016|34,607|6,474|70,542|35,686|
|85,9730 /0|58–68|1.072|26,433|3,983|53,661|31,015|
|85,9730 /0|69–79|1.633|20,011|6,267|60,241|37,116|

Pada impact 154,0178 semua baris EF=1: median naik 6,478 poin, P10 naik 17,310, P90 hampir tetap (+0,149), IQRmenyempit 12,167. Jadi perubahan bukan pergeseran lokasi seragam; kelompok semakin terkonsentrasi di rank tinggi. Proporsi≥95 tidak naik monotonik:23,56%→22,09%→28,25%.

Pada impact 85,9730 semua baris EF=0: median justru turun 14,596 poin, sementara mean turun 7,759 dan IQRakhir sedikit lebih lebar dari awal. Kuantil tidak bergeser paralel. Perbedaan seperti ini dapat tersembunyi di balik median atau mean global yang stabil.

**Pembalikan yang perlu dikoreksi interpretasinya:** discovery mendapati median impact 110,5078 lebih tinggi daripada 134,6432. Komposisi EF sangat berbeda, tetapi mengontrol EF tidak membuat semua ringkasan monotonik. DiEF=0, mean higher-impact minus lower-impact positif pada semua blok, sementara median contrast adalah−4,348/+7,856/−0,378. Kelompok EF=0 pada 110,5078 hanya 51/66/40 baris; pada 134,6432 ada 1.320/1.489/2.383. EF=1 tidak tersedia pada 134,6432, sehingga perbandingan EF=1 tidak dapat dibuat. Ini contoh mean,median dan composition menjawab aspek berbeda, bukan bukti impact yang lebih tinggi menurunkan prioritas secara kausal.

## 4. Mean stabil dapat menyembunyikan perubahan yang saling meniadakan

Pada **common-support population** (bukan semua train), mean early→late 50,1245→50,2920, net **+0,1676**. Symmetric decomposition memberi komponen perubahan komposisi **+1,0504** dan perubahan mean di dalam cell **−0,8828**; jumlah cocok hingga floating-point error. Kelompok 154,0178/EF=1 memberi kontribusiwithin positif+1,0435, sedangkan 85,9730/EF=0−0,6264.

Ini identitas aritmetika, bukan decomposition penyebab atau penjelasan persentase perubahan Pearson. Mean semua train malah 50,0406→50,0271; denominator berbeda karena support exclusion. Stabilitas agregat dapat terjadi bersamaan dengan pergeseran yang berlawanan pada kelompok besar.

## 5. Mengapa marginal threat tampak begitu stabil?

Seluruh 50.608 target tepat sama dengan `round(100×average_rank(target dalam cycle)/n_cycle,4)`. Ini tetap **target-on-target diagnostic**, bukan feature, estimator atau bukti source-generation formula.

Conditional pada identitas numerik tersebut, average ranks selalu mempunyai mean(n+1)/2. Maka mean target sebelum pembulatan harus **50+50/n_cycle**. Pada data aktual, selisih maksimum mean dari nilai ini hanya **0,000002832 poin**. Jadi mean sekitar 50 tidak memberikan bukti tambahan bahwa risiko laten, composition atau hubungan fitur stabil; nilainya hampir dipaksa oleh geometri rank yang teramati.

Two-sided ECDF distance terhadap uniform paling besar 0,022333 di cycle 62 (n 1.500, tie terbesar 66). Conditional pada midrank geometry, distance sesuai `(max_tie+1)/(2n)` dalam toleransi pembulatan 4 desimal; ini menjelaskan pula deviasi marginal kecil akibat ties. Tidak ada p-value uniformity atau klaim proses sumber sudah diidentifikasi. Ketidakstabilan hubungan rank dengan impact tetap mungkin walau skor relatif selalu hampir uniform.

## 6. Missingness dan penjelasan alternatif

Stream-present Pearson 0,511→0,677→0,703; stream-missing 0,401→0,598→0,586 dengan n 1.009/1.072/1.323. Share missingstream 7,45%→6,41%→6,51%. Arah early→late tetap pada kedua kelompok, sehingga missingstream saja bukan penjelasan lengkap. Ini subgroup sensitivity tanpa adjustment semua fitur, bukan nilai tambah text yang sudah dibuktikan. Org/module missingness juga dicatat sebagai konteks; tidak menambah pertanyaan EDA anggota lain.

Alternatif yang belum tersingkir: komposisi fitur lain, perubahan mutu/konteks observasi, ketergantungan record dalam cycle, dan perbedaan skala relatif dengan risiko absolut. Tidak ada data laten atau label test untuk menguji penjelasan sumber. Support selection juga mengecualikan sebagian kelompok langka dan low-tail secara tidak seimbang.

## 7. TP1 ke TP2 dan implikasi

Pelajaran yang bertahan dari REG-Lite B: periksa seluruh distribusi, median dan tails; MAE dan signed bias berbeda; subgroup yang sulit dapat tertutup oleh average. Phase 16 B2 memberi alasan memeriksa regime heterogeneity. Namun target TP1 hanya 80 nilai dengan top 10 mencakup 79,10% baris, sedangkan TP2 mempunyai 34.333 nilai dan top 10 hanya 1,08%. Multiclass satu kelas per skor TP2 tidak dibenarkan dari recipe TP1. Label/OOF/model TP1 tidak digabungkan keTP2.

EDA menunjukkan **risiko generalisasi kondisional** ke cycle baru dan alasan untuk membandingkan forward validation, tetapi belum mengukur prediction bias, MAE masa depan atau manfaat recency/cyclefeature. Menguatnya korelasi pada train akhir tidak berarti test lebih mudah. Menggunakan label target untuk feature, rank ulang foldtarget, atau calibration dengan labeltest tetap dilarang.

Rancangan eksperimen siap direview di `docs/REGRESSION_EXPERIMENT_REVIEW.md`: baseline median yang interpretabel; respons nonlinear impact×EF; ablation waktu/history; conditional quantiles; penanganan relative rank; serta text/missingness bila baseline sudah dipahami. Tidak ada recipe yang dijalankan. MAE dan validation windows perlu disepakati serta dicocokkan dengan rubric sebelum training.

## Kesimpulan milestone dan keputusan berikutnya

Pertanyaan utama **terjawab secara deskriptif** dan tidak perlu diganti. Temuan awal valid, dengan penajaman: perubahan tidak monotonik dan tidak satu arah, marginal stabil sebagian besar konsisten dengan geometri skala rank, dan composition control memperkuat bukti perbedaan kondisional tanpa memberi identifikasi kausal. Notebook demo: `03_samuel_formal_eda.ipynb`; supporting investigations tetap pada script/JSON agar notebook utama ringkas.

Milestone berhenti untuk review pengguna atas validation protocol, metric resmi, fitur allowlist dan putaran eksperimen pertama. Tidak ada finalmodel, training, tuning, ensemble atau submission.
