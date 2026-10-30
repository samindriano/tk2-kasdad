# Regression TP2 — audit, perbandingan TP1, dan discovery Samuel

30 Oktober 2026 · **DISCOVERY: pertanyaan EDA belum dikunci.** Analisis hanya deskriptif pada train berlabel; test untuk audit input. Tidak ada training, tuning, prediksi TP2, atau submission.

## Kesimpulan yang didukung data

Melanjutkan **pelajaran** REG-Lite B tentang bentuk distribusi, ambiguitas kondisional dan perilaku target ekstrem relevan. Memindahkan pertanyaan error model atau recipe multiclass TP1 secara langsung belum beralasan: target TP2 berbeda drastis. Arah terkuat sekarang adalah **perubahan distribusi kondisional threat_rank pada impact_load yang sama antarcycle, meskipun marginal target hampir tetap**. Ini rekomendasi kandidat, bukan keputusan final Samuel.

## 1. Audit regression aktual

Arsip `tp2-regression-cyber-vulnerability-kasdd-gasal-2026-2027.zip` dipindahkan dari Downloads ke `data/raw/regression/` dan diekstrak tanpa overwrite. CRC lolos; SHA-256 archive `8d8a6e036bc23382574d432cbf72a3195ffdb1b3047b7db3d255b6b7991d157a`. Receipt sumber, ukuran dan hash tiap CSV: `reports/audits/regression_acquisition.json`. Nama arsip adalah bukti asal lokal unduhan pengguna, bukan autentikasi ulang upstream Kaggle.

| Pemeriksaan | Hasil aktual |
|---|---|
| Shape | Train 50.608 × 19; test 29.421 × 18; sample 29.421 × 2 |
| Target | `threat_rank`: satu-satunya kolom train-only; sample tepat `record_id, threat_rank` |
| Predictor | 17 selain record_id; nama, urutan dan dtype fitur train/test cocok |
| Missing target / impact / cycle | 0 / 0 / 0 pada train; impact dan cycle test juga lengkap |
| Duplikasi | 0 full-row duplicate, duplicate ID atau missing ID; 0 ID overlap train/test |
| Duplicate feature vectors | 0 dalam tiap split tanpa ID/target; 0 shared distinct vectors antar-split |
| Sample | Jumlah baris dan urutan ID sama persis dengan test; nilai sample bukan label test |
| Numeric infinity | Tidak ada pada kolom numerik |
| Waktu | Train cycle 47–79, test 80–91 |
| Lineage lintas tugas | Test regression byte-identical dengan test classification; seluruh fitur bersama train cocok berdasarkan ID |

Profil **setiap kolom** (dtype, missing count/rate, cardinality, kategori/frekuensi, kuantil, IQR outlier, string length), target, dan perbandingan train/test tersedia di `reports/audits/regression_audit.json`; ringkasannya di `regression_audit.md`. Bentuk target dan outlier di bawah tidak dipakai menghapus data.

Missing predictor tetap material: `org_code` 5,31% train vs 23,27% test; `module_code` 7,99% vs 25,14%; `pattern_code` 16,17% vs 8,61%; `signal_stream` 6,73% vs 3,61%. Di antara baris test nonmissing, 20,96% org_code dan 36,77% module_code belum pernah ada di train. Field spread/scope_trace dan telemetry memiliki format campuran. Simpan raw utuh; preprocessing kelak perlu menangani missing, unknown dan parser secara eksplisit di dalam fold.

`scope_pressure` memiliki 6.284 nilai test di luar range train dan KS D=0,2138; ini audit covariate, bukan EDA Abhi atau alasan memilih model. Asal/timing `audit_lane`, `checksum_hint`, impact_load dan entry_friction tetap memerlukan keterangan sumber. Tidak adanya korelasi marginal besar pada checksum bukan sertifikasi bebas leakage. Jangan menambahkan label TP1 melalui join; seluruh analisis TP2 memakai fitur resmi yang sudah ada pada regression.

Materi lokal `TP KASDAD Ganjil 2026_2027.pdf`, halaman 16, mendefinisikan record_id sebagai ID observasi kerentanan dan cycle_code sebagai urutan siklus disclosure (lebih besar = lebih baru). Materi itu membahas TP1 dan kolom bersama, bukan definisi khusus threat_rank TP2. SHA dan parafrase sumber ada di `source_definitions.json`. Kalender absolut serta formula/range resmi target TP2 belum terkonfirmasi.

## 2. Target TP1 dan TP2 berbeda

Keduanya train dengan 50.608 baris dan ID yang sama. `impact_load` resmi TP2 cocok persis dengan label regression TP1 berdasarkan ID. Perbandingan berikut membandingkan **bentuk dan konsentrasi**, bukan menyamakan unit dampak teknis dan prioritas.

| Statistik train | TP1 impact_load | TP2 threat_rank |
|---|---:|---:|
| Nilai unik | 80 | 34.333 |
| Range teramati | 22,6469–157,9725 | 0,0428–100 |
| Mean / median | 105,2715 / 110,5078 | 50,0326 / 50,0282 |
| Q25 / Q75 | 75,9658 / 134,6432 | 25,0476 / 74,9819 |
| P90 / P99 | 154,0178 / 154,0178 | 90,0122 / 99,0291 |
| Modus terbesar | 154,0178: 7.304 baris (14,43%) | 22,6552: 73 baris (0,14%) |
| Top-10 nilai / seluruh baris | 79,10% | 1,08% |

TP1 disebut indeks dampak berskala kontinu dalam materi, tetapi **support train teramati hanya 80 nilai berulang**. TP2 adalah target numerik berbatas dengan support jauh lebih rapat dan marginal hampir uniform; jangan menyebutnya matematika kontinu murni, sebab ada pembulatan dan ties. Target TP2 tidak missing, tidak memiliki IQR outlier global, hanya 149/50.608 nilai integer-like pada toleransi absolut 1e-8, 32 baris tepat 100 dan tidak ada tepat 0. Batas 0–100 di sini masih observasi data, bukan definisi resmi yang diasumsikan.

### Geometri percentile per-cycle

Untuk 50.608/50.608 baris, tepat pada empat desimal:

`threat_rank == round(100 × average_rank(threat_rank di dalam cycle) / n_cycle, 4)`.

Versi pooled seluruh train hanya cocok pada 25 baris. Ini **diagnostik target terhadap target itu sendiri**, bukan feature, predictor, atau metode mengisi target test. Hasilnya konsisten dengan skala percentile relatif per-cycle; tidak membuktikan generator upstream atau skor laten mana yang diurutkan. Ties menggunakan average rank. Distribusi marginal yang hampir sama antarcycle dapat terkait konstruksi skala, sehingga tidak cukup untuk menyimpulkan risiko dasar atau hubungan fitur stabil. Jangan merangking ulang subset validation/test sebagai transformasi target atau mengambil label test dari sample.

## 3. Apa yang sebenarnya dipelajari dari REG-Lite B dan Phase16 B2

REG-Lite B adalah alternatif akademik sederhana pada delivery TP1 9 Oktober; Phase16 B2 adalah cabang recipe berbeda yang sudah dibahas dalam laporan phase. Keduanya tidak diperlakukan sebagai urutan kalender B → B2. Sumber dan hash report ada di `tp1_regression_history.json`; model tidak dijalankan ulang.

**REG-Lite B.** Recipe menyatakan 300-tree/31-leaf LightGBM multiclass, support skor fit-local (min_count=20 pada frozen 84-score axis), TF-IDF 5.000 unigram dan struktur; hasil probability distribution didekode sebagai lower conditional median. Median kondisional selaras dengan minimum expected absolute loss/MAE, sementara distribusi mempertahankan ambiguitas antarskor. Structured Ridge alpha=10 dipakai saat stream kosong. Dukungan empirisnya relevan untuk target TP1 yang terkonsentrasi; tidak otomatis untuk 34.333 label unik TP2.

Saved H12_64 memakai train cycle 47–63 dan validation 64–75. Pooled MAE B=15,717529, Phase14=16,160261, Phase5=17,123116. Suitability akademik **CONDITIONALLY SUITABLE**, tetapi research promotion **FAIL** dan signed-bias gate S6 tetap gagal. Ini reused development, bukan untouched holdout.

| Tercile target H12 (batas dari fit) | n | MAE B | Mean signed error (pred−truth) | Overpredict | Underpredict |
|---|---:|---:|---:|---:|---:|
| Lower | 5.560 | 22,6524 | +22,3716 | 67,30% | 1,26% |
| Middle | 6.700 | 11,9845 | +6,4631 | 35,13% | 16,15% |
| Upper | 7.230 | 13,8439 | −9,6144 | 10,46% | 40,54% |

Over/under adalah tanda error pada fungsi cohort notebook historis; sisanya exact equality. Lower target cenderung diprediksi terlalu tinggi dan upper terlalu rendah dalam **prediksi tersimpan TP1**. Ini konsisten dengan tarikan ke pusat kondisional, bukan bukti mekanisme tunggal. Lower/upper **tercile** ini bukan definisi 5% tail TP2 dan tidak boleh disamakan.

**Phase16 B2.** Soft gate zero-fit mengubah probability Phase15 memakai probability entry_friction Phase14 serta rate EF per score yang dipelajari dari fit, lalu mengambil median; missing stream tetap fallback Phase14 regression. Laporan mengaitkan gain dengan berkurangnya regime confusion. Mean tujuh window B2=14,6193 vs Phase15-all=14,7213, tetapi window development digunakan ulang dan ada regresi subgroup natural missing-org. Ini memberi alasan memeriksa heterogenitas regime dalam EDA, bukan izin melatih gate TP2 atau klaim bahwa error TP1 sudah berulang pada TP2.

## 4. Hubungan impact_load, threat_rank dan cycle_code

Semua 50.608 pasangan lengkap: Pearson **0,6370**, Spearman **0,6211**. Korelasi positif tidak menjamin fungsi monotonik atau target deterministik. Dari 78 nilai impact berulang, 77 memiliki lebih dari satu target.

Periodisasi utama memecah **33 cycle unik terurut** menjadi tiga blok 11 cycle, tanpa melihat target; bukan tiga quantile baris yang dapat memotong cycle.

| Cycle | n | Mean threat | SD threat | Pearson | Spearman |
|---|---:|---:|---:|---:|---:|
| 47–57 | 13.546 | 50,0406 | 28,8683 | 0,4994 | 0,4773 |
| 58–68 | 16.736 | 50,0329 | 28,8679 | 0,6722 | 0,6631 |
| 69–79 | 20.326 | 50,0271 | 28,8679 | 0,6964 | 0,6798 |

Pearson per-cycle berkisar 0,447–0,756; kenaikan antarblok tidak berarti setiap cycle naik monotonik. Mean/SD target hampir tidak berubah, tetapi hubungan kondisionalnya tidak identik. Perubahan komposisi impact, EF dan karakteristik lain masih dapat menjelaskan sebagian perbedaan; belum ada klaim causal drift atau signifikansi statistik.

Contoh dengan **impact persis sama**, bukan sekadar bin lebar:

- Impact 154,0178: n=7.304; threat P10–P90=63,7722–98,2861. Median per blok 83,4860 → 86,2561 → 89,9640. Dampak teknis yang sama dapat memiliki distribusi peringkat berbeda antarperiode.
- Impact 110,5078: n=5.984, median threat=71,2865; impact yang lebih besar 134,6432: n=5.192, median threat justru 58,9862. Pembalikan median terlihat di ketiga blok. Namun 5.827/5.984 baris pada impact 110,5078 memiliki EF=1, sedangkan impact 134,6432 semuanya EF=0: jangan menafsirkannya sebagai efek impact yang sudah dikontrol.
- Impact bin 107,23–124,14: n=11.824; P10–P90 threat 25,16–88,57. Hubungan keseluruhan kuat tetap menyisakan penyebaran besar.
- Tail deskriptif tetap: threat≤5 sebanyak 2.527 (4,99%), threat≥95 sebanyak 2.560 (5,06%). Angka ini proporsi target train, **bukan rate error model**, dan ambang tidak dioptimalkan.

## 5. Lima kandidat pertanyaan EDA Samuel

| Kandidat | Pertanyaan | Bukti dan penilaian |
|---|---|---|
| **1 — rekomendasi utama** | **Bagaimana distribusi threat_rank pada impact_load yang sama berubah dari cycle awal ke akhir, ketika distribusi marginal threat_rank tampak stabil?** | r antarblok 0,499→0,672→0,696; mean/SD target hampir tetap; median pada impact154,0178 bergeser 83,49→89,96. Menggabungkan temuan aktual, temporal dan conditional spread; pertanyaan lama diperluas, belum dikunci. |
| 2 — alternatif kuat | Apakah urutan impact_load selalu sejalan dengan urutan threat_rank, atau ada pembalikan pada tingkat impact yang sering muncul? | Median impact110,5078 vs134,6432 adalah 71,29 vs58,99, dengan n besar dan pola di tiga blok. Periksa common support EF/cycle sebelum mengaitkan pembalikan dengan impact itu sendiri. |
| 3 — kelanjutan tema ekstrem TP1 | Seberapa lebar rentang threat_rank dan peluang masuk tail rendah/tinggi pada impact_load yang sama, serta apakah polanya berubah antarcycle? | 77/78 impact berulang multi-target; P10–P90 pada impact154,0178 mencakup 34,51 poin. Melanjutkan lensa ambiguitas/ekstrem tanpa mengklaim over/underprediction TP2. |
| 4 — pendamping Phase16 | Apakah entry_friction membedakan threat_rank pada impact_load dan periode yang sebanding? | Median marginal EF0=35,52 (n34.706), EF1=78,79 (n15.902); tetapi beberapa impact tidak punya kedua EF. Prioritas lebih rendah karena common-support/komposisi perlu diperiksa, dan harus tetap regression Samuel, bukan EDA classification Rousan. |
| 5 — audit skala pendamping | Seberapa konsisten threat_rank berperilaku sebagai percentile relatif per-cycle, dan apa batas perbandingan angka target antarperiode? | Exact average-rank geometry 50.608/50.608 vs pooled25. Penting untuk interpretasi; lebih cocok subbagian metode/sumber daripada satu-satunya pertanyaan EDA karena ada unsur pemeriksaan target terhadap dirinya sendiri. |

Kandidat 1 paling kaya bukti tanpa menuntut model dan tetap dekat dengan scope Samuel. Kandidat 3 menjadi alternatif jika fokus diinginkan pada kelanjutan tema tail TP1; kandidat 2 dapat menjadi subpertanyaan untuk bentuk nonmonotonik. Pilihan final memerlukan keputusan Samuel/kelompok setelah meninjau bukti.

## 6. Arah lanjutan yang tepat

**Ya untuk melanjutkan prinsip EDA REG-Lite B; belum untuk melanjutkan recipe atau error claim.** Pertahankan fokus pada support target, median dan spread kondisional, tails serta segmentasi periode. TP2 menambah fenomena baru: label padat hampir uniform dengan geometri rank per-cycle dan hubungan impact yang berubah walaupun marginal tetap.

Langkah discovery berikutnya: gunakan frequent exact-impact levels yang hadir cukup banyak pada semua blok (laporkan n dan common support), bandingkan CDF/kuantil kondisional, lalu pisahkan perubahan komposisi EF dari perubahan di dalam kelompok yang benar-benar sebanding. Hindari rata-rata median sebagai distribusi gabungan. Konfirmasi definisi resmi threat_rank dan timing fitur sebelum membekukan validasi. Satu cycle sebaiknya tetap utuh dalam split temporal; jangan merangking target ulang perfold. Model, tuning, correction ekstrem, dan Kaggle submission tetap belum dimulai.

## Reproduksi dan batas bukti

Audit: `python scripts/audit_data.py --tp1-data-dir <raw-TP1-root>`. Diagnostik awal: `python scripts/samuel_eda.py`. Discovery: `python scripts/discover_regression.py --tp1-data-dir <raw-TP1-root>`. Notebook: `02_regression_discovery.ipynb`; JSON agregat: `discovery_summary.json`. Root TP1 berisi classification/regression dan hanya dibaca. Jika TP1 tidak tersedia, discovery TP2 tetap berjalan dengan perbandingan TP1 berstatus missing.

Semua rekomendasi berasal dari train yang sudah dieksplorasi; bukan hipotesis confirmatory praregistrasi. Tidak ada holdout TP2 yang diakses labelnya atau hasil model baru. Referensi TP1 bersifat laporan historis dengan hash/HEAD, bukan klaim replay model pada sesi ini.
