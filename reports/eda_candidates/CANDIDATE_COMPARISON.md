# Perbandingan kandidat EDA Samuel

31 Oktober 2026 · **Rekomendasi untuk dipilih pengguna; belum mengganti pertanyaan resmi.** Semua angka berasal dari kedua Train resmi dan analisis eksploratif tersendiri; skor model/Kaggle tidak digunakan untuk memberi peringkat kandidat.

## Screening dan shortlist

| Kandidat | Kualitas insight | Kredibilitas dan support | Interpretability/grafik | Relevansi | Keputusan |
|---|---|---|---|---|---|
| A · keragaman token pada panjang sama | Menunjukkan asosiasi di luar banyaknya token: weighted artifact+14,20 pp | 71 exact lengths/37.681 support rows; 9.011 versus9.354 tail rows; 64/71 positif; thresholds dan ukuran terkait stabil | Definisi proporsi unik sederhana; satu grafik mean weighted dan semua kontras memperlihatkan counterexample | Konteks informasi teks Classification; tidak membuktikan penyebab gain model | **Rekomendasi pertama** |
| B · metadata volume versus panjang aktual | Asosiasi sangat kuat tetapi hampir sesuai definisi: Spearman0,983 | 47.204 present; hasil jelas | Scatter intuitif, kesimpulannya kurang substantif sebagai pertanyaan utama | Penting sebagai kontrol/diagnostik A dan C | Eliminasi sebagai pertanyaan utama |
| C · availability dan komposisi volume | Gap rank missing+8,12 mengecil dekat nol setelah menyamakan volume; contoh bahaya membaca mean marginal | 3.404 missing; cycle32/33 searah sebelum kontrol; weighted effect bin-sensitive;906 di luar present range | Dua grafik cukup untuk fenomena, tetapi bobot/common support memerlukan penjelasan lebih panjang | Konteks representasi informasi hilang, bukan model-error analysis | **Shortlist kedua** |
| D · `pattern_code` | Heterogenitas antar-code nyata dan relatif bertahan pada support cutoffs | Known codes coverage80,2/75,0/61,2% pada n30/100/300; η² cycle-centered bersifat deskriptif | Profil kategori sulit diterjemahkan menjadi cerita domain karena anonim; equal-code versus weighted berbeda | Konteks heterogenitas kedua task | Cadangan, bukan shortlist utama |
| E · `org_code`/`module_code` | Rentang raw group means terlihat besar, tetapi mudah menjadi cerita kelompok kecil | Median kategori n1; coverage known-code n30 hanya50,4%/23,6% | Banyak ID tidak mempunyai arti domain yang dapat diperiksa | Belum cukup untuk insight end-to-end yang jelas | Eliminasi karena sparsity/semantik |
| F · keselarasan dua outcome | Asosiasi positif disertai5.029 rank-tinggi tanpa indikasi artefak teramati | Semua50.608 verified pairs; cycle dan ambang sensitivity; no target cross-use in models | ECDF dan stacked bars paling intuitif; asosiasi dasar relatif diharapkan | Menjelaskan mengapa kedua task berbeda | **Shortlist ketiga** |

Tidak ada skor numerik gabungan yang menyamarkan penilaian editorial. A dipilih dari kombinasi fokus pertanyaan, effect size, keterbacaan, reasonable robustness, dan perbedaan topik; tidak semata-mata karena teks terdapat pada model yang disubmit. Semua arah memperoleh statistik dan visual awal. C dan D juga memperlihatkan penjelasan alternatif yang melemahkan klaim awal, bukan disembunyikan.

## Explicit non-overlap assessment

Pertanyaan anggota yang menjadi batas:

- **Rousan:** proporsi `artifact_signal` berdasarkan `entry_friction`, dengan hubungan pada berbagai `impact_load`.
- **Abhi:** informasi tambahan `scope_pressure` terhadap `threat_rank` setelah mempertimbangkan `impact_load`.
- **Rizky:** `telemetry_flag` terhadap `artifact_signal`, dengan heterogenitas `module_spread`.
- **Samuel lama:** perubahan hubungan `impact_load`–`threat_rank` antarperiode `cycle_code`, meskipun marginal relatif stabil.

| Kandidat | Terhadap Rousan | Terhadap Abhi | Terhadap Rizky | Terhadap EDA Samuel lama |
|---|---|---|---|---|
| A | Target artifact sama, tetapi fitur utama repetisi/keragaman token; EF/impact bukan fitur utama atau kontrol | Bukan scope_pressure atau incremental-scope analysis | Bukan telemetry/spread; tanpa keduanya sebagai stratifier | Bukan impact–threat; cycle hanya sensitivity, bukan inti pertanyaan |
| B | Metadata teks/observed length, bukan EF/impact proporsi artifact | Tidak meneliti scope_pressure | Tidak memakai telemetry/module_spread | Pertanyaan metadata, bukan temporal association |
| C | Fokus proses availability dan komposisi text_load, bukan EF/impact | Tidak meneliti scope_pressure atau kontrol impact | Telemetry sengaja dikeluarkan dari completeness; module_spread hanya indikator availability sekunder, bukan nilai atau interaksinya dengan telemetry; stream-only menjadi fokus shortlist | Bukan hubungan impact–threat antarcycle; cycle robustness saja tidak menjadikannya salinan EDA lama |
| D | Pattern categories sebagai fitur utama, tidak memakai pasangan EF/impact | Tidak meneliti scope_pressure/impact | Pattern code, bukan telemetry/spread | Profil code, bukan perubahan temporal impact–threat |
| E | Org/module identifiers, bukan EF/impact | Tidak meneliti scope_pressure/impact | `module_code` adalah identitas kategori, berbeda dari `module_spread`; tidak diuji bersama telemetry | Konteks organisasi/modul, bukan temporal impact–threat |
| F | Membandingkan **dua outcome**; tidak memakai EF/impact untuk menjelaskan artifact | Tidak memakai scope_pressure atau kontrol impact | Tidak memakai telemetry/module_spread | Outcome concordance, bukan conditional impact–threat sepanjang waktu |

**Putusan:** keenam arah berbeda substantif dari tiga EDA anggota lain dan EDA temporal Samuel lama. Eliminasi B/E bukan karena overlap, melainkan kualitas insight/support. Tidak ada pertanyaan yang “diselamatkan” dengan mengganti kata sambil mengambil pasangan variabel utama anggota lain. Kesamaan target saja tidak menjadikan pertanyaan sama, sesuai batas pengguna.

## Dibandingkan EDA lama Samuel

| Aspek keputusan | EDA lama | Kandidat A | Kandidat C | Kandidat F |
|---|---|---|---|---|
| Insight lebih kuat? | Sudah kuat: marginal stabil dengan perbedaan conditional/time | Menambahkan fenomena berbeda; tidak terbukti lebih kuat secara universal | Sangat menarik sebagai pembelajaran composition; tidak menetapkan mekanisme | Sangat jelas, tetapi lebih deskriptif |
| Mudah dijelaskan? | Memerlukan korelasi, periode, standardisasi EF/impact, rank geometry | Dua konsep utama: repetisi teks dan prevalence dengan panjang sama | Memerlukan missingness, weighting, common support, bin sensitivity | Paling mudah: dua label tidak sama |
| Grafik intuitif? | Informatif, tetapi beberapa panel diperlukan | Bar weighted+scatter seluruh panjang | Missingness-by-volume+ECDF | Stacked bars+ECDF |
| Kedalaman akademis | Banyak robustness checks yang telah diverifikasi | Exact count, ties, common weights, effect heterogeneity, support sensitivity | Demonstrasi confounding dan positivity secara eksplisit | Grain/join verification, subgroup/threshold sensitivity |
| Konsistensi | Tidak monotonik, tetapi pola early/late bertahan | 64/71 length strata; bukan seluruh panjang;30/33 cycle broad-bin | Pooled positif; weighted near-zero/bin-sensitive, sehingga klaim harus sempit | Positive mean gap33/33 cycle; discordance tetap besar |
| Cakupan | Semua Train; common impact×EF96,09% | Teks present93,27%; exact-length support74,46% semua Train; tail contrast36,29% semua Train | Dua grup semua Train, tetapi906 missing di luar present range; joint support13% | Seluruh Train untuk asosiasi dua outcome |
| Kaitan modeling | Risiko generalisasi Regression | Konteks fitur teks Classification, bukan ukuran nilai tambah model | Availability yang berbeda; bukan ulang analisis kesalahan model | Menjelaskan perbedaan task, bukan fitur silang |
| Biaya mengganti | Tidak ada, sudah selesai dan valid | Rumusan dan grafik baru perlu disetujui; riset pendukung sudah tersedia | Penjelasan demo lebih rumit | Paling ringan, tetapi mungkin terasa terlalu sederhana |

## Rekomendasi dan titik berhenti

**Pilih A sebagai calon pengganti terbaik**, dengan rumusan:

> Pada laporan dengan jumlah token yang sama, apakah keragaman token dalam `signal_stream` berkaitan dengan proporsi `artifact_signal` yang teramati?

Jawaban yang dapat dipertanggungjawabkan: pada71 panjang yang memenuhi support, weighted prevalence24,19% versus38,39%, gap14,20 pp; mayoritas panjang searah tetapi tujuh tidak. Ini asosiasi eksploratif pada data terpilih, bukan bukti semantik, kualitas laporan, kausalitas, independensi temporal, atau generalisasi Test.

**Worth it:** ya jika Samuel mengutamakan pertanyaan teks yang lebih fokus dan visual utama yang ringkas. Keuntungan penggantian bersifat fokus dan komunikasi, bukan klaim bahwa penelitian lama gagal atau model teks otomatis membenarkan EDA baru. Jika coverage luas dan keluasan robustness lebih diutamakan, mempertahankan EDA lama tetap layak. C/F tersedia sebagai alternatif dengan trade-off jelas.

**Tidak ada perubahan resmi otomatis.** Pembagian EDA, notebook formal03, seluruh checkpoint, model, dan Kaggle submissions tetap utuh. Final notebook preparation tetap berhenti. Samuel perlu memilih setelah membaca [temuan lengkap](DISCOVERY_FINDINGS.md) dan [notebook discovery](../../notebooks/samuel/eda_candidate_discovery.ipynb).
