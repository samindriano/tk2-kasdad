# Raw-data EDA: ketersediaan `signal_stream` dan komposisi label

**Pertanyaan:** apakah komposisi `artifact_signal` berbeda saat `signal_stream` tersedia dibanding kosong, dan apakah selisih mentah bertahan setelah menyamakan distribusi `cycle_code`? Ini analisis deskriptif dari official Classification Train.

Train memiliki 50,608 baris unik (SHA-256 `875552a68a3d24b0f3e1966d6ddb5671e6f54305a1c526e2f0a76747942ed91c`); aturan availability sama dengan wrapper frozen: null, kosong, atau whitespace-only dianggap missing. Terdapat 3,404 baris missing (6.73%) dan 47,204 baris present.

Raw prevalence artifact adalah **38.13%** pada stream missing (n=3,404; Wilson 95% 36.51–39.78%) dan **34.47%** pada stream present (n=47,204; Wilson 95% 34.04–34.90%). Gap marginalnya **+3.66 poin persentase**.

### Periode temporal
- Cycle 47–57: present 30.78% (n=12,537); missing 28.94% (n=1,009); gap -1.84 pp.
- Cycle 58–68: present 40.37% (n=15,664); missing 50.37% (n=1,072); gap +10.01 pp.
- Cycle 69–79: present 32.05% (n=19,003); missing 35.22% (n=1,323); gap +3.18 pp.

### Standardisasi menurut exact cycle

Menggunakan cycle dengan ≥20 baris pada masing-masing status stream, terdapat 33/33 cycle yang memenuhi dukungan. Common cycle set mencakup 50,608/50,608 baris (100.00% overall), termasuk 3,404/3,404 missing (100.00%) dan 47,204/47,204 present (100.00%). Bobot setiap cycle ialah proporsi seluruh baris pada common cycle set di cycle itu; jadi kedua status dibandingkan pada distribusi cycle yang sama.

Setelah standardisasi cycle, prevalence menjadi 34.47% present dan 38.43% missing; gap **+3.96 pp**. Arah gap positif pada 21, negatif pada 12, dan nol pada 0 cycle yang didukung (rentang -17.04 hingga +23.73 pp). Cycle mix saja tidak menghilangkan gap marginal, tetapi ini belum memperhitungkan volume teks.

### Sensitivitas terhadap volume teks

Selanjutnya text_load dibagi ke kuantil global 5 dan 10; batas bin dihitung dari seluruh Train tanpa menggunakan target atau availability. Bobot dalam setiap standardisasi sama bagi kedua status: ukuran cell (n_present+n_missing) dibagi total baris pada semua cell yang didukung. Cell harus memiliki minimal 20 baris missing dan 20 present.

- Cycle saja, exact cycle 33/33: 50,608/50,608 baris (100.00%; missing 3,404/3,404, present 47,204/47,204), gap **+3.96 pp**.
- Volume saja, 5 kuantil: 5/5 bins; 50,608/50,608 (100.00%; missing 3,404/3,404, present 47,204/47,204), gap **-3.27 pp**. Dengan 10 kuantil: 10/10 bins; full coverage, gap **-2.26 pp**.
- Periode×volume, 3×5 bins: 15/15 cells; 50,608/50,608 (100.00%; missing 3,404/3,404, present 47,204/47,204), gap **-2.30 pp**.
- Periode×volume, 3×10 bins: 28/30 cells; 47,740/50,608 (94.33%; missing 3,372/3,404 (99.06%), present 44,368/47,204 (93.99%)), gap **-2.15 pp**. Dua cells tidak memenuhi dukungan karena masing-masing hanya berisi 17 dan 15 laporan missing; kedua cell dan denominatornya tercatat di stats.json.

Gap berubah arah: positif setelah penyamaan cycle, tetapi negatif pada volume-only serta period×volume yang didukung. Rentang volume present 20.6456–98.0650; missing 16.3429–77.5241; intersection 20.6456–77.5241. Di luar intersection terdapat 906 missing dan 65 present rows. Ini menunjukkan asosiasi raw peka terhadap komposisi volume dan definisi support; tidak membuktikan text_load sebagai confounder kausal.

### Apa artinya untuk modeling

Raw EDA memberi tahu bahwa status ketersediaan stream berkaitan dengan komposisi label dan berubah menurut waktu; ini belum memberi tahu apakah stream kosong menyebabkan label atau merupakan sinyal proses pengumpulan. Pada diagnostics model yang dibuat kemudian, C2_B hybrid memiliki missing-stream Macro F1 0.710427, sedangkan C1_B structured HGB mencapai 0.781712. C4_HGB merutekan missing rows ke HGB dan menaikkan aggregate primary 0.820633→0.825481 serta H12 0.788581→0.796810. Angka model tersebut adalah bukti retrospektif terpisah, bukan hasil raw EDA ini atau gain baru dari pertanyaan EDA; routing sudah menjadi bagian model final.

text_load berbeda secara deskriptif (lihat median dan kuartil per status di stats.json). Kontras standardisasi adalah deskriptif tanpa CI adjusted; Wilson intervals di atas merangkum setiap rate mentah secara terpisah. Seluruh cycle sudah berulang kali digunakan dalam pengembangan, jadi perbandingan temporal bukan konfirmasi independen.

**Nilai EDA untuk tugas:** ini dapat menjadi cerita EDA-ke-keputusan modeling yang konkret, asalkan dituturkan jujur sebagai profil ketersediaan dan heterogenitas yang kemudian memotivasi routing. Kelemahannya, hipotesis menjadi lebih kuat setelah diagnostics model; jangan menyajikannya seolah discovery pra-modeling atau mengulang skor sebagai analisis EDA. `signal_stream` missing juga proses availability, bukan teks ber-TTR rendah.

Grafik utama: `prevalence_by_availability.png` (rates dan Wilson intervals menurut keseluruhan/periode) dan `cycle_gaps.png` (arah tiap cycle; ukuran titik mengikuti denominator missing). Machine-readable detail, schema, raw hashes, seluruh denominator, dan batas interpretasi ada di `stats.json`.
