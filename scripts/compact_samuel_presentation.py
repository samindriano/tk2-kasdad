"""Concise academic presentation; no model/metric/plot changes."""
import re
import nbformat

EDA = {
0: "## 2. EDA individual: teks kosong dan terisi\n\n**Apakah persentase laporan berlabel `artifact_signal = 1` berbeda antara `signal_stream` kosong dan terisi, setelah mempertimbangkan waktu dan volume laporan?**",
1: "### EDA 1. Data dan metode\n\nTeks **kosong** berarti null, string kosong, atau hanya spasi. Waktu memakai `cycle_code`; volume memakai `text_load`, bukan jumlah token.\n\nPerbandingan disesuaikan dengan memberi bobot yang sama pada kelompok waktu/volume untuk kedua status. Bobot mengikuti jumlah baris pada kelompok yang didukung. Minimum **20 baris per status per kelompok**. Rentang volume memakai kuantil tanpa label; ini penyesuaian kasar, bukan pencocokan nilai persis.",
3: "### EDA 2. Gambaran data\n\nJumlah laporan, label positif, dan volume menurut ketersediaan teks.",
5: "### EDA 3. Perbandingan awal\n\nBandingkan persentase label 1 pada seluruh laporan. Error bar adalah interval Wilson 95% masing-masing kelompok, bukan interval selisihnya.",
7: "### EDA 4. Perubahan sepanjang waktu\n\nSelisih kosong dikurangi terisi tidak positif pada semua cycle maupun periode.",
9: "### EDA 5. Perbedaan volume\n\nHistogram dinormalisasi agar ukuran kelompok yang berbeda tetap dapat dibandingkan. Distribusi volume kedua kelompok berbeda.",
11: "### EDA 6. Menyamakan komposisi\n\nBandingkan hasil awal dengan penyesuaian setiap cycle serta tiga periode x 5/10 kelompok volume. **Gap = persentase kosong - terisi**, dalam poin persentase (pp). Coverage menunjukkan bagian data yang memenuhi minimum support.",
14: "### EDA 7. Interpretasi dan batasan",
16: "### EDA 8. Kesimpulan\n\nPersentase label 1 pada teks kosong lebih tinggi secara keseluruhan (**38,13% vs 34,47%**). Setelah waktu dan volume diperhitungkan, selisih berbalik menjadi sekitar **-2,2 sampai -2,3 pp**. Jadi, perbandingan dipengaruhi komposisi laporan.\n\nIni hubungan deskriptif, bukan sebab-akibat. Kelompok volume masih kasar; pemeriksaan 10 kelompok mencakup **94,33%** baris. Pola official Test belum diketahui."
}
CLS = {
0: "# TP2 Classification - Samuel Indriano\n\n**A06 | KasdudLovers | 2406400524**\n\nTarget: `artifact_signal` (0/1). Metrik: **Macro F1**. Approach individual: **C4_HGB**, hybrid teks+struktur saat teks terisi dan HGB saat kosong, threshold **0,50**.\n\nNotebook memuat EDA dan reproduksi evaluasi cycle **47-75 -> 76-79**. Ini approach individual akademis, bukan ensemble champion kelompok terbaru.",
2: "## 1. Masalah dan data\n\nLabel 1 berarti indikasi artefak eksploit publik teramati; label 0 tidak memastikan artefak tidak ada. Train mencakup cycle 47-79, Test 80-91. Validasi mengikuti waktu dan seluruh preprocessing dipelajari dari data fit.\n\nMacro F1 memberi bobot sama pada F1 kedua kelas. Recipe TP1 dievaluasi ulang karena target TP2 berbeda.",
4: "### Label per cycle\n\nPerubahan proporsi label mendukung penggunaan validasi temporal.",
23: "## 3. Perkembangan pendekatan\n\n1. Majority dan model sederhana sebagai baseline.\n2. LR/HGB dengan fitur terstruktur.\n3. TF-IDF dan hybrid C2_B untuk menambah informasi teks.\n4. C4_HGB mengatasi kelemahan hybrid saat teks kosong melalui routing.\n\nTabel berikut berasal dari eksperimen historis terverifikasi. Primary menggabungkan enam window; H12 menguji horizon lebih panjang dengan sebagian baris yang sama, sehingga bukan bukti independen. TF-IDF memakai unigram-bigram, maksimal 20.000 fitur, `min_df=3`, dan sublinear TF.",
25: "**Pemilihan model.** C4_HGB mencapai Macro F1 **0,8255 primary / 0,7968 H12**, dibanding C2_B **0,8206 / 0,7886**. Threshold inner dan perubahan bobot fitur tidak memberi perbaikan konsisten.\n\n## 4. Reproduksi temporal\n\nFit pada cycle 47-75, evaluasi 76-79. C2_B adalah hybrid Logistic Regression; C1_B adalah structured HGB. Threshold tetap 0,50, tidak dipilih dari label evaluasi.",
27: "### Pemeriksaan reproduksi\n\nImputasi, encoding, scaling, dan TF-IDF dipelajari di dalam fit. Hash prediksi dibandingkan dengan receipt historis W6 tanpa cache model pribadi.",
29: "### Hasil W6\n\nConfusion matrix: baris aktual, kolom prediksi. Periksa precision dan recall bersama Macro F1.",
31: "C4_HGB mempertahankan hybrid pada teks terisi dan memakai HGB pada teks kosong. Perbandingan menggunakan baris evaluasi yang sama.",
33: "Kurva threshold hanya diagnostik; keputusan tetap **0,50**.\n\n## 5. Recipe individual dan Kaggle\n\n- Teks kosong: C1_B structured HGB.\n- Teks terisi: C2_B structured + TF-IDF LR.\n- Routing hanya memakai ketersediaan fitur; ID dan target tidak menjadi predictor.\n\nSkor public di bawah berasal dari manifest submission historis, terpisah dari validasi temporal.",
36: "### Refit penuh (opsional)\n\nDimatikan secara default. Jika diaktifkan, pipeline dilatih pada seluruh Train dan menulis CSV lokal; tidak mengunggah ke Kaggle.",
38: "## 6. Kesimpulan\n\nTeks menambah informasi prediktif, tetapi hybrid lebih lemah ketika teks kosong. Routing C4_HGB memberi **0,8255 primary / 0,7968 H12**; public benchmark approach ini **0,82641**.\n\nEDA memberi konteks komposisi availability, bukan bukti bahwa missingness menyebabkan label atau bahwa EDA menyebabkan kenaikan skor. Window development sudah digunakan berulang; H12 bukan holdout independen dan public score tidak menjamin private score.\n\nPetunjuk data, dependencies, dan bukti eksekusi: [README](README.md)."
}
REG = {
0: "# TP2 Regression - Samuel Indriano\n\n**A06 | KasdudLovers | 2406400524**\n\nTarget: `threat_rank` (0-100). Metrik: **MAE**. Approach individual: **LightGBM L1 + structured + TF-IDF/SVD32**, dengan rank blend yang dipilih melalui inner validation.\n\nNotebook memuat EDA dan reproduksi evaluasi temporal. Ini approach individual akademis, bukan ensemble champion kelompok terbaru.",
1: "## 1. Data dan lingkungan\n\nGunakan official Train Classification untuk EDA dan Train Regression untuk modeling. Dependencies dan lokasi data dijelaskan di [README](README.md). Seluruh preprocessing dipelajari dari fold fit; seed 42 dan dua thread.",
4: "`impact_load` dan `entry_friction` menjadi fitur resmi TP2; label/model TP1 tidak digabungkan. ID dan target lain dikecualikan. Format campuran dan nilai kosong ditangani di pipeline.",
22: "## 3. Perkembangan pendekatan\n\nMAE mengukur rata-rata selisih absolut; median menjadi baseline yang sesuai. TP1 memiliki target diskret yang terkonsentrasi, sedangkan TP2 memiliki **34.333 nilai unik**. Karena itu, multiclass per nilai target TP1 tidak disalin.\n\nTabel berikut adalah hasil historis enam window temporal. Pooled MAE menimbang baris, equal-cycle menimbang cycle. H12 memakai ulang baris primary dan bukan replikasi independen.",
24: "**Alur keputusan:** median -> HGB -> expanded structured -> tambahan teks -> LightGBM+SVD32. Recency, penghapusan fitur, dan percentile mapping murni tidak konsisten membantu.\n\nStructured Ridge **14,17865/15,20582** menjadi hybrid Ridge **12,55131/13,93812** pada primary/H12. Teks membantu dalam perbandingan ini; pilihan estimator tetap memengaruhi hasil.",
26: "## 4. Pipeline dan validasi\n\n1. Parser numerik; imputasi median dan encoding kategori dari fit fold. Unknown category -1 diperlakukan LightGBM sebagai missing.\n2. TF-IDF unigram-bigram, maksimal 20.000 fitur, `min_df=2`; SVD32 (`n_iter=7`, seed 42).\n3. LightGBM `regression_l1`, 400 trees, learning rate 0,05; dua konfigurasi frozen: base (31 leaves/minleaf40/L2=1), regularized (15/80/3).\n4. Inner MAE memilih konfigurasi dan alpha dari {0; 0,25; 0,50; 0,75; 1}.\n\nRank blend: `clip((1-alpha)*raw + alpha*100*average_rank(raw)/n_cycle, 0, 100)`. Rank berasal dari prediksi, bukan label; baris dalam satu cycle harus diprediksi bersama. Mapping mengubah jarak, bukan urutan prediksi.\n\n**Inner:** 47-71 -> 72-75. **Outer:** refit 47-75 -> evaluasi 76-79. Semua preprocessing dibangun ulang per fit.",
29: "## 5. Evaluasi dan keterbatasan\n\nW6 dihitung ulang di atas; angka primary/H12 berasal dari eksperimen historis. LightGBM+text memberi **MAE 11,93160 primary / 14,70404 H12**. Namun **hybrid Ridge lebih baik pada H12: 13,93812**. Model terpilih tidak mendominasi semua horizon.\n\nSeluruh window sudah digunakan dalam development; reproduksi W6 bukan holdout baru.",
31: "Target rendah masih cenderung overpredicted dan target tinggi underpredicted. Bias global dapat menyembunyikan error tersebut. Target subgroup hanya dipakai untuk evaluasi.\n\nSebagian Test memiliki nilai count nol tanpa analog Train; maknanya belum diketahui. Tanpa label Test, risikonya belum dapat diukur. Banyaknya eksperimen juga membatasi kekuatan kesimpulan generalisasi.",
32: "## 6. Refit penuh (opsional)\n\nDimatikan secara default. Recipe deployment historis memakai base dan alpha 0,50, lalu refit seluruh Train. Jika diaktifkan, sel menulis CSV lokal tanpa upload Kaggle; manifest historis tidak ditimpa.",
34: "## 7. Kesimpulan\n\nLightGBM+SVD32 dipilih berdasarkan primary MAE dan analisis error. Teks melengkapi struktur; hybrid Ridge tetap pembanding penting karena lebih baik pada H12. Public benchmark approach ini **11,71363**, terpisah dari validasi internal.\n\nEDA individual membahas `artifact_signal`, bukan `threat_rank`. Bukti manfaat teks untuk Regression berasal dari eksperimen model yang terpisah.\n\nPetunjuk data, dependencies, dan bukti eksekusi: [README](README.md)."
}

def compact(nb, task):
    nb.cells = [c for c in nb.cells if not c.id.startswith("samuel-format-")]
    for index, source in (CLS if task == "Classification" else REG).items():
        assert nb.cells[index].cell_type == "markdown"
        nb.cells[index].source = source
    for cell in nb.cells:
        if cell.id.startswith("samuel-approved-eda-"):
            index = int(cell.id.rsplit("-", 1)[1])
            if cell.cell_type == "markdown":
                cell.source = EDA[index]
                if index == 0:
                    cell.source += "\n\nSumber: official Classification Train. Analisis ini sama pada kedua notebook; tidak memasukkan label Classification ke model Regression." if task == "Regression" else "\n\nSumber: official Classification Train."
            elif index == 15:
                cell.source = """display(Markdown(
    f"**Hasil:** gap awal {eda_marginal_gap_pp:+.2f} pp; setelah penyesuaian cycle "
    f"{eda_cycle_standardized['standardized_missing_minus_present_pp']:+.2f} pp; "
    f"periode x volume: {eda_period_q5['standardized_missing_minus_present_pp']:+.2f} pp (5 kelompok), "
    f"{eda_period_q10['standardized_missing_minus_present_pp']:+.2f} pp (10 kelompok).\\n\\n"
    f"**Dukungan:** minimum 20 baris per status/kelompok. Coverage 10 kelompok "
    f"{eda_period_q10['supported_rows_total']:,}/{len(eda_analysis):,} "
    f"({eda_period_q10['coverage_all_rows']:.2%}); "
    f"{eda_period_q10['cells_supported']}/{eda_period_q10['cells_total']} kelompok didukung.\\n\\n"
    "**Batasan:** kelompok volume masih kasar; faktor lain belum dikontrol. "
    "Hasil bukan sebab-akibat atau konfirmasi independen. Routing berasal dari diagnosis model terdahulu, "
    "bukan ditemukan secara prospektif melalui EDA ini."
))"""
    # Match the TP1 opening: task, individual EDA, short approach, identity.
    title = "Klasifikasi `artifact_signal`" if task == "Classification" else "Regresi `threat_rank`"
    approach = ("C4_HGB memakai hybrid structured + TF-IDF Logistic Regression saat teks terisi, dan structured HGB saat kosong. Metrik utama: **Macro F1**." if task == "Classification" else "LightGBM L1 memakai fitur structured dan TF-IDF/SVD32, dengan rank blend yang dipilih melalui inner validation. Metrik utama: **MAE**.")
    nb.cells[0].source = (f"# TP2 KASDAD - {title}\n\n"
        "**EDA individual:** Apakah persentase laporan berlabel `artifact_signal = 1` berbeda antara `signal_stream` kosong dan terisi, setelah mempertimbangkan waktu dan volume laporan?\n\n"
        + approach + "\n\n**Identitas:** Kelompok **A06 - KasdudLovers** | **Samuel Indriano** | NPM **2406400524**.\n\n"
        "Notebook mencakup data, EDA, preprocessing, pemodelan, dan evaluasi temporal. Approach individual ini berbeda dari ensemble champion kelompok terbaru.")
    # Add short readings directly after the preceding computed table/figure.
    insights = {
        5: "**Insight data.** Teks kosong hanya mencakup 6,73% laporan. Ukuran kelompok yang jauh berbeda perlu diperhatikan saat membaca persentase dan ketidakpastiannya.",
        7: "**Insight perbandingan awal.** Label 1 lebih sering muncul pada kelompok kosong, dengan selisih +3,66 pp. Ini belum memperhitungkan perbedaan waktu dan volume laporan.",
        9: "**Insight waktu.** Gap positif pada 21 dari 33 cycle, tetapi negatif pada 12 lainnya. Periode tengah menunjukkan perbedaan terbesar; satu angka agregat menyembunyikan variasi ini.",
        11: "**Insight volume.** Median `text_load` kelompok kosong 21,81, sedangkan kelompok terisi 31,78. Komposisi volume yang berbeda menjadi alasan untuk membandingkan kelompok waktu/volume yang lebih setara.",
        14: "**Insight penyesuaian.** Menyamakan cycle saja tidak membalik arah gap, sedangkan penyesuaian periode dan volume membalikkannya. Hasil 5 dan 10 kelompok memiliki arah serupa, dengan coverage lebih rendah pada 10 kelompok."
    }
    for c in nb.cells:
        if c.id.startswith("samuel-approved-eda-"):
            i = int(c.id.rsplit("-",1)[1])
            if i in insights: c.source = insights[i] + "\n\n" + c.source
            if i == 0: c.source = c.source.replace("## 2.", "## 3.")
    if task == "Classification":
        nb.cells[2].source = nb.cells[2].source.replace("## 1. Masalah dan data", "## 2. Dataset dan target")
        nb.cells[23].source = nb.cells[23].source.replace("## 3.", "## 4.")
        nb.cells[25].source = nb.cells[25].source.replace("## 4. Reproduksi temporal", "## 5. Pipeline dan validasi temporal")
        nb.cells[25].source = nb.cells[25].source.replace("**Pemilihan model.**", "**Insight pemilihan model.**")
        nb.cells[29].source = nb.cells[29].source.replace("### Hasil W6", "## 6. Evaluasi model")
        nb.cells[31].source = "**Insight W6.** Routing mengurangi false positive dari **608 menjadi 522**, tetapi true positive juga turun dari **1.758 menjadi 1.725** dibanding hybrid C2_B. Jadi, peningkatan keputusan memiliki trade-off: lebih sedikit alarm salah dengan sedikit kehilangan deteksi positif."
        nb.cells[33].source = nb.cells[33].source.replace("## 5. Recipe individual dan Kaggle", "### Recipe individual dan Kaggle")
        nb.cells[36].source = nb.cells[36].source.replace("### Refit penuh (opsional)", "## 7. Refit penuh (opsional)")
        nb.cells[38].source = nb.cells[38].source.replace("## 6.", "## 8.")
        prep = nbformat.v4.new_markdown_cell("## 1. Persiapan\n\nGunakan dependencies dan lokasi data di [README](README.md), lalu Restart & Run All. Refit penuh tetap dimatikan secara default.")
        prep.id = "samuel-format-preparation"; nb.cells.insert(1,prep)
    else:
        nb.cells[1].source = nb.cells[1].source.replace("## 1. Data dan lingkungan", "## 1. Persiapan")
        nb.cells[22].source = nb.cells[22].source.replace("## 3.", "## 4.")
        nb.cells[24].source += "\n\n**Insight model.** Tambahan teks memperbaiki Ridge dalam keluarga model yang sama. Ini evidence prediktif yang terpisah dari EDA availability; EDA tersebut memakai target Classification."
        nb.cells[26].source = nb.cells[26].source.replace("## 4.", "## 5.")
        nb.cells[29].source = nb.cells[29].source.replace("## 5. Evaluasi dan keterbatasan", "## 6. Evaluasi model")
        nb.cells[29].source += "\n\n**Insight pemilihan.** Hasil primary yang lebih baik belum berarti unggul pada semua horizon. Keunggulan Ridge di H12 menjadi alasan untuk tetap menampilkannya sebagai pembanding."
        nb.cells[31].source = "**Insight error.** Pada grafik bias, target rendah cenderung diprediksi terlalu tinggi dan target tinggi terlalu rendah. Prediksi mendekati nilai tengah dapat memberi MAE yang baik secara keseluruhan, tetapi masih kurang tepat pada target ekstrem.\n\n" + nb.cells[31].source.split("\n\n",1)[-1]
        nb.cells[32].source = nb.cells[32].source.replace("## 6.", "## 7.")
        nb.cells[34].source = nb.cells[34].source.replace("## 7.", "## 8.")
        data = nbformat.v4.new_markdown_cell("## 2. Dataset dan target\n\nModel memprediksi `threat_rank` pada skala 0-100. Ringkasan berikut memakai official Regression Train; bagian EDA kemudian memakai Classification Train secara terpisah.")
        data.id = "samuel-format-dataset"; nb.cells.insert(3,data)
    return nb
