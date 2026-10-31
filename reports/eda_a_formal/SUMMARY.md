# EDA individual resmi Samuel — keragaman token

31 Oktober 2026 · Dipilih pengguna setelah discovery dan targeted robustness. Ini EDA yang dapat dijalankan untuk review; penyajian paket final pengumpulan tetap tahap terpisah.

**Pertanyaan:** Pada laporan dengan jumlah token yang sama, apakah keragaman token dalam "signal_stream" berkaitan dengan proporsi "artifact_signal = 1" yang teramati?

Notebook: [11_samuel_formal_eda_a.ipynb](../../notebooks/samuel/11_samuel_formal_eda_a.ipynb). Status resmi dicatat pada [dokumen Samuel](../../docs/SAMUEL_EDA.md). EDA temporal lama, [notebook03](../../notebooks/samuel/03_samuel_formal_eda.ipynb), tetap utuh sebagai arsip penelitian.

## Ringkasan akademis

Official Classification Train mempunyai50.608 baris;47.204 stream tokenizable dan3.404 kosong/blank. TTR adalah jumlah token unik dibagi jumlah token total, memakai regex `[A-Za-z0-9]+` dan lowercase. Ia mengukur repetisi/keragaman token anonim, bukan makna atau kualitas teks. Target0 berarti indikasi artefak tidak teramati, bukan terbukti tidak ada.

Di setiap exact token count, average percentile rank TTR≤0,25 membentuk grup rendah dan≥0,75 grup tinggi; ties tidak dipisah. Syarat n≥120 per panjang dan≥20 tiap tail menghasilkan71 supported lengths/37.681 support rows. Kontras memakai9.011 rendah dan9.354 tinggi. Persentase tertimbang per panjang **24,19% versus38,39%**, gap **+14,20 pp**, dengan64/71 panjang berarah positif. Ukuran tail bukan penyebut langsung persentase tertimbang; kedua grup menggunakan bobot panjang yang sama.

Exact length×periode tetap positif: **+5,65/+15,19/+13,82 pp** pada cycle47–57/58–68/69–79; pooled supported-strata gap **+12,10 pp** pada31.622 support rows (67,0% stream tersedia). Exact length×exact cycle dengan minimal5 baris dan variasi TTR mencakup **43.333 baris (91,80% present)**, ringkasan linear **+7,99 pp per+0,10 TTR**, positif31/33 cycle. Koefisien linear, tail gap, bobot, dan support set berbeda; angka7,99 bukan versi lain dari14,20.

**Kesimpulan:** pada support Train yang dianalisis, keragaman token lebih tinggi berkaitan dengan proporsi artifact=1 yang lebih tinggi pada panjang yang sama. Hubungan tidak universal, tidak kausal, tidak menghilangkan semua confounding, dan belum menjamin official Test. Stream kosong dikecualikan. Evidence TF-IDF dari modeling Classification adalah evidence prediktif terpisah; TTR tidak dibuktikan menyebabkan gain Macro F1 atau menjadi fitur terbaik.

Empat visual: distribusi panjang/TTR; weighted prevalence bersama seluruh71 gap panjang; gap/coverage tiga periode; koefisien exact-cycle. Detail riset tetap pada [discovery](../eda_candidates/DISCOVERY_FINDINGS.md) dan [robustness](../eda_a_robustness/FINDINGS.md), tanpa diduplikasi ke notebook utama.

## Compliance singkat dan non-overlap

PDF resmi: `TP KASDAD Ganjil 2026_2027.pdf`,57 halaman, SHA-256 `386fa63541cc3e9b36b56ee8f8f6c479b3bbaa7753b9482d60763326f4465c0a`.

| Kewajiban tertulis | Evidence notebook |
|---|---|
| p.2: minimal satu pertanyaan EDA individual dari dataset | §1, pertanyaan resmi Samuel dan definisi TTR |
| p.2: eksplorasi, visualisasi sesuai, interpretasi/insight | §2–5, hasil aktual dan empat visual yang dihitung ulang |
| p.9: Restart & Run All, tanpa error, hasil reproducible, tanpa file sementara pribadi | Kode load official Train, fungsi reusable, dan [receipt eksekusi/angka](verification.json) |
| p.10: mampu menjelaskan approach saat demo | Alasan exact length, ties, weighting, estimand waktu dan keterbatasan dijelaskan; pemahaman lisan tetap perlu review Samuel |

Kesimpulan yang menjawab pertanyaan ada di§6. Checklist ini mencakup milestone EDA; ia tidak menyatakan seluruh tugas kelompok siap dikumpulkan. Dua pipeline individual Classification/Regression (pp.3–4), nama notebook task dan ZIP kelompok (pp.7–8) tetap bagian finalisasi pengumpulan yang terpisah.

- **Rousan:** EF/impact→artifact; A memakai TTR dan exact token length, tanpa EF/impact sebagai explanatory characteristic atau kontrol.
- **Abhi:** scope_pressure→threat; A tidak memakai scope_pressure atau target threat_rank.
- **Rizky:** telemetry×module_spread→artifact; A tidak memakai kedua fitur tersebut.

Dokumen pembagian bersama dan pertanyaan anggota lain tidak diedit. Halaman khusus Samuel mencatat keputusan baru, sementara catatan EDA temporal terdahulu dipertahankan sebagai arsip.

## Reproduksi

Gunakan repository dengan `src/` dan `scripts/`, Python/dependensi `requirements.txt`, serta `data/raw/classification/Train.csv`. Alternatif: `TP2_DATA_DIR` menunjuk folder induk `classification/`. Input dipatok SHA-256 `875552a68a3d24b0f3e1966d6ddb5671e6f54305a1c526e2f0a76747942ed91c`. Test, Regression Train, cache eksperimen, kredensial, dan model binaries tidak diperlukan.

```powershell
python -m pip install -r requirements.txt
python -m jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=1800 notebooks/samuel/11_samuel_formal_eda_a.ipynb
python -m unittest tests.test_samuel_eda_a -v
```

Alternatif regenerasi source+execution: `python scripts/build_samuel_formal_eda_a.py --execute`. Pemeriksaan salinan source dengan hanya satu Train: `python scripts/verify_samuel_formal_eda_a.py`. Opsi `--check-preservation` khusus checkout lokal referensi yang masih mempunyai draft modeling tertunda; opsi itu tidak diperlukan pada fresh clone atau untuk menjalankan EDA. Perhitungan menghasilkan hanya agregat dan empat PNG di folder ini; helper writer historis tidak dipanggil. Unit tests memeriksa tokenization/missingness, ties/support yang tidak memakai target, perbedaan weighted versus pooled rates, periode, dan kovarians nol yang tepat. Pemeriksaan angka merekonsiliasi output dengan tabel riset yang telah diverifikasi; receipt menyatakan cakupannya. Tidak ada penelitian kandidat baru, model training, perubahan model, atau Kaggle submission dalam milestone ini.
