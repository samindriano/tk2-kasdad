# Reproduksi dan QA — independent EDA discovery Samuel

Notebook utama: [eda_candidate_discovery.ipynb](../../notebooks/samuel/eda_candidate_discovery.ipynb). Ini notebook riset kandidat, bukan notebook final Classification/Regression dan bukan penggantian otomatis pertanyaan resmi.

## Input dan lingkungan

Gunakan repository beserta `src/`, `scripts/`, dan laporan historis yang sudah ada. Pasang `requirements.txt`; lingkungan referensi memakai Python3.13.5. Kedua official Train diperlukan, tanpa Test atau SampleSubmission:

```text
data/raw/classification/Train.csv
data/raw/regression/Train.csv
```

Alternatif: arahkan `TP2_DATA_DIR` ke folder yang berisi subfolder `classification/` dan `regression/`. Nama Train diselesaikan case-insensitive oleh resolver repository. Hash kedua Train dipatok di skrip; dataset berbeda harus diselidiki, bukan diterima diam-diam. Tidak ada kredensial, cache eksperimen ignored, model binary, atau file sementara pribadi yang dibutuhkan. PDF resmi dicatat provenance-nya dalam scope manifest; tidak diperlukan untuk menjalankan perhitungan numerik.

## Restart & Run All

Jalankan dari dalam repository:

```powershell
python -m pip install -r requirements.txt
python -m jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=1800 notebooks/samuel/eda_candidate_discovery.ipynb
```

Notebook menjalankan empat analisis dari data mentah, membentuk ulang perbandingan dan grafik, kemudian menyimpan output agregat ke folder riset ini. Tidak mengimpor report writers historis atau melakukan fit/upload. Input yang sama menghasilkan statistik deterministik; metadata eksekusi notebook dapat berbeda. Windows dapat memberi warning Jupyter selector-thread fallback tanpa cell error.

Per arah:

```powershell
python scripts/eda_candidates_text.py
python scripts/eda_candidates_missingness.py
python scripts/eda_candidates_categories.py
python scripts/eda_candidates_targets.py
python scripts/build_eda_candidate_discovery.py
```

Perintah terakhir menyusun ulang source notebook tanpa outputs; gunakan `--execute` untuk langsung mengeksekusinya. Jangan menjalankan builder tanpa execution jika ingin mempertahankan outputs tersimpan untuk review.

Untuk pemeriksaan file-isolation yang dipakai pada milestone ini:

```powershell
python scripts/qa_execute_eda_candidates.py
```

Perintah ini membuat salinan sementara berisi source dan notebook saja, tanpa reports awal, cache, atau folder data. Kedua Train diberikan melalui `TP2_DATA_DIR`; kernel baru memakai dependencies yang sudah terpasang. Setelah semua cell berhasil dan JSON/CSV numerik cocok, executed notebook dipromosikan kembali ke repository. Ini menguji ketergantungan file, bukan instalasi dependencies baru atau environment/container yang sepenuhnya baru.

## Pemeriksaan substantif

```powershell
python scripts/verify_eda_candidates.py
python -m unittest tests.test_formal_eda -q
```

Verifier membaca kedua Train secara independen dari fungsi penulis discovery dan merekonsiliasi nilai utama: join satu-ke-satu, exact-token-count TTR dengan set-based tokenization dan SciPy tied ranks, support dan weighted artifact prevalences, asosiasi length proxy, raw missingness gaps, decile standardization,906 out-of-range observations, serta coverage kategori30/100/300 tanpa missing pseudo-category. Ia **tidak menghitung ulang setiap exploratory cell/entropy/bootstrap draw**; receipt membatasi klaim sesuai cakupan pemeriksaan.

`scope_preservation.json` merekam319 tracked files sebelum milestone dan delapan draft lokal dari finalization yang dihentikan. Pemeriksaan byte preservation bersifat khusus checkout referensi: gunakan `--check-preservation --check-paused-drafts` untuk memeriksa keduanya. Opsi kedua menambahkan pemeriksaan draft lokal yang sengaja tidak disertakan pada commit riset ini; jangan memakai opsi itu pada fresh clone yang tidak mempunyai draft tersebut. Default verifier hanya memeriksa angka dan source boundaries, sehingga perbedaan line endings antarplatform tidak dianggap kegagalan numerik. Tidak ada private draft yang diperlukan oleh notebook atau numerical analysis.

## Output dan batas review

- [DISCOVERY_FINDINGS.md](DISCOVERY_FINDINGS.md): semua arah, tiga shortlist, statistik/denominator, counterexamples, rekomendasi.
- [CANDIDATE_COMPARISON.md](CANDIDATE_COMPARISON.md): explicit non-overlap, perbandingan dengan EDA lama, trade-off penggantian.
- `text/`, `missingness/`, `categories/`, `targets/`: aggregate CSV/JSON/plots dan laporan pendukung bila tersedia.
- [verification.json](verification.json): receipt pemeriksaan numerik/preservation.
- [QA.json](QA.json): eksekusi notebook, pemeriksaan visual, tests, dan preservation akhir.

Visual acceptance berasal dari pemeriksaan PNG sebenarnya, bukan keberhasilan eksekusi saja. Hasil receipt yang belum ada berarti pemeriksaan itu belum selesai; jangan menganggap tautan sebagai bukti PASS. Tidak ada raw text, token strings, record-level dataset, model, perubahan formalEDA, atau Kaggle upload yang termasuk keluaran ini. Penelitian berhenti pada review pemilihan kandidat oleh Samuel.
