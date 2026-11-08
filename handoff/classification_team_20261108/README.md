# Handoff Classification Samuel untuk integrasi model tim

Mulai dari folder ini setelah clone `samindriano/tk2-kasdad`, branch `main`.
Tujuannya menyediakan komponen pembanding untuk model Abhi v3, bukan menggantinya
berdasarkan skor public Samuel yang lebih rendah. Nilai Abhi **0,83860** berasal
dari screenshot pengguna; handoff ini tidak memverifikasi akun atau submission Abhi.

## Yang langsung bisa dipakai

- `artifacts/validation_probabilities.csv.gz`: 29.394 baris prediksi temporal,
  termasuk dua origin yang saling tumpang tindih.
- `artifacts/test_probabilities.csv.gz`: 29.421 baris official Test.
- `artifacts/MANIFEST.json`: hash input/output, sumber commit, skema, dan batas interpretasi.
- `RESEARCH_MAP.md`: peta riset, hasil negatif, dan prioritas source yang perlu dibaca.
- `PROMPT_FOR_TEAMMATE.md`: instruksi siap salin untuk GPT/Codex teman.

File probabilitas ini berasal dari prediksi tersimpan. Ekspor tidak menjalankan
training, inference baru, evaluasi kandidat baru, atau Kaggle. Tidak ada label
Train, label Test, raw text, model binary, atau credential dalam paket ini.
`samuel_prediction` adalah prediksi model, bukan target aktual.

## Model yang benar: public 0,83225

Submission terkonfirmasi **58001282** memakai:

```python
p_samuel = np.where(stream_missing, p_a02, 0.5*p_pos + 0.5*p_a02)
threshold = np.where(stream_missing, 0.53230976129477, 0.50)
prediction = (p_samuel >= threshold).astype(int)
```

`p_pos` adalah POS_BAG_OFFSET_V1 asli: model bag-of-words Logistic Regression
dengan koreksi logit berbasis posisi token. `p_a02` adalah A02 asli, koreksi
residual berbasis tetangga leksikal di atas baseline routed Sparse25.
Final missing route memakai **A02**, bukan fallback POS dari versi MIX50 awal.
Ini bukan model global-strength, recency, Poisson, jointpos, atau notebook
akademis C4_HGB. Notebook akademis mendemonstrasikan recipe individual lebih awal.

Kode aplikasi akhir:
[`optimization_v3_cls_a02_missing_temporal_decision_v2.py`](../../scripts/optimization_v3_cls_a02_missing_temporal_decision_v2.py).
Bukti public:
[`SOL_UPLOAD/RESULT.md`](../../reports/optimization_v3/cls_a02_missing_temporal_decision_v2/SOL_UPLOAD/RESULT.md).
Ekspor telah mereproduksi ID dan label CSV submission tersebut persis; hash CSV
canonical tercatat di manifest. CSV submission asli dan model binary tetap lokal.

## Verifikasi dan penggunaan

Dari root repository, dengan Python, NumPy, dan pandas:

```console
python scripts/export_classification_team_handoff.py verify
python -m pytest tests/test_classification_team_handoff.py -q
```

Verifikasi paket tidak membutuhkan bundle NPZ lokal. `export-local` hanya untuk
maintainer yang memiliki input hash-pinned; jangan gunakan untuk clone baru yang
tidak memiliki bundle tersebut. Source training tersedia di repo, tetapi runner
historis memiliki freeze, receipt, dan ledger lokal. Jangan mematikan guard atau
menjalankan ulang kampanye lama hanya untuk memakai paket ini.

```python
from pathlib import Path
import pandas as pd

root = Path('handoff/classification_team_20261108/artifacts')
validation = pd.read_csv(root/'validation_probabilities.csv.gz', float_precision='round_trip')
test = pd.read_csv(root/'test_probabilities.csv.gz', float_precision='round_trip')
unique_validation = validation.loc[validation.is_primary_unique.eq(1)].copy()
```

| Origin | Data fit | Data prediksi | Baris |
|---|---|---|---:|
| E67 | cycle 47–67 | cycle 68–79 | 20.903 |
| E75 | cycle 47–75 | cycle 76–79 | 8.491 |
| E79 | cycle 47–79 | Test cycle 80–91 | 29.421 |

Seluruh ID E75 juga muncul pada E67. Jangan menghitung 29.394 baris sebagai
observasi independen. `is_primary_unique=1` memilih E67 cycle 68–75 dan E75 cycle
76–79, sehingga ada **20.903 ID unik**. Tetap laporkan hasil per origin/periode;
pool unik ini bukan pengganti seluruh protokol temporal historis.

Join target lokal dari official Train melalui `record_id`, dengan pemeriksaan
one-to-one dan hash sumber sesuai manifest. Cocokkan prediksi Abhi melalui
**origin + record_id**, batas fit, dan periode prediksi yang sama. Random-split OOF
Abhi tidak otomatis sebanding dengan prediksi temporal ini walaupun ID sama.
`source_row` adalah posisi baris pada official Train/Test, bukan fitur model.

## Cara menilai peluang gabungan

Bandingkan Abhi v3 standalone dengan komponen Samuel dan gabungannya pada baris
yang benar-benar sama. Mulai dari probabilitas, bukan voting CSV hard-label.
Periksa disagreement dan performa pada present/missing untuk melihat apakah
kesalahan saling melengkapi. `p_samuel` belum terbukti terkalibrasi atau sebanding
skalanya dengan skor Abhi; arithmetic blend tidak otomatis optimal.

Tetapkan satu uji gabungan sederhana dan aturan keputusan sebelum melihat skor
gabungan. Jika perlu memilih bobot/threshold, lakukan pada periode lebih awal dan
evaluasi pada periode lebih akhir, tanpa grid besar. Pertahankan Abhi v3 sebagai
comparator utama. Laporkan Macro F1, denominator, per-cycle/periode, dan precision/
recall; subgroup guards lama adalah kebijakan risiko riset, bukan aturan Kaggle.

Semua development cycle telah dipakai berulang selama kampanye. H12 dan primary
bukan holdout independen. Threshold missing final Samuel dipilih pada development
yang telah terlihat: uji forward sebelumnya justru negatif. Label prediksinya
boleh direplay, tetapi gain pada baris pemilihan threshold bukan bukti generalisasi.
Public 0,83225 juga tidak menjamin transfer ke private 70% Test.

Siapkan maksimal satu CSV gabungan jika bukti cukup; jangan otomatis mengunggah
atau mengubah pilihan final Kaggle. Pertahankan v3 dan submission Samuel asli.
