# CLS_POS_RECENCY_E79_V1 — hasil benchmark eksploratif

## Status

Satu fit gamma E79 pada official Classification Train selesai dan lolos
independent fit QA. Satu aplikasi feature-only pada official Test menghasilkan
CSV lokal; independent application QA juga PASS. Tidak ada Kaggle upload atau
kontak Kaggle. Hasil ini **eksploratif saja**: verdict scientific pilot tetap
FAIL, dan E79 tidak ditetapkan sebagai champion.

## Fit

| Item | Hasil |
|---|---:|
| Official Train | 50.608 baris, cycle 47–79 |
| Gamma fit | 47.204 baris dengan `signal_stream` tersedia |
| Dikecualikan dari gamma fit | 3.404 baris dengan stream kosong |
| Positional features | 17.259 |
| Objective akhir, dihitung ulang independen | 11.004,216689843077 |
| Gradient infinity norm, dihitung ulang independen | 0,0000330555124973 |
| Gamma SHA-256 | `146434e0ac49e2248bbe24c5bffd5193ff7c7e58dd3d456915077527704bfefb` |

Recipe yang dijalankan: bobot `max(0.25, 2**(-(79-cycle_code)/8))`, dinormalisasi
dengan mean terpisah per kelas pada stream-present rows; frozen C2_B `z_BAG`
offset; summed weighted logistic loss plus `0.5 * ||gamma||²`; zero-init
L-BFGS-B dengan opsi pilot yang sama (`maxiter=2000`, `ftol=1e-12`,
`gtol=1e-6`, `maxls=50`). Satu official optimizer start, tanpa preprocessing
fit, shuffled control, atau uniform comparator fit.

## Test application dan perbandingan label

| Item | Hasil |
|---|---:|
| Official Test | 29.421 baris dan ID unik |
| Stream-present / missing | 28.359 / 1.062 |
| Prediksi positif E79 | 6.843 |
| Prediksi positif champion tersimpan | 6.759 |
| Perubahan label dibanding champion | 200 |
| Missing probabilities sama bit-exact dengan A02 | Ya |
| Missing labels sama dengan champion | Ya |

Perubahan label terjadi pada stream-present route. Test tidak memiliki label
yang digunakan di sini; angka positif dan flips merupakan ringkasan perubahan
prediksi, bukan ukuran performa. Karena itu tidak ada Macro F1 Test atau public
score yang dilaporkan.

CSV lokal:
`submissions/cls_pos_recency_e79_v1/classification_cls_pos_recency_e79_v1.csv`

SHA-256:
`6e8f6b08e769135717de4ce1f821af8ec696b847849b516d0f8749da4b59da55`

CSV berisi tepat 29.421 baris, kolom `record_id,artifact_signal`, seluruh ID
Test dalam urutan yang sama, dan label integer 0/1. Independent verifier
memindai 25 submission classification lengkap yang tersedia dan tidak
menemukan duplikat vektor label.

## Recovery adapter Test

Percobaan setup bridge pertama berhenti pada `AttributeError` karena kode lama
merujuk `r3.TEST_REL`, sebelum `read_feature_csv` dipanggil. Tidak ada Test row,
SampleSubmission, prediksi, atau CSV yang dibaca/dihasilkan pada percobaan itu.
Marker dan failure receipt tersebut dipertahankan. Helper recovery additive
memverifikasi bahwa `R1_TEST_REL` adalah file sumber `.py`, lalu menetapkan
alias process-local `r3.TEST_REL` ke **`data/raw/classification/Test.csv`**
dengan hash Test yang dibekukan. Sesudah preflight itu hanya satu transform
Test dan satu persiapan CSV yang benar-benar berjalan.

## Reproducibility dan QA

- Source inventory memverifikasi 51 hash source/input MAIN.
- Prefit source/synthetic QA PASS; freeze/reservation membatasi fit resmi satu.
- Synthetic lifecycle PASS, termasuk persistence/readback, overwrite refusal,
  dan resume tanpa fit kedua.
- Final regression run atas dua file test: 10 passed, 0 failed (9 test model +
  1 recovery; `RECOVERY_UNIT_TEST_QA.json`). QA pertama yang berisi satu
  ekspektasi label keliru tetap disimpan terpisah sebagai riwayat.
- Independent fit verifier menghitung ulang weights, class mass, objective,
  gradient, dan seluruh input digest tanpa memanggil optimizer.
- Independent application verifier memeriksa hash receipt/cache, route
  probability, threshold, champion reference, missing fallback, Test/Sample ID
  order/schema, CSV hash, dan duplikasi label.
- Official Test target dan SampleSubmission target values tidak dibaca.
- Model fit source/recipe tidak berubah; Test dipakai hanya untuk inference.
- Kaggle tidak dihubungi dan tidak ada upload.

## Artefak utama

- `FREEZE.json`, `PREFIT_QA.json`, `RESERVATION.json`
- `FIT_QA.json` dan `FIT_RECEIPT.json`
- `APPLICATION_MANIFEST.json`, `APPLICATION_QA.json`,
  `APPLICATION_RECEIPT.json`, `APPLICATION_RECOVERY.json`
- Script recovery additive: `scripts/cls_pos_recency_e79_recovery_v1.py`

Fit receipt dan application receipt di bawah `reports/` adalah salinan
audit-view dari receipt lokal pada folder `experiments/`; salinan tersebut
memiliki bytes/hash yang sama saat dibuat. Model arrays, gamma NPZ, application
arrays, dan CSV tetap lokal/ignored sesuai kebijakan raw/model artifacts.

## Git

- Fit source commit: `bbac3d998f801702788d3b7e934a0263f99825bf`
- Freeze/reservation commit: `fd96e261d8cdd709f4b431fd9b6c066b72c253be`
- Recovery adapter: `5af6cabdfaa3962177bc822d538d5499615269d3`
- Tidak ada push.
