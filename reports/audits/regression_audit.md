# Audit dataset TP2 — regression

Status: **COMPLETE**.

Grain: satu baris kandidat record; semantik satu vulnerability per record belum dikonfirmasi data dictionary.

Target: `threat_rank`; evidence: VERIFIED_TRAIN_TEST_SAMPLE_SCHEMA.

| File | Rows | Columns | Duplicate rows | Duplicate ID |
|---|---:|---:|---:|---:|
| train | 50,608 | 19 | 0 | 0 |
| test | 29,421 | 18 | 0 | 0 |
| sample | 29,421 | 2 | 0 | 0 |

Nama, dtype setiap kolom, missing, kuantil, IQR, kategori, hash SHA-256 dan hitungan cycle lengkap terdapat dalam JSON pendamping. IQR adalah penanda deskriptif, bukan alasan otomatis menghapus data.

Prediktor selain ID: 17; fitur berurutan cocok: True; ID overlap: 0; shared distinct feature rows: 0.

Duplicate fitur train/test tanpa ID dan target: 0/0.

Target numerik train: 34,333 nilai unik; kuantil {'0.0': 0.0428, '0.01': 1.075707, '0.05': 5.0096, '0.1': 10.0166, '0.25': 25.0476, '0.5': 50.02815, '0.75': 74.9819, '0.9': 90.01224, '0.95': 95.03406499999998, '0.99': 99.0291, '1.0': 100.0}; top-10 nilai mencakup 1.08% baris. Angka ini deskriptif; nama/range teramati tidak menetapkan definisi domain target.

| Kolom | Missing train | Missing test | KS numeric / TV categorical |
|---|---:|---:|---:|
| cycle_code | 0.00% | 0.00% | 1.0000 |
| org_code | 5.31% | 23.27% | 0.5454 |
| module_code | 7.99% | 25.14% | 0.6957 |
| pattern_code | 16.17% | 8.61% | 0.1865 |
| signal_stream | 6.73% | 3.61% | 1.0000 |
| scope_trace | 8.36% | 8.27% | 0.2382 |
| org_spread | 8.61% | 8.39% | 0.2131 |
| module_spread | 8.22% | 8.50% | 0.2210 |
| text_load | 0.00% | 0.00% | 0.1551 |
| scope_pressure | 0.00% | 0.00% | 0.2138 |
| pulse_a | 0.00% | 0.00% | 0.0753 |
| pulse_b | 0.00% | 0.00% | 0.0237 |
| telemetry_flag | 11.27% | 11.11% | 0.0142 |
| audit_lane | 0.00% | 0.00% | 0.0452 |
| checksum_hint | 0.00% | 0.00% | 0.0085 |
| impact_load | 0.00% | 0.00% | 0.0943 |
| entry_friction | 0.00% | 0.00% | 0.0613 |

KS = perbedaan maksimum empirical CDF; TV = total variation proporsi kategori nonmissing. Nilai ini deskriptif, tanpa klaim signifikansi atau penyebab drift; kategori teks ber-cardinality tinggi perlu dibaca bersama unseen rate dan panjang teks di JSON.

## Risiko dan implikasi

- HIGH / confidence tinggi: missing dan format heterogen memerlukan parser terdokumentasi; jangan fit imputasi atau encoding pada gabungan train/test.
- HIGH / confidence pada ketidakpastian: asal pembentukan audit_lane, checksum_hint, dan fitur lintas-target belum diketahui. Korelasi tidak membuktikan atau menyingkirkan leakage; verifikasi ketersediaan sebelum outcome dengan asisten/data dictionary.
- cycle_code adalah kode urut; arti tanggal/interval belum diketahui. Pisahkan train/validation menurut urutan cycle jika semantik waktu dikonfirmasi; jangan gunakan test untuk memilih fitur, threshold, atau model.
- impact_load dan entry_friction sebagai fitur TP2 tidak otomatis leakage hanya karena merupakan target TP1. Jangan menggabungkan label TP1 tambahan; gunakan hanya kolom resmi task TP2.
- Target test tidak tersedia, sehingga target shift train–test tidak dapat diuji. Data dictionary, lineage dan timing fitur tetap UNKNOWN.

Tidak ada training, hyperparameter tuning, atau submission pada audit ini.
