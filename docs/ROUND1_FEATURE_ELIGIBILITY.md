# Feature eligibility sebelum Round1 training

**GO bersyarat untuk controlled coursework comparison**, bukan sertifikasi semua timestamp operasional. Review dilakukan sebelum melihat skor model. Tidak ada feature-target search atau keputusan fitur berdasarkan hasil validation.

## Allowlist

| Fitur raw | Evidence kelayakan untuk task | Perlakuan | Batas evidence |
|---|---|---|---|
| impact_load, entry_friction | Pengguna mengonfirmasi keduanya fitur resmi TP2 yang boleh menjadi predictor | Dua fitur awal seluruh recipe; EF binary | Izin coursework terkonfirmasi; jangan menambah label TP1 melalui join |
| cycle_code | Official Train/Test; materi p16 menjelaskan urutan disclosure, besar=baru | Numerik terurut | Bukan kalender; test lebih baru/extrapolation risk |
| scope_pressure, text_load | Official Train/Test; materi p17 mendeskripsikan intensitas scope dan volume informasi teknis | Numeric, fit-only median jika missing | Konstruk/timestamp rinci belum terkonfirmasi |
| scope_trace, org_spread, module_spread | Official Train/Test; materi p16–17 menyebut indikator scope/spread dengan format campuran | Numeric component, approximate/suffix/missing/parse-failure flags deterministik | Makna ~ tidak disamakan dengan exact count tanpa flag; timestamp rinci belum diketahui |
| telemetry_flag | Official Train/Test; materi p17 boolean/string/missing | Explicit mapping false/inactive/N/0 dan true/active/Y/1; missing/invalid flags | Bukan bukti timing pengumpulan |
| org_code, module_code, pattern_code | Official Train/Test; materi p16 kode organisasi/module/weakness family | Train-only rare grouping dan native categorical; unknown/missing diperlakukan eksplisit | Tidak menganggap besaran kode numerik; novelty/timing/provenance masih risiko |

Semua12rawcandidate tersedia di train/test. Missingness dan cardinality sudah diaudit sebelumnya; tidak diulang dari nol. Review parser menemukan semua nonmissing mixed values berbentuk numeric, ~numeric, atau numeric dengan suffix kelas yang konsisten perfield, tanpa unparsed cell pada kedua file. Parser tetap fail-to-NaN+flag untuk bentuk tidak dikenal. Review format test tidak memakai label dan tidak menentukan parameter model dari skor test.

Unseen category rows terhadap vocabulary train: org_code20,96%, module_code36,77%, pattern_code0,70% dari baris test nonmissing. Ini menjadikan unknown handling penting; official test tidak dipakai fit encoder atau frequency cutoff. Encoder fit-fold memakai min_frequency30, max_categories254, unknown/missing−1, lalu HGB native categorical mask. Numeric imputer/linear scaler hanya belajar dari fit fold.

## Denylist

- `record_id`: identifier; tidak menjadi predictor.
- `threat_rank`, `artifact_signal`: target/label task lain; tidak menjadi input Round1.
- `signal_stream`: text ditunda untuk Round2 sesuai scope.
- `audit_lane`, `checksum_hint`, `pulse_a`, `pulse_b`: provenance/semantik intended prediction-time belum cukup jelas untuk comparison ini. Dikecualikan secara konservatif; tidak disebut terbukti leakage.

## Status sumber

Makna target threat_rank0–100, MAE, artifact_signal0/1, MacroF1 dan izin impact/EF dikonfirmasi pengguna dari halaman resmi; lihat `COMPETITION_INFO.md`. Materi PDF bersama mendukung deskripsi kolom tambahan, dan kehadiran pada official Test membuktikan tersedia dalam supplied scoring inputs. Evidence ini **tidak** menetapkan bahwa seluruh fitur telah ada sebelum outcome pada deployment eksternal.

R1-D ditafsirkan sebagai fitur terstruktur **supplied-schema/semantics eligible, timing-provisional**. Jika tugas kelak menuntut timestamp-certified as-of prediction, sepuluh tambahan di luar impact/EF memerlukan authority lebih lanjut. Round1 tidak mengklaim readiness operasional atau fitur final.
