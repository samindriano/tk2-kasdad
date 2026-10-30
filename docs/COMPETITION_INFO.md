# Informasi resmi kompetisi TP2

Dikonfirmasi pengguna melalui instruksi Round 1 pada 30 Oktober 2026, berdasarkan halaman resmi kompetisi. Sesi ini memakai konfirmasi tersebut sebagai sumber; tidak mengklaim verifikasi live/login Kaggle baru.

Status terbaru: Classification Round 1 juga telah diotorisasi dan dijalankan sebagai milestone terpisah; metode/hasil berada pada `CLASSIFICATION_ROUND1.md` dan `reports/classification_round1/FINDINGS.md`. Kalimat scope regression-only di bawah adalah catatan saat informasi ini pertama dicatat. Untuk classification, kelas 0 berarti indikasi artefak eksploit publik **tidak teramati**, bukan bukti bahwa artefak pasti tidak ada.

| Task | Target dan makna | Metric | Submission schema |
|---|---|---|---|
| Regression | `threat_rank`: indeks prioritas respons kerentanan, skala 0–100 | MAE, lebih rendah lebih baik | `record_id,threat_rank` |
| Classification | `artifact_signal`: indikator artefak eksploit publik, binary 0/1 | Macro F1, lebih tinggi lebih baik | `record_id,artifact_signal` |

Sumber: [Regression](https://www.kaggle.com/competitions/tp2-regression-cyber-vulnerability-kasdd-gasal-2026-2027), [Classification](https://www.kaggle.com/competitions/tp2-classification-cyber-vulnerability-kasdd-gasal-2026-2027).

Test berasal dari cycle disclosure yang lebih baru daripada training. `impact_load` dan `entry_friction` adalah fitur resmi tambahan TP2 dan boleh menjadi predictor pada kedua task. Makna target, metric dan izin kedua fitur tersebut tidak lagi UNKNOWN. Round 1 ini hanya regression Samuel; classification tidak dijalankan.

Yang masih belum terkonfirmasi: formula pembentukan target/percentile, absolute calendar cycle, operational timestamp/provenance rinci fitur lain, dan generalisasi model pada test. Kehadiran suatu fitur dalam file test tidak membuktikan timing operasionalnya untuk penggunaan lain di luar task coursework.

Catatan UNKNOWN pada audit/EDA terdahulu adalah status historis saat materi PDF saja tersedia. Informasi yang dikonfirmasi di sini mengatasi ketidakpastian tersebut tanpa mengubah hasil EDA atau mengklaim source-generation formula sudah diketahui.
