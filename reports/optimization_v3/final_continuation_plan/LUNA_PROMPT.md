# Prompt eksekusi terakhir untuk Luna

```text
Lanjutkan KASDAD TP2 di:
C:\Users\Sam\OneDrive\Documents\KASDAD\TP 2

Baca AGENTS yang berlaku, lalu:
reports/optimization_v3/final_continuation_plan/DECISION.md
reports/optimization_v3/final_continuation_plan/guard_audit.json
reports/optimization_v3/batch1/PAUSE_HANDOFF_01.md

DECISION.md adalah keputusan review terbaru untuk scope lane ini. Dokumen lama tetap evidence/protocol historis, tetapi pilihan empat continuation dan instruksi auto-Kaggle lama tidak mengatasi scope sempit prompt ini.

Selesaikan hanya continuation C16 dan R22: cutoff59/63/71/75 masing-masing. Total8 predictor fit baru. Reuse pilot55/67 utuh. Estimator/fit recipe tidak berubah. R22 mempunyai satu pembanding postprocessing tetap R22_PCT_PRESENT sesuai DECISION; tidak ada fit tambahan atau konfigurasi lain.

Verifikasi root/remote/HEAD/status aktual dan bundle/freeze/input hashes. Orkestra Luna boleh memakai worker untuk implementasi/review independen dengan ownership terpisah; MAIN integrasi dan Git. Satu training job dua thread. Jangan spawn nested worker. Jangan mengubah frozen source/protocol/artifact lama, checkpoint, notebook/EDA, raw data atau pekerjaan rekan.

Buat continuation-only namespace baru:
experiments/optimization_v3/final_continuation/
experiments/runs/optimization_v3_final_continuation/ (ignored)
reports/optimization_v3/final_continuation/
Wrapper/adapter/tests dengan nama optimization_v3_final_continuation*. Jangan mengedit scripts/optimization_v3_postscreen_review.py atau final_guard_audit.py yang sudah menjadi evidence review.

Jangan memanggil runner pilot untuk origins baru. R22 class mengunci cutoff55/67; buat adapter baru yang ekuivalen untuk origin59/63/71/75. Audit diff shared harness C16 yang berubah sejak freeze; metadata-only reservation fix tidak mengizinkan rewrite source frozen. Jika tidak bisa membuktikan ekuivalensi/control alignment, stop dependency itu dengan blocker, bukan refit diam-diam.

Implementasikan dan tes end-to-end sampai independent metric+all-view guard+report+resume, lalu simpan recipe, input/source/package hashes dan amendment scope sebelum official fit. Hormati source raw Train only, fit-only TFIDF/SVD/category transforms, model controls frozen dan feature allowlist lama. Jangan early stopping menggunakan outer labels.

Reserve8 fits atomik di global ledger aktif setelah preflight;94/180 hanya snapshot. Jangan menambah24/16/20 fit dari rencana lama. Control refit hanya bila perlu membuktikan bridge baru, dihitung terpisah dan jelaskan alasannya; utamakan valid existing cache tanpa refit. Cap2threads/job6GiB/reserve2GiB, check lock ownership dan RAM aktual, tidak menghapus lock untuk memaksa run. Preserve per-origin outputs agar resume tidak mengulang fit.

Evaluasi full-six origin55/59/63/67/71/75 menggunakan empat-cycle primary windows disjoint (n38588). H12 tetap dari origin55/67. W5+W6 n15953. Jangan menyebut enam horizon bins dari dua pilot fits sebagai full-six; jangan menyebut labels untouched holdout.

Terapkan exact harmonized guard scope primary,H12,W1..W6 untuk kedua task dan qualification/late-selection criteria di DECISION. Unsupported groups tetap descriptive. Threshold Classification0.50. R22 variant pure percentile dihitung seluruhcycle baru setelah model raw score, baru missing fallback exactchampion; no subset ranking atau bobot sweep.

Jalankan sampai8 fits yang direncanakan selesai atau blocker nyata, bukan berhenti setelah pilot/worker/fit pertama. Jika satu task gagal valid, simpan hasil dan tetap selesaikan task independen lainnya. Tidak membuka rescue R12/R06/C11/C17/R16 atau model lain bila kandidat gagal. Tetapkan hasil sebagai qualified/notqualified menurut aturan yang sudah ditulis, tanpa melonggarkannya.

Simpan summary primary/H12/late/perwindow/groups, original vs variant, kontrol/fallback,fit attempts/time/budget, source/manifest dan tests. Jalankan independent verification. Satu commit lokal coherent boleh setelah valid; jangan push tanpa instruksi tambahan. Berikan final handoff dan status working tree/proses.

STOP setelah review checkpoint continuation. Jangan full-Train refit, inference Test, submission Kaggle, finalisasi notebook, resetbudget, auto-merge/rebase/force-push atau menandai V3 COMPLETE. Tahap finalist akan diputuskan setelah hasil ini direview.
```
