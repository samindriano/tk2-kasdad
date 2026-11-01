# Keputusan shortlist akhir dan audit guard

1 November 2026; audit dari HEAD `1191661`. Kampanye tetap **PAUSED_FOR_REVIEW**. Dokumen ini menggantikan rekomendasi continuation empat kandidat pada review sebelumnya, tetapi tidak menimpa protokol, verdict, ledger, atau hasil historis. Belum ada fit/reservasi/akses Test/Kaggle baru.

## Keputusan

**Dua recipe training saja: C16 dan R22.** Masing-masing empat origin baru 59/63/71/75, jadi delapan predictor fits, bukan 16 atau 20. Model final belum dipilih: champion lama tetap fallback jika continuation gagal.

R22 mempunyai satu policy pembanding tetap, `R22_PCT_PRESENT`, dihitung dari skor mentah **model yang sama**; nol predictor fit tambahan. Tidak ada sweep bobot, threshold, mapping, atau route lain. Ini bukan lane training ketiga.

| Kandidat | Putusan | Alasan |
|---|---|---|
| C16 | Continue recipe asli | Gain primary/H12 dan tidak ada supported subgroup failure tersembunyi pada empat view pilot |
| R22 | Continue recipe asli, satu policy pembanding | Gain tetap ada late, lolos guard harmonized; output yang sama cukup untuk membandingkan mapping |
| C11 | Tidak dipilih | Nilai lanjut rendah: H12/late negatif; family diversity bukan kewajiban menghabiskan slot |
| R06 | Tidak continue recipe asli | W1 missing harm +0,395556 pada n392; sudah pasti melanggar scope individual-window yang kini dipilih |
| C17 | Reserve, tidak dilatih | Redundan dengan C16; bukan konfirmasi independen |
| R21 / R11 | Reserve, tidak dilatih | Lebih lemah daripada R22 dalam penilaian screening; bukan otomatis pengganti bila R22 gagal |
| R12 rescue / R23 rescue / R16 | Archive hypothesis | Belum cukup nilai tambahan untuk menambah fit saat ini; tidak menyatakan hipotesis universal salah |

## Hasil audit dan cakupan yang diseragamkan

[guard_audit.json](guard_audit.json) dihasilkan [script no-fit](../../../scripts/optimization_v3_final_guard_audit.py). Regression dihitung ulang dari bundle, sidecar, serta official Train; Classification menghitung kembali arithmetic guard dari detailed independent verifier historis, bukan membuat ulang semua mask kategorinya. NPZ+sidecar integrity diperiksa oleh loader; input hashes disimpan.

- C16: 32 supported subgroup checks pada primary/H12/W1/W4, semuanya lolos. C11 dan C17 juga 32/32 lolos; alasan tidak dipilih bukan kegagalan guard.
- R22 frozen: seluruh supported tail/missing checks pada empat view tersebut lolos; maksimum harm +0,171021, W4 high-tail n249.
- R06 frozen: gagal W1 missing +0,395556 (n392), walaupun primary/H12 lolos.
- R12 frozen: gagal missing W1/primary/H12. Tidak ada pemutihan verdict.
- R22_PCT_PRESENT: guard empat-view pilot lolos; missing harm nol dengan exact fallback. Ini diagnostik post hoc yang akan dibandingkan secara prospectif terhadap output origin baru, bukan bukti independen.

**Temuan scope historis:** C16/C17/C11 mengambil subgroup decision dari primary saja, walaupun detailed verifier menyediakan empat view. R06 hanya primary/H12. R12/R22 cakupannya lebih luas. Ini menghalangi perbandingan status tanpa memeriksa scope, tetapi bukan bukti gate dimanipulasi setelah skor. Source R06 yang berisi scope sempit cocok dengan freeze prefit, yang dibuat 07:12:04 UTC sebelum fit pertama 07:12:55 UTC.

Untuk continuation baru, scope eksplisit yang berlaku pada **kedua task**: pooled primary, H12, dan **setiap primary window W1–W6**. Tidak menambahkan novelty groups yang historis hanya deskriptif.

- Regression: stream missing, target≤5, target≥95; n≥100; setiap supported group MAE harm≤0,25.
- Classification: entry_friction, stream availability, text_load quantile groups sesuai aturan verifier lama; n≥200 dan ≥30 tiap kelas; F1 harm≤0,01 dan positive recall harm≤0,02. Bin text_load dibuat dari fit prefix masing-masing origin, bukan seluruh Train/score labels. Unsupported tetap deskriptif.
- H12_E/H12_L, per-cycle, dan ranking metrics tetap dilaporkan sebagai diagnostics; tidak diam-diam menjadi guard tambahan per-cycle.

Ini **amendment cakupan untuk keputusan berikutnya**, bukan penerapan ulang diam-diam pada verdict frozen lama. R06 tetap ADVANCE historis dan `NOT_SELECTED_HARMONIZED_SCOPE` untuk shortlist baru.

## Dua policy Regression yang dikunci sebelum continuation

1. **R22_ORIGINAL:** recipe XGBoost pseudo-Huber slope5+structured/SVD32 yang sama dengan R22 pilot. `component=clip(0.5*raw+0.5*pct_cycle(raw),0,100)`, `prediction=clip(0.75*champion+0.25*component,0,100)` untuk seluruh baris.
2. **R22_PCT_PRESENT:** estimator/preprocessing sama persis. Hitung `pct_cycle(raw)=100*average_rank(raw)/n_cycle` pada **semua baris dalam cycle lengkap**, kemudian present=`clip(0.75*champion+0.25*pct_cycle(raw),0,100)`; missing=exact champion. Jangan menghitung rank pada subset present. Availability memakai aturan missing yang sudah diverifikasi.

Policy kedua merupakan satu tambahan konfigurasi postprocessing yang dinyatakan eksplisit sesudah screening. Ia tidak boleh menyamar sebagai recipe lama atau berputar menjadi grid. Batas family lama telah memakai R21/R22; bila family accounting menghitung policy tambahan sebagai konfigurasi ketiga, amendment harus mencatat **pengecualian tunggal ini**, bukan mereset budget/configuration counter. Tidak ada izin perubahan loss/fit/hyperparameter baru.

## Evaluasi dan pemilihan sesudah delapan fit

Primary enam-origin:

| Origin fit end | Score primary |
|---:|---|
|55 (reuse pilot)|56–59 (W1)|
|59 (baru)|60–63 (W2)|
|63 (baru)|64–67 (W3)|
|67 (reuse pilot)|68–71 (W4)|
|71 (baru)|72–75 (W5)|
|75 (baru)|76–79 (W6)|

Primary pooled n38.588 unik; jangan menggandakan score rows dari output12cycle tiap model. W5+W6 n15.953 unik. H12 memakai dua fit55/67 yang sudah ada (n38.588); jangan menghitungnya sebagai origin baru atau konfirmasi independen. Semua label evaluasi sudah pernah digunakan dalam development.

Gate angka utama dipertahankan: Classification primary gain≥0,0005, H12 harm≤0,001, ≥4/6 window nonnegatif; Regression primary gain≥0,03, H12 harm≤0,05, ≥4/6 window nonnegatif. Terapkan harmonized supported-group guards di atas.

Tambahan prospectif untuk shortlist akhir (belum mengubah verdict lama): kandidat harus mempunyai gain positif terhadap champion pada pooled W5+W6. Ini filter deployment risk, bukan bukti bahwa Test pasti menyerupai late Train. Bila tidak lolos, rekomendasikan champion; jangan mencari cutoff/threshold alternatif.

Pilih R22_PCT_PRESENT di atas R22_ORIGINAL hanya jika:

- Lolos seluruh qualification gate terhadap champion, termasuk scope subgroup seragam dan late gain positif.
- MAE pooled primary enam-origin tidak lebih buruk daripada R22_ORIGINAL.
- Pada W5+W6, MAE lebih baik daripada R22_ORIGINAL **minimal 0,03**. Ini ambang materiality prospectif tambahan, bukan dipilih dari hasil continuation.

Jika varian tidak memenuhi ketiganya, pilih original bila original qualified; jika keduanya gagal, pertahankan champion. Tidak ada pembulatan untuk memutuskan threshold. Jangan menjumlahkan gain pilot ke public score.

## Integrasi teknis dan stop

Budget snapshot94/180; delapan continuation akan menjadi102/180 (sisa78), **belum direservasi**. Cek ulang ledger aktif sebelum fit; jangan mereset ke angka snapshot jika berubah. Satu training job dua thread, jobcap6GiB dan reserve2GiB sesuai protokol resource lama.

Runner pilot bukan continuation runner. R22 class mengunci cutoff55/67; buat adapter/wrapper terpisah untuk cutoff baru, dengan recipe sama, source/freeze/output namespace baru. Jangan mengubah source lama untuk melewati guard cutoff. Source shared harness C16 pada freeze lama berbeda dari current; diff commit `5840f4c` hanya mengubah `reserve_card_budget` untuk mendeteksi reservation duplicate/card identity. Tidak ada perubahan fit/rank/prediksi pada diff tersebut. Tetap ikat source current yang diaudit ke freeze baru; jangan menimpa freeze C16 lama atau mengklaim source identity-nya sama.

Luna harus uji synthetic end-to-end, aligned cached controls keenam origin, no fit-only leakage, rank ties/full cycle, empty/unsupported groups, dan resume atomik sebelum official fits. Verifier harus menghitung independently policy/metric dan semua view guard. Reuse pilot bundles yang utuh; jangan melatih ulang cutoff55/67. Implementasi dan freeze baru wajib selesai sebelum skor continuation pertama.

**Stop akhir lane Luna:** report full-six kedua task, dua policy Regression, guard table, pilihan kandidat/fallback, budget dan QA; commit lokal pekerjaan valid, lalu berhenti untuk review. Tidak melakukan full-Train refit, Test inference, Kaggle submission, perubahan notebook atau menyatakan V3 COMPLETE. User dapat memutuskan tahap finalist setelah melihat hasil. Prompt ada di [LUNA_PROMPT.md](LUNA_PROMPT.md).
