# Research Next V1 — checkpoint untuk review riset

Tanggal: 2 November 2026

Status: `RESEARCH_REVIEW_READY`

Queue N00–N06, amendment N04_DIAGNOSTIC, audit lanjutan N01/N02, dan review fixed class blend sudah memiliki bukti tersimpan. Tidak ada kelanjutan model baru yang lolos aturan seleksi queue. N04_DIAGNOSTIC tetap diagnostik; verdict N04 asli tidak berubah.

## N04_DIAGNOSTIC

**Pertanyaan:** apakah density weighting masih memberi manfaat prediktif ketika domain AUC sedikit di bawah 0,55 tetapi ESS dan clipping memenuhi syarat?

Amendment dibuat setelah hasil domain AUC terlihat dan sebelum task MAE dibuka; ini **bukan** analisis outcome-blind. N04 asli tetap `SKIPPED_SCIENTIFIC_PREFLIGHT` pada freeze `eb05a654…`. Freeze amendment terpisah `f9e5cf46…` mempertahankan recipe dan gate N04. Empat domain fits beserta OOF probabilities, weights, receipts, dan recovery arrays dipakai ulang; tidak ada domain classifier yang dilatih ulang.

| Cutoff | Domain AUC fold | ESS fraction | Source clipping | Task contrast di primary |
|---:|---|---:|---:|---|
| 55 | 0.549576, 0.542069 | 0.829 | 0.062 | W1: gain +0.079782 MAE vs R22; +0.037368 vs shuffled weights |
| 67 | 0.592372, 0.604807 | 0.816 | 0.036 | W4: gain −0.017641 MAE vs R22; −0.005598 vs shuffled weights |

Pada primary gabungan W1+W4 (n=9.064), weighted model memberi gain +0.026578 MAE vs R22 dan +0.013904 vs shuffled-weight control. Pada H12 pooled (n=38.588), gain-nya +0.012444 vs R22 dan +0.002635 vs shuffled control. Di enam window, gain vs R22 positif pada 4/6, sedangkan gain vs shuffled weights positif pada 3/6. Hasil H12 cutoff 67 juga campuran: −0.010856 vs R22 tetapi +0.003486 vs shuffled control.

Subgroup missing-stream dan target aktual ≤5/≥95 direkonsiliasi terhadap R22, shuffled weights, protected legacy, dan cache R06. Rincian denominator, MAE, gain/harm seluruh view ada pada [laporan amendment](results/N04_DIAGNOSTIC/amendment_01/SUMMARY.md). Contoh H12 pooled: missing-stream gain +0.038067 vs R22 dan +0.019236 vs shuffle, tetapi harm vs legacy +0.064371; tail ≤5 harm vs R22 −0.029075, gain vs shuffle +0.062729, harm vs legacy −0.707738; tail ≥95 gain vs R22 +0.005277, gain vs shuffle +0.035603, harm vs legacy +0.012862. Tanda positif pada kolom harm vs legacy berarti MAE kandidat lebih tinggi daripada legacy. Seluruh tabel subgroup adalah diagnostik dan tidak menciptakan gate baru.

Pooled primary/H12 mendukung manfaat rata-rata kecil pada dua view yang diuji, sementara window, cutoff, dan subgroup menunjukkan hasil campuran. Dua cutoff tidak cukup untuk menyimpulkan hubungan umum antara domain AUC dan task gain. ESS/clipping yang lolos tidak membuktikan manfaat prediktif. N04_DIAGNOSTIC berstatus `DIAGNOSTIC_ONLY`, bukan kandidat continuation.

## Status queue dan keputusan seleksi

| Lane | Bukti/status yang dipertahankan | Fit resmi |
|---|---|---:|
| N00 | Audit cache dan rekonstruksi fixed arm lulus; zero-fit | 0 |
| N01 class-by-volume | Verdict pilot `RETIRE`; primary kalah dari matched class-only control | 4 |
| N01 class-only diagnostic | `COMPLETE_DIAGNOSTIC_ONLY`; primary +0.000687 Macro F1, tetapi W5+W6 −0.000237; tidak qualified | 4 |
| N02 sketch | Pilot `PILOT_COMPLETE_ADVANCE` tetap historis; audit semua view menemukan dua kegagalan positive-recall pada Q4 text-load H12, sehingga continuation ditahan | 4 |
| N03 router | `RETIRED_PERFORMANCE`; late primary gagal performance dan matched-control gate | 2 |
| N04 asli | `SKIPPED_SCIENTIFIC_PREFLIGHT`; empat domain fits tersimpan, tanpa task fit | 4 |
| N04_DIAGNOSTIC | `DIAGNOSTIC_ONLY`; empat task fits sesudah N05 dan sebelum N06 | 4 |
| N05 interval information | `COMPLETE_INFORMATION`; bukan lane promosi point model | 4 |
| N06 token mapping information | `COMPLETE_INFORMATION`; bukan lane promosi point model | 4 |

N01 class-only adalah kelanjutan diagnostik yang diotorisasi terpisah, bukan perubahan verdict N01 class-by-volume. Fixed blend `0.5*N02 sketch + 0.5*N01 class-only` berstatus `NO_GO_FURTHER_FITS_FOR_THIS_BLEND`; verifier-nya lulus, tetapi primary gate gagal. Lihat [audit N01/N02](results/N01_N02_GUARD_AUDIT/amendment_01/SUMMARY.md), [N01 class-only](results/N01_CLASS_ONLY_DIAGNOSTIC/amendment_01/SUMMARY.md), [N03](results/N03/amendment_01/SUMMARY.md), dan [review fixed blend](results/CLASS_BLEND_REVIEW/amendment_01/SUMMARY.md).

Dengan aturan queue asli, tidak ada kandidat Classification/Regression yang memenuhi syarat untuk continuation berikutnya: N02 ditahan oleh subgroup guard; N01 class-only gagal syarat late W5+W6; N03 retired; N04 asli di-skip; N04_DIAGNOSTIC tidak eligible; N05/N06 hanya information lanes. Tidak ada fit continuation tambahan yang dimulai setelah review fixed blend.

## Urutan, ledger, dan QA

- Ledger [budget.jsonl](../../../experiments/runs/optimization_v3_research_next_v1/budget.jsonl), SHA-256 `e70b0e850b300b380c9c134532926a2317bd76803d23c571a2ab56978a6c0b8a`: 77 event, cap 60, 30 `FIT_STARTED`, 30 `FIT_SAVED`, 30 fit ID unik; start dan save cocok, 0 fit gagal/terinterupsi.
- Reservasi bruto 34 dikurangi 4 slot N04 lama yang dilepas = 30 reservasi aktif. Slot release N04 tetap tersimpan; reservasi baru `N04_DIAGNOSTIC:DIAGNOSTIC` tepat 4 dan digunakan 4 kali. Tidak ada slot yang dihitung dua kali.
- Waktu ledger membuktikan N05 selesai lebih dulu, empat fit amendment N04_DIAGNOSTIC tersimpan berikutnya, lalu N06 mulai. [Preflight/freeze N04_DIAGNOSTIC](../../../experiments/optimization_v3/research_next_v1/frozen/N04_DIAGNOSTIC/amendment_01/preflight.json) mencatat N05 selesai, N06 belum mulai, task MAE belum terlihat, stable-identity check lulus, dan synthetic end-to-end serta persistence/readback/resume lulus.
- [Independent verification N04_DIAGNOSTIC](results/N04_DIAGNOSTIC/amendment_01/verification.json) lulus: empat domain bundle dipakai ulang, empat task bundle diverifikasi, 50 hash input/source diperiksa, rank50 average ties, weight/policy blend, MAE report, dan batas data diverifikasi. [Tes khusus amendment](../../../tests/test_research_next_n04_diagnostic.py) dijalankan pada checkpoint ini: **3 passed**.
- N06 verification awal tetap tersimpan dengan kegagalan accounting `global_and_pilot_caps_respected`; [amendment koreksi zero-fit N06](results/N06/amendment_02_verification_correction/verification.json) merekonsiliasi empat slot N04 yang dilepas, lulus tanpa fit baru dan tanpa mengubah hasil/prediksi.
- Verifier fixed blend: PASS, zero fit. Tidak ada Test/SampleSubmission, full-Train refit, Kaggle access, upload, atau perubahan notebook pada checkpoint ini. Tidak ada fit research yang sedang berjalan.

## Model dan skor publik historis

Angka internal temporal dan skor publik adalah sumber evaluasi berbeda. Catatan Kaggle historis [KAGGLE_RESULTS.md](../../submissions/KAGGLE_RESULTS.md) mencatat C4_HGB initial benchmark ref `57808225`, public Macro F1 `0.82641`, dan R1_D initial benchmark ref `57808231`, public MAE `13.53473`. Catatan itu juga memisahkan internal Round 2 C4_HGB primary/H12 Macro F1 `0.825481/0.796810` dan R1_D primary/H12 MAE `13.349836/15.5772`.

Pada continuation internal, C16 tetap gate-failed dan tidak menjadi champion; R22_ORIGINAL qualified dengan primary MAE gain `0.039699` dan H12 gain `0.124873` terhadap champion menurut [laporan continuation](../final_continuation/CAMPAIGN_SUMMARY.md). Status repo terakhir [C16/R22](../../submissions/c16_r22/SUBMISSION_STATUS.md) menyatakan kedua CSV tidak di-upload dan tidak memiliki public score; hal tersebut tidak dicek ulang lewat Kaggle pada milestone ini. Ada nilai regression Kaggle berbeda `11.71363` dalam [QA akademis historis](../../academic_availability/QA.json); asal perbedaan dengan ref/score pada `KAGGLE_RESULTS.md` belum direkonsiliasi, sehingga angka tersebut tidak dipakai sebagai skor publik terverifikasi di ringkasan ini.

## Stop point

Checkpoint ini siap untuk review lintas lane. Pertahankan seluruh freeze dan receipt. Tidak ada eksperimen, refit, inferensi Test, atau submission berikutnya yang dimulai otomatis.
