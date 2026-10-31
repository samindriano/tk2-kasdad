# Panduan Luna — satu lane atau seluruh antrean end-to-end

**Paket ini disusun 1 November 2026 sebagai research-only handoff. Belum ada kandidat queue yang diimplementasikan atau dilatih. Kampanye V3 tetap belum selesai.** User dapat menjalankan sesi Luna berikutnya dengan salah satu prompt di bagian akhir.

## Bacaan minimum dan sumber kendali

1. Baca instruksi project yang berlaku, file ini, lalu `experiments/optimization_v3/research_backlog/queue.json`.
2. Untuk SINGLE_LANE, baca kartu ID yang dipilih serta recipe lain pada task+family yang sama. Untuk ONE_CHAT_END_TO_END, baca kartu berikutnya ketika akan dikerjakan. Jangan berulang kali memuat seluruh riwayat.
3. Queue dan kartu canonical mengatasi perbedaan dengan `*_draft.md`. Manifest kontrol dan data aktual mengatasi catatan konteks yang stale. Profil model tetap mengikuti pilihan runtime pengguna; jangan otomatis memakai Sol/Astra atau menyatakan berjalan dengan Luna jika runtime tidak demikian.

Checkpoint asal adalah **cb27597bedf50e126bd3182898bb1301c49dbbb2**. Mulai dari HEAD terbaru yang memuat checkpoint dan paket ini. Verifikasi root/remote/branch/HEAD/status; jangan reset working tree agar cocok dengan nama commit lama. Raw data dan source lama dipertahankan. Semua pekerjaan baru memakai namespace batch1 di queue.

## Orchestra diperbolehkan

Gunakan satu Luna MAIN dan configured worker/explorer Luna XHigh sesuai instruksi orchestra yang berlaku untuk sesi Luna pilihan pengguna. Verifikasi effective pins; jangan auto-escalate ke Sol/Astra. Jika delegation tidak tersedia atau pin tidak dapat dipastikan, kerjakan DIRECT dan catat keterbatasannya, bukan berhenti hanya karena tidak ada worker.

MAIN memiliki framing, shared harness, registry, integrasi, keputusan, Git dan Kaggle. Worker menerima pertanyaan, input, namespace/worktree, prohibited actions, deliverable, verification dan stop point yang jelas. Worker tidak spawn worker lain, merge atau upload sendiri. Implementasi adapter dan review independen boleh paralel dengan scope yang tidak overlap; **hanya satu job training dua thread** yang memakai slot compute pada satu waktu. Worker selesai berarti MAIN melanjutkan antrean, bukan seluruh chat berhenti.

## B0 — bangun fondasi sekali

Luna perlu membuat harness kecil yang dapat dipakai ulang, bukan 45 controller dan verifier yang hampir sama. Ini pekerjaan implementasi pertama, belum tersedia sebagai CLI sekarang.

- Satu runner membaca satu ID dari queue. Adapter model dibuat terpisah, misalnya `src/models/optimization_v3_batch1_c03.py`. Loader generik tidak perlu diedit tiap ada adapter baru. Source yang sudah dibekukan tidak diubah oleh kartu berikutnya.
- Gunakan resolver official data `src.utils.paths.task_files`. Fit hanya Train task bersangkutan. Daftar feature eksplisit mengecualikan `record_id`, target dan target task lain.
- Bridge kontrol harus memeriksa input/source/cache hashes, cutoff dan urutan row. Class memakai sparse25 terbaru; Regression memakai tiga komponen incumbent. Jangan memakai hasil kandidat baru sebagai kontrol di tengah batch.
- Kontrak hasil adapter harus konsisten dan diuji **hingga pembentukan seluruh report**, bukan hanya fungsi metric: raw score, row indices, state, fit population, route masks, prediction policies dan fit metadata.
- Uji end-to-end sintetis harus memuat string `record_id`, tied/binary quantile edges, stream kosong, kategori baru, subgroup support nol, nilai missing dan resume setelah satu fit. Metric untuk `n=0` adalah null/unsupported. Uji encoder terawasi dengan perturbasi label satu cycle dan audit dua `PYTHONHASHSEED`.
- Path preflight membuat direktori, lalu memastikan file output belum ada atau identitasnya cocok untuk reuse. Simpan NPZ dan sidecar dengan penulisan eksklusif/atomik. Jangan memakai overwrite NPZ lalu baru menolak sidecar.
- Verifier menghitung route/rank/metric dari raw arrays dan row masks secara terpisah. Memanggil helper metric yang sama dua kali bukan pemeriksaan independen.

Jika bridge baru mengubah fit/prediction handling, lakukan satu replay kontrol cutoff 67 per task dan cocokkan terhadap cache terverifikasi, lalu hitung fit tersebut sebagai control refit. Jika bridge hanya membaca cache yang identitasnya sudah diverifikasi, jangan mengulang seluruh training history.

## Satu kartu: urutan yang tidak boleh ambigu

1. **Preflight.** Baca kartu, fixed recipe, policy, population, views, kondisi, support dan jumlah predictor fit. Periksa dependency/RAM. Jika input tidak ada, `BLOCKED_INPUT`; jika dukungan atau feature variation tidak cukup, `SKIPPED_SUPPORT`; jangan mengubah minimum setelah melihat label/score.
2. **Implementasi.** Salin fungsi source yang valid ke adapter/extension baru. Parameter yang diwarisi ditulis lengkap ke `recipe.json`. Hyperparameter/threshold/bobot yang tidak disebut bukan ruang pencarian tambahan.
3. **Tes dan dry-run.** Jalankan syntax check, tes yang relevan dan synthetic aggregation. Masalah sebelum freeze boleh diperbaiki lalu tes ulang. Tidak ada official fit sebelum langkah ini lulus.
4. **Freeze.** Catat exact recipe, source/input/dependency hashes, fit/scoring rows, support/fallback rule, policy dan gate **sebelum** skor pertama. HEAD hanya konteks; frozen content dan data hashes menjadi identitas.
5. **Pilot.** Jalankan hanya fit yang dijanjikan. Simpan setiap hasil segera. Catat konfigurasi, successful fits, unsaved attempts dan technical retries secara terpisah. Tidak ada outer early stopping, threshold sweep atau tambahan bobot dari hasil pilot.
6. **Verifikasi.** Periksa fit-only vocabulary/IDF/scaler/SVD/category encoder; aligned IDs/rows; availability route; finite/range; metric primary/H12; subgroup denominators dan gate. Encoder target/meta-head harus membuktikan score/label chronology. Solver warnings dicatat, bukan disembunyikan.
7. **Verdict.** `ADVANCE` hanya jika seluruh gate lulus; `RETIRE` jika evidence performa valid gagal; `REWORK` jika correctness belum terbukti. Simpan semua hasil, termasuk yang buruk. Tidak ada perubahan champion atau academic notebook otomatis.

## Definisi view dan mapping

STANDARD: cutoff 55 dan 67 memprediksi 12 cycle masing-masing. Pilot primary adalah W1 (56–59) + W4 (68–71), 9.064 rows. H12 memakai seluruh 38.588 rows, bukan data konfirmasi independen. Continuation menambah cutoff 59/63/71/75 sehingga primary enam window mempunyai 38.588 rows unik.

LATE: cutoff 67 dan 75; pilot primary **W4+W6**, H12_L 68–79, early H12 exact fallback. Denominator dihitung aktual. Kartu late tidak boleh meminjam denominator 9.064 atau mengaku full-six. Sebelum continuation, bekukan support/fallback di semua enam cutoff; jalankan hanya origins yang memenuhi aturan tersebut.

`rank_w(raw)` pada predictor berskala target: dalam **seluruh cycle yang tersedia**, `percentile = 100 * average_rank(raw) / n_cycle`, lalu `clip((1-w)*raw + w*percentile, 0, 100)`. Ties memakai average rank. Reproduksi clipping/order persis seperti helper kontrol bila mengganti komponen lama. Catat kebutuhan full-cycle batch; jangan memakai target validation untuk mapping.

Ranker R12/R13 mempunyai raw score tanpa satuan target: gunakan **pure percentile** seluruh cycle, lalu fixed blend di kartu. Specialist EF/missing memakai **raw clipped target-scale prediction**, tanpa rank per-subset. EF adalah parsed `entry_friction`, bukan `telemetry_flag`.

R17/R18/R19 hanya memakai exact prequential incumbent OOF dari cycle 56 ke atas. Setiap score harus dibuat oleh fit yang berakhir sebelum cycle labelnya, dan semua label calibrator ≤ outer cutoff. Jangan membuat skor current champion 52–55 dari outcome-selected historical weights. Jika cache tersebut hilang atau lineage salah, catat BLOCKED_INPUT. R18/R19 canonical memakai median residual, bukan mean pada draft lama.

Cache registry konkret ada di `queue.json`. Classification memakai `classification_round1_v1/fit{end}__C2_B.npz`, `dual_cls_catboost_v1/fit{end}.npz`, dan sparse pilot55/67 atau full59/63/71/75. Validasi kontrol Round1 melalui `_load_round1_control` beserta fit audits. Regression memakai `dual_reg_v1/stage1_paired_blends.npz` dan dua cache `optimization_v2_regression_repr`. **LGB/Ridge pada cache pertama sudah rank-mapped; jangan dipetakan dua kali.** Raw sparse pada cache kedua memerlukan satu rank50. Jangan menyamakan row order pooled windows dengan urutan global Train; selalu join dengan row indices dan periksa IDs.

Guard Macro F1 subgroup hanya berlaku bila n≥200 serta masing-masing kelas mempunyai ≥30 baris. Kelompok satu kelas/dukungan kecil tetap dilaporkan deskriptif, bukan dipaksa menjadi evidence F1. Semua tie bins/denominator kosong tetap eksplisit.

## ONE_CHAT_END_TO_END: jangan berhenti di pilot

Luna menjalankan seluruh 45 kartu bersyarat, lalu continuation, finalist QA/refit/submission bila layak, dan laporan akhir. **24 ID pertama hanya prefix prioritas, bukan stop point.** Tidak ada persetujuan Sol/user di antara ID atau milestone internal. Kandidat yang gagal secara valid ditutup dengan verdict negatif, lalu chat melanjutkan lane berikutnya.

1. Jalankan pilot sesuai urutan canonical, condition dan family budget. Jangan menciptakan konfigurasi tambahan. Simpan ADVANCE pool, RETIRE dan SKIPPED; jangan mengganti kontrol di tengah batch.
2. Setelah screening, seleksi continuation **menurut profile**: maksimal dua STANDARD/task dari W1+W4/H12; secara terpisah maksimal satu LATE/task dari W4+W6/H12_L. Urutkan hanya dalam profile yang sama menurut gain primary, gain H12, lalu jumlah window nonnegatif. Utamakan family berbeda. Jangan membandingkan raw pilot gains dari mask berbeda.
3. CACHE_SIX sudah mempunyai view enam window dan tidak memerlukan continuation fit. Semua kandidat baru boleh dibandingkan bersama hanya sesudah dievaluasi pada **primary enam window yang sama**, dengan support/fallback rule tetap. Late-only evidence belum cukup untuk promosi.
4. Jalankan continuation yang lolos dengan recipe tetap, verifikasi, kemudian lakukan tahap finalist di bawah. Hasil buruk bukan alasan membuka grid rescue atau berhenti di tengah antrean.
5. Default budget 180 successful candidate predictor fits dan 24 control/verification refits, maksimal dua pilot configurations per task+family. Catat seluruh transform/inner/training calls dan retries. Sembilan ordinal heads berarti sembilan predictor fits per origin.
6. **Reserve seluruh fit kartu sebelum mulai.** Jika sisa budget tidak cukup, SKIP_BUDGET sebelum fit pertama. Seluruh backlog memiliki 98 pilot fits; prefix 24 ID memiliki 44. Budget bukan target yang harus dihabiskan.
7. Satu job training dua thread. Tambahan memory harus muat pada free RAM dikurangi 2 GiB dan cap job 6 GiB. Jangan densify word TF-IDF penuh. Jika tidak muat, SKIP_RESOURCE dan lanjut lane independen.
8. Tidak ada hard stop enam jam secara default. Sekitar delapan jam, simpan checkpoint dan lanjut bila sehat. Batas waktu keras hanya bila user memilihnya atau resource nyata membatasi. Setelah satu fit melewati 15 menit, periksa progres/resource; jangan menyatakan gagal hanya karena observation timeout. Jika benar-benar macet/terbatas, minta graceful interruption pada proses milik sesi, simpan attempt/incomplete state, lalu lanjut independen. Jangan hentikan proses pengguna lain.

### SINGLE_LANE juga harus selesai dari awal sampai akhir

Jika user memilih C03, misalnya, selesaikan family `vocab_allocation` Classification (C03). Jika memilih C16, family `sparse_capacity` mencakup C16 dan C17, dengan maksimal dua recipe yang sudah dideklarasikan. Alurnya: implementasi → tests → freeze → pilot → continuation bila lolos → QA → qualified finalist stage → laporan lane. Jika semua recipe gagal atau support tidak cukup, tutup dengan bukti negatif/skip yang jelas. Jangan berhenti untuk meminta review setelah satu pilot; jangan membuka family lain di mode ini.

## Tahap finalist: refit, inference dan Kaggle sesuai otorisasi V3

Hanya kandidat yang lolos full-six, H12, supported-group guards dan source QA yang masuk tahap ini. Semua gagal berarti champion lama dipertahankan, tidak ada upload, dan laporan riset tetap dituntaskan.

- Bekukan recipe dan source identity; commit source/model preparation yang sudah valid sebelum full-Train fit. Refit komponen dari official Train sesuai population/fallback kartu, tanpa mempelajari preprocessing dari Test. Record input Train tetap 50.608 rows; fit subset yang disengaja tetap dicatat sebagai subset.
- Official Test hanya dibaca untuk inference setelah qualification. Validasi tepat 29.421 ordered IDs sesuai SampleSubmission, tanpa duplicate/missing, kolom tepat; Classification integer 0/1 threshold 0,50; Regression finite [0,100]. Simpan manifest, dependency/source commit/input hashes, fitted-state audit, prediction/CSV hashes dan perbandingan terhadap kontrol.
- Reuse source/API safety implementation yang sudah terverifikasi melalui wrapper namespace baru; jangan mengubah accepted CSV/manifest atau frozen source lama. Fit/prediction cache dan CSV tetap ignored. New external decision/manifest memakai prefix `batch1_`.
- Otorisasi submission berasal dari objective V3 yang sudah ada. Periksa fresh authenticated rules, team, deadline, shared daily allowance, submission history dan budget kampanye sebelum upload. **V3 sudah memakai satu Classification submission ref57837821**; jangan mereset pemakaian budget ketika memakai namespace baru. Baca `experiments/optimization_v3/milestone.json` dan seluruh ledger V3.
- Batas dua initial submissions merupakan instruksi milestone lebih lama. Objective V3 yang lebih baru secara eksplisit menyatakan “Submission boleh dilakukan beberapa kali sepanjang sesuai live Kaggle limits dan aturan tim” serta mengotorisasi submission kandidat kompetitif secara otonom. Rekonsiliasi dan kutipan ada di `experiments/optimization_v3/research_backlog/authorization_note.json`; ini bukan otorisasi upload sembarang atau bukti kuota tersedia.
- Evaluasi terbaik qualifying finalist per task terlebih dahulu. Upload tambahan harus berbeda secara substantif dan memiliki alasan internal kuat serta budget/kuota yang dibekukan, bukan weight/threshold probing terhadap public score.
- Simpan exclusive idempotent reservation sebelum upload, jangan mengulang upload ambigu. Konfirmasi status COMPLETE dan public score dari Kaggle sebelum menyatakan berhasil. Internal primary/H12 dan public dilaporkan terpisah.
- Jika auth, aturan, deadline atau kuota tidak memungkinkan, simpan CSV/manifest dan blocker yang tepat; jangan mencari credential secara tidak aman. Lanjutkan lane independen dan tuntaskan laporan, tanpa mengklaim submission berhasil.

### Kapan lanjut, rework, atau berhenti seluruh batch

| Keadaan | Tindakan otomatis |
|---|---|
| Pilot valid, gate gagal | RETIRE recipe, simpan, lanjut ID berikutnya |
| Pilot valid, gate lolos | ADVANCE candidate pool, lanjut pilot berikutnya; continuation pada akhir screening |
| Condition/support/RAM tidak terpenuhi | SKIPPED dengan alasan dan denominator, lanjut ID independen |
| Controller/report bug setelah fit tersimpan | Pertahankan source/outputs; buat amendment + adapter report versi baru, tanpa refit; verifikasi ulang |
| Bug sebelum fit | Perbaiki source belum frozen, tes ulang; jika tetap gagal dua upaya terarah, REWORK ID dan lanjut independen |
| Source/input/hash/row leakage mismatch | Hentikan fit pada dependency tersebut; bila menyangkut kontrol/global input, STOP seluruh batch dan checkpoint |
| Output partial | Verifikasi pasangan cache/sidecar; reuse jika lengkap dan cocok. Jangan overwrite/hapus untuk memaksa restart |
| User-selected hard deadline / fit cap / resource nyata | Jangan mulai kartu/fit baru; simpan checkpoint dan laporan konsolidasi |

Selama pilot dan continuation: tidak ada Test/Kaggle access. Pada tahap finalist: inference/upload hanya sesuai gate di atas. Sepanjang chat: tidak ada perubahan notebook/EDA resmi atau pekerjaan rekan. Jangan berhenti hanya karena satu worker selesai, satu family gagal, satu upload selesai atau public rank 1 tercapai; lanjutkan scope antrean yang belum tuntas.

## Output yang harus ditemukan saat user bangun

Satu `reports/optimization_v3/batch1/BATCH_FINDINGS.md` dan `batch_ledger.json` berisi seluruh ID: NOT_STARTED/SKIPPED/RETIRED/REWORK/ADVANCE/FULL_SIX_PASS/FAIL; source/freeze/input IDs; actual fits/attempts/time; primary dan H12 dengan view yang benar; subgroup/convergence counterexamples; active verifier dan next action. Tautkan per-ID results/verification. Angka dari laporan lama harus diberi provenance; jangan diisi sebagai hasil fit baru.

Commit hanya path milik batch yang coherent dan terverifikasi. Push secara aman sesuai otorisasi project; jangan force-push/merge/reset/rebase atau mengubah milik rekan. Jika push gagal, simpan commit lokal dan laporkan. Pertahankan working tree resumable. V3 **tidak** otomatis COMPLETE saat satu batch berakhir.

## Prompt satu lane

Setiap [kartu](CATALOG.md) mempunyai prompt dengan family dan ID terisi. Contoh:

> Lanjutkan samindriano/tk2-kasdad memakai Luna MAIN dengan orchestra seperti biasa. Kerjakan lane vocabulary allocation Classification, C03, sesuai `reports/optimization_v3/research_backlog/cards/C03.md` dan `reports/optimization_v3/research_backlog/LUNA_HANDOFF.md`, mode SINGLE_LANE. Selesaikan implementasi, tes, freeze, pilot, full-six bila lolos, QA serta full-Train inference/submission jika seluruh gate V3 terpenuhi. Jangan meminta review atau berhenti di tiap milestone. Bila evidence valid gagal, retire dan tuntaskan laporan lane. Pertahankan seluruh frozen work/EDA/notebook/rekan; commit dan push hasil valid. Akhiri setelah lane selesai, tanpa menganggap seluruh kampanye V3 otomatis selesai.

## Prompt antrean malam siap tempel

> Lanjutkan V3 yang belum selesai di repository samindriano/tk2-kasdad memakai Luna MAIN dan orchestra seperti biasa. Jalankan mode ONE_CHAT_END_TO_END dari `experiments/optimization_v3/research_backlog/queue.json`, menurut `reports/optimization_v3/research_backlog/LUNA_HANDOFF.md` dan kartu per-ID. Verifikasi HEAD/status aktual dan pertahankan checkpoint cb27597 serta seluruh frozen work. Bangun dan uji shared harness B0 jika belum ada. Proses seluruh 45 recipe bersyarat secara berurutan; 24 ID awal hanya prioritas, bukan stop point. Jangan meminta approval Sol/user atau berhenti setelah satu pilot, family, worker, continuation atau submission. Terapkan gate tetap, retire/skip yang gagal secara valid, perbaiki hambatan teknis secara aman dan lanjut lane independen. Seleksi continuation menurut profile yang sama: dua STANDARD dan satu LATE per task; bandingkan finalist pada full-six yang sama. Selesaikan QA, qualified full-Train refit/inference dan submission sesuai otorisasi V3 dengan fresh live rules/team/deadline/kuota, global budget dan idempotency checks; jangan spam atau memakai public score sebagai search. Worker Luna boleh mengerjakan scope independen; MAIN memegang integrasi/Git/Kaggle dan satu job training dua thread. Budget yang dideklarasikan 180 candidate predictor fits serta 24 control/verification refits, tanpa hard stop waktu arbitrer; checkpoint dan lanjut saat context berganti. Jangan mengubah EDA/notebook/rekan atau overwrite frozen evidence. Commit/push pekerjaan valid secara berkala. Tuntaskan laporan seluruh ID, skor internal/public, champion/fallback, QA, retries, blocker dan commit terbaru. Berhenti hanya ketika alur antrean selesai atau ada batas nyata yang terdokumentasi. Jangan menandai goal V3 COMPLETE hanya karena satu lane/batch selesai; lakukan audit objective aslinya terlebih dahulu.
