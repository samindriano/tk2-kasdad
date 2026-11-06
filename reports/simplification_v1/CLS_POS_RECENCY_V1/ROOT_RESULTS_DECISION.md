# Root review: satu benchmark E79 RECENCY

Tanggal 7 November 2026 WIB. Keputusan **APPROVE_EXPLORATORY_E79_BENCHMARK**, scientific verdict tetap FAIL. Root memeriksa source objective/gradient, fit-present preparation, saved score/ledger, dan laporan hasil pada HEAD704a616f76b440a076d89c5726cb0d7859f453dd. Root menjalankan `verify_score_outputs(...,write_report=False)` (PASS,186 metric comparisons,0 additional query-label reads) serta dua file tes pilot (12 passed).

Angka yang menentukan: primary unique20.903 gainF1+0.000354522 (<scientific0.0005); E67H12+0.000980991; E75W6+0.001267681. RECENCY-minus-SHUFFLED primary+0.000392590 dan E75W6+0.001666285. Semua supported guards lulus, missing persis. Counterevidence: E67W5-0.001560; cycle7/12 positif,5/12 negatif; E75 AUC sekitar-0.000006 dan efek ranking hampir datar. Jadi tidak boleh menyebut robust universal temporal mechanism. Overlapping/reused development tetap adaptif. Gate exploratory yang telah dibekukan memang lulus; bukan pelonggaran scientific gate setelah hasil.

Satu refit penuh dan CSV bernilai untuk benchmark terukur di bawah preferensi Sam yang mengizinkan small-gain submissions. Jangan mengubah recipe untuk mengejar public. Tidak ada implicit champion replacement, final-slot selection, atau izin submission tambahan.

## Scope Luna1 berikutnya

Miliki hanya `reports/simplification_v1/CLS_POS_RECENCY_E79_V1/`, `experiments/CLS_POS_RECENCY_E79_V1/`, `submissions/cls_pos_recency_e79_v1/`, dan narrowly named new source/verifier/tests. Raw data, arrays, model/CSV dan receipts row-level tetap ignored; historical pilot sources/freezes tidak diedit. Main worktree read-only; tidak ada push/merge/reset. REG tetap pause.

Izin ini mencakup source preparation, deterministic replay existing fitted transforms, **tepat satu gamma optimizer fit E79**, independent fit/prediction/CSV QA dan persiapan satu CSV. Sebelum optimizer: lengkapkan actual synthetic lifecycle source/default-path/serialization/import checks, commit source+recipe/input hashes+freeze, durable claim cap1. Jika QA/source boundaries tidak dapat dipenuhi tanpa recipe baru, kirim blocker ke root; jangan silently refit preprocessing. Tidak perlu checkpoint persetujuan seremonial untuk fit yang sudah diotorisasi ini.

E79 fit official Classification Train seluruh cycle47-79 (50.608 rows), gamma learned pada stream-present saja. Prefix labels seluruh Train legal; tidak ada evaluasi temporal/threshold/weight selection baru. Raw Train SHA tetap875552a68a3d24b0f3e1966d6ddb5671e6f54305a1c526e2f0a76747942ed91c. Bobot `max(.25,2**(-(79-cycle)/8))`, normalisasi per class present, C1,sum weighted logistic,zero-init,no intercept dan optimizer options sama persis pilot. Gunakan fungsi fit/normalisasi pilot yang dibekukan bila sesuai. `recency_base_weights` pilot membatasi origin67/75: buat helper E79 baru dan buktikan parity formula pada E67/E75; jangan memodifikasi ORIGINS/freeze lama atau menjalankan origin75 sebagai proxy79. Nol SHUFFLED fit E79, nol uniform/base/A02/vocab/IDF/scaler/OHE/preprocessing refit. Optimizer starts1 maksimum, failed/interrupted consumed; saved artifact readback tanpa refit.

## E79 source tersedia: inventory harus substantif

Kesimpulan statis bahwa runner pilot tidak mendukung79 bukan source blocker. Full-Train transform/reference sudah ada pada MAIN:

- `scripts/optimization_v3_cls_pos_global_strength_e79_gamma_v1.py::_build_train_features(root)` membangun ulang dari fitted C2B states/A02-compatible vectorizer, memverifikasi frozen full-Train base/position matrices serta z_BAG; tidak fit preprocessor.
- `_build_feature_matrices_for_rows(root,frame)` di file yang sama membuat transform inference dari fitted states.
- `optimization_v3_cls_pos_global_strength_e79_v3.py` dan R3 descendant replay mendokumentasikan full-Train parity: base50.608x20.427/3.855.766nnz logicalSHA9d47aa36ff15a0cbeead8cf36049f0a699f7618d0ea8890806a2d82140cff59b; position50.608x17.259/1.638.979nnz SHA32268550b8220dbfb43f5c92c33cd84554f0bced446d1459c5c55c3eabbf9942.

Inventory exact paths/hashes dan jalankan safe helper reuse bila memang input/source hashes cocok. Helper gamma historical juga membaca existing verified DF map; jika dipanggil, aligned/global penalty outputs **tidak dipakai** pada recency. Penalty harusI/C1. Boleh ekstrak hanya pure transform logic dengan independent parity checks jika lebih sederhana. Tidak memanggil historical run/fit/one-shot replay stages atau mengubah marker/receipt/ledgernya. Bedakan reproduksi transform dari learned-state fit.

Frozen original C2B bag coefficient/intercept menjadi offset; gamma uniform original untuk reference full champion, bukan global-strength gamma/Poisson. A02 probabilities/state original dan champion operational missing policy harus di-replay sebelum kandidat digunakan. Referensi historical champion CSV MAIN `submissions/optimization_v3/cls_a02_missing_temporal_decision_v2/classification_a02_missing_temporal_e79.csv`, SHA189a779b6100333a2fecfe443689fe7d340a331e828167221c81281a7ece4a4a. Public0.83225 sebelumnya tercatat; root belum memeriksa Kaggle ulang pada review ini.

## Application dan CSV

Izin memakai **feature-only official Test cycle80-91** dan SampleSubmission untuk schema/IDs setelah source freeze dan fit receipt lolos. Tidak ada label Test. Existing Test probability caches boleh digunakan setelah hashes/provenance/IDs/cycles/presence diverifikasi. Test tidak digunakan untuk fitting/statistik bobot/normalisasi/vocab/threshold/seleksi model.

Present `p=.5*sigmoid(z_BAG+Xthirds@gamma_recency79)+.5*original_pA02`, threshold>=.50. Missing original_pA02 dan threshold>=.53230976129477, probabilities/labels persis champion; 1.062 Test missing. Tidak ada tuning blend/gamma strength/threshold sesudah hasil pilot/public.

CSV29421ordered unique officialTest IDs, integer artifact_signal0/1, SampleSubmission schema persis; finite bounded probability artifact, no target columns passed to transforms. Independent verifier merekonstruksi bobot,class masses/ESS,objective/gradient,gamma receipt,prediction arithmetic,original champion reference,missingexact dan CSV. Laporkan jumlah labelflip vs champion/positivecounts serta source/input/csv/vector hashes. Manifest sourcecommit+input+recipe; synthetic tests mencakup zero-fit transforms, exactshape/ID order, readback no-refit, tamper dan corrected namespace imports.

Simpan satu potential CSV dan kirim hasil/QA/commit/paths langsung ke root chat01a19569-16e0-7253-aa14-57a0a3160e03 sebelum mengakhiri giliran. **Root memegang upload** setelah independent CSV review, duplicate check dan durable intent; Luna tidak upload sendiri. Ini hanya satu potential official benchmark upload, no ambiguous retries, no final-slot changes. Jika source/fit/QA gagal, jangan mengirim candidate atau rescue parameter.

Resources tetap user-revised policy: one actual heavy job,2threads,1GiBinitial/512MiBreservegamma dan512MiB/256MiBcached. Bukan blocker blanket4GiB, no scheduler/processkill/idlepoll. Root tidak mengerjakan duplikat saat Luna punya pekerjaan authorized.
