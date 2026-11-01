# Panduan eksekusi Luna â€” Research Next V1

**Entry point operasional.** Queue ini menggantikan bagian execution suggestions pada [catatan riset awal](RESEARCH_LANES.md). Catatan awal tetap evidence/penjelasan; jika recipe atau biaya berbeda, [queue.json](../../../experiments/optimization_v3/research_next_v1/queue.json) dan panduan ini yang berlaku. Semua lane belum diimplementasikan/dilatih ketika paket disusun. Input dan API sudah diperiksa; itu bukan klaim adapter sudah lolos Run All.

## Scope dan authorization saat prompt diluncurkan

Satu chat Luna menyelesaikan **N00 â†’ N01 â†’ N02 â†’ N03 â†’ N04 â†’ N05 â†’ N06**, lalu continuation terbatas sesuai aturan di bawah, QA dan handoff akhir. Tidak berhenti meminta persetujuan per lane, worker atau pilot. N07/FM dan N08/neural **bukan execution queue** karena backend belum diverifikasi; jangan memasang Torch atau mengganti scope untuk mengejarnya.

Prompt pengguna mengizinkan maksimal **60 predictor/auxiliary fits baru** untuk fase ini. Ini budget tambahan fase baru, bukan reset180 atau izin memperbaiki ulang model kampanye lama. Jangan mengubah global counter lama. Catat hash dan total ledger lama ketika start; append `BATCH_ALLOWANCE_AUTHORIZED` pada ledger baru dengan inherited-usage snapshot. Seluruh penggunaan lama dan baru tetap dapat dijumlahkan.

Alokasi maksimum:26 fits seluruh pilot termasuk matched controls;24 continuation;6 control/verification replays yang benar-benar diperlukan;4 technical retries setelah estimator mulai. Semua domain classifiers dan quantile/router heads dihitung. Synthetic unit-test fits dicatat sebagai QA terpisah, tidak disebut official-data research fits. Pre-fit dispatch error tidak menghabiskan predictor fit, tetapi event tetap dicatat. Budget bukan target yang harus dihabiskan.

**Tidak ada Test, SampleSubmission, Kaggle API/browser/upload, full-Train79 refit atau perubahan notebook dalam fase ini.** Izin benchmark tujuh model sebelumnya tidak diwariskan ke lane baru. Jangan membuat submission meskipun kandidat lolos; berhenti pada research-review checkpoint.

## Startup tanpa bentrok atau tergantung auth lama

1. Paling mudah jalankan di chat Luna yang sekarang, sesudah pekerjaan benchmark lama mencapai checkpoint aman. Jika chat lain masih menguasai slot training/Git, jangan ikut menulis source/index atau memulai job. Koordinasikan lewat handoff/ownership; jangan kill proses lain atau hapus lock.
2. Verifikasi root/remote/HEAD/status aktual. Banyak untracked expanded reports adalah pekerjaan agent lama, bukan milik queue baru. Jangan stage/reset/clean file tersebut. Main hanya stage daftar path milik Research Next secara eksplisit. Jangan push tanpa instruksi tambahan.
3. Checkpoint lama boleh terminal dengan blocker terdokumentasi (misalnya auth Kaggle tidak tersedia atau R11 continuation hilang). Itu tidak otomatis memblokir research baru. Namun **N03 wajib memakai R06 yang sudah diverifikasi secara independen**. Report final yang sekarang tersedia adalah `reports/optimization_v3/expanded_benchmark_v1_final_evaluation/R06/{results,verification}.json`, bukan direktori recovery tempat attempt log disimpan. Jika report itu tidak ada/berbeda, N00 menghitung ulang rank/blend/metric/guard dari cache secara independen tanpa refit dan menyimpan receipt baru. Jangan menganggap hash check saja cukup atau mencoba memulihkan R11 menggunakan budget ini.
4. Panggil `python -m scripts.optimization_v3_research_next_preflight` untuk membaca data/cache/API. Ia tidak memanggil loader context training lama dan tidak memeriksa sisa budget175 sebagai syarat membaca cache. Snapshot [queue_readiness.json](queue_readiness.json) sudah membuktikan18 bundle dan semua pasangan enam-origin R22/R06 tersedia/aligned; cek ulang karena file dapat berubah.
5. Freeze Class control **sparse25** dan Reg control **R22_ORIGINAL**, plus protected legacy Reg baseline, untuk seluruh batch ini. Jangan mengganti control menurut public score atau karena lane sebelumnya menang. Semua kandidat juga dilaporkan terhadap R06_ORIGINAL sebagai pembanding sekunder dari cache, tanpa menjadikannya control baru secara diam-diam.

Hasil public terbaru boleh disimpan sebagai observasi terpisah; jangan digunakan untuk menentukan formula/threshold/seed/urutan. Queue dibuat sebelum inspeksi hasil public terbaru, tetapi hasil tersebut sudah diketahui sebelum implementasi; jangan mengklaim riset berikutnya outcome-blind.

## Sumber dan namespace yang boleh dipakai

- Queue: `experiments/optimization_v3/research_next_v1/queue.json`.
- Kartu: [cards/N00.md](cards/N00.md) sampai [N06](cards/N06.md). Baca satu kartu saat mulai mengerjakannya, tidak perlu memuat seluruh sejarah tiap turn.
- Source baru: `src/models/research_next_*.py`, `scripts/optimization_v3_research_next_*.py`, `tests/test_research_next_*.py`.
- Frozen protocol baru: `experiments/optimization_v3/research_next_v1/frozen/{ID}/`.
- Output ignored: `experiments/runs/optimization_v3_research_next_v1/{ID}/`.
- Report baru: `reports/optimization_v3/research_next_v1/results/{ID}/` plus satu `CAMPAIGN_SUMMARY.md` dan ledger ringkas seluruh ID.
- Raw Train melalui `src.utils.paths.task_files` dan hashes queue. Fitur wajib allowlist lama, tanpa record_id/target task lain. ID hanya alignment/receipt.
- Historical source boleh dibaca/reuse; **tidak diedit**. Wrapper sparse Class dan R22 lama tidak menerima `sample_weight`; buat adapter baru, bukan monkeypatch. N02 tidak punya accessor public matrix; copy/wrap construction secara eksplisit dan buktikan no-sketch equivalence pada synthetic fixture.

### Registry cache enam origin

| Komponen | 55/67 | 59/63/71/75 |
|---|---|---|
| R22 | `experiments/runs/optimization_v3_batch1/R22/R22_fit{E}.npz` | `experiments/runs/optimization_v3_final_continuation/R22/R22_fit{E}.npz` |
| R06 | `experiments/runs/optimization_v3_batch1/R06/R06_fit{E}.npz` | `experiments/runs/optimization_v3_expanded_benchmark_v1_recovery_v2/R06/R06_fit{E}.npz` |
| Class control bridge (C16 bundles) | `experiments/runs/optimization_v3_batch1/C16/C16_fit{E}.npz` | `experiments/runs/optimization_v3_final_continuation/C16/C16_fit{E}.npz` |

Semua NPZ mempunyai JSON sidecar dengan hashes. **Schema R22 berbeda:** pilot memakai `prediction`/`baseline_prediction`; continuation memakai `R22_ORIGINAL`/`champion_prediction`. `raw_candidate` tersedia di keduanya. Preflight sudah memuat normalisasi eksplisit tersebut; jangan mengandalkan nama key generic di semua artifact. C16 bridge menyediakan baseline sparse25, C2_B, CB dan raw sparse. Jangan menganggap prediction C16 adalah Class control; yang dipakai adalah `baseline_prediction` di bundle C16.

Pilot55/67 menyimpan12cycle. Untuk OOF training head, ambil hanya primary block56â€“59 dari55 dan68â€“71 dari67; pakai origin khusus lainnya untuk60â€“63/64â€“67/72â€“75/76â€“79. Jangan concatenate seluruh H12 pilot ke OOF primary. Join berdasarkan row indices+record IDs, bukan posisi array mentah. Rank memakai cycle lengkap sebelum filtering availability. Old control vectors yang sudah mapped jangan dirank dua kali.

## B0 â€” implementasi dan kontrak sebelum training resmi

Pakai [research_next_contract.py](../../../src/models/research_next_contract.py) untuk payload baru. Ia sudah mempunyai unit tests untuk dict diagnostics, JSON tuple/list, string IDs, raw scores di luar rentang final, publish atomik, tamper detection dan exact resume. Ia **bukan** runner/model adapter lengkap. Tugas Luna adalah menghubungkan adapter ke kontrak dan membuktikan E2E per lane.

```python
from src.models.research_next_contract import publish_bundle, read_bundle
# arrays and diagnostics stay in ONE payload; mappings are moved to strict JSON.
receipt = publish_bundle(bundle_dir, payload, stable_identity, task=task)
saved = read_bundle(bundle_dir, stable_identity, task=task)
```

Required arrays: `score_rows` integer unik/nonnegative; `score_record_ids` string unik; `prediction`/`baseline_prediction` finite dan range task. Raw predictions finite tetapi boleh di luar[0,100]; jangan mengulang bug validasi raw sparse R21. Dict diagnostics masuk metadata JSON, **tidak dipaksa menjadi object ndarray**. Array mentah tidak boleh pickle. Untuk N05 information, `prediction` boleh exact R22 reference, dengan `raw_q10`/`raw_q90` terpisah; jangan mempromosikan point prediction yang sengaja tidak berubah.

Bundle berupa direktori final berisi `arrays.npz` dan `receipt.json`, dipublikasikan dari staging lengkap. Tidak ada NPZ final tanpa receipt. Resume cukup `read_bundle` dengan identity yang sama, **tanpa refit**. `resume_or_publish_bundle` digunakan hanya jika caller memang sudah mempunyai payload yang sama; jangan membangun payload lewat training ulang hanya untuk memeriksa resume.

### Checklist wajib SETIAP adapter

1. Signature/schema: dispatch task/ID/cutoff benar; new adapter mendukung seluruh cutoffs yang diperlukan, tanpa mengubah class pilot lama. Check weight forwarding untuk N01/N04 dan alpha untuk N05.
2. Synthetic fitâ†’predictâ†’partitionâ†’publishâ†’readâ†’independent reconstructionâ†’metricâ†’group supportâ†’gateâ†’summary. Bukan hanya syntax check. Fixture memuat dict diagnostics, Unicode IDs, tied/empty bins, unknown categories, all-missing numeric, missing text, subset fallback, dan skor raw luar range final.
3. Persistence failure injection sebelum official data: publication terhenti tidak menulis final partial; payload tetap bisa disimpan tanpa refit. Simulasikan resume sesudah satu origin selesai; simulasikan tuple/list JSON roundtrip.
4. Independent verifier tidak boleh sekadar memanggil helper metric/composition model yang sama dua kali. Implementasikan MacroF1 melalui confusion counts, MAE melalui absolute loss, rank melalui independent tied-sort atau scipy reference, dan masks dari source rows.
5. Negative leakage test: perubahan outer labels tidak boleh mengubah fitted state/predictions; metadata source rows/target digest fit tercatat. Model adapter tidak menerima outer y.
6. Compute conservative peak untuk fit+score matrices, FFT/sketch, domain folds dan solver temporary. Jangan memakai232MiB sketch sebagai total RAM. Uji max-origin dimensions sebelum menghabiskan pilot bila feasible lewat target-blind shape estimate.
7. Source/recipe dan stability check: panggil preflight dua kali dengan RAM/ledger snapshots yang berbeda; **stable identity harus sama**. Live RAM, timestamp, total reserved, timer dan retry number bukan identity recipe. Mereka di receipt execution context. Budget key tetap, tidak membuat reservasi baru akibat field dinamis.

**Hard rule:** bila save/schema/report gagal pertama kali, jangan mulai origin berikutnya dengan jalur yang sama. Simpan model/raw scores di memory atau durable recovery staging segera; perbaiki serialization/report via amendment baru dan ulangi publikasi/verifikasi, bukan estimator fit. Bila proses sudah mati dan output tidak ada, catat started-unsaved fit sebagai terpakai, lakukan maksimal satu retry terarah sesudah synthetic fix lulus dan cukup4-fit retry allowance. Tidak ada loop retry tanpa batas.

Jika shared helper berubah setelah card freeze, jangan mengubah identity lama supaya lolos. Audit diff, simpan freeze/output lama dan buat source wrapper/freeze baru sebelum estimator berikutnya; buktikan tidak ada recipe change. Ini bukan alasan mengulang semua fit yang sudah valid.

## Urutan execution dan status

1. **N00**: input/API readiness, QA cache independen, atlas error seluruh controls dan valid expanded candidates. Salah satu historical candidate tidak punya bundle/QA bukan alasan merefitnya. N03 arms A/B harus dihitung ulang sesuai policy queue, khususnya missing fallback A=R22. Old residual-head failure adalah counterexample utama: shallow OOF head bukan jaminan berhasil.
2. N01 sampai N06 berurutan. Untuk tiap ID: implementâ†’synthetic testsâ†’preflightâ†’freezeâ†’reserve seluruh fit pilotâ†’fit serialâ†’publish perorigin segeraâ†’verifyâ†’report verdictâ†’lanjut kartu berikutnya. Tidak meminta review tiap ID. N04 boleh `SKIPPED_SCIENTIFIC_PREFLIGHT` sesudah domain fits bila AUC/overlap/ESS gagal; domain fits tetap dihitung. N05/N06 harus menyelesaikan semua konfigurasi info yang sudah ditetapkan, tidak memilih seed/quantile pemenang.
3. Setelah seluruh pilot selesai, pilih continuation maksimal **satu Classification dan satu Regression**. Class: di antara N01/N02 yang lolos semua gates, urutkan primary gain, H12 gain, jumlah window nonnegatif; exact tie N01. Reg: N03 jika lolos LATE+matched-control; jika tidak, N04 jika lolos STANDARD+matched-control. Prioritas ini ditetapkan dahulu karena profile berbeda, bukan ranking raw gain lintas profile.
4. N01/N02 continuation menambah4 candidate+4 matched-control fits (8). N03 menambah heads59/63/71 (3), origin55 exactA fallback, pilot67/75 reused. N04 menambah8 domain+8 task fits (16). N05/N06 tidak otomatis continuation/promosi point model. Maksimum continuation8+16=24.
5. Semua hasil disimpan: `PILOT_PASS_NOT_SELECTED`, `RETIRED_PERFORMANCE`, `COMPLETE_INFORMATION`, `SKIPPED_SCIENTIFIC_PREFLIGHT`, `TECHNICAL_REWORK`, `DEFERRED_RESOURCE`, atau full-six qualified/not. Jangan menyebut skip/info sebagai model gagal performa.

## Gate dan denom yang tidak boleh ambigu

STANDARD pilot:55/67, primary W1+W4 n9.064; H12 n38.588. LATE pilot:67/75, primary W4+W6 n13.441; H12_L n20.903. Raw gain dua profile tidak boleh diranking langsung. H12_E descriptive untuk LATE;55 exactfallback tidak menjadi hasil baru.

Full-six primary pooled38.588 baris UNIK:55â†’56â€“59,59â†’60â€“63,63â†’64â€“67,67â†’68â€“71,71â†’72â€“75,75â†’76â€“79. W5+W6 pooled15.953. H12 tetap dua origins55/67 dan bukan evidence independen. N03 full H12_E exactA, H12_L head67.

- Classification pilot/full primary gainâ‰¥0.0005; H12 harmâ‰¤0.001; individual primary-window harmâ‰¤0.005. Supported groups nâ‰¥200 danâ‰¥30/kelas: F1 harmâ‰¤0.01, positive recall harmâ‰¤0.02. Fields EF, stream status, fit-defined text_load quartiles. Candidate primary harusâ‰¥0.0002 lebih baik dari matched control. Full:â‰¥4/6 windows nonnegatif dan pooled W5+W6 gain>0.
- Regression pilot/full primary gainâ‰¥0.03; H12 harmâ‰¤0.05; individual primary-window harmâ‰¤0.5. Supported missing/actual yâ‰¤5/yâ‰¥95 nâ‰¥100: harmâ‰¤0.25 terhadap CURRENT R22 dan protected legacy baseline. Actual target-tail hanya mask evaluasi, bukan fitur/router saat inference. Primary candidate harusâ‰¥0.01 lebih baik dari matched control. Full:â‰¥4/6 windows nonnegatif dan pooled W5+W6 gain>0.
- Scope **primary,H12,dan setiap declared individual primary window**; full-six berarti W1..W6. Unsupported/null explicit; jangan menghitung safety flags `test_used=false` sebagai failed gate. N05/N06 tidak dikenai score-promotion gate, tetapi tetap mendapat metric/guard diagnostics.

## Resource, budget dan error yang tidak mematikan seluruh queue

Satu job dua thread; cap6GiB dan minimum free4GiB, reserve2GiB. Check sebelum setiap fit; jangan memaksa fit bila estimasi lebih besar dari freeâˆ’reserve. Bila satu lane belum muat, simpan `DEFERRED_RESOURCE`, kerjakan independent preflight/report lane lain, lalu revisit deferred sekali sesudah pilot lain. Jangan menurunkan vocabulary, sketch dimension, tree count atau min support sesudah melihat skor hanya agar jalan. Jika tetap tidak muat, dokumentasikan batas nyata; tidak mengklaim lane dijalankan.

Reservation sebelum fit adalah untuk seluruh card, termasuk kedua matched controls. Key `(batch,card,stage)` stabil dan idempotent. Append events `FIT_STARTED` tepat sebelum estimator.fit dan `FIT_SAVED` hanya sesudah bundle+independent verification tersimpan. Skipped pre-task N04 boleh merekonsiliasi unused reserved slots tanpa menggandakan counters; ledger harus membedakan reserved, consumed, unused, technical attempts dan remaining capacity. Tidak memakai selisih dua snapshot sebagai bukti jumlah fit baru.

Jika total cap60 tidak cukup, jangan mengambil budget lama, mengurangi resep diam-diam, atau meminta persetujuan perfit. Selesaikan jobs yang bisa dilaksanakan sesuai order/reservation, tandai `SKIPPED_BUDGET`, lalu buat handoff. Shared raw/control corruption adalah STOP dependency; masalah satu lane yang terisolasi tidak menghentikan lane lain.

## Deliverable dan stop point

Satu summary kampanye menjelaskan seluruh7 ID, matched controls, gates, support, actual fits/time/retries/consumedbudget, per-window/H12/late, source/lineage, QA dan next recommendation. Simpan JSON/CSV perlane dan hanya grafik yang menambah informasi; tidak membuat notebook akademis/HTML besar. N05 uncertainty coverage dan N06 dua bijeksi tetap dilaporkan meskipun score buruk.

Main melakukan commit lokal berkala terhadap path Research Next saja sesudah valid; workers tidak Git mutate. Jangan campurkan expanded benchmark lama. Tidak push tanpa instruksi tambahan. Setelah pilot+continuation terpilih+QA selesai, berhenti untuk review. Jangan lanjut final refit/Test/Kaggle atau menandai seluruh V3 complete.

### Satuan budget fit

Cap 60 menghitung panggilan training estimator pada data resmi, termasuk domain classifier, router, setiap quantile head, dan fit yang sudah dimulai tetapi belum tersimpan. Learned preprocessing (imputer/vectorizer/SVD/sketch) dicatat terpisah dan tidak masuk cap 60 estimator-head; tetap wajib fit-only dan diperhitungkan untuk waktu/RAM. Fit estimator synthetic dicatat sebagai QA terpisah. N00 memakai daftar kandidat dan syarat verification eksplisit pada queue; 18 bundles preflight bukan QA seluruh kandidat expanded.
