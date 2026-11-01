# Prompt siap pakai — Research Next V1

Salin blok ini ke **satu chat Luna**. Jika memakai chat Luna kampanye lama, biarkan ia menyimpan checkpoint lamanya dahulu. Prompt ini mengizinkan budget baru ketika pengguna meluncurkannya; paket dokumen sendiri belum menjalankan fit atau mengubah ledger.

```text
Lanjutkan proyek KASDAD TP2 di:
C:\Users\Sam\OneDrive\Documents\KASDAD\TP 2

Saya mengizinkan Research Next V1 dengan maksimal60 predictor/auxiliary fits
BARU sesuai alokasi queue. Ini budget tambahan fase baru, bukan reset budget
historis, dan bukan izin memakai budget baru untuk retry kampanye lama.

Jalankan satu chat end-to-end memakai Luna MAIN dan aturan orchestra proyek.
Baca AGENTS yang berlaku, lalu entry point berikut:

reports/optimization_v3/research_next_v1/LUNA_HANDOFF.md
experiments/optimization_v3/research_next_v1/queue.json
reports/optimization_v3/research_next_v1/queue_readiness.json

Kerjakan seluruh queue N00→N01→N02→N03→N04→N05→N06.
N07/FM dan N08/neural tidak termasuk scope; jangan menginstal Torch atau
menciptakan kandidat tambahan. Baca kartu tiap ID saat mengerjakannya.

Jika pekerjaan benchmark sebelumnya masih aktif, selesaikan/simpan checkpoint
aman pekerjaan itu terlebih dahulu. Preserve semua bundle/freeze/report,
termasuk blocker R11. Jangan membuat training/Git writer kedua yang bertabrakan.
Auth/submission lama yang blocked tidak menghalangi research Train-only ini.

Pertahankan baseline Class sparse25 dan Reg R22_ORIGINAL selama batch;
Reg juga diaudit terhadap protected legacy baseline dan dilaporkan terhadap
R06_ORIGINAL. Hasil public terbaru dicatat terpisah, tidak menentukan recipe,
threshold, weights, seed, route, urutan atau control baru.

INPUTS DAN IMPLEMENTASI
Jalankan read-only preflight yang sudah tersedia.18 historical bundles dan
enam pasangan R22/R06 sudah dicek, tetapi cek ulang hash/source aktual.
Perhatikan alias schema R22 pilot versus continuation di handbook;
jangan memanggil training-context loader lama hanya untuk membaca cache.
Gunakan R06 verification di expanded_benchmark_v1_final_evaluation atau
buat QA independen tanpa refit jika dibutuhkan.

Bangun adapter/runner baru sesuai namespace Research Next; jangan mengedit
source/model/protocol frozen, notebook, raw data atau pekerjaan anggota lain.
Library APIs sudah tersedia; wrappers lama tidak meneruskan sample_weight,
jadi buat adapter baru secara eksplisit. N02 main-effect design harus tepat.

KONTRAK SEBELUM TRAINING RESMI
Pakai src/models/research_next_contract.py untuk output baru.
Dict diagnostics harus ke JSON, bukan NPZ object arrays.
Untuk setiap adapter, wajib synthetic fit→predict→publish→read-back→independent
reconstruction→metric→all-view guard→report, termasuk resume dan failure injection.
Jangan menganggap syntax test atau shared-contract test membuktikan adapter lengkap.

Bekukan recipe, complete input/source/dependency hashes, rows/fit-only bounds,
views/gates, support/fallback dan budget SEBELUM task score baru.
RAM bebas, timestamp dan ledger total adalah execution context, bukan recipe identity.
Reserve seluruh card secara idempotent sebelum fit; auxiliary/domain/head fits
dan started-unsaved fits juga dihitung. Jangan mereset atau menggandakan ledger.

Satu training job,dua thread,cap6GiB,reserve2GiB,live resource check tiapfit.
Jika save/report pertama gagal, JANGAN melatih origin berikutnya lewat jalur
yang sama. Pertahankan model/raw output, perbaiki dan buktikan persistence
tanpa refit bila output masih tersedia. Retry estimator hanya setelah synthetic
fix lolos, ada budget, dan sesuai batas retry yang tertulis.

TUNTASKAN QUEUE
Jangan berhenti meminta review/approval setelah tiap worker,lane,pilot atau
verdict. Gagal performa→retire recipe dan lanjut. Scientific preflight skip
atau informational result bukan technical failure. Tangani technical issue
terisolasi lalu lanjut lane independen. Shared input corruption atau batas
resource/budget nyata harus didokumentasikan, bukan dipaksa menjadi PASS.

Setelah semua pilot, continuation maksimal satu Classification dan satu
Regression mengikuti selection yang SUDAH ditetapkan di queue. Jangan
meranking gain STANDARD versus LATE pada populasi berbeda. Reuse pilot;
primary full-six disjoint38.588rows, bukan overlapping12-cycle concatenation.
N05/N06 hanya information lanes; jangan memilih seed atau mengubahnya menjadi
submission candidate otomatis.

DELIVER DAN STOP
Simpan JSON/CSV, receipts/fit accounting,independent QA, sedikit grafik yang
informatif, dan satu CAMPAIGN_SUMMARY seluruh7ID termasuk skip/fail/info.
Laporkan continuation/fallback dan matched controls, primary/H12/late/groups,
new-phase budget serta inherited historical usage. Commit lokal hanya path
milik fase ini; jangan push tanpa instruksi tambahan.

Tidak ada official Test/SampleSubmission access, Kaggle call/upload,
full-Train79 refit, external/pretrained data, notebook finalization, atau
perubahan champion otomatis. Izin tujuh submission sebelumnya tidak berlaku
untuk lane baru. Berhenti setelah pilot+selected continuation+QA selesai
pada research-review checkpoint; jangan menandai seluruh V3 COMPLETE.
```
