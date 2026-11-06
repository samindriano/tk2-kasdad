# CLS_POS_RECENCY_V1 — satu pilot adaptasi temporal koefisien posisi

Status: ROOT_AUTHORIZED_BOUNDED_PILOT, 7 November 2026 (WIB).

Sam meminta melanjutkan dari champion dan memusatkan pekerjaan pada CLS; REG tetap dipause. Root mengotorisasi satu pilot berikut sampai hasil terverifikasi, tanpa checkpoint kosmetik yang memerlukan balasan Sam. Ini bukan izin pencarian parameter, reset budget lama, atau submission otomatis.

## Keputusan dan bukti

Uji apakah pembobotan data terbaru memperbaiki **gamma positional** di atas bag-logit yang dibekukan. Tidak menambah cabang model/fitur, tidak mengubah A02, dan tidak memperbesar arsitektur deployment. Gamma baru menggantikan gamma uniform, bukan ditambahkan di atasnya.

Pemeriksaan terarah pada source POS/CLS/A02 dan review novelty independen tidak menemukan fit recency pada gamma POS. Ini bukan bukti census lengkap seluruh artefak. A01 lama mengubah sparse LightGBM, bukan gamma positional:

- `TP 2/reports/optimization_v3/classification_recency_a01/FINAL_SUMMARY.md`: H8, floor0.25, class-preserving weights. Primary -0.000809 (NO_GO); E67 H12 +0.001468 terhadap Sparse25 dan +0.001430 terhadap shuffled. Ini bukti campuran dan motivasi lemah, bukan jaminan.
- `TP 2/reports/optimization_v3/positional_bag_offset_v1/RECIPE.json`: gamma uniform dengan z_BAG tetap, summed logistic+C1. Original POS gate gagal. Historical `SOL_CLOSE_NEW_FRONTIER.md` menutup fit posisi saat itu. Izin baru ini secara prospektif membuka hanya satu pertanyaan temporal-weighting; tidak mengubah verdict lama.
- `CLS_POS_EF_CONTEXT_V1/FINAL_SUMMARY.md`: interaction EF menurunkan primary0.001890 dan E75/W6 0.003845; tidak diteruskan.
- `TP 2/reports/optimization_v3/bigram_position_pilot_v1/RESULTS.md` dan `cls_position_error_audit_v1/RESULTS.md`: bigram telah diuji, bukan lane baru.

Hipotesis yang dapat dipatahkan: perubahan kontribusi positional antarsiklus membuat data terbaru lebih berguna untuk mempelajari gamma. RECENCY harus mengalahkan uniform dan kontrol permutasi bobot. Kemenangan atas uniform saja tidak mengisolasi temporal alignment.

## Input dan comparator

Official Classification Train: `C:/Users/Sam/OneDrive/Documents/KASDAD/TP 2/data/raw/classification/Train.csv`, 50.608 baris, SHA256 `875552a68a3d24b0f3e1966d6ddb5671e6f54305a1c526e2f0a76747942ed91c`. Tidak ada Test/SampleSubmission/Kaggle pada pilot.

Gunakan source loader yang sudah diverifikasi dari `scripts/cls_pos_poisson_bag2_v1.py::_load_origin_context`, termasuk frozen bag logits, Xthirds, original gamma, dan original A02. Reuse original-only source/readback bindings bila sesuai; jangan mengambil replica Poisson atau joint-position probabilities. Parent input/source hashes harus diverifikasi, lalu dimasukkan ke freeze baru. MAIN worktree adalah read-only.

Reference full champion: present `0.5*sigmoid(z_BAG + X_POS@gamma_uniform) + 0.5*pA02`, threshold>=0.50; missing original pA02 dengan threshold>=0.53230976129477, exact. Replay comparator terhadap original-only cache sebelum scoring. Missing threshold historis dipilih memakai development OOF: sebut paired adaptive development, bukan validasi threshold sepenuhnya past-only.

## Recipe tunggal

Pada setiap E67/E75, ambil prefix stream-present saja, urutan baris resmi stabil. Prefix labels boleh dibaca untuk fitting dan class normalization; query labels tidak boleh dibaca sampai prediksi kedua origin dan independent prelabel QA tersimpan dan committed.

Untuk baris fit i:

`a_i = max(0.25, 2**(-(origin-cycle_i)/8))`

`w_i = a_i / mean(a_j untuk fit-present class y_j=y_i)`.

Formula H8/floor0.25 dipakai kembali dari A01, tidak dicari dari hasil POS. Total bobot setiap class sama dengan jumlah baris class semula. Tidak ada class-prior shift by construction.

RECENCY memakai w asli. SHUFFLED_WITHIN_CLASS mempermutasikan multiset w dalam masing-masing class prefix-present; `default_rng(42)` di-reset per origin, class0 lalu class1, stable original row order. Simpan permutation/weight hashes. Jangan shuffle dalam cycle (bobot cycle-only akan tidak berubah), jangan shuffle target, dan jangan memakai bobot query. Kontrol mempertahankan class mass, weight distribution dan ESS, tetapi tidak menjamin equal effective degrees of freedom atau mengisolasi semua confounding.

Fit gamma dari nol, tanpa intercept baru, di atas **z_BAG**, bukan z_POS:

`L(g) = sum_i w_i*logaddexp(0, -(2*y_i-1)*(z_BAG_i + X_POS_i@g)) + 0.5*dot(g,g)`

`grad = X_POS.T@(w*(sigmoid(z_BAG+X_POS@g)-y)) + g`.

L-BFGS-B float64: maxiter2000, ftol1e-12, gtol1e-6, maxls50, C1. Summed objective, bukan mean tanpa menyesuaikan penalti. Normalisasi bobot menjaga skala massa loss dan penalti. Optimizer success, finite loss/grad/gamma, final loss<=zero-gamma loss+1e-8 diperlukan. Dua threads. Tidak ada warm-start, grid, seed sweep, retry atau rescue otomatis setelah optimizer mulai.

Original uniform fit memakai juga missing rows yang X_POS-nya nol. Menghilangkan baris tersebut hanya menghilangkan konstanta loss dan tidak mengubah gradien, tetapi dapat memengaruhi stopping relatif. Comparator tetap original gamma yang tersimpan, bukan refit uniform baru. Uji identitas gradient tersebut secara sintetis; jangan klaim koefisien optimizer bit-identical hanya karena gradient matematis sama.

Deployment policy pilot: replace gamma_uniform dengan gamma RECENCY atau SHUFFLED, sigmoid, blend50/50 original A02; threshold/routing/missing persis comparator. Nol base model/vocab/encoder/IDF/A02 fits dan tidak ada preprocessing learned baru.

## Budget dan batas temporal

Empat predictor fits persis, urutan E67_RECENCY, E67_SHUFFLED_WITHIN_CLASS, E75_RECENCY, E75_SHUFFLED_WITHIN_CLASS. Ledger baru terpisah dengan cap4; historical counters tetap utuh. Semua optimizer starts, termasuk failure/interruption, mengonsumsi slot. Unit SAVED hanya boleh dibaca ulang, bukan fit ulang. Pre-fit loader/import abort sebelum optimizer harus dibedakan dari consumed fit; perbaikan teknis sempit diperbolehkan dengan bukti, tanpa mengganti recipe.

E67 fit cycle47-67/query68-79; E75 fit47-75/query76-79. Primary 20.903 unique rows: E67 cycles68-75 + E75 cycles76-79. Laporkan juga E67 W4/W5/W6/H12 dan E75 W6; overlap E67/E75 W6 dan H12 bukan bukti independen. Tidak mengubah horizon setelah skor terlihat.

## QA dan kriteria sebelum query outcomes

Selesaikan satu implementasi ringkas yang menjalankan lifecycle nyata: input/source checks -> durable fit claim -> artifact+receipt -> readback -> predict -> independent prelabel -> score -> independent metrics/gates. Hindari menyalin scorer EF yang memiliki key-before-assignment. Tidak perlu framework/manifest berlapis atau notebook baru.

Sebelum official fit: jalankan actual CLI/default-path/import smoke dan synthetic end-to-end semua view serta supported/unsupported subgroups, IDs, absent/duplicate source, tampered receipt, consumed-fit rejection, saved readback tanpa optimizer, dan scoring outputs. Finite differences harus memverifikasi weighted gradient; w=1 harus mengembalikan objective/gradient uniform dan missing X=0 hanya konstanta loss. Root ad-hoc synthetic check 17x5 pada 7 November menghasilkan max gradient error9.261343913991027e-10; itu bukan pengganti tes implementasi final.

Verifikasi bobot finite positif, class mass rtol1e-12/atol1e-10, equal sorted multiset/ESS antararm, ESS ratio>=0.50/class dan nontrivial weight permutation. Tolerasikan rounding numerik yang wajar; jangan menciptakan blocker karena perbedaan format/dtype yang semantik identik. Prefix-label usage harus dilaporkan jujur. Seluruh query predictions target-free dan immutable sebelum label scoring. Gunakan satu canonical parent loader untuk producer/verifier dan independent objective/metric arithmetic.

Scientific gate, semua harus lulus: unique F1 gain vs full champion>=0.0005; E67H12>=0 dan E75W6>=0; unique RECENCY-minus-SHUFFLED>=0.0002 dan E75W6>=0; supported subgroup harm F1<=0.01/positive recall<=0.02; missing exact dan QA PASS.

Exploratory eligibility terpisah: [unique gain>0 dan E75W6 gain>=-0.0005] ATAU [E75W6 gain>=0.0005 dan unique gain>=-0.0005], serta semua supported subgroup harm F1<=0.02/recall<=0.03, missing exact dan QA PASS. Jangan mengubah scientific failure menjadi PASS. SHUFFLED tidak boleh dipromosikan berdasarkan skor. Ini peluang benchmark terukur, bukan klaim temporal mechanism.

Semua guards dihitung pada E67 W4/W5/W6/H12, E75W6, unique primary: availability, entry_friction, dan inherited fit-only text-load groups, n>=200 dengan>=30/class. Unsupported tetap dilaporkan dengan denominator. Laporkan F1/confusion/precision/recall, AUC/AP/logloss/Brier, cycle directions dan RECENCY-vs-SHUFFLED; jangan membuat public confidence interval.

## Ownership, execution dan stop point

Luna1 (`01a18b41-4be9-7cb2-b8b5-d4cd45ed8e18`) memiliki new lane paths di `C:/Users/Sam/OneDrive/Documents/KASDAD/tk2-simplification-audit-v1`: `reports/simplification_v1/CLS_POS_RECENCY_V1/`, `experiments/CLS_POS_RECENCY_V1/`, narrowly named runner/core/verifier/tests. Jangan ubah ROOT_PLAN.md, historical sources/frozen states, notebook/anggota lain, MAIN, REG atau source fit lain. Scoped local source/freeze commit sebelum official optimizer, committed prelabel sebelum query score; no push/merge/rebase/reset.

Izin root ini cukup untuk menyelesaikan tepat pilot empat fits dan hasil QA setelah seluruh syarat pre-fit dipenuhi; jangan berhenti hanya untuk meminta persetujuan seremonial. Jika ada technical blocker nyata yang tak bisa diperbaiki tanpa mengubah recipe/budget/boundary, simpan checkpoint dan kirim ke root, bukan Sam. Setelah hasil tersedia kirim langsung ke root Sol `01a19569-16e0-7253-aa14-57a0a3160e03` dengan exact counts, sources/commit, negative evidence dan next action; jangan hanya menjawab di chat sendiri.

Jika eligible, siapkan source/inventory E79 saja, tanpa fit/inference Test/CSV/upload sampai root memeriksa hasil. Jika gagal, tutup recipe tanpa mengubah half-life/floor/threshold/blend. Root tetap memutuskan langkah berikut dan submission. Jangan buka lane lain sendiri atau mengaktifkan REG.

Resource policy tetap `RESOURCE_POLICY_USER_REVISED_20261106.json`: one actual heavy job under OS advisory lock, two threads, gamma/feature stage1GiB initial/512MiB actual reserve, cached scoring512MiB/256MiB. Tidak ada blanket2/4GiB floor, scheduler, polling waktu, process kill, atau menghapus lock marker berdasarkan keberadaannya. Estimasi pilot keseluruhan puluhan menit termasuk implementasi/QA; runtime belum diukur untuk recipe ini. Simpan semua binary/row arrays di ignored paths; laporan ringkas saja.
