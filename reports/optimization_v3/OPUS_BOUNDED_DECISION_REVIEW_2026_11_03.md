# Opus: independent TP2 decision review after the FM campaign

## Verified handoff context (Sol, 3 November 2026)

Incoming repository HEAD e130fd5cc1374bc3ff6e1f860f4f82cb4abc2643, clean.
This document is a subsequent documentation commit, not a new experiment.

B01 local reconciliation: CSV submissions/optimization_v3/b01_benchmark_v1/
b01_benchmark.csv has exactly29421 rows,29421 unique IDs, exact official
Classification Test ID order, columns record_id/artifact_signal and only0/1.
SHA256 b30e6d4f732902bab43d9d28683975ac9d1820ce940853f232ba3936ae57a2d5,
matching the preparation manifest. This check inspected IDs/schema/predictions,
not Test labels or evaluation metrics. Kaggle CLI is not on the current PATH.
Local B01 records say PREPARED_FOR_REVIEW_NOT_SUBMITTED; no confirmed B01
submission receipt/score was found in the targeted report/receipt search.
Live Kaggle state is UNKNOWN. An old preparation status is not proof that nobody
subsequently uploaded it manually. Do not duplicate an upload without reconciling.

## Prompt: follow these instructions as an independent reviewer

Anda menjadi reviewer independen Opus untuk TP2 KASDAD di:
C:/Users/Sam/OneDrive/Documents/KASDAD/TP 2

Lakukan SATU review read-only sekitar15-20 menit. Jangan otomatis membenarkan
Sol atau Astra; jangan merasa wajib menemukan ide baru agar review bernilai.
Nilai satu keputusan berikutnya dengan evidence yang tersedia, lalu berhenti.

Konteks tugas: ini coursework kelompok dengan analisis individual yang harus
bisa dijelaskan dan direproduksi, serta kompetisi Kaggle. Classification
menargetkan artifact_signal dengan Macro F1; Regression menargetkan threat_rank
dengan MAE. Official Train cycles47-79 dan Test80-91. Fokus review kali ini
Classification; Regression hanya konteks keputusan proyek, bukan lane baru.
Kesiapan akademis dan skor kompetisi adalah dua tujuan yang perlu dibedakan.
Jangan mengarang rubric, deadline, atau aturan submission. Jika ingin mengklaim
aturan resmi, temukan dan kutip file instruksi TP2 yang benar. PDF resmi belum
dilokalisasi pada handoff ini; ketiadaannya di pencarian sempit bukan bukti aturan
tertentu. Materi kuliah/TP1 bukan pengganti instruksi resmi TP2.

Dataset memakai token anonim. Jangan menyarankan semantic embeddings/pretrained
language models berdasarkan asumsi token mempunyai makna bahasa atau domain.
Jika menyarankan representasi lain, jelaskan mekanisme statistiknya. Jangan
menafsirkan token diversity sebagai kualitas teks, kausalitas, atau bahaya.

### Posisi terakhir dan pertanyaan yang masih terbuka

- Best public Classification yang dilaporkan pengguna: A02 .82930;
  sparse25 .82908. Leaderboard pengguna menunjukkan pesaing .82782.
  Public sekitar30% Test, private70%; ini bukan label, CI, private ranking,
  atau dasar estimasi ceiling. Jangan memakai posisi sementara sebagai alasan
  ilmiah membuka riset baru atau menjanjikan .83.
- R06 best-public-observed Regression11.16620; R-NBR11.19115 lebih buruk.
  Salah upload samuel_rnbr_final_e79.csv ke Classification gagal karena
  threat_rank bukan artifact_signal. Itu error destinasi/schema, bukan skor model.
- Banyak puluhan eksperimen memakai development labels temporal yang sama.
  Primary/H12/history sensitivity overlap dan bukan konfirmasi independen.
- FM V2 akhirnya technical PASS, MECHANISM_FAIL dan COMPETITION_FAIL:
  E67 H12 FM .763788 vs no-pair .804906/C2_B .805106/A02 .811159;
  E75 W6 FM .785334 vs no-pair .809872/C2_B .810384/A02 .819604.
  AUC/loss juga memburuk; no-pair dekat dengan C2_B. Empat fits, tanpa rescue.
  Serialization amendment zero-fit mempertahankan numeric arrays. V1 tetap
  optimizer-inconclusive. Jangan merekomendasikan ulang repair yang sudah selesai.
- Positional offset positif kecil terhadap bag, tetapi gagal ambang .003;
  belum terbukti incremental terhadap A02. Gate risiko bukan scoring Kaggle.
- B01 mengubah centering residual menjadi local-label-rate shrinkage dengan
  geometry dan threshold tetap. Internal gain vs A02: full-primary+.000261,
  new origins+.000519,late+.001762,H12+.003890. Ia gagal primary minimum+.0005
  dan hanya3/5 origin positif (syarat4/5), sehingga tetap NOT_QUALIFIED.
  CSV sudah disiapkan;261 label berbeda dari A02 (1730->1,881->0).
  Tidak ada bukti public score B01 yang terkonfirmasi pada handoff ini.

### Baca terbatas

Mulai dari delapan ringkasan berikut (path relatif repo):
1. reports/optimization_v3/ASTRA_RESEARCH_REVIEW_HANDOFF_2026_11_03.md
2. reports/optimization_v3/learned_pair_pilot_v2/SOL_FINAL_REVIEW.md
3. reports/optimization_v3/a02_continuation_v1/FINAL_HANDOFF.md
4. reports/optimization_v3/a02_next_decision_v1/DECISION.md
5. reports/optimization_v3/a02_b01_centering_v1/B01_OUTPUTS/SUMMARY.md
6. reports/optimization_v3/b01_benchmark_review_v1/SOL_REVIEW.md
7. reports/optimization_v3/classification_next_v1/FINAL_SUMMARY.md
8. reports/optimization_v3/postscreen_review/REVIEW.md

Tambahkan maksimal tiga file source/evidence yang benar-benar menentukan
keputusan, termasuk kandidat historis jika menilai novelty. Jangan membaca
seluruh repo/log, raw CSV, prediction arrays atau JSON audit besar. Jika klaim
tidak bisa dipastikan dalam batas itu, tulis UNKNOWN atau kesimpulan bersyarat.
Header PLANNED bisa usang; hasil actual outrank judul kartu. Reviewer sebelumnya
boleh salah; tantang alasan dan comparator-nya, bukan sekadar kata PASS/FAIL.

### Pertanyaan keputusan

1. Apa penjelasan alternatif terkuat terhadap rekomendasi Sol untuk berhenti
   broad search? Tunjukkan evidence pendukung sekaligus counterevidence.
2. Apakah B01 pantas mendapat satu benchmark publik yang dibekukan, setelah
   status upload direkonsiliasi, meskipun qualification internal gagal?
   Apakah view late/H12 cukup mengubah keputusan, atau selection bias dominan?
3. Apakah ada SATU mekanisme berbeda dengan peluang masuk akal mengalahkan
   A02 yang memang belum diuji? Tidak wajib ada. Periksa sejarah sebelum
   menyarankan model/threshold/routing/blend yang sudah dicoba.
4. Bagaimana private70% dan keterbatasan temporal validation memengaruhi
   keputusan, tanpa menebak label/ranking private atau confidence interval?

### Batas keras dan deliverable

Tidak ada training/optimizer, fit preprocessing, scoring baru/akses label
evaluasi, Test/Kaggle/web/API, install, perubahan file/Git/notebook/EDA, subagent,
atau pesan ke chat lain. Jangan mengubah pekerjaan anggota lain atau mengubah
FAIL historis menjadi PASS. Tidak boleh mengusulkan pseudo-labeling Test,
target leakage, reverse-engineering label, atau pencarian threshold pada public.

Jawab Bahasa Indonesia, maksimal1000 kata:
- Verdict, maksimum3 temuan berurutan dampak, evidence path dan counterevidence.
- Pilih TEPAT SATU tindakan berikutnya:
  A) RECONCILE_THEN_ONE_FROZEN_B01_BENCHMARK,
  B) FINALIZE_EXISTING_MODELS, atau
  C) ONE_NOVEL_BOUNDED_MECHANISM.
- JikaA: alasan menanggung risiko, syarat status/upload/hash/schema, satu CSV
  tetap tanpa tuning, serta cara membatasi penggunaan feedback leaderboard.
- JikaC: hipotesis yang bisa dipatahkan, pemeriksaan novelty, recipe minimum,
  comparator A02 dan matched control, temporal boundaries, maksimal4 predictor
  fits/zero sweep, stop criteria, risiko dan alasan nilai informasinya lebih
  tinggi daripadaA/B. Bukti kuat lebih penting daripada terdengar baru.
- JikaB: alasan yang tetap berlaku walaupun pengguna mampu membayar eksperimen.
- UNKNOWN yang berpotensi mengubah keputusan.

Ini rekomendasi saja, bukan izin mengerjakan tindakan terpilih. Satu review,
lalu STOP. Jangan memoles notebook atau menyiapkan submission saat review.
