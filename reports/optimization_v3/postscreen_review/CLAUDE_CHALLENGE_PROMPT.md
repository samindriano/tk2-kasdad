# Prompt reviewer tandingan

Salin isi blok berikut ke Claude. Ini tugas kritik baca-saja, bukan implementasi/resume.

```text
Kamu adalah reviewer tandingan independen. Tugasmu berusaha mematahkan atau mempersempit kesimpulan review kami, bukan mencari persetujuan dan bukan melaksanakan eksperimen baru.

Repo: C:\Users\Sam\OneDrive\Documents\KASDAD\TP 2
Remote: samindriano/tk2-kasdad
Snapshot screening: 49b3fdf4d9ef027f1c1cbbb6bbec03dce6a495cf.
HEAD berikutnya mungkin hanya menambahkan dokumen/script review. Verifikasi HEAD/status, baca AGENTS yang berlaku; jangan reset atau checkout versi lama.

Kampanye sekarang PAUSED_FOR_REVIEW. Seluruh 45 kartu sudah berstatus: 7 ADVANCE, 36 RETIRED, 2 SKIPPED. Tidak ada fit continuation cutoff59/63/71/75 untuk finalist batch ini. Jangan mengubah pause atau melanjutkan kampanye.

Batas tindakan:
- READ-ONLY repository; output cukup di chat.
- Tidak menulis/mengubah file, Git mutation, training, full-Train refit, Test access, inference, Kaggle/API/submission, atau credential access.
- Analisis kecil dari official Train dan bundle tersimpan diperbolehkan; jaga RAM, baca kolom/array yang diperlukan saja. Temporary scratch di luar repo boleh untuk analisis yang tidak melatih model.
- Jangan mengubah EDA/notebook/model/recipe/gate/ledger. Jangan memanggil runner yang bisa fit; audit source sebelum menjalankan command.
- Jangan menyatakan source QA PASS sebagai bukti seluruh hipotesis benar.

Baca pertama:
1. reports/optimization_v3/postscreen_review/REVIEW.md
2. reports/optimization_v3/postscreen_review/CARD_REVIEW.md
3. reports/optimization_v3/postscreen_review/diagnostics.json
4. scripts/optimization_v3_postscreen_review.py
5. reports/optimization_v3/batch1/PAUSE_HANDOFF_01.md
6. reports/optimization_v3/batch1/batch_ledger.json
7. experiments/optimization_v3/research_backlog/queue.json

Bukti utama tetap recipe/source, result, verifier dan gate per kartu. Registry menunjuk result variants final (beberapa bukan results.json); jangan baca revision awal sebagai hasil final.

Poin yang harus kamu tantang:

A. Mempertahankan continuation C16/C11/R22/R06.
- C16 H12_E +.004953 F1, H12_L +.000830; C17 late hampir nol dan delta berkorelasi .790 dengan C16.
- C11 primary +.000661 tapi H12 -.000310, late -.000947; korelasi delta C16/C11 .105. Apakah keberagaman family cukup layak untuk 4 fit C11, atau ini hanya menjalankan aturan lama dengan bukti lemah? Jelaskan tradeoff tanpa mengubah gate historis.
- R22 MAE gain primary .098176, H12 .124873, late .100466; R06 .061726/.103260/.081977. Delta corr .200; 23/24 dan24/24 cycle pilot positif. Apakah ini konsistensi bermakna atau artefak mapping/pooling?
- Kesimpulan Claude terdahulu bahwa gain hampir hilang late hanya memakai snapshot lebih awal R21; jangan mengulangnya tanpa R22/R06.

B. Rescue tunggal R12_PRESENT_ROUTE.
- R12 asli RETIRED: gain MAE primary .245801, H12 .284490; hanya guard missing yang gagal (harm .768759 W1, .489276 primary, .324499 H12). Tail H12 low/high justru membaik 1.029815/.561555.
- Posthoc policy: hitung rank percentile atas SELURUH cycle, blend .75 champion+.25 ranker percentile seperti R12 asli; setelah itu ganti missing rows dengan exact champion. Jangan rank ulang subset present.
- Dari saved pilot: gain primary .285422, H12 .306170, late .114157. Missing harm nol by construction; present predictions unchanged, tails tetap membaik.
- Matched no-ranker control: .75 champion+.25 percentile(champion), same whole-cycle mapping+present route, justru gain primary -.018399/H12 -.012084. R12 route mengungguli kontrol ini .303821/.318254. Apakah kontrol ini cukup adil? Apa penjelasan lain yang belum dipisahkan?
- Raw ranker mean Spearman early .733412 vs champion .693230, tetapi late .839047 vs .851298; blend MAE late tetap membaik. Jangan mengklaim ranker sendiri selalu lebih baik.
- Proposal hanya 1 policy baru: 4 origin fits tambahan, freeze/amendment baru, status R12 asli tetap RETIRED. Total continuation asli16+rescue4=20, saldo nominal86->66; belum ada reservasi atau otorisasi resume.
- Existing full-six gate tetap, ditambah proposal syarat BARU gain>=.03 pada pooled W2(cut59,cycle60–63),W3(cut63,64–67),W5(cut71,72–75),W6(cut75,76–79):29.524baris unik, bukan pooled seluruh output12cycle tiapfit. Exact missing fallback. Label sudah pernah dilihat pada pilot. Apakah kriteria ini koheren atau terlalu adaptif? Rekomendasikan kriteria sebelum fit, jangan cari ambang dari output baru.
- Apakah rescue pantas diuji, atau tetap prioritaskan16 fit asli dan tutup riset? Jangan wajib menyetujui rescue.

C. Pelajaran kegagalan dan blind spot.
- C19 gagal primary+Q4 recall tetapi late F1+.005160 dan ranking membaik: apakah kami terlalu cepat menaruhnya di archive?
- R23 present-only menghilangkan pilot guard failures, tetapi late gain hanya .0228 dan pilotW4 sedikit negatif: adakah alasan ia lebih layak dari R06 atau R12?
- R16 recency punya late gain .099974 tetapi primary hanya .015729: apakah ini petunjuk history-regime yang lebih penting dari rescue ranking?
- R01/R14 gagal high-tail; missing routing tidak menyelesaikannya. Adakah rescue yang memakai fitur tersedia saat inference, bukan observed y, dan bukti spesifik yang membenarkannya?
- R09 Huber gagal sementara R22 pseudo-Huber berhasil; char Ridge historis gagal sementara R06 char LightGBM berhasil. Apakah interpretasi learner×representation/objective kami terlalu spekulatif?
- Karakter token anonim dapat membawa artefak encoding; jangan menyebut substring punya semantik.
- Target Train cocok rank grid within-cycle dalam toleransi ~5e-5. Ini observasi, bukan proof generative mechanism atau distribusi Test.

D. Validasi dan audit.
- Primary pilot subset H12. Continuation memakai label/baris yang SUDAH muncul dalam pilot, meskipun origin prediction belum dibuat. Tidak ada klaim untouched holdout. Cari jika dokumen kami masih mengisyaratkan independensi secara keliru.
- Enam bin4cycle dari fit55/67 bukan full-six forecast origin55/59/63/67/71/75.
- Moving-block circular length3,2000draw hanya sensitivitas deskriptif sesudah seleksi;12cycle/horizon; bukan CI valid yang mengoreksi45 percobaan.
- Script menghitung ulang35 STANDARD,220artifact hash,70metric comparisons;10kartu lain hanya historical reports. Apakah coverage disampaikan jujur? Cari kesalahan alignment, missing mask, average-rank ties, clipping/order, source-data boundary dan baseline mismatch.
- C18 gate primary-only subgroup pass berbeda dengan detailW1Q3failure; registry juga mengoleksi beberapa boolean false yang sebenarnya expected safety flags. Apakah masalah ini berdampak pada ADVANCE lain atau keputusan continuation? Audit scope, jangan hanya copy statusPASS.
- Kami tidak menyatakan hasil baru Kaggle atau menjumlahkan gain pilot ke public score.

Deliverable di chat:
1. APPROVE / REWORK / REJECT untuk rencana empat continuation asli dan terpisah untuk proposal R12_PRESENT_ROUTE.
2. Maksimal5 temuan yang paling bisa mengubah keputusan; tiap temuan: klaim yang ditantang, bukti path+angka, penjelasan alternatif, tingkat keyakinan, implikasi.
3. Tabel perbedaan pendapat dengan reviewer utama; tandai asumsi/UNKNOWN.
4. Satu urutan tindakan berikutnya yang paling hemat namun informatif, dengan stop criteria. Tidak perlu puluhan recipe baru.

Lakukan review sampai selesai tanpa meminta konfirmasi per langkah. Jangan merasa perlu menemukan kesalahan bila bukti kuat. Jangan menjalankan training/resume. Berhenti setelah kritik dan rekomendasi.
```
