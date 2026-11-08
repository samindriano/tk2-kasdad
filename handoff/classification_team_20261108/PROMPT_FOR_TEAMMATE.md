# Prompt untuk Codex/GPT teman

```text
Bahan Samuel sudah tersedia di GitHub:
https://github.com/samindriano/tk2-kasdad
Branch main, folder handoff/classification_team_20261108/.

Tolong lanjutkan usulmu menguji gabungan dengan model Abhi v3 public 0.83860.
Samuel terbaik public 0.83225 adalah POS + A02 dengan routing/threshold missing,
bukan C4_HGB di notebook akademis. Public score berbeda dari validasi internal.

Baca berurutan:
1. handoff/classification_team_20261108/README.md
2. handoff/classification_team_20261108/RESEARCH_MAP.md
3. handoff/classification_team_20261108/artifacts/MANIFEST.json
4. Source champion dan bukti hasil yang ditunjuk oleh peta tersebut.

Probabilitas validasi dan Test sudah tersedia dalam dua CSV gzip di artifacts/.
Jalankan verifier paket terlebih dahulu. Ada p_pos, p_a02, p_samuel,
stream_missing, record_id, origin, batas fit, dan flag is_primary_unique.
Tidak perlu mengulang puluhan eksperimen atau meminta NPZ/model Samuel dahulu.
Tidak ada label target di ekspor; join target official Train milikmu secara lokal.

Prioritas pekerjaan:
- Pahami recipe Abhi v3 dan batas preprocessing/fit-nya. Audit leakage sebelum
  menafsirkan skor tinggi atau melakukan blend.
- Pastikan prediksi validasi Abhi dan Samuel memakai origin serta batas fit sama:
  E67 fit47-67/query68-79; E75 fit47-75/query76-79. Align origin+record_id,
  bukan sekadar urutan baris. Jika Abhi hanya punya random-split OOF, jangan
  langsung mencampurnya dengan temporal Samuel; siapkan evaluasi kompatibel dulu.
- E67/E75 tumpang tindih 8491 ID. Pool unik memakai E67 cycle68-75 dan E75
  cycle76-79 (20903 baris); jangan menggandakan denominator. Laporkan juga per view.
- Bandingkan v3 standalone, Samuel standalone, serta komponen POS/A02 pada baris
  sama. Periksa complementary errors, present/missing, per-cycle, precision/recall
  dan skala probabilitas sebelum memutuskan ensemble berguna.
- Tetapkan satu benchmark gabungan sederhana sebelum scoring, bukan sweep besar.
  Jika ada pemilihan bobot atau threshold, gunakan periode lebih awal untuk
  pemilihan dan periode lebih akhir untuk evaluasi. V3 tetap comparator utama.
- Baca hasil gagal di peta agar tidak mengulang mekanisme yang sudah diuji.
  Bedakan kalah predictive information, gagal optimisasi, dan trade-off risiko.
  Jangan mengubah verdict eksperimen lama menjadi PASS.
- Semua development data telah dipakai adaptif; jangan menyebut holdout ini fresh.
  Threshold missing Samuel final dipilih pada development yang sama dan forward
  check-nya negatif; jangan menjual gain selection sebagai evidence independen.

Jika gabungan layak: refit/inference hanya komponen Abhi yang memang diperlukan
dan diizinkan pemilik proyek, gunakan probabilitas Test Samuel yang tersedia,
lalu siapkan satu CSV record_id,artifact_signal dengan 29421 ID unik sesuai
SampleSubmission, label integer0/1. Simpan recipe, hash input/output, komponen,
bobot/threshold, batas data, metrik matched validation dan keterbatasannya.
Jangan upload Kaggle atau mengubah final-slot selection tanpa instruksi pemilik.

Jika tidak ada gain yang cukup meyakinkan, katakan jujur dan pertahankan v3;
model dengan public lebih rendah hanya berguna jika kesalahannya melengkapi.
Jangan mengejar angka0.86729 dengan public threshold search atau label Test.

Deliverable: ringkasan temuan, validitas alignment, tabel matched comparison,
keputusan blend/no-blend, dan lokasi CSV+manifest bila dibuat. Sebut UNKNOWN
secara spesifik. Jangan berhenti hanya dengan daftar algoritme alternatif.
```
