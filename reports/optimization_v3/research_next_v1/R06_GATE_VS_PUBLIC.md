# R06: mengapa gagal guard tetapi public lebih baik?

Observasi Kaggle 1 November2026 melalui browser Chrome authenticated, read-only. Semua lima submission terbaru berstatus COMPLETE. File labels dibaca dari halaman; hash upload manual tidak dibandingkan dengan file server Kaggle. Tidak ada upload, perubahan final selection, akses label Test, atau model fit pada pemeriksaan ini.

| Task / file | Public | Pembanding |
|---|---:|---|
| Class c11.csv |0.82739|sparse25 tetap0.82908|
| Class c17.csv |0.82825|sparse25 tetap0.82908|
| Reg r06_original.csv |**11.16620**|R22 sebelumnya11.30535|
| Reg r21.csv |11.29314|sedikit lebih baik daripada R22|
| Reg r11.csv |11.39469|lebih buruk daripada legacy11.35970|

R06 memperbaiki public0.13915 terhadap R22 dan0.19350 terhadap legacy. Halaman leaderboard menyatakan public memakai kira-kira30% Test, private70%; komposisi subset dan variasi paired loss tidak tersedia. Ini bukan bukti bahwa subset IID, representative tiap cycle, atau bahwa private akan mempunyai peringkat sama.

## Internal mendukung signal R06, tetapi tidak superior universal

Perhitungan ulang independen dari bundle+Train, dengan rank50 dan alignment yang telah diperiksa:

| Evaluasi | Legacy | R22 | R06 |
|---|---:|---:|---:|
| Primary enam-origin MAE |11.559390|11.519691|**11.502413**|
| Primary gain vs legacy |—|0.039699|**0.056977**|
| H12 MAE |14.047533|**13.922659**|13.944273|
| W5+W6 gain vs legacy |—|0.021379|0.023589|

R06 menang5/6 windows terhadap legacy; **hanya2/6 terhadap R22** (W3/W5). Aggregate primary menang karena besarnya gain di window tersebut, bukan kemenangan merata. W6 R06 masih lebih buruk daripada legacy0.013929. H12 mengunggulkan R22. Jadi dukungan internal ada, tetapi magnitude public lebih besar dan ranking tidak stabil menurut origin/horizon.

Gate R06 gagal tepat satu group check: missing W1 n392, harm0.395556>0.25. Primary gain, H12 tolerance,5/6 window signs dan positive W5+W6 lolos. Tidak tepat mengatakan R06 gagal karena overall MAE buruk.

Kelompok yang gagal hanya392/38.588=**1.0159%** primary rows. Kontribusi harm kelompok itu ke pooled primary MAE adalah sekitar0.004018; perbaikan di tempat lain lebih besar. **Kaggle MAE mengagregasikan loss, sedangkan guard menolak jika satu supported subgroup melampaui batas. Keduanya mengoptimalkan preferensi yang berbeda.** Batas0.25 adalah keputusan proteksi riset, bukan definisi scoring MAE kompetisi.

Guard failure terjadi pada origin55 dengan12.020 train rows, sedangkan refit79 memakai50.608. Perubahan ukuran data dan regime dapat mengubah manfaat char learner; ini penjelasan yang masuk akal, belum terisolasi sebabnya. Jangan menyatakan full-Train pasti menghilangkan harm missing atau bahwa Test mengikuti late Train.

## Apakah hoki?

Belum dapat dipisahkan secara pasti. Internal primary/H12 memberi gain terhadap legacy, jadi kemenangan public bukan satu-satunya evidence bahwa R06 berguna. Namun relative ranking R06/R22 berubah antarview, sehingga variasi sample public, distribusi cycle/availability/target yang berbeda, learning-curve effects dan selection dari banyak percobaan sama-sama mungkin berkontribusi.

MAE public tunggal tidak menyediakan variance dari per-row loss difference. Tanpa public labels kita tidak dapat menghitung paired standard error/bootstrap yang sah untuk public gap0.13915, atau probabilitas keunggulan private. Jangan membuat confidence interval dari row counts saja. Bootstrap Train tetap conditional development sensitivity, bukan pengganti Test labels.

Implikasi: R06 adalah **best observed public benchmark**, tetap mempunyai catatan gagal guard internal. R22 tetap qualified internal comparator pada queue yang sudah disusun; R06 wajib menjadi secondary comparator. Tidak menghapus failure, menurunkan guard, mengganti weights atau memilih recipe berikutnya semata dari public. Jika user ingin mengubah prioritas dari subgroup protection ke pooled competition MAE, itu harus menjadi amendment tujuan yang berlaku konsisten untuk batch baru, bukan pengecualian tersembunyi untuk satu pemenang.

Detail numerik: [R06_GATE_VS_PUBLIC.json](R06_GATE_VS_PUBLIC.json). Sumber internal final: [hasil R06](../expanded_benchmark_v1_final_evaluation/R06/results.json), [verifikasi R06](../expanded_benchmark_v1_final_evaluation/R06/verification.json), [ringkasan expanded](../expanded_benchmark_v1_final_evaluation/CAMPAIGN_SUMMARY.md). Sumber public: [Regression submissions](https://www.kaggle.com/competitions/tp2-regression-cyber-vulnerability-kasdd-gasal-2026-2027/submissions), [Classification submissions](https://www.kaggle.com/competitions/tp2-classification-cyber-vulnerability-kasdd-gasal-2026-2027/submissions).
