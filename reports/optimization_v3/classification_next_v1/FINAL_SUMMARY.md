# Classification Next V1 — hasil dan checkpoint review

Tanggal: 2 November 2026
Kode persiapan dibekukan pada commit `503f01dfc9d35b285c526f2f42b7548617714efa`; input/source freeze disimpan pada `f804bc3e12a146daa047f63a3614ee8791e75310`. Protokol SHA256 `45a053ab61f13369e6838e92e7a9b7683d0f04a5e505f828ac2a15fb07d27a5a`; official Classification Train SHA256 `875552a68a3d24b0f3e1966d6ddb5671e6f54305a1c526e2f0a76747942ed91c` (50.608 baris, cycle 47–79).

## ClassOnly public receipt

Riwayat Kaggle yang ditempelkan pengguna menunjukkan `class_only.csv` berstatus **Complete**, skor publik **0,82774**, dibanding `v3_cls_sparse25` sebelumnya **0,82908** (delta −0,00134). Repository menyimpan ini sebagai observasi yang dilaporkan pengguna: tidak ada submission ref, waktu UTC persis, metode upload, server-side hash, atau receipt API yang disimpulkan. Event append-only berada di `submission_receipts.jsonl`. Otorisasi ClassOnly dianggap sudah terpakai; tidak ada upload baru pada milestone ini.

## C19 continuation — NO-GO

Empat fit resmi pada cutoff 59, 63, 71, dan 75 selesai dan tersimpan. Cutoff 55/67 menggunakan cache historis yang diverifikasi. Model dan routing sama dengan resep C19 yang dibekukan; tidak ada tuning.

| View | Baris | Macro F1 gain vs sparse25 |
|---|---:|---:|
| W1 | 4.114 | −0,000513 |
| W2 | 6.493 | −0,000781 |
| W3 | 7.078 | +0,003135 |
| W4 | 4.950 | +0,001105 |
| W5 | 7.462 | +0,003378 |
| W6 | 8.491 | −0,002599 |
| Primary full-six | 38.588 | +0,000667 |
| H12 pooled | 38.588 | +0,004152 |
| H12 origin 67 | 20.903 | +0,005160 |
| Late W5–W6 | 15.953 | +0,000286 |

Primary full-six juga memperbaiki ROC-AUC **+0,002497**, average precision **+0,005422**, log loss **−0,004169**, dan Brier **−0,001274**. Beberapa metrik membaik, tetapi gate frozen tetap berlaku: W6 melewati batas harm 0,001; hanya **3 dari 6** window yang nonnegative; serta satu subgroup melampaui F1-harm 0,01. Kegagalan subgroup ada pada **W2/Q3**, 1.450 baris, F1 harm **0,010461**, recall harm **0,017857**. Dari 48 subgroup yang memenuhi dukungan, 0 tidak didukung; tidak ada subgroup yang melewati recall-harm 0,03.

Dengan demikian, primary, H12 pooled, H12 origin 67, dan late W5–W6 lulus; W6, jumlah window nonnegative, dan subgroup F1 gagal. Verdict C19 lama tetap `RETIRE`; continuation juga `NO_GO_GATE_FAILED`. Independent verifier menghitung ulang metrik, gate, routing, ID, dan batas cycle: **PASS**, tanpa mismatch. Full-Train C19 tidak dijalankan. `C19_FULLTRAIN_GATE_DECISION.json` merekam bahwa Test dan SampleSubmission tidak diakses.

## Tree TTR matched pilot — gagal gate diagnostik

Empat fit selesai pada dua origin (55, 67) dan dua arm. COUNT menambahkan exact token count; BOTH menambahkan token count dan TTR. Stream memakai regex lowercase `[A-Za-z0-9]+`; TTR tidak terdefinisi untuk 0 token dan dipertahankan sebagai NaN pada kolom sparse LightGBM. Test sintetik membuktikan fit/predict LightGBM berjalan dengan NaN sparse yang eksplisit.

| View | Baris | BOTH gain vs sparse25 | BOTH minus COUNT Macro F1 |
|---|---:|---:|---:|
| W1 | 4.114 | −0,001130 | −0,000283 |
| W4 | 4.950 | −0,000561 | −0,001029 |
| Primary dua origin | 9.064 | −0,000814 | −0,000686 |
| H12 origin 55 | 17.685 | +0,000675 | −0,000136 |
| H12 origin 67 | 20.903 | +0,000108 | −0,000078 |
| H12 pooled | 38.588 | +0,000369 | −0,000116 |

Pada primary, BOTH minus COUNT ROC-AUC **−0,000142**, AP **+0,000262**, log loss **+0,000261**, dan Brier **+0,000095**. Delta loss yang positif berarti probabilistic fit memburuk. Kedua syarat gain primary gagal: BOTH tidak mencapai +0,0005 terhadap sparse25 dan tidak mencapai +0,0002 terhadap COUNT. Seluruh syarat H12/W1/W4 serta **24/24** subgroup support checks lulus. Matched subgroup terlemah adalah W1/Q3 (772 baris): BOTH minus COUNT Macro F1 **−0,005874**, recall **−0,012658**; ini deskriptif terhadap COUNT, bukan gate tambahan.

Verdict: `FAIL_DIAGNOSTIC_GATE`; pilot berhenti sesuai protokol. Tidak ada lanjutan otomatis, refit full-Train, Test inference, CSV, atau submission untuk TTR.

## Budget, boundary, dan QA

- Total fit resmi di fase Research Next: **48/60** consumed; 47 tersimpan dan 1 kegagalan/interupsi historis tetap dihitung. Kedelapan fit baru (4 C19 + 4 TTR) tersimpan; tidak ada fit retry baru.
- Active reservations: **48/60**; kategori continuation **21/24**. Maksimum 1 C19 full-Train continuation fit tidak dipakai karena gate gagal. Satu slot technical-retry yang sudah ada tetap dicatat; tidak ada ledger reset atau event lama yang dihapus.
- Tidak ada Test/SampleSubmission access, Kaggle API/CLI call, upload, perubahan threshold/bobot, atau perubahan notebook.
- Synthetic E2E preflight: PASS untuk payload, route reconstruction, bundle publish/read, all-view guard path, injected post-bundle failure recovery tanpa refit, dan LightGBM sparse-NaN fit/predict.
- Independent verification C19: PASS; independent verification TTR: PASS. Keduanya menghitung ulang gate secara terpisah.
- Unit tests: `python -m pytest -q tests/test_optimization_v3_classification_next_v1.py` — **6 passed**. `py_compile` dan `git diff --check` lulus.
- Historical C19 pilot memakai satu versi file sumber yang kemudian berubah. Blob file pada commit C19 historis cocok dengan hash freeze; discrepancy dilaporkan di preflight dan tidak mengubah cache atau verdict asli.

## Bukti dan reproducibility

- `C19_CONTINUATION/results.json`, `groups.csv`, `per_cycle.csv`, `independent_verification.json`, `budget_snapshot.json`.
- `TREE_TTR_PILOT/results.json`, `groups.csv`, `per_cycle.csv`, `matched_groups.csv`, `independent_verification.json`, `budget_snapshot.json`.
- `preflight.json`, `synthetic_preflight.json`, `submission_receipts.jsonl`.
- Zero-fit verifikasi: `python -m scripts.optimization_v3_classification_next_v1_verify --c19` dan `--ttr`.
- Unit tests: `python -m pytest -q tests/test_optimization_v3_classification_next_v1.py`.

## Keputusan checkpoint

Pertahankan `v3_cls_sparse25` sebagai champion Classification saat ini. C19 dan tree-TTR tidak memenuhi gate frozen. Milestone berhenti untuk review; tidak ada kandidat baru dibuka dan tidak ada upload Kaggle.
