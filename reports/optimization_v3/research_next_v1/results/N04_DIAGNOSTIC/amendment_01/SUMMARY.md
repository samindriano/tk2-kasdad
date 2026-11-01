# N04_DIAGNOSTIC — Task fits setelah domain screening

Status: `DIAGNOSTIC_ONLY` — tidak eligible untuk continuation.

Amendment dibuat setelah domain AUC diketahui, tetapi sebelum task MAE ditampilkan. Ini bukan analisis outcome-blind. N04 asli tetap `SKIPPED_SCIENTIFIC_PREFLIGHT`; threshold, gate, recipe, dan ledger N04 asli tidak diubah.

Empat domain fits asli dipakai ulang setelah verifikasi bundle, recovery, OOF probabilities, dan hash bobot. Tidak ada domain classifier yang dilatih ulang. Tepat empat task fits dijalankan: weighted dan within-cycle-shuffle pada cutoff 55 dan 67.

## Hasil per window

Gain positif berarti MAE lebih rendah daripada pembanding.

| Window | n | Weighted MAE | R22 MAE | Shuffle MAE | Protected legacy MAE | R06 sekunder MAE | Gain vs R22 | Gain vs shuffle |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| W1 | 4,114 | 15.663873 | 15.743656 | 15.701241 | 15.882112 | 15.816833 | +0.079782 | +0.037368 |
| W2 | 6,493 | 16.078873 | 16.081749 | 16.055361 | 16.192801 | 16.051924 | +0.002876 | -0.023513 |
| W3 | 7,078 | 17.954246 | 18.005136 | 17.958168 | 18.206874 | 18.053194 | +0.050890 | +0.003921 |
| W4 | 4,950 | 11.407949 | 11.390307 | 11.402351 | 11.455005 | 11.396232 | -0.017641 | -0.005598 |
| W5 | 7,462 | 11.464374 | 11.496760 | 11.492898 | 11.577229 | 11.477448 | +0.032386 | +0.028524 |
| W6 | 8,491 | 11.639318 | 11.594417 | 11.626096 | 11.733308 | 11.653450 | -0.044901 | -0.013223 |

## Primary, H12, dan domain context

| View | n | Weighted MAE | R22 MAE | Shuffle MAE | Legacy MAE | R06 MAE | Gain vs R22 | Gain vs shuffle |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Primary W1 + W4 | 9,064 | 13.339643 | 13.366220 | 13.353546 | 13.464396 | 13.402670 | +0.026578 | +0.013904 |
| H12 W1–W6 | 38,588 | 13.910216 | 13.922659 | 13.912851 | 14.047533 | 13.944273 | +0.012444 | +0.002635 |

| Cutoff | Domain fold AUCs | ESS fraction | Source clip fraction | Window utama | Gain vs R22 | Gain vs shuffle |
|---:|---|---:|---:|---|---:|---:|
| 55 | 0.549576, 0.542069 | 0.829 | 0.062 | W1 | +0.079782 | +0.037368 |
| 67 | 0.592372, 0.604807 | 0.816 | 0.036 | W4 | -0.017641 | -0.005598 |

| Cutoff | Fit rows | Primary window | Primary gain vs R22 | Primary gain vs shuffle | H12 n | H12 gain vs R22 | H12 gain vs shuffle |
|---:|---:|---|---:|---:|---:|---:|---:|
| 55 | 12,020 | W1 | +0.079782 | +0.037368 | 17,685 | +0.039983 | +0.001630 |
| 67 | 29,705 | W4 | -0.017641 | -0.005598 | 20,903 | -0.010856 | +0.003486 |

Cutoff 55 berada tipis di bawah batas AUC pada satu fold (0,549576) dan lebih rendah pada fold lain (0,542069), sementara ESS serta clipping memenuhi batas. Cutoff 67 memenuhi AUC, ESS, dan clipping. Dengan dua cutoff saja, hasil ini tidak mengidentifikasi hubungan umum antara AUC domain dan manfaat task.

## Guard subgroup

Setiap window, primary, dan H12 menyimpan denominator, MAE kandidat, R22, shuffled-weight, protected legacy, dan R06 untuk `stream_missing`, target aktual `<=5`, dan target aktual `>=95`. Gain positif berarti kandidat mengurangi MAE terhadap kontrol. Tail target hanya dipakai sebagai mask evaluasi. `supported` mengikuti minimum n=100; hasil ini diagnostik, bukan penerapan gate baru.

| View | Subgroup | n | Gain vs R22 | Gain vs shuffle | Harm vs legacy | Candidate MAE | R06 MAE |
|---|---|---:|---:|---:|---:|---:|---:|
| W1_55_to_56_59 | stream_missing | 392 | +0.202732 | +0.005644 | +0.115590 | 13.948865 | 14.228832 |
| W1_55_to_56_59 | observed_target_le5_evaluation_only | 205 | -0.084176 | +0.177562 | -1.006293 | 20.505915 | 21.445207 |
| W1_55_to_56_59 | observed_target_ge95_evaluation_only | 213 | +0.179027 | +0.009604 | -0.135268 | 10.816638 | 10.996120 |
| W2_55_to_60_63 | stream_missing | 369 | +0.071826 | -0.000589 | -0.005618 | 13.998536 | 14.041066 |
| W2_55_to_60_63 | observed_target_le5_evaluation_only | 322 | +0.069012 | +0.087923 | -0.904281 | 21.846841 | 22.569347 |
| W2_55_to_60_63 | observed_target_ge95_evaluation_only | 325 | -0.035735 | +0.190667 | -0.228327 | 13.757706 | 13.802408 |
| W3_55_to_64_67 | stream_missing | 442 | +0.021974 | +0.042742 | +0.058765 | 16.043184 | 16.270406 |
| W3_55_to_64_67 | observed_target_le5_evaluation_only | 353 | -0.061282 | +0.041011 | -1.093084 | 22.621875 | 23.366625 |
| W3_55_to_64_67 | observed_target_ge95_evaluation_only | 360 | -0.193033 | +0.150483 | -0.284836 | 14.623337 | 14.948626 |
| W4_67_to_68_71 | stream_missing | 342 | -0.031120 | +0.160149 | -0.081673 | 13.778249 | 13.772165 |
| W4_67_to_68_71 | observed_target_le5_evaluation_only | 244 | -0.179196 | +0.098773 | -0.450075 | 9.290475 | 9.381503 |
| W4_67_to_68_71 | observed_target_ge95_evaluation_only | 249 | +0.012905 | -0.052440 | +0.183927 | 7.537954 | 7.530585 |
| W5_67_to_72_75 | stream_missing | 507 | +0.009167 | +0.031444 | +0.000046 | 13.948181 | 13.980048 |
| W5_67_to_72_75 | observed_target_le5_evaluation_only | 362 | -0.137291 | +0.052805 | -0.501818 | 7.605254 | 7.733683 |
| W5_67_to_72_75 | observed_target_ge95_evaluation_only | 374 | +0.084122 | -0.053129 | +0.230526 | 7.748343 | 7.651984 |
| W6_67_to_76_79 | stream_missing | 526 | -0.021969 | -0.079868 | +0.236967 | 14.425375 | 14.409473 |
| W6_67_to_76_79 | observed_target_le5_evaluation_only | 442 | +0.122246 | -0.003309 | -0.429221 | 9.545645 | 9.582056 |
| W6_67_to_76_79 | observed_target_ge95_evaluation_only | 434 | +0.042888 | -0.036069 | +0.227398 | 7.452102 | 7.298686 |
| Primary W1+W4 | stream_missing | 734 | +0.093771 | +0.077634 | +0.023677 | 13.869369 | 14.016053 |
| Primary W1+W4 | observed_target_le5_evaluation_only | 449 | -0.135813 | +0.134745 | -0.704028 | 14.411110 | 14.889430 |
| Primary W1+W4 | observed_target_ge95_evaluation_only | 462 | +0.089494 | -0.023835 | +0.036766 | 9.049555 | 9.128332 |
| H12 W1-W6 | stream_missing | 2,578 | +0.038067 | +0.019236 | +0.064371 | 14.389503 | 14.479334 |
| H12 W1-W6 | observed_target_le5_evaluation_only | 1,928 | -0.029075 | +0.062729 | -0.707738 | 14.763003 | 15.163882 |
| H12 W1-W6 | observed_target_ge95_evaluation_only | 1,955 | +0.005277 | +0.035603 | +0.012862 | 10.255060 | 10.288516 |
| H12 origin 55 | stream_missing | 1,203 | +0.096165 | +0.017363 | +0.057533 | 14.733585 | 14.921343 |
| H12 origin 55 | observed_target_le5_evaluation_only | 880 | -0.018940 | +0.089987 | -1.003781 | 21.845361 | 22.627291 |
| H12 origin 55 | observed_target_ge95_evaluation_only | 898 | -0.047854 | +0.131611 | -0.228908 | 13.407127 | 13.596282 |
| H12 origin 67 | stream_missing | 1,375 | -0.012764 | +0.020875 | +0.070354 | 14.088462 | 14.092617 |
| H12 origin 67 | observed_target_le5_evaluation_only | 1,048 | -0.037586 | +0.039841 | -0.459153 | 8.815985 | 8.896898 |
| H12 origin 67 | observed_target_ge95_evaluation_only | 1,057 | +0.050415 | -0.045962 | +0.218264 | 7.577146 | 7.478323 |

Tail subgroup masks use observed evaluation targets for reporting only; the task fits never receive evaluation targets.

## Verifikasi dan batas interpretasi

Independent verification: `PASS`; domain bundles: 4; task fits: 4; official Train hash, component weights, score IDs, rank50 average ties, .75/.25 policy, ledgers, and report metrics checked.

Amendment ini menguji empat task fits yang sudah diotorisasi. Hasilnya tidak mengubah verdict N04 asli, tidak melonggarkan gate 0,55, tidak memilih threshold baru, dan tidak layak otomatis masuk continuation. Test, Kaggle, notebook, dan full-Train refit tidak digunakan.
