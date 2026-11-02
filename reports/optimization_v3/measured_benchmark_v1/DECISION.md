# Measured Benchmark V1 — keputusan Sol

2 November2026. Milestone baru berdasarkan izin pengguna untuk iterasi terukur dan keputusan submission di Sol. Original Research Next tetap ditutup dengan verdict historisnya; batch benchmark ini terpisah.

## Classification

Siapkan class-only sebagai SATU benchmark exception: full-six gain+0,000687,late−0,000237,W6+0,001000,H12 pooled+0,000596 dan64supportedgroupsPASS. Ini belum qualified menurut gate lama. Lanjutkan N02 exactrecipe pada59/63/71/75 dengan no-sketch matchedcontrol:8fits. Recipe ini memberi pilot primary/H12 positif tetapi recall Q4 H12 gagal guard lama2pp; benchmarktrack membolehkan ceiling3pp secara eksplisit, MacroF1 harm tetap maksimum0,01. Evaluasi full-six sebelum memutuskan CSV N02. Tidak mengubah bobot/threshold/sketchdimensi. Kombinasi class50:50 yang gagal tetap ditutup.

Budget maksimal10plannednewfits(8temporal+2candidatefullTrain), dengan maksimal2technicalretries yang tetap masuk ledger/cap60. Reuse attestedfullTraincontrols; tidak ada izin refitkontrol tambahan. Originalcontinuationcap24tetap;2slotdialokasikan eksplisit untukFULLTRAINbenchmark. Maksimal2Classification submissions boleh dipertimbangkan batchini, tetapi TIDAK ADA upload yang diotorisasi pada tahap persiapan; hasil kembali keSol terlebihdahulu.

## Regression

Satu midpoint R06_ORIGINAL/R22_ORIGINAL50:50 dihitung zero-fit dari enamorigin setelah protokol dibekukan. Gain primaryvsR06+0,016742,H12+0,031242,late+0,022616,4/6window nonnegatif. Namunprimary kurang darihurdle0,02 dan dua lowtailgroups memburuk>0,25 terhadapR22 (meskipun lebihbaikdaripadaR06/legacy). Sesuai protokol ini **NO_GO_MIDPOINT_SUBMISSION**; jangan mencari bobotbaru atau mengecualikan baseline sesudahhasil. Angkanya menunjukkan komplementaritas kecil, bukan bukti midpoint pasti burukdiTest. Tidak ada fit atau submissionRegression baru dalam batchini.

## Koreksi provenance dan sign

CAMPAIGN_SUMMARY stale tentang submission C16/R22 dan11,71363. Luna sedang merekonsiliasi current/historicalKaggle registry secara read-only. InitialR1_D danlgb_text_inner adalah modelberbeda; jangan memaksakan keduanya punya score sama.

Ada kesalahan label/sign pada tabel subgroup N04_DIAGNOSTIC Markdown: kolom GainvsR22 berisi harm_vs_R22. W1missing candidate13,948865 vsR2213,746134 adalah harm+0,202732. H12missing candidate14,389503 vsR2214,351436 adalah harm+0,038067. UnderlyingJSON benar; aggregateprimary/H12 gains tidakberubah. Perbaiki label/prose lewatcorrectionnote tanpa refit/mengubah datahistoris.

## Koordinasi

Sol menyiapkan keputusan, Luna mengeksekusi milestone dan mengirim hasil kembali; Sol berhenti selamaLuna bekerja. User secara eksplisit mengizinkan komunikasi dua arah. Setelah CSVQA/internalresults kembali, Sol menentukan uploadperfile dengan quota/duplicates/sourceverified. Tidak ada monitoring/pollingroot atau penelitian sambilmenunggu.

Protokol canonical: experiments/optimization_v3/measured_benchmark_v1/class_benchmark_protocol.json danreg_midpoint_protocol.json. RawTrain/cache/models/CSV tidak masukGit.

## QA review Regression

Independent verifier PASS:25inputhashes,12cachebundles,11views dan seluruhsubgroup diperiksa. MAIN juga menjalankan ulang verifier melalui fungsi run tanpa menulis ulang receipt; hasil/gates cocok. Python compilation kedua skrip lulus. Tidak ada pytest baru yang diklaim untuk milestone Regression zero-fit ini. Tidak ada officialfit atau aksesTest/Kaggle pada review ini.
