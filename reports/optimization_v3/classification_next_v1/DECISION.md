# Classification Next V1 — keputusan setelah ClassOnly public

2 November2026. Champion Classification tetap sparse25 (public sebelumnya0,82908). Pengguna memberi bukti Kaggle CLASS_ONLY Complete0,82774, selisih−0,00134. Reference, timestamp pasti dan metode upload belum direkonsiliasi; tidak mengklaim serverbytehash atau API receipt. Otorisasi upload ClassOnly sebelumnya sudah terpakai: jangan upload ulang.

## Postmortem tanpa fit

Pada full-six, class-only menaikkan MacroF1+0,000687, tetapi AUC hanya+0,000170, AP−0,000270, logloss+0,000473 danBrier+0,000063. Pada late, logloss/Brier memburuk dan AP turun meskipun AUC hampir tetap. Keuntungan F1 internal bukan bukti kuat ranking/informasi baru.

SavedTest predictions berubah dari6.621 menjadi6.810 prediksipositif:190label0→1 dan1label1→0. Ini **predicted prevalence**, bukan proporsi labelTest sebenarnya. Tidak tersedia labelpublic untuk mengidentifikasi flipmana yang salah, menguraikan gapF1public atau membuat confidenceinterval paired yang sah. Tidak melakukan threshold/weightsweep dari skorpublik.

## Prioritas1 — C19 continuation

C19 merupakan XGBoost binary pada exactstructured/OHE+TFIDF/SVD32, present0,75currentSparse25+0,25C19raw, missingCB,threshold0,50. Model ini sudah ada, bukan resepbaru hasil tuning. Pilotlama gagal primaryhurdle dan satu recallguard; verdictRETIRE tetap historis.

Perbandingan propermetrics pada **H12 yang sama (38.588baris, origin55/67)**:

| Delta vs sparse25 | Class-only | C19 |
|---|---:|---:|
| MacroF1 |+0,000596|+0,004152|
| ROC-AUC |−0,000038|+0,003836|
| Averageprecision |−0,000498|+0,007331|
| Logloss (lebihkecilbaik) |+0,000028|−0,005885|
| Brier (lebihkecilbaik) |−0,000026|−0,002064|

C19AUC/AP/properloss membaik pada keduaorigin, termasuk H12origin67 F1+0,005160. Primarypilot9.064baris hanya+0,000375: jangan menyamakannya dengan full-six class-only38.588baris atau menganggapperbaikanH12 menjamin public. Ini alasan substantif membayar4missingoriginfits, bukan mengulangcapacity/weightgrid.

Luna melengkapi exactC19 pada59/63/71/75 (4fits), reuse55/67; primaryfullsix38.588unique danpilotH12 terpisah. Sebelumfits, audit ulang cachedpilot danall-view guards sesuaibenchmarkriskceiling3pprecall/1ppF1. Originalqualification2pprecall masihdicatat. CSV opsional hanya bila primaryfullsixgain≥0,0005,H12pooled≥0,002,H12origin67≥0,001,lateharm≤0,0005,W6harm≤0,001,≥4/6window nonnegative,allguardslulus. Persisnilai protokolcanonical, tidakdiubahsesudahoutcomebaru.

## Prioritas2 — TTR tree matched pilot

PriorLRablation42fits menunjukkan TTR hampirtidak memberi informasi tambahan setelah hybridTFIDF. Eksperimen baru hanyamenguji gap nonlinear: originalsparseLightGBM +exacttokencount dibanding exacttokencount+TTR. Regexlowercase[A-Za-z0-9]+ samaEDA; count0untuktidakadatoken,TTRNaN(nativeLightGBMmissing) saatn0. Rowlocal, noTFIDFtokenizerchange/noextraindicators/noquality/semantic/causalclaim.

Dua arm×origin55/67=4fits. Baselinecachedsparse25,C2B/CB terverifikasi, no baselinefitbaru. Gateprimary≥0,0005vsbaseline dan≥0,0002vsCountOnly,H12harm≤0,001,W1/W4harm≤0,005;groupn200,eachclass30,F1harm0,01/recallharm0,02 padaall6views. Terlepaslulus/gagal, stopsetelahpilotuntukreview; tidakadaauto8fitcontinuationatauCSV/submissionTTR.

## Budget, QA dan handoff

Current40estimatorfitsconsumed dari cap60. Maksimal9plannednewfits:4C19temporal+4TTRpilot+1C19fullTrainconditional;max1technicalretrybaru. Maksimal50consumed bila semuaeksekusi+retry. Categorycontinuation prior13+9=22≤24;4slotdialokasikan eksplisituntukTTRNEW_PILOT. Technicalretryprior1+max1=2≤4. No capreset/controlrefit.

Dua skrip zero-fit membuktikan row/ID alignment dan hash, hasilF1 sesuai8referencechecks dari laporan terverifikasi; officialTrainhash cocok. CompilechecksPASS, no newpytest suite/noofficialfits. Roottidak mengklaim tesmodelbaru sudahlulus: Luna wajib exactpayloadbuilder E2E sebelumtraining.

Kontrak engineering: simpanmodel/rawprobabilities durably sebelumpublikasi/CSV; mandatorybaseline_prediction jelasprobabilityvslabel; CSVbytes tidakmelewati JSONwriter; semanticintegerreadback; sourceidentitytidakbergantungHEADyangberubah; no objectNPZ; no refituntukreporterror. Wrapperbaru saja, frozenmodel/source/history/notebooks untouched.

Protokol: experiments/optimization_v3/classification_next_v1/protocol.json. Bukti: [CLASS_ONLY_POSTMORTEM.json](CLASS_ONLY_POSTMORTEM.json), [C19_DIAGNOSTIC.json](C19_DIAGNOSTIC.json), [QA.json](QA.json).

SetelahLuna mengeksekusi, kirimhasilkeSol danstop. Solmenentukanuploadperfile setelahQA; tidakadauploadbaru di milestoneini. Soltidakpolling/terusresearchsambilmenunggu.
