# Classification Sparse MLP V1 — next decision

2 November2026. C19 continuation dan TreeTTR sudah selesai pada bb95480; originalverdict dan bukti tetap dipertahankan. C19 menaikkan full-six F1+0,000667,H12+0,004152,AUC/AP/properloss, tetapi W6−0,002599,3/6windowsnonnegative,danW2Q3F1harm0,010461 menggagalkan frozenpolicy. Tidak ada C19fullTrain/CSV/upload. TTR BOTH kalah COUNT:primary−0,000686,H12−0,000116; tutup scalarTTRtree recipe, bukan membatalkan EDAasosiasi.

## Research screening

NB-SVM sempat dipertimbangkan dari literatur, tetapi pemeriksaanrepo menemukan DC_NBSVM sudah diuji fullsix dan negatif(primarysekitar−0,0027,H12−0,0005) pada reports/optimization_v3/classification_diagnostics/FINDINGS.md. LinearSVC,word-char,calibration/stacking juga mempunyai counterevidence. Karena itu tidak melatih ulang metode tersebut dengan nama baru.

FM/DAN pernah direncanakan tetapi belum dieksekusi akibat backend yang belum tersedia. Kandidat baru memakai backend yang sudah terpasang: sklearn1.8 MLPClassifier satuReLUhidden32 pada exactC2_B sparsemain-effectdesign. Ini bukan fullDAN yang pernah direncanakan dan tidakmemerlukanTorch/pretrained/GPU. Dokumentasi resmi mendukung sparseinput dan menjelaskan epoch semantics: https://scikit-learn.org/1.8/modules/generated/sklearn.neural_network.MLPClassifier.html . Ini mendukung feasibility, bukan klaim bahwa hasil neural pasti lebihbaik pada tokenanonim.

## Fixed recipe dan pembanding

ExactC2_B numeric/OHE/TFIDF/missingindicator tetap; fit-onlypreprocessing, noSVD/newfeatures/normalization. MLP32ReLU,Adam,alpha0,01,batch256,lr0,001,seed42,30epoch,early_stoppingFalse,tol0,n_iter_no_change31,warm_startFalse. Semua parameter lengkap pada protocol.json. Epoch danseed tidakdipilih dari outerlabels/publicscore. Shuffle hanyapada FITprefix.

Candidatepresent=0,75Sparse25+0,25MLP; matchedzero-fitallocationcontrol=0,75Sparse25+0,25C2_B; missingCBexact danthreshold0,50. Matchedcontrol menguji apakah headbaru memberi manfaat dibanding sekadar mengurangi bobotsparseexpert. Karena optimizer/model berbeda, ini bukan identifikasi kausal efek nonlinearitas.

Dua pilotorigin55/67, masingmasing satuMLPfit. PrimaryW1+W4=9.064rows,H12=38.588rows, allsixviews danfitdefinedsubgroups sama. Gateprimary≥0,0005vsSparse25 dan≥0,0002vsmatched;H12pooledgain≥0,H12origin67harm≤0,0005,W1/W4harm≤0,003; supportedn200,eachclass30,F1harm≤0,01/recallharm≤0,03. Originalrecall0,02failures tetapdilaporkan. Propermetrics/loss curves dideskripsikan, bukan hyperparametersweep.

## Feasibility dan budget

SyntheticCSR512×256,3.932nonzero,dua fits seedidentik berhasil. Kedua runs selesai30epoch,trainloss0,690887→0,555623,finiteprobabilities danexactreproducibility. ConvergenceWarning fixed30epoch dicatat dan diharapkan; jangan mengklaim konvergen. Ini tidakmembuktikan realC2_Bpipeline/persistence/fullTrainresources sudahlulus. Luna wajib actualpayloadbuilderE2E/matrixparity sebelumofficialfits. Tidakadadatasetresmi dibuka oleh syntheticfeasibility.

48/60officialfits sudahconsumed. Maksimal2newfits+1technicalretrybaru→maksimal51/60. Continuation21+2=23/24; trueMLP_NEW_PILOT stage dicatat. Tidakada controlrefit/continuation/fullTrain/CSV/Kaggle diotorisasi. Semua fitstarted yanggagal tetapconsumed. No categoryreset atau menaikkan cap.

Satujob,duaCPUthreads,6GiBcap+2GiBreserve; estime seluruhCSR+score+Adamstate/gradients/transient+hidden activations. No rawTFIDFdensify; hiddenweights/activations memangdense danharusdihitung. Officialnonfiniteloss/predictions adalah technicalfail; fixedepochwarning adalah statusbudget, bukan alasan menambahepoch.

## Stop dan return

Luna mengimplementasi/QA/synthetic/realfits duaorigin/independentverification, lalu mengirim satu hasilkembali keSol danstop baikPASS/FAIL. Tidakadaautomaticarchitecture/epoch/alpha/thresholdrescue atau submission. Solberhenti selamaLunabekerja. Reporttrainloss/fitquality terpisah supaya kegagalan satu frozenrecipe tidak dianggap bukti bahwa seluruh neuralfamilymustahil.

Entryprotokol: experiments/optimization_v3/classification_mlp_v1/protocol.json. Feasibility: SYNTHETIC_FEASIBILITY.json. Rootfeasibilityskrip scripts/optimization_v3_mlp_feasibility.py; itu bukan runnerofficial. Historicaldata/cache/modelartifacts tetapignored.
