# Affine final-wave result

**Kaggle COMPLETE, ref57994383, public MAE11.04552.** Improvement0.09848 versus direct-position11.14400 and0.12068 versus originalR06 11.16620. New best observed public Regression in recorded campaign; privateunknown.

Fixed map **clip(2.169196428141562 +0.9744451489746011*direct_position_prediction,0,100)** applied toall29421rows including1062 formerexactR06fallbacks. ONE LP fitted on fixed20903 causalOOF Train rows (E67 cycles68-75 +E75 cycles76-79); no preprocessing/baselearnerfits. OOF calibrationfitloss is in-sample, notheldoutvalidation. No postscoreparametertuning.

Rootindependentlyverified manifest/QAsignatures, sourcepins, calibratorstatehash, CSVhash, exact29421IDs/order inheritedfromsignedofficialbase and bit-exactaffineclipreplay. CSV SHA256 `4abc56101e1863520771c125ad9d1f61e9f1530ee43575f2ae0cbd49331d4b56`. Oneupload, no retry/refit/finalslotselectionchange.

Interpretation: this observed benchmark supports a simple scale/location adjustment onthispublicsample; it does not identify whichrows/cycles causedgain or establishprivate superiority. Historical R17riskfailure was onanotherbaseline and remainsunchanged; newcandidateidentityCURRENT_DIRECT_AFFINE_L1_E79. No additional fits or uploads followautomatically.

Finalwave: C10 publicMacroF1.83021 (belowMIX50.83144); R16 scopedprerequisiteNOTFOUND/0fit; affinecandidateCOMPLETE11.04552. Requestedboundedwave ended.
