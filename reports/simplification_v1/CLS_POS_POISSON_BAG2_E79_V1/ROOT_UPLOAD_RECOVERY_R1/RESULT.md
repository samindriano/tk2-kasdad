# CLS POS Poisson Bag2 — root benchmark result

Kaggle confirmed submission **58047602** as COMPLETE with public Macro F1 **0.83024**. The existing champion remains **0.83225** (reference 58001282); the observed difference is **-0.00201**. No final submission selection was changed.

The uploaded CSV SHA-256 is `c83f9fad3592795f14ae8401fbee0155b070d4ce95d7235a65572214b5ef5363`. Independent root readback verified all 29,421 official ordered Test IDs, binary labels, the exact original champion reconstruction, probability averaging before thresholds, and unchanged fallback on 1,062 missing-stream rows. There were 136 changed decisions: 73 from zero to one, and 63 from one to zero. No duplicate vector was found among 26 aligned historical local CSVs.

Exactly one upload was made under Sam's standing measured exploratory benchmark authorization. The failed historical application attempt is preserved. Recovery used the existing two verified final replicas and performed zero additional fits. Test labels were not accessed. Raw predictions and model artifacts remain ignored.

The positive unique-pool local gain of 0.000173362 was small and temporally mixed; the pilot had not met scientific qualification. Public performance provides counterevidence to promoting this fixed bagging recipe. Keep the champion; do not tune replica weights or thresholds against this public result. The independently authorized regression branch continues unchanged.

Evidence: ROOT_QA.json, UPLOAD_INTENT.json, UPLOAD_RESPONSE.json, SCORING_STATUS.json, and the before/after Kaggle submission histories in this directory. Private score remains unknown.
