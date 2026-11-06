# REG POS Poisson Bag2 — independently reviewed public benchmark

Kaggle confirmed submission **58047858** COMPLETE, public MAE **11.04202**. The frozen five-learner plus P50-hinge champion remains **11.01510** (58019539). Candidate harm is **0.02692 MAE** on this public split; private performance is unknown. Final selection slots were not changed.

Independent root review verified the narrow parent digest-format repair, reran the final artifact verifier (PASS), and passed all three repair tests. Root independently reconstructed mean raw POS scores, within-full-cycle T50, the 75/25 DIRECT ensemble and final present-only hinge from frozen NPZ members. All 29,421 ordered Test IDs, finite [0,100] values, CSV round-trip values, and 1,062 missing fallback values matched. No identical vector was found among 23 aligned historical local CSVs. The CSV SHA-256 is `4a7f5f8fb97cf377578f9de34635b54dd636450f4a29f71cfd88ff3b8ad46d55`.

The candidate changes 28,342 predictions; mean absolute change 0.2428882405, maximum 2.3498007390. These are feature-only prediction deltas, not target errors. Exactly two authorized full-Train POS replicas were started/saved, no failures or retries, zero preprocessing fits. Root performed zero new fits and no Test-label access. The original champion was exactly reconstructed before those fits. All other four learner outputs and final hinge were fixed.

The pilot's unique-pool MAE gain was only 0.000905214, with E67 +0.025464739 and E75 -0.020678012; scientific qualification remained FAIL. This upload was one explicitly exploratory benchmark under Sam's standing permission, not evidence that the recipe had robustly improved. Keep the champion; do not use this score to tune bootstrap weights or seeds. Further REG research, if authorized separately, must state its own mechanism and prospective protocol.

Evidence in this directory: ROOT_QA.json, UPLOAD_INTENT.json, UPLOAD_RESPONSE.json, SCORING_STATUS.json and before/after submission histories. Raw CSV/model/prediction binaries remain ignored.
