# Sol CSV review — exploratory hinge candidate

Decision: VERIFIED_CANDIDATE_PENDING_BATCH_SELECTION. No upload yet; collect Classification result before choosing the measured benchmark batch.

Root independently ran the complete final-application verifier under the shared lock and current resource gate: PASS_INDEPENDENT_FINAL_APPLICATION_REPLAY. It confirms 20,903 fit rows, exactly three cumulative LP starts (two historical diagnostic slots plus one final slot), 29,421 CSV rows, exact ordered unique IDs, 1,062 bit-identical raw-DIRECT missing-route predictions and the signed source/freeze/preflight/report lineage. No new fit, output overwrite, Test/Sample read or Kaggle access in this review.

CSV: `submissions/reg_monotone_hinge_p50_v1/hinge_p50_final_exploratory_e79.csv`; SHA256 `a5813fdd384da31611db1e945c7db33199d8ef9044fb71dd2941ef7af6422da9`. Columns: record_id, threat_rank. Finite values range 1.7288079036674198–99.28413951788829.

Final mapping: intercept 1.7288079036674198, low slope 0.9933285599956824, high slope 0.957778072288735, knot50. Apply clipping only on present rows; restore missing raw DIRECT exactly. The mapping changes all28,359 present predictions versus the current global-present-only candidate, mean absolute shift about0.21350, mean signed shift+0.08668, maximum absolute shift0.50376. Current-best ordered IDs match. Root also verified the algebra numerically. Python csv.reader float parsing confirms missing float64 bit identity; pandas default float parsing can introduce decimal-roundtrip differences around2.84e-14 and should not be used to diagnose raw bit corruption.

This final all-OOF calibration fit supplies no new independent validation score. Prior causal diagnostic gain remains+0.006360119 pooled, below+0.02 qualification; p90 and high-target harm remain counterevidence. Scientific qualification stays FAIL, exploratory status stays pending. No further fit in this three-slot lane; do not reset its budget. Root will choose later whether to upload this exact hash once as part of the research batch, without modifying final submission selections.
