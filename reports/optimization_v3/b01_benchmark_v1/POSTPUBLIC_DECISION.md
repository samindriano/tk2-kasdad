# B01: close the frozen public benchmark

Decision: CLOSE_B01_BENCHMARK; KEEP_A02_BEST_PUBLIC_OBSERVED_REFERENCE.

Samuel pasted the Classification submissions page showing:

| File | Status | Public Macro F1 |
|---|---|---:|
| b01_benchmark.csv | Complete | .82822 |
| a02_continuation.csv | Complete | .82930 |
| c19_xgb_svd32.csv | Complete | .82760 |
| samuel_rnbr_final_e79.csv | Error | unavailable |

B01 minus A02 is -.00108 Macro F1 (-.108 percentage points). This is one
user-reported public observation, not live/API verification, a significance
test, a private-ranking estimate, or proof that the mechanism never helps.
The C19 row confirms the later user-reported public submission; old reports
stating that full-Train C19 had not run describe an earlier checkpoint.
The erroneous Regression-file upload to Classification is not a scored model.

The local B01 CSV hash was rechecked against MANIFEST.json:
b30e6d4f732902bab43d9d28683975ac9d1820ce940853f232ba3936ae57a2d5.
The server hash/reference/exact upload time are unavailable. The appended
submission_receipts.jsonl records this limitation. The old PREPARED status
in the frozen preparation manifest is preserved as a historical snapshot,
not treated as current submission status or rewritten after the outcome.

B01's internal gains versus A02 were +.000261 full-primary, +.001762 late,
and +.003890 H12. Its original primary hurdle and four-of-five origin gate
failed. These internal views did not transfer to improvement on the observed
public subset; they were overlapping, adaptively reused development evidence.
No cause of this disagreement is identified by the public score alone.

Maintain the pre-benchmark stopping policy: no threshold, shrinkage, neighbor,
blend or seed tuning and no further submission arising from this result.
Keep B01's NOT_QUALIFIED verdict. FM V2 remains closed under its separate
negative controlled results. A02 remains the best-public-observed
Classification reference; R06 remains the existing Regression reference.
Academic selection still requires faithful recipe/reproducibility checks;
the public winner is not automatically the most appropriate demo model.

Next recommended phase is consolidation and academic finalization, not
automatic new research. This receipt milestone starts neither. No training,
new inference, label scoring, Kaggle action or frozen-model change occurred.
