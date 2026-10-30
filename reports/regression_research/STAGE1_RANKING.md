# Stage 1 — prediction-only within-cycle ranking

Frozen before evaluation: four existing structured controls, raw/clip and rank weights0.25/0.50/0.75/1.00. No model fits, official-Test predictions or Kaggle submissions. The rank transform uses only model scores and cycle labels; average ties,100*rank/N. This is conditional on supplied-batch availability and does not establish the official target-generation formula.

| Model | Policy | Primary MAE | Equal-cycle MAE | H12 MAE |
|---|---|---:|---:|---:|
|R1_C|raw|15.148986|15.254088|15.477893|
|R1_C|clip|15.148986|15.254088|15.477893|
|R1_C|rank25|15.048377|15.129277|15.289767|
|R1_C|rank50|15.111231|15.185750|15.300884|
|R1_C|rank75|15.346841|15.432047|15.529608|
|R1_C|rank100|15.760725|15.872374|15.983442|
|R1_D|raw|13.349836|13.589657|15.577240|
|R1_D|clip|13.347935|13.587643|15.575348|
|R1_D|rank25|13.297560|13.537011|15.548290|
|R1_D|rank50|13.370191|13.610737|15.621319|
|R1_D|rank75|13.555653|13.799321|15.806381|
|R1_D|rank100|13.842403|14.091800|16.107358|
|B_D_RECENT8|raw|13.217154|13.507105|15.794561|
|B_D_RECENT8|clip|13.216236|13.506013|15.793384|
|B_D_RECENT8|rank25|13.148815|13.439145|15.780927|
|B_D_RECENT8|rank50|13.198969|13.488862|15.862427|
|B_D_RECENT8|rank75|13.361475|13.651969|16.044724|
|B_D_RECENT8|rank100|13.632073|13.928145|16.336227|
|A_NO_CODES|raw|14.078601|14.190019|15.053467|
|A_NO_CODES|clip|14.077783|14.189162|15.051767|
|A_NO_CODES|rank25|14.054448|14.161256|15.025571|
|A_NO_CODES|rank50|14.145561|14.251512|15.119652|
|A_NO_CODES|rank75|14.348016|14.458919|15.343531|
|A_NO_CODES|rank100|14.656939|14.782342|15.702852|

**Finding:** full rank mapping worsens every control on pooled primary and H12. A25% blend gives small gains but is not near the baseline target. It improves tails while some EF0 long-horizon groups worsen. Rank mapping changes score spacing/marginal distribution, not pairwise inversions. Do not select a separate blend from each outer window. Next-stage learned choices use past inner temporal folds.

The included incomplete-batch stress randomly retains25/50/75% per cycle using target-blind fixed seeds. It compares subset mapping with full supplied-batch mapping on identical retained validation rows. It tests reference sensitivity, not external source completeness. Test contains complete supplied cycle groups80–91 with1,060–3,317rows/cycle, but completeness relative to the real source universe remains UNKNOWN.

All validation cycles have been seen in prior development. H12 overlaps primary. These fixed-policy comparisons are exploratory; the minimum over several weights is not an unbiased performance estimate. No Kaggle slot is spent on a small-only mapping change at this stage.

Mapping implementation follows SciPy average-tie semantics: https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.rankdata.html . Local frozen receipts and saved validation vectors are authoritative for measured scores.
