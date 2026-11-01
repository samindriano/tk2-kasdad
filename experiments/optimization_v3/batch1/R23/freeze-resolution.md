# R23 freeze resolution

The initial R23 freeze (`freeze.json`, signature
`27208c7f7d917f5149cf9f2f00e778bccc72a162a3ede7f25b85d218d2f777a4`) was
created and committed before any official candidate fit. Synthetic end-to-end
verification then found that the serialized `transformed_feature_names` list
described the pre-OHE structured columns, while the ExtraTrees matrix used the
expanded one-hot columns. The resulting width audit rejected the synthetic
report after fitting.

No official candidate model fit or pilot was run under the initial freeze. The
synthetic test fits used temporary synthetic inputs only. At amendment time,
there were no R23 pilot bundles, pilot report, or results/verification reports
in its run/report paths. The initial recipe and freeze remain preserved
unchanged.

R23 v2 corrects the fitted-state feature-name metadata to enumerate numeric,
expanded OHE, SVD32, and stream-missing columns in design-matrix order. It keeps
the estimator parameters, fitted transformations, feature allowlist, folds,
prediction policy, gates, and official data unchanged. The amended recipe and
freeze are stored as `recipe.v2.json` and `freeze.v2.json`; v2 run artifacts are
isolated under `experiments/runs/optimization_v3_batch1/R23/v2/`, and reports
use `.v2` filenames. The v2 recipe and freeze bind the exact SHA256 and
signature of the superseded v1 artifacts.

The v2 freeze signature is
`30251fd06b02169af6e3de776a33d0dc3e30506b6f32c1dcf7e06f22688b3167` and its
SHA256 is `7ced7df1217958dba286c4e1e109741728fd14387391ebad8709f6a40924e046`.
The v2 recipe and freeze are now stored at the paths above and both bind the
v1 recipe/freeze hashes and v1 freeze signature.

Synthetic QA completed the R23 v2 fit, independent verification, and report
publication path successfully. It used synthetic rows only and does not count
as an official pilot fit.
