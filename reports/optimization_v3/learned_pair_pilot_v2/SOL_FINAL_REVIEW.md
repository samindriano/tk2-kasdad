# Sol review: close the bounded FM lane

Decision: ACCEPT_DIAGNOSTIC_RESULTS; RETIRE_V2_RECIPE; NO_FURTHER_FM_FITS.
Reviewed result checkpoint c8b87c8c84b3a021893790a7510e55679827ba6e, clean.
No new training, prediction scoring, target evaluation or Kaggle access in this
review. Original V1 remains FAIL_OPTIMIZER_INCONCLUSIVE; V2 now provides negative
predictive evidence for its exact trained recipe, not every possible FM design.

| View | n | FM Macro F1 | No-pair | C2_B bag | A02 | FM minus A02 |
|---|---:|---:|---:|---:|---:|---:|
| E67 H12 | 20903 | .763788 | .804906 | .805106 | .811159 | -.047371 |
| E75 W6 | 8491 | .785334 | .809872 | .810384 | .819604 | -.034270 |

MECHANISM_FAIL and COMPETITION_FAIL are material performance failures, not
near-threshold risk rejections. No-pair is close to the faithful C2_B baseline
on both main views; a large common baseline reconstruction error is therefore
not supported by these aggregate results. This does not independently certify
every implementation detail or remove differences in optimization difficulty.

E67 H12 FM AUC .871872 is below no-pair .892198 and A02 .899755; log loss
.448589 is worse than no-pair .390968. Positive recall drops to .576483 versus
no-pair .744611. E75 AUC/log loss also worsen. The evidence is not confined to
threshold .50 or a single subgroup. All12 E67 cycle gains and all4 E75 cycle
gains vs comparators are nonpositive. E75 W6 overlaps E67 W6 and is not an
independent sample. No threshold, blend, regularization or seed rescue follows.

FM fit objectives are lower than no-pair, while temporal performance is worse:
this is consistent with overfitting or a poorly transferring interaction recipe,
not proof of one cause. Stationarity and solver success do not establish good
generalization, global optimality, or that all learned pair information is absent.

Reviewed POSTFIT_RESULTS.md, independent scored-verification status
PASS_INDEPENDENT_METRICS_AND_GATES, completion receipt and Git scope. Independently
checked report-result/verifier hashes against completion metadata and the live
ledger hash. This review did not replay all source code, fit gradients, prediction
arrays, or metrics from raw labels. Reported26 tests and synthetic lifecycle PASS
are executor evidence, not tests rerun by Sol.

The documented zero-fit export amendment preserves13 safe NPY members byte for
byte and reconstructs Unicode IDs from verified source identity. Original object
payload was not deserialized; original malformed archives and failure evidence
are retained. Pre-label PASS precedes query evaluation per completion receipt.
Four V2 fits total; no extra fit during export correction. Ledger:292 events,
114 starts,110 saved,4 historical failures,0 pending; cap117. Hash after completion:
d23c8d365fdf3daae99600ab74a4c02818b0fad27f030fd051a5bdf5c79e8c9f.

Keep A02 Classification and R06 Regression as existing best-public-observed
references from Sam's receipts, without asserting private leaderboard superiority.
No FM refit/full-Train/Test/submission. No automatic next research lane. The
recommended next phase is evidence consolidation and academic finalization;
this decision does not start that work. Stop for Sam's review.
