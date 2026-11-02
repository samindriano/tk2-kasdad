# A02 next decision: ranking versus allocation

This is adaptive, Train-only, zero-fit decision research. The public A02 gain and
all prior development results are already known. No new prediction policy,
threshold, submission, fit, or budget increase is authorized by this analysis.

Before opening the new results, fix these controls:

- Use the four existing audited row-level views, preserving missing-stream and
  W1 fallback predictions exactly. Verify official Train hash, row/cycle/label
  alignment, and unchanged ledger before and after.
- Within each (origin, cycle) eligible present-stream group, compute K_A from
  the frozen A02 labels and K_B from the frozen Sparse25 labels, threshold >=.50.
- Rank Sparse25, A02 and shuffled probabilities at the same K_A; additionally
  rank A02 at K_B. Ties use ascending official source row position. No labels
  are passed to the rank/count selector.
- Repeat the same fixed controls at origin level as a sensitivity view, not a
  choice of the better performing policy. Never combine outcomes across views
  as independent evidence; H12-L overlaps primary.
- Report paired confusion counts and pooled Macro F1. At equal positive count,
  report differences in true positives. Split the total F1 difference along two
  explicitly order-dependent algebraic paths: rank change then count change,
  and count change then rank change. Do not call either a causal attribution.
- Report the five learned primary origins separately. W1 is exact fallback,
  not a sixth learned origin. Produce no row-level transformed predictions.
- A matched-count advantage can motivate further ranking research. It cannot
  select K for deployment or justify a new submission. If it vanishes, reassess
  whether allocation rather than local residual information drove the gain.

No threshold/grid search, fit, Test/SampleSubmission/Kaggle access, or new data.
Use meaningful tests for ties, K=0/K=n, target-free selection, unchanged fallback,
confusion metrics and exact algebraic reconstruction. Record all input hashes.

## Subsequent descriptor question (registered before its results)

The matched-count comparison is positive on four of five origins. Before
selecting a next mechanism, describe the squared-cosine-weighted absolute gap
between each neighbor's past-only OOF probability and the query's Sparse25
probability. Also describe the fraction of weight within absolute gap <=0.1.
These use no query outcome to construct the descriptors. Compare useful and
harmful flips at each learned origin and in the four already fixed boundary
bands. Show all counts rather than demanding a new 20+20 stratum gate. Small
groups remain uncertain. Do not fit a router or optimize a gap threshold.

This examines the assumption that a text neighbor's residual transports across
different baseline-confidence regimes. It does not prove confidence matching
will improve F1; it may be redundant or reduce effective support.
