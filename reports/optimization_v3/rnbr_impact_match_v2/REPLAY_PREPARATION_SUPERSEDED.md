# Initial replay preparation — superseded before any official fit

This immutable preparation snapshot was generated before a synthetic verifier
fixture correction. It is preserved for audit history only and is not a replay
authorization or the freeze used for any official fit.

- Initial replay freeze: `1dd9d8fe078521e9fb0114460e19e61510c48924d8b6d3b87f779182265b2d3c`.
- The synthetic check originally allowed observed and shuffled predictions to
  coincide, so the independent synthetic verifier was strengthened to exercise
  a changed matched-control prediction and to validate the persisted bundle.
- No official TF-IDF replay fit was started from this freeze.
- No ledger event or reservation was written from this freeze.
- Replacement preparation is under the versioned `rnbr_impact_match_v2_1`
  namespace, with a fresh source hash and freeze signature.
