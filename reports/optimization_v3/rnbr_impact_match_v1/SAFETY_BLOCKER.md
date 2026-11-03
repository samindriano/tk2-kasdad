# RNBR_IMPACT_MATCH_V1 — pre-fit safety blocker

**Status: BLOCKED_BEFORE_FITS.** The target-free support calculation passed its
scientific thresholds, but this preparation is not eligible to proceed to the
four task fits.

## Blocking data-boundary issue

The pilot protocol says Classification labels and probabilities are never read.
During `--prepare`, `_verify_a02_text_state()` in
`scripts/optimization_v3_rnbr_impact_match.py` called `joblib.load()` on the
whole A02 `ResidualNeighborPredictor` and then selected
`model.text_space.vectorizer`. The serialized predictor also contains
Classification-derived `bank_oof_probabilities` and `bank_residuals`. Loading
the whole pickle therefore deserialized those fields even though the pilot only
used the vectorizer. The generated `targetfree_preflight.json` assertion that
Classification labels/probabilities were not read is inaccurate and must not be
treated as a valid compliance attestation.

The same full-model load appears in the independent verifier. The E67/E75 A02
observed directories have no separate saved vectorizer artifact. A safe
vectorizer-only reuse route has not been established. Do not start any task fit
unless a separately reviewed, boundary-compliant route is available and the
protocol/freeze is rebuilt before fitting.

This is a technical/safety blocker, **not** a scientific preflight skip. The
support numbers below are recorded as exploratory diagnostics only; they do not
authorize fitting:

| View | Supported positive-cosine queries / present-stream denominator | Coverage |
|---|---:|---:|
| E67 W4 | 4,596 / 4,608 | 99.74% |
| E67 H12 | 19,480 / 19,528 | 99.75% |
| E75 W6 | 7,959 / 7,965 | 99.92% |

Shuffle source-position movement was 87.65% at E67 and 88.34% at E75. These
figures came from geometry constructed after the full model was unpickled and
are therefore not a clean protocol-compliant preflight result.

## Additional implementation blockers found in review

- `_load_original_rnbr_predictions()` reads the recovered E67 bundle paths at
  the wrong JSON level. The current report nests them under
  `serialization_recovery_detail.recovery_event`; verification would fail after
  fitting until corrected.
- The proposed result report does not yet preserve a reference to the inherited
  R06/legacy subgroup risks required by `NEXT_DECISION.md`.

## Work and access accounting

- The preparation commit is `511892f258b10884a75b48fb14667e3a15e1c476`.
- The frozen recipe/preflight files are preserved as historical preparation
  evidence. Their `classification_labels_or_probabilities_read: false` field is
  superseded by this blocker report.
- No Regression target values were opened, no official task fit started, no
  Test or SampleSubmission was opened, and no Kaggle access occurred.
- The shared fit ledger remains unchanged at SHA-256
  `35ca73c9adc64df8b794db29cc8cd39109362bea865a8bc6b1d8caf0c7703ff8` (201
  events; zero RNBR impact-match amendment/reservation/start events).
- The preflight command has exited; no impact-match process remains.
- No push was performed.

Next step: Sol/user review must determine whether a separately stored
vectorizer-only artifact can be used without reading Classification-derived
state. If not, retire this lane. Do not bypass the boundary by unpickling or
reconstructing the vectorizer from Classification model files.
