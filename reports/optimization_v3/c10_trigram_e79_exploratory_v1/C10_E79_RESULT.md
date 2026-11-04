# C10 Trigram E79 — exploratory full-Train artifact

**Status:** one C10 predictor fit and Test inference are complete; CSV and
independent QA are ready for Sol's review. No Kaggle upload was made in this
lane. This is a full-Train inference artifact, not a validation result or model
promotion.

## Frozen recipe and boundaries

- Fit population: official Classification Train, 50,608 rows, cycle 47–79;
  raw Train SHA-256 `875552a68a3d24b0f3e1966d6ddb5671e6f54305a1c526e2f0a76747942ed91c`.
- C10: original word 1–2 TF-IDF plus separate word-trigram TF-IDF, structured
  preprocessing, and LightGBM binary classifier with 400 trees. The signed
  recipe is in `RECIPE.json`; the prefit source is `7a1fb2e7d39b10d060f2376ee95fbcaa38000954`.
- Present stream route: `0.75 × C2_B + 0.25 × C10`; missing stream route:
  verified Sparse25 E79 CatBoost probability bit-exact; fixed threshold 0.50.
- Reused C2_B, CatBoost missing, and Sparse25 components are each documented as
  fitted on all 50,608 official Train rows. The baseline bundle and component
  hashes are pinned in `PREFIT_AUDIT.json`.
- Test was read only after the trained-state freeze commit. Test labels,
  SampleSubmission, and Kaggle were not accessed. No predictive score was
  measured from this full-Train run.

## Fit persistence event

Exactly one official predictor fit started. Model fitting and `joblib.dump`
completed, then the original runner raised Windows `OSError: [Errno 9] Bad file
descriptor` while calling `fsync` on a file opened read-only. The original
`FIT_FAILED` event remains in `BUDGET_LEDGER.jsonl`; the fit was not repeated.

The saved `.partial` model loaded as the frozen C10 class with 50,608 fit rows,
the same target hash as `FIT_STARTED`, 10,000 trigram features, and 400 trees.
A writable-handle recovery copy is byte-identical to the retained original
partial model (SHA-256
`3093fb468bc670e2aebebd8b13c794c6bfa91a1793550016b714d43c217e8f4c`). The
separate `FIT_RECOVERED` event consumes zero extra units; no retry occurred.
Fit duration is unavailable and left null. Details are in `FIT_RECOVERY.json`
and `PERSISTENCE_FAILURE_NOTE.md`.

## Test artifact and QA

- CSV: `submissions/optimization_v3/c10_trigram_e79_exploratory_v1/c10_trigram_e79_exploratory.csv`
- Rows and columns: 29,421; `record_id, artifact_signal`; exact official Test ID
  order with no missing or duplicate IDs; integer labels are only 0/1.
- CSV SHA-256: `7ca190b8fe27f409eb425bef4aaee7e39e73317e8478cf9f8374dbe4db626da1`.
- Saved inference bundle:
  `experiments/runs/optimization_v3_c10_trigram_e79_exploratory_v1/test_predictions.npz`
  (SHA-256 `7b42a5177acbc664839cb83ea80e6d422b5b14693382910923a7459daf50c8dd`).
- Routed rows: 28,359 stream-present and 1,062 missing-stream. Predicted labels:
  22,790 zero and 6,631 one.
- Independent verification passed: the saved blend and missing fallback
  reconstruct exactly; all probabilities are finite and within [0,1]; CSV IDs,
  order, schema, and labels match the saved outputs. The QA reads saved arrays
  and does not rerun model inference.
- The first inference process wrote the CSV and NPZ, then failed while building
  its receipt because it passed an unsupported `dtype=` keyword to
  `array_digest`. A separate finalizer verified the existing files and wrote
  the missing receipt; no Test prediction was recomputed. See
  `INFERENCE_RECEIPT_FAILURE_NOTE.md` and `INFERENCE_BUILD_RECOVERY.json`.

The independent QA record is `QA_VERIFICATION.json`. The historical C10 pilot
verdict remains **RETIRE**: its primary gain was +0.000342846 versus Sparse25,
below the frozen +0.0005 gate; the H12 gain was +0.000878127. This E79 artifact
does not change that verdict or establish new validation performance.
