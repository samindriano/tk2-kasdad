# MIX50 present + A02 missing — E79 artifact

**Status: prepared for Root review; not uploaded.** This is a fixed recomposition of already saved E79 probabilities. It did not fit a model, run model inference, choose a threshold, read Test labels, read SampleSubmission, or contact Kaggle.

## Frozen policy

```text
present: 0.5 * pP + 0.5 * pA
missing: pA
prediction: probability >= 0.50
```

The source, synthetic checks, formula and input hashes were committed before CSV construction in prefit commit `221cc81692f7ce65c029d7db09a1288cc3eb7ab3`. The exact probability bundle hash is `ffc55aa2d18b75438257f3cf0aa205baa3f4c2922180e2b3688b36c352c00271`. Baseline MIX50 was replayed from the same bundle before recomposition.

## Prepared CSV

- Absolute path: `C:\Users\Sam\OneDrive\Documents\KASDAD\TP 2\submissions\optimization_v3\cls_mix50_a02_missing_e79_v1\classification_mix50_present_a02_missing_e79.csv`
- SHA-256: `3d970fa208d094e1e01c462671eeff327e05125bfacde2bacf36259b29fdf0bc`
- Rows / unique official Test IDs: 29,421 / 29,421
- Columns: `record_id, artifact_signal`
- Candidate positive labels: 6,777
- Missing-stream rows: 1,062; present-stream rows: 28,359
- On present-stream rows, probabilities and labels are bit-exact to MIX50.
- Among missing-stream rows, 107 labels changed: 42 changed from 0 to 1 and 65 from 1 to 0. Positive missing labels changed from 261 to 238.
- Full-label scan: 21 existing full-Test classification vectors; no exact duplicate found.

Independent CSV replay, ID/order/availability validation, and the duplicate scan all passed. QA confirms Test features were read only for `record_id` and `signal_stream`; no labels or SampleSubmission were read. The CSV remains local for Root review; this chat did not submit it.

## Files

- Frozen contract: `PREFIT_FREEZE.json`
- Synthetic QA: `SYNTHETIC_QA.json` (3 tests passed)
- Artifact manifest: `MANIFEST.json`
- Local checks: `QA.json`
- CSV: `C:\Users\Sam\OneDrive\Documents\KASDAD\TP 2\submissions\optimization_v3\cls_mix50_a02_missing_e79_v1\classification_mix50_present_a02_missing_e79.csv`
