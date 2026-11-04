# CLS Bigram Position REAL E79 — exploratory benchmark

**Status: candidate prepared for Sol review; no Kaggle upload.**

The benchmark reused the verified full-Train C2_B, original unigram POS,
A02, and C1_B states, then ran exactly one new REAL bigram-position gamma
fit on all 50,608 official Train rows. No BAG/SHUFFLED E79 fit, retry,
test-label read, SampleSubmission access, or Kaggle call occurred.

- Candidate CSV: `submissions/optimization_v3/bigram_position_real_e79_v1/classification_bigram_position_real_e79.csv`
- CSV SHA-256: `4d413a402173762987f5d5813bc75b02666ffede6c18fc00c629a7e3b3b63666`
- Full-Train fit ID: `CLS_BIGRAM_POSITION_REAL_E79_V1_GAMMA`
- Gamma SHA-256: `448aba61b399dd4ef8cca42e3ba7165db017b5a541759b1efb40dbf12ea300dc`
- Test rows/positive predictions: 29421 / 6842
- Missing-stream rows: 1062; present rows with no known bigram: 4
- Zero-gamma equivalence: PASS; maximum present difference 0
- Ledger: 1 start, 1 saved, 0 failed, 0 pending; SHA-256 `8ee720fac6b5be4f2727bad4980299d39688dc339f6e3eaacc0144f0cb3358e9`
- Test labels were not read; the candidate has no public score and remains unsubmitted.

One post-freeze inference attempt stopped before CSV creation because the
historical CGeom receipt names its target-isolation flag
`test_targets_loaded`. A separately committed compatibility adapter maps that
verified false flag in memory to the runner's expected singular name. It did
not change the training source, recipe, fitted state, or historical receipt;
the final inference then passed and created the CSV above.

## Interpretation boundary

This is an exploratory E79 inference artifact, not a new internal-validation
result and not a qualified champion. It does not alter the failed diagnostic
verdict of the E67/E75 Bigram Position pilot or the separate Moments lane.

## One zero-fit explanation for a later review

In the existing pilot, E75 W6 REAL was 0.001482 Macro F1 below MIX50 and
also below both BAG and SHUFFLED. Before authorizing another fit, inspect
whether the saved E75 REAL gamma's prediction changes are concentrated
near the 0.50 threshold and whether those changes increase false positives
or false negatives. This is a prediction/error decomposition only; it does
not justify a new fit or change the historical gate.
