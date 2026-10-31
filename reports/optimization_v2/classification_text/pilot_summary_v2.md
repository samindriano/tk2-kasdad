# CatBoost Native Text Classification Pilot — V2

## Verdict

The pilot **does not pass its preregistered continuation gate**. It did not improve pooled W1+W4 Primary or pooled H12 Macro F1, and W4 and H12_L each lost more than the allowed 0.01 Macro F1. No cutoffs beyond 55 and 67 were fitted.

## Frozen method and audit

V1 was stopped before any native-text fit. Its preflight incorrectly expected the archived `CB_MISSING` model to be trained only on missing-stream rows. The cache audit instead shows that it was fitted on every row through the cutoff, with the stream mask used only to route validation predictions. The pre-fit failure is preserved in [`prefit_failure_v1.json`](prefit_failure_v1.json), byte-identical to its ignored-run copy.

V2 fixes only that controller check. It independently validates both archived `C2_B` and `CB_MISSING` fit fingerprints and outer row/cache identities for both cutoffs **before** the first candidate fit. V1 source, protocol, freeze, and failure evidence remain intact.

The candidate fits all fold-training rows using the frozen expanded structured/native-categorical parser plus CatBoost native `signal_stream` text. The fixed native-text design uses a whitespace tokenizer, one unigram dictionary (`occurence_lower_bound=3`, `max_dictionary_size=20000`), and `BoW` + `NaiveBayes`. Blank streams map to `__KASDAD_EMPTY_STREAM__`; the original availability mask is retained before normalization. There were 3,404 blank/missing streams in official Train and no collision with the sentinel token. The candidate replaces only the present validation branch; the cached missing branch remains byte-for-byte unchanged.

The configuration follows CatBoost's [text feature documentation](https://catboost.ai/docs/en/features/text-features), [feature calcer reference](https://catboost.ai/docs/en/references/text-processing__feature_calcers), and [Python Dictionary reference](https://catboost.ai/docs/en/concepts/python-reference_dictionary). CatBoost documents BoW as token-presence features and NaiveBayes as an online-permutation calcer intended to avoid target leakage.

V2 freeze signature: `25632bd06075e13f4e950bfa4a6353320b1d200d9b2410b787621bcca06656d5`.

## Results

| View | Rows | Current champion | Native text | Delta |
|---|---:|---:|---:|---:|
| W1 | 4,114 | 0.802090 | 0.808886 | +0.006796 |
| W4 | 4,950 | 0.840297 | 0.828055 | -0.012242 |
| H12_E | 17,685 | 0.792671 | 0.797229 | +0.004558 |
| H12_L | 20,903 | 0.803724 | 0.786201 | -0.017523 |
| Primary, pooled W1+W4 | 9,064 | 0.822903 | 0.819333 | -0.003570 |
| H12, pooled E+L | 38,588 | 0.798984 | 0.792357 | -0.006627 |

At threshold 0.50, W1 and H12_E improved, while W4 and H12_L regressed. Both failing views exceed the allowed 0.01 loss. The candidate's pooled Primary AP and Brier were slightly better (`0.846236` vs `0.831262`; `0.123089` vs `0.123633`), but pooled Primary Macro F1 was lower. On pooled H12, AP and Brier were both worse (`0.798288` vs `0.800403`; `0.133002` vs `0.129230`). No threshold changes or post-hoc calibration were attempted.

The missing-stream branch is exactly unchanged in every view. For example, pooled Primary missing-stream Macro F1 is `0.810162` for both routes; pooled H12 missing-stream Macro F1 is `0.772261` for both. The differences therefore come from the present branch only.

## Independent verification

The verifier independently recomputed per-class precision/recall/F1, confusion matrices, average precision, Brier score, stream groups, entry-friction groups, pooled views, and the gate from official Train labels and cached probabilities. It also checked all-row fit fingerprints/targets, exact outer IDs, source/cache hashes, sentinel handling, and branch equality.

Verification status: **PASS**. Details are in [`verification_v2.json`](verification_v2.json); full metrics are in [`pilot_results_v2.json`](pilot_results_v2.json). The model fit and input identity are recorded in [`freeze_v2.json`](../../../experiments/optimization_v2/classification_text/freeze_v2.json).

Only the two preregistered candidate fits ran. Test data, Kaggle APIs, threshold searches, blends, and additional folds were not used. H12 consists of known development cycles and is not an untouched holdout.
