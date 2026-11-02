# Positional text — design-only reserve

## Search result

The scoped search over `reports/optimization_v3/`, `scripts/`, and `src/` for absolute token position, early/middle/late placement, or within-document position permutation found no explicit absolute-position experiment or result. This is a scoped search, not proof that no such analysis exists anywhere in the repository. Existing word unigrams/bigrams already encode token identity and local adjacent order; a candidate must isolate **absolute relative location within the full document**.

## Research question

For `artifact_signal`, does adding coarse absolute token location to the existing lexical text representation improve temporal predictive performance beyond the same ordinary unigram/bigram model at the same total feature budget?

## Provisional design

- Keep the exact existing `signal_stream` missing rule, document normalization, tokenizer, and `(1,2)` word n-gram source from `src/models/classification_models.py`.
- Assign each token occurrence to first, middle, or final third by normalized token index. Add a location-qualified unigram feature for that occurrence while retaining the same ordinary unigram/bigram channel. Do not infer token meaning or remove/rewrite tokens.
- Keep the same structured features, learner, temporal folds, metric, threshold, and total text-feature cap as the verified C2_B baseline.
- Matched control: keep each document's actual token order and ordinary unigram/bigram features unchanged, but permute only the early/middle/late tags across token occurrences within that document. Use a deterministic document-specific permutation from a frozen seed. This preserves each document's token multiset, length, tag counts, and local bigram order while breaking the association between a token and its absolute location.
- Compare position-aware against both the exact ordinary bag-of-words baseline and the matched position-tag permutation. All preprocessing is fit on the training prefix; evaluation labels are joined only after predictions are fixed.

## Still unresolved — not executable

The analyzer implementation, common candidate/control vocabulary construction, and exact rule for sharing the total feature budget have not been frozen. The audit has not verified that a saved C2_B baseline cache exactly matches a six-origin comparison for this design. Freeze these items and run synthetic/persistence checks before any official task fit. Do not implement a custom analyzer or start a fit as part of this design note.

## Cost

A full-six run requires six position-aware task fits plus six matched position-permutation fits (**12 new fits**), assuming the historical C2_B baseline is row-exact and reusable after verification. If it is not, add six baseline fits (**18 total**). Current Research Next capacity is zero, so this lane is a research reserve only and has no fit authorization.
