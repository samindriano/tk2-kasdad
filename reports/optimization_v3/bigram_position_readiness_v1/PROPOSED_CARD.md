# Proposed card: CLS_BIGRAM_POSITION_PILOT_V1

**Status:** proposed only; no fit budget reserved and no fit authorized by this readiness check.

## Question

Does three-part position information for known word-bigram occurrences add predictive information beyond an identical frozen C2B bag + original unigram POS logit, bag bigrams, and a shuffled-position control?

## Fixed data and feature contract

- Train source: official Classification Train only. Use temporal origins E67 and E75: fit `cycle_code <= origin`, query `cycle_code > origin`.
- Reuse frozen E67/E75 bigram vocabulary and IDF from `POSITIONAL_TEXT_PILOT_V1_E{67,75}_C2B_TFIDF`. Verify artifact SHA-256, receipt, Train input SHA-256, fit-cycle maximum, tokenizer parameters, and vocabulary/IDF alignment before use.
- Tokenize each document with the frozen vectorizer preprocessor and tokenizer. For tokens `t[0:n]`, form the full adjacent pair sequence `(t[j], t[j+1])`, with `n_pair = max(0, n-1)`. Assign occurrence `j` to `floor(3*j/n_pair)`. OOV pairs remain in the sequence and affect bin boundaries but receive no feature column.
- Keep only exact pair terms present in that origin's frozen prefix bigram vocabulary. Compact columns in original vectorizer column order; use that term's original IDF. For each known term/bin, use `(1+log(count))*IDF`; concatenate channels and normalize the whole row L2 once.
- Arms: `BAG` uses all known pair counts without position; `REAL` uses observed first/middle/final bins; `SHUFFLED` uses `deterministic_order_shuffle` on the pair-string sequence only with seed 42. Require exact pair multiset, sequence length, bag counts, and three bin sizes to match the real sequence.
- All arms share the same frozen offset `z_original = C2B bag logit + ORIGINAL unigram POS gamma`. Fit only the new arm's gamma. Use the existing POS_BAG_OFFSET objective, regularization, optimizer, convergence criteria, and resume protocol without changes or search.

## Fit plan and frozen route

| Origin | Fit rows | Query rows | BAG fits | REAL fits | SHUFFLED fits |
|---|---:|---:|---:|---:|---:|
| E67 | 29.705 | 20.903 | 1 | 1 | 1 |
| E75 | 42.117 | 8.491 | 1 | 1 | 1 |
| **Total** |  |  | **2** | **2** | **2** |

Total cost: **6 task estimator fits**. No vectorizer, C2B, original POS, A02, or missing-stream HGB refit is included. For arm `a`, `q_a = sigmoid(z_original + P_a @ gamma_a)`. Keep the final present-stream route `0.5*q_a + 0.5*p_A02`, threshold 0.50; keep missing-stream route at frozen `pP`.

If a present row has no known bigram, `P_a` is all zero and its output falls exactly to `0.5*sigmoid(z_original) + 0.5*p_A02`. Keep this deterministic incumbent fallback in every arm. Do not infer a token from the held-out query window or alter the feature vocabulary.

## Evaluation and controls

- Before any fit, independently verify source hashes, all frozen input/model lineages, output row identities, and exact prefix-only fit populations. Reuse the existing predeclared temporal views, subgroup guards, gate, fit accounting, and persistence/resume checks unchanged.
- Report BAG, REAL, SHUFFLED, and frozen anchor on primary, each registered window, H12, and every protected subgroup with denominator. Include negative results.
- Compare REAL against the frozen anchor, BAG, and SHUFFLED under the unchanged predeclared qualification gate. Do not add or relax a gate. If BAG/REAL/SHUFFLED collapse to the same descriptor for an arm or a frozen lineage is uncertain, stop before that fit.
- E67 query covers cycles 68–79 and E75 query covers 76–79; the later query is nested in the earlier one. Treat results as related temporal development evidence, not independent replication or untouched holdout.
- No Test, SampleSubmission, Kaggle score, leaderboard, or new tuning. No causal or semantic claims.

## Readiness evidence already available

- E67 query: 19.525/19.528 stream-present documents contain a known bigram; occurrence coverage 60,31%; every query cycle has at least 525 such documents; 86,41% of prefix bigram types appear.
- E75 query: 7.963/7.965 stream-present documents contain a known bigram; occurrence coverage 60,62%; every query cycle has at least 1.704 such documents; 77,49% of prefix bigram types appear.
- Pair shuffling changes the positional descriptor in 99,90% and 99,86% of feature-bearing query documents, while preserving the pair multiset and bin sizes.
- This establishes feature support and a meaningful null descriptor, not predictive benefit.

## Pre-fit fail-closed checks still required

1. Verify exact hashes and ordered row identities for the frozen `z_original`, `p_A02`, and `pP` artifacts; they were intentionally not opened in this target-free support pass.
2. Verify the exact objective/regularization, fit-only label population, optimizer status, and resumability against the original POS_BAG_OFFSET freeze. Do not reconstruct a near-match or tune a substitute.
3. Verify fit budget reservation and shared ledger before starting; reserve six new unique IDs exactly once. This audit did not reserve or release any fit slots.
4. Verify no other task has a training job active and obey the project's one-job/thread/RAM limits.

If any pre-fit check fails or returns UNKNOWN, do not start a partial fit. Report the blocker and wait for a decision. If all checks pass and the root explicitly authorizes the pilot, execute all six registered fits, verify and report the full comparison, then stop for review. Do not infer approval from this proposed card.
