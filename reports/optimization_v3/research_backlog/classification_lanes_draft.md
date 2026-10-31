# Classification research lane backlog (draft)

**Status:** research plan only. No model was fit for this draft. The owned scope is Classification candidate ideas; execution, gates, and integration remain with MAIN.

## Evidence snapshot and operating baseline

The current verified comparator is the fixed-threshold **sparse25 route**: 75% C2_B + 25% raw word TF-IDF LightGBM on present-stream rows, with the existing CatBoost missing-stream branch unchanged. Its six-window primary Macro F1 is **0.829944** (38,588 rows), H12 is **0.802966** (same chronology reused), all six primary windows improve over the prior route, and the public score is 0.82908 (reference 57837821). This is the comparator for every proposed candidate below; public score is context only, never a gate.

Important evidence and counterexamples:

- `reports/optimization_v3/classification_sparse_full/FULLSIX.md`: sparse25 deltas are positive in W1–W6 (+0.001070 to +0.003989), pooled primary +0.002297, H12 +0.003982. H12 high-text Q4 has a counterexample of −0.002279. The threshold remains 0.50 and the missing route is identical.
- `reports/optimization_v3/classification_diagnostics/FINDINGS.md`: unseen `org_code`, `module_code`, or `pattern_code` affects 17,531 stream-present rows; C2_B error is 17.85% versus 13.42% when all are known. Unseen pattern alone has only 240 rows, too few for a stable claim. This motivates testing fit-derived unknown flags or past-only category statistics, but the error gap alone does not prove predictive value.
- Same diagnostics: C2_B residuals change sign by confidence band (+3.72 pp at p=.20–.40 in 5/6 windows; −3.75 pp at p=.60–.80 in 5/6). Yet the already-tested residual Head A in `reports/optimization_v3/classification_residual_fullsix/FULLSIX.md` loses pooled primary by 0.001430 versus the old champion and loses to sparse25 in all six windows; H12_E is its only favorable sparse25 counterexample. A simple reuse of that head is retired.
- `reports/optimization_v3/supervised_pilot/FULLPILOT.md` and `verification.json`: past-only target summaries over anonymous text tokens failed their fixed blend gate (primary −0.000679, H12 −0.001932 versus sparse25). Do not rerun that same token-target-statistics family.
- `reports/optimization_v2/FINDINGS.md` and `reports/classification_round2/FINDINGS.md`: native-text CatBoost, plain unigram-count/binary Logistic Regression, NB-weighted Logistic Regression, SVC, word+character Logistic Regression, broad text-weight changes, TTR additions, and threshold/allocation alternatives are failed or mixed. The exact same recipes are retired; distinct representation/learner hypotheses below are labeled as such and still need strong gates.
- `reports/autonomous_eda/classification_volume/FINDINGS.md`: the fit-only q80 routing experiment was already evaluated and improved C4_HGB only modestly (+0.000591 primary, +0.001129 H12). Do not repeat its original C2_B route. A zero-fit recomposition using the newer sparse25 score is a separate, narrow check only.
- `reports/optimization_v3/research_backlog/feasibility.json`: a label-blind official-Train scan found 47,204 non-null documents and 93,338 distinct word unigrams under both default and case-sensitive analyzers, with zero rows containing uppercase characters. It used no labels and no fits. Case-preserving text has no supported distinction in this corpus and is excluded. The descriptive full-Train vocabulary count is not a fit vocabulary: each candidate must fit its own vocabulary inside each temporal cutoff.

## Shared pilot protocol proposed for MAIN review

For model cards, use only official Classification Train and the established six temporal windows. Ordinary pilots use cutoffs 55 and 67, reusing the current frozen controls, with W1 (cycles 56–59), W4 (68–71), H12_E (56–67), and H12_L (68–79) reported separately. A card may state a different cutoff set when its feature lineage requires it. The combined W1+W4 subset is a cheap screen, **not** the six-window primary. Preserve missing-stream predictions unless a card explicitly tests that branch. Keep threshold 0.50. Fit every vectorizer, scaler, category vocabulary/statistic, dimensionality reduction, and estimator within its cutoff. Do not inspect official Test, use labels from other tasks, external/pretrained data, Kaggle scores, or teammate artifacts.

Suggested two-stage gate, to be frozen once by MAIN in the queued plan before unattended execution:

1. Advance a pilot only if the fixed W1+W4 pooled Macro F1 delta is at least +0.0005 versus sparse25, H12 pooled delta is at least −0.001, neither W1 nor W4 nor H12_E/L is below −0.005, and supported subgroup checks show no severe harm. Report primary pilot coverage as 9,064 rows and H12 as a repeated development view; do not relabel either as independent confirmation.
2. If advanced, run only the four not-yet-fitted cutoffs (59, 63, 71, 75). Retain only if six-window primary delta is at least +0.0005, H12 delta at least −0.001, at least 4/6 primary windows are nonnegative, and there is no severe supported stream, class, or text-load subgroup harm. Use a predeclared support floor (recommended pooled n≥200 and present in ≥4/6 windows); show smaller groups as descriptive only.

Every fit needs a frozen recipe, input/code hashes, exact row fingerprints, exclusive cache path, independent metric verification, and fit-only preprocessing checks. These gates are proposals for MAIN to integrate into the run queue. All cycles have already been used in model selection; they are adaptive development checks, not fresh holdouts.

## Candidate queue

The ordering favors cheap checks and hypotheses with local evidence. A candidate’s pilot normally requires **two new model fits** (cutoffs 55 and 67); continuation adds four. “Unknown time” means no matching timing was measured and should be recorded in the first pilot before scheduling more work.

### CL00 — Cached route composition audit (zero new fits)

- **Question:** Does the current sparse25 present expert remain the right route when combined with the verified legacy C1_B missing branch, and does the old fit-only q80 rule still help when the high-volume branch is sparse25?
- **Exact check:** Recompose already-saved aligned prediction vectors only. (a) Keep sparse25 present predictions and replace CB_MISSING by C1_B missing probabilities. (b) Keep CB_MISSING; route present rows with `text_load > q80_fit` to the sparse25 blend and the rest to C2_B, where each q80 is calculated from that fold’s fit rows only. Threshold 0.50. Score the six primary windows and H12 without refitting.
- **Why:** Diagnostics show CB_MISSING beats C1_B by +0.0291 F1 on the missing subset overall but loses W4 and H12_L; a changed present branch can alter pooled Macro F1. The earlier q80 route predates the sparse expert.
- **Gate / stop:** Compute once from verified caches. Before scoring, verify every branch has nonzero present/missing support in the six-window primary and high/low q80 support at each cutoff; otherwise mark that route comparison unsupported. Retire both alternatives if neither improves six-window primary without violating H12/window gates. No candidate-model continuation follows this audit.
- **Cost / risk:** Zero fits; cache lineage and row alignment must be verified first. Reusing a label-informed route threshold would be invalid; only the frozen fit-only q80 rule is allowed.

### CL01 — Present-only fit population for the sparse expert

- **Question:** Does fitting the present-stream routed expert only on present rows reduce interference from rows it will never score?
- **Exact recipe:** Current sparse LightGBM architecture and fixed parameters (binary objective, 400 estimators, 15 leaves, min child 50, lambda 3, learning rate .05, seed 42, two threads), same word TF-IDF and 25% blend with C2_B; fit only rows with present `signal_stream`. Keep CB_MISSING unchanged and threshold 0.50. Control is the all-row-fit sparse25 recipe.
- **Why:** The current present expert is trained on all rows then routed only to present rows. A prior present-only token-summary pilot changed both representation and population, so it did **not** isolate this population effect (`supervised_pilot/FULLPILOT.md`). This is a controlled ablation.
- **Pilot / gate:** Two fits at 55/67; advance under shared gate. For full-six, preserve the same present-only population at all cutoffs.
- **Cost / risk:** Two sparse fits; about 10–25 seconds per fit was observed for the word model as fit rows grew in `execution_protocol_draft.md`; present-only may change that. The missing-stream indicator is constant zero in this fit population and cannot be learned; record that expected degeneracy (or remove the constant column equivalently) rather than interpreting it as a signal. This can discard useful structured-label information and is not a leakage correction.

### CL02 — Strictly past-only category-novelty flags

- **Question:** Does a category novelty signal available at prediction time help when a structured ID has not appeared before?
- **Exact recipe:** For each training row in cycle c, compute three flags (`org_code` unseen, `module_code` unseen, `pattern_code` unseen) using only category values observed in cycles <c; do not use labels. For outer validation rows, compute the flags against the fold’s complete fit rows. Missing category values are not marked “unseen” (they remain handled by the existing missingness/preprocessor path). Append eligible flags to the current sparse-LGB structured block. Keep current word TF-IDF, LGB parameters, all-row fit population, 25% blend, missing route, and threshold unchanged.
- **Why:** Unknown-category records are common and show higher C2_B error (17.85% versus 13.42%, `classification_diagnostics/FINDINGS.md`). Existing one-hot encoding maps unknown values through its fit-only policy but does not expose a dedicated per-field novelty flag to the tree.
- **Pilot / gate:** Before fitting, build only the unlabeled novelty flags for each cutoff and verify they vary in training rows. Retain a field only if both flag states have at least 200 fit rows at both pilot cutoffs; if no field qualifies, retire before any model fit. Then run two fits at 55/67, reporting flag support and unknown-any/field denominators; suppress unstable subgroup claims below support floor. Advance under shared gate.
- **Cost / risk:** Two fits if the feature-only support precheck passes, otherwise zero; timing **unmeasured**. A nonzero validation unknown rate is not enough: training flags must have variation, otherwise the learner cannot estimate that effect. All category histories must be cycle-past-only, with no same-cycle look-ahead.

### CL03 — Strictly past-only structured-category target rates

- **Question:** Do past outcome rates for repeated organization/module/pattern IDs add information beyond one-hot/infrequent encoding?
- **Exact recipe:** Add one smoothed log-odds statistic for each of `org_code`, `module_code`, and `pattern_code`. For a training record in cycle c, compute its value using only labels from cycles < c. For category support n≥30, use `(positive_count + 30 * prior_rate) / (n + 30)` and logit-transform that rate; for n<30, use the global prior rate. The prior is prevalence from earlier cycles, or exactly 0.5 when no earlier labels exist (cycle 47). For outer rows use labels through the declared fit cutoff. Append to current sparse structured+word design; keep LGB parameters, all-row population, 25% blend, threshold, and missing branch fixed.
- **Why:** Fit-defined unseen categories affect 17,531 present rows, but the existing treatment is representation-only. This is distinct from the failed anonymous token target summaries: it tests structured IDs with repeated entities, not token meanings.
- **Pilot / gate:** Before fitting, run a label-aligned but model-free feature-support audit for both cutoffs. Require nonconstant ordered statistics for at least one category field at both cutoffs; otherwise retire before fitting. Add a synthetic same-cycle perturbation test and independent manual ordered-stat reconstruction before official fits. Then run two cutoffs 55/67; advance under shared gate and require no subgroup collapse among observed/unseen categories.
- **Cost / risk:** Two fits; runtime **unknown**. High leakage risk if any row uses same-cycle or future labels; stop if strict cycle lineage cannot be proved. Rate stability may be poor for high-cardinality patterns.

### CL04 — Add character-boundary n-grams to sparse LightGBM

- **Question:** Do subword fragments rescue lexical patterns that the word analyzer does not represent well, with a nonlinear learner?
- **Exact recipe:** Keep current word TF-IDF and add fit-only `char_wb` TF-IDF (`ngram_range=(3,5)`, `min_df=3`, `max_features=30000`, `sublinear_tf=True`, lowercase and Unicode accent stripping) to the same fixed sparse LightGBM parameters. Candidate output is the new model at a fixed 25% blend with C2_B on present rows; missing branch and threshold unchanged.
- **Why:** `DC_WORD_CHAR` word+character Logistic Regression lost (`classification_diagnostics/FINDINGS.md`; `reports/optimization_v2/FINDINGS.md`), but that was a linear learner. Current sparse LGB uses nonlinear splits over word TF-IDF and has improved all six windows; char fragments are a different feature/learner interaction, not a claim that the old LR result reverses.
- **Pilot / gate:** Before fitting, use a label-blind support scan at cutoffs 55/67; require at least 100 char n-grams meeting `min_df=3` at each cutoff so this is not just the word-only model again. Then run two fits; advance only under shared gate. Do not run extra char vocabulary variants if this fails.
- **Cost / risk:** Character vocab can be much heavier than word TF-IDF; time and peak memory are **unknown**. At inspection, visible RAM was ~31 GiB with ~7.9 GiB free in the shared execution protocol snapshot; recheck before fitting, serialize runs, use two threads.

### CL05 — Character-only sparse LightGBM (conditional)

- **Question:** Are character fragments complementary to the word model, or did any CL04 gain come only from added dimension/capacity?
- **Exact recipe:** `char_wb` TF-IDF only with CL04 parameters; fixed sparse-LGB parameters; score as a fixed 25% present-stream blend with C2_B; missing route and threshold fixed. Compare paired correct/error transitions against both sparse25 and CL04.
- **Why:** This isolates the character channel. It is new relative to failed word+char LR but is intentionally lower priority than CL04.
- **Pilot / gate:** Run two fits only if CL04 shows a nonnegative primary trend or if the pre-fit plan chooses this one-family decomposition in advance. Before fitting, require at least 100 char n-grams meeting `min_df=3` at both cutoffs. Do not use CL04 outer labels to choose among char parameter variants. Same shared gate.
- **Cost / risk:** Two fits, memory/time **unknown** and likely sensitive to vocabulary size. Retire the whole char family if it fails the fixed gate.

### CL06 — Separate unigram and bigram feature budgets

- **Question:** Is the current shared 20,000-term cap crowding out useful lexical identities or co-occurrences?
- **Exact recipe:** Two fit-only TF-IDF vectorizers: word unigrams (1,1), `min_df=3`, `max_df=.995`, `max_features=10,000`; and word bigrams (2,2), same thresholds/cap. Hstack both with unchanged structured features and train current sparse-LGB parameters; fixed 25% blend with C2_B, missing route, threshold .50.
- **Why:** Feasibility scan (`research_backlog/feasibility.json`) found 93,338 distinct unigrams in official Train while the current combined word vocabulary is capped at 20,000. This does not prove a cap problem, but makes feature-budget allocation testable. Earlier unigram-only LR failed; this keeps both n-gram levels and uses the different nonlinear sparse learner.
- **Pilot / gate:** Before fitting, require at least 100 unigrams and 100 bigrams meeting `min_df=3` in each fold. Then run two fits at 55/67; report learned unigram/bigram vocab sizes and fit-only coverage before metrics. Advance under shared gate.
- **Cost / risk:** Sparse matrix width remains 20,000 before structured features; two vectorizer builds add work. Fit-specific vocabulary remains mandatory. Runtime **unmeasured**.

### CL07 — Lower minimum document frequency for sparse LightGBM

- **Question:** Does current `min_df=3` discard useful rare token identities?
- **Exact recipe:** Current word n-grams 1–2 with `min_df=2`, `max_df=.995`, `max_features=30,000`, `sublinear_tf=True`, all other LGB parameters unchanged; fixed 25% C2_B blend, missing route, threshold .50.
- **Why:** The token scan shows high vocabulary cardinality relative to the cap, but says nothing about label relevance. Rare terms may contain useful patterns; they may equally be noise. Earlier unigram-only LR failures do not answer whether rare 1–2-grams help LightGBM.
- **Pilot / gate:** Two fits at 55/67; report vocabulary size, percentage of features supported by only 2–5 fit documents, fit time, and peak memory. Advance only under shared gate and no material temporal instability.
- **Cost / risk:** Two fits; memory/runtime **unknown**. This is a single fixed sparse-capacity variant, not permission for a min_df/max_features grid.

### CL08 — Explicit token × entry_friction interaction (linear)

- **Question:** Do lexical effects vary across the two observed entry_friction levels?
- **Exact recipe:** C2_B-style structured Logistic Regression. Retain the original main-effect word TF-IDF block and add one sparse interaction block by multiplying each word column by an indicator for `entry_friction=1`; `entry_friction=0` is the reference. Reuse the Round 1 LR parameters (`C=1`, `liblinear`, max_iter=2000), keep threshold .50 and missing route fixed. Treat `entry_friction` as its observed discrete feature; do not create quartiles. This reference coding avoids an exact linear dependency between the main word block and two exhaustive level-specific blocks.
- **Why:** Earlier global text-weight changes failed, but they rescaled one block uniformly. This tests a different hypothesis—level-specific lexical coefficients. `entry_friction` has only two occupied levels, so the interaction is bounded rather than a large numeric grid.
- **Pilot / gate:** Before fitting, verify both raw levels have at least 200 present-stream fit rows at cutoffs 55 and 67 and that the same raw levels exist at both cutoffs. If not, retire before fitting. Then run two fits at 55/67; inspect convergence. Same shared gate versus sparse25; if main effects/interactions fail convergence, retire without increasing iterations post hoc.
- **Cost / risk:** Two sparse linear fits after the feature-support precheck, time **unknown**. Added text interactions raise variance and do not establish that entry_friction causes lexical effects.

### CL09 — Add a fixed trigram channel

- **Question:** Do local three-token sequences add stable signal beyond current unigram/bigram features?
- **Exact recipe:** Preserve the current fit-only word 1–2 TF-IDF block (20,000 cap) and add a separate word trigram-only TF-IDF block (`ngram_range=(3,3)`, `min_df=3`, `max_df=.995`, max_features=10,000, sublinear TF). Fit the same sparse LightGBM parameters, fixed 25% C2_B blend, missing route, threshold .50.
- **Why:** The current model has only 1–2-grams. This directly tests a previously absent context scale without letting a shared cap silently choose the budget. Anonymous/hash-like tokens may make trigram co-occurrences sparse, so this is below CL06.
- **Pilot / gate:** Before fitting, compute cutoff-specific unlabeled trigram support and require at least 100 distinct trigrams meeting `min_df=3` at both cutoffs; otherwise retire before fitting. Then two fits at 55/67; report trigram support and vocabulary growth; same shared gate. Do not continue if the trigram block is mostly low-support or unstable across the two cutoffs.
- **Cost / risk:** Added 10,000 max features; memory and time **unknown**. No semantic interpretation of a token sequence is allowed.

### CL10 — Compact SVD text plus structured HGB

- **Question:** Can a compact, smoothed text projection support nonlinear interactions with structured inputs that raw sparse scoring did not capture?
- **Exact recipe:** Fit current word TF-IDF inside each fold, project to exactly 32 components with `TruncatedSVD(random_state=42)`, append to unchanged fit-only structured numeric/categorical preprocessing, train `HistGradientBoostingClassifier` with Round 1 C1_B parameters (loss `log_loss`, learning_rate .05, max_iter 200, max_leaf_nodes 15, min_samples_leaf 30, L2 1, no early stopping). Route present rows to this model at fixed 25% probability blend with C2_B; missing branch unchanged, threshold .50.
- **Why:** Current sparse LGB is strong but C2_B residuals vary with confidence; a low-dimensional text representation plus structured nonlinearity is a representation hypothesis, not a threshold or weight sweep. No classification SVD result was found in the inspected V2/V3 reports.
- **Pilot / gate:** Before SVD, require at least 32 learned text columns and more than 32 fit documents at both cutoffs; fit all transforms only on each fold. Record the number of nonconstant SVD components. If fewer than 16 components vary in the fit matrix, retire the lane as a degenerate projection before fitting HGB. Then apply the shared gate.
- **Cost / risk:** SVD construction may dominate runtime; **no matching classification timing measured**. The dimensionality is fixed at 32; no component grid.

### CL11 — Random structured partitions plus sparse text LR

- **Question:** Can fixed random partitions of numeric/count metadata supply nonlinear structured interactions while a linear head retains word-level text effects?
- **Exact recipe:** On the fit-only scaled numeric/count structured block, fit `RandomTreesEmbedding(n_estimators=64, max_depth=5, min_samples_leaf=30, random_state=42, n_jobs=2)`; concatenate its sparse leaf indicators with the current word TF-IDF and fit LR `C=1`, `liblinear`, max_iter=2000. Keep categorical one-hot features and missing indicator. Fixed threshold .50; current missing route.
- **Why:** The residual Head A already tested a learned structured residual head and lost; this uses a different fixed random feature map. It is lower confidence because current sparse LGB already models structured and text together.
- **Pilot / gate:** Require at least two nonconstant structured input columns and both target classes at each cutoff. Then two fits at 55/67 only after dry-run validates sparse/dense compatibility, row alignment, and fit-only transforms. Report nonconstant tree-leaf columns before LR; if no leaf column varies, retire as degenerate. Same shared gate.
- **Cost / risk:** New estimator integration and memory are **unmeasured**. Do not add tree count/depth variants if this fixed recipe fails.

### CL12 — Two-score temporal OOF meta-head (conditional)

- **Question:** Do C2_B and sparse-LGB probability scores have complementary conditional information that can be learned from strictly past predictions?
- **Exact recipe:** At cutoffs with sufficient saved OOF predictions, fit one `LogisticRegression(C=1, solver=liblinear, max_iter=2000)` on two inputs only: `logit(clip(p, 1e-6, 1-1e-6))` for C2_B and sparse-LGB. Training scores must be from base models fitted before their entire validation cycles. No structured-CB score or threshold selection; the only score adjustment is the single predeclared logistic head (no separate blend-weight search). Keep early windows on frozen sparse25 fallback until enough past OOF blocks exist; apply the meta-head only where the fixed OOF-support rule is met, and retain the CB missing branch.
- **Why:** Sparse25 disagrees with residual Head A on 892 primary rows; sparse25 has more sparse-only correct than Head-A-only correct (521 versus 371), so a meta-head is uncertain. This is materially different from the failed V2 C2_B/NB or weak-CB score heads: it uses the new sparse candidate score. It is also not a second attempt at residual Head A.
- **Pilot / gate:** First verify that existing cached base scores form complete disjoint past-only blocks for cutoffs 67 and 75, that both score columns vary, and that both classes meet the fixed OOF support rule. If not, stop before fitting and report the gap. Pilot only those two meta-head fits. For any full-six follow-up, fit a head only when at least 5,000 present-stream OOF rows and 1,000 examples of each class are available; otherwise retain the exact sparse25 fallback. Freeze this support rule before scoring. The current latest sparse OOF cache is missing a fit51 base score for cycles 52–55, so an outer-55 meta-fit would require an additional new sparse base fit; do not silently count it as available. Same shared gate with fallback locations fixed in advance.
- **Cost / risk:** Meta-fits themselves are cheap; reconstructing valid OOF base predictions may require extra fits. Exact support and time are **unknown** until cache audit. Prior score calibration/stacking failures and H12 reversals make this low priority.

## Not queued / known retirements

- **Case-preserving analyzer:** no uppercase characters in any of 47,204 non-null Train documents and case-sensitive/default unigram counts match in `research_backlog/feasibility.json`; no observed case variant supports another fit. Accent collisions were not audited by that receipt, so accent-preservation remains unknown; only reopen after a read-only accent-collision scan finds actual collisions.
- **Repeat old residual Head A, native-text CatBoost, plain unigram-only LR, NBSVM weighting, SVC, word+char LR, global text-weight scaling, TTR feature, threshold sweep, temporal sample weighting, or anonymous token target rates:** those exact families already failed or were mixed in the cited V2/V3 reports. Do not include as overnight retries.
- **Model parameters/public-score optimization:** no LightGBM leaf/learning-rate/class-weight grids, public leaderboard tuning, Test-based vocabulary or threshold selection, or more Kaggle submissions.
- **Standalone post-blend temperature scaling:** a monotone temperature applied to the final blended probability cannot change labels at a fixed 0.50 threshold, so it has no Macro F1 experiment value. Per-component calibration before blending would be a different, OOF-dependent experiment, but the existing V2 calibration failures and missing early sparse OOF block make it a low-priority lane; it is not queued here.

## Proposed serial overnight order

1. CL00 cached route composition (if the cache verifier confirms exact alignment).
2. CL01 present-only sparse-fit ablation.
3. CL02 explicit unknown-category flags.
4. CL03 ordered structured-category rates, only after the temporal proof and synthetic test exist.
5. CL06 split unigram/bigram budget.
6. CL04 word+char sparse LGB; CL05 only conditionally.
7. CL08 token × entry_friction interaction.
8. CL07 rare-term vocabulary.
9. CL10 SVD32 + structured HGB.
10. CL09 trigrams.
11. CL11 random structured partitions.
12. CL12 two-score meta-head only if existing OOF cache coverage is verified.

Run the frozen queue serially, one candidate at a time. The queued plan can proceed without a separate human review after every successful card, but each step must obey its predeclared gate and write a verified receipt before the next one starts. Any integrity failure, missing input, non-frozen variant, failed precheck, or ambiguous partial output stops the queue. A failed pilot retires that exact family; it does not authorize adjacent parameter variants. The run order deliberately places the most evidence-backed and lower-cost checks first, and keeps the highest-variance/high-implementation-risk candidates late.

## Time and resource notes

- The current word sparse-LGB continuation took **9.80–25.21 s per fit** as training rows grew; source is summarized in `reports/optimization_v3/research_backlog/execution_protocol_draft.md`. The protocol’s 100.1 s fresh raw-only replay includes three full-Train components and is not a timing estimate for a new model family.
- CL00 needs no fits. For each simple two-cutoff lane plan for two fits plus setup, verification, plots/report, and possible safe recovery. If advanced, add four sparse-style fits; do not run lanes in parallel on this workstation.
- CL04/05, CL07, CL09, CL10, CL11, and any new OOF reconstruction have **unknown runtime/peak memory**. Measure the first fit, recheck free RAM, and stop before the next model if memory or source lineage is unclear. No GPU was verified and PyTorch was absent in the feasibility receipt.

## Main recommendation

The best next Classification sequence is **CL00 → CL01 → CL02 → CL03 → CL06**. It begins with a cache-only route check, then isolates fit population, then targets a documented unknown-category failure mode with simple indicators before the more leakage-sensitive ordered encoding, and tests vocabulary allocation motivated by the label-blind token scan. Character sparse LGB is a reasonable second tier because its representation differs from the failed character LR, but it has less direct support and much higher memory uncertainty. The residual-score head and random/SVD representations should remain late-stage options, not the default overnight workload.
