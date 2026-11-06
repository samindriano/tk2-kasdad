# CLS continuation: position-aware A02 donor kernel

## Decision and authorization

Sam explicitly requested continued CLS research after the Bag2 benchmark; closing one recipe does not close the task. Authorize one new bounded pilot, `CLS_A02_POSITION_KERNEL_V1`, subject to input compatibility and novelty checks below. Luna1 executes; Sol independently reviews results and any eventual E79/upload. REG remains with Luna2. Do not modify historical source, models, protocols, counters, EDA or teammates' work.

Champion reference: full uniform-POS/A02 present blend with operational missing threshold, public 0.83225, submission 58001282. Bag2 scored 0.83024 (58047602) and does not replace it. No Bag2 seed/weight/threshold rescue.

## Hypothesis and counterevidence

A02 retrieves historical residual donors using bag TF-IDF cosine. The successful positional predictor is a separate branch. Test whether conditioning donor similarity on relative token position supplies useful residual information beyond the current complete champion. This differs from CGEOM's diagonal weighting of bag columns using absolute bag-LR coefficients, and from simply blending POS/A02 predictions. Prior CGEOM history reversal and the already strong POS branch are adverse evidence: retrieval may add noise or duplicate existing information.

Read `reports/optimization_v3/a02_coefficient_geometry_v1/{SOL_EXECUTION_HANDOFF.md,RESULTS.md}`, `src/models/classification_a02_neighbors.py`, `src/models/a02_coefficient_geometry_v1.py`, `src/models/optimization_v3_positional_text_pilot_v1.py::build_position_block`, and the current Bag2 pilot's original-probability/source bindings. Check for an equivalent historical positional donor kernel before official construction. If one exists, return its exact path and verdict to Sol; do not silently repeat it or choose another recipe.

## Fixed recipe

Reuse each original A02 prefix-fitted vectorizer, full historical donor bank, residuals `y_bank - p_bank_OOF`, query Sparse25 baseline, and original lexical top32 similarities. No base-model, vectorizer, IDF, calibration, or POS-head refits.

Build original third-position blocks with the existing `build_position_block` using the same prefix vectorizer. This deterministic transformation uses existing unigram vocabulary and IDF, sublinear counts and row L2 normalization. REAL uses original order. SHUFFLED uses the existing deterministic within-document order shuffle, seed42, independently for each document; preserve token multisets and segment sizes. Original lexical bag/bigram matrices remain unchanged in both arms.

For query q and donor d, define `K(q,d) = cosine_bag(q,d) * (0.5 + 0.5*cosine_position(q,d))`. Search top32 positive K over the FULL original bank, with stable source-row-position tie breaks. Missing/zero lexical support retains original fallback. The 0.5 floor preserves positive lexical support even when positional blocks are zero. No coefficient/mixture grid.

Let `M0(q)` equal the original A02 sum of squared top32 lexical cosines. For newly selected donors use `u=K**2`, `w=M0*u/sum(u)`, and `delta=clip(sum(w*residual)/(4+M0),-0.05,0.05)`. New A02 probability is `clip(p_Sparse25+delta,0,1)`. M0=0 means exact baseline fallback; M0>0 with no positive selected support is an integrity failure.

Primary candidate replaces ONLY pA in the current champion: present `0.5*pPOS_original + 0.5*pA_new`, threshold >=0.50; missing retains current pA and threshold >=0.53230976129477 exactly. SHUFFLED is a diagnostic control, never an automatic finalist. Preserve original full champion replay and all missing decisions before evaluation. Disclose that the operational missing threshold was previously learned from global OOF: this is adaptive paired development, not fully past-only calibration.

## Boundaries, budget and prospective decisions

Four new residual-memory predictor constructions (not zero fits): E67/E75 x REAL/SHUFFLED. No optimizer or learned preprocessing fits. Use a fresh lane-local cap4/ledger; preserve historical counters rather than resetting or retroactively spending their caps. E67 fits through67 and queries68-79; E75 fits through75 and queries76-79. All donor cycles <= origin < query cycles. Official row order is not chronological; do not use physical row comparisons as a time gate.

Unique primary: E67 cycles68-75 plus E75 cycles76-79, 20,903 distinct IDs. Report E67 H12 and W4/W5/W6, E75 W6, cycle signs, confusion counts, Macro F1, AUC/AP/logloss/Brier, neighbor overlap, correction-cap rate, mass identity, and support. Do not count overlapping E67/E75 W6 twice. Query labels may be read for scoring only after all four states/prediction bundles pass durable independent prelabel reconstruction. Bank labels already belonging to the historical fitted memory are allowed; query-label cache members must remain unopened until that boundary.

Scientific qualification: REAL unique-pool gain >=0.0005 vs full champion; E67 H12 and E75 W6 gain >=0; REAL vs SHUFFLED unique-pool gain >=0.0002 and E75 gain >=0; existing supported guards pass. Separately, exploratory eligibility follows the already stated small-gain policy: unique-pool gain >0 OR (E75 W6 gain >=0.0005 AND pool gain >=-0.0005), plus supported subgroup F1 harm <=0.02 and positive-recall harm <=0.03, missing invariant and independent QA PASS. Use n>=200 and >=30/class for supported groups; reuse fit-defined text-load quartiles and entry-friction levels. Unsupported is UNKNOWN. Scientific FAIL stays FAIL even if exploratory-eligible. No public-score-based selection or control promotion.

## Execution and return contract

Luna1 may implement, test the actual CLI feature/artifact/receipt/readback/prediction/scoring lifecycle synthetically, freeze exact input/source hashes and recipe, reserve and execute all four units, independently reconstruct predictions, score once and verify all gates without another cosmetic checkpoint. Source-ready review is needed only for a genuine incompatibility or a change to this recipe/budget. All invariants must survive Python -O; preserve partial states and completed units, never refit a consumed unit. Keep producer/consumer array digest algorithms explicit and test their compatibility.

Full-bank search must be blocked/sparse, not one full dense query-by-bank allocation. Use two threads, the shared advisory heavy-job lock, initial free RAM1GiB and actual reserve512MiB; bounded target-free timing/size preflight may set query block size without changing the model. If REG holds the lock, wait for ownership release without consuming a fit. Do not delete lock files or require an idle chat. No schedules.

Keep source and concise results in new lane-only paths; ignore raw predictions/model binaries. Scoped local commits only, no push/merge. No Test/Sample/Kaggle or E79 construction yet. If eligible, prepare E79 source/inventory without executing it, then send results plus exact candidate source/budget to Sol immediately. If scientifically negative, send the evidence to Sol for the next research decision rather than declaring all CLS research exhausted. Questions go to Sol, not Sam. Expected implementation/pilot time is roughly 30-90 minutes; full-bank runtime is uncertain until the target-free preflight.
