# CLS one-sided positional ridge — bounded follow-up

Decision: PREPARE_SOURCE_ONLY. Sam authorized continued measured research and requested collecting several results before selecting a submission batch. Official fits require a new Sol source pass. No Test, E79, CSV or upload is authorized here.

## Evidence and uncertainty

Sol independently replayed the R1.2 mean-one energy diagnostic: 133 checks passed with maximum numerical difference zero, without target access or artifact mutation. The w<1 partition accounts for 95.6468% / 94.4310% of partition L1 logit movement at E67/E75. This is descriptive and does not establish why the prior candidate lost: coefficients were jointly fitted and frequent features naturally cover more rows. The previous mean-one ALIGNED candidate lost to both UNIFORM and PERMUTED; its verdict remains unchanged. This follow-up is post-result adaptive research, not independent confirmation.

## One fixed hypothesis and matched control

Hypothesis: avoiding reduced regularization of frequent tokens while retaining increased regularization of rare tokens improves the existing positional model. Distinguish selective penalties from a global increase in regularization.

For each origin, reuse the exact prefix-only, hash-verified ALIGNED token penalty map w from the previous pilot. Define ONE_SIDED weights v_j=max(1,w_j), repeated across the same three position blocks. Do NOT normalize v again. Define GLOBAL_STRENGTH weights c=mean(v_j) over token coordinates, repeated identically across all coordinates. This matches the mean diagonal penalty only; it does not match effective degrees of freedom or fitted penalty energy. Do not tune c, rescale by outcomes, or add a permutation arm.

Reuse identical frozen bag-logit offsets, positional features, training labels, objective SUM logloss plus 0.5*sum(weight_j*gamma_j^2), optimizer settings, feature order and missing routing. New namespace, exactly four possible predictor fits: E67/E75 x ONE_SIDED/GLOBAL_STRENGTH. Reusing prior DF maps is not a new learned transform; disclose that those maps originally required prefix DF fitting. Do not refit UNIFORM: verify and reuse saved uniform coefficients/predictions. No grid, seed sweep or automatic retry/rescue of a consumed fit.

## Boundaries and decision criteria frozen before fits

Same official Classification Train prefixes: E67 cycles47–67, query68–79 (20,903 rows); E75 cycles47–75, query76–79 (8,491 rows). Same-origin common rows for all controls. E67 H12 overlaps E75 W6, so report the overlap explicitly; unique pool is E67 W4+W5 plus E75 W6, n20,903. All these development labels have already been used by previous campaigns.

Component mechanism gate: ONE_SIDED Macro F1 must exceed BOTH saved UNIFORM and GLOBAL_STRENGTH at E67 H12 and E75 W6 by more than 1e-12. Failure does not retroactively alter the old experiment.

Competition policy: present p=0.5*p_POS+0.5*p_A, threshold>=0.50. Primary missing reference is p_A>=0.50; operational p_A>=0.53230976129477 is a separately labelled post-selected descriptive reference. Missing rows must be bit-identical within each reference. Historical A02 MIX50 used p_P on missing; do not conflate those vectors.

Competition gate versus saved uniform-POS/A02 same-policy reference: gain>=0.0003 on E67 H12, >=0 on E75 W6 and >=0.0003 on unique pool. Each E67 window harm<=0.003; supported subgroup F1 harm<=0.01 and positive recall harm<=0.02, all reported views. Support n>=200 and >=30 per class, otherwise UNKNOWN. Report full and present-only Macro F1, confusion/precision/recall, Brier/logloss, per-cycle signs and inherited availability/entry-friction/prefix-defined text-load groups. Include GLOBAL_STRENGTH results even if selective mechanism fails. Neither arm is automatically eligible for refit or submission; root will choose prospective next action from both results without rewriting gates.

## Source readiness, resources and ownership

Luna1 owns new source/scripts/tests, experiments and reports in cls_pos_one_sided_ridge_v1, except this root handoff. Preserve historical sources/freezes/receipts, peer notebooks and unrelated forensic files. Stage named files only, local coherent commits allowed, no push or merge.

Prepare complete fit/evaluation/independent-verification source before official fits, freeze exact inputs and source hashes. Synthetic end-to-end must exercise the actual orchestration, public loader schema, target-access boundary, optimizer, persistence, ledger and verifier; mocking external data is allowed but mocking away these internal boundaries is not. Include finite-difference gradients, all-ones equivalence, max(1,w) and scalar-control construction, unchanged missing rows, tampered hashes/IDs, interrupted-save resume without duplicate fits. Real identity-only integration preflight must call the same pre-target identity checks used in execution; do not load target members for inventory.

Use current USER_REVISED_20261105 resource policy, shared OS job lock, thread2/job1. Saved matrix replay minimum1.5GiB and reserve0.75GiB; small gamma fit minimum2GiB with estimated incremental1GiB and reserve1GiB. Record estimates separately from measured RSS/OS lifetime peak. No silent floor reduction, killing other apps, schedules or polling. Coordinate directly with Luna2 for lock access.

Return source commit, freeze/preflight hashes, exact commands, synthetic QA, accounting and remaining UNKNOWNs to Sol chat01a19569-16e0-7253-aa14-57a0a3160e03 before stopping. Root reviews before any official fit. Questions/blockers go to Sol, not Sam.
