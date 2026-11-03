# Sol: retire recipe; zero-fit decomposition before another learner

Sol reviewed the positional construction (shared unigram vocabulary/IDF,
sublinear counts and combined positional normalization) and independently
recomputed routed F1 from saved probabilities joined by official Train IDs:

- E67 H12:bag0.805106410,position0.805995828,shuffle0.800017797.
- E75 W6:bag0.810383725,position0.810137508,shuffle0.808658306.

Missing probabilities were exactly equal across arms. Root reran the4 unittest
contracts, allPASS. **Retire this exact three-third recipe**, preserve failed
gates and all historical ledger-incident evidence. No continuation or upload.

It is incorrect to conclude that position contains no information: position
beats shuffled capacity, but barely improves (or loses to) the simpler bag
baseline. Conversely that comparison does not prove robust incremental value.
Same W6 outcomes at two origins show history sensitivity, not independent tests
or a causal history effect: both vocabulary and fitted coefficients change.

## Authorized next scope: positional decomposition, ZERO FITS

New namespace positional_text_diagnostic_v1. Keep cap105 and ledger unchanged.
No model/preprocessor fits, tuning or deployment policy. No Test/Sample/Kaggle,
installs, notebooks or push. This is finite research on the completed pilot:

1. Load saved Classification estimators/preprocessors/matrix artifacts. Rebuild
   score matrices using transform only if necessary and require equality to
   saved matrix/prediction hashes. No fit on import, reconstructed vocabulary,
   new statistics, hidden solver or callback that trains. Reading this task's
   Classification model/label state is allowed; Regression labels are not.
2. Decompose augmented logistic score exactly as intercept + base-feature
   contribution + positional contribution, for both real and shuffled arms.
   Validate sigmoid(sum)=saved raw LR probabilities. Also reconstruct the bag
   model score with its own coefficients. Respect original feature dimensions,
   class order and availability routing.
3. Report a prespecified diagnostic ablation with positional contribution set
   to zero but augmented model's base coefficients/intercept retained. This is
   a counterfactual score diagnostic, not a newly trained model or deployable
   candidate. Compare full augmented, dropped-block and original bag metrics,
   logit contribution summaries and useful/harmful threshold flips on W4/W5/W6,
   E67 H12 and E75 W6. Same missing expert exact. No chosen scaling or threshold.
   This separates adding the channel at scoring from changes in fitted base
   coefficients; it is not a causal attribution or a fair refit ablation.
4. On shared W6 rows, compare the two origins' changes explicitly. Describe
   source-prefix vocabulary coverage/OOV and document length using fixed bins
   <=20,21-40,41-80,>80 tokens. These are descriptive, not subgroup routes.
   No claim that a subgroup will generalize just because it looks favorable.
5. At most one fixed matched-positive-count comparison of position versus bag
   per(origin,cycle),present rows only: use bag's>=0.5 positive count for both
   rankings, tie by source row. Selector gets no outcomes. Compare F1/confusion
   and TP changes as a ranking/allocation diagnostic; generate no Test policy.
   Overlapping cohorts remain explicitly dependent/adaptive.

Write one concise findings report + computed tables, max2 informative PNGs,
source/artifact hashes and reproducibility command. Synthetic tests for logit
partition/equality, empty position block, class order, fixed-K ties/allzero/allone
and exact fallback. Assert live ledger hash before/after and explicit temp-only
test paths. Preserve raw model/token contents outside Git; report aggregates.

Final judgment must choose ONE: NO_GO_MORE_POSITIONAL_FITS or ONE_SPECIFIC
FOLLOWUP_FOR_SOL_REVIEW with an evidence-supported mechanism and exact minimal
cost/control/gates. Do not offer a list of C/bin/vocab/seed/blend variations or
perform them. If decomposition cannot be verified from available saved state,
report the precise missing artifact rather than refit. Routine research choices
authorized, real blockers to Sol. Local scoped commit and callback Sol with
evidence/counterevidence/QA/ledger/commit BEFORE executor final reply, thenSTOP.
