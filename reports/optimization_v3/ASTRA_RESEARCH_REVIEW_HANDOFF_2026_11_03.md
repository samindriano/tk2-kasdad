# Research stop checkpoint for independent Astra review

**User instruction:** stop after the present campaign and conclude before more
research. Luna has received explicit STOP. No failed-FM checkpoint scoring,
outer-label access, optimizer resume or successor experiment was authorized.
This document records the handoff, not an experiment or budget amendment.

Verified incoming HEAD:d94edbbaecd38a3fada326136d6654c34a2b9ff0,main,clean.
Local commits have not been pushed during this campaign. This handoff itself
will be a subsequent local documentation commit. Do not reset/rebase/merge/push.

## Working conclusion from Sol, to challenge rather than assume

Keep A02 Classification and R06 Regression as the best PUBLIC-observed reference
models supplied by Sam, not independently established private winners. Recent
exploration does not yet justify replacing either. Stop adding configurations
until an independent review distinguishes exhausted mechanisms from conservative
gates, repeated-data selection and technical optimization failures.

| Candidate | Public metric reported by Sam | Meaning |
|---|---:|---|
|Classification A02|Macro F1 .82930|Best Classification score in user-provided receipts|
|Classification sparse25|Macro F1 .82908|Previous observed reference|
|Regression R06 original|MAE11.16620|Best Regression score in user-provided receipts|
|Regression R-NBR E79|MAE11.19115|Worse than R06 by .02495|

These are user-pasted Kaggle receipts, not fresh API checks. Public sample labels,
private scores and server artifact hashes are unavailable. Do not infer a
confidence interval or a performance ceiling, or promise .83. Other prepared
artifacts such as B01 must not be silently called submitted or scored; verify
their separate manifests/receipts if they matter to the decision.

## Recent research: what actually happened

| Lane | Observed result | Disposition |
|---|---|---|
|Original Regression R-NBR|Full-primary MAE gain .220627 vs R06; late .148274; H12 harm .048271. Public11.19115 vs R0611.16620.|Internal incremental gates passed, diagnostic/inherited risks retained; one reviewed manual benchmark; public improvement not observed.|
|Exact-impact residual retrieval|W4 harm .139436,H12 harm .056857; W6 gain .191049. Pooled not better than original R-NBR by required margin.|Retired; no submission.|
|Direct neighbor target median|W4/W6/H12 all worse than R06; pooled low/high-tail harms .80766/2.31184.|Retired; no submission.|
|Joint bag+position LR|E67 H12 F1 gain .000889 vs bag,.005978 vs shuffled; E75W6 -.000246 vs bag.|Failed fixed .003 H12/bag and nonnegative E75 gates; retired exact recipe.|
|Frozen-bag positional offset|E67 H12 F1 gain .001151 vs bag,.005650 vs shuffled; E75W6 +.001059 vs bag.|One .003 H12/bag gate failed; other gates passed; no further positional fits.|
|Learned rank16 FM|E67 FM optimizer reached1000 iterations,nfev1044. SUM loss20606.106149->6677.625305; gradient infinity1.345563 SUM,.00004530 mean.|FAIL_OPTIMIZER_INCONCLUSIVE. NO outer predictions/labels/metrics. Other3fits released.|

Original R-NBR validation uses5 learned origins plus W1 fallback; H12 is one
fixed-origin forecast, not a composite overwritten by later fits. Its corrected
evaluation preserved old bug evidence. Original R06 legacy W1 missing-stream
harm .395556,n392 remains an inherited failed guard, not cleared by improvements
relative to R06. New stricter H12 gates did not change old gate verdicts.

Recent findings are NOT proof that all positional/pair/neighbor information is
useless. Real position beat shuffled capacity; freezing bag made the small net
gain positive in both evaluated histories. But drop-block scoring was not a fair
refit ablation; extra capacity, allocation, optimization and history differ.
Direct target median beat shuffle but lost to the strong baseline and hurt tails.
Information versus a null control is not necessarily incremental predictive value.

R-NBR harm did not monotonically grow with horizon: E67 gains for1-4/5-8/9-12
cycles were -.112556/-.041044/-.017145. Horizon decay is not established.

## Fit accounting and final technical state

Authoritative ledger:experiments/runs/optimization_v3_research_next_v1/budget.jsonl.
Root independently checked SHA256
125446878a69bdecf2d658daf259622e568147642a072d2aaf2d8e21a8f47008.
281 events,cap113,active reservations110,starts110,saved106,failed4,pending0.
This is that research-phase ledger, NOT every project fit or distinct estimator.
Some units explicitly count preprocessing and target-memory construction.
Three released unstarted slots are not permission to restart work.

FM terminal parameter NPZ is complete and its SHA was root-verified:
718cf4b4c6f47b12bb3262333c47027080d2810d83498fdacb2e7fe189b6ac82.
344846 parameters,29705 saved fit logits; independent prefix-only loss/gradient
verification passed. Saved nonconverged state is not a converged predictor.
Rescaling its gradient after the fact cannot retroactively make SciPy PASS.
Sol changed proposed mean-objective/200-iteration plan BEFORE fitting to an
equivalent SUM objective with1000/maxfun2000 limits. That choice tightened the
absolute convergence criterion and must be considered in an adversarial review.
No retry/resume or retrospective gate relaxation is authorized.

## Reliability and study-design concerns for Astra

- Repeated adaptive use of the same temporal development outcomes, overlapping
  primary/H12 and histories, plus public feedback. More fits do not create an
  independent validation set. Recent pilot gates differ by mechanism and are
  risk policies, not Kaggle scoring requirements. Question whether .003 F1 was
  too conservative, but do not rewrite original FAIL to PASS.
- Multiple implementation bugs were found before/after fits: shared H12/primary
  map overwrite, raw/routed probability confusion, stale pre-fit ledger guards,
  wrong recovery metadata nesting/hash types, unsorted cycle assumptions and
  NumPy string-ID equality. Corrections retained immutable model outputs; source
  lineages matter more than green summaries. Some checks were independently
  recomputed by Sol, not every source line or all-pairs geometry every turn.
- Whole A02 pickle extraction materialized Classification-derived arrays despite
  an intended vectorizer-only route. Strict never-read attestations were marked
  invalid; two standalone target-free Regression-text TF-IDF replays matched
  historical prefix/document/vocabulary/IDF hashes and avoided that route later.
  No evidence established Class target use in Regression predictions; do not
  equate materialization with demonstrated predictive leakage.
- Synthetic positional harness accidentally appended one live-ledger event and
  then removed it. Both violated append-only discipline; restored hashes do NOT
  establish uninterrupted history. Incident is explicitly recorded. Writer paths
  are now required,synthetic root-bound,production writer guarded,finally hash/
  restoration and failure injection tested. Preserve this evidence in any review.

## Directed reading; expand only when decision-relevant

All paths below relative to repository root. Start here and the final source
checkpoint; do not read all raw CSVs,huge JSONs or logs into context.

Recent authoritative results:
1. reports/optimization_v3/rnbr_continuation_v1/RESULTS.md
2. reports/optimization_v3/rnbr_postpublic_v1/NEXT_DECISION.md
3. reports/optimization_v3/rnbr_impact_match_v2_2/RESULTS.md
4. reports/optimization_v3/rnbr_target_median_v1/RESULTS.md
5. reports/optimization_v3/positional_text_pilot_v1/RESULTS.md
6. reports/optimization_v3/positional_text_diagnostic_v1/RESULTS.md
7. reports/optimization_v3/positional_bag_offset_v1/RESULTS.md
8. reports/optimization_v3/learned_pair_pilot_v1/FAILURE_DIAGNOSTIC.md
9. reports/optimization_v3/learned_pair_preflight_v1/REPORT.md
10. reports/optimization_v3/positional_text_pilot_v1/INTEGRITY_INCIDENT.md

Historical orientation as needed:
- reports/optimization_v3/postscreen_review/REVIEW.md
- reports/optimization_v3/research_next_v1/CAMPAIGN_SUMMARY.md
- reports/optimization_v3/classification_next_v1/FINAL_SUMMARY.md
- reports/optimization_v3/a02_next_decision_v1/DECISION.md
- reports/optimization_v3/b01_benchmark_review_v1/SOL_REVIEW.md
- reports/optimization_v3/research_next_v1/RESEARCH_LANES.md

Read a small number of exact source/tests/receipts to challenge important claims;
stale PLANNED headers are not actual execution status. Source-code hashes and
frozen fit commits differ from later report/evaluator commits; do not conflate.

## Ready-to-use Astra prompt

You are an independent adversarial reviewer for TP2 KASDAD at
C:\Users\Sam\OneDrive\Documents\KASDAD\TP 2. Read this handoff first and verify
current HEAD/status. The user has STOPPED experiments for a strategic conclusion.
This is a read-only review: no training,optimizer resume,new scoring/label access,
installation,Test/Kaggle,subagents,messages to other chats,Git/file mutations or
external source upload. Use existing reports/evidence and targeted source checks.

Do not accept Sol's narrative or gate choices by default. Determine:
1. Which mechanisms genuinely failed to add value,which were merely below a
   risk threshold,and which remain UNKNOWN due to optimization or weak controls?
2. Does our temporal validation/horizon/denominator match Train->Test deployment?
   How much adaptive/public selection has accumulated,without invented CIs?
3. Are strict gates,comparator implementations or optimization choices making us
   miss useful candidates? Conversely,are positive null-control contrasts being
   overinterpreted? State the strongest alternative explanation to Sol.
4. What should the NEXT PHASE be: finalize with existing champions, one narrow
   verification/benchmark milestone, or ONE new mechanism? No algorithm list.

Deliver1500-2200words maximum: verdict,ranked at-most5 findings with evidence
paths,counterevidence/UNKNOWNs; a stop/continue decision for each active mechanism;
exactly ONE recommended next action with scope,cost,controls,boundaries and stop
rule. If more work is recommended,check prior novelty and say what would falsify
it. Separate competition strategy from academic/demo readiness. No .83 promise,
no retroactive PASS,no automatic continuation. One review,thenSTOP for Sam.
