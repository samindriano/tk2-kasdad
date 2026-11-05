# REG reopened: fixed positional row-bootstrap pilot

Sam explicitly requested further REG work after the earlier park decision. Root reopens only this bounded exploratory pilot; previous calibration and architecture-audit verdicts remain historical. The current full champion public MAE is11.01510; the latest triangle benchmark11.04367 did not replace it.

## Hypothesis and fixed candidate

Test estimator-variance reduction in the positional auxiliary of the full five-learner DIRECT+hinge model. This is a replication/variance hypothesis, not a new information source or proven benefit. The inspected positional recipe uses all rows/features (subsample1, frequency0, colsample1); changing only a seed would likely produce an identical model. Instead create two Poisson(1) row-weight replicates with RNG seeds17 and137, normalized to sum n_fit, keeping model seed42 and all original hyperparameters/fitted features. Average original and two replicated raw positional predictions equally, then apply the exact existing T50 mapping and full DIRECT policy. No seed selection or weight grid.

All four non-positional learners, existing R06, present routing, clipping placement and missing fallback are fixed. Pilot evaluation must use the same past-only hinge states as the earlier D1/D2 full-champion comparison, never the final all-OOF hinge on earlier labels. D1: E67 predictions scored on72–75 with hinge fitted68–71. D2: E75 predictions scored on76–79 with hinge fitted from E67 OOF68–75. These are reused development data, not independent/fresh holdout evidence.

## Bounds and gates fixed before new outcomes

Exactly four pilot predictor fits: two replicas at each origin67/75; zero preprocessing or calibrator refits. Conditional final stage may add exactly two full-Train replica fits after root review. Reuse original fitted feature states; prove faithful original-head/R06/full-DIRECT replay and exact IDs before pilot. If those states are unavailable or an exact duplicate is found in a bounded novelty check, return the evidence rather than fabricate comparability.

Scientific qualification: pooled D1/D2 MAE gain>=0.02, each window gain>=0, supported fixed tail/group harm<=0.25, missing bit-exact. Exploratory benchmark eligibility is separately declared: pooled gain>0 OR (D2gain>=0.01 AND pooledgain>=-0.01), D1harm<=0.05, same supported-group/tail guards and missing invariant, independent QA PASS. No retrospective changes to historical gates and no public-score-driven seed, mapping or width selection.

## Execution

Luna2 owns its existing isolated REG worktree. Source/weights/inputs/output paths and fit accounting must be frozen before outcomes; full synthetic CLI persistence/replay must exercise actual feature-result envelopes. Root source review precedes real pilot fits; later root decisions precede full-Train refit and Test/upload. Preserve all consumed fits and failed receipts, do not overwrite historical experiments. No new raw source, dependency installation or model-family search.

Sam's latest relaxed RAM policy applies: vector/CSV jobs start512MiB with256MiB reserve; feature/application work starts1GiB with512MiB reserve. Estimate actual LightGBM training separately from sparse shape/measured evidence, retaining512MiB reserve, two threads and one heavy-job lock. Do not inherit blanket2/4GiB floors or terminate apps. Luna1 continues CLS independently. The bounded task and explicit return-handoff requirement were sent directly to Luna2.
