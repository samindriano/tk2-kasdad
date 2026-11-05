# Sol handoff — R06 center-local transfer source preparation R2

This handoff records source preparation only. It does not authorize a root source pass, fit, or score-target access. Please review the frozen source/protocol and decide whether a diagnostic root pass is appropriate.

The candidate transfers the fixed `center_local` triangle from the complex route to R06 SIMPLE. Its primary comparator is the immutable same-cutoff R4 p50 hinge applied to R06 raw predictions. Failed native R2C states are not used. The original full-DIRECT R4 hinge replay remains a separate diagnostic, and the earlier complex `center_local` result remains historical context.

The frozen triangle is `b=max(0,1-|p-50|/10)` with delta in `[-2,0]`. LOCAL uses the exact lower weighted median over present fit rows with weight `b`; GLOBAL_CONTROL uses the lower empirical median over the same present fit rows. The D1/D2 windows, counts, new cap of five, four diagnostic slots, conditional final slot, zero base/preprocessing fits, metric groups, support rules, and three separate gates are specified in `POLICY_R2.md`.

The conditional final slot is reserved only. Its base/state mapping still requires explicit review and authorization. Source preparation does not create a root pass or reset a prior ledger. No official fit, score target, Test/E79, CSV, or Kaggle access is part of this handoff.

After the source-ready artifacts are generated, the completion message to this Sol thread should carry the exact source commit, prefit-freeze SHA-256, identity-preflight SHA-256, source-ready-verification SHA-256, explicit population counts, synthetic metric QA status, and any unresolved UNKNOWNs. The verifier receipt is the controlling independent check of the non-target identity population and immutable inputs.
