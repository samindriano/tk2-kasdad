# Full REG fixed hinge + center-local benchmark

Client date: 6 November 2026 (Asia/Jakarta).

Root reviewed source commit `fed08ead448967fe7132aed7e5ef5cec1a0287e5`, reran four full-application tests and independent source verification (PASS), and verified AST parity of the prediction formula and CSV helpers with R3. R4 repaired the hash-field lookup; R3 failure and receipts remain preserved. The one completed scalar fit was reused, with delta -1.3955577433171884; zero new model, preprocessing or calibrator fits occurred during application.

Root application pass SHA256: `0cf7c4263f8cf3fbc0a2a4dec1b41b830301880b2ef208a83f126984798953de`. Fresh available RAM was 1,939,922,944 bytes and the shared OS lock was held during application, then released. Actual application and independent verifier passed.

CSV: `submissions/reg_center_local_application_v1/fullreg_center_local_r4.csv`, SHA256 `1519e0e8e2dbc17b4959d968fba04d10a861d28234a126bbf92f114e52d246f3`. Root separately replayed the formula from cached prediction-only E79 arrays, checked 29,421 unique ordered IDs, finite [0,100] values, 1,062 bit-identical missing fallback values, and exact output float bits. There are 6,028 changed predictions versus the full hinge champion. CSV/model data remain outside Git.

Exactly one authorized exploratory submission was made via authenticated Kaggle CLI to the Regression competition. Submission ref **58032751**, message **Samuel Full REG Fixed Hinge Center Local Exploratory R4**, is independently confirmed **COMPLETE**, public MAE **11.04367**. This is **0.02857 worse** than the prior full champion **11.01510** (ref58019539). Initial PENDING receipt and subsequent COMPLETE check are both preserved. No final-selection changes or retry upload occurred.

Decision: retain the existing full champion. The earlier diagnostic mechanism/exploratory checks passed but candidate qualification failed (pooled local gain0.00615004 below0.01). This benchmark does not change those historical verdicts or establish private performance. A positive but small development effect did not transfer to public here; adaptation, temporal variation and sampling remain possible explanations, not proven causes.

Next work delegated to Luna2 is bounded read-only novelty review of at most six summaries/policies for exactly one genuinely untested mechanism, or an honest NO_NEW_MECHANISM. No new fit, width/delta sweep, Test-label access or submission is authorized by that review. CLS full-champion regularization proceeds independently with Luna1. No push or Kaggle final-slot selection was performed.
