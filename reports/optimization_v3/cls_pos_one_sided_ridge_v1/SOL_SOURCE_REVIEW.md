# Sol source review — one-sided positional ridge

ROOT_SOURCE_PASS for the four protocol-defined fits only. Source freeze SHA256 6b7941fb753ec745953a6906097fd48c31daee7ce3abbd2ba87cff0ea67b6843; identity preflight SHA256 fd87e44da5b04fd92d50b93838d060ac54fb477b7aec1cf9430ec59644827149.

Root independently ran all five pytest tests successfully. Coverage includes exact penalty construction, finite-difference gradients, uniform equivalence/missing invariance, protocol digest correction, and synthetic optimizer/transaction/evaluation/independent metric orchestration with interruption recovery. Root also independently replayed the target-free preflight (14 checks) and source-freeze verification under shared lock. All 17 source files and 79 input files match. E75 digest correction is metadata only, preserves the original protocol and rejects the stale pin.

The selective arm is max(1, prior aligned weight), without normalization. The scalar control is the prefix-only coordinate mean of those selective weights; matching mean diagonal penalty does not match effective model complexity. The frozen optimizer/objective and fixed bag logits are reused. Actual runner uses four fixed fit IDs, durable pre-target starts and no consumed-fit retries, with exact cache identity checks before target access.

Execute E67 ONE_SIDED/GLOBAL_STRENGTH then E75 ONE_SIDED/GLOBAL_STRENGTH serially under current RAM gates/shared lock. After all saved fits, prepare and independently verify target-free predictions; commit their immutable manifest before opening query labels for frozen evaluation. Run the independent metric/row/subgroup verifier and reconcile ledger. Mechanism and competition gates remain separate. Query windows overlap; unique pooling and reused-development-label disclosure are mandatory.

Return complete results and counterevidence to Sol before stopping. This pass authorizes no E79 refit, Test/SampleSubmission access, candidate CSV or Kaggle submission. No push, threshold search, extra arms or retry/rescue.
