# CGEOM: bounded zero-fit verifier correction

Sol review: GO_CORRECT_EXPECTED_ID_DTYPE_AND_RESUME_FROZEN_EVALUATION.
Within Sam's authorization to complete this one lane. No additional fit,
preprocessing, nearest-neighbor construction or prediction modification.
Incoming source HEAD ddd64736155a64e70c5453c921aedcbd910397f8, clean.
All four CGEOM memory predictors saved. Prelabel has not passed; no query
evaluation labels or results exist. Existing STOP was for this review only.

Source finding: _verify_unit reconstruction expects query_record_ids from
source['score_ids'] (pandas-derived object array), then compares exact dtype
against the safe stored Unicode array. The stored object-array rejection is
correct and MUST remain. The expected identity representation is the defect.

Authorize a minimal postfit evaluator/verifier amendment, with its own source
commit/hash/receipt and focused tests, that converts only expected query IDs
to explicit lossless Unicode before exact shape/dtype/value/order comparison.
Use source-derived expected IDs, not stored IDs to manufacture expected values.
Test long/non-ASCII IDs and reject truncation, reordering, altered values,
wrong shapes, object-stored arrays and numeric-array tampering. Preserve numeric
dtype/tolerance checks, all source/bank/row/future-cycle checks, and full-bank
reconstruction and correction-mass verification for all four units.

Preserve frozen runner/verifier/core, recipe, FREEZE, model/prediction artifacts,
receipts and pre-fit QA/preflight. Do not rewrite or republish prediction arrays,
use allow_pickle=True, or rerun original fit commands. Keep original failure
diagnostic evidence. A small separate adapter is preferred over copying the
entire runner. Reuse pure frozen routines where safe and auditable.

Because a documentation/evaluation amendment changes current Git HEAD, the
adapter may explicitly distinguish immutable training source identity from
amendment evaluator identity: verify original files against frozen hashes and
original Git blobs, require original fit source to be an ancestor, bind the
amendment source separately. This limited exception replaces only strict
HEAD==fit-commit checks for evaluation. Do not disable hash checks, silently
rewrite frozen source identity, broadly swallow verification errors, or patch
global source/ledger guards without an explicit checked amendment boundary.
If an unchanged-file EOL issue exists, compare exact original Git blob under
its original recorded hash plus declared working-tree normalization evidence;
never claim a raw hash matches when it does not.

Initial live ledger SHA256
bcedd3b589c3c496176fa24e0f5d4da316907541389cecbd8361ce133a5179e7:
302 events,cap121,active118,starts118,saved114,failed4,pending0.
No additional fit/reservation/budget events; ledger unchanged through correction
and prelabel. After independent full numerical/boundary PASS, resume the original
frozen metrics and gates, independent metric verification, normal completion
receipt/event, concise report and scoped local commits. No gate change.
Evaluation targets remain closed until that complete PASS. Fit-prefix/bank
data may be read only under the original technical reconstruction boundaries.

Run focused synthetic QA and actual production-schema readback before official
verification. Synthetic artifacts/ledger live in a temporary isolated root;
production writer stays blocked. Commit amended evaluator before its official
execution. Record old and amended verifier identity and what was corrected.
If a distinct real integrity mismatch emerges, stop and return it to Sol;
do not introduce further fixes beyond this identity/source-lineage amendment.

No Test/Kaggle/submission, notebook/shared-team edits, extra research or push.
Luna owns amendment implementation, verification, evaluation and reports;
Sol stops writing after dispatch. Return result/blocker to KASDAD - TK2 - Sol,
thread01a19569-16e0-7253-aa14-57a0a3160e03,host local, under Sam's existing
result-return authorization. Then STOP.
