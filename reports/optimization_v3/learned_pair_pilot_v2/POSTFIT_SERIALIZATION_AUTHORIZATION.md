# FM V2: zero-fit serialization amendment

Sol decision: GO_ZERO_FIT_SERIALIZATION_REPAIR_AND_RESUME_EVALUATION.
This is the narrowly scoped post-fit review required by SOL_EXECUTION_HANDOFF.md,
within Sam's authorization to complete the FM V2 pilot. No new fit or budget.

Verified incoming HEAD 2e5418c5181e08c77c6c4c0e135835d5f4fc84a2, clean.
All four V2 fits are saved; do not run an optimizer or preprocessing fit again.
Independent pre-label verification failed on an object-dtype score_record_ids
member. Original failure evidence remains POSTFIT_BLOCKER.json, SHA256
561dda827f000bf7e681026e1504731249fa339db08689e5f1e1e95a6483f323.
Ledger SHA256 ccf18357d2d7fcb513e9f36b8fb2ec39688d28ab3d43471b5db345c35fce6b14;
291 events,114 starts,110 saved,4 historical failures,0 pending. Cap117 and
all reservations/gates/fit receipts/model parameters remain unchanged.

Source review: frozen runner _load_origin_unlabeled uses
frame.iloc[score_rows]['record_id'].astype(str).to_numpy(); build_predictions
writes that array directly. Pandas string conversion does not guarantee a
non-object NumPy array. This is an output serialization defect. It is not a
new predictive outcome or a reason to rerun fits. Full numerical identity
still must be demonstrated before resuming evaluation.

## Permitted correction

Create an isolated postfit serialization amendment namespace, new adapter,
independent verifier extension and focused tests. Preserve original frozen
runner/verifier/model code, FREEZE, fit parameters, receipts, both malformed
prediction archives, original readiness metadata and failure diagnostic.
Never overwrite original outputs or silently update their hashes.

Read old NPZ with allow_pickle=False, selecting safe members individually;
never load the object member or enable pickle. Prefer preserving exact raw
NPY member bytes for every member except score_record_ids.npy. In derived
archives, rebuild only this ID array from frozen source row indices and
verified historical score identities/official Train record_id column.
Reading label-free identity columns is allowed. Require equality with the
same-origin historical positional and A02 identities, cycles and row order.
Write explicit fixed-width Unicode with sufficient length, no truncation,
no object dtype. Disclose that original object ID payload values were not
deserialized; corrected IDs are independently re-established from source.

Bind old/new archive hashes and per-member hashes in a create-once amendment
receipt. For all safe original arrays (probabilities, raw logits, pair scores,
fallbacks, masks, rows, cycles and any other numeric/string members), require
byte identity or independently recorded exact dtype/shape/value identity.
No predictions may change. Do not recompute predictions as a substitute for
preserving numerical arrays; saved fit scores may be recomputed only by the
independent verifier as in the original protocol. No refit, threshold, route,
metric, group definition or comparator change.

Create new readiness/prelabel/results metadata under the amendment namespace.
Keep original training source commit/freeze unchanged and explicitly record
the new export/evaluation source commit and amendment authorization separately.
Do not rewrite the frozen source identity to the amendment commit. A new
postfit adapter may validate the original source by its frozen file hashes
and the exact original Git blobs rather than requiring current HEAD to equal
the training commit; this exception is limited to the amendment and must not
weaken any source/input/artifact/receipt check. No broad monkeypatch disabling
source checks. Reuse frozen pure helpers if appropriate, but preserve every
independent verifier condition and both mechanism/competition gates.

## QA and evaluation continuation

Before official derived export, test synthetic object-ID rejection, Unicode
roundtrip including long IDs, unchanged numeric members, same-origin alignment,
tampered numeric/ID rejection, old/new receipt binding, resume without fit and
all original gates. Synthetic files/ledgers are temporary and production ledger
writer must stay blocked; verify original artifacts and ledger remain unchanged.
Commit adapter/verifier/tests before source-bound amendment execution.

After derived export, independent pre-label verification must repeat the full
original checks, including fit-prefix gradient/objective, source matrices,
all four fit receipts, routed probabilities, A02 identities/hashes, budget and
row boundaries, plus amendment numerical-identity checks. Fit-prefix labels
may be reread only for original authorized technical verification. Query
evaluation targets remain closed until the complete verifier returns PASS.

Only after PASS, evaluate the same frozen metrics and gates, independently
verify results, produce report(s) and one normal completion event if required
by the original lifecycle. No new FIT/RESERVE/ALLOWANCE events; ledger remains
byte-identical throughout repair/prelabel. Any later completion event must
bind the actual amendment result and preserve all earlier events.
On another real blocker, retain evidence and return to Sol; no broader rescue.

Luna owns amendment implementation/execution and scoped local commits; Sol
will not concurrently edit those files. No push, Test/Kaggle, notebook edits,
full-Train inference or submission. Return results (or blocker) to KASDAD -
TK2 - Sol, thread01a19569-16e0-7253-aa14-57a0a3160e03,host local, before final
response, under Sam's standing result-return authorization. Then STOP.
