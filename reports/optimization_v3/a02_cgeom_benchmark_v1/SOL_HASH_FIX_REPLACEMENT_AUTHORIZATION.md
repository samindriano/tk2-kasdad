# Three-CSV batch: one pre-estimator hash-check replacement

Sol decision: GO_ONE_TECHNICAL_REPLACEMENT_AND_RESUME_BATCH_PREPARATION.
Scope remains Sam's batch of3 frozen Classification CSVs, no Kaggle upload.
Incoming HEAD4480f74ac909da4641fb9278dd9b536f66d8889d, clean.

Reviewed BLOCKER_C2B_LR_VERIFIER.json and actual runner check at line873:
fit_row_positions_sha256 was compared using positional.sha256_array, whereas
the saved A02 field was created with A02 _array_sha256(rows,'<i8'), a raw-byte
SHA convention. Independently recomputing SHA256(arange(50608,dtype='<i8').tobytes)
gives b3412a08e9f0877cc37ef9615479de0d182a21b9806068c2d576ed06ae495bb2,
matching the original A02 field. Document/missing/population checks also match
per blocker evidence. This is a false lineage rejection, not a model result.
Estimator.fit was never reached and no LR artifact exists. Do not recover a
model that was never trained or reinterpret the failed slot as successful.

## Exact correction and exception

Correct only the hash-convention comparison to the original field's declared
algorithm/dtype. Keep document, row-count/order, prefix, missing-mask, vocabulary,
IDF and input-source checks. Record algorithm identity explicitly; synthetic
tests must demonstrate an altered row/order fails and differing digest formats
cannot be interchanged. Move these feature-only compatibility checks into actual
resume preflight BEFORE the new FIT_STARTED event, and exercise real full-Train
input lineage without calling estimator.fit. Do not weaken gates or fit scope.

Preserve original source snapshots, BLOCKER JSON, failed event and all3 saved
preprocessing artifacts/receipts exactly. Reuse these units after verification;
never refit or silently rebind them to a newer source commit. New LR receipt
records corrected-source identity; consumer provenance records shared original
component sources separately. Commit source and replacement mapping before fit.

Authorize exactly ONE additional technical replacement ID:
A02_CGEOM_BENCHMARK_V1_E79_C2B_LR_HASHFIX_R1.
Map only the failed C2B_LR dependency to this fresh fit ID. Original ID remains
FIT_FAILED/consumed. No deletion/reuse/reset. This is an explicit exception to
the original no-retry policy, limited to this confirmed pre-estimator defect.
No further replacement or retry is authorized if the new fit fails.

Initial ledger313events,cap130,active127,starts122,saved117,failed5,pending0;
SHA c6edde633880213f67c067c993a8c8db840cba015554fb8f7c38f90ce0b20448.
Append prospective cap130->131 and continuation47->48 amendment, other category
caps unchanged, and reserve1 replacement only. Five original NOT_STARTED units
stay reserved; do not reserve them again. Active becomes128. Whole batch cap is
10 consumed units instead of9 (including the original false-check failure);
total starts max128. If all remaining6 units succeed, saved123,failed5,pending0.
Global spare headroom is not more authorization. Preserve complete event history.
Reconcile expected checkpoints against the known failed ID; do not carry old
failed4/starts118 assumptions into resume. Any unrelated failure still blocks.

After source-bound synthetic QA and real feature-only hash preflight PASS,
fit replacement once, then resume the existing5 remaining units serially.
Generate all3 CSVs, immutable manifests, source boundaries, Unicode readbacks,
independent formula/ID/fallback QA and the original batch report. No new models,
recipes, thresholds, label validation or Test targets. Test features only during
authorized inference. Existing resource gates remain unchanged; no installs,
user-process termination, notebook/teammate edits or push.

Luna owns implementation/source commits/resume and reports, Sol stops writing
after dispatch. Routine fixes to this amendment before its fit are allowed;
distinct mismatch -> preserve evidence and return to Sol. Return batch result
or blocker to KASDAD - TK2 - Sol,thread01a19569-16e0-7253-aa14-57a0a3160e03,
host local, under Sam's result-return authorization. STOP after prepared CSVs;
no upload under the current batch-first steering.
