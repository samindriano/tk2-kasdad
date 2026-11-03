# Batch integration repair: real feature pipeline verified before replacement

Sol reviewed stopped HEAD4eeac6b96b97fb05ada7ae02da4536eebeb071a3.
AST audit of all positional.* references found TWO missing script exports:
transform_c2b_structured and build_position_block. Both exist in the model
module. Sol changed only these calls to an explicit positional_core import.
Historical positional source, model recipes and saved artifacts remain unchanged.

Sol ran the actual batch _c2b_matrices on all50608 official Train FEATURE rows,
using the three cached preprocessing units and A02 vectorizer. PASS:
base shape50608x20427; position shape50608x17259; finite matrix data; no target
columns read; no fit invoked; live ledger hash unchanged. This verifies the
production transform branch that previous synthetic tests did not exercise.
It does not certify every later training/inference branch.

GO to one bounded integration replacement and resume batch. First require tests
of actual production callback bindings (including missing HGB, gamma, joint LR,
geometry construction, persistence) with synthetic inputs; do not merely replace
the production callback with a dummy function. No Test features until trained
freeze. No recipe/gate changes. Any distinct failure stops for Sol review.

Preserve two failed LR IDs/events and source snapshots and all3 saved preprocessing
units/receipts. Do not refit successful units. The five old NOT_STARTED IDs were
released; do not treat them as still reserved. Use fresh replacement IDs for
the LR and the five released components (six new IDs), with explicit canonical
component-to-active-ID mapping, new source lineage, and original-source receipts
for shared saved units. This prevents generic release-accounting code from
subtracting re-reserved IDs twice. Record old-to-new maps in the amendment.
Original unstarted IDs remain released; failed IDs remain consumed.

Starting ledger318events,cap131,active123,starts123,saved117,failed6,pending0;
SHA5ea031efca2a09ad0f0c9d0204288412735c051114dae9085e6f828a4cc4aca0.
Authorize cap131->132,continuation48->49, other category caps unchanged;
reserve exactly6 new IDs for the remaining6 logical components, not9/10 units.
Active after reservation129, maximum totalstarts129. Whole batch max11 consumed
units including both pre-estimator failures; if six remaining units succeed,
saved123,failed6,pending0. Global spare headroom is not more permission.
No retry after this integration replacement authorization. Release only
never-started new IDs if failure recurs; do not delete/refund any event.

Commit corrected source/tests/mapping before official work. Validate the shared
components under their original hashes/commits, not a false current-source
attestation. Real feature-only compatibility checks must run before FIT_STARTED.
Freeze/resume logic must explicitly support this amended checkpoint; retain
original preflight/failure evidence. Source-bound QA must not alter ledger.

Finish exactly the three selected CSVs and independent QA/report, then STOP
PREPARED_FOR_BATCH_REVIEW_NOT_SUBMITTED. Kaggle CLI is now installed/authenticated
but this batch-first checkpoint still does not upload. No notebook changes,
push, new research, installs or Test labels. Send result/blocker to Sol chat
thread01a19569-16e0-7253-aa14-57a0a3160e03,host local, under Sam's standing
return-message authorization. Sol stops writing after this dispatch.
