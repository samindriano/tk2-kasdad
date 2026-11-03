# Pre-metric baseline clarification and artifact-reuse boundary

Sol reviewed the reported blocker before any new policy metrics. Original
handoff's phrase "Sparse25 baseline underlying pA" meant the ACTUAL routed
input probability supplied to original A02, not the raw sparse learner score.
Use b=A02 prediction bundle['baseline_prediction'] (or the equivalent saved
baseline_probability field according to the exact original artifact schema).
Require equality to the same-origin C16 cached baseline_prediction on exact
same rows. Do not use control__baseline_sparse_raw. No fourth policy.
ADD_RESIDUAL remains pP+(pA-b) on present rows, then clip; missing exactly pP.
pA-b is the effective correction after A02 output clipping, not necessarily its
unclipped neighbor correction. Preserve this distinction in the manifest.

MIX50 is unchanged. Both formulas still frozen before new outcome access.

## Explicit independent-verification scope

This endgame reuses fixed, hash/receipt-bound prediction artifacts; it does not
re-execute or certify every old fitted component. The frozen legacy loader
expects C2_B/CB arrays absent from the bound C16 NPZs. Preserve that discrepancy
as UNKNOWN_UNAVAILABLE_COMPONENT_DECOMPOSITION. Historical PASS is historical
evidence, not a new successful reconstruction. Do NOT patch the loader to
manufacture arrays, skip a failed check while reporting full replay, or refit.

The following alternative evidence is mandatory for the present derived-policy
scope: original artifact/source/input/receipt hashes; exact row/ID/cycle/mask
alignment; b equality between A02 package and matching C16 baseline cache;
OOF donor/model state identity; independently reconstruct original A02 observed
correction from its saved donors, similarities and residual bank, cap/shrinkage
and final clip, comparing pA and b to original immutable predictions. Preserve
the original prediction package; do not infer b by subtracting a new correction.
All prefixes/donor dates remain valid and no new target labels enter prediction.
If this fails or reveals a concrete provenance mismatch rather than absent
legacy decomposition, STOP to Sol before metrics. If all pass, artifact-based
endgame verification can PASS with the legacy decomposition UNKNOWN disclosed.
Do not relabel it as full six-cache component reconstruction.

E79 uses the same semantic b from the already bit-exact-replayed A02 E79 package,
not raw sparse or B01. Manifest records exact field names and hashes per origin.
No change to numerical gates, candidate priority, zero-fit budget, scope,
single-eligible-upload rule or final-entry selection boundary.
Proceed with source-bound synthetic QA/freeze before metrics, send result or
new real blocker to Sol, then STOP. Sol stops writing after this clarification.
