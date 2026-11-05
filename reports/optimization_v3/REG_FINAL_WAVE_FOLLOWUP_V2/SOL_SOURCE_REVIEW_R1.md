# Root source review R1 — REWORK, zero official fits

Reviewed source commit 9ca8e948f21e37cc989de26899c307375d8a175b. Root reran `python -m pytest -q tests/test_reg_final_wave_followup_v2.py`: **10 passed in4.76s**. Root RAM snapshot after tests: **3.455GiB available**, below4GiB; no official labels/LP/CSV opened or started by this review. Worker identity preflight remains missing after earlier1.88GiB gate failure. No ROOT_SOURCE_PASS issued.

Inspected weighted LP objective/constraints, causal per-window global/present-only versus recency construction, frozen row-origin identity payloads, shared OS lock, accounting/resume, and independent verifier source. Causal GLOBAL_PRESENT_ONLY correctly uses causal per-history affine coefficients; final full-support global is explicitly descriptive. Objective and derivative-free LP constraint orientation appear consistent. Tests support the synthetic scope, not official input readiness. No local/public score inferred.

## Required repair before readiness

Runner `verify_prefit_freeze`, `_verify_root_source_pass`, `_load_official_labels`, and pre-LP start check, plus independent verifier's prefit source check, require live HEAD == frozen source_commit. This prevents persisting a legitimate report/freeze/root-review/diagnostic checkpoint commit and then continuing/resuming later stages even when all executable dependency bytes are unchanged. The handoff requires durable checkpointing; exact HEAD is unnecessarily coupled to output commits.

Replace live-HEAD equality with: frozen source commit is a proven ancestor of live HEAD on exact expected branch; every pinned executable/dependency/protocol/test/QA source byte and every input hash still matches; source checkout has no tracked/staged executable changes. Authorization receipts still bind the exact frozen source commit/active freeze digest and approved fit IDs. Unrelated source modifications must fail. Do not solve by resetting/rebasing history, changing old freeze, weakening digest checks, or declaring old pins valid for changed source.

Add meaningful synthetic regression checks: report-only descendant accepted with unchanged source/input pins; nonancestor commit rejected; executable mutation rejected; durable saved-fit reuse remains zero-refit after report-only descendant. Preserve previous freeze and source-preparation blocker receipts. Publish a versioned successor freeze/preflight/QA mapping appropriate for the repaired executable and state explicit no old official slots consumed. Avoid a new whole workflow or template rewrite.

Small usability check: repeated successful identity-preflight invocation should verify/reuse its immutable receipt plus fresh resource gates rather than attempt to overwrite differing resource snapshots. Add reuse check if absent.

After scoped repair + focused tests, commit source, prepare successor identity freeze, run identity-only preflight only if actualRAM>=4GiB and shared project job gate passes. If resources still unavailable, return source-ready-but-resource-blocked once; no polling/schedules, no official fit, no question to Sam. All questions and readiness callback go to Sol. Root retains independent source approval and diagnostic-only execution decision.
