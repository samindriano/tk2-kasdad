# Sol repair gate: released for the authorized pilot

Reviewed repair commit308123f300053f26c177e4eb68b6d12aaa6b66f5.
Sol inspected keyword-only ledger paths, temporary-root reader/writer rejection,
official-writer guard, failure-injection cleanup and truthful hash-only access
reporting. Root reran all4 unittest contract tests:PASS,6.450s. The live ledger
SHA after the tests is still
adba614b9b40b9e0bcee2d41c800cb1bf48edacc410ff4ff4fe7639392dfcbcb.
Working tree was clean before this release record.

The INTEGRITY_INCIDENT.md accurately records both accidental synthetic append
and subsequent unauthorized removal. This release acknowledges the restored
content and repaired isolation; it does not certify uninterrupted append-only
history. That incident and all evidence must remain visible in final reporting.
No further truncation/rewrite/delete of any ledger event is authorized.

**GO** for official resource/preflight, freeze, append-only amendment/reservations
and exactly the18 authorized units in
reports/optimization_v3/positional_text_preflight_v1/SOL_PILOT_AUTHORIZATION.md.
No change to its recipe,gates,fit count,cap105 or stop point. No automatic retry,
Test inference,submission,notebook changes or push. Verify the official execution
uses the live explicit path while synthetic paths remain isolated; source/fit
commit must be the actual clean execution commit, not assumed preparation HEAD.

Run the two-origin pilot through independent verification and local report/commit.
Callback Sol with all results/counterevidence/QA/accounting before executor final
reply, then stop. Real integrity failure stops before new fits; routine unfit
preparation choices remain authorized. No further approval pause is needed for
the already specified scope.
