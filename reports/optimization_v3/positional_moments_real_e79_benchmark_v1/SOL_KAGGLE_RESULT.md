# Sol upload review and public result

Kaggle confirmed submission57976581 COMPLETE on4November2026 WIB, public Macro
F1 **0.82979**, versus previous best MIX50 **0.83144** (delta **-0.00165**).
Private performance is unknown. Historical POS_MOMENTS_V1 remains gate-failed;
this was an explicitly authorized exploratory benchmark, not model promotion.

Sol reran the independent feature/model/formula verifier with only its QA and
manifest OUTPUT paths redirected to a temporary ignored directory. It returned
PASS_INDEPENDENT_FULL_QA_CANDIDATE_NOT_SUBMITTED for all29,421 rows,1,062 missing,
with original metadata files unchanged. Root separately checked CSV SHA/schema,
binary labels and exact SampleSubmission ID order. One upload was made via the
official already authenticated Kaggle CLI after live duplicate-name history check.

CSV SHA256:809339b9506f58cd3302c4d439ee9ce80df17c5eb48243640bcd7c5c8450177b.
API result, one-attempt intent/response and review scope are recorded beside this
report. No credentials, new fit, inference retry or final-selection change.

Both recent exploratory pushes failed to improve best public: Regression REAL
11.24513 versus R0611.16620; Classification moments .82979 versus MIX50 .83144.
These are separate metrics/tasks, not comparable effect sizes. Public outcomes
do not prove no information exists, but give no promotion evidence for the exact
recipes. Sam explicitly requests continuing bounded research; the next CLS scope
is target-free readiness/design for bigram placement on top of the original
winning POS anchor, not a sweep of moment weights or thresholds. No automatic
new fit or upload follows this report. Local commit only; no push.
