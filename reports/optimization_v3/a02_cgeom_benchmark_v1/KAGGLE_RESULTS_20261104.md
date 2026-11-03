# Classification batch: three confirmed Kaggle results

On4November2026 WIB, Sam authorized uploading the three prepared CSVs and checking
scores. Pre-upload history contained none of these filenames. Each file was
rechecked for frozen SHA,29421 rows,unique ordered official IDs,schema and binary
labels before any upload. Three sequential uploads used official Kaggle CLI2.2.5
with existing authentication; subsequent authenticated submissions API confirmed
COMPLETE and a public score for each. No credentials were logged.

| Candidate | Submission ref | Public Macro F1 | Delta vs A02 .82930 |
|---|---:|---:|---:|
| CGEOM aligned | 57955748 | .82884 | -.00046 |
| Positional bag offset | 57955753 | **.83123** | **+.00193** |
| Joint positional LR | 57955761 | .82911 | -.00019 |

Positional offset is the new best-public-observed Classification reference.
This replaces A02's public-record status, not its historical internal verdicts.
No final-entry selections were changed. No fourth upload or new training follows.
Local uploaded file hashes and API dates/refs are in KAGGLE_RESULTS_20261104.json;
append-only upload attempts, CLI responses and API observations are in
kaggle_submission_receipts_20261104.jsonl. Server-side file hashes/private scores
are not exposed in these observations and remain unknown.

Scientific interpretation: this batch supports retaining a distinction between
failed internal promotion gates and a poor public benchmark. Positional offset
had small positive gains versus C2_B at both pilot histories and failed its
prespecified .003 H12 margin; it was not a large negative experiment like FM.
Its public win does not retroactively qualify it under that protocol or identify
the cause of transfer. The current public subset is not an independent private
test, and repeated public selection remains a limitation.

The external review's proposal to plan final entries deserves separate review,
but its assertion that all remaining differences are below evidence resolution
is not established by the supplied summaries. Per-error Macro F1 increments
cannot be inferred from public row count alone without confusion counts.
Likewise, maximizing unlabeled prediction disagreement alone does not establish
a stronger final portfolio; base quality and error complementarity matter.
No final-slot mechanics, deadline, or selection were changed in this milestone.

Next recommended step is a bounded comparison of frozen positional-offset and
joint models and a final-entry decision, before any further adaptive fitting.
Do not infer Test prevalence or tune a threshold from these three public scores.
