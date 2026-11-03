# Sol review: CGEOM V1 is informative but not qualified

Decision: ACCEPT_DIAGNOSTIC_RESULTS; CLOSE_EXACT_CGEOM_RECIPE; KEEP_A02.
Reviewed main@4c1f900c6bba953d12cb7271e2fd479398884cca, clean.

| View | n | Aligned Macro F1 | A02 | Permuted |
|---|---:|---:|---:|---:|
| E67 H12 | 20903 | .811820 | .811159 | .811130 |
| E75 W6 | 8491 | .819068 | .819604 | .819313 |

E67 H12 aligned gains about .000661 versus A02 and .000690 versus permuted.
E67 W4,W5,W6 also have positive aligned differences versus A02, but the newer
E75 fit on the same W6 rows reverses the sign: -.000536338 versus A02,
-.000245142 versus permuted. This is history-sensitive weak evidence, not the
large broad degradation observed for FM V2. It does not establish a reliable
deployment improvement or prove that coefficient-informed metrics never help.

E67 H12 confusion counts improve by10 fewer FP and3 fewer FN versus A02.
E75 W6 worsens by3 more FP and1 more FN. The later loss is small, but remains
negative against both comparators; no post-outcome gate reinterpretation or
weight/mapping rescue is justified. Multiple gates on the same rows are not
independent evidence:60/62 PASS is not a confidence level or success probability.
All52 subgroup checks passed; the two failures concern global E75 performance.

Accept documented zero-fit expected-ID dtype amendment and completed diagnostic:
original artifacts/fit receipts preserved; independent full-bank numerical and
boundary verification PASS before labels; independent metrics/gates PASS.
Sol inspected the results, amendment/source-identity explanation, scored verifier
and working tree. No new metric computation from raw labels, fits, or independent
full source replay occurred during this review. Executor's9 focused adapter tests
and4 artifact readbacks are reported QA, not tests rerun by Sol.

The exact one-lane milestone is complete. No automatic next lane, full-Train,
Test or submission. Preserve A02 as best-public-observed Classification reference.
Keep all original gates and failed verdicts. Any next research requires a distinct
question and bounded decision; this note starts nothing further.
