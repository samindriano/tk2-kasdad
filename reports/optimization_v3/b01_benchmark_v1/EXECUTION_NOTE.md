# B01 output publication note

The B01 candidate recipe and inputs were frozen before its first Test-feature
inference in `FREEZE.json`. The candidate probabilities and CSV were written
from that frozen code and saved geometry. The first publication attempt then
raised a `NameError` while assembling the manifest because the frozen runner
referenced `ledger_after` instead of its already-verified local ledger value.

No Test target, Test prevalence, accuracy, Kaggle score, or public leaderboard
result was loaded or inspected. The failure occurred after writing the
candidate NPZ/CSV and before publishing the report bundle. To resume without
changing the frozen source or candidate, the follow-up helper
`scripts/optimization_v3_b01_benchmark_resume.py` supplies the current verified
ledger value in memory and reruns the frozen procedure. Its durable writer
requires the existing NPZ and CSV bytes to match exactly before publication;
the final hashes are independently checked in `MANIFEST.json` and `QA.json`.

The resume helper is a publication workaround only. It did not alter the B01
recipe, geometry, bank, fallback, threshold, probabilities, or labels. The
original B01 scientific verdict remains failed/diagnostic-only, and this
benchmark was not uploaded or authorized for submission.
