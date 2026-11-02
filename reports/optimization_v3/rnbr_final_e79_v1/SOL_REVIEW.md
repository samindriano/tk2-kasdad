# Sol review: approved for one manual Regression benchmark

Verdict: **APPROVE_MANUAL_BENCHMARK**, not champion promotion or an upload receipt.
CSV: submissions/optimization_v3_rnbr_final_e79_v1/samuel_rnbr_final_e79.csv.
SHA256: 54316cd4806c1957a87d46806b5eacb59a6f35287d8f31411f12139f4114362c.

Sol reviewed execution source, lineage checks, actual fit artifacts and independent
verifier. A separate read-only computation, without calling the production
prediction function, reconstructed weighted-median correction and final clipping
for all29,421 rows: exact equality with stored raw predictions and deltas. All
36,010 donor targets and IDs matched official Regression Train; residuals and
four-cycle OOF cutoff assignments matched, with every cutoff preceding its donor.
CSV schema/order/unique IDs/range/hash and all1,063 exact R06 fallbacks passed.
CSV parse precision differed from raw arrays by at most the declared5e-14 bound.

Additional independent spot checks transformed33 donor documents and recomputed
top32 neighbors/similarities for36 random present Test queries (seed20261103).
All matched exactly. Full target-free Test record_id/cycle_code/signal_stream
columns matched between tasks. No fitting was performed during Sol review.

Important provenance: this final target bank extends through cycle79, while the
reused A02 vectorizer was fitted on42,117 Train-prefix rows through cycle75.
This is a frozen-transform benchmark, not a full79 preprocessing refit. It does
not leak future labels; representation freshness remains a limitation. Loading
the A02 container to extract its vectorizer does not transfer Classification
labels or predictions into the Regression donor bank.

Mean absolute Test prediction change versus R06 is0.961297; correction quantiles
at0/10/50/90/100 percent are -5/-1.726737/-0.114692/1.261111/5. Those changes are
not observed Test improvement. Primary gain0.220627 and late gain0.148274 remain
development evidence; W4 harm0.112556, H12 harm0.048271 and historical R06
legacy-guard failure remain disclosed. No Test MAE/public score is known.

## Tests and limits

Command: python -m pytest -q tests/test_optimization_v3_rnbr_pilot.py
tests/test_optimization_v3_rnbr_continuation_corrected_eval.py

Result:17 passed,2 failed. Both failures require the historical preflight ledger
hash6fcc4b4d427e3ff03139e0e9ee429f529f7c7bf6ad91ce0eb9a52d55e2503147;
the append-only ledger has legitimately advanced to
35ca73c9adc64df8b794db29cc8cd39109362bea865a8bc6b1d8caf0c7703ff8.
Do not rewrite frozen historical expectations or claim the whole suite passed.
Current E79 synthetic persistence/readback/resume and independent artifact QA
are separately PASS. A small current-state regression-test supplement is the
only remaining executor cleanup requested; no additional official fit.

Manual Kaggle message: Samuel R-NBR Frozen Residual Neighbor Benchmark.
One upload of this exact hash is approved; no automatic upload or repeated
submission is requested. Record any returned score as separate public evidence.
No push, refit, tuning or champion replacement is authorized by this review.
