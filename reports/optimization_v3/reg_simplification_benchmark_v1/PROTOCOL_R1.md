# REG simplification benchmark R1

## Scope and authority

This namespace implements only the two REG simplification candidates selected by the read-only root audit: `NO_POS` and `NO_CHAR`. The audit worktree remains read-only at named commit `0052f9c9aaa564caf6d9292dc9d9fa369adeef4e`; its input freeze head is `a39721172b6d293cb68ee0d120060082d222699e`. Source pins and SHA-256 values are in `INPUT_MANIFEST_R1.json`.

No predictor, preprocessing, or calibrator is fitted. No Test labels are read. The existing fixed final HINGE is reused with intercept `1.7288079036674198`, low slope `0.9933285599956824`, high slope `0.957778072288735`, and knot `50`. It is applied to present rows only; each candidate's specified missing route stays raw.

## Candidate recipes

`NO_POS` reuses the E79 R06 final mapped prediction byte-for-byte as its raw present-row value. For missing rows it uses raw R06, with no HINGE. For present rows it applies the fixed HINGE to the saved final R06 prediction.

`NO_CHAR` builds the present-row raw value from saved components:

```text
raw_present = clip((9/13) * legacyThree + (4/13) * rank50_pos_unclipped, 0, 100)
final_present = fixed_HINGE(raw_present)
raw_missing = legacyThree
final_missing = raw_missing
```

`rank50_pos_unclipped` is computed separately for every complete saved cycle as `0.5 * raw_position + 0.5 * 100 * average_rank(raw_position) / cycle_size`. The full-cycle average-rank tie rule is used. The position auxiliary blend is not clipped before combination. The distinct `legacyThree` missing route is deliberate and must be disclosed when interpreting this candidate.

The runner verifies the exact official `SampleSubmission.csv` `record_id` order, the saved E79 IDs, all source hashes, the R06 reconstruction, both missing routes, output finiteness/range, and local vector duplicates. The independent verifier separately reloads the pinned saved arrays and recomputes both CSV vectors. It also verifies CSV schema, exact row order, exact float round-trip, and the recorded output hash.

## Audit findings and interpretation

The root audit retained `NO_POS` with pooled change `+0.035148178` and `NO_CHAR` with pooled change `+0.002980398`. `NO_CHAR` was less stable: D1 change `-0.026169823`, D2 `+0.028597987`, and maximum supported-group harm `0.247448961`. These are root-audit findings, not new measurements from this build. Do not combine the candidates or reinterpret their routing. The already-uploaded `GLOBAL_INSTEAD_HINGE` result (reported 11.04304) is not resubmitted; prior failed candidates remain failed.

## Resource policy for cached vectors

`EXECUTION_POLICY_R1.json` records the user's resource-only override for the frozen cached-vector/CSV stage. At most two light jobs total may run in parallel (one CLS and one REG); the two REG candidates run serially within REG. Before reading cached vectors, take the exclusive REG lane idempotency lock and measure global available RAM. Require at least 2.5 GiB available, estimate at most 512 MiB incremental use, and preserve at least 1 GiB projected and actual available RAM. Do not acquire the shared heavy-job lock for this cached-vector stage. A failed lock or RAM gate stops the run; it does not poll, lower thresholds, or remove a stale lock automatically.

This versioned resource policy does not change frozen data, Test-label, fit, candidate-selection, or local acceptance boundaries. The build writes `RESOURCE_GATE_R1.json` with the measured start and end values. Do not start the REG vector run until Luna1 has sent the single pre-start coordination note and a fresh RAM snapshot still passes the same gate.

## Build and QA sequence

1. Review this protocol, `EXECUTION_POLICY_R1.json`, and `INPUT_MANIFEST_R1.json`; require all file hashes to match.
2. Run `python scripts/reg_simplification_benchmark_v1.py --check-pins` without loading NPZ arrays.
3. Run the targeted synthetic-only tests with `python -m pytest tests/test_reg_simplification_benchmark_v1.py`.
4. After the coordinated fresh resource gate, run `python scripts/reg_simplification_benchmark_v1.py --build` once. It writes both CSVs serially plus reconstruction, duplicate, resource, and build receipts. Existing outputs or receipts are immutable and cause a stop.
5. Run `python scripts/verify_reg_simplification_benchmark_v1.py` once for each candidate; each independent receipt is immutable. Any UNKNOWN, hash mismatch, local duplicate, row mismatch, or nonzero numeric difference blocks submission.
6. Capture Kaggle competition history in JSON immediately before each candidate's one authorized upload. The exact upload message must not already appear in history. Submit `NO_POS` with `Samuel REG Simplified No POS Hinge Benchmark`; then use a single status read to confirm `COMPLETE` and public score before proceeding serially to `NO_CHAR` with `Samuel REG Simplified No Character Hinge Benchmark`.
7. Each upload writes a durable intent before the CLI side effect and a response afterwards. Timeout or uncertain response consumes the one attempt; never retry. Do not use CLI `--wait`, active polling, or a scheduler. If the single status read is not yet `COMPLETE`, preserve the status and use only a later discrete read after a meaningful wait; never resubmit.

## Artifacts and boundaries

The input manifest is under `experiments/reg_simplification_benchmark_v1/`. Candidate CSVs go under `submissions/reg_simplification_benchmark_v1/`; reconstruction, duplicate, resource, build, independent-verification, Kaggle-history, and upload receipts go under `reports/optimization_v3/reg_simplification_benchmark_v1/`. Do not modify the audit worktree, final selection, or local gates. Do not push, merge, cherry-pick, delete, or reset. Preserve the separate untracked `reg_center_local_application_v1/SOURCE_PREP_BLOCKED_R1.json` checkpoint and all unrelated workspace changes. Stage only the named simplification files.
