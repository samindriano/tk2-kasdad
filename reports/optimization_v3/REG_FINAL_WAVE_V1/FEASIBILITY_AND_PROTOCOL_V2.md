# REG final wave: feasibility and protocol V2

**Status:** Candidate A is `SKIP_A_SCOPE_LIMITED_NOT_FOUND`. Candidate B is ready for source-only prefit review. No Train targets have been parsed for this wave, no candidate calibration has been fit, no Test/Sample file has been read, and no upload has been made.

## Scope and starting state

Writable worktree: `C:\Users\Sam\OneDrive\Documents\KASDAD\tk2-reg-position-failure-mechanism-v1`, branch `codex/reg-position-failure-mechanism-v1`. The V2 package is based on V1 commit `635b6371c503f81ded3a8ac9482f0066a5b5a65c`. V1 remains preserved; V2 files use versioned names. Read-only source roots are `C:\Users\Sam\OneDrive\Documents\KASDAD\tk2-regression-frontier` and `C:\Users\Sam\OneDrive\Documents\KASDAD\TP 2`.

V1 was returned for rework before any candidate-data fit. The fixes are: preserve original full-bundle indices while stitching the E67/E75 rows; exercise shuffled rows and a corrupted duplicate-ID case synthetically; round-trip parse and compare the base CSV bit-for-bit against its signed saved prediction archive; record R16/R06 absence as scope-limited with broader availability UNKNOWN; and run fresh memory and one-job gates immediately before the only candidate LP. The V1 script, protocol, freeze, and QA were not edited.

## Candidate A: bounded feasibility result

Candidate A (`R16_RECENCY_ORIGINAL_E79`) requires an E79 LGB replacement while retaining the exact full-Train Ridge and sparseRank50 vectors and original ranking. The pinned artifacts do not contain those required vectors, so A is skipped without substitution or fitting.

The exact pinned R16 directory `TP 2/experiments/runs/optimization_v3_batch1/R16/` contains `R16_fit55.json`, `R16_fit55.npz`, `R16_fit67.json`, `R16_fit67.npz`, and `pilot_engine_report.json`; there is no E79 state. The pinned R06 output archive (SHA-256 `ce034e82a7063ec5dd873b1126d60731dd291d3852c57936c7aa1ee449a81eba`) contains only `champion`, `raw_candidate`, `rank_mapped_candidate`, `cycle_code`, and `prediction`. The exact R06 manifest, fit state, and attempt record are hash-pinned in `PREFIT_FREEZE_V2.json`; the current fit-state ID is `V3_BATCH1_R06_CHAR_SPARSE_L1`.

The R06 manifest's legacy `champion_reuse` reference is `57835122`. It records only component prediction digests: LGB `8cc839f49b2ac22d9d6bae1ba6922c9e53b8ea51185437ab4ff6df5c42deff34`, Ridge `204d5c343b4ef5392c9d8c7e224ef3d9161d63e47fb24615211459e0458e7475`, and `lgbm_sparse_l1` `66c3c631c7785c53c5946980b459b06e197e0e59b9baf9f2dfc231296cdd6c60`. Digests do not provide the Ridge vector. The pinned R16 source and runner, also hash-bound in the freeze, expose pilot cutoffs 55 and 67, not the needed E79 controls.

Therefore A's machine status is `SKIP_A_SCOPE_LIMITED_NOT_FOUND`; broader workspace availability remains `UNKNOWN` because no broader search was performed. This is not a global unavailability claim. Composite decomposition is prohibited. A uses zero candidate fit units and zero preprocessing fit units.

## Candidate B: fixed recipe and data boundary

Candidate B is `CURRENT_DIRECT_AFFINE_L1_E79`. It uses only the saved REAL E67 and E75 direct-position bundles from the read-only `tk2-regression-frontier` worktree, with their signed receipts, direct-pilot freeze, prelabel receipt, and same-origin R06 baselines pinned by SHA-256.

The row stitch takes E67 cycles 68–75 and E75 cycles 76–79. For late cycles, E75 must match the E67 duplicate subset by physical row, record ID, and cycle. The script retains each selected row's original bundle index, then sorts the assembled support by physical row. It requires exactly 20,903 unique rows and IDs: 12,412 from E67 and 8,491 from E75. E67's fit prefix is cycles 47–67; E75's is cycles 47–75. The fit reads `threat_rank` only on these selected, already-scored query rows. It does not use fit-prefix labels to fit the affine map.

The base file is the existing 29,421-row `direct_position_real_e79.csv` (SHA-256 `49ae7f668aea99e433c2bec72ed5ff51126fb12d888c3185ec048730d767ea6f`), previously uploaded as ref `57988146`. V2 parses decimal values with `pandas.read_csv(float_precision="round_trip")` and requires bit-exact equality with the signed B `inference_B.npz` prediction array and its manifest prediction digest. It also requires exact ordered IDs and the recorded count of 1,062 rows that equal the R06 fallback.

The single permitted fit minimizes in-sample absolute error with the pinned R17 helper, subject to slope `b >= 0`; the intercept is free. After fitting, V2 applies `clip(a + b*p, 0, 100)` to all 29,421 saved base predictions, including all 1,062 former fallback rows. That can affect tail behavior. The fit objective is in-sample only; it makes no validation-win or generalization claim. No score, public leaderboard result, or test outcome is used for candidate selection.

No Test or Sample file is read. No new model prediction, model refit, preprocessing fit, external data access, or upload is part of this recipe. Output row order must exactly match the frozen base CSV and output decimal values must round-trip bit-exactly to the calibrated float64 values.

## Synthetic evidence and execution budgets

`SYNTHETIC_QA_V2.json` is a candidate-blind PASS. It records the R17 synthetic LP solution, a shuffled E67/E75 stitch with the expected original indices, and rejection of a corrupted duplicate ID. The reported maximum reconstruction error is `7.105427357601002e-15`; it reads no Train, Test, or Sample data and performs no candidate fit.

The QA command was invoked once more after its write-once receipt already existed. That invocation completed its synthetic LP and assertions, then refused to overwrite the existing receipt. Count all calls transparently: one V1 synthetic LP, one LP in the persisted V2 QA, and one LP in that V2 no-overwrite retry, for three synthetic LP calls before any candidate fit. The QA JSON records its own persisted run's one call. The V2 command now checks an existing receipt and returns without re-solving, so another invocation does not consume an LP. These are synthetic calls only; candidate-fit budget remains exactly one.

The signed prefit freeze limits the candidate to one calibrator fit, one R17 LP call, zero preprocessing fits, two numerical threads, and one concurrent project job. It requires at least 4 GiB available memory, reserves 2 GiB, and budgets an estimated peak of 512 MiB. Immediately before the LP, the script refreshes the memory check and scans other Python/PowerShell processes for project experiment markers. There is no retry after the LP starts. Any gate failure before the LP stops without fitting; a solver failure after it starts consumes the one attempt and cannot be retried.

## Required review gate and next action

The next action is source-only independent review of the exact V2 script, protocol, synthetic QA, and signed prefit freeze. Sol/root must return an explicit `PASS` for those exact hashes before any candidate-data target read or fit. The review receipt must bind the freeze signature, reviewed artifact hashes, and reviewed commit; the runner rejects a missing or mismatched receipt and requires a clean committed worktree.

Only after that explicit PASS may the runner be invoked once with `--run`. It then verifies the raw Train SHA, reads the selected 20,903 labels, repeats resource and one-job checks immediately before the LP, and writes output artifacts with exclusive-create semantics. A real fit result, if permitted and successful, remains local and exploratory. It must not be uploaded in this lane.
