# REG final wave: feasibility and protocol

**Status:** Candidate A is skipped on a missing exact full-Train control prerequisite. Candidate B is prepared for independent prefit review; no candidate-data calibration has run. The only solver call so far is the synthetic entrypoint QA recorded in `SYNTHETIC_QA.json`.

## Scope and write boundary

Worktree: `C:\Users\Sam\OneDrive\Documents\KASDAD\tk2-reg-position-failure-mechanism-v1`, branch `codex/reg-position-failure-mechanism-v1`, starting HEAD `78a46b4a2bf2a69fe88e49f8e2b001c2ef699443`.

All new files and any eventual output are confined to this worktree. The read-only source roots are `C:\Users\Sam\OneDrive\Documents\KASDAD\tk2-regression-frontier` and `C:\Users\Sam\OneDrive\Documents\KASDAD\TP 2`. The full machine-readable input paths and SHA-256 values are recorded in `experiments/REG_FINAL_WAVE_V1/PREFIT_FREEZE.json`.

## Candidate A — `R16_RECENCY_ORIGINAL_E79`: skip

The recipe requires an E79 LGB replacement while preserving the exact full-Train `.1875 Ridge + .25 sparseRank50` components and original rank mapping. Those full-Train non-LGB components are not available in the pinned artifacts.

`TP 2/experiments/runs/optimization_v3_batch1/R16/` contains only `R16_fit55.{json,npz}`, `R16_fit67.{json,npz}`, and `pilot_engine_report.json`; it has no E79 state. The R06 full-Train Test archive `TP 2/experiments/runs/optimization_v3_expanded_submissions_v1/R06_ORIGINAL/predictions.npz` is SHA-256 `ce034e82a7063ec5dd873b1126d60731dd291d3852c57936c7aa1ee449a81eba`; its only members are `champion`, `raw_candidate`, `rank_mapped_candidate`, `cycle_code`, and `prediction`. It does not contain separate Ridge and sparseRank50 outputs. The R16 OOF loader also has saved controls only for cutoffs 55 and 67. Decomposing the composite R06 output would invent component values, so it is prohibited.

Skip reason: `BLOCKED_EXACT_FULLTRAIN_NONLGB_COMPONENTS_UNAVAILABLE`. Candidate A uses zero fits and zero preprocessing fits. No substitute components or refits will be used.

## Candidate B — `CURRENT_DIRECT_AFFINE_L1_E79`

### Frozen OOF support and baseline

Use only the existing REAL direct-position OOF bundles from the read-only frontier checkout:

| Artifact | SHA-256 | Frozen population |
|---|---|---|
| `experiments/runs/reg_position_direct_l1_v1/predictions/REG_POSITION_DIRECT_L1_V1_E67_REAL.npz` | `116d4c51f43f96a9597b8d1c6a63af10adf022f21ac7c797f92666c7feea425f` | 20,903 query rows, cycles 68–79, fit cycles 47–67 |
| `experiments/runs/reg_position_direct_l1_v1/predictions/REG_POSITION_DIRECT_L1_V1_E75_REAL.npz` | `da8e99eacfa9cb6ba3cfdac776a5e410b390e6ac90618ba22edea144c577ba39` | 8,491 query rows, cycles 76–79, fit cycles 47–75 |
| `experiments/runs/reg_position_direct_l1_v1/FREEZE_V4.json` | `8294ec5197f115bf11fae1a90dacda91fcc1ad789134e228357384d958994097` | Signed temporal origins and same-origin R06 anchors |
| `reports/optimization_v3/reg_position_direct_l1_v1/PRELABEL_VERIFICATION.json` | `7aaf6e4585d29fb92678542c61ea4b7d94256508733f9a5c25161b6c3f90d914` | PASS; query targets had not been read |

Stitch E67 predictions only for cycles 68–75 with E75 predictions only for 76–79. The resulting physical-row and record-ID support must be exactly 20,903 unique records. E67 cycles 76–79 are omitted because E75 scores those same records with a later prefix. Each prefix ends before every cycle it scores. Each origin uses its own frozen R06 baseline.

Reconstruct the current direct policy as `clip(0.75 * same-origin R06 + 0.25 * mapped auxiliary, 0, 100)`, then restore exact R06 values on rows marked missing. Require equality to each saved `prediction` vector before stitching. The selected-row reader `scripts/reg_position_failure_mechanism_v1.py` (SHA-256 `664ad522782e7ea84679aede04c8d9376cc94c05e3138ee90fc054fc974a2bb6`) parses `threat_rank` only for those 20,903 pre-scored rows; it skips unselected Train lines without decoding them.

### Single permitted calibration and Test transform

Call the existing R17 helper in `src/models/optimization_v3_batch1_r17.py` (SHA-256 `1d5d74cdf8215b8df654f5492cfa9ba396873e2bcbf58075550d9da1f331619d`) exactly once on the 20,903 stitched OOF predictions and their selected Train targets. Use `scipy.optimize.linprog(method="highs")` to minimize the sum of absolute errors with intercept free and slope `b >= 0`. No grids, regularization, weights, tail routing, fresh model, or second calibrator are permitted. The minimum support is 5,000 rows.

Apply `clip(a + b * p, 0, 100)` to all 29,421 rows in the existing B prediction CSV, SHA-256 `49ae7f668aea99e433c2bec72ed5ff51126fb12d888c3185ec048730d767ea6f`. Preserve its two-column schema and ordered IDs; verify their digest against the signed existing inference manifest. This transform reads no Test or SampleSubmission file, calls no model `predict`, and reads no test labels.

The base CSV was previously uploaded once as `direct_position_real_e79.csv` (ref `57988146`). The calibrated candidate has the distinct identity and output filename `current_direct_affine_l1_e79.csv`; the local upload-event history contains no upload for that identity. No upload occurs in this worktree; Sol/root owns any later upload decision and action.

The existing base CSV includes 1,062 rows that were exact R06 fallbacks. The affine map intentionally transforms these rows too, so it changes their fallback and carries known tail risk. The LP objective is an in-sample fit objective over the 20,903 OOF rows. It is not a fresh holdout, validation result, or claim of a generalization win.

## Fit and resource budget

- Candidate A: skipped; 0 candidate fits and 0 preprocessing fits.
- Candidate B: at most 1 affine LP fit; 0 preprocessing/model fits; at most 1 LP call.
- Synthetic QA: 1 solver call on deterministic synthetic vectors only; it reads no dataset and is not a candidate fit.
- One job, two threads, at least 4 GiB available RAM before fit with a 2 GiB reserve; estimated LP peak is 512 MiB.
- State and outputs are created once with exclusive file creation, flush, `fsync`, then SHA-256. A started LP cannot be retried.
- No Test/Sample access, Kaggle access, upload, network lookup, public-score tuning, or mutation of either read-only source root.

## Gate and expected outputs

`SYNTHETIC_QA.json` passes with maximum reconstruction error `7.11e-15`. The source, protocol, signed prefit freeze, and synthetic QA must be committed. Sol must independently review that exact artifact set and return PASS; only then can the one-shot `--run` entrypoint read selected Train labels and call the LP. Any failed prerequisite blocks Candidate B without fitting.

On solver success, the runner writes the calibrated CSV, calibrator state, fit receipt, signed inference manifest, local CSV-QA receipt, and isolated fit ledger under this worktree. The output status is `EXPLORATORY_READY_PENDING_ROOT_UPLOAD`; no outcome comparison or upload is performed here.
