# R-NBR Four-Fit Memory Pilot — Results

**Status:** `FOUR_FIT_PILOT_COMPLETE_DIAGNOSTIC_ONLY`

**Decision:** diagnostic only; this pilot does not qualify or replace a model.

**Recipe freeze:** `b4467f22077287c94f18e81f4b5e306307cd9d81646241f487ebf2537d3cab91`

**Execution source commit:** `1069e150563548762ff37df75185d73b35eaf4d0`

**Serialization adapter v2:** `4e72370ed8b6f71f432331a36c1200504b37a1b0f27cdbe7783571549dbd2eec`

## Result

R-NBR observed improved the frozen primary comparison, but its result was mixed across horizons. Primary MAE was **10.392689** on 13,441 rows, a **0.071990** gain versus R06 (10.464680) and a **0.089198** gain versus the within-cycle shuffled control (10.481887). It was also 0.043807 lower than cached R22_ORIGINAL (10.436497).

On the complete E67 H12 horizon (cycles 68–79, 20,903 rows), R-NBR observed MAE was **11.577980**, which is **0.048271 worse** than R06 (11.529709) and 0.066760 worse than cached R22_ORIGINAL (11.511220). It beat the shuffled control by only 0.004046. Therefore the primary gain does not establish a consistent horizon-wide improvement.

| Evaluation view | Rows | R06 MAE | R-NBR observed MAE | Shuffled MAE | Observed minus R06 |
|---|---:|---:|---:|---:|---:|
| Primary: E67 W4 + E75 W6 | 13,441 | 10.464680 | 10.392689 | 10.481887 | −0.071990 |
| H12: E67 cycles 68–79 | 20,903 | 11.529709 | 11.577980 | 11.582025 | +0.048271 |
| E67 W4: cycles 68–71 | 4,950 | 11.396232 | 11.508788 | 11.477601 | +0.112556 |
| E67 W5: cycles 72–75 | 7,462 | 11.477448 | 11.518492 | 11.525104 | +0.041044 |
| E67 W6: cycles 76–79 | 8,491 | 11.653450 | 11.670595 | 11.692925 | +0.017145 |
| E75 W6: cycles 76–79 | 8,491 | 9.921613 | 9.742037 | 9.901416 | −0.179576 |

Missing-stream predictions were unchanged by construction: primary n=868 and H12 n=1,375, with zero MAE change versus R06. On primary subgroups, R-NBR improved over R06 by 0.108801 for `threat_rank >= 95` (n=683) and 0.743565 for `threat_rank <= 5` (n=686). The corresponding H12 gains were 0.031421 (n=1,057) and 0.956944 (n=1,048). These subgroup results are descriptive checks within the authorized Train evaluation.

All predeclared late diagnostic gates evaluated as passing, but the protocol explicitly keeps the pilot diagnostic-only regardless of gate outcome. This result does not change the historical R06 verdict: its archived W1 missing-stream group failure (n=392, +0.395556 MAE harm) remains separate from this late incremental comparison.

## Recovery and fit accounting

The first E67 observed attempt had already persisted all numeric predictions, residuals, and geometry; readback failed because `donor_record_ids` alone used object dtype. We reconstructed that field from the frozen donor row indices and official Regression Train IDs, wrote a new safe-Unicode bundle, and verified the saved residuals and predictions without rebuilding the residual memory. The original NPZ, receipt, and `FIT_FAILED` event remain unchanged. The ledger now has one `FIT_OUTPUT_RECOVERED` event for that same fit ID; no second `FIT_STARTED` or extra fit slot was used.

The first adapter freeze was superseded **before official recovery or any additional fit** after its target-free verifier was found to apply the donor-cycle R06 OOF mapping to query baselines. Adapter v2 uses the frozen origin R06 cache for every query row in that fit horizon, while preserving per-cycle OOF alignment for donor residuals. The original runner, verifier, recipe freeze, gates, and R-NBR formula were not modified.

Four unique authorized fit IDs are complete: one same-fit output recovery without refit plus the three remaining memory fits. The ledger reconciles to cap 70, starts 70, saved 67, failed 3, pending 0. The three failure events include the original E67 readback failure now recovered and two unrelated unresolved historical failures; no failure history was erased or refunded.

| Fit ID | Result | Donor rows | Query rows | Receipt SHA-256 |
|---|---|---:|---:|---|
| `RNBR_PILOT_E67_OBSERVED` | recovered; independently verified; no refit | 16,482 | 20,903 | `df59aee61315094e7887211e2c0d66ded78c3343b23bc6c81fb38ca95639cc96` |
| `RNBR_PILOT_E67_SHUFFLED` | saved and independently verified | 16,482 | 20,903 | `83826b37bf255a03310eaab839427949dd9fe79d075be115465b62105494de83` |
| `RNBR_PILOT_E75_OBSERVED` | saved and independently verified | 28,045 | 8,491 | `755fd9cbc98cc82782ef9a00ef582b5ca65170b19ae5d77dec5d8808c5e80b90` |
| `RNBR_PILOT_E75_SHUFFLED` | saved and independently verified | 28,045 | 8,491 | `39cb85140813ed837fc33d1394455ba633ba7270dd51d7d12f80031931a97e67` |

## Boundaries and QA

- Model donors came from official Regression Train rows only; preprocessing, source caches, and A02 geometry matched the original frozen hashes.
- Query `threat_rank` for cycles 68–79 was read only after all four output bundles were durable and independently verified, for the authorized internal evaluation. Test, SampleSubmission, Kaggle, and full-Train refit were not accessed.
- All four saved fit states passed formula, row/ID/cycle, donor OOF, geometry, receipt, and independent prediction checks. Missing-stream fallback remained unchanged.
- Focused synthetic QA passed: **12 tests** across the recovery adapter and frozen RNBR runner. Coverage includes object-ID input normalization, query-origin versus donor-OOF baseline semantics, pickle-free persistence/resume, corruption rejection, and injected write failure.
- Successor freeze v2 target-free preflight passed. It recorded exactly one object field (`donor_record_ids`), with no target values loaded. Resource guard passed before execution.

## Evidence files

- `PILOT_RESULT.json` — execution status, fit receipts, accounting snapshot, and source boundaries.
- `INDEPENDENT_VERIFICATION.json` — all four bundle checks and primary/H12/window/subgroup metrics.
- `SERIALIZATION_RECOVERY_V2.json` — same-fit recovery lineage and final reconciliation.
- `experiments/optimization_v3/rnbr_pilot_v1/official_freeze_serialization_adapter_02.json` — signed adapter and query/donor baseline semantics.

Final ledger SHA-256: `29ec18718bf0dadae033098c1dec95136bd56d0ec32d9aa5ad13f92cd6a663a9`.
No push was performed.
