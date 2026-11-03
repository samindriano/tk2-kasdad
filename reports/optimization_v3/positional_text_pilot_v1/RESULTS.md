# POSITIONAL_TEXT_PILOT_V1 — execution results

## Decision

**Final pilot status: FAIL_PILOT_GATES at both origins.** The frozen three-third positional recipe is retired under its preregistered rule. No rescue search, threshold tuning, bin/seed/C/vocabulary/block-weight changes, or continuation is authorized by these results. The missing-stream C1_B fallback and all integrity checks completed successfully; they do not offset the failed primary gates.

This is a controlled mechanism pilot, not a full-six qualification, champion comparison, or claim that every possible positional representation is ineffective. E75 is a history-sensitivity check whose W6 labels overlap E67 W6, not an independent replication.

## Frozen question and data boundary

The pilot asked whether coarse first/middle/final-third unigram position information improves temporal Classification Macro F1 beyond (1) faithful unordered C2_B and (2) a deterministic order-shuffled positional control. All three LR arms used the shared fit-only C2_B structured state, scaler, OHE, TF-IDF vocabulary, feature order, and threshold 0.50. Empty normalized `signal_stream` rows route to the same C1_B structured HGB expert for every arm. No model or threshold selection used Test or SampleSubmission.

Official Classification Train contained 50,608 rows on cycles 47–79; SHA-256 `875552a68a3d24b0f3e1966d6ddb5671e6f54305a1c526e2f0a76747942ed91c`.

| Origin | Fit cycles / rows | Score cycles / rows | Present stream | Missing or blank stream |
|---|---:|---:|---:|---:|
| E67 | 47–67 / 29,705 | 68–79 / 20,903 | 19,528 | 1,375 |
| E75 | 47–75 / 42,117 | 76–79 / 8,491 | 7,965 | 526 |

Score labels for each origin were opened only after both prediction bundles were durable and passed independent pre-label verification. The score code read E67 cycles 68–79 and E75 cycles 76–79 only.

## Primary results

Macro F1 is the primary metric. `Δ bag` and `Δ shuffle` compare the position arm with each same-view control. Positive recall is included as a descriptive metric; it does not override Macro F1 gates.

| Origin / view | n | Bag Macro F1 | Position Macro F1 | Δ bag | Shuffled Macro F1 | Δ shuffle | Position positive recall |
|---|---:|---:|---:|---:|---:|---:|---:|
| E67 W4 | 4,950 | 0.842554 | 0.838262 | −0.004292 | 0.836535 | +0.001727 | 0.806800 |
| E67 W5 | 7,462 | 0.799093 | 0.799238 | +0.000146 | 0.798449 | +0.000789 | 0.738837 |
| E67 W6 | 8,491 | 0.784403 | 0.789240 | +0.004837 | 0.776322 | +0.012918 | 0.726000 |
| E67 H12_E67 | 20,903 | 0.805106 | 0.805996 | **+0.000889** | 0.800018 | +0.005978 | 0.752787 |
| E75 W6 | 8,491 | 0.810384 | 0.810138 | **−0.000246** | 0.808658 | +0.001479 | 0.698000 |

The H12_E67 position gain versus bag was below the required `+0.003`, so E67 failed its primary gate even though position cleared the threshold against shuffled control. E75 position was `0.000246` below bag, violating its no-regression requirement, while it remained above shuffled control. All fixed gates and descriptive metric rows are in [RESULTS.json](RESULTS.json).

## Per-cycle direction and variation

The table gives position Macro F1 minus each control for every scored cycle. It makes clear that the pooled result is not uniform across time.

| Cycle | n | E67 Δ bag | E67 Δ shuffle | Cycle | n | E75 Δ bag | E75 Δ shuffle |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 68 | 577 | −0.00787 | −0.00272 | 76 | 1,822 | +0.00163 | −0.00072 |
| 69 | 1,079 | −0.01192 | +0.00373 | 77 | 2,335 | −0.00673 | −0.00176 |
| 70 | 1,605 | −0.00393 | −0.00812 | 78 | 2,146 | +0.00511 | +0.00226 |
| 71 | 1,689 | +0.00077 | +0.01066 | 79 | 2,188 | +0.00013 | +0.00600 |
| 72 | 1,811 | −0.01117 | −0.00950 |  |  |  |  |
| 73 | 1,880 | +0.00543 | +0.00964 |  |  |  |  |
| 74 | 1,814 | +0.00511 | +0.00073 |  |  |  |  |
| 75 | 1,957 | +0.00184 | +0.00285 |  |  |  |  |
| 76 | 1,822 | −0.00367 | +0.00264 |  |  |  |  |
| 77 | 2,335 | +0.00055 | +0.01030 |  |  |  |  |
| 78 | 2,146 | +0.00542 | +0.01846 |  |  |  |  |
| 79 | 2,188 | +0.01578 | +0.01950 |  |  |  |  |

Position exceeded bag on 7/12 E67 cycles and exceeded shuffled position on 9/12. At E75 it exceeded bag on 3/4 cycles and shuffled position on 2/4. The origins use different training history but overlapping W6 labels, so these signs are descriptive rather than independent replication.

## Protected subgroup and support checks

All deltas below are position Macro F1 minus the named control. `Present` and `missing/blank` are the two exact routing groups; the missing group is identical across arms because it uses the shared HGB fallback. `EF` is `entry_friction`. Q4 is the fit-defined text-load fourth quartile. The full JSON retains precision, recall, confusion counts, ROC AUC, average precision, log loss, Brier score, all group denominators, and each gate.

### Stream availability

| Origin / view | Present n | Δ bag | Δ shuffle | Missing/blank n | Missing-arm delta |
|---|---:|---:|---:|---:|---:|
| E67 W4 | 4,608 | −0.004627 | +0.001864 | 342 | 0 |
| E67 W5 | 6,955 | +0.000194 | +0.000862 | 507 | 0 |
| E67 W6 | 7,965 | +0.005205 | +0.013855 | 526 | 0 |
| E67 H12_E67 | 19,528 | +0.000971 | +0.006441 | 1,375 | 0 |
| E75 W6 | 7,965 | −0.000287 | +0.001565 | 526 | 0 |

### Entry friction

Each row reports support and position Macro F1 differences against both controls; all cells are included to show subgroup sign changes.

| Origin / view | EF | n | Δ bag | Δ shuffle |
|---|---:|---:|---:|---:|
| E67 W4 | 0 | 3,418 | −0.002967 | +0.002567 |
| E67 W4 | 1 | 1,532 | −0.007128 | −0.000204 |
| E67 W5 | 0 | 5,188 | −0.001268 | −0.001234 |
| E67 W5 | 1 | 2,274 | +0.002332 | +0.004190 |
| E67 W6 | 0 | 5,738 | +0.003033 | +0.013863 |
| E67 W6 | 1 | 2,753 | +0.007792 | +0.011674 |
| E67 H12_E67 | 0 | 14,344 | +0.000015 | +0.005732 |
| E67 H12_E67 | 1 | 6,559 | +0.002258 | +0.006247 |
| E75 W6 | 0 | 5,738 | +0.001043 | +0.002969 |
| E75 W6 | 1 | 2,753 | −0.002160 | −0.000975 |

### Fit-defined text-load quartile

Q4 thresholds are computed from the respective fit prefix: 36.1952 at E67 and 36.6066 at E75. All declared Q4 views met the fixed support rule (`n ≥ 200`, at least 30 examples in each class). Every Q4 gate passed; this does not repair either failed primary gate.

| Origin / view | Group | n | Δ bag | Δ shuffle | Positive-recall Δ bag |
|---|---|---:|---:|---:|---:|
| E67 W4 | Q4 | 1,765 | −0.001757 | +0.004173 | +0.005882 |
| E67 W4 | Below Q4 | 3,185 | −0.005671 | +0.000342 | −0.001705 |
| E67 W4 | Missing text load | 0 | n/a | n/a | n/a |
| E67 W5 | Q4 | 2,018 | −0.004707 | −0.003247 | +0.013534 |
| E67 W5 | Below Q4 | 5,444 | +0.002060 | +0.002235 | +0.012288 |
| E67 W5 | Missing text load | 0 | n/a | n/a | n/a |
| E67 W6 | Q4 | 2,080 | −0.001232 | +0.010344 | +0.010101 |
| E67 W6 | Below Q4 | 6,411 | +0.006952 | +0.013822 | +0.006641 |
| E67 W6 | Missing text load | 0 | n/a | n/a | n/a |
| E67 H12_E67 | Q4 | 5,863 | −0.002739 | +0.003920 | +0.009814 |
| E67 H12_E67 | Below Q4 | 15,040 | +0.002382 | +0.006795 | +0.006611 |
| E67 H12_E67 | Missing text load | 0 | n/a | n/a | n/a |
| E75 W6 | Q4 | 1,993 | −0.005332 | −0.005911 | +0.003008 |
| E75 W6 | Below Q4 | 6,498 | +0.001410 | +0.003793 | +0.009809 |
| E75 W6 | Missing text load | 0 | n/a | n/a | n/a |

E67 Q4 position-vs-control Macro F1 harms were no worse than 0.004707 and positive recall was not harmed against either control. E75 Q4 showed a `−0.005332` Macro F1 difference versus bag and `−0.005911` versus shuffle, within the frozen `−0.01` limit; its recall differences were +0.003008 vs bag and −0.010526 vs shuffle, within the frozen `−0.02` limit. Below-Q4 and entry-friction direction varied by view, so the evidence does not support a uniform positional benefit.

## Execution accounting and QA

- Exactly **18/18** pilot units were saved: 10 shared preprocessing units and 8 predictor fits. There were no pilot retries, duplicate fits, or pending units. All eight predictor receipts report `PASS_NO_CONVERGENCE_WARNING`.
- Total recorded fit elapsed time was 66.05 seconds. The highest before/after receipt RSS sample was 516,370,432 bytes (about 492 MiB), under the 6 GiB policy ceiling. Receipts sample process RSS around each unit; this is not a continuous operating-system high-water measurement.
- The pilot additions changed live budget accounting from 87 starts / 84 saved / 3 historical failures to 105 starts / 102 saved / 3 historical failures / 0 pending. The 3 failures predate this pilot. Global cap is 105; pilot cap is 64; all other category caps remain unchanged.
- Both prediction bundles passed hash and independent pre-label integrity verification. Independent metric/gate recomputation passed for both origins. It rechecked 3 aggregate metrics, subgroup and per-cycle rows, and all gate predicates per origin.
- Post-fit QA: `python -m unittest tests.test_optimization_v3_positional_text_pilot_v1 -v` — **4/4 passed**.
- Runtime outputs remain in the ignored run directory. [FIT_ACCOUNTING.json](FIT_ACCOUNTING.json) preserves all 18 fit IDs, populations, artifact and receipt hashes, convergence status, timings, and memory samples. [RESULTS.json](RESULTS.json) contains the full per-view, per-cycle, subgroup, gate, and independent-verification records.

## Provenance and integrity note

- Source commit: `3429fb93c2a579ca91f0dfe7fc1a67d06a1b8855`.
- Frozen recipe SHA-256: `179b746cabdd9e3a8b522ce346dc84978a17886709cca08ad92fecc04e381592`.
- Freeze signature: `f310d0340ea4402677ca9e86aa29c43b1536ded902c0f6a6f52cb53bd333c292`.
- E67 matrix manifest: `d8d672c26771944bc2bb6eee907e9bef903959fbc62d9f5df3aaab7f7cb28028`; prediction bundle: `2f22c68aca959baad6cd4244e05f96b702b4b89dfdc83f82c8edde6cf465d444`.
- E75 matrix manifest: `67ec3f9bd376f5f4dad2afbd5e13ee5fd8316126fa3894ef676fa40077166fc5`; prediction bundle: `83e5beb9b10be4d395df4a2292cb464625f356e5e77a76a232b92f007c1b1306`.
- Final live ledger SHA-256: `59ef4f3c91091cf0f81d71fdb95211992ff796672227f764bc9523e7304ed6a1`.
- The earlier append-only ledger integrity incident remains documented in [INTEGRITY_INCIDENT.md](INTEGRITY_INCIDENT.md). No further ledger rewrite or truncation occurred during this run.

No Test/SampleSubmission access, Kaggle submission, notebook edit, push, or follow-on experiment was performed. The required stop point is reached: results are ready for review.
