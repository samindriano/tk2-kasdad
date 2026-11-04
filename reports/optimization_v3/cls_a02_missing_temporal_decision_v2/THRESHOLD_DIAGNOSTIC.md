# A02 Missing-Stream Threshold Pilot — Diagnostic

**Status:** three frozen selectors completed; exploratory only. No estimator or preprocessing was fit, no new model inference was run, and no Test or Kaggle access has occurred.

## Frozen decision

Only the missing-stream decision varied: `original A02 pA >= threshold`. Present-stream labels remained the current best `MIX50 >= 0.50`. Each threshold maximized overall full-row Macro F1 with the present confusion fixed. Candidate cutoffs, tie rule, support gate, stage order, and inputs are in [protocol.json](../../../experiments/optimization_v3/cls_a02_missing_temporal_decision_v2/protocol.json); the exact thresholds and receipts are in [RESULTS.json](RESULTS.json).

All three selections passed the predeclared minimums (1,000 total rows and 100 per class; 200 missing rows and 30 per class). Actual denominators matched the support recorded before selector outcomes:

| Selection | Rows (class 0 / 1) | Missing rows (class 0 / 1) | Selected threshold |
|---|---:|---:|---:|
| D1: E67 cycles 68–71 | 4,950 (3,097 / 1,853) | 342 (202 / 140) | 0.401662741677 |
| D2: E67 cycles 68–75 | 12,412 (8,185 / 4,227) | 849 (533 / 316) | 0.564553584031 |
| Final stitched H12: E67 cycles 68–75 + E75 cycles 76–79 | 20,903 (14,176 / 6,727) | 1,375 (888 / 487) | 0.532309761295 |

The thresholds differ across origins. D1 and D2 evaluations are adaptive development evidence: W5 is used by D2 selection, and W5/W6 are both used by the final stitched selector. They are not fresh holdouts for the final threshold.

## Temporal evaluation

| Frozen threshold from | Evaluation window | Rows / missing rows | Selected Macro F1 | Current best, missing pA at 0.50 | Δ vs current best | Old MIX50, missing pP at 0.50 (descriptive) | Δ vs old MIX50 |
|---|---|---:|---:|---:|---:|---:|---:|
| D1, 0.401662741677 | E67 W5, cycles 72–75 | 7,462 / 507 | 0.802534 | 0.805095 | −0.002561 | 0.806465 | −0.003931 |
| D2, 0.564553584031 | E75 W6, cycles 76–79 | 8,491 / 526 | 0.818042 | 0.819113 | −0.001071 | 0.816099 | +0.001943 |
| Pooled unique W5 + W6 | 15,953 / 1,033 | 0.810837 | 0.812582 | −0.001745 | 0.811707 | −0.000869 |

Pooled values are recomputed from summed confusion counts across the non-overlapping W5 and W6 rows. Selected pooled confusion is `[[9903, 1176], [1358, 3516]]`; current-best confusion is `[[9940, 1139], [1365, 3509]]`. The change removes 7 false negatives and adds 37 false positives, for 30 more total errors.

On missing rows alone, D1 W5 Macro F1 changed from 0.745412 to 0.705103 (−0.040309); D2 W6 changed from 0.794129 to 0.776919 (−0.017210). In W5, missing false positives rose 84→126 while false negatives fell 39→22. In W6, missing false positives fell 36→31 while false negatives rose 56→66. Neither temporal window improved overall Macro F1 against the current best.

## Cycle and protected subgroup checks

Cycle deltas are selected minus current-best Macro F1; a positive sign is not consistent across cycles.

| Evaluation | Per-cycle Δ Macro F1 |
|---|---|
| E67 W5 | 72: −0.001760; 73: +0.000365; 74: −0.010344; 75: +0.000699 |
| E75 W6 | 76: −0.001605; 77: −0.002087; 78: −0.000764; 79: +0.000173 |

All listed groups passed the protected support gate (at least 200 rows and 30 examples per class). Deltas are selected minus current-best Macro F1.

| Group | E67 W5 support (class 0 / 1), Δ | E75 W6 support (class 0 / 1), Δ |
|---|---:|---:|
| entry_friction 0 | 5,188 (3,694 / 1,494), −0.004825 | 5,738 (4,316 / 1,422), −0.000183 |
| entry_friction 1 | 2,274 (1,394 / 880), +0.001689 | 2,753 (1,675 / 1,078), −0.002500 |
| text_load Q1 | 2,116 (1,376 / 740), −0.008581 | 2,244 (1,463 / 781), −0.003032 |
| text_load Q2 | 1,696 (1,111 / 585), −0.000436 | 2,285 (1,674 / 611), −0.000172 |
| text_load Q3 | 1,632 (1,248 / 384), −0.000713 | 1,969 (1,526 / 443), +0.000000 |
| text_load Q4 | 2,018 (1,353 / 665), −0.000053 | 1,993 (1,328 / 665), +0.000000 |

Most supported subgroup comparisons are negative or effectively unchanged. The small positive entry_friction-1 result in W5 does not offset the overall or pooled decline; the earlier text-load Q4 guard is effectively flat in both windows.

## Final stitched selection and interpretation

The final selector chose `0.532309761295` with Macro F1 `0.821941` on the same 20,903 adaptive selection rows. At fixed 0.50, those same rows score `0.821572` for the current best and `0.821519` for old MIX50. The +0.000369 same-row contrast is selection-set performance, not an independent estimate of generalization. Probabilities did not change, so this pilot makes no Brier-score or log-loss claim.

**Decision:** the temporal evidence does not support replacing the current missing-stream threshold as a qualified improvement. The authorized one-CSV E79 application remains an exploratory benchmark only. It must keep every present-stream label identical to current best, disclose these negative internal results, and must not change the champion or be retuned from a public score.

## Provenance and accounting

- Approved prefit commit: `4a9cf4cfbc0b3030473733507841238d958ad92e`; V2 source-pass receipt is committed in the current ancestry.
- Train SHA-256: `875552a68a3d24b0f3e1966d6ddb5671e6f54305a1c526e2f0a76747942ed91c`.
- OOF probability bundle SHA-256: `25d1d2b91348a57d33b3b0e5ce78da29840d115b75e24f9502cc83c1b63ef351`.
- Three reservations, three saved threshold receipts, zero retries; resource gate passed each time (at least 4 GiB available, 2 GiB safety reserve, one project job, two threads).
- Selector receipt SHA-256: D1 `b1e39b9f2843273350de206aae6823588037fc4590ed8f46151566c9bca205c2`; D2 `b94588592e710b0ef04f14987e44b900bec1376e852b44049404a70b85a8b73e`; final `a54b24cab82102d9ff4652dbc19d76291258c3ae12d2c069e7fd3a275e2e060c`.
- Evaluation receipt SHA-256: E67 W5 `57c3ac5741041db058343105a0fd8723006e40af25498d9e83c93cf9c4d79a48`; E75 W6 `610c5a4aa2f3cb886b2eebeae1ba896fba115f8870456453ba3abf7d32c0cce4`.
- Original five-input E79 application freeze remains byte-identical (SHA-256 `b833b5d078a233ae17dcb7d14fecd451cdeac3cd8f63d0b94e9db7a8838cd493`). A versioned successor at `E79_APPLICATION_FREEZE_V2.json` adds the two prefit-pinned Train/OOF hashes required by the seven-input builder guard (SHA-256 `b6d4bee60015bd47eef1e10ceefe0e310e7c3cac804ebfe035bcc60766f1f017`); threshold, final receipt, source commit, and recipe are unchanged.
- The first builder attempt failed closed at the seven-hash equality check before loading the E79 NPZ or parsing Test rows. No E79 probability array or Test field was read in that attempt.
- Run manifest is retained in ignored runtime output `experiments/runs/optimization_v3_cls_a02_missing_temporal_decision_v2/RUN_MANIFEST.json` (SHA-256 `9efe0ddc00dfbfbea1a67f122efecfb390936e2d03e9877e09880b50e459b383`).
- No Test fields, E79 probability bundle, SampleSubmission, or Kaggle endpoint were accessed for this report.
