# CLS simple present-threshold — independent post-fit findings

Status: independent replay PASS. Exactly two authorized diagnostic calibrator fits (D1, D2) and their two score windows were completed. No predictor or preprocessing fit, Test inference, CSV, Kaggle access, or submission occurred. This report evaluates the frozen pilot and does not promote or replace a champion.

## Lineage and independent audit

- Root release: commit 2b06b81672befa926352d6682f60fe95f68b01f4; release SHA256 00a316262db48f841d279d6c7914ca488710e598065adcf3cb8be73c31c0f1f8.
- Frozen producer source commit: aa0bf8a7cb2a7dfec3e41f557921bce31f88f09b.
- Official Train SHA256: 875552a68a3d24b0f3e1966d6ddb5671e6f54305a1c526e2f0a76747942ed91c.
- OOF cache SHA256: 25d1d2b91348a57d33b3b0e5ce78da29840d115b75e24f9502cc83c1b63ef351.
- Ledger SHA256: 96f72ca52b2bf3fc161fdf6913c24c5df9e38eebf13e046462751dba92639460.
- Independent audit replayed both exact threshold optima, fit confusion/support, frozen prediction IDs and values, full/present/cycle/subgroup metrics, pooled metrics, gates, and fit/score ledger accounting: 20 checks passed.
- It read labels only for the frozen Train IDs in each fit and score population. It did not open OOF target members, Test, or SampleSubmission.

## Calibration fits and score windows

| Stage | Fit population | Threshold | Fit Macro F1 | Present fit support (class 0 / 1) | Score population |
|---|---|---:|---:|---:|---|
| D1 | E67 cycles 68–71, 4,950 rows | 0.4924878024998012 | 0.847260815 | 2,895 / 1,713 | E67 cycles 72–75, 7,462 rows |
| D2 | E67 cycles 68–75, 12,412 rows | 0.4854367183801688 | 0.825069203 | 7,652 / 3,911 | E75 cycles 76–79, 8,491 rows |

Both thresholds are inside [0.45, 0.55]; neither is at a permitted interval bound. Their absolute difference is 0.0070510841. The missing-stream decision stayed fixed and unchanged in every score window.

### Full-row and present-row results

| Window / view | Candidate Macro F1 | Fixed pA at 0.50 | Gain | Candidate precision / recall | Complex blend at 0.50 |
|---|---:|---:|---:|---:|---:|
| D1 full | 0.806689260 | 0.806328575 | +0.000360685 | 0.742587 / 0.727885 | 0.806246025 |
| D1 present | 0.810008592 | 0.809612026 | +0.000396566 | 0.752008 / 0.724295 | 0.809530679 |
| D2 full | 0.820558154 | 0.819373915 | +0.001184239 | 0.771650 / 0.716400 | 0.818884188 |
| D2 present | 0.822595419 | 0.821339078 | +0.001256340 | 0.771101 / 0.721769 | 0.820812344 |

The threshold change made 30 D1 and 52 D2 score-window decisions differ from the fixed pA threshold. On pooled rows, the candidate has 40 more true positives and 40 fewer false negatives, with 42 more false positives and 42 fewer true negatives. Pooled precision falls while recall rises, so the small Macro F1 gain reflects a boundary tradeoff.

Pooled across the 15,953 disjoint score IDs, candidate full-row Macro F1 is 0.814079156, versus 0.813299108 for fixed pA (gain +0.000780048). Candidate precision/recall are 0.757100/0.721994; fixed pA gives 0.761936/0.713787. Present-only pooled Macro F1 is 0.816723415 versus 0.815885077 (gain +0.000838338). The secondary complex blend reaches 0.813018496 full-row pooled Macro F1.

### Cycle-level negative results

Per-cycle full-row Macro F1 gains versus fixed pA are:

- D1: cycle 72 +0.000115553; 73 +0.001973251; 74 −0.000060426; 75 −0.000764869.
- D2: cycle 76 +0.001256967; 77 +0.003325638; 78 −0.003167536; 79 +0.002747720.

These are descriptive within-window variations; the locked gate is evaluated on each full window and the specified subgroup guards.

### Subgroup results and guards

All 16 subgroup/window combinations were supported under n >= 200 and at least 30 rows per class; there are no UNKNOWN groups. Every subgroup's positive-recall change is nonnegative. Negative Macro F1 deltas, all within the allowed 0.01 harm, were:

- D1 entry_friction=1: −0.001356723; text_load Q1: −0.000804224; text_load Q3: −0.000531012.
- D2 entry_friction=1: −0.000801959; text_load Q3: −0.001375060.

Availability=missing is unchanged by design. The largest supported subgroup positive-recall gain is +0.014084507 (D2 text_load Q1); all are within the allowed recall guard.

## Frozen gates and interpretation

- Qualification gate: PASS (pooled gain +0.000780048 >= 0.0003; both window gains are positive; all supported subgroup guards pass).
- Exploratory-review gate: PASS (pooled gain and both window gains are nonnegative; subgroup guards pass).
- These are two stage-specific temporal diagnostics, not one untouched fixed-model holdout. D1 score cycles 72–75 also belong to D2's later fit population, so the pooled gate is a prespecified summary across distinct windows, not an independent estimate of generalization.
- The results support only the bounded threshold-calibration pilot on the declared OOF/Train windows. They do not establish Test performance, public leaderboard performance, or a champion change. D1/D2 thresholds differ modestly, and several cycle/subgroup rows are slightly negative despite both aggregate windows clearing the gates.
- The observed POS/A02 public champion remains unchanged. No final fit, E79, Test inference, Kaggle CSV, or upload was authorized by this release.

## Receipts

- Runtime fit/score files: experiments/runs/cls_simple_present_threshold_v1/
- Independent machine-readable post-fit audit: POSTFIT_INDEPENDENT_AUDIT.json
- Independent audit script: scripts/audit_cls_simple_present_threshold_v1_postfit.py
