# MIX50 missing-stream fallback — zero-fit diagnostic

**Decision:** retain the current MIX50 fallback (`pP` on missing streams). The fixed alternative improves proper probability scores in every reported view, but its Macro F1 effect changes with the OOF origin: it is negative across E67 and positive on E75 W6. The unique pooled contrast is effectively flat (+0.000053), and the missing-only Macro F1 changes direction sharply between E67 and E75. This is not stable evidence to change the fallback.

## Fixed question and boundary

Does the already available original A02 probability improve current MIX50 if it is used only for missing-stream rows while present-stream probabilities remain bit-exact?

The single recomposition was fixed as:

```text
candidate = where(missing, original_A02_pA, frozen_MIX50)
prediction = probability >= 0.50
```

No estimator or preprocessing was fit. The target, cycle and `signal_stream` availability were read from official Classification Train only, joined to saved OOF rows by `record_id` with cycle equality checked. The stream-missing rule is `signal_stream.strip() == ""`. The saved OOF bundle formula and hashes were verified, and all five previously published MIX50 view metrics were recomputed exactly against `POLICY_METRICS.json` before interpreting the candidate.

E67 and E75 are overlapping adaptive development views, not independent holdouts. E67 and E75 W6 use the same 8,491 Train IDs under different origin-specific predictions. The pooled unique contrast uses E67 cycles 68–75 plus E75 cycles 76–79, with 20,903 rows and no duplicated W6 IDs.

## Full-row results

All Macro F1 values use the full view denominator at the fixed 0.50 threshold. Confusion counts are ordered `[[TN, FP], [FN, TP]]`.

| View | Rows | Missing | MIX50 Macro F1 | Candidate Macro F1 | Δ Macro F1 | MIX50 → candidate FP | MIX50 → candidate FN | Δ Brier | Δ log loss |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| E67 H12 | 20,903 | 1,375 | 0.811937 | 0.810552 | −0.001384 | 1,759 → 1,766 | 1,683 → 1,700 | −0.000332 | −0.000945 |
| E67 W4 | 4,950 | 342 | 0.846365 | 0.844109 | −0.002256 | 360 → 362 | 353 → 361 | −0.000002 | −0.000294 |
| E67 W5 | 7,462 | 507 | 0.806465 | 0.805095 | −0.001370 | 618 → 615 | 633 → 643 | −0.000436 | −0.001225 |
| E67 W6 | 8,491 | 526 | 0.792531 | 0.791759 | −0.000772 | 781 → 789 | 697 → 696 | −0.000433 | −0.001077 |
| E75 W6 | 8,491 | 526 | 0.816099 | 0.819113 | +0.003014 | 528 → 524 | 737 → 722 | −0.000638 | −0.001815 |
| Unique pooled | 20,903 | 1,375 | 0.821519 | 0.821572 | +0.000053 | 1,506 → 1,501 | 1,723 → 1,726 | −0.000416 | −0.001245 |

The A02 fallback has better Brier score and log loss on every view, while the thresholded Macro F1 is origin-sensitive. On the E67 H12 missing subset, candidate probabilities change 216 decisions; 68 false positives and 52 false negatives are added while 61 false positives and 35 false negatives are removed. Its missing-only Macro F1 falls from 0.763347 to 0.744342 even as Brier improves from 0.159748 to 0.154703 and log loss improves from 0.477035 to 0.462674. These probability-score improvements do not establish better fixed-threshold class decisions.

The per-cycle direction is similarly unstable: E67 is positive in 3/12 cycles and negative in 9/12; E75 W6 is positive in 3/4 and negative in 1/4. These counts are sensitivity summaries, not independent observations. Full cycle deltas, row counts and missing support are in `per_cycle.csv`.

## Supported subgroup checks

The report uses the frozen minimum support rule of at least 200 rows and at least 30 rows in each class. All supported groups, including negative deltas, and all unsupported groups marked `UNKNOWN_UNSUPPORTED` are recorded in `supported_groups.csv` and `RESULTS.json`.

There are eight supported groups in each view. Within E67 H12, 6 are negative, 1 is positive and 1 is unchanged; the missing-stream subgroup falls by 0.019005 Macro F1 and 0.034908 positive recall. In E75 W6, 6 groups are positive and the missing-stream subgroup rises by 0.047484 Macro F1. In the unique pooled view, 4 groups are negative, 3 are positive and 1 is unchanged. The sign changes reinforce that the apparent late-origin gain does not transfer consistently to E67.

## Relation to earlier fallback work

This exact composite was not scored in the scoped historical reports, scripts and experiment records. The neighboring results are not direct controls:

- C01 routed sparse25 present predictions with C1_B missing probabilities and was compared with sparse25 (0.829944 → 0.827770 full-six Macro F1; `reports/optimization_v3/batch1/C01/summary.md`).
- C20 kept its CatBoost missing route and used an inherited q80 present route; it was also compared with sparse25 (0.829944 → 0.828371; `reports/optimization_v3/batch1/C20/summary.md`).
- The V2 calibration work used the earlier C2_B-present plus CatBoost-missing route, not current MIX50 (`reports/optimization_v3/cls_reopen_v1/luna1/CLS_REOPEN_DECISION_AUDIT.md`).
- The MIX50 threshold selector held its missing decision at current `pP`, so it did not test this fallback change (`reports/optimization_v3/cls_mix50_temporal_decision_v1/SELECTOR_DIAGNOSTIC.md`).

Thus the result here is the first located row-matched score for A02-on-missing plus current MIX50-on-present, and it does not overturn the earlier experiments' verdicts.

## Recommendation and reproduction

Keep `pP` as the current missing-stream fallback. The E67 full-row score declines, E75 W6 improves, and the unique pooled Macro F1 gain is negligible. No threshold or origin-specific rescue was evaluated or is recommended from these same opened development outcomes.

Reproduce the audit and focused synthetic checks from the repository root:

```powershell
python -m pytest -q tests/test_optimization_v3_cls_missing_current_mix50_audit_v1.py
python scripts/optimization_v3_cls_missing_current_mix50_audit_v1.py
```

Outputs: `RESULTS.json`, `views.csv`, `missing_branch.csv`, `per_cycle.csv`, `supported_groups.csv`, and `QA.json`. `QA.json` records the Train and OOF SHA-256 checks, baseline cross-check, ID/cycle/availability alignment, exact present and missing probability identities, zero new fits, and no Test access.
