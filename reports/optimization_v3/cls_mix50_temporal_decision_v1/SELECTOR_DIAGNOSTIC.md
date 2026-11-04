# MIX50 Temporal Decision — Diagnostic Results (R5)

## Scope and rule

This diagnostic evaluates a single present-stream threshold for the frozen MIX50 policy using saved Train OOF probabilities. Missing-stream decisions remain exactly the historical MIX50 rule (`pP >= 0.50`). No predictive estimator or preprocessing was fit. All target labels used below came from the official Train file after the matching R5 root approval; Test targets were never loaded.

Selection objective was overall binary Macro F1 with labels `[0, 1]` and `zero_division=0`. Candidate thresholds preserve tied probability blocks; ties within `1e-12` choose closest to 0.50, then lower. Selection support required at least 1,000 rows and 100 examples per class. Each selected/evaluated receipt binds IDs, cycles, probabilities, missing mask, source hashes, and reservation IDs.

## Threshold selections

| Stage | Selection cycles | Rows (class 0 / 1) | Selected present cutoff |
|---|---:|---:|---:|
| D1 | 68–71 | 4,950 (3,097 / 1,853) | 0.503854986 |
| D2 | 68–75 | 12,412 (8,185 / 4,227) | 0.501116544 |
| Final OOF | 68–79 | 20,903 (14,176 / 6,727) | 0.474054103 |

D1 selected on E67 cycles 68–71 and evaluated on E67 cycles 72–75. D2 selected on E67 cycles 68–75, then evaluated on E75 cycles 76–79; D2 is cross-origin transfer. The final cutoff uses the unique 20,903-row OOF set (E67 cycles 68–75 plus E75 cycles 76–79), excluding duplicate E67 W6 IDs. The final selection score is an in-sample selection objective and is not treated as validation evidence.

## Temporal evaluation versus fixed 0.50

| Window | Eval rows | Cutoff | Macro F1 at 0.50 | Macro F1 selected | Δ Macro F1 | FP (0.50 → selected) | FN (0.50 → selected) |
|---|---:|---:|---:|---:|---:|---:|---:|
| W5, cycles 72–75 | 7,462 | 0.503854986 | 0.806465 | 0.806069 | -0.000397 | 618 → 613 | 633 → 639 |
| W6, cycles 76–79 | 8,491 | 0.501116544 | 0.816099 | 0.815499 | -0.000600 | 528 → 526 | 737 → 742 |

Unique pooled W5+W6 support is 15,953 rows. Macro F1 is 0.811707 at 0.50 and 0.811202 with the stage-selected cutoffs (Δ -0.000505). Confusion counts: fixed FP/FN 1146/1370; selected 1139/1381 (four more total errors). These are pooled unique rows, not an average of window F1 values.

## Per-cycle and subgroup changes

Δ below is selected cutoff minus fixed 0.50 Macro F1. Every shown group met the frozen support gate; missing-stream decisions are fixed, so their Δ is zero.

| Slice | Group | W5 rows | W5 Δ | W6 rows | W6 Δ |
|---|---|---:|---:|---:|---:|
| Cycle 72 / 76 | — | 1,811 | -0.000806 | 1,822 | -0.000718 |
| Cycle 73 / 77 | — | 1,880 | -0.000176 | 2,335 | -0.001472 |
| Cycle 74 / 78 | — | 1,814 | +0.000974 | 2,146 | -0.000767 |
| Cycle 75 / 79 | — | 1,957 | -0.001475 | 2,188 | +0.000465 |
| Availability | missing | 507 | +0.000000 | 526 | +0.000000 |
| Availability | present | 6,955 | -0.000435 | 7,965 | -0.000637 |
| entry_friction | 0 | 5,188 | -0.000560 | 5,738 | +0.000099 |
| entry_friction | 1 | 2,274 | -0.000094 | 2,753 | -0.001666 |
| text_load | Q1 | 2,116 | +0.000000 | 2,244 | -0.001081 |
| text_load | Q2 | 1,696 | -0.000145 | 2,285 | -0.000647 |
| text_load | Q3 | 1,632 | -0.000637 | 1,969 | +0.001164 |
| text_load | Q4 | 2,018 | -0.001130 | 1,993 | -0.001413 |

Of eight individual cycles, two improve and six decline. Subgroup differences are small and mixed: present-stream groups decline in both windows; missing groups remain unchanged by design. For text_load Q4, Δ is negative in both windows (−0.001130 W5; −0.001413 W6); W6 Q3 improves by +0.001164. Entry-friction=1 declines by −0.001666 in W6, while entry-friction=0 improves by only +0.000099. No broad or uniform subgroup gain appears.

## Interpretation and decision

The two past-only temporal selectors chose cutoffs only slightly above 0.50, yet each underperformed the fixed threshold on its later window. The pooled difference is −0.000505 Macro F1, with four extra total errors. The final all-OOF selector chose 0.474054, but that cutoff is fitted to the same 20,903 OOF rows and has no independent temporal validation in this run. Treat the resulting E79 CSV as an experimental candidate only; these results do not support changing MIX50’s threshold as a qualified policy or champion.

## E79 artifact QA

- Candidate CSV: `submissions/optimization_v3/cls_mix50_temporal_decision_v1/classification_mix50_temporal_decision_v1.csv`; SHA-256 `8bedcfb9b9c3396b076d37d0640bfc04ebe2ac968531c6365fe787bf11224276`.
- 29,421 rows, unique IDs, exact SampleSubmission ID order, columns `record_id, artifact_signal`; integer labels only.
- Independent duplicate label-vector scan: 0 matches; none found.
- Relative to current MIX50, 251 labels change (251 0→1; 0 1→0); Test targets not loaded.
- Status: prepared locally, not submitted. No Kaggle score exists for this candidate.

## Provenance

- Prefit source commit: `aebf04025ff5fe64ef4209ebbc2d46762fef15d4`; current source/approval HEAD at execution: `8c378ff6d840f58f846a8ff34e50859e5e5beec2`.
- Prefit freeze file/receipt SHA-256: `460fe62687b4f8ba733ff5e2e895ef23cf179957034c6d0a9a6bf3c3ce47547a` / `81eb55f73ffce159b39cd1c80cbf442c4f176019cd91d0d2a8f1b21feb8ce5cd`.
- Selector results receipt: `0eaba0ef1ef8f4420f0ae39e96608e301bc8631596f304fa5ba2fc0a8227a243`; final threshold receipt: `7880871da82151f574eab70a289db5d8b83d42671ad80c37cde37d6ba03b4cb8`.
- E79 application freeze receipt: `1ea6c3c151d208c5443211249320f8c583ebdfb1679a0197fd3116863d9ee7dc`; prediction bundle SHA-256: `ffc55aa2d18b75438257f3cf0aa205baa3f4c2922180e2b3688b36c352c00271`.
- Independent CSV QA receipt: `598828cfbbdedb7f622e247acabdbb327050a8b312ef65c0c86c6d83cae67438` (written with O_EXCL).
