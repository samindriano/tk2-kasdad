# Center-local correction R1 — diagnostic result

Decision: `EXPLORATORY_REVIEW_ELIGIBLE`; the mechanism gate passes, the
candidate qualification gate fails, and the exploratory review gate passes.
This is adaptive post-discovery evidence, not a fresh holdout result. Preserve
the earlier residual memo's no-follow-up recommendation and its counterevidence.

The signed root callback authorized exactly four calibrator fits and the
separate score. Four fits completed, saved predictions were frozen before score
targets were loaded, and the independent verifier passed. No base-model or
preprocessing fit was started. No Test/Sample, E79, CSV, or Kaggle access occurred.

## Frozen identity and verification

- Source HEAD used for fitting, prediction freeze, scoring, and independent
  verification: `0db77c67e2ab9e1153128cdd72724162c630809b`.
- `ROOT_SOURCE_PASS_R1.json` SHA-256:
  `3b5cd6684f2eb084bada74f384ff0d1d44019278c2d8c9128443ae7a28c284ec`.
- `PREFIT_FREEZE_R1.json` SHA-256:
  `9f488498c44d8e308e739050cf7c5d2411b2af74428f60ef7b5d2e27ba1807ba`.
- `IDENTITY_PREFLIGHT_R1.json` SHA-256:
  `bbfd2935d91a23722f0ba79dbfb0013a72625c68676a232500532b65dffaf04b`.
- `PREDICTION_FREEZE_R1.json` SHA-256:
  `fe3ec5ae9ec66d090f7aa5dd98012099707cddbfe3cd1efd0bbdb4917d68398e`.
- `EVALUATION_R1.json` SHA-256:
  `dfae132a2216677547c0f7c06a4924519a9d03f232972064106b72910369d89d`.
- `INDEPENDENT_VERIFICATION_R1.json` SHA-256:
  `a5c856da5aa4caed3e15d65bd3274955ade93569cadc571c034b446702ec6d4b`.
- Independent verification ran while HEAD still matched the frozen source commit.
  Future source verification requires that frozen checkout.

The freeze binds 34 source, reference, input, and novelty file hashes. The
complete inventory and six-document novelty hashes are in
[`SOURCE_READY_R1.md`](SOURCE_READY_R1.md). Fit and score target hashes are
recorded in the fit states, evaluation, and independent verification receipt.

## Fits

The LOCAL optimum is the lower endpoint of the exact bounded weighted-median
solution interval. GLOBAL_CONTROL uses the lower empirical median on the same
present rows. Objectives use the saved HINGE formula without clipping.

| Fit ID | Present rows | Active LOCAL rows | Delta | Bounded optimum interval | Unclipped absolute objective | Minimum derivative |
|---|---:|---:|---:|---:|---:|---:|
| D1_LOCAL | 4,608 | 970 | -0.5144618597605731 | [-0.5144618597605731, -0.5144618597605731] | 51,361.12265326972 | 0.9495400653646829 |
| D1_GLOBAL_CONTROL | 4,608 | — | -0.03211422123678176 | [-0.03211422123678176, -0.015764593246700542] | 51,363.35861958489 | 0.9495400653646829 |
| D2_LOCAL | 11,563 | 2,476 | -1.9644215654237005 | [-1.9644215654237005, -1.9644215654237005] | 129,259.46214528003 | 0.8039491832577933 |
| D2_GLOBAL_CONTROL | 11,563 | — | 0.0 | [0.0, 0.0] | 129,314.13656338603 | 0.9544062282511625 |

Both LOCAL deltas are negative and both corrected functions remain monotone.
The D1 control optimum is nonunique over its bounded interval; D2 control is
the zero boundary solution.

## Score results

The D1 and D2 score populations are 7,462 and 8,491 rows, respectively; their
15,953 pooled record IDs are unique. Gains below are `HINGE MAE - arm MAE`, so
positive values favor the arm.

| Window | Rows | HINGE MAE, all | LOCAL MAE, all | GLOBAL_CONTROL MAE, all | LOCAL gain vs HINGE | LOCAL gain vs control |
|---|---:|---:|---:|---:|---:|---:|
| D1, E67 cycles 72–75 | 7,462 | 11.4048720941 | 11.4023730648 | 11.4063035229 | +0.0024990293 | +0.0039304581 |
| D2, E75 cycles 76–79 | 8,491 | 9.9398315097 | 9.9304729135 | 9.9398315097 | +0.0093585962 | +0.0093585962 |
| Pooled | 15,953 | 10.6251027966 | 10.6189527561 | 10.6257723461 | +0.0061500405 | +0.00681958997 |

Present-only MAE is D1 HINGE/LOCAL/GLOBAL_CONTROL `11.2171489828 / 11.2144677813 /
11.2186847588`, and D2 `9.7305096868 / 9.7205330590 / 9.7305096868`.
All-row mean error and p90 absolute error are, respectively:

| Window | HINGE mean / p90 | LOCAL mean / p90 | GLOBAL_CONTROL mean / p90 |
|---|---:|---:|---:|
| D1 | -0.6911544605 / 24.8430742610 | -0.7426198109 / 24.8182800644 | -0.7210867050 / 24.8548193573 |
| D2 | +0.7233266086 / 23.0272149369 | +0.5339623413 / 22.9599761185 | +0.7233266086 / 23.0272149369 |

### Gates and fixed groups

- **Mechanism gate: PASS.** Both LOCAL deltas are negative. LOCAL beats both
  HINGE and GLOBAL_CONTROL by more than `1e-12` in each window.
- **Candidate gate: FAIL.** Pooled gain over HINGE is `0.0061500405`, below
  the frozen `0.01` threshold. Both window gains are nonnegative and maximum
  supported all-row group harm is only `0.0047197264` (D1).
- **Exploratory review gate: PASS.** Pooled gain is positive; each window's
  harm is below `0.02`; maximum supported all-row group harm is below `0.25`.
- Present-only targeted-region gain over HINGE (`40 < p < 60`) is D1
  `+0.0123823085` and D2 `+0.0481891086`.
- By raw-DIRECT bin, LOCAL gain is exactly zero outside `[40,60)`; in `[40,60)`
  all-row gain is D1 `+0.0114333272` and D2 `+0.0440731226`.

Supported target tails (`all rows / present rows`) show the high-tail tradeoff:

| Window / target tail | Support | LOCAL gain vs HINGE | GLOBAL_CONTROL gain vs HINGE |
|---|---:|---:|---:|
| D1, y <= 5 | 362 / 345 | 0.0000000000 | +0.0293641084 |
| D1, y >= 95 | 374 / 355 | -0.0047197264 | -0.0253307360 |
| D2, y <= 5 | 442 / 422 | +0.0037814642 | 0.0000000000 |
| D2, y >= 95 | 434 / 416 | -0.0046764386 | 0.0000000000 |

Cycle-level LOCAL gains versus HINGE are D1 cycles 72–75:
`-0.0010595357, +0.0082661267, -0.0043562866, +0.0066063165`; D2 cycles
76–79: `+0.0058526037, -0.0035860355, +0.0085682765, +0.0268675812`.
The evaluation JSON contains every all-row and present-only metric (MAE, mean
error, p90, gains, and support) for bins, cycles, and tails.

## Artifact hashes

| Artifact | SHA-256 |
|---|---|
| D1_LOCAL fit state | `2ae3259ae0715051bb06b6eab9e23ec5ae0394832b3455456c16e364064a4291` |
| D1_GLOBAL_CONTROL fit state | `bd26301705cfb20c1fddd011741cc2b30bdbc0912100aa0b26d2944bd131db95` |
| D2_LOCAL fit state | `b818cfd21d0c207c801786945e3fd8e2384487197f75106e70ba88d185a53343` |
| D2_GLOBAL_CONTROL fit state | `47157f177a51d3cc3503f42fe5952bcb9b399cafa90a37d5a2793094fffed808` |
| D1 frozen predictions | `fb424e50a24fa45092af6bbaffb54247649c505cb30b4aa7f5617824f5b13780` |
| D2 frozen predictions | `d5c2cd9b61c85786bfef9c6ad9cac0d1c46ab40e803238e0cbaf8a1170ca67da` |
| D1 score targets, tagged array hash | `a66e65cac3f70191f38a1dc41d6da93f4f8f7736c9524a643ad76ba2874a200a` |
| D2 score targets, tagged array hash | `3ad5e23fc8ec72c1263827668be68fb550e201bf714809c7a737c621216c69b2` |

## Runtime and limits

All fit and score memory checks passed under the shared OS lock with one heavy
job and two threads. For fits, the lowest measured available RAM was
2,322,952,192 bytes (projected 1,786,081,280 after the estimate), above the
2 GiB available and 1 GiB reserve floors. Score evaluation also passed its
1.5 GiB available / 0.75 GiB projected floors. Peak measured process working
set was 146,534,400 bytes during fits and 145,113,088 bytes during scoring.

No rescue, tuning, alternative correction, or final application was run. The
candidate qualification failure does not trigger another fit or reset of the
prior three-slot budget. Broader novelty remains `UNKNOWN` beyond the six
frozen summaries.
