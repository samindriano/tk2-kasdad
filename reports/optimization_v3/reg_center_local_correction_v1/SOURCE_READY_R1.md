# Source-ready R1 — preparation only

Decision: `SOURCE_READY_FOR_ROOT_REVIEW`; the signed root source pass is absent, so no calibrator fit or score target access is authorized by this receipt.

## Identity and immutable receipts

- Worktree: `C:\Users\Sam\OneDrive\Documents\KASDAD\tk2-reg-position-failure-mechanism-v1`
- Branch: `codex/reg-position-failure-mechanism-v1`
- Source commit: `0db77c67e2ab9e1153128cdd72724162c630809b`
- Prefit freeze: `experiments\reg_center_local_correction_v1\PREFIT_FREEZE_R1.json` — SHA-256 `9f488498c44d8e308e739050cf7c5d2411b2af74428f60ef7b5d2e27ba1807ba`
- Non-target preflight: `reports\optimization_v3\reg_center_local_correction_v1\IDENTITY_PREFLIGHT_R1.json` — SHA-256 `bbfd2935d91a23722f0ba79dbfb0013a72625c68676a232500532b65dffaf04b`
- Frozen file pins: 34 source, reference, input, and novelty entries

### Six-document novelty scan

No exact duplicate was found in this bounded scan. Repository-wide novelty remains `UNKNOWN`.

| Frozen summary | SHA-256 |
|---|---|
| `C:\Users\Sam\OneDrive\Documents\KASDAD\tk2-reg-position-failure-mechanism-v1\reports\optimization_v3\reg_monotone_hinge_p50_v1\PROTOCOL_R3.md` | `65bd376d5645f47338dac844be9c0a573a951919c8fccc0e4321da65674646cf` |
| `C:\Users\Sam\OneDrive\Documents\KASDAD\tk2-reg-position-failure-mechanism-v1\reports\optimization_v3\reg_monotone_hinge_p50_v1\SOL_DIAGNOSTIC_DECISION_R4.md` | `538007fd2449b472e1fad657c916760710b1c6dd148287f051959162318ff7ba` |
| `C:\Users\Sam\OneDrive\Documents\KASDAD\tk2-reg-position-failure-mechanism-v1\reports\optimization_v3\reg_monotone_hinge_p50_v1\PROTOCOL_R4.md` | `534192cfa32f181b1a168110517ae6290598a529a02911bf32f93f70753623a1` |
| `C:\Users\Sam\OneDrive\Documents\KASDAD\tk2-reg-position-failure-mechanism-v1\reports\optimization_v3\reg_position_transport_diagnostic_v1\TRANSPORT_DIAGNOSTIC.md` | `5b0840a80ce80cb478d6dbf888c733b7f7cd9f901ee67006b8c4206bd5be0085` |
| `C:\Users\Sam\OneDrive\Documents\KASDAD\tk2-reg-position-failure-mechanism-v1\reports\optimization_v3\regression_position_frontier_v1\FINDINGS.md` | `736837b510762965dd8c41089a88ed943a83b72bb7e56ae48ce41e5ce5703345` |
| `C:\Users\Sam\OneDrive\Documents\KASDAD\tk2-reg-position-failure-mechanism-v1\reports\optimization_v3\regression_residual\FINDINGS.md` | `48465bfc35162315da247600673ae9bd54121dfb12c7315d0a4449dc6b5e81ae` |

### Source and input hashes

The table is the complete 34-entry byte-hash inventory bound by the prefit freeze.

| Pin | SHA-256 | Path |
|---|---|---|
| `input:d1_fit_input` | `d3646fe66040f064faa671f9ccc67f67d8c72b83e6192b681c2857a3163c76bd` | `C:\Users\Sam\OneDrive\Documents\KASDAD\tk2-reg-position-failure-mechanism-v1\experiments\runs\reg_monotone_hinge_p50_v1\R4\fit_inputs\HINGE_D1.npz` |
| `input:d1_global_state` | `e0cb43aff7c99722041404e6db6666db387aaf4a937e248c51f12ba4d5293b06` | `C:\Users\Sam\OneDrive\Documents\KASDAD\tk2-reg-position-failure-mechanism-v1\experiments\runs\REG_FINAL_WAVE_FOLLOWUP_V2\fits\D1_GLOBAL.json` |
| `input:d1_hinge_state` | `a2fe1297e15566bdc86506458a81de4b7b9f4c6c15359d9fbea4b2818ff3838a` | `C:\Users\Sam\OneDrive\Documents\KASDAD\tk2-reg-position-failure-mechanism-v1\experiments\runs\reg_monotone_hinge_p50_v1\R4\fits\HINGE_D1.json` |
| `input:d1_score_rows` | `b996d09f18446eff7f8723c76b5d44d83e0a6d684672c98d38839a2ea81e2643` | `C:\Users\Sam\OneDrive\Documents\KASDAD\tk2-reg-position-failure-mechanism-v1\experiments\runs\reg_monotone_hinge_p50_v1\R4\score_rows\HINGE_D1.npz` |
| `input:d2_fit_input` | `29b1a7d52e9e4f9d502d329071769e260889e81817d82b9e12f73abd0cfc5445` | `C:\Users\Sam\OneDrive\Documents\KASDAD\tk2-reg-position-failure-mechanism-v1\experiments\runs\reg_monotone_hinge_p50_v1\R4\fit_inputs\HINGE_D2.npz` |
| `input:d2_global_state` | `01db7b8973fa0488f0f37b134c6b99e9ae7003b88dc03fd2dbcccdd17245b53f` | `C:\Users\Sam\OneDrive\Documents\KASDAD\tk2-reg-position-failure-mechanism-v1\experiments\runs\REG_FINAL_WAVE_FOLLOWUP_V2\fits\D2_GLOBAL.json` |
| `input:d2_hinge_state` | `9be290f35c89d7fd9199e6838a93a429d5ca5c3049117e4f30506a0b418bc51e` | `C:\Users\Sam\OneDrive\Documents\KASDAD\tk2-reg-position-failure-mechanism-v1\experiments\runs\reg_monotone_hinge_p50_v1\R4\fits\HINGE_D2.json` |
| `input:d2_score_rows` | `8c950bd8fd27b00f89c71a4e87eed0e0270cf385b9aece9cca33caeee91040e6` | `C:\Users\Sam\OneDrive\Documents\KASDAD\tk2-reg-position-failure-mechanism-v1\experiments\runs\reg_monotone_hinge_p50_v1\R4\score_rows\HINGE_D2.npz` |
| `input:r4_diagnostics` | `ea6a77cbb43c836e08136a8c417e02384084deabc1c42fa94226f719e20f7b1e` | `C:\Users\Sam\OneDrive\Documents\KASDAD\tk2-reg-position-failure-mechanism-v1\reports\optimization_v3\reg_monotone_hinge_p50_v1\DIAGNOSTICS_R4.json` |
| `input:r4_identity_freeze` | `97163e0ff187da7ac82f9e84319db89979c00eba0998016dd14ca25910226bfa` | `C:\Users\Sam\OneDrive\Documents\KASDAD\tk2-reg-position-failure-mechanism-v1\experiments\reg_monotone_hinge_p50_v1\IDENTITY_FREEZE_R4.json` |
| `input:r4_root_pass` | `adbeaac6a53afd095eb53ffe372eb89de9ec1f3e47dd7794ecafb163ff394bac` | `C:\Users\Sam\OneDrive\Documents\KASDAD\tk2-reg-position-failure-mechanism-v1\experiments\reg_monotone_hinge_p50_v1\ROOT_SOURCE_PASS_R4.json` |
| `input:r4_verification` | `f89a2e2f21afacd79f88d3253e76a3207e25c7cd29fae5f5e8cbdc21586f9f2f` | `C:\Users\Sam\OneDrive\Documents\KASDAD\tk2-reg-position-failure-mechanism-v1\reports\optimization_v3\reg_monotone_hinge_p50_v1\DIAGNOSTICS_VERIFICATION_R4.json` |
| `input:resource_policy` | `0f281c9ba4552b20a47a8d977bf7ee083cf1009c94024068c4f0ef80ab4cad32` | `C:\Users\Sam\OneDrive\Documents\KASDAD\tk2-reg-position-failure-mechanism-v1\reports\optimization_v3\RESOURCE_POLICY_USER_REVISED_20261105.json` |
| `novelty_summary:r3_protocol` | `65bd376d5645f47338dac844be9c0a573a951919c8fccc0e4321da65674646cf` | `C:\Users\Sam\OneDrive\Documents\KASDAD\tk2-reg-position-failure-mechanism-v1\reports\optimization_v3\reg_monotone_hinge_p50_v1\PROTOCOL_R3.md` |
| `novelty_summary:r4_decision` | `538007fd2449b472e1fad657c916760710b1c6dd148287f051959162318ff7ba` | `C:\Users\Sam\OneDrive\Documents\KASDAD\tk2-reg-position-failure-mechanism-v1\reports\optimization_v3\reg_monotone_hinge_p50_v1\SOL_DIAGNOSTIC_DECISION_R4.md` |
| `novelty_summary:r4_protocol` | `534192cfa32f181b1a168110517ae6290598a529a02911bf32f93f70753623a1` | `C:\Users\Sam\OneDrive\Documents\KASDAD\tk2-reg-position-failure-mechanism-v1\reports\optimization_v3\reg_monotone_hinge_p50_v1\PROTOCOL_R4.md` |
| `novelty_summary:reg_position_transport_diagnostic` | `5b0840a80ce80cb478d6dbf888c733b7f7cd9f901ee67006b8c4206bd5be0085` | `C:\Users\Sam\OneDrive\Documents\KASDAD\tk2-reg-position-failure-mechanism-v1\reports\optimization_v3\reg_position_transport_diagnostic_v1\TRANSPORT_DIAGNOSTIC.md` |
| `novelty_summary:regression_position_frontier_findings` | `736837b510762965dd8c41089a88ed943a83b72bb7e56ae48ce41e5ce5703345` | `C:\Users\Sam\OneDrive\Documents\KASDAD\tk2-reg-position-failure-mechanism-v1\reports\optimization_v3\regression_position_frontier_v1\FINDINGS.md` |
| `novelty_summary:regression_residual_findings` | `48465bfc35162315da247600673ae9bd54121dfb12c7315d0a4449dc6b5e81ae` | `C:\Users\Sam\OneDrive\Documents\KASDAD\tk2-reg-position-failure-mechanism-v1\reports\optimization_v3\regression_residual\FINDINGS.md` |
| `reference_source:hinge_core` | `f51d428fce23cf45575d8bbdc3241cc2a5c976b840665aca33b62b1a9687fb99` | `C:\Users\Sam\OneDrive\Documents\KASDAD\tk2-reg-position-failure-mechanism-v1\scripts\reg_monotone_hinge_p50_v1.py` |
| `reference_source:position_frontier_findings` | `736837b510762965dd8c41089a88ed943a83b72bb7e56ae48ce41e5ce5703345` | `C:\Users\Sam\OneDrive\Documents\KASDAD\tk2-reg-position-failure-mechanism-v1\reports\optimization_v3\regression_position_frontier_v1\FINDINGS.md` |
| `reference_source:r4_decision` | `538007fd2449b472e1fad657c916760710b1c6dd148287f051959162318ff7ba` | `C:\Users\Sam\OneDrive\Documents\KASDAD\tk2-reg-position-failure-mechanism-v1\reports\optimization_v3\reg_monotone_hinge_p50_v1\SOL_DIAGNOSTIC_DECISION_R4.md` |
| `reference_source:r4_protocol_r3` | `65bd376d5645f47338dac844be9c0a573a951919c8fccc0e4321da65674646cf` | `C:\Users\Sam\OneDrive\Documents\KASDAD\tk2-reg-position-failure-mechanism-v1\reports\optimization_v3\reg_monotone_hinge_p50_v1\PROTOCOL_R3.md` |
| `reference_source:r4_protocol_r4` | `534192cfa32f181b1a168110517ae6290598a529a02911bf32f93f70753623a1` | `C:\Users\Sam\OneDrive\Documents\KASDAD\tk2-reg-position-failure-mechanism-v1\reports\optimization_v3\reg_monotone_hinge_p50_v1\PROTOCOL_R4.md` |
| `reference_source:r4_runner` | `dcf35a80bf50e9e83a234b08c7b3cc1b32cdec5e54e89236fb221fa33352225d` | `C:\Users\Sam\OneDrive\Documents\KASDAD\tk2-reg-position-failure-mechanism-v1\scripts\reg_monotone_hinge_p50_v4.py` |
| `reference_source:r4_verifier` | `0a291d18390eacd17ccd7ee1712c99bd4c307263dca442936b5494201d969245` | `C:\Users\Sam\OneDrive\Documents\KASDAD\tk2-reg-position-failure-mechanism-v1\scripts\verify_reg_monotone_hinge_p50_v4.py` |
| `reference_source:residual_findings` | `48465bfc35162315da247600673ae9bd54121dfb12c7315d0a4449dc6b5e81ae` | `C:\Users\Sam\OneDrive\Documents\KASDAD\tk2-reg-position-failure-mechanism-v1\reports\optimization_v3\regression_residual\FINDINGS.md` |
| `reference_source:transport_diagnostic` | `5b0840a80ce80cb478d6dbf888c733b7f7cd9f901ee67006b8c4206bd5be0085` | `C:\Users\Sam\OneDrive\Documents\KASDAD\tk2-reg-position-failure-mechanism-v1\reports\optimization_v3\reg_position_transport_diagnostic_v1\TRANSPORT_DIAGNOSTIC.md` |
| `source:gitattributes` | `03dc531d559a04bb0dcbc07668bde7179bbfbba92bb39fb2ece15c2ac3b7acee` | `C:\Users\Sam\OneDrive\Documents\KASDAD\tk2-reg-position-failure-mechanism-v1\.gitattributes` |
| `source:handoff` | `d33b01ddf4d556cc0be9ae9b647950c5dc2571669f643893d64273a8b829a82a` | `C:\Users\Sam\OneDrive\Documents\KASDAD\tk2-reg-position-failure-mechanism-v1\reports\optimization_v3\reg_center_local_correction_v1\SOL_HANDOFF.md` |
| `source:policy` | `e9a6d09308e2880c97de2a18768d71420e153e29aae10262c49a0892fb746fe8` | `C:\Users\Sam\OneDrive\Documents\KASDAD\tk2-reg-position-failure-mechanism-v1\reports\optimization_v3\reg_center_local_correction_v1\POLICY_R1.md` |
| `source:runner` | `4dab89d9a6d58c2bdcd47001de623a45c8765c3719928db033e72d2c7cabe4f5` | `C:\Users\Sam\OneDrive\Documents\KASDAD\tk2-reg-position-failure-mechanism-v1\scripts\reg_center_local_correction_v1.py` |
| `source:tests` | `1cbfa5e63e31f614e1655bc6541de95b38925fb8d1e8b3c663d183a72d2a50bc` | `C:\Users\Sam\OneDrive\Documents\KASDAD\tk2-reg-position-failure-mechanism-v1\tests\test_reg_center_local_correction_v1.py` |
| `source:verifier` | `5bff4e64811f702cf1219ffc597f3fe7f34910db8ffb7577135fc8d86efc1d76` | `C:\Users\Sam\OneDrive\Documents\KASDAD\tk2-reg-position-failure-mechanism-v1\scripts\verify_reg_center_local_correction_v1.py` |

## Non-target QA

| Population | All rows | Present | Missing | Present rows with b(p)>0 |
|---|---:|---:|---:|---:|
| HINGE_D1 / fit | 4950 | 4608 | 342 | 970 |
| HINGE_D1 / score | 7462 | 6955 | 507 | 1506 |
| HINGE_D2 / fit | 12412 | 11563 | 849 | 2476 |
| HINGE_D2 / score | 8491 | 7965 | 526 | 1649 |

- Pooled score rows: 15953; record IDs and physical rows are unique across both windows.
- Saved HINGE minimum derivatives at delta = -2: D1 0.801552453489198; D2 0.800391339800163.
- Raw DIRECT fallback bits and saved HINGE replay passed.
- Candidate fit targets loaded: `false`; calibrator starts: `0`; score target values loaded: `false`. Test/Sample, E79, CSV, and Kaggle flags remain false.

## Recipe and cost

- Adaptive diagnostic: same-history saved fixed-p50 HINGE base; LOCAL triangle center 50, half-width 10, delta in [-2,0]; exact lower weighted median with weight b(p); matched same-present-fit GLOBAL_CONTROL lower empirical median. Each of the four calibrators counts as one fit. No objective clipping; application clips present rows; missing rows preserve raw DIRECT float64 bits.
- Memory gate: 2,364,690,432 bytes available initially; 536,870,912-byte estimate; projected 1,827,819,520 bytes; required 1,610,612,736 available and 805,306,368 projected. Gate passed.
- Preflight peak process working set: 145,629,184 bytes. Shared OS lock held, one heavy job, two threads. Candidate fit and score compute cost so far: zero starts.

## Unknowns and next gate

- `ROOT_SOURCE_PASS_R1.json` is absent. The four calibrator fits, prediction freeze, score-target read, score metrics, and every candidate/mechanism gate remain unrun/UNKNOWN.
- Broader novelty beyond the six summaries is `UNKNOWN`; results would remain adaptive after the post-discovery choice.
- The exact root callback schema is in `POLICY_R1.md`. Root review is the next action; this receipt does not open fitting or scoring.
