# Public/local forensic comparison: direct-position regression

**Disposition:** keep `REG_POSITION_DIRECT_L1_V1` at FAIL / no promotion. Candidate B is the current public-score leader in the supplied regression snapshot, an observed benchmark result that does not retro-pass the local gate.

## Scope and access

This is a read-only comparison of hash-pinned fitted states, saved direct E67/E75 prediction bundles, already-scored Train query labels, frozen R06 OOF caches, and the supplied Kaggle history snapshots. The script made zero fit or predict calls, parsed no fit-prefix target rows, did not open Test labels, Sample Submission, or a new submission CSV, and did not contact Kaggle. Writes are limited to this report namespace.

The local target support is the previously scored E67 query rows (20,903 rows, cycles 68–79); E75 W6 is the same 8,491 rows as E67 W6. These are adaptive development views, not untouched validation.

## Public benchmark observation

The 2026-11-04 regression snapshot has 18 COMPLETE scored records. B (`direct_position_real_e79.csv`, ref 57988146) scored **11.14400**, the lowest public score in that snapshot, versus R06 ref 57874821 at **11.16620**: a public MAE improvement of **0.02220**. All 18 private scores are blank. The public sample composition/randomness and private outcome are unknown; this is one selected observed score, not a significance result or evidence of private transfer.

The separate classification snapshot is pinned only to prevent cross-task mix-ups; it contributes no regression evidence.

## Recipe and fitted-state comparison

The available evidence shows no material recipe or baseline-definition mismatch between the frozen pilot and B:

- Both use direct `threat_rank`, the same `LGBMRegressor` objective and all resolved parameters: L1 regression, GBDT, 400 trees, learning rate .05, 15 leaves, min child samples 80, lambda 3, alpha 0, max depth -1, seed 42, two jobs, deterministic and column-wise execution. The full parameter receipt is in `feature_state_comparison.json`.
- The ordered structured fields, 36 numeric positions and 3 categorical positions match. Both use `StandardScaler(with_mean=True, with_std=True)` and sparse `OneHotEncoder(handle_unknown=infrequent_if_exist, min_frequency=30, max_categories=254)`. B's structured block is 426 columns; learned category sets differ with the training population.
- Both use `[structured, unigram bag, REAL three-third position suffix, missing indicator]`. The word TF-IDF parameter dictionaries match exactly (word analyzer, 1–2 grams, `min_df=3`, `max_df=.995`, 20,000 max features, sublinear TF, Unicode accent stripping, float64, L2, smooth IDF). The selected unigram bag is row-L2 normalized. The position descriptor uses the same saved three-third counts weighted by `(1+log(count))*shared_IDF`, then row-L2 normalized once.
- Both rank raw auxiliary scores within each complete cycle using average ties and percentile `100*rank/n`, then map `0.5*raw + 0.5*percentile`. Both use `clip(0.75*final_mapped_R06 + 0.25*mapped_aux, 0, 100)`, calculate the whole-cycle rank/blend before masking, and replace missing-text rows with exact final-mapped R06.
- Direct E67 and E75 use their respective prefix-fitted states (29,705 rows through cycle 67; 42,117 through cycle 75). B uses the full 50,608 Train rows through cycle 79, as intended for deployment. Therefore the learned vocabulary, IDF, structured categories, and LGBM input width are different fitted states. That is a train-boundary/state difference, not a different recipe: the 20k-term vocabulary overlaps B by 17,143 terms for E67 (Jaccard .750) and 18,686 for E75 (Jaccard .877); mean absolute IDF differences on common terms are .2286 and .1044. Input widths are 22,985 (E67), 23,263 (E75), and 23,439 (B).
- Both local R06 baselines are exact matches to the hash-pinned final mapped R06 prediction member in their origin cache. B's manifest explicitly binds the baseline to the manifest-bound final mapped R06 prediction and says the raw candidate member was not used as baseline.
- Content pins: direct freeze signature `80614c292413e00fd4234bb1d4da4f3b40f2aea019bb6d510dab035e356b4003`; B trained-state signature `5bef9bb9b9587e1a5880f91578b0e3b8525f6a93f4ab5110e18586ab269c0b4c`. Direct REAL model, E67/E75 and B regressor, TF-IDF, structured, scaler, OHE, prediction bundle, R06 cache, and snapshot file hashes are recorded in `evidence_pins.json`.

B fits cycles 47–79 and predicts cycles 80–91 (12 cycles). E67 predicts 68–79 (12 cycles) from only 21 training cycles; E75 W6 predicts 76–79 (four cycles) from 29 training cycles. No exact 33-cycle fit → 12-future-cycle OOS evaluation exists within Train. These horizon and training-support differences are material to transfer, but are the intended full-fit vs local OOF comparison.

## Saved local results

| View | n | R06 MAE | REAL final MAE | Gain vs R06 | Gain vs BAG | Gain vs SHUFFLED |
|---|---:|---:|---:|---:|---:|---:|
| E67 H12 | 20903 | 11.52971 | 11.50815 | +0.02156 | +0.01160 | +0.01091 |
| E67 W6 | 8491 | 11.65345 | 11.61198 | +0.04147 | +0.02953 | +0.02327 |
| E75 W6 | 8491 | 9.92161 | 9.98642 | -0.06481 | -0.00850 | -0.01314 |

E67 H12's +.02156 gain misses its frozen +.03 R06 gate. E75 W6 harms R06 by .06481; BAG and SHUFFLED also harm R06, and REAL trails them. Each of cycles 76–79 harms R06 at E75. This preserves the local FAIL.

### Fixed tail groups and same-cycle changes

| Origin/view | Group | n | Gain vs R06 (MAE) |
|---|---:|---:|---:|
| E67 H12 | TARGET_LE5 | 1048 | +0.00539 |
| E67 H12 | TARGET_GT5_LT95 | 18798 | +0.02823 |
| E67 H12 | TARGET_GE95 | 1057 | -0.08108 |
| E75 W6 | TARGET_LE5 | 442 | -0.13675 |
| E75 W6 | TARGET_GT5_LT95 | 7615 | -0.05957 |
| E75 W6 | TARGET_GE95 | 434 | -0.08338 |

E67 H12's upper tail (target ≥95) is down .08108 while its low and middle groups are slightly up. All three E75 W6 target groups are down; the largest harm is .13675 for target ≤5. These group values are descriptive and do not override the primary gate.

| Origin | Cycle | n | Gain vs R06 (MAE) |
|---:|---:|---:|---:|
| E67 | 76 | 1822 | +0.01351 |
| E67 | 77 | 2335 | +0.03833 |
| E67 | 78 | 2146 | +0.01306 |
| E67 | 79 | 2188 | +0.09596 |
| E75 | 76 | 1822 | -0.07746 |
| E75 | 77 | 2335 | -0.01071 |
| E75 | 78 | 2146 | -0.10000 |
| E75 | 79 | 2188 | -0.07748 |

On matched W6, R06 MAE falls by 1.73184 from E67 to E75; the direct raw-head MAE falls by 1.55940, so its deficit relative to R06 widens by .17244. Rank50 reduces auxiliary MAE by .29974 at E67 W6 and .09378 at E75 W6, but the mapped head remains worse than R06 in both views.

For present rows, the saved policy correction has the same sign as the R06 residual on 0.517 of E67 W6 rows and 0.484 of E75 W6 rows. Per-cycle correction/residual correlations range 0.041 to 0.175 (E67) and -0.071 to 0.088 (E75). This is weak directional alignment, not a stable correction mechanism.

A deterministic 2,000-resample, seed-42 cycle-cluster bootstrap over the eight-cycle late-unique view gives row-weighted gain -0.03321, with descriptive percentile interval [-0.06430, +0.00605]. It spans zero and is not public/private uncertainty or a significance test.

## Interpretation and prior STOP language

Sampling, transport, and submission selection can plausibly explain why one public score looks better while the local pilot fails, but the evidence cannot identify which mechanism did so. The observed comparison mixes different trained prefixes and different forecast windows, and public sample composition is unknown. A recipe mismatch or raw-vs-final R06 baseline mismatch is not supported by the pinned artifacts.

The earlier `NONE_JUSTIFIED` / STOP conclusion is sound when read as: the frozen local gate failed, and this post-hoc decomposition does not justify another fit, retune, or promotion. It would overstate the evidence if read as proof that the recipe cannot transfer, that B's public score is false, or that the local FAIL should be replaced by PASS.

## Decision by evidence layer

- **Mechanism:** local E75 evidence shows baseline catch-up and weak correction alignment; it does not isolate a causal positional-text effect.
- **Internal risk gate:** direct-position candidate remains **FAIL / no promotion** under the frozen pilot gate.
- **Benchmark admission:** B is a completed E79 public benchmark record, explicitly marked exploratory with prior local gates failed.
- **Public-best status:** B is lowest among the 18 completed regression records in this dated snapshot; private score and generalization remain unknown.

No blanket PASS, significance claim, or private-score inference is warranted.

## Reproduction artifacts

Run `python reports/optimization_v3/public_local_forensic_v1/reg/public_local_forensic_v1.py --repo-root <owned-worktree> --source-root <TP-2-root> --saved-artifact-root <saved-artifact-root>`.

- `public_local_metrics.csv`, `cycle_metrics.csv`, `group_metrics.csv`, and `correction_diagnostics.csv` contain the detailed tables.
- `feature_state_comparison.json`, `cluster_bootstrap.json`, and `public_snapshot_summary.json` hold the state comparison and summary diagnostics.
- `evidence_pins.json` lists every absolute input path and verified SHA-256; `output_hashes.json` pins generated files.
