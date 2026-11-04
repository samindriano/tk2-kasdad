# CLS_BIGRAM_POSITION_PILOT_V1 — frozen execution card

**Status:** authorized, fixed six-fit pilot. This is a temporal development comparison, not an outcome-blind experiment or untouched holdout.

## Question

Does position information for known adjacent token pairs add predictive information beyond the same frozen C2B bag plus original unigram-position logit, the bag bigram arm, and a pair-order shuffled control?

## Execution

- Use official Classification Train only, with E67 fit/query cycles 47–67 / 68–79 and E75 47–75 / 76–79.
- Reuse the frozen prefix TF-IDF vocabulary/IDF and tokenizer. Keep OOV pair occurrences in the sequence so they still determine the three equal-position thirds.
- Fit only one gamma per arm and origin: BAG, REAL, SHUFFLED at each cutoff, in the order in `RECIPE.json`. The fixed offset is reconstructed from the original C2B bag model plus its original unigram-position gamma.
- Use the original POS_BAG_OFFSET pure L-BFGS-B solver and exact sum-loss-plus-L2 objective, starting gamma at zero. One attempt per fit; never retry or sweep.
- Route present rows as `0.5 * sigmoid(z_original + X_arm @ gamma) + 0.5 * p_A02`; route missing rows exactly to pP. Present rows with no known bigram must match frozen MIX50 exactly.
- Keep query labels closed until six saved receipts, prediction bundles, and independent prelabel verification pass.

## Gate and interpretation

REAL must exceed MIX50 by at least 0.0005 Macro F1 on both E67 H12 and E75 W6, must not trail BAG or SHUFFLED on either required view, and may lose no more than 0.003 on E67 W4/W5/W6. On each of the five views, supported fit-defined availability, entry-friction, and text-load groups require n≥200, at least 30 rows per class, Macro F1 harm≤0.01, and positive-recall harm≤0.02. Unsupported groups are UNKNOWN. All comparisons use threshold 0.50 with 1e-12 gate tolerance.

These are engineering gates, not significance tests. The E67 and E75 query windows overlap. BAG is lower-dimensional; REAL versus SHUFFLED is the matched-capacity control. No Test, SampleSubmission, Kaggle, causal, semantic, or champion-promotion conclusion is part of this pilot.

## Reproduction

After reviewing the Sol authorization, run:

```powershell
python scripts/optimization_v3_bigram_position_pilot_v1.py preflight
python scripts/optimization_v3_bigram_position_pilot_v1.py synthetic --temp-root "$env:TEMP\bigram-position-pilot-synthetic"
# Commit source, recipe, tests, and target-free QA before freezing/reserving.
python scripts/optimization_v3_bigram_position_pilot_v1.py freeze
python scripts/optimization_v3_bigram_position_pilot_v1.py reserve
python scripts/optimization_v3_bigram_position_pilot_v1.py run-all
python scripts/optimization_v3_bigram_position_pilot_v1.py predictions
python scripts/optimization_v3_bigram_position_pilot_v1.py verify-prelabel
python scripts/optimization_v3_bigram_position_pilot_v1.py score
```

The six-fit accounting and durable bundles live under the ignored `experiments/runs/optimization_v3_research_next_v1/BIGRAM_POSITION_PILOT_V1/` directory. Final metrics and plots are written under this report directory.
