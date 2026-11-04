# Regression frontier: a frozen R06 anchor with position-aware L1 corrections

**Decision: GO for one six-head pilot, REG_POSITION_L1_V1. No official fit has
been run during this design milestone.** Sam requested fresh Regression research
while Luna 1 continues Classification. The executor is **Luna 2**, thread
01a1a442-590f-7520-a09a-bcd10f292328. Root Sol remains decision owner.

Worktree: `tk2-regression-frontier`, branch `codex/regression-position-research`,
base ba79ec30fd6a8b17f66cfde8186e79282ca1a494. The main TP 2 worktree contains
concurrent Classification changes and is an artifact source only for this lane.

## What is new, and why this is not a promised breakthrough

Classification's positional offset and later MIX50 provide a reason to examine
position in Regression, not proof that it transfers across targets. R06 sees
character n-grams and earlier incumbent components see bag text, but the scoped
Regression novelty audit found no first/middle/last token-position experiment.
The historical `rnbr_readiness_v1/POSITIONAL_TEXT_DESIGN.md` records this gap.

The test freezes R06's final prediction and learns a small L1 head on strictly
past R06 out-of-fold residuals. It adds text-position features rather than
refitting R06 or attempting another rank-map/neighbor recipe. Three arms isolate
whether benefit comes from bag text alone, extra capacity, or actual placement.
Six heads are needed: BAG, REAL, SHUFFLED at E67 and E75.

Strong counterevidence is retained. The old coarse residual head increased W4
MAE from11.455005 to11.613971 and pooled H12 from14.047533 to14.128414; it did
not use token-position text. Original R-NBR had public11.19115 versus R06
11.16620. Exact-impact retrieval had E67 H12 harm0.056857 despite E75 W6
gain0.191049, and direct target median lost to R06 with tail harms. A new feature
channel may still fail because past conditional errors do not transfer.

An independent configured Luna explorer reviewed bounded Regression source and
reports, preferred this OOF correction experiment over refitting an augmented
R06 learner, and found rank/CDF/ordinal lanes heavily covered already. A full
R06+position learner would require new character/structured transforms and
control refits; the small residual pilot answers the incremental-information
question with fewer new learned states and an exact anchor. Do not start both.

## Actual zero-fit readiness and descriptive evidence

Run from this worktree:

    python scripts/regression_position_frontier_audit.py --source-root "<main TP 2 root>"

Root ran this successfully. Official Regression Train SHA256:
`756523ff8156a9cd776e52249f7e2a8f23dcec432b8d10570e55ff6db0e0f082`.
All50,608 Train IDs, cycles and raw signal_stream values match the corresponding
Classification feature columns. The reader excludes Classification targets.
Cached positional row identities, manifest/file hashes, prefix boundaries and
blank rows passed; standalone Regression replay vectorizers are present and
match their original receipts. No model, memory, or preprocessor was fitted.

| View | Query rows | R06 MAE | Present MAE | Missing MAE | Available present OOF bank |
|---|---:|---:|---:|---:|---:|
| E67 H12, cycles68–79 |20,903|11.529709|11.349250|14.092617|16,482|
| E75 W6, cycles76–79 |8,491|9.921613|9.711088|13.109505|28,045|

These are the old predictions on reused development outcomes. W6 overlaps the
E67 view; they are not replications. The OOF bank contains present rows in
cycles56–origin, and every donor's generating cutoff precedes its own cycle.
Its stored residuals equal official Regression target minus FINAL mapped R06
OOF prediction, not the raw character learner output.

R06's low targets still tend to be overpredicted, and high targets underpredicted.
On E67, mean target-minus-prediction is -8.753 for target<=5 and +7.116 for
target>=95; at E75, -8.009 and +5.207. These target-defined slices are descriptive,
never input features or routes. The new pilot must protect tails, not merely
improve a pooled number. Missing rows remain exactly R06; this does not clear its
historical W1 missing-stream guard failure against the older incumbent.

Existing REAL/shuffled position suffixes have17,025 columns at E67 and17,169 at
E75. Fit CSR storage is about11.1 and15.9 MiB respectively before adding the bag
block. This is input storage only, not a LightGBM peak-memory guarantee. The
full numeric/provenance audit is `READINESS_AND_ERROR_AUDIT.json`; the one plot
`r06_cycle_residuals.png` shows why a universal error offset is not assumed.

## Prior lanes that do not justify another broad sweep

- R11 ordinal heads, fixed ten-bin CDF median, and R12 ranking already test
  distribution/rank approaches. They are not new simply because they resemble TP1.
- Earlier character Ridge lost primary despite H12 improvement. Text benefit is
  representation/learner dependent, not automatic.
- N03 learned routing already failed performance/control criteria. No learned
  router or target-defined route is hidden in this pilot.
- Position is treated as anonymous token placement; no semantic, quality or
  causal interpretation is authorized.

Implementation and exact gates are in `PROTOCOL.md`; the copyable executor prompt
is `LUNA_2_PROMPT.md`. The design phase does not access Test/Sample/Kaggle, modify
the shared ledger, or claim a new score. Six pilot fits and any conditional final
work must be counted in the new isolated lane ledger, not silently absorbed into
Luna 1's active shared budget.

Root also supplied pure transform/design/prediction contracts in
`src/models/reg_position_l1_v1.py`; **8 tests passed**, including a synthetic L1
fit, model persistence/replay and Unicode NPZ readback. One sklearn warning about
feature names on the synthetic sparse input remains disclosed. Compile and diff
checks passed. These are initial unit/contract checks, not a claim that the
executor's production entrypoint/lifecycle, six official fits or metrics passed.
