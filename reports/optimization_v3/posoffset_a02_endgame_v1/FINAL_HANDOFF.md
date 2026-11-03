# POS + A02 zero-fit endgame — final handoff

**Status: MIX50 qualified internally; E79 candidate prepared and independently verified; awaiting Sol review. No Kaggle upload was made.**

## Frozen policy comparison

The selected policy is **MIX50** by the predeclared priority. Both frozen
policies passed the internal gate. The experiment was adaptively selected after
the observed public scores of POS (0.83123) and A02 (0.82930); E67 and E75
overlap, so these results are sensitivity views rather than independent
replication. No public score was used to choose between the two policies.

| View | Rows | pP | pA | b (A02 routed baseline) | MIX50 | MIX50 Δ vs pP | ADD_RESIDUAL | ADD_RESIDUAL Δ vs pP |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| E67 H12 | 20,903 | 0.806258 | 0.811159 | 0.808961 | 0.811937 | +0.005679 | 0.808056 | +0.001798 |
| E67 W4 | 4,950 | 0.840145 | 0.844405 | 0.841418 | 0.846365 | +0.006220 | 0.843415 | +0.003269 |
| E67 W5 | 7,462 | 0.798799 | 0.805176 | 0.802502 | 0.806465 | +0.007666 | 0.802267 | +0.003469 |
| E67 W6 | 8,491 | 0.788935 | 0.792999 | 0.791846 | 0.792531 | +0.003595 | 0.788239 | −0.000696 |
| E75 W6 | 8,491 | 0.811443 | 0.819604 | 0.816599 | 0.816099 | +0.004657 | 0.812859 | +0.001416 |

All entries are Macro F1 at threshold 0.50. For MIX50, the required gains were
at least 0.0005 on E67 H12 and E75 W6; E67 W4/W5/W6 all improved, with no
window harm. ADD_RESIDUAL also cleared those gates, though E67 W6 was 0.000696
below pP, within the allowed 0.003 harm.

Across 40 supported group/view cells, MIX50's worst Macro F1 harm versus pP
was 0.001789 and worst positive-recall harm was 0.012987, both within the
limits of 0.01 and 0.02. ADD_RESIDUAL's corresponding worst harms were 0.004353
and 0.007215. There were no unsupported group cells. The fixed priority picks
MIX50 because both policies qualify; no extra selection rule was added.

The primary comparison shows MIX50 improves on the positional route pP in both
required views and all three E67 windows. It exceeds A02 pA on E67 H12 by
0.000778, but is below pA on E75 W6 by 0.003505. The result supports this
specific frozen combination on the examined overlapping views; it is not a
new untouched holdout or a public leaderboard result.

## E79 candidate

- CSV: `submissions/optimization_v3/posoffset_a02_endgame_v1/classification_mix50_e79.csv`
- CSV SHA-256: `48dcbd3f51fa4567e8f3f3a6838efdfaa50f4c2129e628e1b597436918c46ce7`
- Label-vector SHA-256: `74c7198d6c7c2e97dee5c083f6b37d5666cd71c1df50378f044de57928bb36f2`
- Rows/schema: 29,421, columns `record_id,artifact_signal`, unique IDs, integer
  0/1 labels. Positive count: 6,800.
- Test cycles checked: complete 80–91; empty-stream rows: 1,062.
- Candidate definition: missing rows use pP exactly; present rows use
  `0.5*pP + 0.5*pA`; threshold `>= 0.50`.
- Existing-submission duplicate scan: no identical label vector found.
- Preparation: `PREPARED_NOT_SUBMITTED`. No Test targets, fits, or Kaggle API
  calls were made.

The E79 verifier joined the bundle to official Test IDs, cycles, and
`signal_stream` availability; checked SampleSubmission order; matched the
combined prediction bundle, original A02 prediction bundle/receipt/model,
trained-state freeze, C16/Sparse25 cache, and prior batch manifest hashes; and
reconstructed A02 b from the C16 route bit-exactly. It also reproduced the
historical POS and A02 hard labels from their stored probabilities before
building MIX50.

Detailed artifacts:

- [Selected E79 manifest](E79_SELECTED_MANIFEST.json)
- [Independent E79 QA](E79_INDEPENDENT_QA.json)
- [Frozen policy metrics and gates](POLICY_METRICS.json)
- [Frozen-source independent metric verification](INDEPENDENT_VERIFICATION.json)
- [Full premetric A02 model/donor/correction audit](PREMETRIC_AUDIT.json)

## Reproducibility and accounting

- Focused synthetic tests: **10 passed**.
- Independent policy verification: **PASS, 195 checks**.
- Full A02 lineage replay before scoring: **PASS**. It reconstructed the E67
  and E75 model states, TF-IDF donor transforms and Train-only OOF residuals,
  then reproduced 879,776 saved neighbor references and the original capped
  corrections/predictions bit-exactly.
- E79 independent verification: **PASS** for all recorded checks.
- No new estimator fits or ledger events. The shared ledger remains byte-identical
  to the prior E79 batch snapshot: SHA-256
  `8380d884fabb3cf19edfd606b2b23c7839b9267d16c6660e579444e8089f2d22`;
  129 starts, 123 saved, 6 failed, 0 pending.
- The independent metric verifier was invoked with
  `FREEZE_SUCCESSOR_01.json` explicitly because the original frozen verifier's
  default path points to the superseded pre-correction freeze. Its source file
  remained byte-identical to the version committed before metrics; the exact
  invocation and hash are recorded in `INDEPENDENT_VERIFICATION.json`.

## Commit trail and stop point

- Policy code/tests and no-target input preflight: `761453dc47e61586de9c74ec2445dfe1bb25b512`.
- Corrected donor-filter preflight: `deb4c263ecd5d424122ea4aa02184c4ab48c1793`.
- Successor input/source freeze: `7db706cf9dd0b77a9b914ec19d7a2c5c21db56bd`.
- Full premetric artifact audit: `6b6eb773473566c8fda8418466ba3691c49e8e9c`.
- Frozen comparison and independent verification: `da8f0e1613c1da1bb7fa9788432de208b877470e`.
- E79 preparation/verification wrapper: `4565adb483bed13c07b140edf2b3bc67e583c3b5`.

The candidate is ready for Sol's independent review. No upload, additional
policy evaluation, Test-based selection, gate change, or model update is
authorized here. Do not retry an upload from this task.
