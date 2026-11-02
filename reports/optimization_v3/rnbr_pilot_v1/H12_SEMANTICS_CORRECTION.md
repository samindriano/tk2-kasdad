# H12_L semantics correction

## What was wrong

Commit `dbaa3bb0cf3e621daa84537bc8e62f4134153030` contained an incorrect
H12_L implementation in its protocol, synthetic verifier, and preflight
summary. It combined E67 W4, a synthetic R06-only W5, and E75 W6. That mixed
origin view happened to have the expected denominator 20,903, but it was not
H12_L. The E75 prediction could change or hide long-horizon harm, and E67 W5
was incorrectly treated as unchanged R06.

The old artifacts are retained as `PREPARATION_PREFLIGHT_SUPERSEDED_DBAA.json`
and `SYNTHETIC_QA_SUPERSEDED_DBAA.json` in this folder, and are also preserved
in Git history with these hashes:

- `PREPARATION_PREFLIGHT.json`:
  `85e97ae1437ad57cc09a2ee8ee7516473d288e870d12c62484fc86e8145ddcee`
- `SYNTHETIC_QA.json`:
  `d5a993e60e62ff6f90a492efe6f1cb857d3574edc8b6b904426c311c8d2cf2e8`

The twelve tests at that checkpoint passed implementation mechanics but did
not assert view provenance. A matching denominator was therefore insufficient
to detect the semantic error. No official Regression target, official memory,
official candidate prediction, Test row, fit, or Kaggle result was opened or
created under the incorrect view.

## Correct successor definition

- The four planned memories remain unchanged: E67 observed/shuffled and E75
  observed/shuffled.
- E67 uses its past-only donor bank from cycles 56–67 and predicts all saved
  queries in cycles 68–79. This is H12_L, `n=20,903`, compared with the E67
  R06 baseline and E67 shuffled control on those exact rows.
- E75 uses its donor bank from cycles 56–75 and predicts cycles 76–79 only.
- Primary remains E67 W4 (cycles 68–71, `n=4,950`) plus E75 W6 (cycles
  76–79, `n=8,491`), total `n=13,441`.
- E67 W5 (cycles 72–75) uses E67 predictions and is included in H12_L. It is
  not forced to R06. E75 outputs, labels, baseline, and donor bank are excluded
  from H12_L.

## Verification added

The successor preflight inspects the entire saved E67 A02 horizon and confirms
row/ID/cycle alignment with the complete E67 R06 cache; it separately checks
E67 W4 and E75 W6 primary slices. The synthetic verifier now checks the same
origin/cycle provenance, aligns truth by ID + origin + cycle + window, and
uses E67 W5 predictions in H12_L. Tests prove E75 changes cannot affect H12_L,
E75 changes do affect primary W6, E67 W5 corrections are included, cycle 80 is
rejected for E67, and a cycle-68 donor is rejected by the E67 bank boundary.

No weighted-median, alpha, cap, shuffle, or gate rule changed. Historical R06
legacy guard failures remain separate. The fit cap remains 66 and the four-fit
budget request is unchanged and unreserved. This correction does not qualify
the candidate or authorize official fits.
