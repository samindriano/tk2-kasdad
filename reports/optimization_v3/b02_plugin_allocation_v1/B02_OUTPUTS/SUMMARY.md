# B02 label-free plug-in Macro F1 allocation

**Status:** COMPLETED_ADAPTIVE_DIAGNOSTIC

This is a zero-fit adaptive diagnostic. For each fixed probability source and batch, K was chosen from expected two-class confusion counts using probabilities and stream-availability masks only. Missing-stream hard labels and W1 stayed at the existing 0.50 route. Query labels were read only after all K values and predictions were fixed.

The method is narrowly distinct: it selects whole tied-score K values using plug-in expected two-class Macro F1. Prior work already studied label-tuned thresholds, fixed-0.50 calibration heads, and output-count matching. This result is not a novel general thresholding method.

The objective is Macro F1 of the expected confusion counts under the supplied probabilities. It is not the exact expectation of Macro F1 and does not guarantee a beneficial or Bayes-optimal decision rule. H12 overlaps primary rows and is a separate scenario, not an independent holdout.

## B01 main source

Full primary: unchanged 0.50=0.832306; plug-in=0.833299; Δ=+0.000994 (n=38,588).
New-origin aggregate: unchanged=0.825865; plug-in=0.829486; Δ=+0.003621 (n=20,903).
Late: unchanged=0.818720; plug-in=0.823033; Δ=+0.004312.
E67 H12: unchanged=0.815049; plug-in=0.809717; Δ=-0.005332.

All K values, soft objectives, confusion matrices, origin/window metrics, subgroup guards, and A02/Sparse25/shuffled sensitivities are in RESULTS.json. Freeze, hashes, no-fit accounting, and independent QA are in FREEZE.json and QA.json.

Eligibility is only for further review; it does not qualify a candidate, authorize submission, or select a champion. No public/test metric was used.

## Fixed eligibility checks

- PASS — full primary gain vs unchanged B01 ge 0 0005
- PASS — new origin gain vs unchanged B01 ge 0 0005
- FAIL — late and H12 B01 gains nonnegative
- FAIL — at least 4 of 5 origin gains nonnegative
- PASS — no primary window harm over 0 003
- PASS — no supported full primary subgroup harm over fixed guards
