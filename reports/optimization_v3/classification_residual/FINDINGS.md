# V3 Classification — Nonlinear OOF residual pilot

**Status:** `PASS_CONTINUATION_GATE`. Two heads, four outer fits total; no new base model was fitted. Official Test and Kaggle were not read or used.

The correction heads were trained only on stream-present, past-only OOF blocks: cutoff 55 used cycles 52–55 scored by base fit 51; cutoff 67 used cycles 52–67 scored by base fits 51, 55, 59, and 63. Every block is exactly four next cycles, disjoint, and ends no later than its correction cutoff. The head preprocessor fits only on these OOF rows. Outer cutoffs 55 and 67 reuse cached C2_B and CB_MISSING predictions; missing rows retain CB_MISSING byte-for-byte.

Head A uses C2_B score + expanded structured covariates. Head B adds the weak structured CB score as a fixed ablation. Both use the same frozen LightGBM correction settings, logit(C2_B) init margin, 0.5 shrinkage, and fixed 0.50 threshold. No outer outcomes choose inputs, weights, thresholds, or route.

## Held-out views

The pilot covers W1 and W4 plus their existing H12_E/H12_L views. The pooled primary pilot slice combines W1 and W4 only; it is not the full six-window primary metric. H12 views reuse those same two cutoffs and remain adaptive development evidence.

| View | Head | Present F1 current → candidate (Δ) | All-row F1 current → candidate (Δ) | Missing route exact? |
|---|---|---:|---:|---|
| W1 | A_base_structured | 0.7949 → 0.7951 (+0.0001) | 0.8021 → 0.8022 (+0.0001) | True |
| W1 | B_base_structured_plus_CB_ablation | 0.7949 → 0.7918 (-0.0032) | 0.8021 → 0.7993 (-0.0028) | True |
| W4 | A_base_structured | 0.8469 → 0.8467 (-0.0002) | 0.8403 → 0.8401 (-0.0002) | True |
| W4 | B_base_structured_plus_CB_ablation | 0.8469 → 0.8481 (+0.0012) | 0.8403 → 0.8414 (+0.0011) | True |
| W1_plus_W4_pilot_slice | A_base_structured | 0.8237 → 0.8236 (-0.0001) | 0.8229 → 0.8228 (-0.0001) | True |
| W1_plus_W4_pilot_slice | B_base_structured_plus_CB_ablation | 0.8237 → 0.8229 (-0.0008) | 0.8229 → 0.8222 (-0.0007) | True |
| H12_E | A_base_structured | 0.7912 → 0.7961 (+0.0049) | 0.7927 → 0.7972 (+0.0045) | True |
| H12_E | B_base_structured_plus_CB_ablation | 0.7912 → 0.7953 (+0.0041) | 0.7927 → 0.7964 (+0.0037) | True |
| H12_L | A_base_structured | 0.8080 → 0.8109 (+0.0030) | 0.8037 → 0.8065 (+0.0028) | True |
| H12_L | B_base_structured_plus_CB_ablation | 0.8080 → 0.8121 (+0.0041) | 0.8037 → 0.8075 (+0.0038) | True |
| H12_pooled | A_base_structured | 0.8005 → 0.8045 (+0.0040) | 0.7990 → 0.8027 (+0.0038) | True |
| H12_pooled | B_base_structured_plus_CB_ablation | 0.8005 → 0.8047 (+0.0042) | 0.7990 → 0.8029 (+0.0039) | True |

**Frozen pilot gate:** `PASS`. It requires at least one pooled primary pilot-slice or H12 present-stream F1 gain, no individual W1/W4/H12_E/H12_L present-stream loss below −0.01, and exact missing-route preservation. This gate is a continuation screen, not a model-selection guarantee.

## Provenance and limits

Official Train SHA-256: `875552a68a3d24b0f3e1966d6ddb5671e6f54305a1c526e2f0a76747942ed91c`. Source/protocol/OOF inputs were frozen before fitting with signature `8094ba8898558841d66cda0d99ae442d5d3bca2e327cc14c01c37322a199da47`. The full raw input hashes, OOF row IDs and block denominators are in `results.json`; the ignored run directory contains outer vectors, transformed designs and LightGBM text models for independent reconstruction.

Only cycles 56–59 and 68–71 are represented in the pooled primary pilot slice; no claim about W2/W3/W5/W6 or the full six-window primary score is made. The H12 pooled view overlaps its early and late constituent slices and uses the same chronological dataset already studied. If both heads pass, Head B is not preferred automatically: the weak CB score is an ablation, and a small difference must be weighed against the pre-existing negative conditional diagnostics.

Reproduce with `python scripts/optimization_v3_cls_residual.py --freeze-only` before fits, then `python scripts/optimization_v3_cls_residual.py --run`, then `python scripts/optimization_v3_verify_cls_residual.py`. No Kaggle submission or Git operation is part of this pilot.
