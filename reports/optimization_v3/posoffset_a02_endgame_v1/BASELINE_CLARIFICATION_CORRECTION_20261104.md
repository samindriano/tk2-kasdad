# Correction to baseline clarification — 2026-11-04

The `UNKNOWN_UNAVAILABLE_COMPONENT_DECOMPOSITION` exception in
`BASELINE_CLARIFICATION.md` came from an audit of alternate C16 cache artifacts,
not the exact artifacts bound by the frozen A02 preflight. It is superseded for
this endgame and must not be used to relax the required reconstruction.

The authoritative A02 `CACHE_PATHS` in
`scripts/optimization_v3_a02_neighbors.py` point to these C16 caches:

| Cutoff | Exact path | NPZ SHA-256 | Sidecar SHA-256 |
|---:|---|---|---|
| 55 | `experiments/runs/optimization_v3_batch1/C16/C16_fit55.npz` | `7d900c7f66ebcd6f0fcb9d059534bb15cfa47d8073f1119bf046d4a9d0fd54c9` | `e569229e0586bb792b141b64109f8753e97bb6cf8902d94d16b8a114d8d67e20` |
| 59 | `experiments/runs/optimization_v3_final_continuation/C16/C16_fit59.npz` | `bff872a0c5de661a7e2997feb3d1953bbc2caaf1a6a7eedf87ea99f75e15cf26` | `28f602286af2e71542a17d8efb4c91c9914297bd8f8746e8f2dacdbf67334d42` |
| 63 | `experiments/runs/optimization_v3_final_continuation/C16/C16_fit63.npz` | `9194845881e823a010505f253b7e08476c6efeef5a209b9778c058186a700aa0` | `ff84c73861bd8fa107605292b9f69446adb200aea3db009f4b5e0cf088f3eabc` |
| 67 | `experiments/runs/optimization_v3_batch1/C16/C16_fit67.npz` | `7aa025dc1c49d8a76c63d5e8efe27dcbafb600bf0f64a8ed3cbec8db82d31ab3` | `034df0ee0fea7302906638e34fb991fae2a09f1c0f5fba29ed8d469a58c8bf6f` |
| 71 | `experiments/runs/optimization_v3_final_continuation/C16/C16_fit71.npz` | `94e69d5bd83b5edb1ae0337cfe78e8b59a61f2733bcbb580f7c810104f50aa47` | `501784414c91558b15ff48a94dbd9e5a02eb719b0cf1c27ebc4ff21533a7c158` |
| 75 | `experiments/runs/optimization_v3_final_continuation/C16/C16_fit75.npz` | `5e5f25c35cd15b9332c950b78fe8b650cc6a8138a18d6546fa1728ac35c88708` | `7a6f81eeb3d5d87ea53dfd6226a005eebc16415ea7346d695c96b9d061b592ce` |

Each exact NPZ contains `control__c2b_probability` and
`control__cb_probability`; the frozen `_load_cache` requires both. All six
sidecars bind the C16 cutoff and exact NPZ hash. An independent no-target check
reconstructed each `baseline_prediction` bit-exactly as
`where(stream_missing, control__cb_probability,
0.75*control__c2b_probability + 0.25*control__baseline_sparse_raw)`.

For both E67 and E75, the original A02 `baseline_prediction` is bit-exactly
equal to the matching C16 baseline on the same rows. POS and A02 row positions,
IDs, cycles, and stream masks align exactly. This correction records the
wrong-path selection as the cause of the prior apparent missing-key blocker;
it does not alter any historical A02 or POS verdict, model, output, or gate.

The required endgame audit still must independently verify the fitted A02
state, OOF donor residual bank, saved neighbors/similarities, capped correction,
and final probability before any new policy metrics are read.
