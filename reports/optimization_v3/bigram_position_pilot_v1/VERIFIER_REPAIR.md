# Target-blind verifier path repair

The first prelabel verifier invocation returned **FAIL** before opening query labels: `E67 parent prediction file hash mismatch`. The result is preserved in `PRELABEL_VERIFICATION_LEGACY_FAILURE.json`; no model fit, freeze, or gate was changed.

## Diagnosis

The pilot producer's `_load_anchor` reads `POS_CONTRAST_V1/E{origin}/predictions_E{origin}.npz` and stores that file hash in the generic frozen field `parent_prediction_sha256`. The legacy verifier incorrectly compared that value to `POS_BAG_OFFSET_V1/E{origin}/predictions_E{origin}.npz`.

The frozen hash is valid for the intended POS_CONTRAST anchor on both origins:

| Origin | Frozen/actual POS_CONTRAST anchor SHA-256 | Frozen ready-manifest SHA-256 | Result |
|---|---|---|---|
| E67 | `aab800423deb51cb7bdd832fb386153a9b9ff9458635b75a50ff184e19f3ff0b` | `4d29907f657d805b8596edb302c89bb4b9e1777664c43ac510ebb14d66fc75c9` | Match |
| E75 | `50ae10dc1e66c687eb0d11b05f8584d39a5a7dfbe1e2d930c4a0a394c19c240b` | `4d29907f657d805b8596edb302c89bb4b9e1777664c43ac510ebb14d66fc75c9` | Match |

POS_BAG model lineage and original POS gamma receipts were checked separately by the frozen parent integrity verifier and the successor prelabel verifier. The frozen original verifier remains unchanged at SHA-256 `9474c63ed24c9a119be7ffd9ac94991fa821bc6888ca9c2771f8fdfba43b1e8f`.

## Versioned repair and outcome

The versioned verifier is `scripts/verify_optimization_v3_bigram_position_pilot_v1_successor.py`, SHA-256 `1eff6dd9c885a39c9f1591e0bc42f9c77571b1b6fdf35e17ff4fe849386fb492`. It corrects only the anchor path binding, checks both POS_CONTRAST ready entries and hashes, and retains the existing checks for POS_BAG gamma lineage, A02 groups, matrices, candidate routes, fit receipts, and ledger events. New focused tests reject a POS_BAG path in the anchor slot and detect a tampered POS_CONTRAST artifact.

The independent versioned verifier returned **PASS** for E67 and E75 before query labels were opened. Its full result is `PRELABEL_VERIFICATION_SUCCESSOR.json`; `VERIFIER_REPAIR.json` binds the original and successor verifier hashes, freeze signature, and per-origin anchor hashes. Only after that PASS did the pilot's authorized internal Train query-label scoring run. No Test, SampleSubmission, or Kaggle access occurred.

The pilot remains **DIAGNOSTIC_ONLY / gate FAIL**. The versioned path repair does not alter the pilot's original verdict or any scientific threshold.
