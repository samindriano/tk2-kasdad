# B02 new-origin scope reconciliation

**Status:** outcome-aware reporting correction; no new selection or experiment.

The original B02 `new_origin_aggregate` incorrectly used cycles 68-79 (E67/E71/E75), n=20,903. The frozen historical B01 definition is W2/W3/W5: E59/E63/E71 first-four windows, cycles 60-67 and 72-75, n=21,033. The original B02 artifacts remain unchanged.

For B01 on the corrected cohort, fixed 0.50 Macro F1=0.837706; plug-in Macro F1=0.837910; change=+0.000204. The corrected >=0.0005 new-origin gate is FAIL.

The overall diagnostic remains ineligible: the corrected new-origin gain is below its gate, H12 is negative, and only 3/5 origin windows are nonnegative. No K, probability, or prediction changed. Full-primary, late, and H12 values are copied from the original B02 result. See `CORRECTION.json` for all source arms, confusion matrices, subgroups, hashes, K identity checks, and QA.

Fits added: 0. Ledger SHA256 remains `6fcc4b4d427e3ff03139e0e9ee429f529f7c7bf6ad91ce0eb9a52d55e2503147`. No Test/Sample/Kaggle access.
