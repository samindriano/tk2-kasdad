# R04 report index

The pilot decision and metrics are in `results.json` and `gate.json`. The first reporter left `fit_accounting.folds[*].bundle_npz_sha256` null because it read the wrong sidecar key; no model output or metric was affected. The verified, corrected accounting is in `report_amendment_01/results.json`, with independently checked NPZ hashes in `report_amendment_01/amendment.json`. Use that amendment for bundle-hash citations. It did not refit either model.
