import copy
import json

import numpy as np
import pytest

from scripts import optimization_v3_rnbr_continuation as runner
from scripts import optimization_v3_rnbr_official as pilot
from scripts.optimization_v3_rnbr_continuation_corrected_eval import assemble_prediction_maps
from scripts.optimization_v3_rnbr_continuation_recovery_bridge import BridgeError, assert_frozen_source_bytes


def _bundle(rows, values):
    return {"score_rows": np.asarray(rows, dtype=np.int64),
            "prediction": np.asarray(values, dtype=np.float64)}


def _fixture():
    view_rows = {
        "W1": np.array([1]), "W2": np.array([2]), "W3": np.array([3]),
        "W4": np.array([4]), "W5": np.array([5]), "W6": np.array([6]),
        "new_origin_aggregate": np.array([2, 3, 5]),
        "full_primary": np.array([1, 2, 3, 4, 5, 6]),
        "late_W5_W6": np.array([5, 6]),
        "inherited_H12_L_E67": np.array([4, 5, 6]),
    }
    continuation = {
        "RNBR_CONT_E59_OBSERVED": _bundle([2], [59]),
        "RNBR_CONT_E59_SHUFFLED": _bundle([2], [590]),
        "RNBR_CONT_E63_OBSERVED": _bundle([3], [63]),
        "RNBR_CONT_E63_SHUFFLED": _bundle([3], [630]),
        "RNBR_CONT_E71_OBSERVED": _bundle([5], [71]),
        "RNBR_CONT_E71_SHUFFLED": _bundle([5], [710]),
    }
    pilots = {
        (67, "observed"): _bundle([4, 5, 6], [67, 670, 671]),
        (67, "shuffled"): _bundle([4, 5, 6], [167, 1670, 1671]),
        (75, "observed"): _bundle([6], [75]),
        (75, "shuffled"): _bundle([6], [750]),
    }
    r06 = {origin: {1: 55, 2: 59, 3: 63, 4: 67, 5: 71, 6: 75}
           for origin in (55, 59, 63, 67, 71, 75)}
    return view_rows, continuation, pilots, r06


def test_e67_h12_map_cannot_overwrite_primary_w5_or_w6():
    rows, new, pilots, r06 = _fixture()
    before = assemble_prediction_maps(rows, new, pilots, r06, rows["inherited_H12_L_E67"])
    obs, _, h12, _ = before
    assert obs["W5"] == {5: 71.0}
    assert obs["W6"] == {6: 75.0}
    assert h12 == {4: 67.0, 5: 670.0, 6: 671.0}

    changed_e67 = copy.deepcopy(pilots)
    changed_e67[(67, "observed")]["prediction"][1:] += 10000
    after = assemble_prediction_maps(rows, new, changed_e67, r06, rows["inherited_H12_L_E67"])
    obs_after, _, h12_after, _ = after
    assert obs_after["W5"] == obs["W5"]
    assert obs_after["W6"] == obs["W6"]
    assert h12_after != h12


def test_e75_changes_w6_without_changing_h12():
    rows, new, pilots, r06 = _fixture()
    before = assemble_prediction_maps(rows, new, pilots, r06, rows["inherited_H12_L_E67"])
    changed_e75 = copy.deepcopy(pilots)
    changed_e75[(75, "observed")]["prediction"] += 10000
    after = assemble_prediction_maps(rows, new, changed_e75, r06, rows["inherited_H12_L_E67"])
    assert after[0]["W6"] != before[0]["W6"]
    assert after[2] == before[2]


def test_valid_bundle_uses_fit_commit_not_unrelated_current_head(tmp_path, monkeypatch):
    fit_id = "RNBR_CONT_E59_OBSERVED"
    source_commit = "a" * 40
    freeze = {"freeze_signature": "b" * 64}
    arrays = {"score_rows": np.array([10, 11], dtype=np.int64),
              "prediction": np.array([42.0, 43.0]),
              "baseline_prediction": np.array([40.0, 41.0])}
    folder = tmp_path / "bundle"
    folder.mkdir()
    npz_path = folder / "arrays.npz"
    np.savez_compressed(npz_path, **arrays)
    identity = {"fit_id": fit_id, "freeze_signature": freeze["freeze_signature"],
                "source_commit": source_commit}
    receipt = {
        "identity": identity,
        "identity_sha256": pilot.sha256_bytes(runner._canonical(identity)),
        "npz_sha256": runner._sha(npz_path),
        "array_shapes": {k: list(v.shape) for k, v in arrays.items()},
        "array_dtypes": {k: str(v.dtype) for k, v in arrays.items()},
    }
    receipt["integrity_sha256"] = pilot.sha256_bytes(runner._canonical(receipt))
    receipt_path = folder / "receipt.json"
    original_receipt = json.dumps(receipt, sort_keys=True)
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
    monkeypatch.setattr(runner, "_fit_dir", lambda _: folder)

    loaded, _ = runner.load_bundle(fit_id, freeze, source_commit=source_commit)
    assert np.array_equal(loaded["prediction"], arrays["prediction"])

    bad_receipt = dict(receipt)
    bad_receipt["integrity_sha256"] = "0" * 64
    receipt_path.write_text(json.dumps(bad_receipt), encoding="utf-8")
    with pytest.raises(runner.ContinuationError, match="receipt/hash/identity mismatch"):
        runner.load_bundle(fit_id, freeze, source_commit=source_commit)
    receipt_path.write_text(original_receipt, encoding="utf-8")

    with pytest.raises(runner.ContinuationError, match="receipt/hash/identity mismatch"):
        runner.load_bundle(fit_id, {"freeze_signature": "c" * 64}, source_commit=source_commit)

    with npz_path.open("ab") as stream:
        stream.write(b"tamper")
    with pytest.raises(runner.ContinuationError, match="receipt/hash/identity mismatch"):
        runner.load_bundle(fit_id, freeze, source_commit=source_commit)


def test_frozen_source_comparison_is_exact_bytewise():
    assert_frozen_source_bytes(b"source", b"source", "frozen.py") is None
    with pytest.raises(BridgeError, match="Frozen fit source differs"):
        assert_frozen_source_bytes(b"source", b"changed", "frozen.py")

