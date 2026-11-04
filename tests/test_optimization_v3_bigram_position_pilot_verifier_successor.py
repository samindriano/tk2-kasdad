import json

import numpy as np

from scripts import optimization_v3_positional_bag_offset_v1 as offset
from scripts import verify_optimization_v3_bigram_position_pilot_v1_successor as verifier


def _fixture(tmp_path):
    origin = 67
    key = f"E{origin}"
    path = (tmp_path / "experiments/runs/optimization_v3_research_next_v1/POS_CONTRAST_V1"
            / key / f"predictions_{key}.npz")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"frozen contrast anchor")
    relative = "experiments/runs/optimization_v3_research_next_v1/POS_CONTRAST_V1/E67/predictions_E67.npz"
    ready_path = (tmp_path / "experiments/runs/optimization_v3_research_next_v1/POS_CONTRAST_V1"
                  / "PREDICTIONS_READY.json")
    ready_path.parent.mkdir(parents=True, exist_ok=True)
    entry = {"path": relative, "sha256": offset.sha256_file(path)}
    ready_path.write_text(json.dumps({"query_labels_opened": False,
                                      "predictions": {key: entry}}), encoding="utf-8")
    frozen = {"parent_prediction_sha256": entry["sha256"],
              "anchor_bundle": {"ready_sha256": offset.sha256_file(ready_path)}}
    return origin, path, frozen


def test_accepts_hash_bound_pos_contrast_anchor(tmp_path):
    origin, path, frozen = _fixture(tmp_path)
    result = verifier.verify_frozen_pos_contrast_anchor(tmp_path, origin, frozen)
    assert result["checks"]["anchor_file_sha256"] is True
    assert result["path"].endswith("POS_CONTRAST_V1/E67/predictions_E67.npz")


def test_rejects_pos_bag_path_when_pos_contrast_anchor_is_required(tmp_path):
    origin, _, frozen = _fixture(tmp_path)
    wrong = (tmp_path / "experiments/runs/optimization_v3_research_next_v1/POS_BAG_OFFSET_V1"
             / "E67/predictions_E67.npz")
    wrong.parent.mkdir(parents=True, exist_ok=True)
    wrong.write_bytes(b"different POS_BAG bundle")
    with np.testing.assert_raises(AssertionError):
        verifier.verify_frozen_pos_contrast_anchor(tmp_path, origin, frozen, requested_path=wrong)


def test_rejects_tampered_pos_contrast_anchor(tmp_path):
    origin, path, frozen = _fixture(tmp_path)
    path.write_bytes(b"tampered anchor")
    with np.testing.assert_raises(AssertionError):
        verifier.verify_frozen_pos_contrast_anchor(tmp_path, origin, frozen)
