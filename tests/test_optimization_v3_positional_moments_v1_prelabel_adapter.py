from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.optimization_v3_positional_moments_v1_prelabel_adapter import (
    ParentReceiptError,
    normalize_prediction_index,
)


def _write(root: Path, rel: str, content: bytes) -> str:
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return hashlib.sha256(content).hexdigest()


def _fixtures(tmp_path: Path) -> tuple[dict[str, dict[str, str]], list[dict[str, str | int]]]:
    expected: dict[str, dict[str, str]] = {}
    records: list[dict[str, str | int]] = []
    for origin in (67, 75):
        key = f"E{origin}"
        prediction = f"parent/{key}/predictions_{key}.npz"
        manifest = f"parent/MATRIX_MANIFEST_{key}.json"
        verification = f"parent/PREDICTION_VERIFICATION_{key}.json"
        prediction_sha = _write(tmp_path, prediction, f"prediction-{key}".encode())
        manifest_sha = _write(tmp_path, manifest, json.dumps({"origin": origin}).encode())
        verification_sha = _write(tmp_path, verification, json.dumps({"status": "PASS"}).encode())
        expected[key] = {
            "path": prediction,
            "sha256": prediction_sha,
            "matrix_manifest_path": manifest,
            "matrix_manifest_sha256": manifest_sha,
        }
        records.append({
            "origin": origin,
            "prediction_path": prediction,
            "prediction_sha256": prediction_sha,
            "matrix_manifest_path": manifest,
            "matrix_manifest_sha256": manifest_sha,
            "prediction_verification_path": verification,
            "prediction_verification_sha256": verification_sha,
        })
    return expected, records


def test_historical_origin_predictions_schema_is_indexed_by_origin(tmp_path: Path) -> None:
    expected, records = _fixtures(tmp_path)
    ready = {"status": "PREDICTIONS_READY_FOR_SCORE_LABEL_OPENING",
             "query_labels_opened": False, "origin_predictions": list(reversed(records))}
    got = normalize_prediction_index(ready, root=tmp_path, expected=expected,
                                     require_parent_manifests=True)
    assert got == {key: {"path": expected[key]["path"], "sha256": expected[key]["sha256"]}
                   for key in ("E67", "E75")}


def test_current_predictions_mapping_schema_is_accepted(tmp_path: Path) -> None:
    expected, _ = _fixtures(tmp_path)
    ready = {"query_labels_opened": False, "predictions": {
        key: {"path": item["path"], "sha256": item["sha256"]}
        for key, item in expected.items()
    }}
    got = normalize_prediction_index(ready, root=tmp_path, expected=expected,
                                     require_parent_manifests=False)
    assert got == ready["predictions"]


def test_historical_duplicate_or_missing_origin_is_rejected(tmp_path: Path) -> None:
    expected, records = _fixtures(tmp_path)
    duplicate = {"query_labels_opened": False, "origin_predictions": [records[0], records[0]]}
    with pytest.raises(ParentReceiptError, match="Duplicate historical origin"):
        normalize_prediction_index(duplicate, root=tmp_path, expected=expected,
                                   require_parent_manifests=True)
    missing = {"query_labels_opened": False, "origin_predictions": [records[0]]}
    with pytest.raises(ParentReceiptError, match="exactly two"):
        normalize_prediction_index(missing, root=tmp_path, expected=expected,
                                   require_parent_manifests=True)


@pytest.mark.parametrize("field,value", [
    ("origin", 66),
    ("prediction_path", "parent/E75/predictions_E75.npz"),
    ("prediction_sha256", "0" * 64),
    ("matrix_manifest_sha256", "f" * 64),
])
def test_historical_wrong_origin_path_and_hash_are_rejected(tmp_path: Path, field: str, value: object) -> None:
    expected, records = _fixtures(tmp_path)
    records[0][field] = value  # type: ignore[assignment]
    ready = {"query_labels_opened": False, "origin_predictions": records}
    with pytest.raises(ParentReceiptError):
        normalize_prediction_index(ready, root=tmp_path, expected=expected,
                                   require_parent_manifests=True)


def test_current_predictions_missing_wrong_path_or_hash_is_rejected(tmp_path: Path) -> None:
    expected, _ = _fixtures(tmp_path)
    missing = {"query_labels_opened": False, "predictions": {"E67": {"path": expected["E67"]["path"],
                                                                        "sha256": expected["E67"]["sha256"]}}}
    with pytest.raises(ParentReceiptError, match="exactly E67 and E75"):
        normalize_prediction_index(missing, root=tmp_path, expected=expected,
                                   require_parent_manifests=False)
    wrong = {"query_labels_opened": False, "predictions": {
        key: {"path": item["path"], "sha256": item["sha256"]} for key, item in expected.items()
    }}
    wrong["predictions"]["E75"]["sha256"] = "0" * 64
    with pytest.raises(ParentReceiptError, match="differs from frozen path/hash"):
        normalize_prediction_index(wrong, root=tmp_path, expected=expected,
                                   require_parent_manifests=False)
