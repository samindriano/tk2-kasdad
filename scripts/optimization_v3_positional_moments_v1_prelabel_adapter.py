"""Post-fit, label-closed adapter for a historical parent receipt schema.

The frozen independent verifier expects a ``predictions`` mapping for every
parent. The historical POSITIONAL_TEXT_PILOT_V1 receipt instead stores a list
under ``origin_predictions``. This module validates that list against the
already-frozen parent prediction and manifest paths/hashes, then supplies a
temporary in-memory normalized view to the unchanged frozen verifier.
It never edits the historical receipt or frozen verifier source.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path, PurePosixPath
import sys
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import optimization_v3_positional_moments_v1 as moments
from scripts import verify_optimization_v3_positional_moments_v1 as verifier


ADAPTER_ID = "POS_MOMENTS_V1_PARENT_READY_SCHEMA_ADAPTER_V1"
REPORT = moments.REPORT_ROOT / "PRELABEL_SCHEMA_ADAPTER_V1.json"
RUN_REPORT = moments.RUN_ROOT / "PRELABEL_SCHEMA_ADAPTER_V1.json"
FAILURE_REPORT = moments.REPORT_ROOT / "PRELABEL_SCHEMA_ADAPTER_FAILURE_V1.json"
RUN_FAILURE_REPORT = moments.RUN_ROOT / "PRELABEL_SCHEMA_ADAPTER_FAILURE_V1.json"
FAILURE_RECEIPT = {
    "attempt": "frozen verify-prelabel",
    "exit_code": 1,
    "exception": "KeyError: 'predictions'",
    "location": "scripts/verify_optimization_v3_positional_moments_v1.py::_origin_inputs, parent_ready['predictions']",
    "cause": "POSITIONAL_TEXT_PILOT_V1/PREDICTIONS_READY.json uses origin_predictions list schema",
    "query_labels_opened": False,
    "test_or_sample_access": False,
}


class ParentReceiptError(ValueError):
    """A parent prediction receipt does not match the frozen lineage."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_relative_path(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise ParentReceiptError(f"{label} must be a non-empty relative path")
    path = PurePosixPath(value.replace("\\", "/"))
    if path.is_absolute() or ".." in path.parts or path.as_posix() != value.replace("\\", "/"):
        raise ParentReceiptError(f"{label} is not a canonical repository-relative path")
    return path.as_posix()


def normalize_prediction_index(
    ready: Mapping[str, Any],
    *,
    root: Path,
    expected: Mapping[str, Mapping[str, str]],
    require_parent_manifests: bool,
) -> dict[str, dict[str, str]]:
    """Validate either actual receipt schema and return a normalized index.

    ``expected`` is supplied by the signed POS_MOMENTS freeze, never inferred
    from the receipt being checked. For the historical list schema, the parent
    matrix manifest path/hash is also required and checked against that freeze.
    """
    root = Path(root).resolve()
    expected_keys = {"E67", "E75"}
    if set(expected) != expected_keys:
        raise ParentReceiptError("Expected frozen parent origins must be exactly E67 and E75")
    if ready.get("query_labels_opened") is not False:
        raise ParentReceiptError("Parent receipt does not attest label-closed prediction generation")

    if "origin_predictions" in ready:
        if "predictions" in ready:
            raise ParentReceiptError("Ambiguous parent receipt contains both prediction schemas")
        records = ready.get("origin_predictions")
        if not isinstance(records, list) or len(records) != 2:
            raise ParentReceiptError("Historical origin_predictions must contain exactly two records")
        by_origin: dict[str, Mapping[str, Any]] = {}
        for record in records:
            if not isinstance(record, Mapping):
                raise ParentReceiptError("Historical prediction record must be an object")
            origin = record.get("origin")
            if type(origin) is not int or origin not in (67, 75):
                raise ParentReceiptError(f"Invalid or wrong historical origin: {origin!r}")
            key = f"E{origin}"
            if key in by_origin:
                raise ParentReceiptError(f"Duplicate historical origin {key}")
            by_origin[key] = record
        if set(by_origin) != expected_keys:
            raise ParentReceiptError("Historical prediction records are missing a frozen origin")

        normalized: dict[str, dict[str, str]] = {}
        for key in sorted(expected_keys):
            record = by_origin[key]
            frozen = expected[key]
            declared_prediction = _safe_relative_path(record.get("prediction_path"), label=f"{key} prediction_path")
            declared_manifest = _safe_relative_path(record.get("matrix_manifest_path"), label=f"{key} matrix_manifest_path")
            if declared_prediction != frozen["path"]:
                raise ParentReceiptError(f"{key} parent prediction path differs from frozen path")
            if declared_manifest != frozen["matrix_manifest_path"]:
                raise ParentReceiptError(f"{key} parent matrix manifest path differs from frozen path")
            if record.get("prediction_sha256") != frozen["sha256"]:
                raise ParentReceiptError(f"{key} parent prediction receipt hash differs from frozen hash")
            if record.get("matrix_manifest_sha256") != frozen["matrix_manifest_sha256"]:
                raise ParentReceiptError(f"{key} parent matrix manifest receipt hash differs from frozen hash")
            if _sha256(root / declared_prediction) != frozen["sha256"]:
                raise ParentReceiptError(f"{key} parent prediction file hash differs from frozen hash")
            if _sha256(root / declared_manifest) != frozen["matrix_manifest_sha256"]:
                raise ParentReceiptError(f"{key} parent matrix manifest file hash differs from frozen hash")

            verification_path = _safe_relative_path(
                record.get("prediction_verification_path"), label=f"{key} prediction_verification_path"
            )
            verification_hash = record.get("prediction_verification_sha256")
            if not isinstance(verification_hash, str) or _sha256(root / verification_path) != verification_hash:
                raise ParentReceiptError(f"{key} parent prediction verification receipt hash mismatch")
            normalized[key] = {"path": declared_prediction, "sha256": frozen["sha256"]}
        return normalized

    predictions = ready.get("predictions")
    if not isinstance(predictions, Mapping) or set(predictions) != expected_keys:
        raise ParentReceiptError("Predictions mapping must contain exactly E67 and E75")
    normalized = {}
    for key in sorted(expected_keys):
        record = predictions[key]
        if not isinstance(record, Mapping):
            raise ParentReceiptError(f"{key} prediction mapping record must be an object")
        declared_path = _safe_relative_path(record.get("path"), label=f"{key} path")
        frozen = expected[key]
        if declared_path != frozen["path"] or record.get("sha256") != frozen["sha256"]:
            raise ParentReceiptError(f"{key} prediction mapping differs from frozen path/hash")
        if _sha256(root / declared_path) != frozen["sha256"]:
            raise ParentReceiptError(f"{key} prediction bundle file hash differs from frozen hash")
        if require_parent_manifests:
            raise ParentReceiptError("Historical parent manifest fields are required for this adapter")
        normalized[key] = {"path": declared_path, "sha256": frozen["sha256"]}
    return normalized


def _frozen_parent_expected(freeze_data: Mapping[str, Any]) -> dict[str, dict[str, str]]:
    expected: dict[str, dict[str, str]] = {}
    for origin in (67, 75):
        key = f"E{origin}"
        expected[key] = {
            "path": f"experiments/runs/optimization_v3_research_next_v1/POSITIONAL_TEXT_PILOT_V1/{key}/predictions_{key}.npz",
            "sha256": freeze_data["origins"][key]["prediction_bundle_sha256"],
            "matrix_manifest_path": f"experiments/runs/optimization_v3_research_next_v1/POSITIONAL_TEXT_PILOT_V1/MATRIX_MANIFEST_{key}.json",
            "matrix_manifest_sha256": freeze_data["origins"][key]["manifest_sha256"],
        }
    return expected


def run_prelabel_adapter() -> dict[str, Any]:
    freeze_data = moments._load_freeze()
    parent_ready_path = moments.PILOT_RUN / "PREDICTIONS_READY.json"
    parent_ready = moments._read_json(parent_ready_path)
    if parent_ready.get("status") != "PREDICTIONS_READY_FOR_SCORE_LABEL_OPENING":
        raise ParentReceiptError("Historical parent receipt status is not the frozen label-closed status")

    expected = _frozen_parent_expected(freeze_data)
    normalized = normalize_prediction_index(
        parent_ready, root=moments.ROOT, expected=expected, require_parent_manifests=True
    )
    original_reader = verifier._read_json
    parent_ready_resolved = parent_ready_path.resolve()
    read_count = 0

    def _read_json_with_parent_schema_adapter(path: Path) -> Any:
        nonlocal read_count
        if Path(path).resolve() == parent_ready_resolved:
            read_count += 1
            # Normalize only in memory. The historical parent receipt is untouched.
            return {**parent_ready, "predictions": normalized}
        return original_reader(path)

    verifier._read_json = _read_json_with_parent_schema_adapter
    try:
        result = verifier.verify_prelabel_outputs(moments.ROOT, moments.RUN_ROOT, freeze_data)
    finally:
        verifier._read_json = original_reader
    if read_count != 2:
        raise ParentReceiptError(f"Expected verifier to check the parent receipt for both origins; got {read_count}")
    if result.get("status") != "PASS" or result.get("query_labels_read") is not False:
        raise ParentReceiptError("Unchanged frozen verifier did not return a label-closed PASS")

    adapter_hash = _sha256(Path(__file__))
    frozen_verifier_hash = freeze_data["source_hashes"]["verifier"]
    result.update({
        "verification_method": "unchanged frozen verifier with a read-only, in-memory parent schema adapter",
        "adapter_id": ADAPTER_ID,
        "adapter_source_path": Path(__file__).resolve().relative_to(moments.ROOT).as_posix(),
        "adapter_source_sha256": adapter_hash,
        "frozen_verifier_source_sha256": frozen_verifier_hash,
        "historical_parent_ready_sha256": _sha256(parent_ready_path),
        "parent_schema": "origin_predictions list indexed by explicit origin; never rewritten",
        "parent_origin_mapping": normalized,
        "initial_frozen_verifier_failure": FAILURE_RECEIPT,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    })
    return result


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(payload, sort_keys=True, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temp.replace(path)


def main() -> int:
    moments._atomic_json(FAILURE_REPORT, FAILURE_RECEIPT)
    moments._atomic_json(RUN_FAILURE_REPORT, FAILURE_RECEIPT)
    try:
        result = run_prelabel_adapter()
    except Exception as exc:
        failure = {
            "schema_version": 1, "adapter_id": ADAPTER_ID, "status": "FAIL",
            "error_type": type(exc).__name__, "error": str(exc),
            "adapter_source_sha256": _sha256(Path(__file__)),
            "frozen_verifier_source_sha256": moments._load_freeze()["source_hashes"]["verifier"],
            "freeze_signature": moments._load_freeze()["freeze_signature"],
            "query_labels_read": False, "test_or_sample_access": False,
            "initial_frozen_verifier_failure": FAILURE_RECEIPT,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
        }
        _atomic_json(FAILURE_REPORT, failure)
        _atomic_json(RUN_FAILURE_REPORT, failure)
        raise
    # The standard receipt is the amended PASS consumed by the frozen score
    # entrypoint. The original failed attempt remains separately preserved.
    moments._atomic_json(moments.REPORT_ROOT / "PRELABEL_VERIFICATION.json", result)
    moments._atomic_json(moments.RUN_ROOT / "PRELABEL_VERIFICATION.json", result)
    _atomic_json(REPORT, result)
    _atomic_json(RUN_REPORT, result)
    print(json.dumps(result, sort_keys=True, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
