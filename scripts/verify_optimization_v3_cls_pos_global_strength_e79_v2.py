"""Independent, data-free verifier for the E79 compatibility attestation/receipt."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
ATTESTATION_REL = "reports/optimization_v3/cls_pos_global_strength_e79_v2/SOURCE_COMPATIBILITY_ATTESTATION.json"
REPLAY_RECEIPT_REL = "experiments/runs/optimization_v3_cls_pos_global_strength_e79_v2/source_compatibility_replay/REPLAY_RECEIPT.json"
AUTH_REL = "reports/optimization_v3/cls_pos_global_strength_e79_v1/SOL_COMPATIBILITY_REPLAY_AUTHORIZATION.md"
AUTH_SHA256 = "9d0729d1f19793dfb79cc3024be868e44f8ff97020147cd0af2d91ff5df6b6e3"
EXPECTED = {
    "base_shape": [50_608, 20_427],
    "base_nnz": 3_855_766,
    "base_logical_sha256": "9d47aa36ff15a0cbeead8cf36049f0a699f7618d0ea8890806a2d82140cff59b",
    "position_shape": [50_608, 17_259],
    "position_nnz": 1_638_979,
    "position_logical_sha256": "32268550b8220dbfb43f5c92c33cd84554f0bced446d1459c5c55c3eabbf9942",
    "ordered_train_ids_sha256": "9a96fa28e72b08a6022dd005aed877dadcc5e3dd5053200bcdd12711312c26b2",
}


class VerificationError(RuntimeError):
    pass


def _canonical_sha(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False,
                         separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise VerificationError(f"Expected JSON object: {path}")
    return value


def _check_files(root: Path, field: str, manifest: dict[str, str]) -> None:
    for relative, expected in manifest.items():
        path = root / relative
        if not path.is_file():
            raise VerificationError(f"Manifest file missing: {relative}")
        actual = _sha256_file(path)
        if actual != expected:
            raise VerificationError(f"{field} hash mismatch for {relative}: {actual}")


def verify_attestation(root: Path = ROOT) -> dict[str, Any]:
    path = root / ATTESTATION_REL
    if not path.is_file():
        raise VerificationError("Compatibility attestation is missing")
    data = _read_json(path)
    body = dict(data)
    supplied = body.pop("attestation_sha256", None)
    if supplied != _canonical_sha(body):
        raise VerificationError("Compatibility attestation self-digest failed")
    if data.get("status") != "PASS_SOURCE_COMPATIBILITY_ATTESTATION_R2_PENDING_REPLAY":
        raise VerificationError("Unexpected attestation status")
    if _sha256_file(root / AUTH_REL) != AUTH_SHA256:
        raise VerificationError("Authorization document hash changed")
    _check_files(root, "current source", data.get("current_source_sha256", {}))
    _check_files(root, "immutable input", data.get("saved_inputs_and_fitted_state_receipts", {}).get("immutable_file_sha256", {}))
    expected_matrix = data.get("saved_inputs_and_fitted_state_receipts", {}).get("matrix_reference", {})
    if (
        expected_matrix.get("base_shape") != EXPECTED["base_shape"]
        or expected_matrix.get("base_nnz") != EXPECTED["base_nnz"]
        or expected_matrix.get("base_logical_sha256") != EXPECTED["base_logical_sha256"]
        or expected_matrix.get("position_shape") != EXPECTED["position_shape"]
        or expected_matrix.get("position_nnz") != EXPECTED["position_nnz"]
        or expected_matrix.get("position_logical_sha256") != EXPECTED["position_logical_sha256"]
        or expected_matrix.get("ordered_train_ids_sha256") != EXPECTED["ordered_train_ids_sha256"]
    ):
        raise VerificationError("Attestation does not bind the exact historical matrix audit")
    return {
        "status": "PASS_INDEPENDENT_DATA_FREE_ATTESTATION_VERIFICATION",
        "attestation_sha256": supplied,
        "current_source_files_verified": len(data.get("current_source_sha256", {})),
        "immutable_input_files_verified": len(data.get("saved_inputs_and_fitted_state_receipts", {}).get("immutable_file_sha256", {})),
        "matrix_reference": expected_matrix,
        "data_read_by_verifier": False,
    }


def verify_replay_receipt(root: Path = ROOT, receipt_path: Path | None = None) -> dict[str, Any]:
    attestation_result = verify_attestation(root)
    path = receipt_path or root / REPLAY_RECEIPT_REL
    receipt = _read_json(path)
    body = dict(receipt)
    supplied = body.pop("receipt_sha256", None)
    if supplied != _canonical_sha(body):
        raise VerificationError("Replay receipt self-digest failed")
    if receipt.get("status") != "PASS_SAVED_STATE_BASE_AND_POSITION_MATRIX_PARITY":
        raise VerificationError(f"Feature replay did not pass: {receipt.get('status')}")
    if (
        receipt.get("source_compatibility_attestation_sha256") != attestation_result["attestation_sha256"]
        or receipt.get("target_column_requested_or_loaded") is not False
        or receipt.get("test_or_sample_loaded") is not False
        or receipt.get("estimator_fit_calls") != 0
        or receipt.get("prediction_or_inference_calls") != 0
        or receipt.get("new_df_map_created") is not False
        or receipt.get("new_gamma_created") is not False
        or receipt.get("csv_created") is not False
    ):
        raise VerificationError("Replay receipt crossed a forbidden boundary or has wrong attestation")
    matrix = receipt.get("matrices", {})
    base, position = matrix.get("base", {}), matrix.get("position", {})
    if (
        base.get("shape") != EXPECTED["base_shape"]
        or base.get("nnz") != EXPECTED["base_nnz"]
        or base.get("logical_sha256") != EXPECTED["base_logical_sha256"]
        or position.get("shape") != EXPECTED["position_shape"]
        or position.get("nnz") != EXPECTED["position_nnz"]
        or position.get("logical_sha256") != EXPECTED["position_logical_sha256"]
        or receipt.get("ordered_train_ids_sha256") != EXPECTED["ordered_train_ids_sha256"]
        or not matrix.get("checks")
        or not all(matrix["checks"].values())
    ):
        raise VerificationError("Replay matrix evidence does not match exact immutable reference")
    return {
        "status": "PASS_INDEPENDENT_DATA_FREE_REPLAY_VERIFICATION",
        "attestation_sha256": attestation_result["attestation_sha256"],
        "replay_receipt_sha256": supplied,
        "matrices": {"base": base, "position": position},
        "data_read_by_verifier": False,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replay", action="store_true", help="verify the one saved replay receipt")
    parser.add_argument("--receipt", type=Path, help="optional explicit replay receipt path")
    args = parser.parse_args(argv)
    result = verify_replay_receipt(receipt_path=args.receipt) if args.replay else verify_attestation()
    print(json.dumps(result, sort_keys=True, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
