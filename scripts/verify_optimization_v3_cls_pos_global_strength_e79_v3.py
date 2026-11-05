"""Independent, data-free verifier for the E79 R3 attestation and replay receipt."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
ATTESTATION_REL = "reports/optimization_v3/cls_pos_global_strength_e79_v3/SOURCE_COMPATIBILITY_ATTESTATION.json"
REPLAY_RECEIPT_REL = "experiments/runs/optimization_v3_cls_pos_global_strength_e79_v3/source_compatibility_replay/REPLAY_RECEIPT.json"
AUTH_REL = "reports/optimization_v3/cls_pos_global_strength_e79_v1/SOL_COMPATIBILITY_REPLAY_AUTHORIZATION.md"
AUTH_SHA256 = "9d0729d1f19793dfb79cc3024be868e44f8ff97020147cd0af2d91ff5df6b6e3"
TRAIN_REL = "data/raw/classification/Train.csv"
OFFICIAL_EXPECTED_MATRIX = {
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
    if not isinstance(manifest, dict) or not manifest:
        raise VerificationError(f"{field} manifest is empty or invalid")
    for relative, expected in manifest.items():
        path = root / relative
        if not path.is_file():
            raise VerificationError(f"Manifest file missing: {relative}")
        actual = _sha256_file(path)
        if actual != expected:
            raise VerificationError(f"{field} hash mismatch for {relative}: {actual}")


def verify_attestation(
    root: Path = ROOT,
    *,
    expected_matrix: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Validate source/input hashes and the pinned expected matrix without parsing data files."""
    path = root / ATTESTATION_REL
    data = _read_json(path)
    body = dict(data)
    supplied = body.pop("attestation_sha256", None)
    if supplied != _canonical_sha(body):
        raise VerificationError("Compatibility attestation self-digest failed")
    if data.get("status") != "PASS_SOURCE_COMPATIBILITY_ATTESTATION_R3_PENDING_REPLAY":
        raise VerificationError("Unexpected R3 attestation status")
    authorization = data.get("authorization", {})
    if authorization.get("path") != AUTH_REL or authorization.get("sha256") != AUTH_SHA256:
        raise VerificationError("R1 feature-only authorization pin changed")
    if _sha256_file(root / AUTH_REL) != AUTH_SHA256:
        raise VerificationError("Authorization document hash changed")
    current_sources = data.get("current_source_sha256", {})
    immutable = data.get("saved_inputs_and_fitted_state_receipts", {}).get("immutable_file_sha256", {})
    _check_files(root, "current source", current_sources)
    _check_files(root, "immutable input", immutable)
    saved = data.get("saved_inputs_and_fitted_state_receipts", {})
    matrix = saved.get("matrix_reference", {})
    expected = expected_matrix or OFFICIAL_EXPECTED_MATRIX
    if any(matrix.get(key) != value for key, value in expected.items()):
        raise VerificationError("Attestation does not bind the exact expected matrix reference")
    r2 = saved.get("preserved_r2_failed_attempt", {})
    if (
        r2.get("status") != "FAILED_FEATURE_ONLY_REPLAY_NO_RETRY"
        or r2.get("replay_receipt_file_sha256") != "1a001f7843dfa19356adab62b0eadd21aecf1b0be8fa83f3db63a9a354c68ff9"
        or r2.get("replay_receipt_internal_digest_present") is not True
        or r2.get("replay_receipt_internal_sha256") != "7245c94a097f3440fa43938b2d798646f7d31d2f1324354144867cae54762031"
        or r2.get("matrix_digests_persisted") is not False
    ):
        raise VerificationError("Attestation does not preserve exact R2 failed-receipt status")
    return {
        "status": "PASS_INDEPENDENT_DATA_FREE_ATTESTATION_VERIFICATION",
        "attestation_sha256": supplied,
        "current_source_files_verified": len(current_sources),
        "immutable_input_files_verified": len(immutable),
        "matrix_reference": matrix,
        "file_bytes_hashed_data_not_parsed": True,
    }


def verify_replay_receipt(
    root: Path = ROOT,
    receipt_path: Path | None = None,
    *,
    expected_matrix: dict[str, Any] | None = None,
) -> dict[str, Any]:
    attestation_result = verify_attestation(root, expected_matrix=expected_matrix)
    attestation = _read_json(root / ATTESTATION_REL)
    path = receipt_path or root / REPLAY_RECEIPT_REL
    receipt = _read_json(path)
    receipt_body = dict(receipt)
    supplied = receipt_body.pop("receipt_sha256", None)
    if supplied != _canonical_sha(receipt_body):
        raise VerificationError("Replay receipt self-digest failed")
    if receipt.get("status") != "PASS_SAVED_STATE_BASE_AND_POSITION_MATRIX_PARITY":
        raise VerificationError(f"Feature replay did not pass: {receipt.get('status')}")
    saved = attestation["saved_inputs_and_fitted_state_receipts"]
    immutable = saved["immutable_file_sha256"]
    if (
        receipt.get("source_compatibility_attestation_sha256") != attestation_result["attestation_sha256"]
        or receipt.get("source_sha256") != attestation.get("current_source_sha256")
        or receipt.get("input_sha256") != immutable
        or receipt.get("source_commit_parent") != attestation.get("prepared_from_commit")
        or receipt.get("target_column_requested_or_loaded") is not False
        or receipt.get("test_or_sample_loaded") is not False
        or receipt.get("estimator_fit_calls") != 0
        or receipt.get("prediction_or_inference_calls") != 0
        or receipt.get("new_df_map_created") is not False
        or receipt.get("new_gamma_created") is not False
        or receipt.get("csv_created") is not False
    ):
        raise VerificationError("Replay receipt crossed a forbidden boundary or has wrong lineage")
    matrix = receipt.get("matrices", {})
    base, position = matrix.get("base", {}), matrix.get("position", {})
    expected = expected_matrix or OFFICIAL_EXPECTED_MATRIX
    if (
        base.get("shape") != expected["base_shape"]
        or base.get("nnz") != expected["base_nnz"]
        or base.get("logical_sha256") != expected["base_logical_sha256"]
        or position.get("shape") != expected["position_shape"]
        or position.get("nnz") != expected["position_nnz"]
        or position.get("logical_sha256") != expected["position_logical_sha256"]
        or receipt.get("ordered_train_ids_sha256") != expected["ordered_train_ids_sha256"]
    ):
        raise VerificationError("Receipt matrix hashes are not the exact expected actual-matrix metadata")
    required_checks = {
        "base_shape", "base_nnz", "base_logical_sha256", "position_shape", "position_nnz",
        "position_logical_sha256", "ordered_train_ids_sha256", "finite_base", "finite_position",
    }
    checks = matrix.get("checks")
    if not isinstance(checks, dict) or set(checks) != required_checks or not all(value is True for value in checks.values()):
        raise VerificationError("Receipt parity check set is incomplete or contains a failed check")
    rows = expected["base_shape"][0]
    train_sha = immutable.get(TRAIN_REL)
    if receipt.get("train_rows") != rows or receipt.get("train_sha256") != train_sha:
        raise VerificationError("Receipt Train population does not match the attested feature input")
    return {
        "status": "PASS_INDEPENDENT_DATA_FREE_REPLAY_VERIFICATION",
        "attestation_sha256": attestation_result["attestation_sha256"],
        "replay_receipt_sha256": supplied,
        "matrices": {"base": base, "position": position},
        "file_bytes_hashed_data_not_parsed": True,
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
