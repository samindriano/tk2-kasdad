"""Independent source and V2 DF-map verification without labels or Test."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from scripts import optimization_v3_cls_pos_global_strength_e79_dfmap_resource_v2 as v2
from scripts import optimization_v3_cls_pos_global_strength_e79_dfmap_v1 as v1
from scripts import verify_optimization_v3_cls_pos_global_strength_e79_dfmap_v1 as v1_verify


ROOT = Path(__file__).resolve().parents[1]


class VerificationError(RuntimeError):
    """Raised when V2 source or persisted map evidence is inconsistent."""


def verify_resource_preflight(resource: dict[str, Any], locks: list[dict[str, Any]]) -> dict[str, Any]:
    """Independently evaluate one locked resource snapshot before map creation."""
    lock_ok = (len(locks) == 2 and all(item.get("acquired") is True for item in locks)
               and len({item.get("path") for item in locks}) == 2)
    sparse_bound = resource.get("sparse_union_extra_upper_bound", {}).get("total_bytes", 2**63)
    checks = {
        "both_project_locks_acquired": lock_ok,
        "resource_gate_pass_flag": resource.get("gate_pass") is True,
        "initial_available_at_least_1_5_gib": resource.get("available_ram_bytes", 0) >= v2.MIN_AVAILABLE_BYTES,
        "projected_free_at_least_768_mib": resource.get("projected_free_bytes", -1) >= v2.FREE_RESERVE_BYTES,
        "sparse_union_bound_at_most_160_mib": sparse_bound <= v2.EXTRA_SPARSE_BOUND_LIMIT_BYTES,
        "estimate_is_not_claimed_as_measured_peak": resource.get("estimate_is_measured_df_map_peak") is False,
        "estimate_and_reserve_remain_frozen": (
            resource.get("incremental_memory_estimate_bytes") == 768 * v2.MIB
            and resource.get("required_free_reserve_bytes") == 768 * v2.MIB
        ),
    }
    return {
        "status": "PASS_INDEPENDENT_RESOURCE_PREFLIGHT" if all(checks.values()) else "BLOCKED_RESOURCE_PREFLIGHT",
        "checks": checks,
        "resource_snapshot_sha256": v2._canonical_sha256(resource),
        "lock_paths": [item.get("path") for item in locks],
        "targets_test_sample_fit_or_kaggle_access": False,
    }


def verify_freeze(root: Path = ROOT) -> dict[str, Any]:
    return v2.verify_source_freeze(root)


def verify_map(root: Path = ROOT) -> dict[str, Any]:
    freeze_result = verify_freeze(root)
    v2_attempt, v1_attempt, map_path, receipt_path = v2._output_paths(root)
    if not v2_attempt.is_file() or not map_path.is_file() or not receipt_path.is_file():
        raise VerificationError("V2 map lacks its one-shot attempt, shared artifact, or shared receipt")
    attempt = json.loads(v2_attempt.read_text(encoding="utf-8"))
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    body = dict(receipt)
    receipt_digest = body.pop("receipt_sha256", None)
    if receipt_digest != v2._canonical_sha256(body):
        raise VerificationError("V2 DF-map receipt self-digest failed")
    if (v1_attempt != v2_attempt
            or attempt.get("policy_id") != v2.POLICY_ID
            or attempt.get("status") != "ONE_GLOBAL_DF_MAP_ATTEMPT_STARTED_V2"
            or attempt.get("no_retry_across_v1_or_v2") is not True
            or receipt.get("status") != "PASS_E79_DF_MAP_CREATED_TRAIN_ONLY"
            or receipt.get("policy_id") != v2.POLICY_ID
            or receipt.get("dfmap_id") != v1.DFMAP_ID
            or receipt.get("dfmap_resource_v2_freeze_sha256") != freeze_result["freeze_sha256"]
            or receipt.get("dfmap_freeze_sha256") != freeze_result["freeze_sha256"]
            or receipt.get("v1_freeze_sha256") != v2.V1_FREEZE_SHA256
            or receipt.get("no_retry_across_v1_or_v2") is not True
            or receipt.get("v1_runner_invoked") is not False
            or receipt.get("global_v1_v2_map_fit_count") != 1):
        raise VerificationError("V2 receipt violates its frozen single-pass/source identity")
    if (receipt.get("train_sha256") != v2.r3.TRAIN_SHA256
            or receipt.get("train_rows") != v2.r3.EXPECTED_ROWS
            or receipt.get("target_column_requested_or_loaded") is not False
            or receipt.get("test_or_sample_loaded") is not False
            or receipt.get("estimator_fit_calls") != 0
            or receipt.get("prediction_or_inference_calls") != 0
            or receipt.get("vectorizer_fit_calls") != 0
            or receipt.get("structured_preprocessor_fit_calls") != 0
            or receipt.get("newly_learned_preprocessing_fit_calls") != 1
            or receipt.get("df_map_formula_fit_calls") != 1
            or receipt.get("csv_created") is not False
            or receipt.get("kaggle_contacted") is not False):
        raise VerificationError("V2 DF-map receipt violates frozen data/fit boundaries")
    independent_gate = receipt.get("independent_resource_preflight", {})
    if (independent_gate.get("status") != "PASS_INDEPENDENT_RESOURCE_PREFLIGHT"
            or independent_gate.get("resource_snapshot_sha256") != v2._canonical_sha256(receipt.get("resource_before", {}))
            or not all(independent_gate.get("checks", {}).values())):
        raise VerificationError("Independent resource preflight did not pass for the exact fit snapshot")
    after = receipt.get("resource_after", {})
    if after.get("available_ram_bytes", 0) < v2.FREE_RESERVE_BYTES:
        raise VerificationError("Post-map available memory fell below the frozen 768 MiB reserve")
    if receipt.get("artifact_path") != map_path.relative_to(root).as_posix():
        raise VerificationError("V2 map artifact path is outside the shared canonical namespace")
    if v1.sha256_file(map_path) != receipt.get("artifact_sha256"):
        raise VerificationError("V2 map artifact file hash differs from its receipt")
    matrix = receipt.get("position_matrix", {})
    if (matrix.get("shape") != list(v2.r3.POSITION_SHAPE)
            or matrix.get("nnz") != v2.r3.POSITION_NNZ
            or matrix.get("logical_sha256") != v2.r3.POSITION_SHA256
            or matrix.get("data_dtype") != "float64"
            or matrix.get("indices_dtype") not in {"int32", "int64"}
            or matrix.get("indptr_dtype") not in {"int32", "int64"}
            or matrix.get("sparse_union_extra_upper_bound", {}).get("total_bytes", v2.EXTRA_SPARSE_BOUND_LIMIT_BYTES + 1) > v2.EXTRA_SPARSE_BOUND_LIMIT_BYTES):
        raise VerificationError("V2 actual matrix dtype/allocation evidence differs from the approved bounds")
    formula = receipt.get("formula")
    if formula != v1.EXPECTED_FORMULA:
        raise VerificationError("V2 changed the frozen V1 DF-map formula")
    size = int(receipt["statistics"]["retained_unigram_vocabulary_size"])
    array_result = v1_verify.verify_npz_arrays(
        map_path, expected_size=size, expected_rows=v2.r3.EXPECTED_ROWS,
        statistics=receipt["statistics"],
    )
    return {
        "status": "PASS_INDEPENDENT_DFMAP_RESOURCE_V2_VERIFICATION",
        "policy_id": v2.POLICY_ID,
        "freeze_sha256": freeze_result["freeze_sha256"],
        "receipt_sha256": receipt_digest,
        "receipt_file_sha256": v1.sha256_file(receipt_path),
        "artifact_sha256": v1.sha256_file(map_path),
        "formula_verification": array_result,
        "independent_resource_preflight": independent_gate,
        "v1_runner_invoked": False,
        "shared_attempt_marker_owned_by_v2": True,
        "global_map_fit_count": 1,
        "target_test_sample_or_kaggle_access": False,
    }


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--verify-freeze", action="store_true")
    group.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    try:
        result = verify_freeze() if args.verify_freeze else verify_map()
    except (VerificationError, v2.ResourceV2Error, v1_verify.VerificationError, v1.DFMapError) as exc:
        print(json.dumps({"status": "FAIL_INDEPENDENT_DFMAP_RESOURCE_V2_VERIFICATION", "error": str(exc)}, indent=2))
        return 1
    print(json.dumps(result, sort_keys=True, ensure_ascii=False, allow_nan=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
