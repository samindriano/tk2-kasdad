"""Integrity helpers for the independent E79 verifier v3 attestation."""
from __future__ import annotations

import hashlib
import json
import re
from typing import Any, Callable


ORIGINAL_V2_COMMIT = "9b35affa261449aa63ed057b7bd77abe2ed29caf"
ORIGINAL_V2_ADAPTER_FILES = (
    "scripts/reg_position_real_e79_hash_compat_v2.py",
    "scripts/reg_position_real_e79_inference_v2.py",
    "scripts/verify_reg_position_real_e79_benchmark_v2.py",
    "tests/test_reg_position_real_e79_hash_compat_v2.py",
    "reports/optimization_v3/reg_position_real_e79_benchmark_v1/INFERENCE_CONTINUATION_AUTHORIZATION_V2.md",
)


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, ensure_ascii=False,
        separators=(",", ":"), allow_nan=False,
    ).encode("utf-8")).hexdigest()


def validate_manifest_integrity(
    manifest: dict[str, Any],
    actual_file_sha256: str,
    expected_manifest_signature: str,
    expected_file_sha256: str,
) -> dict[str, Any]:
    supplied_signature = manifest.get("manifest_sha256")
    unsigned = {key: value for key, value in manifest.items() if key != "manifest_sha256"}
    canonical_signature = canonical_sha256(unsigned)
    checks = {
        "manifest_file_hash_matches_expected": actual_file_sha256 == expected_file_sha256,
        "embedded_manifest_signature_matches_expected": supplied_signature == expected_manifest_signature,
        "embedded_manifest_signature_valid": supplied_signature == canonical_signature,
    }
    if not all(checks.values()):
        failed = [key for key, passed in checks.items() if not passed]
        raise ValueError("Original inference manifest integrity failed: " + ", ".join(failed))
    return {
        "checks": checks,
        "manifest_sha256": supplied_signature,
        "manifest_file_sha256": actual_file_sha256,
    }


def validate_source_hash_binding(
    expected_inventory: dict[str, str],
    preflight_inventory: Any,
    freeze_inventory: Any,
    observed_boolean_results: Any,
    actual_file_hashes: dict[str, str],
) -> dict[str, Any]:
    expected_keys = set(expected_inventory)
    result_map = observed_boolean_results if isinstance(observed_boolean_results, dict) else {}
    preflight_map = preflight_inventory if isinstance(preflight_inventory, dict) else {}
    freeze_map = freeze_inventory if isinstance(freeze_inventory, dict) else {}
    actual_map = actual_file_hashes if isinstance(actual_file_hashes, dict) else {}
    checks = {
        "expected_inventory_is_sha256_map": (
            bool(expected_inventory)
            and all(isinstance(key, str) and isinstance(value, str)
                    and re.fullmatch(r"[0-9a-f]{64}", value) is not None
                    for key, value in expected_inventory.items())
        ),
        "preflight_inventory_exact": (
            set(preflight_map) == expected_keys and preflight_map == expected_inventory
        ),
        "freeze_inventory_exact": (
            set(freeze_map) == expected_keys and freeze_map == expected_inventory
        ),
        "observed_result_keys_exact": set(result_map) == expected_keys,
        "observed_result_values_are_booleans": (
            set(result_map) == expected_keys
            and all(type(result_map[key]) is bool for key in expected_keys)
        ),
        "observed_results_all_true": (
            set(result_map) == expected_keys
            and all(type(result_map[key]) is bool and result_map[key] is True
                    for key in expected_keys)
        ),
        "actual_file_hashes_exact": (
            set(actual_map) == expected_keys and actual_map == expected_inventory
        ),
    }
    if not all(checks.values()):
        failed = [key for key, passed in checks.items() if not passed]
        raise ValueError("Original frozen source binding failed: " + ", ".join(failed))
    return {
        "checks": checks,
        "pinned_inventory": dict(expected_inventory),
        "preflight_inventory": dict(preflight_map),
        "freeze_inventory": dict(freeze_map),
        "observed_boolean_results": dict(result_map),
        "actual_file_hashes": dict(actual_map),
    }


def validate_historical_adapter_commit(
    recorded_commit: Any,
    current_head: Any,
    manifest_adapter_hashes: Any,
    is_ancestor: Callable[[str, str], bool],
    read_blob: Callable[[str, str], bytes],
) -> dict[str, Any]:
    commit_checks = {
        "recorded_commit_is_approved_v2_commit": recorded_commit == ORIGINAL_V2_COMMIT,
        "recorded_commit_is_full_sha": (
            isinstance(recorded_commit, str)
            and re.fullmatch(r"[0-9a-f]{40}", recorded_commit) is not None
        ),
        "current_head_is_full_sha": (
            isinstance(current_head, str)
            and re.fullmatch(r"[0-9a-f]{40}", current_head) is not None
        ),
    }
    if not all(commit_checks.values()):
        failed = [key for key, passed in commit_checks.items() if not passed]
        raise ValueError("Historical adapter commit identity failed: " + ", ".join(failed))
    if not is_ancestor(recorded_commit, current_head):
        raise ValueError("Historical adapter commit is not an ancestor of current HEAD")
    if not isinstance(manifest_adapter_hashes, dict):
        raise ValueError("Historical adapter file hash map is missing")
    if set(manifest_adapter_hashes) != set(ORIGINAL_V2_ADAPTER_FILES):
        raise ValueError("Historical adapter file hash map has missing or extra paths")
    blob_hashes: dict[str, str] = {}
    for relative in ORIGINAL_V2_ADAPTER_FILES:
        expected = manifest_adapter_hashes[relative]
        if not isinstance(expected, str) or re.fullmatch(r"[0-9a-f]{64}", expected) is None:
            raise ValueError(f"Historical adapter manifest hash is invalid: {relative}")
        digest = hashlib.sha256(read_blob(recorded_commit, relative)).hexdigest()
        blob_hashes[relative] = digest
        if digest != expected:
            raise ValueError(f"Historical adapter Git blob differs from manifest pin: {relative}")
    return {
        "recorded_commit": recorded_commit,
        "current_head": current_head,
        "recorded_commit_is_ancestor": True,
        "manifest_adapter_hashes": dict(manifest_adapter_hashes),
        "git_blob_hashes": blob_hashes,
    }
