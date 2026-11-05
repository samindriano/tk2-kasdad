"""Additive R9 hash-schema adapter for the frozen R8 independent verifier.

The R8 verifier remains byte-for-byte immutable. This adapter corrects only
the documented array-digest serialization, delegates every other frozen
identity, matrix, routing, formula, resource, and metric check to that verifier,
and records its own source/input freeze before any query-target scoring.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

BASE_VERIFIER_REL = "scripts/verify_optimization_v3_cls_pos_frequency_ridge_eval_v1.py"
BASE_EVALUATOR_REL = "scripts/evaluate_optimization_v3_cls_pos_frequency_ridge_v1.py"
MODEL_HASH_SOURCE_REL = "src/models/pos_frequency_ridge_v1.py"
TEST_REL = "tests/test_cls_pos_frequency_ridge_eval_r9_adapter.py"
ADAPTER_FREEZE_REL = "experiments/optimization_v3/cls_pos_frequency_ridge_v1/EVALUATION_VERIFIER_ADAPTER_R9_FREEZE.json"
R8_FREEZE_REL = "experiments/optimization_v3/cls_pos_frequency_ridge_v1/EVALUATION_FREEZE_R8.json"
PRED_MANIFEST_REL = "reports/optimization_v3/cls_pos_frequency_ridge_v1/PREDICTIONS_MANIFEST_R1.json"
AUTH_REL = "reports/optimization_v3/cls_pos_frequency_ridge_v1/SOL_VERIFIER_HASH_ADAPTER_R9_AUTHORIZATION.json"
PRED_REL = "experiments/runs/optimization_v3_cls_pos_frequency_ridge_eval_v1/predictions_r1.npz"
AUTHORIZED_PRED_SHA256 = "babc77e7052c2a4dd0c16f4de932b9391fce6bf4efbc9816df3322ae5485b685"
ADAPTER_VERSION = "R9_ADDITIVE_HASH_SCHEMA_ADAPTER"


def _load_base_verifier():
    path = ROOT / BASE_VERIFIER_REL
    spec = importlib.util.spec_from_file_location("frozen_r8_cls_eval_verifier", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("Could not load the frozen R8 verifier")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


BASE = _load_base_verifier()


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def documented_array_sha256(value: Any) -> str:
    """Reproduce the frozen array schema independently from its documented rule.

    The digest starts with the ASCII hex SHA256 of canonical shape/dtype JSON,
    then incorporates canonical string values or contiguous numeric bytes.
    """
    array = np.asarray(value)
    header_hex = hashlib.sha256(
        _canonical_json({"shape": array.shape, "dtype": str(array.dtype)})
    ).hexdigest()
    digest = hashlib.sha256(header_hex.encode("ascii"))
    if array.dtype.kind in "OUS":
        values = json.dumps(
            array.astype(str).tolist(), ensure_ascii=False, separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
        digest.update(values)
    else:
        digest.update(np.ascontiguousarray(array).tobytes())
    return digest.hexdigest()


def verify_array_digest(value: Any, expected: str, label: str) -> str:
    actual = documented_array_sha256(value)
    if actual != expected:
        raise BASE.VerifyError(f"Independent saved array hash mismatch: {label}")
    return actual


def load_predictions_adapted(manifest: dict[str, Any]) -> dict[str, np.ndarray]:
    schema = manifest.get("prediction_schema")
    if (not isinstance(schema, list) or not schema
            or any(not isinstance(key, str) or key.endswith("_y") for key in schema)
            or len(schema) != len(set(schema))):
        raise BASE.VerifyError("Prediction schema is malformed or contains target arrays")
    if set(schema) & {"E67_y", "E75_y"}:
        raise BASE.VerifyError("Target key is forbidden in prediction archive")
    expected_hashes = manifest.get("prediction_array_hashes")
    if not isinstance(expected_hashes, dict) or set(expected_hashes) != set(schema):
        raise BASE.VerifyError("Prediction array hash keys differ from the frozen schema")
    with np.load(ROOT / BASE.PRED, allow_pickle=False) as archive:
        if set(schema) != set(archive.files):
            raise BASE.VerifyError("Prediction archive member set differs from manifest")
        arrays = {key: np.asarray(archive[key]) for key in schema}
    for key in schema:
        verify_array_digest(arrays[key], expected_hashes[key], key)
    return arrays


def _file_sha256(rel: str) -> str:
    return hashlib.sha256((ROOT / rel).read_bytes()).hexdigest()


def _git(*args: str) -> str:
    import subprocess

    completed = subprocess.run(
        ["git", *args], cwd=ROOT, text=True, capture_output=True, check=False,
    )
    if completed.returncode:
        raise RuntimeError(completed.stderr.strip() or "Git command failed")
    return completed.stdout.strip()


def _committed(rel: str) -> dict[str, str]:
    blob = _git("rev-parse", f"HEAD:{rel}")
    if blob != _git("hash-object", rel):
        raise RuntimeError(f"Required R9 binding is not committed cleanly: {rel}")
    return {"git_blob": blob, "commit": _git("log", "-1", "--format=%H", "--", rel)}


def _read_json(rel: str) -> dict[str, Any]:
    return json.loads((ROOT / rel).read_text(encoding="utf-8"))


def build_adapter_freeze() -> dict[str, Any]:
    # Retain the full R8 source/input verification before binding the adapter.
    base_freeze, manifest = BASE.verify_freeze_and_manifest()
    decision = _read_json(AUTH_REL)
    if decision.get("decision") != "AUTHORIZE_ADDITIVE_INDEPENDENT_VERIFIER_HASH_ADAPTER_ONLY":
        raise RuntimeError("Sol authorization does not authorize this adapter")
    if decision.get("source_prediction_archive_sha256") != AUTHORIZED_PRED_SHA256:
        raise RuntimeError("Sol authorization names a different prediction archive")
    if manifest.get("prediction_sha256") != AUTHORIZED_PRED_SHA256:
        raise RuntimeError("Frozen prediction manifest differs from authorized archive")
    source_paths = [
        str(Path(__file__).resolve().relative_to(ROOT)).replace("\\", "/"),
        TEST_REL, BASE_VERIFIER_REL, BASE_EVALUATOR_REL, MODEL_HASH_SOURCE_REL,
    ]
    source_hashes = {rel: _file_sha256(rel) for rel in source_paths}
    # The frozen R8 verifier/evaluator source hashes must remain exact.
    for rel in (BASE_VERIFIER_REL, BASE_EVALUATOR_REL):
        if base_freeze.get("source_hashes", {}).get(rel) != source_hashes[rel]:
            raise RuntimeError(f"R8 source freeze does not match {rel}")
    for rel, expected in base_freeze.get("source_hashes", {}).items():
        if _file_sha256(rel) != expected:
            raise RuntimeError(f"R8 frozen source changed: {rel}")
    for rel, expected in base_freeze.get("input_hashes", {}).items():
        if _file_sha256(rel) != expected:
            raise RuntimeError(f"R8 frozen input changed: {rel}")
    input_paths = [R8_FREEZE_REL, PRED_MANIFEST_REL, AUTH_REL]
    input_hashes = {rel: _file_sha256(rel) for rel in input_paths}
    input_hashes[PRED_REL] = _file_sha256(PRED_REL)
    if input_hashes[PRED_REL] != AUTHORIZED_PRED_SHA256:
        raise RuntimeError("Prediction archive bytes differ from authorization")
    bindings = {
        rel: _committed(rel)
        for rel in (BASE_VERIFIER_REL, BASE_EVALUATOR_REL, R8_FREEZE_REL,
                    PRED_MANIFEST_REL, AUTH_REL, *source_paths[:2])
    }
    body: dict[str, Any] = {
        "schema_version": 1,
        "adapter_version": ADAPTER_VERSION,
        "status": "FROZEN_ADDITIVE_VERIFIER_REPAIR_BEFORE_QUERY_TARGET_SCORING",
        "parent_r8_freeze": {
            "path": R8_FREEZE_REL,
            "sha256": _file_sha256(R8_FREEZE_REL),
            "commit": bindings[R8_FREEZE_REL]["commit"],
            "source_hashes": base_freeze.get("source_hashes", {}),
            "input_hashes": base_freeze.get("input_hashes", {}),
        },
        "prediction_manifest": {
            "path": PRED_MANIFEST_REL,
            "sha256": input_hashes[PRED_MANIFEST_REL],
            "commit": bindings[PRED_MANIFEST_REL]["commit"],
            "archive_path": PRED_REL,
            "archive_sha256": input_hashes[PRED_REL],
            "archive_members": len(manifest.get("prediction_schema", [])),
        },
        "authorization": {
            "path": AUTH_REL,
            "sha256": input_hashes[AUTH_REL],
            "commit": bindings[AUTH_REL]["commit"],
        },
        "source_hashes": source_hashes,
        "source_bindings": bindings,
        "input_hashes": input_hashes,
        "array_hash_schema": {
            "header": "sha256(canonical_json({shape,dtype})); seed content SHA256 with lowercase ASCII hex header digest",
            "unicode_or_object_strings": "compact UTF-8 JSON of array.astype(str).tolist(), ensure_ascii=false",
            "other_dtypes": "C-contiguous array bytes",
            "gamma_receipts": "same documented logical-array schema as prediction arrays",
        },
        "delegated_checks": [
            "R8 source/input, cache semantics, identities, matrices, routes, formulas, receipts, and resource gates",
            "R8 independent prediction replay and full independent metric reconstruction",
        ],
        "scientific_recipe_or_gate_changes": False,
        "new_fits": 0,
        "test_or_samplesubmission_accessed": False,
        "query_target_labels_read_during_freeze": False,
    }
    body["freeze_sha256"] = hashlib.sha256(_canonical_json(body)).hexdigest()
    return body


def verify_adapter_freeze() -> dict[str, Any]:
    path = ROOT / ADAPTER_FREEZE_REL
    if not path.exists():
        raise RuntimeError("R9 adapter freeze is missing")
    frozen = _read_json(ADAPTER_FREEZE_REL)
    body = dict(frozen)
    claim = body.pop("freeze_sha256", None)
    if claim != hashlib.sha256(_canonical_json(body)).hexdigest():
        raise RuntimeError("R9 adapter freeze signature mismatch")
    for rel, expected in frozen.get("source_hashes", {}).items():
        if _file_sha256(rel) != expected:
            raise RuntimeError(f"R9 frozen adapter source changed: {rel}")
        _committed(rel)
    for rel, expected in frozen.get("input_hashes", {}).items():
        if _file_sha256(rel) != expected:
            raise RuntimeError(f"R9 frozen adapter input changed: {rel}")
    for rel, binding in frozen.get("source_bindings", {}).items():
        committed = _committed(rel)
        if committed != binding:
            raise RuntimeError(f"R9 Git binding changed: {rel}")
    # Re-run every frozen R8 source/input and lineage check, without labels.
    BASE.verify_freeze_and_manifest()
    return frozen


def run_self_tests(*, verify_actual_archive: bool) -> dict[str, Any]:
    vectors = [
        np.asarray([1, -2, 300], dtype=np.int64),
        np.asarray([0.25, -3.5], dtype=np.float64),
        np.asarray([True, False, True], dtype=np.bool_),
        np.asarray(["café", "漏洞", "α"], dtype="U4"),
        np.asarray(["id-a", "id-b"], dtype=object),
        np.asarray([], dtype=np.int32),
        np.arange(24, dtype=np.int16)[::3],
    ]
    digests = [documented_array_sha256(value) for value in vectors]
    if len(set(digests)) != len(vectors):
        raise RuntimeError("R9 schema self-test produced an unexpected digest collision")
    ids = np.asarray(["r1", "r2", "r3"], dtype="U2")
    ids_hash = documented_array_sha256(ids)
    for changed in (ids.astype(object), ids[::-1], np.asarray(["r1", "r2", "r4"], dtype="U2")):
        if documented_array_sha256(changed) == ids_hash:
            raise RuntimeError("R9 dtype/order/value distinction self-test failed")
    # Verify direct rejection of tampered order/value and dtype under frozen hash.
    for label, changed in (("order", ids[::-1]), ("value", np.asarray(["r1", "r2", "r4"], dtype="U2")),
                           ("dtype", ids.astype(object))):
        try:
            verify_array_digest(changed, ids_hash, label)
        except BASE.VerifyError:
            pass
        else:
            raise RuntimeError(f"R9 tampered {label} array was accepted")
    result: dict[str, Any] = {
        "status": "PASS_R9_ARRAY_SCHEMA_SELF_TESTS",
        "dtype_families": [str(a.dtype) for a in vectors],
        "noncontiguous_vector_tested": True,
        "dtype_order_value_tampering_rejected": True,
        "actual_archive_checked": False,
    }
    if verify_actual_archive:
        _, manifest = BASE.verify_freeze_and_manifest()
        arrays = load_predictions_adapted(manifest)
        if len(arrays) != 28:
            raise RuntimeError(f"Expected 28 frozen prediction arrays, got {len(arrays)}")
        if any(key.endswith("_y") for key in arrays):
            raise RuntimeError("Target key present in frozen prediction archive")
        # Check a known saved gamma receipt through the corrected independent helper.
        gamma_rel = "experiments/runs/optimization_v3_cls_pos_frequency_ridge_v1/units/E67_ALIGNED.npz"
        receipt_rel = "experiments/runs/optimization_v3_cls_pos_frequency_ridge_v1/receipts/E67_ALIGNED.json"
        gamma = BASE.load_gamma(gamma_rel, receipt_rel)
        if documented_array_sha256(gamma) != BASE.read_json(ROOT / receipt_rel).get("gamma_sha256"):
            raise RuntimeError("Known gamma receipt did not match corrected independent schema")
        result.update({
            "actual_archive_checked": True,
            "actual_prediction_arrays": len(arrays),
            "actual_array_hashes_pass": True,
            "target_keys_present": [],
            "known_gamma_receipt_checked": True,
            "gamma_elements": int(gamma.size),
        })
    return result


def _metadata(freeze: dict[str, Any], invocation: str) -> dict[str, Any]:
    return {
        "adapter_version": ADAPTER_VERSION,
        "adapter_invocation": invocation,
        "adapter_freeze_sha256": freeze["freeze_sha256"],
        "prediction_archive_sha256": AUTHORIZED_PRED_SHA256,
        "unmodified_r8_cli_status": "FAILED_PRELABEL_HASH_SCHEMA_MISMATCH; adapter result is distinct",
        "query_target_access_forbidden_during_prediction_verification": True,
    }


def _verify_predictions(freeze: dict[str, Any]) -> dict[str, Any]:
    with BASE.SharedJobLock():
        initial = BASE.initial_resource_snapshot()
        if not initial["pass_both_initial_gates"]:
            raise BASE.VerifyError(f"R9 initial RAM gates failed before prelabel replay: {initial}")
        samples = [initial]
        base_freeze, manifest = BASE.verify_freeze_and_manifest()
        arrays = load_predictions_adapted(manifest)
        replay = BASE.independently_replay_saved_predictions(arrays, samples)
        final = BASE.boundary_resource_snapshot("AFTER_R9_ADAPTER_PRELABEL_REPLAY")
        samples.append(final)
        if not final["pass"]:
            raise BASE.VerifyError(f"R9 post-replay RAM boundary failed: {final}")
        return {
            "status": "PASS_R9_ADAPTER_PRELABEL_PREDICTION_HASH_AND_SCHEMA",
            "target_labels_loaded": False,
            "test_or_samplesubmission_accessed": False,
            "prediction_fields": len(arrays),
            "parent_r8_freeze_sha256": _file_sha256(R8_FREEZE_REL),
            "independent_prediction_replay": replay,
            "resource_snapshot": {
                "policy_id": BASE.RESOURCE_POLICY_ID,
                "samples": samples,
                "samples_are_peak_measurements": False,
                "rss_samples_are_current_state_not_peak": True,
            },
        }


def _verify_results(freeze: dict[str, Any]) -> dict[str, Any]:
    with BASE.SharedJobLock():
        initial = BASE.initial_resource_snapshot()
        if not initial["pass_both_initial_gates"]:
            raise BASE.VerifyError(f"R9 initial RAM gates failed before independent metrics: {initial}")
        samples = [initial]
        result = BASE.verify_results(samples)
        return {
            "status": "PASS_R9_ADAPTER_INDEPENDENT_METRIC_AND_ROW_RECONSTRUCTION",
            "adapter_metadata": _metadata(freeze, "--verify-results"),
            **result,
        }


# The R8 implementation keeps its original source bytes. These two narrow
# replacements are the entire adapter surface; all other frozen checks remain.
BASE.load_predictions = load_predictions_adapted
BASE.array_sha256 = documented_array_sha256


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--self-test", action="store_true")
    group.add_argument("--write-freeze", action="store_true")
    group.add_argument("--verify-predictions", action="store_true")
    group.add_argument("--verify-results", action="store_true")
    args = parser.parse_args(argv)
    if args.self_test:
        result = run_self_tests(verify_actual_archive=True)
    elif args.write_freeze:
        run_self_tests(verify_actual_archive=False)
        result = build_adapter_freeze()
        dest = ROOT / ADAPTER_FREEZE_REL
        if dest.exists():
            raise RuntimeError("Refusing to replace an existing R9 adapter freeze")
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(json.dumps(result, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    elif args.verify_predictions:
        run_self_tests(verify_actual_archive=False)
        freeze = verify_adapter_freeze()
        result = _verify_predictions(freeze)
        result["adapter_metadata"] = _metadata(freeze, "--verify-predictions")
    else:
        run_self_tests(verify_actual_archive=False)
        freeze = verify_adapter_freeze()
        result = _verify_results(freeze)
    print(json.dumps(result, indent=2, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
