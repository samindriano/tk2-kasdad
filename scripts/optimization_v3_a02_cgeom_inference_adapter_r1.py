"""Authorized one-expression adapter for the frozen A02 inference replay.

The frozen runner and trained-state hashes stay unchanged. The adapter derives
the replay helper in memory only after verifying the exact original file and
function hashes and the single approved field-name mismatch.
"""
from __future__ import annotations

import ast
import hashlib
import inspect
import json
import sys
import textwrap
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import scripts.optimization_v3_a02_cgeom_benchmark_v1 as batch  # noqa: E402


ADAPTER_PATH = Path(__file__).resolve()
ADAPTER_RELPATH = ADAPTER_PATH.relative_to(ROOT).as_posix()
ADAPTER_RECEIPT = batch.REPORT_ROOT / "INFERENCE_ADAPTER_R1.json"
EXPECTED_FROZEN_RUNNER_SHA256 = "d6af3f3e18787c9cd1e5c52b1463fb30e18398cee6e3dc838e5227cf30a50a3f"
EXPECTED_FROZEN_HELPER_SHA256 = "2c7a41f577e87d358337c8cabfb74e130793782661da760d15c60d3b511205d5"
EXPECTED_A02_TYPE = "src.models.classification_a02_neighbors.ResidualNeighborPredictor"
EXPECTED_CGEOM_TYPE = "src.models.a02_coefficient_geometry_v1.CoefficientGeometryPredictor"
OLD_EXPRESSION = 'result["correction_raw"]'
NEW_EXPRESSION = 'result["correction_unclipped"]'


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _literal_return_keys(method: Any) -> list[str]:
    tree = ast.parse(textwrap.dedent(inspect.getsource(method)))
    for node in reversed(list(ast.walk(tree))):
        if isinstance(node, ast.Return) and isinstance(node.value, ast.Dict):
            keys = [ast.literal_eval(key) for key in node.value.keys]
            if all(isinstance(key, str) for key in keys):
                return keys
    raise batch.BatchError(f"Could not read a literal top-level return dictionary: {method}")


def _derive_verified_helper() -> tuple[str, dict[str, str]]:
    runner_path = Path(batch.__file__).resolve()
    runner_bytes = runner_path.read_bytes()
    runner_sha = _sha256(runner_bytes)
    if runner_sha != EXPECTED_FROZEN_RUNNER_SHA256:
        raise batch.BatchError("Frozen A02 runner source hash changed; refusing adapter")

    freeze = batch._read_json(batch.TRAINED_FREEZE)
    frozen_runner_sha = freeze.get("source_sha256", {}).get("scripts/optimization_v3_a02_cgeom_benchmark_v1.py")
    if frozen_runner_sha != runner_sha:
        raise batch.BatchError("Trained freeze no longer binds the frozen runner source")

    original = inspect.getsource(batch._load_and_replay_a02)
    original_sha = _sha256(original.encode("utf-8"))
    if original_sha != EXPECTED_FROZEN_HELPER_SHA256:
        raise batch.BatchError("Frozen A02 replay helper hash changed; refusing adapter")
    if original.count(OLD_EXPRESSION) != 1:
        raise batch.BatchError("Expected exactly one incorrect correction field expression")
    if original.count('"correction_unclipped": ' + OLD_EXPRESSION) != 1:
        raise batch.BatchError("Incorrect correction field is not the one authorized mapping")

    derived = original.replace(OLD_EXPRESSION, NEW_EXPRESSION, 1)
    if OLD_EXPRESSION in derived or derived.count(NEW_EXPRESSION) != 1:
        raise batch.BatchError("Derived helper differs beyond the one approved key correction")
    compile(derived, str(runner_path), "exec")
    identities = {
        "frozen_runner_sha256": runner_sha,
        "frozen_helper_sha256": original_sha,
        "derived_helper_sha256": _sha256(derived.encode("utf-8")),
    }
    return derived, identities


def _audit_model_predict_contracts() -> dict[str, Any]:
    import joblib

    a02 = joblib.load(batch.A02_MODEL)
    a02_method = type(a02).predict
    a02_keys = _literal_return_keys(a02_method)
    with __import__("numpy").load(batch.A02_TEST_NPZ, allow_pickle=False) as archive:
        saved_keys = sorted(archive.files)
    if (f"{type(a02).__module__}.{type(a02).__name__}" != EXPECTED_A02_TYPE
            or "correction_unclipped" not in a02_keys or "correction_raw" in a02_keys
            or "correction_unclipped" not in saved_keys):
        raise batch.BatchError("Actual A02 producer/cache return contract differs from authorization")

    cgeom_payload = batch._load_unit(batch.INTEGRATION_ACTIVE_ID_BY_SLOT[8])
    cgeom = cgeom_payload["predictor"]
    cgeom_method = type(cgeom).predict
    cgeom_keys = _literal_return_keys(cgeom_method)
    cgeom_consumed = ["candidate_probability", "correction", "neighbor_count",
                      "neighbor_source_rows", "neighbor_ids", "neighbor_similarities"]
    if (f"{type(cgeom).__module__}.{type(cgeom).__name__}" != EXPECTED_CGEOM_TYPE
            or not set(cgeom_consumed).issubset(cgeom_keys)
            or "correction_raw" not in cgeom_keys):
        raise batch.BatchError("Actual CGEOM predictor contract differs from the frozen consumer")

    return {
        "status": "PASS_ACTUAL_MODEL_PREDICT_SIGNATURE_AND_RETURN_KEY_AUDIT",
        "a02": {
            "model_type": f"{type(a02).__module__}.{type(a02).__name__}",
            "predict_signature": str(inspect.signature(a02_method)),
            "predict_return_keys": a02_keys,
            "saved_test_prediction_keys": saved_keys,
            "consumed_correction_key": "correction_unclipped",
            "correction_raw_key_present": "correction_raw" in a02_keys,
        },
        "cgeom": {
            "model_type": f"{type(cgeom).__module__}.{type(cgeom).__name__}",
            "predict_signature": str(inspect.signature(cgeom_method)),
            "predict_return_keys": cgeom_keys,
            "consumed_return_keys": cgeom_consumed,
            "correction_raw_key_preserved": "correction_raw" in cgeom_keys,
        },
        "other_inference_calls": {
            "structured_logistic": "predict_proba and decision_function; both outputs are checked against class order / formula",
            "positional_gamma": "offset_probability plus direct sigmoid formula; outputs compared in formula audit",
            "joint_logistic": "predict_proba and decision_function; outputs compared in formula audit",
        },
    }


def install_verified_helper() -> dict[str, str]:
    derived, identities = _derive_verified_helper()
    namespace = batch.__dict__
    exec(compile(derived, str(Path(batch.__file__).resolve()), "exec"), namespace)
    patched = namespace.get("_load_and_replay_a02")
    if (not callable(patched)
            or patched.__code__.co_filename != str(Path(batch.__file__).resolve())
            or "correction_unclipped" not in patched.__code__.co_consts):
        raise batch.BatchError("Derived replay helper was not installed from the verified one-expression source")
    return identities


def _output_paths() -> list[Path]:
    return [
        batch.OUTPUT_NPZ,
        batch.RUN_ROOT / "inference_receipt.json",
        batch.OUTPUT_ROOT / "classification_cgeom_aligned_e79.csv",
        batch.OUTPUT_ROOT / "classification_posoffset_e79.csv",
        batch.OUTPUT_ROOT / "classification_jointpos_e79.csv",
        ADAPTER_RECEIPT,
    ]


def run_inference() -> dict[str, Any]:
    if ADAPTER_RECEIPT.exists():
        raise batch.BatchError("Refusing to overwrite the adapter receipt")
    existing = [str(path) for path in _output_paths() if path.exists()]
    if existing:
        raise batch.BatchError(f"Inference output already exists; refusing a non-idempotent rerun: {existing}")
    freeze = batch._verify_trained_freeze()
    if batch.source_hashes() != freeze.get("source_sha256"):
        raise batch.BatchError("Frozen runner/core source set differs before adapter inference")
    batch.require_clean_tree()
    audit = _audit_model_predict_contracts()
    derived = install_verified_helper()
    result = batch.infer_three()
    if result.get("status") != "PREDICTIONS_AND_CSVS_CREATED_NOT_UPLOADED":
        raise batch.BatchError("Frozen inference runner did not return the expected completed status")
    return {
        "status": "PASS_THREE_CANDIDATE_CSVS_CREATED_NOT_UPLOADED",
        "inference": result,
        "predict_contract_audit": audit,
        "derived_helper_identity": derived,
    }


def write_adapter_receipt(verification: dict[str, Any]) -> dict[str, Any]:
    if ADAPTER_RECEIPT.exists():
        raise batch.BatchError("Refusing to overwrite the adapter receipt")
    adapter_commit = batch._git("log", "-1", "--format=%H", "--", ADAPTER_RELPATH)
    adapter_sha = batch.sha256_file(ADAPTER_PATH)
    inference_receipt_path = batch.RUN_ROOT / "inference_receipt.json"
    manifest_path = batch.MANIFEST_PATH
    report_path = batch.REPORT_PATH
    if not inference_receipt_path.is_file() or not manifest_path.is_file() or not report_path.is_file():
        raise batch.BatchError("Inference receipt, verified manifest, or report is missing")
    record = {
        "schema_version": 1,
        "status": "PASS_INFERENCE_ADAPTER_AND_OUTPUTS_BOUND",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "adapter_commit": adapter_commit,
        "adapter_path": ADAPTER_RELPATH,
        "adapter_sha256": adapter_sha,
        "frozen_source_commit": batch._read_json(batch.TRAINED_FREEZE)["source_commit"],
        "frozen_runner_sha256": verification["frozen_runner_sha256"],
        "frozen_helper_sha256": verification["frozen_helper_sha256"],
        "derived_helper_sha256": verification["derived_helper_sha256"],
        "approved_single_expression_change": {
            "old_consumer_expression": OLD_EXPRESSION,
            "new_consumer_expression": NEW_EXPRESSION,
            "model_outputs_and_cached_arrays_changed": False,
        },
        "predict_contract_audit": _audit_model_predict_contracts(),
        "inference_receipt_sha256": batch.sha256_file(inference_receipt_path),
        "batch_manifest_sha256": batch.sha256_file(manifest_path),
        "batch_report_sha256": batch.sha256_file(report_path),
        "trained_freeze_sha256": batch.sha256_file(batch.TRAINED_FREEZE),
        "prediction_bundle_sha256": batch.sha256_file(batch.OUTPUT_NPZ),
        "a02_model_sha256": batch.sha256_file(batch.A02_MODEL),
        "a02_cached_test_predictions_sha256": batch.sha256_file(batch.A02_TEST_NPZ),
        "official_inputs_sha256": batch.input_hashes(),
        "csvs": verification["csvs"],
        "csv_verification_status": verification["status"],
        "test_targets_loaded": False,
        "kaggle_contacted": False,
        "upload_performed": False,
    }
    batch.write_json_once(ADAPTER_RECEIPT, record)
    return record


def main() -> None:
    if len(sys.argv) != 2 or sys.argv[1] not in {"infer", "receipt"}:
        raise SystemExit("usage: python -B scripts/optimization_v3_a02_cgeom_inference_adapter_r1.py {infer|receipt}")
    if sys.argv[1] == "infer":
        result = run_inference()
    else:
        result = write_adapter_receipt(batch.verify_outputs())
    print(json.dumps(result, sort_keys=True, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
