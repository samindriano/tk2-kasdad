"""Apply the frozen zero-fit present-only policy for REG E79.

This stage only combines already saved predictions.  ``--freeze`` hashes
inputs and reads JSON metadata without parsing either prediction CSV or NPZ
arrays.  ``--apply`` is permitted only after the source and input freeze has
been created and committed.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
BRANCH = "codex/reg-position-failure-mechanism-v1"
R3_CLOSURE_COMMIT = "a73a760a451358cb735ca50a9ac3207892bb3d71"
N_ROWS = 29421
N_MISSING = 1062
INTERCEPT = 2.169196428141562
SLOPE = 0.9744451489746011

INPUT_PATHS = {
    "global_csv": ROOT / "submissions/reg_final_wave_v2/current_direct_affine_l1_e79.csv",
    "global_manifest": ROOT / "reports/optimization_v3/REG_FINAL_WAVE_V1/INFERENCE_MANIFEST_V2.json",
    "global_state": ROOT / "experiments/runs/REG_FINAL_WAVE_V2/CALIBRATOR_STATE.json",
    "global_fit_receipt": ROOT / "experiments/runs/REG_FINAL_WAVE_V2/FIT_RECEIPT.json",
    "global_qa": ROOT / "reports/optimization_v3/REG_FINAL_WAVE_V1/CSV_QA_V2.json",
    "direct_csv": ROOT / "submissions/reg_current_r06_e79_exploratory_v1/direct_position_real_e79.csv",
    "direct_manifest": ROOT / "reports/optimization_v3/REG_CURRENT_R06_E79_EXPLORATORY_BATCH_V1/INFERENCE_MANIFEST_B.json",
    "direct_qa": ROOT / "reports/optimization_v3/REG_CURRENT_R06_E79_EXPLORATORY_BATCH_V1/CSV_QA_B.json",
    "direct_archive": ROOT / "experiments/runs/REG_CURRENT_R06_E79_EXPLORATORY_BATCH_V1/inference_B.npz",
    "direct_trained_freeze": ROOT / "reports/optimization_v3/REG_CURRENT_R06_E79_EXPLORATORY_BATCH_V1/TRAINED_STATE_FREEZE_B.json",
    "direct_prelabel": ROOT / "reports/optimization_v3/REG_CURRENT_R06_E79_EXPLORATORY_BATCH_V1/PRELABEL_PASS_B.json",
    "decision": ROOT / "reports/optimization_v3/REG_FINAL_WAVE_FOLLOWUP_V2/SOL_DIAGNOSTIC_DECISION.md",
    "r3_source_freeze": ROOT / "experiments/REG_FINAL_WAVE_FOLLOWUP_V2/PREFIT_FREEZE_R3.json",
    "r3_root_source_pass": ROOT / "experiments/REG_FINAL_WAVE_FOLLOWUP_V2/ROOT_SOURCE_PASS.json",
    "r3_identity_preflight": ROOT / "reports/optimization_v3/REG_FINAL_WAVE_FOLLOWUP_V2/IDENTITY_PREFLIGHT_R3.json",
    "r3_diagnostics": ROOT / "reports/optimization_v3/REG_FINAL_WAVE_FOLLOWUP_V2/DIAGNOSTICS.json",
    "r3_diagnostic_ledger": ROOT / "experiments/runs/REG_FINAL_WAVE_FOLLOWUP_V2/lp_ledger.jsonl",
    "r3_closure": ROOT / "reports/optimization_v3/REG_FINAL_WAVE_FOLLOWUP_V2/DIAGNOSTIC_PHASE_CLOSURE_R3.md",
    "previous_r1_freeze": ROOT / "experiments/REG_FINAL_WAVE_FOLLOWUP_V2_B_ONLY/B_ONLY_PREFIT_FREEZE.json",
    "previous_r1_failure_receipt": ROOT / "experiments/runs/REG_FINAL_WAVE_FOLLOWUP_V2_B_ONLY/APPLICATION_GATE_FAILED_R1.json",
}

EXPECTED_INPUT_SHA256 = {
    "global_csv": "4abc56101e1863520771c125ad9d1f61e9f1530ee43575f2ae0cbd49331d4b56",
    "global_manifest": "d29d6a0ffc46d36910e822e1537d2ba7c05543faa656d88f60ad49c0d1477cf1",
    "global_state": "38754194dc6548e1627496c894961e1fa81a365b377b791a722a24a027bb3d8a",
    "global_fit_receipt": "0b37eb61afd32ff5ee69c9b53650950ebf0d593626f41121f4f5f2fa0d03096e",
    "global_qa": "2e252e92b9dbad45e6baac3df620d6819356f8eebeb643340fb3e06f6ace53fa",
    "direct_csv": "49ae7f668aea99e433c2bec72ed5ff51126fb12d888c3185ec048730d767ea6f",
    "direct_manifest": "d2bd7dea17feab1740ca4630bb5cf7ace62caf8746051b3322cc38d168674484",
    "direct_qa": "bc19efcbbf57ae0e621fe1b338f00e1726380ed9a736947f6bf760b8a9fa67c8",
    "direct_archive": "e29b1eabb1692729749b25d88b7cf86d1bdf44c34ce74ab964b124074f5817f2",
    "direct_trained_freeze": "3364e75a8c791cebbd9fd4e2693149fa1429e7e5ef19f68bb1cadb6828b51218",
    "direct_prelabel": "a4771eacfe808e5645977008827ebc9552c288dbed5910606649e3d4670716c3",
    "decision": "12ccdba91b2777c30a132d04a8b58998bcf567bf6c79042333c451d622642026",
    "r3_source_freeze": "70b49ff7dea59edf53622106392c1c5d943b46dcf0f4e792f5211b83453061b8",
    "r3_root_source_pass": "d5d71092e2824596cc7416a22af9de8f074b2123dca36ebf1a7b661c6b8da10f",
    "r3_identity_preflight": "639c080b91121d7d1183fc0de7f6d9f8cec2e68f496f4242cddd072dd4e5f211",
    "r3_diagnostics": "136d4a8c7a565413ec550dca1212f7cda22a2c8ca5d70dbfdd6672ceb14961d8",
    "r3_diagnostic_ledger": "961366c4b0e8ed998aa147b624a91fa05bae9a644b04cba9c5612eeafa9fc33b",
    "r3_closure": "02dd7c9117d3f3fb83f98603cb5e0739ef618b505311212312ef44fa04b12243",
    "previous_r1_freeze": "d6279835f8272ae23ac0e90d42a10f26ae2352bc0cebab47461b65e9e678db1d",
    "previous_r1_failure_receipt": "751ed2267e665cbfc982a97a3b7ba13ff3866de5b09ca02049a4adfde01e7eca",
}

SOURCE_PATHS = {
    "application_source": ROOT / "scripts/reg_final_wave_followup_v2_b_only.py",
    "independent_verifier": ROOT / "scripts/verify_reg_final_wave_followup_v2_b_only.py",
    "tests": ROOT / "tests/test_reg_final_wave_followup_v2_b_only.py",
    "protocol": ROOT / "reports/optimization_v3/REG_FINAL_WAVE_FOLLOWUP_V2_B_ONLY/PROTOCOL.md",
    "previous_synthetic_qa": ROOT / "reports/optimization_v3/REG_FINAL_WAVE_FOLLOWUP_V2_B_ONLY/SYNTHETIC_QA.json",
    "synthetic_qa": ROOT / "reports/optimization_v3/REG_FINAL_WAVE_FOLLOWUP_V2_B_ONLY/SYNTHETIC_QA_R2.json",
}

FREEZE = ROOT / "experiments/REG_FINAL_WAVE_FOLLOWUP_V2_B_ONLY/B_ONLY_PREFIT_FREEZE_R2.json"
OUT_CSV = ROOT / "submissions/reg_final_wave_followup_v2_b_only/global_present_only_e79.csv"
OUT_MANIFEST = ROOT / "reports/optimization_v3/REG_FINAL_WAVE_FOLLOWUP_V2_B_ONLY/INFERENCE_MANIFEST_B_ONLY.json"
APP_RECEIPT = ROOT / "experiments/runs/REG_FINAL_WAVE_FOLLOWUP_V2_B_ONLY/APPLICATION_RECEIPT.json"
INDEPENDENT_QA = ROOT / "reports/optimization_v3/REG_FINAL_WAVE_FOLLOWUP_V2_B_ONLY/INDEPENDENT_QA.json"
SYNTHETIC_QA = SOURCE_PATHS["synthetic_qa"]


class GateError(RuntimeError):
    """An input, freeze, or output failed a fail-closed identity gate."""


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False,
                      separators=(",", ":")).encode("utf-8", errors="surrogatepass")


def canonical_sha(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha_vector(values: np.ndarray) -> str:
    vector = np.ascontiguousarray(np.asarray(values, dtype="<f8"))
    digest = hashlib.sha256(canonical_bytes({"shape": vector.shape, "dtype": str(vector.dtype)}))
    digest.update(vector.tobytes(order="C"))
    return digest.hexdigest()


def sha_raw_f64(values: np.ndarray) -> str:
    """Match the direct-candidate manifest's raw little-endian byte digest."""
    vector = np.ascontiguousarray(np.asarray(values, dtype="<f8"))
    return hashlib.sha256(vector.tobytes(order="C")).hexdigest()


def sha_ordered_ids(values: list[str] | np.ndarray) -> str:
    return hashlib.sha256(canonical_bytes([str(v) for v in values])).hexdigest()


def sha_mask(values: np.ndarray) -> str:
    mask = np.ascontiguousarray(np.asarray(values, dtype=np.bool_))
    digest = hashlib.sha256(canonical_bytes({"shape": mask.shape, "dtype": "bool"}))
    digest.update(mask.astype(np.uint8).tobytes(order="C"))
    return digest.hexdigest()


def signed_body(body: dict[str, Any], field: str = "signature") -> dict[str, Any]:
    return {k: v for k, v in body.items()
            if k != field and not (field == "signature" and k == "signature_method")}


def verify_signature(value: dict[str, Any], label: str, field: str = "signature") -> None:
    if value.get(field) != canonical_sha(signed_body(value, field)):
        raise GateError(f"{label} integrity signature failed")


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise GateError(f"Expected JSON object: {path}")
    return value


def write_once(path: Path, value: dict[str, Any]) -> str:
    payload = json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False,
                         indent=2).encode("utf-8") + b"\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError as exc:
        raise GateError(f"Refusing to overwrite write-once artifact: {path}") from exc
    with os.fdopen(fd, "wb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    return hashlib.sha256(payload).hexdigest()


def git(*args: str) -> str:
    return subprocess.run(["git", "-C", str(ROOT), *args], check=True, text=True,
                          stdout=subprocess.PIPE, stderr=subprocess.PIPE).stdout.strip()


def source_hashes() -> dict[str, str]:
    missing = [name for name, path in SOURCE_PATHS.items() if not path.is_file()]
    if missing:
        raise GateError(f"Required source/freeze files missing: {missing}")
    return {name: sha_file(path) for name, path in SOURCE_PATHS.items()}


def verify_sources_match_commit(commit: str, hashes: dict[str, str]) -> None:
    for name, path in SOURCE_PATHS.items():
        relative = path.relative_to(ROOT).as_posix()
        try:
            committed = subprocess.run(
                ["git", "-C", str(ROOT), "show", f"{commit}:{relative}"],
                check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            ).stdout
        except subprocess.CalledProcessError as exc:
            raise GateError(f"Source file is not present in source commit: {name}") from exc
        committed_hash = hashlib.sha256(committed).hexdigest()
        if committed_hash != hashes[name]:
            raise GateError(f"Live source differs from committed blob: {name}")


def verify_metadata_and_hash_inputs() -> dict[str, str]:
    actual: dict[str, str] = {}
    for name, path in INPUT_PATHS.items():
        if not path.is_file():
            raise GateError(f"Required frozen input missing: {name} ({path})")
        actual[name] = sha_file(path)
        if actual[name] != EXPECTED_INPUT_SHA256[name]:
            raise GateError(f"Frozen input hash changed: {name}")

    global_manifest = read_json(INPUT_PATHS["global_manifest"])
    global_qa = read_json(INPUT_PATHS["global_qa"])
    global_state = read_json(INPUT_PATHS["global_state"])
    global_receipt = read_json(INPUT_PATHS["global_fit_receipt"])
    direct_manifest = read_json(INPUT_PATHS["direct_manifest"])
    direct_qa = read_json(INPUT_PATHS["direct_qa"])
    direct_freeze = read_json(INPUT_PATHS["direct_trained_freeze"])
    direct_prelabel = read_json(INPUT_PATHS["direct_prelabel"])
    verify_signature(global_manifest, "global manifest")
    verify_signature(global_qa, "global QA")
    verify_signature(direct_manifest, "direct manifest")
    verify_signature(direct_qa, "direct QA")
    verify_signature(direct_freeze, "direct trained-state freeze")
    verify_signature(direct_prelabel, "direct prelabel receipt")

    global_id_hash = global_manifest.get("ordered_ids_sha256",
                                         global_manifest.get("ordered_official_test_ids_sha256"))
    direct_id_hash = direct_manifest.get("ordered_official_test_ids_sha256")
    if (global_manifest.get("candidate") != "CURRENT_DIRECT_AFFINE_L1_E79"
            or global_manifest.get("csv_sha256") != actual["global_csv"]
            or global_manifest.get("rows") != N_ROWS
            or global_manifest.get("scientific_status") != "IN_SAMPLE_CALIBRATION_UNVALIDATED"
            or global_manifest.get("candidate_refits") != 0
            or global_manifest.get("preprocessing_fits") != 0
            or global_manifest.get("scipy_lp_calls") != 1
            or global_manifest.get("model_predict_calls") != 0
            or global_manifest.get("intercept") != INTERCEPT
            or global_manifest.get("slope") != SLOPE
            or global_id_hash != direct_id_hash):
        raise GateError("Signed existing global manifest identity or fixed parameters changed")
    if (global_state.get("candidate") != "CURRENT_DIRECT_AFFINE_L1_E79"
            or global_state.get("status") != "FIT_SAVED"
            or global_state.get("intercept") != INTERCEPT
            or global_state.get("slope") != SLOPE
            or global_manifest.get("state_sha256") != actual["global_state"]
            or global_receipt.get("status") != "FIT_AND_OUTPUT_SAVED"
            or global_receipt.get("csv_sha256") != actual["global_csv"]
            or global_receipt.get("state_sha256") != actual["global_state"]
            or global_receipt.get("manifest_sha256") != actual["global_manifest"]
            or global_receipt.get("csv_qa_sha256") != actual["global_qa"]
            or global_receipt.get("candidate_fits") != 1
            or global_receipt.get("lp_calls") != 1
            or global_receipt.get("preprocessing_fits") != 0
            or global_receipt.get("test_features_read") is not False
            or global_receipt.get("test_labels_read") is not False
            or global_receipt.get("upload_performed") is not False):
        raise GateError("Existing global state/fit receipt does not bind its frozen candidate")
    if (global_qa.get("status") != "PASS_LOCAL_OUTPUT_QA"
            or global_qa.get("csv_sha256") != actual["global_csv"]
            or global_qa.get("manifest_sha256") != actual["global_manifest"]
            or global_qa.get("rows") != N_ROWS
            or global_qa.get("columns") != ["record_id", "threat_rank"]):
        raise GateError("Existing global QA is not the exact pinned candidate PASS")

    if (direct_manifest.get("candidate") != "B"
            or direct_manifest.get("csv_sha256") != actual["direct_csv"]
            or direct_manifest.get("raw_prediction_archive_sha256") != actual["direct_archive"]
            or direct_manifest.get("prediction_rows") != N_ROWS
            or direct_manifest.get("missing_rows") != N_MISSING
            or direct_manifest.get("missing_fallback_bit_exact") is not True
            or direct_manifest.get("test_labels_loaded") is not False
            or direct_manifest.get("upload_performed") is not False):
        raise GateError("Signed direct manifest identity or label/upload boundary changed")
    if (direct_qa.get("status") != "PASS_INDEPENDENT_CSV_QA"
            or direct_qa.get("csv_sha256") != actual["direct_csv"]
            or direct_qa.get("inference_manifest_sha256") != actual["direct_manifest"]
            or direct_qa.get("missing_rows") != N_MISSING
            or direct_qa.get("missing_fallback_bit_exact") is not True
            or direct_qa.get("test_labels_read") is not False):
        raise GateError("Signed direct QA is not the exact pinned label-blind PASS")
    if (direct_freeze.get("status") != "TRAINED_STATE_FROZEN_EXPLORATORY_ONLY"
            or direct_freeze.get("candidate") != "B"
            or direct_prelabel.get("status") != "PRELABEL_PASS"
            or direct_prelabel.get("test_labels_loaded") is not False
            or direct_prelabel.get("evaluation_label_values_opened") is not False):
        raise GateError("Direct trained-state or prelabel provenance changed")

    decision = INPUT_PATHS["decision"].read_text(encoding="utf-8")
    if "ADVANCE_ONE_B_ONLY_ARTIFACT" not in decision or "No Kaggle upload by Luna" not in decision:
        raise GateError("Root decision no longer contains the exact B-only boundary")
    root_pass = read_json(INPUT_PATHS["r3_root_source_pass"])
    if root_pass.get("verdict") != "PASS":
        raise GateError("R3 root source-pass receipt is not a PASS")
    prior_freeze = read_json(INPUT_PATHS["previous_r1_freeze"])
    prior_failure = read_json(INPUT_PATHS["previous_r1_failure_receipt"])
    verify_signature(prior_freeze, "prior R1 source/input freeze")
    verify_signature(prior_failure, "prior R1 fail-closed application receipt")
    if (prior_freeze.get("source_commit") != "769bcf26440d5396578bc7ef1973028ba5f14774"
            or prior_freeze.get("status") != "SOURCE_AND_INPUTS_FROZEN_BEFORE_DATA_LOAD"
            or prior_failure.get("status") != "FAIL_CLOSED_BEFORE_OUTPUT"
            or prior_failure.get("source_commit") != prior_freeze.get("source_commit")
            or prior_failure.get("freeze_sha256") != actual["previous_r1_freeze"]
            or prior_failure.get("b_only_output_artifacts_created") is not False
            or prior_failure.get("diagnosis", {}).get("direct_manifest_prediction_sha256") !=
            "fde81b7d9d66c6c428920264151f33aac20331affe9b3185199af923be6abb55"):
        raise GateError("Prior R1 freeze/failure accounting does not bind the known fail-closed attempt")
    return actual


def freeze_source_and_inputs() -> None:
    if git("branch", "--show-current") != BRANCH:
        raise GateError("Unexpected branch for B-only application freeze")
    commit = git("rev-parse", "HEAD")
    subprocess.run(["git", "-C", str(ROOT), "merge-base", "--is-ancestor",
                    R3_CLOSURE_COMMIT, commit], check=True,
                   stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    hashes = source_hashes()
    verify_sources_match_commit(commit, hashes)
    input_hashes = verify_metadata_and_hash_inputs()
    output_existing = [str(path) for path in (OUT_CSV, OUT_MANIFEST, APP_RECEIPT,
                                               INDEPENDENT_QA, FREEZE) if path.exists()]
    if output_existing:
        raise GateError(f"B-only write-once outputs already exist: {output_existing}")
    synthetic = read_json(SYNTHETIC_QA)
    expected_synthetic_sources = {name: digest for name, digest in hashes.items()
                                  if name != "synthetic_qa"}
    if (synthetic.get("status") != "PASS_SYNTHETIC_ONLY"
            or synthetic.get("candidate_data_read") is not False
            or synthetic.get("test_or_sample_read") is not False
            or synthetic.get("train_targets_read") is not False
            or synthetic.get("raw_f64_and_shape_tagged_hashes_are_distinct") is not True
            or synthetic.get("wrong_value_mutation_rejected") is not True
            or synthetic.get("reordered_output_ids_rejected") is not True
            or synthetic.get("source_hashes") != expected_synthetic_sources):
        raise GateError("Committed synthetic QA is not an isolated synthetic PASS")
    body = {
        "schema_version": 1,
        "namespace": "KASDAD_REG_FINAL_WAVE_FOLLOWUP_V2_B_ONLY",
        "application_revision": "R2",
        "status": "SOURCE_AND_INPUTS_FROZEN_BEFORE_DATA_LOAD",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "branch": BRANCH,
        "source_commit": commit,
        "r3_closed_ancestor_commit": R3_CLOSURE_COMMIT,
        "source_hashes": hashes,
        "input_hashes": {name: {"path": str(INPUT_PATHS[name]), "sha256": value}
                         for name, value in input_hashes.items()},
        "policy": {
            "operation": "reuse saved global affine values on present rows and restore saved direct values on the signed 1,062-row missing mask",
            "intercept": INTERCEPT,
            "slope": SLOPE,
            "candidate_refits": 0,
            "preprocessing_fits": 0,
            "lp_calls": 0,
            "model_predict_calls": 0,
            "train_target_access": False,
            "test_or_sample_access": False,
            "test_metric_computation": False,
            "upload": False,
        },
        "expected_output": {
            "path": str(OUT_CSV),
            "rows": N_ROWS,
            "columns": ["record_id", "threat_rank"],
            "missing_rows": N_MISSING,
            "ordered_ids_sha256": read_json(INPUT_PATHS["direct_manifest"])["ordered_official_test_ids_sha256"],
            "write_once": True,
        },
        "data_content_loaded_during_freeze": False,
        "csv_rows_parsed_during_freeze": False,
        "npz_arrays_loaded_during_freeze": False,
    }
    signed = {**body, "signature": canonical_sha(body),
              "signature_method": "SHA-256 canonical JSON integrity seal"}
    digest = write_once(FREEZE, signed)
    print(json.dumps({"status": body["status"], "freeze": str(FREEZE),
                      "sha256": digest, "source_commit": commit,
                      "input_count": len(input_hashes), "source_count": len(hashes),
                      "csv_rows_parsed": False, "npz_arrays_loaded": False}, sort_keys=True))


def read_prediction_csv(path: Path) -> tuple[list[str], np.ndarray]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.reader(stream)
        try:
            header = next(reader)
        except StopIteration as exc:
            raise GateError(f"Empty CSV: {path}") from exc
        if header != ["record_id", "threat_rank"]:
            raise GateError(f"Unexpected CSV schema at {path}: {header}")
        ids: list[str] = []
        values: list[float] = []
        for line_no, row in enumerate(reader, start=2):
            if len(row) != 2 or not row[0]:
                raise GateError(f"Malformed CSV row {line_no} at {path}")
            try:
                value = float(row[1])
            except ValueError as exc:
                raise GateError(f"Non-numeric threat_rank at {path}:{line_no}") from exc
            ids.append(row[0])
            values.append(value)
    return ids, np.asarray(values, dtype=np.float64)


def _unique_id_index(ids: list[str] | np.ndarray, label: str) -> dict[str, int]:
    text_ids = [str(v) for v in ids]
    if len(set(text_ids)) != len(text_ids):
        raise GateError(f"Duplicate record_id values in {label}")
    return {value: index for index, value in enumerate(text_ids)}


def compose_present_only(archive_ids: list[str] | np.ndarray,
                         global_ids: list[str] | np.ndarray,
                         global_values: np.ndarray,
                         direct_ids: list[str] | np.ndarray,
                         direct_values: np.ndarray,
                         missing_mask: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Align saved predictions by ID, then apply the fixed missing-row route."""
    archive_ids = np.asarray([str(v) for v in archive_ids], dtype=str)
    mask = np.asarray(missing_mask, dtype=np.bool_)
    global_values = np.asarray(global_values, dtype=np.float64)
    direct_values = np.asarray(direct_values, dtype=np.float64)
    if archive_ids.ndim != 1 or mask.shape != archive_ids.shape:
        raise GateError("Archive IDs and missing mask are not aligned vectors")
    archive_index = _unique_id_index(archive_ids, "saved direct archive")
    global_index = _unique_id_index(global_ids, "global CSV")
    direct_index = _unique_id_index(direct_ids, "direct CSV")
    if set(archive_index) != set(global_index) or set(archive_index) != set(direct_index):
        raise GateError("Global/direct CSV IDs do not match the saved direct archive")
    if len(global_values) != len(global_index) or len(direct_values) != len(direct_index):
        raise GateError("Prediction vector length differs from its ID sequence")
    g = np.asarray([global_values[global_index[key]] for key in archive_ids], dtype=np.float64)
    d = np.asarray([direct_values[direct_index[key]] for key in archive_ids], dtype=np.float64)
    result = g.copy()
    result[mask] = d[mask]
    return archive_ids, g, d, result


def same_float_bits(left: np.ndarray, right: np.ndarray) -> bool:
    a = np.ascontiguousarray(np.asarray(left, dtype="<f8"))
    b = np.ascontiguousarray(np.asarray(right, dtype="<f8"))
    return a.shape == b.shape and a.tobytes(order="C") == b.tobytes(order="C")


def write_prediction_csv(path: Path, ids: np.ndarray, values: np.ndarray) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError as exc:
        raise GateError(f"Refusing to overwrite output CSV: {path}") from exc
    with os.fdopen(fd, "w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(["record_id", "threat_rank"])
        for record_id, value in zip(ids, values, strict=True):
            writer.writerow([str(record_id), format(float(value), ".17g")])
        stream.flush()
        os.fsync(stream.fileno())
    return sha_file(path)


def load_verified_freeze() -> tuple[dict[str, Any], dict[str, str]]:
    freeze = read_json(FREEZE)
    verify_signature(freeze, "B-only source/input freeze")
    current_commit = git("rev-parse", "HEAD")
    if (freeze.get("status") != "SOURCE_AND_INPUTS_FROZEN_BEFORE_DATA_LOAD"
            or freeze.get("branch") != BRANCH
            or freeze.get("r3_closed_ancestor_commit") != R3_CLOSURE_COMMIT):
        raise GateError("B-only freeze is not for this committed source stage")
    try:
        subprocess.run(["git", "-C", str(ROOT), "merge-base", "--is-ancestor",
                        freeze["source_commit"], current_commit], check=True,
                       stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    except (KeyError, subprocess.CalledProcessError) as exc:
        raise GateError("Current HEAD is not descended from the frozen source commit") from exc
    current_sources = source_hashes()
    if freeze.get("source_hashes") != current_sources:
        raise GateError("B-only application source changed after its freeze")
    actual_inputs = verify_metadata_and_hash_inputs()
    expected_inputs = {name: value["sha256"] for name, value in freeze.get("input_hashes", {}).items()}
    if expected_inputs != actual_inputs:
        raise GateError("Input files changed after the B-only freeze")
    return freeze, actual_inputs


def apply_frozen_policy() -> None:
    freeze, input_hashes = load_verified_freeze()
    if any(path.exists() for path in (OUT_CSV, OUT_MANIFEST, APP_RECEIPT, INDEPENDENT_QA)):
        raise GateError("A B-only output already exists; refusing a second application")

    global_ids, global_values = read_prediction_csv(INPUT_PATHS["global_csv"])
    direct_ids, direct_values = read_prediction_csv(INPUT_PATHS["direct_csv"])
    with np.load(INPUT_PATHS["direct_archive"], allow_pickle=False) as archive:
        required = {"record_id", "prediction", "baseline", "missing"}
        if not required.issubset(archive.files):
            raise GateError(f"Direct E79 archive fields changed: {archive.files}")
        archive_ids = np.asarray(archive["record_id"], dtype=str)
        archived_direct = np.asarray(archive["prediction"], dtype=np.float64)
        baseline = np.asarray(archive["baseline"], dtype=np.float64)
        missing = np.asarray(archive["missing"], dtype=np.bool_)
    if (len(global_ids) != N_ROWS or len(direct_ids) != N_ROWS
            or len(archive_ids) != N_ROWS or int(missing.sum()) != N_MISSING
            or sha_ordered_ids(archive_ids) != freeze["expected_output"]["ordered_ids_sha256"]
            or global_ids != archive_ids.tolist() or direct_ids != archive_ids.tolist()
            or sha_vector(global_values) != read_json(INPUT_PATHS["global_manifest"])["prediction_sha256"]
            or sha_raw_f64(direct_values) != read_json(INPUT_PATHS["direct_manifest"])["prediction_sha256"]
            or not np.isfinite(global_values).all() or not np.isfinite(direct_values).all()
            or not np.isfinite(archived_direct).all()
            or np.any((global_values < 0) | (global_values > 100))
            or np.any((direct_values < 0) | (direct_values > 100))
            or not same_float_bits(direct_values, archived_direct)
            or not same_float_bits(direct_values[missing], archived_direct[missing])
            or not same_float_bits(archived_direct[missing], baseline[missing])):
        raise GateError("E79 saved CSV/archive identity, range, or missing fallback failed")
    ids, aligned_global, aligned_direct, result = compose_present_only(
        archive_ids, global_ids, global_values, direct_ids, direct_values, missing)
    if (len(ids) != N_ROWS or len(set(ids.tolist())) != N_ROWS
            or not np.isfinite(result).all() or np.any((result < 0) | (result > 100))
            or not same_float_bits(result[~missing], aligned_global[~missing])
            or not same_float_bits(result[missing], aligned_direct[missing])):
        raise GateError("B-only row alignment, exact-route, or value-range gate failed")
    if same_float_bits(result, aligned_global) or same_float_bits(result, aligned_direct):
        raise GateError("B-only result is an exact duplicate of an existing candidate vector")

    global_changed = int(np.count_nonzero(
        np.asarray(result, dtype="<f8").view("<u8") !=
        np.asarray(aligned_global, dtype="<f8").view("<u8")))
    direct_changed = int(np.count_nonzero(
        np.asarray(result, dtype="<f8").view("<u8") !=
        np.asarray(aligned_direct, dtype="<f8").view("<u8")))
    if global_changed > N_MISSING:
        raise GateError("Output differs from the saved global vector outside the missing-row budget")

    csv_hash = write_prediction_csv(OUT_CSV, ids, result)
    body = {
        "schema_version": 1,
        "namespace": "KASDAD_REG_FINAL_WAVE_FOLLOWUP_V2_B_ONLY",
        "candidate": "GLOBAL_PRESENT_ONLY_RESTORE_DIRECT_MISSING_E79",
        "application_revision": "R2",
        "status": "B_ONLY_ARTIFACT_READY_PENDING_INDEPENDENT_QA",
        "scientific_status": "TEST_PERFORMANCE_NOT_COMPUTED_OR_CLAIMED",
        "csv_path": str(OUT_CSV),
        "csv_sha256": csv_hash,
        "rows": len(ids),
        "columns": ["record_id", "threat_rank"],
        "ordered_ids_sha256": sha_ordered_ids(ids),
        "prediction_sha256": sha_vector(result),
        "range": [float(result.min()), float(result.max())],
        "missing_rows": int(missing.sum()),
        "missing_mask_sha256": sha_mask(missing),
        "rule": "saved global affine prediction on present rows; original saved direct prediction on exact missing mask",
        "global_reference": {
            "candidate": "CURRENT_DIRECT_AFFINE_L1_E79",
            "csv_sha256": input_hashes["global_csv"],
            "scientific_status": "IN_SAMPLE_CALIBRATION_UNVALIDATED",
        },
        "direct_reference": {
            "candidate": "B",
            "csv_sha256": input_hashes["direct_csv"],
            "archive_sha256": input_hashes["direct_archive"],
        },
        "current_best_comparison": {
            "frozen_reference": "CURRENT_DIRECT_AFFINE_L1_E79",
            "reference_csv_sha256": input_hashes["global_csv"],
            "changed_rows_vs_global": global_changed,
            "changed_rows_vs_direct": direct_changed,
            "all_global_differences_confined_to_missing_mask": global_changed <= N_MISSING,
            "test_metric_computed": False,
            "public_score_used": False,
        },
        "source_commit": freeze["source_commit"],
        "source_hashes": freeze["source_hashes"],
        "freeze_sha256": sha_file(FREEZE),
        "input_hashes": {name: value["sha256"] for name, value in freeze["input_hashes"].items()},
        "candidate_refits": 0,
        "preprocessing_fits": 0,
        "lp_calls": 0,
        "model_predict_calls": 0,
        "train_target_access": False,
        "test_or_sample_access": False,
        "test_metric_computed": False,
        "upload_performed": False,
    }
    manifest = {**body, "signature": canonical_sha(body),
                "signature_method": "SHA-256 canonical JSON integrity seal"}
    manifest_hash = write_once(OUT_MANIFEST, manifest)
    app_body = {
        "status": "PASS_B_ONLY_APPLICATION",
        "candidate": body["candidate"],
        "source_commit": freeze["source_commit"],
        "freeze_sha256": sha_file(FREEZE),
        "csv_sha256": csv_hash,
        "manifest_sha256": manifest_hash,
        "rows": len(ids),
        "missing_rows": int(missing.sum()),
        "changed_rows_vs_global": global_changed,
        "changed_rows_vs_direct": direct_changed,
        "range": body["range"],
        "candidate_refits": 0,
        "preprocessing_fits": 0,
        "lp_calls": 0,
        "model_predict_calls": 0,
        "train_target_access": False,
        "test_or_sample_access": False,
        "test_metric_computed": False,
        "upload_performed": False,
    }
    write_once(APP_RECEIPT, {**app_body, "signature": canonical_sha(app_body),
                             "signature_method": "SHA-256 canonical JSON integrity seal"})
    print(json.dumps({"status": app_body["status"], "csv": str(OUT_CSV),
                      "csv_sha256": csv_hash, "manifest": str(OUT_MANIFEST),
                      "manifest_sha256": manifest_hash, "rows": len(ids),
                      "missing_rows": int(missing.sum()),
                      "changed_rows_vs_global": global_changed,
                      "changed_rows_vs_direct": direct_changed,
                      "range": body["range"], "test_metrics_computed": False,
                      "upload_performed": False}, sort_keys=True))


def write_synthetic_qa() -> None:
    ids_archive = np.asarray(["d", "b", "a", "c"])
    ids_global = ["a", "b", "c", "d"]
    values_global = np.asarray([10.0, 20.0, 30.0, 40.0])
    ids_direct = ["c", "a", "d", "b"]
    values_direct = np.asarray([300.0, 100.0, 400.0, 200.0])
    missing = np.asarray([False, True, False, True])
    ids, global_aligned, direct_aligned, result = compose_present_only(
        ids_archive, ids_global, values_global, ids_direct, values_direct, missing)
    expected = np.asarray([40.0, 200.0, 10.0, 300.0])
    if (ids.tolist() != ["d", "b", "a", "c"]
            or not same_float_bits(result, expected)
            or not same_float_bits(result[~missing], global_aligned[~missing])
            or not same_float_bits(result[missing], direct_aligned[missing])
            or same_float_bits(result, global_aligned)
            or same_float_bits(result, direct_aligned)):
        raise GateError("Synthetic row-alignment, missing-route, or duplicate-vector check failed")
    duplicate_rejected = False
    try:
        compose_present_only(ids_archive, ["a", "a", "c", "d"], values_global,
                             ids_direct, values_direct, missing)
    except GateError:
        duplicate_rejected = True
    if not duplicate_rejected:
        raise GateError("Synthetic duplicate-ID mutation was not rejected")
    wrong_value = result.copy()
    wrong_value[0] = np.nextafter(wrong_value[0], np.inf)
    wrong_value_rejected = not same_float_bits(wrong_value, expected)
    reordered_ids = ids[::-1].tolist()
    output_order_rejected = reordered_ids != ids_archive.tolist()
    source_hash_schemes_distinct = sha_raw_f64(values_global) != sha_vector(values_global)
    if not (wrong_value_rejected and output_order_rejected and source_hash_schemes_distinct):
        raise GateError("Synthetic source-hash/value/order mutation check failed")
    body = {
        "schema_version": 1,
        "status": "PASS_SYNTHETIC_ONLY",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "synthetic_rows": len(ids),
        "synthetic_missing_rows": int(missing.sum()),
        "row_alignment_by_record_id": True,
        "global_present_bit_exact": True,
        "direct_missing_bit_exact": True,
        "duplicate_existing_vector_rejected": True,
        "duplicate_id_mutation_rejected": duplicate_rejected,
        "raw_f64_and_shape_tagged_hashes_are_distinct": source_hash_schemes_distinct,
        "wrong_value_mutation_rejected": wrong_value_rejected,
        "reordered_output_ids_rejected": output_order_rejected,
        "candidate_data_read": False,
        "test_or_sample_read": False,
        "train_targets_read": False,
        "candidate_refits": 0,
        "preprocessing_fits": 0,
        "lp_calls": 0,
        "model_predict_calls": 0,
        "source_hashes": {name: sha_file(path) for name, path in SOURCE_PATHS.items()
                          if name != "synthetic_qa"},
    }
    write_once(SYNTHETIC_QA, body)
    print(json.dumps({"status": body["status"], "receipt": str(SYNTHETIC_QA),
                      "synthetic_rows": len(ids), "candidate_data_read": False}, sort_keys=True))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--synthetic-qa", action="store_true")
    group.add_argument("--freeze", action="store_true")
    group.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    try:
        if args.synthetic_qa:
            write_synthetic_qa()
        elif args.freeze:
            freeze_source_and_inputs()
        else:
            apply_frozen_policy()
    except (GateError, OSError, ValueError, KeyError, subprocess.CalledProcessError) as exc:
        print(json.dumps({"status": "FAIL_CLOSED", "error": str(exc)}, sort_keys=True),
              file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
