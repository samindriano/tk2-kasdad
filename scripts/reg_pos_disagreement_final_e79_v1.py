"""Source-only preparation and root-gated final E79 POS disagreement application."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import zipfile
from typing import Any, Mapping

import numpy as np
import pandas as pd
from scipy.stats import rankdata

ROOT = Path(__file__).resolve().parents[1]
LANE = "REG_POS_DISAGREEMENT_FINAL_E79_V1"
REPORT_ROOT = ROOT / "reports" / "optimization_v3" / LANE
ARTIFACT_ROOT = ROOT / "experiments" / "runs" / LANE
PILOT_REPORT = ROOT / "reports" / "optimization_v3" / "REG_POS_DISAGREEMENT_GATE_V1"
PILOT_ARTIFACT = ROOT / "experiments" / "runs" / "REG_POS_DISAGREEMENT_GATE_V1"
PARENT_MANIFEST = ROOT / "reports" / "optimization_v3" / "REG_CURRENT_R06_E79_EXPLORATORY_BATCH_V1" / "INFERENCE_MANIFEST_B.json"
PARENT_ARCHIVE = ROOT / "experiments" / "runs" / "REG_CURRENT_R06_E79_EXPLORATORY_BATCH_V1" / "inference_B.npz"
HINGE_STATE = ROOT / "experiments" / "runs" / "reg_monotone_hinge_p50_v1" / "FINAL_APPLICATION_R1" / "fits" / "HINGE_FINAL.json"
HINGE_REPORT = ROOT / "reports" / "optimization_v3" / "reg_monotone_hinge_p50_v1" / "FINAL_APPLICATION_REPORT_R1.json"
CHAMPION_CSV = ROOT / "submissions" / "reg_monotone_hinge_p50_v1" / "hinge_p50_final_exploratory_e79.csv"
PARENT_RUNNER = ROOT / "scripts" / "optimization_v3_reg_current_r06_e79_exploratory_v1.py"
HINGE_RUNNER = ROOT / "scripts" / "reg_monotone_hinge_p50_v1.py"
PILOT_RUNNER = ROOT / "scripts" / "reg_pos_disagreement_gate_v1.py"
PILOT_VERIFIER = ROOT / "scripts" / "verify_reg_pos_disagreement_gate_v1.py"
PILOT_VERIFIER_AMENDMENT = PILOT_REPORT / "METRIC_QA_VERIFIER_AMENDMENT_R3.json"
PILOT_SOURCE_R1 = PILOT_ARTIFACT / "SOURCE_FREEZE_R1.json"
PILOT_SOURCE_R2 = PILOT_ARTIFACT / "SOURCE_FREEZE_R2.json"
PILOT_PREDICTION_R2 = PILOT_ARTIFACT / "PREDICTION_FREEZE_R2.json"
PILOT_PRELABEL_R2 = PILOT_REPORT / "PRELABEL_QA_R2.json"
PILOT_SCORE_R2 = PILOT_REPORT / "SCORE_R2.json"
PILOT_METRIC_QA_R3 = PILOT_REPORT / "METRIC_QA_R3.json"
ROOT_HEADLINE_REPLAY = PILOT_REPORT / "ROOT_HEADLINE_REPLAY_R1.json"
LEDGER_PATH = PILOT_ARTIFACT / "lane_ledger.jsonl"
SOURCE_ARCHIVE = ARTIFACT_ROOT / "FINAL_E79_SOURCE_INPUTS_R1.npz"
SOURCE_REPORT = REPORT_ROOT / "SOURCE_PREPARATION_R1.json"
SOURCE_QA_REPORT = REPORT_ROOT / "SOURCE_QA_R1.json"
FINAL_PASS = ROOT / "experiments" / LANE / "ROOT_FINAL_SOURCE_PASS_R1.json"
FINAL_RUN_ROOT = ROOT / "experiments" / "runs" / LANE
FINAL_CSV = ROOT / "submissions" / "reg_pos_disagreement_final_e79_v1.csv"

EXPECTED = {
    "parent_manifest": "d2bd7dea17feab1740ca4630bb5cf7ace62caf8746051b3322cc38d168674484",
    "parent_archive": "e29b1eabb1692729749b25d88b7cf86d1bdf44c34ce74ab964b124074f5817f2",
    "parent_raw_score": "4483e6bf2ff96e8baf65435c59d5eddb6348bf2e797a672afa38585e8a2ea338",
    "hinge_state": "a81b2e8e709a33e9a15036ab040c87c92faabd08f16b48ff7f42916fd23ad99a",
    "hinge_report": "fec7182eace04b425b7dbf1d871dbb39f14a1080e479ba95243803e4b1cdae73",
    "champion_csv": "a5813fdd384da31611db1e945c7db33199d8ef9044fb71dd2941ef7af6422da9",
    "root_headline": "b3e60d52b576577b3c18e48007e5d1e9b36f375f217947734fffc01c508d468c",
}
FINAL_HINGE = {"intercept": 1.7288079036674198, "slope_low": 0.9933285599956824,
               "slope_high": 0.957778072288735, "knot": 50.0}
PARENT_ARCHIVE_MEMBERS = {"record_id", "cycle_code", "missing", "prediction", "baseline", "raw"}
PARENT_READ_MEMBERS = ("record_id", "cycle_code", "missing", "baseline", "raw")
OOF_MEMBERS = ("oof_query_rows", "oof_query_ids", "oof_cycles", "oof_missing",
               "oof_baseline", "oof_pos_t50", "oof_disagreement")
E79_MEMBERS = ("e79_record_id", "e79_cycle_code", "e79_missing", "e79_baseline",
               "e79_raw_pos", "e79_pos_t50", "e79_champion_direct", "e79_champion_final")
ALL_SOURCE_MEMBERS = OOF_MEMBERS + E79_MEMBERS
MIN_POSITIVE_DISAGREEMENTS = 200
OOF_EXPECTED_ROWS = 20903
E79_EXPECTED_ROWS = 29421


class SourceError(RuntimeError):
    pass


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _canonical(value: Any) -> bytes:
    return json.dumps(_jsonable(value), sort_keys=True, ensure_ascii=False, allow_nan=False,
                      separators=(",", ":")).encode("utf-8", errors="surrogatepass")


def _canonical_sha(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha_array(value: Any) -> str:
    array = np.asarray(value)
    header = _canonical({"dtype": array.dtype.str, "shape": list(array.shape)})
    if array.dtype.kind in "OUS":
        payload = bytearray()
        for item in array.reshape(-1).tolist():
            raw = str(item).encode("utf-8", errors="surrogatepass")
            payload.extend(len(raw).to_bytes(8, "little")); payload.extend(raw)
        body = bytes(payload)
    else:
        body = np.ascontiguousarray(array).tobytes()
    return hashlib.sha256(len(header).to_bytes(8, "little") + header + body).hexdigest()


def _array_equal(left: Any, right: Any) -> bool:
    a, b = np.asarray(left), np.asarray(right)
    if a.dtype.kind in "fc" or b.dtype.kind in "fc":
        return bool(np.array_equal(a, b, equal_nan=True))
    return bool(np.array_equal(a, b))


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise SourceError(f"Invalid JSON source: {path}") from exc
    if not isinstance(value, dict):
        raise SourceError(f"Expected JSON object: {path}")
    return value


def _verify_signed(value: Mapping[str, Any], key: str, label: str) -> None:
    body = {name: item for name, item in value.items() if name != key}
    if value.get(key) != _canonical_sha(body):
        raise SourceError(f"Signature mismatch: {label}")


def _write_signed_new(path: Path, body: Mapping[str, Any], key: str) -> dict[str, Any]:
    value = dict(body)
    value[key] = _canonical_sha(value)
    payload = _canonical(value) + b"\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError as exc:
        if path.read_bytes() == payload:
            return value
        raise SourceError(f"Refusing to overwrite changed immutable JSON: {path}") from exc
    with os.fdopen(fd, "wb") as stream:
        stream.write(payload); stream.flush(); os.fsync(stream.fileno())
    return value


def _write_npz_new(path: Path, arrays: Mapping[str, Any]) -> dict[str, Any]:
    values = {key: np.asarray(value) for key, value in arrays.items()}
    buffer = io.BytesIO()
    np.savez_compressed(buffer, **values)
    payload = buffer.getvalue()
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError as exc:
        existing = _read_npz_exact(path, tuple(values))
        if any(not _array_equal(values[key], existing[key]) for key in values):
            raise SourceError(f"Refusing to overwrite changed immutable NPZ: {path}") from exc
    else:
        with os.fdopen(fd, "wb") as stream:
            stream.write(payload); stream.flush(); os.fsync(stream.fileno())
    readback = _read_npz_exact(path, tuple(values))
    if any(not _array_equal(values[key], readback[key]) for key in values):
        raise SourceError(f"NPZ readback differs: {path}")
    try:
        relative_path = path.relative_to(ROOT).as_posix()
    except ValueError:
        relative_path = path.name
    return {"path": relative_path, "sha256": _sha_file(path),
            "members": {key: _sha_array(readback[key]) for key in sorted(readback)}}


def _read_npz_exact(path: Path, members: tuple[str, ...]) -> dict[str, np.ndarray]:
    if not path.is_file():
        raise SourceError(f"Required NPZ is missing: {path}")
    with zipfile.ZipFile(path) as archive:
        names = {name[:-4] for name in archive.namelist() if name.endswith(".npy")}
    if names != set(members):
        raise SourceError(f"NPZ member allowlist mismatch at {path}")
    with np.load(path, allow_pickle=False) as archive:
        return {key: np.asarray(archive[key]).copy() for key in members}


def _read_npz_subset(path: Path, all_members: set[str], read_members: tuple[str, ...]) -> dict[str, np.ndarray]:
    if not path.is_file():
        raise SourceError(f"Required source archive is missing: {path}")
    with zipfile.ZipFile(path) as archive:
        names = {name[:-4] for name in archive.namelist() if name.endswith(".npy")}
    if names != all_members or not set(read_members).issubset(names):
        raise SourceError(f"Parent NPZ inventory/allowlist mismatch: {sorted(names)}")
    with np.load(path, allow_pickle=False) as archive:
        return {key: np.asarray(archive[key]).copy() for key in read_members}


def _load_module(path: Path, name: str):
    if not path.is_file():
        raise SourceError(f"Required module is missing: {path}")
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise SourceError(f"Cannot import required module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _parent_runner():
    return _load_module(PARENT_RUNNER, "reg_current_r06_e79_parent_for_disagreement")


def _hinge_module():
    return _load_module(HINGE_RUNNER, "reg_monotone_hinge_for_disagreement_final_e79")


def _hinge_parameters() -> dict[str, float]:
    value = _read_json(HINGE_STATE)
    solver = value.get("solver_result", {})
    params = {"intercept": solver.get("intercept"), "slope_low": solver.get("slope_low"),
              "slope_high": solver.get("slope_high"), "knot": 50.0}
    if value.get("status") != "LP_COMPLETE" or params != FINAL_HINGE:
        raise SourceError("Frozen final hinge values differ from the pinned champion contract")
    return {key: float(item) for key, item in params.items()}


def _rank50(raw: np.ndarray, cycles: np.ndarray) -> np.ndarray:
    mapped = np.empty(len(raw), dtype=np.float64)
    for cycle in np.unique(cycles):
        mask = cycles == cycle
        mapped[mask] = 100.0 * rankdata(raw[mask], method="average") / int(mask.sum())
    return mapped


def _direct_from_parent_raw(raw: np.ndarray, baseline: np.ndarray, cycles: np.ndarray,
                            missing: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if not (raw.shape == baseline.shape == cycles.shape == missing.shape):
        raise SourceError("Parent raw/baseline/cycle/missing vectors are misaligned")
    # This is the original single POS stream transform; no replica mean is read or used.
    t50 = 0.5 * raw + 0.5 * _rank50(raw, cycles)
    direct = np.clip(0.75 * baseline + 0.25 * t50, 0.0, 100.0)
    direct[missing] = baseline[missing]
    return t50, direct


def _candidate_from_scale(pos_t50: np.ndarray, baseline: np.ndarray,
                          missing: np.ndarray, scale_s: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    disagreement = np.abs(pos_t50 - baseline)
    weight = 0.25 * np.exp(-disagreement / scale_s)
    direct = np.clip(baseline + weight * (pos_t50 - baseline), 0.0, 100.0)
    direct[missing] = baseline[missing]
    return disagreement, weight, direct


def _csv_bytes(ids: np.ndarray, values: np.ndarray) -> bytes:
    if ids.ndim != 1 or values.ndim != 1 or len(ids) != len(values):
        raise SourceError("CSV IDs and prediction values are misaligned")
    frame = pd.DataFrame({"record_id": ids.astype(str), "threat_rank": values.astype(np.float64)})
    return frame.to_csv(index=False, float_format="%.17g", lineterminator="\n").encode("utf-8")


def _read_csv_roundtrip(path: Path) -> tuple[np.ndarray, np.ndarray]:
    frame = pd.read_csv(path, dtype={"record_id": "string"}, float_precision="round_trip")
    if list(frame.columns) != ["record_id", "threat_rank"]:
        raise SourceError("Synthetic/final CSV header differs from the competition schema")
    return (frame["record_id"].astype(str).to_numpy(dtype=str),
            pd.to_numeric(frame["threat_rank"], errors="raise").to_numpy(dtype=np.float64))


def _source_dependencies() -> dict[str, Any]:
    files = {
        "pilot_source_r1": PILOT_SOURCE_R1,
        "pilot_source_r2": PILOT_SOURCE_R2,
        "pilot_prediction_freeze_r2": PILOT_PREDICTION_R2,
        "pilot_prelabel_r2": PILOT_PRELABEL_R2,
        "pilot_score_r2": PILOT_SCORE_R2,
        "pilot_metric_qa_r3": PILOT_METRIC_QA_R3,
        "pilot_verifier_amendment_r3": PILOT_VERIFIER_AMENDMENT,
        "root_headline_replay_r1": ROOT_HEADLINE_REPLAY,
        "lane_ledger": LEDGER_PATH,
        "parent_manifest": PARENT_MANIFEST,
        "parent_archive": PARENT_ARCHIVE,
        "hinge_state": HINGE_STATE,
        "hinge_report": HINGE_REPORT,
        "champion_csv": CHAMPION_CSV,
    }
    missing = [name for name, path in files.items() if not path.is_file()]
    if missing:
        raise SourceError(f"Frozen source dependencies missing: {missing}")
    hashes = {name: _sha_file(path) for name, path in files.items()}
    if (hashes["parent_manifest"] != EXPECTED["parent_manifest"]
            or hashes["parent_archive"] != EXPECTED["parent_archive"]
            or hashes["hinge_state"] != EXPECTED["hinge_state"]
            or hashes["hinge_report"] != EXPECTED["hinge_report"]
            or hashes["champion_csv"] != EXPECTED["champion_csv"]
            or hashes["root_headline_replay_r1"] != EXPECTED["root_headline"]):
        raise SourceError("One or more pinned E79 source hashes differ")

    source_r1 = _read_json(PILOT_SOURCE_R1)
    source_r2 = _read_json(PILOT_SOURCE_R2)
    prediction = _read_json(PILOT_PREDICTION_R2)
    prelabel = _read_json(PILOT_PRELABEL_R2)
    score = _read_json(PILOT_SCORE_R2)
    qa = _read_json(PILOT_METRIC_QA_R3)
    amendment = _read_json(PILOT_VERIFIER_AMENDMENT)
    headline = _read_json(ROOT_HEADLINE_REPLAY)
    _verify_signed(source_r1, "source_signature", "pilot source R1")
    _verify_signed(source_r2, "source_signature", "pilot source R2")
    _verify_signed(prediction, "prediction_signature", "pilot prediction freeze R2")
    _verify_signed(prelabel, "qa_signature", "pilot prelabel QA R2")
    _verify_signed(score, "score_signature", "pilot score R2")
    _verify_signed(qa, "qa_signature", "pilot metric QA R3")
    _verify_signed(amendment, "amendment_signature", "pilot verifier amendment R3")
    if (source_r2.get("status") != "SOURCE_AMENDED_R2"
            or source_r2.get("source_signature_r1") != source_r1.get("source_signature")
            or source_r2.get("source_freeze_r1_sha256") != hashes["pilot_source_r1"]
            or prediction.get("status") != "PREDICTIONS_FROZEN"
            or prediction.get("source_signature") != source_r2.get("source_signature")
            or prelabel.get("status") != "PASS"
            or prelabel.get("source_signature") != source_r2.get("source_signature")
            or prelabel.get("prediction_signature") != prediction.get("prediction_signature")
            or prelabel.get("query_targets_read") is not False
            or score.get("status") != "SCORED"
            or score.get("source_signature") != source_r2.get("source_signature")
            or score.get("prediction_signature") != prediction.get("prediction_signature")
            or score.get("prelabel_qa_sha256") != hashes["pilot_prelabel_r2"]
            or qa.get("status") != "PASS"
            or qa.get("source_signature") != source_r2.get("source_signature")
            or qa.get("prediction_signature") != prediction.get("prediction_signature")
            or qa.get("prelabel_qa_signature") != prelabel.get("qa_signature")
            or qa.get("score_signature") != score.get("score_signature")
            or qa.get("verifier_amendment_sha256") != hashes["pilot_verifier_amendment_r3"]
            or qa.get("target_values_persisted") is not False
            or qa.get("test_or_sample_accessed") is not False
            or qa.get("scientific_qualification") is not True
            or qa.get("exploratory_benchmark_eligible") is not True
            or qa.get("fixed_groups_known") is not True
            or qa.get("maximum_supported_REAL_group_harm") is None
            or qa.get("maximum_supported_REAL_group_harm") > 0.25):
        raise SourceError("Pilot source, prediction, score, or group QA binding failed")
    if (amendment.get("status") != "VERIFIER_AMENDED_R3"
            or amendment.get("source_freeze_sha256") != hashes["pilot_source_r2"]
            or amendment.get("prediction_freeze_sha256") != hashes["pilot_prediction_freeze_r2"]
            or amendment.get("prelabel_qa_sha256") != hashes["pilot_prelabel_r2"]
            or amendment.get("score_sha256") != hashes["pilot_score_r2"]
            or amendment.get("old_verifier_sha256") != deps_source_verifier_hash(source_r2)
            or amendment.get("new_verifier_sha256") != _sha_file(PILOT_VERIFIER)):
        raise SourceError("Pilot metric verifier amendment does not bind frozen R2 scoring inputs")
    if (headline.get("status") != "PASS_ROOT_INDEPENDENT_HEADLINE_METRIC_REPLAY"
            or headline.get("score_sha256") != hashes["pilot_score_r2"]
            or headline.get("test_access") is not False
            or headline.get("new_fits") != 0
            or headline.get("new_candidate_policies") != 0
            or headline.get("windows", {}).get("D1", {}).get("REAL_MAE_gain") != qa.get("D1_REAL_MAE_gain")
            or headline.get("windows", {}).get("D2", {}).get("REAL_MAE_gain") != qa.get("D2_REAL_MAE_gain")
            or headline.get("pooled_REAL_MAE_gain") != qa.get("pooled_REAL_MAE_gain")
            or headline.get("pooled_REAL_over_CONTROL_gain") != qa.get("pooled_REAL_over_CONTROL_gain")):
        raise SourceError("Root headline replay does not bind the independently verified score")
    if (score.get("pooled_rows") != qa.get("pooled_rows")
            or score.get("scientific_performance_gate") is not True
            or score.get("exploratory_performance_gate") is not True):
        raise SourceError("Pilot score eligibility fields differ from metric QA")
    return {"hashes": hashes, "source": source_r2, "prediction": prediction,
            "prelabel": prelabel, "score": score, "metric_qa": qa,
            "headline": headline, "verifier_amendment": amendment}


def deps_source_verifier_hash(source: Mapping[str, Any]) -> str | None:
    hashes = source.get("source_code_hashes", {})
    value = hashes.get("independent_verifier")
    return str(value) if value is not None else None


def _audit_ledger(deps: Mapping[str, Any]) -> dict[str, Any]:
    events = [json.loads(line) for line in LEDGER_PATH.read_text(encoding="utf-8").splitlines() if line]
    policy_type = "deterministic_target_free_policy_construction"
    policy = [item for item in events if item.get("operation_type") == policy_type]
    starts = [item for item in policy if item.get("event") == "OPERATION_STARTED"]
    saved = [item for item in policy if item.get("event") == "OPERATION_SAVED"]
    failed = [item for item in policy if item.get("event") == "OPERATION_FAILED"]
    expected_ids = {"REAL_D1", "CONTROL_D1", "REAL_D1_R2", "CONTROL_D1_R2", "REAL_D2", "CONTROL_D2"}
    if ({item.get("operation_id") for item in starts} != expected_ids or len(starts) != 6
            or {item.get("operation_id") for item in failed} != {"REAL_D1", "CONTROL_D1"}
            or len(failed) != 2 or {item.get("operation_id") for item in saved} != expected_ids - {"REAL_D1", "CONTROL_D1"}
            or len(saved) != 4):
        raise SourceError("Policy ledger does not match six authorized starts, two retained failures and four saves")
    prediction = deps["prediction"]
    artifact_by_op = {item.get("operation_id"): item.get("sha256")
                      for item in prediction.get("predictions", {}).values()}
    for item in saved:
        if item.get("artifact_sha256") != artifact_by_op.get(item.get("operation_id")):
            raise SourceError(f"Policy ledger saved receipt differs: {item.get('operation_id')}")
    for receipt in prediction.get("predictions", {}).values():
        prediction_path = ROOT / receipt["path"]
        if _sha_file(prediction_path) != receipt.get("sha256"):
            raise SourceError(f"Saved policy archive changed: {receipt['path']}")

    scalar_events = [item for item in events if item.get("operation_type") == "target_free_scalar_scale_state"]
    scalar_starts = [item for item in scalar_events if item.get("event") == "OPERATION_STARTED"]
    scalar_saved = [item for item in scalar_events if item.get("event") == "OPERATION_SAVED"]
    scalar_failed = [item for item in scalar_events if item.get("event") == "OPERATION_FAILED"]
    if ({item.get("operation_id") for item in scalar_starts} != {"SCALE_D1", "SCALE_D2"}
            or len(scalar_starts) != 2 or {item.get("operation_id") for item in scalar_saved} != {"SCALE_D1", "SCALE_D2"}
            or len(scalar_saved) != 2 or scalar_failed):
        raise SourceError("Two saved scale states do not reconcile to the policy ledger")
    source = deps["source"]
    for name, receipt in source.get("scale_states", {}).items():
        ledger_receipt = next((item for item in scalar_saved if item.get("operation_id") == f"SCALE_{name}"), None)
        state_path = ROOT / receipt["path"]
        if (ledger_receipt is None or ledger_receipt.get("artifact_sha256") != receipt.get("sha256")
                or _sha_file(state_path) != receipt.get("sha256")):
            raise SourceError(f"Scale ledger receipt differs from source freeze: {name}")

    amendment_events = [item for item in events if item.get("operation_id") == "SOURCE_AMENDMENT_R2"]
    if (sum(item.get("event") == "OPERATION_STARTED" for item in amendment_events) != 1
            or sum(item.get("event") == "OPERATION_SAVED" for item in amendment_events) != 1
            or sum(item.get("event") == "OPERATION_FAILED" for item in amendment_events) != 0
            or not any(item.get("event") == "OPERATION_SAVED"
                       and item.get("artifact_sha256") == deps["hashes"]["pilot_source_r2"]
                       for item in amendment_events)):
        raise SourceError("R2 source amendment lane receipt differs")
    if any(item.get("outputs_written") != [] for item in failed):
        raise SourceError("Consumed failed D1 starts unexpectedly wrote policy artifacts")
    if prediction.get("policy_start_allowance_total") != 6 or prediction.get("successful_policy_artifact_cap") != 4:
        raise SourceError("Prediction freeze does not record the amended policy budget")
    if prediction.get("model_fits_started") != 0 or prediction.get("scale_state_fits_started") != 0:
        raise SourceError("Prediction freeze reports unauthorized additional fits")
    return {"policy_starts": 6, "policy_failures_retained": 2, "policy_artifacts_saved": 4,
            "scale_states_saved": 2, "scale_state_refits": 0,
            "ledger_sha256": deps["hashes"]["lane_ledger"]}


def _load_oof_source(deps: Mapping[str, Any]) -> dict[str, np.ndarray]:
    source = deps["source"]
    pieces: list[dict[str, np.ndarray]] = []
    for origin, low, high in ((67, 68, 75), (75, 76, 79)):
        receipt = source["origins"][str(origin)]["archive"]
        path = ROOT / receipt["path"]
        if _sha_file(path) != receipt["sha256"]:
            raise SourceError(f"Frozen OOF source archive changed at E{origin}")
        arrays = _read_npz_exact(path, tuple(receipt["members"]))
        if {key: _sha_array(value) for key, value in arrays.items()} != receipt["members"]:
            raise SourceError(f"Frozen OOF source member digest changed at E{origin}")
        cycles = arrays["cycles"].astype(np.int64)
        keep = (cycles >= low) & (cycles <= high)
        pieces.append({
            "query_rows": arrays["query_rows"][keep].astype(np.int64),
            "query_ids": arrays["query_ids"][keep].astype(str),
            "cycles": cycles[keep], "missing": arrays["missing"][keep].astype(bool),
            "baseline": arrays["baseline"][keep].astype(np.float64),
            "pos_t50": arrays["pos_t50"][keep].astype(np.float64),
        })
    combined = {key: np.concatenate([part[key] for part in pieces])
                for key in ("query_rows", "query_ids", "cycles", "missing", "baseline", "pos_t50")}
    order = np.argsort(combined["query_rows"], kind="stable")
    combined = {key: value[order] for key, value in combined.items()}
    if (len(combined["query_rows"]) != OOF_EXPECTED_ROWS
            or len(np.unique(combined["query_rows"])) != OOF_EXPECTED_ROWS
            or len(np.unique(combined["query_ids"])) != OOF_EXPECTED_ROWS
            or not np.isfinite(combined["baseline"]).all()
            or not np.isfinite(combined["pos_t50"]).all()):
        raise SourceError("Unique final OOF scalar source population or finiteness check failed")
    disagreement = np.abs(combined["pos_t50"] - combined["baseline"])
    positive = (~combined["missing"]) & (disagreement > 0.0)
    if int(positive.sum()) < MIN_POSITIVE_DISAGREEMENTS:
        raise SourceError("Final OOF scalar source has fewer than 200 positive present disagreements")
    return {
        "oof_query_rows": combined["query_rows"], "oof_query_ids": combined["query_ids"],
        "oof_cycles": combined["cycles"], "oof_missing": combined["missing"],
        "oof_baseline": combined["baseline"], "oof_pos_t50": combined["pos_t50"],
        "oof_disagreement": disagreement,
    }


def _load_e79_parent() -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    manifest = _read_json(PARENT_MANIFEST)
    if (manifest.get("namespace") != "REG_CURRENT_R06_E79_EXPLORATORY_BATCH_V1"
            or manifest.get("candidate") != "B"
            or manifest.get("candidate_id") != "DIRECT_POSITION_REAL_E79"
            or manifest.get("baseline_member_used") != "manifest-bound final mapped R06 prediction"
            or manifest.get("baseline_raw_candidate_member_used") is not False
            or manifest.get("raw_prediction_archive_sha256") != EXPECTED["parent_archive"]
            or manifest.get("raw_score_sha256") != EXPECTED["parent_raw_score"]
            or manifest.get("test_labels_loaded") is not False
            or manifest.get("upload_performed") is not False):
        raise SourceError("E79 parent manifest does not bind the approved raw POS/R06 baseline source")
    parent = _read_npz_subset(PARENT_ARCHIVE, PARENT_ARCHIVE_MEMBERS, PARENT_READ_MEMBERS)
    ids = parent["record_id"].astype(str)
    cycles = parent["cycle_code"].astype(np.int64)
    missing = parent["missing"].astype(bool)
    baseline = parent["baseline"].astype(np.float64)
    raw = parent["raw"].astype(np.float64)
    n = len(ids)
    if (n != E79_EXPECTED_ROWS or len(np.unique(ids)) != n
            or cycles.shape != ids.shape
            or not np.array_equal(np.unique(cycles), np.arange(80, 92, dtype=np.int64))
            or missing.shape != ids.shape or baseline.shape != ids.shape or raw.shape != ids.shape
            or not np.isfinite(baseline).all() or not np.isfinite(raw).all()
            or np.any((baseline < 0.0) | (baseline > 100.0))):
        raise SourceError("Frozen E79 parent input arrays fail row/cycle/range checks")
    producer = _parent_runner()
    raw_digest = producer.sha_array(raw, dtype="<f8")
    if raw_digest != manifest["raw_score_sha256"]:
        raise SourceError("Original raw POS digest differs from the parent producer format")
    t50, direct = _direct_from_parent_raw(raw, baseline, cycles, missing)
    # Cross-check the exact producer implementation without loading its prediction member.
    producer_direct = producer._candidate_output_prediction("B", raw, cycles, missing, baseline)
    if not np.array_equal(direct, producer_direct):
        raise SourceError("Single-stream parent T50/direct formula differs from the frozen producer")
    hinge = _hinge_module().apply_hinge_present_only(direct, missing, _hinge_parameters())
    champion = pd.read_csv(CHAMPION_CSV, dtype={"record_id": "string"}, float_precision="round_trip")
    if list(champion.columns) != ["record_id", "threat_rank"]:
        raise SourceError("Frozen champion CSV schema differs")
    champion_ids = champion["record_id"].astype(str).to_numpy(dtype=str)
    champion_values = pd.to_numeric(champion["threat_rank"], errors="raise").to_numpy(dtype=np.float64)
    if (not np.array_equal(ids, champion_ids)
            or not np.array_equal(hinge.view("<u8"), champion_values.view("<u8"))
            or not np.array_equal(hinge[missing].view("<u8"), baseline[missing].view("<u8"))):
        raise SourceError("Original champion direct/hinge replay failed bit-exact parity")
    arrays = {
        "e79_record_id": ids, "e79_cycle_code": cycles, "e79_missing": missing,
        "e79_baseline": baseline, "e79_raw_pos": raw, "e79_pos_t50": t50,
        "e79_champion_direct": direct, "e79_champion_final": hinge,
    }
    evidence = {
        "archive_member_inventory": sorted(PARENT_ARCHIVE_MEMBERS),
        "members_loaded": list(PARENT_READ_MEMBERS), "unused_parent_prediction_member_loaded": False,
        "raw_digest_format": "parent_runner.sha_array(raw, dtype='<f8')",
        "raw_digest": raw_digest, "raw_digest_matches_parent_receipt": True,
        "t50_formula": "0.5 * original_raw_POS + 0.5 * full-cycle_average_rank_100(original_raw_POS)",
        "champion_direct_formula": "clip(0.75 * R06_baseline + 0.25 * t50, 0, 100); missing uses baseline",
        "producer_direct_replay_bit_exact": True, "champion_ids_exact": True,
        "champion_hinge_replay_bit_exact": True, "champion_rows": n,
        "cycle_counts": {str(int(cycle)): int((cycles == cycle).sum()) for cycle in np.unique(cycles)},
        "missing_rows": int(missing.sum()),
    }
    return arrays, evidence


def _source_code_hashes() -> dict[str, str]:
    paths = {
        "runner": Path(__file__),
        "independent_verifier": ROOT / "scripts" / "verify_reg_pos_disagreement_final_e79_v1.py",
        "source_regression_test": ROOT / "tests" / "test_reg_pos_disagreement_final_e79_source_v1.py",
        "pilot_runner": PILOT_RUNNER, "pilot_verifier": PILOT_VERIFIER,
        "parent_runner": PARENT_RUNNER, "hinge_runner": HINGE_RUNNER,
    }
    missing = [name for name, path in paths.items() if not path.is_file()]
    if missing:
        raise SourceError(f"Source package code inputs missing: {missing}")
    return {name: _sha_file(path) for name, path in paths.items()}


def _source_only() -> dict[str, Any]:
    if FINAL_PASS.exists() or FINAL_RUN_ROOT.exists() or FINAL_CSV.exists():
        raise SourceError("Final pass or E79 output already exists; source-only stage will not continue")
    deps = _source_dependencies()
    ledger = _audit_ledger(deps)
    oof = _load_oof_source(deps)
    e79, parent_evidence = _load_e79_parent()
    arrays = {**oof, **e79}
    if set(arrays) != set(ALL_SOURCE_MEMBERS):
        raise SourceError("Source archive members differ from the explicit allowlist")
    code_hashes = _source_code_hashes()
    synthetic = _synthetic_e2e()
    receipt = _write_npz_new(SOURCE_ARCHIVE, arrays)
    positive_disagreements = int(((~oof["oof_missing"]) & (oof["oof_disagreement"] > 0.0)).sum())
    body = {
        "schema_version": 1, "namespace": LANE, "status": "SOURCE_READY_PENDING_ROOT_SOURCE_PASS",
        "source_script": Path(__file__).relative_to(ROOT).as_posix(),
        "source_code_hashes": code_hashes,
        "source_archive": receipt,
        "source_dependencies": deps["hashes"],
        "pilot_bindings": {
            "source_signature": deps["source"]["source_signature"],
            "prediction_signature": deps["prediction"]["prediction_signature"],
            "prelabel_qa_signature": deps["prelabel"]["qa_signature"],
            "score_signature": deps["score"]["score_signature"],
            "metric_qa_signature": deps["metric_qa"]["qa_signature"],
            "root_headline_score_sha256": deps["headline"]["score_sha256"],
        },
        "pilot_gate_qualification": {
            "scientific": deps["metric_qa"]["scientific_qualification"],
            "exploratory_eligible": deps["metric_qa"]["exploratory_benchmark_eligible"],
            "fixed_groups_known": deps["metric_qa"]["fixed_groups_known"],
            "maximum_supported_REAL_group_harm": deps["metric_qa"]["maximum_supported_REAL_group_harm"],
            "pooled_REAL_MAE_gain": deps["metric_qa"]["pooled_REAL_MAE_gain"],
            "D1_REAL_MAE_gain": deps["metric_qa"]["D1_REAL_MAE_gain"],
            "D2_REAL_MAE_gain": deps["metric_qa"]["D2_REAL_MAE_gain"],
            "pooled_REAL_over_CONTROL_gain": deps["metric_qa"]["pooled_REAL_over_CONTROL_gain"],
        },
        "ledger_qa": ledger,
        "final_scale_source": {
            "oof_rows": OOF_EXPECTED_ROWS, "cycles": [68, 79],
            "origin_windows": {"E67": [68, 75], "E75": [76, 79]},
            "positive_present_disagreements": positive_disagreements,
            "minimum_required": MIN_POSITIVE_DISAGREEMENTS,
            "formula": "median(abs(pos_t50 - baseline)) over unique present OOF rows with strictly positive disagreement",
            "scale_s_computed": False, "scale_state_fit_started": False,
            "target_labels_read_for_final_scale": False,
        },
        "e79_parent_inventory": parent_evidence,
        "final_candidate_contract": {
            "policy": "REAL only; CONTROL remains diagnostic and is not promoted",
            "scale_budget": 1, "application_budget": 1,
            "predictor_fits": 0, "preprocessor_fits": 0, "hinge_fits": 0,
            "original_parent_replay_before_candidate": True,
            "weight": "0.25 * exp(-abs(p - b) / s)",
            "direct_present": "clip(b + weight * (p - b), 0, 100)",
            "missing": "exact frozen R06 baseline bits",
            "hinge": "fixed saved final hinge on present rows only; missing remains baseline",
            "final_scale_state_fit_started": False, "real_e79_application_started": False,
        },
        "synthetic_e2e": synthetic,
        "root_pass_required": FINAL_PASS.relative_to(ROOT).as_posix(),
        "execution_boundary": {
            "test_file_opened": False, "sample_file_opened": False,
            "test_labels_read": False, "sample_rows_read": False,
            "real_e79_candidate_application": False, "real_submission_csv_created": False,
            "kaggle_accessed": False, "upload_performed": False,
            "final_csv_path_after_root_pass": FINAL_CSV.relative_to(ROOT).as_posix(),
        },
    }
    report = _write_signed_new(SOURCE_REPORT, body, "source_manifest_signature")
    return {"status": report["status"], "source_report": SOURCE_REPORT.relative_to(ROOT).as_posix(),
            "source_report_sha256": _sha_file(SOURCE_REPORT),
            "source_manifest_signature": report["source_manifest_signature"],
            "source_archive_sha256": receipt["sha256"],
            "oof_rows": OOF_EXPECTED_ROWS, "e79_parent_rows": E79_EXPECTED_ROWS,
            "positive_present_disagreements": positive_disagreements,
            "scale_state_fit_started": False, "real_e79_application_started": False}


def _verify_root_pass() -> dict[str, Any]:
    if not FINAL_PASS.is_file():
        raise SourceError("Final E79 application requires the root-signed source pass")
    approval = _read_json(FINAL_PASS)
    _verify_signed(approval, "approval_signature", "root final source pass")
    source_report = _read_json(SOURCE_REPORT)
    scope = approval.get("scope", {})
    expected = {
        "source_report_sha256": _sha_file(SOURCE_REPORT),
        "source_manifest_signature": source_report.get("source_manifest_signature"),
        "source_archive_sha256": _sha_file(SOURCE_ARCHIVE),
        "metric_qa_sha256": source_report["source_dependencies"]["pilot_metric_qa_r3"],
        "root_headline_sha256": source_report["source_dependencies"]["root_headline_replay_r1"],
        "prediction_freeze_sha256": source_report["source_dependencies"]["pilot_prediction_freeze_r2"],
        "final_scale_state_fit_cap": 1, "final_application_cap": 1,
        "predictor_fits": 0, "preprocessor_fits": 0, "hinge_fits": 0,
        "test_labels_access_allowed": False, "sample_access_allowed": False,
        "upload_allowed": False,
    }
    if approval.get("status") != "PASS" or approval.get("execute_final_source") is not True:
        raise SourceError("Root source pass does not authorize this final E79 candidate")
    if any(scope.get(key) != value for key, value in expected.items()):
        raise SourceError("Root pass scope does not bind the exact source package and budget")
    return approval


def _append_ledger(path: Path, event: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(_canonical(event).decode("utf-8") + "\n")
        stream.flush(); os.fsync(stream.fileno())


def _single_start(ledger: Path, operation_id: str, operation_type: str) -> None:
    if ledger.exists() and ledger.stat().st_size:
        raise SourceError(f"Final operation already has ledger state; retries are forbidden: {operation_id}")
    _append_ledger(ledger, {"schema_version": 1, "event": "OPERATION_STARTED",
                            "operation_id": operation_id, "operation_type": operation_type})


def _execute_final() -> dict[str, Any]:
    approval = _verify_root_pass()
    if FINAL_RUN_ROOT.exists() or FINAL_CSV.exists():
        raise SourceError("Final E79 output exists; overwrite/retry is forbidden")
    source = _read_json(SOURCE_REPORT)
    _verify_signed(source, "source_manifest_signature", "source preparation")
    deps = _source_dependencies()
    if source.get("source_dependencies") != deps["hashes"]:
        raise SourceError("Frozen source dependencies changed after root pass")
    arrays = _read_npz_exact(SOURCE_ARCHIVE, tuple(source["source_archive"]["members"]))
    if _sha_file(SOURCE_ARCHIVE) != source["source_archive"]["sha256"]:
        raise SourceError("Frozen final source NPZ hash changed")
    if {key: _sha_array(value) for key, value in arrays.items()} != source["source_archive"]["members"]:
        raise SourceError("Frozen final source NPZ member hashes changed")
    hinge_params = _hinge_parameters()
    hinge = _hinge_module()
    positive = (~arrays["oof_missing"].astype(bool)) & (arrays["oof_disagreement"] > 0.0)
    scale_values = arrays["oof_disagreement"][positive].astype(np.float64)
    if len(scale_values) < MIN_POSITIVE_DISAGREEMENTS:
        raise SourceError("Final scalar source is unsupported")
    run = FINAL_RUN_ROOT
    ledger = run / "final_ledger.jsonl"
    _single_start(ledger, "FINAL_SCALE_R1", "target_free_scalar_scale_state")
    try:
        scale_s = float(np.median(scale_values))
        scale_body = {"schema_version": 1, "namespace": LANE,
                      "status": "FIT_COMPLETE", "scale_s": scale_s,
                      "positive_present_disagreements": int(len(scale_values)),
                      "source_archive_sha256": source["source_archive"]["sha256"],
                      "source_report_sha256": _sha_file(SOURCE_REPORT),
                      "labels_used": False, "test_accessed": False}
        scale_state = _write_signed_new(run / "final_scale_state_R1.json", scale_body, "state_signature")
        _append_ledger(ledger, {"schema_version": 1, "event": "OPERATION_SAVED",
                                "operation_id": "FINAL_SCALE_R1", "artifact_sha256": _sha_file(run / "final_scale_state_R1.json")})

        _append_ledger(ledger, {"schema_version": 1, "event": "OPERATION_STARTED",
                                "operation_id": "FINAL_APPLICATION_R1", "operation_type": "deterministic_candidate_application"})
        baseline = arrays["e79_baseline"].astype(np.float64)
        p = arrays["e79_pos_t50"].astype(np.float64)
        missing = arrays["e79_missing"].astype(bool)
        disagreement, weight, direct = _candidate_from_scale(p, baseline, missing, scale_s)
        final = hinge.apply_hinge_present_only(direct, missing, hinge_params)
        if (not np.isfinite(final).all() or np.any((final < 0.0) | (final > 100.0))
                or not np.array_equal(final[missing].view("<u8"), baseline[missing].view("<u8"))):
            raise SourceError("Final candidate range or exact missing fallback failed")
        prediction_arrays = {
            "record_id": arrays["e79_record_id"], "cycle_code": arrays["e79_cycle_code"],
            "missing": missing, "baseline": baseline, "pos_t50": p,
            "disagreement": disagreement, "weight": weight,
            "champion_direct": arrays["e79_champion_direct"],
            "champion_final": arrays["e79_champion_final"],
            "candidate_direct": direct, "candidate_final": final,
        }
        prediction_path = run / "predictions" / "E79.npz"
        prediction_receipt = _write_npz_new(prediction_path, prediction_arrays)
        receipt = {"schema_version": 1, "namespace": LANE,
                   "prediction_path": prediction_path.relative_to(ROOT).as_posix(),
                   "prediction_sha256": prediction_receipt["sha256"],
                   "members": prediction_receipt["members"],
                   "source_manifest_signature": source["source_manifest_signature"],
                   "final_pass_sha256": _sha_file(FINAL_PASS),
                   "scale_state_sha256": _sha_file(run / "final_scale_state_R1.json"),
                   "final_application_count": 1, "predictor_fits": 0,
                   "test_labels_read": False, "sample_accessed": False}
        receipt = _write_signed_new(run / "predictions" / "E79.json", receipt, "receipt_signature")
        readback = _read_npz_exact(prediction_path, tuple(prediction_receipt["members"]))
        replay_disagreement, replay_weight, replay_direct = _candidate_from_scale(
            readback["pos_t50"], readback["baseline"], readback["missing"].astype(bool), scale_s)
        replay_final = hinge.apply_hinge_present_only(replay_direct, readback["missing"].astype(bool), hinge_params)
        if (not np.array_equal(readback["disagreement"], replay_disagreement)
                or not np.array_equal(readback["weight"], replay_weight)
                or not np.array_equal(readback["candidate_direct"], replay_direct)
                or not np.array_equal(readback["candidate_final"], replay_final)):
            raise SourceError("Final prediction archive readback/replay failed")
        _append_ledger(ledger, {"schema_version": 1, "event": "OPERATION_SAVED",
                                "operation_id": "FINAL_APPLICATION_R1", "artifact_sha256": prediction_receipt["sha256"]})
        if FINAL_CSV.exists():
            raise SourceError("Final CSV exists; overwrite is forbidden")
        csv_payload = _csv_bytes(readback["record_id"].astype(str), readback["candidate_final"])
        FINAL_CSV.parent.mkdir(parents=True, exist_ok=True)
        try:
            fd = os.open(FINAL_CSV, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        except FileExistsError as exc:
            raise SourceError("Final CSV appeared during application") from exc
        with os.fdopen(fd, "wb") as stream:
            stream.write(csv_payload); stream.flush(); os.fsync(stream.fileno())
        csv_ids, csv_values = _read_csv_roundtrip(FINAL_CSV)
        if (not np.array_equal(csv_ids, readback["record_id"].astype(str))
                or not np.array_equal(csv_values, readback["candidate_final"])):
            raise SourceError("Final CSV round-trip differs from saved candidate prediction")
        manifest = {
            "schema_version": 1, "namespace": LANE, "status": "CSV_READY_PENDING_INDEPENDENT_QA",
            "source_manifest_signature": source["source_manifest_signature"],
            "final_pass_sha256": _sha_file(FINAL_PASS), "scale_state_signature": scale_state["state_signature"],
            "scale_s": scale_s, "scale_state_fits": 1, "candidate_applications": 1,
            "predictor_fits": 0, "preprocessor_fits": 0, "hinge_fits": 0,
            "prediction_archive_sha256": prediction_receipt["sha256"],
            "prediction_receipt_sha256": _sha_file(run / "predictions" / "E79.json"),
            "csv_path": FINAL_CSV.relative_to(ROOT).as_posix(), "csv_sha256": _sha_file(FINAL_CSV),
            "csv_rows": int(len(csv_ids)), "test_labels_read": False,
            "sample_accessed": False, "kaggle_accessed": False, "upload_performed": False,
            "root_source_pass_signature": approval["approval_signature"],
        }
        _write_signed_new(run / "FINAL_MANIFEST_R1.json", manifest, "manifest_signature")
        return {"status": manifest["status"], "csv_path": manifest["csv_path"],
                "csv_sha256": manifest["csv_sha256"], "rows": manifest["csv_rows"],
                "scale_state_fits": 1, "candidate_applications": 1}
    except Exception as exc:
        _append_ledger(ledger, {"schema_version": 1, "event": "OPERATION_FAILED",
                                "operation_id": "FINAL_APPLICATION_R1", "error": f"{type(exc).__name__}: {exc}"})
        raise


def _synthetic_e2e() -> dict[str, Any]:
    rng = np.random.default_rng(20261106)
    hinge_params = _hinge_parameters()
    hinge = _hinge_module()
    cycles = np.repeat(np.arange(68, 80, dtype=np.int64), 64)
    rows = np.arange(len(cycles), dtype=np.int64)
    missing_oof = (rows % 13) == 0
    baseline_oof = np.clip(45.0 + (cycles - 68) * 0.7 + rng.normal(0, 2.0, len(rows)), 0.0, 100.0)
    pos_oof = np.clip(baseline_oof + rng.normal(0.0, 5.0, len(rows)), 0.0, 100.0)
    pos_oof[rows % 29 == 0] = baseline_oof[rows % 29 == 0]
    delta = np.abs(pos_oof - baseline_oof)
    eligible = (~missing_oof) & (delta > 0.0)
    synthetic_s = float(np.median(delta[eligible]))
    query_ids = np.asarray([f"SYN-E79-{index:04d}" for index in range(96)], dtype=str)
    query_cycles = np.repeat(np.asarray([80, 81, 82], dtype=np.int64), 32)
    missing = (np.arange(96) % 11) == 0
    baseline = np.clip(rng.normal(50.0, 18.0, len(query_ids)), 0.0, 100.0)
    raw = rng.normal(0.0, 2.0, len(query_ids))
    pos = 0.5 * raw + 0.5 * _rank50(raw, query_cycles)
    champion_direct = np.clip(0.75 * baseline + 0.25 * pos, 0.0, 100.0)
    champion_direct[missing] = baseline[missing]
    champion_final = hinge.apply_hinge_present_only(champion_direct, missing, hinge_params)
    disagreement, weight, candidate_direct = _candidate_from_scale(pos, baseline, missing, synthetic_s)
    candidate_final = hinge.apply_hinge_present_only(candidate_direct, missing, hinge_params)
    arrays = {
        "record_id": query_ids, "cycle_code": query_cycles, "missing": missing,
        "baseline": baseline, "pos_t50": pos, "disagreement": disagreement,
        "weight": weight, "champion_direct": champion_direct,
        "champion_final": champion_final, "candidate_direct": candidate_direct,
        "candidate_final": candidate_final,
    }
    with tempfile.TemporaryDirectory(prefix="reg-pos-final-e79-synthetic-") as name:
        temp = Path(name)
        archive = temp / "E79.npz"
        receipt = _write_npz_new(archive, arrays)
        receipt_body = {"schema_version": 1, "synthetic": True,
                        "prediction_sha256": receipt["sha256"], "members": receipt["members"],
                        "scale_state_fits": 1, "candidate_applications": 1}
        signed_receipt = _write_signed_new(temp / "E79.json", receipt_body, "receipt_signature")
        restored = _read_npz_exact(archive, tuple(receipt["members"]))
        if signed_receipt["prediction_sha256"] != _sha_file(archive):
            raise SourceError("Synthetic E79 receipt did not bind the archive")
        replay_d, replay_w, replay_direct = _candidate_from_scale(
            restored["pos_t50"], restored["baseline"], restored["missing"].astype(bool), synthetic_s)
        replay_final = hinge.apply_hinge_present_only(replay_direct, restored["missing"].astype(bool), hinge_params)
        if (not np.array_equal(restored["disagreement"], replay_d)
                or not np.array_equal(restored["weight"], replay_w)
                or not np.array_equal(restored["candidate_direct"], replay_direct)
                or not np.array_equal(restored["candidate_final"], replay_final)):
            raise SourceError("Synthetic E79 receipt/archive application replay failed")
        csv_path = temp / "candidate.csv"
        csv_path.write_bytes(_csv_bytes(query_ids, replay_final))
        csv_ids, csv_values = _read_csv_roundtrip(csv_path)
        if not np.array_equal(csv_ids, query_ids) or not np.array_equal(csv_values, replay_final):
            raise SourceError("Synthetic candidate CSV round-trip failed")
    return {"status": "PASS", "synthetic_oof_rows": int(len(rows)),
            "synthetic_e79_rows": int(len(query_ids)), "temporary_archive_receipt_roundtrip": True,
            "temporary_candidate_csv_roundtrip": True, "real_inputs_read": False,
            "real_scale_state_fit_started": False, "real_e79_application_started": False,
            "test_or_sample_accessed": False, "temporary_outputs_removed": True}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--source-only", action="store_true", help="freeze target-free final source and run synthetic E2E")
    modes.add_argument("--synthetic-e2e", action="store_true", help="run isolated synthetic fit/application/receipt/CSV flow")
    modes.add_argument("--execute", action="store_true", help="requires the exact root-signed final source pass")
    args = parser.parse_args(argv)
    try:
        if args.synthetic_e2e:
            result = _synthetic_e2e()
        elif args.execute:
            result = _execute_final()
        else:
            result = _source_only()
    except Exception as exc:
        print(json.dumps({"status": "BLOCKED", "error": f"{type(exc).__name__}: {exc}"}, sort_keys=True))
        return 1
    print(json.dumps(_jsonable(result), sort_keys=True, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
