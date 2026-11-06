"""Bounded, target-gated POS disagreement pilot for the frozen full REG champion."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
from typing import Any, Mapping
import zipfile

import numpy as np
import pandas as pd
import psutil

ROOT = Path(__file__).resolve().parents[1]
LANE = "REG_POS_DISAGREEMENT_GATE_V1"
CARD_ID = LANE
REPORT_ROOT = ROOT / "reports" / "optimization_v3" / LANE
RECIPE_PATH = REPORT_ROOT / "RECIPE.json"
ROOT_PLAN_PATH = REPORT_ROOT / "ROOT_PLAN.md"
ROOT_AMENDMENT_PATH = REPORT_ROOT / "ROOT_FLOAT_ORDER_AMENDMENT_R2.md"
ARTIFACT_ROOT = ROOT / "experiments" / "runs" / LANE
SOURCE_FREEZE_R1 = ARTIFACT_ROOT / "SOURCE_FREEZE_R1.json"
SOURCE_FREEZE = ARTIFACT_ROOT / "SOURCE_FREEZE_R2.json"
PREDICTION_FREEZE = ARTIFACT_ROOT / "PREDICTION_FREEZE_R2.json"
LEDGER_PATH = ARTIFACT_ROOT / "lane_ledger.jsonl"
PRELABEL_QA_PATH = REPORT_ROOT / "PRELABEL_QA_R2.json"
SCORE_PATH = REPORT_ROOT / "SCORE_R2.json"
METRIC_QA_PATH = REPORT_ROOT / "METRIC_QA_R2.json"
FINAL_SOURCE_PATH = ARTIFACT_ROOT / "FINAL_OOF_DISAGREEMENT_SOURCE_R2.npz"
FINAL_SOURCE_MANIFEST = REPORT_ROOT / "FINAL_SOURCE_R2.json"
VERIFIER_PATH = ROOT / "scripts" / "verify_reg_pos_disagreement_gate_v1.py"
HELPER_PATH = ROOT / "scripts" / "reg_pos_poisson_bag2_v1.py"
FRONTIER_ROOT = Path(r"C:\Users\Sam\OneDrive\Documents\KASDAD\tk2-regression-frontier")
RUNNER_PATH = FRONTIER_ROOT / "scripts" / "reg_position_direct_l1_v1.py"
HINGE_FINAL_PATH = ROOT / "experiments" / "runs" / "reg_monotone_hinge_p50_v1" / "FINAL_APPLICATION_R1" / "fits" / "HINGE_FINAL.json"
HINGE_FINAL_FREEZE_PATH = ROOT / "experiments" / "reg_monotone_hinge_p50_v1" / "FINAL_APPLICATION_FREEZE_R1.json"
SCORE_WINDOWS = {67: (72, 75), 75: (76, 79)}
SCALE_WINDOWS = {"D1": (68, 71), "D2": (68, 75)}
MIN_POSITIVE_DISAGREEMENTS = 200
EXPECTED_POOLED_ROWS = 15953
FEATURE_MEMORY_START = 1024 * 1024**2
FEATURE_MEMORY_RESERVE = 512 * 1024**2
SOURCE_OPERATION_ID = "SOURCE_PASS_CACHE_R3"


class GateError(RuntimeError):
    pass


def _load_module(path: Path, name: str):
    if not path.is_file():
        raise GateError(f"Required source file missing: {path}")
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise GateError(f"Cannot import source module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _helpers():
    bag2 = _load_module(HELPER_PATH, "reg_pos_poisson_bag2_v1_for_disagreement")
    return bag2, bag2._runner()


def _sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
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
    return json.dumps(_jsonable(value), sort_keys=True, ensure_ascii=False,
                      allow_nan=False, separators=(",", ":")).encode("utf-8", errors="surrogatepass")


def _canonical_sha(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _sha_array(values: Any) -> str:
    arr = np.asarray(values)
    header = _canonical({"dtype": arr.dtype.str, "shape": list(arr.shape)})
    if arr.dtype.kind in "OUS":
        payload = bytearray()
        for value in arr.reshape(-1).tolist():
            raw = str(value).encode("utf-8", errors="surrogatepass")
            payload.extend(len(raw).to_bytes(8, "little")); payload.extend(raw)
        body = bytes(payload)
    else:
        body = np.ascontiguousarray(arr).tobytes()
    return hashlib.sha256(len(header).to_bytes(8, "little") + header + body).hexdigest()


def _json_read(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise GateError(f"Invalid JSON: {path}") from exc
    if not isinstance(value, dict):
        raise GateError(f"Expected a JSON object: {path}")
    return value


def _signed(body: Mapping[str, Any], signature_key: str) -> dict[str, Any]:
    result = dict(body)
    result[signature_key] = _canonical_sha(result)
    return result


def _write_json_immutable(path: Path, body: Mapping[str, Any], signature_key: str | None = None) -> dict[str, Any]:
    payload = _signed(body, signature_key) if signature_key else dict(body)
    encoded = _canonical(payload) + b"\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != encoded:
            raise GateError(f"Refusing to overwrite changed immutable JSON: {path}")
        return payload
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(encoded); stream.flush(); os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)
    return payload


def _arrays_equal(left: Mapping[str, np.ndarray], right: Mapping[str, np.ndarray]) -> bool:
    if set(left) != set(right):
        return False
    for key in left:
        a, b = np.asarray(left[key]), np.asarray(right[key])
        if a.dtype.kind in "fc" or b.dtype.kind in "fc":
            if not np.array_equal(a, b, equal_nan=True):
                return False
        elif not np.array_equal(a, b):
            return False
    return True


def _write_npz_immutable(path: Path, arrays: Mapping[str, Any]) -> dict[str, Any]:
    values = {key: np.asarray(value) for key, value in arrays.items()}
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        current = _read_npz(path, tuple(values))
        if not _arrays_equal(values, current):
            raise GateError(f"Refusing to overwrite changed immutable NPZ: {path}")
    else:
        fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".npz", dir=path.parent)
        os.close(fd)
        try:
            with Path(temporary).open("wb") as stream:
                np.savez_compressed(stream, **values)
                stream.flush(); os.fsync(stream.fileno())
            os.replace(temporary, path)
        finally:
            Path(temporary).unlink(missing_ok=True)
        current = _read_npz(path, tuple(values))
        if not _arrays_equal(values, current):
            raise GateError(f"NPZ readback differs immediately after write: {path}")
    try:
        relative_path = path.relative_to(ROOT).as_posix()
    except ValueError:
        relative_path = path.name
    return {"path": relative_path, "sha256": _sha_file(path),
            "members": {key: _sha_array(current[key]) for key in sorted(current)}}


def _read_npz(path: Path, keys: tuple[str, ...]) -> dict[str, np.ndarray]:
    if not path.is_file():
        raise GateError(f"Required NPZ missing: {path}")
    with zipfile.ZipFile(path) as archive:
        available = {name[:-4] for name in archive.namelist() if name.endswith(".npy")}
    if available != set(keys):
        raise GateError(f"NPZ member allowlist mismatch at {path}: expected {sorted(keys)}, got {sorted(available)}")
    with np.load(path, allow_pickle=False) as archive:
        return {key: np.asarray(archive[key]).copy() for key in keys}


def _verify_npz_receipt(path: Path, receipt: Mapping[str, Any]) -> dict[str, np.ndarray]:
    if _sha_file(path) != receipt.get("sha256"):
        raise GateError(f"NPZ file hash differs from receipt: {path}")
    members = receipt.get("members")
    if not isinstance(members, dict):
        raise GateError(f"NPZ member receipt is missing: {path}")
    arrays = _read_npz(path, tuple(members))
    if {key: _sha_array(value) for key, value in arrays.items()} != members:
        raise GateError(f"NPZ member hash differs from receipt: {path}")
    return arrays


def _append_event(event: Mapping[str, Any]) -> None:
    ARTIFACT_ROOT.mkdir(parents=True, exist_ok=True)
    LEDGER_PATH.parent.mkdir(parents=True, exist_ok=True)
    with LEDGER_PATH.open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(_canonical(event).decode("utf-8") + "\n")
        stream.flush(); os.fsync(stream.fileno())


def _ledger() -> list[dict[str, Any]]:
    if not LEDGER_PATH.exists():
        return []
    events = []
    for number, line in enumerate(LEDGER_PATH.read_text(encoding="utf-8").splitlines(), 1):
        try:
            event = json.loads(line)
        except Exception as exc:
            raise GateError(f"Malformed lane ledger at line {number}") from exc
        if not isinstance(event, dict):
            raise GateError(f"Malformed lane ledger event at line {number}")
        events.append(event)
    return events


def _operation_start(operation_id: str, operation_type: str) -> None:
    events = [event for event in _ledger() if event.get("operation_id") == operation_id]
    if any(event.get("event") == "OPERATION_FAILED" for event in events):
        raise GateError(f"Operation has a preserved failure event: {operation_id}")
    if not any(event.get("event") == "OPERATION_STARTED" for event in events):
        _append_event({"schema_version": 1, "event": "OPERATION_STARTED", "operation_id": operation_id,
                       "operation_type": operation_type, "utc": datetime.now(timezone.utc).isoformat()})


def _operation_saved(operation_id: str, operation_type: str, artifact_sha256: str) -> None:
    events = [event for event in _ledger() if event.get("operation_id") == operation_id
              and event.get("event") == "OPERATION_SAVED"]
    if events:
        if len(events) != 1 or events[0].get("artifact_sha256") != artifact_sha256:
            raise GateError(f"Saved lane operation conflicts with current artifact: {operation_id}")
        return
    _append_event({"schema_version": 1, "event": "OPERATION_SAVED", "operation_id": operation_id,
                   "operation_type": operation_type, "artifact_sha256": artifact_sha256,
                   "utc": datetime.now(timezone.utc).isoformat()})


def _resource_check() -> dict[str, int]:
    available = int(psutil.virtual_memory().available)
    required = FEATURE_MEMORY_START + FEATURE_MEMORY_RESERVE
    if available < required:
        raise GateError(f"Feature rebuild blocked: available RAM {available} < required {required}")
    return {"available_ram_bytes": available, "initial_ram_floor_bytes": FEATURE_MEMORY_START,
            "reserve_bytes": FEATURE_MEMORY_RESERVE, "threads": 2}


def _load_origin_payload(bag2: Any, runner: Any, frame: Any, position_freeze: Mapping[str, Any],
                         hinges: Mapping[int, Mapping[str, float]], hinge_module: Any,
                         origin: int) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    # Read the immutable target-free source caches directly. The historical replay
    # helper also validates/reconstructs cached preprocessors; its E67 STRUCTURED
    # receipt currently fails, although the original prediction receipts remain usable.
    fit_rows, query_rows = runner._prefix_rows(frame, origin)
    all_ids = frame.record_id.astype(str).to_numpy(dtype=str)
    all_cycles = frame.cycle_code.to_numpy(dtype=np.int64)
    _, all_missing = runner._text_values(frame.signal_stream)
    query_ids = all_ids[query_rows]
    query_cycles = all_cycles[query_rows]
    missing = np.asarray(all_missing[query_rows], dtype=bool)
    baseline, baseline_audit = runner._read_oof_cache(bag2.SOURCE_ROOT, frame, origin, query_rows)
    fit_id = f"{runner.CARD_ID}_E{origin}_REAL"
    prediction_path, receipt_path = runner._prediction_paths(fit_id, runner.RUN_ROOT)
    model_path, model_receipt_path = runner._unit_paths(fit_id, runner.RUN_ROOT)
    if not model_path.is_file() or not model_receipt_path.is_file():
        raise GateError(f"Original POS model artifact/receipt is missing at E{origin}")
    model_receipt = runner._read_json(model_receipt_path)
    if (model_receipt.get("schema_version") != 1 or model_receipt.get("fit_id") != fit_id
            or model_receipt.get("freeze_signature") != position_freeze.get("freeze_signature")
            or model_receipt.get("artifact_sha256") != _sha_file(model_path)
            or model_receipt.get("receipt_sha256") != runner._canonical_sha(runner._receipt_body(model_receipt))):
        raise GateError(f"Original POS model receipt/hash/schema failed at E{origin}")
    ledger_path = runner.RUN_ROOT / "fit_ledger.jsonl"
    ledger = runner._read_events(ledger_path)
    starts = [event for event in ledger if event.get("event") == "FIT_STARTED" and event.get("fit_id") == fit_id]
    saves = [event for event in ledger if event.get("event") == "FIT_SAVED" and event.get("fit_id") == fit_id]
    fails = [event for event in ledger if event.get("event") == "FIT_FAILED" and event.get("fit_id") == fit_id]
    if (len(starts) != 1 or len(saves) != 1 or fails
            or saves[0].get("receipt_sha256") != _sha_file(model_receipt_path)):
        raise GateError(f"Original POS fit ledger does not prove one completed fit at E{origin}")
    saved = runner._validate_prediction_bundle(
        prediction_path, receipt_path, freeze=position_freeze, fit_id=fit_id, origin=origin,
        arm="REAL", query_rows=query_rows, query_ids=query_ids,
        query_cycles=query_cycles, missing=missing, baseline=baseline)
    prediction_receipt = runner._read_json(receipt_path)
    p = runner._rank50(saved["raw_auxiliary"], query_cycles)
    receipt_digests = {
        "query_ids_sha256": runner._sha_array(query_ids),
        "query_cycles_sha256": runner._sha_array(query_cycles),
        "raw_auxiliary_sha256": runner._sha_array(saved["raw_auxiliary"]),
        "mapped_auxiliary_sha256": runner._sha_array(saved["mapped_auxiliary"]),
        "prediction_sha256": runner._sha_array(saved["prediction"]),
    }
    if any(prediction_receipt.get(key) != value for key, value in receipt_digests.items()):
        raise GateError(f"Original POS prediction member digests failed at E{origin}")
    if prediction_receipt.get("model_receipt_sha256") != model_receipt.get("receipt_sha256"):
        raise GateError(f"Original POS prediction is not bound to its durable model receipt at E{origin}")
    champion_direct = np.clip(0.75 * baseline + 0.25 * p, 0.0, 100.0)
    champion_direct[missing] = baseline[missing]
    if not np.array_equal(champion_direct, saved["prediction"]):
        raise GateError(f"Original full-champion DIRECT reconstruction differs at E{origin}")
    champion_hinge = hinge_module.apply_hinge_present_only(champion_direct, missing, hinges[origin])
    query_frame = frame.iloc[query_rows]
    fit_frame = frame.iloc[fit_rows]
    group_features: dict[str, np.ndarray] = {}
    group_edges: dict[str, list[float]] = {}
    for name in ("impact_load", "text_load"):
        prefix = pd.to_numeric(fit_frame[name], errors="coerce").to_numpy(dtype=np.float64)
        query = pd.to_numeric(query_frame[name], errors="coerce").to_numpy(dtype=np.float64)
        if not np.isfinite(prefix).all() or not np.isfinite(query).all():
            raise GateError(f"Nonfinite frozen group feature {name} at E{origin}")
        edges = np.unique(np.quantile(prefix, [0.25, 0.5, 0.75], method="linear")).astype(float)
        group_features[name] = np.digitize(query, edges, right=False).astype(np.int8)
        group_edges[name] = edges.tolist()
    group_features["entry_friction"] = pd.to_numeric(
        query_frame.entry_friction, errors="coerce").to_numpy(dtype=np.float64)
    item = SimpleNamespace(query_rows=query_rows, query_ids=query_ids,
                           query_cycles=query_cycles, query_missing=missing)
    comparator = bag2._compare_comparator_rows(item, origin, champion_direct, champion_hinge)
    arrays = {
        "query_rows": np.asarray(query_rows, dtype=np.int64),
        "query_ids": np.asarray(query_ids, dtype=str),
        "cycles": np.asarray(query_cycles, dtype=np.int64),
        "missing": missing,
        "baseline": baseline,
        "raw_pos": np.asarray(saved["raw_auxiliary"], dtype=np.float64),
        "pos_t50": p,
        "champion_direct": champion_direct,
        "champion_hinge": np.asarray(champion_hinge, dtype=np.float64),
        "impact_quartile": np.asarray(group_features["impact_load"], dtype=np.int8),
        "text_quartile": np.asarray(group_features["text_load"], dtype=np.int8),
        "entry_friction": np.asarray(group_features["entry_friction"], dtype=np.float64),
    }
    lengths = {len(value) for value in arrays.values()}
    if len(lengths) != 1 or not all(np.isfinite(arrays[key]).all()
                                     for key in ("baseline", "raw_pos", "pos_t50", "champion_direct",
                                                 "champion_hinge", "entry_friction")):
        raise GateError(f"Nonfinite or misaligned source vectors at E{origin}")
    if not np.array_equal(champion_hinge[missing].view("<u8"), baseline[missing].view("<u8")):
        raise GateError(f"Original champion missing fallback is not bit exact at E{origin}")
    info = {"source_method": "direct target-free immutable prediction and OOF caches; cached preprocessor replay bypassed after E67 STRUCTURED receipt mismatch",
            "position_fit_id": fit_id, "position_model_path_sha256": _sha_file(model_path),
            "position_model_receipt_sha256": _sha_file(model_receipt_path),
            "position_prediction_path_sha256": _sha_file(prediction_path),
            "position_prediction_receipt_sha256": _sha_file(receipt_path),
            "position_fit_ledger_sha256": _sha_file(ledger_path),
            "baseline_cache_audit": baseline_audit, "comparator": comparator,
            "prediction_path_sha256": _sha_file(prediction_path),
            "prediction_receipt_sha256": _sha_file(receipt_path),
            "raw_pos_sha256": _sha_array(arrays["raw_pos"]),
            "pos_t50_sha256": _sha_array(p), "baseline_sha256": _sha_array(baseline),
            "champion_direct_sha256": _sha_array(champion_direct),
            "champion_hinge_sha256": _sha_array(champion_hinge),
            "query_rows_sha256": _sha_array(arrays["query_rows"]),
            "query_ids_sha256": _sha_array(arrays["query_ids"]),
            "query_cycles_sha256": _sha_array(arrays["cycles"]),
            "missing_rows": int(missing.sum()),
            "group_edges": group_edges}
    return arrays, info


def _source_code_hashes(bag2: Any) -> dict[str, str]:
    files = {"runner": RUNNER_PATH, "bag2_helper": HELPER_PATH,
             "gate_runner": Path(__file__), "independent_verifier": VERIFIER_PATH,
             "regression_tests": ROOT / "tests" / "test_reg_pos_disagreement_gate_v1.py",
             "recipe": RECIPE_PATH, "root_plan": ROOT_PLAN_PATH,
             "root_amendment": ROOT_AMENDMENT_PATH,
             "hinge_runner": bag2.HINGE_RUNNER,
             "final_hinge": HINGE_FINAL_PATH, "final_hinge_freeze": HINGE_FINAL_FREEZE_PATH}
    missing = [str(path) for path in files.values() if not path.is_file()]
    if missing:
        raise GateError(f"Required frozen source/hinge file missing: {missing}")
    return {name: _sha_file(path) for name, path in files.items()}


def source_amendment_stage() -> dict[str, Any]:
    if not SOURCE_FREEZE_R1.is_file():
        raise GateError("R2 amendment requires the preserved original source freeze")
    original = _json_read(SOURCE_FREEZE_R1)
    original_body = {key: value for key, value in original.items() if key != "source_signature"}
    if original.get("source_signature") != _canonical_sha(original_body) or original.get("status") != "SOURCE_FROZEN":
        raise GateError("Preserved R1 source freeze signature/status failed")
    if (_sha_file(RECIPE_PATH) != original.get("recipe_sha256")
            or _sha_file(ROOT_PLAN_PATH) != original.get("root_plan_sha256")):
        raise GateError("Recipe or initial root plan changed since R1 source freeze")
    bag2, runner = _helpers()
    source_artifacts = bag2._source_artifact_hashes(runner)
    if source_artifacts != original.get("source_artifacts"):
        raise GateError("Original source inputs or cached predictions changed since R1")
    scale_receipts = {}
    ledger = _ledger()
    for name in ("D1", "D2"):
        path = ARTIFACT_ROOT / f"SCALE_{name}_R1.json"
        state = _json_read(path)
        body = {key: value for key, value in state.items() if key != "state_signature"}
        if (state.get("state_signature") != _canonical_sha(body)
                or state.get("state_id") != name or state.get("labels_used") is not False):
            raise GateError(f"Saved target-free scalar state is invalid: {name}")
        events = [event for event in ledger if event.get("operation_id") == f"SCALE_{name}"]
        starts = [event for event in events if event.get("event") == "OPERATION_STARTED"]
        saves = [event for event in events if event.get("event") == "OPERATION_SAVED"]
        failures = [event for event in events if event.get("event") == "OPERATION_FAILED"]
        if (len(starts) != 1 or len(saves) != 1 or failures
                or saves[0].get("artifact_sha256") != _sha_file(path)):
            raise GateError(f"Saved scalar state receipt/ledger is not reusable: {name}")
        scale_receipts[name] = {"path": path.relative_to(ROOT).as_posix(),
                                "sha256": _sha_file(path),
                                "state_signature": state["state_signature"]}
    for origin in (67, 75):
        receipt = original["origins"][str(origin)]["archive"]
        path = ROOT / receipt["path"]
        if _sha_file(path) != receipt["sha256"]:
            raise GateError(f"Preserved source archive hash changed at E{origin}")
        arrays = _read_npz(path, tuple(receipt["members"]))
        if {key: _sha_array(value) for key, value in arrays.items()} != receipt["members"]:
            raise GateError(f"Preserved source archive members changed at E{origin}")
    final_hinge_hashes = {"fit": _sha_file(HINGE_FINAL_PATH), "freeze": _sha_file(HINGE_FINAL_FREEZE_PATH)}
    if final_hinge_hashes != original.get("final_hinge_hashes"):
        raise GateError("Final deployed hinge files changed since original source freeze")
    body = {
        "schema_version": 1, "experiment_id": CARD_ID, "status": "SOURCE_AMENDED_R2",
        "source_signature_r1": original["source_signature"],
        "source_freeze_r1_sha256": _sha_file(SOURCE_FREEZE_R1),
        "recipe_sha256": original["recipe_sha256"],
        "root_plan_sha256": original["root_plan_sha256"],
        "root_amendment_sha256": _sha_file(ROOT_AMENDMENT_PATH),
        "amendment_commit": bag2._git(ROOT, "rev-parse", "HEAD"),
        "local_root": original["local_root"], "local_git_branch": original["local_git_branch"],
        "local_git_head_r1": original["local_git_head"],
        "local_git_head_r2": bag2._git(ROOT, "rev-parse", "HEAD"),
        "source_root": original["source_root"], "official_train_sha256": original["official_train_sha256"],
        "position_freeze_signature": original["position_freeze_signature"],
        "source_artifacts": source_artifacts, "source_code_hashes": _source_code_hashes(bag2),
        "final_hinge_hashes": final_hinge_hashes, "resource_gate": original["resource_gate"],
        "origins": original["origins"], "scale_states": scale_receipts,
        "amendment": {
            "reason": "Correct frozen champion comparator arithmetic order after exact bitwise mismatch; preserve candidate formulas, scale states, hinges, input archives and labels boundary.",
            "policy_formula_changed": False, "weight_or_scale_changed": False,
            "source_archives_reused_byte_for_byte": True, "scale_states_reused_byte_for_byte": True,
            "query_targets_read": False, "test_or_sample_accessed": False,
            "base_pos_preprocessor_calibrator_optimizer_fits": 0,
        },
    }
    _operation_start("SOURCE_AMENDMENT_R2", "immutable_source_freeze_arithmetic_amendment")
    amended = _write_json_immutable(SOURCE_FREEZE, body, "source_signature")
    _operation_saved("SOURCE_AMENDMENT_R2", "immutable_source_freeze_arithmetic_amendment", _sha_file(SOURCE_FREEZE))
    return {"status": amended["status"], "source_signature": amended["source_signature"],
            "source_freeze_sha256": _sha_file(SOURCE_FREEZE), "reused_r1_source": True,
            "reused_saved_scale_states": list(scale_receipts)}


def source_stage() -> dict[str, Any]:
    ARTIFACT_ROOT.mkdir(parents=True, exist_ok=True)
    if SOURCE_FREEZE.exists():
        freeze = verify_source_freeze()
        return {"status": freeze["status"], "source_signature": freeze["source_signature"],
                "source_freeze_sha256": _sha_file(SOURCE_FREEZE), "reused": True}
    if SOURCE_FREEZE_R1.exists():
        return source_amendment_stage()
    # The helper-replay R1 and metadata-assembly R2 failures are preserved in the lane ledger.
    resource = _resource_check()
    bag2, runner = _helpers()
    with bag2._SharedHeavyJobLock():
        position_freeze = bag2._validate_signed_json(bag2.POSITION_FREEZE, "freeze_signature")
        frame = runner._load_feature_frame(bag2.SOURCE_ROOT)
        if "threat_rank" in frame.columns:
            raise GateError("Target-free source loader exposed the target column")
        hinges = bag2._load_hinge_parameters()
        hinge_module = bag2._hinge()
        pending: dict[int, tuple[dict[str, np.ndarray], dict[str, Any]]] = {}
        for origin in (67, 75):
            arrays, info = _load_origin_payload(bag2, runner, frame, position_freeze,
                                                hinges, hinge_module, origin)
            pending[origin] = (arrays, info)
        source_artifacts = bag2._source_artifact_hashes(runner)
        final_hinge_hashes = {"fit": _sha_file(HINGE_FINAL_PATH),
                              "freeze": _sha_file(HINGE_FINAL_FREEZE_PATH)}
        origins: dict[str, Any] = {}
        _operation_start(SOURCE_OPERATION_ID, "target_free_immutable_prediction_cache_reconstruction")
        for origin in (67, 75):
            arrays, info = pending[origin]
            receipt = _write_npz_immutable(ARTIFACT_ROOT / f"SOURCE_E{origin}_R1.npz", arrays)
            origins[str(origin)] = {"archive": receipt, "info": info,
                                    "query_rows": len(arrays["query_rows"]),
                                    "cycles": [int(arrays["cycles"].min()), int(arrays["cycles"].max())],
                                    "hinge_parameters": dict(hinges[origin])}
            del arrays
        source_frame = {
            "schema_version": 1, "experiment_id": CARD_ID, "status": "SOURCE_FROZEN",
            "recipe_sha256": _sha_file(RECIPE_PATH), "root_plan_sha256": _sha_file(ROOT_PLAN_PATH),
            "local_root": str(ROOT.resolve()), "local_git_head": bag2._git(ROOT, "rev-parse", "HEAD"),
            "local_git_branch": bag2._git(ROOT, "branch", "--show-current"),
            "official_train_sha256": _sha_file(runner._contained_file(bag2.SOURCE_ROOT, runner.TRAIN_REL)),
            "source_root": str(bag2.SOURCE_ROOT.resolve()),
            "position_freeze_signature": position_freeze.get("freeze_signature"),
            "source_artifacts": source_artifacts, "source_code_hashes": _source_code_hashes(bag2),
            "final_hinge_hashes": final_hinge_hashes,
            "resource_gate": resource,
            "origins": origins,
            "target_column_loaded": False, "query_targets_read": False,
            "test_or_sample_accessed": False, "model_fits_started": 0,
            "base_pos_vectorizer_calibrator_optimizer_fits": 0,
            "source_replay": "Existing original POS model and immutable R06 OOF caches only; no fit.",
        }
        freeze = _write_json_immutable(SOURCE_FREEZE_R1, source_frame, "source_signature")
    _operation_saved(SOURCE_OPERATION_ID, "target_free_immutable_prediction_cache_reconstruction", _sha_file(SOURCE_FREEZE_R1))
    return {"status": freeze["status"], "source_signature": freeze["source_signature"],
            "source_freeze_sha256": _sha_file(SOURCE_FREEZE_R1), "reused": False}


def verify_source_freeze() -> dict[str, Any]:
    if not SOURCE_FREEZE.is_file():
        raise GateError("Source freeze is absent")
    freeze = _json_read(SOURCE_FREEZE)
    body = {key: value for key, value in freeze.items() if key != "source_signature"}
    if freeze.get("source_signature") != _canonical_sha(body) or freeze.get("status") != "SOURCE_AMENDED_R2":
        raise GateError("Source freeze signature/status failed")
    original = _json_read(SOURCE_FREEZE_R1)
    original_body = {key: value for key, value in original.items() if key != "source_signature"}
    if (original.get("source_signature") != _canonical_sha(original_body)
            or _sha_file(SOURCE_FREEZE_R1) != freeze.get("source_freeze_r1_sha256")
            or original.get("source_signature") != freeze.get("source_signature_r1")):
        raise GateError("R2 source amendment does not bind the preserved R1 freeze")
    if (_sha_file(RECIPE_PATH) != freeze.get("recipe_sha256")
            or _sha_file(ROOT_PLAN_PATH) != freeze.get("root_plan_sha256")
            or _sha_file(ROOT_AMENDMENT_PATH) != freeze.get("root_amendment_sha256")):
        raise GateError("Recipe or root authorization changed after source freeze")
    bag2, runner = _helpers()
    if bag2._source_artifact_hashes(runner) != freeze.get("source_artifacts"):
        raise GateError("Frozen source artifacts changed")
    if _source_code_hashes(bag2) != freeze.get("source_code_hashes"):
        raise GateError("Frozen source or implementation code changed")
    for origin in (67, 75):
        meta = freeze["origins"][str(origin)]["archive"]
        path = ROOT / meta["path"]
        _verify_npz_receipt(path, meta)
    for name in ("D1", "D2"):
        receipt = freeze["scale_states"][name]
        path = ROOT / receipt["path"]
        state = _json_read(path)
        state_body = {key: value for key, value in state.items() if key != "state_signature"}
        if (_sha_file(path) != receipt["sha256"]
                or state.get("state_signature") != _canonical_sha(state_body)
                or state.get("state_signature") != receipt["state_signature"]):
            raise GateError(f"Reused saved scale state changed: {name}")
    return freeze


def _scale_state(data: Mapping[str, np.ndarray], name: str) -> dict[str, Any]:
    low, high = SCALE_WINDOWS[name]
    cycles = np.asarray(data["cycles"], dtype=np.int64)
    present = ~np.asarray(data["missing"], dtype=bool)
    disagreement = np.abs(np.asarray(data["pos_t50"], dtype=np.float64)
                          - np.asarray(data["baseline"], dtype=np.float64))
    mask = (cycles >= low) & (cycles <= high) & present & (disagreement > 0.0)
    values = disagreement[mask]
    if len(values) < MIN_POSITIVE_DISAGREEMENTS:
        return {"schema_version": 1, "experiment_id": CARD_ID, "state_id": name,
                "status": "UNSUPPORTED", "scale_s": None,
                "positive_selection_disagreements": int(len(values)),
                "minimum_required": MIN_POSITIVE_DISAGREEMENTS,
                "source_origin": 67, "selection_cycles": [low, high],
                "selection_rows_sha256": _sha_array(np.asarray(data["query_rows"])[mask]),
                "selection_ids_sha256": _sha_array(np.asarray(data["query_ids"])[mask]),
                "disagreement_sha256": _sha_array(values), "labels_used": False}
    return {"schema_version": 1, "experiment_id": CARD_ID, "state_id": name,
            "status": "FIT_COMPLETE", "scale_s": float(np.median(values)),
            "positive_selection_disagreements": int(len(values)),
            "minimum_required": MIN_POSITIVE_DISAGREEMENTS,
            "source_origin": 67, "selection_cycles": [low, high],
            "selection_rows_sha256": _sha_array(np.asarray(data["query_rows"])[mask]),
            "selection_ids_sha256": _sha_array(np.asarray(data["query_ids"])[mask]),
            "disagreement_sha256": _sha_array(values), "labels_used": False}


def _load_saved_scale_state(receipt: Mapping[str, Any]) -> dict[str, Any]:
    path_value = Path(str(receipt["path"]))
    path = path_value if path_value.is_absolute() else ROOT / path_value
    state = _json_read(path)
    body = {key: value for key, value in state.items() if key != "state_signature"}
    if (_sha_file(path) != receipt.get("sha256")
            or state.get("state_signature") != _canonical_sha(body)
            or state.get("state_signature") != receipt.get("state_signature")
            or state.get("labels_used") is not False):
        raise GateError(f"Saved scale-state receipt failed: {path}")
    return state


def _construct_policy(origin: int, data: Mapping[str, np.ndarray], state: Mapping[str, Any],
                      hinge_module: Any, hinge_params: Mapping[str, float]) -> dict[str, np.ndarray]:
    cycles = np.asarray(data["cycles"], dtype=np.int64)
    missing = np.asarray(data["missing"], dtype=bool)
    baseline = np.asarray(data["baseline"], dtype=np.float64)
    pos = np.asarray(data["pos_t50"], dtype=np.float64)
    disagreement = np.abs(pos - baseline)
    if state.get("status") == "FIT_COMPLETE":
        scale = float(state["scale_s"])
        if not np.isfinite(scale) or scale <= 0:
            raise GateError("Fitted scale must be finite and positive")
        real_weight = 0.25 * np.exp(-disagreement / scale)
    elif state.get("status") == "UNSUPPORTED":
        scale = float("nan")
        real_weight = np.full(len(cycles), 0.25, dtype=np.float64)
    else:
        raise GateError("Unknown scale state")
    control_weight = np.full(len(cycles), 0.25, dtype=np.float64)
    control_n = np.zeros(len(cycles), dtype=np.int64)
    for cycle in np.unique(cycles):
        cycle_mask = cycles == cycle
        present_mask = cycle_mask & ~missing
        n = int(present_mask.sum())
        control_n[cycle_mask] = n
        if n:
            control_weight[cycle_mask] = float(np.mean(real_weight[present_mask]))
    # Preserve the exact parent operation order from the original saved ensemble.
    champion_direct = np.asarray(data["champion_direct"], dtype=np.float64).copy()
    expected_champion_direct = np.clip(0.75 * baseline + 0.25 * pos, 0.0, 100.0)
    expected_champion_direct[missing] = baseline[missing]
    if not np.array_equal(expected_champion_direct, champion_direct):
        raise GateError(f"Champion direct formula does not reproduce frozen source at E{origin}")
    if state.get("status") == "UNSUPPORTED":
        # No scale means no policy change: retain the saved champion bits exactly.
        real_direct = champion_direct.copy()
        control_direct = champion_direct.copy()
        real_hinge = np.asarray(data["champion_hinge"], dtype=np.float64).copy()
        control_hinge = np.asarray(data["champion_hinge"], dtype=np.float64).copy()
    else:
        real_direct = np.clip(baseline + real_weight * (pos - baseline), 0.0, 100.0)
        control_direct = np.clip(baseline + control_weight * (pos - baseline), 0.0, 100.0)
        real_direct[missing] = baseline[missing]
        control_direct[missing] = baseline[missing]
        real_hinge = hinge_module.apply_hinge_present_only(real_direct, missing, hinge_params)
        control_hinge = hinge_module.apply_hinge_present_only(control_direct, missing, hinge_params)
    if (not np.array_equal(real_hinge[missing].view("<u8"), baseline[missing].view("<u8"))
            or not np.array_equal(control_hinge[missing].view("<u8"), baseline[missing].view("<u8"))):
        raise GateError(f"A policy changed missing-row baseline bits at E{origin}")
    return {
        "query_rows": np.asarray(data["query_rows"], dtype=np.int64),
        "query_ids": np.asarray(data["query_ids"], dtype=str),
        "cycles": cycles, "score_mask": (cycles >= SCORE_WINDOWS[origin][0]) & (cycles <= SCORE_WINDOWS[origin][1]),
        "missing": missing, "baseline": baseline, "pos_t50": pos,
        "disagreement": disagreement, "scale_s": np.asarray([scale], dtype=np.float64),
        "real_weight": real_weight, "control_weight": control_weight,
        "control_present_cohort_n": control_n,
        "champion_direct": champion_direct,
        "champion_hinge": np.asarray(data["champion_hinge"], dtype=np.float64),
        "real_direct": real_direct, "control_direct": control_direct,
        "real_hinge": np.asarray(real_hinge, dtype=np.float64),
        "control_hinge": np.asarray(control_hinge, dtype=np.float64),
        "impact_quartile": np.asarray(data["impact_quartile"], dtype=np.int8),
        "text_quartile": np.asarray(data["text_quartile"], dtype=np.int8),
        "entry_friction": np.asarray(data["entry_friction"], dtype=np.float64),
        "control_definition": np.asarray(["cycle_mean_real_weight_present_only"]),
    }


def construct_stage() -> dict[str, Any]:
    freeze = verify_source_freeze()
    bag2, _ = _helpers()
    hinge_module = bag2._hinge()
    inputs: dict[int, dict[str, np.ndarray]] = {}
    for origin in (67, 75):
        receipt = freeze["origins"][str(origin)]["archive"]
        inputs[origin] = _read_npz(ROOT / receipt["path"], tuple(receipt["members"]))
    hinge_params = {origin: freeze["origins"][str(origin)]["hinge_parameters"] for origin in (67, 75)}
    states = {name: _load_saved_scale_state(freeze["scale_states"][name]) for name in ("D1", "D2")}
    state_receipts = dict(freeze["scale_states"])
    # Root authorized six total policy starts with at most four successful artifacts.
    prior_policy_events = [event for event in _ledger()
                           if event.get("operation_type") == "deterministic_target_free_policy_construction"]
    prior_starts = [event for event in prior_policy_events if event.get("event") == "OPERATION_STARTED"]
    prior_saves = [event for event in prior_policy_events if event.get("event") == "OPERATION_SAVED"]
    prior_failures = [event for event in prior_policy_events if event.get("event") == "OPERATION_FAILED"]
    expected_prior_ids = {"REAL_D1", "CONTROL_D1"}
    if ({event.get("operation_id") for event in prior_starts} != expected_prior_ids
            or {event.get("operation_id") for event in prior_failures} != expected_prior_ids
            or len(prior_starts) != 2 or len(prior_failures) != 2 or prior_saves
            or len(prior_starts) + 4 > 6):
        raise GateError("Policy construction allowance or preserved failed-start ledger differs")
    for origin in (67, 75):
        data = inputs[origin]
        expected_parent = np.clip(0.75 * data["baseline"] + 0.25 * data["pos_t50"], 0.0, 100.0)
        expected_parent[data["missing"].astype(bool)] = data["baseline"][data["missing"].astype(bool)]
        if not np.array_equal(expected_parent, data["champion_direct"]):
            raise GateError(f"R2 parent-order source precheck failed at E{origin}")
    pred_receipts = {}
    for name, origin in (("D1", 67), ("D2", 75)):
        policy_source = inputs[origin]
        for policy in ("REAL", "CONTROL"):
            op_id = f"{policy}_{name}_R2" if name == "D1" else f"{policy}_{name}"
            _operation_start(op_id, "deterministic_target_free_policy_construction")
        combined = _construct_policy(origin, policy_source, states[name], hinge_module, hinge_params[origin])
        # The two policy constructions are separately charged even though they share the same immutable arrays.
        for policy in ("REAL", "CONTROL"):
            op_id = f"{policy}_{name}_R2" if name == "D1" else f"{policy}_{name}"
            path = ARTIFACT_ROOT / f"PREDICTION_{policy}_{name}_R2.npz"
            member_subset = {key: value for key, value in combined.items()}
            if policy == "REAL":
                member_subset["selected_direct"] = combined["real_direct"]
                member_subset["selected_hinge"] = combined["real_hinge"]
                member_subset["selected_weight"] = combined["real_weight"]
            else:
                member_subset["selected_direct"] = combined["control_direct"]
                member_subset["selected_hinge"] = combined["control_hinge"]
                member_subset["selected_weight"] = combined["control_weight"]
            receipt = _write_npz_immutable(path, member_subset)
            receipt["operation_id"] = op_id
            pred_receipts[f"{policy}_{name}"] = receipt
            _operation_saved(op_id, "deterministic_target_free_policy_construction", receipt["sha256"])
    body = {
        "schema_version": 1, "experiment_id": CARD_ID, "status": "PREDICTIONS_FROZEN",
        "source_signature": freeze["source_signature"], "recipe_sha256": freeze["recipe_sha256"],
        "states": state_receipts, "predictions": pred_receipts,
        "windows": {str(origin): list(SCORE_WINDOWS[origin]) for origin in (67, 75)},
        "pooled_expected_rows": EXPECTED_POOLED_ROWS,
        "policy_contract": "REAL and CONTROL are the only policy constructions; CHAMPION is the frozen comparator in each archive.",
        "target_column_loaded": False, "query_targets_read": False,
        "test_or_sample_accessed": False, "model_fits_started": 0,
        "scale_state_fits_started": 0, "saved_scale_states_reused": 2,
        "deterministic_policy_constructions": 4,
        "policy_start_allowance_total": 6, "successful_policy_artifact_cap": 4,
    }
    prediction_freeze = _write_json_immutable(PREDICTION_FREEZE, body, "prediction_signature")
    # Verify the distinct primary rows before the independent prelabel pass.
    d1 = _read_npz(ARTIFACT_ROOT / "PREDICTION_REAL_D1_R2.npz", tuple(pred_receipts["REAL_D1"]["members"]))
    d2 = _read_npz(ARTIFACT_ROOT / "PREDICTION_REAL_D2_R2.npz", tuple(pred_receipts["REAL_D2"]["members"]))
    rows1 = d1["query_rows"][d1["score_mask"]]
    rows2 = d2["query_rows"][d2["score_mask"]]
    if len(rows1) + len(rows2) != EXPECTED_POOLED_ROWS or set(rows1.tolist()) & set(rows2.tolist()):
        raise GateError("Unique D1 plus D2 row count or row disjointness failed")
    return {"status": prediction_freeze["status"], "prediction_signature": prediction_freeze["prediction_signature"],
            "prediction_freeze_sha256": _sha_file(PREDICTION_FREEZE),
            "scale_states": {name: states[name]["status"] for name in states},
            "policy_constructions": list(pred_receipts), "primary_rows": len(rows1) + len(rows2)}


def _metric(y: np.ndarray, pred: np.ndarray) -> dict[str, Any]:
    errors = np.asarray(pred, dtype=np.float64) - np.asarray(y, dtype=np.float64)
    return {"rows": int(len(errors)), "mae": float(np.mean(np.abs(errors))),
            "p90_absolute_error": float(np.quantile(np.abs(errors), 0.90)),
            "bias_prediction_minus_target": float(np.mean(errors))}


def _score_groups(y: np.ndarray, arrays: Mapping[str, np.ndarray], mask: np.ndarray) -> dict[str, np.ndarray]:
    missing = np.asarray(arrays["missing"], dtype=bool)[mask]
    groups: dict[str, np.ndarray] = {
        "missing": missing, "present": ~missing,
        "target_lt5": y < 5.0, "target_ge95": y >= 95.0,
        "EF0": np.asarray(arrays["entry_friction"], dtype=np.float64)[mask] == 0.0,
        "EF1": np.asarray(arrays["entry_friction"], dtype=np.float64)[mask] > 0.0,
    }
    for feature, key in (("impact_load", "impact_quartile"), ("text_load", "text_quartile")):
        buckets = np.asarray(arrays[key], dtype=np.int8)[mask]
        for bucket in range(4):
            groups[f"{feature}_Q{bucket + 1}"] = buckets == bucket
    return groups


def _require_prediction_freeze() -> tuple[dict[str, Any], dict[str, Any], Any, Any]:
    source = verify_source_freeze()
    prediction = _json_read(PREDICTION_FREEZE)
    body = {key: value for key, value in prediction.items() if key != "prediction_signature"}
    if (prediction.get("prediction_signature") != _canonical_sha(body)
            or prediction.get("status") != "PREDICTIONS_FROZEN"
            or prediction.get("source_signature") != source.get("source_signature")):
        raise GateError("Prediction freeze signature or source binding failed")
    for name, receipt in prediction.get("predictions", {}).items():
        path = ROOT / receipt["path"]
        _verify_npz_receipt(path, receipt)
    bag2, runner = _helpers()
    return source, prediction, bag2, runner


def score_stage() -> dict[str, Any]:
    source, prediction, bag2, runner = _require_prediction_freeze()
    prelabel = _json_read(PRELABEL_QA_PATH)
    prelabel_body = {key: value for key, value in prelabel.items() if key != "qa_signature"}
    if (prelabel.get("qa_signature") != _canonical_sha(prelabel_body)
            or prelabel.get("status") != "PASS"
            or prelabel.get("source_signature") != source.get("source_signature")
            or prelabel.get("prediction_signature") != prediction.get("prediction_signature")
            or prelabel.get("query_targets_read") is not False):
        raise GateError("Official Train score blocked until independent prelabel QA passes")
    if _sha_file(runner._contained_file(bag2.SOURCE_ROOT, runner.TRAIN_REL)) != source["official_train_sha256"]:
        raise GateError("Official Train input changed after source freeze")
    # Load and validate both prediction sets before reading any query target cell.
    prepared = {}
    for name, origin in (("D1", 67), ("D2", 75)):
        receipt = prediction["predictions"][f"REAL_{name}"]
        path = ROOT / receipt["path"]
        arrays = _read_npz(path, tuple(receipt["members"]))
        if not np.array_equal(arrays["score_mask"],
                              (arrays["cycles"] >= SCORE_WINDOWS[origin][0]) & (arrays["cycles"] <= SCORE_WINDOWS[origin][1])):
            raise GateError(f"Frozen score mask differs from the authorized window: {name}")
        prepared[name] = arrays
    results = {}
    targets = {}
    candidate_parts = {"REAL": [], "CONTROL": []}
    champion_parts = []
    control_parts = []
    group_harms: list[float] = []
    fixed_group_names = {"missing", "target_lt5", "target_ge95", "EF0", "EF1"}
    fixed_groups_known = True
    all_rows = []
    for name, origin in (("D1", 67), ("D2", 75)):
        arrays = prepared[name]
        score_mask = arrays["score_mask"].astype(bool)
        rows = np.asarray(arrays["query_rows"][score_mask], dtype=np.int64)
        ids = np.asarray(arrays["query_ids"][score_mask], dtype=str)
        if not np.array_equal(rows, np.sort(rows)) or len(set(rows.tolist())) != len(rows):
            raise GateError(f"Score rows are not unique increasing source rows: {name}")
        # This call parses only the selected official Train target cells, after the full prelabel QA pass.
        y = runner._read_targets_for_rows(runner._contained_file(bag2.SOURCE_ROOT, runner.TRAIN_REL),
                                          rows, ids, expected_total_rows=50608)
        if not np.isfinite(y).all():
            raise GateError(f"Selected Train target vector is nonfinite: {name}")
        targets[name] = y
        all_rows.extend(rows.tolist())
        local = {key: np.asarray(value)[score_mask] for key, value in arrays.items()
                 if np.asarray(value).ndim > 0 and len(np.asarray(value)) == len(score_mask)}
        policies = {
            "REAL": (local["real_direct"], local["real_hinge"]),
            "CONTROL": (local["control_direct"], local["control_hinge"]),
        }
        champion_direct = local["champion_direct"]
        champion_hinge = local["champion_hinge"]
        groups = _score_groups(y, local, np.ones(len(y), dtype=bool))
        group_metrics = {}
        for policy, (direct, hinged) in policies.items():
            for group_name, group_mask in groups.items():
                n = int(group_mask.sum())
                if n < 100:
                    item = {"status": "UNKNOWN", "rows": n, "harm": None}
                    if policy == "REAL" and group_name in fixed_group_names:
                        fixed_groups_known = False
                else:
                    harm = _metric(y[group_mask], hinged[group_mask])["mae"] - _metric(y[group_mask], champion_hinge[group_mask])["mae"]
                    item = {"status": "MEASURED", "rows": n,
                            "policy_mae": _metric(y[group_mask], hinged[group_mask])["mae"],
                            "champion_mae": _metric(y[group_mask], champion_hinge[group_mask])["mae"],
                            "harm": float(harm)}
                    if policy == "REAL":
                        group_harms.append(float(harm))
                group_metrics.setdefault(policy, {})[group_name] = item
        cycle_metrics = {}
        for cycle in np.unique(local["cycles"]):
            cmask = local["cycles"] == cycle
            cycle_metrics[str(int(cycle))] = {
                "rows": int(cmask.sum()),
                "champion": _metric(y[cmask], champion_hinge[cmask]),
                "REAL": _metric(y[cmask], policies["REAL"][1][cmask]),
                "CONTROL": _metric(y[cmask], policies["CONTROL"][1][cmask]),
            }
        results[name] = {
            "origin": origin, "cycles": list(SCORE_WINDOWS[origin]), "rows": len(y),
            "row_identity_sha256": _canonical_sha({"rows": _sha_array(rows), "ids": _sha_array(ids),
                                                   "cycles": _sha_array(local["cycles"])}),
            "champion_direct": _metric(y, champion_direct), "champion_hinge": _metric(y, champion_hinge),
            "REAL_direct": _metric(y, policies["REAL"][0]), "REAL_hinge": _metric(y, policies["REAL"][1]),
            "CONTROL_direct": _metric(y, policies["CONTROL"][0]), "CONTROL_hinge": _metric(y, policies["CONTROL"][1]),
            "REAL_MAE_gain": _metric(y, champion_hinge)["mae"] - _metric(y, policies["REAL"][1])["mae"],
            "CONTROL_MAE_gain": _metric(y, champion_hinge)["mae"] - _metric(y, policies["CONTROL"][1])["mae"],
            "REAL_over_CONTROL_gain": _metric(y, policies["CONTROL"][1])["mae"] - _metric(y, policies["REAL"][1])["mae"],
            "per_cycle": cycle_metrics, "groups": group_metrics,
            "query_target_rows_read": int(len(y)), "test_or_sample_accessed": False,
        }
        candidate_parts["REAL"].append(policies["REAL"][1])
        candidate_parts["CONTROL"].append(policies["CONTROL"][1])
        champion_parts.append(champion_hinge)
        control_parts.append(policies["CONTROL"][1])
    if len(all_rows) != EXPECTED_POOLED_ROWS or len(set(all_rows)) != len(all_rows):
        raise GateError("Pooled score rows do not match unique 15,953 row plan")
    pooled_y = np.concatenate([targets[name] for name in ("D1", "D2")])
    pooled_champion = np.concatenate(champion_parts)
    pooled_real = np.concatenate(candidate_parts["REAL"])
    pooled_control = np.concatenate(control_parts)
    pooled_gain = _metric(pooled_y, pooled_champion)["mae"] - _metric(pooled_y, pooled_real)["mae"]
    d1_gain, d2_gain = results["D1"]["REAL_MAE_gain"], results["D2"]["REAL_MAE_gain"]
    real_over_control = _metric(pooled_y, pooled_control)["mae"] - _metric(pooled_y, pooled_real)["mae"]
    max_harm = max(group_harms, default=float("inf"))
    missing_exact = all(
        np.array_equal(prepared[name]["real_hinge"][prepared[name]["missing"]].view("<u8"),
                       prepared[name]["baseline"][prepared[name]["missing"]].view("<u8"))
        and np.array_equal(prepared[name]["control_hinge"][prepared[name]["missing"]].view("<u8"),
                           prepared[name]["baseline"][prepared[name]["missing"]].view("<u8"))
        for name in ("D1", "D2"))
    harm_pass = fixed_groups_known and max_harm <= 0.25
    qa_pass = bool(prelabel.get("status") == "PASS" and missing_exact and harm_pass)
    scientific_performance_gate = bool(pooled_gain >= 0.02 and d1_gain >= 0 and d2_gain >= 0
                                       and real_over_control > 0 and harm_pass and missing_exact)
    exploratory_performance_gate = bool((pooled_gain > 0 or (d2_gain >= 0.01 and pooled_gain >= -0.01))
                                        and d1_gain >= -0.05 and harm_pass and missing_exact)
    report_body = {
        "schema_version": 1, "experiment_id": CARD_ID, "status": "SCORED",
        "source_signature": source["source_signature"],
        "prediction_signature": prediction["prediction_signature"],
        "prelabel_qa_sha256": _sha_file(PRELABEL_QA_PATH),
        "pooled_rows": len(all_rows), "pooled_row_identity_sha256": _sha_array(np.asarray(all_rows, dtype=np.int64)),
        "pooled_champion_hinge": _metric(pooled_y, pooled_champion),
        "pooled_REAL_hinge": _metric(pooled_y, pooled_real),
        "pooled_CONTROL_hinge": _metric(pooled_y, pooled_control),
        "pooled_REAL_MAE_gain": pooled_gain,
        "pooled_CONTROL_MAE_gain": _metric(pooled_y, pooled_champion)["mae"] - _metric(pooled_y, pooled_control)["mae"],
        "pooled_REAL_over_CONTROL_gain": real_over_control,
        "windows": results,
        "maximum_supported_REAL_group_harm": max_harm if np.isfinite(max_harm) else None,
        "fixed_groups_known": fixed_groups_known, "missing_invariant_pass": bool(missing_exact),
        "independent_prelabel_qa_pass": prelabel.get("status") == "PASS",
        "independent_metric_qa_pass": False,
        "scientific_performance_gate": scientific_performance_gate,
        "exploratory_performance_gate": exploratory_performance_gate,
        "scientific_qualification": None,
        "exploratory_benchmark_eligible": None,
        "evaluation_disclosure": "Repeatedly reused Train development labels; D1 and D2 have distinct row IDs but are not independent validation because E75 training includes earlier D1 score cycles.",
        "selection_disclosure": "No labels or public score selected the scale or policy. CONTROL is diagnostic only and cannot be promoted after results.",
        "query_targets_read": True, "query_target_rows_read": EXPECTED_POOLED_ROWS,
        "target_values_persisted": False, "test_or_sample_accessed": False, "csv_created": False,
        "fit_counts": {"base": 0, "POS": 0, "vectorizer": 0, "calibrator": 0, "optimizer": 0,
                       "scalar_scale_state": 2, "deterministic_policy_constructions": 4},
    }
    report = _write_json_immutable(SCORE_PATH, report_body, "score_signature")
    return {"status": report["status"], "score_signature": report["score_signature"],
            "score_sha256": _sha_file(SCORE_PATH), "pooled_REAL_MAE_gain": pooled_gain,
            "pooled_REAL_over_CONTROL_gain": real_over_control,
            "D1_gain": d1_gain, "D2_gain": d2_gain,
            "scientific_performance_gate": scientific_performance_gate,
            "exploratory_performance_gate": exploratory_performance_gate,
            "qualification_pending_independent_metric_qa": True}


def prepare_final_source() -> dict[str, Any]:
    score = _json_read(SCORE_PATH)
    score_body = {key: value for key, value in score.items() if key != "score_signature"}
    if score.get("score_signature") != _canonical_sha(score_body) or score.get("status") != "SCORED":
        raise GateError("Scored report signature failed")
    metric_qa = _json_read(METRIC_QA_PATH)
    qa_body = {key: value for key, value in metric_qa.items() if key != "qa_signature"}
    if (metric_qa.get("qa_signature") != _canonical_sha(qa_body)
            or metric_qa.get("status") != "PASS"
            or metric_qa.get("score_signature") != score.get("score_signature")):
        raise GateError("Final source preparation requires independent metric QA PASS")
    if not metric_qa.get("exploratory_benchmark_eligible"):
        return {"status": "NOT_PREPARED", "reason": "Pilot is not exploratory eligible"}
    freeze = verify_source_freeze()
    e67 = _read_npz(ARTIFACT_ROOT / freeze["origins"]["67"]["archive"]["path"],
                    tuple(freeze["origins"]["67"]["archive"]["members"]))
    e75 = _read_npz(ARTIFACT_ROOT / freeze["origins"]["75"]["archive"]["path"],
                    tuple(freeze["origins"]["75"]["archive"]["members"]))
    keep67 = (e67["cycles"] >= 68) & (e67["cycles"] <= 75)
    keep75 = (e75["cycles"] >= 76) & (e75["cycles"] <= 79)
    arrays = {key: np.concatenate((e67[key][keep67], e75[key][keep75]))
              for key in ("query_rows", "query_ids", "cycles", "missing", "baseline", "pos_t50")}
    arrays["disagreement"] = np.abs(arrays["pos_t50"] - arrays["baseline"])
    order = np.argsort(arrays["query_rows"], kind="stable")
    arrays = {key: value[order] for key, value in arrays.items()}
    if len(arrays["query_rows"]) != 15953 or len(set(arrays["query_rows"].tolist())) != 15953:
        raise GateError("Final source is not the unique target-free E67/E75 OOF population")
    receipt = _write_npz_immutable(FINAL_SOURCE_PATH, arrays)
    body = {"schema_version": 1, "experiment_id": CARD_ID, "status": "FINAL_SOURCE_READY_NOT_FIT",
            "source_signature": freeze["source_signature"], "score_signature": score["score_signature"],
            "metric_qa_signature": metric_qa["qa_signature"], "archive": receipt,
            "rows": len(arrays["query_rows"]), "cycles": [68, 79],
            "row_identity_sha256": _canonical_sha({"rows": _sha_array(arrays["query_rows"]),
                                                    "ids": _sha_array(arrays["query_ids"]),
                                                    "cycles": _sha_array(arrays["cycles"])}),
            "scale_fit_performed": False,
            "final_hinge_path": str(HINGE_FINAL_PATH.relative_to(ROOT)),
            "final_hinge_sha256": freeze["final_hinge_hashes"]["fit"],
            "E79_inference_performed": False, "test_or_sample_accessed": False,
            "kaggle_or_submission_action": False}
    manifest = _write_json_immutable(FINAL_SOURCE_MANIFEST, body, "source_manifest_signature")
    return {"status": manifest["status"], "source_manifest_signature": manifest["source_manifest_signature"],
            "archive_sha256": receipt["sha256"], "scale_fit_performed": False}


def synthetic_e2e() -> dict[str, Any]:
    """Exercise state, policy, NPZ receipt/readback and hinge application on synthetic rows only."""
    rng = np.random.default_rng(20261106)
    cycles = np.repeat(np.arange(68, 80, dtype=np.int64), 128)
    rows = np.arange(len(cycles), dtype=np.int64)
    missing = (rows % 13 == 0)
    baseline = np.clip(42.0 + (cycles - 68) * 2.0 + rng.normal(0, 4, len(rows)), 0, 100)
    pos = np.clip(baseline + rng.normal(0, 8, len(rows)), 0, 100)
    pos[rows % 17 == 0] = baseline[rows % 17 == 0]
    champion = np.clip(0.75 * baseline + 0.25 * pos, 0, 100)
    champion[missing] = baseline[missing]
    dummy_hinge = {"intercept": 0.0, "slope_low": 1.0, "slope_high": 1.0}
    class IdentityHinge:
        @staticmethod
        def apply_hinge_present_only(values, missing_mask, parameters):
            result = np.asarray(values, dtype=np.float64).copy()
            result[np.asarray(missing_mask, dtype=bool)] = baseline[np.asarray(missing_mask, dtype=bool)]
            return result
    data = {"cycles": cycles, "query_rows": rows, "query_ids": np.asarray([f"SYN-{v:04d}" for v in rows]),
            "missing": missing, "baseline": baseline, "pos_t50": pos,
            "champion_direct": champion, "champion_hinge": champion.copy(),
            "impact_quartile": (rows % 4).astype(np.int8), "text_quartile": ((rows // 4) % 4).astype(np.int8),
            "entry_friction": (rows % 2).astype(np.float64)}
    outputs = {}
    with tempfile.TemporaryDirectory(prefix="reg-pos-disagreement-synthetic-") as temp:
        temp_root = Path(temp)
        for name, origin in (("D1", 67), ("D2", 75)):
            state = _scale_state(data, name)
            output = _construct_policy(origin, data, state, IdentityHinge, dummy_hinge)
            path = temp_root / f"PREDICTION_REAL_{name}.npz"
            receipt = _write_npz_immutable(path, output)
            readback = _read_npz(path, tuple(receipt["members"]))
            if not _arrays_equal(output, readback):
                raise GateError(f"Synthetic receipt/readback failed for {name}")
            outputs[name] = {"receipt": receipt, "scale_state": state}
        if any(outputs[name]["scale_state"].get("status") != "FIT_COMPLETE"
               or not np.isfinite(outputs[name]["scale_state"].get("scale_s"))
               or outputs[name]["scale_state"]["scale_s"] <= 0 for name in outputs):
            raise GateError("Synthetic scale state is not finite")
        # Verify cycle-wise matched shrink exposure and exact missing fallback.
        for name in outputs:
            arr = _read_npz(temp_root / f"PREDICTION_REAL_{name}.npz", tuple(outputs[name]["receipt"]["members"]))
            if not np.array_equal(arr["real_direct"][missing].view("<u8"), baseline[missing].view("<u8")):
                raise GateError("Synthetic missing fallback was not exact")
            for cycle in np.unique(cycles):
                mask = (cycles == cycle) & ~missing
                if mask.any() and not np.all(arr["control_weight"][cycles == cycle] == np.mean(arr["real_weight"][mask])):
                    raise GateError("Synthetic same-cycle exposure match failed")
        report_body = {"schema_version": 1, "experiment_id": CARD_ID, "status": "PASS",
                       "synthetic_only": True, "deterministic_seed": 20261106,
                       "synthetic_rows": len(rows), "real_files_read": False,
                       "official_targets_read": False, "production_ledger_touched": False,
                       "test_or_sample_accessed": False,
                       "checks": {"scale_states": 2, "policy_applications": 2,
                                  "npz_receipt_and_readback": True, "hinge_application": True,
                                  "cycle_matched_control": True, "missing_exact": True},
                       "temporary_outputs_removed": True,
                       "artifacts": {key: value["receipt"] for key, value in outputs.items()}}
    report = _write_json_immutable(REPORT_ROOT / "SYNTHETIC_QA_R2.json", report_body, "qa_signature")
    return {"status": report["status"], "qa_signature": report["qa_signature"],
            "synthetic_rows": report["synthetic_rows"], "real_files_read": False}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--synthetic-e2e", action="store_true")
    mode.add_argument("--source", action="store_true")
    mode.add_argument("--construct", action="store_true")
    mode.add_argument("--score", action="store_true")
    mode.add_argument("--prepare-final-source", action="store_true")
    args = parser.parse_args(argv)
    try:
        if args.synthetic_e2e:
            result = synthetic_e2e()
        elif args.source:
            result = source_stage()
        elif args.construct:
            result = construct_stage()
        elif args.score:
            result = score_stage()
        else:
            result = prepare_final_source()
    except Exception as exc:
        print(json.dumps({"status": "BLOCKED", "error": f"{type(exc).__name__}: {exc}"}, sort_keys=True))
        return 1
    print(json.dumps(_jsonable(result), sort_keys=True, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
