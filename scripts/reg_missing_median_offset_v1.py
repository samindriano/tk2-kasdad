"""Bounded missing-only median-residual pilot for the frozen REG full champion."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
from typing import Any, Mapping
import zipfile

import numpy as np
import psutil
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
LANE = "REG_MISSING_MEDIAN_OFFSET_V1"
REPORT_ROOT = ROOT / "reports" / "optimization_v3" / LANE
ARTIFACT_ROOT = ROOT / "experiments" / "runs" / LANE
SOURCE_ARCHIVE = ARTIFACT_ROOT / "SOURCE_INPUTS_R1.npz"
SOURCE_REPORT = REPORT_ROOT / "SOURCE_FREEZE_R1.json"
PRELABEL_QA = REPORT_ROOT / "PRELABEL_QA_R1.json"
SCORE_REPORT = REPORT_ROOT / "SCORE_R1.json"
METRIC_QA = REPORT_ROOT / "METRIC_QA_R1.json"
FINAL_SOURCE_ARCHIVE = ARTIFACT_ROOT / "FINAL_SOURCE_INPUTS_R1.npz"
FINAL_SOURCE_REPORT = REPORT_ROOT / "FINAL_SOURCE_R1.json"
RECIPE_PATH = REPORT_ROOT / "RECIPE_R1.json"
ROOT_PLAN_PATH = REPORT_ROOT / "ROOT_PLAN.md"
PILOT_ROOT = ROOT / "experiments" / "runs" / "REG_POS_DISAGREEMENT_GATE_V1"
PILOT_REPORT = ROOT / "reports" / "optimization_v3" / "REG_POS_DISAGREEMENT_GATE_V1"
PILOT_FREEZE_R1 = PILOT_ROOT / "SOURCE_FREEZE_R1.json"
PILOT_FREEZE_R2 = PILOT_ROOT / "SOURCE_FREEZE_R2.json"
PILOT_AMENDMENT_R3 = PILOT_REPORT / "METRIC_QA_VERIFIER_AMENDMENT_R3.json"
PILOT_GATE = ROOT / "scripts" / "reg_pos_disagreement_gate_v1.py"
PILOT_VERIFIER = ROOT / "scripts" / "verify_reg_pos_disagreement_gate_v1.py"
GATE_TEST = ROOT / "tests" / "test_reg_pos_disagreement_gate_v1.py"
BAG2_HELPER = ROOT / "scripts" / "reg_pos_poisson_bag2_v1.py"
HINGE_RUNNER = ROOT / "scripts" / "reg_monotone_hinge_p50_v1.py"
R18_FREEZE = ROOT / "experiments" / "optimization_v3" / "batch1" / "R18" / "freeze.json"
R18_SUMMARY = ROOT / "reports" / "optimization_v3" / "batch1" / "R18" / "summary.md"
R19_RECIPE = ROOT / "experiments" / "optimization_v3" / "batch1" / "R19" / "revision_02" / "recipe.json"
R19_SUMMARY = ROOT / "reports" / "optimization_v3" / "batch1" / "R19" / "summary.md"
MIN_SUPPORT = 100
EXPECTED_POOLED_ROWS = 15953
MEMORY_START = 512 * 1024**2
MEMORY_RESERVE = 256 * 1024**2
SCORE_WINDOWS = {"D1": (72, 75), "D2": (76, 79)}
FIT_WINDOWS = {"D1": (68, 71), "D2": (68, 75)}
OPERATION_IDS = ("FIT_D1_R1", "FIT_D2_R1", "APPLY_D1_R1", "APPLY_D2_R1", "SCORE_R1")


class PilotError(RuntimeError):
    pass


def _sha_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def _canonical(value: Any) -> bytes:
    return json.dumps(_jsonable(value), sort_keys=True, ensure_ascii=False, allow_nan=False,
                      separators=(",", ":")).encode("utf-8", errors="surrogatepass")


def _canonical_sha(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _sha_array(value: Any) -> str:
    a = np.asarray(value)
    head = _canonical({"dtype": a.dtype.str, "shape": list(a.shape)})
    if a.dtype.kind in "OUS":
        body = bytearray()
        for item in a.reshape(-1).tolist():
            b = str(item).encode("utf-8", errors="surrogatepass")
            body.extend(len(b).to_bytes(8, "little")); body.extend(b)
        payload = bytes(body)
    else:
        payload = np.ascontiguousarray(a).tobytes()
    return hashlib.sha256(len(head).to_bytes(8, "little") + head + payload).hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise PilotError(f"Invalid JSON: {path}") from exc
    if not isinstance(value, dict):
        raise PilotError(f"Expected JSON object: {path}")
    return value


def _signed(body: Mapping[str, Any], key: str) -> dict[str, Any]:
    result = dict(body)
    result[key] = _canonical_sha(result)
    return result


def _verify_signed(value: Mapping[str, Any], key: str, label: str) -> None:
    body = {k: v for k, v in value.items() if k != key}
    if value.get(key) != _canonical_sha(body):
        raise PilotError(f"Signature mismatch: {label}")


def _write_json_new(path: Path, body: Mapping[str, Any], key: str) -> dict[str, Any]:
    payload = _canonical(_signed(body, key)) + b"\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != payload:
            raise PilotError(f"Refusing to overwrite changed artifact: {path}")
        return json.loads(payload)
    fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    with os.fdopen(fd, "wb") as stream:
        stream.write(payload); stream.flush(); os.fsync(stream.fileno())
    return json.loads(payload)


def _read_npz_exact(path: Path, keys: tuple[str, ...]) -> dict[str, np.ndarray]:
    if not path.is_file():
        raise PilotError(f"NPZ missing: {path}")
    with zipfile.ZipFile(path) as zf:
        members = {name[:-4] for name in zf.namelist() if name.endswith(".npy")}
    if members != set(keys):
        raise PilotError(f"NPZ allowlist mismatch at {path}: {sorted(members)}")
    with np.load(path, allow_pickle=False) as zf:
        return {key: np.asarray(zf[key]).copy() for key in keys}


def _write_npz_new(path: Path, arrays: Mapping[str, Any]) -> dict[str, Any]:
    values = {k: np.asarray(v) for k, v in arrays.items()}
    def same(left: np.ndarray, right: np.ndarray) -> bool:
        if left.dtype.kind in "fc" or right.dtype.kind in "fc":
            return bool(np.array_equal(left, right, equal_nan=True))
        return bool(np.array_equal(left, right))
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        current = _read_npz_exact(path, tuple(values))
        if any(not same(values[k], current[k]) for k in values):
            raise PilotError(f"Refusing to overwrite changed NPZ: {path}")
    else:
        fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".npz", dir=path.parent)
        os.close(fd)
        temp = Path(temp_name)
        try:
            with temp.open("wb") as stream:
                np.savez_compressed(stream, **values); stream.flush(); os.fsync(stream.fileno())
            os.replace(temp, path)
        finally:
            temp.unlink(missing_ok=True)
        current = _read_npz_exact(path, tuple(values))
        if any(not same(values[k], current[k]) for k in values):
            raise PilotError(f"NPZ round-trip mismatch: {path}")
    try:
        rel = path.relative_to(ROOT).as_posix()
    except ValueError:
        rel = path.name
    return {"path": rel, "sha256": _sha_file(path),
            "members": {k: _sha_array(current[k]) for k in sorted(current)}}


def _verify_npz(path: Path, receipt: Mapping[str, Any]) -> dict[str, np.ndarray]:
    if _sha_file(path) != receipt.get("sha256"):
        raise PilotError(f"NPZ hash differs from receipt: {path}")
    arrays = _read_npz_exact(path, tuple(receipt.get("members", {})))
    if {k: _sha_array(v) for k, v in arrays.items()} != receipt.get("members"):
        raise PilotError(f"NPZ member hash differs from receipt: {path}")
    return arrays


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise PilotError(f"Cannot import module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _gate_module():
    return _load_module(PILOT_GATE, "reg_pos_gate_for_missing_offset")


def _parent_source() -> tuple[dict[str, Any], dict[str, Any], Any, Any, dict[int, dict[str, np.ndarray]]]:
    gate = _gate_module()
    r1, r2 = gate._json_read(PILOT_FREEZE_R1), gate._json_read(PILOT_FREEZE_R2)
    for freeze, key, label in ((r1, "source_signature", "pilot source R1"),
                               (r2, "source_signature", "pilot source R2")):
        body = {k: v for k, v in freeze.items() if k != key}
        if freeze.get(key) != gate._canonical_sha(body):
            raise PilotError(f"{label} signature failed")
    if (r1.get("status") != "SOURCE_FROZEN" or r2.get("status") != "SOURCE_AMENDED_R2"
            or gate._sha_file(PILOT_FREEZE_R1) != r2.get("source_freeze_r1_sha256")
            or r1.get("source_signature") != r2.get("source_signature_r1")):
        raise PilotError("Pilot R1/R2 source freeze lineage failed")
    if (gate._sha_file(gate.RECIPE_PATH) != r2.get("recipe_sha256")
            or gate._sha_file(gate.ROOT_PLAN_PATH) != r2.get("root_plan_sha256")
            or gate._sha_file(gate.ROOT_AMENDMENT_PATH) != r2.get("root_amendment_sha256")):
        raise PilotError("Pilot source recipe/plan amendment changed")
    amendment = gate._json_read(PILOT_AMENDMENT_R3)
    _verify_signed(amendment, "amendment_signature", "pilot verifier R3 amendment")
    if (amendment.get("status") != "VERIFIER_AMENDED_R3"
            or amendment.get("source_freeze_sha256") != gate._sha_file(PILOT_FREEZE_R2)
            or amendment.get("source_signature") != r2.get("source_signature")
            or amendment.get("old_verifier_sha256") != r2.get("source_code_hashes", {}).get("independent_verifier")
            or amendment.get("new_verifier_sha256") != gate._sha_file(gate.VERIFIER_PATH)):
        raise PilotError("Pilot verifier amendment R3 does not bind frozen R2 source")
    bag2, runner = gate._helpers()
    if bag2._source_artifact_hashes(runner) != r2.get("source_artifacts"):
        raise PilotError("Pilot frozen source-artifact inventory changed")
    current_code = gate._source_code_hashes(bag2)
    for name, value in current_code.items():
        if name == "independent_verifier":
            if value != amendment.get("new_verifier_sha256"):
                raise PilotError("Pilot verifier does not match authorized R3 amendment")
        elif r2.get("source_code_hashes", {}).get(name) != value:
            raise PilotError(f"Pilot source code changed after freeze: {name}")
    sources: dict[int, dict[str, np.ndarray]] = {}
    for origin in (67, 75):
        receipt = r2["origins"][str(origin)]["archive"]
        path = ROOT / receipt["path"]
        sources[origin] = gate._verify_npz_receipt(path, receipt)
    return r2, amendment, bag2, runner, sources


def _source_code_hashes() -> dict[str, str]:
    paths = {
        "runner": Path(__file__), "independent_verifier": ROOT / "scripts" / "verify_reg_missing_median_offset_v1.py",
        "regression_test": ROOT / "tests" / "test_reg_missing_median_offset_v1.py",
        "recipe": RECIPE_PATH, "root_plan": ROOT_PLAN_PATH,
        "pilot_gate_runner": PILOT_GATE, "pilot_gate_verifier": PILOT_VERIFIER,
        "bag2_helper": BAG2_HELPER, "hinge_runner": HINGE_RUNNER,
        "r18_freeze": R18_FREEZE, "r18_summary": R18_SUMMARY,
        "r19_recipe": R19_RECIPE, "r19_summary": R19_SUMMARY,
    }
    missing = [name for name, path in paths.items() if not path.is_file()]
    if missing:
        raise PilotError(f"Missing frozen source/code inputs: {missing}")
    return {name: _sha_file(path) for name, path in paths.items()}


def _source_group(arrays: Mapping[str, np.ndarray], mask: np.ndarray) -> dict[str, np.ndarray]:
    out = {
        "missing": np.asarray(arrays["missing"], dtype=bool)[mask],
        "present": ~np.asarray(arrays["missing"], dtype=bool)[mask],
        "entry_friction_EF0": np.asarray(arrays["entry_friction"], dtype=np.float64)[mask] == 0.0,
        "entry_friction_EF1": np.asarray(arrays["entry_friction"], dtype=np.float64)[mask] > 0.0,
    }
    for key, label in (("impact_quartile", "impact_load"), ("text_quartile", "text_load")):
        values = np.asarray(arrays[key], dtype=np.int8)[mask]
        for q in range(4):
            out[f"{label}_Q{q+1}"] = values == q
    return out


def _select(arrays: Mapping[str, np.ndarray], prefix: str, mask: np.ndarray, *, score: bool) -> dict[str, np.ndarray]:
    keys = ["query_rows", "query_ids", "cycles", "missing", "baseline"]
    if score:
        keys += ["champion_hinge", "entry_friction", "impact_quartile", "text_quartile"]
    result = {f"{prefix}_{key}": np.asarray(arrays[key])[mask].copy() for key in keys}
    return result


def _describe(arrays: Mapping[str, np.ndarray], prefix: str) -> dict[str, Any]:
    rows = np.asarray(arrays[f"{prefix}_query_rows"], dtype=np.int64)
    ids = np.asarray(arrays[f"{prefix}_query_ids"], dtype=str)
    cycles = np.asarray(arrays[f"{prefix}_cycles"], dtype=np.int64)
    missing = np.asarray(arrays[f"{prefix}_missing"], dtype=bool)
    if len(np.unique(rows)) != len(rows) or len(np.unique(ids)) != len(ids):
        raise PilotError(f"Duplicate row or ID in {prefix}")
    if not np.array_equal(rows, np.sort(rows)):
        raise PilotError(f"Physical rows are not sorted in {prefix}")
    return {"rows": len(rows), "missing_rows": int(missing.sum()),
            "cycles": [int(cycles.min()), int(cycles.max())],
            "identity_sha256": _canonical_sha({"rows": _sha_array(rows), "ids": _sha_array(ids),
                                                "cycles": _sha_array(cycles)}),
            "rows_sha256": _sha_array(rows), "ids_sha256": _sha_array(ids),
            "missing_sha256": _sha_array(missing)}


def _source_stage() -> dict[str, Any]:
    if any(path.exists() for path in (PRELABEL_QA, SCORE_REPORT, METRIC_QA,
                                      ARTIFACT_ROOT / "FIT_APPLICATION_R1.json")):
        raise PilotError("Pilot source cannot be re-prepared after operations begin")
    r2, amendment, _bag2, _runner, source = _parent_source()
    e67, e75 = source[67], source[75]
    arrays: dict[str, np.ndarray] = {}
    roles = {
        "D1_fit": (e67, (68, 71), True, False),
        "D1_score": (e67, (72, 75), False, True),
        "D2_fit": (e67, (68, 75), True, False),
        "D2_score": (e75, (76, 79), False, True),
    }
    for prefix, (origin, window, missing_only, score) in roles.items():
        cycles = np.asarray(origin["cycles"], dtype=np.int64)
        mask = (cycles >= window[0]) & (cycles <= window[1])
        if missing_only:
            mask &= np.asarray(origin["missing"], dtype=bool)
        arrays.update(_select(origin, prefix, mask, score=score))
    descriptions = {prefix: _describe(arrays, prefix) for prefix in roles}
    for prefix, (origin, window, missing_only, score) in roles.items():
        if descriptions[prefix]["cycles"] != list(window):
            raise PilotError(f"Cycle window differs from frozen recipe for {prefix}")
        if missing_only and not np.asarray(arrays[f"{prefix}_missing"], dtype=bool).all():
            raise PilotError(f"Calibration source contains present rows: {prefix}")
        if score:
            baseline = np.asarray(arrays[f"{prefix}_baseline"], dtype=np.float64)
            champion = np.asarray(arrays[f"{prefix}_champion_hinge"], dtype=np.float64)
            missing = np.asarray(arrays[f"{prefix}_missing"], dtype=bool)
            if not np.isfinite(baseline).all() or not np.isfinite(champion).all():
                raise PilotError(f"Nonfinite frozen prediction in {prefix}")
            if not np.array_equal(champion[missing].view("<u8"), baseline[missing].view("<u8")):
                raise PilotError(f"Frozen champion's missing R06 fallback is not exact in {prefix}")
            if np.asarray(arrays[f"{prefix}_entry_friction"], dtype=np.float64).shape != baseline.shape:
                raise PilotError(f"Fixed REG group arrays are misaligned in {prefix}")
    if (descriptions["D1_fit"]["missing_rows"] < MIN_SUPPORT
            or descriptions["D2_fit"]["missing_rows"] < MIN_SUPPORT
            or descriptions["D1_score"]["missing_rows"] < MIN_SUPPORT
            or descriptions["D2_score"]["missing_rows"] < MIN_SUPPORT):
        raise PilotError("One or more missing selection/score folds has fewer than 100 rows")
    score_rows = np.concatenate([arrays["D1_score_query_rows"], arrays["D2_score_query_rows"]])
    score_ids = np.concatenate([arrays["D1_score_query_ids"], arrays["D2_score_query_ids"]])
    if (len(score_rows) != EXPECTED_POOLED_ROWS or len(np.unique(score_rows)) != EXPECTED_POOLED_ROWS
            or len(np.unique(score_ids)) != EXPECTED_POOLED_ROWS):
        raise PilotError("Pooled D1/D2 score rows are not the planned unique 15,953 rows")
    train_path = _training_path()
    gate = _gate_module()
    train_hash = gate._sha_file(train_path)
    if train_hash != r2.get("official_train_sha256"):
        raise PilotError("Official Train source changed since the parent freeze")
    source_receipts = {
        "pilot_source_freeze_r1": _sha_file(PILOT_FREEZE_R1),
        "pilot_source_freeze_r2": _sha_file(PILOT_FREEZE_R2),
        "pilot_verifier_amendment_r3": _sha_file(PILOT_AMENDMENT_R3),
        "source_e67": r2["origins"]["67"]["archive"]["sha256"],
        "source_e75": r2["origins"]["75"]["archive"]["sha256"],
        "official_train": train_hash,
        "r18_freeze": _sha_file(R18_FREEZE), "r18_summary": _sha_file(R18_SUMMARY),
        "r19_recipe": _sha_file(R19_RECIPE), "r19_summary": _sha_file(R19_SUMMARY),
    }
    archive_receipt = _write_npz_new(SOURCE_ARCHIVE, arrays)
    recipe = _read_json(RECIPE_PATH)
    _verify_signed(recipe, "recipe_signature", "missing median recipe")
    body = {
        "schema_version": 1, "experiment_id": LANE, "status": "SOURCE_FROZEN_PRELABEL",
        "recipe_sha256": _sha_file(RECIPE_PATH), "recipe_signature": recipe["recipe_signature"],
        "root_plan_sha256": _sha_file(ROOT_PLAN_PATH),
        "parent_source_signature": r2["source_signature"],
        "parent_source_freeze_sha256": _sha_file(PILOT_FREEZE_R2),
        "parent_source_verifier_amendment_r3_sha256": _sha_file(PILOT_AMENDMENT_R3),
        "parent_prediction_freeze_sha256": _sha_file(PILOT_ROOT / "PREDICTION_FREEZE_R2.json"),
        "parent_official_train_sha256": train_hash,
        "parent_source_receipts": source_receipts,
        "source_code_hashes": _source_code_hashes(),
        "source_archive": archive_receipt,
        "role_sources": descriptions,
        "pooled_score_rows": EXPECTED_POOLED_ROWS,
        "fit_application_budget": {"scalar_fits": 2, "applications": 2,
                                    "learner_fits": 0, "preprocessor_fits": 0,
                                    "position_fits": 0, "hinge_fits": 0},
        "execution_boundary": {"selection_labels_read": False, "score_labels_read": False,
                               "target_values_persisted": False, "test_or_sample_accessed": False,
                               "kaggle_or_upload": False, "fit_started": False,
                               "application_started": False},
        "temporal_disclosure": "D2 selection cycles 68-75 include D1 score cycles 72-75; those labels are used only for the D2 calibration fit before any D1 metric scoring. D1 and D2 score rows are distinct but not independent validation.",
        "historical_novelty": {
            "R18": "global half-median offset capped at +/-2; retired, not missing-only",
            "R19": "half-median offsets by entry_friction; retired, not missing-only",
            "equivalent_missing_only_full_median_found": False,
        },
    }
    report = _write_json_new(SOURCE_REPORT, body, "source_signature")
    return {"status": report["status"], "source_signature": report["source_signature"],
            "source_archive_sha256": archive_receipt["sha256"],
            "D1_fit_missing": descriptions["D1_fit"]["missing_rows"],
            "D1_score_missing": descriptions["D1_score"]["missing_rows"],
            "D2_fit_missing": descriptions["D2_fit"]["missing_rows"],
            "D2_score_missing": descriptions["D2_score"]["missing_rows"],
            "pooled_score_rows": EXPECTED_POOLED_ROWS, "target_values_read": False}


def _training_path() -> Path:
    _r2, _amend, bag2, runner, _src = _parent_source()
    return runner._contained_file(bag2.SOURCE_ROOT, runner.TRAIN_REL)


def _verified_source() -> tuple[dict[str, Any], dict[str, np.ndarray], Any, Any]:
    r2, _amend, bag2, runner, _parent_arrays = _parent_source()
    report = _read_json(SOURCE_REPORT)
    _verify_signed(report, "source_signature", "missing median source freeze")
    recipe = _read_json(RECIPE_PATH)
    _verify_signed(recipe, "recipe_signature", "missing median recipe")
    if (report.get("status") != "SOURCE_FROZEN_PRELABEL"
            or report.get("parent_source_signature") != r2.get("source_signature")
            or report.get("parent_source_freeze_sha256") != _sha_file(PILOT_FREEZE_R2)
            or report.get("source_code_hashes") != _source_code_hashes()
            or report.get("source_archive", {}).get("sha256") != _sha_file(SOURCE_ARCHIVE)
            or report.get("parent_official_train_sha256") != r2.get("official_train_sha256")
            or report.get("recipe_sha256") != _sha_file(RECIPE_PATH)):
        raise PilotError("Missing median source freeze binding failed")
    arrays = _verify_npz(SOURCE_ARCHIVE, report["source_archive"])
    return report, arrays, bag2, runner


def _operation_events() -> list[dict[str, Any]]:
    path = ARTIFACT_ROOT / "lane_ledger.jsonl"
    if not path.is_file():
        return []
    events = []
    for i, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        try:
            item = json.loads(line)
        except Exception as exc:
            raise PilotError(f"Malformed lane ledger at line {i}") from exc
        if not isinstance(item, dict):
            raise PilotError(f"Malformed lane ledger event at line {i}")
        events.append(item)
    return events


def _start_operation(op_id: str, kind: str) -> None:
    existing = [e for e in _operation_events() if e.get("operation_id") == op_id]
    if existing:
        raise PilotError(f"Operation already started or failed; retry forbidden: {op_id}")
    ARTIFACT_ROOT.mkdir(parents=True, exist_ok=True)
    with (ARTIFACT_ROOT / "lane_ledger.jsonl").open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(_canonical({"schema_version": 1, "event": "OPERATION_STARTED", "operation_id": op_id,
                                 "operation_type": kind, "utc": datetime.now(timezone.utc).isoformat()}).decode()+"\n")
        stream.flush(); os.fsync(stream.fileno())


def _finish_operation(op_id: str, kind: str, artifact_sha: str) -> None:
    with (ARTIFACT_ROOT / "lane_ledger.jsonl").open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(_canonical({"schema_version": 1, "event": "OPERATION_SAVED", "operation_id": op_id,
                                 "operation_type": kind, "artifact_sha256": artifact_sha,
                                 "utc": datetime.now(timezone.utc).isoformat()}).decode()+"\n")
        stream.flush(); os.fsync(stream.fileno())


def _fail_operation(op_id: str, exc: Exception) -> None:
    path = ARTIFACT_ROOT / "lane_ledger.jsonl"
    with path.open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(_canonical({"schema_version": 1, "event": "OPERATION_FAILED", "operation_id": op_id,
                                 "error": f"{type(exc).__name__}: {exc}",
                                 "utc": datetime.now(timezone.utc).isoformat()}).decode()+"\n")
        stream.flush(); os.fsync(stream.fileno())


def _read_allowed_targets(train_path: Path, runner: Any, rows: np.ndarray, ids: np.ndarray) -> np.ndarray:
    y = runner._read_targets_for_rows(train_path, np.asarray(rows, dtype=np.int64),
                                      np.asarray(ids, dtype=str), expected_total_rows=50608)
    y = np.asarray(y, dtype=np.float64)
    if y.shape != np.asarray(rows).shape or not np.isfinite(y).all():
        raise PilotError("Authorized Train target vector has invalid shape or nonfinite values")
    return y


def _median_residual_offset(y: np.ndarray, baseline: np.ndarray) -> tuple[float, np.ndarray]:
    residual = np.asarray(y, dtype=np.float64) - np.asarray(baseline, dtype=np.float64)
    return float(np.median(residual)), residual


def _apply_missing_offset(baseline: np.ndarray, champion: np.ndarray,
                          missing: np.ndarray, delta: float) -> np.ndarray:
    mask = np.asarray(missing, dtype=bool)
    result = np.asarray(champion, dtype=np.float64).copy()
    result[mask] = np.clip(np.asarray(baseline, dtype=np.float64)[mask] + float(delta), 0.0, 100.0)
    return result


def _fit_delta(name: str, arrays: Mapping[str, np.ndarray], train_path: Path, runner: Any,
               source: Mapping[str, Any]) -> dict[str, Any]:
    prefix = f"{name}_fit"
    rows = np.asarray(arrays[f"{prefix}_query_rows"], dtype=np.int64)
    ids = np.asarray(arrays[f"{prefix}_query_ids"], dtype=str)
    baseline = np.asarray(arrays[f"{prefix}_baseline"], dtype=np.float64)
    if len(rows) < MIN_SUPPORT or not np.isfinite(baseline).all():
        raise PilotError(f"Insufficient or invalid missing selection source: {name}")
    if ARTIFACT_ROOT.joinpath("scale_states", f"{name}_R1.json").exists():
        raise PilotError(f"Prior scale state exists; retry forbidden: {name}")
    op_id = f"FIT_{name}_R1"
    _start_operation(op_id, "missing_only_median_y_minus_R06")
    try:
        y = _read_allowed_targets(train_path, runner, rows, ids)
        delta, residual = _median_residual_offset(y, baseline)
        if not np.isfinite(delta):
            raise PilotError(f"Nonfinite missing median residual: {name}")
        body = {"schema_version": 1, "experiment_id": LANE, "state_id": name,
                "status": "FIT_COMPLETE", "delta": delta,
                "residual_definition": "y - R06_baseline; positive raises candidate prediction",
                "fit_rows": len(rows), "missing_fit_rows": len(rows),
                "selection_cycles": list(FIT_WINDOWS[name]),
                "selection_rows_sha256": _sha_array(rows), "selection_ids_sha256": _sha_array(ids),
                "selection_y_sha256": _sha_array(y), "selection_residual_sha256": _sha_array(residual),
                "baseline_sha256": _sha_array(baseline), "source_signature": source["source_signature"],
                "source_archive_sha256": source["source_archive"]["sha256"],
                "target_values_persisted": False, "labels_used": True,
                "labels_read_for": f"{name} calibration selection only"}
        path = ARTIFACT_ROOT / "scale_states" / f"{name}_R1.json"
        state = _write_json_new(path, body, "state_signature")
        _finish_operation(op_id, "missing_only_median_y_minus_R06", _sha_file(path))
        return state
    except Exception as exc:
        _fail_operation(op_id, exc)
        raise


def _apply_delta(name: str, arrays: Mapping[str, np.ndarray], state: Mapping[str, Any],
                 source: Mapping[str, Any]) -> dict[str, Any]:
    prefix = f"{name}_score"
    rows = np.asarray(arrays[f"{prefix}_query_rows"], dtype=np.int64)
    ids = np.asarray(arrays[f"{prefix}_query_ids"], dtype=str)
    cycles = np.asarray(arrays[f"{prefix}_cycles"], dtype=np.int64)
    missing = np.asarray(arrays[f"{prefix}_missing"], dtype=bool)
    baseline = np.asarray(arrays[f"{prefix}_baseline"], dtype=np.float64)
    champion = np.asarray(arrays[f"{prefix}_champion_hinge"], dtype=np.float64)
    delta = float(state["delta"])
    if int(missing.sum()) < MIN_SUPPORT:
        raise PilotError(f"Insufficient missing score support: {name}")
    op_id = f"APPLY_{name}_R1"
    _start_operation(op_id, "deterministic_missing_only_offset_application")
    try:
        candidate = _apply_missing_offset(baseline, champion, missing, delta)
        if not np.array_equal(candidate[~missing].view("<u8"), champion[~missing].view("<u8")):
            raise PilotError(f"Present champion changed: {name}")
        prediction = {
            "query_rows": rows, "query_ids": ids, "cycles": cycles, "missing": missing,
            "baseline": baseline, "champion_hinge": champion, "candidate_hinge": candidate,
            "entry_friction": np.asarray(arrays[f"{prefix}_entry_friction"], dtype=np.float64),
            "impact_quartile": np.asarray(arrays[f"{prefix}_impact_quartile"], dtype=np.int8),
            "text_quartile": np.asarray(arrays[f"{prefix}_text_quartile"], dtype=np.int8),
        }
        path = ARTIFACT_ROOT / "predictions" / f"{name}_R1.npz"
        receipt = _write_npz_new(path, prediction)
        receipt_body = {"schema_version": 1, "experiment_id": LANE, "fold": name,
                        "status": "PREDICTION_SAVED", "prediction": receipt,
                        "state_signature": state["state_signature"],
                        "source_signature": source["source_signature"],
                        "rows": len(rows), "missing_rows": int(missing.sum()),
                        "present_exact": True, "target_values_persisted": False}
        receipt_path = ARTIFACT_ROOT / "predictions" / f"{name}_R1.json"
        signed_receipt = _write_json_new(receipt_path, receipt_body, "receipt_signature")
        _finish_operation(op_id, "deterministic_missing_only_offset_application", _sha_file(receipt_path))
        return {"receipt": signed_receipt, "receipt_sha256": _sha_file(receipt_path),
                "present_exact": True, "changed_missing_rows": int(np.count_nonzero(
                    candidate[missing].view("<u8") != champion[missing].view("<u8")))}
    except Exception as exc:
        _fail_operation(op_id, exc)
        raise


def _fit_apply_stage() -> dict[str, Any]:
    source, arrays, bag2, runner = _verified_source()
    manifest_path = ARTIFACT_ROOT / "FIT_APPLICATION_R1.json"
    if manifest_path.exists() or (ARTIFACT_ROOT / "lane_ledger.jsonl").exists():
        raise PilotError("Fit/application output or ledger exists; retry forbidden")
    available = int(psutil.virtual_memory().available)
    required = MEMORY_START + MEMORY_RESERVE
    if available < required:
        raise PilotError(f"RAM gate blocked before operation: available={available}, required={required}")
    train_path = runner._contained_file(bag2.SOURCE_ROOT, runner.TRAIN_REL)
    if _sha_file(train_path) != source["parent_official_train_sha256"]:
        raise PilotError("Official Train hash changed after target-free source freeze")
    # Selection labels are read only in the matching fold. D2 selection includes
    # cycles 72-75, also used as D1 score rows; they are not metric-scored here.
    with threadpool_limits(limits=2):
        states = {name: _fit_delta(name, arrays, train_path, runner, source)
                  for name in ("D1", "D2")}
        predictions = {name: _apply_delta(name, arrays, states[name], source)
                       for name in ("D1", "D2")}
    body = {"schema_version": 1, "experiment_id": LANE, "status": "FIT_APPLICATION_COMPLETE",
            "source_signature": source["source_signature"],
            "source_archive_sha256": source["source_archive"]["sha256"],
            "states": {name: {"path": f"experiments/runs/{LANE}/scale_states/{name}_R1.json",
                             "sha256": _sha_file(ARTIFACT_ROOT / "scale_states" / f"{name}_R1.json"),
                             "state_signature": states[name]["state_signature"],
                             "delta": states[name]["delta"]} for name in states},
            "predictions": {name: predictions[name] for name in predictions},
            "operation_budget": {"fit_started": 2, "fit_saved": 2, "application_started": 2,
                                  "application_saved": 2, "learner_fits": 0, "preprocessor_fits": 0,
                                  "position_fits": 0, "hinge_fits": 0},
            "resource_snapshot": {"available_ram_bytes": available,
                                  "initial_ram_floor_bytes": MEMORY_START,
                                  "reserve_bytes": MEMORY_RESERVE, "threads": 2},
            "selection_target_rows_read": {name: states[name]["fit_rows"] for name in states},
            "score_metric_labels_read": False,
            "D1_score_cycles_used_as_D2_selection_only": [72, 75],
            "target_values_persisted": False, "test_or_sample_accessed": False,
            "kaggle_or_upload": False}
    manifest = _write_json_new(manifest_path, body, "manifest_signature")
    return {"status": manifest["status"], "manifest_sha256": _sha_file(manifest_path),
            "deltas": {name: states[name]["delta"] for name in states},
            "fit_count": 2, "application_count": 2, "score_labels_read": False}


def _score_stage() -> dict[str, Any]:
    source, source_arrays, bag2, runner = _verified_source()
    pre = _read_json(PRELABEL_QA)
    _verify_signed(pre, "qa_signature", "independent prelabel QA")
    if (pre.get("status") != "PASS" or pre.get("source_signature") != source["source_signature"]
            or pre.get("fit_application_manifest_sha256") != _sha_file(ARTIFACT_ROOT / "FIT_APPLICATION_R1.json")
            or pre.get("score_labels_read") is not False):
        raise PilotError("Score blocked until matching independent prelabel QA PASS")
    if SCORE_REPORT.exists() or any(e.get("operation_id") == "SCORE_R1" for e in _operation_events()):
        raise PilotError("Score operation already exists; repeat scoring forbidden")
    app = _read_json(ARTIFACT_ROOT / "FIT_APPLICATION_R1.json")
    _verify_signed(app, "manifest_signature", "fit/application manifest")
    train_path = runner._contained_file(bag2.SOURCE_ROOT, runner.TRAIN_REL)
    if _sha_file(train_path) != source["parent_official_train_sha256"]:
        raise PilotError("Official Train changed before score pass")
    arrays_by_fold: dict[str, dict[str, np.ndarray]] = {}
    for name in ("D1", "D2"):
        receipt = _read_json(ARTIFACT_ROOT / "predictions" / f"{name}_R1.json")
        _verify_signed(receipt, "receipt_signature", f"prediction receipt {name}")
        arrays_by_fold[name] = _verify_npz(ROOT / receipt["prediction"]["path"], receipt["prediction"])
    score_rows = np.concatenate([arrays_by_fold[k]["query_rows"] for k in ("D1", "D2")])
    score_ids = np.concatenate([arrays_by_fold[k]["query_ids"] for k in ("D1", "D2")]).astype(str)
    if len(score_rows) != EXPECTED_POOLED_ROWS or len(np.unique(score_rows)) != EXPECTED_POOLED_ROWS:
        raise PilotError("Score row identity changed after prelabel QA")
    _start_operation("SCORE_R1", "single_fullchamp_matched_score_pass")
    try:
        fold_data: dict[str, dict[str, Any]] = {}
        labels: dict[str, np.ndarray] = {}
        for name in ("D1", "D2"):
            a = arrays_by_fold[name]
            y = _read_allowed_targets(train_path, runner, a["query_rows"], a["query_ids"])
            labels[name] = y
            fold_data[name] = _score_fold(name, a, y)
        pooled_y = np.concatenate([labels[k] for k in ("D1", "D2")])
        pooled_base = np.concatenate([arrays_by_fold[k]["champion_hinge"] for k in ("D1", "D2")])
        pooled_candidate = np.concatenate([arrays_by_fold[k]["candidate_hinge"] for k in ("D1", "D2")])
        pooled_missing = np.concatenate([arrays_by_fold[k]["missing"] for k in ("D1", "D2")]).astype(bool)
        pooled_meta = {key: np.concatenate([arrays_by_fold[k][key] for k in ("D1", "D2")])
                       for key in ("entry_friction", "impact_quartile", "text_quartile")}
        pooled_groups = _groups(pooled_y, pooled_missing, pooled_meta)
        pooled_metrics = _group_metrics(pooled_y, pooled_base, pooled_candidate, pooled_groups)
        overall_base = _metric(pooled_y, pooled_base)
        overall_candidate = _metric(pooled_y, pooled_candidate)
        pooled_missing_metrics = _missing_metrics(pooled_y, pooled_base, pooled_candidate, pooled_missing)
        fold_gains = [fold_data[k]["overall"]["mae_gain_candidate_vs_champion"] for k in ("D1", "D2")]
        missing_harms = [fold_data[k]["missing"]["mae_harm_candidate_minus_champion"] for k in ("D1", "D2")]
        all_group_harms = [item["harm_candidate_minus_champion"] for item in pooled_metrics.values()
                           if item["status"] == "MEASURED"]
        max_harm = max(all_group_harms, default=None)
        both_supported = all(fold_data[k]["missing"]["rows"] >= MIN_SUPPORT for k in ("D1", "D2"))
        present_exact = all(fold_data[k]["present_bit_exact"] for k in ("D1", "D2"))
        pooled_gain = overall_base["mae"] - overall_candidate["mae"]
        group_harm_pass = max_harm is not None and max_harm <= 0.25
        scientific = bool(both_supported and all(x > 0 for x in [fold_data["D1"]["missing"]["mae_gain_candidate_vs_champion"],
                                                                 fold_data["D2"]["missing"]["mae_gain_candidate_vs_champion"]])
                           and pooled_gain > 0 and group_harm_pass and present_exact)
        exploratory = bool(both_supported and pooled_gain > 0 and all(h <= 0.10 for h in missing_harms)
                           and group_harm_pass and present_exact)
        score_body = {
            "schema_version": 1, "experiment_id": LANE, "status": "SCORED",
            "source_signature": source["source_signature"],
            "source_archive_sha256": source["source_archive"]["sha256"],
            "prelabel_qa_sha256": _sha_file(PRELABEL_QA),
            "fit_application_manifest_sha256": _sha_file(ARTIFACT_ROOT / "FIT_APPLICATION_R1.json"),
            "folds": fold_data, "pooled_rows": len(score_rows),
            "pooled_row_identity_sha256": _canonical_sha({"rows": _sha_array(score_rows), "ids": _sha_array(score_ids)}),
            "pooled_overall": {"champion": overall_base, "candidate": overall_candidate,
                               "mae_gain_candidate_vs_champion": pooled_gain},
            "pooled_missing": pooled_missing_metrics,
            "pooled_fixed_groups": pooled_metrics,
            "maximum_supported_group_harm": max_harm,
            "both_fold_missing_support_pass": both_supported,
            "present_predictions_bit_exact": present_exact,
            "query_target_rows_read": len(score_rows), "target_values_persisted": False,
            "test_or_sample_accessed": False, "kaggle_or_upload": False,
            "selection_disclosure": "D2 calibration selection includes E67 cycles 68-75, including rows also used for D1 score. Those targets entered only the D2 selection fit; D1 metrics are first computed in this single score stage after both output archives passed prelabel QA.",
            "temporal_disclosure": "D1 and D2 score IDs are disjoint but not independent validation; E75 training includes the earlier D1 score cycles.",
            "fit_application_accounting": {"scalar_fits": 2, "applications": 2, "learner_fits": 0,
                                            "preprocessor_fits": 0, "position_fits": 0, "hinge_fits": 0},
            "scientific_gate_without_independent_metric_qa": scientific,
            "exploratory_gate_without_independent_metric_qa": exploratory,
            "independent_metric_qa_pass": False,
        }
        report = _write_json_new(SCORE_REPORT, score_body, "score_signature")
        _finish_operation("SCORE_R1", "single_fullchamp_matched_score_pass", _sha_file(SCORE_REPORT))
        return {"status": report["status"], "score_signature": report["score_signature"],
                "score_sha256": _sha_file(SCORE_REPORT), "pooled_rows": len(score_rows),
                "pooled_mae_gain": pooled_gain, "D1_missing_gain": fold_data["D1"]["missing"]["mae_gain_candidate_vs_champion"],
                "D2_missing_gain": fold_data["D2"]["missing"]["mae_gain_candidate_vs_champion"],
                "scientific_gate_before_metric_qa": scientific,
                "exploratory_gate_before_metric_qa": exploratory}
    except Exception as exc:
        _fail_operation("SCORE_R1", exc)
        raise


def _metric(y: np.ndarray, prediction: np.ndarray) -> dict[str, Any]:
    y = np.asarray(y, dtype=np.float64); p = np.asarray(prediction, dtype=np.float64)
    residual = y - p
    err = p - y
    return {"rows": len(y), "mae": float(np.mean(np.abs(err))),
            "p90_absolute_error": float(np.quantile(np.abs(err), 0.90, method="linear")),
            "median_residual_y_minus_prediction": float(np.median(residual)),
            "mean_residual_y_minus_prediction": float(np.mean(residual)),
            "bias_prediction_minus_target": float(np.mean(err))}


def _missing_metrics(y: np.ndarray, champion: np.ndarray, candidate: np.ndarray,
                     missing: np.ndarray) -> dict[str, Any]:
    mask = np.asarray(missing, dtype=bool)
    base, cand = _metric(y[mask], champion[mask]), _metric(y[mask], candidate[mask])
    return {"rows": int(mask.sum()), "status": "MEASURED" if int(mask.sum()) >= MIN_SUPPORT else "UNKNOWN",
            "champion": base if int(mask.sum()) >= MIN_SUPPORT else None,
            "candidate": cand if int(mask.sum()) >= MIN_SUPPORT else None,
            "mae_gain_candidate_vs_champion": base["mae"] - cand["mae"] if int(mask.sum()) >= MIN_SUPPORT else None,
            "mae_harm_candidate_minus_champion": cand["mae"] - base["mae"] if int(mask.sum()) >= MIN_SUPPORT else None}


def _groups(y: np.ndarray, missing: np.ndarray, meta: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    groups: dict[str, np.ndarray] = {"missing": np.asarray(missing, dtype=bool),
                                    "present": ~np.asarray(missing, dtype=bool),
                                    "target_lt5": np.asarray(y) < 5.0,
                                    "target_ge95": np.asarray(y) >= 95.0,
                                    "EF0": np.asarray(meta["entry_friction"], dtype=np.float64) == 0.0,
                                    "EF1": np.asarray(meta["entry_friction"], dtype=np.float64) > 0.0}
    for key, label in (("impact_quartile", "impact_load"), ("text_quartile", "text_load")):
        bucket = np.asarray(meta[key], dtype=np.int8)
        for i in range(4):
            groups[f"{label}_Q{i+1}"] = bucket == i
    return groups


def _group_metrics(y: np.ndarray, champion: np.ndarray, candidate: np.ndarray,
                   groups: Mapping[str, np.ndarray]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for name, mask0 in groups.items():
        mask = np.asarray(mask0, dtype=bool); n = int(mask.sum())
        if n < MIN_SUPPORT:
            out[name] = {"status": "UNKNOWN", "rows": n, "champion": None,
                         "candidate": None, "harm_candidate_minus_champion": None}
        else:
            base, cand = _metric(y[mask], champion[mask]), _metric(y[mask], candidate[mask])
            out[name] = {"status": "MEASURED", "rows": n, "champion": base,
                         "candidate": cand, "harm_candidate_minus_champion": cand["mae"] - base["mae"]}
    return out


def _score_fold(name: str, arrays: Mapping[str, np.ndarray], y: np.ndarray) -> dict[str, Any]:
    missing = np.asarray(arrays["missing"], dtype=bool)
    champion = np.asarray(arrays["champion_hinge"], dtype=np.float64)
    candidate = np.asarray(arrays["candidate_hinge"], dtype=np.float64)
    meta = {k: arrays[k] for k in ("entry_friction", "impact_quartile", "text_quartile")}
    if not np.array_equal(candidate[~missing].view("<u8"), champion[~missing].view("<u8")):
        raise PilotError(f"Present full-champion bits changed in score fold {name}")
    missing_metric = _missing_metrics(y, champion, candidate, missing)
    group_metrics = _group_metrics(y, champion, candidate, _groups(y, missing, meta))
    cycles: dict[str, Any] = {}
    for cycle in np.unique(np.asarray(arrays["cycles"], dtype=np.int64)):
        cmask = np.asarray(arrays["cycles"], dtype=np.int64) == cycle
        mmask = cmask & missing
        n_missing = int(mmask.sum())
        cycles[str(int(cycle))] = {
            "rows": int(cmask.sum()), "missing_rows": n_missing,
            "missing_status": "MEASURED" if n_missing >= MIN_SUPPORT else "UNKNOWN",
            "champion": _metric(y[cmask], champion[cmask]),
            "candidate": _metric(y[cmask], candidate[cmask]),
            "overall_mae_gain_candidate_vs_champion": _metric(y[cmask], champion[cmask])["mae"]
                - _metric(y[cmask], candidate[cmask])["mae"],
            "missing_champion": _metric(y[mmask], champion[mmask]) if n_missing >= MIN_SUPPORT else None,
            "missing_candidate": _metric(y[mmask], candidate[mmask]) if n_missing >= MIN_SUPPORT else None,
            "missing_mae_gain_candidate_vs_champion": (
                _metric(y[mmask], champion[mmask])["mae"] - _metric(y[mmask], candidate[mmask])["mae"]
                if n_missing >= MIN_SUPPORT else None),
        }
    base, cand = _metric(y, champion), _metric(y, candidate)
    return {"origin": 67 if name == "D1" else 75, "score_cycles": list(SCORE_WINDOWS[name]),
            "rows": len(y), "missing": missing_metric,
            "overall": {"champion": base, "candidate": cand,
                        "mae_gain_candidate_vs_champion": base["mae"] - cand["mae"]},
            "per_cycle": cycles, "fixed_groups": group_metrics,
            "present_bit_exact": True,
            "row_identity_sha256": _canonical_sha({"rows": _sha_array(arrays["query_rows"]),
                                                    "ids": _sha_array(arrays["query_ids"]),
                                                    "cycles": _sha_array(arrays["cycles"])}),
            "score_labels_read": len(y)}


def _final_source_stage() -> dict[str, Any]:
    score = _read_json(SCORE_REPORT); qa = _read_json(METRIC_QA)
    _verify_signed(score, "score_signature", "score report")
    _verify_signed(qa, "qa_signature", "independent metric QA")
    if (qa.get("status") != "PASS" or qa.get("score_signature") != score.get("score_signature")
            or qa.get("exploratory_benchmark_eligible") is not True):
        return {"status": "NOT_PREPARED", "reason": "Pilot did not pass exploratory eligibility"}
    r2, _amendment, _bag2, _runner, src = _parent_source()
    e67, e75 = src[67], src[75]
    keep67 = (e67["cycles"] >= 68) & (e67["cycles"] <= 75)
    keep75 = (e75["cycles"] >= 76) & (e75["cycles"] <= 79)
    keys = ("query_rows", "query_ids", "cycles", "missing", "baseline", "champion_hinge")
    arrays = {key: np.concatenate([e67[key][keep67], e75[key][keep75]]) for key in keys}
    order = np.argsort(arrays["query_rows"], kind="stable")
    arrays = {key: value[order] for key, value in arrays.items()}
    if (len(arrays["query_rows"]) != EXPECTED_POOLED_ROWS
            or len(np.unique(arrays["query_rows"])) != EXPECTED_POOLED_ROWS
            or len(np.unique(arrays["query_ids"].astype(str))) != EXPECTED_POOLED_ROWS):
        raise PilotError("Final source OOF population is not unique")
    receipt = _write_npz_new(FINAL_SOURCE_ARCHIVE, arrays)
    body = {"schema_version": 1, "experiment_id": LANE, "status": "FINAL_SOURCE_READY_PENDING_ROOT_REVIEW",
            "score_signature": score["score_signature"], "metric_qa_signature": qa["qa_signature"],
            "parent_source_signature": r2["source_signature"], "archive": receipt,
            "rows": len(arrays["query_rows"]), "cycles": [68, 79],
            "missing_rows": int(np.asarray(arrays["missing"], dtype=bool).sum()),
            "final_delta_formula": "median(y - R06_baseline) over unique missing OOF rows, E67 cycles 68-75 plus E75 cycles 76-79",
            "present_prediction": "exact frozen causal full-champion hinge bits",
            "final_scalar_fit_started": False, "E79_application_started": False,
            "test_or_sample_accessed": False, "kaggle_or_upload": False}
    report = _write_json_new(FINAL_SOURCE_REPORT, body, "source_signature")
    return {"status": report["status"], "source_signature": report["source_signature"],
            "archive_sha256": receipt["sha256"], "rows": len(arrays["query_rows"]),
            "final_scalar_fit_started": False, "E79_application_started": False}


def _synthetic_e2e() -> dict[str, Any]:
    rng = np.random.default_rng(20261106)
    checks = []
    with tempfile.TemporaryDirectory(prefix="reg-missing-median-offset-") as temp:
        folder = Path(temp)
        for name, offset in (("D1", 2.5), ("D2", -1.75)):
            fit_base = rng.uniform(10, 90, 140).astype(np.float64)
            fit_y = fit_base + offset + rng.normal(0, 1.0, len(fit_base))
            delta, _ = _median_residual_offset(fit_y, fit_base)
            score_n = 180
            missing = np.zeros(score_n, dtype=bool); missing[:125] = True
            baseline = rng.uniform(0, 100, score_n).astype(np.float64)
            champion = rng.uniform(0, 100, score_n).astype(np.float64)
            candidate = _apply_missing_offset(baseline, champion, missing, delta)
            if not np.array_equal(candidate[~missing].view("<u8"), champion[~missing].view("<u8")):
                raise PilotError(f"Synthetic present bits changed: {name}")
            if not np.array_equal(candidate[missing], np.clip(baseline[missing] + delta, 0.0, 100.0)):
                raise PilotError(f"Synthetic missing formula failed: {name}")
            receipt = _write_npz_new(folder / f"{name}.npz", {"baseline": baseline, "missing": missing,
                                          "champion": champion, "candidate": candidate})
            roundtrip = _verify_npz(folder / f"{name}.npz", receipt)
            if not np.array_equal(roundtrip["candidate"], candidate):
                raise PilotError(f"Synthetic receipt readback failed: {name}")
            state = _write_json_new(folder / f"{name}.json", {"fold": name, "delta": delta,
                                          "fit_rows": len(fit_y), "target_values_persisted": False}, "state_signature")
            if abs(state["delta"] - delta) > 0 or state["fit_rows"] < MIN_SUPPORT:
                raise PilotError(f"Synthetic state receipt failed: {name}")
            checks.append({"fold": name, "fit_rows": len(fit_y), "score_missing": int(missing.sum()),
                           "delta": delta, "receipt_sha256": receipt["sha256"]})
    return {"status": "PASS", "synthetic_only": True, "folds": checks,
            "real_inputs_read": False, "official_targets_read": False,
            "production_ledger_touched": False, "temporary_outputs_removed": True}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--synthetic-e2e", action="store_true")
    mode.add_argument("--source", action="store_true")
    mode.add_argument("--fit-apply", action="store_true")
    mode.add_argument("--score", action="store_true")
    mode.add_argument("--prepare-final-source", action="store_true")
    args = parser.parse_args(argv)
    try:
        if args.synthetic_e2e:
            result = _synthetic_e2e()
        elif args.source:
            result = _source_stage()
        elif args.fit_apply:
            result = _fit_apply_stage()
        elif args.score:
            result = _score_stage()
        else:
            result = _final_source_stage()
    except Exception as exc:
        print(json.dumps({"status": "BLOCKED", "error": f"{type(exc).__name__}: {exc}"}, sort_keys=True))
        return 1
    print(json.dumps(_jsonable(result), sort_keys=True, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
