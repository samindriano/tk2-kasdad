"""Independent source, prelabel, metric and final-source QA for REG missing offset R1."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import zipfile
from typing import Any, Mapping

import numpy as np

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


class VerifyError(RuntimeError):
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
        raise VerifyError(f"Invalid JSON: {path}") from exc
    if not isinstance(value, dict):
        raise VerifyError(f"Expected JSON object: {path}")
    return value


def _verify_signed(value: Mapping[str, Any], key: str, label: str) -> None:
    body = {k: v for k, v in value.items() if k != key}
    if value.get(key) != _canonical_sha(body):
        raise VerifyError(f"Signature mismatch: {label}")


def _read_npz(path: Path, keys: tuple[str, ...]) -> dict[str, np.ndarray]:
    if not path.is_file():
        raise VerifyError(f"NPZ missing: {path}")
    with zipfile.ZipFile(path) as zf:
        members = {name[:-4] for name in zf.namelist() if name.endswith(".npy")}
    if members != set(keys):
        raise VerifyError(f"NPZ allowlist mismatch: {path}")
    with np.load(path, allow_pickle=False) as zf:
        return {k: np.asarray(zf[k]).copy() for k in keys}


def _verify_npz(path: Path, receipt: Mapping[str, Any]) -> dict[str, np.ndarray]:
    if _sha_file(path) != receipt.get("sha256"):
        raise VerifyError(f"NPZ hash mismatch: {path}")
    arrays = _read_npz(path, tuple(receipt.get("members", {})))
    if {k: _sha_array(v) for k, v in arrays.items()} != receipt.get("members"):
        raise VerifyError(f"NPZ member hashes mismatch: {path}")
    return arrays


def _source_code_hashes() -> dict[str, str]:
    paths = {
        "runner": ROOT / "scripts" / "reg_missing_median_offset_v1.py",
        "independent_verifier": Path(__file__),
        "regression_test": ROOT / "tests" / "test_reg_missing_median_offset_v1.py",
        "recipe": RECIPE_PATH, "root_plan": ROOT_PLAN_PATH,
        "pilot_gate_runner": PILOT_GATE, "pilot_gate_verifier": PILOT_VERIFIER,
        "bag2_helper": BAG2_HELPER, "hinge_runner": HINGE_RUNNER,
        "r18_freeze": R18_FREEZE, "r18_summary": R18_SUMMARY,
        "r19_recipe": R19_RECIPE, "r19_summary": R19_SUMMARY,
    }
    missing = [k for k, p in paths.items() if not p.is_file()]
    if missing:
        raise VerifyError(f"Frozen code/reference inputs missing: {missing}")
    return {k: _sha_file(p) for k, p in paths.items()}


def _parent_receipts() -> tuple[dict[str, Any], dict[int, dict[str, np.ndarray]]]:
    r1, r2 = _read_json(PILOT_FREEZE_R1), _read_json(PILOT_FREEZE_R2)
    _verify_signed(r1, "source_signature", "pilot source R1")
    _verify_signed(r2, "source_signature", "pilot source R2")
    if (r1.get("status") != "SOURCE_FROZEN" or r2.get("status") != "SOURCE_AMENDED_R2"
            or r2.get("source_freeze_r1_sha256") != _sha_file(PILOT_FREEZE_R1)
            or r1.get("source_signature") != r2.get("source_signature_r1")):
        raise VerifyError("Pilot source freeze lineage failed")
    amendment = _read_json(PILOT_AMENDMENT_R3)
    _verify_signed(amendment, "amendment_signature", "pilot verifier amendment R3")
    if (amendment.get("status") != "VERIFIER_AMENDED_R3"
            or amendment.get("source_freeze_sha256") != _sha_file(PILOT_FREEZE_R2)
            or amendment.get("source_signature") != r2.get("source_signature")
            or amendment.get("old_verifier_sha256") != r2["source_code_hashes"].get("independent_verifier")
            or amendment.get("new_verifier_sha256") != _sha_file(PILOT_VERIFIER)):
        raise VerifyError("Pilot verifier R3 amendment binding failed")
    gate = importlib.util.spec_from_file_location("verify_parent_pilot_gate", PILOT_GATE)
    if gate is None or gate.loader is None:
        raise VerifyError("Unable to load parent gate paths")
    parent = importlib.util.module_from_spec(gate); sys.modules[gate.name] = parent; gate.loader.exec_module(parent)
    bag2, runner = parent._helpers()
    if bag2._source_artifact_hashes(runner) != r2.get("source_artifacts"):
        raise VerifyError("Parent frozen source artifact hashes changed")
    current = parent._source_code_hashes(bag2)
    for name, value in current.items():
        expected = amendment["new_verifier_sha256"] if name == "independent_verifier" else r2["source_code_hashes"].get(name)
        if value != expected:
            raise VerifyError(f"Parent source code receipt mismatch: {name}")
    arrays = {}
    for origin in (67, 75):
        receipt = r2["origins"][str(origin)]["archive"]
        arrays[origin] = _verify_npz(ROOT / receipt["path"], receipt)
    return r2, arrays


def _source() -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    report = _read_json(SOURCE_REPORT)
    _verify_signed(report, "source_signature", "missing offset source freeze")
    recipe = _read_json(RECIPE_PATH)
    _verify_signed(recipe, "recipe_signature", "missing offset recipe")
    parent, parent_arrays = _parent_receipts()
    if (report.get("status") != "SOURCE_FROZEN_PRELABEL"
            or report.get("parent_source_signature") != parent.get("source_signature")
            or report.get("parent_source_freeze_sha256") != _sha_file(PILOT_FREEZE_R2)
            or report.get("source_code_hashes") != _source_code_hashes()
            or report.get("recipe_sha256") != _sha_file(RECIPE_PATH)
            or report.get("root_plan_sha256") != _sha_file(ROOT_PLAN_PATH)
            or report.get("source_archive", {}).get("sha256") != _sha_file(SOURCE_ARCHIVE)):
        raise VerifyError("Frozen source bindings failed")
    arrays = _verify_npz(SOURCE_ARCHIVE, report["source_archive"])
    roles = {
        "D1_fit": (67, (68, 71), True, False),
        "D1_score": (67, (72, 75), False, True),
        "D2_fit": (67, (68, 75), True, False),
        "D2_score": (75, (76, 79), False, True),
    }
    for prefix, (origin, window, missing_only, score) in roles.items():
        source_arrays = parent_arrays[origin]
        cycles = np.asarray(source_arrays["cycles"], dtype=np.int64)
        mask = (cycles >= window[0]) & (cycles <= window[1])
        if missing_only:
            mask &= np.asarray(source_arrays["missing"], dtype=bool)
        keys = ["query_rows", "query_ids", "cycles", "missing", "baseline"]
        if score:
            keys += ["champion_hinge", "entry_friction", "impact_quartile", "text_quartile"]
        for key in keys:
            expected = np.asarray(source_arrays[key])[mask]
            actual = arrays[f"{prefix}_{key}"]
            if _sha_array(actual) != _sha_array(expected):
                raise VerifyError(f"Frozen parent source mismatch: {prefix}.{key}")
        rows = arrays[f"{prefix}_query_rows"].astype(np.int64)
        ids = arrays[f"{prefix}_query_ids"].astype(str)
        selected_cycles = arrays[f"{prefix}_cycles"].astype(np.int64)
        identity = _canonical_sha({"rows": _sha_array(rows), "ids": _sha_array(ids),
                                   "cycles": _sha_array(selected_cycles)})
        if (report.get("role_sources", {}).get(prefix, {}).get("identity_sha256") != identity
                or report["role_sources"][prefix].get("rows") != len(rows)
                or report["role_sources"][prefix].get("missing_rows") != int(
                    np.asarray(arrays[f"{prefix}_missing"], dtype=bool).sum())):
            raise VerifyError(f"Frozen role identity receipt mismatch: {prefix}")
    return report, arrays


def _load_output(name: str) -> tuple[dict[str, Any], dict[str, Any], dict[str, np.ndarray]]:
    state_path = ARTIFACT_ROOT / "scale_states" / f"{name}_R1.json"
    state = _read_json(state_path); _verify_signed(state, "state_signature", f"scale state {name}")
    if (state.get("state_id") != name or state.get("status") != "FIT_COMPLETE"
            or state.get("target_values_persisted") is not False or state.get("labels_used") is not True):
        raise VerifyError(f"Scale state metadata invalid: {name}")
    receipt = _read_json(ARTIFACT_ROOT / "predictions" / f"{name}_R1.json")
    _verify_signed(receipt, "receipt_signature", f"prediction receipt {name}")
    if receipt.get("status") != "PREDICTION_SAVED" or receipt.get("state_signature") != state.get("state_signature"):
        raise VerifyError(f"Prediction receipt/state binding failed: {name}")
    arrays = _verify_npz(ROOT / receipt["prediction"]["path"], receipt["prediction"])
    return state, receipt, arrays


def _ledger() -> list[dict[str, Any]]:
    path = ARTIFACT_ROOT / "lane_ledger.jsonl"
    if not path.is_file():
        return []
    events = []
    for line in path.read_text(encoding="utf-8").splitlines():
        events.append(json.loads(line))
    return events


def source_stage() -> dict[str, Any]:
    report, arrays = _source()
    expected = {"D1_fit": (342, 342), "D1_score": (7462, 507),
                "D2_fit": (849, 849), "D2_score": (8491, 526)}
    checks = {}
    for prefix, n_missing in expected.items():
        rows = arrays[f"{prefix}_query_rows"]
        ids = arrays[f"{prefix}_query_ids"].astype(str)
        cycles = arrays[f"{prefix}_cycles"].astype(np.int64)
        missing = arrays[f"{prefix}_missing"].astype(bool)
        if len(rows) != expected[prefix][0] or len(np.unique(rows)) != len(rows) or len(np.unique(ids)) != len(ids):
            raise VerifyError(f"Duplicate row identity: {prefix}")
        if not np.array_equal(rows, np.sort(rows)):
            raise VerifyError(f"Physical row order invalid: {prefix}")
        if prefix.endswith("fit") and not missing.all():
            raise VerifyError(f"Fit pool includes present rows: {prefix}")
        if int(missing.sum()) != expected[prefix][1]:
            raise VerifyError(f"Missing support differs: {prefix}")
        if (not np.isfinite(arrays[f"{prefix}_baseline"].astype(np.float64)).all()
                or (prefix.endswith("score")
                    and not np.isfinite(arrays[f"{prefix}_champion_hinge"].astype(np.float64)).all())):
            raise VerifyError(f"Nonfinite source vector: {prefix}")
        checks[prefix] = {"rows": len(rows), "missing_rows": int(missing.sum()),
                          "identity_sha256": report["role_sources"][prefix]["identity_sha256"]}
    rows = np.concatenate([arrays["D1_score_query_rows"], arrays["D2_score_query_rows"]])
    if len(rows) != EXPECTED_POOLED_ROWS or len(np.unique(rows)) != EXPECTED_POOLED_ROWS:
        raise VerifyError("Source score rows are not unique 15,953")
    body = {"schema_version": 1, "experiment_id": LANE, "status": "PASS",
            "source_signature": report["source_signature"], "source_archive_sha256": _sha_file(SOURCE_ARCHIVE),
            "target_values_read": False, "score_labels_read": False,
            "test_or_sample_accessed": False, "role_checks": checks,
            "pooled_score_rows": len(rows), "pooled_unique_rows": len(np.unique(rows))}
    out = REPORT_ROOT / "SOURCE_QA_R1.json"
    if out.exists() and out.read_bytes() != _canonical({**body, "qa_signature": _canonical_sha(body)}) + b"\n":
        raise VerifyError("Refusing to overwrite changed source QA")
    result = dict(body); result["qa_signature"] = _canonical_sha(result)
    payload = _canonical(result) + b"\n"
    out.parent.mkdir(parents=True, exist_ok=True)
    if not out.exists(): out.write_bytes(payload)
    return {"status": result["status"], "qa_signature": result["qa_signature"],
            "source_signature": result["source_signature"], "score_labels_read": False}


def prelabel_stage() -> dict[str, Any]:
    source, arrays = _source()
    manifest_path = ARTIFACT_ROOT / "FIT_APPLICATION_R1.json"
    manifest = _read_json(manifest_path); _verify_signed(manifest, "manifest_signature", "fit/application manifest")
    if (manifest.get("status") != "FIT_APPLICATION_COMPLETE"
            or manifest.get("source_signature") != source.get("source_signature")
            or manifest.get("operation_budget") != {"fit_started": 2, "fit_saved": 2,
                "application_started": 2, "application_saved": 2, "learner_fits": 0,
                "preprocessor_fits": 0, "position_fits": 0, "hinge_fits": 0}):
        raise VerifyError("Fit/application manifest budget or source binding failed")
    events = _ledger()
    if any(e.get("event") == "OPERATION_FAILED" for e in events):
        raise VerifyError("Lane ledger contains a preserved operation failure")
    expected_events = [
        ("FIT_D1_R1", "OPERATION_STARTED", "missing_only_median_y_minus_R06"),
        ("FIT_D1_R1", "OPERATION_SAVED", "missing_only_median_y_minus_R06"),
        ("FIT_D2_R1", "OPERATION_STARTED", "missing_only_median_y_minus_R06"),
        ("FIT_D2_R1", "OPERATION_SAVED", "missing_only_median_y_minus_R06"),
        ("APPLY_D1_R1", "OPERATION_STARTED", "deterministic_missing_only_offset_application"),
        ("APPLY_D1_R1", "OPERATION_SAVED", "deterministic_missing_only_offset_application"),
        ("APPLY_D2_R1", "OPERATION_STARTED", "deterministic_missing_only_offset_application"),
        ("APPLY_D2_R1", "OPERATION_SAVED", "deterministic_missing_only_offset_application"),
    ]
    actual_events = [(e.get("operation_id"), e.get("event"), e.get("operation_type")) for e in events]
    if actual_events != expected_events:
        raise VerifyError("Lane ledger operation order or exact start/save count differs")
    if any(e.get("schema_version") != 1 for e in events):
        raise VerifyError("Unexpected lane ledger schema before scoring")
    if any(e.get("operation_id") == "SCORE_R1" for e in events):
        raise VerifyError("Score labels were accessed before prelabel QA")
    saved_hashes = {
        "FIT_D1_R1": _sha_file(ARTIFACT_ROOT / "scale_states" / "D1_R1.json"),
        "FIT_D2_R1": _sha_file(ARTIFACT_ROOT / "scale_states" / "D2_R1.json"),
        "APPLY_D1_R1": _sha_file(ARTIFACT_ROOT / "predictions" / "D1_R1.json"),
        "APPLY_D2_R1": _sha_file(ARTIFACT_ROOT / "predictions" / "D2_R1.json"),
    }
    for event in events:
        if event["event"] == "OPERATION_SAVED" and event.get("artifact_sha256") != saved_hashes.get(
                event.get("operation_id")):
            raise VerifyError(f"Lane ledger saved artifact hash mismatch: {event.get('operation_id')}")
    checks = {}; all_rows = []; all_ids = []
    for name in ("D1", "D2"):
        state, receipt, pred = _load_output(name)
        fit_prefix = f"{name}_fit"; score_prefix = f"{name}_score"
        fit_rows = arrays[f"{fit_prefix}_query_rows"].astype(np.int64)
        fit_ids = arrays[f"{fit_prefix}_query_ids"].astype(str)
        fit_cycles = arrays[f"{fit_prefix}_cycles"].astype(np.int64)
        fit_base = arrays[f"{fit_prefix}_baseline"].astype(np.float64)
        score_rows = arrays[f"{score_prefix}_query_rows"].astype(np.int64)
        score_ids = arrays[f"{score_prefix}_query_ids"].astype(str)
        score_cycles = arrays[f"{score_prefix}_cycles"].astype(np.int64)
        score_missing = arrays[f"{score_prefix}_missing"].astype(bool)
        baseline = arrays[f"{score_prefix}_baseline"].astype(np.float64)
        champion = arrays[f"{score_prefix}_champion_hinge"].astype(np.float64)
        candidate = pred["candidate_hinge"].astype(np.float64)
        delta = float(state["delta"])
        if (state.get("fit_rows") != len(fit_rows) or state.get("selection_rows_sha256") != _sha_array(fit_rows)
                or state.get("selection_ids_sha256") != _sha_array(fit_ids)
                or state.get("baseline_sha256") != _sha_array(fit_base)
                or state.get("selection_cycles") != list([68, 71] if name == "D1" else [68, 75])
                or not np.isfinite(delta)):
            raise VerifyError(f"Selection state does not match frozen missing rows: {name}")
        if (not np.array_equal(pred["query_rows"], score_rows)
                or not np.array_equal(pred["query_ids"].astype(str), score_ids)
                or not np.array_equal(pred["cycles"], score_cycles)
                or not np.array_equal(pred["missing"].astype(bool), score_missing)
                or not np.array_equal(pred["baseline"].view("<u8"), baseline.view("<u8"))
                or not np.array_equal(pred["champion_hinge"].view("<u8"), champion.view("<u8"))):
            raise VerifyError(f"Prediction source identity or parent vector differs: {name}")
        expected = champion.copy(); expected[score_missing] = np.clip(baseline[score_missing] + delta, 0.0, 100.0)
        if not np.array_equal(candidate, expected):
            raise VerifyError(f"Missing-only offset formula/readback failed: {name}")
        if not np.array_equal(candidate[~score_missing].view("<u8"), champion[~score_missing].view("<u8")):
            raise VerifyError(f"Present full-champion bits changed: {name}")
        if int(score_missing.sum()) < MIN_SUPPORT or len(fit_rows) < MIN_SUPPORT:
            raise VerifyError(f"Missing support below 100: {name}")
        all_rows.extend(score_rows.tolist()); all_ids.extend(score_ids.tolist())
        checks[name] = {"fit_rows": len(fit_rows), "score_rows": len(score_rows),
                        "score_missing_rows": int(score_missing.sum()), "delta": delta,
                        "present_bit_exact": True, "formula_readback": True,
                        "prediction_receipt_sha256": receipt["prediction"]["sha256"]}
    if len(all_rows) != EXPECTED_POOLED_ROWS or len(set(all_rows)) != EXPECTED_POOLED_ROWS or len(set(all_ids)) != EXPECTED_POOLED_ROWS:
        raise VerifyError("Fold score rows are not a unique pooled population")
    body = {"schema_version": 1, "experiment_id": LANE, "status": "PASS",
            "source_signature": source["source_signature"],
            "fit_application_manifest_sha256": _sha_file(manifest_path),
            "checks": checks, "pooled_unique_score_rows": len(all_rows),
            "selection_labels_accessed": True, "score_labels_read": False,
            "target_values_persisted": False, "test_or_sample_accessed": False,
            "kaggle_or_upload": False}
    result = dict(body); result["qa_signature"] = _canonical_sha(result)
    payload = _canonical(result) + b"\n"
    if PRELABEL_QA.exists() and PRELABEL_QA.read_bytes() != payload:
        raise VerifyError("Refusing to overwrite changed prelabel QA")
    PRELABEL_QA.parent.mkdir(parents=True, exist_ok=True)
    if not PRELABEL_QA.exists(): PRELABEL_QA.write_bytes(payload)
    return {"status": result["status"], "qa_signature": result["qa_signature"],
            "score_labels_read": False, "pooled_unique_score_rows": len(all_rows)}


def _metric(y: np.ndarray, p: np.ndarray) -> dict[str, Any]:
    err = np.asarray(p, dtype=np.float64) - np.asarray(y, dtype=np.float64)
    residual = -err
    return {"rows": len(err), "mae": float(np.mean(np.abs(err))),
            "p90_absolute_error": float(np.quantile(np.abs(err), 0.90, method="linear")),
            "median_residual_y_minus_prediction": float(np.median(residual)),
            "mean_residual_y_minus_prediction": float(np.mean(residual)),
            "bias_prediction_minus_target": float(np.mean(err))}


def _groups(y: np.ndarray, arrays: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    missing = arrays["missing"].astype(bool)
    out = {"missing": missing, "present": ~missing, "target_lt5": y < 5.0, "target_ge95": y >= 95.0,
           "EF0": arrays["entry_friction"].astype(np.float64) == 0.0,
           "EF1": arrays["entry_friction"].astype(np.float64) > 0.0}
    for key, label in (("impact_quartile", "impact_load"), ("text_quartile", "text_load")):
        values = arrays[key].astype(np.int8)
        for i in range(4): out[f"{label}_Q{i+1}"] = values == i
    return out


def _group_report(y: np.ndarray, base: np.ndarray, candidate: np.ndarray,
                  groups: Mapping[str, np.ndarray]) -> dict[str, Any]:
    out = {}
    for name, mask0 in groups.items():
        mask = np.asarray(mask0, dtype=bool); n = int(mask.sum())
        if n < MIN_SUPPORT:
            out[name] = {"status": "UNKNOWN", "rows": n, "champion": None,
                         "candidate": None, "harm_candidate_minus_champion": None}
        else:
            bm, cm = _metric(y[mask], base[mask]), _metric(y[mask], candidate[mask])
            out[name] = {"status": "MEASURED", "rows": n, "champion": bm, "candidate": cm,
                         "harm_candidate_minus_champion": cm["mae"] - bm["mae"]}
    return out


def _same_tree(left: Any, right: Any, tol: float = 1e-12) -> bool:
    if isinstance(left, dict) and isinstance(right, dict):
        return set(left) == set(right) and all(_same_tree(left[k], right[k], tol) for k in left)
    if isinstance(left, list) and isinstance(right, list):
        return len(left) == len(right) and all(_same_tree(a, b, tol) for a, b in zip(left, right))
    if isinstance(left, (int, float)) and isinstance(right, (int, float)) and not isinstance(left, bool) and not isinstance(right, bool):
        return abs(float(left) - float(right)) <= tol
    return left == right


def _read_targets(rows: np.ndarray, ids: np.ndarray) -> np.ndarray:
    gate_spec = importlib.util.spec_from_file_location("metric_parent_gate", PILOT_GATE)
    if gate_spec is None or gate_spec.loader is None: raise VerifyError("Cannot load parent gate helper")
    gate = importlib.util.module_from_spec(gate_spec); sys.modules[gate_spec.name] = gate; gate_spec.loader.exec_module(gate)
    bag2, runner = gate._helpers()
    train = runner._contained_file(bag2.SOURCE_ROOT, runner.TRAIN_REL)
    source = _read_json(SOURCE_REPORT)
    if _sha_file(train) != source["parent_official_train_sha256"]:
        raise VerifyError("Official Train hash changed before metric QA")
    y = np.asarray(runner._read_targets_for_rows(train, np.asarray(rows, dtype=np.int64),
                                                 np.asarray(ids, dtype=str), expected_total_rows=50608),
                   dtype=np.float64)
    if y.shape != np.asarray(rows).shape or not np.isfinite(y).all():
        raise VerifyError("Independent score target readback invalid")
    return y


def metric_stage() -> dict[str, Any]:
    source, _ = _source()
    pre = _read_json(PRELABEL_QA); _verify_signed(pre, "qa_signature", "prelabel QA")
    if pre.get("status") != "PASS" or pre.get("score_labels_read") is not False:
        raise VerifyError("Independent metric QA blocked before prelabel PASS")
    score = _read_json(SCORE_REPORT); _verify_signed(score, "score_signature", "score report")
    if (score.get("status") != "SCORED" or score.get("source_signature") != source.get("source_signature")
            or score.get("prelabel_qa_sha256") != _sha_file(PRELABEL_QA)
            or score.get("independent_metric_qa_pass") is not False):
        raise VerifyError("Score report/source/prelabel binding failed")
    folds = {}; fold_arrays = {}; all_groups = []
    for name in ("D1", "D2"):
        receipt = _read_json(ARTIFACT_ROOT / "predictions" / f"{name}_R1.json")
        pred = _verify_npz(ROOT / receipt["prediction"]["path"], receipt["prediction"])
        y = _read_targets(pred["query_rows"], pred["query_ids"].astype(str))
        base = pred["champion_hinge"].astype(np.float64)
        candidate = pred["candidate_hinge"].astype(np.float64)
        missing = pred["missing"].astype(bool)
        base_all, cand_all = _metric(y, base), _metric(y, candidate)
        bm, cm = _metric(y[missing], base[missing]), _metric(y[missing], candidate[missing])
        groups = _group_report(y, base, candidate, _groups(y, pred))
        per_cycle = {}
        for cycle in np.unique(pred["cycles"].astype(np.int64)):
            cmask = pred["cycles"].astype(np.int64) == cycle
            mmask = cmask & missing; nmiss = int(mmask.sum())
            per_cycle[str(int(cycle))] = {"rows": int(cmask.sum()), "missing_rows": nmiss,
                "missing_status": "MEASURED" if nmiss >= MIN_SUPPORT else "UNKNOWN",
                "champion": _metric(y[cmask], base[cmask]), "candidate": _metric(y[cmask], candidate[cmask]),
                "overall_mae_gain_candidate_vs_champion": _metric(y[cmask], base[cmask])["mae"]-
                    _metric(y[cmask], candidate[cmask])["mae"],
                "missing_champion": _metric(y[mmask], base[mmask]) if nmiss >= MIN_SUPPORT else None,
                "missing_candidate": _metric(y[mmask], candidate[mmask]) if nmiss >= MIN_SUPPORT else None,
                "missing_mae_gain_candidate_vs_champion": (
                    _metric(y[mmask], base[mmask])["mae"]-_metric(y[mmask], candidate[mmask])["mae"]
                    if nmiss >= MIN_SUPPORT else None)}
        fold = {"origin": 67 if name == "D1" else 75, "score_cycles": [72,75] if name=="D1" else [76,79],
                "rows": len(y), "missing": {"rows": int(missing.sum()), "status": "MEASURED" if int(missing.sum())>=MIN_SUPPORT else "UNKNOWN",
                    "champion": bm if int(missing.sum())>=MIN_SUPPORT else None,
                    "candidate": cm if int(missing.sum())>=MIN_SUPPORT else None,
                    "mae_gain_candidate_vs_champion": bm["mae"]-cm["mae"] if int(missing.sum())>=MIN_SUPPORT else None,
                    "mae_harm_candidate_minus_champion": cm["mae"]-bm["mae"] if int(missing.sum())>=MIN_SUPPORT else None},
                "overall": {"champion": base_all, "candidate": cand_all,
                            "mae_gain_candidate_vs_champion": base_all["mae"]-cand_all["mae"]},
                "per_cycle": per_cycle, "fixed_groups": groups,
                "present_bit_exact": bool(np.array_equal(candidate[~missing].view("<u8"), base[~missing].view("<u8"))),
                "row_identity_sha256": _canonical_sha({"rows": _sha_array(pred["query_rows"]),
                    "ids": _sha_array(pred["query_ids"]), "cycles": _sha_array(pred["cycles"]) }),
                "score_labels_read": len(y)}
        if not fold["present_bit_exact"]: raise VerifyError(f"Present predictions changed in {name}")
        if not _same_tree(fold, score["folds"][name]):
            raise VerifyError(f"Independent fold metric replay differs: {name}")
        folds[name] = fold; fold_arrays[name] = (y, pred)

    ys = np.concatenate([fold_arrays[k][0] for k in ("D1", "D2")])
    pooled_rows = np.concatenate([fold_arrays[k][1]["query_rows"] for k in ("D1", "D2")])
    pooled_ids = np.concatenate([fold_arrays[k][1]["query_ids"] for k in ("D1", "D2")])
    bases = np.concatenate([fold_arrays[k][1]["champion_hinge"] for k in ("D1", "D2")]).astype(np.float64)
    candidates = np.concatenate([fold_arrays[k][1]["candidate_hinge"] for k in ("D1", "D2")]).astype(np.float64)
    missing = np.concatenate([fold_arrays[k][1]["missing"] for k in ("D1", "D2")]).astype(bool)
    meta = {key: np.concatenate([fold_arrays[k][1][key] for k in ("D1", "D2")])
            for key in ("entry_friction", "impact_quartile", "text_quartile")}
    pooled_overall = {"champion": _metric(ys, bases), "candidate": _metric(ys, candidates),
                      "mae_gain_candidate_vs_champion": _metric(ys, bases)["mae"]-_metric(ys, candidates)["mae"]}
    bm, cm = _metric(ys[missing], bases[missing]), _metric(ys[missing], candidates[missing])
    pooled_missing = {"rows": int(missing.sum()), "status": "MEASURED" if int(missing.sum())>=MIN_SUPPORT else "UNKNOWN",
        "champion": bm if int(missing.sum())>=MIN_SUPPORT else None,
        "candidate": cm if int(missing.sum())>=MIN_SUPPORT else None,
        "mae_gain_candidate_vs_champion": bm["mae"]-cm["mae"] if int(missing.sum())>=MIN_SUPPORT else None,
        "mae_harm_candidate_minus_champion": cm["mae"]-bm["mae"] if int(missing.sum())>=MIN_SUPPORT else None}
    pooled_groups = _group_report(ys, bases, candidates, _groups(ys, missing, meta))
    pooled_identity = _canonical_sha({"rows": _sha_array(pooled_rows), "ids": _sha_array(pooled_ids)})
    if (len(ys) != EXPECTED_POOLED_ROWS or len(np.unique(pooled_rows)) != EXPECTED_POOLED_ROWS
            or len(np.unique(pooled_ids.astype(str))) != EXPECTED_POOLED_ROWS
            or score.get("pooled_rows") != EXPECTED_POOLED_ROWS
            or score.get("pooled_row_identity_sha256") != pooled_identity
            or not _same_tree(pooled_overall, score["pooled_overall"])
            or not _same_tree(pooled_missing, score["pooled_missing"])
            or not _same_tree(pooled_groups, score["pooled_fixed_groups"])):
        raise VerifyError("Independent pooled metric/group replay differs")
    harms = [v["harm_candidate_minus_champion"] for v in pooled_groups.values() if v["status"]=="MEASURED"]
    max_harm = max(harms, default=None)
    both_supported = all(folds[k]["missing"]["rows"] >= MIN_SUPPORT for k in ("D1","D2"))
    present_exact = all(folds[k]["present_bit_exact"] for k in ("D1","D2"))
    pooled_gain = pooled_overall["mae_gain_candidate_vs_champion"]
    group_pass = max_harm is not None and max_harm <= 0.25
    scientific = bool(both_supported and folds["D1"]["missing"]["mae_gain_candidate_vs_champion"] > 0
        and folds["D2"]["missing"]["mae_gain_candidate_vs_champion"] > 0 and pooled_gain > 0 and group_pass and present_exact)
    exploratory = bool(both_supported and pooled_gain > 0
        and all(folds[k]["missing"]["mae_harm_candidate_minus_champion"] <= 0.10 for k in ("D1","D2"))
        and group_pass and present_exact)
    if (score.get("maximum_supported_group_harm") != max_harm
            or score.get("both_fold_missing_support_pass") != both_supported
            or score.get("present_predictions_bit_exact") != present_exact
            or score.get("scientific_gate_without_independent_metric_qa") != scientific
            or score.get("exploratory_gate_without_independent_metric_qa") != exploratory):
        raise VerifyError("Independent gate reconstruction differs")
    body = {"schema_version": 1, "experiment_id": LANE, "status": "PASS",
            "source_signature": source["source_signature"], "score_signature": score["score_signature"],
            "score_sha256": _sha_file(SCORE_REPORT), "prelabel_qa_sha256": _sha_file(PRELABEL_QA),
            "independent_metric_replay": "PASS", "independent_target_rows_read": len(ys),
            "target_values_persisted": False, "test_or_sample_accessed": False,
            "pooled_mae_gain": pooled_gain, "D1_missing_gain": folds["D1"]["missing"]["mae_gain_candidate_vs_champion"],
            "D2_missing_gain": folds["D2"]["missing"]["mae_gain_candidate_vs_champion"],
            "maximum_supported_group_harm": max_harm,
            "present_predictions_bit_exact": present_exact,
            "scientific_qualification": scientific,
            "exploratory_benchmark_eligible": exploratory}
    result = dict(body); result["qa_signature"] = _canonical_sha(result)
    payload = _canonical(result) + b"\n"
    if METRIC_QA.exists() and METRIC_QA.read_bytes() != payload:
        raise VerifyError("Refusing to overwrite changed metric QA")
    METRIC_QA.parent.mkdir(parents=True, exist_ok=True)
    if not METRIC_QA.exists(): METRIC_QA.write_bytes(payload)
    return {"status": result["status"], "qa_signature": result["qa_signature"],
            "pooled_mae_gain": pooled_gain, "scientific_qualification": scientific,
            "exploratory_benchmark_eligible": exploratory}


def final_source_stage() -> dict[str, Any]:
    score = _read_json(SCORE_REPORT); _verify_signed(score, "score_signature", "score report")
    qa = _read_json(METRIC_QA); _verify_signed(qa, "qa_signature", "metric QA")
    report = _read_json(FINAL_SOURCE_REPORT); _verify_signed(report, "source_signature", "final source report")
    arrays = _verify_npz(FINAL_SOURCE_ARCHIVE, report["archive"])
    parent, parent_arrays = _parent_receipts()
    e67, e75 = parent_arrays[67], parent_arrays[75]
    keep67 = (np.asarray(e67["cycles"], dtype=np.int64) >= 68) & (np.asarray(e67["cycles"], dtype=np.int64) <= 75)
    keep75 = (np.asarray(e75["cycles"], dtype=np.int64) >= 76) & (np.asarray(e75["cycles"], dtype=np.int64) <= 79)
    expected_arrays = {key: np.concatenate([e67[key][keep67], e75[key][keep75]])
                       for key in ("query_rows", "query_ids", "cycles", "missing", "baseline", "champion_hinge")}
    order = np.argsort(expected_arrays["query_rows"], kind="stable")
    expected_arrays = {key: value[order] for key, value in expected_arrays.items()}
    if report.get("parent_source_signature") != parent.get("source_signature"):
        raise VerifyError("Final source parent signature mismatch")
    if set(arrays) != set(expected_arrays) or any(
            _sha_array(arrays[key]) != _sha_array(expected_arrays[key]) for key in expected_arrays):
        raise VerifyError("Final source archive differs from exact target-free parent OOF rows")
    rows = arrays["query_rows"].astype(np.int64); ids = arrays["query_ids"].astype(str)
    cycles = arrays["cycles"].astype(np.int64)
    if (report.get("status") != "FINAL_SOURCE_READY_PENDING_ROOT_REVIEW"
            or report.get("score_signature") != score.get("score_signature")
            or report.get("metric_qa_signature") != qa.get("qa_signature")
            or qa.get("exploratory_benchmark_eligible") is not True
            or len(rows) != EXPECTED_POOLED_ROWS or len(np.unique(rows)) != EXPECTED_POOLED_ROWS
            or len(np.unique(ids)) != EXPECTED_POOLED_ROWS
            or (int(cycles.min()), int(cycles.max())) != (68,79)
            or report.get("final_scalar_fit_started") is not False
            or report.get("E79_application_started") is not False):
        raise VerifyError("Final source package check failed")
    return {"status": "PASS", "source_signature": report["source_signature"],
            "archive_sha256": report["archive"]["sha256"], "rows": len(rows),
            "final_scalar_fit_started": False, "E79_application_started": False}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--source", action="store_true")
    group.add_argument("--prelabel", action="store_true")
    group.add_argument("--metric", action="store_true")
    group.add_argument("--final-source", action="store_true")
    args = parser.parse_args(argv)
    try:
        result = source_stage() if args.source else prelabel_stage() if args.prelabel else metric_stage() if args.metric else final_source_stage()
    except Exception as exc:
        print(json.dumps({"status": "FAIL", "error": f"{type(exc).__name__}: {exc}"}, sort_keys=True))
        return 1
    print(json.dumps(result, sort_keys=True, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
