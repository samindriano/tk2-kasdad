"""Frozen missing-only monotone affine L1 calibration pilot, R1."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
import time
from typing import Any, Mapping
import zipfile

import numpy as np
import psutil
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
LANE = "REG_MISSING_AFFINE_L1_V1"
REPORT_ROOT = ROOT / "reports" / "optimization_v3" / LANE
ARTIFACT_ROOT = ROOT / "experiments" / "runs" / LANE
SOURCE_REPORT = REPORT_ROOT / "SOURCE_FREEZE_R1.json"
SOURCE_QA = REPORT_ROOT / "SOURCE_QA_R1.json"
PRELABEL_QA = REPORT_ROOT / "PRELABEL_QA_R1.json"
SCORE_REPORT = REPORT_ROOT / "SCORE_R1.json"
METRIC_QA = REPORT_ROOT / "METRIC_QA_R1.json"
FINAL_SOURCE_REPORT = REPORT_ROOT / "FINAL_SOURCE_R1.json"
FINAL_SOURCE_ARCHIVE = ARTIFACT_ROOT / "FINAL_SOURCE_INPUTS_R1.npz"
RECIPE_PATH = REPORT_ROOT / "RECIPE_R1.json"
ROOT_PLAN_PATH = REPORT_ROOT / "ROOT_PLAN.md"
PARENT_LANE = "REG_MISSING_MEDIAN_OFFSET_V1"
PARENT_REPORT_ROOT = ROOT / "reports" / "optimization_v3" / PARENT_LANE
PARENT_ARTIFACT_ROOT = ROOT / "experiments" / "runs" / PARENT_LANE
PARENT_VERIFIER = ROOT / "scripts" / "verify_reg_missing_median_offset_v1.py"
PARENT_RUNNER = ROOT / "scripts" / "reg_missing_median_offset_v1.py"
PARENT_METRIC_VERIFIER = ROOT / "scripts" / "verify_reg_missing_median_offset_metric_r2.py"
PARENT_GUARD_VERIFIER = ROOT / "scripts" / "verify_reg_missing_median_offset_guard_r3.py"
PARENT_GUARD_REPORT = PARENT_REPORT_ROOT / "GROUP_GUARD_SCOPE_QA_R3.json"
PARENT_FREEZE = PARENT_REPORT_ROOT / "SOURCE_FREEZE_R1.json"
PARENT_SOURCE_ARCHIVE = PARENT_ARTIFACT_ROOT / "SOURCE_INPUTS_R1.npz"
PARENT_SOURCE_QA = PARENT_REPORT_ROOT / "SOURCE_QA_R1.json"
PARENT_SCORE = PARENT_REPORT_ROOT / "SCORE_R1.json"
PARENT_METRIC_QA = PARENT_REPORT_ROOT / "METRIC_QA_R2.json"
PARENT_PRELABEL_QA = PARENT_REPORT_ROOT / "PRELABEL_QA_R1.json"
R17_FREEZE = ROOT / "experiments" / "optimization_v3" / "batch1" / "R17" / "freeze.json"
R17_SUMMARY = ROOT / "reports" / "optimization_v3" / "batch1" / "R17" / "summary.md"
R17_RUNNER = ROOT / "scripts" / "optimization_v3_batch1_r17.py"
R17_SOLVER = ROOT / "src" / "models" / "optimization_v3_batch1_r17.py"
R18_FREEZE = ROOT / "experiments" / "optimization_v3" / "batch1" / "R18" / "freeze.json"
R18_SUMMARY = ROOT / "reports" / "optimization_v3" / "batch1" / "R18" / "summary.md"
R19_RECIPE = ROOT / "experiments" / "optimization_v3" / "batch1" / "R19" / "revision_02" / "recipe.json"
R19_SUMMARY = ROOT / "reports" / "optimization_v3" / "batch1" / "R19" / "summary.md"
GLOBAL_AFFINE_REVIEW = ROOT / "reports" / "optimization_v3" / "REG_FINAL_WAVE_FOLLOWUP_V2" / "SOL_DIAGNOSTIC_DECISION.md"
PRESENT_ONLY_PROTOCOL = ROOT / "reports" / "optimization_v3" / "REG_FINAL_WAVE_FOLLOWUP_V2_B_ONLY" / "PROTOCOL.md"
MIN_SUPPORT = 100
MIN_DISTINCT_BASELINES = 20
MIN_BASELINE_IQR = 5.0
EXPECTED_POOLED_ROWS = 15953
MEMORY_START = 512 * 1024 * 1024
MEMORY_RESERVE = 256 * 1024 * 1024
THREADS = 2
FIT_WINDOWS = {"D1": (68, 71), "D2": (68, 75)}
SCORE_WINDOWS = {"D1": (72, 75), "D2": (76, 79)}
OP_TYPES = {
    "FIT_D1_R1": "missing_only_monotone_affine_l1",
    "FIT_D2_R1": "missing_only_monotone_affine_l1",
    "APPLY_D1_R1": "deterministic_missing_only_affine_application",
    "APPLY_D2_R1": "deterministic_missing_only_affine_application",
    "SCORE_R1": "single_fullchamp_matched_score_pass",
}


class PilotError(RuntimeError):
    pass


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(v) for v in value]
    return value


def _canonical(value: Any) -> bytes:
    return json.dumps(_jsonable(value), sort_keys=True, ensure_ascii=False, allow_nan=False,
                      separators=(",", ":")).encode("utf-8", errors="surrogatepass")


def _canonical_sha(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _sha_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _sha_array(value: Any) -> str:
    a = np.asarray(value)
    head = _canonical({"dtype": a.dtype.str, "shape": list(a.shape)})
    if a.dtype.kind in "OUS":
        body = bytearray()
        for item in a.reshape(-1).tolist():
            raw = str(item).encode("utf-8", errors="surrogatepass")
            body.extend(len(raw).to_bytes(8, "little")); body.extend(raw)
        payload = bytes(body)
    else:
        payload = np.ascontiguousarray(a).tobytes()
    return hashlib.sha256(len(head).to_bytes(8, "little") + head + payload).hexdigest()


def _signed(body: Mapping[str, Any], key: str) -> dict[str, Any]:
    result = dict(body)
    result[key] = _canonical_sha(result)
    return result


def _verify_signed(value: Mapping[str, Any], key: str, label: str) -> None:
    body = {k: v for k, v in value.items() if k != key}
    if value.get(key) != _canonical_sha(body):
        raise PilotError(f"Signature mismatch: {label}")


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise PilotError(f"Invalid JSON: {path}") from exc
    if not isinstance(value, dict):
        raise PilotError(f"Expected JSON object: {path}")
    return value


def _write_json_new(path: Path, body: Mapping[str, Any], key: str) -> dict[str, Any]:
    result = _signed(body, key)
    payload = _canonical(result) + b"\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != payload:
            raise PilotError(f"Refusing to overwrite changed immutable output: {path}")
        return result
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("xb") as stream:
        stream.write(payload); stream.flush(); os.fsync(stream.fileno())
    os.replace(temp, path)
    return result


def _read_npz_exact(path: Path, keys: tuple[str, ...]) -> dict[str, np.ndarray]:
    if not path.is_file():
        raise PilotError(f"Required source file is missing: {path}")
    with zipfile.ZipFile(path) as archive:
        members = {item[:-4] for item in archive.namelist() if item.endswith(".npy")}
    if members != set(keys):
        raise PilotError(f"NPZ allowlist mismatch for {path}")
    with np.load(path, allow_pickle=False) as archive:
        return {key: np.asarray(archive[key]).copy() for key in keys}


def _npz_receipt(path: Path, arrays: Mapping[str, Any]) -> dict[str, Any]:
    try:
        relative_path = path.relative_to(ROOT).as_posix()
    except ValueError:
        relative_path = path.as_posix()
    return {"path": relative_path, "sha256": _sha_file(path),
            "members": {key: _sha_array(np.asarray(value)) for key, value in arrays.items()}}


def _write_npz_new(path: Path, arrays: Mapping[str, Any]) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    names = tuple(arrays)
    if path.exists():
        old = _read_npz_exact(path, names)
        for key, value in arrays.items():
            a, b = np.asarray(old[key]), np.asarray(value)
            same = (a.dtype == b.dtype and a.shape == b.shape and
                    (_sha_array(a) == _sha_array(b)))
            if not same:
                raise PilotError(f"Refusing to overwrite changed NPZ: {path}")
        return _npz_receipt(path, arrays)
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("xb") as stream:
        np.savez_compressed(stream, **{k: np.asarray(v) for k, v in arrays.items()})
        stream.flush(); os.fsync(stream.fileno())
    os.replace(temp, path)
    return _npz_receipt(path, arrays)


def _verify_npz(path: Path, receipt: Mapping[str, Any]) -> dict[str, np.ndarray]:
    if _sha_file(path) != receipt.get("sha256"):
        raise PilotError(f"NPZ archive hash mismatch: {path}")
    arrays = _read_npz_exact(path, tuple(receipt.get("members", {})))
    if {k: _sha_array(v) for k, v in arrays.items()} != receipt.get("members"):
        raise PilotError(f"NPZ member hash mismatch: {path}")
    return arrays


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise PilotError(f"Cannot import frozen helper: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _parent_context() -> tuple[Any, dict[str, Any], dict[str, np.ndarray], dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    base = _load_module(PARENT_VERIFIER, "verify_missing_median_parent_for_affine")
    source, source_arrays = base._source()
    _verify_signed(source, "source_signature", "parent median source")
    source_qa = _read_json(PARENT_SOURCE_QA); _verify_signed(source_qa, "qa_signature", "parent median source QA")
    score = _read_json(PARENT_SCORE); _verify_signed(score, "score_signature", "parent median score")
    metric = _read_json(PARENT_METRIC_QA); _verify_signed(metric, "qa_signature", "parent median metric QA")
    prelabel = _read_json(PARENT_PRELABEL_QA); _verify_signed(prelabel, "qa_signature", "parent median prelabel QA")
    guard = _read_json(PARENT_GUARD_REPORT); _verify_signed(guard, "addendum_signature", "parent all-view guard R3")
    if (source.get("status") != "SOURCE_FROZEN_PRELABEL" or source_qa.get("status") != "PASS"
            or source_qa.get("source_signature") != source.get("source_signature")
            or score.get("status") != "SCORED"
            or metric.get("status") != "PASS" or prelabel.get("status") != "PASS"
            or guard.get("status") != "PASS" or guard.get("qualification_status") != "FAIL"
            or source.get("source_signature") != score.get("source_signature")
            or score.get("score_signature") != metric.get("score_signature")
            or metric.get("score_sha256") != _sha_file(PARENT_SCORE)
            or prelabel.get("source_signature") != source.get("source_signature")
            or guard.get("source_inventory", {}).get("inputs", {}).get("score", {}).get("sha256")
                != _sha_file(PARENT_SCORE)
            or guard.get("all_view_group_guard_pass") is not False):
        raise PilotError("Parent signed score/QA/all-view guard lineage failed")
    outputs = {}
    for name in ("D1", "D2"):
        state, receipt, pred = base._load_output(name)
        prefix = f"{name}_score"
        for key in ("query_rows", "query_ids", "cycles", "missing", "baseline", "champion_hinge"):
            original = source_arrays[f"{prefix}_{key}"]
            saved = pred[key]
            if key in ("baseline", "champion_hinge"):
                if not np.array_equal(np.asarray(original, dtype=np.float64).view("<u8"),
                                      np.asarray(saved, dtype=np.float64).view("<u8")):
                    raise PilotError(f"Parent cached median output source differs: {name}.{key}")
            elif not np.array_equal(np.asarray(original).astype(str) if key == "query_ids" else np.asarray(original),
                                    np.asarray(saved).astype(str) if key == "query_ids" else np.asarray(saved)):
                raise PilotError(f"Parent cached median output identity differs: {name}.{key}")
        outputs[name] = {"state": state, "receipt": receipt, "prediction": pred}
    rows = np.concatenate([source_arrays[f"{name}_score_query_rows"] for name in ("D1", "D2")])
    ids = np.concatenate([source_arrays[f"{name}_score_query_ids"].astype(str) for name in ("D1", "D2")])
    if len(rows) != EXPECTED_POOLED_ROWS or len(np.unique(rows)) != EXPECTED_POOLED_ROWS or len(np.unique(ids)) != EXPECTED_POOLED_ROWS:
        raise PilotError("Parent source score identity is not the planned unique pooled population")
    return base, source, source_arrays, score, metric, prelabel, {"guard": guard, "outputs": outputs}


def _support_stats(x0: np.ndarray) -> dict[str, Any]:
    x = np.asarray(x0, dtype=np.float64)
    if len(x) == 0 or not np.isfinite(x).all():
        return {"rows": int(len(x)), "distinct_baselines": 0, "iqr_linear": None,
                "support_pass": False, "failure": "empty_or_nonfinite_baseline"}
    q25, q75 = np.quantile(x, [0.25, 0.75], method="linear")
    distinct = int(np.unique(x).size)
    iqr = float(q75 - q25)
    return {"rows": int(len(x)), "distinct_baselines": distinct, "iqr_linear": iqr,
            "minimum_rows": MIN_SUPPORT, "minimum_distinct_baselines": MIN_DISTINCT_BASELINES,
            "minimum_iqr": MIN_BASELINE_IQR,
            "support_pass": bool(len(x) >= MIN_SUPPORT and distinct >= MIN_DISTINCT_BASELINES
                                  and iqr >= MIN_BASELINE_IQR)}


def _source_code_hashes() -> dict[str, str]:
    paths = {
        "runner": Path(__file__),
        "independent_verifier": ROOT / "scripts" / "verify_reg_missing_affine_l1_v1.py",
        "regression_tests": ROOT / "tests" / "test_reg_missing_affine_l1_v1.py",
        "recipe": RECIPE_PATH,
        "root_plan": ROOT_PLAN_PATH,
        "parent_verifier": PARENT_VERIFIER,
        "parent_runner": PARENT_RUNNER,
        "parent_metric_verifier": PARENT_METRIC_VERIFIER,
        "parent_guard_verifier": PARENT_GUARD_VERIFIER,
        "parent_guard_report": PARENT_GUARD_REPORT,
        "r17_freeze": R17_FREEZE, "r17_summary": R17_SUMMARY,
        "r17_runner": R17_RUNNER, "r17_solver": R17_SOLVER,
        "r18_freeze": R18_FREEZE, "r18_summary": R18_SUMMARY,
        "r19_recipe": R19_RECIPE, "r19_summary": R19_SUMMARY,
        "global_affine_review": GLOBAL_AFFINE_REVIEW,
        "present_only_protocol": PRESENT_ONLY_PROTOCOL,
    }
    missing = [name for name, path in paths.items() if not path.is_file()]
    if missing:
        raise PilotError(f"Frozen source/history inputs missing: {missing}")
    return {name: _sha_file(path) for name, path in paths.items()}


def _history_review() -> dict[str, Any]:
    r17 = _read_json(R17_FREEZE)
    r18 = _read_json(R18_FREEZE)
    r19 = _read_json(R19_RECIPE)
    r17_identity = r17.get("identity", {})
    r17_primary = r17_identity.get("recipe", {})
    r17_status = r17.get("status", r17.get("run", {}).get("status"))
    history = r17_primary.get("history_policy", {})
    support = history.get("support_by_cutoff", {})
    r17_counts = {str(c): int(support.get(str(c), {}).get("rows", 0)) for c in (67, 75)}
    if r17_counts != {"67": 17685, "75": 30097}:
        raise PilotError(f"R17 full-support lineage changed: {r17_counts}")
    r18_policy = json.dumps(r18.get("identity", {}).get("recipe", {}), sort_keys=True).lower()
    r19_policy = json.dumps(r19, sort_keys=True).lower()
    if "0.5*median" not in r18_policy or "ef0" not in r19_policy or "ef1" not in r19_policy:
        raise PilotError("R18/R19 novelty references no longer match their frozen mechanisms")
    return {
        "exact_missing_only_monotone_affine_equivalent_found": False,
        "scope": "local TK2 code, frozen recipes/results, and optimization_v3 research backlog checked; no exact missing-only affine fit on R06 baseline found",
        "references": {
            "R17": {"freeze": R17_FREEZE.relative_to(ROOT).as_posix(),
                    "summary": R17_SUMMARY.relative_to(ROOT).as_posix(),
                    "implementation": R17_RUNNER.relative_to(ROOT).as_posix(),
                    "status": "RETIRE_LATE_ONLY_NO_EXTERNAL_PROMOTION",
                    "fit_population": "all exact champion prequential OOF rows from cycle 56 through cutoff; no missing-only filter",
                    "fit_rows_by_cutoff": r17_counts,
                    "minimum_rows": int(history.get("minimum_rows", 0)),
                    "two_solve_cutoffs": [67, 75],
                    "primary_gain": 0.032732, "W4_gain": 0.032532,
                    "W6_gain": 0.032849, "H12_L_gain": 0.031992,
                    "H12_E": "exact champion fallback"},
            "R18": {"freeze": R18_FREEZE.relative_to(ROOT).as_posix(),
                    "summary": R18_SUMMARY.relative_to(ROOT).as_posix(),
                    "mechanism": "global half-median residual offset capped at +/-2; not affine"},
            "R19": {"recipe": R19_RECIPE.relative_to(ROOT).as_posix(),
                    "summary": R19_SUMMARY.relative_to(ROOT).as_posix(),
                    "mechanism": "entry_friction-stratified half-median offsets; not affine"},
            "global_affine_followup": {"review": GLOBAL_AFFINE_REVIEW.relative_to(ROOT).as_posix(),
                                       "mechanism": "full-support/global affine policy; not fit on missing rows only"},
            "present_only_affine_bypass": {"protocol": PRESENT_ONLY_PROTOCOL.relative_to(ROOT).as_posix(),
                                           "mechanism": "existing full-support global affine on present rows and direct fallback restored on missing rows; no missing-only affine fit"},
            "missing_offset_R3": {"report": PARENT_GUARD_REPORT.relative_to(ROOT).as_posix(),
                                  "mechanism": "missing-only constant correction; no slope"},
        },
        "input_sha256": {
            "r17_freeze": _sha_file(R17_FREEZE), "r17_summary": _sha_file(R17_SUMMARY),
            "r17_solver": _sha_file(R17_SOLVER), "r18_freeze": _sha_file(R18_FREEZE),
            "r18_summary": _sha_file(R18_SUMMARY), "r19_recipe": _sha_file(R19_RECIPE),
            "r19_summary": _sha_file(R19_SUMMARY),
            "global_affine_review": _sha_file(GLOBAL_AFFINE_REVIEW),
            "present_only_protocol": _sha_file(PRESENT_ONLY_PROTOCOL),
            "missing_offset_guard_r3": _sha_file(PARENT_GUARD_REPORT),
        },
        "no_historical_result_relabelled": True,
    }


def _role_source(source_arrays: Mapping[str, np.ndarray], control: Mapping[str, Any], name: str) -> dict[str, dict[str, np.ndarray]]:
    fit_prefix, score_prefix = f"{name}_fit", f"{name}_score"
    fit = {key: np.asarray(source_arrays[f"{fit_prefix}_{key}"]).copy()
           for key in ("query_rows", "query_ids", "cycles", "missing", "baseline")}
    score = {key: np.asarray(source_arrays[f"{score_prefix}_{key}"]).copy()
             for key in ("query_rows", "query_ids", "cycles", "missing", "baseline", "champion_hinge",
                         "entry_friction", "impact_quartile", "text_quartile")}
    score["median_policy"] = np.asarray(control["prediction"]["candidate_hinge"], dtype=np.float64).copy()
    return {"fit": fit, "score": score}


def _source_stage() -> dict[str, Any]:
    if any(path.exists() for path in (SOURCE_REPORT, SOURCE_QA, PRELABEL_QA, SCORE_REPORT, METRIC_QA,
                                      ARTIFACT_ROOT / "FIT_APPLICATION_R1.json")):
        raise PilotError("Source cannot be re-prepared after immutable outputs exist")
    base, parent_source, source_arrays, parent_score, parent_metric, parent_pre, extra = _parent_context()
    parent_control = extra["outputs"]
    roles: dict[str, Any] = {}
    for name in ("D1", "D2"):
        role = _role_source(source_arrays, parent_control[name], name)
        fit, score = role["fit"], role["score"]
        fit_rows = np.asarray(fit["query_rows"], dtype=np.int64)
        fit_ids = np.asarray(fit["query_ids"], dtype=str)
        fit_cycles = np.asarray(fit["cycles"], dtype=np.int64)
        fit_missing = np.asarray(fit["missing"], dtype=bool)
        score_rows = np.asarray(score["query_rows"], dtype=np.int64)
        score_ids = np.asarray(score["query_ids"], dtype=str)
        score_cycles = np.asarray(score["cycles"], dtype=np.int64)
        score_missing = np.asarray(score["missing"], dtype=bool)
        if (not np.all(fit_missing) or len(np.unique(fit_rows)) != len(fit_rows)
                or len(np.unique(fit_ids)) != len(fit_ids) or not np.array_equal(fit_rows, np.sort(fit_rows))
                or not np.array_equal(score_rows, np.sort(score_rows))
                or len(np.unique(score_rows)) != len(score_rows) or len(np.unique(score_ids)) != len(score_ids)):
            raise PilotError(f"Frozen row identity/missing scope failed: {name}")
        if ((int(fit_cycles.min()), int(fit_cycles.max())) != FIT_WINDOWS[name]
                or (int(score_cycles.min()), int(score_cycles.max())) != SCORE_WINDOWS[name]):
            raise PilotError(f"Frozen cycle window changed: {name}")
        stats = _support_stats(fit["baseline"])
        score_missing_count = int(score_missing.sum())
        if not stats["support_pass"] or score_missing_count < MIN_SUPPORT:
            raise PilotError(f"Pre-fit support gate failed without reading labels: {name}")
        baseline = np.asarray(score["baseline"], dtype=np.float64)
        champion = np.asarray(score["champion_hinge"], dtype=np.float64)
        median = np.asarray(score["median_policy"], dtype=np.float64)
        if (not np.isfinite(baseline).all() or not np.isfinite(champion).all() or not np.isfinite(median).all()
                or not np.array_equal(champion[~score_missing].view("<u8"), median[~score_missing].view("<u8"))):
            raise PilotError(f"Parent prediction finite/present-control identity failed: {name}")
        roles[name] = {
            "selection": {"rows": len(fit_rows), "missing_rows": int(fit_missing.sum()),
                          "cycles": [int(fit_cycles.min()), int(fit_cycles.max())],
                          "rows_sha256": _sha_array(fit_rows), "ids_sha256": _sha_array(fit_ids),
                          "baseline_sha256": _sha_array(fit["baseline"]), "support": stats},
            "score": {"rows": len(score_rows), "missing_rows": score_missing_count,
                      "cycles": [int(score_cycles.min()), int(score_cycles.max())],
                      "rows_sha256": _sha_array(score_rows), "ids_sha256": _sha_array(score_ids),
                      "baseline_sha256": _sha_array(score["baseline"]),
                      "champion_sha256": _sha_array(score["champion_hinge"]),
                      "median_policy_sha256": _sha_array(score["median_policy"]),
                      "missing_sha256": _sha_array(score["missing"])},
        }
    score_rows = np.concatenate([source_arrays[f"{name}_score_query_rows"] for name in ("D1", "D2")])
    score_ids = np.concatenate([source_arrays[f"{name}_score_query_ids"].astype(str) for name in ("D1", "D2")])
    if (len(score_rows) != EXPECTED_POOLED_ROWS or len(np.unique(score_rows)) != EXPECTED_POOLED_ROWS
            or len(np.unique(score_ids)) != EXPECTED_POOLED_ROWS):
        raise PilotError("Pooled score identities are not unique 15,953 rows")
    history = _history_review()
    recipe = _read_json(RECIPE_PATH); _verify_signed(recipe, "recipe_signature", "affine recipe")
    train_hash = parent_source.get("parent_official_train_sha256")
    body = {
        "schema_version": 1, "experiment_id": LANE, "status": "SOURCE_FROZEN_PRELABEL",
        "recipe_sha256": _sha_file(RECIPE_PATH), "recipe_signature": recipe["recipe_signature"],
        "root_plan_sha256": _sha_file(ROOT_PLAN_PATH),
        "parent_lane": PARENT_LANE,
        "parent_source_signature": parent_source["source_signature"],
        "parent_source_freeze_sha256": _sha_file(PARENT_FREEZE),
        "parent_source_archive_sha256": _sha_file(PARENT_SOURCE_ARCHIVE),
        "parent_source_qa_sha256": _sha_file(PARENT_SOURCE_QA),
        "parent_score_sha256": _sha_file(PARENT_SCORE), "parent_score_signature": parent_score["score_signature"],
        "parent_metric_qa_sha256": _sha_file(PARENT_METRIC_QA), "parent_metric_qa_signature": parent_metric["qa_signature"],
        "parent_prelabel_qa_sha256": _sha_file(PARENT_PRELABEL_QA),
        "parent_all_view_guard_r3_sha256": _sha_file(PARENT_GUARD_REPORT),
        "parent_all_view_guard_r3_signature": extra["guard"]["addendum_signature"],
        "parent_official_train_sha256": train_hash,
        "parent_controls": {name: {"state_signature": parent_control[name]["state"]["state_signature"],
                                    "state_sha256": _sha_file(PARENT_ARTIFACT_ROOT / "scale_states" / f"{name}_R1.json"),
                                    "receipt_signature": parent_control[name]["receipt"]["receipt_signature"],
                                    "receipt_sha256": _sha_file(PARENT_ARTIFACT_ROOT / "predictions" / f"{name}_R1.json"),
                                    "prediction_sha256": parent_control[name]["receipt"]["prediction"]["sha256"]}
                           for name in ("D1", "D2")},
        "source_code_hashes": _source_code_hashes(),
        "role_sources": roles,
        "pooled_score_rows": EXPECTED_POOLED_ROWS,
        "support_gate": {"minimum_missing_fit_rows": MIN_SUPPORT,
                         "minimum_distinct_R06_baselines": MIN_DISTINCT_BASELINES,
                         "minimum_baseline_IQR": MIN_BASELINE_IQR,
                         "minimum_missing_score_rows": MIN_SUPPORT,
                         "fit_gate_pass": True, "score_gate_pass": True},
        "historical_novelty": history,
        "fit_application_budget": {"affine_fits": 2, "applications": 2, "base_fits": 0,
                                    "POS_fits": 0, "preprocessor_fits": 0, "hinge_fits": 0},
        "execution_boundary": {"selection_labels_read": False, "score_labels_read": False,
                               "target_values_persisted": False, "test_or_sample_accessed": False,
                               "kaggle_or_upload": False, "fit_started": False,
                               "application_started": False},
        "temporal_disclosure": "D2 selection E67 cycles 68-75 includes D1 score cycles 72-75; those targets are used only for D2 selection. D1/D2 score IDs are distinct but not independent validation and are reused development rows.",
        "adaptive_followup_disclosure": "This follows an exposed negative constant missing offset pilot with opposite D1/D2 fitted signs. It tests conditional affine slope only; it is not a fresh holdout and does not authorize sign, shrink, window, cap, penalty, or coefficient rescues.",
    }
    report = _write_json_new(SOURCE_REPORT, body, "source_signature")
    return {"status": report["status"], "source_signature": report["source_signature"],
            "D1_fit_rows": roles["D1"]["selection"]["rows"],
            "D2_fit_rows": roles["D2"]["selection"]["rows"],
            "D1_score_missing": roles["D1"]["score"]["missing_rows"],
            "D2_score_missing": roles["D2"]["score"]["missing_rows"],
            "pooled_score_rows": EXPECTED_POOLED_ROWS, "target_labels_read": False}


def _parent_arrays() -> tuple[Any, dict[str, Any], dict[str, np.ndarray], dict[str, Any], dict[str, Any]]:
    base, parent_source, arrays, _score, _metric, _prelabel, extra = _parent_context()
    role_arrays = {}
    for name in ("D1", "D2"):
        role_arrays[name] = _role_source(arrays, extra["outputs"][name], name)
    return base, parent_source, arrays, extra["outputs"], role_arrays


def _verified_source() -> tuple[dict[str, Any], Any, dict[str, Any], dict[str, dict[str, dict[str, np.ndarray]]]]:
    report = _read_json(SOURCE_REPORT); _verify_signed(report, "source_signature", "affine source freeze")
    recipe = _read_json(RECIPE_PATH); _verify_signed(recipe, "recipe_signature", "affine recipe")
    base, parent_source, _parent_arrays0, controls, role_arrays = _parent_arrays()
    if (report.get("status") != "SOURCE_FROZEN_PRELABEL"
            or report.get("recipe_sha256") != _sha_file(RECIPE_PATH)
            or report.get("root_plan_sha256") != _sha_file(ROOT_PLAN_PATH)
            or report.get("parent_source_signature") != parent_source.get("source_signature")
            or report.get("parent_source_freeze_sha256") != _sha_file(PARENT_FREEZE)
            or report.get("parent_source_archive_sha256") != _sha_file(PARENT_SOURCE_ARCHIVE)
            or report.get("parent_source_qa_sha256") != _sha_file(PARENT_SOURCE_QA)
            or report.get("parent_score_sha256") != _sha_file(PARENT_SCORE)
            or report.get("parent_metric_qa_sha256") != _sha_file(PARENT_METRIC_QA)
            or report.get("parent_prelabel_qa_sha256") != _sha_file(PARENT_PRELABEL_QA)
            or report.get("parent_all_view_guard_r3_sha256") != _sha_file(PARENT_GUARD_REPORT)
            or report.get("source_code_hashes") != _source_code_hashes()
            or report.get("parent_official_train_sha256") != parent_source.get("parent_official_train_sha256")):
        raise PilotError("Affine source freeze bindings failed")
    return report, base, controls, role_arrays


def _read_targets(rows: np.ndarray, ids: np.ndarray) -> np.ndarray:
    base = _load_module(PARENT_VERIFIER, "verify_median_parent_for_affine_target_read")
    y = np.asarray(base._read_targets(np.asarray(rows, dtype=np.int64), np.asarray(ids, dtype=str)), dtype=np.float64)
    if y.shape != np.asarray(rows).shape or not np.isfinite(y).all() or np.any((y < 0.0) | (y > 100.0)):
        raise PilotError("Authorized target readback is invalid")
    return y


def _solve_affine(x0: np.ndarray, y0: np.ndarray) -> dict[str, Any]:
    """One sparse HiGHS LP call through the frozen R17 solver implementation."""
    helper = _load_module(R17_SOLVER, "r17_sparse_affine_solver_for_missing_affine")
    x = np.asarray(x0, dtype=np.float64)
    y = np.asarray(y0, dtype=np.float64)
    if x.shape != y.shape or x.ndim != 1 or len(x) < MIN_SUPPORT or not np.isfinite(x).all() or not np.isfinite(y).all():
        raise PilotError("Affine LP input vectors are empty, invalid, or misaligned")
    started = time.perf_counter()
    with threadpool_limits(limits=THREADS):
        result = helper.solve_nonnegative_l1(y, x, threads=THREADS)
    elapsed = time.perf_counter() - started
    success = bool(result.get("success"))
    a, b = float(result.get("intercept", 0.0)), float(result.get("slope", 1.0))
    if success:
        residual = y - (a + b * x)
        objective = float(np.abs(residual).sum())
        raw_obj = result.get("objective_l1_sum")
        objective_ok = raw_obj is not None and np.isfinite(float(raw_obj)) and np.isclose(
            objective, float(raw_obj), atol=max(1e-5, len(y) * 1e-9), rtol=1e-8)
        # With u_i=|y_i-a-b*x_i|, both standard epigraph inequalities hold.
        residual_abs = np.abs(residual)
        primal_violation = float(max(0.0, -b,
                                    float(np.max(residual - residual_abs, initial=0.0)),
                                    float(np.max(-residual - residual_abs, initial=0.0)),
                                    float(np.max(-residual_abs, initial=0.0))))
        primal_ok = bool(objective_ok and a == a and b >= 0.0 and np.isfinite(a) and np.isfinite(b)
                         and primal_violation <= 1e-9)
        if not primal_ok:
            success = False
    else:
        objective = None
        objective_ok = False
        primal_violation = None
        primal_ok = False
        a, b = 0.0, 1.0
    return {
        "success": success,
        "solver": result.get("solver", "scipy.optimize.linprog(method='highs')"),
        "solver_status": result.get("solver_status"),
        "solver_message": result.get("solver_message"),
        "solver_iterations": result.get("iterations"),
        "solver_seconds": float(result.get("solver_seconds") or elapsed),
        "elapsed_seconds": float(elapsed),
        "intercept": float(a), "slope": float(b),
        "objective_l1_sum": objective,
        "reported_solver_objective_l1_sum": result.get("objective_l1_sum"),
        "objective_recomputed_matches_solver": bool(objective_ok),
        "primal_feasibility": "PASS" if primal_ok else "UNKNOWN_FALLBACK",
        "primal_max_violation": primal_violation,
        "constraint_shape": result.get("constraint_shape"),
        "constraint_nnz": result.get("constraint_nnz"),
        "constraint_csr_bytes": result.get("constraint_csr_bytes"),
        "fallback": not success,
        "fallback_reason": (result.get("fallback_reason") if not success else None),
    }


def _apply_affine(baseline: np.ndarray, champion: np.ndarray, missing: np.ndarray,
                  intercept: float, slope: float, *, fallback: bool = False) -> np.ndarray:
    x = np.asarray(baseline, dtype=np.float64)
    parent = np.asarray(champion, dtype=np.float64)
    mask = np.asarray(missing, dtype=bool)
    if x.shape != parent.shape or x.shape != mask.shape or not np.isfinite(x).all() or not np.isfinite(parent).all():
        raise PilotError("Affine application inputs do not align or contain nonfinite values")
    candidate = parent.copy()
    if not fallback:
        if not np.isfinite(intercept) or not np.isfinite(slope) or slope < 0.0:
            raise PilotError("Affine coefficients invalid")
        candidate[mask] = np.clip(intercept + slope * x[mask], 0.0, 100.0)
    if not np.array_equal(candidate[~mask].view("<u8"), parent[~mask].view("<u8")):
        raise PilotError("Present full-champion bits changed")
    return candidate


def _operation_events(root: Path = ARTIFACT_ROOT) -> list[dict[str, Any]]:
    path = root / "lane_ledger.jsonl"
    if not path.exists():
        return []
    events = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        try:
            item = json.loads(line)
        except Exception as exc:
            raise PilotError(f"Malformed operation ledger at line {line_number}") from exc
        if not isinstance(item, dict):
            raise PilotError(f"Malformed operation ledger record at line {line_number}")
        events.append(item)
    return events


def _append_event(root: Path, record: Mapping[str, Any]) -> None:
    root.mkdir(parents=True, exist_ok=True)
    with (root / "lane_ledger.jsonl").open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(_canonical(record).decode("utf-8") + "\n")
        stream.flush(); os.fsync(stream.fileno())


def _start_operation(op_id: str, root: Path = ARTIFACT_ROOT) -> None:
    if op_id not in OP_TYPES or any(event.get("operation_id") == op_id for event in _operation_events(root)):
        raise PilotError(f"Operation already started or invalid; retry forbidden: {op_id}")
    from datetime import datetime, timezone
    _append_event(root, {"schema_version": 1, "event": "OPERATION_STARTED", "operation_id": op_id,
                         "operation_type": OP_TYPES[op_id], "utc": datetime.now(timezone.utc).isoformat()})


def _finish_operation(op_id: str, artifact_sha: str, root: Path = ARTIFACT_ROOT) -> None:
    from datetime import datetime, timezone
    _append_event(root, {"schema_version": 1, "event": "OPERATION_SAVED", "operation_id": op_id,
                         "operation_type": OP_TYPES[op_id], "artifact_sha256": artifact_sha,
                         "utc": datetime.now(timezone.utc).isoformat()})


def _fail_operation(op_id: str, exc: Exception, root: Path = ARTIFACT_ROOT) -> None:
    from datetime import datetime, timezone
    _append_event(root, {"schema_version": 1, "event": "OPERATION_FAILED", "operation_id": op_id,
                         "error": f"{type(exc).__name__}: {exc}",
                         "utc": datetime.now(timezone.utc).isoformat()})


def _selection_identity(name: str, fit: Mapping[str, np.ndarray]) -> dict[str, Any]:
    rows = np.asarray(fit["query_rows"], dtype=np.int64)
    ids = np.asarray(fit["query_ids"], dtype=str)
    cycles = np.asarray(fit["cycles"], dtype=np.int64)
    baseline = np.asarray(fit["baseline"], dtype=np.float64)
    missing = np.asarray(fit["missing"], dtype=bool)
    if (len(rows) < MIN_SUPPORT or not missing.all() or len(np.unique(rows)) != len(rows)
            or len(np.unique(ids)) != len(ids) or not np.array_equal(rows, np.sort(rows))
            or not np.isfinite(baseline).all()):
        raise PilotError(f"Invalid missing-only selection rows for {name}")
    support = _support_stats(baseline)
    if not support["support_pass"]:
        raise PilotError(f"Pre-fit distinct-baseline/IQR gate failed for {name}")
    return {"rows": rows, "ids": ids, "cycles": cycles, "baseline": baseline,
            "missing": missing, "support": support}


def _fit_fold(name: str, fit: Mapping[str, np.ndarray], source: Mapping[str, Any],
              *, root: Path = ARTIFACT_ROOT, y_override: np.ndarray | None = None) -> dict[str, Any]:
    ident = _selection_identity(name, fit)
    state_path = root / "scale_states" / f"{name}_R1.json"
    if state_path.exists():
        raise PilotError(f"Prior affine state exists; retry forbidden: {name}")
    op_id = f"FIT_{name}_R1"
    _start_operation(op_id, root)
    try:
        y = (_read_targets(ident["rows"], ident["ids"]) if y_override is None
             else np.asarray(y_override, dtype=np.float64))
        if y.shape != ident["baseline"].shape or not np.isfinite(y).all() or np.any((y < 0) | (y > 100)):
            raise PilotError(f"Selection target vector invalid: {name}")
        result = _solve_affine(ident["baseline"], y)
        status = "FIT_COMPLETE" if result["success"] else "FIT_FAILED_FALLBACK"
        state_body = {
            "schema_version": 1, "experiment_id": LANE, "state_id": name, "status": status,
            "intercept": result["intercept"], "slope": result["slope"],
            "prediction_policy": "clip(intercept + slope * R06_baseline, 0, 100) on missing rows; exact saved full champion on present rows",
            "fit_rows": len(ident["rows"]), "missing_fit_rows": int(ident["missing"].sum()),
            "selection_cycles": list(FIT_WINDOWS[name]),
            "selection_rows_sha256": _sha_array(ident["rows"]),
            "selection_ids_sha256": _sha_array(ident["ids"]),
            "selection_cycles_sha256": _sha_array(ident["cycles"]),
            "baseline_sha256": _sha_array(ident["baseline"]),
            "selection_y_sha256": _sha_array(y),
            "selection_residual_sha256": _sha_array(y - (result["intercept"] + result["slope"] * ident["baseline"])),
            "support": ident["support"], "lp": result,
            "source_signature": source["source_signature"],
            "target_values_persisted": False, "labels_used": True,
            "labels_read_for": f"{name} missing-only calibration selection",
        }
        state = _write_json_new(state_path, state_body, "state_signature")
        _finish_operation(op_id, _sha_file(state_path), root)
        return state
    except Exception as exc:
        _fail_operation(op_id, exc, root)
        raise


def _apply_fold(name: str, score: Mapping[str, np.ndarray], state: Mapping[str, Any],
                source: Mapping[str, Any], *, root: Path = ARTIFACT_ROOT) -> dict[str, Any]:
    pred_path = root / "predictions" / f"{name}_R1.npz"
    if pred_path.exists() or pred_path.with_suffix(".json").exists():
        raise PilotError(f"Prior affine prediction exists; retry forbidden: {name}")
    rows = np.asarray(score["query_rows"], dtype=np.int64)
    ids = np.asarray(score["query_ids"], dtype=str)
    cycles = np.asarray(score["cycles"], dtype=np.int64)
    missing = np.asarray(score["missing"], dtype=bool)
    baseline = np.asarray(score["baseline"], dtype=np.float64)
    champion = np.asarray(score["champion_hinge"], dtype=np.float64)
    median = np.asarray(score["median_policy"], dtype=np.float64)
    if (int(missing.sum()) < MIN_SUPPORT or len(rows) != len(ids) or len(np.unique(rows)) != len(rows)
            or len(np.unique(ids)) != len(ids) or not np.array_equal(rows, np.sort(rows))):
        raise PilotError(f"Invalid score source identity or missing support: {name}")
    op_id = f"APPLY_{name}_R1"
    _start_operation(op_id, root)
    try:
        fallback = state.get("status") != "FIT_COMPLETE" or state.get("lp", {}).get("fallback") is True
        candidate = _apply_affine(baseline, champion, missing, float(state["intercept"]),
                                  float(state["slope"]), fallback=fallback)
        if not np.array_equal(median[~missing].view("<u8"), champion[~missing].view("<u8")):
            raise PilotError(f"Prior median control differs from present champion: {name}")
        arrays = {
            "query_rows": rows, "query_ids": ids, "cycles": cycles, "missing": missing,
            "baseline": baseline, "champion_hinge": champion, "median_policy": median,
            "candidate_hinge": candidate,
            "entry_friction": np.asarray(score["entry_friction"], dtype=np.float64),
            "impact_quartile": np.asarray(score["impact_quartile"], dtype=np.int8),
            "text_quartile": np.asarray(score["text_quartile"], dtype=np.int8),
        }
        receipt = _write_npz_new(pred_path, arrays)
        receipt_body = {"schema_version": 1, "experiment_id": LANE, "fold": name,
                        "status": "PREDICTION_SAVED", "prediction": receipt,
                        "state_signature": state["state_signature"],
                        "source_signature": source["source_signature"],
                        "rows": len(rows), "missing_rows": int(missing.sum()),
                        "present_exact": True, "fallback": fallback,
                        "target_values_persisted": False}
        receipt_path = pred_path.with_suffix(".json")
        receipt_doc = _write_json_new(receipt_path, receipt_body, "receipt_signature")
        _finish_operation(op_id, _sha_file(receipt_path), root)
        return {"receipt_signature": receipt_doc["receipt_signature"],
                "receipt_sha256": _sha_file(receipt_path), "prediction_sha256": receipt["sha256"],
                "present_exact": True, "fallback": fallback}
    except Exception as exc:
        _fail_operation(op_id, exc, root)
        raise


def _fit_apply_stage() -> dict[str, Any]:
    source, _base, _controls, roles = _verified_source()
    if (not SOURCE_QA.exists() or (ARTIFACT_ROOT / "FIT_APPLICATION_R1.json").exists()
            or (ARTIFACT_ROOT / "lane_ledger.jsonl").exists()):
        raise PilotError("Independent source QA required or fit/apply outputs already exist; retry forbidden")
    qa = _read_json(SOURCE_QA); _verify_signed(qa, "qa_signature", "independent source QA")
    if qa.get("status") != "PASS" or qa.get("source_signature") != source["source_signature"]:
        raise PilotError("Affine target access blocked until matching source QA PASS")
    support_pass = all(roles[name]["fit"]["support"]["support_pass"]
                       and int(np.asarray(roles[name]["score"]["missing"], dtype=bool).sum()) >= MIN_SUPPORT
                       for name in ("D1", "D2"))
    if not support_pass:
        raise PilotError("Frozen pre-fit support gate is false; labels and fit operations remain closed")
    available = int(psutil.virtual_memory().available)
    required = MEMORY_START + MEMORY_RESERVE
    if available < required:
        raise PilotError(f"RAM gate blocked before fit: available={available}, required={required}")
    train_hash = source["parent_official_train_sha256"]
    base = _load_module(PARENT_VERIFIER, "verify_median_source_for_affine_fit")
    parent_source, _ = base._source()
    if parent_source.get("parent_official_train_sha256") != train_hash:
        raise PilotError("Official Train source hash changed after freeze")
    with threadpool_limits(limits=THREADS):
        states = {name: _fit_fold(name, roles[name]["fit"], source) for name in ("D1", "D2")}
        applications = {name: _apply_fold(name, roles[name]["score"], states[name], source)
                        for name in ("D1", "D2")}
    body = {
        "schema_version": 1, "experiment_id": LANE, "status": "FIT_APPLICATION_COMPLETE",
        "source_signature": source["source_signature"],
        "source_freeze_sha256": _sha_file(SOURCE_REPORT), "source_qa_sha256": _sha_file(SOURCE_QA),
        "states": {name: {"path": f"experiments/runs/{LANE}/scale_states/{name}_R1.json",
                          "sha256": _sha_file(ARTIFACT_ROOT / "scale_states" / f"{name}_R1.json"),
                          "state_signature": states[name]["state_signature"],
                          "intercept": states[name]["intercept"], "slope": states[name]["slope"],
                          "fit_status": states[name]["status"]} for name in states},
        "applications": applications,
        "operation_budget": {"affine_fit_started": 2, "affine_fit_saved": 2,
                              "application_started": 2, "application_saved": 2,
                              "base_fits": 0, "POS_fits": 0, "preprocessor_fits": 0,
                              "hinge_fits": 0},
        "resource_snapshot": {"available_ram_bytes": available,
                              "initial_ram_floor_bytes": MEMORY_START,
                              "reserve_bytes": MEMORY_RESERVE, "threads": THREADS},
        "selection_target_rows_read": {name: states[name]["fit_rows"] for name in states},
        "score_metric_labels_read": False, "target_values_persisted": False,
        "test_or_sample_accessed": False, "kaggle_or_upload": False,
        "D1_score_cycles_used_as_D2_selection_only": [72, 75],
    }
    manifest = _write_json_new(ARTIFACT_ROOT / "FIT_APPLICATION_R1.json", body, "manifest_signature")
    return {"status": manifest["status"], "manifest_sha256": _sha_file(ARTIFACT_ROOT / "FIT_APPLICATION_R1.json"),
            "coefficients": {name: {"intercept": states[name]["intercept"], "slope": states[name]["slope"],
                                     "fit_status": states[name]["status"]} for name in states},
            "fit_count": 2, "application_count": 2, "score_labels_read": False}


def _metric(y: np.ndarray, p: np.ndarray) -> dict[str, Any]:
    target = np.asarray(y, dtype=np.float64); pred = np.asarray(p, dtype=np.float64)
    error = pred - target; residual = -error; absolute = np.abs(error)
    return {"rows": int(target.size), "mae": float(absolute.mean()),
            "p90_absolute_error": float(np.quantile(absolute, 0.90, method="linear")),
            "median_residual_y_minus_prediction": float(np.median(residual)),
            "mean_residual_y_minus_prediction": float(residual.mean()),
            "bias_prediction_minus_target": float(error.mean())}


def _groups(y: np.ndarray, arrays: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    missing = np.asarray(arrays["missing"], dtype=bool)
    masks = {"missing": missing, "present": ~missing,
             "target_lt5": np.asarray(y, dtype=np.float64) < 5.0,
             "target_ge95": np.asarray(y, dtype=np.float64) >= 95.0,
             "EF0": np.asarray(arrays["entry_friction"], dtype=np.float64) == 0.0,
             "EF1": np.asarray(arrays["entry_friction"], dtype=np.float64) > 0.0}
    for key, label in (("impact_quartile", "impact_load"), ("text_quartile", "text_load")):
        buckets = np.asarray(arrays[key], dtype=np.int8)
        for index in range(4): masks[f"{label}_Q{index+1}"] = buckets == index
    return masks


def _group_report(y: np.ndarray, champion: np.ndarray, candidate: np.ndarray,
                  masks: Mapping[str, np.ndarray]) -> dict[str, Any]:
    out = {}
    for name, mask0 in masks.items():
        mask = np.asarray(mask0, dtype=bool); n = int(mask.sum())
        if n < MIN_SUPPORT:
            out[name] = {"status": "UNKNOWN", "rows": n, "champion": None,
                         "candidate": None, "harm_candidate_minus_champion": None}
        else:
            before, after = _metric(y[mask], champion[mask]), _metric(y[mask], candidate[mask])
            out[name] = {"status": "MEASURED", "rows": n, "champion": before,
                         "candidate": after, "harm_candidate_minus_champion": after["mae"] - before["mae"]}
    return out


def _missing_compare(y: np.ndarray, champion: np.ndarray, candidate: np.ndarray,
                     median: np.ndarray, missing: np.ndarray) -> dict[str, Any]:
    mask = np.asarray(missing, dtype=bool)
    before = _metric(y[mask], champion[mask])
    after = _metric(y[mask], candidate[mask])
    prior = _metric(y[mask], median[mask])
    supported = int(mask.sum()) >= MIN_SUPPORT
    return {"rows": int(mask.sum()), "status": "MEASURED" if supported else "UNKNOWN",
            "champion": before if supported else None, "candidate": after if supported else None,
            "median_control": prior if supported else None,
            "mae_gain_candidate_vs_champion": before["mae"] - after["mae"] if supported else None,
            "mae_harm_candidate_minus_champion": after["mae"] - before["mae"] if supported else None,
            "mae_gain_candidate_vs_median_control": prior["mae"] - after["mae"] if supported else None}


def _view_metrics(name: str, arrays: Mapping[str, np.ndarray], y0: np.ndarray) -> dict[str, Any]:
    y = np.asarray(y0, dtype=np.float64)
    champion = np.asarray(arrays["champion_hinge"], dtype=np.float64)
    candidate = np.asarray(arrays["candidate_hinge"], dtype=np.float64)
    median = np.asarray(arrays["median_policy"], dtype=np.float64)
    missing = np.asarray(arrays["missing"], dtype=bool)
    if not np.array_equal(candidate[~missing].view("<u8"), champion[~missing].view("<u8")):
        raise PilotError(f"Present predictions changed in {name}")
    overall_champ, overall_candidate, overall_median = (_metric(y, champion), _metric(y, candidate), _metric(y, median))
    cycles = {}
    cycle_values = np.asarray(arrays["cycles"], dtype=np.int64)
    for cycle in np.unique(cycle_values):
        cmask = cycle_values == cycle; mmask = cmask & missing
        b, c = _metric(y[cmask], champion[cmask]), _metric(y[cmask], candidate[cmask])
        cycles[str(int(cycle))] = {
            "rows": int(cmask.sum()), "missing_rows": int(mmask.sum()),
            "missing_status": "MEASURED" if int(mmask.sum()) >= MIN_SUPPORT else "UNKNOWN",
            "champion": b, "candidate": c,
            "overall_mae_gain_candidate_vs_champion": b["mae"] - c["mae"],
            "missing_mae_gain_candidate_vs_champion": (
                _metric(y[mmask], champion[mmask])["mae"] - _metric(y[mmask], candidate[mmask])["mae"]
                if int(mmask.sum()) >= MIN_SUPPORT else None),
            "missing_mae_gain_candidate_vs_median_control": (
                _metric(y[mmask], median[mmask])["mae"] - _metric(y[mmask], candidate[mmask])["mae"]
                if int(mmask.sum()) >= MIN_SUPPORT else None),
        }
    row_identity = _canonical_sha({"rows": _sha_array(arrays["query_rows"]),
                                   "ids": _sha_array(arrays["query_ids"]),
                                   "cycles": _sha_array(arrays["cycles"])})
    return {
        "view": name, "rows": len(y), "row_identity_sha256": row_identity,
        "missing": _missing_compare(y, champion, candidate, median, missing),
        "overall": {"champion": overall_champ, "candidate": overall_candidate,
                    "median_control": overall_median,
                    "mae_gain_candidate_vs_champion": overall_champ["mae"] - overall_candidate["mae"],
                    "mae_gain_candidate_vs_median_control": overall_median["mae"] - overall_candidate["mae"]},
        "per_cycle": cycles,
        "fixed_groups": _group_report(y, champion, candidate, _groups(y, arrays)),
        "present_bit_exact": True,
        "score_labels_read": int(len(y)),
    }


def _all_view_max(views: Mapping[str, Mapping[str, Any]]) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    per_view = {}
    all_records = []
    for name in ("D1", "D2", "pooled"):
        measured = [{"group": group, "rows": value["rows"],
                     "harm_candidate_minus_champion": value["harm_candidate_minus_champion"]}
                    for group, value in views[name]["fixed_groups"].items() if value["status"] == "MEASURED"]
        maximum = max(measured, key=lambda item: item["harm_candidate_minus_champion"], default=None)
        per_view[name] = maximum
        if maximum is not None: all_records.append({"view": name, **maximum})
    maximum_all = max(all_records, key=lambda item: item["harm_candidate_minus_champion"], default=None)
    return maximum_all, per_view


def _read_prediction(name: str, root: Path = ARTIFACT_ROOT) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    receipt_path = root / "predictions" / f"{name}_R1.json"
    receipt = _read_json(receipt_path); _verify_signed(receipt, "receipt_signature", f"prediction receipt {name}")
    if receipt.get("status") != "PREDICTION_SAVED":
        raise PilotError(f"Invalid prediction receipt: {name}")
    arrays = _verify_npz(ROOT / receipt["prediction"]["path"], receipt["prediction"])
    return receipt, arrays


def _score_stage() -> dict[str, Any]:
    source, base, _controls, roles = _verified_source()
    if SCORE_REPORT.exists() or any(event.get("operation_id") == "SCORE_R1" for event in _operation_events()):
        raise PilotError("Score operation already exists; repeat score forbidden")
    if not PRELABEL_QA.exists():
        raise PilotError("Independent prelabel QA required before score label access")
    pre = _read_json(PRELABEL_QA); _verify_signed(pre, "qa_signature", "independent prelabel QA")
    if (pre.get("status") != "PASS" or pre.get("source_signature") != source["source_signature"]
            or pre.get("score_labels_read") is not False):
        raise PilotError("Score labels blocked until matching prelabel QA PASS")
    prediction_receipts = {name: _read_prediction(name)[0] for name in ("D1", "D2")}
    predictions = {name: _verify_npz(ROOT / prediction_receipts[name]["prediction"]["path"],
                                     prediction_receipts[name]["prediction"])
                   for name in ("D1", "D2")}
    score_rows = np.concatenate([predictions[name]["query_rows"] for name in ("D1", "D2")])
    score_ids = np.concatenate([predictions[name]["query_ids"].astype(str) for name in ("D1", "D2")])
    if (len(score_rows) != EXPECTED_POOLED_ROWS or len(np.unique(score_rows)) != EXPECTED_POOLED_ROWS
            or len(np.unique(score_ids)) != EXPECTED_POOLED_ROWS):
        raise PilotError("Score population identity changed after prelabel QA")
    _start_operation("SCORE_R1")
    try:
        train_hash = source["parent_official_train_sha256"]
        parent_base, parent_source, _, _, _, _, _ = _parent_context()
        if parent_source.get("parent_official_train_sha256") != train_hash:
            raise PilotError("Official Train hash changed before scoring")
        target = {name: _read_targets(predictions[name]["query_rows"], predictions[name]["query_ids"])
                  for name in ("D1", "D2")}
        views = {name: _view_metrics(name, predictions[name], target[name]) for name in ("D1", "D2")}
        pooled_arrays = {key: np.concatenate([predictions[name][key] for name in ("D1", "D2")])
                         for key in predictions["D1"]}
        pooled_y = np.concatenate([target[name] for name in ("D1", "D2")])
        views["pooled"] = _view_metrics("pooled", pooled_arrays, pooled_y)
        all_max, per_view_max = _all_view_max(views)
        group_guard_pass = (all_max is not None and all(
            item is not None and item["harm_candidate_minus_champion"] <= 0.25 for item in per_view_max.values()))
        affine_fits_successful = all(prediction_receipts[name].get("fallback") is False for name in ("D1", "D2"))
        both_missing_supported = all(views[name]["missing"]["rows"] >= MIN_SUPPORT for name in ("D1", "D2"))
        present_exact = all(views[name]["present_bit_exact"] for name in ("D1", "D2", "pooled"))
        pooled_overall_gain = views["pooled"]["overall"]["mae_gain_candidate_vs_champion"]
        pooled_missing_vs_median = views["pooled"]["missing"]["mae_gain_candidate_vs_median_control"]
        scientific = bool(affine_fits_successful and both_missing_supported
                          and all(views[name]["missing"]["mae_gain_candidate_vs_champion"] > 0 for name in ("D1", "D2"))
                          and pooled_overall_gain > 0 and pooled_missing_vs_median > 0
                          and group_guard_pass and present_exact)
        exploratory = bool(affine_fits_successful and both_missing_supported and pooled_overall_gain > 0
                           and all(views[name]["missing"]["mae_harm_candidate_minus_champion"] <= 0.10
                                   for name in ("D1", "D2"))
                           and group_guard_pass and present_exact)
        body = {
            "schema_version": 1, "experiment_id": LANE, "status": "SCORED",
            "source_signature": source["source_signature"],
            "source_freeze_sha256": _sha_file(SOURCE_REPORT),
            "source_qa_sha256": _sha_file(SOURCE_QA),
            "prelabel_qa_sha256": _sha_file(PRELABEL_QA),
            "fit_application_manifest_sha256": _sha_file(ARTIFACT_ROOT / "FIT_APPLICATION_R1.json"),
            "folds": {name: views[name] for name in ("D1", "D2")},
            "pooled": views["pooled"],
            "pooled_rows": len(score_rows),
            "pooled_row_identity_sha256": _canonical_sha({"rows": _sha_array(score_rows), "ids": _sha_array(score_ids)}),
            "fixed_group_guard": {"minimum_support": MIN_SUPPORT, "harm_limit": 0.25,
                                  "views": per_view_max, "maximum_supported_group_harm_all_views": all_max,
                                  "all_views_pass": group_guard_pass},
            "both_fold_missing_support_pass": both_missing_supported,
            "both_affine_fits_successful": affine_fits_successful,
            "present_predictions_bit_exact": present_exact,
            "query_target_rows_read": len(score_rows), "target_values_persisted": False,
            "test_or_sample_accessed": False, "kaggle_or_upload": False,
            "selection_disclosure": "D2 calibration selection E67 cycles 68-75 includes D1 score cycles 72-75. Those targets enter only D2 selection; D1 scoring occurs after prelabel QA.",
            "temporal_disclosure": "D1 and D2 score IDs are distinct but not independent validation; both are reused development rows and E75 training includes earlier cycles.",
            "adaptive_followup_disclosure": "Outcome-exposed follow-up to the negative missing median pilot; opposite D1/D2 offset signs warn of nonstationarity. No rescue tuning was performed.",
            "fit_application_accounting": {"affine_fits": 2, "applications": 2,
                                            "base_fits": 0, "POS_fits": 0,
                                            "preprocessor_fits": 0, "hinge_fits": 0},
            "scientific_gate_without_independent_metric_qa": scientific,
            "exploratory_gate_without_independent_metric_qa": exploratory,
            "independent_metric_qa_pass": False,
        }
        report = _write_json_new(SCORE_REPORT, body, "score_signature")
        _finish_operation("SCORE_R1", _sha_file(SCORE_REPORT))
        return {"status": report["status"], "score_signature": report["score_signature"],
                "pooled_mae_gain": pooled_overall_gain, "pooled_missing_gain_vs_median": pooled_missing_vs_median,
                "maximum_supported_group_harm_all_views": all_max,
                "scientific_gate_before_metric_qa": scientific,
                "exploratory_gate_before_metric_qa": exploratory}
    except Exception as exc:
        _fail_operation("SCORE_R1", exc)
        raise


def _final_source_stage() -> dict[str, Any]:
    score = _read_json(SCORE_REPORT); _verify_signed(score, "score_signature", "affine score")
    qa = _read_json(METRIC_QA); _verify_signed(qa, "qa_signature", "affine metric QA")
    if (qa.get("status") != "PASS" or qa.get("score_signature") != score.get("score_signature")
            or qa.get("exploratory_benchmark_eligible") is not True):
        return {"status": "NOT_PREPARED", "reason": "Pilot did not pass exploratory eligibility"}
    base = _load_module(PARENT_VERIFIER, "median_parent_for_affine_final_source")
    parent_source, _ = base._source()
    r2, arrays = base._parent_receipts()
    if r2.get("source_signature") != parent_source.get("parent_source_signature"):
        raise PilotError("Parent E67/E75 source signature differs for final source preparation")
    e67 = arrays[67]; e75 = arrays[75]
    keep67 = (e67["cycles"] >= 68) & (e67["cycles"] <= 75)
    keep75 = (e75["cycles"] >= 76) & (e75["cycles"] <= 79)
    keys = ("query_rows", "query_ids", "cycles", "missing", "baseline", "champion_hinge")
    final_arrays = {key: np.concatenate([e67[key][keep67], e75[key][keep75]]) for key in keys}
    order = np.argsort(final_arrays["query_rows"], kind="stable")
    final_arrays = {key: value[order] for key, value in final_arrays.items()}
    if (len(final_arrays["query_rows"]) != EXPECTED_POOLED_ROWS
            or len(np.unique(final_arrays["query_rows"])) != EXPECTED_POOLED_ROWS
            or len(np.unique(final_arrays["query_ids"].astype(str))) != EXPECTED_POOLED_ROWS):
        raise PilotError("Final target-free OOF source is not unique")
    archive = _write_npz_new(FINAL_SOURCE_ARCHIVE, final_arrays)
    body = {"schema_version": 1, "experiment_id": LANE,
            "status": "FINAL_SOURCE_READY_PENDING_ROOT_REVIEW",
            "score_signature": score["score_signature"], "metric_qa_signature": qa["qa_signature"],
            "parent_source_signature": parent_source["source_signature"],
            "archive": archive, "rows": len(final_arrays["query_rows"]), "cycles": [68, 79],
            "missing_rows": int(np.asarray(final_arrays["missing"], dtype=bool).sum()),
            "final_fit_plan": "one a,b solve on unique missing OOF E67 cycles 68-75 plus E75 cycles 76-79; no final coefficients fit here",
            "final_scalar_fit_started": False, "E79_application_started": False,
            "test_or_sample_accessed": False, "kaggle_or_upload": False}
    report = _write_json_new(FINAL_SOURCE_REPORT, body, "source_signature")
    return {"status": report["status"], "source_signature": report["source_signature"],
            "archive_sha256": archive["sha256"], "rows": len(final_arrays["query_rows"]),
            "final_fit_started": False, "E79_application_started": False}


def _synthetic_e2e() -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="reg_missing_affine_l1_synthetic_") as temporary:
        root = Path(temporary)
        try:
            _read_npz_exact(root / "missing_required.npz", ("x",))
        except PilotError:
            missing_input_rejected = True
        else:
            missing_input_rejected = False
        if not missing_input_rejected or (root / "lane_ledger.jsonl").exists():
            raise PilotError("Missing-required-input synthetic guard or no-label gate failed")
        states = {}; predictions = {}; score_targets = {}
        supports = {}; fit_inputs = {}; fit_targets = {}; score_inputs = {}; expected_candidates = {}
        for name in ("D1", "D2"):
            x = np.linspace(2.0, 98.0, 140, dtype=np.float64) + (0.01 if name == "D2" else 0.0)
            fit = {"query_rows": np.arange(140, dtype=np.int64) + (0 if name == "D1" else 1000),
                   "query_ids": np.asarray([f"{name}-fit-{i}" for i in range(140)]),
                   "cycles": np.full(140, 70 if name == "D1" else 74, dtype=np.int64),
                   "missing": np.ones(140, dtype=bool), "baseline": x}
            target_fit = 8.0 + (0.72 if name == "D1" else 0.84) * x + (((np.arange(140) % 5) - 2) * 0.1)
            fit_inputs[name] = fit; fit_targets[name] = target_fit
            n = 180
            sx = np.linspace(1.0, 99.0, n, dtype=np.float64)
            missing = (np.arange(n) % 3 != 0)
            champion = np.clip(10.0 + 0.68 * sx, 0.0, 100.0)
            champion[~missing] += 0.25
            median = champion.copy(); median[missing] = np.clip(sx[missing] + (0.5 if name == "D1" else -0.5), 0.0, 100.0)
            score = {"query_rows": np.arange(n, dtype=np.int64) + (10000 if name == "D1" else 20000),
                     "query_ids": np.asarray([f"{name}-score-{i}" for i in range(n)]),
                     "cycles": np.resize(np.arange(72, 76) if name == "D1" else np.arange(76, 80), n),
                     "missing": missing, "baseline": sx, "champion_hinge": champion,
                     "median_policy": median, "entry_friction": np.resize([0.0, 1.0], n),
                     "impact_quartile": np.resize([0, 1, 2, 3], n).astype(np.int8),
                     "text_quartile": np.resize([3, 2, 1, 0], n).astype(np.int8)}
            score_inputs[name] = score
            score_targets[name] = 3.0 + 0.88 * sx + (((np.arange(n) % 7) - 3) * 0.15)
            supports[name] = {"fit_rows": 140, "fit_distinct_x": int(np.unique(x).size),
                              "fit_iqr": float(np.quantile(x, .75, method="linear")-
                                               np.quantile(x, .25, method="linear")),
                              "score_missing": int(missing.sum())}
        for name in ("D1", "D2"):
            states[name] = _fit_fold(name, fit_inputs[name], {"source_signature": "synthetic-source"}, root=root,
                                     y_override=fit_targets[name])
            supports[name]["lp_success"] = states[name]["lp"]["success"]
        for name in ("D1", "D2"):
            score = score_inputs[name]
            expected_candidates[name] = _apply_affine(
                score["baseline"], score["champion_hinge"], score["missing"],
                float(states[name]["intercept"]), float(states[name]["slope"]),
                fallback=states[name].get("status") != "FIT_COMPLETE")
            _apply_fold(name, score, states[name], {"source_signature": "synthetic-source"}, root=root)
            _receipt, recovered = _read_prediction(name, root=root)
            if not np.array_equal(recovered["candidate_hinge"], expected_candidates[name]):
                raise PilotError("Synthetic prediction archive readback differs")
            missing = np.asarray(score["missing"], dtype=bool)
            champion = np.asarray(score["champion_hinge"], dtype=np.float64)
            if not np.array_equal(recovered["candidate_hinge"][~missing].view("<u8"), champion[~missing].view("<u8")):
                raise PilotError("Synthetic present fallback bits changed")
            predictions[name] = recovered
        if set(np.intersect1d(predictions["D1"]["query_ids"], predictions["D2"]["query_ids"])):
            raise PilotError("Synthetic D1/D2 score IDs are not origin-isolated")
        fold_metrics = {name: _view_metrics(name, predictions[name], score_targets[name]) for name in ("D1", "D2")}
        pooled_arrays = {key: np.concatenate([predictions[name][key] for name in ("D1", "D2")])
                         for key in predictions["D1"]}
        pooled_y = np.concatenate([score_targets[name] for name in ("D1", "D2")])
        fold_metrics["pooled"] = _view_metrics("pooled", pooled_arrays, pooled_y)
        maximum, per_view = _all_view_max(fold_metrics)
        synthetic_qa_body = {"schema_version": 1, "status": "PASS_SYNTHETIC_ONLY",
                             "folds": fold_metrics, "per_view_group_max": per_view,
                             "all_view_group_max": maximum,
                             "labels_are_synthetic": True, "target_values_persisted": False}
        synthetic_qa = _signed(synthetic_qa_body, "qa_signature")
        _write_json_new(root / "SYNTHETIC_METRIC_QA.json", synthetic_qa_body, "qa_signature")
        if not _operation_events(root) or any(event.get("event") == "OPERATION_FAILED" for event in _operation_events(root)):
            raise PilotError("Synthetic lifecycle ledger did not finish cleanly")
        event_ids = [event["operation_id"] for event in _operation_events(root)]
        expected_ids = ["FIT_D1_R1", "FIT_D1_R1", "FIT_D2_R1", "FIT_D2_R1",
                        "APPLY_D1_R1", "APPLY_D1_R1", "APPLY_D2_R1", "APPLY_D2_R1"]
        if event_ids != expected_ids:
            raise PilotError("Synthetic lifecycle operation ownership/order differs")
        # A repeated operation ID is rejected before new work starts.
        prior_count = len(_operation_events(root))
        try:
            _start_operation("FIT_D1_R1", root)
        except PilotError:
            retry_rejected = True
        else:
            retry_rejected = False
        if not retry_rejected or len(_operation_events(root)) != prior_count:
            raise PilotError("Synthetic consumed-unit retry was not blocked")
        return {"status": "PASS", "synthetic_only": True,
                "lp_solves": sum(bool(states[name]["lp"]["success"]) for name in states),
                "fit_rows": {name: supports[name]["fit_rows"] for name in supports},
                "score_missing_rows": {name: supports[name]["score_missing"] for name in supports},
                "missing_required_input_rejected": missing_input_rejected,
                "origin_isolation_pass": True, "present_bits_exact": True,
                "one_shot_retry_rejected": retry_rejected,
                "metric_qa": "PASS_SYNTHETIC_ONLY", "production_ledger_touched": False,
                "temporary_outputs_removed": True}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--source", action="store_true")
    mode.add_argument("--fit-apply", action="store_true")
    mode.add_argument("--score", action="store_true")
    mode.add_argument("--prepare-final-source", action="store_true")
    mode.add_argument("--synthetic-e2e", action="store_true")
    args = parser.parse_args(argv)
    try:
        if args.synthetic_e2e: result = _synthetic_e2e()
        elif args.source: result = _source_stage()
        elif args.fit_apply: result = _fit_apply_stage()
        elif args.score: result = _score_stage()
        else: result = _final_source_stage()
    except Exception as exc:
        print(json.dumps({"status": "BLOCKED", "error": f"{type(exc).__name__}: {exc}"}, sort_keys=True))
        return 1
    print(json.dumps(result, sort_keys=True, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
