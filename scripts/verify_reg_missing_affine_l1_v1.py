"""Independent source, prelabel, metric, and final-source QA for affine L1 R1."""
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
LANE = "REG_MISSING_AFFINE_L1_V1"
REPORT_ROOT = ROOT / "reports" / "optimization_v3" / LANE
ARTIFACT_ROOT = ROOT / "experiments" / "runs" / LANE
SOURCE_REPORT_R1 = REPORT_ROOT / "SOURCE_FREEZE_R1.json"
SOURCE_QA_R1 = REPORT_ROOT / "SOURCE_QA_R1.json"
SOURCE_REPORT = REPORT_ROOT / "SOURCE_FREEZE_R2.json"
SOURCE_QA = REPORT_ROOT / "SOURCE_QA_R2.json"
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
PARENT_GUARD_REPORT = PARENT_REPORT_ROOT / "GROUP_GUARD_SCOPE_QA_R3.json"
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
MIN_DISTINCT = 20
MIN_IQR = 5.0
EXPECTED_ROWS = 15953


class VerifyError(RuntimeError):
    pass


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False,
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


def _same_tree(left: Any, right: Any, tolerance: float = 1e-12) -> bool:
    if isinstance(left, dict) and isinstance(right, dict):
        return set(left) == set(right) and all(_same_tree(left[k], right[k], tolerance) for k in left)
    if isinstance(left, list) and isinstance(right, list):
        return len(left) == len(right) and all(_same_tree(a, b, tolerance) for a, b in zip(left, right))
    if (isinstance(left, (int, float)) and not isinstance(left, bool)
            and isinstance(right, (int, float)) and not isinstance(right, bool)):
        return abs(float(left) - float(right)) <= tolerance
    return left == right


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise VerifyError(f"Cannot load helper {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _parent() -> tuple[Any, dict[str, Any], dict[str, np.ndarray], dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    base = _load_module(PARENT_VERIFIER, "verify_parent_missing_median_for_affine")
    source, arrays = base._source()
    score = _read_json(PARENT_REPORT_ROOT / "SCORE_R1.json")
    metric = _read_json(PARENT_REPORT_ROOT / "METRIC_QA_R2.json")
    prelabel = _read_json(PARENT_REPORT_ROOT / "PRELABEL_QA_R1.json")
    guard = _read_json(PARENT_GUARD_REPORT)
    for value, key, label in ((score, "score_signature", "parent median score"),
                              (metric, "qa_signature", "parent median QA"),
                              (prelabel, "qa_signature", "parent median prelabel"),
                              (guard, "addendum_signature", "parent all-view guard")):
        _verify_signed(value, key, label)
    if (source.get("status") != "SOURCE_FROZEN_PRELABEL" or score.get("status") != "SCORED"
            or metric.get("status") != "PASS" or prelabel.get("status") != "PASS"
            or guard.get("status") != "PASS" or guard.get("qualification_status") != "FAIL"
            or source.get("source_signature") != score.get("source_signature")
            or metric.get("score_signature") != score.get("score_signature")
            or metric.get("score_sha256") != _sha_file(PARENT_REPORT_ROOT / "SCORE_R1.json")
            or prelabel.get("source_signature") != source.get("source_signature")
            or guard.get("source_inventory", {}).get("inputs", {}).get("score", {}).get("sha256")
                != _sha_file(PARENT_REPORT_ROOT / "SCORE_R1.json")
            or guard.get("all_view_group_guard_pass") is not False):
        raise VerifyError("Parent median source/score/all-view guard lineage failed")
    controls = {}
    for name in ("D1", "D2"):
        state, receipt, prediction = base._load_output(name)
        controls[name] = {"state": state, "receipt": receipt, "prediction": prediction}
    return base, source, arrays, score, metric, {"prelabel": prelabel, "guard": guard}, controls


def _source_hashes() -> dict[str, str]:
    paths = {
        "runner": ROOT / "scripts" / "reg_missing_affine_l1_v1.py",
        "independent_verifier": Path(__file__),
        "regression_tests": ROOT / "tests" / "test_reg_missing_affine_l1_v1.py",
        "recipe": RECIPE_PATH, "root_plan": ROOT_PLAN_PATH,
        "parent_verifier": PARENT_VERIFIER,
        "parent_runner": ROOT / "scripts" / "reg_missing_median_offset_v1.py",
        "parent_metric_verifier": ROOT / "scripts" / "verify_reg_missing_median_offset_metric_r2.py",
        "parent_guard_verifier": ROOT / "scripts" / "verify_reg_missing_median_offset_guard_r3.py",
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
        raise VerifyError(f"Frozen source/reference files missing: {missing}")
    return {name: _sha_file(path) for name, path in paths.items()}


def _role_arrays(parent_arrays: Mapping[str, np.ndarray], controls: Mapping[str, Any], name: str) -> dict[str, dict[str, np.ndarray]]:
    fit_prefix, score_prefix = f"{name}_fit", f"{name}_score"
    fit = {key: np.asarray(parent_arrays[f"{fit_prefix}_{key}"]).copy()
           for key in ("query_rows", "query_ids", "cycles", "missing", "baseline")}
    score = {key: np.asarray(parent_arrays[f"{score_prefix}_{key}"]).copy()
             for key in ("query_rows", "query_ids", "cycles", "missing", "baseline", "champion_hinge",
                         "entry_friction", "impact_quartile", "text_quartile")}
    score["median_policy"] = np.asarray(controls[name]["prediction"]["candidate_hinge"], dtype=np.float64).copy()
    return {"fit": fit, "score": score}


def _expected_source_code_hashes() -> dict[str, str]:
    return _source_hashes()


def _source() -> tuple[dict[str, Any], Any, dict[str, Any], dict[str, dict[str, dict[str, np.ndarray]]]]:
    report = _read_json(SOURCE_REPORT); _verify_signed(report, "source_signature", "affine source freeze")
    prior_source = _read_json(SOURCE_REPORT_R1)
    _verify_signed(prior_source, "source_signature", "superseded affine source freeze R1")
    prior_qa = _read_json(SOURCE_QA_R1)
    _verify_signed(prior_qa, "qa_signature", "superseded affine source QA R1")
    recipe = _read_json(RECIPE_PATH); _verify_signed(recipe, "recipe_signature", "affine recipe")
    base, parent_source, parent_arrays, _score, _metric, parent_meta, controls = _parent()
    expected_references = {
        "parent_source_freeze_sha256": _sha_file(PARENT_REPORT_ROOT / "SOURCE_FREEZE_R1.json"),
        "parent_source_archive_sha256": _sha_file(PARENT_ARTIFACT_ROOT / "SOURCE_INPUTS_R1.npz"),
        "parent_source_qa_sha256": _sha_file(PARENT_REPORT_ROOT / "SOURCE_QA_R1.json"),
        "parent_score_sha256": _sha_file(PARENT_REPORT_ROOT / "SCORE_R1.json"),
        "parent_metric_qa_sha256": _sha_file(PARENT_REPORT_ROOT / "METRIC_QA_R2.json"),
        "parent_prelabel_qa_sha256": _sha_file(PARENT_REPORT_ROOT / "PRELABEL_QA_R1.json"),
        "parent_all_view_guard_r3_sha256": _sha_file(PARENT_GUARD_REPORT),
    }
    if (report.get("status") != "SOURCE_FROZEN_PRELABEL"
            or report.get("source_revision") != 2
            or report.get("supersedes_source_signature") != prior_source.get("source_signature")
            or report.get("supersedes_source_freeze_sha256") != _sha_file(SOURCE_REPORT_R1)
            or report.get("supersedes_source_qa_signature") != prior_qa.get("qa_signature")
            or report.get("supersedes_source_qa_sha256") != _sha_file(SOURCE_QA_R1)
            or prior_qa.get("status") != "PASS"
            or prior_qa.get("source_signature") != prior_source.get("source_signature")
            or prior_qa.get("score_labels_read") is not False
            or report.get("recipe_sha256") != _sha_file(RECIPE_PATH)
            or report.get("recipe_signature") != recipe.get("recipe_signature")
            or report.get("root_plan_sha256") != _sha_file(ROOT_PLAN_PATH)
            or report.get("parent_source_signature") != parent_source.get("source_signature")
            or report.get("source_code_hashes") != _expected_source_code_hashes()
            or any(report.get(key) != value for key, value in expected_references.items())
            or report.get("parent_score_signature") != _read_json(PARENT_REPORT_ROOT / "SCORE_R1.json").get("score_signature")
            or report.get("parent_metric_qa_signature") != _read_json(PARENT_REPORT_ROOT / "METRIC_QA_R2.json").get("qa_signature")
            or report.get("parent_all_view_guard_r3_signature") != parent_meta["guard"].get("addendum_signature")):
        raise VerifyError("Affine source freeze binding or code receipt failed")
    roles = {name: _role_arrays(parent_arrays, controls, name) for name in ("D1", "D2")}
    return report, base, controls, roles


def _support(baseline: np.ndarray) -> dict[str, Any]:
    x = np.asarray(baseline, dtype=np.float64)
    if len(x) == 0 or not np.isfinite(x).all():
        return {"rows": len(x), "distinct_baselines": 0, "iqr_linear": None, "support_pass": False}
    q1, q3 = np.quantile(x, [0.25, 0.75], method="linear")
    count = int(np.unique(x).size)
    iqr = float(q3 - q1)
    return {"rows": int(len(x)), "distinct_baselines": count, "iqr_linear": iqr,
            "minimum_rows": MIN_SUPPORT, "minimum_distinct_baselines": MIN_DISTINCT,
            "minimum_iqr": MIN_IQR,
            "support_pass": bool(len(x) >= MIN_SUPPORT and count >= MIN_DISTINCT and iqr >= MIN_IQR)}


def source_stage() -> dict[str, Any]:
    report, _base, controls, roles = _source()
    expected = {"D1": (342, 507, [68, 71], [72, 75]),
                "D2": (849, 526, [68, 75], [76, 79])}
    checks = {}
    for name, (fit_n, score_missing_n, fit_cycles, score_cycles) in expected.items():
        fit, score = roles[name]["fit"], roles[name]["score"]
        fit_rows, fit_ids = np.asarray(fit["query_rows"], dtype=np.int64), np.asarray(fit["query_ids"], dtype=str)
        score_rows, score_ids = np.asarray(score["query_rows"], dtype=np.int64), np.asarray(score["query_ids"], dtype=str)
        fcy, scy = np.asarray(fit["cycles"], dtype=np.int64), np.asarray(score["cycles"], dtype=np.int64)
        fmissing, smissing = np.asarray(fit["missing"], dtype=bool), np.asarray(score["missing"], dtype=bool)
        support = _support(fit["baseline"])
        if (len(fit_rows) != fit_n or len(np.unique(fit_rows)) != fit_n or len(np.unique(fit_ids)) != fit_n
                or not np.array_equal(fit_rows, np.sort(fit_rows)) or not fmissing.all()
                or [int(fcy.min()), int(fcy.max())] != fit_cycles
                or not support["support_pass"] or len(score_rows) != (7462 if name == "D1" else 8491)
                or len(np.unique(score_rows)) != len(score_rows) or len(np.unique(score_ids)) != len(score_ids)
                or [int(scy.min()), int(scy.max())] != score_cycles or int(smissing.sum()) != score_missing_n):
            raise VerifyError(f"Independent frozen source/support check failed: {name}")
        for key in ("baseline", "champion_hinge", "median_policy"):
            if not np.isfinite(np.asarray(score[key], dtype=np.float64)).all():
                raise VerifyError(f"Nonfinite frozen score vector: {name}.{key}")
        if not np.array_equal(np.asarray(score["median_policy"])[~smissing].view("<u8"),
                              np.asarray(score["champion_hinge"])[~smissing].view("<u8")):
            raise VerifyError(f"Median control present route not exact: {name}")
        expected_role = report["role_sources"][name]
        if (support != expected_role["selection"]["support"]
                or _sha_array(fit_rows) != expected_role["selection"]["rows_sha256"]
                or _sha_array(score_rows) != expected_role["score"]["rows_sha256"]
                or _sha_array(fit_ids) != expected_role["selection"]["ids_sha256"]
                or _sha_array(score_ids) != expected_role["score"]["ids_sha256"]):
            raise VerifyError(f"Frozen role receipt differs: {name}")
        checks[name] = {"fit_rows": fit_n, "distinct_baselines": support["distinct_baselines"],
                        "baseline_iqr_linear": support["iqr_linear"],
                        "score_rows": len(score_rows), "score_missing_rows": score_missing_n,
                        "support_pass": True, "present_control_exact": True}
    all_rows = np.concatenate([roles[name]["score"]["query_rows"] for name in ("D1", "D2")])
    all_ids = np.concatenate([roles[name]["score"]["query_ids"].astype(str) for name in ("D1", "D2")])
    if len(all_rows) != EXPECTED_ROWS or len(np.unique(all_rows)) != EXPECTED_ROWS or len(np.unique(all_ids)) != EXPECTED_ROWS:
        raise VerifyError("Pooled frozen source identities are not unique 15,953 rows")
    body = {"schema_version": 1, "experiment_id": LANE, "status": "PASS",
            "source_signature": report["source_signature"],
            "source_freeze_sha256": _sha_file(SOURCE_REPORT),
            "parent_source_archive_sha256": _sha_file(PARENT_ARTIFACT_ROOT / "SOURCE_INPUTS_R1.npz"),
            "target_labels_read": False, "score_labels_read": False,
            "pooled_unique_score_rows": len(all_rows), "checks": checks,
            "historical_novelty_check": "PASS_NO_EXACT_MISSING_ONLY_AFFINE_EQUIVALENT_FOUND",
            "test_or_sample_accessed": False}
    result = dict(body); result["qa_signature"] = _canonical_sha(body)
    payload = _canonical(result) + b"\n"
    if SOURCE_QA.exists() and SOURCE_QA.read_bytes() != payload:
        raise VerifyError("Refusing to overwrite changed source QA")
    if not SOURCE_QA.exists():
        SOURCE_QA.write_bytes(payload)
    return {"status": result["status"], "qa_signature": result["qa_signature"],
            "score_labels_read": False, "pooled_unique_score_rows": len(all_rows)}


def _ledger(root: Path = ARTIFACT_ROOT) -> list[dict[str, Any]]:
    path = root / "lane_ledger.jsonl"
    if not path.exists(): return []
    result = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try: value = json.loads(line)
        except Exception as exc: raise VerifyError("Malformed affine operation ledger") from exc
        if not isinstance(value, dict): raise VerifyError("Malformed affine operation event")
        result.append(value)
    return result


def _metric(y: np.ndarray, prediction: np.ndarray) -> dict[str, Any]:
    target = np.asarray(y, dtype=np.float64); pred = np.asarray(prediction, dtype=np.float64)
    error = pred - target; resid = target - pred; absolute = np.abs(error)
    return {"rows": int(len(target)), "mae": float(absolute.mean()),
            "p90_absolute_error": float(np.quantile(absolute, 0.90, method="linear")),
            "median_residual_y_minus_prediction": float(np.median(resid)),
            "mean_residual_y_minus_prediction": float(resid.mean()),
            "bias_prediction_minus_target": float(error.mean())}


def _masks(target: np.ndarray, pred: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    miss = np.asarray(pred["missing"], dtype=bool)
    result = {"missing": miss, "present": ~miss,
              "target_lt5": np.asarray(target, dtype=np.float64) < 5.0,
              "target_ge95": np.asarray(target, dtype=np.float64) >= 95.0,
              "EF0": np.asarray(pred["entry_friction"], dtype=np.float64) == 0.0,
              "EF1": np.asarray(pred["entry_friction"], dtype=np.float64) > 0.0}
    for key, label in (("impact_quartile", "impact_load"), ("text_quartile", "text_load")):
        value = np.asarray(pred[key], dtype=np.int8)
        for q in range(4): result[f"{label}_Q{q+1}"] = value == q
    return result


def _group_report(target: np.ndarray, base: np.ndarray, candidate: np.ndarray,
                  masks: Mapping[str, np.ndarray]) -> dict[str, Any]:
    output = {}
    for name, raw in masks.items():
        mask = np.asarray(raw, dtype=bool); n = int(mask.sum())
        if n < MIN_SUPPORT:
            output[name] = {"status": "UNKNOWN", "rows": n, "champion": None,
                            "candidate": None, "harm_candidate_minus_champion": None}
        else:
            b, c = _metric(target[mask], base[mask]), _metric(target[mask], candidate[mask])
            output[name] = {"status": "MEASURED", "rows": n, "champion": b,
                            "candidate": c, "harm_candidate_minus_champion": c["mae"]-b["mae"]}
    return output


def _missing(target: np.ndarray, base: np.ndarray, candidate: np.ndarray,
             median: np.ndarray, missing: np.ndarray) -> dict[str, Any]:
    mask = np.asarray(missing, dtype=bool); n = int(mask.sum())
    bm, cm, mm = _metric(target[mask], base[mask]), _metric(target[mask], candidate[mask]), _metric(target[mask], median[mask])
    ok = n >= MIN_SUPPORT
    return {"rows": n, "status": "MEASURED" if ok else "UNKNOWN",
            "champion": bm if ok else None, "candidate": cm if ok else None,
            "median_control": mm if ok else None,
            "mae_gain_candidate_vs_champion": bm["mae"]-cm["mae"] if ok else None,
            "mae_harm_candidate_minus_champion": cm["mae"]-bm["mae"] if ok else None,
            "mae_gain_candidate_vs_median_control": mm["mae"]-cm["mae"] if ok else None}


def _view(name: str, pred: Mapping[str, np.ndarray], target: np.ndarray) -> dict[str, Any]:
    y = np.asarray(target, dtype=np.float64)
    b = np.asarray(pred["champion_hinge"], dtype=np.float64)
    c = np.asarray(pred["candidate_hinge"], dtype=np.float64)
    m = np.asarray(pred["median_policy"], dtype=np.float64)
    missing = np.asarray(pred["missing"], dtype=bool)
    if not np.array_equal(c[~missing].view("<u8"), b[~missing].view("<u8")):
        raise VerifyError(f"Present bits changed in {name}")
    cycles = {}
    cycle_values = np.asarray(pred["cycles"], dtype=np.int64)
    for cycle in np.unique(cycle_values):
        cmask = cycle_values == cycle; mmask = cmask & missing; n = int(mmask.sum())
        cb, cc = _metric(y[cmask], b[cmask]), _metric(y[cmask], c[cmask])
        cycles[str(int(cycle))] = {
            "rows": int(cmask.sum()), "missing_rows": n,
            "missing_status": "MEASURED" if n >= MIN_SUPPORT else "UNKNOWN",
            "champion": cb, "candidate": cc,
            "overall_mae_gain_candidate_vs_champion": cb["mae"]-cc["mae"],
            "missing_mae_gain_candidate_vs_champion": (_metric(y[mmask], b[mmask])["mae"]-
                                                         _metric(y[mmask], c[mmask])["mae"] if n>=MIN_SUPPORT else None),
            "missing_mae_gain_candidate_vs_median_control": (_metric(y[mmask], m[mmask])["mae"]-
                                                              _metric(y[mmask], c[mmask])["mae"] if n>=MIN_SUPPORT else None),
        }
    bb, cc, mm = _metric(y,b), _metric(y,c), _metric(y,m)
    identity = _canonical_sha({"rows": _sha_array(pred["query_rows"]),
                               "ids": _sha_array(pred["query_ids"]),
                               "cycles": _sha_array(pred["cycles"])})
    return {"view": name, "rows": len(y), "row_identity_sha256": identity,
            "missing": _missing(y,b,c,m,missing),
            "overall": {"champion": bb, "candidate": cc, "median_control": mm,
                        "mae_gain_candidate_vs_champion": bb["mae"]-cc["mae"],
                        "mae_gain_candidate_vs_median_control": mm["mae"]-cc["mae"]},
            "per_cycle": cycles, "fixed_groups": _group_report(y,b,c,_masks(y,pred)),
            "present_bit_exact": True, "score_labels_read": len(y)}


def _maxima(views: Mapping[str, Mapping[str, Any]]) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    per_view = {}; all_values = []
    for view_name in ("D1", "D2", "pooled"):
        supported = [{"group": group, "rows": value["rows"],
                      "harm_candidate_minus_champion": value["harm_candidate_minus_champion"]}
                     for group, value in views[view_name]["fixed_groups"].items() if value["status"] == "MEASURED"]
        maximum = max(supported, key=lambda item: item["harm_candidate_minus_champion"], default=None)
        per_view[view_name] = maximum
        if maximum is not None: all_values.append({"view": view_name, **maximum})
    return max(all_values, key=lambda item: item["harm_candidate_minus_champion"], default=None), per_view


def prelabel_stage() -> dict[str, Any]:
    source, base, controls, roles = _source()
    manifest_path = ARTIFACT_ROOT / "FIT_APPLICATION_R1.json"
    manifest = _read_json(manifest_path); _verify_signed(manifest, "manifest_signature", "fit/application manifest")
    if (manifest.get("status") != "FIT_APPLICATION_COMPLETE"
            or manifest.get("source_signature") != source.get("source_signature")
            or manifest.get("operation_budget") != {"affine_fit_started": 2, "affine_fit_saved": 2,
                "application_started": 2, "application_saved": 2,
                "base_fits": 0, "POS_fits": 0, "preprocessor_fits": 0, "hinge_fits": 0}):
        raise VerifyError("Fit/application manifest or budget invalid")
    expected_events = [
        ("FIT_D1_R1","OPERATION_STARTED","missing_only_monotone_affine_l1"),
        ("FIT_D1_R1","OPERATION_SAVED","missing_only_monotone_affine_l1"),
        ("FIT_D2_R1","OPERATION_STARTED","missing_only_monotone_affine_l1"),
        ("FIT_D2_R1","OPERATION_SAVED","missing_only_monotone_affine_l1"),
        ("APPLY_D1_R1","OPERATION_STARTED","deterministic_missing_only_affine_application"),
        ("APPLY_D1_R1","OPERATION_SAVED","deterministic_missing_only_affine_application"),
        ("APPLY_D2_R1","OPERATION_STARTED","deterministic_missing_only_affine_application"),
        ("APPLY_D2_R1","OPERATION_SAVED","deterministic_missing_only_affine_application"),
    ]
    events = _ledger()
    if [(event.get("operation_id"),event.get("event"),event.get("operation_type")) for event in events] != expected_events:
        raise VerifyError("Exact fit/application one-shot ledger differs")
    if any(event.get("event") == "OPERATION_FAILED" or event.get("schema_version") != 1 for event in events):
        raise VerifyError("A failed fit/apply operation was recorded")
    saved_artifact_hashes = {
        "FIT_D1_R1": _sha_file(ARTIFACT_ROOT / "scale_states" / "D1_R1.json"),
        "FIT_D2_R1": _sha_file(ARTIFACT_ROOT / "scale_states" / "D2_R1.json"),
        "APPLY_D1_R1": _sha_file(ARTIFACT_ROOT / "predictions" / "D1_R1.json"),
        "APPLY_D2_R1": _sha_file(ARTIFACT_ROOT / "predictions" / "D2_R1.json"),
    }
    for event in events:
        if (event.get("event") == "OPERATION_SAVED"
                and event.get("artifact_sha256") != saved_artifact_hashes.get(event.get("operation_id"))):
            raise VerifyError(f"Operation ledger saved artifact hash mismatch: {event.get('operation_id')}")
    states = {}; predictions = {}; checks = {}; pooled_rows=[]; pooled_ids=[]
    for name in ("D1","D2"):
        state_path = ARTIFACT_ROOT / "scale_states" / f"{name}_R1.json"
        state = _read_json(state_path); _verify_signed(state,"state_signature",f"state {name}")
        receipt_path = ARTIFACT_ROOT / "predictions" / f"{name}_R1.json"
        receipt = _read_json(receipt_path); _verify_signed(receipt,"receipt_signature",f"receipt {name}")
        pred_path = ROOT / receipt["prediction"]["path"]
        with zipfile.ZipFile(pred_path) as zf:
            members = {member[:-4] for member in zf.namelist() if member.endswith(".npy")}
        with np.load(pred_path,allow_pickle=False) as zf:
            pred = {key: np.asarray(zf[key]).copy() for key in members}
        if ({key:_sha_array(value) for key,value in pred.items()} != receipt["prediction"].get("members")
                or _sha_file(pred_path) != receipt["prediction"].get("sha256")):
            raise VerifyError(f"Prediction archive receipt mismatch: {name}")
        fit = roles[name]["fit"]; score = roles[name]["score"]
        fit_rows=np.asarray(fit["query_rows"],dtype=np.int64); fit_ids=np.asarray(fit["query_ids"],dtype=str)
        fit_base=np.asarray(fit["baseline"],dtype=np.float64); fit_cycles=np.asarray(fit["cycles"],dtype=np.int64)
        support=_support(fit_base)
        if (state.get("source_signature") != source["source_signature"]
                or state.get("fit_rows") != len(fit_rows) or state.get("selection_rows_sha256") != _sha_array(fit_rows)
                or state.get("selection_ids_sha256") != _sha_array(fit_ids)
                or state.get("selection_cycles_sha256") != _sha_array(fit_cycles)
                or state.get("baseline_sha256") != _sha_array(fit_base) or state.get("support") != support
                or state.get("selection_cycles") != list((68,71) if name=="D1" else (68,75))):
            raise VerifyError(f"Selection state/source/support mismatch: {name}")
        y = np.asarray(base._read_targets(fit_rows,fit_ids),dtype=np.float64)
        if _sha_array(y) != state.get("selection_y_sha256"):
            raise VerifyError(f"Selection target digest differs on readback: {name}")
        lp = state.get("lp",{}); intercept=float(state["intercept"]); slope=float(state["slope"])
        resid = y-(intercept+slope*fit_base); objective=float(np.abs(resid).sum())
        if state.get("status")=="FIT_COMPLETE":
            if (lp.get("success") is not True or lp.get("solver_status") != 0 or slope<0
                    or lp.get("primal_feasibility")!="PASS"
                    or not np.isclose(objective,float(lp["objective_l1_sum"]),atol=max(1e-5,len(y)*1e-9),rtol=1e-8)
                    or not np.isclose(objective,float(lp["reported_solver_objective_l1_sum"]),atol=max(1e-5,len(y)*1e-9),rtol=1e-8)):
                raise VerifyError(f"LP status/objective/primal/coefficients failed readback: {name}")
        elif state.get("status")!="FIT_FAILED_FALLBACK" or state.get("intercept")!=0.0 or state.get("slope")!=1.0:
            raise VerifyError(f"Unknown fit state or nonidentity fallback: {name}")
        missing=np.asarray(score["missing"],dtype=bool); base_score=np.asarray(score["baseline"],dtype=np.float64)
        champion=np.asarray(score["champion_hinge"],dtype=np.float64); median=np.asarray(score["median_policy"],dtype=np.float64)
        candidate=np.asarray(pred["candidate_hinge"],dtype=np.float64)
        fallback=state.get("status")!="FIT_COMPLETE"
        expected=champion.copy()
        if not fallback: expected[missing]=np.clip(intercept+slope*base_score[missing],0,100)
        if (not np.array_equal(candidate.view("<u8"),expected.view("<u8"))
                or not np.array_equal(candidate[~missing].view("<u8"),champion[~missing].view("<u8"))
                or not np.array_equal(pred["median_policy"][~missing].view("<u8"),champion[~missing].view("<u8"))):
            raise VerifyError(f"Affine formula or present fallback readback failed: {name}")
        for key in ("query_rows","query_ids","cycles","missing","baseline","champion_hinge","median_policy",
                    "entry_friction","impact_quartile","text_quartile"):
            expected_key=np.asarray(score[key])
            actual_key=np.asarray(pred[key])
            if key in ("baseline","champion_hinge","median_policy"):
                same=np.array_equal(expected_key.astype(np.float64).view("<u8"),actual_key.astype(np.float64).view("<u8"))
            elif key=="query_ids": same=np.array_equal(expected_key.astype(str),actual_key.astype(str))
            else: same=np.array_equal(expected_key,actual_key)
            if not same: raise VerifyError(f"Prediction source field mismatch: {name}.{key}")
        pooled_rows.extend(pred["query_rows"].tolist()); pooled_ids.extend(pred["query_ids"].astype(str).tolist())
        states[name]=state; predictions[name]=pred
        checks[name]={"fit_rows":len(fit_rows),"distinct_baselines":support["distinct_baselines"],
                      "baseline_iqr":support["iqr_linear"],"score_rows":len(pred["query_rows"]),
                      "score_missing_rows":int(missing.sum()),"LP_status":lp.get("solver_status"),
                      "intercept":intercept,"slope":slope,"present_bits_exact":True,
                      "fallback":fallback,"target_values_persisted":False}
    if len(pooled_rows)!=EXPECTED_ROWS or len(set(pooled_rows))!=EXPECTED_ROWS or len(set(pooled_ids))!=EXPECTED_ROWS:
        raise VerifyError("Affine D1/D2 score identities are not unique pooled 15,953")
    body={"schema_version":1,"experiment_id":LANE,"status":"PASS",
          "source_signature":source["source_signature"],"fit_application_manifest_sha256":_sha_file(manifest_path),
          "checks":checks,"pooled_unique_score_rows":len(pooled_rows),
          "selection_labels_accessed_for_independent_state_readback":True,"score_labels_read":False,
          "target_values_persisted":False,"test_or_sample_accessed":False,"kaggle_or_upload":False}
    result=dict(body);result["qa_signature"]=_canonical_sha(body);payload=_canonical(result)+b"\n"
    if PRELABEL_QA.exists() and PRELABEL_QA.read_bytes()!=payload: raise VerifyError("Refusing to overwrite changed prelabel QA")
    if not PRELABEL_QA.exists(): PRELABEL_QA.write_bytes(payload)
    return {"status":result["status"],"qa_signature":result["qa_signature"],"score_labels_read":False,
            "pooled_unique_score_rows":len(pooled_rows)}


def _read_output(name: str) -> tuple[dict[str, Any],dict[str,np.ndarray]]:
    path=ARTIFACT_ROOT/"predictions"/f"{name}_R1.json"
    receipt=_read_json(path);_verify_signed(receipt,"receipt_signature",f"prediction {name}")
    prediction_path=ROOT/receipt["prediction"]["path"]
    if _sha_file(prediction_path)!=receipt["prediction"].get("sha256"):
        raise VerifyError(f"Prediction archive hash mismatch: {name}")
    with zipfile.ZipFile(prediction_path) as zf:
        names={member[:-4] for member in zf.namelist() if member.endswith(".npy")}
    if names!=set(receipt["prediction"].get("members",{})): raise VerifyError("Prediction member allowlist mismatch")
    with np.load(prediction_path,allow_pickle=False) as zf:
        arrays={key:np.asarray(zf[key]).copy() for key in names}
    if {k:_sha_array(v) for k,v in arrays.items()}!=receipt["prediction"].get("members"):
        raise VerifyError(f"Prediction member hash mismatch: {name}")
    return receipt,arrays


def _read_targets(rows: np.ndarray, ids: np.ndarray) -> np.ndarray:
    base=_load_module(PARENT_VERIFIER,"parent_target_reader_for_affine_metric_qa")
    y=np.asarray(base._read_targets(np.asarray(rows,dtype=np.int64),np.asarray(ids,dtype=str)),dtype=np.float64)
    if y.shape!=np.asarray(rows).shape or not np.isfinite(y).all(): raise VerifyError("Independent target readback invalid")
    return y


def metric_stage() -> dict[str, Any]:
    source, _base, _controls, _roles = _source()
    pre=_read_json(PRELABEL_QA);_verify_signed(pre,"qa_signature","prelabel QA")
    score=_read_json(SCORE_REPORT);_verify_signed(score,"score_signature","score report")
    if pre.get("status")!="PASS" or pre.get("score_labels_read") is not False:
        raise VerifyError("Metric QA blocked before prelabel PASS")
    if (score.get("status")!="SCORED" or score.get("source_signature")!=source.get("source_signature")
            or score.get("prelabel_qa_sha256")!=_sha_file(PRELABEL_QA)
            or score.get("independent_metric_qa_pass") is not False):
        raise VerifyError("Score report/source/prelabel binding failed")
    folds={};predictions={};targets={}
    for name in ("D1","D2"):
        _receipt,pred=_read_output(name);y=_read_targets(pred["query_rows"],pred["query_ids"].astype(str))
        computed=_view(name,pred,y)
        if not _same_tree(computed,score["folds"][name]): raise VerifyError(f"Independent fold metric replay differs: {name}")
        folds[name]=computed;predictions[name]=pred;targets[name]=y
    pooled={key:np.concatenate([predictions[name][key] for name in ("D1","D2")]) for key in predictions["D1"]}
    pooled_y=np.concatenate([targets[name] for name in ("D1","D2")])
    computed_pooled=_view("pooled",pooled,pooled_y)
    row_ids=np.concatenate([predictions[name]["query_ids"].astype(str) for name in ("D1","D2")])
    row_numbers=np.concatenate([predictions[name]["query_rows"] for name in ("D1","D2")])
    identity=_canonical_sha({"rows":_sha_array(row_numbers),"ids":_sha_array(row_ids)})
    if (not _same_tree(computed_pooled,score["pooled"])
            or score.get("pooled_rows")!=EXPECTED_ROWS
            or score.get("pooled_row_identity_sha256")!=identity
            or len(np.unique(row_numbers))!=EXPECTED_ROWS or len(np.unique(row_ids))!=EXPECTED_ROWS):
        raise VerifyError("Independent pooled metric or row identity differs")
    views={"D1":folds["D1"],"D2":folds["D2"],"pooled":computed_pooled}
    maximum,per_view=_maxima(views)
    if not _same_tree(score.get("fixed_group_guard",{}).get("views"),per_view):
        raise VerifyError("Score all-view group maxima differ from independent reconstruction")
    if not _same_tree(score.get("fixed_group_guard",{}).get("maximum_supported_group_harm_all_views"),maximum):
        raise VerifyError("Score all-view maximum harm differs")
    all_guard=(maximum is not None and all(v is not None and v["harm_candidate_minus_champion"]<=0.25 for v in per_view.values()))
    both_supported=all(folds[name]["missing"]["rows"]>=MIN_SUPPORT for name in ("D1","D2"))
    present_exact=all(views[name]["present_bit_exact"] for name in ("D1","D2","pooled"))
    pooled_gain=computed_pooled["overall"]["mae_gain_candidate_vs_champion"]
    pooled_missing_vs_median=computed_pooled["missing"]["mae_gain_candidate_vs_median_control"]
    both_fits_successful = all(_read_json(ARTIFACT_ROOT/"predictions"/f"{name}_R1.json").get("fallback") is False
                               for name in ("D1","D2"))
    scientific=bool(both_fits_successful and both_supported
                    and all(folds[name]["missing"]["mae_gain_candidate_vs_champion"]>0 for name in ("D1","D2"))
                    and pooled_gain>0 and pooled_missing_vs_median>0 and all_guard and present_exact)
    exploratory=bool(both_fits_successful and both_supported and pooled_gain>0
                     and all(folds[name]["missing"]["mae_harm_candidate_minus_champion"]<=0.10 for name in ("D1","D2"))
                     and all_guard and present_exact)
    if (score.get("scientific_gate_without_independent_metric_qa")!=scientific
            or score.get("exploratory_gate_without_independent_metric_qa")!=exploratory
            or score.get("both_affine_fits_successful")!=both_fits_successful
            or score.get("fixed_group_guard",{}).get("all_views_pass")!=all_guard):
        raise VerifyError("Independent affine gate reconstruction differs")
    body={"schema_version":1,"experiment_id":LANE,"status":"PASS",
          "source_signature":source["source_signature"],"score_signature":score["score_signature"],
          "score_sha256":_sha_file(SCORE_REPORT),"prelabel_qa_sha256":_sha_file(PRELABEL_QA),
          "independent_metric_replay":"PASS","independent_target_rows_read":len(pooled_y),
          "target_values_persisted":False,"test_or_sample_accessed":False,
          "maximum_supported_group_harm_by_view":per_view,
          "maximum_supported_group_harm_all_views":maximum,"all_view_group_guard_pass":all_guard,
          "pooled_overall_mae_gain":pooled_gain,"pooled_missing_mae_gain_vs_median_control":pooled_missing_vs_median,
          "both_affine_fits_successful":both_fits_successful,
          "D1_missing_gain_vs_champion":folds["D1"]["missing"]["mae_gain_candidate_vs_champion"],
          "D2_missing_gain_vs_champion":folds["D2"]["missing"]["mae_gain_candidate_vs_champion"],
          "scientific_qualification":scientific,"exploratory_benchmark_eligible":exploratory,
          "present_predictions_bit_exact":present_exact}
    result=dict(body);result["qa_signature"]=_canonical_sha(body);payload=_canonical(result)+b"\n"
    if METRIC_QA.exists() and METRIC_QA.read_bytes()!=payload: raise VerifyError("Refusing to overwrite changed metric QA")
    if not METRIC_QA.exists(): METRIC_QA.write_bytes(payload)
    return {"status":result["status"],"qa_signature":result["qa_signature"],
            "pooled_mae_gain":pooled_gain,"pooled_missing_gain_vs_median":pooled_missing_vs_median,
            "all_view_group_guard_pass":all_guard,"scientific_qualification":scientific,
            "exploratory_benchmark_eligible":exploratory}


def final_source_stage() -> dict[str, Any]:
    score=_read_json(SCORE_REPORT);_verify_signed(score,"score_signature","score report")
    qa=_read_json(METRIC_QA);_verify_signed(qa,"qa_signature","metric QA")
    report=_read_json(FINAL_SOURCE_REPORT);_verify_signed(report,"source_signature","final source report")
    receipt=report["archive"]
    if _sha_file(FINAL_SOURCE_ARCHIVE)!=receipt.get("sha256"): raise VerifyError("Final source archive hash mismatch")
    with zipfile.ZipFile(FINAL_SOURCE_ARCHIVE) as zf:
        names={member[:-4] for member in zf.namelist() if member.endswith(".npy")}
    with np.load(FINAL_SOURCE_ARCHIVE,allow_pickle=False) as zf: arrays={key:np.asarray(zf[key]).copy() for key in names}
    if {k:_sha_array(v) for k,v in arrays.items()}!=receipt.get("members"):
        raise VerifyError("Final source member hashes mismatch")
    rows=arrays["query_rows"];ids=arrays["query_ids"].astype(str)
    if (report.get("status")!="FINAL_SOURCE_READY_PENDING_ROOT_REVIEW"
            or report.get("score_signature")!=score.get("score_signature")
            or report.get("metric_qa_signature")!=qa.get("qa_signature")
            or qa.get("exploratory_benchmark_eligible") is not True
            or len(rows)!=EXPECTED_ROWS or len(np.unique(rows))!=EXPECTED_ROWS or len(np.unique(ids))!=EXPECTED_ROWS
            or report.get("final_scalar_fit_started") is not False or report.get("E79_application_started") is not False):
        raise VerifyError("Final target-free source package is invalid")
    return {"status":"PASS","rows":len(rows),"final_scalar_fit_started":False,"E79_application_started":False}


def main(argv: list[str] | None = None) -> int:
    parser=argparse.ArgumentParser(description=__doc__)
    group=parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--source",action="store_true");group.add_argument("--prelabel",action="store_true")
    group.add_argument("--metric",action="store_true");group.add_argument("--final-source",action="store_true")
    args=parser.parse_args(argv)
    try:
        result=source_stage() if args.source else prelabel_stage() if args.prelabel else metric_stage() if args.metric else final_source_stage()
    except Exception as exc:
        print(json.dumps({"status":"FAIL","error":f"{type(exc).__name__}: {exc}"},sort_keys=True));return 1
    print(json.dumps(result,sort_keys=True,ensure_ascii=False));return 0


if __name__=="__main__":
    raise SystemExit(main())
