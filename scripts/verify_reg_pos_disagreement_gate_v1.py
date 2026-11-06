"""Independent target-free and post-score metric verification for REG POS gate V1."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys
import zipfile
from typing import Any, Mapping

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
LANE = "REG_POS_DISAGREEMENT_GATE_V1"
CARD_ID = LANE
REPORT_ROOT = ROOT / "reports" / "optimization_v3" / LANE
ARTIFACT_ROOT = ROOT / "experiments" / "runs" / LANE
RECIPE_PATH = REPORT_ROOT / "RECIPE.json"
ROOT_PLAN_PATH = REPORT_ROOT / "ROOT_PLAN.md"
ROOT_AMENDMENT_PATH = REPORT_ROOT / "ROOT_FLOAT_ORDER_AMENDMENT_R2.md"
SOURCE_FREEZE_R1 = ARTIFACT_ROOT / "SOURCE_FREEZE_R1.json"
SOURCE_FREEZE = ARTIFACT_ROOT / "SOURCE_FREEZE_R2.json"
PREDICTION_FREEZE = ARTIFACT_ROOT / "PREDICTION_FREEZE_R2.json"
PRELABEL_QA_PATH = REPORT_ROOT / "PRELABEL_QA_R2.json"
SCORE_PATH = REPORT_ROOT / "SCORE_R2.json"
METRIC_QA_PATH = REPORT_ROOT / "METRIC_QA_R3.json"
VERIFIER_AMENDMENT_PATH = REPORT_ROOT / "METRIC_QA_VERIFIER_AMENDMENT_R3.json"
GATE_RUNNER_PATH = ROOT / "scripts" / "reg_pos_disagreement_gate_v1.py"
HELPER_PATH = ROOT / "scripts" / "reg_pos_poisson_bag2_v1.py"
FRONTIER_ROOT = Path(r"C:\Users\Sam\OneDrive\Documents\KASDAD\tk2-regression-frontier")
RUNNER_PATH = FRONTIER_ROOT / "scripts" / "reg_position_direct_l1_v1.py"
SCORE_WINDOWS = {67: (72, 75), 75: (76, 79)}
SCALE_WINDOWS = {"D1": (68, 71), "D2": (68, 75)}
MIN_POSITIVE_DISAGREEMENTS = 200
EXPECTED_POOLED_ROWS = 15953


class VerifyError(RuntimeError):
    pass


def _load_module(path: Path, name: str):
    if not path.is_file():
        raise VerifyError(f"Required source file missing: {path}")
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise VerifyError(f"Cannot import frozen source: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _helpers():
    bag2 = _load_module(HELPER_PATH, "reg_pos_poisson_bag2_v1_for_gate_qa")
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


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise VerifyError(f"Invalid JSON: {path}") from exc
    if not isinstance(value, dict):
        raise VerifyError(f"Expected a JSON object: {path}")
    return value


def _read_npz(path: Path, keys: tuple[str, ...]) -> dict[str, np.ndarray]:
    if not path.is_file():
        raise VerifyError(f"Required NPZ missing: {path}")
    with zipfile.ZipFile(path) as archive:
        found = {member[:-4] for member in archive.namelist() if member.endswith(".npy")}
    if found != set(keys):
        raise VerifyError(f"NPZ allowlist mismatch at {path}: expected {sorted(keys)}, got {sorted(found)}")
    with np.load(path, allow_pickle=False) as archive:
        return {key: np.asarray(archive[key]).copy() for key in keys}


def _array_equal(left: Any, right: Any) -> bool:
    a, b = np.asarray(left), np.asarray(right)
    if a.dtype.kind in "fc" or b.dtype.kind in "fc":
        return bool(np.array_equal(a, b, equal_nan=True))
    return bool(np.array_equal(a, b))


def _verify_signed(value: Mapping[str, Any], signature_key: str) -> None:
    body = {key: item for key, item in value.items() if key != signature_key}
    if value.get(signature_key) != _canonical_sha(body):
        raise VerifyError(f"Signature mismatch: {signature_key}")


def _write_signed(path: Path, body: Mapping[str, Any], signature_key: str) -> dict[str, Any]:
    value = dict(body)
    value[signature_key] = _canonical_sha(value)
    encoded = _canonical(value) + b"\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != encoded:
            raise VerifyError(f"Refusing to overwrite changed verification report: {path}")
        return value
    path.write_bytes(encoded)
    return value


def _verify_verifier_amendment(freeze: Mapping[str, Any]) -> dict[str, Any]:
    amendment = _read_json(VERIFIER_AMENDMENT_PATH)
    _verify_signed(amendment, "amendment_signature")
    code_hashes = freeze.get("source_code_hashes", {})
    test_path = ROOT / "tests" / "test_metric_qa_verifier_repair_r3.py"
    if (amendment.get("status") != "VERIFIER_AMENDED_R3"
            or amendment.get("source_freeze_sha256") != _sha_file(SOURCE_FREEZE)
            or amendment.get("source_signature") != freeze.get("source_signature")
            or amendment.get("prediction_freeze_sha256") != _sha_file(PREDICTION_FREEZE)
            or amendment.get("prelabel_qa_sha256") != _sha_file(PRELABEL_QA_PATH)
            or amendment.get("score_sha256") != _sha_file(SCORE_PATH)
            or amendment.get("old_verifier_sha256") != code_hashes.get("independent_verifier")
            or amendment.get("new_verifier_sha256") != _sha_file(Path(__file__))
            or amendment.get("targeted_regression_test_sha256") != _sha_file(test_path)):
        raise VerifyError("R3 verifier amendment does not bind the frozen score QA inputs and implementation")
    expected_fix = {
        "old_reference": "bag2._contained_file",
        "new_reference": "runner._contained_file",
        "call_sites": ["Train path hash check", "authorized query-target read"],
        "scope": "metric verifier only; preserve R2 source, predictions, prelabel and score",
    }
    if amendment.get("helper_reference_fix") != expected_fix:
        raise VerifyError("R3 verifier amendment describes an unexpected code change")
    return amendment


def _train_path(bag2: Any, runner: Any) -> Path:
    resolver = getattr(runner, "_contained_file", None)
    if not callable(resolver):
        raise VerifyError("Runner does not expose the required contained-file resolver")
    return Path(resolver(bag2.SOURCE_ROOT, runner.TRAIN_REL))


def _source(bag2: Any, runner: Any) -> dict[str, Any]:
    freeze = _read_json(SOURCE_FREEZE)
    _verify_signed(freeze, "source_signature")
    original = _read_json(SOURCE_FREEZE_R1)
    _verify_signed(original, "source_signature")
    if (freeze.get("status") != "SOURCE_AMENDED_R2"
            or original.get("status") != "SOURCE_FROZEN"
            or freeze.get("source_signature_r1") != original.get("source_signature")
            or freeze.get("source_freeze_r1_sha256") != _sha_file(SOURCE_FREEZE_R1)):
        raise VerifyError("R2 source amendment does not bind preserved R1 source")
    if (_sha_file(RECIPE_PATH) != freeze.get("recipe_sha256")
            or _sha_file(ROOT_PLAN_PATH) != freeze.get("root_plan_sha256")
            or _sha_file(ROOT_AMENDMENT_PATH) != freeze.get("root_amendment_sha256")):
        raise VerifyError("Recipe/root plan differs from immutable source freeze")
    if bag2._source_artifact_hashes(runner) != freeze.get("source_artifacts"):
        raise VerifyError("Original source/cache hashes differ from the frozen source")
    amendment = _verify_verifier_amendment(freeze)
    code = {"runner": RUNNER_PATH, "bag2_helper": HELPER_PATH,
            "gate_runner": GATE_RUNNER_PATH, "independent_verifier": Path(__file__),
            "regression_tests": ROOT / "tests" / "test_reg_pos_disagreement_gate_v1.py",
            "recipe": RECIPE_PATH, "root_plan": ROOT_PLAN_PATH,
            "root_amendment": ROOT_AMENDMENT_PATH,
            "hinge_runner": bag2.HINGE_RUNNER,
            "final_hinge": ROOT / "experiments" / "runs" / "reg_monotone_hinge_p50_v1" / "FINAL_APPLICATION_R1" / "fits" / "HINGE_FINAL.json",
            "final_hinge_freeze": ROOT / "experiments" / "reg_monotone_hinge_p50_v1" / "FINAL_APPLICATION_FREEZE_R1.json"}
    actual_hashes = {name: _sha_file(path) for name, path in code.items()}
    frozen_hashes = freeze.get("source_code_hashes", {})
    actual_non_verifier_hashes = {key: value for key, value in actual_hashes.items()
                                  if key != "independent_verifier"}
    frozen_non_verifier_hashes = {key: value for key, value in frozen_hashes.items()
                                  if key != "independent_verifier"}
    if (actual_non_verifier_hashes != frozen_non_verifier_hashes
            or amendment.get("old_verifier_sha256") != frozen_hashes.get("independent_verifier")):
        raise VerifyError("Frozen source implementation or hinge files changed")
    if _sha_file(ROOT / "experiments" / "runs" / "reg_monotone_hinge_p50_v1" / "FINAL_APPLICATION_R1" / "fits" / "HINGE_FINAL.json") != freeze["final_hinge_hashes"]["fit"]:
        raise VerifyError("Frozen deployed final hinge hash changed")
    if _sha_file(code["final_hinge_freeze"]) != freeze["final_hinge_hashes"]["freeze"]:
        raise VerifyError("Frozen deployed final hinge application freeze changed")
    return freeze


def _read_inputs(source: Mapping[str, Any]) -> dict[int, dict[str, np.ndarray]]:
    output = {}
    for origin in (67, 75):
        receipt = source["origins"][str(origin)]["archive"]
        path = ROOT / receipt["path"]
        if _sha_file(path) != receipt["sha256"]:
            raise VerifyError(f"Source input archive hash changed at E{origin}")
        arrays = _read_npz(path, tuple(receipt["members"]))
        if {key: _sha_array(value) for key, value in arrays.items()} != receipt["members"]:
            raise VerifyError(f"Source input member digest changed at E{origin}")
        if (not np.array_equal(arrays["query_rows"].astype(np.int64), np.sort(arrays["query_rows"].astype(np.int64)))
                or len(np.unique(arrays["query_rows"])) != len(arrays["query_rows"])):
            raise VerifyError(f"Source query physical rows are duplicated or unsorted at E{origin}")
        output[origin] = arrays
    return output


def _prediction_and_scale_files(prediction: Mapping[str, Any]) -> tuple[dict[str, dict[str, np.ndarray]], dict[str, dict[str, Any]]]:
    predictions = {}
    states = {}
    for key, receipt in prediction.get("predictions", {}).items():
        path = ROOT / receipt["path"]
        if _sha_file(path) != receipt["sha256"]:
            raise VerifyError(f"Prediction archive hash changed: {key}")
        arrays = _read_npz(path, tuple(receipt["members"]))
        if {name: _sha_array(value) for name, value in arrays.items()} != receipt["members"]:
            raise VerifyError(f"Prediction array hash changed: {key}")
        predictions[key] = arrays
    for name, receipt in prediction.get("states", {}).items():
        path = ROOT / receipt["path"]
        state = _read_json(path)
        _verify_signed(state, "state_signature")
        if state.get("state_signature") != receipt.get("state_signature") or _sha_file(path) != receipt.get("sha256"):
            raise VerifyError(f"Scale state receipt mismatch: {name}")
        states[name] = state
    if set(predictions) != {"REAL_D1", "CONTROL_D1", "REAL_D2", "CONTROL_D2"} or set(states) != {"D1", "D2"}:
        raise VerifyError("Prediction freeze does not contain exactly two states and four policy outputs")
    return predictions, states


def _expected_state(data: Mapping[str, np.ndarray], name: str) -> dict[str, Any]:
    low, high = SCALE_WINDOWS[name]
    cycles = data["cycles"].astype(np.int64)
    missing = data["missing"].astype(bool)
    delta = np.abs(data["pos_t50"].astype(np.float64) - data["baseline"].astype(np.float64))
    select = (cycles >= low) & (cycles <= high) & ~missing & (delta > 0.0)
    values = delta[select]
    if len(values) < MIN_POSITIVE_DISAGREEMENTS:
        status, scale = "UNSUPPORTED", None
    else:
        status, scale = "FIT_COMPLETE", float(np.median(values))
    return {"status": status, "scale_s": scale, "positive_selection_disagreements": int(len(values)),
            "selection_rows_sha256": _sha_array(data["query_rows"][select]),
            "selection_ids_sha256": _sha_array(data["query_ids"][select]),
            "disagreement_sha256": _sha_array(values)}


def prelabel_stage() -> dict[str, Any]:
    bag2, runner = _helpers()
    source = _source(bag2, runner)
    prediction = _read_json(PREDICTION_FREEZE)
    _verify_signed(prediction, "prediction_signature")
    if (prediction.get("status") != "PREDICTIONS_FROZEN"
            or prediction.get("source_signature") != source.get("source_signature")):
        raise VerifyError("Prediction freeze is not linked to the immutable source freeze")
    if prediction.get("states") != source.get("scale_states"):
        raise VerifyError("Prediction freeze does not reuse the exact source scale-state receipts")
    inputs = _read_inputs(source)
    outputs, states = _prediction_and_scale_files(prediction)
    hinge = bag2._hinge()
    checks = {"source_freeze_signature": True, "source_artifact_hashes": True,
              "explicit_npz_allowlists": True, "target_column_read": False,
              "query_labels_read": False, "test_sample_accessed": False,
              "scale_states_reconstructed": True, "policy_predictions_reconstructed": True,
              "missing_fallback_exact": True, "score_rows_unique": True}
    for name, origin in (("D1", 67), ("D2", 75)):
        expected = _expected_state(inputs[67], name)
        state = states[name]
        if (state.get("status") != expected["status"]
                or state.get("positive_selection_disagreements") != expected["positive_selection_disagreements"]
                or state.get("selection_rows_sha256") != expected["selection_rows_sha256"]
                or state.get("selection_ids_sha256") != expected["selection_ids_sha256"]
                or state.get("disagreement_sha256") != expected["disagreement_sha256"]
                or state.get("scale_s") != expected["scale_s"]):
            raise VerifyError(f"Independent scalar state reconstruction differs: {name}")
        data = inputs[origin]
        cycles = data["cycles"].astype(np.int64)
        missing = data["missing"].astype(bool)
        b = data["baseline"].astype(np.float64)
        p = data["pos_t50"].astype(np.float64)
        d = np.abs(p - b)
        if expected["status"] == "FIT_COMPLETE":
            w = 0.25 * np.exp(-d / float(expected["scale_s"]))
        else:
            w = np.full(len(d), 0.25, dtype=np.float64)
        wc = np.full(len(d), 0.25, dtype=np.float64)
        cohort_n = np.zeros(len(d), dtype=np.int64)
        for cycle in np.unique(cycles):
            cmask = cycles == cycle
            present = cmask & ~missing
            cohort_n[cmask] = int(present.sum())
            if present.any():
                wc[cmask] = float(np.mean(w[present]))
        expected_champion = data["champion_direct"].astype(np.float64).copy()
        parent_order = np.clip(0.75 * b + 0.25 * p, 0.0, 100.0)
        parent_order[missing] = b[missing]
        if not np.array_equal(parent_order, expected_champion):
            raise VerifyError(f"Parent champion arithmetic order differs at E{origin}")
        if expected["status"] == "UNSUPPORTED":
            expected_real = expected_champion.copy()
            expected_control = expected_champion.copy()
        else:
            expected_real = np.clip(b + w * (p - b), 0.0, 100.0)
            expected_control = np.clip(b + wc * (p - b), 0.0, 100.0)
            expected_real[missing] = b[missing]
            expected_control[missing] = b[missing]
        params = source["origins"][str(origin)]["hinge_parameters"]
        champion_hinge = hinge.apply_hinge_present_only(expected_champion, missing, params)
        if not np.array_equal(champion_hinge, data["champion_hinge"]):
            raise VerifyError(f"Frozen champion hinge did not reconstruct at E{origin}")
        if expected["status"] == "UNSUPPORTED":
            real_hinge = champion_hinge.copy()
            control_hinge = champion_hinge.copy()
        else:
            real_hinge = hinge.apply_hinge_present_only(expected_real, missing, params)
            control_hinge = hinge.apply_hinge_present_only(expected_control, missing, params)
        score_mask = (cycles >= SCORE_WINDOWS[origin][0]) & (cycles <= SCORE_WINDOWS[origin][1])
        for policy in ("REAL", "CONTROL"):
            saved = outputs[f"{policy}_{name}"]
            selected_direct = expected_real if policy == "REAL" else expected_control
            selected_hinge = real_hinge if policy == "REAL" else control_hinge
            selected_weight = w if policy == "REAL" else wc
            expected_members = {
                "query_rows": data["query_rows"], "query_ids": data["query_ids"],
                "cycles": cycles, "score_mask": score_mask, "missing": missing,
                "baseline": b, "pos_t50": p, "disagreement": d,
                "scale_s": np.asarray([float(expected["scale_s"]) if expected["scale_s"] is not None else float("nan")]),
                "real_weight": w, "control_weight": wc,
                "control_present_cohort_n": cohort_n,
                "champion_direct": expected_champion,
                "champion_hinge": champion_hinge,
                "real_direct": expected_real, "control_direct": expected_control,
                "real_hinge": real_hinge, "control_hinge": control_hinge,
                "impact_quartile": data["impact_quartile"], "text_quartile": data["text_quartile"],
                "entry_friction": data["entry_friction"],
                "control_definition": np.asarray(["cycle_mean_real_weight_present_only"]),
                "selected_direct": selected_direct, "selected_hinge": selected_hinge,
                "selected_weight": selected_weight,
            }
            if set(saved) != set(expected_members):
                raise VerifyError(f"Policy output member set differs: {policy}_{name}")
            for key, value in expected_members.items():
                if not _array_equal(saved[key], value):
                    raise VerifyError(f"Independent policy reconstruction differs: {policy}_{name}/{key}")
            if not np.array_equal(saved["selected_hinge"][missing].view("<u8"), b[missing].view("<u8")):
                raise VerifyError(f"Missing fallback bit identity failed: {policy}_{name}")
        checks["missing_fallback_exact"] = checks["missing_fallback_exact"] and True
    d1 = outputs["REAL_D1"]; d2 = outputs["REAL_D2"]
    rows1 = d1["query_rows"][d1["score_mask"].astype(bool)]
    rows2 = d2["query_rows"][d2["score_mask"].astype(bool)]
    if len(rows1) + len(rows2) != EXPECTED_POOLED_ROWS or set(rows1.tolist()) & set(rows2.tolist()):
        raise VerifyError("Independent score population or disjointness check failed")
    body = {"schema_version": 1, "experiment_id": CARD_ID, "status": "PASS",
            "source_signature": source["source_signature"],
            "prediction_signature": prediction["prediction_signature"],
            "source_freeze_sha256": _sha_file(SOURCE_FREEZE),
            "prediction_freeze_sha256": _sha_file(PREDICTION_FREEZE),
            "query_targets_read": False, "query_target_rows_read": 0,
            "test_or_sample_accessed": False, "csv_created": False,
            "checks": checks,
            "scale_states": {name: {"status": states[name]["status"],
                                    "positive_disagreements": states[name]["positive_selection_disagreements"],
                                    "selection_cycles": list(SCALE_WINDOWS[name]),
                                    "scale_s": states[name]["scale_s"]} for name in ("D1", "D2")},
            "score_windows": {"D1": {"rows": len(rows1), "cycles": [72, 75]},
                              "D2": {"rows": len(rows2), "cycles": [76, 79]}},
            "unique_primary_rows": len(rows1) + len(rows2),
            "unique_row_identity_sha256": _canonical_sha({"D1": _sha_array(rows1), "D2": _sha_array(rows2)})}
    qa = _write_signed(PRELABEL_QA_PATH, body, "qa_signature")
    return {"status": qa["status"], "qa_signature": qa["qa_signature"],
            "query_targets_read": False, "unique_primary_rows": qa["unique_primary_rows"]}


def _metric(y: np.ndarray, p: np.ndarray) -> dict[str, Any]:
    err = np.asarray(p, dtype=np.float64) - np.asarray(y, dtype=np.float64)
    return {"rows": int(len(err)), "mae": float(np.mean(np.abs(err))),
            "p90_absolute_error": float(np.quantile(np.abs(err), 0.90)),
            "bias_prediction_minus_target": float(np.mean(err))}


def _close(left: Any, right: Any) -> bool:
    if isinstance(left, bool) or isinstance(right, bool):
        return left is right
    if isinstance(left, (int, np.integer)) and isinstance(right, (int, np.integer)):
        return int(left) == int(right)
    try:
        return math.isclose(float(left), float(right), rel_tol=0.0, abs_tol=1e-12)
    except (TypeError, ValueError):
        return left == right


def _assert_metric_match(observed: Mapping[str, Any], expected: Mapping[str, Any], label: str) -> None:
    if set(observed) != set(expected) or any(not _close(observed[key], expected[key]) for key in expected):
        raise VerifyError(f"Metric reconstruction differs: {label}")


def synthetic_metric_qa_stage() -> dict[str, Any]:
    bag2, runner = _helpers()
    train_path = _train_path(bag2, runner)
    if not train_path.is_file():
        raise VerifyError("Synthetic metric QA could not resolve the real Train path")
    rng = np.random.default_rng(20261106)
    target = rng.uniform(0.0, 100.0, 512)
    champion = np.clip(target + rng.normal(0.0, 9.0, len(target)), 0.0, 100.0)
    real = np.clip(target + rng.normal(0.0, 7.0, len(target)), 0.0, 100.0)
    control = np.clip(target + rng.normal(0.0, 8.0, len(target)), 0.0, 100.0)
    for name, prediction in (("champion", champion), ("REAL", real), ("CONTROL", control)):
        error = prediction - target
        expected = {"rows": int(len(target)), "mae": float(np.mean(np.abs(error))),
                    "p90_absolute_error": float(np.quantile(np.abs(error), 0.90)),
                    "bias_prediction_minus_target": float(np.mean(error))}
        _assert_metric_match(_metric(target, prediction), expected, f"synthetic/{name}")
    return {"status": "PASS", "synthetic_rows": int(len(target)),
            "train_path_resolved": True, "train_data_read": False,
            "query_labels_read": False, "test_or_sample_accessed": False}


def metric_stage() -> dict[str, Any]:
    bag2, runner = _helpers()
    source = _source(bag2, runner)
    prediction = _read_json(PREDICTION_FREEZE)
    _verify_signed(prediction, "prediction_signature")
    prelabel = _read_json(PRELABEL_QA_PATH)
    _verify_signed(prelabel, "qa_signature")
    if (prelabel.get("status") != "PASS" or prelabel.get("query_targets_read") is not False
            or prelabel.get("source_signature") != source.get("source_signature")
            or prelabel.get("prediction_signature") != prediction.get("prediction_signature")):
        raise VerifyError("Metric QA cannot proceed without matching independent prelabel PASS")
    score = _read_json(SCORE_PATH)
    _verify_signed(score, "score_signature")
    if (score.get("status") != "SCORED"
            or score.get("source_signature") != source.get("source_signature")
            or score.get("prediction_signature") != prediction.get("prediction_signature")
            or score.get("prelabel_qa_sha256") != _sha_file(PRELABEL_QA_PATH)):
        raise VerifyError("Score report signature or source binding failed")
    train_path = _train_path(bag2, runner)
    if _sha_file(train_path) != source["official_train_sha256"]:
        raise VerifyError("Official Train changed after score freeze")
    outputs, _ = _prediction_and_scale_files(prediction)
    target_by_window: dict[str, np.ndarray] = {}
    window_report = {}
    real_parts, control_parts, champion_parts, all_rows = [], [], [], []
    real_group_harms: list[float] = []
    fixed_groups_known = True
    fixed_group_names = {"missing", "target_lt5", "target_ge95", "EF0", "EF1"}
    for name, origin in (("D1", 67), ("D2", 75)):
        arrays = outputs[f"REAL_{name}"]
        score_mask = arrays["score_mask"].astype(bool)
        rows = arrays["query_rows"][score_mask].astype(np.int64)
        ids = arrays["query_ids"][score_mask].astype(str)
        if len(set(rows.tolist())) != len(rows) or not np.array_equal(rows, np.sort(rows)):
            raise VerifyError(f"Metric QA row population invalid for {name}")
        # The prelabel report is PASS, and only these authorized Train target cells are read here.
        y = runner._read_targets_for_rows(train_path,
                                          rows, ids, expected_total_rows=50608)
        target_by_window[name] = y
        all_rows.extend(rows.tolist())
        local = {key: value[score_mask] for key, value in arrays.items()
                 if value.ndim > 0 and len(value) == len(score_mask)}
        policies = {"REAL": (local["real_direct"], local["real_hinge"]),
                    "CONTROL": (local["control_direct"], local["control_hinge"])}
        champ_d, champ_h = local["champion_direct"], local["champion_hinge"]
        groups = {"missing": local["missing"].astype(bool),
                  "present": ~local["missing"].astype(bool),
                  "target_lt5": y < 5.0, "target_ge95": y >= 95.0,
                  "EF0": local["entry_friction"] == 0.0, "EF1": local["entry_friction"] > 0.0}
        for feature, key in (("impact_load", "impact_quartile"), ("text_load", "text_quartile")):
            for bucket in range(4):
                groups[f"{feature}_Q{bucket + 1}"] = local[key] == bucket
        group_report = {}
        for policy, (direct, hinged) in policies.items():
            for group_name, mask in groups.items():
                n = int(mask.sum())
                if n < 100:
                    entry = {"status": "UNKNOWN", "rows": n, "harm": None}
                    if policy == "REAL" and group_name in fixed_group_names:
                        fixed_groups_known = False
                else:
                    policy_mae = _metric(y[mask], hinged[mask])["mae"]
                    champion_mae = _metric(y[mask], champ_h[mask])["mae"]
                    harm = policy_mae - champion_mae
                    entry = {"status": "MEASURED", "rows": n,
                             "policy_mae": policy_mae, "champion_mae": champion_mae,
                             "harm": float(harm)}
                    if policy == "REAL":
                        real_group_harms.append(float(harm))
                group_report.setdefault(policy, {})[group_name] = entry
        cycles_report = {}
        for cycle in np.unique(local["cycles"]):
            mask = local["cycles"] == cycle
            cycles_report[str(int(cycle))] = {
                "rows": int(mask.sum()), "champion": _metric(y[mask], champ_h[mask]),
                "REAL": _metric(y[mask], policies["REAL"][1][mask]),
                "CONTROL": _metric(y[mask], policies["CONTROL"][1][mask])}
        def gain(prediction_vector: np.ndarray) -> float:
            return _metric(y, champ_h)["mae"] - _metric(y, prediction_vector)["mae"]
        expected_window = {
            "origin": origin, "cycles": list(SCORE_WINDOWS[origin]), "rows": len(y),
            "row_identity_sha256": _canonical_sha({"rows": _sha_array(rows), "ids": _sha_array(ids),
                                                   "cycles": _sha_array(local["cycles"])}),
            "champion_direct": _metric(y, champ_d), "champion_hinge": _metric(y, champ_h),
            "REAL_direct": _metric(y, policies["REAL"][0]), "REAL_hinge": _metric(y, policies["REAL"][1]),
            "CONTROL_direct": _metric(y, policies["CONTROL"][0]), "CONTROL_hinge": _metric(y, policies["CONTROL"][1]),
            "REAL_MAE_gain": gain(policies["REAL"][1]), "CONTROL_MAE_gain": gain(policies["CONTROL"][1]),
            "REAL_over_CONTROL_gain": _metric(y, policies["CONTROL"][1])["mae"] - _metric(y, policies["REAL"][1])["mae"],
            "per_cycle": cycles_report, "groups": group_report,
            "query_target_rows_read": len(y), "test_or_sample_accessed": False}
        observed_window = score["windows"][name]
        if set(observed_window) != set(expected_window):
            raise VerifyError(f"Score report window schema differs: {name}")
        for key, expected in expected_window.items():
            if isinstance(expected, dict):
                if set(observed_window[key]) != set(expected):
                    raise VerifyError(f"Score report sub-schema differs: {name}/{key}")
                if key == "per_cycle":
                    for cycle, metrics in expected.items():
                        for metric_name, metric_value in metrics.items():
                            if isinstance(metric_value, dict):
                                _assert_metric_match(observed_window[key][cycle][metric_name], metric_value,
                                                     f"{name}/{key}/{cycle}/{metric_name}")
                            else:
                                if not _close(observed_window[key][cycle][metric_name], metric_value):
                                    raise VerifyError(f"Score report differs: {name}/{key}/{cycle}/{metric_name}")
                elif key == "groups":
                    for policy, groups_for_policy in expected.items():
                        for group_name, entry in groups_for_policy.items():
                            observed_entry = observed_window[key][policy][group_name]
                            if observed_entry.get("status") != entry["status"] or observed_entry.get("rows") != entry["rows"]:
                                raise VerifyError(f"Score report group support differs: {name}/{policy}/{group_name}")
                            if entry["status"] == "MEASURED":
                                for metric_key in ("policy_mae", "champion_mae", "harm"):
                                    if not _close(observed_entry[metric_key], entry[metric_key]):
                                        raise VerifyError(f"Score report group metric differs: {name}/{policy}/{group_name}/{metric_key}")
                else:
                    _assert_metric_match(observed_window[key], expected, f"{name}/{key}")
            elif not _close(observed_window[key], expected):
                raise VerifyError(f"Score report value differs: {name}/{key}")
        window_report[name] = expected_window
        real_parts.append(policies["REAL"][1]); control_parts.append(policies["CONTROL"][1]); champion_parts.append(champ_h)
    if len(all_rows) != EXPECTED_POOLED_ROWS or len(set(all_rows)) != EXPECTED_POOLED_ROWS:
        raise VerifyError("Metric QA unique primary row count failed")
    pooled_y = np.concatenate([target_by_window[name] for name in ("D1", "D2")])
    pooled_champ = np.concatenate(champion_parts)
    pooled_real = np.concatenate(real_parts)
    pooled_control = np.concatenate(control_parts)
    pooled_metrics = {"pooled_champion_hinge": _metric(pooled_y, pooled_champ),
                      "pooled_REAL_hinge": _metric(pooled_y, pooled_real),
                      "pooled_CONTROL_hinge": _metric(pooled_y, pooled_control)}
    for key, expected in pooled_metrics.items():
        _assert_metric_match(score[key], expected, key)
    pooled_gain = pooled_metrics["pooled_champion_hinge"]["mae"] - pooled_metrics["pooled_REAL_hinge"]["mae"]
    pooled_control_gain = pooled_metrics["pooled_champion_hinge"]["mae"] - pooled_metrics["pooled_CONTROL_hinge"]["mae"]
    real_over_control = pooled_metrics["pooled_CONTROL_hinge"]["mae"] - pooled_metrics["pooled_REAL_hinge"]["mae"]
    if (not _close(score["pooled_REAL_MAE_gain"], pooled_gain)
            or not _close(score["pooled_CONTROL_MAE_gain"], pooled_control_gain)
            or not _close(score["pooled_REAL_over_CONTROL_gain"], real_over_control)):
        raise VerifyError("Pooled gain metrics differ from independent reconstruction")
    d1_gain, d2_gain = window_report["D1"]["REAL_MAE_gain"], window_report["D2"]["REAL_MAE_gain"]
    max_harm = max(real_group_harms, default=float("inf"))
    required_groups_known = fixed_groups_known
    harm_pass = required_groups_known and max_harm <= 0.25
    missing_exact = all(
        np.array_equal(outputs[f"REAL_{name}"]["real_hinge"][outputs[f"REAL_{name}"]["missing"]].view("<u8"),
                       outputs[f"REAL_{name}"]["baseline"][outputs[f"REAL_{name}"]["missing"]].view("<u8"))
        and np.array_equal(outputs[f"CONTROL_{name}"]["control_hinge"][outputs[f"CONTROL_{name}"]["missing"]].view("<u8"),
                           outputs[f"CONTROL_{name}"]["baseline"][outputs[f"CONTROL_{name}"]["missing"]].view("<u8"))
        for name in ("D1", "D2"))
    scientific = bool(pooled_gain >= 0.02 and d1_gain >= 0 and d2_gain >= 0
                      and real_over_control > 0 and harm_pass and missing_exact)
    exploratory = bool((pooled_gain > 0 or (d2_gain >= 0.01 and pooled_gain >= -0.01))
                       and d1_gain >= -0.05 and harm_pass and missing_exact)
    if (score.get("pooled_rows") != EXPECTED_POOLED_ROWS
            or score.get("pooled_row_identity_sha256") != _sha_array(np.asarray(all_rows, dtype=np.int64))
            or score.get("fixed_groups_known") is not fixed_groups_known
            or score.get("missing_invariant_pass") is not missing_exact
            or score.get("scientific_performance_gate") is not scientific
            or score.get("exploratory_performance_gate") is not exploratory):
        raise VerifyError("Score report decision-support fields differ from independent metric QA")
    body = {"schema_version": 1, "experiment_id": CARD_ID, "status": "PASS",
            "source_signature": source["source_signature"],
            "verifier_amendment_sha256": _sha_file(VERIFIER_AMENDMENT_PATH),
            "active_verifier_sha256": _sha_file(Path(__file__)),
            "prediction_signature": prediction["prediction_signature"],
            "prelabel_qa_signature": prelabel["qa_signature"],
            "score_signature": score["score_signature"],
            "source_train_sha256": source["official_train_sha256"],
            "pooled_rows": len(all_rows), "metrics_reproduced": True,
            "independent_target_rows_read": EXPECTED_POOLED_ROWS,
            "target_values_persisted": False, "test_or_sample_accessed": False,
            "fixed_groups_known": fixed_groups_known,
            "maximum_supported_REAL_group_harm": max_harm if math.isfinite(max_harm) else None,
            "missing_invariant_pass": bool(missing_exact),
            "pooled_REAL_MAE_gain": pooled_gain, "D1_REAL_MAE_gain": d1_gain,
            "D2_REAL_MAE_gain": d2_gain, "pooled_REAL_over_CONTROL_gain": real_over_control,
            "scientific_qualification": scientific,
            "exploratory_benchmark_eligible": exploratory,
            "scientific_fail_remains_distinct_from_exploratory_eligibility": True,
            "evaluation_disclosure": score["evaluation_disclosure"],
            "metrics": pooled_metrics}
    qa = _write_signed(METRIC_QA_PATH, body, "qa_signature")
    return {"status": qa["status"], "qa_signature": qa["qa_signature"],
            "scientific_qualification": scientific,
            "exploratory_benchmark_eligible": exploratory,
            "pooled_REAL_MAE_gain": pooled_gain}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--prelabel", action="store_true")
    group.add_argument("--metrics", action="store_true")
    group.add_argument("--synthetic-metrics", action="store_true")
    args = parser.parse_args(argv)
    try:
        if args.prelabel:
            result = prelabel_stage()
        elif args.metrics:
            result = metric_stage()
        else:
            result = synthetic_metric_qa_stage()
    except Exception as exc:
        print(json.dumps({"status": "FAIL", "error": f"{type(exc).__name__}: {exc}"}, sort_keys=True))
        return 1
    print(json.dumps(_jsonable(result), sort_keys=True, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
