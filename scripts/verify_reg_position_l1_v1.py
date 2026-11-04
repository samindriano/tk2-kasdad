"""Independent integrity, metric and eligibility verifier for REG_POSITION_L1_V1.

The prelabel phase verifies frozen bytes, durable artifacts, row identity and the
fixed residual formula without opening Regression query targets. The final phase
recomputes the metrics and gates only after that prelabel receipt exists.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import sys
from typing import Any

import joblib
import numpy as np
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

CARD_ID = "REG_POSITION_L1_V1"
ORIGINS = (67, 75)
ARMS = ("BAG", "REAL", "SHUFFLED")
FIT_IDS = tuple(f"{CARD_ID}_E{origin}_{arm}" for origin in ORIGINS for arm in ARMS)
REPORT_REL = Path("reports/optimization_v3/reg_position_l1_v1")
RUN_REL = Path("experiments/runs/reg_position_l1_v1")
TRAIN_REL = Path("data/raw/regression/Train.csv")
LEDGER_CAP = 6
MIN_GROUP_N = 100
GROUP_HARM_MAX = 0.25
WINDOW_HARM_MAX = 0.10
PRIMARY_GAIN_MIN = 0.03
MECHANISM_GAIN_MIN = 0.01


class VerificationError(RuntimeError):
    pass


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"),
                      allow_nan=False).encode("utf-8")


def _canonical_sha(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha_array(value: Any, *, dtype: str | None = None) -> str:
    array = np.asarray(value, dtype=dtype)
    digest = hashlib.sha256(_canonical({"shape": array.shape, "dtype": str(array.dtype)}))
    if array.dtype.kind in "OUS":
        digest.update(_canonical(array.astype(str).tolist()))
    else:
        digest.update(np.ascontiguousarray(array).tobytes(order="C"))
    return digest.hexdigest()


def _contained(root: Path, relative: str | Path) -> Path:
    root = root.resolve()
    path = (root / relative).resolve()
    if root not in path.parents or not path.is_file():
        raise VerificationError(f"Required input escapes its root or is missing: {relative}")
    return path


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise VerificationError(f"Expected a JSON object: {path}")
    return value


def _freeze(source_root: Path, output_root: Path) -> dict[str, Any]:
    freeze_path = output_root / REPORT_REL / "FREEZE.json"
    freeze = _json(freeze_path)
    body = {key: value for key, value in freeze.items() if key != "freeze_signature"}
    if (freeze.get("card_id") != CARD_ID or freeze.get("status") != "FROZEN_BEFORE_OFFICIAL_FITS"
            or freeze.get("freeze_signature") != _canonical_sha(body)
            or freeze.get("fit_ids_order") != list(FIT_IDS)
            or freeze.get("accounting", {}).get("pilot_cap") != LEDGER_CAP
            or freeze.get("accounting", {}).get("retries") != 0
            or freeze.get("boundaries", {}).get("query_targets_used_for_fit") is not False):
        raise VerificationError("Freeze signature, fit order, accounting or boundary attestation is invalid")
    paths: dict[str, Path] = {}
    for key in freeze.get("source_hashes", {}):
        if not key.startswith("worktree/"):
            raise VerificationError("Unexpected worktree source hash key")
        paths[key] = _contained(ROOT, key.removeprefix("worktree/"))
    for key in freeze.get("input_hashes", {}):
        if not key.startswith("main/"):
            raise VerificationError("Unexpected main input hash key")
        paths[key] = _contained(source_root, key.removeprefix("main/"))
    expected = {**freeze.get("source_hashes", {}), **freeze.get("input_hashes", {})}
    actual = {key: _sha_file(path) for key, path in sorted(paths.items())}
    if actual != expected:
        changed = [key for key in sorted(set(actual) | set(expected)) if actual.get(key) != expected.get(key)]
        raise VerificationError(f"Frozen code or input bytes changed: {changed[:8]}")
    if freeze.get("readiness_audit_sha256") != _sha_file(ROOT / "reports/optimization_v3/regression_position_frontier_v1/READINESS_AND_ERROR_AUDIT.json"):
        raise VerificationError("Zero-fit readiness audit changed after freeze")
    return freeze


def _ledger(output_root: Path, freeze: dict[str, Any]) -> tuple[list[dict[str, Any]], str]:
    path = output_root / RUN_REL / "fit_ledger.jsonl"
    if not path.is_file():
        raise VerificationError("Isolated fit ledger is missing")
    events = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        try:
            event = json.loads(line)
        except Exception as exc:
            raise VerificationError(f"Malformed ledger event {line_no}") from exc
        if not isinstance(event, dict):
            raise VerificationError(f"Ledger event {line_no} is not an object")
        events.append(event)
    reserve = [event for event in events if event.get("event") == "CARD_FITS_RESERVED"]
    starts = [event for event in events if event.get("event") == "FIT_STARTED"]
    saved = [event for event in events if event.get("event") == "FIT_SAVED"]
    failed = [event for event in events if event.get("event") == "FIT_FAILED"]
    if (len(reserve) != 1 or reserve[0].get("fit_ids") != list(FIT_IDS)
            or reserve[0].get("cap") != LEDGER_CAP
            or reserve[0].get("freeze_signature") != freeze["freeze_signature"]
            or [event.get("fit_id") for event in starts] != list(FIT_IDS)
            or [event.get("fit_id") for event in saved] != list(FIT_IDS)
            or failed or len(events) != 1 + 2 * LEDGER_CAP):
        raise VerificationError("Isolated ledger is not exactly six ordered, saved, retry-free fits")
    if any(event.get("shared_main_ledger_written") is not False for event in reserve + starts + saved):
        raise VerificationError("Shared main ledger boundary is not attested false")
    if any(event.get("freeze_signature") != freeze["freeze_signature"] for event in starts + saved):
        raise VerificationError("Ledger events do not match the frozen signature")
    return events, _sha_file(path)


def _artifact_paths(output_root: Path, fit_id: str) -> dict[str, Path]:
    run = output_root / RUN_REL
    return {"model": run / f"{fit_id}.joblib", "predictions": run / f"{fit_id}.npz",
            "receipt": run / f"{fit_id}.receipt.json"}


def _verify_receipt_and_arrays(output_root: Path, freeze: dict[str, Any], fit_id: str,
                               expected_origin: int, expected_arm: str,
                               expected: dict[str, np.ndarray] | None = None) -> dict[str, np.ndarray]:
    paths = _artifact_paths(output_root, fit_id)
    for path in paths.values():
        if not path.is_file():
            raise VerificationError(f"Durable fit artifact missing: {path.name}")
    receipt = _json(paths["receipt"])
    receipt_body = {key: value for key, value in receipt.items() if key != "receipt_sha256"}
    if (receipt.get("receipt_sha256") != _canonical_sha(receipt_body)
            or receipt.get("fit_id") != fit_id
            or receipt.get("freeze_signature") != freeze["freeze_signature"]
            or receipt.get("model_sha256") != _sha_file(paths["model"])
            or receipt.get("predictions_sha256") != _sha_file(paths["predictions"])
            or receipt.get("query_targets_read_for_fit") is not False
            or receipt.get("test_or_sample_accessed") is not False
            or receipt.get("classification_labels_or_probabilities_accessed") is not False
            or receipt.get("shared_main_ledger_written") is not False):
        raise VerificationError(f"Receipt identity, content hash or boundary failed for {fit_id}")
    payload = joblib.load(paths["model"])
    if (not isinstance(payload, dict) or payload.get("fit_id") != fit_id
            or payload.get("freeze_signature") != freeze["freeze_signature"]
            or payload.get("origin") != expected_origin or payload.get("arm") != expected_arm
            or payload.get("parameters_sha256") != receipt.get("parameters_sha256")):
        raise VerificationError(f"Saved model payload identity failed for {fit_id}")
    with np.load(paths["predictions"], allow_pickle=False) as archive:
        arrays = {key: archive[key] for key in archive.files}
    required = {"fit_id", "origin", "arm", "rows", "record_ids", "cycles", "baseline",
                "missing", "head_prediction", "delta", "prediction"}
    if not required.issubset(arrays):
        raise VerificationError(f"Prediction archive lacks required arrays for {fit_id}")
    if (str(arrays["fit_id"].reshape(-1)[0]) != fit_id
            or int(arrays["origin"].reshape(-1)[0]) != expected_origin
            or str(arrays["arm"].reshape(-1)[0]) != expected_arm):
        raise VerificationError(f"Prediction origin/arm/fit ID mismatch for {fit_id}")
    rows = arrays["rows"].astype(np.int64, copy=False)
    ids = arrays["record_ids"].astype(str)
    cycles = arrays["cycles"].astype(np.int64, copy=False)
    baseline = arrays["baseline"].astype(np.float64, copy=False)
    missing = arrays["missing"]
    head = arrays["head_prediction"].astype(np.float64, copy=False)
    delta = arrays["delta"].astype(np.float64, copy=False)
    prediction = arrays["prediction"].astype(np.float64, copy=False)
    if (rows.ndim != 1 or len(set(rows.tolist())) != len(rows) or ids.shape != rows.shape
            or cycles.shape != rows.shape or baseline.shape != rows.shape or missing.shape != rows.shape
            or missing.dtype.kind != "b" or head.shape != rows.shape or delta.shape != rows.shape
            or prediction.shape != rows.shape or np.any(cycles <= expected_origin)
            or not np.isfinite(baseline).all() or not np.isfinite(head).all()
            or not np.isfinite(delta).all() or not np.isfinite(prediction).all()
            or np.any((baseline < 0) | (baseline > 100))
            or np.any((prediction < 0) | (prediction > 100))):
        raise VerificationError(f"Prediction rows, bounds or values are invalid for {fit_id}")
    independent_delta = np.clip(0.25 * head, -5.0, 5.0)
    independent_delta[missing] = 0.0
    independent_prediction = np.clip(baseline + independent_delta, 0.0, 100.0)
    independent_prediction[missing] = baseline[missing]
    if not np.array_equal(delta, independent_delta) or not np.array_equal(prediction, independent_prediction):
        raise VerificationError(f"Fixed shrink/cap/missing/bound formula differs for {fit_id}")
    if not np.array_equal(prediction[missing], baseline[missing]):
        raise VerificationError(f"Missing route is not bit-exact final R06 for {fit_id}")
    if expected is not None:
        for key in ("rows", "record_ids", "cycles", "baseline", "missing"):
            if key not in expected or not np.array_equal(arrays[key], expected[key]):
                raise VerificationError(f"Synthetic expected {key} mismatch for {fit_id}")
    return arrays


def _bundle_path(source_root: Path, origin: int) -> Path:
    folder = "recovered_bundles" if origin == 67 else "bundles"
    rel = Path("experiments/runs/optimization_v3_research_next_v1/RNBR_PILOT_OFFICIAL_V1") / folder / f"RNBR_PILOT_E{origin}_OBSERVED/arrays.npz"
    return _contained(source_root, rel)


def prelabel(source_root: Path, output_root: Path) -> dict[str, Any]:
    source_root = source_root.resolve()
    output_root = output_root.resolve()
    freeze = _freeze(source_root, output_root)
    events, ledger_sha = _ledger(output_root, freeze)
    verified: dict[str, Any] = {}
    for origin in ORIGINS:
        bundle_path = _bundle_path(source_root, origin)
        with np.load(bundle_path, allow_pickle=False) as bundle:
            expected = {"rows": bundle["score_rows"],
                        "record_ids": bundle["score_record_ids"],
                        "cycles": bundle["score_cycles"],
                        "baseline": bundle["baseline_prediction"],
                        "missing": None}
            # Preserve the final mapped R06 member identity. The RNBR candidate
            # prediction and raw candidate are deliberately not used as anchors.
            final_anchor = np.asarray(bundle["baseline_prediction"], dtype=np.float64)
            if final_anchor.shape != expected["rows"].shape or not np.isfinite(final_anchor).all():
                raise VerificationError(f"E{origin} final R06 baseline member is invalid")
        origin_results = {}
        for arm in ARMS:
            fit_id = f"{CARD_ID}_E{origin}_{arm}"
            arrays = _verify_receipt_and_arrays(output_root, freeze, fit_id, origin, arm)
            if (not np.array_equal(arrays["rows"], expected["rows"])
                    or not np.array_equal(arrays["record_ids"].astype(str), expected["record_ids"].astype(str))
                    or not np.array_equal(arrays["cycles"], expected["cycles"])
                    or not np.array_equal(arrays["baseline"], final_anchor)):
                raise VerificationError(f"E{origin} {arm} rows, IDs, cycles or final R06 baseline differ")
            provenance = freeze.get("origins", {}).get(str(origin), {})
            if (len(arrays["rows"]) != provenance.get("query_rows")
                    or _sha_array(arrays["rows"], dtype="<i8") != provenance.get("query_rows_source_positions_sha256")
                    or _sha_array(arrays["record_ids"]) != provenance.get("query_row_ids_sha256")
                    or _sha_array(arrays["baseline"], dtype="<f8") != provenance.get("query_final_r06_prediction_sha256")):
                raise VerificationError(f"E{origin} {arm} frozen query provenance does not reconcile")
            origin_results[arm] = {"n": int(len(arrays["rows"])), "rows_sha256": _sha_array(arrays["rows"], dtype="<i8"),
                                   "record_ids_sha256": _sha_array(arrays["record_ids"]),
                                   "baseline_sha256": _sha_array(arrays["baseline"], dtype="<f8"),
                                   "missing_exact": bool(np.array_equal(arrays["prediction"][arrays["missing"]], arrays["baseline"][arrays["missing"]]))}
        verified[str(origin)] = origin_results
    result = {"schema_version": 1, "status": "PASS", "phase": "prelabel",
              "card_id": CARD_ID, "freeze_signature": freeze["freeze_signature"],
              "all_six_models_and_predictions_durable": True,
              "fit_ids": list(FIT_IDS), "fit_starts": len([e for e in events if e.get("event") == "FIT_STARTED"]),
              "fit_saved": len([e for e in events if e.get("event") == "FIT_SAVED"]),
              "fit_failed": 0, "fit_pending": 0, "fit_retries": 0,
              "fit_ledger_sha256": ledger_sha,
              "classification_labels_or_probabilities_accessed": False,
              "query_targets_read": False, "query_targets_used_for_fit": False,
              "test_or_sample_accessed": False, "kaggle_accessed": False,
              "shared_main_ledger_written": False, "origins": verified}
    path = output_root / REPORT_REL / "PRELABEL_VERIFICATION.json"
    _write_once(path, result)
    return result


def _write_once(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n"
    if path.exists():
        if path.read_text(encoding="utf-8") == payload:
            return
        raise VerificationError(f"Refusing to overwrite different immutable verification output: {path}")
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("w", encoding="utf-8", newline="\n") as stream:
        stream.write(payload); stream.flush(); os.fsync(stream.fileno())
    os.replace(temp, path)


def _read_scoring_rows(path: Path, wanted: dict[int, str]) -> dict[int, dict[str, Any]]:
    selected: dict[int, dict[str, Any]] = {}
    use = ("record_id", "cycle_code", "signal_stream", "impact_load", "text_load", "entry_friction", "threat_rank")
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames is None or any(name not in reader.fieldnames for name in use):
            raise VerificationError("Regression Train scoring columns are incomplete")
        for row_index, row in enumerate(reader):
            if row_index not in wanted:
                continue
            if row_index in selected or row["record_id"] != str(wanted[row_index]):
                raise VerificationError(f"Query scoring identity differs at row {row_index}")
            selected[row_index] = row
    if set(selected) != set(wanted):
        raise VerificationError("Not all frozen query rows were found for final scoring")
    return selected


def _float(value: str, name: str, *, nullable: bool = False) -> float:
    if nullable and (value is None or not value.strip() or value.lower() == "nan"):
        return float("nan")
    try:
        result = float(value)
    except Exception as exc:
        raise VerificationError(f"Invalid numeric {name}") from exc
    if not math.isfinite(result):
        raise VerificationError(f"Nonfinite numeric {name}")
    return result


def _mae(y: np.ndarray, pred: np.ndarray) -> float:
    if len(y) == 0 or y.shape != pred.shape or not np.isfinite(y).all() or not np.isfinite(pred).all():
        raise VerificationError("Metric inputs are empty, nonfinite or misaligned")
    return float(np.mean(np.abs(y - pred)))


def _rho(y: np.ndarray, pred: np.ndarray) -> float | None:
    value = float(spearmanr(y, pred).statistic)
    return value if math.isfinite(value) else None


def _groups(y: np.ndarray, missing: np.ndarray, entry: np.ndarray,
            impact: np.ndarray, text: np.ndarray) -> dict[str, np.ndarray]:
    masks: dict[str, np.ndarray] = {"missing": missing, "target_le5": y <= 5.0, "target_ge95": y >= 95.0,
                                    "EF0": np.isfinite(entry) & (entry == 0.0),
                                    "EF1": np.isfinite(entry) & (entry == 1.0)}
    for name, bins in (("impact_load", impact), ("text_load", text)):
        for level in range(int(np.max(bins)) + 1):
            masks[f"{name}_Q{level + 1}"] = bins == level
    return masks


def _metric_view(y: np.ndarray, arrays: dict[str, dict[str, np.ndarray]], mask: np.ndarray,
                 groups: dict[str, np.ndarray] | None = None) -> dict[str, Any]:
    if mask.shape != y.shape or not mask.any():
        raise VerificationError("A frozen metric view is empty")
    baseline = arrays["BAG"]["baseline"].astype(np.float64)
    base_mae = _mae(y[mask], baseline[mask])
    metrics: dict[str, Any] = {"R06": {"mae": base_mae, "spearman": _rho(y[mask], baseline[mask])}}
    for arm in ARMS:
        prediction = arrays[arm]["prediction"].astype(np.float64)
        head = arrays[arm]["head_prediction"].astype(np.float64)
        arm_mae = _mae(y[mask], prediction[mask])
        metrics[arm] = {"mae": arm_mae, "gain_vs_R06": base_mae - arm_mae,
                        "spearman_y_vs_prediction": _rho(y[mask], prediction[mask]),
                        "spearman_query_residual_vs_head": _rho((y - baseline)[mask], head[mask]),
                        "missing_route_exact": bool(np.array_equal(prediction[arrays[arm]["missing"]],
                                                                      baseline[arrays[arm]["missing"]]))}
        metrics[arm]["mae_harm_vs_R06"] = arm_mae - base_mae
    metrics["REAL"]["gain_vs_BAG"] = metrics["BAG"]["mae"] - metrics["REAL"]["mae"]
    metrics["REAL"]["gain_vs_SHUFFLED"] = metrics["SHUFFLED"]["mae"] - metrics["REAL"]["mae"]
    result: dict[str, Any] = {"n": int(mask.sum()), "metrics": metrics}
    if groups is not None:
        group_result = {}
        for name, group_mask in groups.items():
            selected = mask & group_mask
            n = int(selected.sum())
            if n < MIN_GROUP_N:
                group_result[name] = {"n": n, "status": "UNKNOWN_UNSUPPORTED"}
                continue
            reference = _mae(y[selected], baseline[selected])
            group_result[name] = {"n": n, "status": "SUPPORTED", "R06_mae": reference,
                                  "arms": {arm: {"mae": _mae(y[selected], arrays[arm]["prediction"][selected]),
                                                 "harm_vs_R06": _mae(y[selected], arrays[arm]["prediction"][selected]) - reference}
                                           for arm in ARMS}}
        result["groups"] = group_result
    return result


def _compare_json(expected: Any, actual: Any, path: str = "root") -> None:
    if isinstance(expected, dict) and isinstance(actual, dict):
        if set(expected) != set(actual):
            raise VerificationError(f"Scoring report keys differ at {path}")
        for key in expected:
            _compare_json(expected[key], actual[key], f"{path}.{key}")
    elif isinstance(expected, list) and isinstance(actual, list):
        if len(expected) != len(actual):
            raise VerificationError(f"Scoring report list length differs at {path}")
        for index, (left, right) in enumerate(zip(expected, actual)):
            _compare_json(left, right, f"{path}[{index}]")
    elif isinstance(expected, (int, float)) and not isinstance(expected, bool) and isinstance(actual, (int, float)) and not isinstance(actual, bool):
        if not math.isclose(float(expected), float(actual), rel_tol=0.0, abs_tol=1e-10):
            raise VerificationError(f"Independent scoring value differs at {path}: {expected} vs {actual}")
    elif expected != actual:
        raise VerificationError(f"Independent scoring value differs at {path}: {expected!r} vs {actual!r}")


def _eligibility(views: dict[str, Any], integrity_pass: bool) -> dict[str, Any]:
    results: dict[str, Any] = {}
    for arm in ("REAL", "BAG", "SHUFFLED"):
        checks: dict[str, Any] = {}
        primary = []
        for view in ("E67_H12", "E75_W6"):
            gain = views[view]["metrics"][arm]["gain_vs_R06"]
            passed = gain >= PRIMARY_GAIN_MIN
            primary.append(passed)
            checks[f"{view}_gain_ge_0_03"] = {"status": "PASS" if passed else "FAIL", "value": gain, "threshold": PRIMARY_GAIN_MIN}
        window_pass = True
        for view in ("E67_W4", "E67_W5", "E67_W6"):
            harm = views[view]["metrics"][arm]["mae_harm_vs_R06"]
            passed = harm <= WINDOW_HARM_MAX
            window_pass &= passed
            checks[f"{view}_harm_le_0_10"] = {"status": "PASS" if passed else "FAIL", "value": harm, "threshold": WINDOW_HARM_MAX}
        group_fail = False
        group_unknown = False
        for view in ("E67_H12", "E67_W4", "E67_W5", "E67_W6", "E75_W6"):
            for name, group in views[view].get("groups", {}).items():
                key = f"{view}.{name}"
                if group.get("status") != "SUPPORTED":
                    group_unknown = True
                    checks[f"group_{key}"] = {"status": "UNKNOWN", "n": group.get("n"), "threshold": MIN_GROUP_N}
                    continue
                harm = group["arms"][arm]["harm_vs_R06"]
                passed = harm <= GROUP_HARM_MAX
                group_fail |= not passed
                checks[f"group_{key}"] = {"status": "PASS" if passed else "FAIL", "n": group["n"],
                                           "value": harm, "threshold": GROUP_HARM_MAX}
        exact = all(views[view]["metrics"][arm]["missing_route_exact"]
                    for view in ("E67_H12", "E67_W4", "E67_W5", "E67_W6", "E75_W6"))
        checks["missing_route_exact"] = {"status": "PASS" if exact else "FAIL"}
        checks["integrity_and_accounting"] = {"status": "PASS" if integrity_pass else "FAIL"}
        if not integrity_pass or not exact or not all(primary) or not window_pass or group_fail:
            comp = "FAIL"
        elif group_unknown:
            comp = "UNKNOWN"
        else:
            comp = "PASS"
        results[arm] = {"competition_gate": comp, "checks": checks}
    real_mechanism = {}
    for view in ("E67_H12", "E75_W6"):
        for competitor in ("BAG", "SHUFFLED"):
            gain = views[view]["metrics"]["REAL"][f"gain_vs_{competitor}"]
            key = f"{view}_gain_vs_{competitor}_ge_0_01"
            passed = gain >= MECHANISM_GAIN_MIN
            real_mechanism[key] = {"status": "PASS" if passed else "FAIL", "value": gain,
                                   "threshold": MECHANISM_GAIN_MIN}
    mechanism = "PASS" if all(value["status"] == "PASS" for value in real_mechanism.values()) else "FAIL"
    if results["REAL"]["competition_gate"] == "PASS" and mechanism == "PASS":
        decision = "ELIGIBLE_REAL_POSITION_MECHANISM"
    elif results["BAG"]["competition_gate"] == "PASS":
        decision = "ELIGIBLE_BAG_POSITION_NOT_ESTABLISHED"
    elif any(results[arm]["competition_gate"] == "UNKNOWN" for arm in ("REAL", "BAG")):
        decision = "NO_ELIGIBLE_CANDIDATE_UNKNOWN_GROUP_SUPPORT"
    else:
        decision = "RETIRE_PILOT_NO_CANDIDATE_PASSED"
    return {"arms": results, "REAL_mechanism_gate": mechanism,
            "REAL_mechanism_checks": real_mechanism, "decision": decision,
            "evidence_class": "ADAPTIVE_DEVELOPMENT_ONLY_NOT_UNTOUCHED_VALIDATION"}


def _png_once(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if _sha_file(path) == hashlib.sha256(content).hexdigest():
            return
        raise VerificationError(f"Refusing to overwrite a different PNG: {path}")
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("wb") as stream:
        stream.write(content); stream.flush(); os.fsync(stream.fileno())
    os.replace(temp, path)


def _text_once(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = text.encode("utf-8")
    if path.exists():
        if path.read_bytes() == payload:
            return
        raise VerificationError(f"Refusing to overwrite different findings text: {path}")
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("wb") as stream:
        stream.write(payload); stream.flush(); os.fsync(stream.fileno())
    os.replace(temp, path)


def _render_final_artifacts(output_root: Path, views: dict[str, Any],
                            eligibility: dict[str, Any], receipts: dict[str, dict[str, Any]]) -> dict[str, Any]:
    import io
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    report_root = output_root / REPORT_REL
    labels = ("E67_H12", "E67_W4", "E67_W5", "E67_W6", "E75_W6", "LATE_UNIQUE_E67_W4_PLUS_E75_W6")
    arms = ("BAG", "REAL", "SHUFFLED")
    x = np.arange(len(labels), dtype=np.float64)
    width = 0.24
    buffer = io.BytesIO()
    fig, ax = plt.subplots(figsize=(10.5, 4.8))
    colors = {"BAG": "#3b82f6", "REAL": "#16a34a", "SHUFFLED": "#a855f7"}
    for index, arm in enumerate(arms):
        gains = [views[name]["metrics"][arm]["gain_vs_R06"] for name in labels]
        ax.bar(x + (index - 1) * width, gains, width=width, label=arm, color=colors[arm])
    ax.axhline(0.0, color="#333333", linewidth=0.8)
    ax.axhline(PRIMARY_GAIN_MIN, color="#d97706", linestyle="--", linewidth=1.0,
               label="Primary gain threshold (+0.03)")
    ax.set_xticks(x, labels, rotation=18, ha="right")
    ax.set_ylabel("MAE gain vs final R06 (positive is better)")
    ax.set_title("REG_POSITION_L1_V1: development-view MAE gains")
    ax.legend(ncol=2, frameon=False)
    ax.grid(axis="y", alpha=0.22)
    fig.tight_layout()
    fig.savefig(buffer, format="png", dpi=170, metadata={"Software": f"matplotlib {matplotlib.__version__}"})
    plt.close(fig)
    gain_path = report_root / "REG_POSITION_L1_V1_PRIMARY_GAINS.png"
    _png_once(gain_path, buffer.getvalue())

    gate_views = ("E67_H12", "E67_W4", "E67_W5", "E67_W6", "E75_W6")
    group_rows: list[tuple[str, str, int, float | None]] = []
    for view_name in gate_views:
        for group_name, group in views[view_name].get("groups", {}).items():
            if group.get("status") != "SUPPORTED":
                group_rows.append((view_name, group_name, int(group.get("n", 0)), None))
                continue
            for arm in ("BAG", "REAL"):
                group_rows.append((view_name, f"{group_name} / {arm}", int(group["n"]),
                                   float(group["arms"][arm]["harm_vs_R06"])))
    labels2 = [f"{view} | {name} (n={n})" for view, name, n, _ in group_rows]
    values = [value if value is not None else np.nan for _, _, _, value in group_rows]
    cmap_data = np.array(values, dtype=np.float64).reshape(-1, 1)
    buffer2 = io.BytesIO()
    fig2, ax2 = plt.subplots(figsize=(8.5, max(5.5, len(labels2) * 0.19)))
    masked = np.ma.masked_invalid(cmap_data)
    image = ax2.imshow(masked, aspect="auto", cmap="RdYlGn_r", vmin=-0.5, vmax=0.5)
    ax2.set_xticks([0], ["Group MAE harm vs R06"])
    ax2.set_yticks(np.arange(len(labels2)), labels2, fontsize=6.5)
    for row, (_, _, count, value) in enumerate(group_rows):
        if value is None:
            ax2.text(0, row, f" UNKNOWN (n={count})", ha="center", va="center", fontsize=6.5, color="#333333")
        else:
            ax2.text(0, row, f" {value:+.3f} (n={count})", ha="center", va="center", fontsize=6.5, color="#111111")
    ax2.set_title("Fixed subgroup risk checks; unsupported groups remain UNKNOWN")
    fig2.colorbar(image, ax=ax2, label="MAE harm vs R06")
    fig2.tight_layout()
    fig2.savefig(buffer2, format="png", dpi=170, metadata={"Software": f"matplotlib {matplotlib.__version__}"})
    plt.close(fig2)
    group_path = report_root / "REG_POSITION_L1_V1_GROUP_HARM.png"
    _png_once(group_path, buffer2.getvalue())

    table = ["| View | n | R06 MAE | BAG gain | REAL gain | SHUFFLED gain |",
             "|---|---:|---:|---:|---:|---:|"]
    for name in labels:
        view = views[name]
        metrics = view["metrics"]
        table.append(f"| {name} | {view['n']} | {metrics['R06']['mae']:.6f} | "
                     f"{metrics['BAG']['gain_vs_R06']:+.6f} | {metrics['REAL']['gain_vs_R06']:+.6f} | "
                     f"{metrics['SHUFFLED']['gain_vs_R06']:+.6f} |")
    group_summary = []
    for arm in ("REAL", "BAG", "SHUFFLED"):
        states = [check["status"] for key, check in eligibility["arms"][arm]["checks"].items()
                  if key.startswith("group_")]
        group_summary.append(
            f"- **{arm}:** {sum(state == 'PASS' for state in states)} PASS, "
            f"{sum(state == 'FAIL' for state in states)} FAIL, "
            f"{sum(state == 'UNKNOWN' for state in states)} UNKNOWN group checks; "
            f"competition gate: {eligibility['arms'][arm]['competition_gate']}."
        )
    mechanism = eligibility["REAL_mechanism_checks"]
    findings = [
        "# REG_POSITION_L1_V1 results",
        "",
        f"**Decision:** {eligibility['decision']}",
        "",
        "This is adaptive development evidence with overlapping E67/E75 outcomes. It is not untouched validation or submission authorization.",
        "",
        "## Primary and window metrics",
        "",
        *table,
        "",
        "## Fixed subgroup gates",
        "",
        *group_summary,
        "",
        "## REAL mechanism comparison",
        "",
        *[f"- {key}: {value['status']} ({value['value']:+.6f}; threshold {value['threshold']:.2f})."
          for key, value in mechanism.items()],
        "",
        f"REAL mechanism gate: {eligibility['REAL_mechanism_gate']}.",
        "",
        "## Fit accounting",
        "",
        f"- Starts: {len(receipts)}; saved: {len(receipts)}; failed: 0; retries: 0; isolated cap: 6.",
        "- Query targets were read only after all six saved artifacts passed independent prelabel checks.",
        "- Classification labels/probabilities, Test, Sample Submission, Kaggle, and the shared main ledger were not accessed or written.",
        "- Reported in-sample training MAE is descriptive only; selection uses the frozen development gates.",
        "",
        "## Charts",
        "",
        "- REG_POSITION_L1_V1_PRIMARY_GAINS.png compares MAE gains across all frozen views.",
        "- REG_POSITION_L1_V1_GROUP_HARM.png shows fixed subgroup harm and marks unsupported denominators UNKNOWN.",
        "",
        "A later tranche requires Sol review and its own source/freeze/accounting amendment. No E79 fit or upload is authorized by this pilot.",
        "",
    ]
    findings_path = report_root / "FINDINGS.md"
    _text_once(findings_path, "\n".join(findings))
    return {
        "findings": findings_path.name, "primary_gain_chart": gain_path.name,
        "subgroup_harm_chart": group_path.name,
        "matplotlib_version": matplotlib.__version__,
        "sha256": {gain_path.name: _sha_file(gain_path), group_path.name: _sha_file(group_path),
                   findings_path.name: _sha_file(findings_path)},
    }


def final(source_root: Path, output_root: Path) -> dict[str, Any]:
    source_root = source_root.resolve(); output_root = output_root.resolve()
    freeze = _freeze(source_root, output_root)
    recomputed_prelabel = prelabel(source_root, output_root)
    prelabel_path = output_root / REPORT_REL / "PRELABEL_VERIFICATION.json"
    prelabel_receipt = _json(prelabel_path)
    if (prelabel_receipt.get("status") != "PASS"
            or prelabel_receipt.get("freeze_signature") != freeze["freeze_signature"]
            or prelabel_receipt.get("query_targets_read") is not False
            or recomputed_prelabel != prelabel_receipt):
        raise VerificationError("A passing prelabel report must precede outcome access")
    if _sha_file(_contained(source_root, TRAIN_REL)) != freeze.get("official_train_sha256"):
        raise VerificationError("Regression Train SHA differs from the freeze")
    per_origin: dict[int, dict[str, Any]] = {}
    expected_views: dict[str, Any] = {}
    for origin in ORIGINS:
        bundle_path = _bundle_path(source_root, origin)
        with np.load(bundle_path, allow_pickle=False) as bundle:
            rows = np.asarray(bundle["score_rows"], dtype=np.int64)
            ids = np.asarray(bundle["score_record_ids"], dtype=str)
            cycles = np.asarray(bundle["score_cycles"], dtype=np.int64)
            baseline = np.asarray(bundle["baseline_prediction"], dtype=np.float64)
        wanted = {int(row): str(record_id) for row, record_id in zip(rows, ids)}
        selected = _read_scoring_rows(_contained(source_root, TRAIN_REL), wanted)
        if len(selected) != len(rows):
            raise VerificationError(f"E{origin} selected target count differs")
        y = np.empty(len(rows), dtype=np.float64)
        impact_raw = np.empty(len(rows), dtype=np.float64)
        text_raw = np.empty(len(rows), dtype=np.float64)
        entry = np.empty(len(rows), dtype=np.float64)
        missing = np.empty(len(rows), dtype=bool)
        for slot, (row_index, record_id) in enumerate(zip(rows, ids)):
            row = selected[int(row_index)]
            if row["record_id"] != record_id or int(_float(row["cycle_code"], "cycle_code")) != cycles[slot]:
                raise VerificationError(f"E{origin} target row, ID or cycle mismatch")
            y[slot] = _float(row["threat_rank"], "threat_rank")
            impact_raw[slot] = _float(row["impact_load"], "impact_load")
            text_raw[slot] = _float(row["text_load"], "text_load")
            entry[slot] = _float(row["entry_friction"], "entry_friction", nullable=True)
            missing[slot] = not bool(row["signal_stream"].strip())
        if not np.isfinite(y).all() or np.any((y < 0) | (y > 100)):
            raise VerificationError(f"E{origin} outcomes violate [0,100]")
        edges = freeze["origins"][str(origin)]["input_reconstruction"]["prefix_quartile_edges"]
        impact_bins = np.digitize(impact_raw, np.asarray(edges["impact_load"], dtype=np.float64), right=False)
        text_bins = np.digitize(text_raw, np.asarray(edges["text_load"], dtype=np.float64), right=False)
        arrays_by_arm: dict[str, dict[str, np.ndarray]] = {}
        for arm in ARMS:
            fit_id = f"{CARD_ID}_E{origin}_{arm}"
            arrays = _verify_receipt_and_arrays(output_root, freeze, fit_id, origin, arm)
            if (not np.array_equal(arrays["rows"], rows) or not np.array_equal(arrays["record_ids"].astype(str), ids)
                    or not np.array_equal(arrays["cycles"], cycles) or not np.array_equal(arrays["baseline"], baseline)
                    or not np.array_equal(arrays["missing"], missing)):
                raise VerificationError(f"E{origin} postlabel source identities differ for {arm}")
            arrays_by_arm[arm] = arrays
        group_masks = _groups(y, missing, entry, impact_bins, text_bins)
        per_origin[origin] = {"y": y, "cycles": cycles, "arrays": arrays_by_arm,
                              "groups": group_masks, "rows": rows, "ids": ids}
    cycle67 = per_origin[67]["cycles"]
    y67 = per_origin[67]["y"]
    m67 = {"E67_H12": np.ones(len(y67), dtype=bool),
           "E67_W4": (cycle67 >= 68) & (cycle67 <= 71),
           "E67_W5": (cycle67 >= 72) & (cycle67 <= 75),
           "E67_W6": (cycle67 >= 76) & (cycle67 <= 79)}
    for name, mask in m67.items():
        expected_views[name] = _metric_view(y67, per_origin[67]["arrays"], mask, per_origin[67]["groups"])
    m75 = (per_origin[75]["cycles"] >= 76) & (per_origin[75]["cycles"] <= 79)
    expected_views["E75_W6"] = _metric_view(per_origin[75]["y"], per_origin[75]["arrays"], m75,
                                              per_origin[75]["groups"])
    late_y = np.concatenate([y67[m67["E67_W4"]], per_origin[75]["y"][m75]])
    if len(np.unique(np.concatenate([per_origin[67]["ids"][m67["E67_W4"]], per_origin[75]["ids"][m75]]))) != len(late_y):
        raise VerificationError("Late unique view contains duplicate record IDs")
    late_arrays = {}
    for arm in ARMS:
        late_arrays[arm] = {}
        for key in ("baseline", "head_prediction", "prediction", "missing"):
            late_arrays[arm][key] = np.concatenate([per_origin[67]["arrays"][arm][key][m67["E67_W4"]],
                                                    per_origin[75]["arrays"][arm][key][m75]])
    late_mask = np.ones(len(late_y), dtype=bool)
    late = _metric_view(late_y, late_arrays, late_mask)
    late_metrics = {"R06": late["metrics"]["R06"]}
    for arm in ARMS:
        late_metrics[arm] = {"mae": late["metrics"][arm]["mae"],
                             "gain_vs_R06": late["metrics"][arm]["gain_vs_R06"],
                             "spearman_y_vs_prediction": late["metrics"][arm]["spearman_y_vs_prediction"]}
    expected_views["LATE_UNIQUE_E67_W4_PLUS_E75_W6"] = {"n": late["n"], "metrics": late_metrics,
                                                        "unique_record_ids": True, "groups": "not_applicable"}
    runner_score_path = output_root / REPORT_REL / "SCORE_METRICS.json"
    runner_score = _json(runner_score_path)
    expected_score_keys = {"schema_version": 1, "status": "SCORED_AFTER_PRELABEL_PASS", "card_id": CARD_ID,
                           "freeze_signature": freeze["freeze_signature"], "fit_ids": list(FIT_IDS),
                           "fit_ledger_sha256": prelabel_receipt["fit_ledger_sha256"],
                           "fit_starts": LEDGER_CAP, "fit_saved": LEDGER_CAP, "fit_failed": 0,
                           "prelabel_verification_sha256": _sha_file(prelabel_path),
                           "query_targets_read_after_all_six_saved": True,
                           "query_targets_used_for_fit": False, "test_or_sample_accessed": False,
                           "classification_labels_or_probabilities_accessed": False,
                           "views": expected_views}
    _compare_json(expected_score_keys, runner_score)
    gate = _eligibility(expected_views, integrity_pass=True)
    receipts = {fit_id: _json(_artifact_paths(output_root, fit_id)["receipt"]) for fit_id in FIT_IDS}
    artifacts = _render_final_artifacts(output_root, expected_views, gate, receipts)
    result = {"schema_version": 1, "status": "PASS_INDEPENDENT_METRICS_AND_GATES",
              "card_id": CARD_ID, "freeze_signature": freeze["freeze_signature"],
              "score_metrics_sha256": _sha_file(runner_score_path),
              "prelabel_verification_sha256": _sha_file(prelabel_path),
              "query_targets_read_after_all_six_saved": True,
              "classification_labels_or_probabilities_accessed": False,
              "test_or_sample_accessed": False, "views": expected_views, "eligibility": gate,
              "artifacts": artifacts,
              "fit_diagnostics_descriptive_only": {
                  fit_id: {"in_sample_training_mae_descriptive_only": receipts[fit_id].get("in_sample_training_mae_descriptive_only"),
                           "resumed_from_durable_model_without_refit": receipts[fit_id].get("resumed_from_durable_model_without_refit"),
                           "resource_preflight_available_bytes": receipts[fit_id].get("resource_preflight", {}).get("available_ram_bytes"),
                           "resource_after_fit_available_bytes": receipts[fit_id].get("resource_after_fit", {}).get("available_ram_bytes")}
                  for fit_id in FIT_IDS}}
    _write_once(output_root / REPORT_REL / "INDEPENDENT_FINAL_VERIFICATION.json", result)
    eligibility_output = {"schema_version": 1, "card_id": CARD_ID,
                          "freeze_signature": freeze["freeze_signature"],
                          "status": result["status"], "query_targets_read_after_all_six_saved": True,
                          **gate, "artifacts": artifacts}
    _write_once(output_root / REPORT_REL / "ELIGIBILITY.json", eligibility_output)
    return result


def _synthetic(source_root: Path, output_root: Path) -> dict[str, Any]:
    manifest = _json(source_root / "synthetic_manifest.json")
    freeze = manifest["freeze"]
    if freeze.get("fit_ids_order") != list(FIT_IDS):
        raise VerificationError("Synthetic manifest fit order is invalid")
    events = []
    ledger_path = output_root / RUN_REL / "fit_ledger.jsonl"
    if not ledger_path.is_file():
        raise VerificationError("Synthetic isolated ledger is missing")
    for line_no, line in enumerate(ledger_path.read_text(encoding="utf-8").splitlines(), 1):
        event = json.loads(line)
        if not isinstance(event, dict):
            raise VerificationError(f"Synthetic ledger event {line_no} is invalid")
        events.append(event)
    starts = [e for e in events if e.get("event") == "FIT_STARTED"]
    saved = [e for e in events if e.get("event") == "FIT_SAVED"]
    failed = [e for e in events if e.get("event") == "FIT_FAILED"]
    reserves = [e for e in events if e.get("event") == "CARD_FITS_RESERVED"]
    freeze_sig = freeze["freeze_signature"]
    if (len(reserves) != 1 or reserves[0].get("freeze_signature") != freeze_sig
            or [e.get("fit_id") for e in starts] != list(FIT_IDS)
            or [e.get("fit_id") for e in saved] != list(FIT_IDS) or failed):
        raise VerificationError("Synthetic fit lifecycle or accounting did not reconcile")
    all_arrays: dict[int, dict[str, dict[str, np.ndarray]]] = {}
    for origin in ORIGINS:
        expected = manifest["origins"][str(origin)]
        expected_arrays = {"rows": np.asarray(expected["rows"], dtype=np.int64),
                           "record_ids": np.asarray(expected["record_ids"], dtype=str),
                           "cycles": np.asarray(expected["cycles"], dtype=np.int64),
                           "baseline": np.asarray(expected["final_r06"], dtype=np.float64),
                           "missing": np.asarray(expected["missing"], dtype=bool)}
        if (expected["raw_candidate"] == expected["final_r06"]
                or expected["candidate_prediction"] == expected["final_r06"]):
            raise VerificationError("Synthetic raw/candidate controls must differ from final R06")
        all_arrays[origin] = {}
        for arm in ARMS:
            fit_id = f"{CARD_ID}_E{origin}_{arm}"
            arrays = _verify_receipt_and_arrays(output_root, freeze, fit_id, origin, arm, expected_arrays)
            all_arrays[origin][arm] = arrays
    views: dict[str, Any] = {}
    for origin in ORIGINS:
        data = manifest["origins"][str(origin)]
        y = np.asarray(data["truth"], dtype=np.float64)
        cycles = np.asarray(data["cycles"], dtype=np.int64)
        if y.shape != cycles.shape or y.shape != np.asarray(data["rows"]).shape:
            raise VerificationError("Synthetic outcomes and row metadata are misaligned")
        qgroups = {name: np.asarray(mask, dtype=bool) for name, mask in data["groups"].items()}
        if not np.array_equal(qgroups["target_le5"], y <= 5) or not np.array_equal(qgroups["target_ge95"], y >= 95):
            raise VerificationError("Synthetic tail group denominators do not match selected outcomes")
        origin_arrays = all_arrays[origin]
        if origin == 67:
            specs = {"E67_H12": np.ones(len(y), dtype=bool),
                     "E67_W4": (cycles >= 68) & (cycles <= 71),
                     "E67_W5": (cycles >= 72) & (cycles <= 75),
                     "E67_W6": (cycles >= 76) & (cycles <= 79)}
        else:
            specs = {"E75_W6": (cycles >= 76) & (cycles <= 79)}
        for name, mask in specs.items():
            views[name] = _metric_view(y, origin_arrays, mask, qgroups)
    small_groups_unknown = any(group.get("status") == "UNKNOWN_UNSUPPORTED"
                               for view in views.values() for group in view.get("groups", {}).values())
    if not small_groups_unknown:
        raise VerificationError("Synthetic tail/missing denominators did not exercise UNKNOWN support")
    result = {"status": "PASS_SYNTHETIC_INDEPENDENT_METRICS", "fit_ids": list(FIT_IDS),
              "fit_starts": len(starts), "fit_saved": len(saved), "fit_failed": len(failed),
              "fit_invocations_reported": int(manifest["fit_invocations"]),
              "resume_used_without_refit": bool(manifest["resume_used_without_refit"]),
              "query_targets_are_synthetic_only": True,
              "synthetic_views": views, "unsupported_groups_remain_unknown": True,
              "decision": "SYNTHETIC_QA_ONLY_NOT_PRODUCTION_ELIGIBILITY"}
    _write_once(output_root / "synthetic_independent_verification.json", result)
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("prelabel", "final", "synthetic"), required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        if args.phase == "prelabel":
            result = prelabel(args.source_root, args.output_root)
        elif args.phase == "final":
            result = final(args.source_root, args.output_root)
        else:
            result = _synthetic(args.source_root, args.output_root)
        print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
        return 0
    except Exception as exc:
        print(json.dumps({"status": "FAIL", "error_type": type(exc).__name__, "error": str(exc)},
                         ensure_ascii=False), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
