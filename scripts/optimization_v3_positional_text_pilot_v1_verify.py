"""Independent, fit-free prediction integrity and metric verification."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.models.classification_models import _text_values


THRESHOLD = 0.50
ROUTE_NAMES = ("bag", "position", "shuffled_position")


def _file_sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _json_sha(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, ensure_ascii=False,
                         allow_nan=False, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _array_sha(value: Any) -> str:
    array = np.asarray(value)
    header = json.dumps({"shape": array.shape, "dtype": str(array.dtype)}, sort_keys=True,
                        ensure_ascii=False, allow_nan=False,
                        separators=(",", ":")).encode("utf-8")
    h = hashlib.sha256(hashlib.sha256(header).hexdigest().encode("ascii"))
    if array.dtype.kind in "OUS":
        h.update(json.dumps(array.astype(str).tolist(), ensure_ascii=False,
                            separators=(",", ":")).encode("utf-8"))
    else:
        h.update(np.ascontiguousarray(array).tobytes())
    return h.hexdigest()


def _csr_sha(matrix: sparse.spmatrix) -> str:
    value = sparse.csr_matrix(matrix, dtype=np.float64, copy=True)
    value.sum_duplicates()
    value.sort_indices()
    header = json.dumps({"shape": value.shape, "dtype": str(value.dtype)}, sort_keys=True,
                        ensure_ascii=False, allow_nan=False,
                        separators=(",", ":")).encode("utf-8")
    h = hashlib.sha256(hashlib.sha256(header).hexdigest().encode("ascii"))
    h.update(np.asarray(value.indptr, dtype="<i8").tobytes())
    h.update(np.asarray(value.indices, dtype="<i8").tobytes())
    h.update(np.asarray(value.data, dtype="<f8").tobytes())
    return h.hexdigest()


def _expected_rows(frame: pd.DataFrame, origin: int) -> tuple[np.ndarray, int, int]:
    cycles = frame["cycle_code"].to_numpy(dtype=np.int64)
    if origin == 67:
        return np.flatnonzero((cycles >= 68) & (cycles <= 79)).astype(np.int64), 68, 79
    if origin == 75:
        return np.flatnonzero((cycles >= 76) & (cycles <= 79)).astype(np.int64), 76, 79
    raise ValueError(f"Unsupported origin {origin}")


def verify_prediction_bundle(
    frame: pd.DataFrame, origin: int, prediction_path: Path,
    matrix_manifest_path: Path,
) -> dict[str, Any]:
    """Verify durable row alignment, matrices, arm dimensions, and route logic.

    This function never reads target labels and never fits any object.
    """
    manifest = json.loads(matrix_manifest_path.read_text(encoding="utf-8"))
    with np.load(prediction_path, allow_pickle=False) as archive:
        prediction = {key: archive[key] for key in archive.files}
    rows, low, high = _expected_rows(frame, origin)
    ids = frame.iloc[rows]["record_id"].astype(str).to_numpy()
    cycles = frame.iloc[rows]["cycle_code"].to_numpy(dtype=np.int64)
    if (prediction.get("target") is not None or any(
            key.casefold() in {"y", "target", "labels", "score_labels", "y_score"}
            for key in prediction)):
        raise AssertionError("Prediction archive contains target labels")
    if (not np.array_equal(prediction.get("origin"), np.asarray([origin], dtype=np.int64))
            or not np.array_equal(prediction.get("score_rows"), rows)
            or not np.array_equal(prediction.get("score_record_ids"), ids)
            or not np.array_equal(prediction.get("score_cycles"), cycles)):
        raise AssertionError(f"E{origin} prediction rows/IDs/cycles do not match official Train features")
    if ids.size != np.unique(ids).size or int(cycles.min()) != low or int(cycles.max()) != high:
        raise AssertionError(f"E{origin} scored identity/cycle support failed")

    present = ~_text_values(frame.iloc[rows]["signal_stream"])[1]
    if not np.array_equal(prediction.get("present_mask"), present):
        raise AssertionError(f"E{origin} availability mask differs from signal_stream")
    model_names = ("bag", "position", "shuffled_position", "missing_hgb")
    for name in model_names:
        probability = np.asarray(prediction[f"prob_{name}"], dtype=np.float64)
        if (probability.shape != (len(rows),) or not np.isfinite(probability).all()
                or np.any((probability < 0.0) | (probability > 1.0))):
            raise AssertionError(f"E{origin} invalid probability vector: {name}")
    for name in ROUTE_NAMES:
        routed = np.asarray(prediction[f"prob_route_{name}"], dtype=np.float64)
        raw = np.asarray(prediction[f"prob_{name}"], dtype=np.float64)
        missing = np.asarray(prediction["prob_missing_hgb"], dtype=np.float64)
        if not np.array_equal(routed[present], raw[present]):
            raise AssertionError(f"E{origin} present route changed its {name} LR probability")
        if not np.array_equal(routed[~present], missing[~present]):
            raise AssertionError(f"E{origin} missing fallback changed for {name}")

    if str(prediction["matrix_manifest_sha256"].item()) != _file_sha(matrix_manifest_path):
        raise AssertionError(f"E{origin} prediction references another matrix manifest")
    matrices = manifest["matrices"]
    for name, item in matrices.items():
        path = Path(item["path"])
        if not path.is_absolute():
            path = Path(manifest.get("project_root", matrix_manifest_path.parents[3])) / path
        if not path.is_file() or _file_sha(path) != item["sha256"]:
            raise AssertionError(f"E{origin} matrix cache file hash mismatch: {name}")
        if path.suffix == ".npz":
            matrix = sparse.load_npz(path).tocsr()
            digest = _csr_sha(matrix)
        else:
            matrix = np.load(path, allow_pickle=False)
            digest = _array_sha(matrix)
        if list(matrix.shape) != item["shape"] or digest != item["logical_sha256"]:
            raise AssertionError(f"E{origin} matrix cache content mismatch: {name}")
    shapes = {name: item["shape"] for name, item in matrices.items()}
    if (shapes["fit_position"] != shapes["fit_shuffled_position"]
            or shapes["score_position"] != shapes["score_shuffled_position"]):
        raise AssertionError(f"E{origin} observed/shuffled feature dimensions differ")
    fit_rows = np.flatnonzero((frame["cycle_code"].to_numpy(dtype=np.int64) >= 47)
                              & (frame["cycle_code"].to_numpy(dtype=np.int64) <= origin))
    if manifest.get("fit_row_metadata", {}).get("rows") != len(fit_rows):
        raise AssertionError(f"E{origin} fit row count differs from frozen temporal origin")
    if (manifest.get("fit_row_metadata", {}).get("row_indices_sha256") != _array_sha(fit_rows)
            or manifest.get("fit_row_metadata", {}).get("record_ids_sha256")
            != _array_sha(frame.iloc[fit_rows]["record_id"].astype(str).to_numpy())):
        raise AssertionError(f"E{origin} fit row/ID hashes differ from the origin contract")
    return {
        "status": "PASS_INDEPENDENT_PREDICTION_INTEGRITY",
        "origin": origin,
        "score_rows": len(rows),
        "present_rows": int(present.sum()),
        "missing_rows": int((~present).sum()),
        "cycle_min_max": [low, high],
        "prediction_sha256": _file_sha(prediction_path),
        "matrix_manifest_sha256": _file_sha(matrix_manifest_path),
        "matrix_shapes": shapes,
        "query_labels_read": False,
        "fit_or_transform_fitted": False,
    }


def _metric(y: np.ndarray, probability: np.ndarray) -> dict[str, Any]:
    y = np.asarray(y, dtype=np.int8)
    p = np.asarray(probability, dtype=np.float64)
    if len(y) == 0:
        return {"n": 0, "positive": 0, "negative": 0, "macro_f1": None,
                "positive_f1": None, "precision": None, "positive_recall": None,
                "accuracy": None, "tn": 0, "fp": 0, "fn": 0, "tp": 0,
                "roc_auc": None, "average_precision": None, "log_loss": None, "brier": None}
    pred = (p >= THRESHOLD).astype(np.int8)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel().tolist()
    both = np.unique(y).size == 2
    return {
        "n": int(len(y)), "positive": int(y.sum()), "negative": int((y == 0).sum()),
        "macro_f1": float(f1_score(y, pred, labels=[0, 1], average="macro", zero_division=0)),
        "positive_f1": float(f1_score(y, pred, pos_label=1, zero_division=0)),
        "precision": float(precision_score(y, pred, pos_label=1, zero_division=0)),
        "positive_recall": float(recall_score(y, pred, pos_label=1, zero_division=0)),
        "accuracy": float(accuracy_score(y, pred)),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
        "roc_auc": float(roc_auc_score(y, p)) if both else None,
        "average_precision": float(average_precision_score(y, p)) if both else None,
        "log_loss": float(log_loss(y, p, labels=[0, 1])),
        "brier": float(brier_score_loss(y, p)),
    }


def _group_label(value: Any) -> str:
    if pd.isna(value):
        return "MISSING"
    return str(value)


def score_origin(
    frame: pd.DataFrame, origin: int, prediction_path: Path, y_score: np.ndarray,
    score_low: int, score_high: int, *, fit_text_q75: float,
) -> dict[str, Any]:
    rows, expected_low, expected_high = _expected_rows(frame, origin)
    if (score_low, score_high) != (expected_low, expected_high):
        raise ValueError("Scoring cycle range differs from origin contract")
    y = np.asarray(y_score, dtype=np.int8)
    if y.shape != (len(rows),) or not np.isin(y, (0, 1)).all():
        raise ValueError("Score labels must align to the prediction row list")
    with np.load(prediction_path, allow_pickle=False) as archive:
        prediction = {key: archive[key] for key in archive.files}
    if not np.array_equal(prediction["score_rows"], rows):
        raise ValueError("Score label row positions do not align to predictions")
    cycles = frame.iloc[rows]["cycle_code"].to_numpy(dtype=np.int64)
    present = np.asarray(prediction["present_mask"], dtype=bool)
    entry_friction = frame.iloc[rows]["entry_friction"].to_numpy(dtype=object)
    text_load = pd.to_numeric(frame.iloc[rows]["text_load"], errors="coerce").to_numpy(dtype=np.float64)
    q4 = np.isfinite(text_load) & (text_load >= fit_text_q75)
    q4_missing = ~np.isfinite(text_load)
    view_masks: dict[str, np.ndarray] = {}
    if origin == 67:
        view_masks = {
            "W4": (cycles >= 68) & (cycles <= 71),
            "W5": (cycles >= 72) & (cycles <= 75),
            "W6": (cycles >= 76) & (cycles <= 79),
            "H12_E67": (cycles >= 68) & (cycles <= 79),
        }
    else:
        view_masks = {"W6": (cycles >= 76) & (cycles <= 79)}

    probability_by_arm = {
        arm: np.asarray(prediction[f"prob_route_{arm}"], dtype=np.float64)
        for arm in ROUTE_NAMES
    }
    metrics_rows: list[dict[str, Any]] = []
    subgroup_rows: list[dict[str, Any]] = []
    cycle_rows: list[dict[str, Any]] = []
    q4_support_rows: list[dict[str, Any]] = []
    for view, view_mask in view_masks.items():
        for arm, probability in probability_by_arm.items():
            metrics_rows.append({"origin": origin, "view": view, "arm": arm,
                                 **_metric(y[view_mask], probability[view_mask])})
        subgroup_masks: list[tuple[str, str, np.ndarray]] = [
            ("stream_availability", "present", present),
            ("stream_availability", "missing_or_blank", ~present),
            ("fit_defined_text_load_q4", "Q4", q4),
            ("fit_defined_text_load_q4", "below_Q4", np.isfinite(text_load) & ~q4),
            ("fit_defined_text_load_q4", "missing_text_load", q4_missing),
        ]
        ef_values = sorted({_group_label(value) for value in entry_friction})
        for value in ef_values:
            subgroup_masks.append(("entry_friction", value,
                                   np.asarray([_group_label(x) == value for x in entry_friction], dtype=bool)))
        for group_name, group_value, group_mask in subgroup_masks:
            mask = view_mask & group_mask
            for arm, probability in probability_by_arm.items():
                subgroup_rows.append({
                    "origin": origin, "view": view, "group": group_name,
                    "group_value": group_value, "arm": arm,
                    **_metric(y[mask], probability[mask]),
                    "fit_text_load_q75": float(fit_text_q75),
                })
        for cycle in sorted(np.unique(cycles[view_mask]).tolist()):
            mask = view_mask & (cycles == cycle)
            for arm, probability in probability_by_arm.items():
                cycle_rows.append({"origin": origin, "view": view, "cycle_code": int(cycle),
                                   "arm": arm, **_metric(y[mask], probability[mask])})

    metric_lookup = {(row["view"], row["arm"]): row for row in metrics_rows}
    q4_lookup = {(row["view"], row["arm"]): row for row in subgroup_rows
                 if row["group"] == "fit_defined_text_load_q4" and row["group_value"] == "Q4"}
    checks: list[dict[str, Any]] = []
    if origin == 67:
        for comparator in ("bag", "shuffled_position"):
            actual = (metric_lookup[("H12_E67", "position")]["macro_f1"]
                      - metric_lookup[("H12_E67", comparator)]["macro_f1"])
            checks.append({"gate": f"H12_E67_gain_vs_{comparator}_min_0.003",
                           "actual": actual, "threshold": 0.003,
                           "pass": actual >= 0.003})
        for view in ("W4", "W5", "W6"):
            for comparator in ("bag", "shuffled_position"):
                actual = (metric_lookup[(view, "position")]["macro_f1"]
                          - metric_lookup[(view, comparator)]["macro_f1"])
                checks.append({"gate": f"{view}_harm_vs_{comparator}_max_0.005",
                               "actual": actual, "threshold": -0.005,
                               "pass": actual >= -0.005})
    else:
        for comparator in ("bag", "shuffled_position"):
            actual = (metric_lookup[("W6", "position")]["macro_f1"]
                      - metric_lookup[("W6", comparator)]["macro_f1"])
            checks.append({"gate": f"E75_W6_gain_vs_{comparator}_min_0",
                           "actual": actual, "threshold": 0.0,
                           "pass": actual >= 0.0})
    for view in view_masks:
        candidate = q4_lookup[(view, "position")]
        supported = candidate["n"] >= 200 and candidate["positive"] >= 30 and candidate["negative"] >= 30
        q4_support_rows.append({
            "view": view, "status": "SUPPORTED" if supported else "UNSUPPORTED",
            "n": candidate["n"], "positive": candidate["positive"],
            "negative": candidate["negative"], "minimum_n": 200,
            "minimum_each_class": 30,
        })
        for comparator in ("bag", "shuffled_position"):
            control = q4_lookup[(view, comparator)]
            if supported:
                f1_gain = candidate["macro_f1"] - control["macro_f1"]
                recall_gain = candidate["positive_recall"] - control["positive_recall"]
                checks.append({"gate": f"{view}_text_load_Q4_macro_f1_harm_vs_{comparator}_max_0.01",
                               "actual": f1_gain, "threshold": -0.01, "pass": f1_gain >= -0.01,
                               "support": {"n": candidate["n"], "positive": candidate["positive"],
                                           "negative": candidate["negative"]}})
                checks.append({"gate": f"{view}_text_load_Q4_positive_recall_harm_vs_{comparator}_max_0.02",
                               "actual": recall_gain, "threshold": -0.02, "pass": recall_gain >= -0.02,
                               "support": {"n": candidate["n"], "positive": candidate["positive"],
                                           "negative": candidate["negative"]}})
    gates_pass = all(item["pass"] for item in checks)
    q4_all_supported = all(item["status"] == "SUPPORTED" for item in q4_support_rows)
    if not gates_pass:
        status = "FAIL_PILOT_GATES"
        interpretation = "exact three-third recipe retires; no bin/seed/C/vocabulary/block-weight rescue"
    elif not q4_all_supported:
        status = "INCOMPLETE_UNSUPPORTED_Q4"
        interpretation = "Q4 gate is not fully evaluated; result does not advance until support is reviewed"
    else:
        status = "PASS_PILOT_GATES"
        interpretation = "mechanism advances for review only; no full-six qualification or champion replacement"
    result = {
        "origin": origin,
        "status": status,
        "fit_defined_text_load_q75": float(fit_text_q75),
        "views": metrics_rows,
        "subgroups": subgroup_rows,
        "per_cycle": cycle_rows,
        "q4_support": q4_support_rows,
        "gate_checks": checks,
        "interpretation": interpretation,
    }
    return result


def _metric_independent(y: np.ndarray, probability: np.ndarray) -> dict[str, Any]:
    """Direct confusion-count implementation, separate from sklearn F1 helpers."""
    y = np.asarray(y, dtype=np.int8)
    probability = np.asarray(probability, dtype=np.float64)
    if not len(y):
        return _metric(y, probability)
    predicted = (probability >= THRESHOLD).astype(np.int8)
    tn = int(np.sum((y == 0) & (predicted == 0)))
    fp = int(np.sum((y == 0) & (predicted == 1)))
    fn = int(np.sum((y == 1) & (predicted == 0)))
    tp = int(np.sum((y == 1) & (predicted == 1)))
    pos_recall = tp / (tp + fn) if tp + fn else 0.0
    pos_precision = tp / (tp + fp) if tp + fp else 0.0
    pos_f1 = (2 * pos_precision * pos_recall / (pos_precision + pos_recall)
              if pos_precision + pos_recall else 0.0)
    neg_recall = tn / (tn + fp) if tn + fp else 0.0
    neg_precision = tn / (tn + fn) if tn + fn else 0.0
    neg_f1 = (2 * neg_precision * neg_recall / (neg_precision + neg_recall)
              if neg_precision + neg_recall else 0.0)
    both = np.unique(y).size == 2
    return {
        "n": int(len(y)), "positive": int(y.sum()), "negative": int((y == 0).sum()),
        "macro_f1": float((pos_f1 + neg_f1) / 2), "positive_f1": float(pos_f1),
        "precision": float(pos_precision), "positive_recall": float(pos_recall),
        "accuracy": float((tn + tp) / len(y)), "tn": tn, "fp": fp, "fn": fn, "tp": tp,
        "roc_auc": float(roc_auc_score(y, probability)) if both else None,
        "average_precision": float(average_precision_score(y, probability)) if both else None,
        "log_loss": float(log_loss(y, probability, labels=[0, 1])),
        "brier": float(np.mean((probability - y) ** 2)),
    }


def verify_metrics_independently(result: dict[str, Any], frame: pd.DataFrame,
                                 origin: int, prediction_path: Path,
                                 y_score: np.ndarray, q75: float) -> dict[str, Any]:
    """Recompute every view/subgroup/cycle metric from labels and prediction arrays."""
    with np.load(prediction_path, allow_pickle=False) as archive:
        prediction = {key: archive[key] for key in archive.files}
    rows, _, _ = _expected_rows(frame, origin)
    y = np.asarray(y_score, dtype=np.int8)
    if (y.shape != (len(rows),) or not np.isin(y, (0, 1)).all()
            or not np.array_equal(prediction.get("score_rows"), rows)):
        raise AssertionError("Independent metric labels do not align with the frozen score rows")
    cycles = frame.iloc[rows]["cycle_code"].to_numpy(dtype=np.int64)
    present = ~_text_values(frame.iloc[rows]["signal_stream"])[1]
    text_load = pd.to_numeric(frame.iloc[rows]["text_load"], errors="coerce").to_numpy(dtype=np.float64)
    q4 = np.isfinite(text_load) & (text_load >= q75)
    ef = frame.iloc[rows]["entry_friction"].to_numpy(dtype=object)
    if origin == 67:
        views = {"W4": (cycles >= 68) & (cycles <= 71),
                 "W5": (cycles >= 72) & (cycles <= 75),
                 "W6": (cycles >= 76) & (cycles <= 79),
                 "H12_E67": (cycles >= 68) & (cycles <= 79)}
    else:
        views = {"W6": (cycles >= 76) & (cycles <= 79)}
    probabilities = {arm: np.asarray(prediction[f"prob_route_{arm}"], dtype=np.float64)
                     for arm in ROUTE_NAMES}

    def compare(row: dict[str, Any], mask: np.ndarray) -> None:
        expected = _metric_independent(y[mask], probabilities[row["arm"]][mask])
        for key, expected_value in expected.items():
            actual = row.get(key)
            if isinstance(expected_value, float):
                if actual is None or not np.isclose(float(actual), expected_value, atol=1e-12, rtol=0.0):
                    raise AssertionError(f"Independent {key} mismatch for {row}")
            elif actual != expected_value:
                raise AssertionError(f"Independent {key} mismatch for {row}")

    for row in result["views"]:
        compare(row, views[row["view"]])
    for row in result["subgroups"]:
        if row["group"] == "stream_availability":
            subgroup = present if row["group_value"] == "present" else ~present
        elif row["group"] == "fit_defined_text_load_q4":
            if row["group_value"] == "Q4":
                subgroup = q4
            elif row["group_value"] == "below_Q4":
                subgroup = np.isfinite(text_load) & ~q4
            else:
                subgroup = ~np.isfinite(text_load)
        elif row["group"] == "entry_friction":
            subgroup = np.asarray([_group_label(v) == row["group_value"] for v in ef], dtype=bool)
        else:
            raise AssertionError(f"Unexpected subgroup in result: {row['group']}")
        compare(row, views[row["view"]] & subgroup)
    for row in result["per_cycle"]:
        compare(row, views[row["view"]] & (cycles == row["cycle_code"]))

    by_view_arm = {(row["view"], row["arm"]): row for row in result["views"]}
    q4_by_view_arm = {(row["view"], row["arm"]): row for row in result["subgroups"]
                      if row["group"] == "fit_defined_text_load_q4" and row["group_value"] == "Q4"}
    expected_support = []
    for view in views:
        candidate = q4_by_view_arm[(view, "position")]
        supported = candidate["n"] >= 200 and candidate["positive"] >= 30 and candidate["negative"] >= 30
        expected_support.append({
            "view": view, "status": "SUPPORTED" if supported else "UNSUPPORTED",
            "n": candidate["n"], "positive": candidate["positive"],
            "negative": candidate["negative"], "minimum_n": 200,
            "minimum_each_class": 30,
        })
    if result.get("q4_support") != expected_support:
        raise AssertionError(f"Independent Q4 support verification mismatch for E{origin}")
    expected_checks = []
    if origin == 67:
        for comparator in ("bag", "shuffled_position"):
            gain = (by_view_arm[("H12_E67", "position")]["macro_f1"]
                    - by_view_arm[("H12_E67", comparator)]["macro_f1"])
            expected_checks.append((f"H12_E67_gain_vs_{comparator}_min_0.003", gain, 0.003, gain >= 0.003))
        for view in ("W4", "W5", "W6"):
            for comparator in ("bag", "shuffled_position"):
                gain = (by_view_arm[(view, "position")]["macro_f1"]
                        - by_view_arm[(view, comparator)]["macro_f1"])
                expected_checks.append((f"{view}_harm_vs_{comparator}_max_0.005", gain, -0.005, gain >= -0.005))
    else:
        for comparator in ("bag", "shuffled_position"):
            gain = (by_view_arm[("W6", "position")]["macro_f1"]
                    - by_view_arm[("W6", comparator)]["macro_f1"])
            expected_checks.append((f"E75_W6_gain_vs_{comparator}_min_0", gain, 0.0, gain >= 0.0))
    for view in views:
        candidate = q4_by_view_arm[(view, "position")]
        supported = candidate["n"] >= 200 and candidate["positive"] >= 30 and candidate["negative"] >= 30
        if not supported:
            continue
        for comparator in ("bag", "shuffled_position"):
            control = q4_by_view_arm[(view, comparator)]
            f1_gain = candidate["macro_f1"] - control["macro_f1"]
            recall_gain = candidate["positive_recall"] - control["positive_recall"]
            expected_checks.append((f"{view}_text_load_Q4_macro_f1_harm_vs_{comparator}_max_0.01",
                                    f1_gain, -0.01, f1_gain >= -0.01))
            expected_checks.append((f"{view}_text_load_Q4_positive_recall_harm_vs_{comparator}_max_0.02",
                                    recall_gain, -0.02, recall_gain >= -0.02))
    actual_checks = {row["gate"]: row for row in result["gate_checks"]}
    if len(actual_checks) != len(result["gate_checks"]) or len(actual_checks) != len(expected_checks):
        raise AssertionError(f"Independent gate count mismatch for E{origin}")
    for gate, actual, threshold, passed in expected_checks:
        row = actual_checks[gate]
        if (not np.isclose(float(row["actual"]), actual, atol=1e-12, rtol=0.0)
                or row["threshold"] != threshold or row["pass"] != passed):
            raise AssertionError(f"Independent gate mismatch: {gate}")
    gate_checks_pass = all(item[3] for item in expected_checks)
    q4_all_supported = all(item["status"] == "SUPPORTED" for item in expected_support)
    expected_status = ("FAIL_PILOT_GATES" if not gate_checks_pass else
                       "INCOMPLETE_UNSUPPORTED_Q4" if not q4_all_supported else
                       "PASS_PILOT_GATES")
    if result.get("status") != expected_status:
        raise AssertionError(f"Independent overall pilot status mismatch for E{origin}")
    return {"origin": origin, "status": "PASS_METRIC_AND_GATE_RECOMPUTATION",
            "metric_row_count": len(result["views"]),
            "subgroup_row_count": len(result["subgroups"]),
            "cycle_row_count": len(result["per_cycle"]),
            "gate_check_count": len(result["gate_checks"]),
            "independent_method": "direct confusion-count formulas plus independent support/view masks"}


__all__ = ["score_origin", "verify_metrics_independently", "verify_prediction_bundle"]
