"""Independent, in-memory verifier for CLS_PPMI_CONTEXT_V1_CORRECTION_R1."""
from __future__ import annotations

import hashlib
import json
from typing import Any, Mapping

import numpy as np
from sklearn.metrics import (accuracy_score, average_precision_score, confusion_matrix,
                             f1_score, log_loss, precision_score, recall_score,
                             roc_auc_score)


ORIGINS = (67, 75)
PRESENT_THRESHOLD = 0.50
MISSING_THRESHOLD = 0.53230976129477
EXPECTED_VIEWS = {
    "E67_W4": (67, 68, 71, 4_950),
    "E67_W5": (67, 72, 75, 7_462),
    "E67_W6": (67, 76, 79, 8_491),
    "E67_H12": (67, 68, 79, 20_903),
    "E75_W6": (75, 76, 79, 8_491),
}
ALL_VIEWS = (*EXPECTED_VIEWS, "PRIMARY_UNIQUE")


class IndependentVerificationError(RuntimeError):
    """Independent correction verifier could not reproduce a saved claim."""


def _rows_digest(values: Any) -> str:
    return hashlib.sha256(np.ascontiguousarray(np.asarray(values, dtype="<i8")).tobytes()).hexdigest()


def _ids_digest(values: Any) -> str:
    encoded = json.dumps([str(value) for value in np.asarray(values).reshape(-1)],
                         ensure_ascii=False, allow_nan=False, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _metric(y: np.ndarray, p: np.ndarray, threshold: np.ndarray) -> dict[str, Any]:
    decision = (p >= threshold).astype(np.int64)
    cm = confusion_matrix(y, decision, labels=[0, 1])
    per_class = f1_score(y, decision, labels=[0, 1], average=None, zero_division=0)
    macro = float(f1_score(y, decision, labels=[0, 1], average="macro", zero_division=0))
    unique = np.unique(y)
    return {
        "n": int(len(y)), "positive": int(y.sum()), "negative": int(len(y) - y.sum()),
        "confusion_matrix_labels_0_1": cm.astype(int).tolist(),
        "class_f1": {"0": float(per_class[0]), "1": float(per_class[1])},
        "macro_f1": macro, "sklearn_macro_f1": macro,
        "positive_class_f1": float(per_class[1]),
        "precision": float(precision_score(y, decision, zero_division=0)),
        "positive_recall": float(recall_score(y, decision, zero_division=0)),
        "negative_recall": float(recall_score(y, decision, pos_label=0, zero_division=0)),
        "accuracy": float(accuracy_score(y, decision)),
        "average_precision": float(average_precision_score(y, p)) if len(unique) == 2 else None,
        "roc_auc": float(roc_auc_score(y, p)) if len(unique) == 2 else None,
        "log_loss": float(log_loss(y, np.clip(p, 1e-15, 1.0 - 1e-15), labels=[0, 1])),
        "brier_score": float(np.mean(np.square(p - y))),
    }


def _bundle(y: np.ndarray, arrays: Mapping[str, np.ndarray],
            threshold: np.ndarray) -> dict[str, Any]:
    views = {
        "champion": _metric(y, arrays["p_champion"], threshold),
        "candidate_real": _metric(y, arrays["p_candidate_real"], threshold),
        "control_shuffled": _metric(y, arrays["p_control_shuffled"], threshold),
    }
    views["gains"] = {
        "candidate_real_minus_champion_macro_f1": views["candidate_real"]["macro_f1"] - views["champion"]["macro_f1"],
        "candidate_real_minus_control_shuffled_macro_f1": views["candidate_real"]["macro_f1"] - views["control_shuffled"]["macro_f1"],
        "control_shuffled_minus_champion_macro_f1": views["control_shuffled"]["macro_f1"] - views["champion"]["macro_f1"],
    }
    return views


def _rebuild_view(name: str, origin: int, mask: np.ndarray, data: Mapping[str, np.ndarray],
                  labels: Mapping[str, int]) -> dict[str, Any]:
    ids = data["score_record_ids"][mask].astype(str)
    if any(record_id not in labels for record_id in ids.tolist()):
        raise IndependentVerificationError(f"No selected label in memory for {name}")
    y = np.asarray([labels[record_id] for record_id in ids], dtype=np.int64)
    threshold = data["decision_thresholds"][mask].astype(np.float64)
    arrays = {key: data[key][mask].astype(np.float64)
              for key in ("p_champion", "p_candidate_real", "p_control_shuffled")}
    cycles = data["score_cycles"][mask].astype(np.int64)
    rows = data["score_rows"][mask].astype(np.int64)
    present = data["present_mask"][mask].astype(bool)
    return {
        "name": name, "rows": int(mask.sum()), "origin": origin,
        "cycle_min": int(cycles.min()), "cycle_max": int(cycles.max()),
        "present_rows": int(present.sum()), "missing_rows": int((~present).sum()),
        "metrics": _bundle(y, arrays, threshold),
        "rows_array": rows, "ids_array": ids, "cycles_array": cycles,
        "rows_sha256": _rows_digest(rows), "ids_sha256": _ids_digest(ids),
        "labels_array": y, "present_array": present,
        "entry_friction_array": data["entry_friction"][mask].astype(str),
        "text_group_array": data["text_load_group"][mask].astype(str),
        "threshold_array": threshold, "prediction_arrays": arrays,
    }


def _rebuild_subgroups(view: Mapping[str, Any]) -> dict[str, Any]:
    dimensions = {
        "availability": np.where(view["present_array"], "present", "missing"),
        "entry_friction": view["entry_friction_array"],
        "inherited_text_load_quartile": view["text_group_array"],
    }
    groups: dict[str, Any] = {}
    for dimension, values in dimensions.items():
        for category in sorted(set(values.tolist())):
            selected = values == category
            y = view["labels_array"][selected]
            support = {"n": int(selected.sum()), "positive": int(y.sum()),
                       "negative": int(len(y) - y.sum())}
            supported = support["n"] >= 200 and support["positive"] >= 30 and support["negative"] >= 30
            arrays = {key: value[selected] for key, value in view["prediction_arrays"].items()}
            threshold = view["threshold_array"][selected]
            metrics = _bundle(y, arrays, threshold) if len(y) else None
            f1_harm = recall_harm = None
            if metrics is not None:
                f1_harm = max(0.0, metrics["champion"]["macro_f1"] - metrics["candidate_real"]["macro_f1"])
                recall_harm = max(0.0, metrics["champion"]["positive_recall"]
                                  - metrics["candidate_real"]["positive_recall"])
            groups[f"{dimension}={category}"] = {
                "dimension": dimension, "category": category, "support": support,
                "supported_for_gate": bool(supported), "metrics": metrics,
                "candidate_macro_f1_harm": f1_harm,
                "candidate_positive_recall_harm": recall_harm,
            }
    return groups


def _rebuild_primary(data: Mapping[int, Mapping[str, np.ndarray]],
                     labels: Mapping[str, int]) -> dict[str, Any]:
    pieces = []
    for origin, low, high in ((67, 68, 75), (75, 76, 79)):
        item = data[origin]
        mask = (item["score_cycles"] >= low) & (item["score_cycles"] <= high)
        pieces.append(_rebuild_view(f"PRIMARY_E{origin}_{low}_{high}", origin, mask, item, labels))
    ids = np.concatenate([piece["ids_array"] for piece in pieces])
    rows = np.concatenate([piece["rows_array"] for piece in pieces])
    if len(set(ids.tolist())) != 20_903 or len(set(rows.tolist())) != 20_903:
        raise IndependentVerificationError("Primary unique view has duplicate row IDs or source rows")
    order = np.argsort(rows, kind="stable")
    joined = {}
    for key in ("labels_array", "cycles_array", "present_array", "entry_friction_array",
                "text_group_array", "threshold_array"):
        joined[key] = np.concatenate([piece[key] for piece in pieces])[order]
    joined["prediction_arrays"] = {
        key: np.concatenate([piece["prediction_arrays"][key] for piece in pieces])[order]
        for key in ("p_champion", "p_candidate_real", "p_control_shuffled")
    }
    joined["rows_array"] = rows[order]
    joined["ids_array"] = ids[order]
    joined["rows_sha256"] = _rows_digest(joined["rows_array"])
    joined["ids_sha256"] = _ids_digest(joined["ids_array"])
    joined["present_rows"] = int(joined["present_array"].sum())
    joined["missing_rows"] = int((~joined["present_array"]).sum())
    joined["rows"] = int(len(rows))
    joined["cycle_min"] = int(joined["cycles_array"].min())
    joined["cycle_max"] = int(joined["cycles_array"].max())
    joined["origin"] = "E67_68_75+E75_76_79"
    joined["name"] = "PRIMARY_UNIQUE"
    joined["metrics"] = _bundle(joined["labels_array"], joined["prediction_arrays"],
                                 joined["threshold_array"])
    return joined


def _rebuild_gates(views: Mapping[str, Mapping[str, Any]],
                   subgroup_views: Mapping[str, Mapping[str, Any]],
                   missing_route_exact: bool) -> dict[str, Any]:
    primary = views["PRIMARY_UNIQUE"]["metrics"]
    e67_h12 = views["E67_H12"]["metrics"]
    e75_w6 = views["E75_W6"]["metrics"]
    primary_gain = primary["gains"]["candidate_real_minus_champion_macro_f1"]
    e67_gain = e67_h12["gains"]["candidate_real_minus_champion_macro_f1"]
    e75_gain = e75_w6["gains"]["candidate_real_minus_champion_macro_f1"]
    primary_control = primary["gains"]["candidate_real_minus_control_shuffled_macro_f1"]
    e75_real_shuffle = e75_w6["gains"]["candidate_real_minus_control_shuffled_macro_f1"]
    supported = [group for groups in subgroup_views.values() for group in groups.values()
                 if group["supported_for_gate"]]
    group_views = {name: sum(group["supported_for_gate"] for group in groups.values())
                   for name, groups in subgroup_views.items()}
    worst_f1 = max((float(group["candidate_macro_f1_harm"]) for group in supported), default=None)
    worst_recall = max((float(group["candidate_positive_recall_harm"]) for group in supported), default=None)
    scientific_checks = {
        "primary_gain_vs_champion_ge_0_0005": primary_gain >= 0.0005,
        "e67_h12_gain_nonnegative": e67_gain >= 0.0,
        "e75_w6_gain_nonnegative": e75_gain >= 0.0,
        "primary_candidate_minus_control_ge_0_0002": primary_control >= 0.0002,
        "e75_real_minus_shuffled_nonnegative": e75_real_shuffle >= 0.0,
        "all_view_supported_subgroup_macro_f1_harm_le_0_01": worst_f1 is not None and worst_f1 <= 0.01,
        "all_view_supported_subgroup_positive_recall_harm_le_0_02": worst_recall is not None and worst_recall <= 0.02,
        "prelabel_qa_and_missing_route_exact": bool(missing_route_exact),
    }
    exploratory_pattern = ((primary_gain > 0 and e75_gain >= -0.0005)
                           or (e75_gain >= 0.0005 and primary_gain >= -0.0005))
    exploratory_checks = {
        "candidate_gain_pattern": bool(exploratory_pattern),
        "all_view_supported_subgroup_macro_f1_harm_le_0_02": worst_f1 is not None and worst_f1 <= 0.02,
        "all_view_supported_subgroup_positive_recall_harm_le_0_03": worst_recall is not None and worst_recall <= 0.03,
        "prelabel_qa_and_missing_route_exact": bool(missing_route_exact),
    }
    mechanism_checks = {
        "primary_candidate_minus_control_ge_0_0002": primary_control >= 0.0002,
        "e75_real_minus_shuffled_nonnegative": e75_real_shuffle >= 0.0,
    }
    return {
        "scientific": {"status": "PASS" if all(scientific_checks.values()) else "FAIL",
                       "checks": scientific_checks,
                       "primary_gain": primary_gain, "e67_h12_gain": e67_gain,
                       "e75_w6_gain": e75_gain,
                       "primary_candidate_minus_control": primary_control,
                       "e75_real_minus_shuffled": e75_real_shuffle,
                       "worst_supported_macro_f1_harm_all_views": worst_f1,
                       "worst_supported_positive_recall_harm_all_views": worst_recall,
                       "supported_subgroup_count_across_views": int(len(supported)),
                       "supported_groups_by_view": group_views},
        "exploratory": {"status": "ELIGIBLE" if all(exploratory_checks.values()) else "NOT_ELIGIBLE",
                         "checks": exploratory_checks},
        "mechanism": {"status": "SEPARATED_FROM_SHUFFLED_CONTROL"
                      if all(mechanism_checks.values()) else "NOT_SEPARATED_FROM_SHUFFLED_CONTROL",
                      "checks": mechanism_checks},
    }


def _compare_numeric(expected: Any, actual: Any, path: str, errors: list[str]) -> None:
    if isinstance(expected, dict):
        if not isinstance(actual, dict) or set(expected) != set(actual):
            errors.append(f"{path}: key contract differs")
            return
        for key in expected:
            _compare_numeric(expected[key], actual[key], f"{path}.{key}", errors)
    elif isinstance(expected, list):
        if not isinstance(actual, list) or len(expected) != len(actual):
            errors.append(f"{path}: list contract differs")
            return
        for index, (left, right) in enumerate(zip(expected, actual, strict=True)):
            _compare_numeric(left, right, f"{path}[{index}]", errors)
    elif isinstance(expected, (int, float)) and not isinstance(expected, bool):
        if not isinstance(actual, (int, float)) or abs(float(expected) - float(actual)) > 1e-12:
            errors.append(f"{path}: numeric mismatch")
    elif expected != actual:
        errors.append(f"{path}: value mismatch")


def verify_in_memory(predictions: Mapping[int, Mapping[str, Any]],
                     labels_by_id: Mapping[str, int],
                     scored: Mapping[str, Any],
                     anchors: Mapping[str, Any]) -> dict[str, Any]:
    """Recompute every view, subgroup, frozen metric, and gate from the one RAM vector."""
    if len(labels_by_id) != 20_903:
        raise IndependentVerificationError("Verifier received other than the one authorized 20,903-row label map")
    data: dict[int, dict[str, np.ndarray]] = {}
    for origin in ORIGINS:
        source = predictions[origin]
        data[origin] = {key: np.asarray(source[key]).reshape(-1) for key in (
            "score_rows", "score_record_ids", "score_cycles", "present_mask", "p_champion",
            "p_candidate_real", "p_control_shuffled", "decision_thresholds",
            "entry_friction", "text_load_group")}
    rebuilt: dict[str, dict[str, Any]] = {}
    for name, (origin, low, high, count) in EXPECTED_VIEWS.items():
        item = data[origin]
        mask = (item["score_cycles"] >= low) & (item["score_cycles"] <= high)
        if int(mask.sum()) != count:
            raise IndependentVerificationError(f"Independent row count mismatch for {name}")
        rebuilt[name] = _rebuild_view(name, origin, mask, item, labels_by_id)
    rebuilt["PRIMARY_UNIQUE"] = _rebuild_primary(data, labels_by_id)
    if set(labels_by_id) != set(rebuilt["PRIMARY_UNIQUE"]["ids_array"].tolist()):
        raise IndependentVerificationError("In-memory label map is not exactly the selected primary ID set")
    if set(scored.get("views", {})) != set(ALL_VIEWS):
        raise IndependentVerificationError("Saved corrected result does not contain exactly six intended views")
    expected_sources = {name: (f"E{spec[0]}" if name != "PRIMARY_UNIQUE"
                               else "E67_cycles_68_75 + E75_cycles_76_79")
                        for name, spec in EXPECTED_VIEWS.items()}
    expected_sources["PRIMARY_UNIQUE"] = "E67_cycles_68_75 + E75_cycles_76_79"
    for name in ALL_VIEWS:
        view = rebuilt[name]
        saved = scored["views"][name]
        expected_source = expected_sources[name]
        if saved.get("rows") != view["rows"] or saved.get("prediction_source") != expected_source \
                or saved.get("source_rows_sha256") != view["rows_sha256"] \
                or saved.get("record_ids_sha256") != view["ids_sha256"] \
                or saved.get("present_rows") != view["present_rows"] \
                or saved.get("missing_rows") != view["missing_rows"] \
                or saved.get("cycle_min") != view["cycle_min"] \
                or saved.get("cycle_max") != view["cycle_max"]:
            raise IndependentVerificationError(f"Saved view identity differs for {name}")
        for metric_name in ("champion", "candidate_real", "control_shuffled", "gains"):
            errors: list[str] = []
            _compare_numeric(view["metrics"][metric_name], saved["metrics"].get(metric_name),
                             f"views.{name}.metrics.{metric_name}", errors)
            if errors:
                raise IndependentVerificationError("; ".join(errors[:3]))
    ids = {name: set(rebuilt[name]["ids_array"].tolist()) for name in ALL_VIEWS}
    rows = {name: set(rebuilt[name]["rows_array"].tolist()) for name in ALL_VIEWS}
    if ids["E67_W4"] & ids["E67_W5"] or ids["E67_W4"] & ids["E67_W6"] \
            or ids["E67_W5"] & ids["E67_W6"]:
        raise IndependentVerificationError("E67 W4/W5/W6 ID sets overlap")
    if rows["E67_W4"] & rows["E67_W5"] or rows["E67_W4"] & rows["E67_W6"] \
            or rows["E67_W5"] & rows["E67_W6"]:
        raise IndependentVerificationError("E67 W4/W5/W6 source rows overlap")
    if ids["E67_W4"] | ids["E67_W5"] | ids["E67_W6"] != ids["E67_H12"] \
            or rows["E67_W4"] | rows["E67_W5"] | rows["E67_W6"] != rows["E67_H12"]:
        raise IndependentVerificationError("E67 windows do not exactly cover H12")
    if ids["E67_W6"] != ids["E75_W6"] or rows["E67_W6"] != rows["E75_W6"]:
        raise IndependentVerificationError("E67 and E75 W6 source IDs/rows differ")
    if ids["PRIMARY_UNIQUE"] != ids["E67_W4"] | ids["E67_W5"] | ids["E75_W6"] \
            or rows["PRIMARY_UNIQUE"] != rows["E67_W4"] | rows["E67_W5"] | rows["E75_W6"]:
        raise IndependentVerificationError("Primary unique view uses the wrong source windows")
    subgroup_views = {name: _rebuild_subgroups(view) for name, view in rebuilt.items()}
    if set(scored.get("subgroups_by_view", {})) != set(ALL_VIEWS):
        raise IndependentVerificationError("Subgroup guards are not present for all six views")
    for name in ALL_VIEWS:
        errors: list[str] = []
        _compare_numeric(subgroup_views[name], scored["subgroups_by_view"][name],
                         f"subgroups_by_view.{name}", errors)
        if errors:
            raise IndependentVerificationError("; ".join(errors[:3]))
    missing_exact = True
    for origin in ORIGINS:
        item = data[origin]
        missing = ~item["present_mask"].astype(bool)
        missing_exact = missing_exact and np.array_equal(item["p_candidate_real"][missing], item["p_champion"][missing])
        missing_exact = missing_exact and np.array_equal(item["p_control_shuffled"][missing], item["p_champion"][missing])
    gates = _rebuild_gates(rebuilt, subgroup_views, bool(missing_exact))
    errors = []
    _compare_numeric(gates, scored.get("gates"), "gates", errors)
    if errors:
        raise IndependentVerificationError("; ".join(errors[:3]))
    overlap_expected = {
        "E67_W4_vs_E67_W5_shared_ids": 0,
        "E67_W4_vs_E67_W6_shared_ids": 0,
        "E67_W5_vs_E67_W6_shared_ids": 0,
        "E67_H12_equals_disjoint_W4_W5_W6": True,
        "E67_W6_equals_E75_W6_ids_and_rows": True,
        "PRIMARY_UNIQUE_uses_E67_W4_W5_and_E75_W6": True,
        "E67_H12_and_PRIMARY_UNIQUE_shared_ids": len(ids["E67_H12"] & ids["PRIMARY_UNIQUE"]),
    }
    if scored.get("window_overlap_checks") != overlap_expected:
        raise IndependentVerificationError("Saved window overlap receipt differs from independent row/ID audit")
    anchors_checked: dict[str, Any] = {}
    for view_name, expected_by_model in anchors.items():
        if view_name not in rebuilt:
            raise IndependentVerificationError(f"Unknown authorized macro-F1 anchor view {view_name}")
        anchors_checked[view_name] = {}
        for model_name, expected in expected_by_model.items():
            actual = rebuilt[view_name]["metrics"][model_name]["macro_f1"]
            passed = abs(actual - float(expected)) <= 1e-12
            anchors_checked[view_name][model_name] = {"expected": float(expected),
                                                       "actual": actual, "pass": passed}
            if not passed:
                raise IndependentVerificationError(f"Authorized macro-F1 anchor mismatch: {view_name}.{model_name}")
    return {
        "status": "PASS", "method": "independent sklearn MacroF1 and metric recomputation from the single in-memory selected target vector",
        "views_recomputed": len(rebuilt), "subgroup_views_recomputed": len(subgroup_views),
        "subgroup_groups_recomputed": int(sum(len(groups) for groups in subgroup_views.values())),
        "all_frozen_metrics_recomputed": True, "all_gates_recomputed": True,
        "all_supported_subgroups_recomputed_across_all_views": True,
        "macro_f1_root_anchors": anchors_checked,
        "query_target_vector_rows": len(labels_by_id),
        "additional_target_reads": 0,
    }


def synthetic_gate_verification() -> dict[str, Any]:
    """Unit fixture for the E75 real-minus-shuffle sign, independent of production data."""
    # Small exact-support vectors are enough to lock the comparison direction.
    y = np.asarray([0, 0, 1, 1], dtype=np.int64)
    champ = _metric(y, np.asarray([0.1, 0.1, 0.9, 0.9]), np.full(4, 0.5))
    real = _metric(y, np.asarray([0.1, 0.8, 0.9, 0.9]), np.full(4, 0.5))
    shuffled = _metric(y, np.asarray([0.8, 0.8, 0.9, 0.9]), np.full(4, 0.5))
    real_minus_shuffled = real["macro_f1"] - shuffled["macro_f1"]
    shuffled_minus_champion = shuffled["macro_f1"] - champ["macro_f1"]
    if not (real_minus_shuffled > 0 and shuffled_minus_champion < 0):
        raise IndependentVerificationError("E75 sign-direction fixture is not adversarial")
    return {"status": "PASS", "real_minus_shuffled": real_minus_shuffled,
            "shuffled_minus_champion": shuffled_minus_champion,
            "signs_differ": True, "intended_gate_source": "real_minus_shuffled"}
