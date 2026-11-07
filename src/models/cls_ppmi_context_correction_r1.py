"""Pure scoring logic for the additive CLS PPMI contract correction R1."""
from __future__ import annotations

import hashlib
import json
from typing import Any, Mapping

import numpy as np
from sklearn.metrics import (average_precision_score, f1_score, log_loss,
                             roc_auc_score)


PILOT_ID = "CLS_PPMI_CONTEXT_V1"
CORRECTION_ID = "CLS_PPMI_CONTEXT_V1_CORRECTION_R1"
ORIGINS = (67, 75)
MISSING_THRESHOLD = 0.53230976129477
PRESENT_THRESHOLD = 0.50
PRIMARY_ROWS = 20_903
WINDOW_SPECS = (
    ("E67_W4", 67, 68, 71, 4_950),
    ("E67_W5", 67, 72, 75, 7_462),
    ("E67_W6", 67, 76, 79, 8_491),
    ("E67_H12", 67, 68, 79, 20_903),
    ("E75_W6", 75, 76, 79, 8_491),
)
VIEW_NAMES = tuple(item[0] for item in WINDOW_SPECS) + ("PRIMARY_UNIQUE",)
TEXT_LOAD_GROUPS = ("0", "1", "2", "3")
ENTRY_FRICTION_GROUPS = ("0", "1")
PROBABILITY_KEYS = {
    "champion": "p_champion",
    "candidate_real": "p_candidate_real",
    "control_shuffled": "p_control_shuffled",
}


class CorrectionError(RuntimeError):
    """Fail-closed contract violation for the corrected scoring pass."""


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, ensure_ascii=False,
                         allow_nan=False, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def bytes_sha256(values: Any) -> str:
    return hashlib.sha256(np.ascontiguousarray(values).tobytes()).hexdigest()


def id_sequence_sha256(values: Any) -> str:
    sequence = [str(value) for value in np.asarray(values).reshape(-1)]
    return canonical_sha256(sequence)


def _as_vector(prediction: Mapping[str, Any], key: str, dtype: Any | None = None) -> np.ndarray:
    if key not in prediction:
        raise CorrectionError(f"Frozen prediction is missing {key}")
    values = np.asarray(prediction[key]).reshape(-1)
    if dtype is not None:
        values = values.astype(dtype, copy=False)
    return values


def _metric_manual_confusion(y: np.ndarray, probability: np.ndarray,
                             threshold: np.ndarray) -> dict[str, Any]:
    decision = (probability >= threshold).astype(np.int64)
    code = 2 * y.astype(np.int64) + decision
    counts = np.bincount(code, minlength=4)
    tn, fp, fn, tp = (int(counts[0]), int(counts[1]), int(counts[2]), int(counts[3]))
    f1_zero_denom = 2 * tn + fp + fn
    f1_one_denom = 2 * tp + fp + fn
    f1_zero = (2.0 * tn / f1_zero_denom) if f1_zero_denom else 0.0
    f1_one = (2.0 * tp / f1_one_denom) if f1_one_denom else 0.0
    macro = (f1_zero + f1_one) / 2.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    specificity = tn / (tn + fp) if (tn + fp) else 0.0
    unique_labels = np.unique(y)
    ap = float(average_precision_score(y, probability)) if len(unique_labels) == 2 else None
    auc = float(roc_auc_score(y, probability)) if len(unique_labels) == 2 else None
    sklearn_macro = float(f1_score(y, decision, labels=[0, 1], average="macro", zero_division=0))
    if not np.isclose(macro, sklearn_macro, rtol=0.0, atol=1e-14):
        raise CorrectionError("Manual MacroF1 differs from sklearn labels=[0,1], average=macro")
    return {
        "n": int(len(y)), "positive": int(y.sum()), "negative": int(len(y) - y.sum()),
        "confusion_matrix_labels_0_1": [[tn, fp], [fn, tp]],
        "class_f1": {"0": float(f1_zero), "1": float(f1_one)},
        "macro_f1": float(macro), "sklearn_macro_f1": sklearn_macro,
        "positive_class_f1": float(f1_one), "precision": float(precision),
        "positive_recall": float(recall), "negative_recall": float(specificity),
        "accuracy": float((tn + tp) / len(y)), "average_precision": ap,
        "roc_auc": auc,
        "log_loss": float(log_loss(y, np.clip(probability, 1e-15, 1.0 - 1e-15), labels=[0, 1])),
        "brier_score": float(np.mean((probability - y) ** 2)),
    }


def _metric_bundle(y: np.ndarray, predictions: Mapping[str, np.ndarray],
                   thresholds: np.ndarray) -> dict[str, Any]:
    metrics = {name: _metric_manual_confusion(y, np.asarray(predictions[key], dtype=np.float64), thresholds)
               for name, key in PROBABILITY_KEYS.items()}
    champion = metrics["champion"]
    real = metrics["candidate_real"]
    shuffled = metrics["control_shuffled"]
    metrics["gains"] = {
        "candidate_real_minus_champion_macro_f1": real["macro_f1"] - champion["macro_f1"],
        "candidate_real_minus_control_shuffled_macro_f1": real["macro_f1"] - shuffled["macro_f1"],
        "control_shuffled_minus_champion_macro_f1": shuffled["macro_f1"] - champion["macro_f1"],
    }
    return metrics


def validate_predictions(predictions: Mapping[int, Mapping[str, Any]]) -> dict[str, Any]:
    if set(int(origin) for origin in predictions) != set(ORIGINS):
        raise CorrectionError("Exactly E67 and E75 frozen prediction arrays are required")
    normalized: dict[int, dict[str, np.ndarray]] = {}
    for origin in ORIGINS:
        raw = predictions[origin]
        required = {
            "origin", "score_rows", "score_record_ids", "score_cycles", "present_mask",
            "p_champion", "p_candidate_real", "p_control_shuffled", "decision_thresholds",
            "entry_friction", "text_load_group",
        }
        if not required.issubset(raw):
            raise CorrectionError(f"E{origin} prediction arrays lack required fields")
        if any("label" in key.lower() or key == "artifact_signal" for key in raw):
            raise CorrectionError(f"E{origin} prediction archive contains a label field")
        rows = _as_vector(raw, "score_rows", np.int64)
        ids = _as_vector(raw, "score_record_ids").astype(str)
        cycles = _as_vector(raw, "score_cycles", np.int64)
        present = _as_vector(raw, "present_mask", bool)
        p_champion = _as_vector(raw, "p_champion", np.float64)
        p_real = _as_vector(raw, "p_candidate_real", np.float64)
        p_shuffled = _as_vector(raw, "p_control_shuffled", np.float64)
        thresholds = _as_vector(raw, "decision_thresholds", np.float64)
        entry = _as_vector(raw, "entry_friction").astype(str)
        text_group = _as_vector(raw, "text_load_group").astype(str)
        n = len(rows)
        arrays = (ids, cycles, present, p_champion, p_real, p_shuffled, thresholds, entry, text_group)
        if any(len(value) != n for value in arrays):
            raise CorrectionError(f"E{origin} prediction arrays are misaligned")
        if int(np.asarray(raw["origin"]).reshape(-1)[0]) != origin:
            raise CorrectionError(f"E{origin} archive origin field is wrong")
        if len(set(rows.tolist())) != n or len(set(ids.tolist())) != n:
            raise CorrectionError(f"E{origin} score rows and IDs must each be unique")
        if np.any(rows < 0) or np.any(rows >= 50_608) or not np.isin(cycles, np.arange(47, 80)).all():
            raise CorrectionError(f"E{origin} source rows/cycles are outside frozen Train bounds")
        expected_cycles = np.arange(68, 80) if origin == 67 else np.arange(76, 80)
        if set(cycles.tolist()) != set(expected_cycles.tolist()):
            raise CorrectionError(f"E{origin} query cycles do not match the frozen contract")
        if not np.isfinite(np.column_stack((p_champion, p_real, p_shuffled, thresholds))).all():
            raise CorrectionError(f"E{origin} probability/threshold arrays are nonfinite")
        if any(np.any((p < 0.0) | (p > 1.0)) for p in (p_champion, p_real, p_shuffled)):
            raise CorrectionError(f"E{origin} probability is outside [0,1]")
        expected_thresholds = np.where(present, PRESENT_THRESHOLD, MISSING_THRESHOLD)
        if not np.array_equal(thresholds, expected_thresholds):
            raise CorrectionError(f"E{origin} frozen threshold route changed")
        if not np.array_equal(p_real[~present], p_champion[~present]) \
                or not np.array_equal(p_shuffled[~present], p_champion[~present]):
            raise CorrectionError(f"E{origin} missing-stream probability route is not exact")
        if not set(entry).issubset(set(ENTRY_FRICTION_GROUPS)):
            raise CorrectionError(f"E{origin} entry-friction groups are outside frozen categories")
        if not set(text_group).issubset(set(TEXT_LOAD_GROUPS)):
            raise CorrectionError(f"E{origin} inherited text-load groups are not four frozen quartiles")
        normalized[origin] = {
            "score_rows": rows, "score_record_ids": ids, "score_cycles": cycles,
            "present_mask": present, "p_champion": p_champion,
            "p_candidate_real": p_real, "p_control_shuffled": p_shuffled,
            "decision_thresholds": thresholds, "entry_friction": entry,
            "text_load_group": text_group,
        }
    if len(normalized[67]["score_rows"]) != 20_903 or len(normalized[75]["score_rows"]) != 8_491:
        raise CorrectionError("Frozen origin score row counts changed")
    e67 = normalized[67]
    e75 = normalized[75]
    e67_last = e67["score_cycles"] >= 76
    e75_all = np.ones(len(e75["score_cycles"]), dtype=bool)
    if not np.array_equal(e67["score_rows"][e67_last], e75["score_rows"][e75_all]) \
            or not np.array_equal(e67["score_record_ids"][e67_last], e75["score_record_ids"][e75_all]):
        raise CorrectionError("E67/E75 cycles 76-79 must map to identical source rows and IDs")
    if set(e67["text_load_group"].tolist()) != set(TEXT_LOAD_GROUPS) \
            or set(e75["text_load_group"].tolist()) != set(TEXT_LOAD_GROUPS):
        raise CorrectionError("Frozen inherited text-load quartiles are incomplete")
    return {origin: normalized[origin] for origin in ORIGINS}


def selection_binding(predictions: Mapping[int, Mapping[str, Any]]) -> dict[str, Any]:
    data = validate_predictions(predictions)
    e67 = data[67]
    e75 = data[75]
    primary_67 = (e67["score_cycles"] >= 68) & (e67["score_cycles"] <= 75)
    primary_75 = (e75["score_cycles"] >= 76) & (e75["score_cycles"] <= 79)
    rows = np.concatenate((e67["score_rows"][primary_67], e75["score_rows"][primary_75]))
    ids = np.concatenate((e67["score_record_ids"][primary_67], e75["score_record_ids"][primary_75]))
    if len(rows) != PRIMARY_ROWS or len(ids) != PRIMARY_ROWS \
            or len(set(rows.tolist())) != PRIMARY_ROWS or len(set(ids.tolist())) != PRIMARY_ROWS:
        raise CorrectionError("Selected primary source rows/IDs must be exactly 20,903 unique records")
    order = np.argsort(rows, kind="stable")
    ordered_rows = rows[order]
    ordered_ids = ids[order]
    return {
        "selected_rows": ordered_rows,
        "selected_ids": ordered_ids,
        "selected_rows_sha256": bytes_sha256(ordered_rows.astype("<i8", copy=False)),
        "selected_ids_sha256": id_sequence_sha256(ordered_ids),
        "selected_rows_count": int(len(ordered_rows)),
        "origins": {"E67_cycles_68_75": int(primary_67.sum()),
                    "E75_cycles_76_79": int(primary_75.sum())},
    }


def labels_to_map(selected_ids: Any, labels: Any) -> dict[str, int]:
    ids = np.asarray(selected_ids).astype(str).reshape(-1)
    y = np.asarray(labels, dtype=np.int64).reshape(-1)
    if len(ids) != len(y) or len(set(ids.tolist())) != len(ids):
        raise CorrectionError("Selected target vector IDs/labels are not one-to-one")
    if len(y) != PRIMARY_ROWS or not np.isin(y, (0, 1)).all():
        raise CorrectionError("Selected target vector must contain exactly 20,903 binary labels")
    return {str(record_id): int(label) for record_id, label in zip(ids, y, strict=True)}


def _view_payload(name: str, origin_source: str, data: Mapping[str, np.ndarray],
                  mask: np.ndarray, labels_by_id: Mapping[str, int]) -> dict[str, Any]:
    ids = data["score_record_ids"][mask].astype(str)
    missing = [record_id for record_id in ids.tolist() if record_id not in labels_by_id]
    if missing:
        raise CorrectionError(f"Selected in-memory labels do not cover view {name}")
    y = np.asarray([labels_by_id[record_id] for record_id in ids], dtype=np.int64)
    thresholds = data["decision_thresholds"][mask]
    predictions = {
        "p_champion": data["p_champion"][mask],
        "p_candidate_real": data["p_candidate_real"][mask],
        "p_control_shuffled": data["p_control_shuffled"][mask],
    }
    cycles = data["score_cycles"][mask]
    rows = data["score_rows"][mask]
    present = data["present_mask"][mask]
    return {
        "name": name, "prediction_source": origin_source,
        "rows": int(len(rows)), "cycle_min": int(cycles.min()), "cycle_max": int(cycles.max()),
        "source_rows_sha256": bytes_sha256(rows.astype("<i8", copy=False)),
        "record_ids_sha256": id_sequence_sha256(ids),
        "present_rows": int(present.sum()), "missing_rows": int((~present).sum()),
        "label_counts": {"positive": int(y.sum()), "negative": int(len(y) - y.sum())},
        "metrics": _metric_bundle(y, predictions, thresholds),
        "_rows": rows, "_ids": ids, "_cycles": cycles, "_labels": y,
        "_present": present, "_entry_friction": data["entry_friction"][mask],
        "_text_load_group": data["text_load_group"][mask],
        "_thresholds": thresholds, "_predictions": predictions,
    }


def _subgroups_for_view(view: Mapping[str, Any]) -> dict[str, Any]:
    groups = {
        "availability": np.where(view["_present"], "present", "missing"),
        "entry_friction": view["_entry_friction"].astype(str),
        "inherited_text_load_quartile": view["_text_load_group"].astype(str),
    }
    result: dict[str, Any] = {}
    for dimension, values in groups.items():
        for category in sorted(set(values.tolist())):
            mask = values == category
            y = view["_labels"][mask]
            support = {"n": int(len(y)), "positive": int(y.sum()),
                       "negative": int(len(y) - y.sum())}
            supported = support["n"] >= 200 and support["positive"] >= 30 and support["negative"] >= 30
            subset_predictions = {key: value[mask] for key, value in view["_predictions"].items()}
            metrics = _metric_bundle(y, subset_predictions, view["_thresholds"][mask]) if len(y) else None
            f1_harm = recall_harm = None
            if metrics is not None:
                f1_harm = max(0.0, metrics["champion"]["macro_f1"] - metrics["candidate_real"]["macro_f1"])
                recall_harm = max(0.0, metrics["champion"]["positive_recall"]
                                  - metrics["candidate_real"]["positive_recall"])
            result[f"{dimension}={category}"] = {
                "dimension": dimension, "category": category, "support": support,
                "supported_for_gate": bool(supported), "metrics": metrics,
                "candidate_macro_f1_harm": f1_harm,
                "candidate_positive_recall_harm": recall_harm,
            }
    return result


def _public_view(view: Mapping[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in view.items() if not key.startswith("_")}


def score_correction(predictions: Mapping[int, Mapping[str, Any]],
                     labels_by_id: Mapping[str, int], *,
                     prelabel_verification_sha256: str) -> dict[str, Any]:
    data = validate_predictions(predictions)
    selection = selection_binding(predictions)
    if set(labels_by_id) != set(selection["selected_ids"].tolist()):
        raise CorrectionError("In-memory label ID set differs from frozen selected primary IDs")
    # Ensure no extra vector entry can be silently ignored.
    if len(labels_by_id) != PRIMARY_ROWS:
        raise CorrectionError("In-memory label map contains rows outside the 20,903-row authorization")
    views: dict[str, dict[str, Any]] = {}
    for name, origin, low, high, expected_rows in WINDOW_SPECS:
        origin_data = data[origin]
        mask = (origin_data["score_cycles"] >= low) & (origin_data["score_cycles"] <= high)
        if int(mask.sum()) != expected_rows:
            raise CorrectionError(f"{name} row count changed: {int(mask.sum())} != {expected_rows}")
        views[name] = _view_payload(name, f"E{origin}", origin_data, mask, labels_by_id)
    e67, e75 = data[67], data[75]
    m67 = (e67["score_cycles"] >= 68) & (e67["score_cycles"] <= 75)
    m75 = (e75["score_cycles"] >= 76) & (e75["score_cycles"] <= 79)
    primary_parts = [
        _view_payload("PRIMARY_E67_68_75", "E67", e67, m67, labels_by_id),
        _view_payload("PRIMARY_E75_76_79", "E75", e75, m75, labels_by_id),
    ]
    primary_rows = np.concatenate([part["_rows"] for part in primary_parts])
    primary_ids = np.concatenate([part["_ids"] for part in primary_parts])
    if len(primary_rows) != PRIMARY_ROWS or len(set(primary_ids.tolist())) != PRIMARY_ROWS:
        raise CorrectionError("PRIMARY_UNIQUE must contain exactly 20,903 unique records")
    order = np.argsort(primary_rows, kind="stable")
    primary_parts[0]["_rows"] = primary_rows[order]
    primary_parts[0]["_ids"] = primary_ids[order]
    combined = {
        key: np.concatenate([part[key] for part in primary_parts])[order]
        for key in ("_cycles", "_labels", "_present", "_entry_friction", "_text_load_group", "_thresholds")
    }
    combined_predictions = {
        key: np.concatenate([part["_predictions"][key] for part in primary_parts])[order]
        for key in ("p_champion", "p_candidate_real", "p_control_shuffled")
    }
    p_mask = np.ones(PRIMARY_ROWS, dtype=bool)
    views["PRIMARY_UNIQUE"] = {
        "name": "PRIMARY_UNIQUE", "prediction_source": "E67_cycles_68_75 + E75_cycles_76_79",
        "rows": PRIMARY_ROWS, "cycle_min": int(combined["_cycles"].min()),
        "cycle_max": int(combined["_cycles"].max()),
        "source_rows_sha256": bytes_sha256(primary_rows[order].astype("<i8", copy=False)),
        "record_ids_sha256": id_sequence_sha256(primary_ids[order]),
        "present_rows": int(combined["_present"].sum()),
        "missing_rows": int((~combined["_present"]).sum()),
        "label_counts": {"positive": int(combined["_labels"].sum()),
                         "negative": int(PRIMARY_ROWS - combined["_labels"].sum())},
        "metrics": _metric_bundle(combined["_labels"], combined_predictions, combined["_thresholds"]),
        "_rows": primary_rows[order], "_ids": primary_ids[order],
        **combined, "_predictions": combined_predictions,
    }
    if views["E67_W4"]["_ids"].tolist() and views["E67_W5"]["_ids"].tolist():
        ids_w4, ids_w5, ids_w6 = (set(views[name]["_ids"].tolist())
                                  for name in ("E67_W4", "E67_W5", "E67_W6"))
        if ids_w4 & ids_w5 or ids_w4 & ids_w6 or ids_w5 & ids_w6:
            raise CorrectionError("E67 W4/W5/W6 must be disjoint cycle windows")
        if ids_w4 | ids_w5 | ids_w6 != set(views["E67_H12"]["_ids"].tolist()):
            raise CorrectionError("E67 W4/W5/W6 do not exactly cover E67 H12")
        if ids_w6 != set(views["E75_W6"]["_ids"].tolist()):
            raise CorrectionError("E67 W6 and E75 W6 do not share the same source rows")
    else:
        raise CorrectionError("Corrected E67 windows are empty")
    primary_expected_ids = ids_w4 | ids_w5 | set(views["E75_W6"]["_ids"].tolist())
    if primary_expected_ids != set(views["PRIMARY_UNIQUE"]["_ids"].tolist()):
        raise CorrectionError("PRIMARY_UNIQUE source IDs differ from E67 W4+W5 and E75 W6")
    for name, _, _, _, expected_rows in WINDOW_SPECS:
        if views[name]["rows"] != expected_rows:
            raise CorrectionError(f"Output row count mismatch for {name}")
    subgroups = {name: _subgroups_for_view(view) for name, view in views.items()}
    supported = [group for by_view in subgroups.values() for group in by_view.values()
                 if group["supported_for_gate"]]
    worst_f1 = max((float(group["candidate_macro_f1_harm"]) for group in supported), default=None)
    worst_recall = max((float(group["candidate_positive_recall_harm"])
                        for group in supported), default=None)
    group_views = {name: sum(group["supported_for_gate"] for group in items.values())
                   for name, items in subgroups.items()}
    all_missing_exact = all(
        np.array_equal(data[origin]["p_candidate_real"][~data[origin]["present_mask"]],
                       data[origin]["p_champion"][~data[origin]["present_mask"]])
        and np.array_equal(data[origin]["p_control_shuffled"][~data[origin]["present_mask"]],
                           data[origin]["p_champion"][~data[origin]["present_mask"]])
        for origin in ORIGINS)
    primary = views["PRIMARY_UNIQUE"]["metrics"]
    e67_h12 = views["E67_H12"]["metrics"]
    e75_w6 = views["E75_W6"]["metrics"]
    p_gain = primary["gains"]["candidate_real_minus_champion_macro_f1"]
    e67_gain = e67_h12["gains"]["candidate_real_minus_champion_macro_f1"]
    e75_gain = e75_w6["gains"]["candidate_real_minus_champion_macro_f1"]
    primary_control_gain = primary["gains"]["candidate_real_minus_control_shuffled_macro_f1"]
    e75_real_minus_shuffled = e75_w6["gains"]["candidate_real_minus_control_shuffled_macro_f1"]
    scientific_checks = {
        "primary_gain_vs_champion_ge_0_0005": p_gain >= 0.0005,
        "e67_h12_gain_nonnegative": e67_gain >= 0.0,
        "e75_w6_gain_nonnegative": e75_gain >= 0.0,
        "primary_candidate_minus_control_ge_0_0002": primary_control_gain >= 0.0002,
        "e75_real_minus_shuffled_nonnegative": e75_real_minus_shuffled >= 0.0,
        "all_view_supported_subgroup_macro_f1_harm_le_0_01": worst_f1 is not None and worst_f1 <= 0.01,
        "all_view_supported_subgroup_positive_recall_harm_le_0_02": worst_recall is not None and worst_recall <= 0.02,
        "prelabel_qa_and_missing_route_exact": bool(prelabel_verification_sha256) and all_missing_exact,
    }
    exploratory_formula = ((p_gain > 0 and e75_gain >= -0.0005)
                           or (e75_gain >= 0.0005 and p_gain >= -0.0005))
    exploratory_checks = {
        "candidate_gain_pattern": bool(exploratory_formula),
        "all_view_supported_subgroup_macro_f1_harm_le_0_02": worst_f1 is not None and worst_f1 <= 0.02,
        "all_view_supported_subgroup_positive_recall_harm_le_0_03": worst_recall is not None and worst_recall <= 0.03,
        "prelabel_qa_and_missing_route_exact": bool(prelabel_verification_sha256) and all_missing_exact,
    }
    mechanism_checks = {
        "primary_candidate_minus_control_ge_0_0002": primary_control_gain >= 0.0002,
        "e75_real_minus_shuffled_nonnegative": e75_real_minus_shuffled >= 0.0,
    }
    scientific_pass = all(scientific_checks.values())
    exploratory_pass = all(exploratory_checks.values())
    gates = {
        "scientific": {"status": "PASS" if scientific_pass else "FAIL", "checks": scientific_checks,
                       "primary_gain": p_gain, "e67_h12_gain": e67_gain, "e75_w6_gain": e75_gain,
                       "primary_candidate_minus_control": primary_control_gain,
                       "e75_real_minus_shuffled": e75_real_minus_shuffled,
                       "worst_supported_macro_f1_harm_all_views": worst_f1,
                       "worst_supported_positive_recall_harm_all_views": worst_recall,
                       "supported_subgroup_count_across_views": int(len(supported)),
                       "supported_groups_by_view": group_views},
        "exploratory": {"status": "ELIGIBLE" if exploratory_pass else "NOT_ELIGIBLE",
                         "checks": exploratory_checks},
        "mechanism": {"status": "SEPARATED_FROM_SHUFFLED_CONTROL"
                      if all(mechanism_checks.values()) else "NOT_SEPARATED_FROM_SHUFFLED_CONTROL",
                      "checks": mechanism_checks},
    }
    if scientific_pass:
        overall = "SCIENTIFIC_PASS"
    elif exploratory_pass:
        overall = "EXPLORATORY_ONLY_ELIGIBLE"
    else:
        overall = "NO_ADVANCE"
    cycle_signs: dict[str, Any] = {}
    primary_view = views["PRIMARY_UNIQUE"]
    for cycle in sorted(set(primary_view["_cycles"].tolist())):
        mask = primary_view["_cycles"] == cycle
        bundle = _metric_bundle(primary_view["_labels"][mask],
                                {key: value[mask] for key, value in primary_view["_predictions"].items()},
                                primary_view["_thresholds"][mask])
        cycle_signs[str(int(cycle))] = {
            "rows": int(mask.sum()),
            "candidate_minus_champion_macro_f1": bundle["gains"]["candidate_real_minus_champion_macro_f1"],
            "candidate_minus_control_macro_f1": bundle["gains"]["candidate_real_minus_control_shuffled_macro_f1"],
            "control_minus_champion_macro_f1": bundle["gains"]["control_shuffled_minus_champion_macro_f1"],
        }
    overlap = {
        "E67_W4_vs_E67_W5_shared_ids": 0,
        "E67_W4_vs_E67_W6_shared_ids": 0,
        "E67_W5_vs_E67_W6_shared_ids": 0,
        "E67_H12_equals_disjoint_W4_W5_W6": True,
        "E67_W6_equals_E75_W6_ids_and_rows": True,
        "PRIMARY_UNIQUE_uses_E67_W4_W5_and_E75_W6": True,
        "E67_H12_and_PRIMARY_UNIQUE_shared_ids": int(len(set(views["E67_H12"]["_ids"].tolist())
                                                            & set(views["PRIMARY_UNIQUE"]["_ids"].tolist()))),
    }
    public_views = {name: _public_view(view) for name, view in views.items()}
    public_subgroups = {name: groups for name, groups in subgroups.items()}
    return {
        "schema_version": 1, "pilot_id": PILOT_ID, "correction_id": CORRECTION_ID,
        "metric_contract": "MacroF1=mean(F1(label0),F1(label1)); sklearn labels=[0,1], average=macro",
        "overall": overall, "views": public_views, "subgroups_by_view": public_subgroups,
        "gates": gates, "cycle_signs_primary_unique": cycle_signs,
        "window_overlap_checks": overlap,
        "selected_rows": selection["selected_rows_count"],
        "selected_rows_sha256": selection["selected_rows_sha256"],
        "selected_ids_sha256": selection["selected_ids_sha256"],
        "selected_label_vector": {"rows": PRIMARY_ROWS,
                                   "positive": int(sum(labels_by_id.values())),
                                   "negative": int(PRIMARY_ROWS - sum(labels_by_id.values()))},
    }


def training_macro_f1_readback(fit_summary: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Correct historical train-diagnostic F1 from already-saved confusion matrices."""
    entries = fit_summary.get("fit_summaries", [])
    result: list[dict[str, Any]] = []
    for item in entries:
        quality = item.get("fit_quality", {})
        matrix = np.asarray(quality.get("confusion_matrix_labels_0_1"), dtype=np.int64)
        if matrix.shape != (2, 2) or np.any(matrix < 0):
            raise CorrectionError(f"Missing frozen fit confusion matrix for {item.get('fit_id')}")
        tn, fp, fn, tp = (int(matrix[0, 0]), int(matrix[0, 1]),
                          int(matrix[1, 0]), int(matrix[1, 1]))
        f1_0 = 2 * tn / (2 * tn + fp + fn) if (2 * tn + fp + fn) else 0.0
        f1_1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0
        sklearn_value = float(quality["f1"])
        result.append({
            "fit_id": item["fit_id"], "rows": int(quality["n"]),
            "original_positive_class_f1": sklearn_value,
            "corrected_macro_f1_from_saved_confusion": float((f1_0 + f1_1) / 2.0),
            "class_f1": {"0": float(f1_0), "1": float(f1_1)},
            "confusion_matrix_labels_0_1": matrix.tolist(),
        })
    if len(result) != 4:
        raise CorrectionError("Expected four frozen HGB fit diagnostics")
    return result


def synthetic_prediction_fixture() -> tuple[dict[int, dict[str, np.ndarray]], dict[str, int]]:
    """Build exact-shape synthetic prediction arrays without reading official inputs."""
    cycle_counts = {
        68: 577, 69: 1_079, 70: 1_605, 71: 1_689,
        72: 1_811, 73: 1_880, 74: 1_814, 75: 1_957,
        76: 1_822, 77: 2_335, 78: 2_146, 79: 2_188,
    }
    rows: list[int] = []
    cycles: list[int] = []
    cursor = 100
    for cycle, count in cycle_counts.items():
        rows.extend(range(cursor, cursor + count))
        cycles.extend([cycle] * count)
        cursor += count
    row_array = np.asarray(rows, dtype=np.int64)
    cycle_array = np.asarray(cycles, dtype=np.int64)
    ids = np.asarray([f"SYNTHETIC-{row:05d}" for row in row_array], dtype=str)
    present = (row_array % 13) != 0
    labels_all = (row_array % 2).astype(np.int64)
    champion = np.where(labels_all == 1, 0.8, 0.2).astype(np.float64)
    candidate = champion.copy()
    control = champion.copy()
    thresholds = np.where(present, PRESENT_THRESHOLD, MISSING_THRESHOLD).astype(np.float64)
    entry = np.where(row_array % 3 == 0, "1", "0").astype(str)
    text_group = (row_array % 4).astype(str)

    def pack(mask: np.ndarray, origin: int) -> dict[str, np.ndarray]:
        return {
            "origin": np.asarray([origin], dtype=np.int64),
            "score_rows": row_array[mask].copy(),
            "score_record_ids": ids[mask].copy(),
            "score_cycles": cycle_array[mask].copy(),
            "present_mask": present[mask].copy(),
            "p_champion": champion[mask].copy(),
            "p_candidate_real": candidate[mask].copy(),
            "p_control_shuffled": control[mask].copy(),
            "decision_thresholds": thresholds[mask].copy(),
            "entry_friction": entry[mask].copy(),
            "text_load_group": text_group[mask].copy(),
        }

    predictions = {
        67: pack(np.ones(len(row_array), dtype=bool), 67),
        75: pack(cycle_array >= 76, 75),
    }
    selection = selection_binding(predictions)
    selected_set = set(selection["selected_ids"].tolist())
    label_map = {str(record_id): int(label) for record_id, label in zip(ids, labels_all, strict=True)
                 if str(record_id) in selected_set}
    return predictions, label_map


def synthetic_qa() -> dict[str, Any]:
    """Small, label-free contract fixtures for corrected metrics and row routing."""
    cases: dict[str, str] = {}
    cm = np.asarray([[90, 10], [5, 5]], dtype=np.int64)
    f1_zero = 180 / 195
    f1_one = 10 / 25
    expected = (f1_zero + f1_one) / 2
    y = np.asarray([0] * 100 + [1] * 10, dtype=np.int64)
    decision = np.asarray([0] * 90 + [1] * 10 + [0] * 5 + [1] * 5, dtype=np.int64)
    p = np.where(decision == 1, 0.8, 0.2)
    scored = _metric_manual_confusion(y, p, np.full(len(y), 0.5))
    if scored["confusion_matrix_labels_0_1"] != cm.tolist() or not np.isclose(scored["macro_f1"], expected):
        raise CorrectionError("imbalanced MacroF1 confusion fixture failed")
    cases["imbalanced_asymmetric_confusion_macro_f1"] = "PASS"
    if np.isclose(scored["macro_f1"], scored["positive_class_f1"]):
        raise CorrectionError("imbalance fixture did not distinguish MacroF1 from binary positive F1")
    cases["macro_f1_differs_from_old_positive_f1"] = "PASS"
    perfect = _metric_manual_confusion(np.asarray([0, 0, 1, 1]),
                                      np.asarray([0.1, 0.2, 0.8, 0.9]),
                                      np.full(4, 0.5))
    if perfect["macro_f1"] != 1.0:
        raise CorrectionError("perfect-prediction MacroF1 fixture failed")
    cases["perfect_prediction_macro_f1"] = "PASS"
    constant = _metric_manual_confusion(np.asarray([0, 0, 0, 1]),
                                        np.asarray([0.49, 0.49, 0.49, 0.49]),
                                        np.full(4, 0.5))
    if constant["confusion_matrix_labels_0_1"] != [[3, 0], [1, 0]]:
        raise CorrectionError("constant-class prediction fixture failed")
    tied = _metric_manual_confusion(np.asarray([0, 1]), np.asarray([0.5, 0.5]),
                                    np.full(2, 0.5))
    if tied["confusion_matrix_labels_0_1"] != [[0, 1], [0, 1]]:
        raise CorrectionError("threshold tie must classify as positive")
    cases["constant_class_and_threshold_tie"] = "PASS"
    return {"schema_version": 1, "correction_id": CORRECTION_ID,
            "status": "PASS", "cases": cases,
            "case_count": len(cases), "official_source_opened": False,
            "query_targets_read": False, "test_or_sample_opened": False}
