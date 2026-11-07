"""Independent, pure-array metric verification for CLS_DAN_V1.

No data is opened here. The evaluator passes its single in-memory query-label
vector and predictions to this module after the frozen prelabel commit.
"""
from __future__ import annotations

from typing import Any, Mapping

import numpy as np
from scipy.stats import rankdata
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)


MISSING_THRESHOLD = 0.53230976129477
PRESENT_THRESHOLD = 0.50
METRIC_KEYS = ("macro_f1", "precision", "positive_recall", "roc_auc",
               "average_precision", "log_loss", "brier_score")


def _validate(y: Any, probability: Any, present: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    target = np.asarray(y, dtype=np.int8).reshape(-1)
    p = np.asarray(probability, dtype=np.float64).reshape(-1)
    is_present = np.asarray(present, dtype=bool).reshape(-1)
    if (len(target) == 0 or len(target) != len(p) or len(target) != len(is_present)
            or not np.isin(target, (0, 1)).all() or not np.isfinite(p).all()
            or np.any((p < 0.0) | (p > 1.0))):
        raise ValueError("Metric arrays must be nonempty, aligned, binary and finite")
    return target, p, is_present


def manual_metrics(y: Any, probability: Any, present: Any) -> dict[str, Any]:
    """Recompute binary metrics without sklearn's classification metric calls."""
    target, p, is_present = _validate(y, probability, present)
    thresholds = np.where(is_present, PRESENT_THRESHOLD, MISSING_THRESHOLD)
    predicted = (p >= thresholds).astype(np.int8)
    tn = int(np.count_nonzero((target == 0) & (predicted == 0)))
    fp = int(np.count_nonzero((target == 0) & (predicted == 1)))
    fn = int(np.count_nonzero((target == 1) & (predicted == 0)))
    tp = int(np.count_nonzero((target == 1) & (predicted == 1)))
    precision = tp / (tp + fp) if tp + fp else 0.0
    positive_recall = tp / (tp + fn) if tp + fn else 0.0
    negative_recall = tn / (tn + fp) if tn + fp else 0.0
    positive_f1 = 2.0 * precision * positive_recall / (precision + positive_recall) if precision + positive_recall else 0.0
    negative_precision = tn / (tn + fn) if tn + fn else 0.0
    negative_f1 = 2.0 * negative_precision * negative_recall / (negative_precision + negative_recall) if negative_precision + negative_recall else 0.0
    positive_count = int(target.sum())
    negative_count = int(len(target) - positive_count)

    if positive_count and negative_count:
        ranks = rankdata(p, method="average")
        roc_auc = float((float(np.sum(ranks[target == 1]))
                         - positive_count * (positive_count + 1) / 2.0)
                        / (positive_count * negative_count))
        order = np.argsort(-p, kind="mergesort")
        sorted_p = p[order]
        sorted_y = target[order].astype(np.int64)
        end_positions = np.r_[np.flatnonzero(sorted_p[:-1] != sorted_p[1:]), len(sorted_p) - 1]
        cumulative_tp = np.cumsum(sorted_y)[end_positions]
        increments = np.diff(np.r_[0, cumulative_tp])
        average_precision = float(np.sum((cumulative_tp / (end_positions + 1)) * increments)
                                  / positive_count)
    elif positive_count:
        roc_auc = None
        average_precision = 1.0
    else:
        roc_auc = None
        average_precision = 0.0

    eps = np.finfo(np.float64).eps
    clipped = np.clip(p, eps, 1.0 - eps)
    logloss = float(np.mean(-(target * np.log(clipped) + (1 - target) * np.log(1.0 - clipped))))
    return {
        "n": int(len(target)), "positive": positive_count, "negative": negative_count,
        "predicted_positive": int(predicted.sum()),
        "confusion_matrix": [[tn, fp], [fn, tp]],
        "macro_f1": 0.5 * (positive_f1 + negative_f1),
        "precision": precision,
        "positive_recall": positive_recall,
        "roc_auc": roc_auc,
        "average_precision": average_precision,
        "log_loss": logloss,
        "brier_score": float(np.mean((p - target) ** 2)),
    }


def verify_against_sklearn(y: Any, probability: Any, present: Any,
                           atol: float = 1e-12) -> dict[str, Any]:
    """Compare independent formulas against sklearn's reference metrics."""
    target, p, is_present = _validate(y, probability, present)
    manual = manual_metrics(target, p, is_present)
    predicted = (p >= np.where(is_present, PRESENT_THRESHOLD, MISSING_THRESHOLD)).astype(np.int8)
    sklearn_metrics: dict[str, Any] = {
        "n": int(len(target)),
        "positive": int(target.sum()),
        "negative": int(len(target) - target.sum()),
        "predicted_positive": int(predicted.sum()),
        "confusion_matrix": confusion_matrix(target, predicted, labels=[0, 1]).tolist(),
        "macro_f1": float(f1_score(target, predicted, labels=[0, 1], average="macro", zero_division=0)),
        "precision": float(precision_score(target, predicted, pos_label=1, zero_division=0)),
        "positive_recall": float(recall_score(target, predicted, pos_label=1, zero_division=0)),
        "roc_auc": float(roc_auc_score(target, p)) if np.unique(target).size == 2 else None,
        "average_precision": float(average_precision_score(target, p)),
        "log_loss": float(log_loss(target, p, labels=[0, 1])),
        "brier_score": float(brier_score_loss(target, p)),
    }
    deltas: dict[str, float] = {}
    exact_counts = all(manual[key] == sklearn_metrics[key]
                       for key in ("n", "positive", "negative", "predicted_positive", "confusion_matrix"))
    for key in METRIC_KEYS:
        if manual[key] is None or sklearn_metrics[key] is None:
            if manual[key] is not sklearn_metrics[key]:
                deltas[key] = float("inf")
            continue
        deltas[key] = abs(float(manual[key]) - float(sklearn_metrics[key]))
    status = exact_counts and all(value <= atol for value in deltas.values())
    return {"status": "PASS" if status else "FAIL", "atol": atol,
            "manual": manual, "sklearn": sklearn_metrics, "absolute_deltas": deltas}


def verify_metric_map(y: Any, views: Mapping[str, Mapping[str, Any]],
                      atol: float = 1e-12) -> dict[str, Any]:
    """Verify every named model/view using one already-loaded label vector."""
    output = {}
    for name, arrays in views.items():
        output[name] = verify_against_sklearn(y, arrays["probability"], arrays["present"], atol)
    return {
        "status": "PASS" if all(value["status"] == "PASS" for value in output.values()) else "FAIL",
        "query_label_vector_reused_in_memory": True,
        "views": output,
    }
