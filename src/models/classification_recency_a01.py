"""A01 recency weighting for the frozen sparse Classification task model.

This module contains fit-only weighting and estimator primitives. It does not
load official data, choose a split, access held-out labels, or write artifacts.
"""
from __future__ import annotations

import hashlib
import json
from typing import Any

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from lightgbm import LGBMClassifier

from src.models.classification_models import _TFIDF_PARAMS, _text_values
from src.models.optimization_v3_sparse_classification import PARAMS
from src.models.regression_research_text import (
    HYBRID_FEATURES,
    RegressionTextModel,
    _array_digest,
)
from src.models.round1_models import _fit_fingerprint


CARD_ID = "CLASSIFICATION_RECENCY_A01"
ORIGINS = (55, 67)
WEIGHT_FLOOR = 0.25
WEIGHT_HALF_LIFE = 8.0
SHUFFLE_SEED = 42
PRESENT_BASELINE_WEIGHT = 0.75
EXPERT_WEIGHT = 0.25
THRESHOLD = 0.50
FEATURE_NAMES = tuple(HYBRID_FEATURES)


def _int64_sha(values: Any) -> str:
    arr = np.asarray(values, dtype="<i8")
    return hashlib.sha256(arr.tobytes(order="C")).hexdigest()


def _canonical_sha(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False,
                         separators=(",", ":")).encode("utf-8", errors="surrogatepass")
    return hashlib.sha256(payload).hexdigest()


def recency_weight_plan(cycles: Any, target: Any, cutoff: int) -> dict[str, Any]:
    """Return RECENCY and same-class shuffled weights for one temporal origin.

    Inputs must contain fit rows in their official stable row order. The RNG is
    restarted at seed 42 per origin, and class 0 is permuted before class 1.
    Weights are normalized within class so each class retains its original mass.
    """
    if cutoff not in ORIGINS:
        raise ValueError(f"A01 cutoff must be one of {ORIGINS}")
    raw_cycles = np.asarray(cycles)
    raw_target = np.asarray(target)
    if (raw_cycles.ndim != 1 or raw_target.ndim != 1 or len(raw_cycles) == 0
            or len(raw_cycles) != len(raw_target)):
        raise ValueError("A01 cycles and targets must be nonempty aligned vectors")
    try:
        cycle_values = raw_cycles.astype(np.int64)
    except (TypeError, ValueError) as exc:
        raise ValueError("A01 cycle_code must be integer-valued") from exc
    if not np.array_equal(raw_cycles, cycle_values):
        raise ValueError("A01 cycle_code must be integer-valued")
    if np.any((cycle_values < 47) | (cycle_values > cutoff)):
        raise ValueError("A01 fit rows must belong to cycles 47 through the requested cutoff")
    if not np.isin(raw_target, (0, 1)).all() or set(np.unique(raw_target)) != {0, 1}:
        raise ValueError("A01 fit target must contain both binary classes")
    y = raw_target.astype(np.int8)

    raw = np.maximum(WEIGHT_FLOOR, np.power(2.0, -(cutoff - cycle_values) / WEIGHT_HALF_LIFE))
    recency = np.empty(len(y), dtype=np.float64)
    for class_value in (0, 1):
        mask = y == class_value
        recency[mask] = raw[mask] / float(raw[mask].mean())

    rng = np.random.default_rng(SHUFFLE_SEED)
    shuffled = np.empty_like(recency)
    permutation_sources = np.arange(len(y), dtype=np.int64)
    permutation_by_class: dict[str, list[int]] = {}
    for class_value in (0, 1):
        positions = np.flatnonzero(y == class_value)
        local_permutation = rng.permutation(len(positions)).astype(np.int64)
        permutation_sources[positions] = positions[local_permutation]
        shuffled[positions] = recency[positions[local_permutation]]
        permutation_by_class[str(class_value)] = local_permutation.tolist()

    def effective_sample_size(weights: np.ndarray) -> float:
        return float(weights.sum() ** 2 / np.square(weights).sum())

    per_class = {}
    for class_value in (0, 1):
        mask = y == class_value
        n_class = int(mask.sum())
        class_recency = recency[mask]
        class_shuffled = shuffled[mask]
        per_class[str(class_value)] = {
            "n": n_class,
            "recency_mass": float(class_recency.sum()),
            "shuffled_mass": float(class_shuffled.sum()),
            "recency_ess": effective_sample_size(class_recency),
            "shuffled_ess": effective_sample_size(class_shuffled),
            "recency_ess_ratio": effective_sample_size(class_recency) / n_class,
            "shuffled_ess_ratio": effective_sample_size(class_shuffled) / n_class,
        }
    recency_ess = effective_sample_size(recency)
    shuffled_ess = effective_sample_size(shuffled)
    checks = {
        "finite_positive_weights": bool(np.isfinite(recency).all() and np.isfinite(shuffled).all()
                                        and np.all(recency > 0) and np.all(shuffled > 0)),
        "global_means_one": bool(np.isclose(recency.mean(), 1.0, rtol=0, atol=1e-12)
                                  and np.isclose(shuffled.mean(), 1.0, rtol=0, atol=1e-12)),
        "class_masses_preserved": all(np.isclose(item["recency_mass"], item["shuffled_mass"],
                                                  rtol=0, atol=1e-12)
                                       and np.isclose(item["recency_mass"], item["n"],
                                                      rtol=0, atol=1e-12)
                                       for item in per_class.values()),
        "each_class_ess_ratio_at_least_0_50": all(
            item["recency_ess_ratio"] >= 0.50 and item["shuffled_ess_ratio"] >= 0.50
            for item in per_class.values()
        ),
    }
    diagnostics = {
        "card_id": CARD_ID, "cutoff": int(cutoff), "fit_rows": int(len(y)),
        "cycle_min": int(cycle_values.min()), "cycle_max": int(cycle_values.max()),
        "class_counts": {str(c): int(np.sum(y == c)) for c in (0, 1)},
        "weight_formula": "a_i=max(0.25, 2^(-(cutoff-cycle_i)/8)); divide by within-class mean",
        "weight_floor": WEIGHT_FLOOR, "half_life_cycles": WEIGHT_HALF_LIFE,
        "shuffle": "default_rng(42) separately per origin; class 0 then class 1; stable fit row order",
        "permutation_indices_by_class": permutation_by_class,
        "permutation_source_rows_sha256": _int64_sha(permutation_sources),
        "recency_weight_sha256": _array_digest(recency),
        "shuffled_weight_sha256": _array_digest(shuffled),
        "recency_ess": recency_ess, "shuffled_ess": shuffled_ess,
        "recency_ess_ratio": recency_ess / len(y), "shuffled_ess_ratio": shuffled_ess / len(y),
        "by_class": per_class, "preflight_checks": checks,
        "score_rows_used_for_weight_fit": False,
    }
    if not all(checks.values()):
        raise RuntimeError(f"A01 scientific weight preflight failed at cutoff {cutoff}: {checks}")
    return {
        "recency_weights": recency,
        "shuffled_weights": shuffled,
        "permutation_source_rows": permutation_sources,
        "diagnostics": diagnostics,
    }


class RecencySparseClassifier:
    """Exact sparse LightGBM recipe with externally planned sample weights."""

    input_feature_names = FEATURE_NAMES

    def __init__(self, cutoff: int, weight_arm: str):
        if int(cutoff) not in ORIGINS:
            raise ValueError(f"A01 cutoff must be one of {ORIGINS}")
        if weight_arm not in {"RECENCY", "SHUFFLED_WITHIN_CLASS"}:
            raise ValueError("A01 weight_arm must be RECENCY or SHUFFLED_WITHIN_CLASS")
        self.cutoff = int(cutoff)
        self.weight_arm = weight_arm
        self._fitted = False

    def _select(self, frame: pd.DataFrame) -> pd.DataFrame:
        if (not isinstance(frame, pd.DataFrame) or frame.columns.has_duplicates
                or len(frame.columns) != len(self.input_feature_names)
                or set(frame.columns) != set(self.input_feature_names)):
            raise ValueError("A01 requires exactly the structured + signal_stream feature allowlist")
        if {"record_id", "artifact_signal", "threat_rank"} & set(frame.columns):
            raise ValueError("A01 model features cannot contain row identity or target columns")
        return frame.loc[:, list(self.input_feature_names)].copy()

    def fit(self, frame: pd.DataFrame, target: Any, sample_weight: Any) -> "RecencySparseClassifier":
        selected = self._select(frame)
        raw_y = np.asarray(target)
        weights = np.asarray(sample_weight, dtype=np.float64)
        if (raw_y.ndim != 1 or len(raw_y) != len(selected) or not np.isin(raw_y, (0, 1)).all()
                or set(np.unique(raw_y)) != {0, 1}):
            raise ValueError("A01 fit requires aligned labels containing both classes")
        y = raw_y.astype(np.int8)
        if (weights.shape != (len(y),) or not np.isfinite(weights).all() or np.any(weights <= 0)
                or not np.isclose(weights.mean(), 1.0, rtol=0, atol=1e-12)):
            raise ValueError("A01 sample weights must be positive, finite, aligned, and globally mean-one")
        cycles = pd.to_numeric(selected["cycle_code"], errors="raise").to_numpy(dtype=np.int64)
        if (not len(selected) or np.any((cycles < 47) | (cycles > self.cutoff))
                or not selected.index.is_unique or not selected.index.is_monotonic_increasing):
            raise ValueError("A01 fit rows must preserve ordered 47-through-cutoff temporal history")
        class_mass = {str(c): float(weights[y == c].sum()) for c in (0, 1)}
        class_counts = {str(c): int(np.sum(y == c)) for c in (0, 1)}
        if any(not np.isclose(class_mass[str(c)], class_counts[str(c)], rtol=0, atol=1e-10)
               for c in (0, 1)):
            raise ValueError("A01 estimator weights must preserve each class's original mass")

        structured, self._processor, self._scaler, self._encoder, self._numeric_positions, structured_state = (
            RegressionTextModel._structured_fit(selected)
        )
        documents, missing = _text_values(selected["signal_stream"])
        self._vectorizer = TfidfVectorizer(**dict(_TFIDF_PARAMS))
        text = self._vectorizer.fit_transform(documents).tocsr()
        design = sparse.hstack((structured, text,
                                sparse.csr_matrix(missing.astype(np.float64).reshape(-1, 1))),
                               format="csr", dtype=np.float64)
        if design.shape[0] != len(y) or not np.isfinite(design.data).all():
            raise RuntimeError("A01 sparse fit design is misaligned or nonfinite")
        self._estimator = LGBMClassifier(**dict(PARAMS))
        self._estimator.fit(design, y, sample_weight=weights)
        ordered_vocab = [(str(token), int(index)) for token, index in
                         sorted(self._vectorizer.vocabulary_.items(), key=lambda item: item[1])]
        state = {
            "id": f"A01_SPARSE_RECENCY_{self.weight_arm}", "card_id": CARD_ID,
            "cutoff": self.cutoff, "weight_arm": self.weight_arm,
            "fit_rows": int(len(y)), "fit_cycle_min": int(cycles.min()),
            "fit_cycle_max": int(cycles.max()),
            "fit_cycle_counts": {str(int(cycle)): int(count) for cycle, count in
                                 zip(*np.unique(cycles, return_counts=True))},
            "input_feature_names": list(self.input_feature_names),
            "lightgbm_parameters": dict(PARAMS),
            "tfidf_parameters": {key: (list(value) if isinstance(value, tuple)
                                        else value.__name__ if isinstance(value, type) else value)
                                 for key, value in _TFIDF_PARAMS.items()},
            "fit_data_sha256": _fit_fingerprint(selected, self.input_feature_names, y),
            "target_sha256": _array_digest(y), "sample_weight_sha256": _array_digest(weights),
            "sample_weight_used": True, "sample_weight_mean": float(weights.mean()),
            "sample_weight_min": float(weights.min()), "sample_weight_max": float(weights.max()),
            "sample_weight_class_mass": class_mass, "fit_class_counts": class_counts,
            "vocabulary_size": int(len(ordered_vocab)),
            "vocabulary_sha256": _canonical_sha(ordered_vocab),
            "idf_sha256": _array_digest(self._vectorizer.idf_),
            "structured_preprocessing": structured_state,
            "design_shape": [int(value) for value in design.shape],
            "design_nnz": int(design.nnz), "design_data_sha256": _array_digest(design.data),
            "model_sha256": hashlib.sha256(
                self._estimator.booster_.model_to_string().encode("utf-8")
            ).hexdigest(),
            "validation_data_passed_to_fit": False,
            "validation_labels_passed_to_fit": False,
            "early_stopping_used": False,
            "test_data_used": False,
        }
        state["state_sha256"] = _canonical_sha(state)
        self._state = state
        self._fitted = True
        return self

    def predict_probability(self, frame: pd.DataFrame) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("A01 classifier must be fitted before prediction")
        selected = self._select(frame)
        structured, unknown = RegressionTextModel._structured_transform(
            selected, self._processor, self._scaler, self._encoder, self._numeric_positions
        )
        documents, missing = _text_values(selected["signal_stream"])
        text = self._vectorizer.transform(documents).tocsr()
        design = sparse.hstack((structured, text,
                                sparse.csr_matrix(missing.astype(np.float64).reshape(-1, 1))),
                               format="csr", dtype=np.float64)
        probability = np.asarray(self._estimator.predict_proba(design)[:, 1], dtype=np.float64)
        if (probability.shape != (len(selected),) or not np.isfinite(probability).all()
                or np.any((probability < 0) | (probability > 1))):
            raise RuntimeError("A01 returned invalid positive-class probabilities")
        self._diagnostics = {"n_predictions": int(len(probability)),
                             "stream_missing_rows": int(missing.sum()),
                             "unknown_structured_categories": unknown}
        return probability

    def audit_state(self) -> dict[str, Any]:
        if not self._fitted:
            raise RuntimeError("A01 classifier must be fitted before audit")
        return json.loads(json.dumps(self._state, sort_keys=True, allow_nan=False))


def route_probabilities(raw_recency: Any, raw_shuffled: Any, raw_sparse25: Any,
                        c2b_probability: Any, cb_probability: Any,
                        stream_missing: Any) -> dict[str, np.ndarray]:
    """Rebuild sparse25 and both A01 routes with the frozen availability policy."""
    recency, shuffled, sparse25_raw, c2b, cb = (
        np.asarray(value, dtype=np.float64)
        for value in (raw_recency, raw_shuffled, raw_sparse25, c2b_probability, cb_probability)
    )
    missing = np.asarray(stream_missing, dtype=bool)
    if (recency.ndim != 1 or len(recency) == 0 or any(value.shape != recency.shape
            for value in (shuffled, sparse25_raw, c2b, cb, missing))):
        raise ValueError("A01 route requires nonempty, aligned one-dimensional vectors")
    if any(not np.isfinite(values).all() for values in (recency, shuffled, sparse25_raw, c2b, cb)):
        raise ValueError("A01 route probabilities must be finite")
    if any(np.any((values < 0) | (values > 1))
           for values in (recency, shuffled, sparse25_raw, c2b, cb)):
        raise ValueError("A01 route probabilities must lie in [0,1]")
    baseline = np.where(missing, cb, PRESENT_BASELINE_WEIGHT * c2b + EXPERT_WEIGHT * sparse25_raw)
    candidate = np.where(missing, cb, PRESENT_BASELINE_WEIGHT * c2b + EXPERT_WEIGHT * recency)
    matched = np.where(missing, cb, PRESENT_BASELINE_WEIGHT * c2b + EXPERT_WEIGHT * shuffled)
    return {
        "sparse25_baseline": baseline,
        "recency_candidate": candidate,
        "shuffled_weight_control": matched,
        "stream_missing": missing,
        "baseline_label": (baseline >= THRESHOLD).astype(np.int8),
        "recency_label": (candidate >= THRESHOLD).astype(np.int8),
        "shuffled_label": (matched >= THRESHOLD).astype(np.int8),
    }


__all__ = [
    "CARD_ID", "EXPERT_WEIGHT", "FEATURE_NAMES", "ORIGINS", "PARAMS",
    "RecencySparseClassifier", "SHUFFLE_SEED", "THRESHOLD", "WEIGHT_FLOOR",
    "WEIGHT_HALF_LIFE", "recency_weight_plan", "route_probabilities",
]
