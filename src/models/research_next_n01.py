"""N01: fixed sparse classifier with fit-only class-by-text-load sample weights."""
from __future__ import annotations

import hashlib
import json
from typing import Any

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer

from src.models.classification_models import _TFIDF_PARAMS, _text_values
from src.models.optimization_v3_sparse_classification import PARAMS as BASE_PARAMS
from src.models.regression_research_text import HYBRID_FEATURES, RegressionTextModel, _array_digest
from src.models.round1_models import _fit_fingerprint, _number_array


CARD_ID = "N01"
FEATURE_NAMES = tuple(HYBRID_FEATURES)
PARAMS = dict(BASE_PARAMS)
PILOT_ORIGINS = (55, 67)
QUANTILES = (0.25, 0.50, 0.75)
MINIMUM_CLASS_BIN_N = 100
WEIGHT_CLIP = (0.5, 2.0)
PRESENT_EXPERT_WEIGHT = 0.25
THRESHOLD = 0.50


class WeightSupportError(ValueError):
    """The frozen class-by-volume weighting design lacks required support."""


def class_by_volume_weight_plan(
    frame: pd.DataFrame,
    target: Any,
    *,
    minimum_cell_n: int = MINIMUM_CLASS_BIN_N,
) -> dict[str, Any]:
    """Build candidate and matched class-only weights using only fit rows."""
    if not isinstance(frame, pd.DataFrame) or "text_load" not in frame:
        raise TypeError("N01 weight planning requires fit rows with text_load")
    raw_y = np.asarray(target)
    if raw_y.shape != (len(frame),) or not np.isin(raw_y, (0, 1)).all():
        raise ValueError("N01 needs a row-aligned binary target with both classes")
    y = raw_y.astype(np.int8)
    if set(np.unique(y)) != {0, 1}:
        raise ValueError("N01 needs a row-aligned binary target with both classes")
    if minimum_cell_n < 1:
        raise ValueError("minimum_cell_n must be positive")
    parsed, missing, parse_fail = _number_array(frame["text_load"])
    finite = np.isfinite(parsed)
    finite_values = parsed[finite]
    if not len(finite_values):
        raise WeightSupportError("N01 has no finite fit text_load values")
    edges = np.unique(np.quantile(finite_values, QUANTILES).astype(np.float64))
    bins = np.full(len(frame), -1, dtype=np.int64)
    bins[finite] = np.searchsorted(edges, parsed[finite], side="right").astype(np.int64)
    levels = np.unique(bins)
    counts: dict[str, dict[str, int]] = {}
    cell_weights: dict[tuple[int, int], float] = {}
    n_fit = len(frame)
    n_groups = len(levels)
    for class_value in (0, 1):
        counts[str(class_value)] = {}
        for bin_value in levels:
            cell_n = int(np.sum((y == class_value) & (bins == bin_value)))
            counts[str(class_value)][str(int(bin_value))] = cell_n
            if cell_n < minimum_cell_n:
                raise WeightSupportError(
                    f"N01 class/bin support below {minimum_cell_n}: "
                    f"class={class_value}, bin={int(bin_value)}, n={cell_n}"
                )
            cell_weights[(class_value, int(bin_value))] = float(
                np.sqrt(n_fit / (2.0 * n_groups * cell_n))
            )

    raw_candidate = np.asarray(
        [cell_weights[(int(class_value), int(bin_value))] for class_value, bin_value in zip(y, bins)],
        dtype=np.float64,
    )
    clipped_candidate = np.clip(raw_candidate, *WEIGHT_CLIP)
    candidate = clipped_candidate / float(clipped_candidate.mean())
    class_means = {int(class_value): float(candidate[y == class_value].mean()) for class_value in (0, 1)}
    class_only = np.asarray([class_means[int(class_value)] for class_value in y], dtype=np.float64)
    candidate_mass = {int(class_value): float(candidate[y == class_value].sum()) for class_value in (0, 1)}
    class_only_mass = {int(class_value): float(class_only[y == class_value].sum()) for class_value in (0, 1)}
    if (not np.isfinite(candidate).all() or not np.isfinite(class_only).all()
            or np.any(candidate <= 0) or np.any(class_only <= 0)
            or not np.isclose(candidate.mean(), 1.0, rtol=0, atol=1e-12)
            or not np.isclose(class_only.mean(), 1.0, rtol=0, atol=1e-12)
            or any(not np.isclose(candidate_mass[c], class_only_mass[c], rtol=0, atol=1e-10)
                   for c in (0, 1))):
        raise RuntimeError("N01 weight vectors are invalid or class masses differ")
    return {
        "candidate_weights": candidate,
        "class_only_weights": class_only,
        "bins": bins,
        "diagnostics": {
            "n_fit": int(n_fit),
            "quantiles": list(QUANTILES),
            "edges": edges.tolist(),
            "bin_levels": [int(value) for value in levels],
            "actual_fit_bin_count": int(n_groups),
            "missing_bin_present_in_fit": bool(np.any(bins == -1)),
            "actual_missing_rows": int(missing.sum()),
            "parse_failure_rows": int(parse_fail.sum()),
            "finite_text_load_rows": int(finite.sum()),
            "minimum_cell_n": int(minimum_cell_n),
            "class_bin_counts": counts,
            "candidate_clipped_weight_min": float(clipped_candidate.min()),
            "candidate_clipped_weight_max": float(clipped_candidate.max()),
            "candidate_weight_mean_after_normalization": float(candidate.mean()),
            "class_only_weight_mean": float(class_only.mean()),
            "candidate_class_masses": {str(key): value for key, value in candidate_mass.items()},
            "class_only_class_masses": {str(key): value for key, value in class_only_mass.items()},
            "class_only_weight_by_class": {str(key): value for key, value in class_means.items()},
            "candidate_weight_sha256": _array_digest(candidate),
            "class_only_weight_sha256": _array_digest(class_only),
            "score_rows_used_for_edges_or_weights": False,
        },
    }


class ClassByVolumeSparseClassifier:
    """The original sparse classifier recipe with sample_weight explicitly passed."""

    input_feature_names = FEATURE_NAMES

    def __init__(self, cutoff: int, weight_kind: str):
        if int(cutoff) not in PILOT_ORIGINS:
            raise ValueError(f"N01 cutoff must be one of {PILOT_ORIGINS}")
        if weight_kind not in {"class_by_volume", "class_only"}:
            raise ValueError("weight_kind must be class_by_volume or class_only")
        self.cutoff = int(cutoff)
        self.weight_kind = weight_kind
        self._fitted = False

    def _select(self, frame: pd.DataFrame) -> pd.DataFrame:
        if (not isinstance(frame, pd.DataFrame) or frame.columns.has_duplicates
                or len(frame.columns) != len(FEATURE_NAMES)
                or set(frame.columns) != set(FEATURE_NAMES)):
            raise ValueError("N01 requires the exact structured + signal_stream feature allowlist")
        if {"record_id", "artifact_signal", "threat_rank"} & set(frame.columns):
            raise ValueError("N01 feature allowlist contains an identity or target")
        return frame.loc[:, list(FEATURE_NAMES)].copy()

    def fit(self, frame: pd.DataFrame, target: Any, sample_weight: Any) -> "ClassByVolumeSparseClassifier":
        selected = self._select(frame)
        raw_y = np.asarray(target)
        if raw_y.shape != (len(selected),) or not np.isin(raw_y, (0, 1)).all():
            raise ValueError("N01 fit requires aligned binary labels with both classes")
        y = raw_y.astype(np.int8)
        weights = np.asarray(sample_weight, dtype=np.float64)
        if set(np.unique(y)) != {0, 1}:
            raise ValueError("N01 fit requires aligned binary labels with both classes")
        if (weights.shape != (len(selected),) or not np.isfinite(weights).all()
                or np.any(weights <= 0) or not np.isclose(weights.mean(), 1.0, rtol=0, atol=1e-12)):
            raise ValueError("N01 estimator sample weights must be finite, positive, aligned and mean-one")
        cycles = pd.to_numeric(selected["cycle_code"], errors="raise").to_numpy(dtype=np.int64)
        if (len(selected) == 0 or np.any((cycles < 47) | (cycles > self.cutoff))
                or not selected.index.is_unique or not selected.index.is_monotonic_increasing):
            raise ValueError("N01 fit rows violate the frozen ordered temporal population")

        structured, self._processor, self._scaler, self._encoder, self._numeric_positions, structured_state = (
            RegressionTextModel._structured_fit(selected)
        )
        documents, missing = _text_values(selected["signal_stream"])
        self._vectorizer = TfidfVectorizer(**dict(_TFIDF_PARAMS))
        text = self._vectorizer.fit_transform(documents).tocsr()
        missing_column = sparse.csr_matrix(missing.astype(np.float64).reshape(-1, 1))
        design = sparse.hstack((structured, text, missing_column), format="csr", dtype=np.float64)
        if design.shape[0] != len(y) or not np.isfinite(design.data).all():
            raise RuntimeError("N01 sparse fit design is misaligned or nonfinite")
        self._estimator = LGBMClassifier(**dict(PARAMS))
        # Required change for N01: use the estimator's actual sample_weight fit API.
        self._estimator.fit(design, y, sample_weight=weights)
        ordered_vocab = [(str(token), int(index)) for token, index in
                         sorted(self._vectorizer.vocabulary_.items(), key=lambda item: item[1])]
        self._state = {
            "id": f"N01_SPARSE_CLASS_BY_VOLUME_{self.weight_kind.upper()}",
            "card_id": CARD_ID, "cutoff": self.cutoff, "weight_kind": self.weight_kind,
            "fit_rows": int(len(y)), "fit_cycle_min": int(cycles.min()), "fit_cycle_max": int(cycles.max()),
            "fit_cycle_counts": {str(int(cycle)): int(count) for cycle, count in
                                 zip(*np.unique(cycles, return_counts=True))},
            "input_feature_names": list(FEATURE_NAMES), "base_parameters": dict(PARAMS),
            "tfidf_parameters": {key: (list(value) if isinstance(value, tuple)
                                        else value.__name__ if isinstance(value, type) else value)
                                 for key, value in _TFIDF_PARAMS.items()},
            "fit_data_sha256": _fit_fingerprint(selected, FEATURE_NAMES, y),
            "target_sha256": _array_digest(y), "sample_weight_sha256": _array_digest(weights),
            "sample_weight_used": True, "sample_weight_mean": float(weights.mean()),
            "sample_weight_min": float(weights.min()), "sample_weight_max": float(weights.max()),
            "vocabulary_size": int(len(ordered_vocab)),
            "vocabulary_sha256": hashlib.sha256(json.dumps(ordered_vocab, ensure_ascii=False,
                                                              separators=(",", ":"),
                                                              allow_nan=False).encode("utf-8")).hexdigest(),
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
        digest_payload = json.dumps(self._state, sort_keys=True, separators=(",", ":"), allow_nan=False)
        self._state["state_sha256"] = hashlib.sha256(digest_payload.encode("utf-8")).hexdigest()
        self._fitted = True
        return self

    def predict_probability(self, frame: pd.DataFrame) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("N01 classifier must be fitted before prediction")
        selected = self._select(frame)
        structured, unknown = RegressionTextModel._structured_transform(
            selected, self._processor, self._scaler, self._encoder, self._numeric_positions
        )
        documents, missing = _text_values(selected["signal_stream"])
        text = self._vectorizer.transform(documents).tocsr()
        design = sparse.hstack(
            (structured, text, sparse.csr_matrix(missing.astype(np.float64).reshape(-1, 1))),
            format="csr", dtype=np.float64,
        )
        result = np.asarray(self._estimator.predict_proba(design)[:, 1], dtype=np.float64)
        if (result.shape != (len(selected),) or not np.isfinite(result).all()
                or np.any((result < 0) | (result > 1))):
            raise RuntimeError("N01 returned invalid positive-class probabilities")
        self._diagnostics = {"n_predictions": int(len(result)),
                             "missing_stream_rows": int(missing.sum()),
                             "unknown_structured_categories": unknown}
        return result

    def audit_state(self) -> dict[str, Any]:
        if not self._fitted:
            raise RuntimeError("N01 classifier must be fitted before audit")
        return json.loads(json.dumps(self._state, sort_keys=True, allow_nan=False))


def route_expert_probabilities(
    raw_candidate: Any,
    raw_class_only: Any,
    c2b_probability: Any,
    cb_probability: Any,
    sparse_baseline_raw: Any,
    stream_missing: Any,
) -> dict[str, np.ndarray]:
    """Apply the frozen blend to two new experts and reconstruct sparse25."""
    arrays = [np.asarray(value) for value in
              (raw_candidate, raw_class_only, c2b_probability, cb_probability, sparse_baseline_raw, stream_missing)]
    n = len(arrays[0])
    if any(array.shape != (n,) for array in arrays) or n == 0:
        raise ValueError("N01 routing inputs must be nonempty aligned vectors")
    raw_candidate, raw_class_only, c2b, cb, sparse_raw = [array.astype(np.float64) for array in arrays[:5]]
    missing = arrays[5].astype(bool)
    if any(not np.isfinite(array).all() for array in (raw_candidate, raw_class_only, c2b, cb, sparse_raw)):
        raise ValueError("N01 routing inputs must be finite")
    if any(np.any((array < 0) | (array > 1)) for array in
           (raw_candidate, raw_class_only, c2b, cb, sparse_raw)):
        raise ValueError("N01 classification probabilities must lie in [0,1]")
    current = np.where(missing, cb, 0.75 * c2b + PRESENT_EXPERT_WEIGHT * sparse_raw)
    candidate = np.where(missing, cb, 0.75 * c2b + PRESENT_EXPERT_WEIGHT * raw_candidate)
    matched = np.where(missing, cb, 0.75 * c2b + PRESENT_EXPERT_WEIGHT * raw_class_only)
    return {"current": current, "candidate": candidate, "matched_control": matched,
            "stream_missing": missing}


__all__ = [
    "CARD_ID", "ClassByVolumeSparseClassifier", "FEATURE_NAMES", "MINIMUM_CLASS_BIN_N",
    "PARAMS", "PILOT_ORIGINS", "THRESHOLD", "WeightSupportError",
    "class_by_volume_weight_plan", "route_expert_probabilities",
]
