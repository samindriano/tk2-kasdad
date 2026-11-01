"""R11 cumulative ordinal median regression candidate.

Nine binary heads estimate P(threat_rank > threshold) for thresholds 10..90.
The heads share fit-only structured, TF-IDF, and SVD32 preprocessing. Survival
probabilities are projected to a nonincreasing sequence before a linear
0.5-crossing estimate is returned. No semantic interpretation is assigned to
the SVD components.
"""
from __future__ import annotations

import hashlib
import json
from typing import Any, Mapping

import lightgbm
import numpy as np
import pandas as pd
import sklearn
from scipy.stats import rankdata
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from lightgbm import LGBMClassifier

from src.models.regression_research_lgb_text import _select_exact
from src.models.regression_research_tabular import _tree_summary
from src.models.regression_research_text import (
    HYBRID_FEATURES,
    STRUCTURED_FEATURES,
    RegressionTextModel,
    _SVD_PARAMS,
    _TFIDF_PARAMS,
    _array_digest,
    _json_digest,
    _safe_builtin,
    _target,
    _text_documents,
)
from src.models.regression_round2_models import _SelectedPreprocessor
from src.models.round1_models import _fit_fingerprint


CARD_ID = "R11"
FEATURE_NAMES = tuple(HYBRID_FEATURES)
THRESHOLDS = tuple(range(10, 100, 10))
MIN_CLASS_SUPPORT = 100
TFIDF_PARAMETERS: dict[str, Any] = dict(_TFIDF_PARAMS)
SVD_PARAMETERS: dict[str, Any] = dict(_SVD_PARAMS)
HEAD_PARAMETERS: dict[str, Any] = {
    "objective": "binary",
    "metric": "binary_logloss",
    "learning_rate": 0.05,
    "n_estimators": 250,
    "num_leaves": 15,
    "min_child_samples": 30,
    "reg_lambda": 3.0,
    "n_jobs": 2,
    "random_state": 42,
    "verbosity": -1,
    "deterministic": True,
    "force_col_wise": True,
}
BASELINE_WEIGHT = 0.75
CANDIDATE_WEIGHT = 0.25
FINAL_RANGE = (0.0, 100.0)


def threshold_targets(target: Any) -> np.ndarray:
    """Return columns for the fixed binary targets y>10,...,y>90."""
    y = np.asarray(target, dtype=np.float64)
    if y.ndim != 1 or not len(y) or not np.isfinite(y).all() or np.any((y < 0) | (y > 100)):
        raise ValueError("R11 target must be a nonempty finite vector in [0,100]")
    return np.column_stack([y > threshold for threshold in THRESHOLDS]).astype(np.uint8)


def threshold_support(target: Any, minimum: int = MIN_CLASS_SUPPORT) -> dict[str, Any]:
    """Report both class denominators at each ordinal threshold."""
    labels = threshold_targets(target)
    result: dict[str, Any] = {}
    for index, threshold in enumerate(THRESHOLDS):
        positive = int(labels[:, index].sum())
        negative = int(len(labels) - positive)
        result[str(threshold)] = {
            "positive": positive,
            "negative": negative,
            "minimum_per_class": int(minimum),
            "supported": positive >= minimum and negative >= minimum,
        }
    return result


def validated_cycle_codes(values: Any, *, name: str = "R11 cycle_code") -> np.ndarray:
    """Parse cycles and reject missing, nonfinite, fractional, or out-of-range values."""
    try:
        numeric = pd.to_numeric(pd.Series(values), errors="raise").to_numpy(dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must contain numeric values") from exc
    int64_limit = float(2**63)
    if (numeric.ndim != 1 or not len(numeric) or not np.isfinite(numeric).all()
            or not np.equal(numeric, np.floor(numeric)).all()
            or np.any(numeric < -int64_limit) or np.any(numeric >= int64_limit)):
        raise ValueError(f"{name} must be nonempty, finite, integer-valued, and fit int64")
    return numeric.astype(np.int64)


def pava_nonincreasing(survival: Any) -> np.ndarray:
    """Project each row to nonincreasing probabilities using unweighted PAVA."""
    values = np.asarray(survival, dtype=np.float64)
    one_row = values.ndim == 1
    if one_row:
        values = values.reshape(1, -1)
    if (values.ndim != 2 or values.shape[1] != len(THRESHOLDS) or not len(values)
            or not np.isfinite(values).all() or np.any((values < 0.0) | (values > 1.0))):
        raise ValueError("R11 survival probabilities must be a finite n-by-9 matrix in [0,1]")
    projected = np.empty_like(values)
    for row_index, row in enumerate(values):
        means: list[float] = []
        weights: list[int] = []
        for value in row:
            means.append(float(value))
            weights.append(1)
            while len(means) >= 2 and means[-2] < means[-1]:
                total = weights[-2] + weights[-1]
                pooled = (means[-2] * weights[-2] + means[-1] * weights[-1]) / total
                means[-2:] = [pooled]
                weights[-2:] = [total]
        projected[row_index] = np.repeat(np.asarray(means), np.asarray(weights))
    return projected[0] if one_row else projected


def median_from_survival(survival: Any) -> np.ndarray:
    """Interpolate the 0.5 survival crossing on the 0..100 decile grid."""
    values = np.asarray(survival, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != len(THRESHOLDS) or not len(values):
        raise ValueError("R11 median conversion requires a nonempty n-by-9 matrix")
    if not np.isfinite(values).all() or np.any((values < 0.0) | (values > 1.0)):
        raise ValueError("R11 median conversion requires finite probabilities in [0,1]")
    if np.any(np.diff(values, axis=1) > 1e-10):
        raise ValueError("R11 survival probabilities must be nonincreasing")
    grid = np.asarray([0.0, *map(float, THRESHOLDS), 100.0])
    augmented = np.column_stack((np.ones(len(values)), values, np.zeros(len(values))))
    output = np.empty(len(values), dtype=np.float64)
    for i, row in enumerate(augmented):
        crossing = np.flatnonzero(row <= 0.5)
        if not len(crossing):
            output[i] = 100.0
            continue
        right = int(crossing[0])
        if right == 0 or row[right] == 0.5:
            output[i] = grid[right]
            continue
        left = right - 1
        denominator = row[left] - row[right]
        if denominator <= 0.0:
            raise ValueError("R11 survival crossing is not strictly interpolable")
        fraction = (row[left] - 0.5) / denominator
        output[i] = grid[left] + fraction * (grid[right] - grid[left])
    return np.clip(output, *FINAL_RANGE)


def rank50_prediction(raw_prediction: Any, cycle_codes: Any) -> np.ndarray:
    """Map target-scale predictions to the fixed within-cycle raw/rank50 blend."""
    raw = np.asarray(raw_prediction, dtype=np.float64)
    cycles = pd.to_numeric(pd.Series(cycle_codes), errors="raise").to_numpy(dtype=np.float64)
    if raw.ndim != 1 or cycles.shape != raw.shape or not len(raw):
        raise ValueError("R11 rank50 inputs must be nonempty aligned vectors")
    if not np.isfinite(raw).all():
        raise ValueError("R11 rank50 inputs must be finite and cycle codes integer-valued")
    cycle_values = validated_cycle_codes(cycles)
    mapped = np.empty(len(raw), dtype=np.float64)
    for cycle in np.unique(cycle_values):
        mask = cycle_values == cycle
        percentile = 100.0 * rankdata(raw[mask], method="average") / int(mask.sum())
        mapped[mask] = 0.5 * raw[mask] + 0.5 * percentile
    return np.clip(mapped, *FINAL_RANGE)


def compose_r11_prediction(baseline_prediction: Any, raw_candidate: Any,
                           cycle_codes: Any) -> np.ndarray:
    baseline = np.asarray(baseline_prediction, dtype=np.float64)
    raw = np.asarray(raw_candidate, dtype=np.float64)
    cycles = np.asarray(cycle_codes)
    if (baseline.ndim != 1 or baseline.shape != raw.shape or baseline.shape != cycles.shape
            or not len(baseline) or not np.isfinite(baseline).all()
            or np.any((baseline < 0.0) | (baseline > 100.0)) or not np.isfinite(raw).all()):
        raise ValueError("R11 composition inputs must be aligned and finite; baseline in [0,100]")
    return np.clip(BASELINE_WEIGHT * baseline + CANDIDATE_WEIGHT * rank50_prediction(raw, cycles), *FINAL_RANGE)


def _vectorizer_state(vectorizer: TfidfVectorizer | None, fallback: str | None) -> dict[str, Any]:
    parameters = {
        key: list(value) if isinstance(value, tuple) else value
        for key, value in (vectorizer or TfidfVectorizer(**TFIDF_PARAMETERS)).get_params(deep=False).items()
    }
    if vectorizer is None:
        return {"parameters": parameters, "vocabulary_size": 0, "vocabulary_sha256": None,
                "idf_sha256": None, "fallback": fallback}
    ordered = [(str(token), int(index)) for token, index in
               sorted(vectorizer.vocabulary_.items(), key=lambda item: item[1])]
    return {"parameters": parameters, "vocabulary_size": len(ordered),
            "vocabulary_sha256": _json_digest(ordered), "idf_sha256": _array_digest(vectorizer.idf_),
            "fallback": None}


class CumulativeOrdinalSVD32:
    """Fit shared feature transforms and nine cumulative binary LightGBM heads."""

    input_feature_names = FEATURE_NAMES

    def __init__(self) -> None:
        self._fitted = False
        self._diagnostics: dict[str, Any] = {"n_predictions": 0}

    @staticmethod
    def _select(X: pd.DataFrame) -> pd.DataFrame:
        selected = _select_exact(X)
        if not len(selected):
            raise ValueError("R11 feature frames must be nonempty")
        return selected

    def fit(self, X: pd.DataFrame, y: Any) -> "CumulativeOrdinalSVD32":
        self._fitted = False
        selected = self._select(X)
        target = _target(y, len(selected))
        if np.any(target == 0.0):
            raise ValueError("R11 boundary assumption requires every fit target to be strictly above 0")
        labels = threshold_targets(target)
        support = threshold_support(target)
        unsupported = {key: item for key, item in support.items() if not item["supported"]}
        if unsupported:
            raise ValueError(f"R11 threshold class support below {MIN_CLASS_SUPPORT}: {unsupported}")
        self.fit_rows = int(len(target))
        self.fit_fingerprint = _fit_fingerprint(selected, FEATURE_NAMES, target)
        self.target_sha256 = _array_digest(target)
        self.threshold_support = support

        self.preprocessor = _SelectedPreprocessor(tuple(STRUCTURED_FEATURES))
        structured, categorical_mask = self.preprocessor.fit_transform(
            selected.loc[:, list(STRUCTURED_FEATURES)]
        )
        self.categorical_positions = [i for i, is_cat in enumerate(categorical_mask) if is_cat]
        documents, missing = _text_documents(selected["signal_stream"])
        self.vectorizer, tfidf, text_fallback = RegressionTextModel._fit_vectorizer(documents)
        if self.vectorizer is None or tfidf.shape[1] <= int(SVD_PARAMETERS.get("n_components", 32)):
            raise ValueError("R11 requires a fit-only TF-IDF vocabulary wider than 32 columns")
        self.svd = TruncatedSVD(n_components=32, **dict(SVD_PARAMETERS))
        text_projection = np.asarray(self.svd.fit_transform(tfidf), dtype=np.float64)
        if text_projection.shape != (len(selected), 32) or not np.isfinite(text_projection).all():
            raise RuntimeError("R11 SVD32 fit returned an invalid projection")
        self.vectorizer_fallback = text_fallback
        self.fit_missing_stream_rows = int(missing.sum())
        self.feature_names = (
            list(self.preprocessor.transformed_feature_names)
            + [f"signal_stream_svd_{i + 1}" for i in range(32)]
            + ["signal_stream_missing"]
        )
        design = np.column_stack((np.asarray(structured, dtype=np.float64),
                                  text_projection, missing.astype(np.float64)))
        if design.shape != (len(selected), len(self.feature_names)) or not np.isfinite(design).all():
            raise RuntimeError("R11 training design is misaligned or nonfinite")
        design_frame = pd.DataFrame(design, columns=self.feature_names)
        self.design_sha256 = _array_digest(design)
        self.heads: list[LGBMClassifier] = []
        self.head_model_digests: list[str] = []
        self.head_tree_summaries: list[dict[str, Any]] = []
        self.head_fit_calls = 0
        self.heads_support: dict[str, dict[str, int]] = {}
        for index, threshold in enumerate(THRESHOLDS):
            head = LGBMClassifier(**dict(HEAD_PARAMETERS))
            head.fit(design_frame, labels[:, index], categorical_feature=self.categorical_positions)
            if not np.array_equal(np.asarray(head.classes_), np.array([0, 1])):
                raise RuntimeError(f"R11 threshold {threshold} head did not fit both binary classes")
            self.heads.append(head)
            self.head_fit_calls += 1
            self.heads_support[str(threshold)] = {
                "negative": int(np.sum(labels[:, index] == 0)),
                "positive": int(np.sum(labels[:, index] == 1)),
            }
            model_text = head.booster_.model_to_string(num_iteration=-1)
            self.head_model_digests.append(hashlib.sha256(model_text.encode("utf-8")).hexdigest())
            self.head_tree_summaries.append(_tree_summary(head.booster_))
        self._state = self._make_state(target, design.shape)
        self._fitted = True
        self._diagnostics = {"n_predictions": 0}
        return self

    def _make_state(self, target: np.ndarray, design_shape: tuple[int, int]) -> dict[str, Any]:
        payload = {
            "id": "V3_BATCH1_R11_CUMULATIVE_ORDINAL_SVD32",
            "input_feature_names": list(FEATURE_NAMES),
            "fit_rows": self.fit_rows,
            "fit_data_sha256": self.fit_fingerprint,
            "target_sha256": self.target_sha256,
            "target_range": [float(target.min()), float(target.max())],
            "thresholds": list(THRESHOLDS),
            "minimum_class_support": MIN_CLASS_SUPPORT,
            "threshold_support": self.heads_support,
            "head_parameters": dict(HEAD_PARAMETERS),
            "head_fit_calls": self.head_fit_calls,
            "head_model_sha256": self.head_model_digests,
            "head_tree_summaries": self.head_tree_summaries,
            "structured_state": self.preprocessor.audit_state(),
            "categorical_feature_positions": list(self.categorical_positions),
            "tfidf": _vectorizer_state(self.vectorizer, self.vectorizer_fallback),
            "svd_parameters": {**dict(SVD_PARAMETERS), "n_components": 32},
            "svd_components_sha256": _array_digest(self.svd.components_),
            "svd_explained_variance_ratio": np.asarray(
                self.svd.explained_variance_ratio_, dtype=np.float64
            ).tolist(),
            "design_shape": [int(design_shape[0]), int(design_shape[1])],
            "design_sha256": self.design_sha256,
            "fit_missing_stream_rows": self.fit_missing_stream_rows,
            "lightgbm_version": lightgbm.__version__,
            "scikit_learn_version": sklearn.__version__,
            "median_rule": "row-wise nonincreasing PAVA; interpolate where survival probability crosses 0.5 on grid 0,10,...,100",
            "discretization_assumption": "Linear interpolation across adjacent decile thresholds approximates a continuous conditional median. S(0)=1 is valid only when the fit fold has no target equal to 0; fit rejects zero targets. S(100)=0 follows strict y>100 being false.",
            "validation_data_passed_to_fit": False,
            "validation_labels_passed_to_fit": False,
            "test_data_used": False,
            "predictor_fit_calls": int(self.head_fit_calls),
        }
        payload["state_sha256"] = _json_digest(_safe_builtin(payload))
        return json.loads(json.dumps(_safe_builtin(payload), sort_keys=True, allow_nan=False))

    def _transform(self, X: pd.DataFrame) -> pd.DataFrame:
        selected = self._select(X)
        structured, _ = self.preprocessor.transform(selected.loc[:, list(STRUCTURED_FEATURES)])
        documents, missing = _text_documents(selected["signal_stream"])
        tfidf = self.vectorizer.transform(documents)
        projected = np.asarray(self.svd.transform(tfidf), dtype=np.float64)
        design = np.column_stack((np.asarray(structured, dtype=np.float64),
                                  projected, missing.astype(np.float64)))
        if design.shape != (len(selected), len(self.feature_names)) or not np.isfinite(design).all():
            raise RuntimeError("R11 score design is misaligned or nonfinite")
        return pd.DataFrame(design, columns=self.feature_names)

    def predict_survival(self, X: pd.DataFrame) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("R11 must be fitted before prediction")
        design = self._transform(X)
        raw = np.empty((len(design), len(THRESHOLDS)), dtype=np.float64)
        for index, head in enumerate(self.heads):
            classes = np.asarray(head.classes_)
            positive_column = int(np.flatnonzero(classes == 1)[0])
            raw[:, index] = np.asarray(head.predict_proba(design), dtype=np.float64)[:, positive_column]
        if not np.isfinite(raw).all() or np.any((raw < 0.0) | (raw > 1.0)):
            raise RuntimeError("R11 LightGBM heads returned invalid survival probabilities")
        projected = pava_nonincreasing(raw)
        self._diagnostics = {
            "n_predictions": int(len(design)),
            "raw_survival_monotonicity_violations": int(np.sum(np.diff(raw, axis=1) > 1e-10)),
            "pava_adjusted_rows": int(np.sum(np.any(np.abs(projected - raw) > 1e-12, axis=1))),
        }
        return projected

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        prediction = median_from_survival(self.predict_survival(X))
        if prediction.shape != (len(X),) or not np.isfinite(prediction).all():
            raise RuntimeError("R11 returned invalid median predictions")
        return prediction

    def audit_state(self) -> dict[str, Any]:
        if not self._fitted:
            raise RuntimeError("R11 cannot expose fitted state before training")
        return json.loads(json.dumps(self._state, sort_keys=True, allow_nan=False))

    def prediction_diagnostics(self) -> dict[str, Any]:
        return json.loads(json.dumps(_safe_builtin(self._diagnostics), sort_keys=True, allow_nan=False))


def fit_predict(fit_X: pd.DataFrame, fit_y: Any, score_X: pd.DataFrame, *,
                controls: Mapping[str, Any], card_id: str, cutoff: int) -> dict[str, Any]:
    if card_id != CARD_ID or int(cutoff) not in {55, 67}:
        raise ValueError("R11 adapter accepts only card R11 at cutoffs 55 and 67")
    fit_selected = _select_exact(fit_X)
    score_selected = _select_exact(score_X)
    if (not fit_selected.index.is_unique or not fit_selected.index.is_monotonic_increasing
            or not score_selected.index.is_unique or not score_selected.index.is_monotonic_increasing):
        raise ValueError("R11 fit and score rows must preserve increasing Train positions")
    fit_cycles = validated_cycle_codes(fit_selected["cycle_code"], name="R11 fit cycle_code")
    score_cycles = validated_cycle_codes(score_selected["cycle_code"], name="R11 score cycle_code")
    if (np.any((fit_cycles < 47) | (fit_cycles > int(cutoff)))
            or np.any((score_cycles <= int(cutoff)) | (score_cycles > int(cutoff) + 12))):
        raise ValueError("R11 fit/score rows violate the frozen temporal horizon")
    if not {"baseline_prediction", "cycle_code"}.issubset(controls):
        raise ValueError("R11 control bridge requires baseline_prediction and cycle_code")
    baseline = np.asarray(controls["baseline_prediction"], dtype=np.float64)
    cycle_control = validated_cycle_codes(controls["cycle_code"], name="R11 control cycle_code")
    if (baseline.shape != (len(score_selected),) or cycle_control.shape != baseline.shape
            or not np.array_equal(cycle_control, score_cycles)):
        raise ValueError("R11 score rows do not align with the row-exact frozen control bridge")
    model = CumulativeOrdinalSVD32().fit(fit_selected, fit_y)
    raw = model.predict(score_selected)
    mapped = rank50_prediction(raw, score_cycles)
    prediction = np.clip(BASELINE_WEIGHT * baseline + CANDIDATE_WEIGHT * mapped, *FINAL_RANGE)
    return {
        "raw_candidate": raw,
        "prediction": prediction,
        "fit_predictor_count": int(model.head_fit_calls),
        "fit_metadata": {
            "cutoff": int(cutoff),
            "fit_rows": int(len(fit_selected)),
            "score_rows": int(len(score_selected)),
            "policy": "clip(0.75*frozen_three_component_champion + 0.25*rank50(raw R11 target-scale median),0,100)",
            "rank_mapping": "per complete future cycle; average ties; percentile=100*rank/n; rank50=0.5*raw+0.5*percentile; clip[0,100]",
            "vectorizer_fit_calls": 1,
            "svd_fit_calls": 1,
            "predictor_fit_calls": int(model.head_fit_calls),
            "thresholds": list(THRESHOLDS),
            "threshold_support": model.threshold_support,
            "validation_data_passed_to_fit": False,
            "validation_labels_passed_to_fit": False,
            "test_data_used": False,
        },
        "state": model.audit_state(),
        "prediction_diagnostics": model.prediction_diagnostics(),
        "audit_arrays": {
            "baseline_prediction": baseline,
            "candidate_rank50": mapped,
            "cycle_code": score_cycles,
        },
    }


__all__ = [
    "BASELINE_WEIGHT", "CARD_ID", "CANDIDATE_WEIGHT", "CumulativeOrdinalSVD32",
    "FEATURE_NAMES", "HEAD_PARAMETERS", "MIN_CLASS_SUPPORT", "SVD_PARAMETERS",
    "THRESHOLDS", "TFIDF_PARAMETERS", "compose_r11_prediction", "fit_predict",
    "median_from_survival", "pava_nonincreasing", "rank50_prediction",
    "threshold_support", "threshold_targets",
    "validated_cycle_codes",
]
