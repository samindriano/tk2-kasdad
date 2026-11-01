"""C12: character-only sparse boosting classification pilot.

The structured block, character TF-IDF, sparse LightGBM parameters, and
availability route are fit and applied inside the frozen temporal protocol.
The word TF-IDF block is intentionally absent from the candidate design.
"""
from __future__ import annotations

import hashlib
import json
import warnings
from typing import Any, Mapping

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer

from src.models.classification_models import _text_values
from src.models.optimization_v3_sparse_classification import PARAMS as SPARSE_PARAMS
from src.models.regression_research_text import (
    HYBRID_FEATURES,
    RegressionTextModel,
    _array_digest,
    _json_digest,
)
from src.models.round1_models import _fit_fingerprint


CARD_ID = "C12"
FEATURE_NAMES = tuple(HYBRID_FEATURES)
CHAR_TFIDF_PARAMS = {
    "analyzer": "char_wb",
    "ngram_range": (3, 5),
    "min_df": 3,
    "max_features": 30_000,
    "sublinear_tf": True,
    "lowercase": True,
    "strip_accents": "unicode",
    "dtype": np.float64,
}
THREADS = 2
AUTHORIZED_CUTOFFS = frozenset({55, 67})


def _safe_params(params: Mapping[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in params.items():
        if isinstance(value, tuple):
            value = list(value)
        elif isinstance(value, type):
            value = value.__name__
        elif callable(value):
            value = f"{value.__module__}.{value.__qualname__}"
        result[str(key)] = value
    return result


def _ordered_vocabulary(vectorizer: TfidfVectorizer) -> list[tuple[str, int]]:
    return [
        (str(token), int(index))
        for token, index in sorted(vectorizer.vocabulary_.items(), key=lambda item: item[1])
    ]


class CharOnlySparseClassifier:
    """Fit-only structured/OHE plus one fit-only ``char_wb`` TF-IDF block."""

    def __init__(self) -> None:
        self.input_feature_names = FEATURE_NAMES
        self._fitted = False

    def _select(self, frame: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(frame, pd.DataFrame):
            raise TypeError("C12 requires pandas feature frames")
        if frame.columns.has_duplicates or set(frame.columns) != set(FEATURE_NAMES):
            raise ValueError("C12 requires the exact structured + signal_stream feature allowlist")
        if set(FEATURE_NAMES) & {"record_id", "artifact_signal", "threat_rank"}:
            raise ValueError("C12 feature allowlist contains an identity or protected target")
        return frame.loc[:, list(FEATURE_NAMES)].copy()

    def fit(self, frame: pd.DataFrame, target: Any) -> "CharOnlySparseClassifier":
        self._fitted = False
        selected = self._select(frame)
        raw_target = np.asarray(target)
        if raw_target.ndim != 1 or len(raw_target) != len(selected) or len(selected) == 0:
            raise ValueError("C12 requires a nonempty row-aligned binary fit target")
        try:
            numeric_target = raw_target.astype(np.float64)
        except (TypeError, ValueError) as exc:
            raise ValueError("C12 fit target must contain only binary values") from exc
        if (not np.isfinite(numeric_target).all()
                or not np.isin(numeric_target, [0.0, 1.0]).all()
                or len(np.unique(numeric_target)) != 2):
            raise ValueError("C12 requires both classes in a finite binary fit target")
        y = numeric_target.astype(np.int8)

        (structured, self._processor, self._scaler, self._encoder,
         self._numeric_positions, structured_state) = RegressionTextModel._structured_fit(selected)
        documents, missing = _text_values(selected["signal_stream"])
        self._char_vectorizer = TfidfVectorizer(**dict(CHAR_TFIDF_PARAMS))
        char_text = self._char_vectorizer.fit_transform(documents).tocsr()
        missing_column = sparse.csr_matrix(missing.astype(np.float64).reshape(-1, 1))
        design = sparse.hstack((structured, char_text, missing_column), format="csr", dtype=np.float64)
        if (not sparse.isspmatrix_csr(design)
                or design.shape[0] != len(y)
                or not np.isfinite(design.data).all()):
            raise ValueError("C12 sparse training design is misaligned or nonfinite")

        parameters = dict(SPARSE_PARAMS)
        parameters.update({"n_jobs": THREADS, "deterministic": True, "force_col_wise": True})
        self._estimator = LGBMClassifier(**parameters)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            self._estimator.fit(design, y)
        self._fit_warnings = [f"{item.category.__name__}: {item.message}" for item in caught]

        vocabulary = _ordered_vocabulary(self._char_vectorizer)
        self._state: dict[str, Any] = {
            "id": "C12_CHAR_ONLY_SPARSE_LIGHTGBM",
            "card_id": CARD_ID,
            "fit_rows": int(len(y)),
            "input_feature_names": list(FEATURE_NAMES),
            "fit_data_sha256": _fit_fingerprint(selected, FEATURE_NAMES, y),
            "target_sha256": _array_digest(y, dtype="<i8"),
            "structured_transform": structured_state,
            "vectorizers": {
                "char_wb": {
                    "parameters": _safe_params(CHAR_TFIDF_PARAMS),
                    "vocabulary_size": int(len(vocabulary)),
                    "vocabulary_sha256": _json_digest(vocabulary),
                    "idf_sha256": _array_digest(self._char_vectorizer.idf_),
                    "nnz": int(char_text.nnz),
                },
            },
            "text_blocks": ["char_wb"],
            "word_tfidf_used": False,
            "design_shape": [int(value) for value in design.shape],
            "design_nnz": int(design.nnz),
            "design_data_sha256": _array_digest(design.data),
            "lightgbm_parameters": parameters,
            "lightgbm_model_sha256": hashlib.sha256(
                self._estimator.booster_.model_to_string().encode("utf-8")
            ).hexdigest(),
            "validation_data_passed_to_fit": False,
            "validation_labels_passed_to_fit": False,
            "early_stopping_used": False,
            "test_data_used": False,
            "fit_warnings": list(self._fit_warnings),
        }
        self._state["state_sha256"] = _json_digest(self._state)
        self._fit_metadata = {
            "fit_rows": int(len(y)),
            "training_design_nnz": int(design.nnz),
            "training_char_text_nnz": int(char_text.nnz),
            "training_text_nnz": int(char_text.nnz),
            "training_missing_stream_rows": int(missing.sum()),
            "validation_data_passed_to_fit": False,
            "validation_labels_passed_to_fit": False,
            "early_stopping_used": False,
            "fit_warnings": list(self._fit_warnings),
        }
        self._fitted = True
        return self

    def predict_proba(self, frame: pd.DataFrame) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("C12 model must be fitted before prediction")
        selected = self._select(frame)
        structured, unknown = RegressionTextModel._structured_transform(
            selected, self._processor, self._scaler, self._encoder, self._numeric_positions
        )
        documents, missing = _text_values(selected["signal_stream"])
        char_text = self._char_vectorizer.transform(documents).tocsr()
        missing_column = sparse.csr_matrix(missing.astype(np.float64).reshape(-1, 1))
        design = sparse.hstack((structured, char_text, missing_column), format="csr", dtype=np.float64)
        expected_width = int(self._state["design_shape"][1])
        if (design.shape != (len(selected), expected_width)
                or not np.isfinite(design.data).all()):
            raise RuntimeError("C12 score design differs from its fitted sparse schema")
        probability = np.asarray(
            self._estimator.booster_.predict(design, num_threads=THREADS), dtype=np.float64
        )
        if (probability.shape != (len(selected),)
                or not np.isfinite(probability).all()
                or np.any((probability < 0.0) | (probability > 1.0))):
            raise RuntimeError("C12 returned invalid positive-class probabilities")
        self._prediction_metadata = {
            "score_rows": int(len(selected)),
            "score_char_text_nnz": int(char_text.nnz),
            "score_text_nnz": int(char_text.nnz),
            "score_design_nnz": int(design.nnz),
            "unknown_structured_categories": unknown,
            "score_missing_stream_rows": int(missing.sum()),
        }
        return probability

    def audit_state(self) -> dict[str, Any]:
        if not self._fitted:
            raise RuntimeError("C12 model must be fitted before audit")
        return json.loads(json.dumps(self._state, sort_keys=True, allow_nan=False))


def fit_predict(
    fit_features: pd.DataFrame,
    fit_target: np.ndarray,
    score_features: pd.DataFrame,
    *,
    controls: Mapping[str, np.ndarray],
    card_id: str,
    cutoff: int,
) -> dict[str, Any]:
    """Fit one C12 expert and apply the frozen C2_B/CB availability route."""
    if str(card_id) != CARD_ID:
        raise ValueError(f"C12 adapter cannot execute {card_id!r}")
    cutoff = int(cutoff)
    if cutoff not in AUTHORIZED_CUTOFFS:
        raise ValueError(f"C12 pilot does not authorize cutoff {cutoff}")
    if not isinstance(controls, Mapping):
        raise TypeError("C12 controls must be a mapping")
    required = {
        "baseline_prediction", "baseline_sparse_raw", "c2b_probability",
        "cb_probability", "stream_missing", "control_rows",
    }
    absent = sorted(required - set(controls))
    if absent:
        raise ValueError(f"C12 route controls omitted required fields: {absent}")
    if not isinstance(score_features, pd.DataFrame):
        raise TypeError("C12 score features must be a pandas DataFrame")

    selected_score = CharOnlySparseClassifier()._select(score_features)
    _documents, score_missing = _text_values(selected_score["signal_stream"])
    missing = np.asarray(controls["stream_missing"], dtype=bool)
    baseline = np.asarray(controls["baseline_prediction"], dtype=np.float64)
    baseline_sparse = np.asarray(controls["baseline_sparse_raw"], dtype=np.float64)
    c2b = np.asarray(controls["c2b_probability"], dtype=np.float64)
    cb = np.asarray(controls["cb_probability"], dtype=np.float64)
    control_rows = np.asarray(controls["control_rows"], dtype=np.int64)
    row_positions = score_features.index.to_numpy(dtype=np.int64)
    if not (missing.shape == baseline.shape == baseline_sparse.shape == c2b.shape == cb.shape
            == control_rows.shape == row_positions.shape == (len(selected_score),)):
        raise ValueError("C12 route controls are not aligned to score rows")
    if (not np.array_equal(control_rows, row_positions)
            or len(np.unique(control_rows)) != len(control_rows)
            or not np.array_equal(missing, score_missing)
            or not np.isfinite(baseline).all()
            or not np.isfinite(baseline_sparse).all()
            or not np.isfinite(c2b).all()
            or not np.isfinite(cb).all()
            or np.any((baseline < 0.0) | (baseline > 1.0))
            or np.any((baseline_sparse < 0.0) | (baseline_sparse > 1.0))
            or np.any((c2b < 0.0) | (c2b > 1.0))
            or np.any((cb < 0.0) | (cb > 1.0))):
        raise ValueError("C12 route rows, mask, or frozen control probabilities are invalid")
    reconstructed_baseline = np.where(missing, cb, 0.75 * c2b + 0.25 * baseline_sparse)
    if not np.allclose(baseline, reconstructed_baseline, rtol=0.0, atol=1e-12):
        raise ValueError("C12 frozen baseline does not reconstruct from row-aligned C2_B/CB controls")

    model = CharOnlySparseClassifier().fit(fit_features, fit_target)
    raw = model.predict_proba(score_features)
    prediction = np.where(missing, cb, 0.75 * c2b + 0.25 * raw)
    if (prediction.shape != raw.shape
            or not np.isfinite(prediction).all()
            or np.any((prediction < 0.0) | (prediction > 1.0))):
        raise RuntimeError("C12 availability-routed probability is invalid")

    return {
        "prediction": prediction,
        "raw_candidate": raw,
        "fit_predictor_count": 1,
        "state": {
            "card_id": CARD_ID,
            "cutoff": cutoff,
            "blend_weight_frozen_c2b": 0.75,
            "blend_weight_new_sparse": 0.25,
            "threshold": 0.50,
            "missing_stream_route": "frozen_cb_probability",
            "present_stream_route": "0.75*frozen_c2b + 0.25*c12_char_only_probability",
            "model": model.audit_state(),
        },
        "fit_metadata": {
            **model._fit_metadata,
            **model._prediction_metadata,
            "cutoff": cutoff,
            "predictor_fits": 1,
        },
        "audit_arrays": {
            "control_rows": control_rows,
            "stream_missing": missing,
            "baseline_prediction": baseline,
            "baseline_sparse_raw": baseline_sparse,
            "frozen_c2b_probability": c2b,
            "frozen_cb_probability": cb,
        },
    }


__all__ = [
    "AUTHORIZED_CUTOFFS", "CARD_ID", "CHAR_TFIDF_PARAMS", "FEATURE_NAMES",
    "CharOnlySparseClassifier", "fit_predict",
]
