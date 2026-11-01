"""R06 character TF-IDF + structured sparse L1 regression candidate.

This module owns one fixed temporal-fold predictor and the card's fixed output
composition. The caller owns data loading, fold selection, controls, freezing,
and persistence.
"""
from __future__ import annotations

import hashlib
from typing import Any, Mapping

import lightgbm
import numpy as np
import pandas as pd
import sklearn
from scipy import sparse
from scipy.stats import rankdata
from sklearn.feature_extraction.text import TfidfVectorizer

from src.models.optimization_v2_regression import SPARSE_LGB_PARAMETERS
from src.models.regression_research_text import (
    HYBRID_FEATURES,
    RegressionTextModel,
    _array_digest,
    _json_digest,
    _safe_builtin,
    _target,
    _text_documents,
)
from src.models.round1_models import _fit_fingerprint


FEATURE_NAMES = tuple(HYBRID_FEATURES)
CHAR_TFIDF_PARAMETERS: dict[str, Any] = {
    "analyzer": "char_wb",
    "ngram_range": (3, 5),
    "min_df": 3,
    "max_features": 30_000,
    "sublinear_tf": True,
    "dtype": np.float64,
}
BASELINE_WEIGHT = 0.75
CANDIDATE_WEIGHT = 0.25
FINAL_RANGE = (0.0, 100.0)


def _vectorizer_state(
    vectorizer: TfidfVectorizer | None,
    fallback: str | None,
) -> dict[str, Any]:
    parameters: dict[str, Any] = {
        **CHAR_TFIDF_PARAMETERS,
        "ngram_range": list(CHAR_TFIDF_PARAMETERS["ngram_range"]),
        "dtype": "float64",
        # Explicitly record sklearn defaults that are part of the fitted recipe.
        "lowercase": True,
        "strip_accents": None,
        "norm": "l2",
        "use_idf": True,
        "smooth_idf": True,
        "token_pattern": None,
    }
    if vectorizer is None:
        return {
            "parameters": parameters,
            "vocabulary_size": 0,
            "vocabulary_sha256": None,
            "idf_sha256": None,
            "fallback": fallback,
        }
    ordered = [
        (str(token), int(index))
        for token, index in sorted(vectorizer.vocabulary_.items(), key=lambda pair: pair[1])
    ]
    return {
        "parameters": parameters,
        "vocabulary_size": len(ordered),
        "vocabulary_sha256": _json_digest(ordered),
        "idf_sha256": _array_digest(vectorizer.idf_),
        "fallback": None,
    }


def rank50_prediction(raw_prediction: Any, cycle_codes: Any) -> np.ndarray:
    """Within each complete cycle, blend raw score and average-rank percentile."""
    raw = np.asarray(raw_prediction, dtype=np.float64)
    cycles_numeric = pd.to_numeric(pd.Series(cycle_codes), errors="raise")
    if cycles_numeric.isna().any():
        raise ValueError("R06 cycle codes must be present")
    cycles_float = cycles_numeric.to_numpy(dtype=np.float64)
    if not np.isfinite(cycles_float).all() or not np.equal(cycles_float, np.floor(cycles_float)).all():
        raise ValueError("R06 cycle codes must be finite integers")
    cycles = cycles_float.astype(np.int64)
    if raw.ndim != 1 or cycles.shape != raw.shape or not len(raw) or not np.isfinite(raw).all():
        raise ValueError("R06 raw scores and cycles must be nonempty, finite, aligned vectors")
    mapped = np.empty(len(raw), dtype=np.float64)
    for cycle in np.unique(cycles):
        mask = cycles == cycle
        percentile = 100.0 * rankdata(raw[mask], method="average") / int(mask.sum())
        mapped[mask] = 0.5 * raw[mask] + 0.5 * percentile
    result = np.clip(mapped, *FINAL_RANGE)
    if not np.isfinite(result).all():
        raise RuntimeError("R06 rank50 output is nonfinite")
    return result


def compose_r06_prediction(
    baseline_prediction: Any,
    raw_candidate: Any,
    cycle_codes: Any,
) -> np.ndarray:
    """Apply the frozen three-component champion + R06 rank50 policy."""
    baseline = np.asarray(baseline_prediction, dtype=np.float64)
    raw = np.asarray(raw_candidate, dtype=np.float64)
    cycles = pd.to_numeric(pd.Series(cycle_codes), errors="raise").to_numpy(dtype=np.float64)
    if (baseline.ndim != 1 or baseline.shape != raw.shape or baseline.shape != cycles.shape
            or not len(baseline)):
        raise ValueError("R06 composition inputs must be nonempty aligned vectors")
    if (not np.isfinite(baseline).all() or np.any((baseline < 0.0) | (baseline > 100.0))
            or not np.isfinite(raw).all() or not np.isfinite(cycles).all()):
        raise ValueError("R06 composition inputs must be finite; baseline must be within [0,100]")
    mapped = rank50_prediction(raw, cycles)
    result = np.clip(BASELINE_WEIGHT * baseline + CANDIDATE_WEIGHT * mapped, *FINAL_RANGE)
    if not np.isfinite(result).all():
        raise RuntimeError("R06 composed output is nonfinite")
    return result


class R06CharacterSparseL1Hybrid:
    """Fit-only char_wb TF-IDF + structured/OHE + LightGBM regression_l1."""

    def __init__(self) -> None:
        self.input_feature_names = FEATURE_NAMES
        self._fitted = False
        self._diagnostics: dict[str, Any] = {"n_predictions": 0}

    def _select(self, X: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(X, pd.DataFrame):
            raise TypeError("R06 X must be a pandas DataFrame")
        if (X.columns.has_duplicates or set(X.columns) != set(FEATURE_NAMES)
                or len(X.columns) != len(FEATURE_NAMES)):
            absent = sorted(set(FEATURE_NAMES) - set(X.columns))
            extra = sorted(set(X.columns) - set(FEATURE_NAMES))
            raise ValueError(f"R06 requires the exact feature allowlist; absent={absent}, extra={extra}")
        return X.loc[:, list(FEATURE_NAMES)].copy()

    @staticmethod
    def _fit_vectorizer(documents: list[str]):
        if not any(documents):
            return None, sparse.csr_matrix((len(documents), 0), dtype=np.float64), "all_training_text_empty"
        vectorizer = TfidfVectorizer(**CHAR_TFIDF_PARAMETERS)
        try:
            matrix = vectorizer.fit_transform(documents).tocsr()
        except ValueError as exc:
            message = str(exc).casefold()
            if not any(token in message for token in ("empty vocabulary", "after pruning, no terms remain")):
                raise
            return None, sparse.csr_matrix((len(documents), 0), dtype=np.float64), "empty_vocabulary_after_filters"
        if not sparse.isspmatrix_csr(matrix) or not np.isfinite(matrix.data).all():
            raise RuntimeError("R06 char TF-IDF did not produce a finite CSR matrix")
        return vectorizer, matrix, None

    def fit(self, X: pd.DataFrame, y: Any) -> "R06CharacterSparseL1Hybrid":
        self._fitted = False
        selected = self._select(X)
        target = _target(y, len(selected))
        self.fit_rows = int(len(target))
        self.fit_fingerprint = _fit_fingerprint(selected, FEATURE_NAMES, target)
        self.target_sha256 = _array_digest(target)

        documents, missing = _text_documents(selected["signal_stream"])
        self.vectorizer, text_matrix, text_fallback = self._fit_vectorizer(documents)
        (structured_matrix, self.processor, self.scaler, self.encoder,
         self.numeric_positions, structured_state) = RegressionTextModel._structured_fit(selected)
        missing_column = sparse.csr_matrix(missing.astype(np.float64).reshape(-1, 1))
        design = sparse.hstack(
            (structured_matrix, text_matrix, missing_column), format="csr", dtype=np.float64
        )
        design.sum_duplicates()
        design.sort_indices()
        if (not sparse.isspmatrix_csr(design) or design.shape[0] != self.fit_rows
                or not np.isfinite(design.data).all()):
            raise RuntimeError("R06 sparse fit design is misaligned or nonfinite")

        self.estimator = lightgbm.LGBMRegressor(**dict(SPARSE_LGB_PARAMETERS))
        self.estimator.fit(design, target)
        model_text = self.estimator.booster_.model_to_string(num_iteration=-1)
        self.state = {
            "id": "V3_BATCH1_R06_CHAR_SPARSE_L1",
            "recipe": "char_wb TF-IDF ngrams 3-5 + structured/OHE + missing indicator; LightGBM regression_l1",
            "only_intended_change": "character TF-IDF block; keep original structured/OHE and sparse L1 model",
            "input_feature_names": list(FEATURE_NAMES),
            "fit_rows": self.fit_rows,
            "fit_data_sha256": self.fit_fingerprint,
            "target_sha256": self.target_sha256,
            "target_range": [float(target.min()), float(target.max())],
            "training_design_shape": [int(design.shape[0]), int(design.shape[1])],
            "training_design_nnz": int(design.nnz),
            "structured_width": int(structured_matrix.shape[1]),
            "char_tfidf_width": int(text_matrix.shape[1]),
            "char_tfidf_nnz": int(text_matrix.nnz),
            "char_tfidf": _vectorizer_state(self.vectorizer, text_fallback),
            "structured_transform": structured_state,
            "missing_stream_fit_rows": int(missing.sum()),
            "missing_stream_indicator": True,
            "sparse_design": True,
            "word_tfidf_used": False,
            "svd_used": False,
            "native_categorical_features_used": False,
            "parameters": dict(SPARSE_LGB_PARAMETERS),
            "trees_fitted": int(self.estimator.booster_.current_iteration()),
            "feature_importance_sha256": _array_digest(self.estimator.feature_importances_, dtype="<i8"),
            "fitted_model_sha256": hashlib.sha256(model_text.encode("utf-8")).hexdigest(),
            "lightgbm_version": lightgbm.__version__,
            "scikit_learn_version": sklearn.__version__,
            "validation_data_passed_to_fit": False,
            "validation_labels_passed_to_fit": False,
            "early_stopping_used": False,
            "test_data_used": False,
        }
        self.state["state_sha256"] = _json_digest(_safe_builtin(self.state))
        self._fitted = True
        self._diagnostics = {"n_predictions": 0}
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("R06 model must be fitted before prediction")
        selected = self._select(X)
        documents, missing = _text_documents(selected["signal_stream"])
        if self.vectorizer is None:
            text_matrix = sparse.csr_matrix((len(selected), 0), dtype=np.float64)
            unseen_count = sum(len(document) for document in documents)
        else:
            text_matrix = self.vectorizer.transform(documents).tocsr()
            analyzer = self.vectorizer.build_analyzer()
            vocabulary = self.vectorizer.vocabulary_
            unseen_count = sum(
                token not in vocabulary for document in documents for token in analyzer(document)
            )
        structured, unknown = RegressionTextModel._structured_transform(
            selected, self.processor, self.scaler, self.encoder, self.numeric_positions
        )
        missing_column = sparse.csr_matrix(missing.astype(np.float64).reshape(-1, 1))
        design = sparse.hstack((structured, text_matrix, missing_column), format="csr", dtype=np.float64)
        design.sum_duplicates()
        design.sort_indices()
        if (design.shape[1] != self.state["training_design_shape"][1]
                or not np.isfinite(design.data).all()):
            raise RuntimeError("R06 prediction design differs from fit schema or is nonfinite")
        raw = np.asarray(self.estimator.predict(design), dtype=np.float64)
        if raw.shape != (len(selected),) or not np.isfinite(raw).all():
            raise RuntimeError("R06 LightGBM returned invalid raw scores")
        self._diagnostics = {
            "n_predictions": int(len(raw)),
            "signal_stream_missing_rows": int(missing.sum()),
            "unseen_char_ngram_count": int(unseen_count),
            "unknown_category_rows": dict(sorted(unknown.items())),
        }
        return raw

    def audit_state(self) -> dict[str, Any]:
        if not self._fitted:
            raise RuntimeError("R06 cannot expose fitted state before training")
        return _safe_builtin(self.state)

    def prediction_diagnostics(self) -> dict[str, Any]:
        return _safe_builtin(self._diagnostics)


def fit_predict(
    fit_X: pd.DataFrame,
    fit_y: Any,
    score_X: pd.DataFrame,
    *,
    controls: Mapping[str, Any],
    card_id: str,
    cutoff: int,
) -> dict[str, Any]:
    if card_id != "R06":
        raise ValueError(f"R06 adapter received card {card_id!r}")
    required = {"baseline_prediction", "cycle_code"}
    if not required.issubset(controls):
        raise ValueError(f"R06 frozen-control bridge missing {sorted(required - set(controls))}")
    model = R06CharacterSparseL1Hybrid().fit(fit_X, fit_y)
    raw_candidate = model.predict(score_X)
    baseline = np.asarray(controls["baseline_prediction"], dtype=np.float64)
    cycles = np.asarray(controls["cycle_code"])
    prediction = compose_r06_prediction(baseline, raw_candidate, cycles)
    mapped = rank50_prediction(raw_candidate, cycles)
    return {
        "raw_candidate": raw_candidate,
        "prediction": prediction,
        "fit_predictor_count": 1,
        "fit_metadata": {
            "cutoff": int(cutoff),
            "fit_rows": int(len(fit_X)),
            "score_rows": int(len(score_X)),
            "policy": "clip(0.75*frozen_three_component_champion + 0.25*rank50(raw R06),0,100)",
            "rank_mapping": "per complete future cycle; average ties; percentile=100*rank/n; rank50=0.5*raw+0.5*percentile; clip[0,100]",
            "vectorizer_fit_calls": 1,
            "predictor_fit_calls": 1,
            "validation_data_passed_to_fit": False,
            "validation_labels_passed_to_fit": False,
            "test_data_used": False,
        },
        "state": model.audit_state(),
        "audit_arrays": {
            "baseline_prediction": baseline,
            "candidate_rank50": mapped,
            "cycle_code": np.asarray(cycles, dtype=np.int64),
        },
    }


__all__ = [
    "BASELINE_WEIGHT", "CANDIDATE_WEIGHT", "CHAR_TFIDF_PARAMETERS", "FEATURE_NAMES",
    "R06CharacterSparseL1Hybrid", "SPARSE_LGB_PARAMETERS", "compose_r06_prediction",
    "fit_predict", "rank50_prediction",
]
