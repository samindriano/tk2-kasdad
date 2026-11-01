"""R05 binary word-presence sparse regression candidate.

Only the text representation changes from the V2 sparse L1 model: word 1--2
CountVectorizer features are binary, with no IDF weighting or row normalization.
Fold construction, target access, controls, and persistence live in the runner.
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
from sklearn.feature_extraction.text import CountVectorizer

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
COUNT_PARAMETERS: dict[str, Any] = {
    "ngram_range": (1, 2),
    "min_df": 2,
    "max_features": 20_000,
    "binary": True,
}
RIDGE25_WEIGHT = 0.75
SPARSE_RANK50_WEIGHT = 0.25
FINAL_RANGE = (0.0, 100.0)


def _count_state(vectorizer: CountVectorizer | None, fallback: str | None) -> dict[str, Any]:
    params: dict[str, Any] = {}
    if vectorizer is None:
        # Report all configured parameters and the relevant sklearn defaults.
        params = {
            **COUNT_PARAMETERS,
            "lowercase": True,
            "token_pattern": r"(?u)\\b\\w\\w+\\b",
            "strip_accents": None,
            "dtype": "int64",
            "use_idf": False,
            "row_normalization": None,
        }
        params["ngram_range"] = list(params["ngram_range"])
        return {
            "parameters": params,
            "vocabulary_size": 0,
            "vocabulary_sha256": None,
            "fallback": fallback,
        }
    for key, value in vectorizer.get_params(deep=False).items():
        if isinstance(value, tuple):
            params[key] = list(value)
        elif isinstance(value, type):
            params[key] = value.__name__
        elif isinstance(value, np.generic):
            params[key] = value.item()
        else:
            params[key] = value
    params["use_idf"] = False
    params["row_normalization"] = None
    ordered = [(str(token), int(index)) for token, index in
               sorted(vectorizer.vocabulary_.items(), key=lambda pair: pair[1])]
    return {
        "parameters": params,
        "vocabulary_size": len(ordered),
        "vocabulary_sha256": _json_digest(ordered),
        "fallback": None,
    }


def rank50_prediction(raw_prediction: Any, cycle_codes: Any) -> np.ndarray:
    """Map raw scores per full cycle using average ranks, then clip to target scale."""
    raw = np.asarray(raw_prediction, dtype=np.float64)
    cycles = pd.to_numeric(pd.Series(cycle_codes), errors="raise").to_numpy(dtype=np.int64)
    if raw.ndim != 1 or cycles.shape != raw.shape or not len(raw) or not np.isfinite(raw).all():
        raise ValueError("R05 raw scores and cycles must be nonempty, finite, aligned vectors")
    mapped = np.empty(len(raw), dtype=np.float64)
    for cycle in np.unique(cycles):
        mask = cycles == cycle
        percentile = 100.0 * rankdata(raw[mask], method="average") / int(mask.sum())
        mapped[mask] = 0.5 * raw[mask] + 0.5 * percentile
    result = np.clip(mapped, *FINAL_RANGE)
    if not np.isfinite(result).all():
        raise RuntimeError("R05 rank50 output is nonfinite")
    return result


def compose_r05_prediction(ridge25: Any, raw_sparse: Any, cycle_codes: Any) -> np.ndarray:
    ridge = np.asarray(ridge25, dtype=np.float64)
    raw = np.asarray(raw_sparse, dtype=np.float64)
    cycles = pd.to_numeric(pd.Series(cycle_codes), errors="raise").to_numpy(dtype=np.int64)
    if ridge.ndim != 1 or ridge.shape != raw.shape or ridge.shape != cycles.shape or not len(ridge):
        raise ValueError("R05 policy inputs must be nonempty aligned vectors")
    if not np.isfinite(ridge).all() or np.any((ridge < 0) | (ridge > 100)):
        raise ValueError("Frozen Ridge25 predictions must be finite and within [0,100]")
    mapped = rank50_prediction(raw, cycles)
    return np.clip(RIDGE25_WEIGHT * ridge + SPARSE_RANK50_WEIGHT * mapped, *FINAL_RANGE)


class R05BinaryPresenceSparseHybrid:
    """Structured/OHE + binary word-presence CSR + LightGBM regression_l1."""

    def __init__(self) -> None:
        self.input_feature_names = FEATURE_NAMES
        self._fitted = False
        self._diagnostics: dict[str, Any] = {"n_predictions": 0}

    def _select(self, X: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(X, pd.DataFrame):
            raise TypeError("R05 X must be a pandas DataFrame")
        if X.columns.has_duplicates or set(X.columns) != set(FEATURE_NAMES) or len(X.columns) != len(FEATURE_NAMES):
            absent = sorted(set(FEATURE_NAMES) - set(X.columns))
            extra = sorted(set(X.columns) - set(FEATURE_NAMES))
            raise ValueError(f"R05 requires the exact feature allowlist; absent={absent}, extra={extra}")
        return X.loc[:, list(FEATURE_NAMES)].copy()

    @staticmethod
    def _fit_vectorizer(documents: list[str]):
        if not any(documents):
            return None, sparse.csr_matrix((len(documents), 0), dtype=np.float64), "all_training_text_empty"
        vectorizer = CountVectorizer(**COUNT_PARAMETERS)
        try:
            matrix = vectorizer.fit_transform(documents).tocsr()
        except ValueError as exc:
            message = str(exc).casefold()
            if not any(token in message for token in ("empty vocabulary", "after pruning, no terms remain")):
                raise
            return None, sparse.csr_matrix((len(documents), 0), dtype=np.float64), "empty_vocabulary_after_filters"
        if matrix.nnz and not np.isin(matrix.data, (0, 1)).all():
            raise RuntimeError("R05 binary CountVectorizer emitted a nonbinary value")
        return vectorizer, matrix, None

    def fit(self, X: pd.DataFrame, y: Any) -> "R05BinaryPresenceSparseHybrid":
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
        design = sparse.hstack((structured_matrix, text_matrix, missing_column),
                               format="csr", dtype=np.float64)
        design.sum_duplicates()
        design.sort_indices()
        if design.shape[0] != self.fit_rows or not np.isfinite(design.data).all():
            raise RuntimeError("R05 sparse fit matrix is misaligned or nonfinite")

        self.estimator = lightgbm.LGBMRegressor(**dict(SPARSE_LGB_PARAMETERS))
        self.estimator.fit(design, target)
        model_text = self.estimator.booster_.model_to_string(num_iteration=-1)
        self.state = {
            "id": "V3_BATCH1_R05_BINARY_WORD_PRESENCE_SPARSE_L1",
            "recipe": "binary CountVectorizer word 1-2 + structured/OHE + missing indicator; LightGBM regression_l1",
            "only_intended_change": "word 1-2 TF-IDF -> binary CountVectorizer presence; min_df=2 and cap=20,000 retained",
            "input_feature_names": list(FEATURE_NAMES),
            "fit_rows": self.fit_rows,
            "fit_data_sha256": self.fit_fingerprint,
            "target_sha256": self.target_sha256,
            "target_range": [float(target.min()), float(target.max())],
            "training_design_shape": [int(design.shape[0]), int(design.shape[1])],
            "training_design_nnz": int(design.nnz),
            "structured_width": int(structured_matrix.shape[1]),
            "count_width": int(text_matrix.shape[1]),
            "count_nnz": int(text_matrix.nnz),
            "count_vectorizer": _count_state(self.vectorizer, text_fallback),
            "structured_transform": structured_state,
            "missing_stream_fit_rows": int(missing.sum()),
            "missing_stream_indicator": True,
            "sparse_design": True,
            "svd_used": False,
            "native_categorical_features_used": False,
            "parameters": dict(SPARSE_LGB_PARAMETERS),
            "trees_fitted": int(self.estimator.booster_.current_iteration()),
            "feature_importance_sha256": _array_digest(self.estimator.feature_importances_, dtype="<i8"),
            "fitted_model_sha256": hashlib.sha256(model_text.encode("utf-8")).hexdigest(),
            "lightgbm_version": lightgbm.__version__,
            "scikit_learn_version": sklearn.__version__,
            "validation_data_passed_to_fit": False,
            "early_stopping_used": False,
            "test_data_used": False,
        }
        self.state["state_sha256"] = _json_digest(_safe_builtin(self.state))
        self._fitted = True
        self._diagnostics = {"n_predictions": 0}
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("R05 model must be fitted before prediction")
        selected = self._select(X)
        documents, missing = _text_documents(selected["signal_stream"])
        if self.vectorizer is None:
            text_matrix = sparse.csr_matrix((len(selected), 0), dtype=np.float64)
            unseen_token_count = sum(len(document.split()) for document in documents)
        else:
            text_matrix = self.vectorizer.transform(documents).tocsr()
            analyzer = self.vectorizer.build_analyzer()
            vocabulary = self.vectorizer.vocabulary_
            unseen_token_count = sum(token not in vocabulary for document in documents for token in analyzer(document))
        structured, unknown = RegressionTextModel._structured_transform(
            selected, self.processor, self.scaler, self.encoder, self.numeric_positions
        )
        missing_column = sparse.csr_matrix(missing.astype(np.float64).reshape(-1, 1))
        design = sparse.hstack((structured, text_matrix, missing_column), format="csr", dtype=np.float64)
        design.sum_duplicates()
        design.sort_indices()
        if design.shape[1] != self.state["training_design_shape"][1] or not np.isfinite(design.data).all():
            raise RuntimeError("R05 score design differs from the fitted schema or is nonfinite")
        prediction = np.asarray(self.estimator.predict(design), dtype=np.float64)
        if prediction.shape != (len(selected),) or not np.isfinite(prediction).all():
            raise RuntimeError("R05 LightGBM returned invalid raw scores")
        self._diagnostics = {
            "n_predictions": int(len(prediction)),
            "signal_stream_missing_rows": int(missing.sum()),
            "unseen_token_count": int(unseen_token_count),
            "unknown_category_rows": dict(sorted(unknown.items())),
        }
        return prediction

    def audit_state(self) -> dict[str, Any]:
        if not self._fitted:
            raise RuntimeError("R05 cannot expose fitted state before training")
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
    if card_id != "R05":
        raise ValueError(f"R05 adapter received card {card_id!r}")
    required = {"baseline_prediction", "ridge25_prediction", "baseline_sparse_raw", "cycle_code"}
    if not required.issubset(controls):
        raise ValueError(f"R05 frozen-control bridge missing {sorted(required - set(controls))}")
    model = R05BinaryPresenceSparseHybrid().fit(fit_X, fit_y)
    raw_candidate = model.predict(score_X)
    cycles = np.asarray(controls["cycle_code"], dtype=np.int64)
    baseline_reconstructed = compose_r05_prediction(
        controls["ridge25_prediction"], controls["baseline_sparse_raw"], cycles
    )
    baseline_saved = np.asarray(controls["baseline_prediction"], dtype=np.float64)
    if baseline_saved.shape != baseline_reconstructed.shape or not np.allclose(
        baseline_saved, baseline_reconstructed, atol=1e-10, rtol=0.0
    ):
        raise ValueError("R05 frozen champion control does not reconstruct from row-aligned caches")
    prediction = compose_r05_prediction(controls["ridge25_prediction"], raw_candidate, cycles)
    mapped = rank50_prediction(raw_candidate, cycles)
    return {
        "raw_candidate": raw_candidate,
        "prediction": prediction,
        "fit_predictor_count": 1,
        "fit_metadata": {
            "cutoff": int(cutoff),
            "fit_rows": int(len(fit_X)),
            "score_rows": int(len(score_X)),
            "policy": "clip(0.75*Ridge25 + 0.25*rank50(raw R05 candidate),0,100)",
            "rank_mapping": "per complete future cycle; average ties; percentile=100*rank/n; rank50=0.5*raw+0.5*percentile; clip[0,100]",
            "vectorizer_fit_calls": 1,
            "predictor_fit_calls": 1,
            "validation_labels_passed_to_estimator": False,
            "test_data_used": False,
        },
        "state": model.audit_state(),
        "audit_arrays": {"candidate_rank50": mapped},
    }


__all__ = [
    "COUNT_PARAMETERS", "FEATURE_NAMES", "R05BinaryPresenceSparseHybrid",
    "compose_r05_prediction", "fit_predict", "rank50_prediction",
]
