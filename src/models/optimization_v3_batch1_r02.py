"""R02: V2 sparse hybrid regression with a wider fit-only word vocabulary.

The inherited SparseHybridRegressor recipe is copied narrowly so that the
only estimator/preprocessing change is TF-IDF ``max_features=60000`` instead
of 20000. Fold construction, frozen-control loading, and output persistence
belong to the scripts/optimization_v3_batch1_r02.py bridge.
"""
from __future__ import annotations

import hashlib
import json
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
    _TFIDF_PARAMS,
    _array_digest,
    _json_digest,
    _safe_builtin,
    _target,
    _text_documents,
)
from src.models.round1_models import _fit_fingerprint


FEATURE_NAMES = tuple(HYBRID_FEATURES)
TFIDF_PARAMETERS: dict[str, Any] = {**_TFIDF_PARAMS, "max_features": 60_000}
EXPECTED_BASE_TFIDF_PARAMETERS: dict[str, Any] = dict(_TFIDF_PARAMS)
if EXPECTED_BASE_TFIDF_PARAMETERS.get("max_features") != 20_000:
    raise RuntimeError("Inherited V2 TF-IDF cap changed; R02 must change only max_features 20,000 to 60,000")
if {key: value for key, value in TFIDF_PARAMETERS.items() if key != "max_features"} != {
    key: value for key, value in EXPECTED_BASE_TFIDF_PARAMETERS.items() if key != "max_features"
}:
    raise RuntimeError("R02 changed a TF-IDF parameter other than max_features")

RIDGE25_WEIGHT = 0.75
SPARSE_RANK50_WEIGHT = 0.25
FINAL_RANGE = (0.0, 100.0)


def _text_state(vectorizer: TfidfVectorizer | None, fallback: str | None) -> dict[str, Any]:
    """Audit the fitted R02 vectorizer, including the one changed cap."""
    params: dict[str, Any] = {}
    for key, value in TFIDF_PARAMETERS.items():
        if isinstance(value, tuple):
            params[key] = list(value)
        elif isinstance(value, type):
            params[key] = value.__name__
        elif isinstance(value, np.generic):
            params[key] = value.item()
        else:
            params[key] = value
    # TfidfVectorizer defaults to L2 normalization; retain that inherited
    # behavior and report it explicitly without adding a recipe override.
    params["effective_default_norm"] = "l2"
    if vectorizer is None:
        return {
            "parameters": params,
            "vocabulary_size": 0,
            "vocabulary_sha256": None,
            "idf_sha256": None,
            "fallback": fallback,
        }
    ordered = [(str(token), int(index)) for token, index in
               sorted(vectorizer.vocabulary_.items(), key=lambda pair: pair[1])]
    return {
        "parameters": params,
        "vocabulary_size": len(ordered),
        "vocabulary_sha256": _json_digest(ordered),
        "idf_sha256": _array_digest(vectorizer.idf_),
        "fallback": None,
    }


def rank50_prediction(raw_prediction: Any, cycle_codes: Any) -> np.ndarray:
    """Map target-scale predictions by average-rank percentile within full cycles."""
    raw = np.asarray(raw_prediction, dtype=np.float64)
    cycles = pd.to_numeric(pd.Series(cycle_codes), errors="raise").to_numpy(dtype=np.int64)
    if raw.ndim != 1 or cycles.shape != raw.shape or len(raw) == 0:
        raise ValueError("R02 scores and cycle codes must be nonempty and aligned")
    if not np.isfinite(raw).all():
        raise ValueError("R02 raw scores must be finite")
    mapped = np.empty(len(raw), dtype=np.float64)
    for cycle in np.unique(cycles):
        mask = cycles == cycle
        percentile = 100.0 * rankdata(raw[mask], method="average") / int(mask.sum())
        mapped[mask] = 0.5 * raw[mask] + 0.5 * percentile
    result = np.clip(mapped, *FINAL_RANGE)
    if not np.isfinite(result).all():
        raise RuntimeError("R02 rank50 mapping produced nonfinite values")
    return result


def compose_r02_prediction(ridge25: Any, raw_sparse: Any, cycle_codes: Any) -> np.ndarray:
    """Apply fixed Ridge25 + 25% within-cycle sparse rank50 policy."""
    ridge = np.asarray(ridge25, dtype=np.float64)
    raw = np.asarray(raw_sparse, dtype=np.float64)
    cycles = pd.to_numeric(pd.Series(cycle_codes), errors="raise").to_numpy(dtype=np.int64)
    if ridge.ndim != 1 or ridge.shape != raw.shape or ridge.shape != cycles.shape or not len(ridge):
        raise ValueError("R02 policy inputs must be nonempty aligned vectors")
    if not np.isfinite(ridge).all() or np.any((ridge < 0) | (ridge > 100)):
        raise ValueError("Frozen Ridge25 predictions must be finite and within [0,100]")
    sparse_rank50 = rank50_prediction(raw, cycles)
    return np.clip(RIDGE25_WEIGHT * ridge + SPARSE_RANK50_WEIGHT * sparse_rank50, *FINAL_RANGE)


class R02WideVocabularySparseHybrid:
    """V2 sparse LightGBM-L1 model with only a 60k word vocabulary cap."""

    def __init__(self) -> None:
        self.input_feature_names = FEATURE_NAMES
        self._fitted = False
        self._diagnostics: dict[str, Any] = {"n_predictions": 0}

    def _select(self, X: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(X, pd.DataFrame):
            raise TypeError("R02 X must be a pandas DataFrame")
        if X.columns.has_duplicates or set(X.columns) != set(FEATURE_NAMES) or len(X.columns) != len(FEATURE_NAMES):
            absent = sorted(set(FEATURE_NAMES) - set(X.columns))
            extra = sorted(set(X.columns) - set(FEATURE_NAMES))
            raise ValueError(f"R02 requires the exact V2 feature allowlist; absent={absent}, extra={extra}")
        return X.loc[:, list(FEATURE_NAMES)].copy()

    @staticmethod
    def _fit_vectorizer(documents: list[str]):
        if not any(documents):
            return None, sparse.csr_matrix((len(documents), 0), dtype=np.float64), "all_training_text_empty"
        vectorizer = TfidfVectorizer(**TFIDF_PARAMETERS)
        try:
            matrix = vectorizer.fit_transform(documents).tocsr()
        except ValueError as exc:
            message = str(exc).casefold()
            if not any(token in message for token in ("empty vocabulary", "after pruning, no terms remain")):
                raise
            return None, sparse.csr_matrix((len(documents), 0), dtype=np.float64), "empty_vocabulary_after_filters"
        return vectorizer, matrix, None

    def fit(self, X: pd.DataFrame, y: Any) -> "R02WideVocabularySparseHybrid":
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
        design = sparse.hstack((structured_matrix, text_matrix, missing_column), format="csr", dtype=np.float64)
        design.sum_duplicates()
        design.sort_indices()
        if design.shape[0] != self.fit_rows or not np.isfinite(design.data).all():
            raise RuntimeError("R02 sparse fit matrix is misaligned or nonfinite")

        self.estimator = lightgbm.LGBMRegressor(**dict(SPARSE_LGB_PARAMETERS))
        # The estimator receives only the current temporal fit features and its
        # matching fit target. Score-window labels are not accepted here.
        self.estimator.fit(design, target)
        model_text = self.estimator.booster_.model_to_string(num_iteration=-1)
        self.state = {
            "id": "V3_BATCH1_R02_WIDE_VOCAB_SPARSE_L1",
            "recipe": "V2 sparse hybrid TF-IDF + structured/OHE + missing indicator; LightGBM L1",
            "only_intended_change": "TF-IDF max_features 20,000 -> 60,000",
            "input_feature_names": list(FEATURE_NAMES),
            "fit_rows": self.fit_rows,
            "fit_data_sha256": self.fit_fingerprint,
            "target_sha256": self.target_sha256,
            "target_range": [float(target.min()), float(target.max())],
            "training_design_shape": [int(design.shape[0]), int(design.shape[1])],
            "training_design_nnz": int(design.nnz),
            "structured_width": int(structured_matrix.shape[1]),
            "tfidf_width": int(text_matrix.shape[1]),
            "tfidf": _text_state(self.vectorizer, text_fallback),
            "structured_transform": structured_state,
            "missing_stream_fit_rows": int(missing.sum()),
            "missing_stream_indicator": True,
            "sparse_design": True,
            "svd_used": False,
            "native_categorical_features_used": False,
            "parameters": dict(SPARSE_LGB_PARAMETERS),
            "trees_fitted": int(self.estimator.booster_.current_iteration()),
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
            raise RuntimeError("R02 model must be fitted before prediction")
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
            raise RuntimeError("R02 score design differs from the fitted schema or contains nonfinite values")
        prediction = np.asarray(self.estimator.predict(design), dtype=np.float64)
        if prediction.shape != (len(selected),) or not np.isfinite(prediction).all():
            raise RuntimeError("R02 sparse LightGBM returned invalid raw target-scale predictions")
        self._diagnostics = {
            "n_predictions": int(len(prediction)),
            "signal_stream_missing_rows": int(missing.sum()),
            "unseen_tfidf_token_count": int(unseen_token_count),
            "unknown_category_rows": dict(sorted(unknown.items())),
        }
        return prediction

    def audit_state(self) -> dict[str, Any]:
        if not self._fitted:
            raise RuntimeError("R02 cannot expose fitted state before training")
        return json.loads(json.dumps(_safe_builtin(self.state), sort_keys=True, allow_nan=False))

    def prediction_diagnostics(self) -> dict[str, Any]:
        return json.loads(json.dumps(_safe_builtin(self._diagnostics), sort_keys=True, allow_nan=False))


def fit_predict(
    fit_X: pd.DataFrame,
    fit_y: Any,
    score_X: pd.DataFrame,
    *,
    controls: Mapping[str, Any],
    card_id: str,
    cutoff: int,
) -> dict[str, Any]:
    """Fit R02 on past rows and return its fixed composite score predictions."""
    if card_id != "R02":
        raise ValueError(f"R02 adapter received card {card_id!r}")
    required = {"baseline_prediction", "ridge25_prediction", "baseline_sparse_raw", "cycle_code"}
    if not required.issubset(controls):
        raise ValueError(f"R02 control bridge missing {sorted(required - set(controls))}")
    model = R02WideVocabularySparseHybrid().fit(fit_X, fit_y)
    raw_candidate = model.predict(score_X)
    cycle_codes = controls["cycle_code"]
    baseline_reconstructed = compose_r02_prediction(
        controls["ridge25_prediction"], controls["baseline_sparse_raw"], cycle_codes
    )
    baseline_saved = np.asarray(controls["baseline_prediction"], dtype=np.float64)
    if baseline_saved.shape != baseline_reconstructed.shape or not np.allclose(
        baseline_saved, baseline_reconstructed, atol=1e-10, rtol=0.0
    ):
        raise ValueError("R02 frozen current-champion control does not reconstruct from row-aligned V2 caches")
    prediction = compose_r02_prediction(controls["ridge25_prediction"], raw_candidate, cycle_codes)
    mapped_candidate = rank50_prediction(raw_candidate, cycle_codes)
    if prediction.shape != (len(score_X),) or not np.isfinite(prediction).all():
        raise RuntimeError("R02 fixed policy returned invalid or misaligned predictions")
    return {
        "raw_candidate": raw_candidate,
        "prediction": prediction,
        "fit_predictor_count": 1,
        "fit_metadata": {
            "cutoff": int(cutoff),
            "fit_rows": int(len(fit_X)),
            "score_rows": int(len(score_X)),
            "policy": "clip(0.75*Ridge25 + 0.25*rank50(raw widened-vocabulary sparse LGB), 0, 100)",
            "rank_mapping": "per complete future cycle; average ties; percentile=100*rank/n; rank50=0.5*raw+0.5*percentile; clip[0,100]",
            "current_champion_reconstruction_max_abs_error": float(np.max(np.abs(baseline_saved - baseline_reconstructed))),
            "validation_labels_passed_to_estimator": False,
            "test_data_used": False,
        },
        "state": model.audit_state(),
        "audit_arrays": {"candidate_rank50": mapped_candidate},
    }


__all__ = [
    "EXPECTED_BASE_TFIDF_PARAMETERS", "FEATURE_NAMES", "R02WideVocabularySparseHybrid",
    "SPARSE_RANK50_WEIGHT", "TFIDF_PARAMETERS", "compose_r02_prediction", "fit_predict",
    "rank50_prediction",
]
