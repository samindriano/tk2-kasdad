"""Batch1 R01: sparse hybrid LinearSVR with a fixed champion auxiliary blend.

The adapter only receives past-fit features/labels and future score features.
Validation labels are never an argument to this module.
"""
from __future__ import annotations

import hashlib
import json
import warnings
from typing import Any

import numpy as np
import pandas as pd
import sklearn
from scipy import sparse
from scipy.stats import rankdata
from sklearn.exceptions import ConvergenceWarning
from sklearn.svm import LinearSVR

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
LINEAR_SVR_PARAMETERS: dict[str, Any] = {
    "epsilon": 0.0,
    "loss": "epsilon_insensitive",
    "dual": True,
    "C": 1.0,
    "tol": 1e-4,
    "max_iter": 20_000,
    "random_state": 42,
}
TARGET_TRANSFORM = "z=(threat_rank-50)/50"
TARGET_INVERSE = "prediction=50*z+50"


class R01ConvergenceError(RuntimeError):
    """The predeclared LinearSVR iteration budget did not converge."""

    def __init__(self, message: str, state: dict[str, Any]):
        super().__init__(message)
        self.state = json.loads(json.dumps(_safe_builtin(state), sort_keys=True, allow_nan=False))
        self.n_iter = int(state["n_iter"])
        self.max_iter = int(LINEAR_SVR_PARAMETERS["max_iter"])
        self.converged = bool(state["converged"])
        self.warning_messages = list(state["convergence_warning_messages"])
        self.cutoff: int | None = None
        self.fit_rows: int | None = None


def _select_features(X: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(X, pd.DataFrame):
        raise TypeError("R01 features must be a pandas DataFrame")
    if X.columns.has_duplicates or set(X.columns) != set(FEATURE_NAMES) or len(X.columns) != len(FEATURE_NAMES):
        absent = sorted(set(FEATURE_NAMES) - set(X.columns))
        extra = sorted(set(X.columns) - set(FEATURE_NAMES))
        raise ValueError(f"R01 requires the exact V2 sparse-hybrid allowlist; absent={absent}, extra={extra}")
    return X.loc[:, list(FEATURE_NAMES)].copy()


def _csr_sha256(matrix: sparse.spmatrix) -> str:
    csr = sparse.csr_matrix(matrix, dtype=np.float64)
    csr.sort_indices()
    digest = hashlib.sha256()
    digest.update(np.asarray(csr.shape, dtype="<i8").tobytes())
    digest.update(np.asarray(csr.indptr, dtype="<i8").tobytes())
    digest.update(np.asarray(csr.indices, dtype="<i8").tobytes())
    digest.update(np.asarray(csr.data, dtype="<f8").tobytes())
    return digest.hexdigest()


def rank50_prediction(raw_prediction: Any, cycle_codes: Any) -> np.ndarray:
    """Apply the historical fixed within-cycle 50% score/rank map."""
    raw = np.asarray(raw_prediction, dtype=np.float64)
    cycles = pd.to_numeric(pd.Series(cycle_codes), errors="raise").to_numpy(dtype=np.int64)
    if raw.ndim != 1 or cycles.shape != raw.shape or len(raw) == 0:
        raise ValueError("R01 raw scores and cycle codes must be nonempty and aligned")
    if not np.isfinite(raw).all():
        raise ValueError("R01 raw scores must be finite")
    mapped = np.empty(len(raw), dtype=np.float64)
    for cycle in np.unique(cycles):
        mask = cycles == cycle
        percentile = 100.0 * rankdata(raw[mask], method="average") / int(mask.sum())
        mapped[mask] = 0.5 * raw[mask] + 0.5 * percentile
    return np.clip(mapped, 0.0, 100.0)


class SparseLinearSVR:
    """V2-identical fit-only sparse hybrid transforms plus LinearSVR."""

    def __init__(self) -> None:
        self._fitted = False
        self.input_feature_names = FEATURE_NAMES

    def fit(self, X: pd.DataFrame, y: Any) -> "SparseLinearSVR":
        self._fitted = False
        selected = _select_features(X)
        target = _target(y, len(selected))
        self.fit_rows = int(len(target))
        self.fit_fingerprint = _fit_fingerprint(selected, FEATURE_NAMES, target)
        self.target_sha256 = _array_digest(target)

        documents, missing = _text_documents(selected["signal_stream"])
        self.vectorizer, text_matrix, text_fallback = RegressionTextModel._fit_vectorizer(documents)
        (structured_matrix, self.processor, self.scaler, self.encoder,
         self.numeric_positions, structured_state) = RegressionTextModel._structured_fit(selected)
        missing_column = sparse.csr_matrix(missing.astype(np.float64).reshape(-1, 1))
        design = sparse.hstack((structured_matrix, text_matrix, missing_column),
                               format="csr", dtype=np.float64)
        design.sum_duplicates()
        design.sort_indices()
        if design.shape[0] != self.fit_rows or not np.isfinite(design.data).all():
            raise RuntimeError("R01 training matrix is misaligned or nonfinite")

        self.estimator = LinearSVR(**dict(LINEAR_SVR_PARAMETERS))
        scaled_target = (target - 50.0) / 50.0
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", ConvergenceWarning)
            self.estimator.fit(design, scaled_target)
        convergence_messages = [str(item.message) for item in caught
                                if issubclass(item.category, ConvergenceWarning)]
        n_iter_raw = np.asarray(self.estimator.n_iter_)
        if n_iter_raw.size != 1:
            raise RuntimeError(f"Unexpected LinearSVR n_iter_ shape: {n_iter_raw.shape}")
        self.n_iter = int(n_iter_raw.reshape(-1)[0])
        max_iter_reached = self.n_iter >= int(LINEAR_SVR_PARAMETERS["max_iter"])
        self.converged = not convergence_messages and not max_iter_reached

        self.state = {
            "id": "V3_BATCH1_R01_SPARSE_LINEARSVR",
            "recipe": "V2 sparse hybrid TF-IDF + structured/OHE + missing indicator; LinearSVR",
            "model_family": "sklearn.svm.LinearSVR",
            "input_feature_names": list(FEATURE_NAMES),
            "fit_rows": self.fit_rows,
            "fit_data_sha256": self.fit_fingerprint,
            "target_sha256": self.target_sha256,
            "target_transform": TARGET_TRANSFORM,
            "target_inverse": TARGET_INVERSE,
            "parameters": dict(LINEAR_SVR_PARAMETERS),
            "parameters_effective": self.estimator.get_params(deep=False),
            "n_iter": self.n_iter,
            "converged": bool(self.converged),
            "convergence_warning_messages": convergence_messages,
            "training_design_shape": [int(x) for x in design.shape],
            "training_design_nnz": int(design.nnz),
            "training_design_sha256": _csr_sha256(design),
            "structured_transform": structured_state,
            "tfidf": RegressionTextModel._text_state(self.vectorizer, text_fallback),
            "missing_stream_fit_rows": int(missing.sum()),
            "svd_used": False,
            "validation_data_passed_to_fit": False,
            "early_stopping_used": False,
            "test_data_used": False,
            "scikit_learn_version": sklearn.__version__,
        }
        self.state["state_sha256"] = _json_digest(self.state)
        if not self.converged:
            # The engine truncates FIT_FAILED.error to 1,000 chars. Keep the
            # required convergence facts at the start of that durable receipt.
            prefix = (
                f"R01_CONVERGENCE_FAILED n_iter={self.n_iter} "
                f"max_iter={LINEAR_SVR_PARAMETERS['max_iter']} "
                f"converged={self.converged} warning_count={len(convergence_messages)}; state="
            )
            raise R01ConvergenceError(
                prefix + json.dumps(_safe_builtin(self.state), sort_keys=True, allow_nan=False),
                self.state,
            )
        self._fitted = True
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("R01 LinearSVR must converge successfully before prediction")
        selected = _select_features(X)
        documents, missing = _text_documents(selected["signal_stream"])
        if self.vectorizer is None:
            text_matrix = sparse.csr_matrix((len(selected), 0), dtype=np.float64)
        else:
            text_matrix = self.vectorizer.transform(documents).tocsr()
        structured, _ = RegressionTextModel._structured_transform(
            selected, self.processor, self.scaler, self.encoder, self.numeric_positions
        )
        missing_column = sparse.csr_matrix(missing.astype(np.float64).reshape(-1, 1))
        design = sparse.hstack((structured, text_matrix, missing_column),
                               format="csr", dtype=np.float64)
        design.sum_duplicates()
        design.sort_indices()
        if design.shape[1] != self.state["training_design_shape"][1] or not np.isfinite(design.data).all():
            raise RuntimeError("R01 score matrix differs from the fit schema or contains nonfinite values")
        prediction = np.asarray(self.estimator.predict(design), dtype=np.float64) * 50.0 + 50.0
        if prediction.shape != (len(selected),) or not np.isfinite(prediction).all():
            raise RuntimeError("R01 LinearSVR returned invalid target-scale predictions")
        return prediction

    def audit_state(self) -> dict[str, Any]:
        if not self._fitted:
            raise RuntimeError("R01 cannot expose fitted state before convergence")
        return json.loads(json.dumps(_safe_builtin(self.state), sort_keys=True, allow_nan=False))


def fit_predict(
    fit_features: pd.DataFrame,
    fit_target: np.ndarray,
    score_features: pd.DataFrame,
    *,
    controls: dict[str, np.ndarray],
    card_id: str,
    cutoff: int,
) -> dict[str, Any]:
    """Fit one temporal fold and apply only the frozen R_AUX25 policy."""
    if card_id != "R01":
        raise ValueError(f"R01 adapter cannot execute {card_id}")
    fit_x = _select_features(fit_features)
    score_x = _select_features(score_features)
    model = SparseLinearSVR()
    try:
        model.fit(fit_x, fit_target)
    except R01ConvergenceError as exc:
        exc.cutoff = int(cutoff)
        exc.fit_rows = int(len(fit_x))
        raise
    raw = model.predict(score_x)
    baseline = np.asarray(controls.get("baseline_prediction"), dtype=np.float64)
    cycles = pd.to_numeric(score_x["cycle_code"], errors="raise").to_numpy(dtype=np.int64)
    if baseline.shape != raw.shape or not np.isfinite(baseline).all() or np.any((baseline < 0) | (baseline > 100)):
        raise ValueError("R01 champion control is invalid or misaligned")
    candidate_rank50 = rank50_prediction(raw, cycles)
    prediction = np.clip(0.75 * baseline + 0.25 * candidate_rank50, 0.0, 100.0)
    return {
        "prediction": prediction,
        "raw_candidate": raw,
        "fit_predictor_count": 1,
        "state": model.audit_state(),
        "fit_metadata": {
            "cutoff": int(cutoff), "fit_rows": model.fit_rows,
            "n_iter": model.n_iter, "converged": model.converged,
            "target_transform": TARGET_TRANSFORM,
            "policy": ".75 frozen champion +.25 candidate rank50; clip[0,100]",
        },
        "audit_arrays": {"score_cycles": cycles},
    }


__all__ = [
    "FEATURE_NAMES", "LINEAR_SVR_PARAMETERS", "R01ConvergenceError",
    "SparseLinearSVR", "fit_predict", "rank50_prediction",
]
