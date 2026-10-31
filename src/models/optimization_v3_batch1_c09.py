"""C09: C2_B hybrid logistic regression with a reference-coded text x EF block.

All parsing, fit-only structured transforms, TF-IDF, and LogisticRegression
parameters remain owned by ``ClassificationModel``. This adapter only appends
``TF-IDF * I(entry_friction == 1)`` as a sparse block.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
import warnings

from src.models.classification_models import (
    ClassificationModel,
    _LR_PARAMS,
    _json_digest,
    _text_values,
)
from src.models.round1_models import _number_array


def entry_friction_one_mask(frame: pd.DataFrame) -> tuple[np.ndarray, dict[str, int]]:
    """Return the deterministic parsed ``entry_friction == 1`` indicator.

    Missing, invalid, and non-binary values do not activate the interaction;
    the unchanged C2_B structured block still represents the parsed/missing
    feature state. The official Train folds are expected to be binary/complete.
    """
    if not isinstance(frame, pd.DataFrame) or "entry_friction" not in frame:
        raise ValueError("entry_friction is required to build the C09 interaction")
    numeric, missing, invalid = _number_array(frame["entry_friction"])
    finite = np.isfinite(numeric)
    one = (numeric == 1.0) & (missing == 0.0) & (invalid == 0.0) & finite
    zero = (numeric == 0.0) & (missing == 0.0) & (invalid == 0.0) & finite
    audit = {
        "rows": int(len(frame)),
        "ef1_rows": int(one.sum()),
        "ef0_rows": int(zero.sum()),
        "missing_rows": int(missing.sum()),
        "invalid_rows": int(invalid.sum()),
        "other_finite_rows": int((finite & ~one & ~zero).sum()),
    }
    return one.astype(bool, copy=False), audit


def append_reference_interaction(
    base_design: Any,
    text_design: Any,
    ef1_mask: Any,
) -> tuple[sparse.csr_matrix, sparse.csr_matrix]:
    """Append the EF1-only text block without densifying or altering ``base``."""
    base = sparse.csr_matrix(base_design, dtype=np.float64)
    text = sparse.csr_matrix(text_design, dtype=np.float64)
    mask = np.asarray(ef1_mask, dtype=bool)
    if base.shape[0] != text.shape[0] or mask.shape != (base.shape[0],):
        raise ValueError("C09 base, text, and entry_friction rows must align")
    if not np.isfinite(base.data).all() or not np.isfinite(text.data).all():
        raise ValueError("C09 design blocks must be finite")
    interaction = text.multiply(mask.astype(np.float64).reshape(-1, 1)).tocsr()
    interaction.eliminate_zeros()
    extended = sparse.hstack((base, interaction), format="csr", dtype=np.float64)
    return extended, interaction


class C09TextEntryFrictionClassifier(ClassificationModel):
    """The original C2_B model plus one sparse TF-IDF x entry_friction block."""

    def __init__(self) -> None:
        super().__init__({"id": "C2_B"})
        self.model_id = "C09"
        self._c09_fit_ef1: np.ndarray | None = None
        self._c09_fit_ef_audit: dict[str, int] | None = None
        self._c09_design_state: dict[str, Any] = {}
        self._c09_fit_attempt: dict[str, Any] = {
            "attempted": False,
            "status": "NOT_STARTED",
            "convergence_warning_count": 0,
            "warning_categories": [],
        }

    def fit(self, X: pd.DataFrame, y: Any) -> "C09TextEntryFrictionClassifier":
        selected = self._select(X)
        self._c09_fit_ef1, self._c09_fit_ef_audit = entry_friction_one_mask(selected)
        try:
            super().fit(selected, y)
        finally:
            self._c09_fit_ef1 = None
        return self

    def _fit_logistic(self, matrix: Any, target: np.ndarray) -> None:
        if self.kind != "lr_hybrid":
            raise RuntimeError("C09 must retain the original C2_B hybrid LogisticRegression")
        if self._c09_fit_ef1 is None or self._c09_fit_ef_audit is None:
            raise RuntimeError("C09 fit-time entry_friction indicator is missing")
        base = sparse.csr_matrix(matrix, dtype=np.float64)
        text_width = int(self._transform_state.get("text_columns", -1))
        if text_width < 0 or not self._transform_state.get("stream_missing_indicator", False):
            raise RuntimeError("C09 requires the original C2_B text and stream-missing blocks")
        if self._c09_fit_ef1.shape != (base.shape[0],):
            raise RuntimeError("C09 entry_friction indicator is not aligned with fit rows")
        # C2_B block order is [structured, main TF-IDF, stream-missing]. The
        # interaction is strictly appended after every original C2_B column.
        text_start = base.shape[1] - text_width - 1
        if text_start < 0:
            raise RuntimeError("C09 could not locate the original TF-IDF block")
        text = base[:, text_start : text_start + text_width].tocsr()
        extended, interaction = append_reference_interaction(base, text, self._c09_fit_ef1)
        csr_bytes = lambda matrix: int(matrix.data.nbytes + matrix.indices.nbytes + matrix.indptr.nbytes)
        if interaction.nnz > text.nnz:
            raise RuntimeError("C09 interaction unexpectedly exceeds the main TF-IDF nonzero count")
        # Reproduce the parent's exact liblinear configuration and warning
        # failure, while retaining an auditable count (the parent swallows the
        # warning object before raising its RuntimeError).
        estimator = LogisticRegression(**_LR_PARAMS)
        self._c09_fit_attempt = {
            "attempted": True,
            "status": "RUNNING",
            "parameters": dict(_LR_PARAMS),
            "convergence_warning_count": 0,
            "warning_categories": [],
        }
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", ConvergenceWarning)
            try:
                estimator.fit(extended, target)
            except Exception as exc:
                self._c09_fit_attempt.update({
                    "status": "FIT_EXCEPTION",
                    "exception_type": type(exc).__name__,
                    "exception_message": str(exc),
                    "convergence_warning_count": sum(
                        issubclass(item.category, ConvergenceWarning) for item in caught
                    ),
                    "warning_categories": [item.category.__name__ for item in caught],
                })
                raise
        convergence_count = sum(issubclass(item.category, ConvergenceWarning) for item in caught)
        self._estimator = estimator
        self._c09_fit_attempt.update({
            "status": "CONVERGENCE_WARNING" if convergence_count else "FIT_SUCCESS",
            "convergence_warning_count": int(convergence_count),
            "warning_categories": [item.category.__name__ for item in caught],
            "n_iter": [int(value) for value in np.asarray(estimator.n_iter_).ravel()],
        })
        if convergence_count:
            raise RuntimeError("C09 LogisticRegression did not converge")
        self._c09_design_state = {
            "base_design_shape": [int(v) for v in base.shape],
            "extended_design_shape": [int(v) for v in extended.shape],
            "structured_width": int(text_start),
            "main_tfidf_width": int(text_width),
            "stream_missing_width": 1,
            "interaction_width": int(interaction.shape[1]),
            "base_nnz": int(base.nnz),
            "main_tfidf_nnz": int(text.nnz),
            "interaction_nnz": int(interaction.nnz),
            "extended_nnz": int(extended.nnz),
            "base_csr_storage_bytes": csr_bytes(base),
            "interaction_csr_storage_bytes": csr_bytes(interaction),
            "extended_csr_storage_bytes": csr_bytes(extended),
            "feature_order": ["C2_B structured", "main TF-IDF", "stream_missing", "TF-IDF x EF1"],
            "interaction_reference_level": "entry_friction=0",
            "interaction_sparse_csr": bool(sparse.isspmatrix_csr(interaction)),
            "entry_friction_fit": dict(self._c09_fit_ef_audit),
            "fit_attempt": dict(self._c09_fit_attempt),
        }

    def _make_audit_state(self) -> None:
        super()._make_audit_state()
        self._state["c09_interaction"] = {
            "recipe": "append TF-IDF * I(parsed entry_friction == 1)",
            "feature_order": list(self._c09_design_state.get("feature_order", [])),
            "reference_level": "entry_friction=0",
            "design": dict(self._c09_design_state),
            "entry_friction_fit": dict(self._c09_fit_ef_audit or {}),
            "uses_validation_labels": False,
            "uses_telemetry_flag_for_interaction": False,
            "fit_attempt": dict(self._c09_fit_attempt),
        }
        self._state["model_state_sha256"] = _json_digest(
            {key: value for key, value in self._state.items() if key != "model_state_sha256"}
        )

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("Classifier must be fitted before predict_proba")
        selected = self._select(X)
        self._diagnostics = {"n_predictions": int(len(selected))}
        structured = sparse.csr_matrix(self._transform_structured(selected), dtype=np.float64)
        text = sparse.csr_matrix(self._transform_text(selected["signal_stream"]), dtype=np.float64)
        _, missing = _text_values(selected["signal_stream"])
        missing_matrix = sparse.csr_matrix(missing.astype(np.float64).reshape(-1, 1))
        base = sparse.hstack((structured, text, missing_matrix), format="csr", dtype=np.float64)
        ef1, audit = entry_friction_one_mask(selected)
        extended, interaction = append_reference_interaction(base, text, ef1)
        expected_width = int(self._c09_design_state.get("extended_design_shape", [0, 0])[1])
        if extended.shape != (len(selected), expected_width):
            raise RuntimeError("C09 prediction design does not match the fitted schema")
        if not np.isfinite(extended.data).all():
            raise RuntimeError("C09 prediction design contains non-finite values")
        self._diagnostics["entry_friction_interaction"] = {
            **audit,
            "interaction_nnz": int(interaction.nnz),
            "extended_nnz": int(extended.nnz),
        }
        probabilities = np.asarray(self._estimator.predict_proba(extended)[:, 1], dtype=float)
        if probabilities.shape != (len(selected),) or not np.isfinite(probabilities).all():
            raise RuntimeError("C09 LogisticRegression returned invalid probabilities")
        if ((probabilities < 0.0) | (probabilities > 1.0)).any():
            raise RuntimeError("C09 LogisticRegression probabilities fall outside [0,1]")
        return probabilities


__all__ = [
    "C09TextEntryFrictionClassifier",
    "append_reference_interaction",
    "entry_friction_one_mask",
]
