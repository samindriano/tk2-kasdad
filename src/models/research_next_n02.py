"""N02: C2_B main effects plus one fixed degree-two TensorSketch arm."""
from __future__ import annotations

import hashlib
import json
from typing import Any

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.kernel_approximation import PolynomialCountSketch
from sklearn.preprocessing import normalize

from src.models.classification_models import ClassificationModel, _LR_PARAMS, _text_values


CARD_ID = "N02"
SKETCH_PARAMS = {
    "degree": 2,
    "n_components": 1024,
    "gamma": 1.0,
    "coef0": 0.0,
    "random_state": 42,
}
BLEND_WEIGHT = 0.25
CURRENT_ROUTE_WEIGHT = 0.75
THRESHOLD = 0.50


def _array_sha256(values: Any) -> str:
    array = np.ascontiguousarray(np.asarray(values))
    return hashlib.sha256(array.view(np.uint8)).hexdigest()


def _sparse_audit(matrix: sparse.spmatrix) -> dict[str, Any]:
    csr = matrix.tocsr(copy=False)
    if csr.dtype != np.float64:
        raise TypeError(f"N02 sparse design must be float64, received {csr.dtype}")
    if not np.isfinite(csr.data).all():
        raise ValueError("N02 sparse design contains nonfinite data")
    return {
        "shape": [int(value) for value in csr.shape],
        "nnz": int(csr.nnz),
        "dtype": str(csr.dtype),
        "data_sha256": _array_sha256(csr.data),
        "indices_sha256": _array_sha256(csr.indices),
        "indptr_sha256": _array_sha256(csr.indptr),
    }


def _sketch_state(sketch: PolynomialCountSketch, *, fit_rows: int, input_features: int) -> dict[str, Any]:
    arrays = {}
    for name, value in sorted(vars(sketch).items()):
        if name.endswith("_") and isinstance(value, np.ndarray):
            arrays[name] = {
                "shape": [int(dimension) for dimension in value.shape],
                "dtype": str(value.dtype),
                "sha256": _array_sha256(value),
            }
    state = {
        "class": "sklearn.kernel_approximation.PolynomialCountSketch",
        "parameters": dict(SKETCH_PARAMS),
        "fit_input": "fit-fold C2_B main effects normalized on a copy, rowwise L2",
        "fit_rows": int(fit_rows),
        "input_features": int(input_features),
        "output_components": int(SKETCH_PARAMS["n_components"]),
        "fitted_random_arrays": arrays,
        "fit_rows_only": True,
    }
    state["state_sha256"] = hashlib.sha256(
        json.dumps(state, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    ).hexdigest()
    return state


class C2BInteractionClassifier:
    """Fit a C2_B-compatible Logistic Regression head with optional TensorSketch."""

    def __init__(self, *, include_sketch: bool):
        self.include_sketch = bool(include_sketch)
        self._base: ClassificationModel | None = None
        self._sketch: PolynomialCountSketch | None = None
        self._fit_main_effects: sparse.csr_matrix | None = None
        self._design_shape: tuple[int, int] | None = None
        self._state: dict[str, Any] = {}
        self._fitted = False

    @staticmethod
    def _initialize_c2b_design(
        X: pd.DataFrame, y: Any
    ) -> tuple[ClassificationModel, sparse.csr_matrix, np.ndarray]:
        """Reuse C2_B's exact fit-only transforms without fitting its head."""
        base = ClassificationModel({"id": "C2_B"})
        selected = base._select(X).copy()
        target = base._target(y, len(selected), require_both=True)
        base._fit_rows = int(len(target))
        from src.models.round1_models import _fit_fingerprint

        base._fit_fingerprint = _fit_fingerprint(selected, tuple(base.input_feature_names), target)
        base._positive_rate = float(target.mean())
        base._estimator = None
        base._structured_preprocessor = None
        base._text_vectorizer = None
        base._linear_scaler = None
        base._linear_encoder = None
        base._lookup_rates = {}
        base._lookup_counts = {}
        base._fallback_all_empty = False
        base._fallback_vocab_empty = False
        base._transform_state = {}
        base._hgb_categorical_mask = []
        base._native_vocabulary_hashes = {}

        structured = base._fit_structured(selected)
        text = base._fit_text(selected["signal_stream"])
        _documents, missing = _text_values(selected["signal_stream"])
        missing_matrix = sparse.csr_matrix(missing.astype(np.float64).reshape(-1, 1))
        main_effects = sparse.hstack((structured, text, missing_matrix), format="csr", dtype=np.float64)
        base._transform_state["structured_columns"] = int(main_effects.shape[1] - text.shape[1])
        base._transform_state["text_columns"] = int(text.shape[1])
        base._transform_state["stream_missing_indicator"] = True
        if (main_effects.shape[0] != len(target) or main_effects.dtype != np.float64
                or not np.isfinite(main_effects.data).all()):
            raise RuntimeError("N02 exact C2_B fit matrix is invalid")
        return base, main_effects, target

    def _transform_c2b_design(self, X: pd.DataFrame) -> sparse.csr_matrix:
        if self._base is None:
            raise RuntimeError("N02 C2_B transforms have not been fitted")
        selected = self._base._select(X)
        structured = self._base._transform_structured(selected)
        text = self._base._transform_text(selected["signal_stream"])
        _documents, missing = _text_values(selected["signal_stream"])
        missing_matrix = sparse.csr_matrix(missing.astype(np.float64).reshape(-1, 1))
        matrix = sparse.hstack((structured, text, missing_matrix), format="csr", dtype=np.float64)
        if matrix.shape[1] != self._fit_main_effects.shape[1]:
            raise RuntimeError("N02 transformed C2_B main-effect schema differs from fit")
        if matrix.dtype != np.float64 or not np.isfinite(matrix.data).all():
            raise RuntimeError("N02 transformed C2_B design is not finite float64")
        return matrix

    def _compose_design(self, main_effects: sparse.csr_matrix, *, fit: bool) -> sparse.csr_matrix:
        if not self.include_sketch:
            return main_effects.tocsr(copy=False)
        if fit:
            self._sketch = PolynomialCountSketch(**dict(SKETCH_PARAMS))
            sketch_values = self._sketch.fit_transform(
                normalize(main_effects.copy(), norm="l2", axis=1, copy=False)
            )
        else:
            if self._sketch is None:
                raise RuntimeError("N02 sketch transformer was not fitted")
            sketch_values = self._sketch.transform(
                normalize(main_effects.copy(), norm="l2", axis=1, copy=False)
            )
        dense = np.asarray(sketch_values, dtype=np.float64)
        if (dense.ndim != 2 or dense.shape != (main_effects.shape[0], SKETCH_PARAMS["n_components"])
                or not np.isfinite(dense).all()):
            raise RuntimeError("N02 PolynomialCountSketch returned an invalid dense sketch")
        sketch_sparse = sparse.csr_matrix(dense, dtype=np.float64)
        composed = sparse.hstack((main_effects, sketch_sparse), format="csr", dtype=np.float64)
        if composed.shape[0] != main_effects.shape[0] or not np.isfinite(composed.data).all():
            raise RuntimeError("N02 augmented design is invalid")
        return composed

    def fit(self, X: pd.DataFrame, y: Any) -> "C2BInteractionClassifier":
        self._fitted = False
        base, main_effects, target = self._initialize_c2b_design(X, y)
        self._base = base
        self._fit_main_effects = main_effects
        design = self._compose_design(main_effects, fit=True)
        base._fit_logistic(design, target)
        base._fitted = True
        base._make_audit_state()
        self._design_shape = tuple(int(value) for value in design.shape)
        self._state = {
            "id": "N02_C2B_TENSOR_SKETCH" if self.include_sketch else "N02_C2B_NO_SKETCH_CONTROL",
            "card_id": CARD_ID,
            "include_sketch": self.include_sketch,
            "fit_rows": int(len(target)),
            "input_features": list(base.input_feature_names),
            "main_effect_design": _sparse_audit(main_effects),
            "estimator_design": _sparse_audit(design),
            "head_parameters": dict(_LR_PARAMS),
            "main_effect_state": base.audit_state(),
            "sketch_state": (_sketch_state(self._sketch, fit_rows=len(target),
                                            input_features=main_effects.shape[1])
                             if self._sketch is not None else None),
            "preprocessing_fit_rows_only": True,
            "validation_rows_or_labels_passed_to_fit": False,
            "test_data_used": False,
        }
        self._state["state_sha256"] = hashlib.sha256(
            json.dumps(self._state, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
        ).hexdigest()
        self._fitted = True
        return self

    def transform_design(self, X: pd.DataFrame) -> sparse.csr_matrix:
        if not self._fitted:
            raise RuntimeError("N02 model must be fitted before transform")
        main_effects = self._transform_c2b_design(X)
        return self._compose_design(main_effects, fit=False)

    def predict_probability(self, X: pd.DataFrame) -> np.ndarray:
        if not self._fitted or self._base is None:
            raise RuntimeError("N02 model must be fitted before prediction")
        design = self.transform_design(X)
        probability = np.asarray(self._base._estimator.predict_proba(design)[:, 1], dtype=np.float64)
        if (probability.shape != (len(X),) or not np.isfinite(probability).all()
                or np.any((probability < 0) | (probability > 1))):
            raise RuntimeError("N02 returned invalid positive-class probabilities")
        return probability

    def audit_state(self) -> dict[str, Any]:
        if not self._fitted:
            raise RuntimeError("N02 model must be fitted before audit")
        return json.loads(json.dumps(self._state, sort_keys=True, allow_nan=False))


def route_expert_probabilities(
    raw_candidate: Any,
    raw_no_sketch: Any,
    current_route: Any,
    cb_probability: Any,
    stream_missing: Any,
) -> dict[str, np.ndarray]:
    """Apply the queue's fixed .75 current-route + .25 expert policy."""
    arrays = [np.asarray(value) for value in
              (raw_candidate, raw_no_sketch, current_route, cb_probability, stream_missing)]
    n = len(arrays[0])
    if n == 0 or any(array.shape != (n,) for array in arrays):
        raise ValueError("N02 route inputs must be nonempty and row-aligned")
    candidate_raw, control_raw, current, cb = [array.astype(np.float64) for array in arrays[:4]]
    missing = arrays[4].astype(bool)
    if any(not np.isfinite(array).all() for array in (candidate_raw, control_raw, current, cb)):
        raise ValueError("N02 route inputs must be finite")
    if any(np.any((array < 0) | (array > 1)) for array in (candidate_raw, control_raw, current, cb)):
        raise ValueError("N02 route probabilities must lie in [0,1]")
    candidate = np.where(missing, cb, CURRENT_ROUTE_WEIGHT * current + BLEND_WEIGHT * candidate_raw)
    no_sketch = np.where(missing, cb, CURRENT_ROUTE_WEIGHT * current + BLEND_WEIGHT * control_raw)
    baseline = np.where(missing, cb, current)
    return {"candidate": candidate, "no_sketch_control": no_sketch,
            "current_route": baseline, "stream_missing": missing}


__all__ = [
    "BLEND_WEIGHT", "CARD_ID", "CURRENT_ROUTE_WEIGHT", "C2BInteractionClassifier",
    "SKETCH_PARAMS", "THRESHOLD", "route_expert_probabilities",
]
