"""Frozen Round 2 classification adapters built on the Round 1 wrappers.

This module changes only the pre-registered linear feature geometry. Input
parsing, structured preprocessing, text vectorization, class probabilities,
and convergence handling remain owned by ``classification_models``.
"""
from __future__ import annotations

import copy
import re
from typing import Any

import numpy as np
import pandas as pd
from scipy import sparse

from src.models.classification_models import (
    ClassificationModel,
    _json_digest,
    _text_values,
    build_classifier,
)
from src.models.round1_models import EXPANDED_FEATURES, TWO_FEATURES


_CONTROL_RECIPES = {"C1_A", "C1_B", "C2_A", "C2_B"}
_VARIANTS = {
    "C3_B": {"base": "C1_A", "variant": "structured_indicator", "weight": 1.0,
             "indicator": True},
    "C3_C": {"base": "C2_B", "variant": "hybrid_no_indicator", "weight": 1.0,
             "indicator": False},
    "C5_LOW": {"base": "C2_B", "variant": "hybrid_text_half", "weight": 0.5,
               "indicator": True},
    "C5_HIGH": {"base": "C2_B", "variant": "hybrid_text_double", "weight": 2.0,
                "indicator": True},
}


def stream_missing_mask(frame: pd.DataFrame | pd.Series) -> np.ndarray:
    """Return the Round 1 blank/missing semantics for ``signal_stream``."""
    if isinstance(frame, pd.DataFrame):
        if frame.columns.has_duplicates:
            raise ValueError("frame has duplicate feature columns")
        if "signal_stream" not in frame.columns:
            raise ValueError("frame must contain signal_stream")
        series = frame["signal_stream"]
    elif isinstance(frame, pd.Series):
        series = frame
    else:
        raise TypeError("frame must be a pandas DataFrame or Series")
    _, missing = _text_values(series)
    return missing.copy()


class _Round2Variant(ClassificationModel):
    """Single frozen adapter that modifies only the LR design matrix."""

    def __init__(self, recipe_id: str):
        config = _VARIANTS[recipe_id]
        super().__init__({"id": config["base"]})
        self.model_id = recipe_id
        self._round2_base_id = config["base"]
        self._round2_variant = config["variant"]
        self._round2_text_weight = float(config["weight"])
        self._round2_missing_indicator = bool(config["indicator"])
        if recipe_id == "C3_B":
            # Let the parent's normal fingerprinting bind the extra evidence
            # stream to the same fit rows as the structured features.
            self.input_feature_names = [*EXPANDED_FEATURES, "signal_stream"]
        self._round2_dimensions: dict[str, int] = {}
        self._round2_fit_missing: np.ndarray | None = None

    def fit(self, X: pd.DataFrame, y: Any) -> "_Round2Variant":
        self._round2_fit_missing = (
            stream_missing_mask(self._select(X)) if self._round2_missing_indicator else None
        )
        super().fit(X, y)
        self._round2_fit_missing = None
        return self

    def _fit_logistic(self, matrix: Any, target: np.ndarray) -> None:
        matrix = sparse.csr_matrix(matrix, dtype=np.float64)
        if self._round2_variant == "structured_indicator":
            if self._round2_fit_missing is None or len(self._round2_fit_missing) != matrix.shape[0]:
                raise RuntimeError("C3_B stream indicator is not aligned to the fit rows")
            structured_width = int(matrix.shape[1])
            indicator = sparse.csr_matrix(self._round2_fit_missing.astype(np.float64).reshape(-1, 1))
            matrix = sparse.hstack((matrix, indicator), format="csr", dtype=np.float64)
            self._round2_dimensions = {
                "structured": structured_width, "text": 0, "missing_indicator": 1,
                "total": int(matrix.shape[1]),
            }
        else:
            text_width = int(self._transform_state["text_columns"])
            full_width = int(matrix.shape[1])
            indicator_width = int(self._round2_missing_indicator)
            structured_width = full_width - text_width - 1
            if structured_width < 0:
                raise RuntimeError("Round 1 hybrid feature blocks do not reconcile")
            if self._round2_text_weight != 1.0 and text_width:
                left = matrix[:, :structured_width]
                text = matrix[:, structured_width:structured_width + text_width] * self._round2_text_weight
                right = matrix[:, structured_width + text_width:]
                matrix = sparse.hstack((left, text, right), format="csr", dtype=np.float64)
            if not self._round2_missing_indicator:
                matrix = matrix[:, :-1].tocsr()
            self._round2_dimensions = {
                "structured": structured_width, "text": text_width,
                "missing_indicator": indicator_width, "total": int(matrix.shape[1]),
            }

        self._transform_state.update({
            "structured_columns": self._round2_dimensions["structured"],
            "text_columns": self._round2_dimensions["text"],
            "stream_missing_indicator": bool(self._round2_dimensions["missing_indicator"]),
            "stream_missing_indicator_columns": self._round2_dimensions["missing_indicator"],
            "total_feature_count": self._round2_dimensions["total"],
            "text_block_weight": self._round2_text_weight,
        })
        super()._fit_logistic(matrix, target)

    def _make_audit_state(self) -> None:
        super()._make_audit_state()
        state = self._state
        state["round2_variant"] = {
            "id": self.model_id,
            "base_recipe": self._round2_base_id,
            "variant": self._round2_variant,
            "text_block_weight": self._round2_text_weight,
            "stream_missing_indicator": bool(self._round2_dimensions.get("missing_indicator", 0)),
            "block_dimensions": dict(self._round2_dimensions),
            "structured_transform_sha256": _structured_transform_fingerprint(self),
            "text_transform_sha256": _text_transform_fingerprint(self),
        }
        state["model_state_sha256"] = _json_digest(
            {key: value for key, value in state.items() if key != "model_state_sha256"}
        )

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("Classifier must be fitted before predict_proba")
        selected = self._select(X)
        self._diagnostics = {"n_predictions": int(len(selected))}
        structured = self._transform_structured(selected)
        if self._round2_variant == "structured_indicator":
            _, missing = _text_values(selected["signal_stream"])
            self._diagnostics["signal_stream_missing_rows"] = int(missing.sum())
            design = sparse.hstack((
                structured,
                sparse.csr_matrix(missing.astype(float).reshape(-1, 1)),
            ), format="csr", dtype=np.float64)
        else:
            text = self._transform_text(selected["signal_stream"])
            if self._round2_text_weight != 1.0:
                text = text * self._round2_text_weight
            blocks = [structured, text]
            if self._round2_missing_indicator:
                blocks.append(sparse.csr_matrix(
                    stream_missing_mask(selected).astype(float).reshape(-1, 1)
                ))
            design = sparse.hstack(blocks, format="csr", dtype=np.float64)
        probabilities = np.asarray(self._estimator.predict_proba(design)[:, 1], dtype=float)
        if probabilities.shape != (len(selected),) or not np.isfinite(probabilities).all():
            raise RuntimeError("Classifier returned invalid positive-class probabilities")
        if ((probabilities < 0) | (probabilities > 1)).any():
            raise RuntimeError("Classifier probabilities fall outside [0, 1]")
        return probabilities


def _structured_transform_fingerprint(model: ClassificationModel) -> str | None:
    processor = getattr(model, "_structured_preprocessor", None)
    if processor is None:
        return None
    state = model._state
    transformed = state.get("transformed_features", {})
    scaler = state.get("scaler", {})
    return _json_digest({
        "preprocessor": processor.audit_state(),
        "linear_feature_schema": {
            "numeric_names": list(model._linear_numeric_names),
            "category_names": list(model._category_names),
            "encoded_category_counts": transformed.get("encoded_category_counts", []),
            "encoded_category_sha256": transformed.get("encoded_category_sha256"),
            "infrequent_category_sha256": transformed.get("infrequent_category_sha256"),
        },
        "linear_scaler_sha256": {
            "mean": scaler.get("mean_sha256"),
            "scale": scaler.get("scale_sha256"),
        },
        "one_hot_encoder": state.get("parameters", {}).get("one_hot_encoder"),
    })


def _text_transform_fingerprint(model: ClassificationModel) -> str | None:
    if model.kind not in {"lr_hybrid", "lr_text"}:
        return None
    return _json_digest(model._text_state())


def build_round2_model(recipe: dict[str, Any]) -> ClassificationModel:
    """Build an unchanged Round 1 control or one fixed Round 2 adapter."""
    if not isinstance(recipe, dict):
        raise TypeError("recipe must be a dictionary")
    if recipe.get("kind") == "control":
        recipe_id = recipe.get("id")
        if recipe_id not in _CONTROL_RECIPES or recipe.get("source") != recipe_id:
            raise ValueError("control recipe id and source must match a frozen Round 1 recipe")
        if set(recipe) != {"id", "kind", "source"}:
            raise ValueError("control recipe accepts only id, kind, and source")
        return build_classifier({"id": recipe_id})
    if recipe.get("kind") == "variant":
        recipe_id = recipe.get("id")
        if recipe_id not in _VARIANTS or recipe.get("variant") != _VARIANTS[recipe_id]["variant"]:
            raise ValueError("unknown Round 2 variant or variant label mismatch")
        if set(recipe) != {"id", "kind", "variant"}:
            raise ValueError("variant recipe accepts only id, kind, and variant")
        return _Round2Variant(recipe_id)
    raise ValueError("recipe kind must be control or variant")


def _component_design(model: ClassificationModel, X: pd.DataFrame):
    """Prepare linear block matrices without retaining diagnostic side effects."""
    if not model._fitted:
        raise RuntimeError("Classifier must be fitted before linear_components")
    selected = model._select(X)
    previous_diagnostics = copy.deepcopy(model._diagnostics)
    try:
        structured = sparse.csr_matrix((len(selected), 0), dtype=np.float64)
        text = sparse.csr_matrix((len(selected), 0), dtype=np.float64)
        mask = np.zeros(len(selected), dtype=bool)
        if model.kind == "lr_two":
            numeric = sparse.csr_matrix(
                model._linear_scaler.transform(model._numeric_two(selected)), dtype=np.float64
            )
            categorical = sparse.csr_matrix((len(selected), 0), dtype=np.float64)
        else:
            if model.kind in {"lr_structured", "lr_hybrid"}:
                structured = sparse.csr_matrix(model._transform_structured(selected), dtype=np.float64)
                numeric_width = len(model._linear_numeric_names)
                numeric = structured[:, :numeric_width]
                categorical = structured[:, numeric_width:]
            else:
                numeric = sparse.csr_matrix((len(selected), 0), dtype=np.float64)
                categorical = sparse.csr_matrix((len(selected), 0), dtype=np.float64)

            if model.kind in {"lr_hybrid", "lr_text"}:
                text = model._transform_text(selected["signal_stream"])
                mask = stream_missing_mask(selected)
            is_round2 = isinstance(model, _Round2Variant)
            include_indicator = (model._round2_missing_indicator if is_round2 else
                                 model.kind in {"lr_hybrid", "lr_text"})
            weight = model._round2_text_weight if is_round2 else 1.0
            if model.kind == "lr_structured" and is_round2:
                mask = stream_missing_mask(selected)

        is_round2 = isinstance(model, _Round2Variant)
        include_indicator = (model._round2_missing_indicator if is_round2 else
                             model.kind in {"lr_hybrid", "lr_text"})
        weight = model._round2_text_weight if is_round2 else 1.0
        weighted_text = text * weight
        blocks = [structured, weighted_text]
        if include_indicator:
            blocks.append(sparse.csr_matrix(mask.astype(float).reshape(-1, 1)))
        design = numeric if model.kind == "lr_two" else sparse.hstack(
            blocks, format="csr", dtype=np.float64
        )
        return selected, numeric, categorical, text, mask, weight, include_indicator, design
    finally:
        model._diagnostics = previous_diagnostics


def linear_components(model: ClassificationModel, X: pd.DataFrame) -> dict[str, np.ndarray]:
    """Return per-row additive logit components for a fitted linear classifier."""
    if model.kind not in {"lr_two", "lr_structured", "lr_hybrid", "lr_text"}:
        return {}
    selected, numeric, categorical, text, mask, weight, include_indicator, design = _component_design(model, X)
    coefficients = np.asarray(model._estimator.coef_[0], dtype=float)
    if design.shape[1] != len(coefficients):
        raise RuntimeError("Linear component blocks do not match the fitted coefficient width")
    structured_width = numeric.shape[1] + categorical.shape[1]
    text_width = text.shape[1]
    intercept = np.full(len(selected), float(model._estimator.intercept_[0]), dtype=float)
    numeric_term = np.asarray(numeric.dot(coefficients[:numeric.shape[1]])).reshape(-1)
    cat_start = numeric.shape[1]
    categorical_term = np.asarray(
        categorical.dot(coefficients[cat_start:structured_width])
    ).reshape(-1)
    text_start = structured_width
    text_term = np.asarray(
        text.dot(coefficients[text_start:text_start + text_width] * weight)
    ).reshape(-1)
    missing_index = structured_width + text_width
    missing_term = np.zeros(len(selected), dtype=float)
    if include_indicator:
        missing_term = mask.astype(float) * coefficients[missing_index]
    reconstructed = intercept + numeric_term + categorical_term + text_term + missing_term
    direct = np.asarray(model._estimator.decision_function(design), dtype=float).reshape(-1)
    if not np.allclose(reconstructed, direct, rtol=1e-10, atol=1e-10):
        raise RuntimeError("Linear additive components do not reconstruct the fitted logits")
    return {
        "intercept": intercept,
        "numeric": numeric_term,
        "categorical": categorical_term,
        "text": text_term,
        "missing_indicator": missing_term,
    }


def coefficient_summary(model: ClassificationModel) -> dict[str, Any]:
    """Return safe coefficient norms and numeric-field coefficients only."""
    if not model._fitted:
        raise RuntimeError("Classifier must be fitted before coefficient_summary")
    if model.kind not in {"lr_two", "lr_structured", "lr_hybrid", "lr_text"}:
        return {"id": model.model_id, "kind": model.kind, "linear": False}

    coef = np.asarray(model._estimator.coef_[0], dtype=float)
    if model.kind == "lr_two":
        numeric_names = list(TWO_FEATURES)
        numeric_width = len(numeric_names)
        structured_width = numeric_width
        text_width = 0
    elif model.kind in {"lr_structured", "lr_hybrid"}:
        numeric_names = list(model._linear_numeric_names)
        numeric_width = len(numeric_names)
        if model.kind == "lr_structured":
            structured_width = len(coef) - int(
                isinstance(model, _Round2Variant) and model._round2_missing_indicator
            )
            text_width = 0
        else:
            is_round2 = isinstance(model, _Round2Variant)
            indicator = model._round2_missing_indicator if is_round2 else True
            text_width = int(model._text_state()["vocabulary_size"])
            structured_width = len(coef) - text_width - int(indicator)
    else:  # lr_text
        numeric_names = []
        numeric_width = 0
        text_width = int(model._text_state()["vocabulary_size"])
        structured_width = 0

    numeric_coefficients = {
        name: float(value)
        for name, value in zip(numeric_names, coef[:numeric_width])
        if re.fullmatch(r"[A-Za-z0-9_]+", str(name))
    }
    categorical = coef[numeric_width:structured_width]
    text = coef[structured_width:structured_width + text_width]
    variant = isinstance(model, _Round2Variant)
    weight = model._round2_text_weight if variant else 1.0
    indicator = model._round2_missing_indicator if variant else model.kind in {"lr_hybrid", "lr_text"}
    missing_coef = float(coef[-1]) if indicator else 0.0
    state = model.audit_state()
    if "model_state_sha256" in state:
        del state["model_state_sha256"]
    summary = {
        "id": model.model_id,
        "kind": model.kind,
        "linear": True,
        "model_C": float(model._estimator.C),
        "intercept": float(model._estimator.intercept_[0]),
        "numeric_coefficients": numeric_coefficients,
        "numeric_nonzero_count": int(np.count_nonzero(coef[:numeric_width])),
        "numeric_l2_norm": float(np.linalg.norm(coef[:numeric_width])),
        "categorical_dimension": int(len(categorical)),
        "categorical_nonzero_count": int(np.count_nonzero(categorical)),
        "categorical_l2_norm": float(np.linalg.norm(categorical)),
        "text_dimension": int(len(text)),
        "text_nonzero_count": int(np.count_nonzero(text)),
        "text_coefficient_l2_norm": float(np.linalg.norm(text)),
        "text_effective_original_basis_l2_norm": float(weight * np.linalg.norm(text)),
        "text_block_weight": float(weight),
        "stream_missing_indicator": bool(indicator),
        "stream_missing_indicator_coefficient": missing_coef,
        "stream_missing_indicator_nonzero_count": int(bool(indicator and missing_coef != 0.0)),
        "coefficient_l2_norm": float(np.linalg.norm(coef)),
        "structured_transform_sha256": _structured_transform_fingerprint(model),
        "text_transform_sha256": _text_transform_fingerprint(model),
        "fit_fingerprint_sha256": model._fit_fingerprint,
        "model_state_sha256": model.audit_state()["model_state_sha256"],
    }
    if variant:
        summary["block_dimensions"] = dict(model._round2_dimensions)
    else:
        summary["block_dimensions"] = {
            "structured": int(structured_width),
            "numeric": int(numeric_width),
            "categorical": int(len(categorical)),
            "text": int(text_width),
            "missing_indicator": int(indicator),
            "total": int(len(coef)),
        }
    return summary


__all__ = [
    "build_round2_model",
    "stream_missing_mask",
    "linear_components",
    "coefficient_summary",
]
