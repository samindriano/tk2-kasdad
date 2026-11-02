"""Frozen sparse MLP adapter for the Classification Research Next pilot.

This module reuses the existing fit-only C2_B transforms and applies one fixed
sklearn MLPClassifier head. It never loads data or chooses a model recipe.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.neural_network import MLPClassifier

from src.models.research_next_n02 import C2BInteractionClassifier


ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = ROOT / "experiments/optimization_v3/classification_mlp_v1/protocol.json"


def read_protocol() -> dict[str, Any]:
    return json.loads(PROTOCOL.read_text(encoding="utf-8"))


def build_fit_design(X: pd.DataFrame, y: Any) -> tuple[Any, sparse.csr_matrix, np.ndarray]:
    """Fit only C2_B preprocessing on X/y; deliberately do not fit a C2_B head."""
    base, design, target = C2BInteractionClassifier._initialize_c2b_design(X, y)
    if not sparse.isspmatrix_csr(design):
        raise RuntimeError("C2_B fit design must remain CSR")
    if design.dtype != np.float64 or not np.isfinite(design.data).all():
        raise RuntimeError("C2_B fit design must be finite float64 CSR")
    return base, design, target


def transform_score_design(base: Any, X: pd.DataFrame) -> sparse.csr_matrix:
    """Transform score rows with the already fitted C2_B preprocessing state."""
    selected = base._select(X)
    structured = base._transform_structured(selected)
    text = base._transform_text(selected["signal_stream"])
    from src.models.classification_models import _text_values

    _documents, missing = _text_values(selected["signal_stream"])
    missing_matrix = sparse.csr_matrix(missing.astype(np.float64).reshape(-1, 1))
    matrix = sparse.hstack((structured, text, missing_matrix), format="csr", dtype=np.float64)
    if matrix.shape[1] != base._transform_state["structured_columns"] + base._transform_state["text_columns"]:
        raise RuntimeError("C2_B score design feature count differs from fit state")
    if matrix.dtype != np.float64 or not np.isfinite(matrix.data).all():
        raise RuntimeError("C2_B score design must be finite float64 CSR")
    return matrix


def make_estimator(protocol: Mapping[str, Any] | None = None) -> MLPClassifier:
    """Return the exact estimator declared in the frozen JSON protocol."""
    protocol = read_protocol() if protocol is None else dict(protocol)
    model = protocol.get("model", {})
    if model.get("class") != "sklearn.neural_network.MLPClassifier":
        raise ValueError("Frozen protocol does not name sklearn MLPClassifier")
    parameters = model.get("parameters")
    if not isinstance(parameters, dict):
        raise ValueError("Frozen MLP parameter mapping is missing")
    return MLPClassifier(**parameters)


def route_probabilities(
    candidate_kind: str,
    baseline_sparse25: Any,
    raw_mlp: Any,
    c2b_probability: Any,
    cb_probability: Any,
    stream_missing: Any,
) -> np.ndarray:
    """Apply the frozen candidate or zero-fit matched allocation policy."""
    baseline = np.asarray(baseline_sparse25, dtype=np.float64)
    mlp = np.asarray(raw_mlp, dtype=np.float64)
    c2b = np.asarray(c2b_probability, dtype=np.float64)
    cb = np.asarray(cb_probability, dtype=np.float64)
    missing = np.asarray(stream_missing, dtype=bool)
    if not (baseline.ndim == mlp.ndim == c2b.ndim == cb.ndim == missing.ndim == 1):
        raise ValueError("All route inputs must be vectors")
    if not (baseline.shape == mlp.shape == c2b.shape == cb.shape == missing.shape):
        raise ValueError("All route inputs must have identical shape")
    for name, values in (("sparse25", baseline), ("MLP", mlp), ("C2_B", c2b), ("CB", cb)):
        if not np.isfinite(values).all() or np.any((values < 0) | (values > 1)):
            raise ValueError(f"{name} probabilities must be finite and in [0, 1]")
    if candidate_kind == "candidate":
        present = 0.75 * baseline + 0.25 * mlp
    elif candidate_kind == "matched_control":
        present = 0.75 * baseline + 0.25 * c2b
    elif candidate_kind == "sparse25":
        present = baseline.copy()
    elif candidate_kind == "c2b_diagnostic":
        present = c2b.copy()
    else:
        raise ValueError(f"Unknown route policy: {candidate_kind}")
    routed = np.where(missing, cb, present)
    if not np.isfinite(routed).all() or np.any((routed < 0) | (routed > 1)):
        raise RuntimeError("Routed probability is invalid")
    return routed.astype(np.float64, copy=False)


def sparse_nbytes(matrix: sparse.csr_matrix) -> int:
    matrix = matrix.tocsr(copy=False)
    return int(matrix.data.nbytes + matrix.indices.nbytes + matrix.indptr.nbytes)


def estimate_peak_bytes(fit_matrix: sparse.csr_matrix, score_matrix: sparse.csr_matrix,
                        *, hidden_units: int = 32, batch_size: int = 256) -> dict[str, int]:
    """Conservative bounded estimate for CSR, Adam state, and dense activations."""
    n_fit, n_features = fit_matrix.shape
    n_score = score_matrix.shape[0]
    parameter_count = (int(n_features) + 1) * int(hidden_units) + (int(hidden_units) + 1) * 2
    parameter_bytes = parameter_count * np.dtype(np.float64).itemsize
    # Parameters, gradients, and both Adam moments plus a copy for fitting state.
    optimizer_state_bytes = 5 * parameter_bytes
    hidden_activation_bytes = 6 * max(int(batch_size), 1) * int(hidden_units) * 8
    score_probability_bytes = int(n_score) * 2 * 8
    csr_bytes = sparse_nbytes(fit_matrix) + sparse_nbytes(score_matrix)
    dataframe_allowance = (int(n_fit) + int(n_score)) * 13 * 256
    # Include three CSR-equivalents for transient sparse products/slices and a
    # fixed 1 GiB process/library allowance; no TF-IDF densification is used.
    estimated = 3 * csr_bytes + optimizer_state_bytes + hidden_activation_bytes + score_probability_bytes + dataframe_allowance + 1024**3
    return {
        "fit_csr_bytes": sparse_nbytes(fit_matrix),
        "score_csr_bytes": sparse_nbytes(score_matrix),
        "model_parameter_count": int(parameter_count),
        "optimizer_state_bytes": int(optimizer_state_bytes),
        "hidden_activation_bytes": int(hidden_activation_bytes),
        "score_probability_bytes": int(score_probability_bytes),
        "dataframe_allowance_bytes": int(dataframe_allowance),
        "fixed_runtime_allowance_bytes": 1024**3,
        "estimated_additional_peak_bytes": int(estimated),
    }


__all__ = [
    "PROTOCOL", "ROOT", "build_fit_design", "estimate_peak_bytes", "make_estimator",
    "read_protocol", "route_probabilities", "sparse_nbytes", "transform_score_design",
]
