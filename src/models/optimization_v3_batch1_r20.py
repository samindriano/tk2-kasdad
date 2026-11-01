"""R20: fold-internal top-128 token identities with sparse LightGBM L1.

Every learned transform and the estimator are fit only on the temporal fit
rows. The adapter receives no validation labels and does not load data.
"""
from __future__ import annotations

import hashlib
import warnings
from collections import Counter, defaultdict
from typing import Any

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.stats import rankdata
from sklearn.feature_extraction.text import TfidfVectorizer
import lightgbm
from lightgbm import LGBMRegressor

from src.models.optimization_v2_regression import SPARSE_LGB_PARAMETERS
from src.models.regression_research_text import (
    HYBRID_FEATURES,
    RegressionTextModel,
    _array_digest,
    _json_digest,
    _text_documents,
    _target,
)


CARD_ID = "R20"
FEATURE_NAMES = HYBRID_FEATURES
PILOT_CUTOFFS = (55, 67)
LIGHTGBM_PARAMETERS = dict(SPARSE_LGB_PARAMETERS)
MIN_TOKEN_SUPPORT = 30
TOKEN_SMOOTHING = 30.0
SELECTED_TOKEN_COUNT = 128
TOKEN_ANALYZER_PARAMETERS = {
    "analyzer": "word", "ngram_range": (1, 1), "lowercase": True,
    "strip_accents": None, "token_pattern": r"(?u)\b\w\w+\b",
}
_CONTROL_KEYS = (
    "baseline_prediction",
    "baseline_lgb_svd32",
    "baseline_hybrid_ridge",
    "baseline_sparse_raw",
    "baseline_sparse_rank50",
    "control_rows",
)


def _select_exact(X: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(X, pd.DataFrame):
        raise TypeError("R20 inputs must be pandas DataFrames")
    if X.columns.has_duplicates or set(X.columns) != set(FEATURE_NAMES) or len(X.columns) != len(FEATURE_NAMES):
        raise ValueError("R20 inputs must contain exactly the frozen hybrid feature allowlist")
    if set(FEATURE_NAMES) & {"record_id", "artifact_signal", "threat_rank"}:
        raise ValueError("R20 feature allowlist contains an identity or target")
    return X.loc[:, list(FEATURE_NAMES)].copy()


def rank50_prediction(raw_prediction: Any, cycle_codes: Any) -> np.ndarray:
    """Map target-scale predictions to the frozen within-cycle rank50 scale."""
    raw = np.asarray(raw_prediction, dtype=np.float64)
    cycles = np.asarray(cycle_codes)
    if raw.ndim != 1 or cycles.shape != raw.shape or raw.size == 0:
        raise ValueError("R20 rank50 inputs must be nonempty aligned vectors")
    if not np.isfinite(raw).all() or cycles.dtype.kind not in "iu":
        raise ValueError("R20 rank50 needs finite predictions and integer cycle codes")
    mapped = np.empty(raw.shape, dtype=np.float64)
    for cycle in np.unique(cycles):
        mask = cycles == cycle
        percentile = 100.0 * rankdata(raw[mask], method="average") / int(mask.sum())
        mapped[mask] = 0.5 * raw[mask] + 0.5 * percentile
    mapped = np.clip(mapped, 0.0, 100.0)
    if not np.isfinite(mapped).all():
        raise ValueError("R20 rank50 mapping returned nonfinite values")
    return mapped


def reconstruct_r20(raw_prediction: Any, baseline_prediction: Any,
                    cycle_codes: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Rebuild the candidate and mapped component from raw score arrays."""
    raw = np.asarray(raw_prediction, dtype=np.float64)
    baseline = np.asarray(baseline_prediction, dtype=np.float64)
    if raw.shape != baseline.shape or raw.ndim != 1 or raw.size == 0:
        raise ValueError("R20 raw and baseline predictions must be aligned vectors")
    if not np.isfinite(baseline).all() or np.any((baseline < 0.0) | (baseline > 100.0)):
        raise ValueError("R20 baseline predictions must be finite in [0,100]")
    mapped = rank50_prediction(raw, cycle_codes)
    candidate = np.clip(0.75 * baseline + 0.25 * mapped, 0.0, 100.0)
    return candidate, baseline.copy(), mapped


def _cycle_codes(features: pd.DataFrame, *, label: str) -> np.ndarray:
    values = pd.to_numeric(features["cycle_code"], errors="raise").to_numpy(dtype=np.float64)
    if values.ndim != 1 or not np.isfinite(values).all() or not np.equal(values, np.floor(values)).all():
        raise ValueError(f"{label} cycle_code must be finite integer values")
    cycles = values.astype(np.int64)
    if np.any((cycles < 47) | (cycles > 79)):
        raise ValueError(f"{label} cycle_code is outside the frozen Train range 47-79")
    return cycles


def _token_analyzer():
    return TfidfVectorizer(**TOKEN_ANALYZER_PARAMETERS).build_analyzer()


def _document_token_sets(docs: list[str]) -> list[tuple[str, ...]]:
    analyzer = _token_analyzer()
    return [tuple(sorted(set(analyzer(str(doc))))) for doc in docs]


def eligible_token_support_count(docs: list[str]) -> dict[str, int]:
    document_tokens = _document_token_sets(docs)
    support: Counter[str] = Counter()
    for tokens in document_tokens:
        support.update(set(tokens))
    return {
        "fit_rows": int(len(document_tokens)),
        "vocabulary_size": int(len(support)),
        "support_ge_minimum": int(sum(n >= MIN_TOKEN_SUPPORT for n in support.values())),
        "minimum_document_support": MIN_TOKEN_SUPPORT,
    }


def select_token_identities(
    document_tokens: list[tuple[str, ...]], target: Any,
) -> tuple[list[str], dict[str, Any]]:
    values = np.asarray(target, dtype=np.float64)
    if values.ndim != 1 or len(values) != len(document_tokens) or values.size == 0:
        raise ValueError("R20 selection needs aligned nonempty fit documents and targets")
    if not np.isfinite(values).all() or np.any((values < 0.0) | (values > 100.0)):
        raise ValueError("R20 selection targets must be finite threat_rank values in [0,100]")
    support: Counter[str] = Counter()
    sums: defaultdict[str, float] = defaultdict(float)
    for tokens, value in zip(document_tokens, values):
        for token in set(tokens):
            support[token] += 1
            sums[token] += float(value)
    fit_mean = float(values.mean())
    eligible = []
    for token, n in support.items():
        if n < MIN_TOKEN_SUPPORT:
            continue
        token_mean = sums[token] / n
        score = abs((n / (n + TOKEN_SMOOTHING)) * (token_mean - fit_mean))
        eligible.append((token, int(n), float(token_mean), float(score)))
    ranked = sorted(eligible, key=lambda item: (-item[3], item[0]))
    if len(ranked) < SELECTED_TOKEN_COUNT:
        raise ValueError(
            f"R20 requires {SELECTED_TOKEN_COUNT} eligible tokens with support >= "
            f"{MIN_TOKEN_SUPPORT}; found {len(ranked)}"
        )
    selected_records = ranked[:SELECTED_TOKEN_COUNT]
    selected_tokens = [record[0] for record in selected_records]
    audit = {
        "fit_mean": fit_mean,
        "fit_document_count": int(values.size),
        "eligible_token_count": int(len(ranked)),
        "minimum_document_support": MIN_TOKEN_SUPPORT,
        "smoothing": TOKEN_SMOOTHING,
        "selected_token_count": SELECTED_TOKEN_COUNT,
        "ordering": "descending absolute shrunken mean deviation; exact ties by Unicode lexical ascending",
        "selected_tokens_sha256": _json_digest(selected_tokens),
        "selected_token_records": [
            {"token": token, "support": n, "token_mean": mean, "selection_score": score}
            for token, n, mean, score in selected_records
        ],
    }
    return selected_tokens, audit


def _presence_matrix(
    document_tokens: list[tuple[str, ...]], selected_tokens: list[str],
) -> sparse.csr_matrix:
    if len(selected_tokens) != SELECTED_TOKEN_COUNT or len(set(selected_tokens)) != SELECTED_TOKEN_COUNT:
        raise ValueError("R20 selected token identities must be exactly 128 unique columns")
    lookup = {token: column for column, token in enumerate(selected_tokens)}
    indices: list[int] = []
    indptr = [0]
    for tokens in document_tokens:
        columns = sorted({lookup[token] for token in tokens if token in lookup})
        indices.extend(columns)
        indptr.append(len(indices))
    index_array = np.asarray(indices, dtype=np.int32)
    pointer_array = np.asarray(indptr, dtype=np.int64)
    data = np.ones(len(index_array), dtype=np.float64)
    matrix = sparse.csr_matrix(
        (data, index_array, pointer_array),
        shape=(len(document_tokens), SELECTED_TOKEN_COUNT),
        dtype=np.float64,
    )
    matrix.sort_indices()
    if matrix.shape != (len(document_tokens), SELECTED_TOKEN_COUNT):
        raise RuntimeError("R20 token-presence matrix changed its fixed schema")
    return matrix


def _csr_digest(matrix: sparse.csr_matrix) -> str:
    csr = matrix.tocsr(copy=False)
    digest = hashlib.sha256()
    digest.update(np.asarray(csr.shape, dtype="<i8").tobytes())
    digest.update(np.asarray(csr.indptr, dtype="<i8").tobytes())
    digest.update(np.asarray(csr.indices, dtype="<i8").tobytes())
    digest.update(np.asarray(csr.data, dtype="<f8").tobytes())
    return digest.hexdigest()


class TokenIdentitySparseL1Regressor:
    """Sparse LightGBM L1 with exactly 128 identities selected inside the outer fit."""

    def __init__(self, cutoff: int):
        if int(cutoff) not in PILOT_CUTOFFS:
            raise ValueError(f"R20 cutoff must be one of {PILOT_CUTOFFS}")
        self.cutoff = int(cutoff)
        self._fitted = False
        self._fit_warnings: list[dict[str, str]] = []
        self._state: dict[str, Any] = {}
        self._diagnostics: dict[str, Any] = {}

    def fit(self, X: pd.DataFrame, y: Any) -> "TokenIdentitySparseL1Regressor":
        self._fitted = False
        selected = _select_exact(X)
        target = _target(y, len(selected))
        if np.any((target < 0.0) | (target > 100.0)):
            raise ValueError("R20 target must be threat_rank in [0,100]")
        cycles = _cycle_codes(selected, label="R20 fit features")
        if np.any(cycles > self.cutoff):
            raise ValueError("R20 fit features contain a cycle after the model cutoff")
        if not selected.index.is_unique or not selected.index.is_monotonic_increasing:
            raise ValueError("R20 fit rows must retain unique increasing source positions")

        docs, missing = _text_documents(selected["signal_stream"])
        document_tokens = _document_token_sets(docs)
        self._selected_tokens, selection = select_token_identities(document_tokens, target)
        token_matrix = _presence_matrix(document_tokens, self._selected_tokens)

        structured, self._processor, self._scaler, self._encoder, self._numeric_positions, structured_state = (
            RegressionTextModel._structured_fit(selected)
        )
        missing_column = sparse.csr_matrix(missing.astype(np.float64).reshape(-1, 1))
        design = sparse.hstack((structured, token_matrix, missing_column), format="csr", dtype=np.float64)
        if design.shape[0] != len(target) or not np.isfinite(design.data).all():
            raise RuntimeError("R20 sparse design is misaligned or nonfinite")

        self._estimator = LGBMRegressor(**dict(LIGHTGBM_PARAMETERS))
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            self._estimator.fit(design, target)
        self._fit_warnings = [
            {"category": item.category.__name__, "message": str(item.message)} for item in caught
        ]
        effective = dict(self._estimator.get_params(deep=False))
        structured_names = list(self._processor.transformed_feature_names)
        token_feature_names = [f"signal_token_presence::{token}" for token in self._selected_tokens]
        self._state = {
            "card_id": CARD_ID,
            "estimator": "lightgbm.LGBMRegressor",
            "lightgbm_version": lightgbm.__version__,
            "parameters_explicit": dict(LIGHTGBM_PARAMETERS),
            "parameters_effective": effective,
            "objective": "regression_l1",
            "fit_rows": int(len(target)),
            "fit_cycle_min": int(cycles.min()),
            "fit_cycle_max": int(cycles.max()),
            "fit_source_rows_sha256": _array_digest(selected.index.to_numpy(dtype="<i8"), dtype="<i8"),
            "fit_data_sha256": _json_digest({
                "features": [str(x) for x in selected.columns],
                "row_positions": selected.index.to_numpy(dtype="<i8").tolist(),
                "target_sha256": _array_digest(target),
            }),
            "target_sha256": _array_digest(target),
            "feature_names": list(FEATURE_NAMES),
            "selection": selection,
            "token_analyzer_parameters": dict(TOKEN_ANALYZER_PARAMETERS),
            "structured_preprocessing": structured_state,
            "missing_stream_indicator": True,
            "sparse_design": True,
            "transformed_feature_names": structured_names + token_feature_names + ["signal_stream_missing"],
            "training_design_shape": [int(design.shape[0]), int(design.shape[1])],
            "training_design_nnz": int(design.nnz),
            "training_design_sha256": _csr_digest(design),
            "fit_missing_stream_rows": int(missing.sum()),
            "validation_data_passed_to_fit": False,
            "validation_labels_passed_to_fit": False,
            "test_data_used_for_fit": False,
            "early_stopping_used": False,
            "fit_warnings": list(self._fit_warnings),
        }
        self._state["state_sha256"] = _json_digest(self._state)
        self._fitted = True
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("R20 model must be fitted before predict")
        selected = _select_exact(X)
        structured, unknown = RegressionTextModel._structured_transform(
            selected, self._processor, self._scaler, self._encoder, self._numeric_positions
        )
        docs, missing = _text_documents(selected["signal_stream"])
        document_tokens = _document_token_sets(docs)
        token_matrix = _presence_matrix(document_tokens, self._selected_tokens)
        missing_column = sparse.csr_matrix(missing.astype(np.float64).reshape(-1, 1))
        design = sparse.hstack((structured, token_matrix, missing_column), format="csr", dtype=np.float64)
        if (design.shape[1] != int(self._state["training_design_shape"][1])
                or not np.isfinite(design.data).all()):
            raise RuntimeError("R20 score design is nonfinite or has a different fixed schema")
        prediction = np.asarray(self._estimator.booster_.predict(design), dtype=np.float64)
        if prediction.shape != (len(selected),) or not np.isfinite(prediction).all():
            raise RuntimeError("R20 LightGBM predictions are misaligned or nonfinite")
        self._diagnostics = {
            "n_predictions": int(len(prediction)),
            "signal_stream_missing_rows": int(missing.sum()),
            "selected_token_presence_nnz": int(token_matrix.nnz),
            "unknown_category_rows": dict(sorted(unknown.items())),
        }
        return prediction

    def audit_state(self) -> dict[str, Any]:
        if not self._fitted:
            raise RuntimeError("R20 model must be fitted before audit_state")
        return dict(self._state)

    def prediction_diagnostics(self) -> dict[str, Any]:
        return dict(self._diagnostics)


def _validated_controls(controls: dict[str, Any], n_score: int) -> tuple[dict[str, np.ndarray], np.ndarray]:
    missing = [name for name in _CONTROL_KEYS if name not in controls]
    if missing or n_score <= 0:
        raise ValueError(f"R20 controls are missing fields or empty: {missing}")
    arrays: dict[str, np.ndarray] = {}
    for name in _CONTROL_KEYS[:-1]:
        values = np.asarray(controls[name], dtype=np.float64)
        if values.shape != (n_score,) or not np.isfinite(values).all():
            raise ValueError(f"R20 control {name} is misaligned or nonfinite")
        if name != "baseline_sparse_raw" and np.any((values < 0.0) | (values > 100.0)):
            raise ValueError(f"R20 control {name} is outside [0,100]")
        arrays[name] = values
    rows = np.asarray(controls["control_rows"])
    if (rows.ndim != 1 or rows.shape != (n_score,) or rows.dtype.kind not in "iu"
            or len(np.unique(rows)) != n_score or np.any(np.diff(rows.astype(np.int64)) <= 0)):
        raise ValueError("R20 control rows must be unique increasing aligned positions")
    rebuilt = np.clip(
        0.75 * (0.75 * arrays["baseline_lgb_svd32"]
                + 0.25 * arrays["baseline_hybrid_ridge"])
        + 0.25 * arrays["baseline_sparse_rank50"], 0.0, 100.0
    )
    if not np.allclose(rebuilt, arrays["baseline_prediction"], rtol=0.0, atol=1e-10):
        raise ValueError("R20 frozen three-component champion does not reconstruct from exact controls")
    if np.any(~np.isfinite(arrays["baseline_sparse_raw"])):
        raise ValueError("R20 sparse raw control is nonfinite")
    return arrays, rows.astype(np.int64, copy=False)


def fit_predict(fit_features: pd.DataFrame, fit_target: np.ndarray,
                score_features: pd.DataFrame, *, controls: dict[str, Any],
                card_id: str, cutoff: int) -> dict[str, Any]:
    """Fit one temporal fold and apply the fixed R_AUX25 policy."""
    if card_id != CARD_ID or int(cutoff) not in PILOT_CUTOFFS:
        raise ValueError("R20 adapter received a different card or cutoff")
    fit_x = _select_exact(fit_features)
    score_x = _select_exact(score_features)
    if (not fit_x.index.is_unique or not fit_x.index.is_monotonic_increasing
            or not score_x.index.is_unique or not score_x.index.is_monotonic_increasing):
        raise ValueError("R20 fit and score rows must preserve increasing source row positions")
    target = _target(fit_target, len(fit_x))
    fit_cycles = _cycle_codes(fit_x, label="R20 fit features")
    score_cycles = _cycle_codes(score_x, label="R20 score features")
    if np.any(fit_cycles > int(cutoff)) or np.any((score_cycles <= int(cutoff)) | (score_cycles > int(cutoff) + 12)):
        raise ValueError("R20 row cycles violate the frozen temporal horizon")
    control, control_rows = _validated_controls(controls, len(score_x))
    if not np.array_equal(score_x.index.to_numpy(dtype=np.int64), control_rows):
        raise ValueError("R20 controls are not aligned to the score row positions")

    model = TokenIdentitySparseL1Regressor(cutoff=int(cutoff)).fit(fit_x, target)
    raw = model.predict(score_x)
    prediction, baseline, mapped = reconstruct_r20(raw, control["baseline_prediction"], score_cycles)
    return {
        "prediction": prediction,
        "baseline_prediction": baseline,
        "raw_candidate": raw,
        "fit_predictor_count": 1,
        "state": model.audit_state(),
        "fit_metadata": {
            "cutoff": int(cutoff),
            "fit_rows": int(len(fit_x)),
            "fit_cycle_min": int(fit_cycles.min()),
            "fit_cycle_max": int(fit_cycles.max()),
            "feature_names": list(FEATURE_NAMES),
            "policy": ".75 current three-component champion + .25 within-cycle average-rank rank50(sparse LightGBM raw prediction), clipped [0,100]",
            "control_rows_verified_unique_increasing": True,
            "score_labels_passed_to_adapter": False,
            "fit_warnings": list(model._fit_warnings),
        },
        "audit_arrays": {"score_cycles": score_cycles, "candidate_rank50": mapped,
                         "control_rows": control_rows},
    }


__all__ = [
    "CARD_ID", "FEATURE_NAMES", "PILOT_CUTOFFS", "LIGHTGBM_PARAMETERS",
    "MIN_TOKEN_SUPPORT", "SELECTED_TOKEN_COUNT", "TOKEN_ANALYZER_PARAMETERS",
    "TokenIdentitySparseL1Regressor", "eligible_token_support_count", "fit_predict",
    "rank50_prediction", "reconstruct_r20", "select_token_identities",
]
