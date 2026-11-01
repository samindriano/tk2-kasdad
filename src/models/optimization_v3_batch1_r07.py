"""R07 NMF32 + structured native-category LightGBM regression candidate.

The adapter owns one fixed temporal-fold predictor and output composition. The
runner owns data loading, fold selection, controls, freezing, and persistence.
NMF components are an anonymous numerical representation; this module assigns
no semantic meaning to any factor.
"""
from __future__ import annotations

import json
import warnings
from typing import Any, Mapping

import lightgbm
import numpy as np
import pandas as pd
import sklearn
from scipy import sparse
from scipy.stats import rankdata
from sklearn.decomposition import NMF
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_extraction.text import TfidfVectorizer
from lightgbm import LGBMRegressor

from src.models.regression_research_lgb_text import _select_exact
from src.models.regression_research_tabular import COMMON_PARAMETERS, PARAMETER_SETS, _tree_summary
from src.models.regression_research_text import (
    HYBRID_FEATURES,
    STRUCTURED_FEATURES,
    RegressionTextModel,
    _TFIDF_PARAMS,
    _array_digest,
    _json_digest,
    _safe_builtin,
    _target,
    _text_documents,
)
from src.models.regression_round2_models import _SelectedPreprocessor
from src.models.round1_models import _fit_fingerprint


FEATURE_NAMES = tuple(HYBRID_FEATURES)
TFIDF_PARAMETERS: dict[str, Any] = dict(_TFIDF_PARAMS)
NMF_PARAMETERS: dict[str, Any] = {
    "n_components": 32,
    "init": "nndsvda",
    "solver": "cd",
    "beta_loss": "frobenius",
    "max_iter": 200,
    "random_state": 42,
}
LIGHTGBM_PARAMETERS: dict[str, Any] = {
    **COMMON_PARAMETERS,
    **PARAMETER_SETS["base"],
}
BASELINE_WEIGHT = 0.75
CANDIDATE_WEIGHT = 0.25
FINAL_RANGE = (0.0, 100.0)


def _vectorizer_state(vectorizer: TfidfVectorizer | None, fallback: str | None) -> dict[str, Any]:
    effective_vectorizer = vectorizer or TfidfVectorizer(**TFIDF_PARAMETERS)
    params = {
        key: list(value) if isinstance(value, tuple)
        else value.__name__ if isinstance(value, type)
        else value
        for key, value in effective_vectorizer.get_params(deep=False).items()
    }
    if vectorizer is None:
        return {
            "parameters": params,
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
        "parameters": params,
        "vocabulary_size": len(ordered),
        "vocabulary_sha256": _json_digest(ordered),
        "idf_sha256": _array_digest(vectorizer.idf_),
        "fallback": None,
    }


def _effective_nmf_parameters(estimator: NMF) -> dict[str, Any]:
    """Return sklearn's effective public NMF parameters in JSON-safe form."""
    return _safe_builtin(estimator.get_params(deep=False))


class _ContiguousCoordinateDescentNMF(NMF):
    """Preserve nndsvda initialization while satisfying the sklearn CD updater.

    sklearn 1.8.0 returns a Fortran-contiguous initial W for ``nndsvda`` but
    its compiled coordinate-descent updater requires C-contiguous W. This
    layout-only normalization preserves initialized values and all recipe
    parameters while avoiding a runtime failure on the installed version.
    """

    def _check_w_h(self, X, W, H, update_H):
        W_checked, H_checked = super()._check_w_h(X, W, H, update_H)
        return np.ascontiguousarray(W_checked), np.ascontiguousarray(H_checked)


def rank50_prediction(raw_prediction: Any, cycle_codes: Any) -> np.ndarray:
    """Blend target-scale score and average-rank percentile within each cycle."""
    raw = np.asarray(raw_prediction, dtype=np.float64)
    cycles_numeric = pd.to_numeric(pd.Series(cycle_codes), errors="raise")
    if cycles_numeric.isna().any():
        raise ValueError("R07 cycle codes must be present")
    cycles_float = cycles_numeric.to_numpy(dtype=np.float64)
    if not np.isfinite(cycles_float).all() or not np.equal(cycles_float, np.floor(cycles_float)).all():
        raise ValueError("R07 cycle codes must be finite integers")
    cycles = cycles_float.astype(np.int64)
    if raw.ndim != 1 or cycles.shape != raw.shape or not len(raw) or not np.isfinite(raw).all():
        raise ValueError("R07 raw scores and cycles must be nonempty, finite, aligned vectors")
    mapped = np.empty(len(raw), dtype=np.float64)
    for cycle in np.unique(cycles):
        mask = cycles == cycle
        percentile = 100.0 * rankdata(raw[mask], method="average") / int(mask.sum())
        mapped[mask] = 0.5 * raw[mask] + 0.5 * percentile
    result = np.clip(mapped, *FINAL_RANGE)
    if not np.isfinite(result).all():
        raise RuntimeError("R07 rank50 output is nonfinite")
    return result


def compose_r07_prediction(
    baseline_prediction: Any,
    raw_candidate: Any,
    cycle_codes: Any,
) -> np.ndarray:
    """Apply the frozen champion-plus-R07 rank50 policy and clip to [0,100]."""
    baseline = np.asarray(baseline_prediction, dtype=np.float64)
    raw = np.asarray(raw_candidate, dtype=np.float64)
    cycles = pd.to_numeric(pd.Series(cycle_codes), errors="raise").to_numpy(dtype=np.float64)
    if baseline.ndim != 1 or baseline.shape != raw.shape or baseline.shape != cycles.shape or not len(baseline):
        raise ValueError("R07 composition inputs must be nonempty aligned vectors")
    if (not np.isfinite(baseline).all() or np.any((baseline < 0.0) | (baseline > 100.0))
            or not np.isfinite(raw).all() or not np.isfinite(cycles).all()):
        raise ValueError("R07 composition inputs must be finite; baseline must be within [0,100]")
    mapped = rank50_prediction(raw, cycles)
    result = np.clip(BASELINE_WEIGHT * baseline + CANDIDATE_WEIGHT * mapped, *FINAL_RANGE)
    if not np.isfinite(result).all():
        raise RuntimeError("R07 composed output is nonfinite")
    return result


def _empty_factors(n_rows: int) -> np.ndarray:
    return np.empty((n_rows, 0), dtype=np.float64)


class NMF32StructuredLGB:
    """Fit-only TF-IDF/NMF32 text representation plus structured LightGBM."""

    def __init__(self) -> None:
        self.input_feature_names = FEATURE_NAMES
        self._fitted = False
        self._diagnostics: dict[str, Any] = {"n_predictions": 0}
        self._score_transform_warnings: list[dict[str, str]] = []

    @staticmethod
    def _select(X: pd.DataFrame) -> pd.DataFrame:
        # Use the project's established exact input allowlist validation.
        return _select_exact(X)

    @staticmethod
    def _fit_text(docs: list[str]) -> tuple[TfidfVectorizer | None, sparse.csr_matrix, str | None]:
        return RegressionTextModel._fit_vectorizer(docs)

    def fit(self, X: pd.DataFrame, y: Any) -> "NMF32StructuredLGB":
        self._fitted = False
        selected = self._select(X)
        target = _target(y, len(selected))
        self.fit_rows = int(len(target))
        self.fit_fingerprint = _fit_fingerprint(selected, FEATURE_NAMES, target)
        self.target_sha256 = _array_digest(target)

        self.preprocessor = _SelectedPreprocessor(tuple(STRUCTURED_FEATURES))
        structured, categorical_mask = self.preprocessor.fit_transform(
            selected.loc[:, list(STRUCTURED_FEATURES)]
        )
        self.categorical_positions = [i for i, is_cat in enumerate(categorical_mask) if is_cat]

        docs, missing = _text_documents(selected["signal_stream"])
        self.vectorizer, tfidf, text_fallback = self._fit_text(docs)
        if not sparse.isspmatrix_csr(tfidf):
            tfidf = tfidf.tocsr()
        if not np.isfinite(tfidf.data).all() or (tfidf.data < -1e-12).any():
            raise RuntimeError("R07 requires finite nonnegative TF-IDF values")

        feature_count = int(tfidf.shape[1])
        if self.vectorizer is None:
            self.nmf = None
            text_factors = _empty_factors(len(selected))
            self.nmf_fallback = text_fallback
        elif feature_count < int(NMF_PARAMETERS["n_components"]):
            # Fail closed to structured-only when a fold cannot support NMF32.
            self.nmf = None
            text_factors = _empty_factors(len(selected))
            self.nmf_fallback = "vocabulary_smaller_than_32_components"
        else:
            self.nmf = _ContiguousCoordinateDescentNMF(**dict(NMF_PARAMETERS))
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                text_factors = np.asarray(self.nmf.fit_transform(tfidf), dtype=np.float64)
            self.nmf_fit_warnings = [
                {"category": item.category.__name__, "message": str(item.message)}
                for item in caught
            ]
            self.nmf_convergence_warnings = [
                item for item in self.nmf_fit_warnings if item["category"] == "ConvergenceWarning"
            ]
            self.nmf_fallback = None
            if text_factors.shape != (len(selected), int(NMF_PARAMETERS["n_components"])):
                raise RuntimeError("R07 NMF fit returned an invalid factor matrix shape")
            if not np.isfinite(text_factors).all() or (text_factors < -1e-12).any():
                raise RuntimeError("R07 NMF fit returned nonfinite or negative factors")

        self._fit_missing_stream_rows = int(missing.sum())
        self._transformed_feature_names = (
            list(self.preprocessor.transformed_feature_names)
            + [f"signal_stream_nmf_{i + 1}" for i in range(text_factors.shape[1])]
            + ["signal_stream_missing"]
        )
        design = np.column_stack((
            np.asarray(structured, dtype=np.float64),
            text_factors,
            missing.astype(np.float64),
        ))
        if design.shape != (len(selected), len(self._transformed_feature_names)) or not np.isfinite(design).all():
            raise RuntimeError("R07 LightGBM fit design is misaligned or nonfinite")
        self._training_design_sha256 = _array_digest(design)
        self._training_design_shape = [int(design.shape[0]), int(design.shape[1])]
        design_frame = pd.DataFrame(design, columns=self._transformed_feature_names)

        self.estimator = LGBMRegressor(**dict(LIGHTGBM_PARAMETERS))
        self.estimator.fit(design_frame, target, categorical_feature=self.categorical_positions)
        self._tree_audit = _tree_summary(self.estimator.booster_)
        self._vectorizer_state = _vectorizer_state(self.vectorizer, self.nmf_fallback)
        self._state = self._make_state(target, text_factors)
        self._fitted = True
        self._diagnostics = {"n_predictions": 0}
        return self

    def _make_state(self, target: np.ndarray, text_factors: np.ndarray) -> dict[str, Any]:
        nmf_present = self.nmf is not None
        fit_warnings = getattr(self, "nmf_fit_warnings", [])
        convergence_warnings = getattr(self, "nmf_convergence_warnings", [])
        n_iter = int(self.nmf.n_iter_) if nmf_present else None
        max_iter = int(NMF_PARAMETERS["max_iter"])
        payload = {
            "id": "V3_BATCH1_R07_NMF32_STRUCTURED_LGB_BASE",
            "recipe": "fit-only word TF-IDF (1,2) + NMF32; structured native-category design + base LightGBM regression_l1",
            "factor_semantics": "not interpreted; numeric representation only",
            "input_feature_names": list(FEATURE_NAMES),
            "fit_rows": self.fit_rows,
            "fit_data_sha256": self.fit_fingerprint,
            "target_sha256": self.target_sha256,
            "target_range": [float(target.min()), float(target.max())],
            "training_design_shape": self._training_design_shape,
            "training_design_sha256": self._training_design_sha256,
            "transformed_feature_names": self._transformed_feature_names,
            "structured_preprocessing": self.preprocessor.audit_state(),
            "structured_categorical_positions": list(self.categorical_positions),
            "tfidf": self._vectorizer_state,
            "nmf": {
                "parameters": _effective_nmf_parameters(self.nmf) if nmf_present else _safe_builtin(NMF_PARAMETERS),
                "fit_performed": nmf_present,
                "fallback": self.nmf_fallback,
                "n_components_requested": int(NMF_PARAMETERS["n_components"]),
                "n_components_actual": int(text_factors.shape[1]),
                "components_sha256": _array_digest(self.nmf.components_) if nmf_present else None,
                "factors_sha256": _array_digest(text_factors),
                "factors_nonnegative": bool(text_factors.size == 0 or np.all(text_factors >= -1e-12)),
                "n_iter_": n_iter,
                "max_iter": max_iter,
                "converged": bool(n_iter is not None and n_iter < max_iter),
                "convergence_status": (
                    "NO_TEXT_FACTORIZATION" if not nmf_present else
                    "CONVERGED_BEFORE_MAX_ITER" if n_iter < max_iter else
                    "MAX_ITER_OR_CONVERGENCE_WARNING"
                ),
                "reconstruction_err_": float(self.nmf.reconstruction_err_) if nmf_present else None,
                "fit_warnings": fit_warnings,
                "fit_convergence_warning_count": len(convergence_warnings),
                "fit_convergence_warnings": {
                    "count": len(convergence_warnings),
                    "texts": [item["message"] for item in convergence_warnings],
                },
                "score_transform_convergence_warning_count": 0,
                "score_transform_convergence_warnings": {"count": 0, "texts": []},
                "initialization_memory_layout": "C-contiguous arrays to satisfy installed sklearn coordinate-descent extension; numerical nndsvda initialization values preserved",
                "score_transform_iteration_count_exposed": False,
            },
            "fit_missing_stream_rows": self._fit_missing_stream_rows,
            "lightgbm": {
                "parameters": dict(LIGHTGBM_PARAMETERS),
                "categorical_feature_argument": list(self.categorical_positions),
                "tree_complexity": dict(self._tree_audit),
                "version": lightgbm.__version__,
            },
            "sklearn_version": sklearn.__version__,
            "validation_data_passed_to_fit": False,
            "validation_labels_passed_to_fit": False,
            "early_stopping_used": False,
            "test_data_used_for_fit": False,
        }
        payload["state_sha256"] = _json_digest(_safe_builtin(payload))
        return json.loads(json.dumps(_safe_builtin(payload), sort_keys=True, allow_nan=False))

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("R07 model must be fitted before predict")
        selected = self._select(X)
        structured, unknown = self.preprocessor.transform(selected.loc[:, list(STRUCTURED_FEATURES)])
        docs, missing = _text_documents(selected["signal_stream"])
        if self.vectorizer is None:
            tfidf = sparse.csr_matrix((len(selected), 0), dtype=np.float64)
            unseen_tokens = sum(len(text.split()) for text in docs)
        else:
            tfidf = self.vectorizer.transform(docs).tocsr()
            analyzer = self.vectorizer.build_analyzer()
            vocabulary = self.vectorizer.vocabulary_
            unseen_tokens = sum(token not in vocabulary for doc in docs for token in analyzer(doc))
        if self.nmf is None:
            text_factors = _empty_factors(len(selected))
            score_warnings: list[dict[str, str]] = []
        else:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                text_factors = np.asarray(self.nmf.transform(tfidf), dtype=np.float64)
            score_warnings = [
                {"category": item.category.__name__, "message": str(item.message)}
                for item in caught
            ]
        if text_factors.shape[0] != len(selected) or not np.isfinite(text_factors).all():
            raise RuntimeError("R07 NMF score factors are misaligned or nonfinite")
        if text_factors.size and (text_factors < -1e-12).any():
            raise RuntimeError("R07 NMF score factors contain negative values")
        design = np.column_stack((
            np.asarray(structured, dtype=np.float64),
            text_factors,
            missing.astype(np.float64),
        ))
        if design.shape[1] != self._training_design_shape[1] or not np.isfinite(design).all():
            raise RuntimeError("R07 prediction design differs from fitted schema or is nonfinite")
        design_frame = pd.DataFrame(design, columns=self._transformed_feature_names)
        raw = np.asarray(self.estimator.predict(design_frame), dtype=np.float64)
        if raw.shape != (len(selected),) or not np.isfinite(raw).all():
            raise RuntimeError("R07 LightGBM returned invalid raw target-scale predictions")
        self._diagnostics = {
            "n_predictions": int(len(raw)),
            "signal_stream_missing_rows": int(missing.sum()),
            "unseen_tfidf_token_count": int(unseen_tokens),
            "unknown_category_rows": dict(sorted(unknown.items())),
            "nmf_score_transform_warnings": score_warnings,
            "nmf_score_transform_convergence_warning_count": sum(
                item["category"] == "ConvergenceWarning" for item in score_warnings
            ),
        }
        self._score_transform_warnings.extend(score_warnings)
        if self.nmf is not None:
            convergence = [
                item for item in self._score_transform_warnings
                if item["category"] == "ConvergenceWarning"
            ]
            self._state["nmf"]["score_transform_convergence_warnings"] = {
                "count": len(convergence),
                "texts": [item["message"] for item in convergence],
            }
            self._state["nmf"]["score_transform_convergence_warning_count"] = len(convergence)
            self._refresh_state_hash()
        return raw

    def _refresh_state_hash(self) -> None:
        payload = {key: value for key, value in self._state.items() if key != "state_sha256"}
        payload["state_sha256"] = _json_digest(_safe_builtin(payload))
        self._state = json.loads(json.dumps(_safe_builtin(payload), sort_keys=True, allow_nan=False))

    def audit_state(self) -> dict[str, Any]:
        if not self._fitted:
            raise RuntimeError("R07 cannot expose fitted state before training")
        return json.loads(json.dumps(self._state, sort_keys=True, allow_nan=False))

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
    if card_id != "R07":
        raise ValueError(f"R07 adapter received card {card_id!r}")
    required = {"baseline_prediction", "cycle_code"}
    if not required.issubset(controls):
        raise ValueError(f"R07 frozen-control bridge missing {sorted(required - set(controls))}")
    model = NMF32StructuredLGB().fit(fit_X, fit_y)
    raw_candidate = model.predict(score_X)
    baseline = np.asarray(controls["baseline_prediction"], dtype=np.float64)
    cycles = np.asarray(controls["cycle_code"])
    prediction = compose_r07_prediction(baseline, raw_candidate, cycles)
    mapped = rank50_prediction(raw_candidate, cycles)
    return {
        "raw_candidate": raw_candidate,
        "prediction": prediction,
        "fit_predictor_count": 1,
        "fit_metadata": {
            "cutoff": int(cutoff),
            "fit_rows": int(len(fit_X)),
            "score_rows": int(len(score_X)),
            "policy": "clip(0.75*frozen_three_component_champion + 0.25*rank50(raw R07),0,100)",
            "rank_mapping": "per complete future cycle; average ties; percentile=100*rank/n; rank50=0.5*raw+0.5*percentile; clip[0,100]",
            "vectorizer_fit_calls": 1,
            "nmf_fit_calls": int(model.nmf is not None),
            "predictor_fit_calls": 1,
            "nmf_convergence_status": model.audit_state()["nmf"]["convergence_status"],
            "nmf_fit_convergence_warnings": model.audit_state()["nmf"]["fit_convergence_warnings"],
            "nmf_score_transform_convergence_warnings": model.audit_state()["nmf"]["score_transform_convergence_warnings"],
            "validation_data_passed_to_fit": False,
            "validation_labels_passed_to_fit": False,
            "test_data_used": False,
        },
        "state": model.audit_state(),
        "prediction_diagnostics": model.prediction_diagnostics(),
        "audit_arrays": {
            "baseline_prediction": baseline,
            "candidate_rank50": mapped,
            "cycle_code": np.asarray(cycles, dtype=np.int64),
        },
    }


__all__ = [
    "BASELINE_WEIGHT", "CANDIDATE_WEIGHT", "FEATURE_NAMES", "LIGHTGBM_PARAMETERS",
    "NMF32StructuredLGB", "NMF_PARAMETERS", "TFIDF_PARAMETERS",
    "compose_r07_prediction", "fit_predict", "rank50_prediction",
]
