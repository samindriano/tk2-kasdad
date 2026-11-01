"""R08: row-local token scalars appended to the frozen LGB SVD32 design.

All learned structured, TF-IDF, SVD and LightGBM components are fitted on the
temporal fit rows only. Token statistics are deterministic transforms of each
individual signal_stream and do not use targets or cross-row information.
"""
from __future__ import annotations

import re
import hashlib
from typing import Any

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.stats import rankdata
from sklearn.decomposition import TruncatedSVD
from lightgbm import LGBMRegressor

from src.models.regression_research_lgb_text import (
    RegressionResearchLGBText,
    _select_exact,
)
from src.models.regression_research_text import (
    HYBRID_FEATURES,
    STRUCTURED_FEATURES,
    RegressionTextModel,
    _SVD_PARAMS,
    _array_digest,
    _json_digest,
    _target,
    _text_documents,
)
from src.models.regression_round2_models import _SelectedPreprocessor
from src.models.round1_models import _fit_fingerprint
from src.models.regression_research_tabular import _tree_summary
from src.models.regression_research_tabular import COMMON_PARAMETERS, PARAMETER_SETS


CARD_ID = "R08"
TOKENIZATION = r"[A-Za-z0-9]+, applied after lowercase normalization"
FEATURE_NAMES = HYBRID_FEATURES
PILOT_CUTOFFS = (55, 67)
LIGHTGBM_PARAMETERS = {**COMMON_PARAMETERS, **PARAMETER_SETS["base"]}
SCALAR_FEATURE_NAMES = (
    "signal_stream_token_count",
    "signal_stream_unique_token_count",
    "signal_stream_ttr",
    "signal_stream_character_count",
)
TOKEN_RE = re.compile(r"[A-Za-z0-9]+")


def text_scalar_features(stream: pd.Series) -> np.ndarray:
    """Return count, unique count, TTR, and raw character count per row.

    Tokenization uses the same trimmed documents as the existing text research
    models; character count matches the EDA's fill-null-and-count raw strings,
    retaining whitespace. TTR is 0 for empty/tokenless rows, as fixed by R08.
    """
    if not isinstance(stream, pd.Series):
        raise TypeError("R08 signal_stream must be a pandas Series")
    docs, _missing = _text_documents(stream)
    raw_values = stream.fillna("").astype(str).to_numpy(dtype=object)
    output = np.zeros((len(docs), len(SCALAR_FEATURE_NAMES)), dtype=np.float64)
    for row, doc in enumerate(docs):
        tokens = TOKEN_RE.findall(doc.lower())
        token_count = len(tokens)
        unique_count = len(set(tokens))
        output[row] = (
            float(token_count),
            float(unique_count),
            float(unique_count / token_count) if token_count else 0.0,
            float(len(raw_values[row])),
        )
    if output.shape != (len(stream), 4) or not np.isfinite(output).all():
        raise RuntimeError("R08 text scalar transform returned invalid values")
    return output


class R08TextScalarLGB(RegressionResearchLGBText):
    """The original base SVD32 LightGBM recipe plus four fixed row scalars."""

    def __init__(self):
        super().__init__(params_id="base")
        self.model_id = "R08_LIGHTGBM_SVD32_BASE_TEXT_SCALARS"
        if self.actual_parameters != LIGHTGBM_PARAMETERS:
            raise RuntimeError("R08 must retain the original base LightGBM parameters")

    def fit(self, X: pd.DataFrame, y: Any) -> "R08TextScalarLGB":
        self._fitted = False
        selected = _select_exact(X)
        target = _target(y, len(selected))
        self._fit_rows = int(len(target))
        self._fit_fingerprint = _fit_fingerprint(selected, self.input_feature_names, target)
        self._target_sha256 = _array_digest(target)

        self._preprocessor = _SelectedPreprocessor(tuple(STRUCTURED_FEATURES))
        structured, categorical_mask = self._preprocessor.fit_transform(
            selected.loc[:, list(STRUCTURED_FEATURES)]
        )
        categorical_mask = list(categorical_mask)
        self._categorical_positions = [i for i, is_cat in enumerate(categorical_mask) if is_cat]

        docs, missing = _text_documents(selected["signal_stream"])
        self._vectorizer, text_matrix, fallback = RegressionTextModel._fit_vectorizer(docs)
        self._svd_components_actual = min(32, text_matrix.shape[1] - 1) if text_matrix.shape[1] > 1 else 0
        if self._svd_components_actual:
            from sklearn.decomposition import TruncatedSVD
            from src.models.regression_research_text import _SVD_PARAMS

            self._svd = TruncatedSVD(n_components=self._svd_components_actual, **dict(_SVD_PARAMS))
            text_dense = self._svd.fit_transform(text_matrix)
        else:
            self._svd = None
            text_dense = np.empty((len(selected), 0), dtype=np.float64)

        scalars = text_scalar_features(selected["signal_stream"])
        design = np.column_stack((structured, text_dense, missing.astype(np.float64), scalars))
        if not np.isfinite(design).all():
            raise RuntimeError("R08 structured + SVD32 + scalar design contains nonfinite values")
        self._base_transformed_feature_names = (
            list(self._preprocessor.transformed_feature_names)
            + [f"signal_stream_svd_{i + 1}" for i in range(self._svd_components_actual)]
            + ["signal_stream_missing"]
        )
        self._transformed_feature_names = self._base_transformed_feature_names + list(SCALAR_FEATURE_NAMES)
        if len(self._transformed_feature_names) != design.shape[1]:
            raise RuntimeError("R08 feature names do not match the fitted design matrix")
        design_frame = pd.DataFrame(design, columns=self._transformed_feature_names)

        from lightgbm import LGBMRegressor
        from src.models.regression_research_tabular import _tree_summary

        self._estimator = LGBMRegressor(**dict(self.actual_parameters))
        self._estimator.fit(design_frame, target, categorical_feature=self._categorical_positions)
        self._tree_audit = _tree_summary(self._estimator.booster_)
        self._training_design_sha256 = _array_digest(design)
        self._training_design_shape = [int(design.shape[0]), int(design.shape[1])]
        self._fit_missing_stream_rows = int(missing.sum())
        self._vectorizer_state = RegressionTextModel._text_state(self._vectorizer, fallback)
        self._structured_mask = categorical_mask
        self._state = self._make_state()
        self._state.pop("state_sha256", None)
        self._state["derived_row_local_features"] = {
            "names": list(SCALAR_FEATURE_NAMES),
            "tokenization": TOKENIZATION,
            "token_text_normalization": "existing _text_values: missing to empty string; otherwise str(value).strip()",
            "empty_ttr": 0.0,
            "character_count": "Python len() of signal_stream.fillna('').astype(str), preserving whitespace",
            "fit_only_learned": False,
            "fit_design_scalar_sha256": _array_digest(scalars),
            "scalars_sha256": _json_digest({
                "names": list(SCALAR_FEATURE_NAMES),
                "array_sha256": _array_digest(scalars),
                "shape": list(scalars.shape),
            }),
            "repeated_fraction_excluded_as_exact_complement_of_ttr": True,
        }
        cycles = pd.to_numeric(selected["cycle_code"], errors="raise").to_numpy(dtype=np.int64)
        fit_rows = selected.index.to_numpy(dtype="<i8")
        self._state.update({
            "card_id": CARD_ID,
            "feature_names": list(FEATURE_NAMES),
            "fit_cycle_min": int(cycles.min()),
            "fit_cycle_max": int(cycles.max()),
            "fit_source_rows_sha256": hashlib.sha256(fit_rows.tobytes(order="C")).hexdigest(),
            "validation_labels_passed_to_fit": False,
        })
        self._state["id"] = self.model_id
        self._state["state_sha256"] = _json_digest(self._state)
        self._fitted = True
        self._diagnostics = {"n_predictions": 0}
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("Model must be fitted before predict")
        selected = _select_exact(X)
        structured, unknown = self._preprocessor.transform(
            selected.loc[:, list(STRUCTURED_FEATURES)]
        )
        docs, missing = _text_documents(selected["signal_stream"])
        if self._vectorizer is None:
            text_matrix = sparse.csr_matrix((len(docs), 0), dtype=np.float64)
            unseen_tokens = sum(len(text.split()) for text in docs)
        else:
            text_matrix = self._vectorizer.transform(docs).tocsr()
            analyzer = self._vectorizer.build_analyzer()
            vocabulary = self._vectorizer.vocabulary_
            unseen_tokens = sum(token not in vocabulary for text in docs for token in analyzer(text))
        text_dense = self._svd.transform(text_matrix) if self._svd is not None else np.empty((len(docs), 0))
        scalars = text_scalar_features(selected["signal_stream"])
        design = np.column_stack((structured, text_dense, missing.astype(np.float64), scalars))
        design_frame = pd.DataFrame(design, columns=self._transformed_feature_names)
        prediction = np.asarray(self._estimator.predict(design_frame), dtype=np.float64)
        if prediction.shape != (len(selected),) or not np.isfinite(prediction).all():
            raise RuntimeError("R08 LightGBM returned invalid predictions")
        self._diagnostics = {
            "n_predictions": int(len(prediction)),
            "signal_stream_missing_rows": int(missing.sum()),
            "unseen_tfidf_token_count": int(unseen_tokens),
            "unknown_category_rows": dict(sorted(unknown.items())),
            "text_scalar_names": list(SCALAR_FEATURE_NAMES),
        }
        return prediction


_CONTROL_KEYS = (
    "baseline_prediction", "baseline_lgb_svd32", "baseline_hybrid_ridge",
    "baseline_sparse_raw", "baseline_sparse_rank50", "control_rows",
)


def _cycle_codes(features: pd.DataFrame, *, label: str) -> np.ndarray:
    values = pd.to_numeric(features["cycle_code"], errors="raise").to_numpy(dtype=np.float64)
    if (values.ndim != 1 or not np.isfinite(values).all()
            or not np.equal(values, np.floor(values)).all()):
        raise ValueError(f"{label} cycle_code must be finite integer values")
    cycles = values.astype(np.int64)
    if np.any((cycles < 47) | (cycles > 79)):
        raise ValueError(f"{label} cycle_code is outside the frozen Train range 47-79")
    return cycles


def rank50_prediction(raw_prediction: Any, cycle_codes: Any) -> np.ndarray:
    """Mix target scale and average-rank percentile separately within cycle."""
    raw = np.asarray(raw_prediction, dtype=np.float64)
    cycles = np.asarray(cycle_codes)
    if raw.ndim != 1 or cycles.shape != raw.shape or raw.size == 0:
        raise ValueError("R08 rank50 inputs must be nonempty aligned vectors")
    if not np.isfinite(raw).all() or cycles.dtype.kind not in "iu":
        raise ValueError("R08 rank50 requires finite predictions and integer cycles")
    mapped = np.empty(raw.shape, dtype=np.float64)
    for cycle in np.unique(cycles):
        mask = cycles == cycle
        percentile = 100.0 * rankdata(raw[mask], method="average") / int(mask.sum())
        mapped[mask] = 0.5 * raw[mask] + 0.5 * percentile
    mapped = np.clip(mapped, 0.0, 100.0)
    if not np.isfinite(mapped).all():
        raise ValueError("R08 rank50 returned nonfinite predictions")
    return mapped


def reconstruct_r08(raw_prediction: Any, baseline_prediction: Any,
                    cycle_codes: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    raw = np.asarray(raw_prediction, dtype=np.float64)
    baseline = np.asarray(baseline_prediction, dtype=np.float64)
    if raw.shape != baseline.shape or raw.ndim != 1 or raw.size == 0:
        raise ValueError("R08 raw and baseline predictions must be aligned")
    if not np.isfinite(baseline).all() or np.any((baseline < 0) | (baseline > 100)):
        raise ValueError("R08 baseline predictions must be finite in [0,100]")
    mapped = rank50_prediction(raw, cycle_codes)
    candidate = np.clip(0.75 * baseline + 0.25 * mapped, 0.0, 100.0)
    return candidate, baseline.copy(), mapped


def _validated_controls(controls: dict[str, Any], n_score: int) -> tuple[dict[str, np.ndarray], np.ndarray]:
    missing = [name for name in _CONTROL_KEYS if name not in controls]
    if missing or n_score <= 0:
        raise ValueError(f"R08 controls are missing fields or empty: {missing}")
    arrays: dict[str, np.ndarray] = {}
    for name in _CONTROL_KEYS[:-1]:
        values = np.asarray(controls[name], dtype=np.float64)
        if values.shape != (n_score,) or not np.isfinite(values).all():
            raise ValueError(f"R08 control {name} is misaligned or nonfinite")
        if name != "baseline_sparse_raw" and np.any((values < 0.0) | (values > 100.0)):
            raise ValueError(f"R08 control {name} is outside [0,100]")
        arrays[name] = values
    rows = np.asarray(controls["control_rows"])
    if (rows.ndim != 1 or rows.shape != (n_score,) or rows.dtype.kind not in "iu"
            or len(np.unique(rows)) != n_score or np.any(np.diff(rows.astype(np.int64)) <= 0)):
        raise ValueError("R08 control rows must be unique, increasing, and aligned")
    rebuilt = np.clip(
        0.75 * (0.75 * arrays["baseline_lgb_svd32"] + 0.25 * arrays["baseline_hybrid_ridge"])
        + 0.25 * arrays["baseline_sparse_rank50"], 0.0, 100.0,
    )
    if not np.allclose(rebuilt, arrays["baseline_prediction"], rtol=0.0, atol=1e-10):
        raise ValueError("R08 incumbent does not reconstruct from frozen controls")
    if not np.isfinite(arrays["baseline_sparse_raw"]).all():
        raise ValueError("R08 sparse raw control is nonfinite")
    return arrays, rows.astype(np.int64, copy=False)


def fit_predict(fit_features: pd.DataFrame, fit_target: np.ndarray,
                score_features: pd.DataFrame, *, controls: dict[str, Any],
                card_id: str, cutoff: int) -> dict[str, Any]:
    """Fit one fold and apply the frozen auxiliary rank50 blend."""
    if card_id != CARD_ID or int(cutoff) not in PILOT_CUTOFFS:
        raise ValueError("R08 adapter received an unexpected card or cutoff")
    fit_x = _select_exact(fit_features)
    score_x = _select_exact(score_features)
    if (not fit_x.index.is_unique or not fit_x.index.is_monotonic_increasing
            or not score_x.index.is_unique or not score_x.index.is_monotonic_increasing):
        raise ValueError("R08 fold rows must preserve unique increasing source positions")
    target = _target(fit_target, len(fit_x))
    fit_cycles = _cycle_codes(fit_x, label="R08 fit features")
    score_cycles = _cycle_codes(score_x, label="R08 score features")
    if np.any(fit_cycles > int(cutoff)) or np.any((score_cycles <= int(cutoff)) | (score_cycles > int(cutoff) + 12)):
        raise ValueError("R08 rows violate the frozen temporal horizon")
    control, control_rows = _validated_controls(controls, len(score_x))
    if not np.array_equal(score_x.index.to_numpy(dtype=np.int64), control_rows):
        raise ValueError("R08 controls are not aligned to score row positions")

    model = R08TextScalarLGB().fit(fit_x, target)
    raw = model.predict(score_x)
    candidate, baseline, mapped = reconstruct_r08(raw, control["baseline_prediction"], score_cycles)
    return {
        "prediction": candidate,
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
            "derived_features": list(SCALAR_FEATURE_NAMES),
            "policy": ".75 exact current three-component champion + .25 within-cycle average-rank rank50(LightGBM raw prediction), clipped [0,100]",
            "control_rows_verified_unique_increasing": True,
            "score_labels_passed_to_fit": False,
            "test_data_used_for_fit": False,
        },
        "audit_arrays": {"score_cycles": score_cycles, "candidate_rank50": mapped,
                         "control_rows": control_rows},
    }


def build_r08_model() -> R08TextScalarLGB:
    return R08TextScalarLGB()


__all__ = [
    "CARD_ID", "FEATURE_NAMES", "LIGHTGBM_PARAMETERS", "PILOT_CUTOFFS",
    "TOKENIZATION", "SCALAR_FEATURE_NAMES", "text_scalar_features",
    "R08TextScalarLGB", "build_r08_model", "rank50_prediction",
    "reconstruct_r08", "fit_predict",
]
