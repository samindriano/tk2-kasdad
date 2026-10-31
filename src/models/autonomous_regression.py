"""SVD128 LightGBM ablation for the autonomous regression research lane.

This adapter holds the validated Stage 5 LightGBM/structured/TF-IDF recipe
fixed and changes only the requested TruncatedSVD width from 32 to 128.
All learned transforms are fit on the rows passed to ``fit``.
"""
from __future__ import annotations

import json
from typing import Any

import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from lightgbm import LGBMRegressor

from src.models.regression_research_lgb_text import (
    LGB_TEXT_PARAMS,
    RegressionResearchLGBText,
    _select_exact,
)
from src.models.regression_research_text import (
    RegressionTextModel,
    _SVD_PARAMS,
    _array_digest,
    _json_digest,
    _target,
    _text_documents,
)
from src.models.regression_round2_models import _SelectedPreprocessor
from src.models.regression_research_text import STRUCTURED_FEATURES


class AutonomousRegressionSVD128(RegressionResearchLGBText):
    """Same fixed LightGBM recipe as Stage 5, with 128 SVD components."""

    def __init__(self, params_id: str = "base"):
        super().__init__(params_id=params_id)
        self.model_id = f"LIGHTGBM_SVD128_{params_id.upper()}"
        self.n_components_requested = 128

    def fit(self, X: pd.DataFrame, y: Any) -> "AutonomousRegressionSVD128":
        self._fitted = False
        selected = _select_exact(X)
        target = _target(y, len(selected))
        self._fit_rows = int(len(target))
        from src.models.round1_models import _fit_fingerprint

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
        self._svd_components_actual = (
            min(self.n_components_requested, text_matrix.shape[1] - 1)
            if text_matrix.shape[1] > 1 else 0
        )
        if self._svd_components_actual:
            self._svd = TruncatedSVD(
                n_components=self._svd_components_actual, **dict(_SVD_PARAMS)
            )
            text_dense = self._svd.fit_transform(text_matrix)
        else:
            self._svd = None
            text_dense = np.empty((len(selected), 0), dtype=np.float64)

        design = np.column_stack((structured, text_dense, missing.astype(np.float64)))
        if not np.isfinite(design).all():
            raise RuntimeError("SVD128 LightGBM design contains nonfinite values")
        self._transformed_feature_names = (
            list(self._preprocessor.transformed_feature_names)
            + [f"signal_stream_svd_{i + 1}" for i in range(self._svd_components_actual)]
            + ["signal_stream_missing"]
        )
        if len(self._transformed_feature_names) != design.shape[1]:
            raise RuntimeError("SVD128 feature-name schema does not match the design matrix")
        design_frame = pd.DataFrame(design, columns=self._transformed_feature_names)

        self._estimator = LGBMRegressor(**dict(self.actual_parameters))
        self._estimator.fit(
            design_frame, target, categorical_feature=self._categorical_positions
        )
        from src.models.regression_research_tabular import _tree_summary

        self._tree_audit = _tree_summary(self._estimator.booster_)
        self._training_design_sha256 = _array_digest(design)
        self._training_design_shape = [int(design.shape[0]), int(design.shape[1])]
        self._fit_missing_stream_rows = int(missing.sum())
        self._vectorizer_state = RegressionTextModel._text_state(self._vectorizer, fallback)
        self._structured_mask = categorical_mask
        self._state = self._make_state()
        self._fitted = True
        self._diagnostics = {"n_predictions": 0}
        return self

    def _make_state(self) -> dict[str, Any]:
        payload = super()._make_state()
        payload.pop("state_sha256", None)
        payload["svd"]["n_components_requested"] = self.n_components_requested
        payload["state_sha256"] = _json_digest(payload)
        return json.loads(json.dumps(payload, sort_keys=True, allow_nan=False))


def build_autonomous_regression(params_id: str = "base") -> AutonomousRegressionSVD128:
    """Build the paired SVD128 ablation for a frozen Stage 5 parameter choice."""
    if params_id not in LGB_TEXT_PARAMS:
        raise ValueError(f"params_id must be one of {list(LGB_TEXT_PARAMS)}")
    return AutonomousRegressionSVD128(params_id=params_id)


__all__ = ["AutonomousRegressionSVD128", "build_autonomous_regression"]
