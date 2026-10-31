"""Fixed CatBoost replacement candidate for the C1_B missing-stream expert."""
from __future__ import annotations

import hashlib
import json
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier

from src.models.classification_models import _fit_fingerprint, _is_missing, _to_builtin
from src.models.round1_models import EXPANDED_FEATURES, _HGBPreprocessor


_CATBOOST_PARAMS = {
    "loss_function": "Logloss",
    "iterations": 500,
    "depth": 6,
    "learning_rate": 0.05,
    "random_seed": 42,
    "thread_count": 2,
    "verbose": False,
    "allow_writing_files": False,
    "use_best_model": False,
}
_MISSING_CATEGORY = "__KASDAD_MISSING_CATEGORY__"


def _json_digest(value: Any) -> str:
    encoded = json.dumps(_to_builtin(value), sort_keys=True, allow_nan=False,
                         ensure_ascii=False, separators=(",", ":")).encode("utf-8", errors="surrogatepass")
    return hashlib.sha256(encoded).hexdigest()


def _file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class CatBoostMissingExpert:
    """A structured CatBoost estimator with the frozen Round 1 expanded parser."""

    input_feature_names = list(EXPANDED_FEATURES)

    def __init__(self):
        self.model_id = "CB_MISSING"
        self._preprocessor = _HGBPreprocessor("expanded")
        self._estimator: CatBoostClassifier | None = None
        self._fitted = False
        self._state: dict[str, Any] = {}
        self._diagnostics: dict[str, Any] = {}
        self._cat_feature_names: list[str] = []
        self._output_feature_names: list[str] = []
        self._training_positive_rate: float | None = None

    def _select(self, frame: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(frame, pd.DataFrame):
            raise TypeError("X must be a pandas DataFrame")
        if frame.columns.has_duplicates:
            raise ValueError("X has duplicate columns")
        absent = [name for name in self.input_feature_names if name not in frame.columns]
        if absent:
            raise ValueError(f"X is missing required expanded features: {absent}")
        return frame.loc[:, self.input_feature_names].copy()

    @staticmethod
    def _target(y: Any, rows: int) -> np.ndarray:
        target = np.asarray(y)
        if target.ndim != 1 or len(target) != rows or rows == 0:
            raise ValueError("y must be a nonempty vector aligned with X")
        try:
            numeric = target.astype(float)
        except (TypeError, ValueError) as exc:
            raise ValueError("artifact_signal must contain binary values") from exc
        if not np.isfinite(numeric).all() or not np.isin(numeric, (0.0, 1.0)).all():
            raise ValueError("artifact_signal must contain finite binary values 0/1")
        if not np.array_equal(np.unique(numeric), np.array([0.0, 1.0])):
            raise ValueError("CatBoost classifier requires both training classes")
        return numeric.astype(np.int8)

    def _feature_frame(self, selected: pd.DataFrame, *, fit: bool) -> pd.DataFrame:
        if fit:
            processed, _ = self._preprocessor.fit_transform(selected)
            self._output_feature_names = list(self._preprocessor.transformed_feature_names)
            category_positions = set(self._preprocessor.categorical_positions)
            self._cat_feature_names = [
                f"{self._output_feature_names[position].removesuffix('__ordinal')}__native"
                for position in sorted(category_positions)
            ]
        else:
            processed, unknown_counts = self._preprocessor.transform(selected)
            category_positions = set(self._preprocessor.categorical_positions)
            self._diagnostics["unknown_category_rows"] = unknown_counts

        raw, names, _, category_names, category_raw, _ = self._preprocessor._raw_matrix(
            selected, fit_categories=False
        )
        if list(names) != self._preprocessor.transformed_feature_names:
            raise RuntimeError("Expanded preprocessing schema changed")
        if fit:
            self._cat_feature_names = [
                f"{names[position].removesuffix('__ordinal')}__native"
                for position in sorted(category_positions)
            ]
        output_names: list[str] = []
        columns: dict[str, Any] = {}
        ordinal_to_category = {
            f"{name}__ordinal": name for name in category_names
        }
        for position, name in enumerate(names):
            if position in category_positions:
                original_name = ordinal_to_category[name]
                output_name = f"{original_name}__native"
                values = category_raw[original_name]
                columns[output_name] = [
                    _MISSING_CATEGORY if _is_missing(value) else str(value)
                    for value in values
                ]
            else:
                output_name = name
                values = np.asarray(processed[:, position], dtype=float)
                if not np.isfinite(values).all():
                    raise RuntimeError(f"Expanded numeric feature {name!r} was not imputed")
                columns[output_name] = values
            output_names.append(output_name)
        if output_names != [
            name if index not in category_positions else ordinal_to_category[name] + "__native"
            for index, name in enumerate(names)
        ]:
            raise RuntimeError("CatBoost output feature order changed")
        self._output_feature_names = output_names
        return pd.DataFrame(columns, columns=output_names)

    def fit(self, X: pd.DataFrame, y: Any) -> "CatBoostMissingExpert":
        self._fitted = False
        selected = self._select(X)
        target = self._target(y, len(selected))
        self._training_positive_rate = float(target.mean())
        self._diagnostics = {}
        design = self._feature_frame(selected, fit=True)
        if not self._cat_feature_names or not set(self._cat_feature_names).issubset(design.columns):
            raise RuntimeError("CatBoost native categorical columns failed to reconcile")
        estimator = CatBoostClassifier(**_CATBOOST_PARAMS)
        estimator.fit(design, target, cat_features=self._cat_feature_names,
                      use_best_model=False, verbose=False)
        self._estimator = estimator
        self._fit_rows = int(len(target))
        self._fit_fingerprint = _fit_fingerprint(selected, tuple(self.input_feature_names), target)
        self._fitted = True
        self._make_audit_state()
        return self

    def _make_audit_state(self) -> None:
        if self._estimator is None or self._training_positive_rate is None:
            raise RuntimeError("CatBoost must be fitted before state creation")
        with tempfile.TemporaryDirectory(prefix="kasdad_cb_state_") as temp_dir:
            model_path = Path(temp_dir) / "candidate.cbm"
            self._estimator.save_model(str(model_path), format="cbm")
            binary_sha256 = _file_digest(model_path)
        importance = np.asarray(self._estimator.get_feature_importance(), dtype=np.float64)
        params = _to_builtin(self._estimator.get_all_params())
        self._state = {
            "id": self.model_id,
            "kind": "catboost_missing_expert",
            "input_feature_names": list(self.input_feature_names),
            "fit_rows": self._fit_rows,
            "fit_data_sha256": self._fit_fingerprint,
            "positive_prevalence": self._training_positive_rate,
            "catboost_version": __import__("catboost").__version__,
            "parameters": params,
            "tree_count": int(self._estimator.tree_count_),
            "native_category_features": list(self._cat_feature_names),
            "output_feature_names": list(self._output_feature_names),
            "fit_only_preprocessor": self._preprocessor.audit_state(),
            "model_binary_sha256": binary_sha256,
            "feature_importance_sha256": hashlib.sha256(importance.tobytes(order="C")).hexdigest(),
            "decision_threshold": 0.50,
            "eval_set_used": False,
            "early_stopping_used": False,
        }
        self._state["model_state_sha256"] = _json_digest(self._state)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if not self._fitted or self._estimator is None:
            raise RuntimeError("CatBoost expert must be fitted before prediction")
        selected = self._select(X)
        self._diagnostics = {"n_predictions": int(len(selected))}
        design = self._feature_frame(selected, fit=False)
        probabilities = np.asarray(self._estimator.predict_proba(design)[:, 1], dtype=float)
        if probabilities.shape != (len(selected),) or not np.isfinite(probabilities).all():
            raise RuntimeError("CatBoost returned invalid probabilities")
        if ((probabilities < 0) | (probabilities > 1)).any():
            raise RuntimeError("CatBoost probabilities lie outside [0,1]")
        return probabilities

    def audit_state(self) -> dict[str, Any]:
        if not self._fitted:
            raise RuntimeError("CatBoost expert must be fitted before audit_state")
        return json.loads(json.dumps(_to_builtin(self._state), ensure_ascii=False,
                                    sort_keys=True, allow_nan=False))

    def prediction_diagnostics(self) -> dict[str, Any]:
        return json.loads(json.dumps(_to_builtin(self._diagnostics), sort_keys=True, allow_nan=False))


def build_catboost_classifier() -> CatBoostMissingExpert:
    """Build the one fixed missing-stream expert; loads no data and fits nothing."""
    return CatBoostMissingExpert()
