"""Fixed native-text CatBoost replacement for the C2_B present-stream expert.

This adapter reuses the audited expanded structured parser from the existing
CatBoost missing-stream expert.  Its text dictionary and text calcers are
created internally by CatBoost from the fit rows supplied to ``fit``; callers
own all splitting and routing decisions.
"""
from __future__ import annotations

import hashlib
import json
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier

from src.models.campaign_classification_catboost import CatBoostMissingExpert
from src.models.classification_models import _fit_fingerprint, _is_missing, _text_values, _to_builtin
from src.models.round1_models import EXPANDED_FEATURES


TEXT_FEATURE = "signal_stream"
EMPTY_STREAM_SENTINEL = "__KASDAD_EMPTY_STREAM__"
INPUT_FEATURE_NAMES = [*EXPANDED_FEATURES, TEXT_FEATURE]
_MISSING_CATEGORY = "__KASDAD_MISSING_CATEGORY__"
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

# CatBoost's documented Python Dictionary argument is spelled
# ``occurence_lower_bound`` (one 'r' after "oc"). The value means the minimum
# total token occurrence count for dictionary inclusion.
_TEXT_PROCESSING = {
    "tokenizers": [{
        "tokenizer_id": "Space",
        "delimiter": " ",
        "separator_type": "ByDelimiter",
        "lowercasing": "false",
    }],
    "dictionaries": [{
        "dictionary_id": "Unigram",
        "gram_order": "1",
        "occurence_lower_bound": "3",
        "max_dictionary_size": "20000",
    }],
    "feature_processing": {
        "default": [{
            "tokenizers_names": ["Space"],
            "dictionaries_names": ["Unigram"],
            "feature_calcers": ["BoW", "NaiveBayes"],
        }],
    },
}


def _json_digest(value: Any) -> str:
    encoded = json.dumps(
        _to_builtin(value), sort_keys=True, allow_nan=False,
        ensure_ascii=False, separators=(",", ":"),
    ).encode("utf-8", errors="surrogatepass")
    return hashlib.sha256(encoded).hexdigest()


def _file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class CatBoostNativeTextPresentExpert:
    """Expanded structured + raw native text expert, used on present rows only.

    The temporal runner fits this expert on every fold-training row. Missing or
    blank streams become one fixed token during model fitting. At evaluation,
    the runner applies the original availability mask and routes only present
    rows through this expert; the missing branch remains the frozen cache.
    """

    input_feature_names = list(INPUT_FEATURE_NAMES)

    def __init__(self) -> None:
        self.model_id = "CB_NATIVE_TEXT_PRESENT"
        self._structured = CatBoostMissingExpert()
        self._estimator: CatBoostClassifier | None = None
        self._fitted = False
        self._state: dict[str, Any] = {}
        self._diagnostics: dict[str, Any] = {}
        self._output_feature_names: list[str] = []
        self._cat_feature_names: list[str] = []
        self._fit_rows = 0
        self._fit_fingerprint = ""

    def _select(self, frame: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(frame, pd.DataFrame):
            raise TypeError("X must be a pandas DataFrame")
        if frame.columns.has_duplicates:
            raise ValueError("X has duplicate columns")
        absent = [name for name in self.input_feature_names if name not in frame.columns]
        if absent:
            raise ValueError(f"X is missing required expanded/text features: {absent}")
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

    @staticmethod
    def _text(series: pd.Series) -> list[str]:
        values, missing = _text_values(series)
        if any(not is_missing and EMPTY_STREAM_SENTINEL in value.split()
               for value, is_missing in zip(values, missing)):
            raise ValueError("Raw signal_stream contains the reserved empty-stream sentinel token")
        return [EMPTY_STREAM_SENTINEL if is_missing else value
                for value, is_missing in zip(values, missing)]

    def _design(self, selected: pd.DataFrame, *, fit: bool) -> pd.DataFrame:
        structured_input = selected.loc[:, list(EXPANDED_FEATURES)]
        structured = self._structured._feature_frame(structured_input, fit=fit)
        design = structured.copy()
        design[TEXT_FEATURE] = self._text(selected[TEXT_FEATURE])
        names = list(design.columns)
        if names != [*self._structured._output_feature_names, TEXT_FEATURE]:
            raise RuntimeError("Native-text feature order changed")
        if fit:
            self._cat_feature_names = list(self._structured._cat_feature_names)
            self._output_feature_names = names
        elif names != self._output_feature_names:
            raise RuntimeError("Native-text model feature schema changed after fit")
        return design

    def fit(self, X: pd.DataFrame, y: Any) -> "CatBoostNativeTextPresentExpert":
        self._fitted = False
        selected = self._select(X)
        target = self._target(y, len(selected))
        _, fit_stream_missing = _text_values(selected[TEXT_FEATURE])
        texts = self._text(selected[TEXT_FEATURE])
        self._fit_rows = int(len(target))
        self._fit_fingerprint = _fit_fingerprint(
            selected, tuple(self.input_feature_names), target,
        )
        design = self._design(selected, fit=True)
        if not self._cat_feature_names or not set(self._cat_feature_names).issubset(design.columns):
            raise RuntimeError("CatBoost native categorical columns failed to reconcile")
        estimator = CatBoostClassifier(**_CATBOOST_PARAMS, text_processing=_TEXT_PROCESSING)
        estimator.fit(
            design,
            target,
            cat_features=self._cat_feature_names,
            text_features=[TEXT_FEATURE],
            use_best_model=False,
            verbose=False,
        )
        self._estimator = estimator
        self._text_train_sha256 = _json_digest(texts)
        self._fit_stream_missing_rows = int(fit_stream_missing.sum())
        self._fitted = True
        self._make_audit_state()
        return self

    def _make_audit_state(self) -> None:
        if self._estimator is None:
            raise RuntimeError("CatBoost must be fitted before state creation")
        with tempfile.TemporaryDirectory(prefix="kasdad_cb_native_text_state_") as temp_dir:
            model_path = Path(temp_dir) / "candidate.cbm"
            self._estimator.save_model(str(model_path), format="cbm")
            binary_sha256 = _file_digest(model_path)
        text_feature_indices = [int(value) for value in self._estimator.get_text_feature_indices()]
        expected_text_index = self._output_feature_names.index(TEXT_FEATURE)
        if text_feature_indices != [expected_text_index]:
            raise RuntimeError("CatBoost did not retain signal_stream as its sole native text feature")
        importance = np.asarray(self._estimator.get_feature_importance(), dtype=np.float64)
        self._state = {
            "id": self.model_id,
            "kind": "catboost_native_text_present_expert",
            "input_feature_names": list(self.input_feature_names),
            "output_feature_names": list(self._output_feature_names),
            "native_category_features": list(self._cat_feature_names),
            "text_feature_names": [TEXT_FEATURE],
            "text_feature_indices": text_feature_indices,
            "fit_rows": self._fit_rows,
            "fit_data_sha256": self._fit_fingerprint,
            "fit_text_sha256": self._text_train_sha256,
            "fit_stream_missing_rows": self._fit_stream_missing_rows,
            "empty_stream_sentinel": EMPTY_STREAM_SENTINEL,
            "catboost_version": __import__("catboost").__version__,
            "parameters": _to_builtin(self._estimator.get_all_params()),
            "fixed_parameters": dict(_CATBOOST_PARAMS),
            "text_processing": json.loads(json.dumps(_TEXT_PROCESSING)),
            "fit_only_structured_preprocessor": self._structured._preprocessor.audit_state(),
            "tree_count": int(self._estimator.tree_count_),
            "model_binary_sha256": binary_sha256,
            "feature_importance_sha256": hashlib.sha256(importance.tobytes(order="C")).hexdigest(),
            "decision_threshold": 0.50,
            "eval_set_used": False,
            "early_stopping_used": False,
            "test_labels_or_test_text_used": False,
        }
        self._state["model_state_sha256"] = _json_digest(self._state)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if not self._fitted or self._estimator is None:
            raise RuntimeError("CatBoost native-text expert must be fitted before prediction")
        selected = self._select(X)
        design = self._design(selected, fit=False)
        probabilities = np.asarray(self._estimator.predict_proba(design)[:, 1], dtype=float)
        if probabilities.shape != (len(selected),) or not np.isfinite(probabilities).all():
            raise RuntimeError("CatBoost returned invalid native-text probabilities")
        if ((probabilities < 0) | (probabilities > 1)).any():
            raise RuntimeError("CatBoost native-text probabilities lie outside [0,1]")
        unknown = {
            name: int(sum(
                not _is_missing(value) and str(value).strip() not in seen
                for value in selected[name].to_numpy(dtype=object)
            ))
            for name, seen in self._structured._preprocessor.seen_categories.items()
        }
        self._diagnostics = {
            "n_predictions": int(len(selected)),
            "unknown_native_categories_by_feature": unknown,
            "text_column_passed_raw": True,
            "text_dictionary_refit": False,
        }
        return probabilities

    def audit_state(self) -> dict[str, Any]:
        if not self._fitted:
            raise RuntimeError("CatBoost native-text expert must be fitted before audit_state")
        return json.loads(json.dumps(_to_builtin(self._state), ensure_ascii=False,
                                    sort_keys=True, allow_nan=False))

    def prediction_diagnostics(self) -> dict[str, Any]:
        return json.loads(json.dumps(_to_builtin(self._diagnostics), ensure_ascii=False,
                                    sort_keys=True, allow_nan=False))


def build_catboost_native_text_classifier() -> CatBoostNativeTextPresentExpert:
    """Build the fixed candidate without loading data or performing a fit."""
    return CatBoostNativeTextPresentExpert()
