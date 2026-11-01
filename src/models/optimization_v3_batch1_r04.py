"""R04: sparse LightGBM-L1 with stateless word/bigram hashing."""
from __future__ import annotations

import hashlib
import json
from typing import Any, Mapping

import lightgbm
import numpy as np
import pandas as pd
import sklearn
from scipy import sparse
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.utils import murmurhash3_32

from src.models.optimization_v2_regression import SPARSE_LGB_PARAMETERS
from src.models.optimization_v3_batch1_r02 import (
    RIDGE25_WEIGHT,
    SPARSE_RANK50_WEIGHT,
    compose_r02_prediction,
    rank50_prediction,
)
from src.models.regression_research_text import (
    HYBRID_FEATURES,
    RegressionTextModel,
    _array_digest,
    _json_digest,
    _safe_builtin,
    _target,
    _text_documents,
)
from src.models.round1_models import _fit_fingerprint


CARD_ID = "R04"
FEATURE_NAMES = tuple(HYBRID_FEATURES)
PILOT_CUTOFFS = (55, 67)
HASHING_PARAMETERS: dict[str, Any] = {
    "analyzer": "word",
    "ngram_range": (1, 2),
    "n_features": 262_144,
    "alternate_sign": False,
    "norm": None,
    "binary": False,
    "lowercase": True,
    "strip_accents": "unicode",
    "token_pattern": r"(?u)\b\w\w+\b",
    "dtype": np.float64,
}
FINAL_RANGE = (0.0, 100.0)
_CONTROL_KEYS = (
    "baseline_prediction", "ridge25_prediction", "baseline_sparse_raw",
    "cycle_code", "control_rows",
)


def _params_json(values: Mapping[str, Any]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for key, value in values.items():
        if isinstance(value, tuple):
            output[key] = list(value)
        elif isinstance(value, type):
            output[key] = value.__name__
        elif isinstance(value, np.generic):
            output[key] = value.item()
        else:
            output[key] = value
    return output


def _validate_cycles(values: Any, *, label: str) -> np.ndarray:
    try:
        cycles = pd.to_numeric(pd.Series(values), errors="raise").to_numpy(dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"R04 {label} cycle_code must be numeric") from exc
    if (cycles.ndim != 1 or not np.isfinite(cycles).all()
            or not np.equal(cycles, np.floor(cycles)).all()):
        raise ValueError(f"R04 {label} cycle_code must contain finite integer values")
    return cycles.astype(np.int64)


def hashing_diagnostics(
    documents: list[str], matrix: sparse.spmatrix, *,
    vectorizer: HashingVectorizer | None = None,
) -> dict[str, Any]:
    """Count global ngram identities and exact bucket collisions without labels.

    ``collision_excess`` counts distinct analyzer ngrams beyond occupied hash
    buckets. Repeated occurrences of the same ngram are not counted as
    collisions. ``matrix_nnz`` separately records occupied per-row cells.
    """
    if not isinstance(documents, list) or any(not isinstance(doc, str) for doc in documents):
        raise TypeError("R04 collision diagnostics require a list of text documents")
    csr = matrix.tocsr(copy=False)
    if csr.shape[0] != len(documents):
        raise ValueError("R04 hashing matrix rows do not align with documents")
    if int(csr.shape[1]) != int(HASHING_PARAMETERS["n_features"]):
        raise ValueError("R04 hashing matrix width differs from the frozen feature space")
    if not np.isfinite(csr.data).all():
        raise ValueError("R04 hashing matrix contains nonfinite values")
    hasher = vectorizer if vectorizer is not None else HashingVectorizer(**dict(HASHING_PARAMETERS))
    analyzer = hasher.build_analyzer()
    seen_ngrams: set[str] = set()
    bucket_counts: dict[int, int] = {}
    occurrence_count = 0
    for document in documents:
        for ngram in analyzer(document):
            occurrence_count += 1
            if ngram in seen_ngrams:
                continue
            seen_ngrams.add(ngram)
            signed_hash = int(murmurhash3_32(ngram, seed=0, positive=False))
            bucket = abs(signed_hash) % int(HASHING_PARAMETERS["n_features"])
            bucket_counts[bucket] = bucket_counts.get(bucket, 0) + 1

    # Confirm that our documented bucket formula matches sklearn's stateless
    # transformer for a bounded prefix of rows before reporting collisions.
    check_rows = min(len(documents), 128)
    for row in range(check_rows):
        expected_buckets = {
            abs(int(murmurhash3_32(ngram, seed=0, positive=False)))
            % int(HASHING_PARAMETERS["n_features"])
            for ngram in analyzer(documents[row])
        }
        actual_buckets = set(int(value) for value in csr.indices[csr.indptr[row]:csr.indptr[row + 1]])
        if expected_buckets != actual_buckets:
            raise RuntimeError("R04 collision bucket formula differs from HashingVectorizer output")

    occupied_buckets = len(bucket_counts)
    collision_excess = len(seen_ngrams) - occupied_buckets
    collision_buckets = sum(count > 1 for count in bucket_counts.values())
    if collision_excess < 0:
        raise RuntimeError("R04 collision count cannot be negative")
    return {
        "document_count": int(len(documents)),
        "ngram_occurrence_count": int(occurrence_count),
        "unique_ngram_identities": int(len(seen_ngrams)),
        "occupied_hash_buckets": int(occupied_buckets),
        "collision_bucket_count": int(collision_buckets),
        "collision_excess": int(collision_excess),
        "matrix_shape": [int(csr.shape[0]), int(csr.shape[1])],
        "matrix_nnz": int(csr.nnz),
        "matrix_csr_bytes": int(csr.data.nbytes + csr.indices.nbytes + csr.indptr.nbytes),
        "bucket_self_check_rows": int(check_rows),
        "collision_definition": "distinct word/bigram identities beyond occupied buckets across this text batch; repeated identical terms are not collisions",
        "hash_function": "sklearn murmurhash3_32(seed=0), bucket=abs(signed_hash)%n_features",
    }


class R04HashingSparseHybrid:
    """Fit-only structured/OHE plus stateless word/bigram hashing and sparse L1."""

    input_feature_names = FEATURE_NAMES

    def __init__(self, cutoff: int):
        if isinstance(cutoff, bool) or int(cutoff) not in PILOT_CUTOFFS:
            raise ValueError(f"R04 cutoff must be one of {PILOT_CUTOFFS}")
        self.cutoff = int(cutoff)
        self._fitted = False

    def _select(self, frame: pd.DataFrame) -> pd.DataFrame:
        if (not isinstance(frame, pd.DataFrame) or frame.columns.has_duplicates
                or set(frame.columns) != set(FEATURE_NAMES)
                or len(frame.columns) != len(FEATURE_NAMES)):
            raise ValueError("R04 requires the exact structured + signal_stream feature allowlist")
        if set(FEATURE_NAMES) & {"record_id", "threat_rank", "artifact_signal"}:
            raise ValueError("R04 feature allowlist contains an identity or target")
        return frame.loc[:, list(FEATURE_NAMES)].copy()

    def _hash_documents(self, documents: list[str]) -> sparse.csr_matrix:
        matrix = self._vectorizer.transform(documents).tocsr()
        matrix.sum_duplicates()
        matrix.sort_indices()
        if matrix.shape != (len(documents), int(HASHING_PARAMETERS["n_features"])):
            raise RuntimeError("R04 hasher returned a different sparse design shape")
        if not np.isfinite(matrix.data).all():
            raise RuntimeError("R04 hasher returned nonfinite sparse values")
        return matrix

    def fit(self, frame: pd.DataFrame, target: Any) -> "R04HashingSparseHybrid":
        self._fitted = False
        selected = self._select(frame)
        y = _target(target, len(selected))
        cycles = _validate_cycles(selected["cycle_code"], label="fit")
        if len(cycles) == 0 or np.any((cycles < 47) | (cycles > self.cutoff)):
            raise ValueError("R04 fit rows contain a cycle outside 47..cutoff")
        if not selected.index.is_unique or not selected.index.is_monotonic_increasing:
            raise ValueError("R04 fit rows must preserve unique increasing Train positions")

        self.fit_rows = int(len(selected))
        self.fit_fingerprint = _fit_fingerprint(selected, FEATURE_NAMES, y)
        self.target_sha256 = _array_digest(y)
        documents, missing = _text_documents(selected["signal_stream"])
        self._vectorizer = HashingVectorizer(**dict(HASHING_PARAMETERS))
        text_matrix = self._hash_documents(documents)
        collision_state = hashing_diagnostics(documents, text_matrix, vectorizer=self._vectorizer)
        (structured, self._processor, self._scaler, self._encoder,
         self._numeric_positions, structured_state) = RegressionTextModel._structured_fit(selected)
        missing_column = sparse.csr_matrix(np.asarray(missing, dtype=np.float64).reshape(-1, 1))
        design = sparse.hstack((structured, text_matrix, missing_column), format="csr", dtype=np.float64)
        design.sum_duplicates()
        design.sort_indices()
        if design.shape[0] != len(y) or not sparse.isspmatrix_csr(design) or not np.isfinite(design.data).all():
            raise RuntimeError("R04 sparse fit design is misaligned, non-CSR, or nonfinite")

        self._estimator = lightgbm.LGBMRegressor(**dict(SPARSE_LGB_PARAMETERS))
        self._estimator.fit(design, y)
        model_text = self._estimator.booster_.model_to_string(num_iteration=-1)
        self._state = {
            "id": "V3_BATCH1_R04_HASHING_SPARSE_L1",
            "recipe": "fit-only structured/OHE + stateless word/bigram HashingVectorizer + missing indicator; LightGBM regression_l1",
            "input_feature_names": list(FEATURE_NAMES),
            "fit_rows": self.fit_rows,
            "fit_cycle_min": int(cycles.min()),
            "fit_cycle_max": int(cycles.max()),
            "fit_cycle_counts": {str(int(cycle)): int(count)
                                 for cycle, count in zip(*np.unique(cycles, return_counts=True))},
            "fit_data_sha256": self.fit_fingerprint,
            "target_sha256": self.target_sha256,
            "hashing_vectorizer_parameters": _params_json(HASHING_PARAMETERS),
            "learned_vocabulary": False,
            "learned_idf": False,
            "text_collision_diagnostics": collision_state,
            "structured_preprocessing": structured_state,
            "training_design_shape": [int(design.shape[0]), int(design.shape[1])],
            "training_design_nnz": int(design.nnz),
            "training_design_csr_bytes": int(design.data.nbytes + design.indices.nbytes + design.indptr.nbytes),
            "training_design_data_sha256": _array_digest(design.data),
            "missing_stream_fit_rows": int(np.asarray(missing, dtype=bool).sum()),
            "missing_stream_indicator": True,
            "sparse_design": True,
            "parameters": dict(SPARSE_LGB_PARAMETERS),
            "trees_fitted": int(self._estimator.booster_.current_iteration()),
            "fitted_model_sha256": hashlib.sha256(model_text.encode("utf-8")).hexdigest(),
            "lightgbm_version": lightgbm.__version__,
            "scikit_learn_version": sklearn.__version__,
            "validation_data_passed_to_fit": False,
            "validation_labels_passed_to_fit": False,
            "early_stopping_used": False,
            "test_data_used": False,
        }
        self._state["state_sha256"] = _json_digest(_safe_builtin(self._state))
        self._fitted = True
        self._diagnostics = {"n_predictions": 0}
        return self

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("R04 model must be fitted before predict")
        selected = self._select(frame)
        documents, missing = _text_documents(selected["signal_stream"])
        text_matrix = self._hash_documents(documents)
        collision_state = hashing_diagnostics(documents, text_matrix, vectorizer=self._vectorizer)
        structured, unknown = RegressionTextModel._structured_transform(
            selected, self._processor, self._scaler, self._encoder, self._numeric_positions
        )
        missing_column = sparse.csr_matrix(np.asarray(missing, dtype=np.float64).reshape(-1, 1))
        design = sparse.hstack((structured, text_matrix, missing_column), format="csr", dtype=np.float64)
        design.sum_duplicates()
        design.sort_indices()
        if (not sparse.isspmatrix_csr(design)
                or design.shape[1] != self._state["training_design_shape"][1]
                or not np.isfinite(design.data).all()):
            raise RuntimeError("R04 score design differs from fit schema or contains invalid values")
        prediction = np.asarray(self._estimator.predict(design), dtype=np.float64)
        if prediction.shape != (len(selected),) or not np.isfinite(prediction).all():
            raise RuntimeError("R04 sparse LightGBM returned invalid raw target-scale predictions")
        self._diagnostics = {
            "n_predictions": int(len(prediction)),
            "missing_stream_rows": int(np.asarray(missing, dtype=bool).sum()),
            "unknown_category_rows": dict(sorted(unknown.items())),
            "text_collision_diagnostics": collision_state,
        }
        return prediction

    def audit_state(self) -> dict[str, Any]:
        if not self._fitted:
            raise RuntimeError("R04 cannot expose fitted state before training")
        return json.loads(json.dumps(_safe_builtin(self._state), sort_keys=True, allow_nan=False))

    def prediction_diagnostics(self) -> dict[str, Any]:
        return json.loads(json.dumps(_safe_builtin(self._diagnostics), sort_keys=True, allow_nan=False))


def _validate_controls(controls: Mapping[str, Any], score_x: pd.DataFrame) -> tuple[dict[str, np.ndarray], np.ndarray]:
    missing = [name for name in _CONTROL_KEYS if name not in controls]
    if missing:
        raise ValueError(f"R04 control bridge missing {missing}")
    n_score = len(score_x)
    if n_score == 0:
        raise ValueError("R04 score features cannot be empty")
    arrays: dict[str, np.ndarray] = {}
    for name in _CONTROL_KEYS[:-1]:
        values = np.asarray(controls[name], dtype=np.float64)
        if values.shape != (n_score,) or not np.isfinite(values).all():
            raise ValueError(f"R04 control {name} is nonfinite or misaligned")
        if name in {"baseline_prediction", "ridge25_prediction"} and np.any((values < 0) | (values > 100)):
            raise ValueError(f"R04 control {name} must be in [0,100]")
        arrays[name] = values
    cycles = _validate_cycles(arrays["cycle_code"], label="score control")
    frame_cycles = _validate_cycles(score_x["cycle_code"], label="score feature")
    if not np.array_equal(cycles, frame_cycles):
        raise ValueError("R04 control cycle codes differ from score features")
    rows = np.asarray(controls["control_rows"])
    if (rows.shape != (n_score,) or rows.ndim != 1 or rows.dtype.kind not in "iu"
            or len(np.unique(rows)) != n_score or np.any(np.diff(rows.astype(np.int64)) <= 0)):
        raise ValueError("R04 control rows must be increasing unique source positions")
    feature_rows = score_x.index.to_numpy()
    if feature_rows.dtype.kind not in "iu" or not np.array_equal(feature_rows.astype(np.int64), rows.astype(np.int64)):
        raise ValueError("R04 controls do not match score feature row positions")
    if not np.array_equal(np.sort(np.unique(cycles)), np.arange(cycles.min(), cycles.max() + 1)):
        raise ValueError("R04 rank50 cycle batch must contain complete consecutive cycles")
    reconstructed = compose_r02_prediction(arrays["ridge25_prediction"], arrays["baseline_sparse_raw"], cycles)
    if not np.allclose(arrays["baseline_prediction"], reconstructed, atol=1e-10, rtol=0.0):
        raise ValueError("R04 frozen Ridge25 baseline does not reconstruct from row-aligned sparse control")
    return arrays, rows.astype(np.int64, copy=False)


def fit_predict(
    fit_X: pd.DataFrame,
    fit_y: Any,
    score_X: pd.DataFrame,
    *,
    controls: Mapping[str, Any],
    card_id: str,
    cutoff: int,
) -> dict[str, Any]:
    if card_id != CARD_ID or isinstance(cutoff, bool) or int(cutoff) not in PILOT_CUTOFFS:
        raise ValueError("R04 adapter only accepts its declared card and cutoffs 55/67")
    model = R04HashingSparseHybrid(int(cutoff))
    fit_x = model._select(fit_X)
    score_x = model._select(score_X)
    for frame, label in ((fit_x, "fit"), (score_x, "score")):
        if not frame.index.is_unique or not frame.index.is_monotonic_increasing:
            raise ValueError(f"R04 {label} rows must preserve unique increasing Train positions")
    fit_cycles = _validate_cycles(fit_x["cycle_code"], label="fit")
    score_cycles = _validate_cycles(score_x["cycle_code"], label="score")
    if (len(fit_cycles) == 0 or len(score_cycles) == 0
            or np.any((fit_cycles < 47) | (fit_cycles > int(cutoff)))
            or np.any((score_cycles <= int(cutoff)) | (score_cycles > int(cutoff) + 12))):
        raise ValueError("R04 fit/score rows violate the frozen temporal horizon")
    if set(np.unique(score_cycles).tolist()) != set(range(int(cutoff) + 1, int(cutoff) + 13)):
        raise ValueError("R04 score rows must include all 12 cycles for rank50 mapping")
    arrays, control_rows = _validate_controls(controls, score_x)
    if not np.array_equal(score_x.index.to_numpy(dtype=np.int64), control_rows):
        raise ValueError("R04 controls do not match score row positions")
    model.fit(fit_x, fit_y)
    raw_candidate = model.predict(score_x)
    candidate_rank50 = rank50_prediction(raw_candidate, score_cycles)
    prediction = np.clip(
        RIDGE25_WEIGHT * arrays["ridge25_prediction"]
        + SPARSE_RANK50_WEIGHT * candidate_rank50,
        *FINAL_RANGE,
    )
    baseline_rank50 = rank50_prediction(arrays["baseline_sparse_raw"], score_cycles)
    baseline_reconstructed = np.clip(
        RIDGE25_WEIGHT * arrays["ridge25_prediction"]
        + SPARSE_RANK50_WEIGHT * baseline_rank50,
        *FINAL_RANGE,
    )
    if not np.allclose(arrays["baseline_prediction"], baseline_reconstructed, atol=1e-10, rtol=0.0):
        raise RuntimeError("R04 reconstructed baseline differs after candidate fit")
    if (prediction.shape != (len(score_x),) or not np.isfinite(prediction).all()
            or np.any((prediction < 0) | (prediction > 100))):
        raise RuntimeError("R04 candidate policy produced invalid regression predictions")
    return {
        "raw_candidate": raw_candidate,
        "prediction": prediction,
        "fit_predictor_count": 1,
        "state": model.audit_state(),
        "fit_metadata": {
            **model.prediction_diagnostics(),
            "cutoff": int(cutoff),
            "fit_rows": int(len(fit_x)),
            "score_rows": int(len(score_x)),
            "fit_cycle_min": int(fit_cycles.min()),
            "fit_cycle_max": int(fit_cycles.max()),
            "candidate_rank50_range": [float(candidate_rank50.min()), float(candidate_rank50.max())],
            "baseline_reconstruction_max_abs_error": float(np.max(np.abs(arrays["baseline_prediction"] - baseline_reconstructed))),
            "validation_labels_passed_to_estimator": False,
            "test_data_used": False,
            "policy": "clip(0.75*Ridge25 + 0.25*rank50(new raw HashingVectorizer sparse LGB), 0, 100)",
            "rank_mapping": "per full 12-cycle score batch; within-cycle average ties; percentile=100*rank/n; rank50=0.5*raw+0.5*percentile; clip[0,100]",
        },
        "audit_arrays": {"candidate_rank50": candidate_rank50, "baseline_rank50": baseline_rank50},
    }


__all__ = [
    "CARD_ID", "FEATURE_NAMES", "HASHING_PARAMETERS", "PILOT_CUTOFFS",
    "R04HashingSparseHybrid", "SPARSE_LGB_PARAMETERS", "fit_predict",
    "hashing_diagnostics",
]
