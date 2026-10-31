"""R03 split unigram/bigram sparse regression adapter.

The adapter preserves the established structured/OHE + sparse LightGBM L1
recipe while learning separate capped TF-IDF blocks inside each temporal fit.
No data are loaded here; the runner owns source, fold, cache and freeze checks.
"""
from __future__ import annotations

import hashlib
import json
import os
from typing import Any, Mapping

import lightgbm
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from scipy import sparse
from scipy.stats import rankdata
from sklearn.feature_extraction.text import TfidfVectorizer

from src.models.optimization_v2_regression import SPARSE_LGB_PARAMETERS
from src.models.regression_research_text import (
    HYBRID_FEATURES,
    RegressionTextModel,
    _TFIDF_PARAMS,
    _array_digest,
    _safe_builtin,
    _target,
    _text_documents,
)
from src.models.round1_models import _fit_fingerprint


CARD_ID = "R03"
FEATURE_NAMES = tuple(HYBRID_FEATURES)
TFIDF_MAX_FEATURES_PER_BLOCK = 30_000
TFIDF_MIN_DF = 2
TFIDF_SUBLINEAR_TF = True
RANK_WEIGHT = 0.5
RIDGE25_WEIGHT = 0.75
SPARSE_WEIGHT = 0.25
FINAL_RANGE = (0.0, 100.0)
THREADS = 2
JOB_MEMORY_CAP_BYTES = 6 * 1024 ** 3
FREE_MEMORY_RESERVE_BYTES = 2 * 1024 ** 3
LIGHTGBM_MEMORY_OVERHEAD_BYTES = 512 * 1024 ** 2
EXPECTED_SCORE_CYCLES = {
    55: tuple(range(56, 68)),
    67: tuple(range(68, 80)),
}


class R03IntegrityError(ValueError):
    """Raised when an R03 row, feature, policy, or score contract is invalid."""


def _vector(name: str, value: Any, *, dtype: Any = np.float64) -> np.ndarray:
    result = np.asarray(value, dtype=dtype)
    if result.ndim != 1:
        raise R03IntegrityError(f"{name} must be one-dimensional")
    return result


def _stable_json_sha256(value: Any) -> str:
    body = json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False,
                      separators=(",", ":"), default=str).encode("utf-8", errors="surrogatepass")
    return hashlib.sha256(body).hexdigest()


def split_tfidf_parameters() -> dict[str, dict[str, Any]]:
    """Return inherited TF-IDF settings with only the card-declared edits."""
    base = dict(_TFIDF_PARAMS)
    base.update({"min_df": TFIDF_MIN_DF,
                 "max_features": TFIDF_MAX_FEATURES_PER_BLOCK,
                 "sublinear_tf": TFIDF_SUBLINEAR_TF})
    unigram = {**base, "ngram_range": (1, 1)}
    bigram = {**base, "ngram_range": (2, 2)}
    return {"unigram": unigram, "bigram": bigram}


def _available_physical_memory_bytes() -> int | None:
    """Read available physical RAM without adding a runtime dependency."""
    if os.name == "nt":
        try:
            import ctypes

            class _MemoryStatus(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_ulong),
                    ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong),
                    ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
                ]

            status = _MemoryStatus()
            status.dwLength = ctypes.sizeof(_MemoryStatus)
            if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
                return None
            return int(status.ullAvailPhys)
        except Exception:
            return None
    try:
        return int(os.sysconf("SC_AVPHYS_PAGES") * os.sysconf("SC_PAGE_SIZE"))
    except (AttributeError, OSError, ValueError):
        return None


def estimate_sparse_memory(
    fit_x: pd.DataFrame, *, available_memory_bytes: int | None = None,
) -> dict[str, Any]:
    """Conservatively estimate sparse-design peak memory before any fit.

    The nonzero bound uses unique per-document analyzer terms and caps each
    block at its frozen maximum. It is an estimate, not observed fit memory;
    the policy multiplies the CSR bound for temporaries and reserves histogram
    and process overhead explicitly.
    """
    if not isinstance(fit_x, pd.DataFrame) or "signal_stream" not in fit_x:
        raise R03IntegrityError("Memory preflight needs a fit-only signal_stream frame")
    docs, _missing = _text_documents(fit_x["signal_stream"])
    params = split_tfidf_parameters()
    unigram_analyzer = TfidfVectorizer(**params["unigram"]).build_analyzer()
    bigram_analyzer = TfidfVectorizer(**params["bigram"]).build_analyzer()
    text_nnz_upper = 0
    for document in docs:
        text_nnz_upper += min(len(set(unigram_analyzer(document))), TFIDF_MAX_FEATURES_PER_BLOCK)
        text_nnz_upper += min(len(set(bigram_analyzer(document))), TFIDF_MAX_FEATURES_PER_BLOCK)

    categories = 3  # org_code, module_code, pattern_code in inherited OHE
    structured_source_count = len(FEATURE_NAMES) - categories - 1  # exclude categories and text
    structured_width_upper = structured_source_count * 5 + categories * 254
    n_rows = len(fit_x)
    structured_nnz_upper = n_rows * structured_width_upper
    total_nnz_upper = text_nnz_upper + structured_nnz_upper + n_rows  # missing-stream indicator
    csr_bytes_upper = total_nnz_upper * 16 + (n_rows + 1) * 8
    n_features_upper = structured_width_upper + 2 * TFIDF_MAX_FEATURES_PER_BLOCK + 1
    # Three simultaneous sparse representations plus a deliberately generous
    # two-copy, two-thread LightGBM histogram allowance.
    histogram_bytes = n_features_upper * 256 * 16 * THREADS * 2
    estimated_peak = 3 * csr_bytes_upper + histogram_bytes + LIGHTGBM_MEMORY_OVERHEAD_BYTES
    available = (_available_physical_memory_bytes() if available_memory_bytes is None
                 else int(available_memory_bytes))
    usable_free = None if available is None else max(0, available - FREE_MEMORY_RESERVE_BYTES)
    allowed = None if usable_free is None else min(JOB_MEMORY_CAP_BYTES, usable_free)
    status = "UNKNOWN_RESOURCE" if allowed is None else (
        "PASS" if estimated_peak <= allowed else "BLOCKED_RESOURCE")
    return {
        "status": status,
        "fit_rows": int(n_rows),
        "unigram_bigram_nnz_upper_bound": int(text_nnz_upper),
        "structured_width_upper_bound": int(structured_width_upper),
        "csr_bytes_upper_bound": int(csr_bytes_upper),
        "estimated_peak_bytes": int(estimated_peak),
        "estimated_peak_is_observed": False,
        "available_physical_bytes": available,
        "free_memory_reserve_bytes": FREE_MEMORY_RESERVE_BYTES,
        "job_cap_bytes": JOB_MEMORY_CAP_BYTES,
        "allowed_bytes": allowed,
        "threads": THREADS,
        "concurrent_training_jobs": 1,
        "sparse_only": True,
    }


def _param_state(vectorizer: TfidfVectorizer) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in vectorizer.get_params(deep=False).items():
        if isinstance(value, tuple):
            value = list(value)
        elif isinstance(value, type):
            value = value.__name__
        elif callable(value):
            value = f"{value.__module__}.{value.__qualname__}"
        out[str(key)] = value
    return _safe_builtin(out)


def _fit_text_block(
    docs: list[str], params: Mapping[str, Any], *, block_name: str,
) -> tuple[TfidfVectorizer | None, sparse.csr_matrix, dict[str, Any]]:
    """Fit one independent block on the current temporal training documents."""
    if not docs or not any(docs):
        empty = sparse.csr_matrix((len(docs), 0), dtype=np.float64)
        return None, empty, {"fallback": "all_training_text_empty", "width": 0,
                             "nnz": 0, "parameters": dict(params)}
    vectorizer = TfidfVectorizer(**dict(params))
    try:
        matrix = vectorizer.fit_transform(docs).tocsr()
    except ValueError as exc:
        message = str(exc).casefold()
        if not any(token in message for token in ("empty vocabulary", "after pruning, no terms remain")):
            raise
        empty = sparse.csr_matrix((len(docs), 0), dtype=np.float64)
        return None, empty, {"fallback": "empty_vocabulary_after_filters", "width": 0,
                             "nnz": 0, "parameters": dict(params)}
    if matrix.shape[0] != len(docs) or not np.isfinite(matrix.data).all():
        raise R03IntegrityError(f"{block_name} TF-IDF fit matrix is misaligned or nonfinite")
    ordered = [(str(token), int(index)) for token, index in
               sorted(vectorizer.vocabulary_.items(), key=lambda item: item[1])]
    state = {
        "fallback": None,
        "width": int(matrix.shape[1]),
        "nnz": int(matrix.nnz),
        "parameters": _param_state(vectorizer),
        "vocabulary_sha256": _stable_json_sha256(ordered),
        "idf_sha256": _array_digest(vectorizer.idf_),
    }
    return vectorizer, matrix, state


def rank50_within_cycle(raw_prediction: Any, cycle_code: Any) -> np.ndarray:
    """Map a target-scale prediction to its fixed 50% within-cycle rank blend.

    Ties use average ranks, each cycle is ranked over all score rows supplied
    for that temporal origin, and the final component is clipped to [0, 100].
    """
    raw = _vector("raw_prediction", raw_prediction)
    cycles = _vector("cycle_code", cycle_code, dtype=np.int64)
    if len(raw) == 0 or raw.shape != cycles.shape:
        raise R03IntegrityError("Raw predictions and cycle codes are empty or misaligned")
    if not np.isfinite(raw).all():
        raise R03IntegrityError("Raw sparse predictions must be finite")
    result = np.empty(len(raw), dtype=np.float64)
    for cycle in np.unique(cycles):
        selected = cycles == cycle
        n_cycle = int(selected.sum())
        percentile = 100.0 * rankdata(raw[selected], method="average") / n_cycle
        result[selected] = (1.0 - RANK_WEIGHT) * raw[selected] + RANK_WEIGHT * percentile
    result = np.clip(result, *FINAL_RANGE)
    if not np.isfinite(result).all():
        raise R03IntegrityError("Rank50 component is nonfinite")
    return result


def apply_fixed_route(
    *, cutoff: int, raw_prediction: Any, ridge25_prediction: Any,
    cycle_code: Any, row_positions: Any | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Reconstruct the frozen .75 Ridge25 + .25 sparse-rank50 candidate route."""
    raw = _vector("raw_prediction", raw_prediction)
    ridge = _vector("ridge25_prediction", ridge25_prediction)
    cycles = _vector("cycle_code", cycle_code, dtype=np.int64)
    if len(raw) == 0 or raw.shape != ridge.shape or raw.shape != cycles.shape:
        raise R03IntegrityError("R03 route components are empty or misaligned")
    if not np.isfinite(ridge).all() or np.any((ridge < 0.0) | (ridge > 100.0)):
        raise R03IntegrityError("Frozen Ridge25 predictions must be finite and within [0,100]")
    expected = EXPECTED_SCORE_CYCLES.get(int(cutoff))
    if expected is None or tuple(sorted(np.unique(cycles).tolist())) != expected:
        raise R03IntegrityError(f"Cutoff {cutoff} does not contain its exact STANDARD 12-cycle score view")
    if row_positions is not None:
        rows = _vector("row_positions", row_positions, dtype=np.int64)
        if rows.shape != raw.shape or np.any(rows < 0) or len(np.unique(rows)) != len(rows):
            raise R03IntegrityError("R03 score row positions are invalid or duplicated")
    rank50 = rank50_within_cycle(raw, cycles)
    prediction = np.clip(RIDGE25_WEIGHT * ridge + SPARSE_WEIGHT * rank50, *FINAL_RANGE)
    if not np.isfinite(prediction).all():
        raise R03IntegrityError("R03 candidate route returned nonfinite predictions")
    return prediction, rank50


class SplitBudgetSparseRegressor:
    """Structured/OHE + separate unigram/bigram sparse LightGBM L1 model."""

    def __init__(self) -> None:
        self.input_feature_names = tuple(FEATURE_NAMES)
        self._fitted = False

    def _select(self, X: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(X, pd.DataFrame):
            raise TypeError("X must be a pandas DataFrame")
        if X.columns.has_duplicates:
            raise R03IntegrityError("Input has duplicate feature names")
        if set(X.columns) != set(self.input_feature_names) or len(X.columns) != len(self.input_feature_names):
            extra = sorted(set(X.columns) - set(self.input_feature_names))
            absent = sorted(set(self.input_feature_names) - set(X.columns))
            raise R03IntegrityError(f"Input differs from frozen feature allowlist; extra={extra}, missing={absent}")
        return X.loc[:, list(self.input_feature_names)].copy()

    def fit(self, X: pd.DataFrame, y: Any) -> "SplitBudgetSparseRegressor":
        self._fitted = False
        selected = self._select(X)
        target = _target(y, len(selected))
        if len(target) == 0:
            raise R03IntegrityError("R03 cannot fit on an empty temporal training fold")
        self._fit_rows = int(len(target))
        self._fit_fingerprint = _fit_fingerprint(selected, self.input_feature_names, target)
        self._target_sha256 = _array_digest(target)
        docs, missing = _text_documents(selected["signal_stream"])
        params = split_tfidf_parameters()
        self._vectorizer_unigram, unigram, unigram_state = _fit_text_block(
            docs, params["unigram"], block_name="unigram")
        self._vectorizer_bigram, bigram, bigram_state = _fit_text_block(
            docs, params["bigram"], block_name="bigram")
        (structured, self._processor, self._scaler, self._encoder,
         self._numeric_positions, structured_state) = RegressionTextModel._structured_fit(selected)
        missing_column = sparse.csr_matrix(missing.astype(np.float64).reshape(-1, 1))
        design = sparse.hstack((structured, unigram, bigram, missing_column),
                               format="csr", dtype=np.float64)
        if not sparse.isspmatrix_csr(design):
            raise R03IntegrityError("R03 design must remain sparse CSR")
        if design.shape[0] != len(target) or not np.isfinite(design.data).all():
            raise R03IntegrityError("R03 fit matrix is misaligned or nonfinite")
        self._design_shape = [int(design.shape[0]), int(design.shape[1])]
        self._structured_width = int(structured.shape[1])
        self._unigram_width = int(unigram.shape[1])
        self._bigram_width = int(bigram.shape[1])
        self._estimator = LGBMRegressor(**dict(SPARSE_LGB_PARAMETERS))
        self._estimator.fit(design, target)
        model_text = self._estimator.booster_.model_to_string(num_iteration=-1)
        estimator_state = {
            "parameters": dict(SPARSE_LGB_PARAMETERS),
            "objective": "regression_l1",
            "target_transform": "identity",
            "trees_fitted": int(self._estimator.booster_.current_iteration()),
            "feature_importance_sha256": _array_digest(
                self._estimator.feature_importances_, dtype="<i8"),
            "fitted_model_sha256": hashlib.sha256(model_text.encode("utf-8")).hexdigest(),
        }
        self._fit_state = {
            "card_id": CARD_ID,
            "model_family": "lightgbm.LGBMRegressor",
            "recipe": "structured_ohe_plus_split_tfidf_unigram30k_bigram30k",
            "fit_rows": self._fit_rows,
            "fit_data_sha256": self._fit_fingerprint,
            "target_sha256": self._target_sha256,
            "input_features": list(self.input_feature_names),
            "tfidf_unigram": unigram_state,
            "tfidf_bigram": bigram_state,
            "structured_transform": structured_state,
            "missing_stream_indicator": True,
            "sparse_design": True,
            "training_design_shape": self._design_shape,
            "structured_width": self._structured_width,
            "unigram_width": self._unigram_width,
            "bigram_width": self._bigram_width,
            "estimator": estimator_state,
            "validation_data_passed_to_fit": False,
            "early_stopping_used": False,
            "new_predictor_fits": 1,
        }
        self._fit_state["state_sha256"] = _stable_json_sha256(self._fit_state)
        self._fitted = True
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("R03 model must be fitted before predict")
        selected = self._select(X)
        docs, missing = _text_documents(selected["signal_stream"])
        if self._vectorizer_unigram is None:
            unigram = sparse.csr_matrix((len(docs), 0), dtype=np.float64)
        else:
            unigram = self._vectorizer_unigram.transform(docs).tocsr()
        if self._vectorizer_bigram is None:
            bigram = sparse.csr_matrix((len(docs), 0), dtype=np.float64)
        else:
            bigram = self._vectorizer_bigram.transform(docs).tocsr()
        structured, _unknown = RegressionTextModel._structured_transform(
            selected, self._processor, self._scaler, self._encoder, self._numeric_positions)
        missing_column = sparse.csr_matrix(missing.astype(np.float64).reshape(-1, 1))
        design = sparse.hstack((structured, unigram, bigram, missing_column),
                               format="csr", dtype=np.float64)
        if design.shape[1] != self._design_shape[1] or not np.isfinite(design.data).all():
            raise R03IntegrityError("R03 score matrix differs from the fitted sparse schema")
        prediction = np.asarray(self._estimator.predict(design), dtype=np.float64)
        if prediction.shape != (len(selected),) or not np.isfinite(prediction).all():
            raise R03IntegrityError("Sparse LightGBM returned invalid raw predictions")
        return prediction

    def audit_state(self) -> dict[str, Any]:
        if not self._fitted:
            raise RuntimeError("R03 model is not fitted")
        return json.loads(json.dumps(_safe_builtin(self._fit_state), sort_keys=True, allow_nan=False))


def fit_predict(
    fit_x: Any, fit_y: Any, score_x: Any, *, controls: Mapping[str, Any],
    card_id: str, cutoff: int,
) -> dict[str, Any]:
    """Generic batch1 adapter: one sparse predictor fit per temporal origin."""
    if str(card_id) != CARD_ID:
        raise R03IntegrityError(f"R03 adapter cannot serve card {card_id!r}")
    cutoff = int(cutoff)
    if cutoff not in EXPECTED_SCORE_CYCLES:
        raise R03IntegrityError(f"R03 pilot does not authorize cutoff {cutoff}")
    required = {"baseline_prediction", "ridge25_prediction", "old_sparse_rank50", "control_rows"}
    missing = sorted(required - set(controls))
    if missing:
        raise R03IntegrityError(f"Frozen control bridge omitted fields: {missing}")
    if not isinstance(fit_x, pd.DataFrame) or not isinstance(score_x, pd.DataFrame):
        raise R03IntegrityError("R03 adapter requires pandas feature frames")
    resource = estimate_sparse_memory(fit_x)
    if resource["status"] != "PASS":
        raise R03IntegrityError(
            f"R03 resource preflight did not pass before model fit: {resource['status']}"
        )
    if "cycle_code" not in score_x:
        raise R03IntegrityError("Score cycle_code is missing")
    row_positions = score_x.index.to_numpy(dtype=np.int64)
    control_rows = _vector("control_rows", controls["control_rows"], dtype=np.int64)
    if not np.array_equal(control_rows, row_positions):
        raise R03IntegrityError("Frozen controls do not match score row positions and order")
    cycles = pd.to_numeric(score_x["cycle_code"], errors="raise").to_numpy(dtype=np.int64)
    baseline = _vector("baseline_prediction", controls["baseline_prediction"])
    ridge25 = _vector("ridge25_prediction", controls["ridge25_prediction"])
    old_rank = _vector("old_sparse_rank50", controls["old_sparse_rank50"])
    if not (baseline.shape == ridge25.shape == old_rank.shape == cycles.shape):
        raise R03IntegrityError("Cached control arrays are not score-row aligned")
    old_rebuilt = np.clip(RIDGE25_WEIGHT * ridge25 + SPARSE_WEIGHT * old_rank, *FINAL_RANGE)
    if not np.allclose(old_rebuilt, baseline, rtol=0.0, atol=1e-10):
        raise R03IntegrityError("Accepted v2_reg_sparse_lgb25 control reconstruction differs")

    model = SplitBudgetSparseRegressor().fit(fit_x, fit_y)
    raw = model.predict(score_x)
    prediction, rank50 = apply_fixed_route(
        cutoff=cutoff, raw_prediction=raw, ridge25_prediction=ridge25,
        cycle_code=cycles, row_positions=row_positions,
    )
    return {
        "raw_candidate": raw,
        "prediction": prediction,
        "fit_predictor_count": 1,
        "fit_metadata": {"model": model.audit_state(), "resource_preflight": resource},
        "state": {
            "card_id": CARD_ID,
            "policy": "R_SPARSE_REPLACE",
            "cutoff": cutoff,
            "rank_weight": RANK_WEIGHT,
            "ridge25_weight": RIDGE25_WEIGHT,
            "sparse_weight": SPARSE_WEIGHT,
            "rank_population": "all rows in the complete 12-cycle score horizon for each cycle",
        },
        "audit_arrays": {"candidate_rank50": rank50},
    }


__all__ = [
    "CARD_ID", "EXPECTED_SCORE_CYCLES", "FEATURE_NAMES", "R03IntegrityError",
    "SplitBudgetSparseRegressor", "apply_fixed_route", "fit_predict",
    "estimate_sparse_memory", "rank50_within_cycle", "split_tfidf_parameters",
]
