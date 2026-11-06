"""Controlled L1/L2/monotone-L2 source pipeline for REG_IMPACT_MONOTONE_L2_V1.

All preprocessing is prefix-only and target-free. Official learner fitting is
kept separate from feature preparation so the same design can be reused by the
three prespecified arms at each origin.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Mapping

import lightgbm
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from scipy.stats import rankdata
from sklearn.decomposition import TruncatedSVD

from src.models.regression_research_lgb_text import build_lgb_text
from src.models.regression_research_text import (
    HYBRID_FEATURES,
    STRUCTURED_FEATURES,
    RegressionTextModel,
    _SVD_PARAMS,
    _array_digest,
    _text_documents,
)
from src.models.regression_round2_models import _SelectedPreprocessor


ORIGINS = (67, 75)
ARM_IDS = ("A_L1_FAITHFUL", "B_L2_CONTROL", "C_L2_MONOTONE")
EXPECTED_LIGHTGBM_VERSION = "4.6.0"
EXPECTED_TRAIN_SHA256 = "756523ff8156a9cd776e52249f7e2a8f23dcec432b8d10570e55ff6db0e0f082"
TARGET_COLUMN = "threat_rank"
ID_COLUMN = "record_id"
MAX_CYCLE = 79


@dataclass
class PrefixDesignBundle:
    """Target-free transformed fit/query matrices for one temporal origin."""

    origin: int
    fit_design: np.ndarray
    query_design: np.ndarray
    fit_rows: np.ndarray
    query_rows: np.ndarray
    fit_ids: np.ndarray
    query_ids: np.ndarray
    fit_cycles: np.ndarray
    query_cycles: np.ndarray
    feature_names: tuple[str, ...]
    categorical_indices: tuple[int, ...]
    impact_index: int
    preprocessing_state: dict[str, Any]
    tfidf_vocabulary: dict[str, int]
    tfidf_idf: np.ndarray
    svd_components: np.ndarray


def json_safe(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    raise TypeError(f"unsupported JSON value type: {type(value).__name__}")


def json_digest(value: Any) -> str:
    payload = json.dumps(json_safe(value), sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def file_sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _string_vector_digest(values: Any) -> str:
    items = np.asarray(values).astype(str).tolist()
    return hashlib.sha256("\n".join(items).encode("utf-8")).hexdigest()


def _cycle_vector(values: Any, *, label: str) -> np.ndarray:
    numeric = pd.to_numeric(pd.Series(values), errors="raise").to_numpy(dtype=np.float64)
    if (numeric.ndim != 1 or not len(numeric) or not np.isfinite(numeric).all()
            or not np.equal(numeric, np.floor(numeric)).all()):
        raise ValueError(f"{label} must be nonempty finite integer cycle codes")
    cycles = numeric.astype(np.int64)
    if np.any((cycles < 47) | (cycles > MAX_CYCLE)):
        raise ValueError(f"{label} falls outside frozen Train cycles 47..{MAX_CYCLE}")
    return cycles


def _feature_frame(frame: pd.DataFrame, *, label: str) -> pd.DataFrame:
    if not isinstance(frame, pd.DataFrame) or frame.columns.has_duplicates:
        raise TypeError(f"{label} must be a DataFrame with unique columns")
    missing = sorted(set(HYBRID_FEATURES) - set(frame.columns))
    if missing:
        raise ValueError(f"{label} is missing declared model features: {missing}")
    forbidden = {TARGET_COLUMN, "artifact_signal", "audit_lane", "checksum_hint"}
    if forbidden.intersection(frame.columns):
        raise ValueError(f"{label} contains protected columns: {sorted(forbidden.intersection(frame.columns))}")
    if ID_COLUMN not in frame.columns:
        raise ValueError(f"{label} is missing identity column {ID_COLUMN}")
    if frame[ID_COLUMN].isna().any():
        raise ValueError(f"{label} record_id values contain nulls")
    ids = frame[ID_COLUMN].astype(str)
    if ids.str.strip().eq("").any() or not ids.is_unique:
        raise ValueError(f"{label} record_id values must be nonempty and unique")
    return frame.loc[:, [ID_COLUMN, *HYBRID_FEATURES]].copy()


def _transform_query(
    selected: pd.DataFrame,
    preprocessor: _SelectedPreprocessor,
    vectorizer: Any,
    svd: TruncatedSVD | None,
    feature_names: tuple[str, ...],
) -> tuple[np.ndarray, np.ndarray]:
    structured, _unknown = preprocessor.transform(
        selected.loc[:, list(STRUCTURED_FEATURES)]
    )
    docs, missing = _text_documents(selected["signal_stream"])
    text_matrix = vectorizer.transform(docs).tocsr()
    text_dense = (
        svd.transform(text_matrix)
        if svd is not None
        else np.empty((len(docs), 0), dtype=np.float64)
    )
    design = np.column_stack((structured, text_dense, missing.astype(np.float64)))
    if design.shape != (len(selected), len(feature_names)) or not np.isfinite(design).all():
        raise RuntimeError("query SVD32 design is misaligned or nonfinite")
    return np.asarray(design, dtype=np.float64), np.asarray(missing, dtype=bool)


def prepare_prefix_design(
    frame: pd.DataFrame,
    origin: int,
    *,
    source_train_sha256: str,
    source_kind: str = "official",
) -> PrefixDesignBundle:
    """Fit structured, TF-IDF, and SVD transforms on rows <= origin only.

    This function accepts features and identity only; it has no target argument
    and rejects a target column if one is accidentally passed.
    """
    origin = int(origin)
    if origin not in ORIGINS:
        raise ValueError(f"origin must be one of {ORIGINS}")
    if source_kind not in {"official", "synthetic"}:
        raise ValueError("source_kind must be 'official' or 'synthetic'")
    if source_kind == "official" and source_train_sha256 != EXPECTED_TRAIN_SHA256:
        raise ValueError("official Train SHA-256 differs from the pinned R01/R16 source")
    if len(source_train_sha256) != 64 or any(c not in "0123456789abcdef" for c in source_train_sha256):
        raise ValueError("source_train_sha256 must be a lowercase SHA-256 digest")
    source = _feature_frame(frame, label=f"E{origin} source frame")
    cycles = _cycle_vector(source["cycle_code"], label=f"E{origin} source cycle_code")
    fit_rows = np.flatnonzero(cycles <= origin).astype(np.int64)
    query_rows = np.flatnonzero(cycles > origin).astype(np.int64)
    if not len(fit_rows) or not len(query_rows):
        raise ValueError(f"E{origin} requires nonempty fit and query populations")
    fit_cycles = cycles[fit_rows]
    query_cycles = cycles[query_rows]
    if np.any(fit_cycles > origin) or np.any(query_cycles <= origin):
        raise RuntimeError("prefix/query row partition leaked across origin")

    selected = source.loc[:, list(HYBRID_FEATURES)]
    fit_x = selected.iloc[fit_rows].copy()
    query_x = selected.iloc[query_rows].copy()
    preprocessor = _SelectedPreprocessor(tuple(STRUCTURED_FEATURES))
    structured_fit, categorical_mask = preprocessor.fit_transform(
        fit_x.loc[:, list(STRUCTURED_FEATURES)]
    )
    categorical_indices = tuple(
        i for i, is_category in enumerate(categorical_mask) if is_category
    )

    fit_docs, fit_missing = _text_documents(fit_x["signal_stream"])
    vectorizer, text_matrix, fallback = RegressionTextModel._fit_vectorizer(fit_docs)
    actual_svd_components = min(32, text_matrix.shape[1] - 1) if text_matrix.shape[1] > 1 else 0
    svd: TruncatedSVD | None = None
    if actual_svd_components:
        svd = TruncatedSVD(n_components=actual_svd_components, **dict(_SVD_PARAMS))
        text_fit = svd.fit_transform(text_matrix)
        svd_components = np.asarray(svd.components_, dtype=np.float64)
    else:
        text_fit = np.empty((len(fit_x), 0), dtype=np.float64)
        svd_components = np.empty((0, text_matrix.shape[1]), dtype=np.float64)

    feature_names = tuple(
        list(preprocessor.transformed_feature_names)
        + [f"signal_stream_svd_{i + 1}" for i in range(actual_svd_components)]
        + ["signal_stream_missing"]
    )
    fit_design = np.column_stack((structured_fit, text_fit, fit_missing.astype(np.float64)))
    if fit_design.shape != (len(fit_rows), len(feature_names)) or not np.isfinite(fit_design).all():
        raise RuntimeError("prefix SVD32 design is misaligned or nonfinite")

    query_design, _query_missing = _transform_query(
        query_x, preprocessor, vectorizer, svd, feature_names
    )
    impact_matches = [i for i, name in enumerate(feature_names) if name == "impact_load"]
    if len(impact_matches) != 1:
        raise RuntimeError(f"expected one numeric impact_load value column, found {impact_matches}")
    impact_index = int(impact_matches[0])
    if impact_index in categorical_indices:
        raise RuntimeError("impact_load value column is marked categorical")

    vectorizer_state = RegressionTextModel._text_state(vectorizer, fallback)
    preprocessing_state = {
        "origin": origin,
        "source_kind": source_kind,
        "source_train_sha256": source_train_sha256,
        "fit_rows": int(len(fit_rows)),
        "query_rows": int(len(query_rows)),
        "fit_cycle_min": int(fit_cycles.min()),
        "fit_cycle_max": int(fit_cycles.max()),
        "query_cycle_counts": {
            str(int(cycle)): int(np.sum(query_cycles == cycle))
            for cycle in np.unique(query_cycles)
        },
        "fit_rows_sha256": _array_digest(fit_rows),
        "query_rows_sha256": _array_digest(query_rows),
        "fit_record_ids_sha256": _string_vector_digest(source.iloc[fit_rows][ID_COLUMN]),
        "query_record_ids_sha256": _string_vector_digest(source.iloc[query_rows][ID_COLUMN]),
        "transformed_feature_names": list(feature_names),
        "categorical_feature_indices": list(categorical_indices),
        "impact_load_value_index": impact_index,
        "learned_transform_count": 3,
        "learned_transform_fit_rows": {
            "structured_preprocessor": int(len(fit_rows)),
            "tfidf_vectorizer": int(len(fit_rows)),
            "svd32": int(len(fit_rows)),
        },
        "structured_preprocessing": preprocessor.audit_state(),
        "tfidf": vectorizer_state,
        "tfidf_vocabulary_size": int(len(vectorizer.vocabulary_)),
        "tfidf_vocabulary_sha256": json_digest(vectorizer.vocabulary_),
        "tfidf_idf_sha256": _array_digest(vectorizer.idf_),
        "svd": {
            **dict(_SVD_PARAMS),
            "n_components_requested": 32,
            "n_components_actual": int(actual_svd_components),
            "components_sha256": _array_digest(svd_components),
        },
        "fit_design_shape": [int(v) for v in fit_design.shape],
        "query_design_shape": [int(v) for v in query_design.shape],
        "fit_design_sha256": _array_digest(fit_design),
        "query_design_sha256": _array_digest(query_design),
        "target_column_read": False,
        "test_data_read": False,
        "sample_submission_read": False,
    }
    return PrefixDesignBundle(
        origin=origin,
        fit_design=np.asarray(fit_design, dtype=np.float64),
        query_design=np.asarray(query_design, dtype=np.float64),
        fit_rows=fit_rows,
        query_rows=query_rows,
        fit_ids=source.iloc[fit_rows][ID_COLUMN].astype(str).to_numpy(dtype=str),
        query_ids=source.iloc[query_rows][ID_COLUMN].astype(str).to_numpy(dtype=str),
        fit_cycles=fit_cycles,
        query_cycles=query_cycles,
        feature_names=feature_names,
        categorical_indices=categorical_indices,
        impact_index=impact_index,
        preprocessing_state=preprocessing_state,
        tfidf_vocabulary=dict(vectorizer.vocabulary_),
        tfidf_idf=np.asarray(vectorizer.idf_, dtype=np.float64),
        svd_components=svd_components,
    )


def base_lgb_parameters() -> dict[str, Any]:
    if lightgbm.__version__ != EXPECTED_LIGHTGBM_VERSION:
        raise RuntimeError(
            f"LightGBM version mismatch: expected {EXPECTED_LIGHTGBM_VERSION}, got {lightgbm.__version__}"
        )
    return dict(build_lgb_text("base").actual_parameters)


def monotone_constraint_vector(bundle: PrefixDesignBundle) -> list[int]:
    width = len(bundle.feature_names)
    if bundle.fit_design.ndim != 2 or bundle.fit_design.shape[1] != width:
        raise ValueError("fit design width differs from transformed feature schema")
    matches = [i for i, name in enumerate(bundle.feature_names) if name == "impact_load"]
    if matches != [bundle.impact_index] or bundle.impact_index in bundle.categorical_indices:
        raise ValueError("impact_load value index is not uniquely verified numeric/noncategorical")
    constraints = [0] * width
    constraints[bundle.impact_index] = 1
    if any(constraints[i] != 0 for i in bundle.categorical_indices):
        raise RuntimeError("categorical features cannot carry monotone constraints")
    return constraints


def arm_parameters(bundle: PrefixDesignBundle, arm_id: str) -> dict[str, Any]:
    params = base_lgb_parameters()
    if arm_id == "A_L1_FAITHFUL":
        params["objective"] = "regression_l1"
    elif arm_id == "B_L2_CONTROL":
        params["objective"] = "regression"
    elif arm_id == "C_L2_MONOTONE":
        params["objective"] = "regression"
        params["monotone_constraints"] = monotone_constraint_vector(bundle)
        params["monotone_constraints_method"] = "advanced"
    else:
        raise ValueError(f"arm_id must be one of {ARM_IDS}")
    return params


def fit_arm(
    bundle: PrefixDesignBundle,
    fit_target: Any,
    arm_id: str,
    *,
    synthetic_params: Mapping[str, Any] | None = None,
) -> tuple[LGBMRegressor, np.ndarray, dict[str, Any]]:
    target = np.asarray(fit_target, dtype=np.float64)
    if target.shape != (len(bundle.fit_rows),) or not np.isfinite(target).all():
        raise ValueError("fit target must be finite and align exactly with prefix rows")
    if np.any((target < 0.0) | (target > 100.0)):
        raise ValueError("fit target falls outside the frozen 0..100 scale")
    params = arm_parameters(bundle, arm_id)
    if synthetic_params:
        if arm_id not in ARM_IDS:
            raise ValueError("unknown arm for synthetic parameter overrides")
        if set(synthetic_params) - {"n_estimators"}:
            raise ValueError("synthetic smoke may override only n_estimators")
        params.update(synthetic_params)
    fit_frame = pd.DataFrame(bundle.fit_design, columns=list(bundle.feature_names))
    query_frame = pd.DataFrame(bundle.query_design, columns=list(bundle.feature_names))
    model = LGBMRegressor(**params)
    model.fit(
        fit_frame,
        target,
        categorical_feature=list(bundle.categorical_indices),
    )
    raw = np.asarray(model.predict(query_frame), dtype=np.float64)
    if raw.shape != (len(bundle.query_rows),) or not np.isfinite(raw).all():
        raise RuntimeError(f"{arm_id} returned invalid query predictions")
    monotonicity = (
        check_raw_head_monotonicity(model, bundle)
        if arm_id == "C_L2_MONOTONE"
        else {"status": "NOT_APPLICABLE", "reason": "arm is unconstrained"}
    )
    meta = {
        "arm_id": arm_id,
        "origin": bundle.origin,
        "objective": params["objective"],
        "actual_parameters": params,
        "parameters_sha256": json_digest(params),
        "categorical_feature_indices": list(bundle.categorical_indices),
        "feature_names_sha256": json_digest(list(bundle.feature_names)),
        "fit_rows": int(len(bundle.fit_rows)),
        "query_rows": int(len(bundle.query_rows)),
        "fit_design_sha256": _array_digest(bundle.fit_design),
        "query_design_sha256": _array_digest(bundle.query_design),
        "target_column_passed_to_preprocessor": False,
        "validation_data_passed_to_fit": False,
        "early_stopping_used": False,
        "test_data_used_for_fit": False,
        "monotonicity_check": monotonicity,
        "synthetic_parameter_overrides": dict(synthetic_params or {}),
    }
    return model, raw, meta


def check_raw_head_monotonicity(
    model: LGBMRegressor,
    bundle: PrefixDesignBundle,
    *,
    tolerance: float = 1e-10,
) -> dict[str, Any]:
    values = np.unique(bundle.fit_design[:, bundle.impact_index])
    values = np.asarray(values[np.isfinite(values)], dtype=np.float64)
    if not len(values):
        raise ValueError("no finite observed impact_load values in prefix design")
    anchors = np.unique(np.linspace(0, len(bundle.fit_design) - 1, min(3, len(bundle.fit_design))).astype(int))
    deltas: list[float] = []
    for row in anchors:
        counterfactuals = np.repeat(bundle.fit_design[row : row + 1], len(values), axis=0)
        counterfactuals[:, bundle.impact_index] = values
        predictions = np.asarray(
            model.predict(pd.DataFrame(counterfactuals, columns=list(bundle.feature_names))),
            dtype=np.float64,
        )
        delta = np.diff(predictions)
        if len(delta):
            deltas.extend(delta.tolist())
    minimum_step = float(min(deltas)) if deltas else 0.0
    if minimum_step < -tolerance:
        raise RuntimeError(f"monotone raw head violated increasing impact_load: min step={minimum_step}")
    return {
        "status": "PASS",
        "impact_load_index": int(bundle.impact_index),
        "feature_width": int(bundle.fit_design.shape[1]),
        "categorical_indices": list(bundle.categorical_indices),
        "observed_value_count": int(len(values)),
        "anchor_rows": [int(x) for x in anchors],
        "minimum_prediction_step": minimum_step,
        "tolerance": float(tolerance),
        "raw_head_only": True,
        "ensemble_monotonicity_claim": False,
    }


def rank50_map(raw: Any, cycles: Any, *, clip: bool = True) -> np.ndarray:
    scores = np.asarray(raw, dtype=np.float64)
    cycle_values = np.asarray(cycles)
    if (scores.ndim != 1 or cycle_values.ndim != 1 or cycle_values.shape != scores.shape
            or not len(scores) or not np.isfinite(scores).all()
            or cycle_values.dtype.kind not in "iu"):
        raise ValueError("rank50 mapping requires aligned finite scores and integer cycles")
    percentile = np.empty_like(scores)
    for cycle in np.unique(cycle_values):
        mask = cycle_values == cycle
        percentile[mask] = 100.0 * rankdata(scores[mask], method="average") / int(mask.sum())
    mapped = 0.5 * scores + 0.5 * percentile
    return np.clip(mapped, 0.0, 100.0) if clip else mapped


def reconstruct_nested_champion(
    lgb_mapped: Any,
    ridge_mapped: Any,
    word_raw: Any,
    char_raw: Any,
    pos_raw: Any,
    cycles: Any,
    missing: Any,
) -> dict[str, np.ndarray]:
    """Replay the saved five-leaf route in its original nested arithmetic order."""
    cycles_array = np.asarray(cycles)
    missing_mask = np.asarray(missing, dtype=bool)
    values = [np.asarray(x, dtype=np.float64) for x in
              (lgb_mapped, ridge_mapped, word_raw, char_raw, pos_raw)]
    if (cycles_array.ndim != 1 or missing_mask.shape != cycles_array.shape
            or any(x.shape != cycles_array.shape or not np.isfinite(x).all() for x in values)):
        raise ValueError("champion leaves, cycles, and missing mask must be aligned finite vectors")
    lgb, ridge, word, char, pos = values
    if (np.any((lgb < 0) | (lgb > 100)) or np.any((ridge < 0) | (ridge > 100))):
        raise ValueError("mapped LGB and Ridge leaves must remain within [0,100]")
    mapped_word = rank50_map(word, cycles_array)
    mapped_char = rank50_map(char, cycles_array)
    mapped_pos = rank50_map(pos, cycles_array, clip=False)
    legacy = np.clip(0.5625 * lgb + 0.1875 * ridge + 0.25 * mapped_word, 0.0, 100.0)
    r06 = np.clip(0.75 * legacy + 0.25 * mapped_char, 0.0, 100.0)
    direct = np.clip(0.75 * r06 + 0.25 * mapped_pos, 0.0, 100.0)
    direct[missing_mask] = r06[missing_mask]
    return {
        "mapped_lgb": lgb,
        "mapped_ridge": ridge,
        "word_raw": word,
        "char_raw": char,
        "pos_raw": pos,
        "mapped_word": mapped_word,
        "mapped_char": mapped_char,
        "mapped_pos": mapped_pos,
        "legacy": legacy,
        "r06": r06,
        "direct": direct,
        "missing": missing_mask,
    }


def apply_saved_hinge(values: Any, hinge_state: Mapping[str, Any], missing: Any) -> np.ndarray:
    scores = np.asarray(values, dtype=np.float64)
    missing_mask = np.asarray(missing, dtype=bool)
    if scores.ndim != 1 or missing_mask.shape != scores.shape or not np.isfinite(scores).all():
        raise ValueError("hinge values and missing mask must be aligned finite vectors")
    required = {"intercept", "slope_low", "slope_high"}
    if required - set(hinge_state):
        raise ValueError(f"saved hinge state missing parameters: {sorted(required-set(hinge_state))}")
    output = scores.copy()
    present = ~missing_mask
    output[present] = np.clip(
        float(hinge_state["intercept"])
        + float(hinge_state["slope_low"]) * np.minimum(scores[present], 50.0)
        + float(hinge_state["slope_high"]) * np.maximum(scores[present] - 50.0, 0.0),
        0.0,
        100.0,
    )
    return output


def replace_lgb_slot_present_only(
    incumbent: Mapping[str, np.ndarray],
    new_lgb_raw: Any,
    cycles: Any,
    hinge_state: Mapping[str, Any],
) -> dict[str, np.ndarray]:
    """Replace only the present-row LGB leaf; preserve missing champion bits."""
    missing = np.asarray(incumbent["missing"], dtype=bool)
    cycles_array = np.asarray(cycles, dtype=np.int64)
    new_mapped = rank50_map(new_lgb_raw, cycles_array)
    rebuilt = reconstruct_nested_champion(
        new_mapped,
        incumbent["mapped_ridge"],
        incumbent["word_raw"],
        incumbent["char_raw"],
        incumbent["pos_raw"],
        cycles_array,
        missing,
    )
    candidate_direct = rebuilt["direct"].copy()
    candidate_direct[missing] = np.asarray(incumbent["direct"], dtype=np.float64)[missing]
    incumbent_final = apply_saved_hinge(incumbent["direct"], hinge_state, missing)
    candidate_final = apply_saved_hinge(candidate_direct, hinge_state, missing)
    candidate_final[missing] = incumbent_final[missing]
    if not np.array_equal(candidate_final[missing], incumbent_final[missing]):
        raise RuntimeError("missing champion rows changed during present-only LGB replacement")
    return {
        "new_lgb_mapped": new_mapped,
        "candidate_direct": candidate_direct,
        "incumbent_final": incumbent_final,
        "candidate_final": candidate_final,
        "missing_exact": np.array_equal(candidate_final[missing], incumbent_final[missing]),
    }


def mae(y: Any, prediction: Any) -> float:
    truth = np.asarray(y, dtype=np.float64)
    pred = np.asarray(prediction, dtype=np.float64)
    if truth.ndim != 1 or pred.shape != truth.shape or not len(truth):
        raise ValueError("MAE vectors must be nonempty and aligned")
    if not np.isfinite(truth).all() or not np.isfinite(pred).all():
        raise ValueError("MAE vectors must be finite")
    return float(np.mean(np.abs(truth - pred)))


def synthetic_frame(rows: int = 264) -> tuple[pd.DataFrame, np.ndarray]:
    """Deterministic toy features/labels used only by the smoke lifecycle."""
    i = np.arange(rows, dtype=np.int64)
    docs = [
        f"alert_{j % 40} module_{(j // 40) % 5} common_signal risk_context"
        for j in i
    ]
    frame = pd.DataFrame({
        ID_COLUMN: [f"SYN_{j:06d}" for j in i],
        "impact_load": 20.0 + (i % 61),
        "entry_friction": (i % 2).astype(float),
        "cycle_code": 47 + (i % 33),
        "scope_pressure": 68.0 + (i % 19),
        "text_load": 1.0 + (i % 9),
        "scope_trace": [f"{j % 8} nodes" for j in i],
        "org_spread": [f"{j % 7} nodes" for j in i],
        "module_spread": [f"{j % 5} nodes" for j in i],
        "telemetry_flag": ["active" if j % 3 else "inactive" for j in i],
        "org_code": [f"org_{j % 6}" for j in i],
        "module_code": [f"module_{j % 4}" for j in i],
        "pattern_code": [f"pattern_{j % 3}" for j in i],
        "signal_stream": docs,
    })
    target = 22.0 + 0.25 * frame["impact_load"].to_numpy(dtype=np.float64)
    target += 4.0 * frame["entry_friction"].to_numpy(dtype=np.float64)
    target += frame["signal_stream"].str.contains("alert_1", regex=False).to_numpy(dtype=np.float64) * 7.0
    return frame, np.clip(target, 0.0, 100.0)


__all__ = [
    "ARM_IDS",
    "EXPECTED_LIGHTGBM_VERSION",
    "EXPECTED_TRAIN_SHA256",
    "ORIGINS",
    "PrefixDesignBundle",
    "apply_saved_hinge",
    "arm_parameters",
    "base_lgb_parameters",
    "check_raw_head_monotonicity",
    "file_sha256",
    "fit_arm",
    "json_digest",
    "json_safe",
    "mae",
    "monotone_constraint_vector",
    "prepare_prefix_design",
    "rank50_map",
    "reconstruct_nested_champion",
    "replace_lgb_slot_present_only",
    "synthetic_frame",
]
