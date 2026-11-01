"""R15: structured-only L1 expert for missing/blank signal streams."""
from __future__ import annotations

import hashlib
import warnings
from typing import Any

import numpy as np
import pandas as pd

from src.models.optimization_v3_batch1 import stream_missing_mask
from src.models.regression_research_tabular import RegressionResearchTabular
from src.models.regression_research_text import HYBRID_FEATURES
from src.models.round1_models import EXPANDED_FEATURES


CARD_ID = "R15"
FEATURE_NAMES = tuple(HYBRID_FEATURES)
STRUCTURED_FEATURE_NAMES = tuple(EXPANDED_FEATURES)
PILOT_CUTOFFS = (67, 75)
EXPERT_WEIGHT = 0.10
CHAMPION_WEIGHT = 0.90
TARGET_RANGE = (0.0, 100.0)
_CONTROL_KEYS = ("baseline_prediction", "stream_missing", "control_rows")


def _select_exact(frame: pd.DataFrame) -> pd.DataFrame:
    if (not isinstance(frame, pd.DataFrame) or frame.columns.has_duplicates
            or set(frame.columns) != set(FEATURE_NAMES) or len(frame.columns) != len(FEATURE_NAMES)):
        raise ValueError("R15 requires the exact structured plus signal_stream feature allowlist")
    if set(STRUCTURED_FEATURE_NAMES) & {"record_id", "threat_rank", "artifact_signal"}:
        raise ValueError("R15 structured feature allowlist contains an identity or target")
    return frame.loc[:, list(FEATURE_NAMES)].copy()


def _cycle_values(frame: pd.DataFrame, *, cutoff: int, label: str) -> np.ndarray:
    values = pd.to_numeric(frame["cycle_code"], errors="raise").to_numpy(dtype=np.float64)
    if (values.ndim != 1 or len(values) == 0 or not np.isfinite(values).all()
            or not np.equal(values, np.floor(values)).all()):
        raise ValueError(f"{label} cycle_code must be a nonempty finite integer vector")
    cycles = values.astype(np.int64)
    if np.any((cycles < 47) | (cycles > 79)):
        raise ValueError(f"{label} cycle_code falls outside the official Train range 47..79")
    return cycles


def _validated_controls(controls: dict[str, Any], n_score: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    missing = [name for name in _CONTROL_KEYS if name not in controls]
    if missing or n_score <= 0:
        raise ValueError(f"R15 controls are incomplete or score rows are empty: {missing}")
    baseline = np.asarray(controls["baseline_prediction"], dtype=np.float64)
    stream_missing = np.asarray(controls["stream_missing"], dtype=bool)
    rows = np.asarray(controls["control_rows"])
    if (baseline.shape != (n_score,) or stream_missing.shape != (n_score,)
            or not np.isfinite(baseline).all() or np.any((baseline < 0.0) | (baseline > 100.0))):
        raise ValueError("R15 baseline probabilities/stream mask are misaligned or outside [0,100]")
    if (rows.shape != (n_score,) or rows.dtype.kind not in "iu"
            or len(np.unique(rows)) != n_score or np.any(np.diff(rows.astype(np.int64)) <= 0)):
        raise ValueError("R15 control rows must be unique increasing aligned Train positions")
    return baseline, stream_missing, rows.astype(np.int64, copy=False)


def reconstruct_r15_arrays(arrays: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    raw = np.asarray(arrays["raw_candidate"], dtype=np.float64)
    baseline = np.asarray(arrays["baseline_prediction"], dtype=np.float64)
    missing = np.asarray(arrays["control__stream_missing"], dtype=bool)
    if raw.ndim != 1 or raw.shape != baseline.shape or raw.shape != missing.shape or not len(raw):
        raise ValueError("R15 saved candidate, baseline, and missing mask do not align")
    if not np.isfinite(raw).all() or not np.isfinite(baseline).all():
        raise ValueError("R15 saved route contains nonfinite values")
    if np.any((baseline < 0.0) | (baseline > 100.0)):
        raise ValueError("R15 saved incumbent predictions fall outside [0,100]")
    expert_clipped = np.clip(raw, *TARGET_RANGE)
    candidate = np.where(
        missing,
        np.clip(CHAMPION_WEIGHT * baseline + EXPERT_WEIGHT * expert_clipped, *TARGET_RANGE),
        baseline,
    )
    if not np.isfinite(candidate).all():
        raise ValueError("R15 route reconstructed nonfinite predictions")
    return candidate, baseline.copy()


def fit_predict(fit_features: pd.DataFrame, fit_target: np.ndarray,
                score_features: pd.DataFrame, *, controls: dict[str, Any],
                card_id: str, cutoff: int) -> dict[str, Any]:
    """Fit one structured L1 expert on missing-stream rows and apply fixed 10% route."""
    cutoff = int(cutoff)
    if card_id != CARD_ID or cutoff not in PILOT_CUTOFFS:
        raise ValueError("R15 adapter only accepts card R15 and cutoffs 67/75")
    fit_x = _select_exact(fit_features)
    score_x = _select_exact(score_features)
    if (not fit_x.index.is_unique or not fit_x.index.is_monotonic_increasing
            or not score_x.index.is_unique or not score_x.index.is_monotonic_increasing):
        raise ValueError("R15 fit and score rows must preserve increasing official Train positions")
    fit_cycles = _cycle_values(fit_x, cutoff=cutoff, label="R15 fit")
    score_cycles = _cycle_values(score_x, cutoff=cutoff, label="R15 score")
    if np.any(fit_cycles > cutoff) or np.any((score_cycles <= cutoff) | (score_cycles > 79)):
        raise ValueError("R15 rows violate the frozen temporal cutoff/view")

    target = np.asarray(fit_target, dtype=np.float64)
    if (target.shape != (len(fit_x),) or not np.isfinite(target).all()
            or np.any((target < 0.0) | (target > 100.0))):
        raise ValueError("R15 fit target must be aligned, finite, and within [0,100]")
    fit_missing = stream_missing_mask(fit_x["signal_stream"].to_numpy(dtype=object))
    score_missing = stream_missing_mask(score_x["signal_stream"].to_numpy(dtype=object))
    baseline, control_missing, control_rows = _validated_controls(controls, len(score_x))
    if (not np.array_equal(control_rows, score_x.index.to_numpy(dtype=np.int64))
            or not np.array_equal(control_missing, score_missing)):
        raise ValueError("R15 controls do not match score rows and null/blank stream masks")
    fit_rows = np.flatnonzero(fit_missing)
    score_missing_rows = np.flatnonzero(score_missing)
    if len(fit_rows) < 1 or len(score_missing_rows) < 1:
        raise ValueError("R15 expert requires nonempty missing-stream fit and score rows")

    model = RegressionResearchTabular(params_id="base", features=STRUCTURED_FEATURE_NAMES)
    warning_records: list[dict[str, str]] = []
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model.fit(fit_x.iloc[fit_rows].loc[:, list(STRUCTURED_FEATURE_NAMES)], target[fit_rows])
        raw_missing = np.asarray(
            model.predict(score_x.iloc[score_missing_rows].loc[:, list(STRUCTURED_FEATURE_NAMES)]),
            dtype=np.float64,
        )
        warning_records = [
            {"category": item.category.__name__, "message": str(item.message)}
            for item in caught
        ]
    if raw_missing.shape != (len(score_missing_rows),) or not np.isfinite(raw_missing).all():
        raise RuntimeError("R15 structured expert returned invalid raw predictions")

    # Present rows carry a finite zero sentinel in raw_candidate; the saved mask
    # makes clear that those values are unused and the incumbent stays unchanged.
    raw = np.zeros(len(score_x), dtype=np.float64)
    raw[score_missing_rows] = raw_missing
    clipped = np.clip(raw, *TARGET_RANGE)
    candidate = np.where(
        score_missing,
        np.clip(CHAMPION_WEIGHT * baseline + EXPERT_WEIGHT * clipped, *TARGET_RANGE),
        baseline,
    )
    if (not np.isfinite(candidate).all() or np.any((candidate < 0.0) | (candidate > 100.0))
            or not np.array_equal(candidate[~score_missing], baseline[~score_missing])):
        raise RuntimeError("R15 fixed missing-stream route is invalid or changed a present-stream prediction")

    fit_rows_hash = hashlib.sha256(
        np.asarray(fit_x.index.to_numpy(dtype="<i8")[fit_rows], dtype="<i8").tobytes(order="C")
    ).hexdigest()
    state = {
        "id": "V3_BATCH1_R15_MISSING_STREAM_STRUCTURED_L1",
        "cutoff": cutoff,
        "fit_population": "null, blank, or whitespace-only signal_stream rows within the official temporal prefix",
        "fit_rows": int(len(fit_rows)),
        "fit_cycle_min": int(fit_cycles[fit_rows].min()),
        "fit_cycle_max": int(fit_cycles[fit_rows].max()),
        "fit_source_row_positions_sha256": fit_rows_hash,
        "fit_missing_mask_sha256": hashlib.sha256(fit_missing.tobytes(order="C")).hexdigest(),
        "feature_names": list(STRUCTURED_FEATURE_NAMES),
        "signal_stream_used_as_feature": False,
        "record_id_or_target_used_as_feature": False,
        "model": model.audit_state(),
        "clip_range": list(TARGET_RANGE),
        "route": "missing only: 0.90 exact incumbent + 0.10 clipped raw structured expert; present: exact incumbent",
        "expert_weight": EXPERT_WEIGHT,
        "champion_weight": CHAMPION_WEIGHT,
        "percentile_or_rank_mapping": False,
        "test_data_used": False,
    }
    return {
        "prediction": candidate,
        "raw_candidate": raw,
        "fit_predictor_count": 1,
        "state": state,
        "fit_metadata": {
            "cutoff": cutoff,
            "fit_rows_all": int(len(fit_x)),
            "fit_missing_stream_rows": int(len(fit_rows)),
            "score_rows": int(len(score_x)),
            "score_missing_stream_rows": int(len(score_missing_rows)),
            "score_present_stream_rows": int((~score_missing).sum()),
            "structured_feature_names": list(STRUCTURED_FEATURE_NAMES),
            "expert_weight": EXPERT_WEIGHT,
            "champion_weight": CHAMPION_WEIGHT,
            "expert_predictions_computed_for_present_rows": False,
            "validation_data_passed_to_fit": False,
            "validation_labels_passed_to_fit": False,
            "score_labels_passed_to_adapter": False,
            "fit_warnings": warning_records,
        },
        "audit_arrays": {
            "score_cycles": score_cycles,
            "stream_missing": score_missing,
            "expert_missing_rows": np.isin(np.arange(len(score_x)), score_missing_rows),
            "raw_expert": raw,
            "control_rows": control_rows,
        },
    }


__all__ = [
    "CARD_ID", "EXPERT_WEIGHT", "FEATURE_NAMES", "PILOT_CUTOFFS",
    "STRUCTURED_FEATURE_NAMES", "fit_predict", "reconstruct_r15_arrays",
]
