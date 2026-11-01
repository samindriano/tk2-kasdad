"""Research Next V1 N03: a past-only router over fixed R22/R06 arms."""
from __future__ import annotations

from typing import Any
import warnings

import numpy as np
from scipy.stats import rankdata
from sklearn.ensemble import HistGradientBoostingRegressor


CARD_ID = "N03"
MINIMUM_PRESENT_OOF_ROWS = 2_000
ROUTE_THRESHOLD = -0.03
FEATURE_NAMES = (
    "A", "B", "B_minus_A", "abs_B_minus_A",
    "parsed_impact_load", "parsed_entry_friction", "parsed_text_load",
)
ROUTER_PARAMETERS: dict[str, Any] = {
    "loss": "squared_error",
    "learning_rate": 0.1,
    "max_depth": 2,
    "max_leaf_nodes": 4,
    "max_iter": 100,
    "min_samples_leaf": 200,
    "l2_regularization": 10.0,
    "early_stopping": False,
    "random_state": 42,
}


def rank50_complete_cycle(raw: Any, cycles: Any) -> np.ndarray:
    """Map scores on the complete supplied cycle populations using average ties."""
    values = np.asarray(raw, dtype=np.float64)
    group = np.asarray(cycles)
    if values.ndim != 1 or group.shape != values.shape or values.size == 0:
        raise ValueError("rank50 inputs must be nonempty aligned vectors")
    if group.dtype.kind not in "iu" or not np.isfinite(values).all():
        raise ValueError("rank50 inputs must contain finite scores and integer cycles")
    out = np.empty(values.shape, dtype=np.float64)
    for cycle in np.unique(group):
        mask = group == cycle
        pct = 100.0 * rankdata(values[mask], method="average") / int(mask.sum())
        out[mask] = np.clip(0.5 * values[mask] + 0.5 * pct, 0.0, 100.0)
    return out


def compose_arms(
    legacy: Any, raw_r22: Any, raw_r06: Any, cycles: Any, stream_missing: Any,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return exact R22 arm A, fixed present-stream arm B, both mapped arms."""
    legacy_arr = np.asarray(legacy, dtype=np.float64)
    missing = np.asarray(stream_missing, dtype=bool)
    r22 = np.asarray(raw_r22, dtype=np.float64)
    r06 = np.asarray(raw_r06, dtype=np.float64)
    cycle_arr = np.asarray(cycles)
    if (legacy_arr.ndim != 1 or any(x.shape != legacy_arr.shape for x in (missing, r22, r06, cycle_arr))
            or not np.isfinite(legacy_arr).all() or np.any((legacy_arr < 0) | (legacy_arr > 100))):
        raise ValueError("N03 arm inputs must be finite, aligned vectors; legacy must be in [0,100]")
    mapped22 = rank50_complete_cycle(r22, cycle_arr)
    mapped06 = rank50_complete_cycle(r06, cycle_arr)
    arm_a = np.clip(0.75 * legacy_arr + 0.25 * mapped22, 0.0, 100.0)
    present_b = np.clip(0.50 * legacy_arr + 0.25 * mapped22 + 0.25 * mapped06, 0.0, 100.0)
    arm_b = np.where(missing, arm_a, present_b)
    if not np.array_equal(arm_b[missing], arm_a[missing]):
        raise RuntimeError("N03 missing-stream fallback must equal arm A exactly")
    return arm_a, arm_b, mapped22, mapped06


def router_features(
    arm_a: Any, arm_b: Any, impact_load: Any, entry_friction: Any, text_load: Any,
) -> np.ndarray:
    """Build the exact ordered inference feature matrix; cycle/ID/target are absent."""
    a = np.asarray(arm_a, dtype=np.float64)
    b = np.asarray(arm_b, dtype=np.float64)
    numeric = [np.asarray(x, dtype=np.float64) for x in (impact_load, entry_friction, text_load)]
    if a.ndim != 1 or b.shape != a.shape or any(x.shape != a.shape for x in numeric):
        raise ValueError("N03 router features must be aligned vectors")
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("N03 arm features must be finite")
    return np.column_stack((a, b, b - a, np.abs(b - a), *numeric)).astype(np.float64, copy=False)


def fit_router(features: Any, loss_difference: Any) -> tuple[HistGradientBoostingRegressor, np.ndarray, np.ndarray]:
    """Fit fit-only median imputation and the frozen shallow router on past OOF rows."""
    x = np.asarray(features, dtype=np.float64)
    y = np.asarray(loss_difference, dtype=np.float64)
    if x.ndim != 2 or x.shape[1] != len(FEATURE_NAMES) or y.shape != (len(x),):
        raise ValueError("N03 router fit data has an invalid shape")
    if len(y) < MINIMUM_PRESENT_OOF_ROWS or not np.isfinite(y).all():
        raise ValueError("N03 router fit requires at least 2,000 finite present OOF rows")
    if not np.isfinite(x[:, :4]).all():
        raise ValueError("N03 arm features cannot be missing")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        medians = np.nanmedian(x, axis=0)
    all_missing = ~np.isfinite(medians)
    medians[all_missing] = 0.0
    prepared = np.where(np.isfinite(x), x, medians.reshape(1, -1))
    if not np.isfinite(prepared).all():
        raise RuntimeError("N03 fit-only median imputation returned nonfinite features")
    model = HistGradientBoostingRegressor(**dict(ROUTER_PARAMETERS))
    model.fit(prepared, y)
    return model, medians, all_missing


def predict_route(
    model: HistGradientBoostingRegressor,
    medians: Any,
    features: Any,
    arm_a: Any,
    arm_b: Any,
    stream_missing: Any,
    *,
    cutoff: int,
    present_oof_rows: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Use B only below the frozen predicted-loss threshold; otherwise exact A."""
    x = np.asarray(features, dtype=np.float64)
    a = np.asarray(arm_a, dtype=np.float64)
    b = np.asarray(arm_b, dtype=np.float64)
    missing = np.asarray(stream_missing, dtype=bool)
    median = np.asarray(medians, dtype=np.float64)
    if (x.ndim != 2 or x.shape[1] != len(FEATURE_NAMES) or x.shape[0] != len(a)
            or b.shape != a.shape or missing.shape != a.shape or median.shape != (len(FEATURE_NAMES),)):
        raise ValueError("N03 router score data has an invalid shape")
    if cutoff <= 55 or present_oof_rows < MINIMUM_PRESENT_OOF_ROWS:
        return a.copy(), np.zeros(len(a), dtype=np.float64), np.zeros(len(a), dtype=bool)
    prepared = np.where(np.isfinite(x), x, median.reshape(1, -1))
    predicted_difference = np.asarray(model.predict(prepared), dtype=np.float64)
    if not np.isfinite(predicted_difference).all():
        raise RuntimeError("N03 router predicted a nonfinite loss difference")
    routed = (~missing) & (predicted_difference < ROUTE_THRESHOLD)
    candidate = np.where(routed, b, a)
    return candidate, predicted_difference, routed


def constant_arm_control(
    arm_a: Any, arm_b: Any, stream_missing: Any, past_loss_difference: Any,
) -> tuple[np.ndarray, float, bool]:
    """Past-only constant control; threshold is applied to the mean past loss delta."""
    a = np.asarray(arm_a, dtype=np.float64)
    b = np.asarray(arm_b, dtype=np.float64)
    missing = np.asarray(stream_missing, dtype=bool)
    delta = np.asarray(past_loss_difference, dtype=np.float64)
    if a.ndim != 1 or b.shape != a.shape or missing.shape != a.shape or delta.ndim != 1 or not len(delta):
        raise ValueError("N03 constant-control inputs are invalid")
    if not np.isfinite(delta).all():
        raise ValueError("N03 constant-control past losses must be finite")
    mean_delta = float(delta.mean())
    choose_b = mean_delta < ROUTE_THRESHOLD
    return np.where((~missing) & choose_b, b, a), mean_delta, choose_b


def mae(y_true: Any, prediction: Any) -> float:
    """Independent absolute-loss metric used for report reconstruction."""
    y = np.asarray(y_true, dtype=np.float64)
    p = np.asarray(prediction, dtype=np.float64)
    if y.ndim != 1 or p.shape != y.shape or not np.isfinite(y).all() or not np.isfinite(p).all():
        raise ValueError("N03 MAE inputs must be aligned and finite")
    return float(np.abs(y - p).sum(dtype=np.float64) / len(y))
