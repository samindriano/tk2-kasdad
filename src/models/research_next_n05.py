"""N05 conditional quantile interval diagnostics (information-only lane)."""
from __future__ import annotations

from typing import Any, Mapping

import numpy as np
from lightgbm import LGBMRegressor
from scipy.stats import rankdata, spearmanr


CARD_ID = "N05"
ALPHAS = (0.1, 0.9)
QUANTILE_PARAMETERS: dict[str, Any] = {
    "boosting_type": "gbdt",
    "objective": "quantile",
    "learning_rate": 0.05,
    "n_estimators": 400,
    "num_leaves": 31,
    "min_child_samples": 40,
    "min_child_weight": 0.001,
    "reg_alpha": 0,
    "reg_lambda": 1,
    "subsample": 1,
    "subsample_freq": 0,
    "colsample_bytree": 1,
    "random_state": 42,
    "n_jobs": 2,
    "verbosity": -1,
    "deterministic": True,
    "force_col_wise": True,
}


def _finite_vector(values: Any, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or not len(array) or not np.isfinite(array).all():
        raise ValueError(f"{name} must be a nonempty finite vector")
    return array


def fit_quantile_head(
    design: Any,
    target: Any,
    *,
    alpha: float,
    cutoff: int,
    parameters: Mapping[str, Any] | None = None,
) -> tuple[LGBMRegressor, dict[str, Any]]:
    """Fit one fixed LightGBM quantile head; caller owns fit-row selection."""
    if alpha not in ALPHAS:
        raise ValueError(f"N05 alpha must be one of {ALPHAS}")
    x = np.asarray(design, dtype=np.float64)
    y = _finite_vector(target, "target")
    if (x.ndim != 2 or len(x) != len(y) or not np.isfinite(x).all()
            or np.any((y < 0) | (y > 100))):
        raise ValueError("N05 fit design and target must be finite, aligned, and target-bounded")
    recipe = dict(QUANTILE_PARAMETERS if parameters is None else parameters)
    if recipe.get("objective") != "quantile":
        raise ValueError("N05 estimator objective must remain quantile")
    recipe["alpha"] = float(alpha)
    estimator = LGBMRegressor(**recipe)
    estimator.fit(x, y)
    model_text = estimator.booster_.model_to_string().encode("utf-8")
    import hashlib

    state = {
        "estimator": "lightgbm.LGBMRegressor",
        "parameters": {key: (value.item() if isinstance(value, np.generic) else value)
                       for key, value in estimator.get_params(deep=False).items()},
        "alpha": float(alpha), "cutoff": int(cutoff), "fit_rows": int(len(y)),
        "fit_design_shape": [int(value) for value in x.shape],
        "fit_design_sha256": array_sha256(x), "target_sha256": array_sha256(y),
        "model_sha256": hashlib.sha256(model_text).hexdigest(),
        "score_target_passed_to_fit": False, "test_data_used": False,
    }
    return estimator, state


def array_sha256(values: Any, dtype: str = "<f8") -> str:
    import hashlib

    array = np.asarray(values, dtype=dtype)
    return hashlib.sha256(array.tobytes(order="C")).hexdigest()


def predict_quantile_head(estimator: LGBMRegressor, design: Any) -> np.ndarray:
    x = np.asarray(design, dtype=np.float64)
    if x.ndim != 2 or not np.isfinite(x).all():
        raise ValueError("N05 score design must be a finite matrix")
    prediction = np.asarray(estimator.predict(x), dtype=np.float64)
    if prediction.shape != (len(x),) or not np.isfinite(prediction).all():
        raise RuntimeError("N05 quantile head returned invalid predictions")
    return prediction


def make_interval(lower_raw: Any, upper_raw: Any) -> dict[str, Any]:
    """Clip endpoint scores, sort them, and retain the pre-sort crossing rate."""
    lower = _finite_vector(lower_raw, "lower_raw")
    upper = _finite_vector(upper_raw, "upper_raw")
    if lower.shape != upper.shape:
        raise ValueError("N05 interval endpoints must align")
    crossed = lower > upper
    lower_clipped = np.clip(lower, 0.0, 100.0)
    upper_clipped = np.clip(upper, 0.0, 100.0)
    return {
        "lower": np.minimum(lower_clipped, upper_clipped),
        "upper": np.maximum(lower_clipped, upper_clipped),
        "raw_crossing": crossed,
        "raw_crossing_rate": float(crossed.mean()),
        "clipped_endpoint_rows": int(np.sum((lower != lower_clipped) | (upper != upper_clipped))),
    }


def past_residual_interval(
    point_prediction: Any,
    past_target: Any,
    past_prediction: Any,
) -> dict[str, Any]:
    """Use fixed 10th/90th signed residual quantiles from past unique OOF rows."""
    point = _finite_vector(point_prediction, "point_prediction")
    target = _finite_vector(past_target, "past_target")
    prediction = _finite_vector(past_prediction, "past_prediction")
    if target.shape != prediction.shape or np.any((point < 0) | (point > 100)):
        raise ValueError("N05 past residual inputs or R22 points are invalid")
    residual = target - prediction
    quantiles = np.quantile(residual, ALPHAS, method="linear")
    raw_lower = point + quantiles[0]
    raw_upper = point + quantiles[1]
    interval = make_interval(raw_lower, raw_upper)
    interval.update({
        "past_residual_q10": float(quantiles[0]),
        "past_residual_q90": float(quantiles[1]),
        "past_residual_rows": int(len(residual)),
        "past_residual_sha256": array_sha256(residual),
    })
    return interval


def pinball_loss(target: Any, prediction: Any, alpha: float) -> float:
    if alpha not in ALPHAS:
        raise ValueError(f"N05 alpha must be one of {ALPHAS}")
    y = _finite_vector(target, "target")
    p = _finite_vector(prediction, "prediction")
    if y.shape != p.shape:
        raise ValueError("N05 pinball target and prediction lengths differ")
    residual = y - p
    return float(np.mean(np.maximum(alpha * residual, (alpha - 1.0) * residual)))


def interval_metrics(target: Any, lower: Any, upper: Any,
                     point_prediction: Any | None = None) -> dict[str, Any]:
    y = _finite_vector(target, "target")
    lo = _finite_vector(lower, "lower")
    hi = _finite_vector(upper, "upper")
    if y.shape != lo.shape or y.shape != hi.shape or np.any(lo > hi):
        raise ValueError("N05 interval metrics require aligned ordered endpoints")
    width = hi - lo
    inside = (y >= lo) & (y <= hi)
    reference = ((lo + hi) / 2.0 if point_prediction is None
                 else _finite_vector(point_prediction, "point_prediction"))
    if reference.shape != y.shape:
        raise ValueError("N05 point prediction and interval lengths differ")
    absolute_error = np.abs(y - reference)
    if np.ptp(width) == 0 or np.ptp(absolute_error) == 0:
        rho_value = None
    else:
        rho = spearmanr(width, absolute_error)
        rho_value = float(rho.statistic) if np.isfinite(rho.statistic) else None
    return {
        "n": int(len(y)), "coverage_80": float(inside.mean()),
        "mean_width": float(width.mean()), "median_width": float(np.median(width)),
        "mean_absolute_reference_error": float(absolute_error.mean()),
        "width_error_reference": "provided point prediction" if point_prediction is not None else "interval midpoint",
        "width_absolute_error_spearman": rho_value,
        "width_absolute_error_spearman_reason": None if rho_value is not None else "constant values",
        "pinball_lower_alpha_0_1": pinball_loss(y, lo, 0.1),
        "pinball_upper_alpha_0_9": pinball_loss(y, hi, 0.9),
    }


def safe_interval_metrics(target: Any, lower: Any, upper: Any,
                          point_prediction: Any | None = None) -> dict[str, Any]:
    """Return an explicit empty-group row instead of failing a descriptive table."""
    y = np.asarray(target, dtype=np.float64)
    if y.ndim != 1:
        raise ValueError("N05 group target must be one-dimensional")
    if len(y):
        return interval_metrics(y, lower, upper, point_prediction)
    return {
        "n": 0, "coverage_80": None, "mean_width": None, "median_width": None,
        "mean_absolute_reference_error": None, "width_error_reference": (
            "provided point prediction" if point_prediction is not None else "interval midpoint"
        ), "width_absolute_error_spearman": None,
        "width_absolute_error_spearman_reason": "empty group",
        "pinball_lower_alpha_0_1": None, "pinball_upper_alpha_0_9": None,
    }


def width_rank_groups(width: Any, groups: int = 5) -> np.ndarray:
    """Assign descriptive approximately equal-count width bins, preserving ties."""
    values = _finite_vector(width, "width")
    if not isinstance(groups, int) or groups < 1:
        raise ValueError("N05 descriptive width groups must be a positive integer")
    ranks = rankdata(values, method="average")
    bins = np.floor((ranks - 1.0) * groups / len(values)).astype(np.int64) + 1
    return np.clip(bins, 1, groups)


def point_model_eligibility() -> dict[str, Any]:
    """N05 is informational and cannot be promoted into point-model selection."""
    return {"eligible": False, "reason": "information-only lane; no point-prediction promotion"}

