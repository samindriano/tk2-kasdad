"""R17 sparse LP calibration for exact prequential champion predictions."""
from __future__ import annotations

import hashlib
import json
import time
import warnings
from typing import Any, Mapping

import numpy as np
from scipy import sparse
from scipy.optimize import OptimizeWarning, linprog


CARD_ID = "R17"
FULL_CUTOFFS = (55, 59, 63, 67, 71, 75)
PILOT_CUTOFFS = (67, 75)
MIN_HISTORY_ROWS = 5_000
FINAL_RANGE = (0.0, 100.0)
SCORE_CYCLES_BY_CUTOFF = {
    55: tuple(range(56, 68)),
    59: tuple(range(60, 64)),
    63: tuple(range(64, 68)),
    67: tuple(range(68, 80)),
    71: tuple(range(72, 76)),
    75: tuple(range(76, 80)),
}
SOURCE_CUTOFF_BY_CYCLE = {
    **{cycle: 55 for cycle in range(56, 60)},
    **{cycle: 59 for cycle in range(60, 64)},
    **{cycle: 63 for cycle in range(64, 68)},
    **{cycle: 67 for cycle in range(68, 72)},
    **{cycle: 71 for cycle in range(72, 76)},
    **{cycle: 75 for cycle in range(76, 80)},
}
FEATURE_NAMES = ("cycle_code",)  # Generic harness context; calibration uses OOF rows only.


class R17IntegrityError(ValueError):
    """Raised when history, score rows, or calibration inputs are not exact."""


def _vector(name: str, values: Any, *, dtype: Any) -> np.ndarray:
    result = np.asarray(values, dtype=dtype)
    if result.ndim != 1:
        raise R17IntegrityError(f"{name} must be one-dimensional")
    return result


def _array_digest(values: Any) -> str:
    array = np.asarray(values)
    if array.dtype == object:
        payload = "\n".join(str(value) for value in array.tolist()).encode("utf-8")
        descriptor = b"utf8\0"
    else:
        stable = np.ascontiguousarray(array)
        payload = stable.tobytes()
        descriptor = stable.dtype.str.encode("ascii") + b"\0" + repr(stable.shape).encode("ascii") + b"\0"
    return hashlib.sha256(descriptor + payload).hexdigest()


def _validate_history(
    *, cutoff: int, history: Mapping[str, Any],
    source_cutoff_by_cycle: Mapping[int, int], score_cycles: np.ndarray,
    cycle_by_row: np.ndarray, minimum_rows: int,
) -> dict[str, np.ndarray]:
    rows = _vector("history_rows", history["rows"], dtype=np.int64)
    cycles = _vector("history_cycles", history["cycles"], dtype=np.int64)
    prediction = _vector("history_prediction", history["prediction"], dtype=np.float64)
    if cutoff == 55:
        if len(rows) or len(cycles) or len(prediction):
            raise R17IntegrityError("Cutoff 55 has no exact champion OOF history and must fall back unchanged")
        return {"rows": rows, "cycles": cycles, "prediction": prediction,
                "source_cutoff": np.empty(0, dtype=np.int64)}
    expected_cycles = np.arange(56, cutoff + 1, dtype=np.int64)
    if not (rows.shape == cycles.shape == prediction.shape) or len(rows) == 0:
        raise R17IntegrityError("R17 OOF history arrays must be nonempty and row-aligned")
    if len(np.unique(rows)) != len(rows) or np.any(rows < 0) or np.any(rows >= len(cycle_by_row)):
        raise R17IntegrityError("R17 OOF history rows must be unique and within official Train bounds")
    expected_rows = np.flatnonzero(np.isin(cycle_by_row, expected_cycles)).astype(np.int64)
    if not np.array_equal(rows, expected_rows):
        raise R17IntegrityError("R17 OOF history rows do not exactly cover the expected Train cycle rows")
    if not np.array_equal(cycles, cycle_by_row[rows]):
        raise R17IntegrityError("R17 OOF cycle values do not align to the exact Train row positions")
    if not np.array_equal(np.unique(cycles), expected_cycles):
        raise R17IntegrityError(
            f"R17 history must contain every OOF cycle 56..{cutoff}; observed={np.unique(cycles).tolist()}"
        )
    if not np.isfinite(prediction).all() or np.any((prediction < 0.0) | (prediction > 100.0)):
        raise R17IntegrityError("R17 exact-champion OOF predictions must be finite and in [0,100]")
    try:
        source = np.asarray([source_cutoff_by_cycle[int(cycle)] for cycle in cycles], dtype=np.int64)
    except KeyError as exc:
        raise R17IntegrityError(f"No exact OOF source cutoff recorded for cycle {exc.args[0]}") from exc
    if np.any(source >= cycles) or np.any(source > cutoff):
        raise R17IntegrityError("R17 OOF history contains a same/future or beyond-cutoff source fit")
    expected_source = np.asarray([SOURCE_CUTOFF_BY_CYCLE.get(int(cycle), -1) for cycle in cycles], dtype=np.int64)
    if not np.array_equal(source, expected_source):
        raise R17IntegrityError("R17 OOF source cutoffs differ from the canonical six-fold mapping")
    if len(score_cycles) == 0 or np.any(cycles >= int(np.min(score_cycles))):
        raise R17IntegrityError("R17 calibration history overlaps/follows its score cycles")
    if len(rows) < minimum_rows:
        # Complete, correctly mapped but small history is a declared fallback.
        return {"rows": rows, "cycles": cycles, "source_cutoff": source,
                "prediction": prediction}
    return {"rows": rows, "cycles": cycles, "source_cutoff": source,
            "prediction": prediction}


def solve_nonnegative_l1(
    target: Any, prediction: Any, *, threads: int = 2,
) -> dict[str, Any]:
    """Solve min sum|y-a-b*p| using a sparse LP and constrain b >= 0."""
    y = _vector("history target", target, dtype=np.float64)
    p = _vector("history prediction", prediction, dtype=np.float64)
    if y.shape != p.shape or len(y) == 0:
        raise R17IntegrityError("LP history target and prediction must be nonempty and aligned")
    if not np.isfinite(y).all() or not np.isfinite(p).all():
        raise R17IntegrityError("LP history target and prediction must be finite")
    if np.any((y < 0.0) | (y > 100.0)) or np.any((p < 0.0) | (p > 100.0)):
        raise R17IntegrityError("LP history target/prediction values must be in [0,100]")
    n = len(y)
    positions = np.arange(n, dtype=np.int64)
    # Variables are [a, b, u_0..u_(n-1)]. The two inequalities encode
    # y-a-b*p <= u and a+b*p-y <= u. The matrix has 6n nonzero entries.
    row_index = np.concatenate((
        positions, positions, positions,
        n + positions, n + positions, n + positions,
    ))
    column_index = np.concatenate((
        np.zeros(n, dtype=np.int64), np.ones(n, dtype=np.int64), 2 + positions,
        np.zeros(n, dtype=np.int64), np.ones(n, dtype=np.int64), 2 + positions,
    ))
    values = np.concatenate((
        -np.ones(n, dtype=np.float64), -p, -np.ones(n, dtype=np.float64),
        np.ones(n, dtype=np.float64), p, -np.ones(n, dtype=np.float64),
    ))
    constraints = sparse.coo_matrix(
        (values, (row_index, column_index)), shape=(2 * n, n + 2), dtype=np.float64
    ).tocsr()
    rhs = np.concatenate((-y, y))
    objective = np.concatenate((np.zeros(2, dtype=np.float64), np.ones(n, dtype=np.float64)))
    bounds = [(None, None), (0.0, None), *((0.0, None) for _ in range(n))]
    started = time.perf_counter()
    try:
        # SciPy passes HiGHS-specific options through and warns because it does
        # not validate the `threads` key itself. Keep the requested thread cap.
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=OptimizeWarning, message="Unrecognized options detected:.*threads")
            result = linprog(
                objective, A_ub=constraints, b_ub=rhs, bounds=bounds,
                method="highs", options={"threads": int(threads)},
            )
    except Exception as exc:  # A solver failure uses the declared exact-champion fallback.
        return {
            "success": False, "solver": "scipy.optimize.linprog(method='highs')",
            "solver_status": None, "solver_message": f"{type(exc).__name__}: {exc}",
            "intercept": 0.0, "slope": 1.0, "objective_l1_sum": None,
            "iterations": None, "solver_seconds": float(time.perf_counter() - started),
            "history_rows": int(n), "constraint_shape": [int(2 * n), int(n + 2)],
            "constraint_nnz": int(constraints.nnz),
            "constraint_csr_bytes": int(constraints.data.nbytes + constraints.indices.nbytes + constraints.indptr.nbytes),
            "fallback": True, "fallback_reason": "solver_exception",
        }
    elapsed = float(time.perf_counter() - started)
    if not bool(getattr(result, "success", False)):
        return {
            "success": False, "solver": "scipy.optimize.linprog(method='highs')",
            "solver_status": int(getattr(result, "status", -1)),
            "solver_message": str(getattr(result, "message", "missing solver message")),
            "intercept": 0.0, "slope": 1.0, "objective_l1_sum": None,
            "iterations": getattr(result, "nit", None), "solver_seconds": elapsed,
            "history_rows": int(n), "constraint_shape": [int(2 * n), int(n + 2)],
            "constraint_nnz": int(constraints.nnz),
            "constraint_csr_bytes": int(constraints.data.nbytes + constraints.indices.nbytes + constraints.indptr.nbytes),
            "fallback": True, "fallback_reason": "solver_non_success",
        }
    solution = np.asarray(getattr(result, "x", []), dtype=np.float64)
    if solution.shape != (n + 2,) or not np.isfinite(solution).all():
        valid = False
    else:
        valid = bool(solution[1] >= -1e-9)
    intercept = float(solution[0]) if valid else 0.0
    slope = max(0.0, float(solution[1])) if valid else 1.0
    residual_objective = float(np.abs(y - intercept - slope * p).sum()) if valid else None
    raw_objective = getattr(result, "fun", None)
    try:
        lp_objective = float(raw_objective) if raw_objective is not None and np.isfinite(float(raw_objective)) else None
    except (TypeError, ValueError, OverflowError):
        lp_objective = None
    objective_matches = bool(
        valid and lp_objective is not None and residual_objective is not None
        and np.isclose(residual_objective, lp_objective, atol=max(1e-5, n * 1e-9), rtol=1e-8)
    )
    if not objective_matches:
        valid = False
    return {
        "success": bool(valid), "solver": "scipy.optimize.linprog(method='highs')",
        "solver_status": int(getattr(result, "status", -1)), "solver_message": str(getattr(result, "message", "missing solver message")),
        "intercept": intercept if valid else 0.0,
        "slope": slope if valid else 1.0,
        "objective_l1_sum": residual_objective if valid else lp_objective,
        "iterations": getattr(result, "nit", None), "solver_seconds": elapsed,
        "history_rows": int(n), "constraint_shape": [int(2 * n), int(n + 2)],
        "constraint_nnz": int(constraints.nnz),
        "constraint_csr_bytes": int(constraints.data.nbytes + constraints.indices.nbytes + constraints.indptr.nbytes),
        "fallback": not valid,
        "fallback_reason": None if valid else "invalid_or_inconsistent_solver_solution",
    }


def apply_affine(baseline_prediction: Any, intercept: float, slope: float) -> np.ndarray:
    baseline = _vector("baseline_prediction", baseline_prediction, dtype=np.float64)
    if len(baseline) == 0 or not np.isfinite(baseline).all():
        raise R17IntegrityError("R17 baseline predictions must be nonempty and finite")
    if np.any((baseline < 0.0) | (baseline > 100.0)):
        raise R17IntegrityError("R17 exact-champion baseline must lie in [0,100]")
    if not np.isfinite(intercept) or not np.isfinite(slope) or slope < 0.0:
        raise R17IntegrityError("R17 affine coefficients must be finite with a nonnegative slope")
    return np.clip(float(intercept) + float(slope) * baseline, *FINAL_RANGE)


def fit_predict(
    fit_x: Any,
    fit_y: Any,
    score_x: Any,
    *,
    controls: Mapping[str, Any],
    card_id: str,
    cutoff: int,
    history_by_cutoff: Mapping[int, Mapping[str, Any]],
    source_cutoff_by_cycle: Mapping[int, int],
    cycle_by_row: Any,
    minimum_rows: int = MIN_HISTORY_ROWS,
) -> dict[str, Any]:
    """Calibrate only on validated OOF rows whose labels are in the fit prefix."""
    cutoff = int(cutoff)
    if str(card_id) != CARD_ID or cutoff not in FULL_CUTOFFS:
        raise R17IntegrityError(f"R17 cannot serve card={card_id!r}, cutoff={cutoff!r}")
    fit_rows = _vector("fit row positions", fit_x.index.to_numpy(), dtype=np.int64)
    score_rows = _vector("score row positions", score_x.index.to_numpy(), dtype=np.int64)
    labels = _vector("fit_y", fit_y, dtype=np.float64)
    baseline = _vector("baseline_prediction", controls["baseline_prediction"], dtype=np.float64)
    score_cycles = _vector("score_cycle", controls["score_cycle"], dtype=np.int64)
    cycle_values = _vector("cycle_by_row", cycle_by_row, dtype=np.int64)
    if (len(fit_rows) != len(labels) or len(score_rows) != len(baseline)
            or len(score_rows) != len(score_cycles)
            or np.any(fit_rows < 0) or np.any(score_rows < 0)
            or np.any(fit_rows >= len(cycle_values)) or np.any(score_rows >= len(cycle_values))):
        raise R17IntegrityError("R17 Train-prefix, score, and cycle vectors are misaligned")
    if (len(np.unique(fit_rows)) != len(fit_rows) or len(np.unique(score_rows)) != len(score_rows)
            or np.intersect1d(fit_rows, score_rows).size):
        raise R17IntegrityError("R17 fit and score row positions must be unique and disjoint")
    expected_fit_rows = np.flatnonzero(cycle_values <= cutoff).astype(np.int64)
    if not np.array_equal(fit_rows, expected_fit_rows):
        raise R17IntegrityError("R17 fit rows do not exactly match the temporal Train prefix")
    if not np.array_equal(cycle_values[score_rows], score_cycles):
        raise R17IntegrityError("R17 score-cycle values do not align with exact Train rows")
    if not np.array_equal(pd_numeric_int(fit_x["cycle_code"]), cycle_values[fit_rows]):
        raise R17IntegrityError("R17 fit cycle features do not align with Train row positions")
    if not np.isfinite(labels).all():
        raise R17IntegrityError("R17 fit-prefix labels must be finite")
    if not np.isfinite(baseline).all() or np.any((baseline < 0.0) | (baseline > 100.0)):
        raise R17IntegrityError("R17 baseline probabilities must be finite and within [0,100]")
    expected_score_cycles = np.asarray(SCORE_CYCLES_BY_CUTOFF[cutoff], dtype=np.int64)
    if not np.array_equal(np.unique(score_cycles), expected_score_cycles):
        raise R17IntegrityError(f"R17 score cycles differ from the frozen cutoff-{cutoff} policy")
    expected_score_rows = np.flatnonzero(np.isin(cycle_values, expected_score_cycles)).astype(np.int64)
    if not np.array_equal(score_rows, expected_score_rows):
        raise R17IntegrityError("R17 score rows do not exactly match the frozen cycle windows")

    history = _validate_history(
        cutoff=cutoff, history=history_by_cutoff[cutoff],
        source_cutoff_by_cycle=source_cutoff_by_cycle,
        score_cycles=score_cycles, cycle_by_row=cycle_values, minimum_rows=minimum_rows,
    )
    base_state = {
        "card_id": CARD_ID, "cutoff": cutoff,
        "history_cycles": [int(x) for x in np.unique(history["cycles"])],
        "history_rows": int(len(history["rows"])),
        "history_row_positions_sha256": _array_digest(history["rows"]),
        "history_cycle_sha256": _array_digest(history["cycles"]),
        "history_source_cutoff_sha256": _array_digest(history["source_cutoff"]),
        "history_prediction_sha256": _array_digest(history["prediction"]),
        "score_rows": int(len(score_rows)),
        "score_row_positions_sha256": _array_digest(score_rows),
        "baseline_prediction_sha256": _array_digest(baseline),
        "minimum_history_rows": int(minimum_rows),
        "threshold_or_weight_search": False,
        "outer_score_labels_passed_to_calibrator": False,
        "OOF_target_cache_opened": False,
        "test_data_used": False,
    }

    if cutoff == 55 or len(history["rows"]) < int(minimum_rows):
        reason = "no_exact_champion_history" if cutoff == 55 else "history_below_5000_rows"
        output = baseline.copy()
        solver_state = {
            "success": False, "fallback": True, "fallback_reason": reason,
            "intercept": 0.0, "slope": 1.0, "solver": None,
            "solver_status": None, "solver_message": reason,
            "objective_l1_sum": None, "iterations": None,
            "solver_seconds": 0.0, "history_rows": int(len(history["rows"])),
            "constraint_shape": None, "constraint_nnz": 0,
            "constraint_csr_bytes": 0,
        }
        fit_count = 0
    else:
        fit_positions = {int(row): position for position, row in enumerate(fit_rows)}
        missing_rows = [int(row) for row in history["rows"] if int(row) not in fit_positions]
        if missing_rows:
            raise R17IntegrityError(f"R17 OOF history rows are outside fit prefix: {missing_rows[:5]}")
        if np.any(history["cycles"] > cutoff):
            raise R17IntegrityError("R17 calibration includes a future-cycle target")
        selected_target = np.asarray(
            [labels[fit_positions[int(row)]] for row in history["rows"]], dtype=np.float64
        )
        if not np.isfinite(selected_target).all() or np.any((selected_target < 0.0) | (selected_target > 100.0)):
            raise R17IntegrityError("R17 selected past OOF targets are invalid")
        solver_state = solve_nonnegative_l1(selected_target, history["prediction"], threads=2)
        output = apply_affine(baseline, solver_state["intercept"], solver_state["slope"])
        fit_count = 1
        base_state["history_target_sha256"] = _array_digest(selected_target)

    base_state.update(solver_state)
    base_state["candidate_prediction_sha256"] = _array_digest(output)
    base_state["fit_predictor_count"] = int(fit_count)
    base_state["state_sha256"] = hashlib.sha256(json.dumps(
        base_state, sort_keys=True, allow_nan=False, separators=(",", ":")
    ).encode("utf-8")).hexdigest()
    return {
        "prediction": output,
        "fit_predictor_count": fit_count,
        "fit_metadata": {
            "calibration_solve_count": int(fit_count),
            "preprocessing_fits": 0,
            "solver": solver_state.get("solver"),
            "solver_status": solver_state.get("solver_status"),
            "history_rows": int(len(history["rows"])),
        },
        "audit_arrays": {
            "intercept": np.full(len(output), solver_state["intercept"], dtype=np.float64),
            "slope": np.full(len(output), solver_state["slope"], dtype=np.float64),
            "solver_success": np.full(len(output), int(bool(solver_state["success"])), dtype=np.int8),
        },
        "state": base_state,
    }


def pd_numeric_int(values: Any) -> np.ndarray:
    """Convert an already numeric cycle feature without importing pandas in the model module."""
    result = np.asarray(values)
    try:
        numeric = result.astype(np.int64)
    except (TypeError, ValueError) as exc:
        raise R17IntegrityError("R17 cycle feature must be integer-valued") from exc
    if not np.array_equal(result, numeric):
        raise R17IntegrityError("R17 cycle feature contains noninteger values")
    return numeric


def reconstruct_r17(
    *, baseline_prediction: Any, history_target: Any, history_prediction: Any,
    fallback: bool = False,
) -> tuple[np.ndarray, float, float, dict[str, Any]]:
    """Independent report-time reconstruction from a stored history and control."""
    baseline = _vector("baseline_prediction", baseline_prediction, dtype=np.float64)
    if fallback:
        return baseline.copy(), 0.0, 1.0, {"status": "FALLBACK_IDENTITY", "solver_called": False}
    fit = solve_nonnegative_l1(history_target, history_prediction, threads=2)
    prediction = apply_affine(baseline, fit["intercept"], fit["slope"])
    return prediction, float(fit["intercept"]), float(fit["slope"]), fit


__all__ = [
    "CARD_ID", "FEATURE_NAMES", "FINAL_RANGE", "FULL_CUTOFFS",
    "MIN_HISTORY_ROWS", "PILOT_CUTOFFS", "R17IntegrityError",
    "SCORE_CYCLES_BY_CUTOFF", "apply_affine", "fit_predict",
    "reconstruct_r17", "solve_nonnegative_l1",
]
