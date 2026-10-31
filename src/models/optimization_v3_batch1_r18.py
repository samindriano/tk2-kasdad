"""R18 past-only median location correction for the frozen V3 champion.

This adapter adds no predictor fit. Its only learned quantity is the scalar
half-median residual from four complete, earlier-cycle champion OOF blocks.
The runner owns cache provenance, row/cycle/source-cutoff validation, and the
pilot evaluation; this module owns the bounded scalar transform only. The
operational default is a read-only ``--preflight-only`` review. After MAIN
review, the runner may be invoked explicitly with ``--freeze-only`` and then
``--run-frozen``; those stages must remain separate invocations.
"""
from __future__ import annotations

from typing import Any, Mapping

import numpy as np


CARD_ID = "R18"
MIN_HISTORY_ROWS = 500
OFFSET_SCALE = 0.5
OFFSET_LIMIT = 2.0
FINAL_RANGE = (0.0, 100.0)
POLICY_BY_CUTOFF: dict[int, dict[str, Any]] = {
    67: {
        "history_cycles": (64, 65, 66, 67), "source_cutoff": 63,
        "score_cycles": tuple(range(68, 80)),
    },
    75: {
        "history_cycles": (72, 73, 74, 75), "source_cutoff": 71,
        "score_cycles": tuple(range(76, 80)),
    },
}


class R18IntegrityError(ValueError):
    """An R18 input violates its frozen temporal or alignment contract."""


class R18InsufficientHistory(RuntimeError):
    """The card's fixed support rule requires an exact-champion fallback."""


def _vector(name: str, value: Any, *, dtype: Any) -> np.ndarray:
    result = np.asarray(value, dtype=dtype)
    if result.ndim != 1:
        raise R18IntegrityError(f"{name} must be one-dimensional")
    return result


def estimate_recent_offset(
    *,
    history_rows: Any,
    history_cycle: Any,
    history_source_cutoff: Any,
    history_target: Any,
    history_prediction: Any,
    expected_history_cycles: tuple[int, ...],
    expected_source_cutoff: int,
    score_cycle: Any,
    minimum_rows: int = MIN_HISTORY_ROWS,
) -> tuple[float, dict[str, Any]]:
    """Estimate the fixed R18 offset after strict temporal/alignment checks.

    Positive ``target - prediction`` means the champion underpredicts, so the
    resulting positive offset raises future predictions.
    """
    rows = _vector("history_rows", history_rows, dtype=np.int64)
    cycles = _vector("history_cycle", history_cycle, dtype=np.int64)
    source_cutoffs = _vector("history_source_cutoff", history_source_cutoff, dtype=np.int64)
    target = _vector("history_target", history_target, dtype=np.float64)
    prediction = _vector("history_prediction", history_prediction, dtype=np.float64)
    scored_cycles = _vector("score_cycle", score_cycle, dtype=np.int64)
    aligned = (rows, cycles, source_cutoffs, target, prediction)
    if len({len(x) for x in aligned}) != 1 or len(rows) == 0:
        raise R18IntegrityError("OOF history arrays must be nonempty and row-aligned")
    if len(np.unique(rows)) != len(rows):
        raise R18IntegrityError("OOF history row positions must be unique")
    if np.any(rows < 0):
        raise R18IntegrityError("OOF history row positions cannot be negative")
    if not np.isfinite(target).all() or not np.isfinite(prediction).all():
        raise R18IntegrityError("OOF history targets and predictions must be finite")
    if len(scored_cycles) == 0:
        raise R18IntegrityError("R18 requires at least one future score cycle")

    expected = tuple(int(x) for x in expected_history_cycles)
    observed_cycles = tuple(sorted(int(x) for x in np.unique(cycles)))
    if observed_cycles != expected:
        raise R18InsufficientHistory(
            f"Expected four complete OOF cycles {expected}; found {observed_cycles}"
        )
    if len(rows) < int(minimum_rows):
        raise R18InsufficientHistory(
            f"Aligned OOF support {len(rows)} is below minimum {minimum_rows}"
        )
    if not np.array_equal(np.unique(source_cutoffs), np.asarray([expected_source_cutoff], dtype=np.int64)):
        raise R18IntegrityError("OOF rows do not share the expected source fit cutoff")
    if np.any(source_cutoffs >= cycles):
        raise R18IntegrityError("An OOF source cutoff is not earlier than its scored cycle")
    if int(expected_source_cutoff) >= int(np.min(scored_cycles)):
        raise R18IntegrityError("The OOF source cutoff is not earlier than future score cycles")
    if np.any(cycles >= int(np.min(scored_cycles))):
        raise R18IntegrityError("An OOF history cycle overlaps or follows future score cycles")
    if np.any((scored_cycles < 0) | (scored_cycles > 10_000)):
        raise R18IntegrityError("Invalid future score cycle values")

    raw_median = float(np.median(target - prediction))
    delta = float(np.clip(OFFSET_SCALE * raw_median, -OFFSET_LIMIT, OFFSET_LIMIT))
    if not np.isfinite(delta):
        raise R18IntegrityError("R18 offset is nonfinite")
    return delta, {
        "history_rows": int(len(rows)),
        "history_cycles": list(expected),
        "history_source_cutoff": int(expected_source_cutoff),
        "score_cycle_min": int(np.min(scored_cycles)),
        "score_cycle_max": int(np.max(scored_cycles)),
        "median_target_minus_prediction": raw_median,
        "offset_scale": OFFSET_SCALE,
        "offset_limit": OFFSET_LIMIT,
        "delta": delta,
        "positive_delta_raises_underpredictions": True,
    }


def apply_offset(baseline_prediction: Any, delta: float) -> np.ndarray:
    """Apply a scalar location correction and enforce the target range."""
    baseline = _vector("baseline_prediction", baseline_prediction, dtype=np.float64)
    if len(baseline) == 0 or not np.isfinite(baseline).all():
        raise R18IntegrityError("Champion predictions must be nonempty and finite")
    if np.any((baseline < 0.0) | (baseline > 100.0)):
        raise R18IntegrityError("Exact-champion predictions must already be within [0,100]")
    if not np.isfinite(delta) or abs(float(delta)) > OFFSET_LIMIT:
        raise R18IntegrityError("R18 delta must be finite and capped within [-2,2]")
    result = np.clip(baseline + float(delta), *FINAL_RANGE)
    if not np.isfinite(result).all():
        raise R18IntegrityError("R18 clipped predictions are nonfinite")
    return result


def predict_with_policy(
    *, cutoff: int, baseline_prediction: Any, score_cycle: Any,
    history: Mapping[str, Any] | None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Apply R18 at supported late cutoffs; otherwise return exact champion.

    Cutoff 55 and early/unsupported blocks use the exact champion unchanged.
    The late pilot only permits cutoff 67 and 75. Invalid row/cutoff mappings
    fail closed; only genuinely absent/undersupported history falls back.
    """
    baseline = _vector("baseline_prediction", baseline_prediction, dtype=np.float64)
    cycles = _vector("score_cycle", score_cycle, dtype=np.int64)
    if baseline.shape != cycles.shape or len(baseline) == 0:
        raise R18IntegrityError("Score cycles and exact-champion predictions are misaligned")
    if not np.isfinite(baseline).all() or np.any((baseline < 0.0) | (baseline > 100.0)):
        raise R18IntegrityError("Exact-champion predictions must be finite and in [0,100]")
    cutoff = int(cutoff)
    if cutoff == 55:
        if tuple(sorted(np.unique(cycles).tolist())) != tuple(range(56, 68)):
            raise R18IntegrityError("Cutoff55 unchanged view must score exactly cycles 56–67")
        return baseline.copy(), {
            "status": "EXACT_CHAMPION_FALLBACK",
            "reason": "cutoff55_unchanged",
            "delta": 0.0,
            "support_rows": 0,
        }
    if cutoff not in POLICY_BY_CUTOFF:
        return baseline.copy(), {
            "status": "EXACT_CHAMPION_FALLBACK",
            "reason": "unsupported_early_or_unlisted_cutoff",
            "delta": 0.0,
            "support_rows": 0,
        }
    frozen = POLICY_BY_CUTOFF[cutoff]
    if tuple(sorted(np.unique(cycles).tolist())) != tuple(frozen["score_cycles"]):
        raise R18IntegrityError(f"Cutoff {cutoff} score cycles differ from the frozen LATE view")
    if history is None:
        return baseline.copy(), {
            "status": "EXACT_CHAMPION_FALLBACK",
            "reason": "OOF_history_unavailable",
            "delta": 0.0,
            "support_rows": 0,
        }

    try:
        delta, audit = estimate_recent_offset(
            history_rows=history["rows"],
            history_cycle=history["cycle"],
            history_source_cutoff=history["source_cutoff"],
            history_target=history["target"],
            history_prediction=history["prediction"],
            expected_history_cycles=frozen["history_cycles"],
            expected_source_cutoff=int(frozen["source_cutoff"]),
            score_cycle=cycles,
        )
    except R18InsufficientHistory as exc:
        return baseline.copy(), {
            "status": "EXACT_CHAMPION_FALLBACK",
            "reason": "insufficient_or_incomplete_OOF_history",
            "support_detail": str(exc),
            "delta": 0.0,
            "support_rows": int(len(np.asarray(history.get("rows", [])))),
        }
    prediction = apply_offset(baseline, delta)
    return prediction, {"status": "CORRECTED", **audit}


def fit_predict(
    fit_x: Any, fit_y: Any, score_x: Any, *, controls: Mapping[str, Any],
    card_id: str, cutoff: int,
) -> dict[str, Any]:
    """Batch1 adapter API. No predictor or preprocessing estimator is fit."""
    if str(card_id) != CARD_ID:
        raise R18IntegrityError(f"R18 adapter cannot serve card {card_id!r}")
    del fit_x, fit_y, score_x  # generic engine arguments are intentionally unused
    required = {"baseline_prediction", "score_cycle"}
    missing = sorted(required - set(controls))
    if missing:
        raise R18IntegrityError(f"R18 controls omit required fields: {missing}")
    frozen = POLICY_BY_CUTOFF.get(int(cutoff))
    history = None
    if frozen is not None:
        history_fields = {"history_rows", "history_cycle", "history_source_cutoff",
                          "history_target", "history_prediction"}
        if history_fields.issubset(controls):
            history = {
                "rows": controls["history_rows"],
                "cycle": controls["history_cycle"],
                "source_cutoff": controls["history_source_cutoff"],
                "target": controls["history_target"],
                "prediction": controls["history_prediction"],
            }
    prediction, state = predict_with_policy(
        cutoff=int(cutoff), baseline_prediction=controls["baseline_prediction"],
        score_cycle=controls["score_cycle"], history=history,
    )
    return {
        "prediction": prediction,
        "fit_predictor_count": 0,
        "fit_metadata": {"predictor_fits": 0, "preprocessing_fits": 0},
        "state": {"card_id": CARD_ID, "policy": "R_CALIBRATE", **state},
        "audit_arrays": {"delta": np.full(len(prediction), float(state["delta"]), dtype=np.float64)},
    }
