"""Pure R-NBR math and validation primitives used by synthetic QA in B0.

This module has no dataset loader, target access, ledger mutation, fitted text
transformer, or official inference entry point. The only current caller is the
synthetic lifecycle script. A later official runner requires a separately
authorized budget and a separately frozen source boundary.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np


PILOT_ORIGINS = (67, 75)
PILOT_QUERY_HORIZONS = {67: 79, 75: 79}
PILOT_QUERY_WINDOWS = {67: (68, 79), 75: (76, 79)}
PILOT_ARMS = ("observed", "shuffled")
ALPHA = 0.25
CORRECTION_CAP = 5.0
MAX_NEIGHBORS = 32
MIN_NEIGHBORS = 5
SHUFFLE_SEED = 42
EXPECTED_SOURCE_CUTOFFS = {
    **{cycle: 55 for cycle in range(56, 60)},
    **{cycle: 59 for cycle in range(60, 64)},
    **{cycle: 63 for cycle in range(64, 68)},
    **{cycle: 67 for cycle in range(68, 72)},
    **{cycle: 71 for cycle in range(72, 76)},
    **{cycle: 75 for cycle in range(76, 80)},
}


class RnbrProtocolError(ValueError):
    """A frozen identity, geometry, or prediction invariant was violated."""


@dataclass(frozen=True)
class SyntheticCase:
    """In-memory synthetic-only fixture; it cannot load or name official data."""

    dataset_tag: str
    origin: int
    donor_rows: np.ndarray
    donor_ids: np.ndarray
    donor_cycles: np.ndarray
    donor_source_cutoffs: np.ndarray
    donor_present: np.ndarray
    donor_r06_predictions: np.ndarray
    donor_targets: np.ndarray
    query_rows: np.ndarray
    query_ids: np.ndarray
    query_cycles: np.ndarray
    query_windows: np.ndarray
    query_missing: np.ndarray
    query_r06_predictions: np.ndarray
    neighbor_rows: np.ndarray
    neighbor_ids: np.ndarray
    neighbor_cosines: np.ndarray
    neighbor_counts: np.ndarray

    def __post_init__(self) -> None:
        if self.dataset_tag != "RNBR_SYNTHETIC_FIXTURE_V1":
            raise RnbrProtocolError("Only the named synthetic fixture is accepted in B0")
        if self.origin not in PILOT_ORIGINS:
            raise RnbrProtocolError("Synthetic pilot origin must be 67 or 75")


def _vector(values: Any, name: str, *, dtype: Any | None = None) -> np.ndarray:
    out = np.asarray(values, dtype=dtype)
    if out.ndim != 1:
        raise RnbrProtocolError(f"{name} must be one-dimensional")
    return out


def _aligned(values: Any, name: str, n: int, *, dtype: Any | None = None) -> np.ndarray:
    out = _vector(values, name, dtype=dtype)
    if len(out) != n:
        raise RnbrProtocolError(f"{name} has {len(out)} rows; expected {n}")
    return out


def _finite(values: np.ndarray, name: str) -> None:
    if not np.isfinite(values).all():
        raise RnbrProtocolError(f"{name} contains a nonfinite value")


def fixed_prediction_decile(predictions: Any) -> np.ndarray:
    """Fixed, label-free bins: min(9, floor(clip(prediction,0,100)/10))."""
    values = _vector(predictions, "R06 predictions", dtype=np.float64)
    _finite(values, "R06 predictions")
    return np.minimum(9, np.floor(np.clip(values, 0.0, 100.0) / 10.0)).astype(np.int8)


def weighted_median_residual(residuals: Any, cosines: Any, source_rows: Any) -> float:
    """Weighted median of residuals using cosine-squared weights.

    Equal residuals are grouped after stable ordering by residual then source
    row. If cumulative mass is half to numerical tolerance (1e-12 of total),
    return the midpoint to the next distinct residual.
    """
    residual = _vector(residuals, "neighbor residuals", dtype=np.float64)
    cosine = _aligned(cosines, "neighbor cosines", len(residual), dtype=np.float64)
    rows = _aligned(source_rows, "neighbor source rows", len(residual), dtype=np.int64)
    _finite(residual, "neighbor residuals")
    _finite(cosine, "neighbor cosines")
    if len(residual) == 0 or len(np.unique(rows)) != len(rows):
        raise RnbrProtocolError("Weighted median requires nonempty, unique neighbor rows")
    if np.any((cosine <= 0.0) | (cosine > 1.0 + 1e-12)):
        raise RnbrProtocolError("Weighted median cosines must be positive and at most one")
    weights = np.square(cosine)
    order = np.lexsort((rows, residual))
    ordered_values = residual[order]
    ordered_weights = weights[order]
    unique_values, first = np.unique(ordered_values, return_index=True)
    grouped_weights = np.add.reduceat(ordered_weights, first)
    total = float(grouped_weights.sum())
    if not np.isfinite(total) or total <= 0.0:
        raise RnbrProtocolError("Weighted median total weight must be finite and positive")
    half = total / 2.0
    tolerance = 1e-12 * total
    cumulative = 0.0
    for index, (value, weight) in enumerate(zip(unique_values, grouped_weights)):
        cumulative += float(weight)
        if cumulative > half + tolerance:
            return float(value)
        if abs(cumulative - half) <= tolerance:
            if index + 1 < len(unique_values):
                return float((value + unique_values[index + 1]) / 2.0)
            return float(value)
    return float(unique_values[-1])


def correction_delta(weighted_median: float) -> float:
    """Frozen formula: delta = clip(0.25 * weighted_median, -5, +5)."""
    value = float(weighted_median)
    if not np.isfinite(value):
        raise RnbrProtocolError("Weighted median must be finite")
    return float(np.clip(ALPHA * value, -CORRECTION_CAP, CORRECTION_CAP))


def shuffle_residuals_within_cycle_decile(
    residuals: Any,
    cycles: Any,
    r06_predictions: Any,
    source_rows: Any,
    *,
    seed: int = SHUFFLE_SEED,
) -> np.ndarray:
    """Shuffle values within cycle × fixed R06-prediction decile.

    Cell traversal is ascending cycle then decile; row order within each cell
    is ascending source position. A new seed-42 generator is used per origin.
    """
    residual = _vector(residuals, "donor residuals", dtype=np.float64)
    n = len(residual)
    cycle = _aligned(cycles, "donor cycles", n, dtype=np.int64)
    prediction = _aligned(r06_predictions, "donor R06 predictions", n, dtype=np.float64)
    rows = _aligned(source_rows, "donor source rows", n, dtype=np.int64)
    _finite(residual, "donor residuals")
    _finite(prediction, "donor R06 predictions")
    if len(np.unique(rows)) != n or not isinstance(seed, (int, np.integer)):
        raise RnbrProtocolError("Shuffle requires unique source rows and an integer seed")
    deciles = fixed_prediction_decile(prediction)
    shuffled = residual.copy()
    rng = np.random.default_rng(int(seed))
    for cycle_value in np.unique(cycle):
        for decile_value in range(10):
            members = np.flatnonzero((cycle == cycle_value) & (deciles == decile_value))
            if len(members) < 2:
                continue
            members = members[np.argsort(rows[members], kind="stable")]
            shuffled[members] = residual[members][rng.permutation(len(members))]
    return shuffled


def _canonical_neighbor_id(raw_id: Any) -> str:
    # Existing A02 geometry records typed NumPy Unicode IDs as "str_:<id>".
    return f"str_:{str(raw_id)}"


def validate_saved_geometry(
    *,
    origin: int,
    donor_rows: Any,
    donor_ids: Any,
    donor_cycles: Any,
    donor_source_cutoffs: Any,
    donor_present: Any,
    query_rows: Any,
    query_ids: Any,
    query_cycles: Any,
    query_missing: Any,
    neighbor_rows: Any,
    neighbor_ids: Any,
    neighbor_cosines: Any,
    neighbor_counts: Any,
) -> dict[str, int]:
    """Fail-closed row/ID/cycle/source validation for one saved geometry block."""
    if origin not in PILOT_ORIGINS:
        raise RnbrProtocolError("Only pilot origins 67 and 75 are permitted")
    dr = _vector(donor_rows, "donor rows", dtype=np.int64)
    di = _aligned(donor_ids, "donor IDs", len(dr)).astype(str)
    dc = _aligned(donor_cycles, "donor cycles", len(dr), dtype=np.int64)
    ds = _aligned(donor_source_cutoffs, "donor R06 source cutoffs", len(dr), dtype=np.int64)
    dp = _aligned(donor_present, "donor present mask", len(dr), dtype=bool)
    qr = _vector(query_rows, "query rows", dtype=np.int64)
    qi = _aligned(query_ids, "query IDs", len(qr)).astype(str)
    qc = _aligned(query_cycles, "query cycles", len(qr), dtype=np.int64)
    qm = _aligned(query_missing, "query missing mask", len(qr), dtype=bool)
    nr = np.asarray(neighbor_rows, dtype=np.int64)
    ni = np.asarray(neighbor_ids).astype(str)
    nc = np.asarray(neighbor_cosines, dtype=np.float64)
    count = _aligned(neighbor_counts, "neighbor counts", len(qr), dtype=np.int64)
    if len(np.unique(dr)) != len(dr) or len(np.unique(di)) != len(di):
        raise RnbrProtocolError("Donor source rows and IDs must be unique")
    if len(np.unique(qr)) != len(qr) or len(np.unique(qi)) != len(qi):
        raise RnbrProtocolError("Query source rows and IDs must be unique")
    if np.intersect1d(dr, qr).size or set(di).intersection(qi):
        raise RnbrProtocolError("Donor and query identities overlap")
    query_horizon = PILOT_QUERY_HORIZONS[origin]
    if (np.any((dc < 56) | (dc > origin)) or np.any(ds >= dc)
            or np.any(~dp) or np.any(qc <= origin) or np.any(qc > query_horizon)):
        raise RnbrProtocolError("Donor/query temporal or present-stream boundary failed")
    for cycle, source in zip(dc, ds):
        if EXPECTED_SOURCE_CUTOFFS.get(int(cycle)) != int(source):
            raise RnbrProtocolError(f"R06 donor cutoff does not match cycle {cycle}")
    if (nr.ndim != 2 or nr.shape[0] != len(qr) or nr.shape[1] > MAX_NEIGHBORS
            or ni.shape != nr.shape or nc.shape != nr.shape or np.any(count < 0)
            or np.any(count > nr.shape[1])):
        raise RnbrProtocolError("Saved neighbor array shapes or counts are invalid")
    _finite(nc, "saved neighbor cosines")
    if np.any((nc < 0.0) | (nc > 1.0 + 1e-12)):
        raise RnbrProtocolError("Saved cosine similarities are outside [0,1]")
    donor_lookup = {int(row): i for i, row in enumerate(dr)}
    neighbor_rows_checked = 0
    for i, n_value in enumerate(count):
        k = int(n_value)
        active_rows = nr[i, :k]
        if (np.any(nr[i, k:] != -1) or np.any(ni[i, k:] != "")
                or np.any(nc[i, k:] != 0.0)):
            raise RnbrProtocolError("Saved neighbor padding is invalid")
        if qm[i] and k != 0:
            raise RnbrProtocolError("Missing query must have no neighbors")
        if k == 0:
            continue
        if np.any(active_rows < 0) or len(np.unique(active_rows)) != k:
            raise RnbrProtocolError("Active neighbor source rows must be valid and unique")
        if np.any(nc[i, :k] <= 0.0):
            raise RnbrProtocolError("Only positive-cosine neighbors may be saved")
        expected_order = sorted(range(k), key=lambda j: (-nc[i, j], active_rows[j]))
        if expected_order != list(range(k)):
            raise RnbrProtocolError("Neighbor order must be descending cosine, then ascending source row")
        for j, row in enumerate(active_rows):
            donor_index = donor_lookup.get(int(row))
            if donor_index is None:
                raise RnbrProtocolError(f"Neighbor row {row} is outside the frozen donor bank")
            if ni[i, j] != _canonical_neighbor_id(di[donor_index]):
                raise RnbrProtocolError(f"Neighbor ID does not match source row {row}")
            if dc[donor_index] >= qc[i] or dc[donor_index] > origin:
                raise RnbrProtocolError("A neighbor is not strictly past its query")
        neighbor_rows_checked += k
    return {"query_rows": int(len(qr)), "donor_rows": int(len(dr)),
            "neighbor_rows_checked": int(neighbor_rows_checked),
            "missing_queries": int(qm.sum()),
            "fallback_queries": int(np.sum(qm | (count < MIN_NEIGHBORS))),
            "status": 1}


def validate_donor_value_lineage(
    *,
    origin: int,
    donor_rows: Any,
    donor_ids: Any,
    donor_cycles: Any,
    donor_source_cutoffs: Any,
    target_rows: Any,
    target_ids: Any,
    target_cycles: Any,
    target_values: Any,
    r06_rows: Any,
    r06_ids: Any,
    r06_cycles: Any,
    r06_predictions: Any,
) -> dict[str, int]:
    """Validate donor-only target and R06 OOF vectors without opening data.

    Callers must provide target arrays already restricted to donor rows. The
    function deliberately returns accounting only, not residuals or a fitted
    memory. B0 exercises it with the named synthetic fixture only.
    """
    dr = _vector(donor_rows, "geometry donor rows", dtype=np.int64)
    di = _aligned(donor_ids, "geometry donor IDs", len(dr)).astype(str)
    dc = _aligned(donor_cycles, "geometry donor cycles", len(dr), dtype=np.int64)
    ds = _aligned(donor_source_cutoffs, "geometry donor source cutoffs", len(dr), dtype=np.int64)
    tr = _aligned(target_rows, "donor target rows", len(dr), dtype=np.int64)
    ti = _aligned(target_ids, "donor target IDs", len(dr)).astype(str)
    tc = _aligned(target_cycles, "donor target cycles", len(dr), dtype=np.int64)
    tv = _aligned(target_values, "donor target values", len(dr), dtype=np.float64)
    rr = _aligned(r06_rows, "R06 donor OOF rows", len(dr), dtype=np.int64)
    ri = _aligned(r06_ids, "R06 donor OOF IDs", len(dr)).astype(str)
    rc = _aligned(r06_cycles, "R06 donor OOF cycles", len(dr), dtype=np.int64)
    rp = _aligned(r06_predictions, "R06 donor OOF predictions", len(dr), dtype=np.float64)
    _finite(tv, "donor target values")
    _finite(rp, "R06 donor OOF predictions")
    if (np.any((tv < 0.0) | (tv > 100.0)) or np.any((rp < 0.0) | (rp > 100.0))
            or len(np.unique(dr)) != len(dr)):
        raise RnbrProtocolError("Donor targets/predictions must be unique-row finite values in [0,100]")
    if (not np.array_equal(tr, dr) or not np.array_equal(rr, dr)
            or not np.array_equal(ti, di) or not np.array_equal(ri, di)
            or not np.array_equal(tc, dc) or not np.array_equal(rc, dc)):
        raise RnbrProtocolError("Donor target or R06 OOF row/ID/cycle alignment differs from geometry")
    if origin not in PILOT_ORIGINS or np.any((dc < 56) | (dc > origin)):
        raise RnbrProtocolError("Donor value lineage is outside the frozen origin history")
    for cycle, cutoff in zip(dc, ds):
        if EXPECTED_SOURCE_CUTOFFS.get(int(cycle)) != int(cutoff) or int(cutoff) >= int(cycle):
            raise RnbrProtocolError(f"Donor R06 OOF cutoff is invalid for cycle {cycle}")
    return {"donor_rows_aligned": int(len(dr)), "target_rows_aligned": int(len(tr)),
            "r06_oof_rows_aligned": int(len(rr)), "status": 1}


def predict_from_saved_neighbors(
    *,
    donor_rows: Any,
    donor_residuals: Any,
    query_r06_predictions: Any,
    query_missing: Any,
    neighbor_rows: Any,
    neighbor_cosines: Any,
    neighbor_counts: Any,
) -> dict[str, np.ndarray]:
    """Apply only the synthetic/approved array-level correction formula.

    This function has no dataset reader. The current B0 caller supplies only
    synthetic residuals; an official caller is deliberately not implemented.
    """
    dr = _vector(donor_rows, "donor rows", dtype=np.int64)
    residual = _aligned(donor_residuals, "donor residuals", len(dr), dtype=np.float64)
    _finite(residual, "donor residuals")
    baseline = _vector(query_r06_predictions, "query R06 predictions", dtype=np.float64)
    _finite(baseline, "query R06 predictions")
    if np.any((baseline < 0.0) | (baseline > 100.0)):
        raise RnbrProtocolError("R06 baseline predictions must be in [0,100]")
    missing = _aligned(query_missing, "query missing mask", len(baseline), dtype=bool)
    rows = np.asarray(neighbor_rows, dtype=np.int64)
    cosines = np.asarray(neighbor_cosines, dtype=np.float64)
    counts = _aligned(neighbor_counts, "neighbor counts", len(baseline), dtype=np.int64)
    if (rows.ndim != 2 or rows.shape[0] != len(baseline) or rows.shape != cosines.shape
            or rows.shape[1] > MAX_NEIGHBORS or np.any(counts < 0) or np.any(counts > rows.shape[1])):
        raise RnbrProtocolError("Prediction neighbor arrays are misaligned")
    if len(np.unique(dr)) != len(dr):
        raise RnbrProtocolError("Donor rows must be unique")
    lookup = {int(row): i for i, row in enumerate(dr)}
    delta = np.zeros(len(baseline), dtype=np.float64)
    n_used = np.zeros(len(baseline), dtype=np.int16)
    for i, count_value in enumerate(counts):
        k = int(count_value)
        if missing[i] or k < MIN_NEIGHBORS:
            continue
        source_rows = rows[i, :k]
        try:
            donor_indices = np.asarray([lookup[int(row)] for row in source_rows], dtype=np.int64)
        except KeyError as exc:
            raise RnbrProtocolError("Neighbor source row is outside donor residual bank") from exc
        median = weighted_median_residual(
            residual[donor_indices], cosines[i, :k], source_rows
        )
        delta[i] = correction_delta(median)
        n_used[i] = k
    candidate = np.clip(baseline + delta, 0.0, 100.0)
    if (not np.isfinite(candidate).all() or not np.isfinite(delta).all()
            or np.any(delta[missing] != 0.0)
            or not np.array_equal(candidate[missing], baseline[missing])):
        raise RnbrProtocolError("R-NBR output or unchanged missing fallback is invalid")
    return {"baseline": baseline.copy(), "delta": delta, "prediction": candidate,
            "neighbor_count_used": n_used}


def plan_pilot_fits(*, freeze_signature: str) -> list[dict[str, Any]]:
    """Return the four requested task-memory slots; does not reserve or start them."""
    if not freeze_signature or len(freeze_signature) != 64:
        raise RnbrProtocolError("A SHA-256 freeze signature is required in the request schema")
    plan = []
    for origin in PILOT_ORIGINS:
        for arm in PILOT_ARMS:
            plan.append({
                "fit_id": f"RNBR_PILOT_E{origin}_{arm.upper()}",
                "card_id": "REGRESSION_RNBR_PILOT",
                "stage": "LATE_TWO_ORIGIN_DIAGNOSTIC",
                "category": "pilot",
                "origin": origin,
                "arm": arm,
                "task_predictor_fit_count": 1,
                "freeze_signature": freeze_signature,
                "status": "REQUEST_ONLY_NOT_RESERVED",
            })
    if len({item["fit_id"] for item in plan}) != 4:
        raise RnbrProtocolError("Pilot fit plan must contain four unique IDs")
    return plan


def evaluate_incremental_gates(metrics: Mapping[str, Any]) -> dict[str, Any]:
    """Diagnostic thresholds versus R06; this never qualifies a global model."""
    required = {"primary", "shuffle_primary", "H12_L", "windows", "groups"}
    if not required.issubset(metrics):
        raise RnbrProtocolError(f"Gate metrics missing {sorted(required - set(metrics))}")
    primary = metrics["primary"]
    shuffle = metrics["shuffle_primary"]
    h12 = metrics["H12_L"]
    checks: dict[str, bool] = {
        "primary_MAE_gain_vs_R06_at_least_0_03": float(primary["R06_mae"]) - float(primary["candidate_mae"]) >= 0.03,
        "primary_MAE_gain_vs_shuffle_at_least_0_01": float(shuffle["candidate_mae"]) - float(primary["candidate_mae"]) >= 0.01,
        "H12_L_harm_vs_R06_at_most_0_05": float(h12["candidate_mae"]) - float(h12["R06_mae"]) <= 0.05,
    }
    for name in ("W4", "W6"):
        row = metrics["windows"][name]
        checks[f"{name}_harm_vs_R06_below_0_25"] = float(row["candidate_mae"]) - float(row["R06_mae"]) < 0.25
    subgroup_checks = {}
    for name in ("missing_stream", "target_le_5", "target_ge_95"):
        row = metrics["groups"].get(name)
        if row is None or int(row["n"]) < 100:
            subgroup_checks[name] = {"supported": False, "pass": None,
                                     "n": 0 if row is None else int(row["n"])}
        else:
            harm = float(row["candidate_mae"]) - float(row["R06_mae"])
            subgroup_checks[name] = {"supported": True, "pass": harm <= 0.25,
                                     "n": int(row["n"]), "harm_vs_R06": harm}
            checks[f"{name}_harm_vs_R06_at_most_0_25"] = harm <= 0.25
    return {"checks": checks, "subgroups": subgroup_checks,
            "all_supported_incremental_diagnostic_checks_pass": all(checks.values()),
            "global_qualification_claim": False,
            "interpretation": "diagnostic-only; original R06 verdict and guards remain unchanged"}


__all__ = [
    "ALPHA", "CORRECTION_CAP", "EXPECTED_SOURCE_CUTOFFS", "MAX_NEIGHBORS",
    "MIN_NEIGHBORS", "PILOT_ARMS", "PILOT_ORIGINS", "RnbrProtocolError",
    "PILOT_QUERY_HORIZONS", "PILOT_QUERY_WINDOWS",
    "SHUFFLE_SEED", "SyntheticCase", "correction_delta", "evaluate_incremental_gates",
    "fixed_prediction_decile", "plan_pilot_fits", "predict_from_saved_neighbors",
    "shuffle_residuals_within_cycle_decile", "validate_donor_value_lineage", "validate_saved_geometry",
    "weighted_median_residual",
]
