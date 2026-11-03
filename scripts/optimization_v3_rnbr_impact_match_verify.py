from __future__ import annotations

import os
for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_name] = "2"

import json
from pathlib import Path
import sys
from typing import Any, Mapping

import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import optimization_v3_rnbr_impact_match as runner  # noqa: E402
from scripts import optimization_v3_rnbr_official as official  # noqa: E402
from src.models.classification_a02_neighbors import _documents  # noqa: E402
from src.models.regression_neighbor_rnbr import (  # noqa: E402
    EXPECTED_SOURCE_CUTOFFS,
    MAX_NEIGHBORS,
    MIN_NEIGHBORS,
    fixed_prediction_decile,
)
from src.models import regression_rnbr_impact_match as impact  # noqa: E402

_LIMITER = threadpool_limits(limits=2)
VERIFY_PATH = runner.REPORT_DIR / "verification.json"
RESULTS_PATH = runner.REPORT_DIR / "results.json"
SUMMARY_PATH = runner.REPORT_DIR / "RESULTS.md"


class ImpactMatchVerificationError(RuntimeError):
    pass


def _independent_top32(query_matrix, donor_matrix, query_impact, donor_impact,
                       donor_rows, stream_missing):
    """Separate row-loop reconstruction; does not call the production builder."""
    q = query_matrix.tocsr()
    d = donor_matrix.tocsr()
    qi = np.asarray(query_impact, dtype=np.float64)
    di = np.asarray(donor_impact, dtype=np.float64)
    dr = np.asarray(donor_rows, dtype=np.int64)
    missing = np.asarray(stream_missing, dtype=bool)
    out_rows = np.full((q.shape[0], MAX_NEIGHBORS), -1, dtype=np.int64)
    out_cos = np.zeros((q.shape[0], MAX_NEIGHBORS), dtype=np.float64)
    out_count = np.zeros(q.shape[0], dtype=np.int16)
    for query_index in range(q.shape[0]):
        if missing[query_index] or not np.isfinite(qi[query_index]):
            continue
        eligible = np.flatnonzero(di == qi[query_index])
        if not len(eligible):
            continue
        scores = np.asarray((q[query_index] @ d[eligible].T).toarray()).reshape(-1)
        if not np.isfinite(scores).all():
            raise ImpactMatchVerificationError("Independent cosine calculation returned nonfinite values")
        positive = np.flatnonzero(scores > 0.0)
        if not len(positive):
            continue
        order = sorted(positive.tolist(), key=lambda j: (-float(scores[j]), int(dr[eligible[j]])))
        chosen = np.asarray(order[:MAX_NEIGHBORS], dtype=np.int64)
        count = len(chosen)
        out_rows[query_index, :count] = dr[eligible[chosen]]
        out_cos[query_index, :count] = scores[chosen]
        out_count[query_index] = count
    return out_rows, out_cos, out_count


def _independent_source_rows(residual, cycles, impacts, r06, rows):
    values = np.asarray(residual, dtype=np.float64)
    cycle = np.asarray(cycles, dtype=np.int64)
    load = np.asarray(impacts, dtype=np.float64)
    baseline = np.asarray(r06, dtype=np.float64)
    source_rows = np.asarray(rows, dtype=np.int64)
    decile = fixed_prediction_decile(baseline)
    cells: dict[tuple[int, float, int], list[int]] = {}
    for i in range(len(source_rows)):
        cells.setdefault((int(cycle[i]), float(load[i]), int(decile[i])), []).append(i)
    source_ix = np.arange(len(source_rows), dtype=np.int64)
    rng = np.random.default_rng(42)
    for key in sorted(cells.keys()):
        positions = np.asarray(cells[key], dtype=np.int64)
        positions = positions[np.argsort(source_rows[positions], kind="stable")]
        if len(positions) > 1:
            source_ix[positions] = positions[rng.permutation(len(positions))]
    mapped = values[source_ix]
    return source_rows[source_ix], mapped, {
        "source_positions_moved": int(np.sum(source_ix != np.arange(len(source_ix)))),
        "source_position_movement_rate": float(np.mean(source_ix != np.arange(len(source_ix)))) if len(source_ix) else 0.0,
        "residual_values_moved": int(np.sum(mapped != values)),
        "residual_value_movement_rate": float(np.mean(mapped != values)) if len(values) else 0.0,
    }


def _independent_weighted_median(residuals, similarities, source_rows):
    residual = np.asarray(residuals, dtype=np.float64)
    weights = np.square(np.asarray(similarities, dtype=np.float64))
    rows = np.asarray(source_rows, dtype=np.int64)
    if (residual.ndim != 1 or weights.shape != residual.shape or rows.shape != residual.shape
            or not np.isfinite(residual).all() or not np.isfinite(weights).all()
            or np.any(weights <= 0.0) or len(np.unique(rows)) != len(rows)):
        raise ImpactMatchVerificationError("Independent weighted-median inputs are invalid")
    order = np.lexsort((rows, residual))
    values, weights = residual[order], weights[order]
    unique, first = np.unique(values, return_index=True)
    masses = np.add.reduceat(weights, first)
    total = float(masses.sum())
    running = 0.0
    for i, (value, mass) in enumerate(zip(unique, masses)):
        running += float(mass)
        half = total / 2.0
        if running > half + total * 1e-12:
            return float(value)
        if abs(running - half) <= total * 1e-12:
            return float((value + unique[i + 1]) / 2.0) if i + 1 < len(unique) else float(value)
    return float(unique[-1])


def _independent_prediction(baseline, missing, rows, similarities, counts,
                            donor_rows, donor_residuals):
    base = np.asarray(baseline, dtype=np.float64)
    fallback = np.asarray(missing, dtype=bool)
    neighbor_rows = np.asarray(rows, dtype=np.int64)
    cosines = np.asarray(similarities, dtype=np.float64)
    n_count = np.asarray(counts, dtype=np.int64)
    source_rows = np.asarray(donor_rows, dtype=np.int64)
    residual = np.asarray(donor_residuals, dtype=np.float64)
    lookup = {int(row): i for i, row in enumerate(source_rows)}
    median = np.zeros(len(base), dtype=np.float64)
    delta = np.zeros(len(base), dtype=np.float64)
    for i, count_value in enumerate(n_count):
        count = int(count_value)
        if fallback[i] or count < MIN_NEIGHBORS:
            continue
        rows_i = neighbor_rows[i, :count]
        try:
            donor_ix = np.asarray([lookup[int(row)] for row in rows_i], dtype=np.int64)
        except KeyError as exc:
            raise ImpactMatchVerificationError("Neighbor is outside the frozen residual memory") from exc
        median[i] = _independent_weighted_median(residual[donor_ix], cosines[i, :count], rows_i)
        delta[i] = np.clip(0.25 * median[i], -5.0, 5.0)
    prediction = np.clip(base + delta, 0.0, 100.0)
    if (not np.isfinite(prediction).all() or not np.array_equal(prediction[fallback], base[fallback])
            or not np.array_equal(delta[fallback], np.zeros(int(fallback.sum())))):
        raise ImpactMatchVerificationError("Independent R06 fallback/formula reconstruction failed")
    return {"prediction": prediction, "delta": delta, "weighted_median": median}


def _load_geometry_files(freeze: Mapping[str, Any]) -> dict[int, dict[str, np.ndarray]]:
    report = runner._read_json(runner.PREFLIGHT_PATH)
    result = {}
    for origin in runner.ORIGINS:
        spec = report["origin_support"][str(origin)]
        path = runner.ROOT / spec["geometry_path_ignored"]
        if impact.sha256_file(path) != freeze["geometry_file_sha256"][str(origin)]:
            raise ImpactMatchVerificationError(f"E{origin} target-free geometry file changed")
        with np.load(path, allow_pickle=False) as archive:
            result[origin] = {key: np.asarray(archive[key]).copy() for key in archive.files}
    return result


def _load_fit_bundle(fit_id: str, freeze: Mapping[str, Any]) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    arrays, receipt = impact.load_bundle(runner._fit_dir(fit_id), fit_id=fit_id,
                                         freeze_signature=freeze["freeze_signature"])
    events = runner._ledger_events()
    starts = [event for event in events if event.get("fit_id") == fit_id and event.get("event") == "FIT_STARTED"]
    saved = [event for event in events if event.get("fit_id") == fit_id and event.get("event") == "FIT_SAVED"]
    failed = [event for event in events if event.get("fit_id") == fit_id and event.get("event") == "FIT_FAILED"]
    if (len(starts) != 1 or len(saved) != 1 or failed
            or saved[0].get("receipt_sha256") != impact.sha256_file(runner._fit_dir(fit_id) / "receipt.json")
            or starts[0].get("freeze_signature") != freeze["freeze_signature"]
            or starts[0].get("source_commit") != receipt["metadata"].get("source_commit")
            or receipt["metadata"].get("fit_recipe") != "RNBR_IMPACT_MATCH_V1 frozen protocol"
            or receipt["metadata"].get("query_targets_read") is not False
            or receipt["metadata"].get("classification_labels_or_probabilities_read") is not False
            or receipt["metadata"].get("test_or_sample_submission_opened") is not False):
        raise ImpactMatchVerificationError(f"Fit receipt/accounting/source boundaries fail: {fit_id}")
    return arrays, receipt


def _recompute_geometry(origin, train, docs, impacts, base_freeze):
    spec = base_freeze["a02_saved_geometry"][str(origin)]["observed"]
    model_path = runner.ROOT / spec["model"]
    if impact.sha256_file(model_path) != spec["model_sha256"]:
        raise ImpactMatchVerificationError(f"A02 E{origin} vectorizer artifact hash differs")
    model = joblib.load(model_path)
    vectorizer = getattr(getattr(model, "text_space", None), "vectorizer", None)
    if vectorizer is None:
        raise ImpactMatchVerificationError(f"A02 E{origin} fitted vectorizer is unavailable")
    donor_rows = np.flatnonzero((train["cycles"] >= 56) & (train["cycles"] <= origin)
                                & train["present"] & np.isfinite(impacts)).astype(np.int64)
    low, high = runner.QUERY_CYCLES[origin]
    query_rows = np.flatnonzero((train["cycles"] >= low) & (train["cycles"] <= high)).astype(np.int64)
    donor_matrix = vectorizer.transform(docs[donor_rows].tolist()).tocsr()
    query_matrix = vectorizer.transform(docs[query_rows].tolist()).tocsr()
    qmissing = ~train["present"][query_rows]
    rows, cosines, counts = _independent_top32(
        query_matrix, donor_matrix, impacts[query_rows], impacts[donor_rows], donor_rows, qmissing
    )
    return {"donor_rows": donor_rows, "query_rows": query_rows, "neighbor_rows": rows,
            "neighbor_similarities": cosines, "neighbor_count": counts}


def _mae(y, prediction) -> float:
    a = np.asarray(y, dtype=np.float64)
    b = np.asarray(prediction, dtype=np.float64)
    if a.shape != b.shape or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ImpactMatchVerificationError("Metric arrays are not aligned finite vectors")
    return float(np.mean(np.abs(a - b)))


def _view_entries(geometry, fits):
    return {
        "W4": [(67, i) for i, cycle in enumerate(geometry[67]["query_cycles"]) if 68 <= cycle <= 71],
        "W6": [(75, i) for i, cycle in enumerate(geometry[75]["query_cycles"]) if 76 <= cycle <= 79],
        "POOLED_W4_W6": ([(67, i) for i, cycle in enumerate(geometry[67]["query_cycles"]) if 68 <= cycle <= 71]
                         + [(75, i) for i, cycle in enumerate(geometry[75]["query_cycles"]) if 76 <= cycle <= 79]),
        "H12_E67": [(67, i) for i in range(len(geometry[67]["query_rows"]))],
    }


def _view_metrics(view_entries, y_lookup, geometry, fits, old_rnbr):
    rows, y, candidate, shuffled, r06, original, supported, stream_missing, impact_missing = [], [], [], [], [], [], [], [], []
    donor_ages, cosines = [], []
    for origin, local_index in view_entries:
        geo = geometry[origin]
        observed = fits[(origin, "observed")]
        control = fits[(origin, "shuffled")]
        row = int(geo["query_rows"][local_index])
        rows.append(row)
        y.append(float(y_lookup[row]))
        candidate.append(float(observed["prediction"][local_index]))
        shuffled.append(float(control["prediction"][local_index]))
        r06.append(float(geo["baseline_prediction"][local_index]))
        original.append(float(old_rnbr[origin]["prediction"][local_index]))
        count = int(geo["neighbor_count"][local_index])
        supported.append(count >= MIN_NEIGHBORS and not bool(geo["query_effective_missing"][local_index]))
        stream_missing.append(bool(geo["query_stream_missing"][local_index]))
        impact_missing.append(bool(geo["query_impact_missing"][local_index]))
        if count:
            source_rows = geo["neighbor_rows"][local_index, :count]
            source_cycles = {int(r): int(c) for r, c in zip(geo["donor_rows"], geo["donor_cycles"])}
            donor_ages.extend([origin - source_cycles[int(r)] for r in source_rows])
            cosines.extend(geo["neighbor_similarities"][local_index, :count].tolist())
    arrays = {key: np.asarray(value) for key, value in {
        "rows": rows, "y": y, "candidate": candidate, "shuffled": shuffled,
        "R06": r06, "original_RNBR": original, "supported": supported,
        "stream_missing": stream_missing, "impact_missing": impact_missing,
    }.items()}
    metrics = {name: _mae(arrays["y"], arrays[name]) for name in ("R06", "original_RNBR", "candidate", "shuffled")}
    common = arrays["supported"].astype(bool)
    common_metrics = ({name: _mae(arrays["y"][common], arrays[name][common])
                       for name in ("R06", "original_RNBR", "candidate", "shuffled")}
                      if common.any() else {})
    return {
        "n": len(rows),
        "supported_rows_ge5": int(common.sum()),
        "fallback_rows": int(len(rows) - common.sum()),
        "fallback_coverage_total_rows": float((len(rows) - common.sum()) / max(1, len(rows))),
        "present_stream_rows": int((~arrays["stream_missing"]).sum()),
        "coverage_of_present_stream": float(common.sum() / max(1, (~arrays["stream_missing"]).sum())),
        "stream_missing_rows": int(arrays["stream_missing"].sum()),
        "impact_missing_rows": int(arrays["impact_missing"].sum()),
        "mae": metrics,
        "gain_vs_R06": {name: metrics["R06"] - metrics[name]
                        for name in ("candidate", "shuffled", "original_RNBR")},
        "common_supported_n": int(common.sum()),
        "common_supported_mae": common_metrics,
        "common_supported_gain_vs_R06": ({name: common_metrics["R06"] - common_metrics[name]
                                           for name in ("candidate", "shuffled", "original_RNBR")}
                                          if common_metrics else {}),
        "matched_impact_neighbors": {
            "mean_positive_cosine": float(np.mean(cosines)) if cosines else 0.0,
            "mean_donor_age_cycles": float(np.mean(donor_ages)) if donor_ages else 0.0,
            "max_donor_age_cycles": int(max(donor_ages)) if donor_ages else 0,
        },
        "row_keys": arrays,
    }


def verify_all() -> dict[str, Any]:
    freeze, protocol, preflight = runner._current_freeze()
    if preflight.get("status") != "PASS":
        raise ImpactMatchVerificationError("Cannot evaluate a scientific-preflight skip")
    base_freeze = runner._read_json(runner.BASE_FREEZE_PATH)
    train = official.load_targetfree_train()
    frame = pd.read_csv(runner.TRAIN, usecols=["record_id", "cycle_code", "signal_stream", "impact_load"],
                        dtype={"record_id": str, "impact_load": "float64"})
    impacts = frame["impact_load"].to_numpy(dtype=np.float64)
    docs, missing = _documents(frame["signal_stream"])
    docs = np.asarray(docs, dtype=object)
    if not np.array_equal(missing, ~train["present"]):
        raise ImpactMatchVerificationError("Independent target-free stream normalization differs")
    base = runner._check_base_freeze(train=train)
    r06_cache = base["r06_cache"]
    geometry = _load_geometry_files(freeze)
    geometry_checks = {}
    for origin in runner.ORIGINS:
        independent = _recompute_geometry(origin, train, docs, impacts, base_freeze)
        stored = geometry[origin]
        for key, expected in independent.items():
            if not np.array_equal(stored[key], expected):
                raise ImpactMatchVerificationError(f"Independent E{origin} geometry differs: {key}")
        for key, expected in (
            ("query_ids", train["ids"][independent["query_rows"]].astype(str)),
            ("query_cycles", train["cycles"][independent["query_rows"]].astype(np.int64)),
            ("query_impact_load", impacts[independent["query_rows"]]),
            ("donor_ids", train["ids"][independent["donor_rows"]].astype(str)),
            ("donor_cycles", train["cycles"][independent["donor_rows"]].astype(np.int64)),
            ("donor_impact_load", impacts[independent["donor_rows"]]),
        ):
            if not np.array_equal(stored[key], expected, equal_nan=True):
                raise ImpactMatchVerificationError(f"Independent E{origin} identity/feature join differs: {key}")
        baseline = runner._baseline_for_origin(
            r06_cache, origin, independent["query_rows"], stored["query_ids"].astype(str)
        )
        if not np.array_equal(stored["baseline_prediction"], baseline):
            raise ImpactMatchVerificationError(f"R06 fixed-origin E{origin} baseline differs")
        geometry_checks[str(origin)] = {
            "status": "PASS", "query_rows": len(independent["query_rows"]),
            "donor_rows": len(independent["donor_rows"]),
            "top32_neighbors_recomputed": int(independent["neighbor_count"].sum()),
            "exact_impact_and_strict_time": True,
        }

    # First verify all four durable bundles and their ledger receipts. Query
    # targets remain unopened until geometry, source boundaries, receipts, and
    # task accounting have passed below.
    loaded: dict[tuple[int, str], dict[str, np.ndarray]] = {}
    fit_receipts = {}
    for origin in runner.ORIGINS:
        for arm in runner.ARMS:
            fit_id = f"RNBR_IMPACT_MATCH_E{origin}_{arm.upper()}"
            arrays, receipt = _load_fit_bundle(fit_id, freeze)
            loaded[(origin, arm)] = arrays
            fit_receipts[fit_id] = {
                "receipt_sha256": impact.sha256_file(runner._fit_dir(fit_id) / "receipt.json"),
                "npz_sha256": impact.sha256_file(runner._fit_dir(fit_id) / "arrays.npz"),
                "source_commit": receipt["metadata"]["source_commit"],
            }

    original = {origin: runner._load_original_rnbr_predictions(origin, base_freeze, train)
                for origin in runner.ORIGINS}
    donor_joins, shuffle_audit, formula_audit = {}, {}, {}
    for origin in runner.ORIGINS:
        observed = loaded[(origin, "observed")]
        control = loaded[(origin, "shuffled")]
        geo = geometry[origin]
        expected_rows = geo["donor_rows"].astype(np.int64)
        expected_cycles = geo["donor_cycles"].astype(np.int64)
        for arm_data in (observed, control):
            for key, expected in (
                ("donor_rows", expected_rows), ("donor_ids", geo["donor_ids"].astype(str)),
                ("donor_cycles", expected_cycles), ("donor_impact_load", geo["donor_impact_load"]),
                ("score_rows", geo["query_rows"]), ("score_record_ids", geo["query_ids"].astype(str)),
                ("score_cycles", geo["query_cycles"]), ("query_impact_load", geo["query_impact_load"]),
                ("baseline_prediction", geo["baseline_prediction"]),
                ("neighbor_rows", geo["neighbor_rows"]), ("neighbor_ids", geo["neighbor_ids"]),
                ("neighbor_similarities", geo["neighbor_similarities"]),
                ("neighbor_count", geo["neighbor_count"]),
            ):
                if not np.array_equal(arm_data[key], expected, equal_nan=True):
                    raise ImpactMatchVerificationError(f"E{origin} arm geometry/prefix differs for {key}")
        # Read only donor targets after all four bundle receipts are verified.
        donor_targets = official.read_regression_targets(expected_rows, purpose=f"RNBR_IMPACT_MATCH_E{origin}_INDEPENDENT_DONOR_AUDIT")
        if (not np.array_equal(donor_targets["ids"].astype(str), geo["donor_ids"].astype(str))
                or not np.array_equal(donor_targets["cycles"].astype(np.int64), expected_cycles)):
            raise ImpactMatchVerificationError(f"E{origin} donor target row/ID/cycle join differs")
        donor_r06 = official.aligned_r06_predictions(expected_rows, expected_cycles, r06_cache)
        cutoffs = np.asarray([EXPECTED_SOURCE_CUTOFFS[int(c)] for c in expected_cycles], dtype=np.int64)
        if np.any(cutoffs >= expected_cycles):
            raise ImpactMatchVerificationError(f"E{origin} donor cutoff is not strictly earlier")
        residuals = donor_targets["target"] - donor_r06
        if not np.array_equal(observed["donor_r06_predictions"], donor_r06):
            raise ImpactMatchVerificationError(f"E{origin} stored R06 OOF donor predictions differ")
        if not np.array_equal(observed["donor_residuals"], residuals):
            raise ImpactMatchVerificationError(f"E{origin} donor residuals differ from target minus R06 OOF")
        if not np.array_equal(control["donor_residuals"], residuals):
            raise ImpactMatchVerificationError(f"E{origin} shuffled arm donor residual basis differs")
        if not np.array_equal(observed["donor_source_cutoffs"], cutoffs) or not np.array_equal(control["donor_source_cutoffs"], cutoffs):
            raise ImpactMatchVerificationError(f"E{origin} saved R06 donor cutoff map differs")
        source_rows, residual_used, movement = _independent_source_rows(
            residuals, expected_cycles, geo["donor_impact_load"], donor_r06, expected_rows
        )
        if not np.array_equal(observed["shuffle_source_rows"], expected_rows):
            raise ImpactMatchVerificationError(f"E{origin} observed arm unexpectedly permuted residual sources")
        if (not np.array_equal(control["shuffle_source_rows"], source_rows)
                or not np.array_equal(control["donor_residuals_used"], residual_used)):
            raise ImpactMatchVerificationError(f"E{origin} exact-cycle/impact/decile shuffle differs")
        shuffle_audit[str(origin)] = movement
        for arm, arrays in (("observed", observed), ("shuffled", control)):
            effective_missing = geo["query_effective_missing"].astype(bool)
            prediction = _independent_prediction(
                geo["baseline_prediction"], effective_missing,
                geo["neighbor_rows"], geo["neighbor_similarities"], geo["neighbor_count"],
                expected_rows, arrays["donor_residuals_used"],
            )
            if (not np.array_equal(arrays["prediction"], prediction["prediction"])
                    or not np.array_equal(arrays["delta"], prediction["delta"])
                    or not np.array_equal(arrays["neighbor_count_used"],
                                          np.where((~effective_missing) & (geo["neighbor_count"] >= MIN_NEIGHBORS), geo["neighbor_count"], 0))):
                raise ImpactMatchVerificationError(f"E{origin} independent prediction/formula differs: {arm}")
            if not np.array_equal(arrays["prediction"][effective_missing | (geo["neighbor_count"] < MIN_NEIGHBORS)],
                                  geo["baseline_prediction"][effective_missing | (geo["neighbor_count"] < MIN_NEIGHBORS)]):
                raise ImpactMatchVerificationError(f"E{origin} unsupported/missing rows are not exact R06 fallback")
            formula_audit[f"E{origin}_{arm}"] = {
                "status": "PASS", "prediction_rows": len(prediction["prediction"]),
                "prediction_sha256": impact.sha256_array(prediction["prediction"], "<f8"),
                "delta_sha256": impact.sha256_array(prediction["delta"], "<f8"),
                "exact_fallback_rows": int(np.sum(effective_missing | (geo["neighbor_count"] < MIN_NEIGHBORS))),
            }
        donor_joins[str(origin)] = {
            "status": "PASS", "donor_rows": len(expected_rows),
            "donor_target_sha256": impact.sha256_array(donor_targets["target"], "<f8"),
            "donor_residual_sha256": impact.sha256_array(residuals, "<f8"),
            "donor_cycles": [int(expected_cycles.min()), int(expected_cycles.max())],
            "r06_source_cutoffs": {str(int(c)): int(np.sum(cutoffs == c)) for c in np.unique(cutoffs)},
        }

    # Query truth is opened only after all four durable bundles, source/receipt
    # identities, target-free geometry, donor boundaries, shuffles and formulas
    # pass independent verification.
    query_rows = np.unique(np.concatenate([geometry[67]["query_rows"], geometry[75]["query_rows"]])).astype(np.int64)
    query_truth = official.read_regression_targets(query_rows, purpose="RNBR_IMPACT_MATCH_V1_POST_BUNDLE_EVALUATION")
    if (not np.array_equal(query_truth["ids"].astype(str), train["ids"][query_rows])
            or not np.array_equal(query_truth["cycles"].astype(np.int64), train["cycles"][query_rows])):
        raise ImpactMatchVerificationError("Query evaluation target rows/IDs/cycles differ")
    y_lookup = {int(row): float(value) for row, value in zip(query_rows, query_truth["target"])}
    old_predictions = {}
    for origin in runner.ORIGINS:
        old = original[origin]
        if not np.array_equal(old["rows"], geometry[origin]["query_rows"]):
            raise ImpactMatchVerificationError(f"Original RNBR E{origin} query rows differ")
        if not np.array_equal(old["baseline"], geometry[origin]["baseline_prediction"]):
            raise ImpactMatchVerificationError(f"Original RNBR E{origin} R06 baseline differs")
        old_predictions[origin] = old
    fits = {(origin, arm): loaded[(origin, arm)] for origin in runner.ORIGINS for arm in runner.ARMS}
    views = _view_entries(geometry, fits)
    metrics: dict[str, Any] = {}
    view_arrays = {}
    for view, entries in views.items():
        view_result = _view_metrics(entries, y_lookup, geometry, fits, old_predictions)
        view_arrays[view] = view_result.pop("row_keys")
        metrics[view] = view_result

    subgroup_report: dict[str, Any] = {}
    subgroup_gates: dict[str, bool] = {}
    for view, entries in views.items():
        arrays = view_arrays[view]
        groups = {
            "target_le_5": arrays["y"] <= 5.0,
            "target_ge_95": arrays["y"] >= 95.0,
        }
        out = {}
        for name, mask in groups.items():
            count = int(mask.sum())
            if count:
                harm = _mae(arrays["y"][mask], arrays["candidate"][mask]) - _mae(arrays["y"][mask], arrays["R06"][mask])
            else:
                harm = None
            checked = count >= 100
            passed = True if not checked else (harm is not None and harm <= 0.25)
            subgroup_gates[f"{view}_{name}"] = passed
            out[name] = {"n": count, "harm_vs_R06": harm, "gate_applies_n_ge_100": checked,
                         "gate_pass": passed, "unsupported_if_n_lt_100": not checked}
        missing = arrays["stream_missing"]
        missing_n = int(missing.sum())
        missing_exact = bool(np.array_equal(arrays["candidate"][missing], arrays["R06"][missing]))
        out["signal_stream_missing"] = {"n": missing_n, "increment_vs_R06_exact_zero": missing_exact,
                                          "gate_pass": missing_exact}
        subgroup_gates[f"{view}_missing_exact"] = missing_exact
        impact_missing = arrays["impact_missing"]
        impact_missing_n = int(impact_missing.sum())
        impact_missing_exact = bool(np.array_equal(arrays["candidate"][impact_missing], arrays["R06"][impact_missing]))
        out["impact_load_missing"] = {"n": impact_missing_n,
                                      "increment_vs_R06_exact_zero": impact_missing_exact,
                                      "gate_pass": impact_missing_exact}
        subgroup_gates[f"{view}_impact_missing_exact"] = impact_missing_exact
        subgroup_report[view] = out

    gate_values = {
        "W4_gain_vs_R06_nonnegative": metrics["W4"]["gain_vs_R06"]["candidate"] >= 0.0,
        "W6_gain_vs_R06_nonnegative": metrics["W6"]["gain_vs_R06"]["candidate"] >= 0.0,
        "pooled_gain_vs_R06_at_least_0_03": metrics["POOLED_W4_W6"]["gain_vs_R06"]["candidate"] >= 0.03,
        "H12_gain_vs_R06_nonnegative": metrics["H12_E67"]["gain_vs_R06"]["candidate"] >= 0.0,
        "pooled_gain_vs_matched_shuffle_at_least_0_01": metrics["POOLED_W4_W6"]["gain_vs_R06"]["candidate"]
                                                       - metrics["POOLED_W4_W6"]["gain_vs_R06"]["shuffled"] >= 0.01,
        "H12_gain_vs_matched_shuffle_nonnegative": metrics["H12_E67"]["gain_vs_R06"]["candidate"]
                                                   - metrics["H12_E67"]["gain_vs_R06"]["shuffled"] >= 0.0,
        "pooled_gain_vs_original_RNBR_at_least_0_03": metrics["POOLED_W4_W6"]["gain_vs_R06"]["candidate"]
                                                      - metrics["POOLED_W4_W6"]["gain_vs_R06"]["original_RNBR"] >= 0.03,
        "H12_gain_vs_original_RNBR_nonnegative": metrics["H12_E67"]["gain_vs_R06"]["candidate"]
                                                  - metrics["H12_E67"]["gain_vs_R06"]["original_RNBR"] >= 0.0,
        **subgroup_gates,
    }
    gates_pass = all(gate_values.values())
    per_cycle = {}
    h12_entries = views["H12_E67"]
    for cycle in range(68, 80):
        selected = [(origin, i) for origin, i in h12_entries if int(geometry[origin]["query_cycles"][i]) == cycle]
        entries_metric = _view_metrics(selected, y_lookup, geometry, fits, old_predictions)
        local_cosines, local_ages, local_fallback = [], [], 0
        for origin, i in selected:
            count = int(geometry[origin]["neighbor_count"][i])
            local_fallback += int(count < MIN_NEIGHBORS or geometry[origin]["query_effective_missing"][i])
            if count:
                local_cosines.extend(geometry[origin]["neighbor_similarities"][i, :count].tolist())
                donor_cycle_lookup = {int(r): int(c) for r, c in zip(geometry[origin]["donor_rows"], geometry[origin]["donor_cycles"])}
                local_ages.extend([origin - donor_cycle_lookup[int(r)]
                                   for r in geometry[origin]["neighbor_rows"][i, :count]])
        per_cycle[str(cycle)] = {"n": entries_metric["n"], "mae": entries_metric["mae"],
                                 "gain_vs_R06": entries_metric["gain_vs_R06"],
                                 "gain_sign_candidate_vs_R06": "positive" if entries_metric["gain_vs_R06"]["candidate"] > 0
                                 else ("negative" if entries_metric["gain_vs_R06"]["candidate"] < 0 else "zero"),
                                 "fallback_rows": local_fallback,
                                 "fallback_rate": local_fallback / max(1, entries_metric["n"]),
                                 "mean_positive_cosine": float(np.mean(local_cosines)) if local_cosines else 0.0,
                                 "mean_donor_age_cycles": float(np.mean(local_ages)) if local_ages else 0.0}
    ledger_events = runner._ledger_events()
    starts = [event for event in ledger_events if event.get("event") == "FIT_STARTED"
              and event.get("reservation_key") == runner.RESERVATION_KEY]
    saved = [event for event in ledger_events if event.get("event") == "FIT_SAVED"
             and event.get("reservation_key") == runner.RESERVATION_KEY]
    failed = [event for event in ledger_events if event.get("event") == "FIT_FAILED"
              and event.get("reservation_key") == runner.RESERVATION_KEY]
    accounting_pass = (len(starts) == 4 and len(saved) == 4 and not failed
                       and {event.get("fit_id") for event in starts} == set(runner.FIT_IDS)
                       and {event.get("fit_id") for event in saved} == set(runner.FIT_IDS)
                       and runner._ledger_snapshot()["pending"] == 0)
    if not accounting_pass:
        raise ImpactMatchVerificationError("Four-fit accounting does not reconcile")
    integrity = {
        "status": "PASS",
        "targetfree_geometry": geometry_checks,
        "donor_target_and_R06_lineage": donor_joins,
        "independent_formula": formula_audit,
        "shuffle_source_and_value_movement": shuffle_audit,
        "four_fit_receipts": fit_receipts,
        "ledger_accounting": {"status": "PASS", "fit_started": len(starts), "fit_saved": len(saved),
                              "fit_failed": len(failed), "snapshot": runner._ledger_snapshot()},
        "query_targets_opened_after_four_bundle_integrity_pass": True,
        "classification_labels_or_probabilities_read": False,
        "official_test_or_sample_submission_opened": False,
        "kaggle_accessed": False,
    }
    decision = "ADVANCE_PILOT_ONLY" if gates_pass else "RETIRE_EXACT_RECIPE"
    results = {
        "status": decision,
        "pilot_id": "RNBR_IMPACT_MATCH_V1",
        "qualification": "diagnostic pilot only; no continuation, full-Train refit, Test inference, or upload is authorized",
        "user_reported_public_context": {
            "R_NBR_public_MAE": 11.19115,
            "R06_public_MAE": 11.16620,
            "difference": 0.02495,
            "receipt_id_or_csv_hash_verified_here": False,
            "used_for_tuning": False,
        },
        "freeze_signature": freeze["freeze_signature"],
        "source_commit": starts[0].get("source_commit"),
        "preflight": preflight,
        "metrics": metrics,
        "subgroups": subgroup_report,
        "per_cycle_H12_E67": per_cycle,
        "gates": {name: {"pass": bool(value)} for name, value in gate_values.items()},
        "all_gates_pass": gates_pass,
        "integrity": integrity,
        "limits": [
            "All evaluation rows are historical official Train rows and reuse the development archive.",
            "The lane was selected after adaptive/public feedback; these rows are not an independent holdout.",
            "Exact impact equality can reduce text-neighbor choice and may match on a redundant structured feature.",
            "A positive pilot result does not qualify a final model or establish Test/Kaggle improvement.",
        ],
        "public_score_used_for_weight_tuning": False,
    }
    verify = {"status": "PASS", "freeze_signature": freeze["freeze_signature"],
              "geometry_checks": geometry_checks, "donor_lineage": donor_joins,
              "formula_checks": formula_audit, "accounting": integrity["ledger_accounting"],
              "test_or_sample_submission_opened": False, "kaggle_accessed": False,
              "query_targets_read_after_integrity": True}
    runner._write_once(VERIFY_PATH, verify)
    runner._write_once(RESULTS_PATH, results)
    summary = [
        "# RNBR exact-impact-match diagnostic pilot",
        "",
        f"Decision: **{decision}**. This result is diagnostic only.",
        "",
        f"- Freeze: `{freeze['freeze_signature']}`",
        f"- Source commit: `{results['source_commit']}`",
        f"- Four fit accounting: {len(starts)} started, {len(saved)} saved, {len(failed)} failed.",
        "- No official Test, SampleSubmission, Kaggle, or Classification labels/probabilities were accessed.",
        "",
        "| View | n | R06 MAE | Exact-impact observed MAE | Matched shuffle MAE | Original R-NBR MAE | Gain vs R06 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for view in ("W4", "W6", "POOLED_W4_W6", "H12_E67"):
        item = metrics[view]
        summary.append(
            f"| {view} | {item['n']:,} | {item['mae']['R06']:.6f} | {item['mae']['candidate']:.6f} | "
            f"{item['mae']['shuffled']:.6f} | {item['mae']['original_RNBR']:.6f} | "
            f"{item['gain_vs_R06']['candidate']:+.6f} |"
        )
    summary.extend(["", "## Gates", ""])
    for name, passed in gate_values.items():
        summary.append(f"- {'PASS' if passed else 'FAIL'} `{name}`")
    summary.extend(["", "All numerical detail, coverage, subgroup values, movement audit, and counterevidence are in `results.json`.", ""])
    summary_text = "\n".join(summary) + "\n"
    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    if SUMMARY_PATH.exists():
        if SUMMARY_PATH.read_text(encoding="utf-8") != summary_text:
            raise ImpactMatchVerificationError("Existing impact-match markdown summary differs")
    else:
        with SUMMARY_PATH.open("x", encoding="utf-8", newline="\n") as stream:
            stream.write(summary_text)
            stream.flush()
    return results


if __name__ == "__main__":
    print(json.dumps(verify_all(), indent=2, ensure_ascii=False, allow_nan=False))

