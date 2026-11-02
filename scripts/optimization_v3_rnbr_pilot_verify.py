"""Independent verifier for synthetic R-NBR bundles only.

This file intentionally does not import the model implementation. It recomputes
residual assignment, weighted medians, predictions, MAE, subgroups, and diagnostic
gates from synthetic bundles. It has no path to official datasets or ledger.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


ORIGINS = (67, 75)
ARMS = ("observed", "shuffled")
QUERY_STARTS = {67: 68, 75: 76}
QUERY_HORIZONS = {67: 79, 75: 79}
ALPHA = 0.25
DELTA_CAP = 5.0
MIN_NEIGHBORS = 5
MAX_NEIGHBORS = 32


class IndependentVerificationError(RuntimeError):
    pass


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
                      allow_nan=False).encode("utf-8")


def _sha_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha_file(path: Path) -> str:
    return _sha_bytes(path.read_bytes())


def _array_digest(array: np.ndarray) -> str:
    return _sha_bytes(np.ascontiguousarray(array).tobytes(order="C"))


def _expect(condition: bool, message: str) -> None:
    if not condition:
        raise IndependentVerificationError(message)


def _read_bundle(directory: Path, origin: int, arm: str) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    stem = f"E{origin}_{arm}"
    npz = directory / f"{stem}.npz"
    sidecar_path = directory / f"{stem}.json"
    _expect(npz.is_file() and sidecar_path.is_file(), f"Missing synthetic bundle pair: {stem}")
    sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    identity = sidecar.get("identity", {})
    _expect(identity.get("dataset_tag") == "RNBR_SYNTHETIC_FIXTURE_V1", "Non-synthetic bundle blocked")
    _expect(identity.get("origin") == origin and identity.get("arm") == arm, "Synthetic bundle identity mismatch")
    _expect(identity.get("fit_id") == f"SYNTHETIC_RNBR_E{origin}_{arm.upper()}", "Synthetic fit ID mismatch")
    integrity = {key: value for key, value in sidecar.items() if key != "integrity_sha256"}
    _expect(sidecar.get("identity_sha256") == _sha_bytes(_canonical(identity)), "Synthetic identity hash mismatch")
    _expect(sidecar.get("integrity_sha256") == _sha_bytes(_canonical(integrity)), "Synthetic sidecar integrity mismatch")
    _expect(sidecar.get("npz_sha256") == _sha_file(npz), "Synthetic bundle SHA mismatch")
    _expect(sidecar.get("metadata", {}).get("synthetic_task_memory_fit_count") == 1,
            "Synthetic task-memory fit accounting mismatch")
    _expect(sidecar.get("metadata", {}).get("official_train_read") is False
            and sidecar.get("metadata", {}).get("official_target_read") is False
            and sidecar.get("metadata", {}).get("official_candidate_prediction") is False
            and sidecar.get("metadata", {}).get("official_ledger_mutated") is False,
            "Synthetic bundle contains a non-synthetic authorization claim")
    with np.load(npz, allow_pickle=False) as archive:
        arrays = {key: np.asarray(archive[key]).copy() for key in archive.files}
    return arrays, sidecar


def _decile(prediction: np.ndarray) -> np.ndarray:
    return np.minimum(9, np.floor(np.clip(prediction, 0.0, 100.0) / 10.0)).astype(np.int8)


def _shuffle_independently(residual: np.ndarray, cycles: np.ndarray,
                           predictions: np.ndarray, rows: np.ndarray) -> np.ndarray:
    result = residual.copy()
    deciles = _decile(predictions)
    rng = np.random.default_rng(42)
    for cycle in sorted(set(map(int, cycles))):
        for decile in range(10):
            indices = np.flatnonzero((cycles == cycle) & (deciles == decile))
            indices = sorted(indices, key=lambda index: int(rows[index]))
            if len(indices) > 1:
                source_values = residual[np.asarray(indices, dtype=np.int64)]
                result[np.asarray(indices, dtype=np.int64)] = source_values[rng.permutation(len(indices))]
    return result


def _weighted_median_independently(values: np.ndarray, cosines: np.ndarray,
                                   source_rows: np.ndarray) -> float:
    weights = np.asarray(cosines, dtype=np.float64) ** 2
    order = sorted(range(len(values)), key=lambda i: (float(values[i]), int(source_rows[i])))
    total = float(weights.sum())
    if total <= 0.0 or not np.isfinite(total):
        raise IndependentVerificationError("Independent weighted median has no positive mass")
    half = total / 2.0
    tolerance = total * 1e-12
    grouped: list[tuple[float, float]] = []
    for index in order:
        value = float(values[index])
        weight = float(weights[index])
        if grouped and grouped[-1][0] == value:
            grouped[-1] = (value, grouped[-1][1] + weight)
        else:
            grouped.append((value, weight))
    cumulative = 0.0
    for index, (value, weight) in enumerate(grouped):
        cumulative += weight
        if cumulative > half + tolerance:
            return value
        if abs(cumulative - half) <= tolerance:
            return (value + grouped[index + 1][0]) / 2.0 if index + 1 < len(grouped) else value
    return grouped[-1][0]


def _independent_predictions(arrays: dict[str, np.ndarray], arm: str) -> tuple[np.ndarray, np.ndarray]:
    donor_targets = arrays["donor_targets"].astype(np.float64)
    donor_base = arrays["donor_r06_predictions"].astype(np.float64)
    donor_rows = arrays["donor_rows"].astype(np.int64)
    original_residuals = donor_targets - donor_base
    _expect(np.all(np.isfinite(original_residuals)), "Synthetic donor residual is nonfinite")
    if arm == "observed":
        expected_memory = original_residuals
    else:
        expected_memory = _shuffle_independently(
            original_residuals,
            arrays["donor_cycles"].astype(np.int64),
            donor_base,
            donor_rows,
        )
    _expect(np.array_equal(expected_memory, arrays["memory_residuals"]),
            f"Synthetic {arm} target-memory residuals differ from independent reconstruction")

    lookup = {int(row): index for index, row in enumerate(donor_rows)}
    base = arrays["query_r06_predictions"].astype(np.float64)
    missing = arrays["query_missing"].astype(bool)
    n_rows = arrays["neighbor_rows"].astype(np.int64)
    cos = arrays["neighbor_cosines"].astype(np.float64)
    counts = arrays["neighbor_counts"].astype(np.int64)
    deltas = np.zeros(len(base), dtype=np.float64)
    used = np.zeros(len(base), dtype=np.int16)
    for i, count in enumerate(counts):
        k = int(count)
        if missing[i] or k < MIN_NEIGHBORS:
            continue
        selected_rows = n_rows[i, :k]
        selected = np.asarray([lookup[int(row)] for row in selected_rows], dtype=np.int64)
        median = _weighted_median_independently(expected_memory[selected], cos[i, :k], selected_rows)
        # Frozen amendment: scale first, then cap the correction.
        deltas[i] = float(np.clip(0.25 * median, -5.0, 5.0))
        used[i] = k
    predicted = np.clip(base + deltas, 0.0, 100.0)
    _expect(np.array_equal(predicted, arrays["prediction"]),
            f"Synthetic {arm} prediction differs from independent recomputation")
    _expect(np.array_equal(deltas, arrays["delta"]), f"Synthetic {arm} delta differs")
    _expect(np.array_equal(used, arrays["neighbor_count_used"]), f"Synthetic {arm} support differs")
    _expect(np.array_equal(predicted[missing], base[missing]), "Missing-stream fallback changed")
    return predicted, base


def _mae(y: np.ndarray, prediction: np.ndarray) -> float:
    if len(y) == 0 or len(y) != len(prediction):
        raise IndependentVerificationError("MAE vectors must be nonempty and aligned")
    return float(np.abs(y.astype(np.float64) - prediction.astype(np.float64)).sum() / len(y))


def _group_metrics(y: np.ndarray, base: np.ndarray, observed: np.ndarray,
                   mask: np.ndarray) -> dict[str, Any]:
    if not np.any(mask):
        return {"n": 0, "R06_mae": None, "candidate_mae": None, "harm_vs_R06": None}
    r06_mae = _mae(y[mask], base[mask])
    candidate_mae = _mae(y[mask], observed[mask])
    return {"n": int(mask.sum()), "R06_mae": r06_mae,
            "candidate_mae": candidate_mae, "harm_vs_R06": candidate_mae - r06_mae}


def _concat_view_parts(parts: list[dict[str, Any]], *, source_origin: int | None = None) -> dict[str, Any]:
    keys = ("ids", "cycles", "truth", "baseline", "observed", "shuffled", "missing", "source_origin")
    joined = {key: np.concatenate([part[key] for part in parts]) for key in keys}
    joined["declared_source_origin"] = source_origin
    return joined


def compose_validation_views(
    query_meta: dict[int, dict[str, np.ndarray]],
    prediction_map: dict[tuple[int, str], np.ndarray],
    baseline_map: dict[int, np.ndarray],
    truth_arrays: dict[str, np.ndarray],
) -> dict[str, dict[str, Any]]:
    """Compose views by immutable origin and cycle, never by matching n alone.

    H12_L is the full E67 prediction horizon (cycles 68-79). E75 has no input
    to H12_L. Primary remains E67 W4 plus E75 W6.
    """
    ids = truth_arrays["query_ids"].astype(str)
    origins = truth_arrays["query_origin"].astype(np.int64)
    cycles = truth_arrays["query_cycles"].astype(np.int64)
    truth_windows = truth_arrays["query_window"].astype(str)
    labels = truth_arrays["query_truth"].astype(np.float64)
    _expect(len(np.unique(ids)) == len(ids), "Synthetic truth IDs are not unique")
    _expect(len(ids) == len(origins) == len(cycles) == len(truth_windows) == len(labels),
            "Synthetic truth arrays are misaligned")
    truth_lookup = {record_id: (int(origin), int(cycle), str(window), float(label))
                    for record_id, origin, cycle, window, label in zip(ids, origins, cycles, truth_windows, labels)}

    indexed_meta: dict[int, dict[str, np.ndarray]] = {}
    for origin in ORIGINS:
        meta = query_meta[origin]
        query_ids = meta["query_ids"].astype(str)
        query_cycles = meta["query_cycles"].astype(np.int64)
        query_windows = meta["query_windows"].astype(str)
        _expect(len(query_ids) == len(np.unique(query_ids)), f"Duplicate query IDs at E{origin}")
        _expect(len(query_ids) == len(query_cycles) == len(query_windows),
                f"Query row/cycle/window lengths differ at E{origin}")
        labels_aligned = np.empty(len(query_ids), dtype=np.float64)
        for i, record_id in enumerate(query_ids):
            if record_id not in truth_lookup:
                raise IndependentVerificationError(f"Truth ID missing for E{origin}: {record_id}")
            truth_origin, truth_cycle, truth_window, label = truth_lookup[record_id]
            _expect(truth_origin == origin and truth_cycle == int(query_cycles[i])
                    and truth_window == str(query_windows[i]),
                    f"Truth origin/cycle differs from query geometry at E{origin}/{record_id}")
            labels_aligned[i] = label
        indexed_meta[origin] = {**meta, "aligned_truth": labels_aligned}

    def extract(origin: int, low: int, high: int, *, source_name: str) -> dict[str, Any]:
        meta = indexed_meta[origin]
        qcycles = meta["query_cycles"].astype(np.int64)
        mask = (qcycles >= low) & (qcycles <= high)
        _expect(int(mask.sum()) > 0, f"Empty synthetic view {source_name}")
        return {
            "ids": meta["query_ids"].astype(str)[mask],
            "cycles": qcycles[mask],
            "truth": meta["aligned_truth"][mask],
            "baseline": baseline_map[origin][mask],
            "observed": prediction_map[(origin, "observed")][mask],
            "shuffled": prediction_map[(origin, "shuffled")][mask],
            "missing": meta["query_missing"].astype(bool)[mask],
            "source_origin": np.full(int(mask.sum()), origin, dtype=np.int16),
            "declared_source_origin": origin,
            "view_source": source_name,
        }

    w4_e67 = extract(67, 68, 71, source_name="E67_W4_PRIMARY")
    w5_e67 = extract(67, 72, 75, source_name="E67_W5_H12_L")
    w6_e67 = extract(67, 76, 79, source_name="E67_W6_H12_L")
    w6_e75 = extract(75, 76, 79, source_name="E75_W6_PRIMARY")
    h12_l = extract(67, 68, 79, source_name="E67_H12_L")
    return {
        "W4_PRIMARY": w4_e67,
        "W5_E67": w5_e67,
        "W6_PRIMARY": w6_e75,
        "W6_E67_H12_L": w6_e67,
        "primary": _concat_view_parts([w4_e67, w6_e75]),
        "H12_L": h12_l,
    }


def verify_synthetic_run(bundle_dir: Path, truth_path: Path) -> dict[str, Any]:
    # First verify and independently recompute every prediction without loading query truth.
    arms: dict[tuple[int, str], dict[str, np.ndarray]] = {}
    prediction_map: dict[tuple[int, str], np.ndarray] = {}
    baseline_map: dict[int, np.ndarray] = {}
    query_meta: dict[int, dict[str, np.ndarray]] = {}
    receipts = []
    for origin in ORIGINS:
        for arm in ARMS:
            arrays, sidecar = _read_bundle(bundle_dir, origin, arm)
            prediction, baseline = _independent_predictions(arrays, arm)
            arms[(origin, arm)] = arrays
            prediction_map[(origin, arm)] = prediction
            baseline_map[origin] = baseline
            query_meta[origin] = {key: arrays[key] for key in
                                  ("query_rows", "query_ids", "query_cycles", "query_windows", "query_missing")}
            query_cycles = arrays["query_cycles"].astype(np.int64)
            expected_cycles = np.arange(QUERY_STARTS[origin], QUERY_HORIZONS[origin] + 1, dtype=np.int64)
            _expect(np.array_equal(np.unique(query_cycles), expected_cycles),
                    f"E{origin} synthetic query horizon is incomplete or exceeds its declaration")
            donor_cycles = arrays["donor_cycles"].astype(np.int64)
            _expect(np.all((donor_cycles >= 56) & (donor_cycles <= origin)),
                    f"E{origin} synthetic donor bank exceeds its origin")
            expected_window_by_cycle = np.asarray([
                "W4" if 68 <= cycle <= 71 else "W5" if 72 <= cycle <= 75 else "W6"
                for cycle in query_cycles
            ], dtype=str)
            _expect(np.array_equal(arrays["query_windows"].astype(str), expected_window_by_cycle),
                    f"E{origin} synthetic cycle-to-window mapping differs")
            receipts.append({"fit_id": sidecar["identity"]["fit_id"],
                             "npz_sha256": sidecar["npz_sha256"],
                             "sidecar_sha256": _sha_file(bundle_dir / f"E{origin}_{arm}.json")})
    for origin in ORIGINS:
        observed = arms[(origin, "observed")]
        shuffled = arms[(origin, "shuffled")]
        for field in ("query_rows", "query_ids", "query_cycles", "query_windows", "query_missing",
                      "query_r06_predictions", "neighbor_rows", "neighbor_ids",
                      "neighbor_cosines", "neighbor_counts"):
            _expect(np.array_equal(observed[field], shuffled[field]),
                    f"Observed/shuffled synthetic geometry differs: E{origin}/{field}")

    with np.load(truth_path, allow_pickle=False) as truth_archive:
        truth = {key: np.asarray(truth_archive[key]).copy() for key in truth_archive.files}
    truth_hash = _sha_file(truth_path)
    expected_ids = np.concatenate([query_meta[o]["query_ids"].astype(str) for o in ORIGINS])
    _expect(np.array_equal(truth["query_ids"].astype(str), expected_ids), "Synthetic query truth IDs do not align")
    views = compose_validation_views(query_meta, prediction_map, baseline_map, truth)
    for origin in ORIGINS:
        meta_mask = truth["query_origin"].astype(np.int64) == origin
        _expect(int(meta_mask.sum()) == len(query_meta[origin]["query_ids"]),
                f"Truth row count does not match source origin E{origin}")

    primary = views["primary"]
    h12 = views["H12_L"]
    y_primary = primary["truth"]
    base_primary = primary["baseline"]
    candidate_primary = primary["observed"]
    shuffle_primary = primary["shuffled"]
    missing_primary = primary["missing"]
    y_h12 = h12["truth"]
    base_h12 = h12["baseline"]
    candidate_h12 = h12["observed"]
    shuffle_h12 = h12["shuffled"]
    w5_e67 = views["W5_E67"]
    w5_candidate_has_correction = bool(np.any(w5_e67["observed"] != w5_e67["baseline"]))
    _expect(w5_candidate_has_correction, "E67 W5 candidate was incorrectly forced to unchanged R06")

    windows: dict[str, Any] = {}
    for name, view in (("W4", views["W4_PRIMARY"]), ("W6", views["W6_PRIMARY"])):
        windows[name] = {"n": len(view["truth"]), "source_origin": int(view["source_origin"][0]),
                         "cycle_range": [int(view["cycles"].min()), int(view["cycles"].max())],
                         "R06_mae": _mae(view["truth"], view["baseline"]),
                         "candidate_mae": _mae(view["truth"], view["observed"]),
                         "shuffle_mae": _mae(view["truth"], view["shuffled"])}
    groups = {
        "missing_stream": _group_metrics(y_primary, base_primary, candidate_primary, missing_primary),
        "target_le_5": _group_metrics(y_primary, base_primary, candidate_primary, y_primary <= 5.0),
        "target_ge_95": _group_metrics(y_primary, base_primary, candidate_primary, y_primary >= 95.0),
    }
    metrics = {
        "primary": {"n": len(y_primary), "R06_mae": _mae(y_primary, base_primary),
                    "candidate_mae": _mae(y_primary, candidate_primary)},
        "shuffle_primary": {"candidate_mae": _mae(y_primary, shuffle_primary)},
        "H12_L": {"n": len(y_h12), "R06_mae": _mae(y_h12, base_h12),
                  "candidate_mae": _mae(y_h12, candidate_h12),
                  "shuffle_mae": _mae(y_h12, shuffle_h12)},
        "windows": windows,
        "groups": groups,
    }
    # Standalone duplicate of protocol gates; do not import the fit implementation.
    gate_checks = {
        "primary_gain_vs_R06_at_least_0_03": metrics["primary"]["R06_mae"] - metrics["primary"]["candidate_mae"] >= 0.03,
        "primary_gain_vs_shuffle_at_least_0_01": metrics["shuffle_primary"]["candidate_mae"] - metrics["primary"]["candidate_mae"] >= 0.01,
        "H12_L_harm_vs_R06_at_most_0_05": metrics["H12_L"]["candidate_mae"] - metrics["H12_L"]["R06_mae"] <= 0.05,
        "W4_harm_vs_R06_below_0_25": windows["W4"]["candidate_mae"] - windows["W4"]["R06_mae"] < 0.25,
        "W6_harm_vs_R06_below_0_25": windows["W6"]["candidate_mae"] - windows["W6"]["R06_mae"] < 0.25,
    }
    for name, row in groups.items():
        if row["n"] >= 100:
            gate_checks[f"{name}_harm_vs_R06_at_most_0_25"] = row["harm_vs_R06"] <= 0.25
    _expect(all(gate_checks.values()), "Synthetic fixture no longer reaches its declared positive QA gate path")
    view_provenance_checks = {
        "H12_L_source_origin_is_E67_only": bool(np.all(h12["source_origin"] == 67)),
        "H12_L_query_cycles_are_68_through_79": bool(np.array_equal(np.unique(h12["cycles"]), np.arange(68, 80))),
        "H12_L_ids_match_full_E67_query_horizon": bool(np.array_equal(
            h12["ids"], query_meta[67]["query_ids"].astype(str))),
        "primary_W4_source_is_E67_cycles_68_71": windows["W4"]["source_origin"] == 67
        and windows["W4"]["cycle_range"] == [68, 71],
        "primary_W6_source_is_E75_cycles_76_79": windows["W6"]["source_origin"] == 75
        and windows["W6"]["cycle_range"] == [76, 79],
        "E67_W5_corrections_are_included_in_H12_L": w5_candidate_has_correction
        and bool(set(w5_e67["ids"]).issubset(set(h12["ids"]))),
        "E75_rows_are_excluded_from_H12_L": not bool(set(query_meta[75]["query_ids"].astype(str))
                                                        .intersection(set(h12["ids"]))),
    }
    _expect(all(view_provenance_checks.values()), "Synthetic validation-view provenance failed")
    return {
        "status": "PASS",
        "synthetic_only": True,
        "synthetic_dataset_tag": "RNBR_SYNTHETIC_FIXTURE_V1",
        "supersedes_prior_B0_commit": "dbaa3bb0cf3e621daa84537bc8e62f4134153030",
        "prior_H12_semantics_incorrect": True,
        "official_data_loaded": False,
        "official_target_loaded": False,
        "query_truth_loaded_after_bundle_predictions": True,
        "no_official_metric_replay": True,
        "synthetic_primary_population": "W4 + W6 only",
        "synthetic_primary_n": int(len(y_primary)),
        "synthetic_H12_L_n": int(len(y_h12)),
        "synthetic_H12_L_source_origin": 67,
        "synthetic_H12_L_query_cycle_range": [68, 79],
        "synthetic_H12_L_note": "Full E67 horizon, cycles 68-79 from E67 R06/observed/shuffled; E75 excluded; overlaps primary W4",
        "view_provenance_validation": {"status": "PASS", "checks": view_provenance_checks},
        "synthetic_W5_E67_rows": int(len(w5_e67["truth"])),
        "synthetic_W5_E67_candidate_changed_R06": w5_candidate_has_correction,
        "metrics": metrics,
        "incremental_gates": {"checks": gate_checks,
                              "all_pass": all(gate_checks.values()),
                              "global_qualification_claim": False},
        "observed_vs_shuffle_geometry_identical": True,
        "independent_prediction_recompute": "PASS",
        "synthetic_fit_receipts": receipts,
        "truth_bundle_sha256": truth_hash,
        "historical_R06_guard_failure_to_report_separately": {
            "source": "reports/optimization_v3/expanded_benchmark_v1_final_evaluation/R06/verification.json",
            "origin": 55, "group": "signal_stream_missing", "n": 392,
            "harm_vs_legacy": 0.3955562085105022, "unchanged_by_this_present_stream_protocol": True,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle-dir", required=True, type=Path)
    parser.add_argument("--truth", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    try:
        report = verify_synthetic_run(args.bundle_dir, args.truth)
        encoded = json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
        if args.output.exists():
            if args.output.read_text(encoding="utf-8") != encoded:
                raise IndependentVerificationError("Refusing to overwrite a different independent report")
        else:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            with args.output.open("x", encoding="utf-8", newline="\n") as stream:
                stream.write(encoded)
                stream.flush()
        print(json.dumps({"status": report["status"], "synthetic_only": True,
                          "primary_n": report["synthetic_primary_n"],
                          "h12_l_n": report["synthetic_H12_L_n"],
                          "all_gates_pass": report["incremental_gates"]["all_pass"]}))
        return 0
    except Exception as exc:
        print(json.dumps({"status": "FAIL", "error": f"{type(exc).__name__}: {exc}"}))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
