"""Synthetic-only R-NBR lifecycle: memory, prediction, persistence, resume.

This command never loads official Train/Test, never constructs an official
Regression target memory, and never writes the official fit ledger.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.models.regression_neighbor_rnbr import (  # noqa: E402
    PILOT_ARMS,
    PILOT_ORIGINS,
    RnbrProtocolError,
    SyntheticCase,
    plan_pilot_fits,
    predict_from_saved_neighbors,
    shuffle_residuals_within_cycle_decile,
    validate_saved_geometry,
)


LEDGER = ROOT / "experiments/runs/optimization_v3_research_next_v1/budget.jsonl"
EXPECTED_LEDGER_SHA256 = "6fcc4b4d427e3ff03139e0e9ee429f529f7c7bf6ad91ce0eb9a52d55e2503147"
SYNTHETIC_TAG = "RNBR_SYNTHETIC_FIXTURE_V1"
PROTOCOL_ID = "RNBR_LATE_TWO_ORIGIN_DIAGNOSTIC_B0"
PROTOCOL_IDENTITY = {
    "protocol_id": PROTOCOL_ID,
    "target": "Regression threat_rank",
    "residual": "donor threat_rank minus its past-only R06 OOF prediction",
    "donor_cycles": {"E67": [56, 67], "E75": [56, 75]},
    "r06_oof_source_cutoffs": {
        "56-59": 55, "60-63": 59, "64-67": 63,
        "68-71": 67, "72-75": 71, "76-79": 75,
    },
    "neighbor_geometry": {
        "source": "verified saved target-free A02 geometry",
        "maximum_positive_cosine_neighbors": 32,
        "ordering": "descending cosine then ascending source row",
        "similarity_weight": "cosine_squared",
        "minimum_neighbors": 5,
    },
    "weighted_median": {
        "order": "ascending residual then source row",
        "exact_half_between_distinct_values": "midpoint",
        "numeric_tolerance_fraction_of_total_weight": 1e-12,
    },
    "candidate_formula": "clip(R06 + clip(0.25*weighted_median_R06_OOF_residual,-5,+5),0,100)",
    "neighbor_limit": 32,
    "similarity_weight": "cosine_squared",
    "min_neighbors": 5,
    "fallback": "unchanged R06 for missing stream or fewer than five neighbors",
    "rerank": False,
    "shuffle": {
        "cells": "exact cycle_code x fixed prediction decile",
        "decile": "min(9,floor(clip(R06_prediction,0,100)/10))",
        "rng": "numpy default_rng(42), reset once per origin",
        "traversal": "ascending cycle then decile; ascending stable source row within cell",
    },
    "origins": [67, 75],
    "query_horizons": {"E67": [68, 79], "E75": [76, 79]},
    "primary_query_windows": {"E67": [68, 71], "E75": [76, 79]},
    "primary_view": "W4+W6",
    "H12_L_view": "E67 predictions for all cycles 68-79 only, including W4/W5/W6",
    "arms": ["observed", "shuffled"],
    "diagnostic_gates": {
        "primary_gain_vs_R06_min": 0.03,
        "primary_gain_vs_shuffle_min": 0.01,
        "H12_L_max_harm_vs_R06": 0.05,
        "each_W4_W6_max_harm_vs_R06_exclusive": 0.25,
        "supported_subgroup_min_n": 100,
        "supported_subgroup_max_harm_vs_R06": 0.25,
        "missing_incremental_harm": 0.0,
        "qualification": "diagnostic only; cannot change historical R06 verdict",
    },
    "resource_policy": {"one_training_job_at_a_time": True, "threads": 2,
                        "memory_ceiling_gib": 6, "reserved_gib": 2,
                        "new_tfidf_fits": 0, "all_pairs_matrix": False},
}


class SyntheticBundleError(RuntimeError):
    pass


class SyntheticInterruption(RuntimeError):
    pass


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
                      allow_nan=False).encode("utf-8")


def _sha_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _array_digest(array: np.ndarray) -> str:
    payload = np.ascontiguousarray(array).tobytes(order="C")
    return _sha_bytes(payload)


def _protocol_signature() -> str:
    return _sha_bytes(_canonical(PROTOCOL_IDENTITY))


QUERY_GROUP_COUNTS = (120, 120, 240, 120, 1)
QUERY_GROUP_NEIGHBOR_DECILES = (0, 9, 5, None, 5)
WINDOW_START_CYCLE = {"W4": 68, "W5": 72, "W6": 76}


def _query_windows_for_origin(origin: int) -> tuple[str, ...]:
    return ("W4", "W5", "W6") if origin == 67 else ("W6",)


def _window_cycle_rows(window: str) -> np.ndarray:
    start = WINDOW_START_CYCLE[window]
    n = sum(QUERY_GROUP_COUNTS)
    base, remainder = divmod(n, 4)
    per_cycle = np.full(4, base, dtype=np.int64)
    per_cycle[:remainder] += 1
    return np.repeat(np.arange(start, start + 4, dtype=np.int64), per_cycle)


def _window_base_values(window: str) -> np.ndarray:
    if window == "W5":
        group_values = (30.0, 70.0, 50.0, 30.0, 50.0)
    else:
        group_values = (4.0, 94.0, 50.0, 20.0, 50.0)
    return np.concatenate([np.full(n, value) for n, value in zip(QUERY_GROUP_COUNTS, group_values)])


def _query_truth_fixture(origin: int) -> np.ndarray:
    # Synthetic labels are generated only after all four prediction bundles exist.
    truth_by_window = {
        "W4": (5.0, 95.0, 51.0, 21.0, 51.0),
        "W5": (34.0, 74.0, 54.0, 34.0, 54.0),
        "W6": (5.0, 95.0, 51.0, 21.0, 51.0),
    }
    return np.concatenate([
        np.concatenate([np.full(n, value) for n, value in zip(QUERY_GROUP_COUNTS, truth_by_window[window])])
        for window in _query_windows_for_origin(origin)
    ])


def make_synthetic_case(origin: int) -> SyntheticCase:
    """Build a small deterministic fixture, never reading project data."""
    if origin not in PILOT_ORIGINS:
        raise ValueError("Only E67 and E75 fixtures are defined")
    row_start = 20_000 if origin == 67 else 30_000
    donor_rows = np.arange(row_start, row_start + 240, dtype=np.int64)
    donor_ids = np.asarray([f"SYN_D_E{origin}_{i:03d}" for i in range(240)], dtype="<U24")
    donor_cycles = np.full(240, origin, dtype=np.int64)
    donor_source_cutoffs = np.full(240, origin - 4, dtype=np.int64)
    donor_present = np.ones(240, dtype=bool)
    donor_r06_predictions = np.repeat(np.asarray([4.0, 50.0, 94.0]), 80)
    donor_residuals = np.zeros(240, dtype=np.float64)
    donor_residuals[[0, 1, 2, 3, 4, 80, 81, 82, 83, 84, 160, 161, 162, 163, 164]] = 2.0
    donor_targets = donor_r06_predictions + donor_residuals

    windows = _query_windows_for_origin(origin)
    query_base = np.concatenate([_window_base_values(window) for window in windows])
    one_window_missing = np.concatenate([
        np.zeros(sum(QUERY_GROUP_COUNTS[:3]), dtype=bool),
        np.ones(QUERY_GROUP_COUNTS[3], dtype=bool),
        np.zeros(QUERY_GROUP_COUNTS[4], dtype=bool),
    ])
    query_missing = np.tile(one_window_missing, len(windows))
    query_rows = np.arange(50_000 if origin == 67 else 60_000,
                           (50_000 if origin == 67 else 60_000) + len(query_base), dtype=np.int64)
    query_ids = np.asarray([f"SYN_Q_E{origin}_{i:03d}" for i in range(len(query_rows))], dtype="<U24")
    query_cycles = np.concatenate([_window_cycle_rows(window) for window in windows])
    query_windows = np.concatenate([
        np.full(sum(QUERY_GROUP_COUNTS), window, dtype="<U2") for window in windows
    ])

    neighbors = np.full((len(query_rows), 32), -1, dtype=np.int64)
    neighbor_ids = np.full((len(query_rows), 32), "", dtype="<U40")
    cosines = np.zeros((len(query_rows), 32), dtype=np.float64)
    counts = np.zeros(len(query_rows), dtype=np.int16)
    positive_cosines = np.asarray([0.95, 0.92, 0.90, 0.88, 0.86], dtype=np.float64)
    group_starts = {0: 0, 5: 80, 9: 160}
    offset = 0
    for _window in windows:
        for group_n, decile in zip(QUERY_GROUP_COUNTS, QUERY_GROUP_NEIGHBOR_DECILES):
            for query_index in range(offset, offset + group_n):
                if decile is None:
                    continue
                start = group_starts[decile]
                k = 4 if group_n == 1 else 5
                source_rows = donor_rows[start:start + k]
                neighbors[query_index, :k] = source_rows
                neighbor_ids[query_index, :k] = [f"str_:{value}" for value in donor_ids[start:start + k]]
                cosines[query_index, :k] = positive_cosines[:k]
                counts[query_index] = k
            offset += group_n

    return SyntheticCase(
        dataset_tag=SYNTHETIC_TAG,
        origin=origin,
        donor_rows=donor_rows,
        donor_ids=donor_ids,
        donor_cycles=donor_cycles,
        donor_source_cutoffs=donor_source_cutoffs,
        donor_present=donor_present,
        donor_r06_predictions=donor_r06_predictions,
        donor_targets=donor_targets,
        query_rows=query_rows,
        query_ids=query_ids,
        query_cycles=query_cycles,
        query_windows=query_windows,
        query_missing=query_missing,
        query_r06_predictions=query_base,
        neighbor_rows=neighbors,
        neighbor_ids=neighbor_ids,
        neighbor_cosines=cosines,
        neighbor_counts=counts,
    )


def _build_arrays(case: SyntheticCase, arm: str) -> dict[str, np.ndarray]:
    if case.dataset_tag != SYNTHETIC_TAG or arm not in PILOT_ARMS:
        raise SyntheticBundleError("Synthetic-only origin/arm identity failed")
    validate_saved_geometry(
        origin=case.origin,
        donor_rows=case.donor_rows,
        donor_ids=case.donor_ids,
        donor_cycles=case.donor_cycles,
        donor_source_cutoffs=case.donor_source_cutoffs,
        donor_present=case.donor_present,
        query_rows=case.query_rows,
        query_ids=case.query_ids,
        query_cycles=case.query_cycles,
        query_missing=case.query_missing,
        neighbor_rows=case.neighbor_rows,
        neighbor_ids=case.neighbor_ids,
        neighbor_cosines=case.neighbor_cosines,
        neighbor_counts=case.neighbor_counts,
    )
    observed_residuals = case.donor_targets - case.donor_r06_predictions
    memory_residuals = observed_residuals.copy()
    if arm == "shuffled":
        memory_residuals = shuffle_residuals_within_cycle_decile(
            observed_residuals, case.donor_cycles, case.donor_r06_predictions,
            case.donor_rows, seed=42,
        )
    prediction = predict_from_saved_neighbors(
        donor_rows=case.donor_rows,
        donor_residuals=memory_residuals,
        query_r06_predictions=case.query_r06_predictions,
        query_missing=case.query_missing,
        neighbor_rows=case.neighbor_rows,
        neighbor_cosines=case.neighbor_cosines,
        neighbor_counts=case.neighbor_counts,
    )
    # Query labels are deliberately absent from this constructor call and bundle.
    return {
        "donor_rows": case.donor_rows,
        "donor_ids": case.donor_ids,
        "donor_cycles": case.donor_cycles,
        "donor_source_cutoffs": case.donor_source_cutoffs,
        "donor_present": case.donor_present,
        "donor_r06_predictions": case.donor_r06_predictions,
        "donor_targets": case.donor_targets,
        "memory_residuals": memory_residuals,
        "query_rows": case.query_rows,
        "query_ids": case.query_ids,
        "query_cycles": case.query_cycles,
        "query_windows": case.query_windows,
        "query_missing": case.query_missing,
        "query_r06_predictions": case.query_r06_predictions,
        "neighbor_rows": case.neighbor_rows,
        "neighbor_ids": case.neighbor_ids,
        "neighbor_cosines": case.neighbor_cosines,
        "neighbor_counts": case.neighbor_counts,
        "delta": prediction["delta"],
        "prediction": prediction["prediction"],
        "neighbor_count_used": prediction["neighbor_count_used"],
    }


def _bundle_identity(origin: int, arm: str) -> dict[str, Any]:
    return {"dataset_tag": SYNTHETIC_TAG, "protocol_id": PROTOCOL_ID,
            "freeze_signature": _protocol_signature(), "origin": origin,
            "arm": arm, "fit_id": f"SYNTHETIC_RNBR_E{origin}_{arm.upper()}"}


def _write_bundle(npz_path: Path, sidecar_path: Path,
                  identity: dict[str, Any], arrays: dict[str, np.ndarray]) -> dict[str, Any]:
    if npz_path.exists() or sidecar_path.exists():
        raise SyntheticBundleError("Refusing to overwrite or complete a partial synthetic bundle")
    npz_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = npz_path.with_suffix(".npz.tmp")
    with temporary.open("xb") as stream:
        np.savez_compressed(stream, **arrays)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, npz_path)
    sidecar = {
        "schema_version": 1,
        "identity": identity,
        "identity_sha256": _sha_bytes(_canonical(identity)),
        "npz_sha256": _sha_file(npz_path),
        "metadata": {"synthetic_task_memory_fit_count": 1,
                     "official_train_read": False,
                     "official_target_read": False,
                     "official_candidate_prediction": False,
                     "official_ledger_mutated": False},
    }
    sidecar["integrity_sha256"] = _sha_bytes(_canonical(sidecar))
    with sidecar_path.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(sidecar, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    return sidecar


def _load_bundle(npz_path: Path, sidecar_path: Path,
                 expected_identity: dict[str, Any]) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    if not npz_path.is_file() or not sidecar_path.is_file():
        raise SyntheticBundleError(f"Incomplete bundle; fail closed: {npz_path.name}")
    sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    integrity = {key: value for key, value in sidecar.items() if key != "integrity_sha256"}
    if (sidecar.get("identity") != expected_identity
            or sidecar.get("identity_sha256") != _sha_bytes(_canonical(expected_identity))
            or sidecar.get("integrity_sha256") != _sha_bytes(_canonical(integrity))
            or sidecar.get("npz_sha256") != _sha_file(npz_path)):
        raise SyntheticBundleError(f"Synthetic bundle verification failed; no rebuild: {npz_path.name}")
    with np.load(npz_path, allow_pickle=False) as archive:
        arrays = {key: np.asarray(archive[key]).copy() for key in archive.files}
    return arrays, sidecar


def _write_npz_once(path: Path, arrays: dict[str, np.ndarray]) -> str:
    if path.exists():
        with np.load(path, allow_pickle=False) as archive:
            if set(archive.files) != set(arrays):
                raise SyntheticBundleError(f"Existing synthetic file has wrong arrays: {path.name}")
            for name, expected in arrays.items():
                if not np.array_equal(archive[name], expected):
                    raise SyntheticBundleError(f"Existing synthetic file differs: {path.name}/{name}")
        return _sha_file(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("xb") as stream:
        np.savez_compressed(stream, **arrays)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(tmp, path)
    return _sha_file(path)


def _write_text_once(path: Path, text: str) -> None:
    encoded = text if text.endswith("\n") else text + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_text(encoding="utf-8") != encoded:
            raise SyntheticBundleError(f"Refusing to overwrite different synthetic output: {path.name}")
        return
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(encoded)
        stream.flush()
        os.fsync(stream.fileno())


def _ledger_hash() -> str:
    value = _sha_file(LEDGER)
    if value != EXPECTED_LEDGER_SHA256:
        raise RuntimeError(f"Official budget ledger changed: expected {EXPECTED_LEDGER_SHA256}, got {value}")
    return value


def _truth_arrays() -> dict[str, np.ndarray]:
    query_truths = []
    query_ids = []
    query_origin = []
    query_window = []
    query_cycles = []
    query_base = []
    query_missing = []
    for origin in PILOT_ORIGINS:
        case = make_synthetic_case(origin)
        query_truths.append(_query_truth_fixture(origin))
        query_ids.append(case.query_ids)
        query_origin.append(np.full(len(case.query_ids), origin, dtype=np.int16))
        query_window.append(case.query_windows)
        query_cycles.append(case.query_cycles)
        query_base.append(case.query_r06_predictions)
        query_missing.append(case.query_missing)
    return {
        "query_truth": np.concatenate(query_truths),
        "query_ids": np.concatenate(query_ids),
        "query_origin": np.concatenate(query_origin),
        "query_window": np.concatenate(query_window),
        "query_cycles": np.concatenate(query_cycles),
        "query_r06_predictions": np.concatenate(query_base),
        "query_missing": np.concatenate(query_missing),
    }


def _write_truth_after_predictions(run_dir: Path) -> str:
    arrays = _truth_arrays()
    path = run_dir / "synthetic_truth_for_independent_verifier.npz"
    return _write_npz_once(path, arrays)


def _build_or_resume_bundles(run_dir: Path) -> dict[str, Any]:
    bundles_dir = run_dir / "bundles"
    bundles_dir.mkdir(parents=True, exist_ok=True)
    built = 0
    loaded = 0
    receipts = []
    for origin in PILOT_ORIGINS:
        case = make_synthetic_case(origin)
        for arm in PILOT_ARMS:
            stem = f"E{origin}_{arm}"
            npz_path = bundles_dir / f"{stem}.npz"
            sidecar_path = bundles_dir / f"{stem}.json"
            identity = _bundle_identity(origin, arm)
            if npz_path.exists() or sidecar_path.exists():
                arrays, sidecar = _load_bundle(npz_path, sidecar_path, identity)
                loaded += 1
            else:
                arrays = _build_arrays(case, arm)
                sidecar = _write_bundle(npz_path, sidecar_path, identity, arrays)
                built += 1
            receipts.append({"fit_id": identity["fit_id"], "origin": origin,
                             "arm": arm, "npz_sha256": sidecar["npz_sha256"],
                             "sidecar_sha256": _sha_file(sidecar_path),
                             "synthetic_fit_count": sidecar["metadata"]["synthetic_task_memory_fit_count"]})
    return {"synthetic_fit_builds_this_invocation": built,
            "synthetic_bundles_reused_this_invocation": loaded,
            "receipts": receipts}


def run_synthetic_lifecycle(run_dir: Path, *, interrupt_after_bundles: bool = False,
                            verifier_script: Path | None = None) -> dict[str, Any]:
    ledger_before = _ledger_hash()
    run_dir.mkdir(parents=True, exist_ok=True)
    lifecycle = _build_or_resume_bundles(run_dir)
    if len(lifecycle["receipts"]) != 4 or len({row["fit_id"] for row in lifecycle["receipts"]}) != 4:
        raise SyntheticBundleError("Expected exactly four unique synthetic memory receipts")
    interrupt_marker = run_dir / "INTERRUPTED_AFTER_DURABLE_BUNDLES.json"
    if interrupt_after_bundles:
        marker = {"status": "SYNTHETIC_INTERRUPTION_INJECTED",
                  "durable_fit_ids": [row["fit_id"] for row in lifecycle["receipts"]],
                  "durable_bundle_sha256": [row["npz_sha256"] for row in lifecycle["receipts"]],
                  "official_fit_starts": 0}
        _write_text_once(interrupt_marker, json.dumps(marker, indent=2, ensure_ascii=False,
                                                       allow_nan=False) + "\n")
        raise SyntheticInterruption("Injected stop after four durable synthetic bundles; no official fit was started")

    truth_path = run_dir / "synthetic_truth_for_independent_verifier.npz"
    truth_sha = _write_truth_after_predictions(run_dir)
    verifier_script = verifier_script or ROOT / "scripts/optimization_v3_rnbr_pilot_verify.py"
    report_path = run_dir / "INDEPENDENT_QA.json"
    result = subprocess.run(
        [sys.executable, "-B", str(verifier_script), "--bundle-dir", str(run_dir / "bundles"),
         "--truth", str(truth_path), "--output", str(report_path)],
        cwd=ROOT, capture_output=True, text=True, encoding="utf-8", check=False,
    )
    if result.returncode != 0:
        raise SyntheticBundleError(f"Independent verifier failed: {result.stderr[-1000:]}")
    verifier_report = json.loads(report_path.read_text(encoding="utf-8"))
    ledger_after = _ledger_hash()
    if ledger_before != ledger_after:
        raise RuntimeError("Synthetic B0 modified the official fit ledger")
    final = {
        **verifier_report,
        "status": verifier_report["status"],
        "synthetic_only": True,
        "synthetic_fit_plan": plan_pilot_fits(freeze_signature=_protocol_signature()),
        "synthetic_memory_builds_total": sum(item["synthetic_fit_count"] for item in lifecycle["receipts"]),
        "synthetic_fit_receipts": lifecycle["receipts"],
        "truth_bundle_sha256": truth_sha,
        "query_targets_accessed_after_predictions_frozen": True,
        "official_train_rows_read": 0,
        "official_targets_read": 0,
        "official_target_memory_constructors": 0,
        "official_candidate_predictions": 0,
        "official_fit_starts": 0,
        "official_ledger_sha256_before": ledger_before,
        "official_ledger_sha256_after": ledger_after,
        "official_ledger_unchanged": True,
        "failure_injection": {
            "interrupted_after_durable_bundles": interrupt_marker.is_file(),
            "resume_reused_verified_bundles": 4 if interrupt_marker.is_file() else 0,
            "marker_sha256": _sha_file(interrupt_marker) if interrupt_marker.is_file() else None,
        },
    }
    output = run_dir / "SYNTHETIC_QA.json"
    _write_text_once(output, json.dumps(final, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    return final


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path,
                        help="Temporary, synthetic-only bundle directory")
    parser.add_argument("--interrupt-after-bundles", action="store_true",
                        help="Inject an interruption after all four synthetic bundles are durable")
    parser.add_argument("--output", type=Path,
                        help="Optional copy of the synthetic QA report; never an official artifact")
    args = parser.parse_args()
    try:
        report = run_synthetic_lifecycle(args.run_dir, interrupt_after_bundles=args.interrupt_after_bundles)
    except SyntheticInterruption as exc:
        print(json.dumps({"status": "SYNTHETIC_INTERRUPTION_INJECTED", "message": str(exc)}))
        return 73
    except Exception as exc:
        print(json.dumps({"status": "FAIL", "error": f"{type(exc).__name__}: {exc}"}))
        return 1
    if args.output:
        _write_text_once(args.output, json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    print(json.dumps({"status": report["status"], "synthetic_only": True,
                      "fits": report["synthetic_memory_builds_total"],
                      "official_fit_starts": report["official_fit_starts"],
                      "ledger_unchanged": report["official_ledger_unchanged"]}))
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
