"""Synthetic end-to-end, persistence, accounting, and safe-resume checks.

This module intentionally does not read any dataset. It builds tiny in-memory
arrays and uses a temporary copy of the append-only fit ledger.
"""
from __future__ import annotations

import hashlib
import json
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import optimization_v3_rnbr_target_median_v1 as runner
from src.models.regression_neighbor_rnbr_target_median import (
    predict_from_target_median,
    shuffle_targets_within_cycle_decile,
    weighted_median_target,
)


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
                      allow_nan=False).encode("utf-8")


def _input(origin: int) -> dict[str, np.ndarray]:
    n_donor = 40
    base_row = 1000 if origin == 67 else 2000
    donor_rows = np.arange(base_row, base_row + n_donor, dtype=np.int64)
    donor_cycles = np.full(n_donor, 56, dtype=np.int64)
    donor_r06 = np.full(n_donor, 5.0, dtype=np.float64)
    query_rows = np.arange((5000 if origin == 67 else 6000),
                           (5000 if origin == 67 else 6000) + 4, dtype=np.int64)
    query_cycles = np.asarray([68, 69, 70, 71] if origin == 67 else [76, 77, 78, 79], dtype=np.int64)
    neighbor_rows = np.full((4, 32), -1, dtype=np.int64)
    neighbor_cosines = np.zeros((4, 32), dtype=np.float64)
    neighbor_counts = np.asarray([5, 4, 0, 5], dtype=np.int16)
    for i, count in enumerate(neighbor_counts):
        if count:
            start = i * 5
            neighbor_rows[i, :count] = donor_rows[start:start + count]
            neighbor_cosines[i, :count] = np.linspace(0.9, 0.5, count)
    return {
        "donor_rows": donor_rows,
        "donor_ids": np.asarray([f"SYN-D-{origin}-{i}" for i in range(n_donor)], dtype="U64"),
        "donor_cycles": donor_cycles,
        "donor_source_cutoffs": np.full(n_donor, 55, dtype=np.int64),
        "donor_r06_predictions": donor_r06,
        "donor_r06_oof": donor_r06,
        "query_rows": query_rows,
        "query_ids": np.asarray([f"SYN-Q-{origin}-{i}" for i in range(4)], dtype="U64"),
        "query_cycles": query_cycles,
        "query_missing": np.asarray([False, False, False, True]),
        "query_baseline": np.asarray([50.0, 42.0, 30.0, 66.0]),
        "neighbor_rows": neighbor_rows,
        "neighbor_ids": np.where(neighbor_rows >= 0, "SYN-NEIGHBOR", "").astype("U64"),
        "neighbor_cosines": neighbor_cosines,
        "neighbor_counts": neighbor_counts,
        "original_prediction": np.asarray([52.0, 42.0, 30.0, 66.0]),
        "original_bundle_sha256": np.asarray(["0" * 64], dtype="U64"),
    }


def run_synthetic_qa(output_path: Path = runner.SYNTHETIC_PATH) -> dict[str, Any]:
    # A minimal distinction case: the direct target median and residual median
    # intentionally differ, so accidentally reusing residual semantics fails.
    target = weighted_median_target(np.asarray([40.0, 80.0]), np.asarray([1.0, 1.0]), np.asarray([1, 2]))
    residual = weighted_median_target(np.asarray([30.0, 10.0]), np.asarray([1.0, 1.0]), np.asarray([1, 2]))
    assert target == 60.0 and residual == 20.0
    direct_prediction = float(np.clip(50.0 + np.clip(0.25 * (target - 50.0), -5, 5), 0, 100))
    residual_prediction = float(np.clip(50.0 + np.clip(0.25 * residual, -5, 5), 0, 100))
    assert direct_prediction == 52.5 and residual_prediction == 55.0

    # Exact-half midpoint, duplicate-target grouping, deterministic shuffle,
    # within-cell target multiset preservation, and exact R06 fallback.
    assert weighted_median_target(np.asarray([10.0, 30.0]), np.ones(2), np.asarray([2, 1])) == 20.0
    assert weighted_median_target(np.asarray([10.0, 10.0, 40.0]), np.ones(3), np.asarray([3, 1, 2])) == 10.0
    cycles = np.repeat(np.asarray([56, 57]), 32)
    predictions = np.tile(np.asarray([5.0] * 32), 2)
    rows = np.arange(100, 164, dtype=np.int64)
    targets = np.linspace(0, 100, len(rows))
    shuffled1, source_rows1, audit1 = shuffle_targets_within_cycle_decile(targets, cycles, predictions, rows, seed=42)
    shuffled2, source_rows2, audit2 = shuffle_targets_within_cycle_decile(targets, cycles, predictions, rows, seed=42)
    assert np.array_equal(shuffled1, shuffled2) and np.array_equal(source_rows1, source_rows2)
    assert np.array_equal(audit1["source_positions"], audit2["source_positions"])
    assert np.array_equal(np.sort(shuffled1), np.sort(targets)) and audit1["fraction_moved"] >= 0.8

    fallback = predict_from_target_median(
        donor_rows=np.arange(10), donor_targets=np.arange(10, dtype=np.float64),
        query_r06_predictions=np.asarray([40.0, 30.0]), query_missing=np.asarray([False, True]),
        neighbor_rows=np.asarray([[0, 1, 2, 3, -1], [0, 1, 2, 3, 4]]),
        neighbor_cosines=np.asarray([[0.9, 0.8, 0.7, 0.6, 0], [0.9, 0.8, 0.7, 0.6, 0.5]]),
        neighbor_counts=np.asarray([4, 5]),
    )
    assert np.array_equal(fallback["prediction"], np.asarray([40.0, 30.0]))
    assert not fallback["median_applied"].any()

    # Exercise the actual runner persistence format and fit ledger functions
    # against temporary synthetic data and a temporary copy of the live ledger.
    live_events = runner._ledger_events(runner.LEDGER_PATH)
    live_snapshot = runner.ledger_snapshot(live_events)
    live_ledger_hash = runner._sha_file(runner.LEDGER_PATH)
    if any(live_snapshot.get(key) != value for key, value in runner.PREVIOUS_LEDGER_SNAPSHOT.items()):
        raise AssertionError(f"Synthetic lifecycle requires the frozen pre-amendment ledger: {live_snapshot}")
    source_commit = runner._git("rev-parse", "HEAD")
    freeze_body = {
        "schema_version": 1, "pilot_id": runner.PILOT_ID,
        "status": "SYNTHETIC_ONLY", "source_commit": source_commit,
        "input_hashes": {"synthetic": "0" * 64}, "geometry_hashes": {"67": {}, "75": {}},
        "ledger_sha_before": live_ledger_hash,
    }
    freeze_body["freeze_signature"] = hashlib.sha256(_canonical(freeze_body)).hexdigest()
    resource = {"jobs": 1, "threads": 2, "process_rss_before_fit_bytes": 0,
                "available_ram_before_fit_bytes": 8 * 1024**3,
                "max_process_rss_bytes": runner.RESOURCE_MAX_RSS_BYTES,
                "minimum_free_ram_bytes": runner.RESOURCE_MIN_FREE_BYTES,
                "estimated_incremental_memory_bytes": 512 * 1024**2}
    target_vectors = {origin: np.linspace(5, 95, 40, dtype=np.float64) for origin in runner.ORIGINS}
    with tempfile.TemporaryDirectory(prefix="rnbr-target-median-synthetic-") as temp:
        temp_root = Path(temp)
        temp_ledger = temp_root / "budget.jsonl"
        shutil.copyfile(runner.LEDGER_PATH, temp_ledger)
        artifact_root = temp_root / "artifacts"
        budget = runner._apply_cap_amendment_and_reserve(temp_ledger, freeze_body)
        for origin in runner.ORIGINS:
            origin_input = _input(origin)
            for arm in runner.ARMS:
                fit_id = f"{runner.PILOT_ID}_E{origin}_{arm.upper()}"
                runner._start_fit(temp_ledger, freeze_body, fit_id, origin, arm, resource)
                built = runner._build_fit_bundle(
                    fit_id, origin, arm, freeze_body, origin_input,
                    target_payload={"target": target_vectors[origin]}, resource=resource,
                    run_root=artifact_root,
                )
                checked = runner._verify_memory_arrays(
                    fit_id, built["arrays"], origin_input,
                    expected_target_actual=target_vectors[origin],
                )
                assert checked["status"] == "PASS"
                if (origin, arm) == (75, "shuffled"):
                    # Simulate a process interruption after durable receipt/NPZ
                    # but before FIT_SAVED, then reconcile from saved state.
                    arrays, receipt = runner._load_bundle(fit_id, freeze_body, run_root=artifact_root)
                    resumed = runner._verify_memory_arrays(
                        fit_id, arrays, origin_input, expected_target_actual=target_vectors[origin]
                    )
                    assert resumed["prediction_sha256"] == checked["prediction_sha256"]
                    runner._finish_fit(temp_ledger, fit_id, "FIT_SAVED",
                                       receipt_sha256=runner._sha_file(built["receipt_path"]))
                    assert receipt["fit_id"] == fit_id
                else:
                    runner._finish_fit(temp_ledger, fit_id, "FIT_SAVED",
                                       receipt_sha256=runner._sha_file(built["receipt_path"]))
        before_resume = runner.ledger_snapshot(runner._ledger_events(temp_ledger))
        starts_before = before_resume["fit_starts"]
        # A completed-run resume is read-only: verify every saved NPZ/receipt,
        # then repeat the idempotent terminal reconciliation.
        for origin in runner.ORIGINS:
            origin_input = _input(origin)
            for arm in runner.ARMS:
                fit_id = f"{runner.PILOT_ID}_E{origin}_{arm.upper()}"
                arrays, receipt = runner._load_bundle(fit_id, freeze_body, run_root=artifact_root)
                runner._verify_memory_arrays(fit_id, arrays, origin_input,
                                             expected_target_actual=target_vectors[origin])
                runner._finish_fit(temp_ledger, fit_id, "FIT_SAVED",
                                   receipt_sha256=runner._sha_file(runner._bundle_dir(fit_id, artifact_root) / "receipt.json"))
                assert receipt["freeze_signature"] == freeze_body["freeze_signature"]
        after_resume = runner.ledger_snapshot(runner._ledger_events(temp_ledger))
        assert after_resume == before_resume and after_resume["fit_starts"] == starts_before
        assert after_resume["fit_starts"] == 87 and after_resume["fit_saved"] == 84 and after_resume["fit_pending"] == 0
        bundle_hashes = {}
        for origin in runner.ORIGINS:
            for arm in runner.ARMS:
                fit_id = f"{runner.PILOT_ID}_E{origin}_{arm.upper()}"
                bundle_hashes[fit_id] = runner._sha_file(runner._bundle_dir(fit_id, artifact_root) / "arrays.npz")

    report = {
        "status": "PASS_SYNTHETIC_END_TO_END",
        "target_or_test_data_accessed": False,
        "fit_ledger_used": "temporary_copy_only",
        "live_ledger_sha256_before_after": {"before": live_ledger_hash, "after": runner._sha_file(runner.LEDGER_PATH)},
        "live_ledger_unchanged": runner._sha_file(runner.LEDGER_PATH) == live_ledger_hash,
        "direct_target_median": target, "residual_median_distinguishing_case": residual,
        "direct_prediction_distinguishing_case": direct_prediction,
        "residual_prediction_distinguishing_case": residual_prediction,
        "seed42_shuffle_fraction_moved": float(audit1["fraction_moved"]),
        "synthetic_fit_starts_and_saves": {"starts": after_resume["fit_starts"] - 83,
                                            "saved": after_resume["fit_saved"] - 80,
                                            "pending": after_resume["fit_pending"]},
        "resume_added_starts": after_resume["fit_starts"] - starts_before,
        "temporary_bundle_npz_hashes": bundle_hashes,
        "temporary_budget_reservation": budget,
        "assertions": ["direct target median differs from residual median", "exact-half tie midpoint",
                       "duplicate targets grouped", "seed42 shuffle deterministic and multiset-preserving",
                       "missing and fewer-than-five fallback exactly to R06", "four temporary fits persisted",
                       "interrupted saved receipt reconciled without refit", "completed resume adds no starts",
                       "temporary ledger reaches cap 87 with four saves", "live ledger unchanged"],
    }
    if not report["live_ledger_unchanged"]:
        raise AssertionError("Synthetic QA unexpectedly changed the live ledger")
    runner._write_json_once(output_path, report)
    return report


if __name__ == "__main__":
    print(json.dumps(run_synthetic_qa(), ensure_ascii=False, indent=2))
