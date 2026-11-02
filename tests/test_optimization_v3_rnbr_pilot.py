from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.optimization_v3_rnbr_pilot_synthetic import (  # noqa: E402
    EXPECTED_LEDGER_SHA256,
    make_synthetic_case,
)
from src.models.regression_neighbor_rnbr import (  # noqa: E402
    RnbrProtocolError,
    correction_delta,
    fixed_prediction_decile,
    plan_pilot_fits,
    predict_from_saved_neighbors,
    shuffle_residuals_within_cycle_decile,
    validate_saved_geometry,
    validate_donor_value_lineage,
    weighted_median_residual,
)


def test_weighted_median_exact_half_returns_midpoint() -> None:
    assert weighted_median_residual([-1.0, 1.0], [1.0, 1.0], [9, 2]) == 0.0


def test_weighted_median_respects_cosine_squared_weight() -> None:
    assert weighted_median_residual([7.0, 2.0, 9.0], [1.0, 0.5, 0.5], [2, 1, 3]) == 7.0


@pytest.mark.parametrize(("median", "expected"), [(30.0, 5.0), (8.0, 2.0), (-30.0, -5.0)])
def test_correction_scales_before_capping(median: float, expected: float) -> None:
    assert correction_delta(median) == expected


def test_fixed_prediction_deciles_include_fixed_boundaries() -> None:
    assert fixed_prediction_decile([-1, 0, 9.99, 10, 99.99, 100, 105]).tolist() == [0, 0, 0, 1, 9, 9, 9]


def test_shuffle_is_deterministic_and_preserves_each_exact_cell_multiset() -> None:
    residuals = np.asarray([1, 2, 3, 4, 5, 6, 7, 8], dtype=float)
    cycles = np.asarray([56, 56, 56, 56, 57, 57, 57, 57])
    predictions = np.asarray([1, 1, 12, 12, 1, 1, 12, 12], dtype=float)
    rows = np.asarray([30, 10, 40, 20, 70, 50, 80, 60])
    first = shuffle_residuals_within_cycle_decile(residuals, cycles, predictions, rows)
    second = shuffle_residuals_within_cycle_decile(residuals, cycles, predictions, rows)
    assert np.array_equal(first, second)
    deciles = fixed_prediction_decile(predictions)
    for cycle in np.unique(cycles):
        for decile in np.unique(deciles):
            mask = (cycles == cycle) & (deciles == decile)
            assert sorted(first[mask].tolist()) == sorted(residuals[mask].tolist())


def test_geometry_integrity_is_fail_closed() -> None:
    case = make_synthetic_case(67)
    kwargs = {
        "origin": case.origin, "donor_rows": case.donor_rows, "donor_ids": case.donor_ids,
        "donor_cycles": case.donor_cycles, "donor_source_cutoffs": case.donor_source_cutoffs,
        "donor_present": case.donor_present, "query_rows": case.query_rows,
        "query_ids": case.query_ids, "query_cycles": case.query_cycles,
        "query_missing": case.query_missing, "neighbor_rows": case.neighbor_rows,
        "neighbor_ids": case.neighbor_ids, "neighbor_cosines": case.neighbor_cosines,
        "neighbor_counts": case.neighbor_counts,
    }
    assert validate_saved_geometry(**kwargs)["status"] == 1
    corrupted_ids = case.neighbor_ids.copy()
    corrupted_ids[0, 0] = "str_:WRONG"
    kwargs["neighbor_ids"] = corrupted_ids
    with pytest.raises(RnbrProtocolError, match="Neighbor ID does not match"):
        validate_saved_geometry(**kwargs)


def test_donor_target_and_r06_oof_lineage_requires_exact_row_id_cycle_match() -> None:
    case = make_synthetic_case(67)
    kwargs = {
        "origin": case.origin, "donor_rows": case.donor_rows, "donor_ids": case.donor_ids,
        "donor_cycles": case.donor_cycles, "donor_source_cutoffs": case.donor_source_cutoffs,
        "target_rows": case.donor_rows.copy(), "target_ids": case.donor_ids.copy(),
        "target_cycles": case.donor_cycles.copy(), "target_values": case.donor_targets.copy(),
        "r06_rows": case.donor_rows.copy(), "r06_ids": case.donor_ids.copy(),
        "r06_cycles": case.donor_cycles.copy(), "r06_predictions": case.donor_r06_predictions.copy(),
    }
    assert validate_donor_value_lineage(**kwargs)["status"] == 1
    kwargs["target_ids"] = case.donor_ids[::-1].copy()
    with pytest.raises(RnbrProtocolError, match="row/ID/cycle alignment"):
        validate_donor_value_lineage(**kwargs)


def test_missing_and_under_supported_queries_fall_back_to_r06() -> None:
    case = make_synthetic_case(67)
    prediction = predict_from_saved_neighbors(
        donor_rows=case.donor_rows,
        donor_residuals=case.donor_targets - case.donor_r06_predictions,
        query_r06_predictions=case.query_r06_predictions,
        query_missing=case.query_missing,
        neighbor_rows=case.neighbor_rows,
        neighbor_cosines=case.neighbor_cosines,
        neighbor_counts=case.neighbor_counts,
    )
    assert np.array_equal(prediction["prediction"][case.query_missing], case.query_r06_predictions[case.query_missing])
    assert prediction["delta"][-1] == 0.0  # present query with only four neighbors
    assert prediction["neighbor_count_used"][-1] == 0
    assert prediction["delta"][0] > 0.0


def test_pilot_plan_is_request_only_and_exactly_four_unique_slots() -> None:
    plan = plan_pilot_fits(freeze_signature="a" * 64)
    assert len(plan) == len({row["fit_id"] for row in plan}) == 4
    assert {(row["origin"], row["arm"]) for row in plan} == {
        (67, "observed"), (67, "shuffled"), (75, "observed"), (75, "shuffled")
    }
    assert all(row["status"] == "REQUEST_ONLY_NOT_RESERVED" for row in plan)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_synthetic_failure_resume_independent_verifier_and_corruption_guard(tmp_path: Path) -> None:
    script = ROOT / "scripts/optimization_v3_rnbr_pilot_synthetic.py"
    run_dir = tmp_path / "lifecycle"
    ledger = ROOT / "experiments/runs/optimization_v3_research_next_v1/budget.jsonl"
    assert _sha256(ledger) == EXPECTED_LEDGER_SHA256
    env = os.environ.copy()
    first = subprocess.run(
        [sys.executable, "-B", str(script), "--run-dir", str(run_dir), "--interrupt-after-bundles"],
        cwd=ROOT, env=env, capture_output=True, text=True, check=False,
    )
    assert first.returncode == 73
    assert len(list((run_dir / "bundles").glob("E*.npz"))) == 4
    assert not (run_dir / "SYNTHETIC_QA.json").exists()

    resumed = subprocess.run(
        [sys.executable, "-B", str(script), "--run-dir", str(run_dir)],
        cwd=ROOT, env=env, capture_output=True, text=True, check=False,
    )
    assert resumed.returncode == 0, resumed.stdout + resumed.stderr
    result = json.loads((run_dir / "SYNTHETIC_QA.json").read_text(encoding="utf-8"))
    assert result["status"] == "PASS"
    assert result["synthetic_memory_builds_total"] == 4
    assert result["official_fit_starts"] == 0
    assert result["official_ledger_unchanged"] is True
    assert result["failure_injection"]["interrupted_after_durable_bundles"] is True
    assert result["failure_injection"]["resume_reused_verified_bundles"] == 4
    assert result["synthetic_primary_n"] == 2 * 601
    assert result["synthetic_H12_L_n"] == 2 * 601 + 300
    assert result["independent_prediction_recompute"] == "PASS"
    assert _sha256(ledger) == EXPECTED_LEDGER_SHA256

    bundle = run_dir / "bundles/E67_observed.npz"
    with bundle.open("ab") as stream:
        stream.write(b"corruption")
    corrupted_sha = _sha256(bundle)
    failed = subprocess.run(
        [sys.executable, "-B", str(script), "--run-dir", str(run_dir)],
        cwd=ROOT, env=env, capture_output=True, text=True, check=False,
    )
    assert failed.returncode == 1
    assert "verification failed" in (failed.stdout + failed.stderr).lower()
    assert _sha256(bundle) == corrupted_sha  # no hidden rebuild or overwrite
    assert _sha256(ledger) == EXPECTED_LEDGER_SHA256
