from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "reg_missing_affine_l1_v1_test", ROOT / "scripts" / "reg_missing_affine_l1_v1.py"
)
assert SPEC is not None and SPEC.loader is not None
pilot = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(pilot)


def test_sparse_lp_recovers_known_monotone_affine_relationship():
    x = np.linspace(0.0, 100.0, 121)
    noise = ((np.arange(len(x)) % 5) - 2) * 0.05
    y = 7.0 + 0.65 * x + noise
    result = pilot._solve_affine(x, y)
    assert result["success"] is True
    assert result["solver"] == "scipy.optimize.linprog(method='highs')"
    assert result["solver_status"] == 0
    assert result["slope"] >= 0.0
    assert result["primal_feasibility"] == "PASS"
    assert np.isclose(result["intercept"], 7.0, atol=1e-7)
    assert np.isclose(result["slope"], 0.65, atol=1e-7)
    assert np.isclose(result["objective_l1_sum"], np.abs(y - (result["intercept"] + result["slope"] * x)).sum())


def test_support_requires_rows_distinct_predictions_and_iqr():
    good = np.linspace(2.0, 98.0, 120)
    assert pilot._support_stats(good)["support_pass"] is True
    assert pilot._support_stats(np.linspace(2.0, 98.0, 19))["support_pass"] is False
    assert pilot._support_stats(np.full(120, 40.0))["support_pass"] is False
    assert pilot._support_stats(np.linspace(40.0, 44.0, 120))["support_pass"] is False


def test_application_changes_missing_only_and_clips():
    baseline = np.asarray([0.5, 99.0, 70.0, 25.0], dtype=np.float64)
    champion = np.asarray([0.5, 99.0, 42.0, 62.0], dtype=np.float64)
    missing = np.asarray([True, True, False, False])
    candidate = pilot._apply_affine(baseline, champion, missing, 5.0, 1.5)
    assert np.array_equal(candidate, np.asarray([5.75, 100.0, 42.0, 62.0]))
    assert np.array_equal(candidate[~missing].view("<u8"), champion[~missing].view("<u8"))
    fallback = pilot._apply_affine(baseline, champion, missing, 99.0, 3.0, fallback=True)
    assert np.array_equal(fallback.view("<u8"), champion.view("<u8"))


def test_npz_receipt_roundtrip_preserves_string_ids_and_arrays(tmp_path):
    payload = {"rows": np.arange(5, dtype=np.int64),
               "ids": np.asarray(["D1-a", "D1-b", "D1-c", "D1-d", "D1-e"]),
               "prediction": np.linspace(1.0, 5.0, 5)}
    path = tmp_path / "prediction.npz"
    receipt = pilot._write_npz_new(path, payload)
    result = pilot._verify_npz(path, receipt)
    assert all(np.array_equal(result[key], payload[key]) for key in payload)


def test_actual_synthetic_cli_lifecycle_runs_fit_apply_receipts_and_metric_qa():
    result = pilot._synthetic_e2e()
    assert result["status"] == "PASS"
    assert result["synthetic_only"] is True
    assert result["lp_solves"] == 2
    assert result["missing_required_input_rejected"] is True
    assert result["origin_isolation_pass"] is True
    assert result["present_bits_exact"] is True
    assert result["one_shot_retry_rejected"] is True
    assert result["metric_qa"] == "PASS_SYNTHETIC_ONLY"
    assert result["production_ledger_touched"] is False


def test_one_shot_ledger_rejects_repeated_operation_without_append(tmp_path):
    pilot._start_operation("FIT_D1_R1", tmp_path)
    before = (tmp_path / "lane_ledger.jsonl").read_bytes()
    with pytest.raises(pilot.PilotError):
        pilot._start_operation("FIT_D1_R1", tmp_path)
    assert (tmp_path / "lane_ledger.jsonl").read_bytes() == before
