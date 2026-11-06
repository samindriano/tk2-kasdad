from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "scripts" / "reg_pos_disagreement_final_e79_v1.py"
SPEC = importlib.util.spec_from_file_location("reg_pos_disagreement_final_e79_source_test", RUNNER)
runner = importlib.util.module_from_spec(SPEC)
assert SPEC is not None and SPEC.loader is not None
SPEC.loader.exec_module(runner)


def test_exact_champion_order_and_shrink_candidate_are_separate():
    baseline = np.asarray([19.5, 53.3, 99.0, 40.2], dtype=np.float64)
    pos = np.asarray([20.7, 46.1, 95.0, 61.4], dtype=np.float64)
    missing = np.asarray([False, False, True, False])
    champion = np.clip(0.75 * baseline + 0.25 * pos, 0.0, 100.0)
    champion[missing] = baseline[missing]
    expected_candidate = np.clip(baseline + 0.25 * np.exp(-np.abs(pos - baseline) / 3.0)
                                 * (pos - baseline), 0.0, 100.0)
    expected_candidate[missing] = baseline[missing]
    assert np.array_equal(runner._candidate_from_scale(pos, baseline, missing, 3.0)[2], expected_candidate)
    assert np.array_equal(champion[missing].view("<u8"), baseline[missing].view("<u8"))


def test_synthetic_final_e79_lifecycle_writes_only_temporary_outputs():
    result = runner._synthetic_e2e()
    assert result["status"] == "PASS"
    assert result["temporary_archive_receipt_roundtrip"] is True
    assert result["temporary_candidate_csv_roundtrip"] is True
    assert result["real_inputs_read"] is False
    assert result["real_scale_state_fit_started"] is False
    assert result["real_e79_application_started"] is False
    assert result["test_or_sample_accessed"] is False
    assert result["temporary_outputs_removed"] is True
