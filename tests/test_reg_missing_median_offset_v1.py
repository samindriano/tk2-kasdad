from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "reg_missing_median_offset_v1_test", ROOT / "scripts" / "reg_missing_median_offset_v1.py"
)
pilot = importlib.util.module_from_spec(SPEC)
assert SPEC is not None and SPEC.loader is not None
SPEC.loader.exec_module(pilot)


def test_median_uses_y_minus_r06_baseline_sign():
    baseline = np.asarray([20.0, 30.0, 40.0, 50.0])
    target = baseline + np.asarray([2.0, 4.0, -2.0, 6.0])
    delta, residual = pilot._median_residual_offset(target, baseline)
    assert np.array_equal(residual, target - baseline)
    assert delta == 3.0


def test_application_changes_missing_only_and_clips():
    baseline = np.asarray([0.5, 99.0, 70.0, 25.0], dtype=np.float64)
    champion = np.asarray([0.5, 99.0, 42.0, 62.0], dtype=np.float64)
    missing = np.asarray([True, True, False, False])
    candidate = pilot._apply_missing_offset(baseline, champion, missing, 5.0)
    assert np.array_equal(candidate[~missing].view("<u8"), champion[~missing].view("<u8"))
    assert np.array_equal(candidate[missing], np.asarray([5.5, 100.0]))


def test_synthetic_prediction_receipt_roundtrip(tmp_path):
    arrays = {
        "rows": np.arange(4, dtype=np.int64),
        "prediction": np.asarray([1.0, 2.0, 3.0, 4.0]),
        "ids": np.asarray(["a", "b", "c", "d"]),
    }
    path = tmp_path / "synthetic.npz"
    receipt = pilot._write_npz_new(path, arrays)
    recovered = pilot._verify_npz(path, receipt)
    assert set(recovered) == set(arrays)
    assert all(np.array_equal(recovered[k], arrays[k]) for k in arrays)

