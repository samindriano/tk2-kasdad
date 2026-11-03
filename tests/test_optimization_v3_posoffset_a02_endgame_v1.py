from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.optimization_v3_posoffset_a02_endgame_v1 import (
    build_submission,
    combine,
    _write_json_once,
    _write_npz_once,
)


def test_both_frozen_formulas_and_exact_missing_route() -> None:
    p_pos = np.asarray([0.2, 0.8, 0.45, 0.99])
    p_a = np.asarray([0.8, 0.2, 0.75, 0.01])
    baseline = np.asarray([0.1, 0.9, 0.25, 0.9])
    missing = np.asarray([False, False, True, True])
    mix = combine("MIX50", p_pos, p_a, baseline, missing)
    residual = combine("ADD_RESIDUAL", p_pos, p_a, baseline, missing)
    np.testing.assert_array_equal(mix, [0.5, 0.5, 0.45, 0.99])
    np.testing.assert_allclose(residual, [0.9, 0.1, 0.45, 0.99], rtol=0, atol=1e-15)


@pytest.mark.parametrize("bad", [np.asarray([0.1, np.nan]), np.asarray([0.1, 1.1])])
def test_invalid_probabilities_fail_closed(bad: np.ndarray) -> None:
    with pytest.raises(ValueError):
        combine("MIX50", bad, np.asarray([0.2, 0.3]), np.asarray([0.4, 0.5]), np.asarray([False, False]))


def test_shape_mismatch_and_unknown_policy_fail_closed() -> None:
    with pytest.raises(ValueError):
        combine("MIX50", [0.1], [0.2, 0.3], [0.4], [False])
    with pytest.raises(ValueError):
        combine("MIX75", [0.1], [0.2], [0.4], [False])


def test_synthetic_end_to_end_unicode_ids_submission_and_npz_readback(tmp_path) -> None:
    ids = np.asarray(["समिती-α", "漏洞-β", "équipement-三"], dtype=str)
    p_pos = np.asarray([0.48, 0.71, 0.49])
    p_a = np.asarray([0.56, 0.62, 0.91])
    b = np.asarray([0.44, 0.70, 0.40])
    missing = np.asarray([False, True, False])
    probs = combine("ADD_RESIDUAL", p_pos, p_a, b, missing)
    frame = build_submission(ids, probs, ids.copy())
    assert list(frame.columns) == ["record_id", "artifact_signal"]
    assert frame.record_id.tolist() == ids.tolist()
    assert frame.artifact_signal.dtype == np.dtype("int8")
    assert frame.artifact_signal.tolist() == [1, 1, 1]

    path = tmp_path / "saved.npz"
    arrays = {"ids": ids, "probabilities": probs, "missing": missing,
              "labels": frame.artifact_signal.to_numpy(dtype=np.int8)}
    _write_npz_once(path, arrays)
    with np.load(path, allow_pickle=False) as loaded:
        assert set(loaded.files) == set(arrays)
        for key, value in arrays.items():
            np.testing.assert_array_equal(loaded[key], value)
    _write_npz_once(path, arrays)  # idempotent resume
    with pytest.raises(RuntimeError):
        _write_npz_once(path, {**arrays, "probabilities": np.asarray([0.0, 0.0, 0.0])})


def test_submission_rejects_permuted_and_duplicate_ids() -> None:
    ids = np.asarray(["A", "B"])
    with pytest.raises(ValueError):
        build_submission(ids[::-1], [0.2, 0.8], ids)
    with pytest.raises(ValueError):
        build_submission(["A", "A"], [0.2, 0.8], ["A", "A"])


def test_json_persistence_is_idempotent_and_detects_tampering(tmp_path) -> None:
    path = tmp_path / "receipt.json"
    receipt = {"schema": 1, "unicode": "β", "value": [1, 2]}
    _write_json_once(path, receipt)
    original = hashlib.sha256(path.read_bytes()).hexdigest()
    _write_json_once(path, receipt)
    assert hashlib.sha256(path.read_bytes()).hexdigest() == original
    with pytest.raises(RuntimeError):
        _write_json_once(path, {**receipt, "value": [1, 3]})
    path.write_text(path.read_text(encoding="utf-8").replace("β", "δ"), encoding="utf-8")
    assert hashlib.sha256(path.read_bytes()).hexdigest() != original


def test_csv_writer_roundtrip_and_integer_target(tmp_path) -> None:
    ids = np.asarray(["A-01", "B-02"])
    frame = build_submission(ids, np.asarray([0.49999, 0.5]), ids)
    path = tmp_path / "submission.csv"
    frame.to_csv(path, index=False)
    reopened = pd.read_csv(path, dtype={"record_id": str})
    assert list(reopened.columns) == ["record_id", "artifact_signal"]
    assert reopened.record_id.tolist() == ids.tolist()
    assert reopened.artifact_signal.tolist() == [0, 1]
