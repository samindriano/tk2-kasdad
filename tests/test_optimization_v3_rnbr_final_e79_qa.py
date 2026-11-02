import hashlib
import json

import numpy as np
import pytest

from scripts.optimization_v3_rnbr_final_e79 import _predict
from src.models.regression_neighbor_rnbr import correction_delta, weighted_median_residual
from src.models.rnbr_final_e79_qa import (
    validate_csv_roundtrip,
    validate_e79_fit_accounting,
    validate_e79_fit_receipt,
)


def _canonical_sha256(payload):
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _receipt():
    body = {
        "fit_id": "RNBR_FINAL_E79_OBSERVED",
        "status": "FIT_OUTPUT_DURABLE",
        "freeze_signature": "f" * 64,
        "bundle_sha256": "b" * 64,
        "bank_matrix_file_sha256": "d" * 64,
        "note": "synthetic fixture",
    }
    return {**body, "receipt_sha256": _canonical_sha256(body)}


def test_weighted_median_midpoint_formula_and_clipping():
    median = weighted_median_residual(
        np.array([0.0, 10.0]), np.array([1.0, 1.0]), np.array([8, 3])
    )
    assert median == 5.0  # exact half mass returns the midpoint of adjacent values
    assert correction_delta(median) == 1.25
    assert correction_delta(40.0) == 5.0
    assert correction_delta(-40.0) == -5.0


def test_missing_and_under_supported_queries_fall_back_to_exact_r06():
    donor_rows = np.array([10, 11, 12, 13, 14], dtype=np.int64)
    donor_residuals = np.full(5, -8.0)
    neighbor_rows = np.full((3, 5), -1, dtype=np.int64)
    similarities = np.zeros((3, 5), dtype=np.float64)
    neighbor_rows[:, :5] = donor_rows
    similarities[:, :5] = 1.0
    counts = np.array([5, 4, 5], dtype=np.int64)
    missing = np.array([True, False, False])
    baseline = np.array([25.0, 50.0, 99.0])

    prediction, delta, median = _predict(
        baseline, missing, neighbor_rows, similarities, counts, donor_rows, donor_residuals
    )

    assert prediction[0] == baseline[0]  # missing stream
    assert prediction[1] == baseline[1]  # only four positive neighbors
    assert delta[:2].tolist() == [0.0, 0.0]
    assert median[:2].tolist() == [0.0, 0.0]
    assert prediction[2] == 97.0


def test_csv_roundtrip_accepts_declared_float_tolerance():
    raw = np.array([0.0, 13.25, 100.0], dtype=np.float64)
    parsed = raw.copy()
    parsed[1] += 2.0e-14
    assert validate_csv_roundtrip(raw, parsed, atol=5e-14) == pytest.approx(2.0e-14)


def test_csv_roundtrip_rejects_error_shape_and_nonfinite_values():
    raw = np.array([1.0, 2.0], dtype=np.float64)
    with pytest.raises(ValueError, match="exceeds tolerance"):
        validate_csv_roundtrip(raw, raw + np.array([0.0, 6e-14]), atol=5e-14)
    with pytest.raises(ValueError, match="aligned"):
        validate_csv_roundtrip(raw, np.array([1.0]))
    with pytest.raises(ValueError, match="finite"):
        validate_csv_roundtrip(raw, np.array([1.0, np.nan]))


def test_receipt_accepts_valid_payload_and_rejects_hash_or_lineage_tampering():
    receipt = _receipt()
    kwargs = {
        "expected_fit_id": "RNBR_FINAL_E79_OBSERVED",
        "expected_freeze_signature": "f" * 64,
        "expected_bundle_sha256": "b" * 64,
        "expected_bank_matrix_sha256": "d" * 64,
    }
    assert validate_e79_fit_receipt(receipt, **kwargs) is None

    wrong_hash = {**receipt, "receipt_sha256": "0" * 64}
    with pytest.raises(ValueError, match="self-hash"):
        validate_e79_fit_receipt(wrong_hash, **kwargs)

    changed_payload = {**receipt, "bundle_sha256": "c" * 64}
    with pytest.raises(ValueError, match="self-hash"):
        validate_e79_fit_receipt(changed_payload, **kwargs)

    wrong_binding = {**receipt, "receipt_sha256": _canonical_sha256(
        {key: value for key, value in receipt.items() if key != "receipt_sha256"}
    )}
    with pytest.raises(ValueError, match="bundle_sha256"):
        validate_e79_fit_receipt(
            wrong_binding, **{**kwargs, "expected_bundle_sha256": "c" * 64}
        )


def test_fit_accounting_requires_one_ordered_save_with_receipt_hash():
    receipt_sha = "a" * 64
    events = [
        {"fit_id": "E79", "event": "FIT_STARTED"},
        {"fit_id": "OTHER", "event": "FIT_STARTED"},
        {"fit_id": "E79", "event": "FIT_SAVED", "receipt_sha256": receipt_sha},
    ]
    assert validate_e79_fit_accounting(events, fit_id="E79", receipt_file_sha256=receipt_sha) is None

    bad_hash = [*events[:2], {**events[2], "receipt_sha256": "0" * 64}]
    with pytest.raises(ValueError, match="receipt hash"):
        validate_e79_fit_accounting(bad_hash, fit_id="E79", receipt_file_sha256=receipt_sha)
    with pytest.raises(ValueError, match="one start and one save"):
        validate_e79_fit_accounting(
            [*events, events[0]], fit_id="E79", receipt_file_sha256=receipt_sha
        )
    with pytest.raises(ValueError, match="one start and one save"):
        validate_e79_fit_accounting(
            [*events, {"fit_id": "E79", "event": "FIT_FAILED"}],
            fit_id="E79",
            receipt_file_sha256=receipt_sha,
        )
