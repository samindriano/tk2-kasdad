"""Pure integrity checks for the supplemental E79 artifact verifier.

These helpers do not read project data, mutate artifacts, or access the budget
ledger. They make the verifier's receipt, accounting, and CSV round-trip checks
directly testable with synthetic values.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Mapping, Sequence

import numpy as np


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def validate_e79_fit_receipt(
    receipt: Mapping[str, Any],
    *,
    expected_fit_id: str,
    expected_freeze_signature: str,
    expected_bundle_sha256: str,
    expected_bank_matrix_sha256: str,
) -> None:
    """Validate the E79 receipt self-hash and its frozen artifact bindings."""
    body = {key: value for key, value in receipt.items() if key != "receipt_sha256"}
    actual_receipt_sha256 = hashlib.sha256(_canonical_bytes(body)).hexdigest()
    expected_fields = {
        "fit_id": expected_fit_id,
        "status": "FIT_OUTPUT_DURABLE",
        "freeze_signature": expected_freeze_signature,
        "bundle_sha256": expected_bundle_sha256,
        "bank_matrix_file_sha256": expected_bank_matrix_sha256,
    }
    if receipt.get("receipt_sha256") != actual_receipt_sha256:
        raise ValueError("E79 receipt self-hash does not match its payload")
    for field, expected in expected_fields.items():
        if receipt.get(field) != expected:
            raise ValueError(f"E79 receipt field {field!r} does not match the expected artifact")


def validate_e79_fit_accounting(
    events: Sequence[Mapping[str, Any]],
    *,
    fit_id: str,
    receipt_file_sha256: str,
) -> None:
    """Require exactly one ordered start/save pair linked to the receipt hash."""
    fit_events = [event for event in events if event.get("fit_id") == fit_id]
    starts = [i for i, event in enumerate(fit_events) if event.get("event") == "FIT_STARTED"]
    saved = [i for i, event in enumerate(fit_events) if event.get("event") == "FIT_SAVED"]
    failed = [event for event in fit_events if event.get("event") == "FIT_FAILED"]
    if len(starts) != 1 or len(saved) != 1 or failed:
        raise ValueError("E79 fit accounting must contain one start and one save, with no failure")
    if starts[0] >= saved[0]:
        raise ValueError("E79 FIT_SAVED must follow its unique FIT_STARTED event")
    if fit_events[saved[0]].get("receipt_sha256") != receipt_file_sha256:
        raise ValueError("E79 saved ledger event does not match the durable receipt hash")


def validate_csv_roundtrip(
    raw_predictions: Any,
    parsed_predictions: Any,
    *,
    atol: float = 5e-14,
) -> float:
    """Validate finite, aligned prediction arrays and return max absolute error."""
    raw = np.asarray(raw_predictions, dtype=np.float64)
    parsed = np.asarray(parsed_predictions, dtype=np.float64)
    if raw.ndim != 1 or parsed.ndim != 1 or raw.shape != parsed.shape:
        raise ValueError("E79 CSV predictions must be aligned one-dimensional arrays")
    if not np.isfinite(raw).all() or not np.isfinite(parsed).all():
        raise ValueError("E79 CSV predictions must contain only finite values")
    if not np.isfinite(atol) or atol < 0:
        raise ValueError("E79 CSV round-trip tolerance must be finite and nonnegative")
    max_abs_error = float(np.max(np.abs(raw - parsed))) if raw.size else 0.0
    if max_abs_error > atol:
        raise ValueError(
            f"E79 CSV serialization error {max_abs_error:.17g} exceeds tolerance {atol:.17g}"
        )
    return max_abs_error
