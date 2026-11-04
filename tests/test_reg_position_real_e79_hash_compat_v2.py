from __future__ import annotations

import numpy as np
import pytest

from scripts.reg_position_real_e79_hash_compat_v2 import (
    legacy_raw_le_f8_sha256,
    structured_f8_sha256,
    validate_legacy_prediction_manifest_digest,
)


def test_legacy_raw_byte_digest_differs_from_structured_array_digest() -> None:
    values = np.asarray([0.0, 1.25, 20.5, 100.0], dtype=np.float64)
    raw_digest = legacy_raw_le_f8_sha256(values)
    structured_digest = structured_f8_sha256(values)

    assert raw_digest != structured_digest
    audit = validate_legacy_prediction_manifest_digest(raw_digest, values)
    assert audit["manifest_prediction_sha256"] == raw_digest
    assert audit["legacy_raw_le_f8_sha256"] == raw_digest
    assert audit["structured_array_sha256"] == structured_digest


def test_legacy_manifest_contract_rejects_wrong_or_structured_digest() -> None:
    values = np.asarray([3.0, 14.125, 91.0], dtype=np.float64)
    with pytest.raises(ValueError, match="manifest_matches_legacy_raw_le_f8"):
        validate_legacy_prediction_manifest_digest("0" * 64, values)
    with pytest.raises(ValueError, match="manifest_matches_legacy_raw_le_f8"):
        validate_legacy_prediction_manifest_digest(structured_f8_sha256(values), values)
