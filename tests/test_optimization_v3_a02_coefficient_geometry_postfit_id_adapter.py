from __future__ import annotations

import numpy as np
import pytest

from scripts.optimization_v3_a02_coefficient_geometry_postfit_id_adapter import (
    AmendmentError,
    adapt_source_ids,
    assert_array_match,
    explicit_unicode_ids,
)


def test_expected_ids_convert_losslessly_with_long_and_non_ascii_values():
    values = np.asarray(["记录-雪-λ-" + "x" * 350, "café-данные"], dtype=object)
    converted = explicit_unicode_ids(values)
    assert converted.dtype.kind == "U"
    assert converted.tolist() == values.tolist()
    assert len(converted[0]) > 300
    assert_array_match("query_record_ids", values, converted)


@pytest.mark.parametrize(
    "mutate",
    [
        lambda ids: np.asarray([str(ids[0])[:12], ids[1]], dtype="<U12"),
        lambda ids: ids[::-1].copy(),
        lambda ids: np.asarray(["altered-value", ids[1]], dtype=str),
        lambda ids: ids[:1].copy(),
        lambda ids: ids.astype(object),
    ],
    ids=["truncation", "reordering", "altered-value", "wrong-shape", "stored-object"],
)
def test_expected_id_comparison_rejects_identity_or_storage_tampering(mutate):
    expected = np.asarray(["query-雪-" + "q" * 100, "record-λ-002"], dtype=object)
    valid = explicit_unicode_ids(expected)
    tampered = mutate(valid)
    with pytest.raises(AmendmentError):
        assert_array_match("query_record_ids", expected, tampered)


def test_source_adapter_changes_only_expected_id_representation():
    rows = np.asarray([14, 27], dtype=np.int64)
    source = {"score_ids": np.asarray(["id-α", "id-β"], dtype=object), "score_rows": rows}
    adapted = adapt_source_ids(source)
    assert adapted is not source
    assert adapted["score_ids"].dtype.kind == "U"
    assert adapted["score_ids"].tolist() == ["id-α", "id-β"]
    assert adapted["score_rows"] is rows


def test_numeric_array_dtype_tolerance_shape_and_value_checks_remain_strict():
    expected = np.asarray([0.25, 0.75], dtype=np.float64)
    assert_array_match("candidate_probability", expected, expected.copy())
    within_frozen_tolerance = np.asarray([0.25 + 5e-13, 0.75], dtype=np.float64)
    assert_array_match("candidate_probability", expected, within_frozen_tolerance)
    with pytest.raises(AmendmentError, match="numeric values"):
        assert_array_match("candidate_probability", expected, np.asarray([0.25001, 0.75], dtype=np.float64))
    with pytest.raises(AmendmentError, match="dtype"):
        assert_array_match("candidate_probability", expected, expected.astype(np.float32))
    with pytest.raises(AmendmentError, match="shape"):
        assert_array_match("candidate_probability", expected, expected[:1])
    with pytest.raises(AmendmentError, match="object array"):
        assert_array_match("candidate_probability", expected, expected.astype(object))


def test_exact_non_string_expected_ids_fail_closed():
    with pytest.raises(AmendmentError, match="non-null strings"):
        explicit_unicode_ids(np.asarray(["ok", None], dtype=object))
