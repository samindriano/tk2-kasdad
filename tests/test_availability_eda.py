import numpy as np
import pandas as pd
import pytest

from src.eda.availability import (
    add_global_quantile_bins,
    prepare_availability_frame,
    prevalence_table,
    standardized_contrast,
    wilson_interval,
)


def _small_frame():
    return pd.DataFrame(
        {
            "record_id": ["a", "b", "c", "d", "e", "f"],
            "cycle_code": [47, 47, 58, 58, 69, 69],
            "signal_stream": [None, "x", "", "y", "z", "w"],
            "text_load": [10, 10, 20, 20, 30, 30],
            "artifact_signal": [1, 0, 1, 0, 1, 0],
        }
    )


def test_prepare_frame_preserves_availability_and_period_boundaries():
    train = _small_frame()
    result = prepare_availability_frame(train, [True, False, True, False, False, False])
    assert result["availability"].tolist() == ["missing", "present", "missing", "present", "present", "present"]
    assert result["period"].tolist() == ["47–57", "47–57", "58–68", "58–68", "69–79", "69–79"]
    assert result["artifact_signal"].dtype.kind in "iu"


def test_prepare_frame_rejects_invalid_target_and_misaligned_mask():
    train = _small_frame()
    with pytest.raises(ValueError, match="one value per training row"):
        prepare_availability_frame(train, [True])
    train.loc[0, "artifact_signal"] = 2
    with pytest.raises(ValueError, match="binary 0/1"):
        prepare_availability_frame(train, [True] * len(train))


def test_prevalence_table_retains_positive_count_and_denominator():
    frame = prepare_availability_frame(_small_frame(), [True, False, True, False, False, False])
    table = prevalence_table(frame, ["availability"])
    missing = table.set_index("availability").loc["missing"]
    assert missing["n"] == 2
    assert missing["positive"] == 2
    assert missing["prevalence"] == 1.0


def test_standardized_contrast_uses_common_pooled_cell_weights_and_reports_support():
    rows = []
    # Cell A: present 9/90, missing 2/10. Cell B: present 50/100, missing 9/10.
    for cell, availability, positives, n in [
        ("A", "present", 9, 90),
        ("A", "missing", 2, 10),
        ("B", "present", 50, 100),
        ("B", "missing", 9, 10),
        ("C", "present", 20, 30),  # unsupported: missing group has no rows
    ]:
        rows.extend(
            {"stratum": cell, "availability": availability, "artifact_signal": int(i < positives)}
            for i in range(n)
        )
    frame = pd.DataFrame(rows)
    summary, cells = standardized_contrast(frame, ["stratum"], min_support=1)
    expected_present = (100 / 210) * 0.10 + (110 / 210) * 0.50
    expected_missing = (100 / 210) * 0.20 + (110 / 210) * 0.90
    assert summary["standardized_prevalence"]["present"] == pytest.approx(expected_present)
    assert summary["standardized_prevalence"]["missing"] == pytest.approx(expected_missing)
    assert summary["cells_total"] == 3
    assert summary["cells_supported"] == 2
    assert summary["supported_rows_total"] == 210
    assert summary["covered_rows_by_availability"] == {"present": 190, "missing": 20}
    assert summary["standardized_missing_minus_present_pp"] == pytest.approx(100 * (expected_missing - expected_present))
    assert not cells.set_index("stratum").loc["C", "supported"]


def test_quantile_edges_use_text_load_only_and_wilson_interval_is_bounded():
    frame = prepare_availability_frame(_small_frame(), [True, False, True, False, False, False])
    binned, edges = add_global_quantile_bins(frame, 3)
    changed_target = frame.copy()
    changed_target["artifact_signal"] = 1 - changed_target["artifact_signal"]
    binned_changed, edges_changed = add_global_quantile_bins(changed_target, 3)
    assert np.array_equal(binned["text_load_q3"], binned_changed["text_load_q3"])
    assert np.array_equal(edges, edges_changed)
    low, high = wilson_interval(16, 101)
    assert 0 <= low < 16 / 101 < high <= 1

