from __future__ import annotations

import numpy as np

from scripts.optimization_v3_batch1_r15_amendment import _expected_fit_population
from src.models.optimization_v3_batch1_r15 import FEATURE_NAMES


def test_amendment_recomputes_missing_fit_row_and_data_fingerprints():
    frame = {
        "impact_load": [1.0, 2.0, 3.0, 4.0],
        "entry_friction": [0.0, 1.0, 0.0, 1.0],
        "cycle_code": [47, 55, 60, 67],
        "scope_pressure": [1.0, 2.0, 3.0, 4.0],
        "text_load": [4.0, 3.0, 2.0, 1.0],
        "scope_trace": ["1", "2", "3", "4"],
        "org_spread": ["1", "2", "3", "4"],
        "module_spread": ["1", "2", "3", "4"],
        "telemetry_flag": [0, 1, 0, 1],
        "org_code": ["a", "a", "b", "b"],
        "module_code": ["m1", "m1", "m2", "m2"],
        "pattern_code": ["p1", "p2", "p1", "p2"],
        "signal_stream": ["stream", None, "  ", "stream"],
    }
    import pandas as pd

    context = {
        "frame": pd.DataFrame(frame, columns=FEATURE_NAMES),
        "cycles": np.array([47, 55, 60, 67]),
        "target": np.array([10.0, 20.0, 30.0, 40.0]),
        "fit_rows_by_cutoff": {67: np.arange(4)},
    }
    receipt = _expected_fit_population(context, 67)
    assert receipt["all_fit_rows"] == 4
    assert receipt["missing_fit_rows"] == 2
    assert receipt["fit_cycle_min"] == 55
    assert receipt["fit_cycle_max"] == 60
    assert len(receipt["fit_source_row_positions_sha256"]) == 64
    assert len(receipt["fit_missing_mask_sha256"]) == 64
    assert len(receipt["fit_data_sha256"]) == 64
