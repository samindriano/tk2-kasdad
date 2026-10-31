import numpy as np
import pandas as pd

from scripts import optimization_v3_supervised_pilot_postfit_v3 as postfit


def test_subgroup_metrics_use_aligned_rows_and_zero_support_is_null():
    frame = pd.DataFrame({
        "cycle_code": [56, 56, 56, 56],
        "entry_friction": [0, 0, 1, 1],
        "signal_stream": ["alpha", "", "beta", "gamma"],
    })
    target = np.array([0.0, 2.0, 80.0, 90.0])
    rows = np.arange(4)
    predictions = {
        "champion": np.array([1.0, 2.0, 82.0, 89.0]),
        "raw": np.array([5.0, 2.0, 80.0, 100.0]),
        "rank50": np.array([0.0, 3.0, 81.0, 91.0]),
        "blend25": np.array([2.0, 2.0, 80.0, 90.0]),
    }

    production = postfit.compare_reg_view_fixed(frame, target, rows, predictions)
    independent = postfit.regression_view_independent(frame, target, rows, predictions)

    # EF0 uses only rows 0-1: champion MAE=0.5, raw MAE=2.5, gain=-2.0.
    assert production["subgroups"]["EF0"]["mae_gain_vs_champion"]["raw"] == -2.0
    assert independent["subgroups"]["EF0"]["mae_gain_vs_champion"]["raw"] == -2.0
    # There are no y>=95 rows in this fixture; retain denominator=0 as null metrics.
    assert production["subgroups"]["tail_ge95"]["n"] == 0
    assert production["subgroups"]["tail_ge95"]["metrics"]["raw"]["mae"] is None
    assert production["subgroups"]["tail_ge95"]["mae_gain_vs_champion"]["raw"] is None
    assert independent["subgroups"]["tail_ge95"]["metrics"]["raw"]["mae"] is None
