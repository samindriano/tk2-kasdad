import numpy as np
import pandas as pd

from scripts import optimization_v3_supervised_pilot as pilot
from scripts.optimization_v3_supervised_pilot_postfit_v2 import adapt_classification_fit


def test_postfit_adapter_reconstructs_frozen_classification_return_schema_for_views():
    cycles = np.repeat(np.arange(56, 80), 8)
    rows_all = np.arange(len(cycles), dtype=np.int64)
    missing_all = (rows_all % 3 == 0)
    frame = pd.DataFrame({
        "cycle_code": cycles,
        "signal_stream": np.where(missing_all, "", "alpha shared token"),
    })
    target = (rows_all % 2).astype(np.int8)

    def record(end, low, high):
        rows = np.flatnonzero((cycles >= low) & (cycles <= high))
        missing = missing_all[rows]
        cb = np.full(len(rows), 0.4)
        raw = np.linspace(0.1, 0.9, len(rows))
        c2b = np.full(len(rows), 0.35)
        latest = np.full(len(rows), 0.45)
        direct = np.where(missing, cb, raw)
        blend = np.where(missing, cb, 0.75 * latest + 0.25 * raw)
        arrays = {
            "rows": rows,
            "cycles": cycles[rows],
            "raw_probability": np.where(missing, 0.0, raw),
            "initial_control": np.where(missing, cb, c2b),
            "latest_control": np.where(missing, cb, latest),
            "direct": direct,
            "blend25": blend,
            "c2b_present": c2b,
            "cb_missing": cb,
            "missing": missing.astype(np.uint8),
        }
        return adapt_classification_fit(end, arrays, {"fit_rows": 1}, 0.01, "fixture-cache-hash")

    records = {55: record(55, 56, 67), 67: record(67, 68, 79)}
    views = pilot.build_class_views(frame, target, records)

    assert set(views) == {"W1", "W4", "H12_E", "H12_L", "primary_pilot", "H12"}
    assert records[55]["policies"]["direct"].shape == records[55]["rows"].shape
    assert np.all(records[55]["policies"]["direct"][records[55]["missing"]] == 0.4)
    assert "blend25_vs_latest_sparse25" in views["primary_pilot"]["comparisons"]
    assert views["primary_pilot"]["n"] == 64
