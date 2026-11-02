from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from scripts import optimization_v3_rnbr_continuation as runner
from scripts import optimization_v3_rnbr_continuation_synthetic as synthetic
from scripts import optimization_v3_rnbr_continuation_verify as verifier


def test_synthetic_end_to_end_save_resume_and_corruption_rejection():
    result = synthetic.run()
    assert result["status"] == "PASS"
    assert "origin-cache query baseline distinct from donor OOF" in result["checks"]


def test_origin_and_query_window_plan_are_exact():
    assert runner.ORIGINS == (59, 63, 71)
    assert runner.QUERY_CYCLES == {59: (60, 63), 63: (64, 67), 71: (72, 75)}
    assert runner.EXPECTED_ROWS == {59: 6493, 63: 7078, 71: 7462}
    protocol_path = Path("experiments/optimization_v3/rnbr_continuation_v1/protocol.json")
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    h12 = protocol["views"]["inherited_H12_L"]
    assert h12 == {"origin": 67, "cycles": [68, 79], "rows": 20903,
                   "source": "existing E67 pilot only; never combine E71/E75"}


def test_independent_weighted_median_handles_exact_half_mass():
    values = np.asarray([10.0, 0.0])
    similarities = np.asarray([1.0, 1.0])
    source_rows = np.asarray([5, 2])
    assert verifier._median(values, similarities, source_rows) == 5.0
    assert runner.weighted_median_residual(values, similarities, source_rows) == 5.0


def test_future_or_wrong_neighbor_id_is_rejected():
    train, _, geom, _ = synthetic.synthetic_case()
    bad = {k: np.asarray(v).copy() for k, v in geom.items()}
    bad["neighbor_ids"][0, 0] = "str_:wrong"
    with pytest.raises(runner.ContinuationError, match="typed ID"):
        runner.validate_geometry(59, train, bad, np.arange(10, dtype=np.int64))


def test_snapshot_preserves_starts_saved_failed_and_pending_accounting():
    events = [{"event": "BATCH_ALLOWANCE_AUTHORIZED", "cap": 70,
               "category_caps": {"pilot": 38, "continuation": 31, "control_replay": 0, "technical_retry": 1}},
              {"event": "CARD_FITS_RESERVED", "reservation_key": "r", "category": "pilot", "fits": 38},
              {"event": "CARD_FITS_RESERVED", "reservation_key": "c", "category": "continuation", "fits": 31},
              {"event": "CARD_FITS_RESERVED", "reservation_key": "t", "category": "technical_retry", "fits": 1}]
    for i in range(70):
        events.append({"event": "FIT_STARTED", "fit_id": f"f{i}"})
        if i < 67:
            events.append({"event": "FIT_SAVED", "fit_id": f"f{i}"})
        else:
            events.append({"event": "FIT_FAILED", "fit_id": f"f{i}"})
    events.append({"event": "FIT_OUTPUT_RECOVERED", "fit_id": "f67", "fit_slot_consumed_again": False})
    snap = runner._snapshot(events)
    assert snap["global_cap"] == 70
    assert snap["started"] == 70 and snap["saved"] == 67 and snap["failed"] == 3
    assert snap["pending"] == 0 and snap["active_reserved"] == 70
