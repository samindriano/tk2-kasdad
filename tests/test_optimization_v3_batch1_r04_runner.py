from __future__ import annotations

import json

import numpy as np
import pandas as pd

import scripts.optimization_v3_batch1_r04 as r04_runner
import src.models.optimization_v3_batch1 as batch
from src.models.optimization_v3_batch1_r04 import FEATURE_NAMES, fit_predict
from src.models.optimization_v3_batch1_r02 import compose_r02_prediction


def _synthetic_frame() -> tuple[pd.DataFrame, np.ndarray]:
    """Build a small temporal Train-shaped frame with duplicates per cycle."""
    cycles = np.repeat(np.arange(47, 80, dtype=np.int64), 3)
    row_ids = np.arange(len(cycles), dtype=np.int64)
    frame = pd.DataFrame({
        "record_id": [f"synthetic-{i:04d}" for i in row_ids],
        "cycle_code": cycles,
        # All model features are constant within a cycle, which guarantees ties
        # among each three-row rank batch while retaining temporal variation.
        "impact_load": (cycles * 7) % 101,
        "entry_friction": np.zeros(len(cycles), dtype=np.int8),
        "scope_pressure": cycles % 9,
        "text_load": cycles % 6,
        "scope_trace": [f"trace-{cycle % 4}" for cycle in cycles],
        "org_spread": cycles % 5,
        "module_spread": cycles % 7,
        "telemetry_flag": cycles % 2,
        "org_code": [f"org-{cycle % 4}" for cycle in cycles],
        "module_code": [f"module-{cycle % 5}" for cycle in cycles],
        "pattern_code": [f"pattern-{cycle % 3}" for cycle in cycles],
        "signal_stream": [f"amber birch cycle{cycle}" for cycle in cycles],
    })
    target = ((cycles * 13) % 97).astype(np.float64)
    return frame, target


def test_r04_actual_adapter_runs_through_synthetic_verification_and_gate(tmp_path, monkeypatch):
    queue = batch.load_queue(r04_runner.QUEUE_PATH)
    card = next(item for item in queue["cards"] if item["id"] == "R04")
    views = batch.evaluation_views_for_card(queue, "R04")
    frame, target = _synthetic_frame()
    temp_train = tmp_path / "synthetic_train.csv"
    temp_train.write_text("synthetic fixture only\n", encoding="utf-8")
    freeze_identity = batch.make_freeze_identity(
        card=card,
        feature_names=FEATURE_NAMES,
        input_files={"synthetic_train": temp_train},
        source_files=[r04_runner.__file__, fit_predict.__code__.co_filename],
        evaluation_views=views,
        packages=[],
    )
    signature = batch.json_sha256(freeze_identity)

    # Keep the generic runner's model/verification path real, but disable its
    # durable protocol freeze and budget reservation in this synthetic test.
    monkeypatch.setattr(batch, "ensure_frozen", lambda _path, _identity: signature)
    reservation_calls = []

    def no_live_budget_reservation(path, **kwargs):
        reservation_calls.append((path, kwargs))
        assert str(path).startswith(str(tmp_path))
        return {"status": "TEST_ONLY_NO_RESERVATION"}

    monkeypatch.setattr(batch, "reserve_card_budget", no_live_budget_reservation)

    artifact_dir = tmp_path / "run_artifacts"
    event_path = tmp_path / "events.jsonl"
    engine_report_path = tmp_path / "pilot_engine_report.json"

    def controls(cutoff: int, score_rows: np.ndarray):
        cycles = frame.iloc[score_rows]["cycle_code"].to_numpy(dtype=np.int64)
        ridge = np.full(len(score_rows), 50.0, dtype=np.float64)
        old_sparse_raw = np.full(len(score_rows), 25.0, dtype=np.float64)
        return {
            "baseline_prediction": compose_r02_prediction(ridge, old_sparse_raw, cycles),
            "ridge25_prediction": ridge,
            "baseline_sparse_raw": old_sparse_raw,
            "cycle_code": cycles,
            "control_rows": score_rows.astype(np.int64, copy=False),
        }

    received_fit_ids = []

    def adapter(fit_x, fit_y, score_x, *, controls, card_id, cutoff):
        assert "record_id" not in fit_x.columns
        assert "record_id" not in score_x.columns
        received_fit_ids.append((cutoff, fit_y.shape[0]))
        return fit_predict(
            fit_x, fit_y, score_x,
            controls=controls, card_id=card_id, cutoff=cutoff,
        )

    report = batch.run_pilot(
        card=card,
        frame=frame,
        target=target,
        feature_names=FEATURE_NAMES,
        adapter=adapter,
        control_loader=controls,
        freeze_identity=freeze_identity,
        freeze_path=tmp_path / "synthetic_freeze.json",
        artifact_dir=artifact_dir,
        event_path=event_path,
        budget_ledger_path=tmp_path / "synthetic_budget_ledger.jsonl",
        max_successful_fits=180,
        evaluation_views=views,
        threads=2,
        report_path=engine_report_path,
    )

    assert report["card_id"] == "R04"
    assert report["test_data_used"] is False
    assert report["fit_count_matches_card"] is True
    assert received_fit_ids == [(55, 27), (67, 63)]
    assert len(reservation_calls) == 1

    # Each validation cycle has three identical-feature rows, so all receive the
    # average tie rank. Confirm the actual adapter's saved rank arrays preserve it.
    for cutoff in (55, 67):
        bundle_path = artifact_dir / f"R04_fit{cutoff}.npz"
        with np.load(bundle_path, allow_pickle=False) as bundle:
            rows = bundle["score_rows"]
            raw_candidate = bundle["raw_candidate"]
            candidate_rank = bundle["audit__candidate_rank50"]
            saved_ids = bundle["score_record_ids"].astype(str)
        expected_ids = frame.iloc[rows]["record_id"].astype(str).to_numpy()
        np.testing.assert_array_equal(saved_ids, expected_ids)
        for cycle in np.unique(frame.iloc[rows]["cycle_code"]):
            in_cycle = frame.iloc[rows]["cycle_code"].to_numpy() == cycle
            assert int(in_cycle.sum()) == 3
            np.testing.assert_array_equal(raw_candidate[in_cycle], np.repeat(raw_candidate[in_cycle][0], 3))
            expected_ranked = 0.5 * raw_candidate[in_cycle][0] + 0.5 * (100.0 * 2.0 / 3.0)
            np.testing.assert_allclose(candidate_rank[in_cycle], expected_ranked, rtol=0, atol=1e-12)

    monkeypatch.setattr(r04_runner, "RUN_DIR", artifact_dir)
    view_rows = {
        name: sum(
            int(((frame["cycle_code"] >= low) & (frame["cycle_code"] <= high)).sum())
            for _cutoff, low, high in segments
        )
        for name, segments in views.items()
    }
    monkeypatch.setattr(r04_runner, "EXPECTED_VIEW_ROWS", view_rows)
    gate_config = queue["evaluation"]["regression_gate"]
    synthetic_queue = {"evaluation": {"regression_gate": gate_config}}
    context = {
        "identity": freeze_identity,
        "frame": frame,
        "target": target,
        "cycles": frame["cycle_code"].to_numpy(dtype=np.int64),
        "views": views,
    }
    verification, subgroup, gate = r04_runner._verify_and_gate(context, report, synthetic_queue)

    assert verification["status"] == "PASS"
    assert verification["test_data_used"] is False
    assert verification["kaggle_access"] is False
    assert gate["status"] in {"ADVANCE", "RETIRE"}
    assert subgroup["views"]["primary"]["stream_missing"] == {
        "n": 0,
        "supported": False,
        "evidence_status": "DESCRIPTIVE_UNSUPPORTED",
        "mae_harm_candidate_minus_control": None,
        "numeric_guard_applies": True,
    }
    assert subgroup["views"]["H12"]["entry_friction_1"]["n"] == 0
    assert subgroup["views"]["H12"]["entry_friction_1"]["supported"] is False
    assert subgroup["views"]["H12"]["entry_friction_1"]["mae_harm_candidate_minus_control"] is None

    # All files created by the dry run stay under pytest's temporary directory.
    assert all(tmp_path in path.parents for path in (artifact_dir, event_path, engine_report_path))
    with event_path.open("r", encoding="utf-8") as handle:
        events = [json.loads(line) for line in handle if line.strip()]
    assert all(event["card_id"] == "R04" for event in events)
