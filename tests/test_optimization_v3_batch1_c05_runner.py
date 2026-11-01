"""Synthetic end-to-end checks for C05 pilot aggregation and independent gate."""
from __future__ import annotations

import numpy as np
import pandas as pd
import json

import scripts.optimization_v3_batch1_c05 as runner
import scripts.optimization_v3_batch1_verify as verifier
import src.models.optimization_v3_batch1 as batch
from src.models.optimization_v3_batch1_c05 import FEATURE_NAMES, fit_predict


def _synthetic_train() -> tuple[pd.DataFrame, np.ndarray]:
    cycles = np.repeat(np.arange(47, 80, dtype=np.int64), 12)
    replica = np.tile(np.arange(12, dtype=np.int64), 33)
    row_positions = np.arange(len(cycles), dtype=np.int64)
    streams = []
    for cycle, rep in zip(cycles, replica, strict=True):
        if rep == 0 and cycle % 3 == 0:
            streams.append(None)
        elif rep == 1 and cycle % 4 == 0:
            streams.append("   ")
        else:
            streams.append(f"amber birch {cycle % 7} token{rep % 5} shared")
    frame = pd.DataFrame({
        "record_id": [f"synthetic-{index:05d}" for index in row_positions],
        "cycle_code": cycles,
        "org_code": [f"org-{rep % 4}" for rep in replica],
        "module_code": [f"module-{rep % 5}" for rep in replica],
        "pattern_code": [f"pattern-{rep % 3}" for rep in replica],
        "signal_stream": streams,
        "scope_trace": [f"trace-{(cycle + rep) % 6}" for cycle, rep in zip(cycles, replica, strict=True)],
        "org_spread": [str((cycle + rep) % 5) for cycle, rep in zip(cycles, replica, strict=True)],
        "module_spread": [str((cycle + 2 * rep) % 7) for cycle, rep in zip(cycles, replica, strict=True)],
        "text_load": (cycles + replica) % 9,
        "scope_pressure": (3 * cycles + replica) % 17,
        "pulse_a": (cycles + 2 * replica) % 13,
        "pulse_b": (2 * cycles + replica) % 11,
        "telemetry_flag": (cycles + replica) % 2,
        "audit_lane": [f"lane-{rep % 4}" for rep in replica],
        "checksum_hint": [f"hint-{(cycle + rep) % 8}" for cycle, rep in zip(cycles, replica, strict=True)],
        "impact_load": (7 * cycles + 3 * replica) % 101,
        "entry_friction": (cycles + replica) % 2,
    })
    target = ((cycles * 5 + replica * 7) % 11 < 5).astype(np.int8)
    return frame, target


def _synthetic_controls(frame: pd.DataFrame, views):
    cycles = frame["cycle_code"].to_numpy(dtype=np.int64)
    controls = {}
    baseline_by_row = np.full(len(frame), np.nan, dtype=np.float64)
    for cutoff in (55, 67):
        rows = np.flatnonzero((cycles > cutoff) & (cycles <= cutoff + 12))
        missing = batch.stream_missing_mask(frame.iloc[rows]["signal_stream"].tolist())
        positions = np.arange(len(rows), dtype=np.float64)
        c2b = 0.22 + (positions % 7) * 0.07
        cb = 0.28 + (positions % 5) * 0.09
        old_sparse = 0.18 + (positions % 9) * 0.06
        baseline = np.where(missing, cb, 0.75 * c2b + 0.25 * old_sparse)
        baseline_by_row[rows] = baseline
        controls[cutoff] = {
            "rows": rows,
            "baseline_prediction": baseline,
            "baseline_sparse_raw": old_sparse,
            "c2b_probability": c2b,
            "cb_probability": cb,
            "stream_missing": missing,
        }
    expected_rows = {}
    expected_metrics = {}
    for name, segments in views.items():
        pieces = [
            np.flatnonzero((cycles >= low) & (cycles <= high))
            for _cutoff, low, high in segments
        ]
        rows = np.concatenate(pieces)
        expected_rows[name] = int(len(rows))
        predicted = baseline_by_row[rows] >= 0.50
        expected_metrics[name] = verifier._macro_f1_independent(frame.attrs["target"][rows], predicted)
    return controls, expected_rows, expected_metrics


def test_c05_actual_adapter_aggregates_and_passes_independent_verification(tmp_path, monkeypatch):
    queue = batch.load_queue(runner.QUEUE_PATH)
    card = next(item for item in queue["cards"] if item["id"] == "C05")
    views = batch.evaluation_views_for_card(queue, "C05")
    frame, target = _synthetic_train()
    frame.attrs["target"] = target
    controls, expected_rows, expected_metrics = _synthetic_controls(frame, views)
    monkeypatch.setattr(runner, "EXPECTED_VIEW_ROWS", expected_rows)
    monkeypatch.setattr(runner, "EXPECTED_CONTROL_METRICS", expected_metrics)

    train_marker = tmp_path / "synthetic_train_marker.txt"
    train_marker.write_text("synthetic test fixture only\n", encoding="utf-8")
    freeze_identity = batch.make_freeze_identity(
        card=card,
        feature_names=FEATURE_NAMES,
        input_files={"synthetic_train_marker": train_marker},
        source_files=[runner.__file__, fit_predict.__code__.co_filename],
        evaluation_views=views,
        packages=[],
    )
    signature = batch.json_sha256(freeze_identity)

    def loader(context):
        return runner._control_loader(context)

    context = {
        "identity": freeze_identity,
        "frame": frame,
        "features": frame.loc[:, list(FEATURE_NAMES)].copy(),
        "target": target,
        "cycles": frame["cycle_code"].to_numpy(dtype=np.int64),
        "views": views,
        "controls": controls,
        "public": {
            "input_sha256": {"synthetic_train_marker": "test-only"},
            "source_sha256": {"runner": "test-only", "adapter": "test-only"},
            "baseline_control_metrics": {
                name: {"rows": expected_rows[name], "macro_f1": expected_metrics[name]}
                for name in views
            },
        },
    }

    artifact_dir = tmp_path / "run_artifacts"
    ledger_path = tmp_path / "batch_ledger.jsonl"
    attempts_path = tmp_path / "attempts.jsonl"
    engine_report_path = tmp_path / "pilot_engine_report.json"
    runner.append_event(attempts_path, {
        "card_id": "C05",
        "event": "PREFLIGHT_PASS",
        "resource_estimates": {"synthetic_only": True},
    })
    report = batch.run_pilot(
        card=card,
        frame=frame,
        target=target,
        feature_names=FEATURE_NAMES,
        adapter=fit_predict,
        control_loader=loader(context),
        freeze_identity=freeze_identity,
        freeze_path=tmp_path / "synthetic_freeze.json",
        artifact_dir=artifact_dir,
        event_path=attempts_path,
        budget_ledger_path=ledger_path,
        max_successful_fits=180,
        evaluation_views=views,
        threads=2,
        report_path=engine_report_path,
    )

    assert report["card_id"] == "C05"
    assert report["test_data_used"] is False
    assert report["fit_count_matches_card"] is True
    assert report["cutoffs"] == [55, 67]
    assert signature == report["freeze_signature"]
    assert set(report["views"]) == {"W1", "W4", "primary", "H12"}
    for name, n_rows in expected_rows.items():
        assert report["views"][name]["rows"] == n_rows

    monkeypatch.setattr(runner, "RUN_DIR", artifact_dir)
    monkeypatch.setattr(runner, "LEDGER_PATH", ledger_path)
    monkeypatch.setattr(runner, "ATTEMPTS_PATH", attempts_path)
    cycles = frame["cycle_code"].to_numpy(dtype=np.int64)
    monkeypatch.setattr(runner, "EXPECTED_FIT_ROWS", {
        55: int(np.sum(cycles <= 55)),
        67: int(np.sum(cycles <= 67)),
    })
    verification, gate = runner._gate_and_verify(context, report)
    assert verification["status"] == "INDEPENDENT_QA_PASS"
    assert verification["test_data_used"] is False
    assert verification["kaggle_access"] is False
    assert verification["cutoffs"] == [55, 67]
    assert gate["status"] in {"ADVANCE", "RETIRE"}
    accounting = runner._fit_accounting(context)
    assert accounting["budget_reserved_predictor_fits"] == 2
    assert accounting["successful_predictor_fits"] == 2
    assert all(value["failed_attempt_count"] == 0 for value in accounting["cutoff_fits"].values())
    local_events = [json.loads(line) for line in attempts_path.read_text(encoding="utf-8").splitlines()]
    ledger_events = [json.loads(line) for line in ledger_path.read_text(encoding="utf-8").splitlines()]
    assert {event["event"] for event in local_events} >= {"PREFLIGHT_PASS", "FIT_STARTED", "FIT_SAVED"}
    assert any(event["event"] == "BUDGET_RESERVED" for event in ledger_events)
    assert not any(event["event"] in {"FIT_STARTED", "FIT_SAVED"} for event in ledger_events)

    report_dir = tmp_path / "reports"
    monkeypatch.setattr(runner, "REPORT_DIR", report_dir)
    monkeypatch.setattr(runner, "RESULTS_PATH", report_dir / "results.json")
    monkeypatch.setattr(runner, "VERIFICATION_PATH", report_dir / "verification.json")
    monkeypatch.setattr(runner, "GATE_PATH", report_dir / "gate.json")
    monkeypatch.setattr(runner, "SUMMARY_PATH", report_dir / "summary.json")
    runner._publish(context, report, verification, gate, accounting)
    saved_results = runner._read_json(report_dir / "results.json")
    saved_gate = runner._read_json(report_dir / "gate.json")
    saved_verification = runner._read_json(report_dir / "verification.json")
    assert saved_results["status"] == "PILOT_COMPLETE_" + gate["status"]
    assert saved_results["test_data_used"] is False
    assert saved_results["kaggle_access"] is False
    assert saved_gate["status"] == gate["status"]
    assert saved_verification["status"] == "INDEPENDENT_QA_PASS"

    for cutoff in (55, 67):
        path = artifact_dir / f"C05_fit{cutoff}.npz"
        with np.load(path, allow_pickle=False) as arrays:
            rows = arrays["score_rows"]
            stored_ids = arrays["score_record_ids"].astype(str)
            np.testing.assert_array_equal(
                stored_ids,
                frame.iloc[rows]["record_id"].astype(str).to_numpy(),
            )
            np.testing.assert_array_equal(
                arrays["control__control_rows"], rows.astype(np.int64)
            )
