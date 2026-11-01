from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import scripts.optimization_v3_final_continuation_cls as runner
from src.models.optimization_v3_final_continuation_c16 import (
    FinalContinuationC16Classifier,
    NEW_CUTOFFS,
)


WINDOW_CYCLE_COUNTS = {
    56: 1029, 57: 1029, 58: 1028, 59: 1028,
    60: 1624, 61: 1623, 62: 1623, 63: 1623,
    64: 1770, 65: 1770, 66: 1769, 67: 1769,
    68: 1238, 69: 1238, 70: 1237, 71: 1237,
    72: 1866, 73: 1866, 74: 1865, 75: 1865,
    76: 2123, 77: 2123, 78: 2123, 79: 2122,
}


def _synthetic_context() -> dict:
    counts = {cycle: 50 for cycle in range(47, 56)}
    counts.update(WINDOW_CYCLE_COUNTS)
    cycles = np.concatenate([
        np.full(counts[cycle], cycle, dtype=np.int64)
        for cycle in range(47, 80)
    ])
    n = len(cycles)
    row = np.arange(n, dtype=np.int64)
    frame = pd.DataFrame({
        "record_id": [f"synthetic-{i:06d}" for i in row],
        "cycle_code": cycles,
        "artifact_signal": (row % 4 == 0).astype(np.int8),
        "signal_stream": [None if i % 11 == 0 else f"tok{i % 53} alpha beta" for i in row],
        "entry_friction": [np.nan if i % 13 == 0 else i % 2 for i in row],
        "text_load": [np.nan if i % 19 == 0 else float(i % 113) for i in row],
        "impact_load": row.astype(float) % 10,
        "scope_pressure": row.astype(float) % 7,
        "scope_trace": row.astype(float) % 5,
        "org_spread": row.astype(float) % 9,
        "module_spread": row.astype(float) % 11,
        "telemetry_flag": row.astype(float) % 2,
        "org_code": [f"org-{i % 3}" for i in row],
        "module_code": [f"module-{i % 5}" for i in row],
        "pattern_code": [f"pattern-{i % 4}" for i in row],
    })
    y = frame["artifact_signal"].to_numpy(dtype=np.int8)
    records: dict[int, dict] = {}
    controls: dict[int, dict] = {}
    for cutoff in runner.ALL_ORIGINS:
        low, high = ((cutoff + 1, cutoff + 12) if cutoff in runner.PILOT_CUTOFFS
                     else (cutoff + 1, cutoff + 4))
        rows = np.flatnonzero((cycles >= low) & (cycles <= high))
        missing = runner._stream_missing(frame, rows)
        c2b = np.where(rows % 2 == 0, 0.43, 0.61).astype(np.float64)
        cb = np.where(rows % 3 == 0, 0.68, 0.32).astype(np.float64)
        old_sparse = np.where(rows % 5 == 0, 0.7, 0.35).astype(np.float64)
        base = np.where(missing, cb, 0.75 * c2b + 0.25 * old_sparse)
        raw = np.where(rows % 7 == 0, 0.74, 0.29).astype(np.float64)
        candidate = np.where(missing, cb, 0.75 * c2b + 0.25 * raw)
        ctrl = {
            "baseline_prediction": base,
            "baseline_sparse_raw": old_sparse,
            "c2b_probability": c2b,
            "cb_probability": cb,
            "stream_missing": missing,
            "control_rows": rows,
        }
        controls[cutoff] = ctrl
        record = {
            "score_rows": rows,
            "score_record_ids": frame.iloc[rows]["record_id"].astype(str).to_numpy(),
            "prediction": candidate,
            "baseline_prediction": base,
            "raw_candidate": raw,
            "control_arrays": ctrl,
            "metadata": {"fit_predictor_count": 1},
        }
        records[cutoff] = record
    return {"frame": frame, "target": y, "cycles": cycles,
            "records": records, "controls": controls}


def _fake_fit_predict(fit_features, fit_target, score_features, *, controls, card_id, cutoff):
    assert card_id == "C16"
    assert cutoff in NEW_CUTOFFS
    assert "artifact_signal" not in fit_features.columns
    assert "artifact_signal" not in score_features.columns
    assert len(fit_features) == len(fit_target)
    assert np.array_equal(score_features.index.to_numpy(), controls["control_rows"])
    raw = np.where(controls["control_rows"] % 7 == 0, 0.74, 0.29).astype(np.float64)
    prediction = np.where(
        controls["stream_missing"], controls["cb_probability"],
        0.75 * controls["c2b_probability"] + 0.25 * raw,
    )
    return {
        "prediction": prediction,
        "raw_candidate": raw,
        "fit_predictor_count": 1,
        "state": {"cutoff": int(cutoff), "fit_rows": len(fit_target)},
        "fit_metadata": {"score_labels_passed_to_fit": False},
        "audit_arrays": {"stream_missing": controls["stream_missing"]},
    }


def test_c16_adapter_is_cutoff_scoped_without_fitting():
    for cutoff in NEW_CUTOFFS:
        model = FinalContinuationC16Classifier(cutoff)
        assert model.cutoff == cutoff
        assert model._fitted is False
    for cutoff in (55, 67, 59.5, 80):
        with pytest.raises(ValueError):
            FinalContinuationC16Classifier(cutoff)


def test_synthetic_six_window_h12_and_guard_report_recomputes_independently():
    context = _synthetic_context()
    report = runner.build_report(context["frame"], context["target"], context["records"], "synthetic-freeze")
    verified = runner.verify_report_independently(
        context["frame"], context["target"], context["records"], report, "synthetic-freeze",
    )
    assert verified["status"] == "PASS"
    assert report["views"]["primary"]["rows"] == runner.EXPECTED_PRIMARY_ROWS
    assert report["views"]["H12"]["rows"] == runner.EXPECTED_H12_ROWS
    assert report["qualification"]["late_W5_W6"]["rows"] == runner.EXPECTED_LATE_ROWS
    assert set(report["group_guards"]) == set(runner.GATE_VIEW_NAMES)


def test_verifier_rejects_record_id_mismatch_and_route_mutation():
    context = _synthetic_context()
    report = runner.build_report(context["frame"], context["target"], context["records"], "sig")
    broken_ids = {key: dict(value) for key, value in context["records"].items()}
    broken_ids[59]["score_record_ids"] = broken_ids[59]["score_record_ids"].copy()
    broken_ids[59]["score_record_ids"][0] = "wrong-id"
    with pytest.raises(runner.FinalContinuationIntegrityError, match="coverage/record_id"):
        runner.verify_report_independently(context["frame"], context["target"], broken_ids, report, "sig")

    broken_route = {key: dict(value) for key, value in context["records"].items()}
    broken_route[63]["prediction"] = broken_route[63]["prediction"].copy()
    broken_route[63]["prediction"][0] = 0.123
    with pytest.raises(runner.FinalContinuationIntegrityError, match="route reconstruction"):
        runner.verify_report_independently(context["frame"], context["target"], broken_route, report, "sig")


def test_text_load_quantiles_use_origin_prefix_only_and_small_groups_are_descriptive():
    context = _synthetic_context()
    frame = context["frame"]
    rows = np.flatnonzero((context["cycles"] >= 60) & (context["cycles"] <= 63))
    _, initial_edges = runner._text_load_labels_for_origin(frame, 59, rows)
    changed = frame.copy()
    changed.loc[changed.cycle_code > 59, "text_load"] = 1e9
    _, changed_edges = runner._text_load_labels_for_origin(changed, 59, rows)
    assert initial_edges == changed_edges

    small_rows = np.arange(40, dtype=np.int64)
    origins = np.full(len(small_rows), 55, dtype=np.int64)
    small = runner._group_metrics(
        frame, context["target"][small_rows], small_rows, origins,
        np.full(len(small_rows), 0.6), np.full(len(small_rows), 0.5),
    )
    assert all(not group["supported"] and group["guard_applies"] is False
               for table in small["groups"].values() for group in table.values())


def test_single_cutoff_bundle_resume_interruption_and_pilot_files_are_untouched(tmp_path, monkeypatch):
    context = _synthetic_context()
    monkeypatch.setattr(runner, "RUN_DIR", tmp_path / "isolated-run")
    monkeypatch.setattr(runner, "EVENT_PATH", runner.RUN_DIR / "events.jsonl")
    runner.RUN_DIR.mkdir(parents=True)
    pilot_npz = runner.RUN_DIR / "C16_fit55.npz"
    pilot_json = runner.RUN_DIR / "C16_fit55.json"
    pilot_npz.write_bytes(b"historical pilot immutable sentinel")
    pilot_json.write_text('{"frozen":"pilot"}', encoding="utf-8")
    before = (hashlib.sha256(pilot_npz.read_bytes()).hexdigest(),
              hashlib.sha256(pilot_json.read_bytes()).hexdigest())
    calls = []
    monkeypatch.setattr(runner, "fit_predict", lambda *a, **kw: calls.append(kw["cutoff"]) or _fake_fit_predict(*a, **kw))

    first, did_fit = runner.execute_cutoff(context, "synthetic-freeze", 59, require_frozen=False)
    assert did_fit is True and calls == [59]
    first_sha = hashlib.sha256((runner.RUN_DIR / "C16_fit59.npz").read_bytes()).hexdigest()
    resumed, did_fit = runner.execute_cutoff(context, "synthetic-freeze", 59, require_frozen=False)
    assert did_fit is False and calls == [59]
    assert np.array_equal(first["prediction"], resumed["prediction"])
    assert hashlib.sha256((runner.RUN_DIR / "C16_fit59.npz").read_bytes()).hexdigest() == first_sha
    after = (hashlib.sha256(pilot_npz.read_bytes()).hexdigest(),
             hashlib.sha256(pilot_json.read_bytes()).hexdigest())
    assert after == before


def test_resume_after_interruption_reuses_saved_origin_without_retraining(tmp_path, monkeypatch):
    context = _synthetic_context()
    monkeypatch.setattr(runner, "RUN_DIR", tmp_path / "resume-run")
    monkeypatch.setattr(runner, "EVENT_PATH", runner.RUN_DIR / "events.jsonl")
    calls = []

    def interrupt_on_63(*args, **kwargs):
        cutoff = kwargs["cutoff"]
        calls.append(cutoff)
        if cutoff == 63:
            raise RuntimeError("synthetic interruption")
        return _fake_fit_predict(*args, **kwargs)

    monkeypatch.setattr(runner, "fit_predict", interrupt_on_63)
    with pytest.raises(RuntimeError, match="synthetic interruption"):
        runner.execute_new_fits(context, "synthetic-freeze", lock=False, require_frozen=False)
    assert calls == [59, 63]
    assert (runner.RUN_DIR / "C16_fit59.npz").is_file()
    assert not (runner.RUN_DIR / "C16_fit63.npz").exists()

    monkeypatch.setattr(runner, "fit_predict", lambda *args, **kwargs: calls.append(kwargs["cutoff"]) or _fake_fit_predict(*args, **kwargs))
    records, attempts = runner.execute_new_fits(
        context, "synthetic-freeze", lock=False, require_frozen=False,
    )
    assert calls == [59, 63, 63, 71, 75]
    assert attempts == {"fit_calls_this_invocation": 3, "resumed_bundle_count": 1, "new_bundle_count": 3}
    report = runner.build_report(context["frame"], context["target"], records, "synthetic-freeze")
    assert runner.verify_report_independently(
        context["frame"], context["target"], records, report, "synthetic-freeze",
    )["status"] == "PASS"

