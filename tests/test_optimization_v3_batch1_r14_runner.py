import hashlib

import numpy as np
import pandas as pd
import pytest

from src.models import optimization_v3_batch1 as engine
from src.models import optimization_v3_batch1_r14 as adapter
from scripts import optimization_v3_batch1_r14 as runner


def _synthetic_train():
    rows = []
    target = []
    for cycle in range(47, 80):
        for within in range(20):
            ef1 = within < 10
            row = {name: 1.0 for name in adapter.FEATURE_NAMES}
            row["cycle_code"] = cycle
            row["entry_friction"] = "1" if ef1 else "0"
            row["telemetry_flag"] = "inactive" if not ef1 else "active"
            row["org_code"] = "NEW_ORG" if cycle >= 68 else f"ORG_{within % 3}"
            row["module_code"] = f"MOD_{within % 4}"
            row["pattern_code"] = f"PAT_{within % 2}"
            row["signal_stream"] = "" if within % 7 == 0 else f"token_{within % 5} token_shared"
            rows.append({"record_id": f"row-{cycle}-{within}", **row})
            target.append(float(10 + 2 * (cycle - 47) + within / 10))
    return pd.DataFrame(rows), np.asarray(target, dtype=np.float64)


def _controls(frame, views):
    cycles = frame["cycle_code"].to_numpy(dtype=np.int64)
    controls = {}
    for cutoff in (55, 67):
        expected_cycles = runner.r03.EXPECTED_SCORE_CYCLES[cutoff]
        rows = np.flatnonzero(np.isin(cycles, expected_cycles))
        ridge = np.full(len(rows), 50.0, dtype=np.float64)
        raw = np.linspace(5.0, 95.0, len(rows), dtype=np.float64)
        rank = runner._independent_rank50(raw, cycles[rows])
        baseline = np.clip(0.75 * ridge + 0.25 * rank, 0.0, 100.0)
        controls[cutoff] = {
            "score_rows": rows,
            "score_record_ids": frame.iloc[rows]["record_id"].astype(str).to_numpy(dtype=str),
            "score_cycle": cycles[rows],
            "ridge25_prediction": ridge,
            "old_sparse_raw": raw,
            "old_sparse_rank50": rank,
            "baseline_prediction": baseline,
        }
    return controls


def _control_loader(controls):
    def load(cutoff, score_rows):
        saved = controls[int(cutoff)]
        assert np.array_equal(saved["score_rows"], score_rows)
        return {
            "baseline_prediction": saved["baseline_prediction"],
            "ridge25_prediction": saved["ridge25_prediction"],
            "old_sparse_raw": saved["old_sparse_raw"],
            "old_sparse_rank50": saved["old_sparse_rank50"],
            "control_rows": saved["score_rows"],
        }
    return load


def _fake_adapter(calls):
    def fit_predict(fit_x, fit_y, score_x, *, controls, card_id, cutoff):
        calls.append(int(cutoff))
        fit_mask = adapter.ef1_mask(fit_x["entry_friction"])
        score_mask = adapter.ef1_mask(score_x["entry_friction"])
        selected_x = fit_x.loc[fit_mask, list(adapter.FEATURE_NAMES)]
        selected_y = np.asarray(fit_y, dtype=np.float64)[fit_mask]
        raw = np.zeros(len(score_x), dtype=np.float64)
        raw[score_mask] = 20.0
        prediction = adapter.compose_r14_prediction(
            raw, controls["baseline_prediction"], score_mask,
        )
        return {
            "prediction": prediction,
            "raw_candidate": raw,
            "audit_arrays": {"expert_applied_mask": score_mask.astype(np.uint8)},
            "fit_metadata": {
                "ef1_fit_rows": int(fit_mask.sum()),
                "expert_fit_population": "parsed entry_friction == 1",
                "score_labels_passed": False,
                "resource_preflight": {"status": "PASS"},
            },
            "state": {
                "fit_rows": int(fit_mask.sum()),
                "fit_data_sha256": runner._fit_fingerprint_independent(selected_x, selected_y),
                "target_sha256": hashlib.sha256(
                    np.asarray(selected_y, dtype="<f8").tobytes(order="C")
                ).hexdigest(),
            },
            "fit_predictor_count": 1,
        }
    return fit_predict


def test_synthetic_end_to_end_aggregates_verifies_and_resumes_after_one_fit(tmp_path, monkeypatch):
    frame, target = _synthetic_train()
    cycles = frame["cycle_code"].to_numpy(dtype=np.int64)
    queue = runner.load_queue(runner.QUEUE_PATH)
    card = next(item for item in queue["cards"] if item["id"] == "R14")
    views = runner.evaluation_views_for_card(queue, "R14")
    controls = _controls(frame, views)
    calls = []
    adapter_fn = _fake_adapter(calls)
    artifact_dir = tmp_path / "runs" / "R14"
    context = {"frame": frame, "cycles": cycles, "views": views,
               "controls": controls, "identity": {"synthetic": "R14"},
               "artifact_dir": artifact_dir}
    ledger = tmp_path / "batch_ledger.jsonl"
    attempts = tmp_path / "attempts.jsonl"
    freeze_path = tmp_path / "freeze.json"

    # Save the first exact 12-cycle origin, then prove the same bundle is reused
    # when the complete two-origin pilot resumes.
    with pytest.raises(engine.BatchIntegrityError, match="fit count mismatch"):
        engine.run_pilot(
            card=card, artifact_dir=artifact_dir, budget_ledger_path=ledger,
            max_successful_fits=180, frame=frame, target=target,
            feature_names=adapter.FEATURE_NAMES, adapter=adapter_fn,
            control_loader=_control_loader(controls), freeze_identity=context["identity"],
            freeze_path=freeze_path, event_path=attempts,
            evaluation_views={"H12_E": [(55, 56, 67)]}, threads=2,
        )
    report = engine.run_pilot(
        card=card, artifact_dir=artifact_dir, budget_ledger_path=ledger,
        max_successful_fits=180, frame=frame, target=target,
        feature_names=adapter.FEATURE_NAMES, adapter=adapter_fn,
        control_loader=_control_loader(controls), freeze_identity=context["identity"],
        freeze_path=freeze_path, event_path=attempts, evaluation_views=views,
        threads=2, report_path=tmp_path / "engine_report.json",
    )
    assert calls == [55, 67]
    events = [line for line in attempts.read_text(encoding="utf-8").splitlines() if line]
    assert sum('"event": "RESUMED_VERIFIED"' in line for line in events) == 1
    assert report["fit_predictor_count"] == 2
    assert report["views"]["primary"]["rows"] == 160

    # Model a crash after both bundles were saved but before the final report:
    # a later invocation verifies both bundles again without counting new fits.
    report = engine.run_pilot(
        card=card, artifact_dir=artifact_dir, budget_ledger_path=ledger,
        max_successful_fits=180, frame=frame, target=target,
        feature_names=adapter.FEATURE_NAMES, adapter=adapter_fn,
        control_loader=_control_loader(controls), freeze_identity=context["identity"],
        freeze_path=freeze_path, event_path=attempts, evaluation_views=views,
        threads=2, report_path=tmp_path / "engine_report.json",
    )
    monkeypatch.setattr(runner, "RUN_DIR", artifact_dir)
    monkeypatch.setattr(runner, "ATTEMPTS_PATH", attempts)
    monkeypatch.setattr(runner, "FREEZE_PATH", freeze_path)
    monkeypatch.setattr(runner, "LEDGER_PATH", ledger)
    accounting = runner._fit_accounting()
    assert calls == [55, 67]
    assert accounting["verified_successful_predictor_fits"] == 2
    assert accounting["cutoffs"]["55"]["resume_verified_event_count"] == 2
    assert accounting["cutoffs"]["67"]["resume_verified_event_count"] == 1

    # Control denominators/MAE are frozen references in production; this
    # synthetic test binds them to the small constructed fixture.
    baseline_only = {
        cutoff: {"score_rows": data["score_rows"],
                 "prediction": data["baseline_prediction"],
                 "baseline_prediction": data["baseline_prediction"]}
        for cutoff, data in controls.items()
    }
    expected_rows = {}
    expected_primary = expected_h12 = None
    for name, segments in views.items():
        rows, _, base = runner._view_rows(frame, cycles, baseline_only, segments)
        expected_rows[name] = len(rows)
        if name == "primary":
            expected_primary = float(np.abs(target[rows] - base).mean())
        elif name == "H12":
            expected_h12 = float(np.abs(target[rows] - base).mean())
    monkeypatch.setattr(runner, "EXPECTED_VIEW_ROWS", expected_rows)
    monkeypatch.setattr(runner, "EXPECTED_CONTROL_METRICS",
                        {"primary_mae": expected_primary, "H12_mae": expected_h12})
    verification, gate = runner._verify_pilot(context, report, target)
    assert verification["status"] == "INDEPENDENT_QA_PASS"
    assert verification["fit_population_independently_verified"] is True
    assert gate["status"] in {"ADVANCE", "RETIRE"}
    assert verification["views"]["H12_E"]["rows"] == 240
    assert verification["views"]["H12_L"]["rows"] == 240


def test_r14_budget_preflight_rejects_matching_key_with_wrong_card_identity(tmp_path, monkeypatch):
    ledger = tmp_path / "batch_ledger.jsonl"
    ledger.write_text(
        '{"event":"BUDGET_RESERVED","budget_key":"R14:PILOT",'
        '"card_id":"C99","predictor_fits":2}\n',
        encoding="utf-8",
    )
    monkeypatch.setattr(runner, "LEDGER_PATH", ledger)
    snapshot = runner._budget_snapshot()
    assert snapshot["status"] == "BLOCKED_BUDGET"
    assert snapshot["existing_R14_reservation"]["card_id"] == "C99"


def test_r14_budget_preflight_rejects_duplicate_identical_reservations(tmp_path, monkeypatch):
    ledger = tmp_path / "batch_ledger.jsonl"
    line = '{"event":"BUDGET_RESERVED","budget_key":"R14:PILOT",' \
           '"card_id":"R14","predictor_fits":2}'
    ledger.write_text(line + "\n" + line + "\n", encoding="utf-8")
    monkeypatch.setattr(runner, "LEDGER_PATH", ledger)
    snapshot = runner._budget_snapshot()
    assert snapshot["status"] == "BLOCKED_LEDGER"
    assert "duplicate budget reservation" in snapshot["error"]
