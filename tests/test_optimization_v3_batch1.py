from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.models.optimization_v3_batch1 import (
    BatchIntegrityError, category_novelty_flags, ensure_frozen,
    evaluation_views_for_card, fit_only_quantile_bins, load_queue,
    make_freeze_identity, metric, read_bundle, run_pilot, temporal_rows,
    write_bundle_exclusive, past_cycle_smoothed_target_rates,
    exclusive_training_slot, reserve_card_budget, stable_unique_tokens,
    stream_missing_mask, supported_group_metric,
)
from scripts.optimization_v3_batch1_verify import (
    _classification_group_metrics, reconstruct_c03, verify_report,
)
from src.models.optimization_v3_batch1_c03 import FEATURE_NAMES, SplitUnigramBigramClassifier, fit_predict as fit_predict_c03


def test_queue_is_complete_and_ordered():
    queue = load_queue("experiments/optimization_v3/research_backlog/queue.json")
    assert len(queue["cards"]) == 45
    assert queue["experiment_order"][0] == "C03"
    assert queue["experiment_order"][-1] == "R11"
    assert evaluation_views_for_card(queue, "C03")["primary"] == [(55, 56, 59), (67, 68, 71)]
    assert evaluation_views_for_card(queue, "R18")["H12_L"] == [(67, 68, 79)]
    assert len(evaluation_views_for_card(queue, "C20")["primary"]) == 6


def test_target_identifier_allowlist_rejected(tmp_path):
    source = tmp_path / "source.py"
    data = tmp_path / "train.csv"
    source.write_text("pass\n", encoding="utf-8")
    data.write_text("x\n1\n", encoding="utf-8")
    with pytest.raises(BatchIntegrityError, match="leakage"):
        make_freeze_identity(card={"id": "C03"}, feature_names=["impact_load", "artifact_signal"],
            input_files={"train": data}, source_files=[source], evaluation_views={}, packages=[])


def test_tied_classification_metrics_and_regression_metrics():
    result = metric("classification", np.array([1, 1, 0, 0]), np.array([0.9, 0.2, 0.3, 0.8]))
    assert result == {"n": 4, "macro_f1": 0.5}
    reg = metric("regression", np.array([0.0, 10.0]), np.array([2.0, 7.0]))
    assert reg == {"n": 2, "mae": 2.5}


def test_temporal_rows_cut_off_and_keep_string_record_ids_out_of_features():
    frame = pd.DataFrame({"record_id": ["a", "b", "c"], "cycle_code": [47, 55, 56]})
    fit, score = temporal_rows(frame, 55)
    assert fit.tolist() == [0, 1]
    assert score.tolist() == [2]
    assert frame.iloc[fit].record_id.tolist() == ["a", "b"]


def test_bundle_is_hash_bound_exclusive_and_resume_safe(tmp_path):
    path = tmp_path / "pilot.npz"
    identity = {"card_id": "C03", "cutoff": 55}
    arrays = {"rows": np.array([1, 2]), "prediction": np.array([0.2, 0.8])}
    write_bundle_exclusive(path, arrays, identity, {"fit_predictor_count": 1})
    resumed, meta = read_bundle(path, identity)
    assert np.array_equal(resumed["rows"], arrays["rows"])
    assert meta["metadata"]["fit_predictor_count"] == 1
    with pytest.raises(FileExistsError):
        write_bundle_exclusive(path, arrays, identity, {})
    with pytest.raises(BatchIntegrityError, match="identity"):
        read_bundle(path, {"card_id": "C03", "cutoff": 67})
    path.with_suffix(".json").write_text("{}\n", encoding="utf-8")
    with pytest.raises(BatchIntegrityError, match="schema"):
        read_bundle(path, identity)


def test_bundle_detects_sidecar_metadata_tampering(tmp_path):
    path = tmp_path / "pilot.npz"
    write_bundle_exclusive(path, {"rows": np.array([1])}, {"id": "C03"}, {"fit_predictor_count": 1})
    sidecar = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
    sidecar["metadata"]["fit_predictor_count"] = 7
    path.with_suffix(".json").write_text(json.dumps(sidecar), encoding="utf-8")
    with pytest.raises(BatchIntegrityError, match="Sidecar"):
        read_bundle(path, {"id": "C03"})


def test_orphaned_bundle_is_preserved_and_blocked(tmp_path):
    path = tmp_path / "pilot.npz"
    path.write_bytes(b"partial")
    with pytest.raises(BatchIntegrityError, match="Incomplete"):
        read_bundle(path, {"id": "C03"})
    assert path.read_bytes() == b"partial"


def test_freeze_is_immutable(tmp_path):
    freeze = tmp_path / "freeze.json"
    identity = {"recipe": {"id": "C03"}, "data": "abc"}
    signature = ensure_frozen(freeze, identity)
    assert ensure_frozen(freeze, identity) == signature
    with pytest.raises(BatchIntegrityError, match="differs"):
        ensure_frozen(freeze, {"recipe": {"id": "C03"}, "data": "changed"})


def test_exclusive_training_slot_and_atomic_budget_reservation(tmp_path):
    lock = tmp_path / "slot.lock"
    with exclusive_training_slot(lock):
        with pytest.raises(BatchIntegrityError, match="exclusive slot"):
            with exclusive_training_slot(lock):
                pass
    ledger = tmp_path / "ledger.jsonl"
    reserved = reserve_card_budget(ledger, budget_key="C03:PILOT", card_id="C03",
                                   predictor_fits=2, cap=3)
    assert reserved["reserved_total"] == 2
    assert reserve_card_budget(ledger, budget_key="C03:PILOT", card_id="C03",
                               predictor_fits=2, cap=3)["status"] == "EXISTING_RESERVATION"
    with pytest.raises(BatchIntegrityError, match="budget blocked"):
        reserve_card_budget(ledger, budget_key="R01:PILOT", card_id="R01",
                            predictor_fits=2, cap=3)


def test_budget_reservation_rejects_matching_key_with_wrong_card_identity(tmp_path):
    ledger = tmp_path / "ledger.jsonl"
    ledger.write_text(json.dumps({"event": "BUDGET_RESERVED", "budget_key": "R14:PILOT",
                                  "card_id": "C99", "predictor_fits": 2}) + "\n",
                      encoding="utf-8")
    with pytest.raises(BatchIntegrityError, match="Existing reservation differs"):
        reserve_card_budget(ledger, budget_key="R14:PILOT", card_id="R14",
                            predictor_fits=2, cap=180)


def test_budget_reservation_rejects_duplicate_identical_reservation_rows(tmp_path):
    ledger = tmp_path / "ledger.jsonl"
    event = {"event": "BUDGET_RESERVED", "budget_key": "R14:PILOT",
             "card_id": "R14", "predictor_fits": 2}
    ledger.write_text(json.dumps(event) + "\n" + json.dumps(event) + "\n", encoding="utf-8")
    with pytest.raises(BatchIntegrityError, match="Duplicate existing budget reservation"):
        reserve_card_budget(ledger, budget_key="R14:PILOT", card_id="R14",
                            predictor_fits=2, cap=180)


def test_synthetic_run_separates_labels_freezes_saves_and_verifies(tmp_path):
    cycles = np.arange(47, 80, dtype=int)
    frame = pd.DataFrame({"record_id": [f"r{i}" for i in cycles],
        "cycle_code": cycles, "impact_load": cycles.astype(float),
        "artifact_signal": (cycles % 2).astype(int), "threat_rank": cycles.astype(float)})
    target = frame.artifact_signal.to_numpy()
    source = tmp_path / "adapter.py"
    data = tmp_path / "train.csv"
    source.write_text("# frozen synthetic adapter\n", encoding="utf-8")
    data.write_text("synthetic source\n", encoding="utf-8")
    views = {"primary": [(55, 56, 59), (67, 68, 71)],
             "H12": [(55, 56, 67), (67, 68, 79)]}
    identity = make_freeze_identity(card={"id": "C03", "task": "classification"},
        feature_names=["cycle_code", "impact_load"], input_files={"train": data},
        source_files=[source], evaluation_views=views, packages=[])
    calls = []

    def adapter(fit_x, fit_y, score_x, *, controls, card_id, cutoff):
        assert "record_id" not in fit_x and "artifact_signal" not in fit_x and "threat_rank" not in fit_x
        assert "artifact_signal" not in score_x and "threat_rank" not in score_x
        assert fit_x.cycle_code.max() <= cutoff
        calls.append(cutoff)
        return {"prediction": np.where(score_x.cycle_code.to_numpy() % 2, 0.9, 0.1),
                "raw_candidate": np.full(len(score_x), 0.5), "fit_predictor_count": 1,
                "state": {"cutoff": cutoff}, "fit_metadata": {"fit_rows": len(fit_y)}}

    def controls(cutoff, score_rows):
        return {"baseline_prediction": np.full(len(score_rows), 0.5)}

    freeze = tmp_path / "freeze.json"
    freeze_signature = ensure_frozen(freeze, identity)
    report = run_pilot(frame=frame, target=target,
        card={"id": "C03", "task": "classification", "pilot_predictor_fits": 2},
        feature_names=["cycle_code", "impact_load"], adapter=adapter,
        control_loader=controls, freeze_identity=identity, freeze_path=freeze,
        artifact_dir=tmp_path / "artifacts", event_path=tmp_path / "attempts.jsonl",
        evaluation_views=views, report_path=tmp_path / "pilot.json")
    assert report["fit_count_matches_card"] is True
    assert len(calls) == 2
    verification = verify_report(report=report, frame=frame, target=target,
        artifact_dir=tmp_path / "artifacts", freeze_signature=freeze_signature,
        evaluation_views=views)
    assert verification["status"] == "PASS"
    assert verification["views"]["primary"]["n"] == 8
    bad_dir = tmp_path / "bad-artifacts"
    bad_dir.mkdir()
    with np.load(tmp_path / "artifacts" / "C03_fit55.npz", allow_pickle=False) as saved:
        arrays55 = {key: saved[key].copy() for key in saved.files}
    bad_rows = arrays55["score_rows"][:-1]
    arrays55 = {key: value[:-1] if value.shape[:1] == arrays55["score_rows"].shape else value
                for key, value in arrays55.items()}
    bad_identity = {"freeze_signature": freeze_signature, "card_id": "C03", "cutoff": 55,
                    "score_rows": bad_rows.tolist()}
    write_bundle_exclusive(bad_dir / "C03_fit55.npz", arrays55, bad_identity,
                           {"fit_predictor_count": 1})
    import shutil
    shutil.copy2(tmp_path / "artifacts" / "C03_fit67.npz", bad_dir / "C03_fit67.npz")
    shutil.copy2(tmp_path / "artifacts" / "C03_fit67.json", bad_dir / "C03_fit67.json")
    with pytest.raises(BatchIntegrityError, match="exactly match"):
        verify_report(report=report, frame=frame, target=target,
            artifact_dir=bad_dir, freeze_signature=freeze_signature,
            evaluation_views=views)
    resumed = run_pilot(frame=frame, target=target,
        card={"id": "C03", "task": "classification", "pilot_predictor_fits": 2},
        feature_names=["cycle_code", "impact_load"], adapter=adapter,
        control_loader=controls, freeze_identity=identity, freeze_path=freeze,
        artifact_dir=tmp_path / "artifacts", event_path=tmp_path / "attempts.jsonl",
        evaluation_views=views, report_path=tmp_path / "pilot.json")
    assert resumed == report
    assert calls == [55, 67]


def test_tied_binary_quantiles_empty_streams_unseen_categories_and_zero_support():
    fit_bins, score_bins, edges = fit_only_quantile_bins([0, 0, 1, 1], [0, 1, np.nan], n_bins=4)
    assert edges == [0.0, 0.5, 1.0]
    assert fit_bins.tolist() == [0, 0, 1, 1]
    assert score_bins.tolist() == [0, 1, -1]
    assert stream_missing_mask([None, np.nan, "", "  ", "token"]).tolist() == [True, True, True, True, False]
    missing, unseen, vocabulary = category_novelty_flags(["A", "A", None], ["A", "NEW", "  "])
    assert vocabulary == ["A"]
    assert missing.tolist() == [False, False, True]
    assert unseen.tolist() == [False, True, False]
    assert supported_group_metric("classification", np.array([]), np.array([]), 30) == {
        "n": 0, "supported": False, "metric": None}


def test_classification_subgroups_use_fit_defined_bins_and_keep_small_groups_descriptive():
    n = 24
    frame = pd.DataFrame({
        "record_id": [f"r{i}" for i in range(n)],
        "cycle_code": np.repeat([1, 2], 12),
        "signal_stream": [" " if i % 3 == 0 else "token" for i in range(n)],
        "entry_friction": np.where(np.arange(n) % 2, 1.0, 0.0),
        "text_load": np.arange(n, dtype=float),
        "org_code": ["known" if i < 12 else "new" for i in range(n)],
    })
    rows = np.arange(12, 24)
    y = (rows % 2).astype(int)
    result = _classification_group_metrics(frame, y,
        np.linspace(0.1, 0.9, len(rows)), np.full(len(rows), 0.5),
        rows, np.full(len(rows), 1))
    assert result["groups"]["signal_stream"]["missing"]["n"] == 4
    assert result["groups"]["entry_friction"]["1"]["n"] == 6
    assert "text_load_q" in result["groups"]
    assert result["groups"]["org_code_novelty"]["unseen"]["n"] == 12
    assert all(not group["supported"] for field in result["groups"].values()
               for group in field.values())
    assert result["supported_guard_failures"] == []


def test_target_rates_exclude_same_cycle_labels_and_vocabulary_order_is_stable():
    train_categories = np.array(["A", "A", "A", "B"], dtype=object)
    train_target = np.array([0.0, 1.0, 1.0, 0.0])
    train_cycles = np.array([1, 1, 2, 2])
    score_categories = np.array(["A", "A", "A"], dtype=object)
    score_cycles = np.array([1, 2, 3])
    train_rate, score_rate = past_cycle_smoothed_target_rates(
        train_categories, train_target, train_cycles, score_categories, score_cycles,
        prior=0.5, smoothing=1.0)
    perturbed = train_target.copy()
    perturbed[train_cycles == 2] = 1.0 - perturbed[train_cycles == 2]
    train_rate_2, score_rate_2 = past_cycle_smoothed_target_rates(
        train_categories, perturbed, train_cycles, score_categories, score_cycles,
        prior=0.5, smoothing=1.0)
    assert train_rate[train_cycles == 2].tolist() == train_rate_2[train_cycles == 2].tolist()
    assert score_rate[score_cycles == 2].tolist() == score_rate_2[score_cycles == 2].tolist()
    assert train_rate[train_cycles == 3].tolist() == train_rate_2[train_cycles == 3].tolist()
    assert score_rate[score_cycles == 3].tolist() != score_rate_2[score_cycles == 3].tolist()
    assert stable_unique_tokens(["b a", "c a"]) == ["a", "b", "c"]


def test_interrupt_after_first_origin_resumes_without_refit_and_mismatch_blocks(tmp_path):
    cycles = np.arange(47, 80, dtype=int)
    frame = pd.DataFrame({"record_id": [f"id-{i}" for i in cycles],
        "cycle_code": cycles, "x": cycles.astype(float)})
    target = (cycles % 2).astype(int)
    views = {"primary": [(55, 56, 59), (67, 68, 71)]}
    source = tmp_path / "adapter.py"
    data = tmp_path / "train.csv"
    source.write_text("pass\n", encoding="utf-8")
    data.write_text("x\n", encoding="utf-8")
    identity = make_freeze_identity(card={"id": "C03", "task": "classification"},
        feature_names=["x"], input_files={"train": data}, source_files=[source],
        evaluation_views=views, packages=[])
    calls = []
    def adapter(fit_x, fit_y, score_x, *, controls, card_id, cutoff):
        calls.append(cutoff)
        if cutoff == 67 and calls.count(67) == 1:
            raise RuntimeError("synthetic interruption")
        return {"prediction": np.full(len(score_x), 0.5), "fit_predictor_count": 1}
    control = lambda cutoff, rows: {"baseline_prediction": np.full(len(rows), 0.5)}
    kwargs = dict(frame=frame, target=target,
        card={"id": "C03", "task": "classification", "pilot_predictor_fits": 2},
        feature_names=["x"], adapter=adapter, control_loader=control,
        freeze_identity=identity, freeze_path=tmp_path / "freeze.json",
        artifact_dir=tmp_path / "partial", event_path=tmp_path / "partial.jsonl",
        evaluation_views=views)
    with pytest.raises(RuntimeError, match="interruption"):
        run_pilot(**kwargs)
    assert (tmp_path / "partial" / "C03_fit55.npz").exists()
    assert not (tmp_path / "partial" / "C03_fit67.npz").exists()
    report = run_pilot(**kwargs)
    assert report["fit_count_matches_card"] is True
    assert calls == [55, 67, 67]
    mismatch_kwargs = {**kwargs, "artifact_dir": tmp_path / "wrong-count",
        "event_path": tmp_path / "wrong-count.jsonl", "freeze_path": tmp_path / "wrong-freeze.json",
        "card": {"id": "C03", "task": "classification", "pilot_predictor_fits": 3}}
    before_mismatch_calls = list(calls)
    with pytest.raises(BatchIntegrityError, match="reservation differs"):
        run_pilot(**mismatch_kwargs)
    assert calls == before_mismatch_calls
    assert not (tmp_path / "wrong-count" / "pilot.json").exists()
    zero_fit_adapter = lambda fit_x, fit_y, score_x, **kwargs: {
        "prediction": np.full(len(score_x), 0.5), "fit_predictor_count": 0}
    with pytest.raises(BatchIntegrityError, match="Card fit count mismatch"):
        run_pilot(**{**kwargs, "adapter": zero_fit_adapter,
            "artifact_dir": tmp_path / "zero-fit", "event_path": tmp_path / "zero-fit.jsonl",
            "freeze_path": tmp_path / "zero-fit-freeze.json",
            "budget_ledger_path": tmp_path / "zero-fit-ledger.jsonl",
            "report_path": tmp_path / "zero-fit-pilot.json"})
    assert not (tmp_path / "zero-fit-pilot.json").exists()


def test_vocabulary_digest_is_stable_across_python_hash_seeds():
    code = ("from src.models.optimization_v3_batch1 import stable_unique_tokens, json_sha256; "
            "print(json_sha256(stable_unique_tokens(['z a', 'b z'])))")
    digests = []
    for seed in ("1", "8675309"):
        env = dict(os.environ)
        env["PYTHONHASHSEED"] = seed
        run = subprocess.run([sys.executable, "-c", code], cwd=Path.cwd(), env=env,
                             capture_output=True, text=True, check=True)
        digests.append(run.stdout.strip())
    assert digests[0] == digests[1]


def test_c03_fit_only_transform_and_routing_smoke_on_synthetic_rows():
    n = 80
    rows = np.arange(n)
    X = pd.DataFrame({
        "impact_load": (rows % 13).astype(float),
        "entry_friction": (rows % 2).astype(int),
        "cycle_code": (47 + rows % 12).astype(int),
        "scope_pressure": (rows % 9).astype(float),
        "text_load": (rows % 7).astype(float),
        "scope_trace": [str(i % 5) for i in rows],
        "org_spread": (rows % 4).astype(int),
        "module_spread": (rows % 6).astype(int),
        "telemetry_flag": (rows % 2).astype(int),
        "org_code": [f"org-{i % 4}" for i in rows],
        "module_code": [f"mod-{i % 5}" for i in rows],
        "pattern_code": [f"pat-{i % 3}" for i in rows],
        "signal_stream": [
            f"alpha {('red', 'blue', 'green', 'yellow')[i % 4]} gamma "
            f"{('delta', 'epsilon', 'zeta', 'eta', 'theta', 'iota', 'kappa')[i % 7]} omega"
            for i in rows
        ],
    }).loc[:, list(FEATURE_NAMES)]
    target = (rows % 2).astype(np.int8)
    model = SplitUnigramBigramClassifier().fit(X.iloc[:64], target[:64])
    state = model.audit_state()
    prediction = model.predict_proba(X.iloc[64:])
    assert state["validation_data_passed_to_fit"] is False
    assert state["unigram"]["vocabulary_size"] > 0
    assert state["bigram"]["vocabulary_size"] > 0
    assert prediction.shape == (16,)
    assert np.isfinite(prediction).all()
    controls = {
        "stream_missing": np.array([False, True] * 8),
        "c2b_probability": np.full(16, 0.4),
        "cb_probability": np.full(16, 0.7),
    }
    result = fit_predict_c03(X.iloc[:64], target[:64], X.iloc[64:],
                             controls=controls, card_id="C03", cutoff=55)
    assert result["fit_predictor_count"] == 1
    assert result["prediction"].shape == (16,)
    assert np.array_equal(result["prediction"][controls["stream_missing"]], np.full(8, 0.7))
    assert np.isfinite(result["raw_candidate"]).all()
    synthetic_arrays = {
        "raw_candidate": result["raw_candidate"],
        "control__stream_missing": controls["stream_missing"],
        "control__c2b_probability": controls["c2b_probability"],
        "control__cb_probability": controls["cb_probability"],
        "control__baseline_sparse_raw": np.full(16, 0.2),
    }
    reconstructed, baseline = reconstruct_c03(synthetic_arrays)
    assert np.array_equal(reconstructed, result["prediction"])
    assert np.allclose(baseline[controls["stream_missing"]], 0.7)
