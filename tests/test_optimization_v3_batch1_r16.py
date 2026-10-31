from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from scripts import optimization_v3_batch1_r16 as r16_runner
from src.models.optimization_v3_batch1 import json_sha256, write_bundle_exclusive
from src.models.optimization_v3_batch1_r16 import (
    ExponentiallyWeightedSVDLGB,
    FEATURE_NAMES,
    normalized_recency_weights,
    rank50_prediction,
    replace_svd_component,
)


def test_freeze_match_survives_json_normalizing_integer_mapping_keys() -> None:
    identity = {"expected_fit_rows": {55: 12_020, 67: 29_705}}
    signature = json_sha256(identity)
    persisted = json.loads(json.dumps(identity))

    assert persisted != identity
    assert r16_runner._freeze_matches({"identity": persisted, "signature": signature}, identity)
    assert not r16_runner._freeze_matches({"identity": persisted, "signature": "wrong"}, identity)


def test_recency_weights_keep_all_rows_and_have_mean_one() -> None:
    cycles = np.array([47, 47, 51, 55, 55], dtype=np.int64)
    weights, audit = normalized_recency_weights(cycles, 55)

    assert weights.shape == cycles.shape
    assert audit["fit_rows"] == len(cycles)
    assert audit["fit_cycle_min"] == 47
    assert audit["fit_cycle_max"] == 55
    assert np.mean(weights) == pytest.approx(1.0, abs=1e-12)
    assert weights[0] == pytest.approx(weights[1])
    assert weights[-1] > weights[2] > weights[0]
    assert audit["sha256"]


@pytest.mark.parametrize("cycles, cutoff", [([46, 55], 55), ([47, 56], 55), ([], 55)])
def test_recency_weights_reject_out_of_fit_or_empty_cycles(cycles, cutoff) -> None:
    with pytest.raises(ValueError):
        normalized_recency_weights(cycles, cutoff)


def test_rank50_uses_average_ties_within_each_cycle() -> None:
    raw = np.array([10.0, 10.0, 30.0, 10.0, 30.0])
    cycles = np.array([56, 56, 56, 57, 57], dtype=np.int64)

    mapped = rank50_prediction(raw, cycles)

    expected = np.array([
        0.5 * 10 + 0.5 * 100 * 1.5 / 3,
        0.5 * 10 + 0.5 * 100 * 1.5 / 3,
        0.5 * 30 + 0.5 * 100 * 3 / 3,
        0.5 * 10 + 0.5 * 100 * 1 / 2,
        0.5 * 30 + 0.5 * 100 * 2 / 2,
    ])
    np.testing.assert_allclose(mapped, expected, rtol=0, atol=1e-12)


def test_replacement_route_changes_only_the_svd_lgb_component() -> None:
    new_lgb = np.array([20.0, 40.0, 100.0])
    ridge = np.array([30.0, 50.0, 80.0])
    sparse = np.array([40.0, 60.0, 70.0])

    prediction = replace_svd_component(new_lgb, ridge, sparse)

    expected = np.clip(0.5625 * new_lgb + 0.1875 * ridge + 0.25 * sparse, 0.0, 100.0)
    np.testing.assert_allclose(prediction, expected, rtol=0, atol=1e-12)
    assert not np.array_equal(prediction, ridge)


def test_replacement_route_rejects_misaligned_or_out_of_range_controls() -> None:
    with pytest.raises(ValueError):
        replace_svd_component(np.array([20.0]), np.array([30.0, 40.0]), np.array([50.0]))
    with pytest.raises(ValueError):
        replace_svd_component(np.array([101.0]), np.array([30.0]), np.array([50.0]))


def test_weighted_svd_lgb_fit_accepts_and_audits_all_past_rows() -> None:
    n_rows = 90
    indices = np.arange(n_rows, dtype=np.int64)
    cycles = np.repeat(np.arange(47, 56, dtype=np.int64), 10)
    values = {}
    for name in FEATURE_NAMES:
        if name == "signal_stream":
            values[name] = [
                f"alpha{idx % 37} beta{idx % 29} gamma{idx % 23} delta{idx % 19} epsilon{idx % 17}"
                for idx in indices
            ]
        elif name in {"org_code", "module_code", "pattern_code"}:
            values[name] = [f"cat{idx % 3}" for idx in indices]
        elif name == "cycle_code":
            values[name] = cycles
        else:
            values[name] = (indices % 11).astype(float)
    frame = pd.DataFrame(values, index=indices).loc[:, list(FEATURE_NAMES)]
    target = (indices * 7 % 101).astype(float)

    model = ExponentiallyWeightedSVDLGB(cutoff=55).fit(frame, target)

    state = model.audit_state()
    assert state["fit_rows"] == n_rows
    assert state["source_fit_cycle_max"] == 55
    assert len(state["source_fit_cycles"]) == n_rows
    assert state["sample_weighting"]["fit_rows"] == n_rows
    assert state["sample_weighting"]["mean"] == pytest.approx(1.0, abs=1e-12)
    prediction = model.predict(frame.iloc[:12])
    assert prediction.shape == (12,)
    assert np.isfinite(prediction).all()


def test_standard_pilot_group_sources_explicitly_include_h12_halves() -> None:
    early_n = r16_runner.EXPECTED_VIEW_ROWS["H12_E"]
    late_n = r16_runner.EXPECTED_VIEW_ROWS["H12_L"]
    cycles = np.concatenate((
        np.resize(np.arange(56, 68, dtype=np.int64), early_n),
        np.resize(np.arange(68, 80, dtype=np.int64), late_n),
    ))
    early_rows = np.arange(early_n, dtype=np.int64)
    late_rows = np.arange(early_n, early_n + late_n, dtype=np.int64)
    per_cutoff = {
        55: {"rows": early_rows, "candidate": np.zeros(early_n), "baseline": np.ones(early_n)},
        67: {"rows": late_rows, "candidate": np.zeros(late_n), "baseline": np.ones(late_n)},
    }
    primary_n = r16_runner.EXPECTED_VIEW_ROWS["primary"]
    view_arrays = {
        "primary": {"rows": np.arange(primary_n), "candidate": np.zeros(primary_n),
                    "baseline": np.ones(primary_n)},
        "H12": {"rows": np.arange(early_n + late_n), "candidate": np.zeros(early_n + late_n),
                "baseline": np.ones(early_n + late_n)},
    }
    standard_context = {"cycles": cycles, "views": {
        "W1": [(55, 56, 59)], "W4": [(67, 68, 71)],
        "primary": [(55, 56, 59), (67, 68, 71)],
        "H12": [(55, 56, 67), (67, 68, 79)],
    }}

    sources = r16_runner._pilot_group_sources(standard_context, per_cutoff, view_arrays)

    assert set(sources) == {"primary", "H12_E", "H12_L"}
    assert len(sources["H12_E"]["rows"]) == early_n
    assert len(sources["H12_L"]["rows"]) == late_n


def test_synthetic_r16_verification_reaches_gate_and_report_publication(tmp_path, monkeypatch) -> None:
    fit_early_n = r16_runner.EXPECTED_FIT_ROWS[55]
    fit_late_n = r16_runner.EXPECTED_FIT_ROWS[67] - r16_runner.EXPECTED_VIEW_ROWS["H12_E"]
    early_n = r16_runner.EXPECTED_VIEW_ROWS["H12_E"]
    late_n = r16_runner.EXPECTED_VIEW_ROWS["H12_L"]
    primary_early_n = r16_runner.EXPECTED_VIEW_ROWS["W1"]
    primary_late_n = r16_runner.EXPECTED_VIEW_ROWS["W4"]
    cycles = np.concatenate((
        np.resize(np.arange(47, 56, dtype=np.int64), fit_early_n),
        np.resize(np.arange(56, 60, dtype=np.int64), primary_early_n),
        np.resize(np.arange(60, 68, dtype=np.int64), early_n - primary_early_n),
        np.resize(np.arange(68, 72, dtype=np.int64), primary_late_n),
        np.resize(np.arange(72, 80, dtype=np.int64), late_n - primary_late_n),
    ))
    rows_early = np.arange(fit_early_n, fit_early_n + early_n, dtype=np.int64)
    rows_late = np.arange(fit_early_n + early_n, len(cycles), dtype=np.int64)
    row_index = np.arange(len(cycles), dtype=np.int64)
    target = (row_index % 101).astype(np.float64)
    frame = pd.DataFrame({
        "record_id": [f"synthetic-{index}" for index in row_index],
        "cycle_code": cycles,
        "signal_stream": [None if index % 7 == 0 else "synthetic token" for index in row_index],
    })
    identity = {"card_id": "R16", "synthetic_only": True}
    signature = json_sha256(identity)
    run_dir = tmp_path / "runs"
    run_dir.mkdir()
    monkeypatch.setattr(r16_runner, "RUN_DIR", run_dir)
    for name in ("RESULTS_PATH", "VERIFICATION_PATH", "GATE_PATH", "SUMMARY_PATH"):
        monkeypatch.setattr(r16_runner, name, tmp_path / f"{name}.json")

    controls = {}
    per_cutoff = {}
    report_views = {}
    for cutoff, rows in ((55, rows_early), (67, rows_late)):
        local_rows = len(rows)
        old_lgb = np.full(local_rows, 44.0)
        ridge = np.full(local_rows, 51.0)
        sparse = np.full(local_rows, 57.0)
        baseline = np.clip(0.5625 * old_lgb + 0.1875 * ridge + 0.25 * sparse, 0, 100)
        raw = (np.arange(local_rows, dtype=np.float64) % 317) / 4.0
        mapped = rank50_prediction(raw, cycles[rows])
        candidate = replace_svd_component(mapped, ridge, sparse)
        controls[cutoff] = {
            "score_rows": rows,
            "old_lgb_svd32": old_lgb,
            "ridge_prediction": ridge,
            "sparse_rank50": sparse,
            "baseline_prediction": baseline,
        }
        per_cutoff[cutoff] = {"rows": rows, "candidate": candidate,
                              "baseline": baseline, "raw": raw, "rank50": mapped}
        fit_cycles = cycles[cycles <= cutoff]
        _weights, weight_summary = normalized_recency_weights(fit_cycles, cutoff)
        arrays = {
            "score_rows": rows,
            "score_record_ids": frame.iloc[rows]["record_id"].astype(str).to_numpy(dtype=str),
            "baseline_prediction": baseline,
            "prediction": candidate,
            "raw_candidate": raw,
            "audit__candidate_rank50": mapped,
        }
        bundle_identity = {"freeze_signature": signature, "card_id": "R16",
                           "cutoff": cutoff, "score_rows": rows.tolist()}
        metadata = {
            "fit_predictor_count": 1,
            "fit_rows": len(fit_cycles),
            "score_rows": local_rows,
            "fit_seconds": 0.1,
            "fit_metadata": {
                "sample_weighting": weight_summary,
                "resource_preflight": {"status": "PASS", "concurrent_training_jobs": 1},
            },
            "state": {"cutoff": cutoff, "source_fit_cycles": fit_cycles.tolist(),
                      "sample_weighting": weight_summary},
        }
        write_bundle_exclusive(run_dir / f"R16_fit{cutoff}.npz", arrays,
                               bundle_identity, metadata)

    views = r16_runner.EXPECTED_VIEWS
    view_arrays = {}
    for name, segments in views.items():
        row_parts = []
        candidate_parts = []
        baseline_parts = []
        for cutoff, low, high in segments:
            fold = per_cutoff[cutoff]
            selected = (cycles[fold["rows"]] >= low) & (cycles[fold["rows"]] <= high)
            row_parts.append(fold["rows"][selected])
            candidate_parts.append(fold["candidate"][selected])
            baseline_parts.append(fold["baseline"][selected])
        view_arrays[name] = {"rows": np.concatenate(row_parts),
                             "candidate": np.concatenate(candidate_parts),
                             "baseline": np.concatenate(baseline_parts)}
        report_views[name] = {
            "rows": len(view_arrays[name]["rows"]),
            "candidate": {"mae": r16_runner._mae(
                target[view_arrays[name]["rows"]], view_arrays[name]["candidate"])},
            "baseline": {"mae": r16_runner._mae(
                target[view_arrays[name]["rows"]], view_arrays[name]["baseline"])},
        }
    context = {
        "identity": identity,
        "target": target,
        "frame": frame,
        "cycles": cycles,
        "controls": controls,
        "views": views,
        "public": {"input_sha256": {}, "source_sha256": {}},
    }
    engine_report = {
        "card_id": "R16",
        "status": "PILOT_SAVED_AWAITING_INDEPENDENT_QA",
        "fit_predictor_count": 2,
        "cutoffs": [55, 67],
        "freeze_signature": signature,
        "views": report_views,
    }

    verification = r16_runner._verify_pilot(context, engine_report)
    r16_runner._publish(context, engine_report, verification,
                         {"total_fit_seconds": 0.2})

    assert verification["status"] == "INDEPENDENT_QA_PASS"
    assert set(verification["groups"]) == {"primary", "H12_E", "H12_L"}
    assert verification["views"]["primary"]["rows"] == r16_runner.EXPECTED_VIEW_ROWS["primary"]
    assert (tmp_path / "RESULTS_PATH.json").is_file()
    assert (tmp_path / "VERIFICATION_PATH.json").is_file()
    assert (tmp_path / "GATE_PATH.json").is_file()
    assert (tmp_path / "SUMMARY_PATH.json").is_file()
    assert json.loads((tmp_path / "GATE_PATH.json").read_text())["decision"] == verification["decision"]
