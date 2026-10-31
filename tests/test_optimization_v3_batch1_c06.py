from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scripts import optimization_v3_batch1_c06 as c06_runner
from scripts.optimization_v3_batch1_verify import verify_report
from src.models.classification_round2_models import stream_missing_mask
from src.models.optimization_v3_batch1 import make_freeze_identity, run_pilot
from src.models.optimization_v3_batch1_c06 import (
    BLEND_WEIGHT,
    CATEGORY_COLUMNS,
    FEATURE_NAMES,
    MINIMUM_CLASS_SUPPORT,
    NOVELTY_FEATURE_NAMES,
    SPARSE_PARAMS,
    FitChronologicalCategoryNovelty,
    fit_predict,
    validate_flag_support,
)
from src.models.optimization_v3_sparse_classification import PARAMS as FROZEN_SPARSE_PARAMS


def _features(cycles: np.ndarray, *, row_offset: int = 0) -> pd.DataFrame:
    cycles = np.asarray(cycles, dtype=np.int64)
    row = np.arange(row_offset, row_offset + len(cycles), dtype=np.int64)
    within_cycle = np.arange(len(cycles), dtype=np.int64) % 60
    category_values = {}
    for field, prefix in zip(CATEGORY_COLUMNS, ("org", "module", "pattern")):
        values = []
        for local_index, (cycle, position) in enumerate(zip(cycles, within_cycle)):
            if int(row_offset + local_index) % 4 == 0:
                values.append(None)
            elif position < 30:
                values.append(f"{prefix}-{int(cycle) - 47}-{position}")
            else:
                values.append(f"{prefix}-0-{position - 30}")
        category_values[field] = values
    stream = [None if int(index) % 4 == 0 else f"common alpha word{int(index) % 11} beta" for index in row]
    values: dict[str, object] = {
        "impact_load": (row % 17).astype(float),
        "entry_friction": (row % 2).astype(float),
        "cycle_code": cycles,
        "scope_pressure": (row % 13).astype(float),
        "text_load": (row % 2).astype(float),
        "scope_trace": [str(value % 5) for value in row],
        "org_spread": (row % 4).astype(float),
        "module_spread": (row % 6).astype(float),
        "telemetry_flag": (row % 2).astype(float),
        **category_values,
        "signal_stream": stream,
    }
    frame = pd.DataFrame(values)
    if set(frame.columns) != set(FEATURE_NAMES):
        raise AssertionError("C06 test feature builder must match the sparse feature allowlist")
    return frame.loc[:, list(FEATURE_NAMES)]


def _synthetic_train() -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    cycles = np.repeat(np.arange(47, 80, dtype=np.int64), 60)
    features = _features(cycles)
    row = np.arange(len(cycles), dtype=np.int64)
    target = ((row % 7) < 3).astype(np.int8)
    frame = features.copy()
    frame.insert(0, "record_id", [f"c06-row-{index:05d}" for index in row])
    frame["artifact_signal"] = target
    return frame, target, cycles


def _controls(rows: np.ndarray, score: pd.DataFrame) -> dict[str, np.ndarray]:
    missing = stream_missing_mask(score)
    c2b = np.full(len(rows), 0.50, dtype=np.float64)
    cb = np.full(len(rows), 0.48, dtype=np.float64)
    old_sparse = np.full(len(rows), 0.52, dtype=np.float64)
    baseline = np.where(missing, cb, 0.75 * c2b + BLEND_WEIGHT * old_sparse)
    return {
        "baseline_prediction": baseline,
        "baseline_sparse_raw": old_sparse,
        "c2b_probability": c2b,
        "cb_probability": cb,
        "stream_missing": missing,
        "control_rows": np.asarray(rows, dtype=np.int64),
    }


def test_ordered_fit_flags_wait_until_cycle_end_and_missing_is_not_novel() -> None:
    X = pd.DataFrame({
        "org_code": ["same", "same", "same", "new", "new", None],
        "module_code": ["m0", "m0", "m0", "m1", "m1", None],
        "pattern_code": ["p0", "p0", "p0", "p1", "p1", None],
    })
    cycles = np.array([47, 47, 48, 48, 48, 49], dtype=np.int64)
    encoder = FitChronologicalCategoryNovelty(minimum_class_support=0)
    flags = encoder.fit_transform(X, cycles)

    # Both occurrences first seen in cycle 47 are novel; the later cycle is seen.
    assert flags[0, 0] == flags[1, 0] == 1
    assert flags[2, 0] == 0
    # New values remain novel for every occurrence in their first cycle.
    assert flags[3, 0] == flags[4, 0] == 1
    # NA is represented only by the missing feature.
    assert flags[5, 0] == 0 and flags[5, 1] == 1
    np.testing.assert_array_equal(flags[:, 2], np.array([1, 1, 0, 1, 1, 0], dtype=np.uint8))

    score = pd.DataFrame({
        "org_code": ["fit-known", "score-only", None],
        "module_code": ["fit-known", "score-only", None],
        "pattern_code": ["fit-known", "score-only", None],
    })
    # Replace the known first score token with a value actually seen during fit.
    score.loc[0, "org_code"] = "same"
    score.loc[0, "module_code"] = "m0"
    score.loc[0, "pattern_code"] = "p0"
    transformed = encoder.transform(score)
    assert transformed[0, 0] == 0
    assert transformed[1, 0] == 1
    assert transformed[2, 0] == 0 and transformed[2, 1] == 1


def test_flag_support_gate_requires_at_least_thirty_zeros_and_ones() -> None:
    balanced = np.tile(np.array([[0, 1, 0, 1, 0, 1], [1, 0, 1, 0, 1, 0]], dtype=np.uint8), (30, 1))
    support = validate_flag_support(balanced)
    assert set(support) == set(NOVELTY_FEATURE_NAMES)
    assert all(item["zeros"] == item["ones"] == 30 for item in support.values())
    with pytest.raises(ValueError, match="below 30"):
        validate_flag_support(balanced[:-1])


def test_c06_preserves_frozen_sparse_classifier_parameters() -> None:
    assert SPARSE_PARAMS == dict(FROZEN_SPARSE_PARAMS)
    assert MINIMUM_CLASS_SUPPORT == 30
    assert len(NOVELTY_FEATURE_NAMES) == 6


def test_c06_two_fold_synthetic_pilot_reaches_independent_report_aggregation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    frame, target, cycles = _synthetic_train()
    views = {
        "W1": [(55, 56, 59)],
        "W4": [(67, 68, 71)],
        "primary": [(55, 56, 59), (67, 68, 71)],
        "H12": [(55, 56, 67), (67, 68, 79)],
    }
    controls: dict[int, dict[str, np.ndarray]] = {}
    for cutoff in (55, 67):
        rows = np.flatnonzero((cycles > cutoff) & (cycles <= cutoff + 12))
        score = frame.iloc[rows].loc[:, list(FEATURE_NAMES)]
        controls[cutoff] = _controls(rows, score)

    source = tmp_path / "synthetic_c06_source.py"
    train = tmp_path / "synthetic_c06_train.csv"
    source.write_text("synthetic C06 source identity\n", encoding="utf-8")
    frame.to_csv(train, index=False)
    card = {"id": "C06", "task": "classification", "pilot_predictor_fits": 2}
    identity = make_freeze_identity(
        card=card, feature_names=FEATURE_NAMES,
        input_files={"synthetic_classification_train": train},
        source_files=[source], evaluation_views=views, packages=[],
    )
    run_dir = tmp_path / "runs"
    freeze_path = tmp_path / "freeze.json"
    event_path = tmp_path / "attempts.jsonl"
    ledger_path = tmp_path / "batch_ledger.jsonl"
    report_path = tmp_path / "results.json"
    report = run_pilot(
        card=card, artifact_dir=run_dir, budget_ledger_path=ledger_path,
        max_successful_fits=180, frame=frame, target=target,
        feature_names=FEATURE_NAMES, adapter=fit_predict,
        control_loader=lambda cutoff, rows: controls[cutoff],
        freeze_identity=identity, freeze_path=freeze_path, event_path=event_path,
        evaluation_views=views, threads=2, report_path=report_path,
    )
    verification = verify_report(
        report=report, frame=frame, target=target, artifact_dir=run_dir,
        freeze_signature=json.loads(freeze_path.read_text(encoding="utf-8"))["signature"],
        evaluation_views=views,
        prediction_reconstructors={"C06": c06_runner.reconstruct_c06},
    )
    model_state_verification = c06_runner._verify_c06_fit_states(
        report=report, frame=frame, target=target, artifact_dir=run_dir,
    )

    assert report["fit_count_matches_card"] is True
    assert verification["status"] == "PASS"
    assert model_state_verification["status"] == "PASS"
    assert verification["views"]["primary"]["n"] == 8 * 60
    assert verification["views"]["H12"]["n"] == 24 * 60
    assert len(np.unique(frame.iloc[np.flatnonzero(cycles > 55)]["record_id"].astype(str))) == 1_440
    # Tied training values produce sparse quantile bins; absent bins remain explicit empty groups.
    text_groups = verification["views"]["primary"]["subgroup_diagnostics"]["groups"]["text_load_q"]
    assert any(int(record["n"]) == 0 for record in text_groups.values())
    for cutoff in (55, 67):
        sidecar = json.loads((run_dir / f"C06_fit{cutoff}.json").read_text(encoding="utf-8"))
        state = sidecar["metadata"]["state"]
        novelty_state = state["chronological_category_novelty_state"]
        assert state["id"] == "V3_BATCH1_C06_ORDERED_CATEGORY_NOVELTY_SPARSE_LGB"
        assert state["target_rates_used"] is False
        assert novelty_state["validation_or_test_categories_used_for_fit_features"] is False
        assert novelty_state["minimum_class_support"] == 30


def test_reconstruct_c06_preserves_frozen_present_blend_and_missing_fallback() -> None:
    arrays = {
        "raw_candidate": np.array([0.10, 0.90, 0.50]),
        "control__stream_missing": np.array([False, True, False]),
        "control__c2b_probability": np.array([0.20, 0.30, 0.70]),
        "control__cb_probability": np.array([0.40, 0.50, 0.60]),
        "control__baseline_sparse_raw": np.array([0.80, 0.10, 0.30]),
    }
    candidate, baseline = c06_runner.reconstruct_c06(arrays)
    np.testing.assert_allclose(candidate, np.array([0.175, 0.50, 0.65]), rtol=0, atol=1e-15)
    np.testing.assert_allclose(baseline, np.array([0.35, 0.50, 0.60]), rtol=0, atol=1e-15)
