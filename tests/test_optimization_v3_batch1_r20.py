from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scripts import optimization_v3_batch1_r20 as r20_runner
from scripts.optimization_v3_batch1_verify import verify_report
from scripts.optimization_v3_batch1_r20 import _selection_audit_independent
from src.models.optimization_v3_batch1 import ensure_frozen, load_queue, make_freeze_identity, run_pilot
from src.models.optimization_v3_batch1_r20 import (
    FEATURE_NAMES,
    LIGHTGBM_PARAMETERS,
    MIN_TOKEN_SUPPORT,
    SELECTED_TOKEN_COUNT,
    TokenIdentitySparseL1Regressor,
    _presence_matrix,
    eligible_token_support_count,
    fit_predict,
    rank50_prediction,
    reconstruct_r20,
    select_token_identities,
)
from src.models.optimization_v2_regression import SPARSE_LGB_PARAMETERS


def _synthetic_train() -> pd.DataFrame:
    rows_per_cycle = 144
    cycles = np.repeat(np.arange(47, 80, dtype=np.int64), rows_per_cycle)
    rows = np.arange(len(cycles), dtype=np.int64)
    streams = []
    for i in rows:
        tokens = [f"term{(int(i) % 160 + offset) % 160:03d}" for offset in range(4)]
        streams.append("" if i % 97 == 0 else " ".join(tokens))
    data: dict[str, object] = {
        "record_id": [f"synthetic-id-{i}" for i in rows],
        "impact_load": (rows % 17).astype(float),
        "entry_friction": (rows % 2).astype(float),
        "cycle_code": cycles,
        "scope_pressure": (rows % 11).astype(float),
        "text_load": (rows % 9).astype(float),
        "scope_trace": (rows % 8).astype(float),
        "org_spread": (rows % 6).astype(float),
        "module_spread": (rows % 7).astype(float),
        "telemetry_flag": (rows % 2).astype(float),
        "org_code": [f"org-{int(cycle % 4)}" if cycle <= 55 else "future-org" for cycle in cycles],
        "module_code": [f"module-{int(i % 5)}" for i in rows],
        "pattern_code": [f"pattern-{int(i % 3)}" for i in rows],
        "signal_stream": streams,
        "threat_rank": np.clip(30.0 + (rows % 31) * 1.5 + (cycles - 47) * 0.25, 0, 100),
    }
    frame = pd.DataFrame(data)
    assert set(FEATURE_NAMES) == set(frame.columns) - {"record_id", "threat_rank"}
    return frame


def _controls(score_rows: np.ndarray) -> dict[str, np.ndarray]:
    n = len(score_rows)
    lgb = np.full(n, 48.0, dtype=np.float64)
    ridge = np.full(n, 52.0, dtype=np.float64)
    sparse_raw = np.full(n, 50.0, dtype=np.float64)
    sparse_rank = np.full(n, 50.0, dtype=np.float64)
    baseline = np.clip(0.75 * (0.75 * lgb + 0.25 * ridge) + 0.25 * sparse_rank, 0, 100)
    return {
        "baseline_prediction": baseline,
        "baseline_lgb_svd32": lgb,
        "baseline_hybrid_ridge": ridge,
        "baseline_sparse_raw": sparse_raw,
        "baseline_sparse_rank50": sparse_rank,
        "control_rows": np.asarray(score_rows, dtype=np.int64),
    }


def _reconstruct_arrays(arrays: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    prediction, baseline, mapped = reconstruct_r20(
        arrays["raw_candidate"], arrays["baseline_prediction"], arrays["audit__score_cycles"],
    )
    np.testing.assert_allclose(mapped, arrays["audit__candidate_rank50"], atol=1e-12, rtol=0)
    return prediction, baseline


def test_r20_inherits_sparse_l1_parameters_without_search_or_extra_dependency():
    assert LIGHTGBM_PARAMETERS == SPARSE_LGB_PARAMETERS
    assert LIGHTGBM_PARAMETERS["objective"] == "regression_l1"
    assert LIGHTGBM_PARAMETERS["n_jobs"] == 2


def test_runner_recipe_describes_sparse_lightgbm_token_identity_experiment():
    queue = load_queue(r20_runner.QUEUE_PATH)
    card = next(item for item in queue["cards"] if item["id"] == "R20")
    recipe = r20_runner._recipe(card, {"primary": [(55, 56, 67)]}, r20_runner.CARD_SHA256)
    assert recipe["estimator"]["class"] == "lightgbm.LGBMRegressor"
    assert recipe["estimator"]["parameters_explicit"] == LIGHTGBM_PARAMETERS
    preprocessing = recipe["preprocessing"]
    assert preprocessing["selected_token_count"] == SELECTED_TOKEN_COUNT
    assert preprocessing["token_support"].startswith("binary document presence")
    assert preprocessing["token_fit_scope"].startswith("outer-fit rows")
    assert preprocessing["sparse_design"] is True
    assert preprocessing["text_tfidf_or_svd"] is False
    assert "xgboost" not in json.dumps(recipe).casefold()


def test_support_shortage_skips_every_runner_mode_without_freeze_or_fit(monkeypatch):
    skipped = {
        "public": {
            "status": "SKIPPED_SUPPORT", "candidate_fit_performed": False,
            "freeze_or_output_written": False,
        }
    }
    monkeypatch.setattr(r20_runner, "_prepare", lambda _card, _queue: skipped)
    assert r20_runner.run_card({}, {}, preflight_only=True)["status"] == "SKIPPED_SUPPORT"
    assert r20_runner.run_card({}, {}, freeze_only=True)["freeze_or_output_written"] is False
    assert r20_runner.run_card({}, {}, run_frozen_pilot=True)["candidate_fit_performed"] is False


def test_selection_uses_binary_document_presence_support_shrinkage_and_lexical_ties():
    tokens = [f"token{i:03d}" for i in range(SELECTED_TOKEN_COUNT)]
    document_tokens = [tuple(tokens + (["rare29"] if row < 29 else [])) for row in range(30)]
    document_tokens += [tuple() for _ in range(10)]
    target = np.array([60.0] * 30 + [20.0] * 10)
    selected, audit = select_token_identities(document_tokens, target)
    assert selected == sorted(tokens)
    assert audit["fit_mean"] == 50.0
    assert audit["eligible_token_count"] == SELECTED_TOKEN_COUNT
    assert audit["selected_token_records"][0]["support"] == 30
    assert audit["selected_token_records"][0]["selection_score"] == 5.0
    assert "rare29" not in selected


def test_fixed_presence_schema_deduplicates_tokens_and_keeps_empty_rows():
    selected = [f"token{i:03d}" for i in range(SELECTED_TOKEN_COUNT)]
    docs = [(selected[0], selected[0], selected[1]), tuple(), (selected[127],)]
    matrix = _presence_matrix(docs, selected)
    assert matrix.shape == (3, SELECTED_TOKEN_COUNT)
    assert matrix.nnz == 3
    assert matrix[0, 0] == 1.0 and matrix[0, 1] == 1.0
    assert matrix[1].nnz == 0
    assert matrix[2, 127] == 1.0


def test_rank50_uses_average_ties_and_fixed_25_percent_blend():
    raw = np.array([10.0, 10.0, 30.0, 40.0])
    cycles = np.array([56, 56, 56, 56], dtype=np.int64)
    mapped = rank50_prediction(raw, cycles)
    np.testing.assert_array_equal(mapped, np.array([23.75, 23.75, 52.5, 70.0]))
    candidate, baseline, saved = reconstruct_r20(raw, np.full(4, 50.0), cycles)
    np.testing.assert_array_equal(baseline, np.full(4, 50.0))
    np.testing.assert_array_equal(saved, mapped)
    np.testing.assert_array_equal(candidate, np.clip(37.5 + 0.25 * mapped, 0, 100))


def test_r20_rejects_leaked_features_and_selection_without_128_supported_tokens():
    frame = _synthetic_train()
    X = frame.loc[:, list(FEATURE_NAMES)].iloc[:40].copy()
    model = TokenIdentitySparseL1Regressor(55)
    with pytest.raises(ValueError, match="exactly the frozen"):
        model.fit(X.assign(threat_rank=1.0), frame.threat_rank.iloc[:40].to_numpy())
    with pytest.raises(ValueError, match="found 0"):
        select_token_identities([tuple([f"rare{i}"]) for i in range(40)], np.full(40, 50.0))


def test_r20_runs_two_real_synthetic_fits_through_persisted_report_and_independent_reconstruction(tmp_path: Path):
    frame = _synthetic_train()
    target = frame["threat_rank"].to_numpy(dtype=np.float64)
    views = {
        "W1": [(55, 56, 59)],
        "W4": [(67, 68, 71)],
        "primary": [(55, 56, 59), (67, 68, 71)],
        "H12": [(55, 56, 67), (67, 68, 79)],
    }
    source = tmp_path / "adapter.py"
    source.write_text("# synthetic R20 source identity\n", encoding="utf-8")
    data = tmp_path / "train.csv"
    frame.to_csv(data, index=False)
    identity = make_freeze_identity(
        card={"id": "R20", "task": "regression"},
        feature_names=FEATURE_NAMES,
        input_files={"synthetic_train": data},
        source_files=[source],
        evaluation_views=views,
        packages=[],
    )
    freeze_path = tmp_path / "freeze.json"
    signature = ensure_frozen(freeze_path, identity)
    calls: list[int] = []

    def adapter(fit_x, fit_y, score_x, *, controls, card_id, cutoff):
        assert "record_id" not in fit_x and "threat_rank" not in fit_x
        assert "record_id" not in score_x and "threat_rank" not in score_x
        assert fit_x.cycle_code.max() <= cutoff
        calls.append(cutoff)
        return fit_predict(fit_x, fit_y, score_x, controls=controls, card_id=card_id, cutoff=cutoff)

    report = run_pilot(
        frame=frame,
        target=target,
        card={"id": "R20", "task": "regression", "pilot_predictor_fits": 2},
        feature_names=FEATURE_NAMES,
        adapter=adapter,
        control_loader=lambda cutoff, rows: _controls(rows),
        freeze_identity=identity,
        freeze_path=freeze_path,
        artifact_dir=tmp_path / "artifacts",
        event_path=tmp_path / "attempts.jsonl",
        evaluation_views=views,
        report_path=tmp_path / "pilot.json",
    )
    verification = verify_report(
        report=report,
        frame=frame,
        target=target,
        artifact_dir=tmp_path / "artifacts",
        freeze_signature=signature,
        evaluation_views=views,
        prediction_reconstructors={"R20": _reconstruct_arrays},
    )
    assert report["fit_count_matches_card"] is True
    assert report["views"]["primary"]["rows"] == 1152
    assert report["views"]["H12"]["rows"] == 3456
    assert verification["status"] == "PASS"
    assert calls == [55, 67]
    support = eligible_token_support_count(frame.loc[frame["cycle_code"] <= 55, "signal_stream"].fillna("").astype(str).tolist())
    assert support["support_ge_minimum"] >= SELECTED_TOKEN_COUNT
    independent = _selection_audit_independent(
        frame.loc[frame["cycle_code"] <= 55, list(FEATURE_NAMES)],
        target[frame["cycle_code"].to_numpy(dtype=np.int64) <= 55],
    )
    assert independent["fit_document_count"] == int(np.sum(frame["cycle_code"] <= 55))
    assert len(independent["selected_token_records"]) == SELECTED_TOKEN_COUNT
    for cutoff in (55, 67):
        sidecar = json.loads((tmp_path / "artifacts" / f"R20_fit{cutoff}.json").read_text(encoding="utf-8"))
        state = sidecar["metadata"]["state"]
        assert state["validation_data_passed_to_fit"] is False
        assert state["validation_labels_passed_to_fit"] is False
        assert state["test_data_used_for_fit"] is False
        assert state["objective"] == "regression_l1"
        assert state["selection"]["selected_token_count"] == 128
        assert state["selection"]["eligible_token_count"] >= 128
        assert state["fit_cycle_max"] == cutoff
        assert state["sparse_design"] is True
