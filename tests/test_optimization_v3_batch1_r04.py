from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest
from scipy import sparse
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.utils import murmurhash3_32

from scripts import optimization_v3_batch1_r04 as r04_runner
from src.models.optimization_v3_batch1_r02 import compose_r02_prediction, rank50_prediction
from src.models.optimization_v3_batch1 import BatchIntegrityError, json_sha256
from src.models.optimization_v3_batch1_r04 import (
    CARD_ID,
    FEATURE_NAMES,
    HASHING_PARAMETERS,
    PILOT_CUTOFFS,
    R04HashingSparseHybrid,
    SPARSE_LGB_PARAMETERS,
    fit_predict,
    hashing_diagnostics,
)
from src.models.optimization_v2_regression import SPARSE_LGB_PARAMETERS as ORIGINAL_SPARSE_LGB_PARAMETERS


def test_r02_dependency_requires_same_official_train_hash(tmp_path, monkeypatch):
    report_dir = tmp_path / "R02"
    report_dir.mkdir()
    results = {
        "card_id": "R02", "status": "RETIRE",
        "fit_accounting": {
            "reserved_predictor_fits": 2, "successful_predictor_fits": 2,
            "fit_saved": 2,
            "fits": {"55": {"fit_predictor_count": 1},
                     "67": {"fit_predictor_count": 1}},
        },
    }
    verification = {"card_id": "R02", "status": "PASS"}
    gate = {"card_id": "R02", "status": "RETIRE"}
    identity = {"input_sha256": {"official_regression_train": "different-train-hash"}}
    signature = json_sha256(identity)
    results["freeze_signature"] = signature
    verification["freeze_signature"] = signature
    freeze = {"identity": identity, "signature": signature}
    for name, value in (("results.json", results), ("verification.json", verification),
                        ("gate.json", gate)):
        (report_dir / name).write_text(json.dumps(value), encoding="utf-8")
    freeze_path = tmp_path / "freeze.json"
    freeze_path.write_text(json.dumps(freeze), encoding="utf-8")
    monkeypatch.setattr(r04_runner, "R02_REPORT_DIR", report_dir)
    monkeypatch.setattr(r04_runner, "R02_FREEZE_PATH", freeze_path)

    with pytest.raises(BatchIntegrityError, match="different official Regression Train hash"):
        r04_runner._validate_r02_completion()


def test_r04_recipe_omits_volatile_memory_snapshot_values():
    first = {"55": {"fit_rows": 12_020, "estimated_peak_gib": 1.55,
                     "available_ram_gib": 7.8, "allowed_job_ram_gib": 5.8,
                     "status": "PASS"}}
    second = {"55": {"fit_rows": 12_020, "estimated_peak_gib": 1.55,
                      "available_ram_gib": 6.4, "allowed_job_ram_gib": 4.4,
                      "status": "PASS"}}

    first_summary = r04_runner._hashing_recipe_summary(first)
    second_summary = r04_runner._hashing_recipe_summary(second)
    assert first_summary == second_summary
    assert first_summary["55"] == {"fit_rows": 12_020, "estimated_peak_gib": 1.55}


def _synthetic_features(cycles: np.ndarray, start: int = 0) -> pd.DataFrame:
    row = np.arange(start, start + len(cycles), dtype=np.int64)
    colors = ("red", "blue", "green", "yellow")
    tokens = ("amber", "birch", "cedar", "dahlia", "elm", "fir")
    frame = pd.DataFrame({
        "impact_load": (row % 17).astype(float),
        "entry_friction": (row % 2).astype(int),
        "cycle_code": cycles.astype(int),
        "scope_pressure": (row % 11).astype(float),
        "text_load": (row % 9).astype(float),
        "scope_trace": [str(value % 5) for value in row],
        "org_spread": (row % 4).astype(int),
        "module_spread": (row % 6).astype(int),
        "telemetry_flag": (row % 2).astype(int),
        "org_code": [f"org-{value % 7}" for value in row],
        "module_code": [f"module-{value % 8}" for value in row],
        "pattern_code": [f"pattern-{value % 3}" for value in row],
        "signal_stream": [
            None if value % 17 == 0 else
            f"alpha {colors[value % len(colors)]} gamma {tokens[value % len(tokens)]} omega"
            for value in row
        ],
    })
    return frame.loc[:, list(FEATURE_NAMES)]


def _synthetic_controls(score_x: pd.DataFrame) -> dict[str, np.ndarray]:
    n = len(score_x)
    ridge = np.linspace(8.0, 92.0, n, dtype=np.float64)
    old_sparse = np.linspace(4.0, 96.0, n, dtype=np.float64)[::-1].copy()
    cycles = score_x["cycle_code"].to_numpy(dtype=np.int64)
    baseline = compose_r02_prediction(ridge, old_sparse, cycles)
    return {
        "baseline_prediction": baseline,
        "ridge25_prediction": ridge,
        "baseline_sparse_raw": old_sparse,
        "cycle_code": cycles,
        "control_rows": score_x.index.to_numpy(dtype=np.int64),
    }


def _scores_for_12_cycles(cutoff: int = 55, rows_per_cycle: int = 3) -> pd.DataFrame:
    cycles = np.repeat(np.arange(cutoff + 1, cutoff + 13, dtype=np.int64), rows_per_cycle)
    return _synthetic_features(cycles, start=10_000)


def test_r04_frozen_recipe_and_l1_parameters_are_exact() -> None:
    assert CARD_ID == "R04"
    assert PILOT_CUTOFFS == (55, 67)
    assert HASHING_PARAMETERS == {
        "analyzer": "word",
        "ngram_range": (1, 2),
        "n_features": 262_144,
        "alternate_sign": False,
        "norm": None,
        "binary": False,
        "lowercase": True,
        "strip_accents": "unicode",
        "token_pattern": r"(?u)\b\w\w+\b",
        "dtype": np.float64,
    }
    assert SPARSE_LGB_PARAMETERS == ORIGINAL_SPARSE_LGB_PARAMETERS
    assert "record_id" not in FEATURE_NAMES
    assert "threat_rank" not in FEATURE_NAMES
    assert "artifact_signal" not in FEATURE_NAMES


def test_hashing_diagnostics_count_unique_identity_collisions_not_repeats() -> None:
    # Find a real collision in the frozen space rather than relying on a lucky
    # hand-picked pair. HashVectorizer's occupied bucket is abs(signed hash)%N.
    first_for_bucket: dict[int, str] = {}
    pair: tuple[str, str] | None = None
    for index in range(25_000):
        token = f"token{index:05d}"
        bucket = abs(int(murmurhash3_32(token, seed=0, positive=False))) % HASHING_PARAMETERS["n_features"]
        if bucket in first_for_bucket:
            pair = (first_for_bucket[bucket], token)
            break
        first_for_bucket[bucket] = token
    assert pair is not None

    vectorizer = HashingVectorizer(**dict(HASHING_PARAMETERS))
    docs = [f"{pair[0]} {pair[1]} {pair[0]}", "cedar cedar cedar"]
    matrix = vectorizer.transform(docs).tocsr()
    stats = hashing_diagnostics(docs, matrix, vectorizer=vectorizer)
    assert sparse.isspmatrix_csr(matrix)
    assert stats["matrix_nnz"] == int(matrix.nnz)
    assert stats["ngram_occurrence_count"] >= stats["unique_ngram_identities"]
    assert stats["collision_excess"] >= 1
    assert stats["collision_bucket_count"] >= 1
    assert stats["bucket_self_check_rows"] == len(docs)
    # Repeated identical terms create counts but do not count as identity collisions.
    repeated_only = ["cedar cedar cedar"]
    repeated_matrix = vectorizer.transform(repeated_only).tocsr()
    repeated_stats = hashing_diagnostics(repeated_only, repeated_matrix, vectorizer=vectorizer)
    assert repeated_stats["collision_excess"] == 0
    assert repeated_stats["ngram_occurrence_count"] > repeated_stats["unique_ngram_identities"]


def test_r04_fit_transform_is_deterministic_sparse_and_has_no_learned_vocabulary() -> None:
    fit_cycles = 47 + np.arange(132, dtype=np.int64) % 9
    fit_x = _synthetic_features(fit_cycles)
    y = ((fit_x["impact_load"].to_numpy(dtype=float) * 2 + fit_x["entry_friction"].to_numpy()) % 101).astype(float)

    model = R04HashingSparseHybrid(55).fit(fit_x, y)
    score_x = _scores_for_12_cycles(55, rows_per_cycle=2)
    raw = model.predict(score_x)
    state = model.audit_state()

    other = R04HashingSparseHybrid(55).fit(fit_x, y)
    raw_again = other.predict(score_x)
    state_again = other.audit_state()

    assert raw.shape == (len(score_x),)
    assert np.isfinite(raw).all()
    assert state["fit_rows"] == len(fit_x)
    assert state["fit_cycle_min"] == 47
    assert state["fit_cycle_max"] == 55
    assert state["training_design_shape"][0] == len(fit_x)
    assert state["training_design_shape"][1] >= HASHING_PARAMETERS["n_features"]
    assert state["training_design_nnz"] > 0
    assert state["sparse_design"] is True
    assert state["learned_vocabulary"] is False
    assert state["learned_idf"] is False
    assert "vocabulary_" not in vars(model._vectorizer)
    assert "idf_" not in vars(model._vectorizer)
    assert state["text_collision_diagnostics"]["matrix_nnz"] > 0
    assert state["text_collision_diagnostics"]["bucket_self_check_rows"] == min(128, len(fit_x))
    assert state["validation_data_passed_to_fit"] is False
    assert state["validation_labels_passed_to_fit"] is False
    assert state["test_data_used"] is False
    assert state["fitted_model_sha256"]
    assert state_again["text_collision_diagnostics"] == state["text_collision_diagnostics"]
    np.testing.assert_array_equal(raw_again, raw)


def test_rank50_uses_average_ties_within_each_cycle() -> None:
    raw = np.array([10.0, 10.0, 20.0, 5.0, 15.0], dtype=np.float64)
    cycles = np.array([56, 56, 56, 57, 57], dtype=np.int64)
    mapped = rank50_prediction(raw, cycles)
    expected = np.array([30.0, 30.0, 60.0, 27.5, 57.5], dtype=np.float64)
    np.testing.assert_array_equal(mapped, expected)


def test_fit_predict_reconstructs_baseline_and_candidate_policy() -> None:
    fit_cycles = 47 + np.arange(132, dtype=np.int64) % 9
    fit_x = _synthetic_features(fit_cycles)
    fit_y = ((fit_x["impact_load"].to_numpy(dtype=float) * 3 + fit_x["text_load"].to_numpy()) % 101).astype(float)
    score_x = _scores_for_12_cycles(55, rows_per_cycle=2)
    controls = _synthetic_controls(score_x)

    result = fit_predict(fit_x, fit_y, score_x, controls=controls, card_id="R04", cutoff=55)
    expected_candidate_rank = rank50_prediction(result["raw_candidate"], controls["cycle_code"])
    expected_candidate = np.clip(
        0.75 * controls["ridge25_prediction"] + 0.25 * expected_candidate_rank, 0.0, 100.0
    )
    expected_baseline = compose_r02_prediction(
        controls["ridge25_prediction"], controls["baseline_sparse_raw"], controls["cycle_code"]
    )

    assert result["fit_predictor_count"] == 1
    assert result["fit_metadata"]["validation_labels_passed_to_estimator"] is False
    assert result["fit_metadata"]["test_data_used"] is False
    assert result["state"]["parameters"] == SPARSE_LGB_PARAMETERS
    assert np.isfinite(result["prediction"]).all()
    assert np.all((result["prediction"] >= 0.0) & (result["prediction"] <= 100.0))
    np.testing.assert_array_equal(result["prediction"], expected_candidate)
    assert result["fit_metadata"]["baseline_reconstruction_max_abs_error"] <= 1e-10
    np.testing.assert_allclose(controls["baseline_prediction"], expected_baseline, rtol=0, atol=1e-10)
    np.testing.assert_array_equal(result["audit_arrays"]["candidate_rank50"], expected_candidate_rank)
    np.testing.assert_array_equal(result["audit_arrays"]["baseline_rank50"], rank50_prediction(
        controls["baseline_sparse_raw"], controls["cycle_code"]
    ))


def test_r04_rejects_bad_cutoffs_cycles_features_controls_and_rows() -> None:
    fit_cycles = 47 + np.arange(96, dtype=np.int64) % 9
    fit_x = _synthetic_features(fit_cycles)
    y = np.linspace(0.0, 100.0, len(fit_x))
    score_x = _scores_for_12_cycles(55, rows_per_cycle=1)
    controls = _synthetic_controls(score_x)

    with pytest.raises(ValueError, match="cutoffs 55/67"):
        fit_predict(fit_x, y, score_x, controls=controls, card_id="R04", cutoff=59)
    extra = score_x.assign(record_id=np.arange(len(score_x)))
    with pytest.raises(ValueError, match="exact structured"):
        R04HashingSparseHybrid(55)._select(extra)
    future_fit = fit_x.copy()
    future_fit.loc[future_fit.index[-1], "cycle_code"] = 56
    with pytest.raises(ValueError, match="temporal horizon"):
        fit_predict(future_fit, y, score_x, controls=controls, card_id="R04", cutoff=55)
    wrong_base = dict(controls)
    wrong_base["baseline_prediction"] = np.zeros(len(score_x), dtype=np.float64)
    with pytest.raises(ValueError, match="baseline does not reconstruct"):
        fit_predict(fit_x, y, score_x, controls=wrong_base, card_id="R04", cutoff=55)
    wrong_rows = dict(controls)
    wrong_rows["control_rows"] = controls["control_rows"][::-1].copy()
    with pytest.raises(ValueError, match="increasing unique"):
        fit_predict(fit_x, y, score_x, controls=wrong_rows, card_id="R04", cutoff=55)
    missing_cycle = score_x.loc[score_x["cycle_code"] != 60].copy()
    missing_controls = _synthetic_controls(missing_cycle)
    with pytest.raises(ValueError, match="all 12 cycles"):
        fit_predict(fit_x, y, missing_cycle, controls=missing_controls, card_id="R04", cutoff=55)
