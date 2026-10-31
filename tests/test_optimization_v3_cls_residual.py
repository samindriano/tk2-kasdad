"""Synthetic guards for the frozen classification OOF residual pilot."""
import numpy as np
import pandas as pd
import pytest

from src.models.optimization_v3_residual import OOFResidualHead
from src.models.round1_models import EXPANDED_FEATURES
from scripts.optimization_v3_cls_residual import training_input_hash, validate_oof_blocks


def synthetic_cycles():
    return np.repeat(np.arange(47, 72, dtype=int), 2)


def block(cycle, cutoff):
    return {"fit_cutoff": cutoff,
            "rows": np.flatnonzero((cycle >= cutoff + 1) & (cycle <= cutoff + 4))}


def test_oof_blocks_are_exact_next_four_cycles_and_disjoint():
    cycle = synthetic_cycles()
    blocks = [block(cycle, 51), block(cycle, 55)]
    rows = validate_oof_blocks(blocks, cycle, 59)
    np.testing.assert_array_equal(rows, np.flatnonzero((cycle >= 52) & (cycle <= 59)))
    assert len(np.unique(rows)) == len(rows)
    shuffled_cycle = cycle[np.random.default_rng(5).permutation(len(cycle))]
    shuffled_rows = validate_oof_blocks([block(shuffled_cycle, 51), block(shuffled_cycle, 55)], shuffled_cycle, 59)
    np.testing.assert_array_equal(np.sort(shuffled_rows), np.flatnonzero((shuffled_cycle >= 52) & (shuffled_cycle <= 59)))


def test_oof_duplicate_or_missing_rows_fail_closed():
    cycle = synthetic_cycles()
    duplicate = [block(cycle, 51), block(cycle, 55)]
    duplicate[1]["rows"] = np.concatenate([duplicate[1]["rows"], duplicate[0]["rows"][:1]])
    with pytest.raises(ValueError, match="exactly its next four cycles"):
        validate_oof_blocks(duplicate, cycle, 59)
    incomplete = [block(cycle, 51), block(cycle, 55)]
    incomplete[1]["rows"] = incomplete[1]["rows"][1:]
    with pytest.raises(ValueError, match="exactly its next four cycles"):
        validate_oof_blocks(incomplete, cycle, 59)


def test_oof_rows_cannot_cross_outer_correction_cutoff():
    cycle = synthetic_cycles()
    with pytest.raises(ValueError, match="exact frozen sequence"):
        validate_oof_blocks([block(cycle, 51), block(cycle, 55), block(cycle, 59)], cycle, 59)
    with pytest.raises(ValueError, match="reaches beyond"):
        validate_oof_blocks([block(cycle, 51), block(cycle, 55), block(cycle, 59)], cycle, 61)


def test_raw_training_fingerprint_tracks_row_feature_label_and_score_inputs():
    X = pd.DataFrame({"impact_load": [1.0, 2.0], "entry_friction": [0, 1]})
    rows = np.array([4, 7], dtype=np.int64)
    y = np.array([0, 1], dtype=np.int8)
    scores = np.array([[0.2], [0.8]], dtype=float)
    baseline = training_input_hash(rows, X, y, scores)
    assert baseline != training_input_hash(rows[::-1], X.iloc[::-1].reset_index(drop=True), y[::-1], scores[::-1])
    assert baseline != training_input_hash(rows, X, 1 - y, scores)
    assert baseline != training_input_hash(rows, X, y, 1 - scores)


def test_residual_head_state_and_fixed_probability_boundary_are_auditable():
    rng = np.random.default_rng(17)
    n = 320
    X = pd.DataFrame({
        "impact_load": rng.normal(90, 20, n),
        "entry_friction": rng.integers(0, 2, n),
        "cycle_code": np.tile(np.arange(52, 68), 20),
        "scope_pressure": rng.normal(0, 1, n),
        "text_load": rng.integers(1, 80, n),
        "scope_trace": [f"{int(v)} modules" for v in rng.integers(1, 30, n)],
        "org_spread": [f"{int(v)} orgs" for v in rng.integers(1, 15, n)],
        "module_spread": [f"{int(v)} modules" for v in rng.integers(1, 20, n)],
        "telemetry_flag": rng.choice(["active", "inactive"], n),
        "org_code": rng.choice(["org-a", "org-b", "org-c"], n),
        "module_code": rng.choice(["m-a", "m-b", "m-c", "m-d"], n),
        "pattern_code": rng.choice(["p-a", "p-b", "p-c"], n),
    }, columns=list(EXPANDED_FEATURES))
    y = ((X["impact_load"].to_numpy() + 0.6 * X["text_load"].to_numpy() + rng.normal(0, 10, n)) > 110).astype(np.int8)
    scores = np.clip(0.12 + 0.75 * ((y * 0.7) + rng.random(n) * 0.3), 1e-3, 1 - 1e-3)[:, None]
    head = OOFResidualHead("classification", ["C2_B"]).fit(X, scores, y)
    state = head.audit_state()
    pred = head.predict(X.iloc[:40].copy(), scores[:40])
    assert state["rows"] == n
    assert state["score_names"] == ["C2_B"]
    for key in ("fit_data_sha256", "fit_scores_sha256", "target_sha256", "booster_sha256"):
        assert isinstance(state[key], str) and len(state[key]) == 64
    assert head.shrinkage == 0.5
    assert np.isfinite(pred).all()
    assert ((pred >= 0) & (pred <= 1)).all()
