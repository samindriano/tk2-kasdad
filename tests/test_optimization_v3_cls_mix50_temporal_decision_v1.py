from __future__ import annotations

import numpy as np
import pytest
from sklearn.metrics import f1_score

from scripts.optimization_v3_cls_mix50_temporal_decision_v1 import (
    _counts_for_rule,
    _macro_f1_from_counts,
    _synthetic_end_to_end,
    _validate_oof_boundaries,
    evaluate,
    select_threshold,
    threshold_candidates,
)


def _supported_case():
    p = np.tile(np.asarray([0.0, 0.0, 0.2, 0.5, 0.5, 0.8, 1.0]), 200)
    y = np.tile(np.asarray([0, 1, 0, 1, 0, 0, 1], dtype=np.int8), 200)
    missing = np.tile(np.asarray([False, False, True, False, False, False, True]), 200)
    return p, y, missing


def test_threshold_candidates_preserve_ties_and_extreme_allocations():
    p = np.asarray([0.0, 0.0, 0.5, 0.5, 1.0])
    candidates = threshold_candidates(p)
    assert 0.0 in candidates
    assert 0.5 in candidates
    assert float(np.nextafter(1.0, np.inf)) in candidates
    for threshold in candidates:
        labels = (p >= threshold).astype(np.int8)
        assert np.all(labels[p == 0.0] == labels[p == 0.0][0])
        assert np.all(labels[p == 0.5] == labels[p == 0.5][0])


def test_adjacent_floats_use_upper_score_as_safe_cutpoint():
    low = np.float64(0.5)
    high = np.nextafter(low, np.inf)
    candidates = threshold_candidates(np.asarray([low, high]))
    assert high in candidates
    assert (np.asarray([low, high]) >= high).tolist() == [False, True]


def test_selector_matches_bruteforce_and_sklearn_macro_f1():
    p, y, missing = _supported_case()
    result = select_threshold(p, missing, y)
    assert result["status"] == "SELECTED"
    assert result["selection_fit_count"] == 1
    candidate_scores = []
    for threshold in threshold_candidates(p[~missing]):
        counts = _counts_for_rule(y, p, missing, float(threshold))
        own = _macro_f1_from_counts(counts)
        pred = (p >= 0.5).astype(np.int8)
        pred[~missing] = (p[~missing] >= threshold).astype(np.int8)
        sklearn_score = f1_score(y, pred, labels=[0, 1], average="macro", zero_division=0)
        assert own == pytest.approx(sklearn_score, abs=1e-15)
        candidate_scores.append((float(threshold), own))
    maximum = max(score for _, score in candidate_scores)
    tied = [cut for cut, score in candidate_scores if maximum - score <= 1e-12]
    expected = min(tied, key=lambda cut: (abs(cut - 0.5), cut))
    assert result["threshold"] == expected
    assert result["selected_macro_f1"] == pytest.approx(maximum, abs=1e-15)


def test_missing_branch_is_fixed_at_half_independent_of_selected_threshold():
    p, y, missing = _supported_case()
    chosen = evaluate(y, p, missing, np.repeat(68, len(y)), 0.9)
    fixed = evaluate(y, p, missing, np.repeat(68, len(y)), 0.5)
    assert chosen["overall"]["rows"] == len(y)
    # Missing branch confusion is invariant to the learned present threshold.
    assert chosen["availability"]["missing"]["confusion_matrix"] == fixed["availability"]["missing"]["confusion_matrix"]
    # Present branch can change when its threshold changes.
    assert chosen["availability"]["present"]["confusion_matrix"] != fixed["availability"]["present"]["confusion_matrix"]


def test_insufficient_support_falls_back_without_selection():
    result = select_threshold(np.asarray([0.1, 0.9]), np.asarray([False, False]), np.asarray([0, 1]))
    assert result["status"] == "FIXED_0_50_INSUFFICIENT_SUPPORT"
    assert result["threshold"] == 0.5
    assert result["selection_fit_count"] == 0


def test_oof_origin_boundary_rejects_wrong_or_duplicate_w6_ids():
    e67_cycles = np.repeat(np.arange(68, 80, dtype=np.int16), 2)
    e67_ids = np.asarray([f"id-{cycle}-{i}" for cycle in range(68, 80) for i in range(2)], dtype="U16")
    e75_cycles = np.repeat(np.arange(76, 80, dtype=np.int16), 2)
    e75_ids = np.asarray([f"id-{cycle}-{i}" for cycle in range(76, 80) for i in range(2)], dtype="U16")
    def origin(ids, cycles):
        n = len(ids)
        return {"ids": ids, "cycles": cycles, "rows": np.arange(n), "pP": np.zeros(n),
                "pA": np.zeros(n), "MIX50": np.zeros(n), "missing": np.zeros(n, dtype=bool),
                "group_availability": np.full(n, "present", dtype="U7"),
                "group_text_load": np.full(n, "Q1", dtype="U2"),
                "group_entry_friction": np.full(n, "0", dtype="U1")}
    oof = {"67": origin(e67_ids, e67_cycles), "75": origin(e75_ids, e75_cycles)}
    _validate_oof_boundaries(oof)
    bad = {"67": oof["67"], "75": dict(oof["75"])}
    bad["75"]["ids"] = bad["75"]["ids"].copy()
    bad["75"]["ids"][0] = "duplicate-id"
    with pytest.raises(RuntimeError, match="exact same unique Train IDs"):
        _validate_oof_boundaries(bad)


def test_synthetic_end_to_end_checks_temporal_persistence_and_resume(tmp_path):
    result = _synthetic_end_to_end(tmp_path)
    assert result["status"] == "PASS"
    assert result["selection_stages"] == 3
    assert result["reserved_slots"] == 3
    assert result["estimator_fits"] == 0
    assert result["evaluation_targets_requested_after_receipt"] is True
    assert result["e79_probability_loader_guard"] == "PASS"
    assert result["persistence_resume"].startswith("PASS")
    assert result["budget_cap_and_no_retry"] == "PASS"
    assert result["all_cutoff_metrics_match_bruteforce"] is True
