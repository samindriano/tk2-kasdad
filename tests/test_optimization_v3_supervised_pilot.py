import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.optimization_v3_supervised_pilot import (
    _synthetic_frame,
    classification_fit_mask,
    classification_routes,
    rank50_within_cycle,
    regression_candidates,
    token_boundary_preflight,
)
from src.models.optimization_v3_token_stats import OrderedTokenStats


ROOT = Path(__file__).resolve().parents[1]


def test_ordered_token_statistics_match_manual_past_only_sample_and_perturbation():
    texts = ["shared alpha", "shared beta", "shared alpha", "shared alpha",
             "shared alpha", "shared beta", "shared alpha", "shared alpha",
             "shared alpha", "shared alpha", "shared beta", "shared alpha"]
    cycles = np.repeat([47, 48, 49], 4)
    target = np.array([0., 1., 0., 1., 1., 1., 0., 0., 0., 1., 0., 1.])
    original_encoder = OrderedTokenStats()
    original = original_encoder.fit_transform(texts, cycles, target)

    changed_target = target.copy()
    changed_target[4:8] = 1.0 - changed_target[4:8]
    changed_encoder = OrderedTokenStats()
    changed = changed_encoder.fit_transform(texts, cycles, changed_target)

    # Cycle 47 has no prior labels; cycle 48 must not depend on its own labels.
    np.testing.assert_array_equal(original[:8], changed[:8])
    # Cycle 49 can use cycle 48 labels for tokens with previous document support.
    assert not np.array_equal(original[8:], changed[8:])
    state = original_encoder.audit_state()
    for item in state["lineage"]:
        assert item["labels_previously_seen"] == int(np.sum(cycles < item["cycle"]))


def test_classification_fit_population_is_present_only_and_cutoff_bounded():
    frame, _, _ = _synthetic_frame(120)
    mask = classification_fit_mask(frame, 50)
    cycles = frame.cycle_code.to_numpy(dtype=int)
    missing = frame.signal_stream.isna() | frame.signal_stream.fillna("").str.strip().eq("")
    assert mask.dtype == bool
    assert np.all((cycles[mask] >= 47) & (cycles[mask] <= 50))
    assert not missing.to_numpy()[mask].any()
    assert not mask[cycles > 50].any()


def test_classification_routes_preserve_missing_expert_and_fixed_blend():
    missing = np.array([False, True, False, True])
    raw = np.array([0.2, 0.0, 0.7, 0.0])
    c2b = np.array([0.4, 0.1, 0.6, 0.2])
    latest = np.array([0.35, 0.1, 0.8, 0.2])
    cb = np.array([0.5, 0.4, 0.6, 0.3])
    routes = classification_routes(missing, raw, c2b, latest, cb)
    np.testing.assert_array_equal(routes["blend25"][missing], cb[missing])
    np.testing.assert_array_equal(routes["direct"][missing], cb[missing])
    np.testing.assert_allclose(routes["blend25"][~missing], 0.75 * latest[~missing] + 0.25 * raw[~missing])
    np.testing.assert_array_equal(routes["initial_control"][~missing], c2b[~missing])


def test_regression_rank_and_blend_are_cycle_local_and_clipped():
    raw = np.array([-2.0, 50.0, 50.0, 105.0, 10.0, 20.0])
    cycles = np.array([60, 60, 60, 60, 61, 61])
    champion = np.array([10.0, 20.0, 30.0, 40.0, 95.0, 5.0])
    rank = rank50_within_cycle(raw, cycles)
    # For cycle 60, tied values receive the same average-rank percentile.
    assert rank[1] == rank[2]
    output = regression_candidates(champion, raw, cycles)
    np.testing.assert_array_equal(output["rank50"], rank)
    np.testing.assert_allclose(output["blend25"], np.clip(0.75 * champion + 0.25 * rank, 0, 100))
    assert np.all((output["rank50"] >= 0) & (output["rank50"] <= 100))
    assert np.all((output["blend25"] >= 0) & (output["blend25"] <= 100))


def test_protocol_declares_present_only_class_fit_and_all_row_regression():
    protocol_path = ROOT / "experiments/optimization_v3/supervised_pilot/protocol.json"
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    assert "present-stream rows only" in protocol["classification_policies"]["fit_population"].lower()
    assert "All official Regression Train rows" in protocol["regression_policies"]["fit_population"]
    assert "jointly reflect representation and fit-population differences" in protocol["comparison_interpretation"]


def test_boundary_preflight_reports_pass():
    receipt = token_boundary_preflight()
    assert receipt["status"] == "PASS"
    assert receipt["same_cycle_label_perturbation_leaves_current_and_past_features_unchanged"]

