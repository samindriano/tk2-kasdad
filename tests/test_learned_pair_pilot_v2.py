from __future__ import annotations

import numpy as np
import pytest
from scipy import sparse

from src.models.learned_pair_pilot_v1 import (
    fm_pair_score,
    fm_pair_score_and_v_gradient,
    initialize_parameters,
    objective_gradient_mean,
    objective_gradient_sum,
    score_logits,
    unpack_parameters,
)


def test_fm_identity_and_weighted_factor_gradient_match_pair_enumeration() -> None:
    q = sparse.csr_matrix([[1.0, 2.0, 0.0], [0.5, 0.0, 1.5]])
    v = np.array([[0.2, -0.4], [0.3, 0.1], [-0.2, 0.5]])
    weights = np.array([0.7, -0.3])
    score = fm_pair_score(q, v)
    explicit = []
    for row in q.toarray():
        total = 0.0
        for j in range(len(row)):
            for k in range(j + 1, len(row)):
                total += row[j] * row[k] * float(v[j] @ v[k])
        explicit.append(total)
    assert np.allclose(score, explicit, rtol=0, atol=1e-14)
    score2, grad = fm_pair_score_and_v_gradient(q, v, weights)
    assert np.array_equal(score, score2)
    flat = v.reshape(-1)
    numeric = np.zeros_like(flat)
    eps = 1e-6
    for i in range(flat.size):
        plus, minus = flat.copy(), flat.copy()
        plus[i] += eps
        minus[i] -= eps
        numeric[i] = (weights @ fm_pair_score(q, plus.reshape(v.shape))
                      - weights @ fm_pair_score(q, minus.reshape(v.shape))) / (2 * eps)
    assert np.allclose(grad.reshape(-1), numeric, rtol=1e-6, atol=1e-8)


def test_sum_objective_chunk_gradient_matches_finite_difference_all_blocks() -> None:
    rng = np.random.default_rng(123)
    x = sparse.csr_matrix(rng.uniform(0.0, 1.4, (13, 5)) * (rng.random((13, 5)) < 0.55))
    y = np.array([0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 1, 0, 1], dtype=np.int8)
    theta = initialize_parameters(5, 2, seed=42)
    theta[:5] = rng.normal(0, 0.15, 5)
    theta[5] = 0.2
    value, grad = objective_gradient_sum(theta, x, y, rank=2, chunk_rows=4)
    numeric = np.zeros_like(theta)
    eps = 1e-6
    for i in range(theta.size):
        plus, minus = theta.copy(), theta.copy()
        plus[i] += eps
        minus[i] -= eps
        fp = objective_gradient_sum(plus, x, y, rank=2, chunk_rows=4)[0]
        fm = objective_gradient_sum(minus, x, y, rank=2, chunk_rows=4)[0]
        numeric[i] = (fp - fm) / (2 * eps)
    assert np.isfinite(value)
    assert np.allclose(grad, numeric, rtol=2e-5, atol=3e-6)


@pytest.mark.parametrize("with_factors", [False, True])
def test_sum_mean_and_chunk_full_equalities_including_uneven_chunk(with_factors: bool) -> None:
    rng = np.random.default_rng(72)
    x = sparse.csr_matrix(rng.normal(size=(17, 6)) * (rng.random((17, 6)) < 0.4))
    y = rng.integers(0, 2, size=17, dtype=np.int8)
    theta = initialize_parameters(6, 3, seed=42, with_factors=with_factors)
    theta += rng.normal(0, 0.015, theta.size)
    full = objective_gradient_sum(theta, x, y, rank=3, with_factors=with_factors, chunk_rows=17)
    uneven = objective_gradient_sum(theta, x, y, rank=3, with_factors=with_factors, chunk_rows=5)
    mean = objective_gradient_mean(theta, x, y, rank=3, with_factors=with_factors, chunk_rows=4)
    assert np.allclose(full[0], uneven[0], rtol=1e-13, atol=1e-13)
    assert np.allclose(full[1], uneven[1], rtol=1e-12, atol=1e-12)
    assert np.allclose(full[0], 17 * mean[0], rtol=1e-13, atol=1e-13)
    assert np.allclose(full[1], 17 * mean[1], rtol=1e-12, atol=1e-12)


def test_signed_logaddexp_objective_stays_finite_at_extreme_logits() -> None:
    x = sparse.csr_matrix([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    y = np.array([0, 1, 0], dtype=np.int8)
    theta = np.array([1200.0, -1200.0, 0.0])
    value, grad = objective_gradient_sum(theta, x, y, rank=2, with_factors=False, chunk_rows=2)
    assert np.isfinite(value)
    assert np.isfinite(grad).all()


def test_single_active_feature_has_zero_pair_score_and_factor_gradient() -> None:
    q = sparse.csr_matrix([[1.0, 0.0, 0.0], [0.0, -2.0, 0.0]])
    v = np.arange(6, dtype=np.float64).reshape(3, 2) / 10
    score, grad = fm_pair_score_and_v_gradient(q, v, np.array([1.0, -0.5]))
    assert np.array_equal(score, np.zeros(2))
    assert np.array_equal(grad, np.zeros_like(v))


def test_class_order_and_parameter_blocks_are_explicit() -> None:
    candidate = initialize_parameters(4, 2, seed=42)
    w, b, v = unpack_parameters(candidate, 4, 2)
    assert np.array_equal(w, np.zeros(4))
    assert b == 0.0
    assert v is not None and v.shape == (4, 2) and np.linalg.norm(v) > 0
    assert np.array_equal(candidate, initialize_parameters(4, 2, seed=42))
    control = initialize_parameters(4, 2, seed=42, with_factors=False)
    assert control.shape == (5,)
    assert unpack_parameters(control, 4, 2, with_factors=False)[2] is None


def test_optimizer_moves_nonzero_factor_block_on_controlled_toy() -> None:
    from scipy.optimize import minimize
    from scipy.special import expit

    rng = np.random.default_rng(29)
    dense = rng.normal(size=(48, 5))
    dense[np.abs(dense) < 0.6] = 0.0
    x = sparse.csr_matrix(dense)
    truth_w = np.array([0.25, -0.2, 0.1, 0.0, 0.18])
    truth_v = np.random.default_rng(5).normal(0, 0.2, (5, 2))
    from src.models.learned_pair_pilot_v1 import row_l2_normalize
    logits = np.asarray(x @ truth_w).reshape(-1) + fm_pair_score(row_l2_normalize(x), truth_v)
    y = (expit(logits) > 0.52).astype(np.int8)
    initial = initialize_parameters(5, 2, seed=42)
    initial_v = initial[6:].copy()
    result = minimize(lambda p: objective_gradient_sum(p, x, y, rank=2, chunk_rows=13),
                      initial, method="L-BFGS-B", jac=True,
                      options={"maxiter": 500, "maxfun": 2000, "maxcor": 10,
                               "maxls": 50, "ftol": 1e-9, "gtol": len(y) * 1e-4})
    assert result.success, result.message
    assert np.isfinite(result.fun) and np.isfinite(result.jac).all()
    assert np.linalg.norm(result.x[6:] - initial_v) > 1e-7
    assert result.fun <= objective_gradient_sum(initial, x, y, rank=2, chunk_rows=13)[0] + 1e-8
    assert np.isfinite(score_logits(result.x, x, rank=2)[0]).all()


def test_synthetic_lifecycle_persistence_independent_verifier_and_resume(tmp_path) -> None:
    from scripts.optimization_v3_learned_pair_pilot_v2 import run_synthetic_lifecycle

    report = run_synthetic_lifecycle(tmp_path / "learned-pair-synthetic")
    assert report["status"] == "PASS_SYNTHETIC_END_TO_END"
    assert report["official_fit_calls"] == 0
    assert report["synthetic_optimizer_fit_calls"] == 1
    assert report["optimizer_reruns_on_resume"] == 0
    assert report["tampered_matrix_rejected"] is True
    assert report["production_writer_blocked"] is True
    assert report["cached_bag_route_formula_pass"] is True
    assert report["raw_routed_bag_swap_rejected"] is True
    assert report["terminal_failure_consumed_started_id"] == "LEARNED_PAIR_PILOT_V2_E67_FM"
    assert report["unstarted_ids_released_once"][0] == "LEARNED_PAIR_PILOT_V2_E67_NOPAIR"
    assert len(report["unstarted_ids_released_once"]) == 3
    assert report["release_idempotent"] is True
    assert report["live_budget_ledger_sha256_before"] == report["live_budget_ledger_sha256_after"]
    assert report["independent_verification"]["status"] == "PASS"


def test_failed_started_fit_releases_only_unstarted_ids() -> None:
    from scripts.optimization_v3_learned_pair_pilot_v2 import FIT_IDS, _unstarted_fit_ids

    assert _unstarted_fit_ids(FIT_IDS, {FIT_IDS[0]}) == list(FIT_IDS[1:])
    assert _unstarted_fit_ids(FIT_IDS, set(FIT_IDS)) == []


def test_new_pilot_gates_use_e67_h12_e75_w6_windows_and_supported_groups() -> None:
    from scripts.optimization_v3_learned_pair_pilot_v2 import _gates_for_origin

    views = []
    for view, score in {
        "H12_E67": {"fm": 0.8000, "bag": 0.7994, "nopair": 0.7997, "a02_observed": 0.7994},
        "W4": {"fm": 0.7950, "bag": 0.8000, "nopair": 0.7980, "a02_observed": 0.7980},
        "W5": {"fm": 0.7980, "bag": 0.8000, "nopair": 0.7990, "a02_observed": 0.7990},
        "W6": {"fm": 0.8010, "bag": 0.8000, "nopair": 0.8005, "a02_observed": 0.8000},
    }.items():
        for arm, f1 in score.items():
            views.append({"view": view, "arm": arm, "macro_f1": f1})
    groups = []
    for arm, f1, recall in (("fm", 0.75, 0.62), ("bag", 0.74, 0.60),
                            ("nopair", 0.745, 0.61), ("a02_observed", 0.74, 0.60)):
        groups.append({"view": "H12_E67", "group": "availability", "group_value": "present",
                       "arm": arm, "n": 200, "positive": 30, "negative": 170,
                       "macro_f1": f1, "positive_recall": recall})
    e67 = _gates_for_origin(67, views, groups)
    assert all(gate["pass"] for gate in e67)
    required = {gate["gate"] for gate in e67}
    assert "E67_H12_gain_min_0.0005_vs_bag" in required
    assert "E67_H12_gain_min_0.0002_vs_nopair" in required
    assert "E67_H12_gain_min_0.0005_vs_a02_observed" in required
    assert any("H12_E67_availability_present_macro_f1_harm_vs_nopair" in x for x in required)
    e75 = _gates_for_origin(75, [
        {"view": "W6", "arm": "fm", "macro_f1": 0.80},
        {"view": "W6", "arm": "bag", "macro_f1": 0.79},
        {"view": "W6", "arm": "nopair", "macro_f1": 0.80},
        {"view": "W6", "arm": "a02_observed", "macro_f1": 0.799},
    ], [])
    assert all(gate["pass"] for gate in e75)
    assert len(e75) == 3
    assert {gate["track"] for gate in e67 if gate["pass"] is not None} == {"mechanism", "competition", "both"}


def test_unsupported_subgroup_is_unknown_and_does_not_block_numeric_gates() -> None:
    from scripts.optimization_v3_learned_pair_pilot_v2 import _gates_for_origin

    views = []
    for view, scores in {
        "H12_E67": {"fm": 0.8010, "bag": 0.8000, "nopair": 0.8005, "a02_observed": 0.8000},
        "W4": {"fm": 0.8000, "bag": 0.7990, "nopair": 0.7990, "a02_observed": 0.7990},
        "W5": {"fm": 0.8000, "bag": 0.7990, "nopair": 0.7990, "a02_observed": 0.7990},
        "W6": {"fm": 0.8000, "bag": 0.7990, "nopair": 0.7990, "a02_observed": 0.7990},
    }.items():
        for arm, macro_f1 in scores.items():
            views.append({"view": view, "arm": arm, "macro_f1": macro_f1})
    unsupported = [{"view": "H12_E67", "group": "text_load", "group_value": "Q4",
                    "arm": arm, "n": 199, "positive": 99, "negative": 100,
                    "macro_f1": 0.1, "positive_recall": 0.1}
                   for arm in ("fm", "bag", "nopair", "a02_observed")]
    gates = _gates_for_origin(67, views, unsupported)
    support_gate = next(gate for gate in gates if gate["gate"].endswith("_support"))
    assert support_gate["pass"] is None
    assert support_gate["status"] == "UNKNOWN_UNSUPPORTED"
    evaluable = [gate for gate in gates if gate["pass"] is not None]
    assert evaluable and all(gate["pass"] is True for gate in evaluable)


def test_audit_receipt_is_create_once_and_rejects_identity_drift(tmp_path) -> None:
    from scripts.optimization_v3_learned_pair_pilot_v2 import (
        LearnedPairPilotError, _create_once_receipt,
    )

    path = tmp_path / "access.json"
    first = _create_once_receipt(path, {"created": "t1", "input": "abc"}, ("input",))
    resumed = _create_once_receipt(path, {"created": "t2", "input": "abc"}, ("input",))
    assert resumed == first
    assert resumed["created"] == "t1"
    with pytest.raises(LearnedPairPilotError):
        _create_once_receipt(path, {"created": "t3", "input": "changed"}, ("input",))


def test_cached_bag_baseline_preserves_raw_present_and_hgb_missing_route() -> None:
    from scripts.optimization_v3_learned_pair_pilot_v2 import (
        LearnedPairPilotError, _validate_cached_bag_route,
    )

    present = np.array([True, False, True, False])
    raw = np.array([0.2, 0.1, 0.8, 0.9])
    fallback = np.array([0.6, 0.7, 0.4, 0.3])
    routed = np.array([0.2, 0.7, 0.8, 0.3])
    validated = _validate_cached_bag_route(raw, routed, present, fallback)
    assert np.array_equal(validated, routed)
    assert not np.array_equal(raw[~present], fallback[~present])
    with pytest.raises(LearnedPairPilotError):
        _validate_cached_bag_route(routed, raw, present, fallback)


def test_v2_gtol_and_independent_mean_gradient_gate() -> None:
    from scripts.optimization_v3_learned_pair_pilot_v2 import (
        _optimizer_convergence_gate, _solver_options,
    )

    assert _solver_options(29_705)["gtol"] == pytest.approx(2.9705)
    assert _solver_options(42_117)["gtol"] == pytest.approx(4.2117)
    passed = _optimizer_convergence_gate(
        solver_success=True, finite_state=True, initial_loss=10, final_loss=9,
        gradient_inf_sum=1.0, n_fit=10_000)
    assert passed["optimizer_converged"] is True
    assert passed["normalized_gradient_inf_norm"] == pytest.approx(1e-4)
    failed_stationarity = _optimizer_convergence_gate(
        solver_success=True, finite_state=True, initial_loss=10, final_loss=9,
        gradient_inf_sum=1.0001, n_fit=10_000)
    assert failed_stationarity["optimizer_converged"] is False
    assert failed_stationarity["stationarity_guard"] is False
    failed_scipy = _optimizer_convergence_gate(
        solver_success=False, finite_state=True, initial_loss=10, final_loss=1,
        gradient_inf_sum=0.01, n_fit=10_000)
    assert failed_scipy["optimizer_converged"] is False


def test_v2_pre_amend_accounting_preserves_historical_releases() -> None:
    from scripts.optimization_v3_learned_pair_pilot_v2 import _expected_pre_amend_snapshot

    expected = _expected_pre_amend_snapshot({"event_rows": 281})
    assert expected["global_cap"] == 113
    assert expected["effective_category_caps"] == {
        "continuation": 38, "control_replay": 2, "pilot": 72, "technical_retry": 1}
    assert expected["active_reservations_by_category"] == {
        "continuation": 38, "control_replay": 2, "pilot": 69, "technical_retry": 1}
    assert (expected["active_reservations"], expected["fit_starts"], expected["fit_saved"],
            expected["fit_failed"], expected["fit_pending"]) == (110, 110, 106, 4, 0)


def test_a02_exact_origin_reference_is_required_and_row_bound(tmp_path, monkeypatch) -> None:
    import hashlib
    import json
    from scripts import optimization_v3_learned_pair_pilot_v2 as pilot

    origin = 67
    ref_root = tmp_path / "a02"
    folder = ref_root / f"E{origin}" / "observed"
    folder.mkdir(parents=True)
    monkeypatch.setattr(pilot, "ROOT", tmp_path)
    monkeypatch.setattr(pilot, "A02_REFERENCE_ROOT", ref_root)
    freeze_path = tmp_path / "a02_freeze.json"
    freeze_path.write_text(json.dumps({"freeze_signature": "frozen-a02"}), encoding="utf-8")
    monkeypatch.setattr(pilot, "A02_FREEZE", freeze_path)
    fit_rows = np.array([0, 1, 2], dtype=np.int64)
    score_rows = np.array([3, 4], dtype=np.int64)
    ids = np.array(["r3", "r4"])
    cycles = np.array([68, 69], dtype=np.int64)
    probability = np.array([0.25, 0.75])
    state = {"card_id": "CLASSIFICATION_A02_NEIGHBOR_RESIDUAL",
             "origin_cycle": origin, "residual_arm": "observed"}
    (folder / "fit_state.json").write_text(json.dumps(state), encoding="utf-8")
    model_bytes = b"synthetic model receipt bytes"
    (folder / "model.joblib").write_bytes(model_bytes)
    receipt = {
        "fit_id": "A02_E67_OBSERVED", "origin_cycle": origin, "residual_arm": "observed",
        "freeze_signature": "frozen-a02", "model_sha256": hashlib.sha256(model_bytes).hexdigest(),
        "state_sha256": pilot._canonical_sha256(state), "query_labels_passed_to_fit": False,
        "test_data_used": False, "task_estimator_fit_count": 1,
        "text_prefix_rows": len(fit_rows),
        "text_prefix_row_positions_sha256": hashlib.sha256(fit_rows.astype("<i8").tobytes()).hexdigest(),
    }
    receipt["receipt_sha256"] = pilot._canonical_sha256(receipt)
    (folder / "fit_receipt.json").write_text(json.dumps(receipt), encoding="utf-8")
    archive_path = folder / "predictions.npz"
    np.savez_compressed(archive_path, score_rows=score_rows, score_record_ids=ids,
                        score_cycles=cycles, prediction=probability,
                        labels=np.array([1, 0], dtype=np.int8))
    sidecar = {
        "fit_id": receipt["fit_id"], "fit_receipt_sha256": receipt["receipt_sha256"],
        "origin_cycle": origin, "residual_arm": "observed", "freeze_signature": "frozen-a02",
        "npz_sha256": pilot._sha256_file(archive_path), "score_rows": len(score_rows),
        "score_row_positions_sha256": hashlib.sha256(score_rows.astype("<i8").tobytes()).hexdigest(),
        "score_record_ids_sha256": hashlib.sha256(json.dumps(ids.tolist(), separators=(",", ":"),
                                                           ensure_ascii=False).encode()).hexdigest(),
        "query_labels_passed_to_prediction": False,
    }
    (folder / "predictions.json").write_text(json.dumps(sidecar), encoding="utf-8")
    verified = pilot._verify_a02_observed(origin, fit_rows, score_rows, ids, cycles)
    assert verified["status"] == "ALIGNED_VERIFIED_REFERENCE_ONLY"
    with pytest.raises(pilot.LearnedPairPilotError):
        pilot._verify_a02_observed(origin, fit_rows, score_rows, ids[::-1], cycles)
    with pytest.raises(pilot.LearnedPairPilotError):
        pilot._verify_a02_observed(75, fit_rows, score_rows, ids, cycles)


def test_mechanism_and_competition_gates_are_separate() -> None:
    from scripts.optimization_v3_learned_pair_pilot_v2 import _gates_for_origin
    from scripts.verify_optimization_v3_learned_pair_pilot_v2 import _gates_independent

    views = []
    for view, scores in {
        "H12_E67": {"fm": 0.8010, "bag": 0.8000, "nopair": 0.8005, "a02_observed": 0.8010},
        "W4": {"fm": 0.8000, "bag": 0.7990, "nopair": 0.7990, "a02_observed": 0.7990},
        "W5": {"fm": 0.8000, "bag": 0.7990, "nopair": 0.7990, "a02_observed": 0.7990},
        "W6": {"fm": 0.8000, "bag": 0.7990, "nopair": 0.7990, "a02_observed": 0.7990},
    }.items():
        views.extend({"view": view, "arm": arm, "macro_f1": score} for arm, score in scores.items())
    gates = _gates_for_origin(67, views, [])
    independent = _gates_independent(67, views, [])
    assert gates == independent
    assert all(g["pass"] for g in gates if g["track"] == "mechanism")
    assert not next(g for g in gates if g["gate"] == "E67_H12_gain_min_0.0005_vs_a02_observed")["pass"]
