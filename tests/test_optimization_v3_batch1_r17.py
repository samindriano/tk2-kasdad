from __future__ import annotations

from functools import partial
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from scripts import optimization_v3_batch1_r17 as r17runner
from scripts.optimization_v3_batch1_verify import verify_report
from src.models.optimization_v3_batch1 import make_freeze_identity, run_pilot
from src.models import optimization_v3_batch1_r17 as r17
from src.models.optimization_v3_batch1 import BatchIntegrityError


LATE_VIEWS = {
    "W4": [(67, 68, 71)],
    "W6": [(75, 76, 79)],
    "primary": [(67, 68, 71), (75, 76, 79)],
    "H12_L": [(67, 68, 79)],
}


def _synthetic_frame(per_cycle: int = 16) -> tuple[pd.DataFrame, np.ndarray]:
    cycles = np.repeat(np.arange(47, 80, dtype=np.int64), per_cycle)
    rows = np.arange(len(cycles), dtype=np.int64)
    frame = pd.DataFrame({
        "record_id": [f"r17-synthetic-{row}" for row in rows],
        "cycle_code": cycles,
        "entry_friction": (rows % 2).astype(float),
        "signal_stream": ["" if row % 5 == 0 else f"token{row % 19}" for row in rows],
        "text_load": (rows % 13).astype(float),
    })
    target = np.clip(12.0 + 0.7 * cycles + 0.2 * (rows % 7), 0.0, 100.0)
    return frame, target


def _source_cutoff_by_cycle() -> dict[int, int]:
    return {
        cycle: cutoff
        for cutoff, low in ((55, 56), (59, 60), (63, 64), (67, 68), (71, 72), (75, 76))
        for cycle in range(low, low + 4)
    }


def _history(frame: pd.DataFrame, prediction: np.ndarray, cutoff: int) -> dict[str, np.ndarray]:
    cycles = frame["cycle_code"].to_numpy(dtype=np.int64)
    rows = np.flatnonzero((cycles >= 56) & (cycles <= cutoff))
    return {
        "rows": rows.astype(np.int64),
        "cycles": cycles[rows],
        "prediction": np.asarray(prediction, dtype=np.float64)[rows],
    }


def _controls(frame: pd.DataFrame, rows: np.ndarray) -> dict[str, np.ndarray]:
    cycles = frame["cycle_code"].to_numpy(dtype=np.int64)[rows]
    baseline = 20.0 + (rows % 97).astype(np.float64) * 0.6
    return {"baseline_prediction": baseline, "score_cycle": cycles}


def test_sparse_lp_recovers_nonnegative_affine_map_and_b_zero_boundary() -> None:
    p = np.linspace(5.0, 60.0, 80)
    y = 3.25 + 1.4 * p
    solved = r17.solve_nonnegative_l1(y, p)
    assert solved["success"] is True
    assert solved["slope"] >= 0.0
    assert solved["solver_status"] == 0
    assert solved["constraint_shape"] == [160, 82]
    assert solved["constraint_nnz"] == 6 * len(y)
    assert solved["constraint_csr_bytes"] > 0
    assert solved["intercept"] == pytest.approx(3.25, abs=1e-7)
    assert solved["slope"] == pytest.approx(1.4, abs=1e-8)

    boundary = r17.solve_nonnegative_l1(np.full(30, 47.0), np.linspace(0.0, 100.0, 30))
    assert boundary["success"] is True
    assert boundary["slope"] == pytest.approx(0.0, abs=1e-8)
    assert boundary["intercept"] == pytest.approx(47.0, abs=1e-8)


def test_lp_tied_predictions_clipping_and_solver_failure_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    tied = r17.solve_nonnegative_l1(np.array([12.0, 15.0, 18.0, 21.0]), np.full(4, 20.0))
    assert tied["success"] is True
    assert tied["slope"] >= 0.0
    clipped = r17.apply_affine(np.array([0.0, 10.0, 90.0]), -20.0, 2.0)
    np.testing.assert_array_equal(clipped, np.array([0.0, 0.0, 100.0]))

    monkeypatch.setattr(r17, "linprog", lambda *args, **kwargs: SimpleNamespace(
        success=False, status=2, message="synthetic infeasible", nit=3,
    ))
    fallback = r17.solve_nonnegative_l1(np.array([1.0, 2.0]), np.array([3.0, 4.0]))
    assert fallback["success"] is False
    assert fallback["fallback"] is True
    assert fallback["fallback_reason"] == "solver_non_success"
    assert fallback["intercept"] == 0.0 and fallback["slope"] == 1.0

    monkeypatch.setattr(r17, "linprog", lambda *args, **kwargs: SimpleNamespace(
        success=True, status=0, message="synthetic missing objective", nit=1,
        x=np.array([1.0, 1.0, 0.0, 0.0]),
    ))
    malformed_success = r17.solve_nonnegative_l1(np.array([1.0, 2.0]), np.array([3.0, 4.0]))
    assert malformed_success["success"] is False
    assert malformed_success["fallback"] is True
    assert malformed_success["fallback_reason"] == "invalid_or_inconsistent_solver_solution"


def test_fit_predict_uses_only_oof_rows_in_the_fit_prefix_and_falls_back_below_support() -> None:
    frame, target = _synthetic_frame()
    cycles = frame["cycle_code"].to_numpy(dtype=np.int64)
    oof_prediction = np.clip(4.0 + (target - 4.0) / 1.25, 0.0, 100.0)
    history_by_cutoff = {
        cutoff: _history(frame, oof_prediction, cutoff) for cutoff in r17.FULL_CUTOFFS
    }
    fit_rows = np.flatnonzero(cycles <= 67)
    score_rows = np.flatnonzero((cycles >= 68) & (cycles <= 79))
    fit_x = frame.iloc[fit_rows].loc[:, ["cycle_code"]]
    score_x = frame.iloc[score_rows].loc[:, ["cycle_code"]]
    adapter = partial(
        r17.fit_predict,
        history_by_cutoff=history_by_cutoff,
        source_cutoff_by_cycle=_source_cutoff_by_cycle(),
        cycle_by_row=cycles,
        minimum_rows=50,
    )
    result = adapter(
        fit_x, target[fit_rows], score_x,
        controls=_controls(frame, score_rows), card_id="R17", cutoff=67,
    )
    assert result["fit_predictor_count"] == 1
    assert result["state"]["history_cycles"] == list(range(56, 68))
    assert result["state"]["history_rows"] == 12 * 16
    assert result["state"]["outer_score_labels_passed_to_calibrator"] is False
    assert result["state"]["slope"] >= 0.0
    np.testing.assert_allclose(result["prediction"], np.clip(
        result["state"]["intercept"] + result["state"]["slope"] * _controls(frame, score_rows)["baseline_prediction"],
        0.0, 100.0,
    ))

    with pytest.raises(r17.R17IntegrityError, match="score rows do not exactly match"):
        incomplete_rows = score_rows[:-1]
        incomplete_controls = {
            name: value[:-1] for name, value in _controls(frame, score_rows).items()
        }
        r17.fit_predict(
            fit_x, target[fit_rows], frame.iloc[incomplete_rows].loc[:, ["cycle_code"]],
            controls=incomplete_controls, card_id="R17", cutoff=67,
            history_by_cutoff=history_by_cutoff,
            source_cutoff_by_cycle=_source_cutoff_by_cycle(),
            cycle_by_row=cycles, minimum_rows=50,
        )

    small_fit_rows = np.flatnonzero(cycles <= 59)
    small_score_rows = np.flatnonzero((cycles >= 60) & (cycles <= 63))
    small_adapter = partial(
        r17.fit_predict,
        history_by_cutoff=history_by_cutoff,
        source_cutoff_by_cycle=_source_cutoff_by_cycle(),
        cycle_by_row=cycles,
        minimum_rows=5_000,
    )
    small = small_adapter(
        frame.iloc[small_fit_rows].loc[:, ["cycle_code"]], target[small_fit_rows],
        frame.iloc[small_score_rows].loc[:, ["cycle_code"]],
        controls=_controls(frame, small_score_rows), card_id="R17", cutoff=59,
    )
    assert small["fit_predictor_count"] == 0
    assert small["state"]["fallback_reason"] == "history_below_5000_rows"
    np.testing.assert_array_equal(small["prediction"], _controls(frame, small_score_rows)["baseline_prediction"])


def test_fit_predict_rejects_future_or_misaligned_oof_history() -> None:
    frame, target = _synthetic_frame()
    cycles = frame["cycle_code"].to_numpy(dtype=np.int64)
    oof_prediction = np.full(len(frame), 50.0)
    history_by_cutoff = {67: _history(frame, oof_prediction, 67)}
    fit_rows = np.flatnonzero(cycles <= 67)
    score_rows = np.flatnonzero((cycles >= 68) & (cycles <= 79))
    fit_x = frame.iloc[fit_rows].loc[:, ["cycle_code"]]
    score_x = frame.iloc[score_rows].loc[:, ["cycle_code"]]
    controls = _controls(frame, score_rows)
    with pytest.raises(r17.R17IntegrityError, match="do not align"):
        broken = dict(history_by_cutoff)
        broken_cycles = broken[67]["cycles"].copy()
        broken_cycles[-16:] = 66
        broken[67] = {**broken[67], "cycles": broken_cycles}
        r17.fit_predict(
            fit_x, target[fit_rows], score_x, controls=controls, card_id="R17", cutoff=67,
            history_by_cutoff=broken, source_cutoff_by_cycle=_source_cutoff_by_cycle(),
            cycle_by_row=cycles, minimum_rows=50,
        )

    # Preserve the complete set of cycle labels while swapping row assignments.
    broken = dict(history_by_cutoff)
    broken_cycles = broken[67]["cycles"].copy()
    cycle_56 = np.flatnonzero(broken_cycles == 56)
    cycle_57 = np.flatnonzero(broken_cycles == 57)
    left, right = int(cycle_56[-1]), int(cycle_57[0])
    broken_cycles[left], broken_cycles[right] = broken_cycles[right], broken_cycles[left]
    broken[67] = {**broken[67], "cycles": broken_cycles}
    with pytest.raises(r17.R17IntegrityError, match="do not align"):
        r17.fit_predict(
            fit_x, target[fit_rows], score_x, controls=controls, card_id="R17", cutoff=67,
            history_by_cutoff=broken, source_cutoff_by_cycle=_source_cutoff_by_cycle(),
            cycle_by_row=cycles, minimum_rows=50,
        )

    broken = dict(history_by_cutoff)
    wrong_source = _source_cutoff_by_cycle()
    wrong_source[56] = 54
    with pytest.raises(r17.R17IntegrityError, match="canonical six-fold mapping"):
        r17.fit_predict(
            fit_x, target[fit_rows], score_x, controls=controls, card_id="R17", cutoff=67,
            history_by_cutoff=history_by_cutoff, source_cutoff_by_cycle=wrong_source,
            cycle_by_row=cycles, minimum_rows=50,
        )


def test_generic_late_pilot_runs_and_independent_metrics_reconstruct(
    tmp_path: Path,
) -> None:
    frame, target = _synthetic_frame(per_cycle=12)
    cycles = frame["cycle_code"].to_numpy(dtype=np.int64)
    oof_prediction = np.clip(5.0 + (target - 5.0) / 1.1, 0.0, 100.0)
    history_by_cutoff = {cutoff: _history(frame, oof_prediction, cutoff) for cutoff in r17.PILOT_CUTOFFS}
    control_by_cutoff = {
        67: _controls(frame, np.flatnonzero((cycles >= 68) & (cycles <= 79))),
        75: _controls(frame, np.flatnonzero((cycles >= 76) & (cycles <= 79))),
    }
    source = tmp_path / "synthetic_r17_source.py"
    train_path = tmp_path / "synthetic_r17_train.csv"
    source.write_text("synthetic R17 source identity\n", encoding="utf-8")
    frame.to_csv(train_path, index=False)
    card = {"id": "R17", "task": "regression", "pilot_predictor_fits": 2}
    identity = make_freeze_identity(
        card=card, feature_names=r17.FEATURE_NAMES,
        input_files={"synthetic_train": train_path}, source_files=[source],
        evaluation_views=LATE_VIEWS, packages=["scipy"],
    )
    adapter = partial(
        r17.fit_predict,
        history_by_cutoff=history_by_cutoff,
        source_cutoff_by_cycle=_source_cutoff_by_cycle(),
        cycle_by_row=cycles,
        minimum_rows=50,
    )

    run_dir = tmp_path / "runs"
    report = run_pilot(
        card=card, artifact_dir=run_dir, budget_ledger_path=tmp_path / "ledger.jsonl",
        max_successful_fits=180, frame=frame, target=target,
        feature_names=r17.FEATURE_NAMES, adapter=adapter,
        control_loader=lambda cutoff, rows: control_by_cutoff[cutoff],
        freeze_identity=identity, freeze_path=tmp_path / "freeze.json",
        event_path=tmp_path / "attempts.jsonl", evaluation_views=LATE_VIEWS,
        threads=2, report_path=tmp_path / "engine_report.json",
    )

    def reconstruct(arrays: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
        first_cycle = int(np.min(cycles[np.asarray(arrays["score_rows"], dtype=np.int64)]))
        cutoff = 67 if first_cycle == 68 else 75
        history = history_by_cutoff[cutoff]
        fit_history_rows = history["rows"]
        fit_prediction = history["prediction"]
        fit_target = target[fit_history_rows]
        solved = r17.solve_nonnegative_l1(fit_target, fit_prediction, threads=2)
        expected = r17.apply_affine(
            arrays["baseline_prediction"], solved["intercept"], solved["slope"]
        )
        return expected, np.asarray(arrays["baseline_prediction"], dtype=np.float64)

    verification = verify_report(
        report=report, frame=frame, target=target, artifact_dir=run_dir,
        freeze_signature=report["freeze_signature"], evaluation_views=LATE_VIEWS,
        prediction_reconstructors={"R17": reconstruct},
    )
    assert report["fit_count_matches_card"] is True
    assert report["fit_predictor_count"] == 2
    assert verification["status"] == "PASS"
    assert verification["views"]["primary"]["n"] == sum(
        int(np.isin(cycles, np.arange(low, high + 1)).sum())
        for _, low, high in LATE_VIEWS["primary"]
    )


def test_resource_receipts_fail_closed_when_status_disagrees_with_rss() -> None:
    signature = "a" * 64
    valid_postfit = {
        "freeze_signature": signature,
        "maximum_job_rss_bytes": r17runner.MAX_JOB_RSS_BYTES,
        "includes_candidate_and_independent_verification": True,
        "status": "PASS",
        "job_peak_rss_within_cap": True,
        "resource": {"process_peak_rss_bytes": r17runner.MAX_JOB_RSS_BYTES - 1},
    }
    assert r17runner._validate_resource_postfit_doc(valid_postfit, signature) is True
    tampered = {**valid_postfit, "job_peak_rss_within_cap": False}
    with pytest.raises(BatchIntegrityError, match="disagrees with measured peak RSS"):
        r17runner._validate_resource_postfit_doc(tampered, signature)

    valid_preflight = {
        "freeze_signature": signature, "status": "PASS", "before_candidate_fit": True,
        "resource": {
            "status": "PASS", "available_ram_bytes": 5 * r17runner.GIB,
            "process_rss_bytes": 100,
            "predicted_process_peak_rss_bytes": 100 + r17runner.GIB,
            "policy": {"estimated_additional_lp_rss_bytes": r17runner.GIB},
        },
    }
    r17runner._validate_resource_preflight_doc(valid_preflight, signature)
    tampered_preflight = {
        **valid_preflight,
        "resource": {**valid_preflight["resource"], "predicted_process_peak_rss_bytes": 6 * r17runner.GIB + 1},
    }
    with pytest.raises(BatchIntegrityError, match="does not satisfy the frozen RAM gate"):
        r17runner._validate_resource_preflight_doc(tampered_preflight, signature)
