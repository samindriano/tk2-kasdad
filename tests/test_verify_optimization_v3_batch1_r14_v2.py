import pytest

from scripts import verify_optimization_v3_batch1_r14_v2 as verifier


def _view_metrics(primary_rows=9064, primary_baseline=13.464395831218011):
    return {
        "primary": {"rows": primary_rows, "baseline_mae": primary_baseline},
        "H12": {
            "rows": 38588,
            "baseline_mae": verifier.EXPECTED_CONTROL_METRICS["H12_mae"],
        },
    }


def test_control_references_are_bound_to_their_actual_populations():
    metrics = _view_metrics()

    verifier._verify_r14_control_populations(metrics)
    audit = verifier._control_metric_population_audit(metrics)

    assert audit["R14_primary_W1_W4"]["rows"] == 9064
    assert audit["V2_fullsix_primary_reference"]["rows"] == 38588
    assert audit["V2_fullsix_primary_reference"]["asserted_equal_to_R14_primary"] is False


@pytest.mark.parametrize(
    "metrics",
    [
        _view_metrics(primary_rows=38588),
        _view_metrics(primary_baseline=11.559390078665364),
    ],
)
def test_control_check_rejects_cross_population_primary_comparison(metrics):
    with pytest.raises(verifier.R14RunnerError):
        verifier._verify_r14_control_populations(metrics)


def test_guarded_harm_aggregation_reads_supported_group_rows_directly():
    groups = {
        "primary": {
            "tail_high_ge95": {"supported": True, "harm_candidate_minus_baseline": 0.32},
            "stream_missing": {"supported": True, "harm_candidate_minus_baseline": 0.04},
        },
        "H12_E": {
            "tail_high_ge95": {"supported": True, "harm_candidate_minus_baseline": 0.31},
            "tail_low_le5": {"supported": False, "harm_candidate_minus_baseline": 99.0},
        },
        "H12_L": {
            "tail_high_ge95": {"supported": True, "harm_candidate_minus_baseline": 0.556},
            "EF1": {"supported": True, "harm_candidate_minus_baseline": 9.0},
        },
    }

    assert verifier._max_guarded_harm(groups) == 0.556


def test_guarded_harm_is_none_when_no_tail_or_missing_group_is_supported():
    groups = {
        view: {
            "tail_high_ge95": {"supported": False, "harm_candidate_minus_baseline": None},
            "EF1": {"supported": True, "harm_candidate_minus_baseline": 5.0},
        }
        for view in ("primary", "H12_E", "H12_L")
    }

    assert verifier._max_guarded_harm(groups) is None
