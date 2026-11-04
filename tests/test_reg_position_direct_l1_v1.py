from __future__ import annotations

import numpy as np

import scripts.reg_position_direct_l1_v1 as pilot
import scripts.verify_reg_position_direct_l1_v1 as verifier


def test_same_row_values_keep_each_historical_digest_convention() -> None:
    rows = np.asarray([0, 1, 2, 17, 250], dtype=np.int64)
    assert np.array_equal(rows, np.arange(251, dtype=np.int64)[rows])
    assert pilot._sha_raw_array(rows, dtype="<i8") != pilot._position_sha_array(rows)


def test_rank50_is_full_cycle_average_rank_before_any_mask() -> None:
    raw = np.asarray([4.0, 1.0, 1.0, 3.0, 8.0, 8.0], dtype=np.float64)
    cycles = np.asarray([68, 68, 68, 68, 69, 69], dtype=np.int64)
    mapped = pilot._rank50(raw, cycles)
    expected_68 = 0.5 * raw[:4] + 0.5 * (100.0 * np.asarray([4.0, 1.5, 1.5, 3.0]) / 4.0)
    expected_69 = 0.5 * raw[4:] + 0.5 * 75.0
    assert np.array_equal(mapped, np.r_[expected_68, expected_69])
    assert np.array_equal(mapped, verifier._verified_rank50(raw, cycles))


def test_group_support_threshold_reports_unsupported_as_unknown() -> None:
    check = pilot._synthetic_metric_support_check()
    assert check["n100_supported_harm"]["status"] == "SUPPORTED"
    assert check["n100_supported_harm"]["harm_vs_R06"] == -0.5
    assert check["n99_unknown"] == {"n": 99, "status": "UNKNOWN_UNSUPPORTED"}


def test_independent_metric_reports_group_support_and_harm() -> None:
    y = np.arange(100, dtype=np.float64)
    baseline = y + 1.0
    prediction = y + 0.25
    all_rows = np.ones(100, dtype=bool)
    supported = verifier._metric_view(
        y, baseline, {"BAG": prediction, "REAL": prediction, "SHUFFLED": prediction},
        all_rows, {"group": all_rows})
    assert supported["groups"]["group"]["status"] == "SUPPORTED"
    assert abs(supported["groups"]["group"]["arms"]["REAL"]["harm_vs_R06"] + 0.75) < 1e-12
    unsupported = verifier._metric_view(
        y[:99], baseline[:99], {"BAG": prediction[:99], "REAL": prediction[:99],
                                "SHUFFLED": prediction[:99]},
        np.ones(99, dtype=bool), {"group": np.ones(99, dtype=bool)})
    assert unsupported["groups"]["group"] == {"n": 99, "status": "UNKNOWN_UNSUPPORTED"}


def test_selected_target_reader_skips_unselected_csv_record(tmp_path) -> None:
    source = tmp_path / "selected_targets_only.csv"
    source.write_bytes(
        b"record_id,threat_rank\n"
        b"r0,12\n"
        b"r1,\xff\"unclosed target text\n"
        b"r2,99\n"
    )
    observed = pilot._read_targets_for_rows(
        source, np.asarray([0, 2], dtype=np.int64), np.asarray(["r0", "r2"], dtype=str),
        expected_total_rows=3,
    )
    assert np.array_equal(observed, np.asarray([12.0, 99.0]))


def test_actual_entrypoint_synthetic_resume_and_tamper_audit() -> None:
    result = pilot.synthetic_e2e()
    assert result["status"] == "PASS"
    assert result["production_fit_units"] == 0
    assert result["learned_fit_call_total"] == 12
    assert result["ledger_starts"] == result["ledger_saves"] == 12
    assert all(bool(value) for key, value in result["checks"].items()
               if key != "metric_group_support_fixture")
    assert result["checks"]["metric_group_support_fixture"]["n99_unknown"]["status"] == "UNKNOWN_UNSUPPORTED"
