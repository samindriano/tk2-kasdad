from pathlib import Path
import importlib.util
import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("handoff", ROOT / "scripts/export_classification_team_handoff.py")
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)

@pytest.fixture(scope="module")
def tables():
    folder = ROOT / "handoff/classification_team_20261108/artifacts"
    return tuple(pd.read_csv(folder / name, float_precision="round_trip") for name in ["validation_probabilities.csv.gz", "test_probabilities.csv.gz"])

def test_exported_bundle_replays_confirmed_champion():
    result = m.verify(m.DEFAULT)
    assert result["status"] == "PASS" and result["test_predicted_positive"] == 6759

def test_policy_routing_and_boundary_ties():
    p, threshold, labels = m.policy([.8, .1, .4], [.2, m.MISSING_THRESHOLD, .7], [0, 1, 1])
    np.testing.assert_array_equal(labels, [1, 1, 1])
    np.testing.assert_allclose(p, [.5, m.MISSING_THRESHOLD, .7], rtol=0, atol=0)
    assert threshold[0] == .5

def test_bad_probabilities_and_shapes_fail():
    with pytest.raises(ValueError): m.policy([np.nan], [.1], [0])
    with pytest.raises(ValueError): m.policy([1.1], [.1], [0])
    with pytest.raises(ValueError): m.policy([.1, .2], [.1], [0])

def test_duplicate_or_label_injection_fails(tables):
    val, test = tables
    broken = test.copy(); broken.loc[1, "record_id"] = broken.loc[0, "record_id"]
    with pytest.raises(ValueError): m.validate_tables(val, broken)
    with pytest.raises(ValueError): m.validate_tables(val.assign(artifact_signal=0), test)

def test_wrong_probability_or_primary_mask_fails(tables):
    val, test = tables
    wrong = test.copy(); wrong.loc[0, "p_samuel"] += .001
    with pytest.raises(ValueError): m.validate_tables(val, wrong)
    wrong = val.copy(); wrong.loc[0, "is_primary_unique"] ^= 1
    with pytest.raises(ValueError): m.validate_tables(wrong, test)

def test_bad_availability_or_temporal_boundary_fails(tables):
    val, test = tables
    wrong = test.copy(); wrong.loc[0, "stream_missing"] = 2
    with pytest.raises(ValueError): m.validate_tables(val, wrong)
    wrong = val.copy(); wrong.loc[0, "cycle_code"] = 67
    with pytest.raises(ValueError): m.validate_tables(wrong, test)
