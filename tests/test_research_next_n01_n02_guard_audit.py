"""Guard against mistakenly authorizing from primary-only subgroup support."""

from scripts.optimization_v3_research_next_n01_n02_guard_audit import guard_gate_passes


def test_primary_pass_does_not_override_supported_h12_subgroup_failure() -> None:
    gains = {"primary": 0.0010, "W1": 0.0008, "W4": 0.0002, "H12_pooled": 0.0001}
    supported_group_harms = [
        # n, class_0, class_1, F1 harm, positive-recall harm
        (300, 150, 150, 0.020, 0.030),
    ]
    assert not guard_gate_passes(gains, supported_group_harms, matched_gain=0.0003)


def test_supported_groups_must_meet_both_harm_limits() -> None:
    gains = {"primary": 0.0010, "W1": 0.0008, "W4": 0.0002, "H12_pooled": 0.0001}
    supported_group_harms = [(300, 150, 150, 0.005, 0.010)]
    assert guard_gate_passes(gains, supported_group_harms, matched_gain=0.0003)


def test_unsupported_group_is_descriptive_and_does_not_fail_gate() -> None:
    gains = {"primary": 0.0010, "W1": 0.0008, "W4": 0.0002, "H12_pooled": 0.0001}
    unsupported_group_harms = [(199, 99, 100, 0.050, 0.050)]
    assert guard_gate_passes(gains, unsupported_group_harms, matched_gain=0.0003)

