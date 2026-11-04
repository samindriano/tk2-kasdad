from scripts.optimization_v3_cls_pos_a02_meta_v1_progress_recovery import ledger_is_extension


def test_ledger_extension_requires_exact_prior_prefix():
    assert ledger_is_extension(["a", "b"], ["a", "b", "c"])
    assert ledger_is_extension([], ["a"])
    assert not ledger_is_extension(["a", "b"], ["a", "x", "b"])
    assert not ledger_is_extension(["a", "b"], ["a"])
