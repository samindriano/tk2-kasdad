import ast
from pathlib import Path

from scripts import optimization_v3_rnbr_impact_match_v2 as runner


ROOT = Path(__file__).resolve().parents[1]


def _function_node(source: str, name: str) -> ast.FunctionDef:
    module = ast.parse(source)
    return next(node for node in module.body if isinstance(node, ast.FunctionDef) and node.name == name)


def test_v2_has_no_full_a02_model_load_route():
    runner_source = (ROOT / "scripts/optimization_v3_rnbr_impact_match_v2.py").read_text(encoding="utf-8")
    verifier_source = (ROOT / "scripts/optimization_v3_rnbr_impact_match_verify_v2.py").read_text(encoding="utf-8")
    runner_tree = ast.parse(runner_source)
    loads = [node for node in ast.walk(runner_tree)
             if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
             and isinstance(node.func.value, ast.Name) and node.func.value.id == "joblib"
             and node.func.attr == "load"]
    assert len(loads) == 1
    assert isinstance(loads[0].args[0], ast.Name)
    assert loads[0].args[0].id == "vectorizer_path"
    assert "joblib.load" not in verifier_source
    assert "optimization_v3_rnbr_impact_match as runner" not in verifier_source
    assert "model_path" not in runner_source and "joblib.load(model" not in runner_source


def test_replay_function_is_target_free():
    source = (ROOT / "scripts/optimization_v3_rnbr_impact_match_v2.py").read_text(encoding="utf-8")
    for function_name in ("_read_replay_inputs", "run_replays"):
        node = _function_node(source, function_name)
        function_source = ast.get_source_segment(source, node)
        assert "read_regression_targets" not in function_source
        assert "threat_rank" not in function_source
    assert runner._protocol()["replay_fit_kind"] == "TARGET_FREE_TFIDF_REPLAY"
    assert runner._protocol()["replay_fit_count"] == 2
    assert runner._protocol()["fit_count"] == 4


def test_completed_replays_cannot_be_run_again():
    before = runner._sha_file(runner.LEDGER)
    try:
        runner.run_replays()
    except runner.ImpactMatchRunError as exc:
        assert "already saved" in str(exc)
        assert "prohibited" in str(exc)
    else:
        raise AssertionError("completed replay fits must never be refit")
    assert before == runner._sha_file(runner.LEDGER)


def test_v22_comparator_receipt_synthetic_distinguishes_raw_and_internal_hashes():
    before = runner._sha_file(runner.LEDGER)
    result = runner._comparator_repair_synthetic_qa()
    after = runner._sha_file(runner.LEDGER)
    assert result["status"] == "PASS"
    assert result["replay_synthetic_sha256"] == runner._sha_file(runner.REPLAY_SYNTHETIC_PATH)
    assert result["receipt_internal_integrity_verified"]
    assert result["raw_receipt_file_sha_recorded_separately"]
    assert result["incorrect_raw_vs_internal_hash_comparison_rejected"]
    assert before == after


def test_synthetic_executor_persistence_resume_and_verifier_guards_do_not_mutate_ledger(
    tmp_path, monkeypatch,
):
    before = runner._sha_file(runner.LEDGER)
    monkeypatch.setattr(runner, "_replay_receipt_path", lambda origin: tmp_path / f"E{origin}_receipt.json")
    result = runner.synthetic_qa()
    after = runner._sha_file(runner.LEDGER)
    assert result["status"] == "PASS"
    replay = result["target_free_text_replay_synthetic"]
    assert replay["fit_transform_calls"] == 1
    assert replay["standalone_persistence_readback"]
    assert replay["durable_resume_without_refit"]
    assert replay["missing_state_rejected"]
    assert replay["vectorizer_tamper_rejected"]
    assert replay["nested_recovery_path"] == ["nested/recovered.npz", "nested/recovered.json"]
    assert replay["swapped_views_rejected"]
    assert replay["comparator_swapped_ids_rejected"]
    assert replay["verifier_and_diagnostic_report_persistence"]
    assert before == after
