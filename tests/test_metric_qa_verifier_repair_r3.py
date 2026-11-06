from __future__ import annotations

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VERIFIER = ROOT / "scripts" / "verify_reg_pos_disagreement_gate_v1.py"
SPEC = importlib.util.spec_from_file_location("reg_pos_metric_verifier_r3_test", VERIFIER)
qa = importlib.util.module_from_spec(SPEC)
assert SPEC is not None and SPEC.loader is not None
SPEC.loader.exec_module(qa)


def test_synthetic_metric_qa_resolves_runner_train_path_without_reading_labels():
    bag2, runner = qa._helpers()
    assert not hasattr(bag2, "_contained_file")
    assert callable(getattr(runner, "_contained_file", None))
    assert callable(getattr(runner, "_read_targets_for_rows", None))
    assert hasattr(runner, "TRAIN_REL")

    train_path = qa._train_path(bag2, runner)
    assert train_path.is_file()
    assert train_path.name == Path(runner.TRAIN_REL).name

    result = qa.synthetic_metric_qa_stage()
    assert result["status"] == "PASS"
    assert result["synthetic_rows"] == 512
    assert result["train_path_resolved"] is True
    assert result["train_data_read"] is False
    assert result["query_labels_read"] is False
