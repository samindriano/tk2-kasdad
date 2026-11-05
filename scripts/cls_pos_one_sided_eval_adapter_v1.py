"""Additive adapter for the frozen CLS one-sided positional-ridge evaluator.

The parent evaluator has a runtime defect: it uses ``ARMS`` without importing
it. This adapter supplies that exact frozen tuple from the campaign model
module at runtime. It does not edit, copy, or replace the frozen evaluator,
runner, verifier, recipe, or existing fit artifacts.

This source-preparation build intentionally has no command to generate real
predictions or score query labels. Those actions require a separate root review
after this adapter and its additive freeze are inspected.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts import evaluate_optimization_v3_cls_pos_frequency_ridge_v1 as OLD_EVAL
from scripts import evaluate_optimization_v3_cls_pos_one_sided_ridge_v1 as EVAL
from scripts import optimization_v3_cls_pos_one_sided_ridge_v1 as RUNNER
from src.models.pos_one_sided_ridge_v1 import ARMS, FIT_IDS


CAMPAIGN_ID = "CLS_POS_ONE_SIDED_RIDGE_V1"
ADAPTER_ID = "CLS_POS_ONE_SIDED_EVALUATION_ADAPTER_V1"
ADAPTER_REL = "scripts/cls_pos_one_sided_eval_adapter_v1.py"
QA_REL = "reports/optimization_v3/cls_pos_one_sided_ridge_v1/EVALUATION_ADAPTER_QA_V2.json"
FREEZE_REL = "experiments/optimization_v3/cls_pos_one_sided_ridge_v1/EVALUATION_ADAPTER_FREEZE_V2.json"
EVALUATOR_REL = "scripts/evaluate_optimization_v3_cls_pos_one_sided_ridge_v1.py"
VERIFIER_REL = "scripts/verify_optimization_v3_cls_pos_one_sided_ridge_v1.py"
EXPECTED_ROOT_PASS_SHA256 = "86b92a8e1e5c401fce0661de139976ea4203822f86938716ead8b4badf3599ea"


class AdapterError(RuntimeError):
    """Raised when additive adapter lineage or frozen evaluator bindings drift."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, allow_nan=False, sort_keys=True,
                         separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def path(relative: str) -> Path:
    return ROOT / relative


def git_text(*args: str) -> str:
    proc = subprocess.run(["git", *args], cwd=ROOT, capture_output=True,
                          text=True, check=False)
    if proc.returncode:
        raise AdapterError(f"git {' '.join(args)} failed: {proc.stderr.strip()}")
    return proc.stdout.strip()


def install_frozen_arm_binding() -> dict[str, Any]:
    """Supply the exact two frozen arms without changing evaluator source bytes."""
    expected = tuple(ARMS)
    existing = getattr(EVAL, "ARMS", None)
    if existing is not None and tuple(existing) != expected:
        raise AdapterError(f"Frozen evaluator ARMS binding conflicts with model recipe: {existing!r}")
    EVAL.ARMS = expected
    if tuple(EVAL.ARMS) != ("ONE_SIDED", "GLOBAL_STRENGTH"):
        raise AdapterError("Adapter did not bind the exact protocol arm order")
    return {
        "evaluator_module": EVALUATOR_REL,
        "evaluator_source_sha256": sha256_file(path(EVALUATOR_REL)),
        "binding_source": "src.models.pos_one_sided_ridge_v1.ARMS",
        "arms": list(EVAL.ARMS),
        "evaluator_source_modified": False,
    }


def require_frozen_parent() -> tuple[dict[str, Any], dict[str, Any]]:
    freeze = RUNNER.verify_source_freeze(require_frozen_commit=True)
    root_pass = RUNNER.require_root_source_pass()
    root_pass_path = RUNNER._path(RUNNER.ROOT_SOURCE_PASS_REL)
    root_pass_sha = sha256_file(root_pass_path)
    if (root_pass_sha != EXPECTED_ROOT_PASS_SHA256
            or root_pass.get("authorized_fit_ids") != list(FIT_IDS)
            or root_pass.get("source_freeze_sha256") != sha256_file(RUNNER._path(RUNNER.SOURCE_FREEZE_REL))
            or root_pass.get("source_preflight_sha256") != sha256_file(RUNNER._path(RUNNER.SOURCE_PREFLIGHT_REL))):
        raise AdapterError("Current ROOT_SOURCE_PASS is not the reviewed four-fit parent pass")
    frozen_sources = freeze.get("source_hashes", {})
    for relative in (EVALUATOR_REL, VERIFIER_REL,
                     "scripts/optimization_v3_cls_pos_one_sided_ridge_v1.py"):
        if frozen_sources.get(relative) != sha256_file(path(relative)):
            raise AdapterError(f"Frozen parent source hash changed: {relative}")
    return freeze, root_pass


def audit_real_fit_ledger() -> dict[str, Any]:
    """Run the unmodified evaluator's actual ledger and receipt audit target-free."""
    require_frozen_parent()
    binding = install_frozen_arm_binding()
    audit = EVAL._verify_fit_ledger()
    expected = list(FIT_IDS)
    if (audit.get("fit_ids_in_order") != expected or audit.get("fit_starts") != 4
            or audit.get("fit_saves") != 4 or audit.get("fit_failures") != 0
            or audit.get("all_gamma_receipts_validated") is not True):
        raise AdapterError("Frozen evaluator did not validate the exact four saved fits")
    return {
        "status": "PASS_TARGET_FREE_REAL_FIT_LEDGER_AUDIT",
        "adapter_binding": binding,
        "evaluator_ledger_audit": audit,
        "target_values_opened": False,
        "query_labels_read": False,
        "prediction_bundle_created": False,
        "test_or_samplesubmission_accessed": False,
        "official_prediction_started": False,
    }


def _fit_bindings() -> dict[str, Any]:
    events = RUNNER.read_events(RUNNER._path(RUNNER.LEDGER_REL))
    starts = [event.get("fit_id") for event in events if event.get("event") == "FIT_STARTED"]
    saves = [event.get("fit_id") for event in events if event.get("event") == "FIT_SAVED"]
    failures = [event.get("fit_id") for event in events if event.get("event") == "FIT_FAILED"]
    if starts != list(FIT_IDS) or saves != list(FIT_IDS) or failures:
        raise AdapterError("Four ordered saved-fit events and zero failures are required for adapter freeze")
    entries = []
    for fit_id in FIT_IDS:
        artifact, receipt_path = RUNNER.fit_paths(fit_id)
        receipt = RUNNER.read_json(receipt_path)
        if (receipt.get("fit_id") != fit_id
                or receipt.get("artifact_sha256") != sha256_file(artifact)
                or receipt.get("fit_target_sha256") != receipt.get("target_sha256_expected_from_frozen_bag_accounting")):
            raise AdapterError(f"Fit receipt/artifact lineage failed for {fit_id}")
        entries.append({
            "fit_id": fit_id,
            "fit_rows": receipt.get("fit_rows"),
            "fit_ids_sha256": receipt.get("fit_ids_sha256"),
            "artifact_path": artifact.relative_to(ROOT).as_posix(),
            "artifact_sha256": sha256_file(artifact),
            "receipt_path": receipt_path.relative_to(ROOT).as_posix(),
            "receipt_sha256": sha256_file(receipt_path),
            "gamma_sha256": receipt.get("gamma_sha256"),
            "target_sha256": receipt.get("fit_target_sha256"),
            "target_hash_is_provenance_only": True,
        })
    return {"fit_ids": starts, "fit_failures": failures,
            "fit_ledger_sha256": sha256_file(RUNNER._path(RUNNER.LEDGER_REL)),
            "fits": entries}


def _adapter_source_hashes() -> dict[str, str]:
    relatives = (
        ADAPTER_REL,
        EVALUATOR_REL,
        VERIFIER_REL,
        "scripts/optimization_v3_cls_pos_one_sided_ridge_v1.py",
        "src/models/pos_one_sided_ridge_v1.py",
        "tests/test_cls_pos_one_sided_eval_adapter_v1.py",
    )
    return {relative: sha256_file(path(relative)) for relative in relatives}


def build_adapter_freeze(qa: dict[str, Any]) -> dict[str, Any]:
    freeze, _ = require_frozen_parent()
    if qa.get("status") != "PASS_ADDITIVE_EVALUATION_ADAPTER_SOURCE_QA":
        raise AdapterError("Passing additive adapter QA is required before adapter freeze")
    adapter_blob = git_text("hash-object", ADAPTER_REL)
    if git_text("rev-parse", f"HEAD:{ADAPTER_REL}") != adapter_blob:
        raise AdapterError("Adapter source must be committed unchanged before its freeze")
    parent_source_hashes = freeze["source_hashes"]
    inputs = _adapter_source_hashes()
    qa_path = path(QA_REL)
    if git_text("rev-parse", f"HEAD:{QA_REL}") != git_text("hash-object", QA_REL):
        raise AdapterError("Additive adapter QA receipt must be committed before freezing its lineage")
    if qa.get("adapter_source_hashes") != inputs:
        raise AdapterError("Adapter QA source hashes differ from the committed source bundle")
    if inputs.get(EVALUATOR_REL) != parent_source_hashes.get(EVALUATOR_REL):
        raise AdapterError("Additive adapter freeze detected a change to the frozen evaluator")
    if inputs.get(VERIFIER_REL) != parent_source_hashes.get(VERIFIER_REL):
        raise AdapterError("Additive adapter freeze detected a change to the frozen verifier")
    body: dict[str, Any] = {
        "schema_version": 2,
        "adapter_id": ADAPTER_ID,
        "campaign_id": CAMPAIGN_ID,
        "status": "SOURCE_READY_ROOT_REVIEW_REQUIRED_BEFORE_REAL_PREDICTIONS",
        "adapter_commit": git_text("rev-parse", "HEAD"),
        "adapter_source_path": ADAPTER_REL,
        "adapter_source_sha256": inputs[ADAPTER_REL],
        "adapter_qa_path": QA_REL,
        "adapter_qa_sha256": sha256_file(qa_path),
        "frozen_parent_source_commit": freeze["source_commit"],
        "frozen_parent_source_hashes": parent_source_hashes,
        "frozen_source_freeze_sha256": sha256_file(RUNNER._path(RUNNER.SOURCE_FREEZE_REL)),
        "frozen_identity_preflight_sha256": sha256_file(RUNNER._path(RUNNER.SOURCE_PREFLIGHT_REL)),
        "frozen_root_source_pass_sha256": sha256_file(RUNNER._path(RUNNER.ROOT_SOURCE_PASS_REL)),
        "frozen_root_source_pass_commit": git_text("log", "-1", "--format=%H", "--",
                                                   RUNNER.ROOT_SOURCE_PASS_REL),
        "adapter_input_source_hashes": inputs,
        "fits": _fit_bindings(),
        "source_and_fit_lineage_verified": True,
        "query_predictions_created": False,
        "query_labels_read": False,
        "test_or_samplesubmission_accessed": False,
        "kaggle_accessed": False,
        "real_prediction_or_evaluation_authorized": False,
        "requires_separate_root_review_before_real_prediction": True,
        "frozen_originals_modified": False,
        "frozen_recipe_or_gates_modified": False,
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    body["freeze_signature"] = canonical_sha256(body)
    return body


def verify_adapter_freeze(*, require_committed: bool = True) -> dict[str, Any]:
    freeze_path = path(FREEZE_REL)
    if not freeze_path.is_file():
        raise AdapterError("Additive evaluation adapter freeze is missing")
    if require_committed and git_text("rev-parse", f"HEAD:{FREEZE_REL}") != git_text("hash-object", FREEZE_REL):
        raise AdapterError("Additive evaluation adapter freeze must be committed unchanged")
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    body = dict(freeze)
    claimed = body.pop("freeze_signature", None)
    if claimed != canonical_sha256(body):
        raise AdapterError("Additive evaluation adapter freeze signature mismatch")
    if (freeze.get("schema_version") != 2
            or freeze.get("adapter_id") != ADAPTER_ID
            or freeze.get("campaign_id") != CAMPAIGN_ID
            or freeze.get("status") != "SOURCE_READY_ROOT_REVIEW_REQUIRED_BEFORE_REAL_PREDICTIONS"):
        raise AdapterError("Additive evaluation adapter freeze identity or status mismatch")
    require_frozen_parent()
    if (_adapter_source_hashes() != freeze.get("adapter_input_source_hashes")
            or sha256_file(path(QA_REL)) != freeze.get("adapter_qa_sha256")
            or git_text("rev-parse", f"HEAD:{QA_REL}") != git_text("hash-object", QA_REL)
            or sha256_file(RUNNER._path(RUNNER.SOURCE_FREEZE_REL)) != freeze.get("frozen_source_freeze_sha256")
            or sha256_file(RUNNER._path(RUNNER.ROOT_SOURCE_PASS_REL)) != freeze.get("frozen_root_source_pass_sha256")):
        raise AdapterError("Additive freeze source or parent-pass hash mismatch")
    if _fit_bindings() != freeze.get("fits"):
        raise AdapterError("Additive freeze fit ledger/receipt/artifact bindings changed")
    if (freeze.get("real_prediction_or_evaluation_authorized") is not False
            or freeze.get("query_predictions_created") is not False
            or freeze.get("query_labels_read") is not False
            or freeze.get("test_or_samplesubmission_accessed") is not False
            or freeze.get("kaggle_accessed") is not False
            or freeze.get("frozen_originals_modified") is not False
            or freeze.get("frozen_recipe_or_gates_modified") is not False
            or freeze.get("requires_separate_root_review_before_real_prediction") is not True):
        raise AdapterError("Additive freeze no longer records the source-review stop boundary")
    return {
        "status": "PASS_INDEPENDENT_ADDITIVE_ADAPTER_FREEZE",
        "freeze_signature": claimed,
        "fit_ids": freeze["fits"]["fit_ids"],
        "real_predictions_authorized": False,
        "query_labels_read": False,
    }


def write_adapter_freeze(qa: dict[str, Any]) -> dict[str, Any]:
    value = build_adapter_freeze(qa)
    path(FREEZE_REL).parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(value, ensure_ascii=False, allow_nan=False,
                         sort_keys=True, indent=2) + "\n"
    if path(FREEZE_REL).exists():
        if path(FREEZE_REL).read_text(encoding="utf-8") != encoded:
            raise AdapterError("Refusing to overwrite immutable additive adapter freeze")
    else:
        with path(FREEZE_REL).open("x", encoding="utf-8", newline="\n") as stream:
            stream.write(encoded)
            stream.flush()
    return value


def run_source_qa() -> dict[str, Any]:
    """Run the actual saved-fit ledger check and synthetic prepare entry path."""
    with OLD_EVAL.SharedJobLock():
        initial = RUNNER.require_resource("saved_replay", "before additive adapter source QA")
        binding = install_frozen_arm_binding()
        real_fit_audit = audit_real_fit_ledger()
        import importlib.util
        test_path = path("tests/test_cls_pos_one_sided_eval_adapter_v1.py")
        spec = importlib.util.spec_from_file_location("cls_pos_one_sided_eval_adapter_tests", test_path)
        if spec is None or spec.loader is None:
            raise AdapterError("Additive adapter test module cannot be loaded")
        test_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(test_module)
        synthetic = test_module.run_synthetic_prepare_entry_qa()
        boundary = RUNNER.require_actual_boundary("saved_replay", "after additive adapter source QA")
    if (synthetic.get("status") != "PASS"
            or synthetic.get("query_labels_blocked_until_manifest_commit") is not True
            or synthetic.get("query_labels_joined_after_commit") is not True):
        raise AdapterError("Synthetic prediction preparation or label boundary QA failed")
    return {
        "status": "PASS_ADDITIVE_EVALUATION_ADAPTER_SOURCE_QA",
        "source_commit": git_text("rev-parse", "HEAD"),
        "adapter_binding": binding,
        "real_fit_audit": real_fit_audit,
        "synthetic_prepare_entry": synthetic,
        "prior_harness_attempts": [
            {
                "status": "FAILED_TEST_HARNESS_PRECONDITION_ONLY",
                "cause": "synthetic test expected ARMS absent after the same-process real ledger audit had correctly installed the adapter binding",
                "campaign_fit_repeated": False,
                "official_predictions_created": False,
                "query_labels_scored": False,
            },
            {
                "status": "FAILED_QA_AGGREGATOR_FIELD_NAME_ONLY",
                "cause": "aggregate QA checked manifest_commit_guard_passed while the test emits query_labels_blocked_until_manifest_commit and query_labels_joined_after_commit",
                "synthetic_prepare_completed": True,
                "campaign_fit_repeated": False,
                "official_predictions_created": False,
            },
        ],
        "resource_snapshot": {
            "policy": RUNNER.RESOURCE_POLICIES["saved_replay"],
            "initial_under_shared_lock": initial,
            "actual_boundary": boundary,
            "samples_are_peak_measurements": False,
        },
        "target_values_opened_by_real_fit_audit": False,
        "official_predictions_created": False,
        "query_labels_scored": False,
        "test_or_samplesubmission_accessed": False,
        "kaggle_accessed": False,
    }


def write_qa_receipt(value: dict[str, Any]) -> None:
    target = path(QA_REL)
    encoded = json.dumps(value, ensure_ascii=False, allow_nan=False,
                         sort_keys=True, indent=2) + "\n"
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        if target.read_text(encoding="utf-8") != encoded:
            raise AdapterError("Refusing to overwrite additive adapter QA receipt")
    else:
        with target.open("x", encoding="utf-8", newline="\n") as stream:
            stream.write(encoded)
            stream.flush()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--source-qa", action="store_true")
    mode.add_argument("--write-adapter-freeze", action="store_true")
    mode.add_argument("--verify-adapter-freeze", action="store_true")
    args = parser.parse_args(argv)
    try:
        if args.source_qa:
            result = run_source_qa()
            result["adapter_source_hashes"] = _adapter_source_hashes()
            write_qa_receipt(result)
        elif args.write_adapter_freeze:
            qa = json.loads(path(QA_REL).read_text(encoding="utf-8"))
            result = write_adapter_freeze(qa)
        else:
            result = verify_adapter_freeze()
        print(json.dumps(result, ensure_ascii=False, allow_nan=False, sort_keys=True, indent=2))
        return 0
    except (AdapterError, RUNNER.OneSidedRunnerError, EVAL.OneSidedEvaluationError,
            OLD_EVAL.EvaluationError) as exc:
        print(json.dumps({"status": "BLOCKED", "error": str(exc)}, ensure_ascii=False,
                         sort_keys=True, indent=2), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
