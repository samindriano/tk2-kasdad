"""Single authorized score recovery using the unchanged frozen rows and predictions."""
from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import time
from typing import Any

import numpy as np

AUDIT_ROOT = Path(__file__).resolve().parents[1]
REPORT_ROOT = AUDIT_ROOT / "reports/simplification_v1/CLS_POS_EF_CONTEXT_V1"
RUN_ROOT = AUDIT_ROOT / "experiments/CLS_POS_EF_CONTEXT_V1"
SOURCE_REL = "scripts/cls_pos_ef_context_v1.py"
RECOVERY_REL = "scripts/recover_cls_pos_ef_context_v1_score.py"
AMENDMENT_PATH = REPORT_ROOT / "SCORE_RECOVERY_AMENDMENT.json"
AUTHORITY_REL = "reports/simplification_v1/CLS_POS_EF_CONTEXT_V1/ROOT_SCORE_RECOVERY_R1.json"
AUTHORITY_SHA256 = "fea379ed88f854db954162677613b115f1060453d5f6c0982c6acc9defe5421c"
AUTHORITY_COMMIT = "bfdde7915f80b31ef665e3be804dc0fd412b9c44"
AUTHORIZATION_SIGNATURE = "3c9827509dd7ea202ef1aabc5d01893d8e614314b035a68511c8e1e075038ec1"
RECOVERY_ID = "CLS_POS_EF_CONTEXT_V1_SCORE_RECOVERY_01"


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def sha256_text_normalized(path: Path) -> str:
    return sha256_bytes(path.read_bytes().replace(b"\r\n", b"\n"))


def git(*args: str) -> str:
    return subprocess.check_output(["git", "-C", str(AUDIT_ROOT), *args]).decode("utf-8").strip()


def load_fit_adapter():
    adapter_path = AUDIT_ROOT / "scripts/run_cls_pos_ef_context_v1_amended.py"
    spec = importlib.util.spec_from_file_location("cls_pos_ef_context_v1_fit_adapter_for_score", adapter_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("Cannot load the committed fit adapter")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _assigns_name(statement: ast.stmt, name: str) -> bool:
    targets = []
    if isinstance(statement, ast.Assign):
        targets = statement.targets
    elif isinstance(statement, ast.AnnAssign):
        targets = [statement.target]
    return any(isinstance(target, ast.Name) and target.id == name for target in targets)


def patch_score_ast(source_text: str) -> tuple[ast.Module, dict[str, Any]]:
    tree = ast.parse(source_text, filename=SOURCE_REL)
    score_functions = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "score_metrics"]
    if len(score_functions) != 1:
        raise RuntimeError("Expected exactly one frozen score_metrics function")
    function = score_functions[0]

    prior_guards = [node for node in ast.walk(function)
                    if isinstance(node, ast.If) and isinstance(node.test, ast.Name) and node.test.id == "prior"]
    if len(prior_guards) != 1:
        raise RuntimeError("Expected exactly one prior-query-label guard")
    guard = prior_guards[0]
    guard.test = ast.BoolOp(op=ast.And(), values=[
        ast.Name(id="prior", ctx=ast.Load()),
        ast.UnaryOp(op=ast.Not(), operand=ast.Name(id="SCORE_RECOVERY_AUTHORIZED", ctx=ast.Load()))])

    subgroup_loops = [node for node in ast.walk(function)
                      if isinstance(node, ast.For) and isinstance(node.target, ast.Name) and node.target.id == "value"]
    if len(subgroup_loops) != 1:
        raise RuntimeError("Expected exactly one subgroup value loop")
    subgroup = subgroup_loops[0]
    metric_positions = [i for i, statement in enumerate(subgroup.body) if _assigns_name(statement, "metrics")]
    key_positions = [i for i, statement in enumerate(subgroup.body) if _assigns_name(statement, "key")]
    if len(metric_positions) != 1 or len(key_positions) != 1 or key_positions[0] != metric_positions[0] + 1:
        raise RuntimeError("Frozen subgroup key/metric statements differ from the audited failure site")
    key_statement = subgroup.body.pop(key_positions[0])
    subgroup.body.insert(metric_positions[0], key_statement)

    ast.fix_missing_locations(tree)
    details = {"source_path": SOURCE_REL,
        "prior_guard_change_count": 1,
        "subgroup_key_move_count": 1,
        "subgroup_key_expression": "f'{kind}={value}'",
        "subgroup_metric_id_uses_key_after_assignment": True,
        "no_metric_formula_or_gate_changes": True,
        "patched_ast_sha256": sha256_bytes(ast.dump(tree, annotate_fields=True,
            include_attributes=False).encode("utf-8"))}
    return tree, details


def load_root_authority(pilot, freeze: dict[str, Any]) -> tuple[dict[str, Any], str]:
    authority_path = AUDIT_ROOT / AUTHORITY_REL
    authority_bytes = authority_path.read_bytes()
    pilot.require(sha256_bytes(authority_bytes) == AUTHORITY_SHA256,
                  "Committed root recovery authority hash mismatch")
    committed = subprocess.check_output(["git", "-C", str(AUDIT_ROOT), "show",
                                         f"{AUTHORITY_COMMIT}:{AUTHORITY_REL}"])
    pilot.require(committed == authority_bytes, "Root recovery authority differs from its authorizing commit")
    subprocess.check_call(["git", "-C", str(AUDIT_ROOT), "merge-base", "--is-ancestor",
                           AUTHORITY_COMMIT, "HEAD"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    authority = json.loads(authority_bytes.decode("utf-8"))
    expected = {
        "status": "APPROVED_ONE_SAME_INPUT_SCORE_RECOVERY",
        "authorization_signature": AUTHORIZATION_SIGNATURE,
        "allowed_scoring_recovery_read_count": 1,
        "candidate_id": freeze["candidate_id"],
        "failed_original_ledger_sha256": "55cb26e4d8868656b45bba2d63346f7afd5c832d9c26c1e622ae7385167d4e02",
        "committed_prelabel_sha256": "f36efa106c4b270a7f71c1435864d4ebd320f8bdd859d6758b5f029f0e0ad796",
        "prediction_readiness_sha256": "61a8f137fc11cf19435fb11c5765e26e733e4631ebf703c47c7bd6f05068d030",
        "query_rows_sha256": "ca439ef7bc8f5be60cd7dcb7456c7f5c5be9455338bcf75940ee6f00b52247ab",
        "query_ids_sha256": "541f9e7620ae2663fd762e385d60e6eddad6e4834dac5df63dfef75631f3ece9",
        "new_fits_allowed": 0,
        "prediction_or_model_changes_allowed": False,
        "test_or_sample_access_allowed": False,
        "kaggle_access_allowed": False,
        "metric_qa_same_row_reconstruction_allowed": True,
        "original_score_output_absent": True,
        "original_prelabel_boundary_respected": True,
        "preserve_original_ledger_and_frozen_sources": True,
    }
    pilot.require(all(authority.get(k) == v for k, v in expected.items()),
                  "Root authority content does not match the exact approved recovery scope")
    pilot.require(authority.get("failed_original_query_event") == {
        "candidate_id": freeze["candidate_id"],
        "event": "QUERY_LABELS_OPENED",
        "freeze_signature": freeze["freeze_signature"],
        "prelabel_sha256": expected["committed_prelabel_sha256"],
        "purpose": "single Train temporal metric evaluation",
        "query_ids_sha256": expected["query_ids_sha256"],
        "query_rows_sha256": expected["query_rows_sha256"],
        "rows": 20903,
    }, "Root authority does not bind the exact prior label-open event")
    pilot.require(freeze["source_commit"] == "21001878edbf7685c23453abb8897ecb999cfdfd"
                  and freeze["freeze_signature"] == "d5a6d6165b26a7d1b1f6faa6e793a503ce15a963b2d75a24e4b099e553c4b30c",
                  "Recovery authority is for a different frozen source or candidate")
    return authority, sha256_bytes(authority_bytes)


def verify_recovery_amendment(pilot, freeze: dict[str, Any], tree: ast.Module,
                              patch_details: dict[str, Any], authority_hash: str) -> tuple[dict[str, Any], str]:
    amendment = pilot.read_json(AMENDMENT_PATH)
    pilot.require(amendment.get("status") == "ONE_SCORE_RECOVERY_AUTHORIZED"
                  and amendment.get("recovery_id") == RECOVERY_ID
                  and amendment.get("parent_freeze_signature") == freeze["freeze_signature"]
                  and amendment.get("parent_source_commit") == freeze["source_commit"],
                  "Recovery amendment does not bind to the frozen pilot")
    pilot.require(amendment.get("root_authority_path") == AUTHORITY_REL
                  and amendment.get("root_authority_sha256") == authority_hash
                  and amendment.get("root_authority_commit") == AUTHORITY_COMMIT
                  and amendment.get("authorization_signature") == AUTHORIZATION_SIGNATURE,
                  "Recovery amendment does not bind the committed Sol authority")
    pilot.require(amendment.get("frozen_score_source_sha256") == freeze["source_hashes"][SOURCE_REL]
                  and sha256_file(AUDIT_ROOT / SOURCE_REL) == freeze["source_hashes"][SOURCE_REL],
                  "Frozen score source changed")
    recovery_hash = sha256_file(AUDIT_ROOT / RECOVERY_REL)
    pilot.require(recovery_hash == amendment.get("recovery_wrapper_sha256"), "Recovery wrapper hash mismatch")
    pilot.require(patch_details["patched_ast_sha256"] == amendment.get("patched_ast_sha256"),
                  "Deterministic score repair does not match the committed amendment")
    head = git("rev-parse", "HEAD")
    code_blob = subprocess.check_output(["git", "-C", str(AUDIT_ROOT), "show", f"HEAD:{RECOVERY_REL}"])
    pilot.require(sha256_bytes(code_blob) == recovery_hash, "Recovery wrapper must be committed before use")
    manifest_rel = AMENDMENT_PATH.relative_to(AUDIT_ROOT).as_posix()
    manifest_blob = subprocess.check_output(["git", "-C", str(AUDIT_ROOT), "show", f"HEAD:{manifest_rel}"])
    pilot.require(sha256_bytes(manifest_blob) == sha256_text_normalized(AMENDMENT_PATH),
                  "Recovery amendment manifest must be committed before use")
    dirty = git("status", "--porcelain", RECOVERY_REL, manifest_rel)
    pilot.require(not dirty, "Recovery code/manifest has uncommitted changes")
    return amendment, head


def validate_preconditions(pilot, freeze: dict[str, Any]) -> dict[str, Any]:
    authority, authority_hash = load_root_authority(pilot, freeze)
    pilot.verify_committed_prelabel()
    readiness = pilot._verified_ready(freeze)
    if pilot.SCORE_PATH.exists() or pilot.METRIC_VERIFY_PATH.exists():
        raise pilot.PilotError("A durable score already exists; recovery must not run")
    ledger_path = RUN_ROOT / "fit_ledger.jsonl"
    ledger_hash_before = pilot.sha256_file(ledger_path)
    pilot.require(ledger_hash_before == authority["failed_original_ledger_sha256"],
                  "Raw ledger is not byte-identical to the failed original attempt bound by Sol")
    pilot.require(pilot.sha256_file(pilot.PRELABEL_PATH) == authority["committed_prelabel_sha256"],
                  "Committed prelabel hash differs from root recovery authority")
    pilot.require(pilot.sha256_file(pilot.READY_PATH) == authority["prediction_readiness_sha256"],
                  "Prediction readiness hash differs from root recovery authority")
    events = pilot.events(ledger_path)
    opens = [e for e in events if e.get("event") == "QUERY_LABELS_OPENED"]
    recoveries = [e for e in events if e.get("event") == "SCORE_RECOVERY_STARTED"]
    pilot.require(opens == [authority["failed_original_query_event"]] and not recoveries,
                  "Recovery requires the exact single prior label-open event and no earlier recovery")
    starts = [e for e in events if e.get("event") == "FIT_STARTED"]
    terminals = [e for e in events if e.get("event") in {"FIT_SAVED", "FIT_FAILED"}]
    pilot.require([e.get("fit_id") for e in starts] == freeze["fit_ids"]
                  and [e.get("event") for e in terminals] == ["FIT_SAVED"] * 4,
                  "Four frozen fit receipts are not complete")
    bundles = {o: pilot._load_saved_predictions(RUN_ROOT / f"E{o}/predictions_E{o}.npz") for o in (67, 75)}
    rows = np.asarray(bundles[67]["row_positions"], dtype=np.int64)
    ids = np.asarray(bundles[67]["record_ids"]).astype(str)
    rows75 = np.asarray(bundles[75]["row_positions"], dtype=np.int64)
    ids75 = np.asarray(bundles[75]["record_ids"]).astype(str)
    cycles67 = np.asarray(bundles[67]["cycles"], dtype=np.int64)
    cycles75 = np.asarray(bundles[75]["cycles"], dtype=np.int64)
    expected_overlap = (cycles67 >= 76) & (cycles67 <= 79)
    pilot.require(len(rows) == 20903 and pilot.sha256_array(rows) == opens[0].get("query_rows_sha256")
                  and pilot.sha256_array(ids) == opens[0].get("query_ids_sha256"),
                  "Prior query-read IDs/rows differ from the committed prediction pool")
    pilot.require(len(rows75) == 8491 and np.all((cycles75 >= 76) & (cycles75 <= 79))
                  and np.array_equal(rows75, rows[expected_overlap])
                  and np.array_equal(ids75, ids[expected_overlap])
                  and np.array_equal(cycles75, cycles67[expected_overlap])
                  and np.isin(rows75, rows).all(),
                  "E75 saved predictions are not the exact cycle 76-79 subset of E67")
    pilot.require(readiness["input_sha256"] == freeze["input_sha256"],
                  "Readiness input hash differs from frozen input")
    return {"root_authority_sha256": authority_hash,
        "root_authority_commit": AUTHORITY_COMMIT,
        "ledger_sha256_before_recovery": ledger_hash_before,
        "prelabel_sha256": pilot.sha256_file(pilot.PRELABEL_PATH),
        "readiness_sha256": pilot.sha256_file(pilot.READY_PATH),
        "prior_query_label_open_events": len(opens), "recovery_attempts_before": len(recoveries),
        "rows": int(len(rows)), "query_rows_sha256": pilot.sha256_array(rows),
        "query_ids_sha256": pilot.sha256_array(ids),
        "E75_rows": int(len(rows75)), "E75_is_exact_E67_cycle_76_79_subset": True,
        "fit_starts": len(starts), "fit_saved": len(terminals)}


def run(official_root: Path) -> dict[str, Any]:
    fit_adapter = load_fit_adapter()
    pilot = fit_adapter.load_frozen_driver()
    root = official_root.resolve()
    freeze = pilot.verify_freeze(root)
    fit_adapter.load_audit_core(pilot, freeze)
    source_path = AUDIT_ROOT / SOURCE_REL
    source_text = source_path.read_text(encoding="utf-8")
    tree, patch_details = patch_score_ast(source_text)
    preconditions = validate_preconditions(pilot, freeze)
    amendment, head = verify_recovery_amendment(pilot, freeze, tree, patch_details,
                                                preconditions["root_authority_sha256"])

    pilot.append_event(RUN_ROOT / "fit_ledger.jsonl", {"event": "SCORE_RECOVERY_STARTED",
        "recovery_id": RECOVERY_ID, "candidate_id": freeze["candidate_id"],
        "freeze_signature": freeze["freeze_signature"], "source_commit": freeze["source_commit"],
        "recovery_commit": head, "prelabel_sha256": pilot.sha256_file(pilot.PRELABEL_PATH),
        "prior_query_open_count": preconditions["prior_query_label_open_events"],
        "attempt_number": 2, "same_rows_sha256": preconditions["query_rows_sha256"],
        "same_ids_sha256": preconditions["query_ids_sha256"], "score_formula_changed": False,
        "model_or_predictions_changed": False, "retry_allowed": False})

    namespace: dict[str, Any] = {"__name__": "cls_pos_ef_context_v1_score_recovery",
        "__file__": str(source_path), "SCORE_RECOVERY_AUTHORIZED": True}
    try:
        exec(compile(tree, str(source_path), "exec"), namespace)
        result = namespace["score_metrics"](root)
        score_hash = pilot.sha256_file(pilot.SCORE_PATH)
        verification_hash = pilot.sha256_file(pilot.METRIC_VERIFY_PATH)
        pilot.append_event(RUN_ROOT / "fit_ledger.jsonl", {"event": "SCORE_RECOVERY_SAVED",
            "recovery_id": RECOVERY_ID, "candidate_id": freeze["candidate_id"],
            "freeze_signature": freeze["freeze_signature"], "score_sha256": score_hash,
            "metric_verification_sha256": verification_hash, "query_label_open_count": 2})
        return {"status": "SCORE_RECOVERY_SAVED", "recovery_id": RECOVERY_ID,
            "recovery_commit": head, "patch": patch_details, "preconditions": preconditions,
            "score_sha256": score_hash, "metric_verification_sha256": verification_hash,
            "score": result}
    except BaseException as exc:
        current = pilot.events(RUN_ROOT / "fit_ledger.jsonl")
        opened = len([e for e in current if e.get("event") == "QUERY_LABELS_OPENED"])
        pilot.append_event(RUN_ROOT / "fit_ledger.jsonl", {"event": "SCORE_RECOVERY_FAILED",
            "recovery_id": RECOVERY_ID, "freeze_signature": freeze["freeze_signature"],
            "query_label_open_count": opened, "error_type": type(exc).__name__,
            "error": str(exc), "retry_allowed": False})
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--official-root", required=True)
    args = parser.parse_args()
    print(json.dumps(run(Path(args.official_root)), sort_keys=True, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
