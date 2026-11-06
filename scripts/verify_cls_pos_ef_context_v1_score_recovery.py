"""Read-only, one-shot-aware verification of the authorized score recovery."""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path
import subprocess
from typing import Any

AUDIT_ROOT = Path(__file__).resolve().parents[1]
REPORT_ROOT = AUDIT_ROOT / "reports/simplification_v1/CLS_POS_EF_CONTEXT_V1"
RUN_ROOT = AUDIT_ROOT / "experiments/CLS_POS_EF_CONTEXT_V1"
SOURCE_REL = "scripts/verify_cls_pos_ef_context_v1.py"
VERIFY_REL = "scripts/verify_cls_pos_ef_context_v1_score_recovery.py"
AMENDMENT_PATH = REPORT_ROOT / "SCORE_RECOVERY_AMENDMENT.json"
AUTHORITY_PATH = REPORT_ROOT / "ROOT_SCORE_RECOVERY_R1.json"
OUTPUT_PATH = REPORT_ROOT / "SCORE_RECOVERY_VERIFICATION.json"
AUTHORITY_COMMIT = "bfdde7915f80b31ef665e3be804dc0fd412b9c44"
AUTHORITY_SHA256 = "fea379ed88f854db954162677613b115f1060453d5f6c0982c6acc9defe5421c"
AUTHORIZATION_SIGNATURE = "3c9827509dd7ea202ef1aabc5d01893d8e614314b035a68511c8e1e075038ec1"
RECOVERY_ID = "CLS_POS_EF_CONTEXT_V1_SCORE_RECOVERY_01"


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(Path(path).read_bytes())


def git(*args: str) -> str:
    return subprocess.check_output(["git", "-C", str(AUDIT_ROOT), *args], text=True).strip()


def require(ok: bool, message: str) -> None:
    if not ok:
        raise RuntimeError(message)


def patch_verify_score_ast(source_text: str) -> tuple[ast.Module, dict[str, Any]]:
    tree = ast.parse(source_text, filename=SOURCE_REL)
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "verify_score"]
    require(len(functions) == 1, "Expected one frozen verify_score function")
    function = functions[0]
    matches = []
    for node in ast.walk(function):
        if not isinstance(node, ast.If):
            continue
        if any(isinstance(child, ast.Constant) and child.value == "Query-label accounting differs from the single committed prelabel pass"
               for child in ast.walk(node)):
            matches.append(node)
    require(len(matches) == 1, "Expected the exact frozen single-read accounting guard")
    matches[0].test = ast.UnaryOp(op=ast.Not(), operand=ast.Name(id="RECOVERY_LEDGER_VALIDATED", ctx=ast.Load()))
    read_counts = [n for n in ast.walk(function) if isinstance(n, ast.Dict)
                   for k, v in zip(n.keys, n.values)
                   if isinstance(k, ast.Constant) and k.value == "query_label_reads"]
    require(len(read_counts) == 1, "Expected one query-read count in the frozen result")
    result_dict = read_counts[0]
    index = next(i for i, k in enumerate(result_dict.keys)
                 if isinstance(k, ast.Constant) and k.value == "query_label_reads")
    result_dict.values[index] = ast.Call(func=ast.Name(id="len", ctx=ast.Load()),
                                        args=[ast.Name(id="opened", ctx=ast.Load())], keywords=[])
    ast.fix_missing_locations(tree)
    return tree, {
        "source_path": SOURCE_REL,
        "ledger_guard_replacement_count": 1,
        "reported_query_label_reads": "len(opened): failed original plus authorized recovery",
        "replacement": "require the recovery wrapper's exact two-event/same-row ledger validation",
        "metric_formula_or_gate_changed": False,
        "patched_ast_sha256": sha256_bytes(ast.dump(tree, annotate_fields=True,
            include_attributes=False).encode("utf-8")),
    }


def load_and_verify_authority() -> tuple[dict[str, Any], str]:
    raw = AUTHORITY_PATH.read_bytes()
    digest = sha256_bytes(raw)
    require(digest == AUTHORITY_SHA256, "Root recovery authority hash mismatch")
    committed = subprocess.check_output(["git", "-C", str(AUDIT_ROOT), "show",
                                         f"{AUTHORITY_COMMIT}:{AUTHORITY_PATH.relative_to(AUDIT_ROOT).as_posix()}"])
    require(committed == raw, "Root recovery authority differs from its approved commit")
    subprocess.check_call(["git", "-C", str(AUDIT_ROOT), "merge-base", "--is-ancestor",
                           AUTHORITY_COMMIT, "HEAD"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    authority = json.loads(raw.decode("utf-8"))
    require(authority.get("status") == "APPROVED_ONE_SAME_INPUT_SCORE_RECOVERY"
            and authority.get("authorization_signature") == AUTHORIZATION_SIGNATURE
            and authority.get("allowed_scoring_recovery_read_count") == 1
            and authority.get("new_fits_allowed") == 0
            and authority.get("test_or_sample_access_allowed") is False
            and authority.get("kaggle_access_allowed") is False
            and authority.get("metric_qa_same_row_reconstruction_allowed") is True,
            "Root authority does not permit this read-only verification")
    return authority, digest


def validate_recovery_ledger(pilot, authority: dict[str, Any]) -> dict[str, Any]:
    events = pilot.events(pilot.RUN_ROOT / "fit_ledger.jsonl")
    opened = [e for e in events if e.get("event") == "QUERY_LABELS_OPENED"]
    expected_open = authority["failed_original_query_event"]
    require(len(opened) == 2 and opened[0] == expected_open and opened[1] == expected_open,
            "The only two label-open events must be the authorized identical row/ID read")
    started = [e for e in events if e.get("event") == "SCORE_RECOVERY_STARTED"]
    saved = [e for e in events if e.get("event") == "SCORE_RECOVERY_SAVED"]
    failed = [e for e in events if e.get("event") == "SCORE_RECOVERY_FAILED"]
    require(len(started) == len(saved) == 1 and not failed,
            "Recovery ledger must contain one successful attempt and no failed/repeated attempts")
    start, terminal = started[0], saved[0]
    require(start.get("recovery_id") == terminal.get("recovery_id") == RECOVERY_ID
            and start.get("candidate_id") == terminal.get("candidate_id") == authority["candidate_id"]
            and start.get("freeze_signature") == terminal.get("freeze_signature")
            and start.get("same_rows_sha256") == authority["query_rows_sha256"]
            and start.get("same_ids_sha256") == authority["query_ids_sha256"]
            and start.get("prior_query_open_count") == 1 and start.get("attempt_number") == 2
            and start.get("retry_allowed") is False
            and start.get("score_formula_changed") is False
            and start.get("model_or_predictions_changed") is False
            and terminal.get("query_label_open_count") == 2,
            "Recovery event fields do not match the exact authorized transition")
    starts = [e for e in events if e.get("event") == "FIT_STARTED"]
    fit_terminals = [e for e in events if e.get("event") in {"FIT_SAVED", "FIT_FAILED"}]
    freeze = pilot.read_json(pilot.FREEZE_PATH)
    require([e.get("fit_id") for e in starts] == freeze["fit_ids"]
            and [e.get("event") for e in fit_terminals] == ["FIT_SAVED"] * 4,
            "Fit accounting changed during score recovery")
    score_hash = pilot.sha256_file(pilot.SCORE_PATH)
    metric_hash = pilot.sha256_file(pilot.METRIC_VERIFY_PATH)
    require(terminal.get("score_sha256") == score_hash
            and terminal.get("metric_verification_sha256") == metric_hash,
            "Recovery-saved event does not bind score and metric output bytes")
    score = pilot.read_json(pilot.SCORE_PATH)
    require(score.get("prelabel_sha256") == authority["committed_prelabel_sha256"]
            and score.get("query_labels_opened") is True
            and score.get("test_or_sample_access") is False,
            "Recovered score violates the committed prelabel or source boundary")
    return {
        "status": "PASS_RECOVERY_LEDGER",
        "total_label_open_events": len(opened),
        "original_failed_attempt_reads": 1,
        "authorized_recovery_reads": 1,
        "recovery_attempt_count": 1,
        "fit_started": len(starts),
        "fit_saved": len(fit_terminals),
        "ledger_sha256": pilot.sha256_file(pilot.RUN_ROOT / "fit_ledger.jsonl"),
        "score_sha256": score_hash,
        "metric_verification_sha256": metric_hash,
    }


def run() -> dict[str, Any]:
    from scripts import cls_pos_ef_context_v1 as pilot

    authority, authority_hash = load_and_verify_authority()
    amendment = json.loads(AMENDMENT_PATH.read_text(encoding="utf-8"))
    require(amendment.get("root_authority_sha256") == authority_hash
            and amendment.get("root_authority_commit") == AUTHORITY_COMMIT
            and amendment.get("authorization_signature") == AUTHORIZATION_SIGNATURE,
            "Recovery amendment does not bind the approved authority")
    wrapper_hash = sha256_file(AUDIT_ROOT / VERIFY_REL)
    require(amendment.get("metric_verifier_wrapper_sha256") == wrapper_hash,
            "Recovery verification wrapper hash differs from its committed amendment")
    source_path = AUDIT_ROOT / SOURCE_REL
    frozen_verifier_hash = sha256_file(source_path)
    require(amendment.get("frozen_metric_verifier_sha256") == frozen_verifier_hash,
            "Frozen metric verifier changed since recovery was authorized")
    source = source_path.read_text(encoding="utf-8")
    tree, patch = patch_verify_score_ast(source)
    require(patch["patched_ast_sha256"] == amendment.get("patched_metric_verifier_ast_sha256"),
            "Deterministic metric guard adapter differs from the committed amendment")
    recovery_report = validate_recovery_ledger(pilot, authority)
    namespace: dict[str, Any] = {"__name__": "cls_pos_ef_context_v1_score_recovery_verify",
        "__file__": str(source_path), "RECOVERY_LEDGER_VALIDATED": True}
    exec(compile(tree, str(source_path), "exec"), namespace)
    metric_result = namespace["verify_score"]()
    output = {
        "schema_version": 1,
        "status": "PASS_RECOVERED_SCORE_AND_INDEPENDENT_METRIC_QA",
        "recovery_id": RECOVERY_ID,
        "authorization_signature": AUTHORIZATION_SIGNATURE,
        "root_authority_sha256": authority_hash,
        "recovery_ledger": recovery_report,
        "frozen_metric_verification": metric_result,
        "metric_verifier_adapter": patch,
        "metric_formula_or_gate_changed": False,
        "label_reads_during_verification": 0,
        "test_or_sample_access": False,
        "kaggle_access": False,
    }
    output["verification_signature"] = pilot.canonical_sha256(output)
    pilot.write_once(OUTPUT_PATH, output)
    return output


def main() -> int:
    result = run()
    print(json.dumps(result, sort_keys=True, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
