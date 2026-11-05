"""One-shot R3 feature replay from an attested-source descendant checkout.

This adapter preserves the frozen R3 runner and verifier. It changes only the
source-lineage check needed for a later checkout whose exact attested paths and
immutable inputs remain byte-identical.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from contextlib import ExitStack
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from threadpoolctl import threadpool_limits

from scripts import optimization_v3_cls_pos_global_strength_e79_v3 as r3
from scripts import verify_optimization_v3_cls_pos_global_strength_e79_v3 as verifier


ROOT = Path(__file__).resolve().parents[1]
CONTINUATION_ID = "CLS_POS_GLOBAL_STRENGTH_E79_R3_DESCENDANT_REPLAY_V1"
PRODUCER_COMMIT = "f6eb5938f9884a3b154ebd4bb6a5daa66ee2239d"
AUTHORIZATION_COMMIT = "71192b65c4ba0b1e31fd9273476d00b28f96b343"
PROTOCOL_REL = "reports/optimization_v3/cls_pos_global_strength_e79_v3/DESCENDANT_REPLAY_CONTINUATION.md"
FREEZE_REL = "reports/optimization_v3/cls_pos_global_strength_e79_v3/DESCENDANT_REPLAY_FREEZE.json"
WRAPPER_REL = "scripts/optimization_v3_cls_pos_global_strength_e79_r3_descendant_replay_v1.py"
TEST_REL = "tests/test_optimization_v3_cls_pos_global_strength_e79_r3_descendant_replay_v1.py"
ATTRIBUTES_REL = ".gitattributes"
CONTINUATION_FILE_PATHS = (ATTRIBUTES_REL, PROTOCOL_REL, WRAPPER_REL, TEST_REL)
OUTPUT_ROOT_REL = "experiments/runs/optimization_v3_cls_pos_global_strength_e79_v3/descendant_replay_v1"
ATTEMPT_NAME = "ATTEMPT.started.json"
RECEIPT_NAME = "REPLAY_RECEIPT.json"


class ContinuationError(RuntimeError):
    """Raised when the descendant continuation contract is not met."""


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False,
                         separators=(",", ":")).encode("utf-8")
    return sha256_bytes(payload)


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ContinuationError(f"Expected a JSON object: {path}")
    return value


def _git(root: Path, *args: str, check: bool = True) -> str:
    completed = subprocess.run(
        ["git", "-C", str(root), *args], check=False, stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if check and completed.returncode != 0:
        detail = completed.stderr.decode("utf-8", errors="replace").strip()
        raise ContinuationError(f"git {' '.join(args)} failed: {detail}")
    return completed.stdout.decode("utf-8", errors="strict").strip()


def _git_blob_sha256(root: Path, commit: str, relative: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(root), "show", f"{commit}:{relative}"],
        check=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    if completed.returncode != 0:
        detail = completed.stderr.decode("utf-8", errors="replace").strip()
        raise ContinuationError(f"Committed path unavailable at {commit}:{relative}: {detail}")
    return sha256_bytes(completed.stdout)


def verify_ancestry(root: Path, producer_commit: str = PRODUCER_COMMIT,
                    authorization_commit: str = AUTHORIZATION_COMMIT) -> dict[str, str]:
    """Require the exact producer parent and producer ancestry, not live HEAD equality."""
    parent = _git(root, "rev-parse", f"{producer_commit}^")
    if parent != authorization_commit:
        raise ContinuationError(
            f"Producer parent changed: expected {authorization_commit}, got {parent}"
        )
    head = _git(root, "rev-parse", "HEAD")
    completed = subprocess.run(
        ["git", "-C", str(root), "merge-base", "--is-ancestor", producer_commit, head],
        check=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    if completed.returncode != 0:
        raise ContinuationError(f"Producer {producer_commit} is not an ancestor of HEAD {head}")
    return {"producer_commit": producer_commit, "producer_parent": parent,
            "invocation_head": head}


def verify_committed_paths(root: Path, commit: str,
                           expected_sha256: dict[str, str]) -> dict[str, str]:
    """Require each pinned path to match both the worktree and selected commit."""
    verified: dict[str, str] = {}
    for relative, expected in sorted(expected_sha256.items()):
        path = root / Path(relative)
        try:
            path.resolve().relative_to(root.resolve())
        except ValueError as exc:
            raise ContinuationError(f"Pinned path escapes repository: {relative}") from exc
        if not path.is_file():
            raise ContinuationError(f"Pinned path is missing: {relative}")
        worktree_hash = sha256_file(path)
        tree_hash = _git_blob_sha256(root, commit, relative)
        if worktree_hash != expected or tree_hash != expected:
            raise ContinuationError(
                f"Pinned path differs in worktree or commit: {relative}; "
                f"expected={expected}, worktree={worktree_hash}, tree={tree_hash}"
            )
        verified[relative] = expected
    return verified


def _tracked_worktree_clean(root: Path) -> None:
    status = _git(root, "status", "--porcelain", "--untracked-files=no")
    if status:
        raise ContinuationError(f"Tracked worktree must be clean before replay: {status}")


def build_freeze(root: Path = ROOT) -> dict[str, Any]:
    """Create the immutable pre-replay freeze after protocol, wrapper, and tests are committed."""
    _tracked_worktree_clean(root)
    lineage = verify_ancestry(root)
    attestation = r3._verify_attestation(root)
    prepared_head = _git(root, "rev-parse", "HEAD")
    continuation_paths = {
        relative: sha256_file(root / relative)
        for relative in CONTINUATION_FILE_PATHS
    }
    verify_committed_paths(root, prepared_head, continuation_paths)
    attestation_file_sha256 = sha256_file(root / r3.ATTESTATION_REL)
    verify_committed_paths(root, prepared_head, {r3.ATTESTATION_REL: attestation_file_sha256})
    body = {
        "schema_version": 1,
        "continuation_id": CONTINUATION_ID,
        "status": "FROZEN_FOR_ONE_SHOT_DESCENDANT_REPLAY",
        "prepared_from_head": prepared_head,
        "producer_commit": lineage["producer_commit"],
        "authorization_commit": lineage["producer_parent"],
        "attestation_path": r3.ATTESTATION_REL,
        "attestation_file_sha256": attestation_file_sha256,
        "attestation_sha256": attestation["attestation_sha256"],
        "attested_source_sha256": dict(attestation["current_source_sha256"]),
        "immutable_input_sha256": dict(attestation["saved_inputs_and_fitted_state_receipts"]["immutable_file_sha256"]),
        "continuation_file_sha256": continuation_paths,
        "official_feature_train_sha256": r3.TRAIN_SHA256,
        "official_feature_train_rows": r3.EXPECTED_ROWS,
        "target_columns_forbidden": ["artifact_signal", "threat_rank"],
        "test_and_sample_forbidden": True,
        "estimator_fit_and_inference_forbidden": True,
        "one_shot_output_root": OUTPUT_ROOT_REL,
    }
    payload = dict(body)
    payload["freeze_sha256"] = canonical_sha256(body)
    path = root / FREEZE_REL
    if path.exists():
        raise ContinuationError(f"Refusing to replace an existing continuation freeze: {FREEZE_REL}")
    r3._write_once(path, (json.dumps(payload, sort_keys=True, ensure_ascii=False, allow_nan=False, indent=2) + "\n").encode("utf-8"))
    return payload


def verify_source_freeze(root: Path = ROOT) -> tuple[dict[str, Any], dict[str, Any]]:
    """Verify the committed continuation freeze, R3 attestation, inputs, and ancestry."""
    _tracked_worktree_clean(root)
    lineage = verify_ancestry(root)
    freeze_path = root / FREEZE_REL
    if not freeze_path.is_file():
        raise ContinuationError(f"Continuation freeze is missing: {FREEZE_REL}")
    freeze = read_json(freeze_path)
    freeze_body = dict(freeze)
    supplied_digest = freeze_body.pop("freeze_sha256", None)
    if supplied_digest != canonical_sha256(freeze_body):
        raise ContinuationError("Continuation freeze self-digest is invalid")
    if freeze.get("continuation_id") != CONTINUATION_ID:
        raise ContinuationError("Unexpected continuation freeze identity")
    if freeze.get("status") != "FROZEN_FOR_ONE_SHOT_DESCENDANT_REPLAY":
        raise ContinuationError("Continuation freeze status is not the expected pre-replay state")
    if freeze.get("producer_commit") != PRODUCER_COMMIT or freeze.get("authorization_commit") != AUTHORIZATION_COMMIT:
        raise ContinuationError("Continuation freeze source lineage differs from authorization")
    frozen_head = freeze.get("prepared_from_head")
    if not isinstance(frozen_head, str) or not frozen_head:
        raise ContinuationError("Freeze does not name its prepared source commit")
    is_ancestor = subprocess.run(
        ["git", "-C", str(root), "merge-base", "--is-ancestor", frozen_head, lineage["invocation_head"]],
        check=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    if is_ancestor.returncode != 0:
        raise ContinuationError("Prepared continuation source commit is not an ancestor of invocation HEAD")
    if _git(root, "rev-parse", "--verify", f"{frozen_head}^{{commit}}", check=False) == "":
        raise ContinuationError("Prepared source commit is not available in repository history")

    attestation = r3._verify_attestation(root)
    attestation_path = root / r3.ATTESTATION_REL
    if sha256_file(attestation_path) != freeze.get("attestation_file_sha256"):
        raise ContinuationError("R3 attestation file hash differs from the continuation freeze")
    if attestation.get("attestation_sha256") != freeze.get("attestation_sha256"):
        raise ContinuationError("R3 attestation internal digest differs from the continuation freeze")
    if (
        freeze.get("official_feature_train_sha256") != r3.TRAIN_SHA256
        or freeze.get("official_feature_train_rows") != r3.EXPECTED_ROWS
        or freeze.get("target_columns_forbidden") != ["artifact_signal", "threat_rank"]
        or freeze.get("test_and_sample_forbidden") is not True
        or freeze.get("estimator_fit_and_inference_forbidden") is not True
        or freeze.get("one_shot_output_root") != OUTPUT_ROOT_REL
    ):
        raise ContinuationError("Continuation freeze does not preserve the approved Train-only replay boundary")

    attested_sources = dict(attestation.get("current_source_sha256", {}))
    expected_sources = freeze.get("attested_source_sha256")
    if expected_sources != attested_sources:
        raise ContinuationError("Frozen current-source manifest differs from R3 attestation")
    verified_sources = verify_committed_paths(root, lineage["invocation_head"], attested_sources)
    verify_committed_paths(
        root, lineage["invocation_head"],
        {r3.ATTESTATION_REL: str(freeze["attestation_file_sha256"])},
    )

    continuation_paths = freeze.get("continuation_file_sha256")
    if not isinstance(continuation_paths, dict) or set(continuation_paths) != set(CONTINUATION_FILE_PATHS):
        raise ContinuationError("Continuation freeze must bind attributes, protocol, wrapper, and tests")
    verified_continuation = verify_committed_paths(
        root, lineage["invocation_head"], continuation_paths,
    )
    verify_committed_paths(root, frozen_head, continuation_paths)
    verify_committed_paths(root, frozen_head, {r3.ATTESTATION_REL: str(freeze["attestation_file_sha256"])})
    verify_committed_paths(root, lineage["invocation_head"], {FREEZE_REL: sha256_file(freeze_path)})
    if freeze.get("immutable_input_sha256") != attestation["saved_inputs_and_fitted_state_receipts"]["immutable_file_sha256"]:
        raise ContinuationError("Frozen immutable-input manifest differs from the R3 attestation")
    return attestation, {
        **lineage,
        "prepared_from_head": frozen_head,
        "freeze_sha256": supplied_digest,
        "attestation_file_sha256": freeze["attestation_file_sha256"],
        "attestation_sha256": attestation["attestation_sha256"],
        "attested_source_files_verified": len(verified_sources),
        "continuation_files_verified": len(verified_continuation),
        "immutable_input_files_verified": len(freeze["immutable_input_sha256"]),
    }


def _output_paths(root: Path) -> tuple[Path, Path]:
    out_root = root / OUTPUT_ROOT_REL
    return out_root / ATTEMPT_NAME, out_root / RECEIPT_NAME


def preflight(root: Path = ROOT) -> dict[str, Any]:
    attestation, lineage = verify_source_freeze(root)
    attempt_path, receipt_path = _output_paths(root)
    if attempt_path.exists() or receipt_path.exists():
        raise ContinuationError("Descendant replay already has an attempt marker or receipt; no retry")
    locks: list[dict[str, Any]] = []
    with ExitStack() as stack:
        locks.append(stack.enter_context(r3._shared_job_lock()))
        locks.append(stack.enter_context(r3._cgeom_local_lock(root)))
        resource = r3._resource_snapshot("descendant continuation read-only preflight")
    return {
        "status": "PASS_PREFLIGHT" if resource["gate_pass"] else "BLOCKED_RESOURCE_GATE",
        "continuation_id": CONTINUATION_ID,
        "lineage": lineage,
        "locks": locks,
        "resource": resource,
        "attempt_marker_exists": attempt_path.exists(),
        "receipt_exists": receipt_path.exists(),
        "attestation_status": attestation.get("status"),
        "no_feature_transform_or_replay_started": True,
    }


def run_feature_replay(root: Path = ROOT) -> dict[str, Any]:
    attestation, lineage = verify_source_freeze(root)
    attempt_path, receipt_path = _output_paths(root)
    if attempt_path.exists() or receipt_path.exists():
        raise ContinuationError("Descendant replay already has an attempt marker or receipt; no retry")
    locks: list[dict[str, Any]] = []
    with ExitStack() as stack:
        locks.append(stack.enter_context(r3._shared_job_lock()))
        locks.append(stack.enter_context(r3._cgeom_local_lock(root)))
        resource_before = r3._resource_snapshot("immediately before one descendant feature-only replay")
        if not resource_before["gate_pass"]:
            return {
                "status": "BLOCKED_RESOURCE_GATE",
                "continuation_id": CONTINUATION_ID,
                "lineage": lineage,
                "locks": locks,
                "resource_before": resource_before,
                "attempt_marker_written": False,
                "receipt_written": False,
                "feature_replay_started": False,
            }

        attempt_body = {
            "schema_version": 1,
            "continuation_id": CONTINUATION_ID,
            "status": "ONE_REPLAY_ATTEMPT_STARTED",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "producer_commit": PRODUCER_COMMIT,
            "authorization_commit": AUTHORIZATION_COMMIT,
            "invocation_head": lineage["invocation_head"],
            "prepared_from_head": lineage["prepared_from_head"],
            "freeze_sha256": lineage["freeze_sha256"],
            "attestation_sha256": attestation["attestation_sha256"],
            "locks": locks,
            "resource_before": resource_before,
            "no_retry_if_interrupted": True,
        }
        r3._write_once(
            attempt_path,
            (json.dumps(attempt_body, sort_keys=True, ensure_ascii=False, allow_nan=False, indent=2) + "\n").encode("utf-8"),
        )
        started = time.perf_counter()
        commit_info = {"head": PRODUCER_COMMIT, "parent": AUTHORIZATION_COMMIT}
        receipt_body: dict[str, Any]
        try:
            with threadpool_limits(limits=r3.MAX_THREADS):
                receipt_body = r3._execute_feature_only_replay(
                    root, attestation, commit_info, locks, resource_before,
                )
            receipt_body.update({
                "continuation_id": CONTINUATION_ID,
                "source_invocation_head": lineage["invocation_head"],
                "continuation_prepared_from_head": lineage["prepared_from_head"],
                "continuation_freeze_sha256": lineage["freeze_sha256"],
                "continuation_attestation_file_sha256": lineage["attestation_file_sha256"],
                "descendant_source_lineage_verified": True,
            })
        except Exception as exc:  # noqa: BLE001 - durable no-retry failure is required after the marker
            receipt_body = {
                "schema_version": 1,
                "replay_id": "CLS_POS_GLOBAL_STRENGTH_E79_TARGET_FREE_FEATURE_REPLAY_V3",
                "continuation_id": CONTINUATION_ID,
                "status": "FAILED_FEATURE_ONLY_REPLAY_NO_RETRY",
                "started_at_utc": attempt_body["created_at_utc"],
                "source_commit": PRODUCER_COMMIT,
                "source_commit_parent": AUTHORIZATION_COMMIT,
                "source_invocation_head": lineage["invocation_head"],
                "continuation_prepared_from_head": lineage["prepared_from_head"],
                "continuation_freeze_sha256": lineage["freeze_sha256"],
                "source_compatibility_attestation_sha256": attestation["attestation_sha256"],
                "source_sha256": attestation["current_source_sha256"],
                "input_sha256": attestation["saved_inputs_and_fitted_state_receipts"]["immutable_file_sha256"],
                "locks": locks,
                "resource_before": resource_before,
                "target_column_requested_or_loaded": False,
                "test_or_sample_loaded": False,
                "estimator_fit_calls": 0,
                "prediction_or_inference_calls": 0,
                "new_df_map_created": False,
                "new_gamma_created": False,
                "csv_created": False,
                "error_type": type(exc).__name__,
                "error": str(exc),
                "resource_after_failure": r3._resource_snapshot("after failed descendant feature replay"),
            }
        receipt_body["elapsed_seconds"] = float(time.perf_counter() - started)
        receipt_body["descendant_source_lineage_verified"] = True
        return r3.write_and_readback_receipt(receipt_path, receipt_body)


def verify_persisted_replay(root: Path = ROOT) -> dict[str, Any]:
    attempt_path, receipt_path = _output_paths(root)
    if not attempt_path.is_file():
        raise ContinuationError("Descendant replay attempt marker is missing")
    if not receipt_path.is_file():
        raise ContinuationError("Descendant replay receipt is missing")
    attempt = read_json(attempt_path)
    if attempt.get("continuation_id") != CONTINUATION_ID or attempt.get("no_retry_if_interrupted") is not True:
        raise ContinuationError("Descendant replay attempt marker is invalid")
    receipt = read_json(receipt_path)
    body = dict(receipt)
    supplied = body.pop("receipt_sha256", None)
    if supplied != canonical_sha256(body):
        raise ContinuationError("Descendant replay receipt self-digest is invalid")
    if receipt.get("continuation_id") != CONTINUATION_ID:
        raise ContinuationError("Persisted receipt has wrong continuation identity")
    if receipt.get("descendant_source_lineage_verified") is not True:
        raise ContinuationError("Persisted receipt does not record verified descendant lineage")
    result = verifier.verify_replay_receipt(root, receipt_path)
    return {
        **result,
        "continuation_id": CONTINUATION_ID,
        "attempt_marker_sha256": sha256_file(attempt_path),
        "receipt_file_sha256": sha256_file(receipt_path),
        "source_invocation_head": receipt.get("source_invocation_head"),
        "continuation_freeze_sha256": receipt.get("continuation_freeze_sha256"),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--build-freeze", action="store_true", help="write the pre-replay freeze after code commit")
    group.add_argument("--preflight", action="store_true", help="read-only lineage, lock, and RAM gate")
    group.add_argument("--feature-replay", action="store_true", help="run the one-shot target-free replay")
    group.add_argument("--verify-replay", action="store_true", help="independently verify the saved receipt")
    args = parser.parse_args(argv)
    try:
        if args.build_freeze:
            result = build_freeze()
        elif args.preflight:
            result = preflight()
        elif args.feature_replay:
            result = run_feature_replay()
        else:
            result = verify_persisted_replay()
    except (ContinuationError, r3.ReplayError, r3.SharedLockHeld, r3.LocalLockHeld) as exc:
        print(json.dumps({"status": "BLOCKED_OR_FAILED", "error_type": type(exc).__name__, "error": str(exc)}, indent=2))
        return 2
    print(json.dumps(result, sort_keys=True, ensure_ascii=False, indent=2))
    return 0 if result.get("status", "").startswith("PASS") else 2


if __name__ == "__main__":
    raise SystemExit(main())
