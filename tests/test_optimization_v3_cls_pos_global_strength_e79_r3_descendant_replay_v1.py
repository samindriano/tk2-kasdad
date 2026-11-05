from __future__ import annotations

import json
import subprocess
from contextlib import nullcontext
from pathlib import Path

import pytest

from scripts import optimization_v3_cls_pos_global_strength_e79_r3_descendant_replay_v1 as continuation
from scripts import optimization_v3_cls_pos_global_strength_e79_v3 as r3


def _git(root: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(root), *args], check=True, stdout=subprocess.PIPE,
        stderr=subprocess.PIPE, text=True,
    )
    return result.stdout.strip()


def _commit(root: Path, message: str) -> str:
    _git(root, "add", "--all")
    _git(root, "-c", "user.name=Test", "-c", "user.email=test@example.invalid", "commit", "-m", message)
    return _git(root, "rev-parse", "HEAD")


def _repo(tmp_path: Path) -> tuple[Path, str, str, str]:
    root = tmp_path / "repo"
    root.mkdir()
    _git(root, "init", "-q")
    (root / ".gitattributes").write_bytes(b"* -text\n")
    (root / "attested.txt").write_bytes(b"pinned bytes\n")
    auth = _commit(root, "authorization")
    (root / "attested.txt").write_bytes(b"pinned bytes\n")
    (root / "producer.txt").write_bytes(b"producer\n")
    producer = _commit(root, "producer")
    (root / "unrelated.txt").write_bytes(b"later unrelated change\n")
    descendant = _commit(root, "descendant")
    return root, auth, producer, descendant


def test_accepts_exact_producer_ancestor_with_unrelated_descendant(tmp_path: Path) -> None:
    root, auth, producer, descendant = _repo(tmp_path)

    result = continuation.verify_ancestry(root, producer, auth)
    pinned = continuation.verify_committed_paths(
        root, descendant, {"attested.txt": continuation.sha256_file(root / "attested.txt")},
    )

    assert result["producer_commit"] == producer
    assert result["producer_parent"] == auth
    assert result["invocation_head"] == descendant
    assert pinned == {"attested.txt": continuation.sha256_file(root / "attested.txt")}


def test_rejects_nonancestor_and_wrong_producer_parent(tmp_path: Path) -> None:
    root, auth, producer, _descendant = _repo(tmp_path)
    _git(root, "checkout", "-q", auth)
    (root / "side.txt").write_bytes(b"divergent branch\n")
    side_head = _commit(root, "divergent descendant")

    with pytest.raises(continuation.ContinuationError, match="not an ancestor"):
        continuation.verify_ancestry(root, producer, auth)
    with pytest.raises(continuation.ContinuationError, match="parent changed"):
        continuation.verify_ancestry(root, producer, side_head)


def test_rejects_changed_attested_worktree_or_commit_blob(tmp_path: Path) -> None:
    root, _auth, _producer, descendant = _repo(tmp_path)
    expected = continuation.sha256_file(root / "attested.txt")
    (root / "attested.txt").write_text("altered bytes\n", encoding="utf-8")

    with pytest.raises(continuation.ContinuationError, match="Pinned path differs"):
        continuation.verify_committed_paths(root, descendant, {"attested.txt": expected})


def _fake_attestation() -> dict[str, object]:
    return {
        "status": "PASS_SOURCE_COMPATIBILITY_ATTESTATION_R3_PENDING_REPLAY",
        "attestation_sha256": "synthetic-attestation",
        "current_source_sha256": {"x.py": "hash-x"},
        "saved_inputs_and_fitted_state_receipts": {"immutable_file_sha256": {"Train.csv": "hash-train"}},
    }


def _fake_lineage() -> dict[str, str]:
    return {
        "producer_commit": continuation.PRODUCER_COMMIT,
        "producer_parent": continuation.AUTHORIZATION_COMMIT,
        "invocation_head": "synthetic-descendant-head",
        "prepared_from_head": "synthetic-prepared-head",
        "freeze_sha256": "synthetic-freeze",
        "attestation_file_sha256": "synthetic-attestation-file",
    }


def test_resource_block_does_not_consume_one_shot_attempt(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    monkeypatch.setattr(continuation, "verify_source_freeze", lambda _root: (_fake_attestation(), _fake_lineage()))
    monkeypatch.setattr(r3, "_shared_job_lock", lambda: nullcontext({"acquired": True, "lock": "shared"}))
    monkeypatch.setattr(r3, "_cgeom_local_lock", lambda _root: nullcontext({"acquired": True, "lock": "local"}))
    monkeypatch.setattr(r3, "_resource_snapshot", lambda _stage: {"gate_pass": False, "available_ram_bytes": 1})

    result = continuation.run_feature_replay(root)
    attempt, receipt = continuation._output_paths(root)

    assert result["status"] == "BLOCKED_RESOURCE_GATE"
    assert result["attempt_marker_written"] is False
    assert result["receipt_written"] is False
    assert not attempt.exists()
    assert not receipt.exists()


def test_post_marker_failure_persists_receipt_and_forbids_retry(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    monkeypatch.setattr(continuation, "verify_source_freeze", lambda _root: (_fake_attestation(), _fake_lineage()))
    monkeypatch.setattr(r3, "_shared_job_lock", lambda: nullcontext({"acquired": True, "lock": "shared"}))
    monkeypatch.setattr(r3, "_cgeom_local_lock", lambda _root: nullcontext({"acquired": True, "lock": "local"}))
    monkeypatch.setattr(r3, "_resource_snapshot", lambda _stage: {"gate_pass": True, "available_ram_bytes": 3_000_000_000})
    monkeypatch.setattr(r3, "_execute_feature_only_replay", lambda *_args: (_ for _ in ()).throw(RuntimeError("synthetic failure")))

    result = continuation.run_feature_replay(root)
    attempt, receipt = continuation._output_paths(root)
    saved = json.loads(receipt.read_text(encoding="utf-8"))

    assert result["status"] == "FAILED_FEATURE_ONLY_REPLAY_NO_RETRY"
    assert attempt.is_file() and receipt.is_file()
    body = dict(saved)
    digest = body.pop("receipt_sha256")
    assert digest == continuation.canonical_sha256(body)
    assert saved["error"] == "synthetic failure"
    with pytest.raises(continuation.ContinuationError, match="no retry"):
        continuation.run_feature_replay(root)
