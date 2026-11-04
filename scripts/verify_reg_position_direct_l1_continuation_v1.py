"""Scoped, versioned continuation policy for the frozen REG_POSITION_DIRECT_L1_V1 pilot."""
from __future__ import annotations

import copy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
from typing import Any, Iterable, Mapping

ROOT = Path(__file__).resolve().parents[1]
import scripts.reg_position_direct_l1_v1 as pilot  # noqa: E402
import scripts.verify_reg_position_direct_l1_v1 as original_verifier  # noqa: E402

CARD_ID = pilot.CARD_ID
CONTINUATION_REL = pilot.REPORT_REL / "CONTINUATION_V1.json"
CONTINUATION_PATH = ROOT / CONTINUATION_REL
COMPLETION_REL = pilot.REPORT_REL / "CONTINUATION_COMPLETION_V1.json"
COMPLETION_PATH = ROOT / COMPLETION_REL
CONTINUATION_PRELABEL_REL = pilot.REPORT_REL / "CONTINUATION_PRELABEL_VERIFICATION_V1.json"
CONTINUATION_PRELABEL_PATH = ROOT / CONTINUATION_PRELABEL_REL
CONTINUATION_FINAL_REL = pilot.REPORT_REL / "CONTINUATION_FINAL_VERIFICATION_V1.json"
CONTINUATION_FINAL_PATH = ROOT / CONTINUATION_FINAL_REL
BUDGET_REL = "experiments/runs/optimization_v3_research_next_v1/budget.jsonl"
BUDGET_PREFIX_LINE_COUNT = 361
BUDGET_AMENDMENT_ID = "CLS_BIGRAM_POSITION_PILOT_V1_CAP149_2026_11_04"
BUDGET_CARD_ID = "CLS_BIGRAM_POSITION_PILOT_V1"
BUDGET_STAGE = "SIX_FIT_TEMPORAL_PILOT"
BUDGET_RESERVATION_KEY = f"{BUDGET_CARD_ID}:{BUDGET_STAGE}"
BUDGET_FREEZE_SIGNATURE = "63bb97834144f50e27a5bd6bf90603f04b8c303e7c234401c7e988a1cf71cebb"
BUDGET_SOURCE_COMMIT = "f2b39043afe612030ab33e24338c24b8e3d89594"
BUDGET_AUTHORIZATION = "reports/optimization_v3/bigram_position_readiness_v1/SOL_PILOT_AUTHORIZATION.md"
BUDGET_FIT_IDS = tuple(
    f"{BUDGET_CARD_ID}_E{origin}_{arm}"
    for origin in (67, 75) for arm in ("BAG", "REAL", "SHUFFLED")
)
BUDGET_SUCCESSOR_REASON = "prefit ID arrays changed from object to Unicode; no fit had started"
BUDGET_RUN_REL = "experiments/runs/optimization_v3_research_next_v1/BIGRAM_POSITION_PILOT_V1"

# These four status entries reconstruct the V4 prefit snapshot. Their canonical
# hash is checked against the immutable historical hash stored in FREEZE_V4.
BASELINE_STATUS_LINES = (
    "?? scripts/optimization_v3_bigram_position_pilot_v1.py",
    "?? scripts/verify_optimization_v3_bigram_position_pilot_v1.py",
    "?? src/models/optimization_v3_bigram_position_pilot_v1.py",
    "?? tests/test_optimization_v3_bigram_position_pilot_v1.py",
)

AUTHORIZED_MAIN_PATHS = frozenset({
    "reports/optimization_v3/bigram_position_pilot_v1",
    "scripts/optimization_v3_bigram_position_pilot_v1.py",
    "scripts/verify_optimization_v3_bigram_position_pilot_v1.py",
    "src/models/optimization_v3_bigram_position_pilot_v1.py",
    "tests/test_optimization_v3_bigram_position_pilot_v1.py",
})

LOCAL_COMMIT_PATHS = frozenset({
    "scripts/reg_position_direct_l1_v1.py",
    "scripts/verify_reg_position_direct_l1_v1.py",
    "scripts/continue_reg_position_direct_l1_v1.py",
    "scripts/verify_reg_position_direct_l1_continuation_v1.py",
    "tests/test_reg_position_direct_l1_v1.py",
    "tests/test_reg_position_direct_l1_continuation_v1.py",
    "reports/optimization_v3/reg_position_direct_l1_v1",
})


class ContinuationError(RuntimeError):
    """Fail-closed continuation audit error."""


def _read_json(path: Path) -> dict[str, Any]:
    try:
        result = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise ContinuationError(f"Invalid continuation JSON: {path}") from exc
    if not isinstance(result, dict):
        raise ContinuationError(f"Expected a JSON object: {path}")
    return result


def _body_signature(value: Mapping[str, Any], signature_key: str) -> str:
    body = {key: item for key, item in value.items() if key != signature_key}
    return pilot._canonical_sha(body)


def _status_path(line: str) -> str:
    if len(line) < 4 or line[2] != " ":
        raise ContinuationError(f"Unrecognized git status entry: {line!r}")
    return line[3:].replace("\\", "/").rstrip("/")


def _normalize_paths(paths: Iterable[str]) -> set[str]:
    return {str(path).replace("\\", "/").strip("/") for path in paths if str(path).strip("/")}


def _within_allowed(path: str, allowed: set[str]) -> bool:
    return any(path == prefix or path.startswith(prefix + "/") for prefix in allowed)


def _require_digest_match(expected: str, actual: str, label: str) -> None:
    if not expected or expected != actual:
        raise ContinuationError(f"Pinned digest mismatch: {label}")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _require_event_fields(event: Mapping[str, Any], fields: set[str], label: str) -> None:
    allowed = fields | {"at_utc"}
    if set(event) != allowed and set(event) != fields:
        raise ContinuationError(f"Unexpected {label} event fields: {sorted(set(event) ^ allowed)}")
    timestamp = event.get("at_utc")
    if timestamp is not None:
        if not isinstance(timestamp, str):
            raise ContinuationError(f"Invalid timestamp in {label} event")
        try:
            datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
        except ValueError as exc:
            raise ContinuationError(f"Invalid timestamp in {label} event") from exc


def validate_budget_extension(
    raw: bytes,
    *,
    expected_prefix_sha256: str,
    expected_prefix_line_count: int = BUDGET_PREFIX_LINE_COUNT,
    valid_source_commits: Iterable[str] | None = None,
) -> dict[str, Any]:
    """Validate the exact V4 ledger prefix and only the authorized Bigram suffix."""
    lines = raw.splitlines(keepends=True)
    if (len(lines) < expected_prefix_line_count
            or any(not line.endswith(b"\n") or line.endswith(b"\r\n") for line in lines)):
        raise ContinuationError("Budget ledger has a missing line or an unexpected line ending")
    prefix = b"".join(lines[:expected_prefix_line_count])
    prefix_sha256 = _sha256_bytes(prefix)
    if prefix_sha256 != expected_prefix_sha256:
        raise ContinuationError("Original budget ledger raw prefix changed")
    if any(not line.strip() for line in lines):
        raise ContinuationError("Budget ledger contains an empty physical line")

    events: list[dict[str, Any]] = []
    for index, line in enumerate(lines[expected_prefix_line_count:], expected_prefix_line_count + 1):
        try:
            event = json.loads(line.decode("utf-8"))
        except Exception as exc:
            raise ContinuationError(f"Invalid appended budget event at line {index}") from exc
        if not isinstance(event, dict) or not isinstance(event.get("event"), str):
            raise ContinuationError(f"Invalid appended budget event at line {index}")
        events.append(event)

    valid_commits = None if valid_source_commits is None else set(valid_source_commits)

    def check_source_commit(commit: Any, *, initial: bool = False) -> None:
        if not isinstance(commit, str) or len(commit) != 40 or any(ch not in "0123456789abcdef" for ch in commit):
            raise ContinuationError("Malformed Bigram source commit in appended budget event")
        if initial and commit != BUDGET_SOURCE_COMMIT:
            raise ContinuationError("Bigram reservation does not use its authorized source commit")
        if valid_commits is not None and commit not in valid_commits:
            raise ContinuationError("Bigram event source commit is not in the authorized source history")

    def check_sha(value: Any, label: str, *, length: int = 64) -> None:
        if (not isinstance(value, str) or len(value) != length
                or any(ch not in "0123456789abcdef" for ch in value)):
            raise ContinuationError(f"Malformed {label} in appended Bigram event")

    amendment = {
        "event": "BUDGET_CAP_AMENDMENT", "amendment_id": BUDGET_AMENDMENT_ID,
        "card_id": BUDGET_CARD_ID, "stage": BUDGET_STAGE,
        "previous_global_cap": 143, "global_cap": 149,
        "previous_caps": {"continuation": 52, "control_replay": 2, "pilot": 88, "technical_retry": 1},
        "effective_caps": {"continuation": 52, "control_replay": 2, "pilot": 94, "technical_retry": 1},
        "added_fits": 6, "source_commit": BUDGET_SOURCE_COMMIT,
        "freeze_signature": BUDGET_FREEZE_SIGNATURE, "authorization": BUDGET_AUTHORIZATION,
    }
    reservation = {
        "event": "CARD_FITS_RESERVED", "reservation_key": BUDGET_RESERVATION_KEY,
        "card_id": BUDGET_CARD_ID, "stage": BUDGET_STAGE, "category": "pilot", "fits": 6,
        "fit_ids": list(BUDGET_FIT_IDS), "amendment_id": BUDGET_AMENDMENT_ID,
        "source_commit": BUDGET_SOURCE_COMMIT, "freeze_signature": BUDGET_FREEZE_SIGNATURE,
    }
    amendment_index = reservation_index = None
    successor: dict[str, Any] | None = None
    started: list[str] = []
    terminals: dict[str, str] = {}
    pending: str | None = None
    accepted_types = {"BUDGET_CAP_AMENDMENT", "CARD_FITS_RESERVED", "PREFIT_FREEZE_SUCCESSOR_BOUND",
                      "FIT_STARTED", "FIT_SAVED", "FIT_FAILED"}

    for event_index, event in enumerate(events):
        kind = event["event"]
        if kind not in accepted_types:
            raise ContinuationError(f"Unapproved event in appended budget suffix: {kind}")
        if kind == "BUDGET_CAP_AMENDMENT":
            _require_event_fields(event, set(amendment), kind)
            if amendment_index is not None or any(event.get(k) != v for k, v in amendment.items()):
                raise ContinuationError("Bigram budget amendment is duplicated or differs from authorization")
            check_source_commit(event.get("source_commit"), initial=True)
            if event.get("freeze_signature") != BUDGET_FREEZE_SIGNATURE:
                raise ContinuationError("Bigram amendment freeze signature differs from authorization")
            amendment_index = event_index
        elif kind == "CARD_FITS_RESERVED":
            _require_event_fields(event, set(reservation), kind)
            if reservation_index is not None or any(event.get(k) != v for k, v in reservation.items()):
                raise ContinuationError("Bigram reservation is duplicated or differs from authorization")
            check_source_commit(event.get("source_commit"), initial=True)
            reservation_index = event_index
        elif kind == "PREFIT_FREEZE_SUCCESSOR_BOUND":
            expected = {
                "event": kind, "reservation_key": BUDGET_RESERVATION_KEY,
                "card_id": BUDGET_CARD_ID, "stage": BUDGET_STAGE,
                "fit_ids": list(BUDGET_FIT_IDS), "parent_freeze_signature": BUDGET_FREEZE_SIGNATURE,
                "reason": BUDGET_SUCCESSOR_REASON,
            }
            _require_event_fields(event, set(expected) | {"freeze_signature", "source_commit"}, kind)
            if successor is not None or any(event.get(k) != v for k, v in expected.items()):
                raise ContinuationError("Bigram successor binding is duplicated or differs from its frozen reservation")
            if pending is not None or started:
                raise ContinuationError("Bigram successor freeze was bound after a fit had started")
            check_sha(event.get("freeze_signature"), "successor freeze signature")
            if event.get("freeze_signature") == BUDGET_FREEZE_SIGNATURE:
                raise ContinuationError("Bigram successor freeze signature did not change")
            check_source_commit(event.get("source_commit"))
            successor = event
        else:
            fit_id = event.get("fit_id")
            if fit_id not in BUDGET_FIT_IDS:
                raise ContinuationError("Appended Bigram fit event has an unauthorized fit ID")
            if kind == "FIT_STARTED":
                fields = {"event", "fit_id", "reservation_key", "card_id", "stage", "origin", "arm",
                          "source_commit", "freeze_signature", "resource_snapshot"}
                _require_event_fields(event, fields, kind)
                if reservation_index is None or pending is not None:
                    raise ContinuationError("Bigram fit start lacks a reservation or overlaps a pending fit")
                expected_fit_id = BUDGET_FIT_IDS[len(started)] if len(started) < len(BUDGET_FIT_IDS) else None
                if fit_id != expected_fit_id:
                    raise ContinuationError("Bigram fit starts are not a prefix of the authorized order")
                origin = int(fit_id.split("_E", 1)[1].split("_", 1)[0])
                arm = fit_id.rsplit("_", 1)[1]
                freeze_sig = successor.get("freeze_signature") if successor else BUDGET_FREEZE_SIGNATURE
                source_commit = successor.get("source_commit") if successor else BUDGET_SOURCE_COMMIT
                if (event.get("reservation_key") != BUDGET_RESERVATION_KEY
                        or event.get("card_id") != BUDGET_CARD_ID or event.get("stage") != BUDGET_STAGE
                        or event.get("origin") != origin or event.get("arm") != arm
                        or event.get("freeze_signature") != freeze_sig
                        or event.get("source_commit") != source_commit):
                    raise ContinuationError("Bigram fit start does not match its reserved freeze identity")
                check_source_commit(event.get("source_commit"))
                resources = event.get("resource_snapshot")
                required_resources = {"process_rss_bytes", "available_memory_bytes", "initial_threshold_bytes",
                                      "reserve_threshold_bytes", "other_research_processes"}
                if (not isinstance(resources, dict) or set(resources) != required_resources
                        or not isinstance(resources.get("other_research_processes"), list)):
                    raise ContinuationError("Bigram FIT_STARTED resource snapshot is malformed")
                started.append(fit_id)
                pending = fit_id
            else:
                fields = ({"event", "fit_id", "reservation_key", "card_id", "stage", "source_commit",
                           "freeze_signature", "artifact_sha256", "receipt_sha256"}
                          if kind == "FIT_SAVED" else
                          {"event", "fit_id", "reservation_key", "card_id", "stage", "source_commit",
                           "freeze_signature", "reason"})
                _require_event_fields(event, fields, kind)
                if pending != fit_id or fit_id in terminals:
                    raise ContinuationError("Bigram terminal receipt has no unique preceding start")
                freeze_sig = successor.get("freeze_signature") if successor else BUDGET_FREEZE_SIGNATURE
                source_commit = successor.get("source_commit") if successor else BUDGET_SOURCE_COMMIT
                if (event.get("reservation_key") != BUDGET_RESERVATION_KEY
                        or event.get("card_id") != BUDGET_CARD_ID or event.get("stage") != BUDGET_STAGE
                        or event.get("freeze_signature") != freeze_sig
                        or event.get("source_commit") != source_commit):
                    raise ContinuationError("Bigram terminal receipt differs from its started fit identity")
                check_source_commit(event.get("source_commit"))
                if kind == "FIT_SAVED":
                    check_sha(event.get("artifact_sha256"), "Bigram artifact SHA256")
                    check_sha(event.get("receipt_sha256"), "Bigram receipt SHA256")
                elif not isinstance(event.get("reason"), str) or not event["reason"]:
                    raise ContinuationError("Bigram FIT_FAILED reason is missing")
                terminals[fit_id] = kind
                pending = None

    if (amendment_index != 0 or reservation_index != 1
            or len([e for e in events if e.get("event") == "BUDGET_CAP_AMENDMENT"]) != 1
            or len([e for e in events if e.get("event") == "CARD_FITS_RESERVED"]) != 1):
        raise ContinuationError("Authorized Bigram amendment/reservation pair is incomplete or out of order")
    suffix = b"".join(lines[expected_prefix_line_count:])
    return {
        "expected_prefix_sha256": expected_prefix_sha256,
        "prefix_sha256": prefix_sha256,
        "prefix_line_count": expected_prefix_line_count,
        "suffix_line_count": len(events),
        "suffix_sha256": _sha256_bytes(suffix),
        "current_sha256": _sha256_bytes(raw),
        "current_line_count": len(lines),
        "authorized_events": events,
        "bigram_fit_started_ids": started,
        "bigram_fit_terminal_events": terminals,
        "bigram_pending_fit_id": pending,
    }


def _validate_bigram_saved_receipts(source_root: Path, budget_extension: Mapping[str, Any]) -> None:
    for event in budget_extension.get("authorized_events", []):
        if event.get("event") != "FIT_SAVED":
            continue
        fit_id = str(event["fit_id"])
        artifact = source_root / BUDGET_RUN_REL / "fits" / f"{fit_id}.npz"
        receipt_path = source_root / BUDGET_RUN_REL / "receipts" / f"{fit_id}.json"
        if not artifact.is_file() or not receipt_path.is_file():
            raise ContinuationError(f"Bigram FIT_SAVED has no durable artifact/receipt: {fit_id}")
        _require_digest_match(str(event["artifact_sha256"]), pilot._sha_file(artifact), f"Bigram {fit_id} artifact")
        _require_digest_match(str(event["receipt_sha256"]), pilot._sha_file(receipt_path), f"Bigram {fit_id} receipt")
        receipt = _read_json(receipt_path)
        for key in ("fit_id", "card_id", "stage", "source_commit", "freeze_signature", "artifact_sha256"):
            if receipt.get(key) != event.get(key):
                raise ContinuationError(f"Bigram saved receipt identity mismatch: {fit_id} {key}")


def validate_scoped_reconciliation(
    *,
    historical_status_sha256: str,
    baseline_status_lines: Iterable[str],
    current_status_lines: Iterable[str],
    allowed_paths: Iterable[str],
    current_head: str,
    frozen_head: str,
    current_is_descendant: bool,
    changed_commit_paths: Iterable[str],
    frozen_source_hashes_match: bool,
    checkpoint_artifacts_match: bool,
) -> dict[str, Any]:
    """Allow only declared concurrent-owner changes; keep every other input pinned."""
    baseline = list(baseline_status_lines)
    current = list(current_status_lines)
    allowed = _normalize_paths(allowed_paths)
    if pilot._canonical_sha(baseline) != historical_status_sha256:
        raise ContinuationError("Reconstructed status baseline does not match the V4 historical hash")
    if current_head != frozen_head and not current_is_descendant:
        raise ContinuationError("Current source HEAD is not the frozen HEAD or its descendant")

    baseline_paths = [_status_path(line) for line in baseline]
    current_paths = [_status_path(line) for line in current]
    baseline_outside = [line for line, path in zip(baseline, baseline_paths)
                        if not _within_allowed(path, allowed)]
    current_outside = [line for line, path in zip(current, current_paths)
                       if not _within_allowed(path, allowed)]
    if current_outside != baseline_outside:
        raise ContinuationError("An outside-path worktree status entry changed from the V4 snapshot")

    for line, path in zip(current, current_paths):
        if line not in baseline and not _within_allowed(path, allowed):
            raise ContinuationError(f"Unexpected new source worktree path: {path}")
    changed = _normalize_paths(changed_commit_paths)
    unexpected_commits = sorted(path for path in changed if not _within_allowed(path, allowed))
    if unexpected_commits:
        raise ContinuationError(f"Source descendant includes out-of-scope paths: {unexpected_commits}")
    if not frozen_source_hashes_match:
        raise ContinuationError("A frozen source, data, or cache hash changed")
    if not checkpoint_artifacts_match:
        raise ContinuationError("A frozen checkpoint model, receipt, prediction, or ledger changed")

    added_status = [line for line in current if line not in baseline]
    return {
        "status": "PASS",
        "historical_status_sha256": historical_status_sha256,
        "baseline_status_lines": baseline,
        "current_status_lines": current,
        "outside_baseline_status_lines": baseline_outside,
        "outside_current_status_lines": current_outside,
        "authorized_current_paths": sorted(path for path in set(current_paths)
                                            if _within_allowed(path, allowed)),
        "new_status_lines": added_status,
        "changed_commit_paths": sorted(changed),
        "frozen_source_hashes_match": True,
        "checkpoint_artifacts_match": True,
        "current_head": current_head,
        "frozen_head": frozen_head,
    }


def _is_ancestor(source_root: Path, frozen_head: str, current_head: str) -> bool:
    if current_head == frozen_head:
        return True
    result = subprocess.run(
        ["git", "merge-base", "--is-ancestor", frozen_head, current_head],
        cwd=source_root, capture_output=True, text=True, check=False,
    )
    return result.returncode == 0


def _git_status_lines(source_root: Path) -> list[str]:
    # Preserve the leading XY status columns; pilot._git intentionally strips output.
    result = subprocess.check_output(["git", "status", "--porcelain"], cwd=source_root, text=True,
                                     encoding="utf-8")
    return result.splitlines()


def _owner_path_hashes(source_root: Path) -> dict[str, str]:
    hashes: dict[str, str] = {}
    for relative in sorted(AUTHORIZED_MAIN_PATHS):
        path = pilot._contained_file(source_root, Path(relative))
        if not path.exists():
            raise ContinuationError(f"Authorized concurrent-owner path is missing: {relative}")
        if path.is_symlink():
            raise ContinuationError(f"Symlink is not allowed in concurrent-owner path: {relative}")
        if path.is_file():
            hashes[relative] = pilot._sha_file(path)
            continue
        if not path.is_dir():
            raise ContinuationError(f"Authorized concurrent-owner path is not a file/directory: {relative}")
        children = sorted(path.rglob("*"))
        files = [child for child in children if child.is_file()]
        for child in children:
            if child.is_symlink():
                raise ContinuationError(f"Symlink is not allowed in concurrent-owner directory: {child}")
        if not files:
            raise ContinuationError(f"Authorized concurrent-owner directory is empty: {relative}")
        for child in files:
            key = child.relative_to(source_root).as_posix()
            hashes[key] = pilot._sha_file(child)
    return hashes


def validate_source_context(freeze: Mapping[str, Any], source_root: Path = pilot.SOURCE_ROOT) -> dict[str, Any]:
    source_root = source_root.resolve()
    if str(source_root) != freeze.get("source_root"):
        raise ContinuationError("Continuation source root differs from the immutable V4 path")
    if freeze.get("card_id") != CARD_ID or freeze.get("freeze_signature") != _body_signature(freeze, "freeze_signature"):
        raise ContinuationError("The original V4 freeze signature or card ID is invalid")
    if pilot.FREEZE_PATH.name != "FREEZE_V4.json":
        raise ContinuationError("Continuation adapter must use the original FREEZE_V4.json")

    current_head = pilot._git(source_root, "rev-parse", "HEAD")
    frozen_head = str(freeze.get("source_git_head", ""))
    current_branch = pilot._git(source_root, "branch", "--show-current")
    frozen_branch = str(freeze.get("source_git_branch", ""))
    if current_branch != frozen_branch:
        raise ContinuationError("Source branch changed from the V4 snapshot")
    descendant = _is_ancestor(source_root, frozen_head, current_head)
    if not descendant:
        raise ContinuationError("Current source HEAD is not descended from the V4 snapshot")
    changed_commit_paths = pilot._git(source_root, "diff", "--name-only", frozen_head, current_head).splitlines()
    baseline = list(BASELINE_STATUS_LINES)
    current_status = _git_status_lines(source_root)
    current_hashes = pilot._source_hashes(source_root)
    expected_hashes = freeze.get("source_hashes", {})
    history = set(pilot._git(source_root, "rev-list", "--ancestry-path",
                             f"{frozen_head}..{current_head}").splitlines())
    history.add(frozen_head)
    budget_path = pilot._contained_file(source_root, Path(BUDGET_REL))
    budget_extension = validate_budget_extension(
        budget_path.read_bytes(),
        expected_prefix_sha256=str(expected_hashes.get(BUDGET_REL, "")),
        expected_prefix_line_count=BUDGET_PREFIX_LINE_COUNT,
        valid_source_commits=history,
    )
    _validate_bigram_saved_receipts(source_root, budget_extension)
    source_hashes_match = (
        set(current_hashes) == set(expected_hashes)
        and all(current_hashes.get(path) == expected_hashes.get(path)
                for path in expected_hashes if path != BUDGET_REL)
        and current_hashes.get(BUDGET_REL) == budget_extension["current_sha256"]
    )
    reconciliation = validate_scoped_reconciliation(
        historical_status_sha256=str(freeze.get("source_worktree_status_sha256", "")),
        baseline_status_lines=baseline,
        current_status_lines=current_status,
        allowed_paths=AUTHORIZED_MAIN_PATHS,
        current_head=current_head,
        frozen_head=frozen_head,
        current_is_descendant=descendant,
        changed_commit_paths=changed_commit_paths,
        frozen_source_hashes_match=source_hashes_match,
        checkpoint_artifacts_match=True,
    )

    frame = pilot._load_feature_frame(source_root)  # Feature-only; target column remains closed.
    origins = {origin: pilot._load_origin(source_root, frame, origin) for origin in pilot.ORIGINS}
    current_body = pilot._freeze_body(source_root, frame, origins)
    compare_keys = (
        "source_root", "source_git_branch", "official_train_sha256",
        "feature_columns", "classification_regression_feature_alignment", "recipe_sha256",
        "resolved_parameters", "fit_order", "fit_cap", "resource_gate",
    )
    if any(current_body.get(key) != freeze.get(key) for key in compare_keys):
        raise ContinuationError("Frozen input hashes, feature rows, IDs, prefix contracts, or parameters changed")
    current_body_hashes = current_body.get("source_hashes", {})
    if (set(current_body_hashes) != set(expected_hashes)
            or any(current_body_hashes.get(path) != expected_hashes.get(path)
                   for path in expected_hashes if path != BUDGET_REL)
            or current_body_hashes.get(BUDGET_REL) != budget_extension["current_sha256"]):
        raise ContinuationError("Current source hash map differs outside the authorized budget append")
    expected_origins = copy.deepcopy(freeze.get("origins", {}))
    current_origins = copy.deepcopy(current_body.get("origins", {}))
    if set(current_origins) != set(expected_origins):
        raise ContinuationError("Frozen temporal origin contracts changed")
    for origin in expected_origins:
        expected_compatibility = expected_origins[origin].get("positional_vectorizer_compatibility", {})
        current_compatibility = current_origins[origin].get("positional_vectorizer_compatibility", {})
        if (expected_compatibility.get("fit_accounting_sha256") != expected_hashes.get(BUDGET_REL)
                or current_compatibility.get("fit_accounting_sha256") != budget_extension["current_sha256"]):
            raise ContinuationError("Positional vectorizer accounting hash does not match the authorized budget prefix/append")
        current_compatibility["fit_accounting_sha256"] = expected_compatibility["fit_accounting_sha256"]
    if current_origins != expected_origins:
        raise ContinuationError("Frozen input hashes, feature rows, IDs, prefix contracts, or parameters changed")
    if not bool(current_body.get("targets_read") is False):
        raise ContinuationError("Source reconciliation unexpectedly loaded targets")

    return {
        "reconciliation": reconciliation,
        "current_source_branch": current_branch,
        "current_source_head": current_head,
        "current_source_status_sha256": pilot._canonical_sha(current_status),
        "authorized_owner_file_hashes": _owner_path_hashes(source_root),
        "source_feature_rows": len(frame),
        "source_hashes_match": True,
        "feature_row_and_prefix_contracts_match": True,
        "query_targets_read": False,
        "budget_log_extension": budget_extension,
        "frame": frame,
        "origins": origins,
        "current_freeze_body": current_body,
    }


def _load_continuation_audit() -> dict[str, Any]:
    audit = _read_json(CONTINUATION_PATH)
    if (audit.get("status") != "AUTHORIZED_PENDING_TWO_UNITS"
            or audit.get("continuation_signature") != _body_signature(audit, "continuation_signature")
            or audit.get("card_id") != CARD_ID):
        raise ContinuationError("Continuation authorization audit signature/status failed")
    return audit


def _verify_local_commit_scope(freeze: Mapping[str, Any], audit: Mapping[str, Any]) -> dict[str, Any]:
    base = str(freeze.get("local_git_head", ""))
    current = pilot._git(ROOT, "rev-parse", "HEAD")
    descendant = _is_ancestor(ROOT, base, current)
    if not descendant:
        raise ContinuationError("Local task branch is not descended from the original V4 code HEAD")
    changed = _normalize_paths(pilot._git(ROOT, "diff", "--name-only", base, current).splitlines())
    allowed = _normalize_paths(LOCAL_COMMIT_PATHS)
    unexpected = sorted(path for path in changed if not _within_allowed(path, allowed))
    if unexpected:
        raise ContinuationError(f"Local continuation commit contains out-of-scope files: {unexpected}")
    required = _normalize_paths(audit.get("required_local_commit_paths", ()))
    missing_required = sorted(path for path in required if path not in changed)
    if missing_required:
        raise ContinuationError(f"Required pinned implementation/audit paths are not committed: {missing_required}")
    source_files = audit.get("continuation_source_file_hashes", {})
    for relative, expected in source_files.items():
        path = pilot._contained_file(ROOT, Path(relative))
        _require_digest_match(str(expected), pilot._sha_file(path), relative)
    return {"base_head": base, "current_head": current,
            "committed_paths": sorted(changed), "scope_status": "PASS"}


def verify_continuation(source_root: Path = pilot.SOURCE_ROOT, *, require_completion: bool = False) -> dict[str, Any]:
    audit = _load_continuation_audit()
    freeze = _read_json(pilot.FREEZE_PATH)
    source_context = validate_source_context(freeze, source_root)
    local_commit = _verify_local_commit_scope(freeze, audit)

    original_pins = audit.get("preserved_checkpoint_files", {})
    for relative, item in original_pins.items():
        path = pilot._contained_file(ROOT, Path(relative))
        _require_digest_match(str(item.get("sha256", "")), pilot._sha_file(path), relative)
    for unit_id, item in audit.get("ten_unit_checkpoint", {}).items():
        artifact, receipt = pilot._unit_paths(unit_id)
        _require_digest_match(item["artifact_sha256"], pilot._sha_file(artifact), f"{unit_id} artifact")
        _require_digest_match(item["receipt_sha256"], pilot._sha_file(receipt), f"{unit_id} receipt")
        pilot._validate_unit(pilot.RUN_ROOT, unit_id, freeze)

    events = pilot._read_events(pilot.LEDGER_PATH)
    starts = [event.get("fit_id") for event in events if event.get("event") == "FIT_STARTED"]
    saves = [event.get("fit_id") for event in events if event.get("event") == "FIT_SAVED"]
    failed = [event for event in events if event.get("event") == "FIT_FAILED"]
    if (len(starts) < 10 or len(starts) > 12 or starts != saves
            or starts != list(pilot.FIT_IDS[:len(starts)]) or failed):
        raise ContinuationError("Continuation ledger is not a clean ordered prefix of the 12-unit cap")
    for fit_id in pilot.FIT_IDS[10:len(starts)]:
        pilot._validate_unit(pilot.RUN_ROOT, fit_id, freeze)
    ledger_bytes = pilot.LEDGER_PATH.read_bytes()
    checkpoint_size = int(audit["checkpoint_ledger_size_bytes"])
    if (len(ledger_bytes) < checkpoint_size
            or hashlib_sha256(ledger_bytes[:checkpoint_size]) != audit.get("checkpoint_ledger_sha256")):
        raise ContinuationError("Original 10-unit ledger prefix changed")
    if len(starts) == 10 and hashlib_sha256(ledger_bytes) != audit.get("checkpoint_ledger_sha256"):
        raise ContinuationError("The initial 10-unit ledger changed before continuation")

    if require_completion:
        completion = _read_json(COMPLETION_PATH)
        if (completion.get("status") != "PASS"
                or completion.get("completion_signature") != _body_signature(completion, "completion_signature")
                or completion.get("continuation_signature") != audit.get("continuation_signature")):
            raise ContinuationError("Continuation completion audit is absent or invalid")
        if completion.get("new_fit_ids") != list(pilot.FIT_IDS[10:12]):
            raise ContinuationError("Completion audit lists fit IDs outside the two authorized E75 units")
        if starts != list(pilot.FIT_IDS) or saves != list(pilot.FIT_IDS) or failed:
            raise ContinuationError("Completion ledger is not exactly 12 ordered starts/saves")
        ready = _read_json(pilot.RUN_ROOT / "PREDICTIONS_READY.json")
        if (ready.get("freeze_signature") != freeze.get("freeze_signature")
                or ready.get("fit_units_started") != 12 or ready.get("fit_units_saved") != 12
                or ready.get("query_targets_read") is not False
                or ready.get("ledger_sha256") != pilot._sha_file(pilot.LEDGER_PATH)):
            raise ContinuationError("Runner readiness record is missing or differs from the signed completion")
        for fit_id in pilot.FIT_IDS:
            artifact, receipt = pilot._unit_paths(fit_id)
            record = completion.get("models", {}).get(fit_id, {})
            _require_digest_match(record.get("artifact_sha256", ""), pilot._sha_file(artifact),
                                  f"{fit_id} completion artifact")
            _require_digest_match(record.get("receipt_sha256", ""), pilot._sha_file(receipt),
                                  f"{fit_id} completion receipt")
        for fit_id in pilot.FIT_IDS:
            bundle, receipt = pilot._prediction_paths(fit_id)
            record = completion.get("predictions", {}).get(fit_id, {})
            _require_digest_match(record.get("bundle_sha256", ""), pilot._sha_file(bundle),
                                  f"{fit_id} completion prediction")
            _require_digest_match(record.get("receipt_sha256", ""), pilot._sha_file(receipt),
                                  f"{fit_id} completion prediction receipt")
    return {"status": "PASS", "continuation_signature": audit["continuation_signature"],
            "source_reconciliation": source_context["reconciliation"],
            "authorized_owner_file_hashes": source_context["authorized_owner_file_hashes"],
            "local_commit_scope": local_commit,
            "checkpoint_units": len(audit.get("ten_unit_checkpoint", {})),
            "query_targets_read": False}


def hashlib_sha256(value: bytes) -> str:
    import hashlib
    return hashlib.sha256(value).hexdigest()


def prelabel_verify(source_root: Path = pilot.SOURCE_ROOT) -> dict[str, Any]:
    continuation = verify_continuation(source_root, require_completion=True)
    original = original_verifier.prelabel_verify(source_root)
    result = {
        "schema_version": 1, "card_id": CARD_ID, "status": original["status"],
        "verified_utc": datetime.now(timezone.utc).isoformat(),
        "continuation_signature": continuation["continuation_signature"],
        "freeze_signature": original["freeze_signature"],
        "query_targets_read": False,
        "original_prelabel_report_sha256": pilot._sha_file(pilot.PRELABEL_PATH),
        "ledger_sha256": original["ledger"]["ledger_sha256"],
        "fit_units_started": original["fit_units_started"],
        "fit_units_saved": original["fit_units_saved"],
        "fit_units_failed": original["fit_units_failed"],
        "models_verified": original["models_verified"],
        "predictions_verified": original["predictions_verified"],
        "source_reconciliation": continuation["source_reconciliation"],
    }
    result["verification_signature"] = pilot._canonical_sha(result)
    pilot._atomic_json(CONTINUATION_PRELABEL_PATH, result)
    return result


def score_after_prelabel(source_root: Path = pilot.SOURCE_ROOT) -> dict[str, Any]:
    continuation = verify_continuation(source_root, require_completion=True)
    prelabel = _read_json(CONTINUATION_PRELABEL_PATH)
    if (prelabel.get("status") != "PASS" or prelabel.get("query_targets_read") is not False
            or prelabel.get("continuation_signature") != continuation["continuation_signature"]
            or prelabel.get("verification_signature") != _body_signature(prelabel, "verification_signature")):
        raise ContinuationError("Scoring blocked until versioned continuation prelabel QA passes")
    result = original_verifier.score_after_prelabel(source_root)
    final = {
        "schema_version": 1, "card_id": CARD_ID, "verification_status": "PASS",
        "candidate_status": result.get("status"),
        "verified_utc": datetime.now(timezone.utc).isoformat(),
        "continuation_signature": continuation["continuation_signature"],
        "freeze_signature": result["freeze_signature"],
        "query_targets_accessed": result["query_targets_accessed"],
        "results_sha256": pilot._sha_file(pilot.REPORT_ROOT / "RESULTS.json"),
        "report_sha256": pilot._sha_file(pilot.REPORT_ROOT / "REPORT.md"),
        "fit_budget_cap": 12, "fit_units_started": 12, "fit_units_saved": 12,
        "remaining_fit_units": 0,
        "promotion_or_upload": "NOT_AUTHORIZED_BY_THIS_PILOT",
    }
    final["verification_signature"] = pilot._canonical_sha(final)
    pilot._atomic_json(CONTINUATION_FINAL_PATH, final)
    return {"result": result, "continuation_final": final}


def main(argv: list[str] | None = None) -> int:
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("verify", "prelabel", "score"))
    parser.add_argument("--source-root", type=Path, default=pilot.SOURCE_ROOT)
    args = parser.parse_args(argv)
    try:
        if args.command == "verify":
            result = verify_continuation(args.source_root, require_completion=True)
        elif args.command == "prelabel":
            result = prelabel_verify(args.source_root)
        else:
            result = score_after_prelabel(args.source_root)
        print(json.dumps(result, sort_keys=True, indent=2))
        return 0
    except (ContinuationError, pilot.PilotError) as exc:
        print(f"NO-GO: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
