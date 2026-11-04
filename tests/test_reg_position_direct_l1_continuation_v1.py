from __future__ import annotations

import json
import hashlib

import pytest

import scripts.continue_reg_position_direct_l1_v1 as runner
import scripts.reg_position_direct_l1_v1 as pilot
import scripts.verify_reg_position_direct_l1_continuation_v1 as continuation


def _reconcile(*, baseline=None, current=None, head="base", frozen="base",
               descendant=True, commits=(), frozen_hashes=True, checkpoint=True):
    base = list(continuation.BASELINE_STATUS_LINES if baseline is None else baseline)
    status = list(base if current is None else current)
    return continuation.validate_scoped_reconciliation(
        historical_status_sha256=pilot._canonical_sha(base),
        baseline_status_lines=base,
        current_status_lines=status,
        allowed_paths=continuation.AUTHORIZED_MAIN_PATHS,
        current_head=head,
        frozen_head=frozen,
        current_is_descendant=descendant,
        changed_commit_paths=commits,
        frozen_source_hashes_match=frozen_hashes,
        checkpoint_artifacts_match=checkpoint,
    )


def test_v4_historical_status_reconstructs_from_preexisting_owner_paths() -> None:
    freeze = pilot._read_json(pilot.FREEZE_PATH)
    assert pilot._canonical_sha(continuation.BASELINE_STATUS_LINES) == freeze["source_worktree_status_sha256"]


def test_continuation_fit_plan_is_only_the_two_remaining_e75_units() -> None:
    assert runner.EXPECTED_PRIOR_FITS == tuple(pilot.FIT_IDS[:10])
    assert runner.ALLOWED_NEW_FITS == (
        "REG_POSITION_DIRECT_L1_V1_E75_REAL",
        "REG_POSITION_DIRECT_L1_V1_E75_SHUFFLED",
    )


def test_authorized_new_bigram_report_file_is_allowed() -> None:
    report = "?? reports/optimization_v3/bigram_position_pilot_v1/"
    result = _reconcile(current=[report, *continuation.BASELINE_STATUS_LINES])
    assert result["status"] == "PASS"
    assert result["new_status_lines"] == [report]
    assert result["outside_current_status_lines"] == []


def test_authorized_bigram_commit_on_descendant_head_is_allowed() -> None:
    committed = "scripts/verify_optimization_v3_bigram_position_pilot_v1.py"
    current = [line for line in continuation.BASELINE_STATUS_LINES
               if not line.endswith(committed)]
    result = _reconcile(head="authorized-descendant", commits=[committed], current=current)
    assert result["status"] == "PASS"
    assert result["changed_commit_paths"] == [committed]


def test_modified_frozen_source_or_input_is_rejected() -> None:
    with pytest.raises(continuation.ContinuationError, match="frozen source"):
        _reconcile(frozen_hashes=False)


def test_unexpected_worktree_or_committed_path_is_rejected() -> None:
    unexpected = "?? data/raw/regression/unapproved.csv"
    with pytest.raises(continuation.ContinuationError, match="outside-path worktree status"):
        _reconcile(current=[*continuation.BASELINE_STATUS_LINES, unexpected])
    with pytest.raises(continuation.ContinuationError, match="out-of-scope paths"):
        _reconcile(head="descendant", commits=["src/models/unapproved.py"])


def test_outside_path_status_must_remain_at_historical_snapshot() -> None:
    outside = " M src/data_contract.py"
    baseline = [*continuation.BASELINE_STATUS_LINES, outside]
    with pytest.raises(continuation.ContinuationError, match="outside-path worktree status"):
        _reconcile(baseline=baseline, current=continuation.BASELINE_STATUS_LINES)


def test_tampered_checkpoint_artifact_is_rejected() -> None:
    with pytest.raises(continuation.ContinuationError, match="Pinned digest mismatch"):
        continuation._require_digest_match("a" * 64, "b" * 64, "E75_REAL unit artifact")
    with pytest.raises(continuation.ContinuationError, match="checkpoint model"):
        _reconcile(checkpoint=False)


def _bigram_amendment() -> dict:
    return {
        "event": "BUDGET_CAP_AMENDMENT",
        "amendment_id": continuation.BUDGET_AMENDMENT_ID,
        "card_id": continuation.BUDGET_CARD_ID,
        "stage": continuation.BUDGET_STAGE,
        "previous_global_cap": 143,
        "global_cap": 149,
        "previous_caps": {"continuation": 52, "control_replay": 2, "pilot": 88, "technical_retry": 1},
        "effective_caps": {"continuation": 52, "control_replay": 2, "pilot": 94, "technical_retry": 1},
        "added_fits": 6,
        "source_commit": continuation.BUDGET_SOURCE_COMMIT,
        "freeze_signature": continuation.BUDGET_FREEZE_SIGNATURE,
        "authorization": continuation.BUDGET_AUTHORIZATION,
        "at_utc": "2026-11-04T09:10:15.107483+00:00",
    }


def _bigram_reservation() -> dict:
    return {
        "event": "CARD_FITS_RESERVED",
        "reservation_key": continuation.BUDGET_RESERVATION_KEY,
        "card_id": continuation.BUDGET_CARD_ID,
        "stage": continuation.BUDGET_STAGE,
        "category": "pilot",
        "fits": 6,
        "fit_ids": list(continuation.BUDGET_FIT_IDS),
        "amendment_id": continuation.BUDGET_AMENDMENT_ID,
        "source_commit": continuation.BUDGET_SOURCE_COMMIT,
        "freeze_signature": continuation.BUDGET_FREEZE_SIGNATURE,
        "at_utc": "2026-11-04T09:10:15.109345+00:00",
    }


def _budget_bytes(events: list[dict]) -> tuple[bytes, str]:
    prefix = b"{}\n" * continuation.BUDGET_PREFIX_LINE_COUNT
    rows = b"".join(json.dumps(event, sort_keys=True, separators=(",", ":")).encode() + b"\n"
                    for event in events)
    return prefix + rows, hashlib.sha256(prefix).hexdigest()


def test_budget_extension_accepts_authorized_amendment_and_reservation() -> None:
    raw, prefix_sha = _budget_bytes([_bigram_amendment(), _bigram_reservation()])
    result = continuation.validate_budget_extension(
        raw, expected_prefix_sha256=prefix_sha,
        valid_source_commits={continuation.BUDGET_SOURCE_COMMIT},
    )
    assert result["prefix_sha256"] == prefix_sha
    assert result["suffix_line_count"] == 2
    assert result["current_sha256"] == hashlib.sha256(raw).hexdigest()
    assert result["bigram_fit_started_ids"] == []


def test_budget_extension_accepts_bound_successor_and_its_fit_receipts() -> None:
    successor_commit = "a" * 40
    successor_signature = "b" * 64
    successor = {
        "event": "PREFIT_FREEZE_SUCCESSOR_BOUND",
        "reservation_key": continuation.BUDGET_RESERVATION_KEY,
        "card_id": continuation.BUDGET_CARD_ID,
        "stage": continuation.BUDGET_STAGE,
        "fit_ids": list(continuation.BUDGET_FIT_IDS),
        "parent_freeze_signature": continuation.BUDGET_FREEZE_SIGNATURE,
        "freeze_signature": successor_signature,
        "source_commit": successor_commit,
        "reason": continuation.BUDGET_SUCCESSOR_REASON,
        "at_utc": "2026-11-04T09:15:00+00:00",
    }
    fit_id = continuation.BUDGET_FIT_IDS[0]
    start = {
        "event": "FIT_STARTED", "fit_id": fit_id,
        "reservation_key": continuation.BUDGET_RESERVATION_KEY,
        "card_id": continuation.BUDGET_CARD_ID, "stage": continuation.BUDGET_STAGE,
        "origin": 67, "arm": "BAG", "source_commit": successor_commit,
        "freeze_signature": successor_signature,
        "resource_snapshot": {"process_rss_bytes": 1, "available_memory_bytes": 2,
                              "initial_threshold_bytes": 3, "reserve_threshold_bytes": 4,
                              "other_research_processes": []},
        "at_utc": "2026-11-04T09:16:00+00:00",
    }
    saved = {
        "event": "FIT_SAVED", "fit_id": fit_id,
        "reservation_key": continuation.BUDGET_RESERVATION_KEY,
        "card_id": continuation.BUDGET_CARD_ID, "stage": continuation.BUDGET_STAGE,
        "source_commit": successor_commit, "freeze_signature": successor_signature,
        "artifact_sha256": "c" * 64, "receipt_sha256": "d" * 64,
        "at_utc": "2026-11-04T09:17:00+00:00",
    }
    raw, prefix_sha = _budget_bytes([_bigram_amendment(), _bigram_reservation(), successor, start, saved])
    result = continuation.validate_budget_extension(
        raw, expected_prefix_sha256=prefix_sha,
        valid_source_commits={continuation.BUDGET_SOURCE_COMMIT, successor_commit},
    )
    assert result["bigram_fit_started_ids"] == [fit_id]
    assert result["bigram_fit_terminal_events"] == {fit_id: "FIT_SAVED"}
    assert result["bigram_pending_fit_id"] is None


def test_budget_extension_rejects_any_old_prefix_mutation() -> None:
    raw, prefix_sha = _budget_bytes([_bigram_amendment(), _bigram_reservation()])
    altered = b"x" + raw[1:]
    with pytest.raises(continuation.ContinuationError, match="raw prefix changed"):
        continuation.validate_budget_extension(
            altered, expected_prefix_sha256=prefix_sha,
            valid_source_commits={continuation.BUDGET_SOURCE_COMMIT},
        )


def test_budget_extension_rejects_unknown_card_event_or_fit_id() -> None:
    unrelated = {**_bigram_amendment(), "card_id": "OTHER_CARD"}
    raw, prefix_sha = _budget_bytes([unrelated, _bigram_reservation()])
    with pytest.raises(continuation.ContinuationError, match="differs from authorization"):
        continuation.validate_budget_extension(
            raw, expected_prefix_sha256=prefix_sha,
            valid_source_commits={continuation.BUDGET_SOURCE_COMMIT},
        )

    wrong_fit = {
        "event": "FIT_STARTED", "fit_id": "UNAUTHORIZED_FIT",
        "reservation_key": continuation.BUDGET_RESERVATION_KEY,
        "card_id": continuation.BUDGET_CARD_ID, "stage": continuation.BUDGET_STAGE,
        "origin": 67, "arm": "BAG", "source_commit": continuation.BUDGET_SOURCE_COMMIT,
        "freeze_signature": continuation.BUDGET_FREEZE_SIGNATURE,
        "resource_snapshot": {"process_rss_bytes": 1, "available_memory_bytes": 2,
                              "initial_threshold_bytes": 3, "reserve_threshold_bytes": 4,
                              "other_research_processes": []},
        "at_utc": "2026-11-04T09:16:00+00:00",
    }
    raw, prefix_sha = _budget_bytes([_bigram_amendment(), _bigram_reservation(), wrong_fit])
    with pytest.raises(continuation.ContinuationError, match="unauthorized fit ID"):
        continuation.validate_budget_extension(
            raw, expected_prefix_sha256=prefix_sha,
            valid_source_commits={continuation.BUDGET_SOURCE_COMMIT},
        )
