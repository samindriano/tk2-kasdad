from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pytest

from scripts import optimization_v3_rnbr_target_median_v1 as runner
from scripts import optimization_v3_rnbr_target_median_v1_synthetic as synthetic


def _synthetic_freeze(ledger_path: Path) -> dict:
    body = {
        "source_commit": runner._git("rev-parse", "HEAD"),
        "freeze_signature": "f" * 64,
        "ledger_sha_before": runner._sha_file(ledger_path),
    }
    return body


def test_completed_fit_ledger_resume_is_idempotent_and_live_file_is_unchanged(
    synthetic_pre_fit_budget_ledger: Path,
) -> None:
    live_path = runner.LEDGER_PATH
    live_hash = runner._sha_file(live_path)
    freeze = _synthetic_freeze(synthetic_pre_fit_budget_ledger)
    before = runner.ledger_snapshot(runner._ledger_events(synthetic_pre_fit_budget_ledger))
    assert before["fit_starts"] == 83 and before["fit_saved"] == 80 and before["fit_failed"] == 3

    runner._apply_cap_amendment_and_reserve(synthetic_pre_fit_budget_ledger, freeze)
    resource = {"jobs": 1, "threads": 2, "process_rss_before_fit_bytes": 0,
                "available_ram_before_fit_bytes": 8 * 1024**3}
    for fit_id in runner.FIT_IDS:
        origin = int(fit_id.split("_E", 1)[1].split("_", 1)[0])
        arm = fit_id.rsplit("_", 1)[1].lower()
        runner._start_fit(synthetic_pre_fit_budget_ledger, freeze, fit_id, origin, arm, resource)
        runner._finish_fit(synthetic_pre_fit_budget_ledger, fit_id, "FIT_SAVED",
                           receipt_sha256=(fit_id.encode("utf-8").hex() * 8)[:64])

    completed_hash = runner._sha_file(synthetic_pre_fit_budget_ledger)
    complete = runner.ledger_snapshot(runner._ledger_events(synthetic_pre_fit_budget_ledger))
    assert complete["global_cap"] == 87
    assert complete["fit_starts"] == 87 and complete["fit_saved"] == 84
    assert complete["fit_failed"] == 3 and complete["fit_pending"] == 0

    resumed = runner._apply_cap_amendment_and_reserve(synthetic_pre_fit_budget_ledger, freeze)
    assert resumed["ledger_sha256"] == completed_hash
    assert runner._sha_file(synthetic_pre_fit_budget_ledger) == completed_hash
    assert runner.ledger_snapshot(runner._ledger_events(synthetic_pre_fit_budget_ledger)) == complete
    assert runner._sha_file(live_path) == live_hash


def test_frozen_source_hash_mismatch_is_rejected_without_writing() -> None:
    live_path = runner.LEDGER_PATH
    live_hash = runner._sha_file(live_path)
    preflight = {"status": "PASS_TARGET_FREE_PREFLIGHT",
                 "input_hashes": {"synthetic": "abc"}, "geometry_hashes": {"67": {}, "75": {}}}
    freeze = {"source_hashes": runner._source_hashes(),
              "protocol_sha256": runner._sha_file(runner.PROTOCOL_PATH),
              "input_hashes": copy.deepcopy(preflight["input_hashes"]),
              "geometry_hashes": copy.deepcopy(preflight["geometry_hashes"])}
    runner._source_and_input_hashes_match(freeze, preflight)
    tampered = copy.deepcopy(freeze)
    source_key = next(iter(tampered["source_hashes"]))
    tampered["source_hashes"][source_key] = "0" * 64
    with pytest.raises(runner.TargetMedianRunError, match="source code hash differs"):
        runner._source_and_input_hashes_match(tampered, preflight)
    assert runner._sha_file(live_path) == live_hash


def test_synthetic_bundle_file_and_receipt_hash_tampering_is_rejected(tmp_path: Path) -> None:
    live_path = runner.LEDGER_PATH
    live_hash = runner._sha_file(live_path)
    run_root = tmp_path / "bundle_root"
    origin_input = synthetic._input(67)
    freeze = {
        "source_commit": runner._git("rev-parse", "HEAD"),
        "freeze_signature": "a" * 64,
        "input_hashes": {"synthetic": "1234"},
        "geometry_hashes": {"67": {"geometry": "test-only"}},
    }
    payload = {"target": np.linspace(0, 100, len(origin_input["donor_rows"]))}
    built = runner._build_fit_bundle(
        f"{runner.PILOT_ID}_E67_OBSERVED", 67, "observed", freeze, origin_input,
        target_payload=payload, resource={"synthetic": True}, run_root=run_root,
    )
    fit_id = f"{runner.PILOT_ID}_E67_OBSERVED"
    runner._load_bundle(fit_id, freeze, run_root=run_root)
    npz_path = built["npz_path"]
    original_npz = npz_path.read_bytes()
    npz_path.write_bytes(original_npz + b"tampered")
    with pytest.raises(runner.TargetMedianRunError, match="receipt or boundary attestation"):
        runner._load_bundle(fit_id, freeze, run_root=run_root)
    npz_path.write_bytes(original_npz)

    receipt_path = built["receipt_path"]
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["source_commit"] = "0" * 40
    receipt_path.write_text(json.dumps(receipt, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    with pytest.raises(runner.TargetMedianRunError, match="receipt or boundary attestation"):
        runner._load_bundle(fit_id, freeze, run_root=run_root)
    assert runner._sha_file(live_path) == live_hash
