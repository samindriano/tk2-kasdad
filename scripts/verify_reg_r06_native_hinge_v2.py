"""Independent non-target verifier for the R06-native hinge source package."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import sys
from typing import Any

import numpy as np

from scripts import reg_r06_native_hinge_v2 as pilot


class VerificationError(RuntimeError):
    pass


def _need(condition: bool, message: str) -> None:
    if not condition:
        raise VerificationError(message)


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _signed_json(path: Path) -> tuple[dict[str, Any], str]:
    _need(path.is_file(), f"Missing signed JSON: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    signature = data.pop("signature", None)
    _need(signature == pilot.canonical_sha(data), f"Invalid signature: {path}")
    data["signature"] = signature
    return data, _sha(path)


def _read_npz(path: Path, members: tuple[str, ...]) -> tuple[dict[str, np.ndarray], list[str]]:
    with np.load(path, allow_pickle=False) as archive:
        all_members = list(archive.files)
        _need(set(members).issubset(all_members), f"Missing NPZ member in {path}")
        selected = {name: np.array(archive[name], copy=True) for name in members}
    return selected, all_members


def _check_row_cache_alignment(freeze: dict[str, Any],
                               preflight: dict[str, Any]) -> dict[str, Any]:
    paths = {name: Path(item["path"]) for name, item in freeze["input_hashes"].items()}
    labels, label_members = _read_npz(paths["oof_label_and_identity_bundle"], (
        "physical_rows", "record_ids", "prediction_origins", "cycles",
        "missing_mask", "direct_predictions"))
    ids = labels["record_ids"].astype(str)
    cycles = labels["cycles"].astype(int)
    rows = labels["physical_rows"].astype(int)
    _need(len(ids) == pilot.EXPECTED_ROWS and len(set(ids.tolist())) == len(ids),
          "Cached label/ID identity is not unique")
    _need("targets" in label_members, "Cached label bundle lost the expected target member")
    _need(np.array_equal(np.unique(cycles), np.arange(68, 80)),
          "Cached label cycles changed")
    index = {record_id: i for i, record_id in enumerate(ids.tolist())}
    raw = np.full(len(ids), np.nan, dtype=np.float64)
    origins = np.full(len(ids), -1, dtype=np.int16)
    observed_members = {}
    for origin, cycle_member, allowed in (
            (67, "audit__cycle_code", set(range(68, 76))),
            (75, "score_cycles", set(range(76, 80)))):
        arrays, names = _read_npz(paths[f"r06_fit{origin}_npz"], (
            "score_rows", "score_record_ids", "prediction", cycle_member))
        cache_ids = arrays["score_record_ids"].astype(str)
        cache_cycles = np.asarray(arrays[cycle_member], dtype=int)
        cache_rows = np.asarray(arrays["score_rows"], dtype=int)
        predictions = np.asarray(arrays["prediction"], dtype=np.float64)
        mask = np.isin(cache_cycles, list(allowed))
        actual = {(str(cache_ids[j]), int(cache_cycles[j]), int(cache_rows[j]))
                  for j in np.flatnonzero(mask)}
        expected = {(str(ids[j]), int(cycles[j]), int(rows[j]))
                    for j in range(len(ids)) if int(cycles[j]) in allowed}
        _need(actual == expected, f"Fit{origin} row IDs, cycles, or positions changed")
        for j in np.flatnonzero(mask):
            target_i = index[str(cache_ids[j])]
            _need(origins[target_i] == -1, "Duplicate R06 source assignment")
            raw[target_i] = predictions[j]
            origins[target_i] = origin
        observed_members[str(origin)] = names
    _need(not np.isnan(raw).any(), "R06 OOF vector has rows without a source")
    _need(np.array_equal(origins, labels["prediction_origins"]),
          "R06 E67/E75 source assignments differ from cached identity")
    missing = labels["missing_mask"].astype(bool)
    _need(np.array_equal(labels["direct_predictions"][missing], raw[missing]),
          "Cached DIRECT missing fallback does not match raw R06 bit-for-bit")
    _need(label_members == preflight["npz_member_audit"]["label_bundle_members_seen"],
          "Label NPZ member inventory changed")
    _need(observed_members == preflight["npz_member_audit"]["r06_cache_members_seen"],
          "R06 NPZ member inventory changed")
    return {
        "rows": len(ids),
        "unique_record_ids": len(set(ids.tolist())),
        "unique_physical_rows": len(set(rows.tolist())),
        "cycle_counts": {str(c): int(np.sum(cycles == c)) for c in range(68, 80)},
        "r06_source_counts": {str(origin): int(np.sum(origins == origin))
                              for origin in (67, 75)},
        "r06_prediction_sha256": pilot.array_sha256(raw),
        "missing_rows": int(missing.sum()),
        "targets_member_seen_but_not_loaded": True,
        "target_values_loaded": False,
        "test_labels_read": False,
    }


def _check_r4_comparator_lineage(freeze: dict[str, Any]) -> dict[str, Any]:
    records = freeze["input_hashes"]
    paths = {name: Path(records[name]["path"]) for name in (
        "r4_root_source_pass", "r4_diagnostics", "r4_diagnostics_verification",
        "r4_hinge_d1_state", "r4_hinge_d2_state", "r4_hinge_d1_input",
        "r4_hinge_d2_input", "r4_lp_ledger")}
    root_pass = json.loads(paths["r4_root_source_pass"].read_text(encoding="utf-8"))
    diagnostics = json.loads(paths["r4_diagnostics"].read_text(encoding="utf-8"))
    verification = json.loads(paths["r4_diagnostics_verification"].read_text(encoding="utf-8"))
    root_pass_sha = _sha(paths["r4_root_source_pass"])
    _need(root_pass.get("verdict") == "PASS"
          and root_pass.get("authorized_fit_ids") == ["HINGE_D1", "HINGE_D2"],
          "R4 historical cutoff fits were not authorized by the frozen root pass")
    _need(diagnostics.get("root_source_pass_sha256") == root_pass_sha
          and diagnostics.get("diagnostic_fit_ids") == ["HINGE_D1", "HINGE_D2"],
          "R4 diagnostics do not bind the frozen root pass and D1/D2 states")
    _need(verification.get("status") == "PASS_INDEPENDENT_DIAGNOSTICS_REPLAY"
          and verification.get("root_source_pass_sha256") == root_pass_sha,
          "R4 independent replay did not pass against the frozen root pass")
    ledger = [json.loads(line) for line in paths["r4_lp_ledger"].read_text(
        encoding="utf-8").splitlines() if line.strip()]
    expected = {
        "HINGE_D1": ("r4_hinge_d1_state", "r4_hinge_d1_input", 4_950),
        "HINGE_D2": ("r4_hinge_d2_state", "r4_hinge_d2_input", 12_412),
    }
    frozen_history = freeze.get("comparators", {}).get("historical_cutoff_states", {})
    result = {}
    for fit_id, (state_key, input_key, rows) in expected.items():
        state_path, input_path = paths[state_key], paths[input_key]
        state = json.loads(state_path.read_text(encoding="utf-8"))
        solver = state.get("solver_result", {})
        fit_input = state.get("fit_input", {})
        history = frozen_history.get(fit_id, {})
        events = [event for event in ledger if event.get("fit_id") == fit_id]
        completed = [event for event in events if event.get("event") == "LP_COMPLETE"]
        reserved = [event for event in events if event.get("event") == "RESERVE"]
        _need(state.get("status") == "LP_COMPLETE" and state.get("fit_id") == fit_id
              and solver.get("success") is True and solver.get("fit_rows") == rows
              and fit_input.get("fit_rows") == rows
              and fit_input.get("fit_input_bundle_sha256") == _sha(input_path),
              f"Frozen R4 {fit_id} state is not a saved same-cutoff calibrator")
        _need(history.get("state_sha256") == _sha(state_path)
              and history.get("fit_input_sha256") == _sha(input_path)
              and history.get("fit_rows") == rows,
              f"Prefit freeze does not bind the saved R4 {fit_id} state and input")
        _need(len(completed) == 1 and len(reserved) == 1
              and completed[0].get("state_sha256") == _sha(state_path)
              and reserved[0].get("fit_input", {}).get(
                  "fit_input_bundle_sha256") == _sha(input_path)
              and reserved[0].get("root_source_pass_sha256") == root_pass_sha,
              f"R4 {fit_id} state differs from its immutable ledger history")
        result[fit_id] = {"state_sha256": _sha(state_path),
                          "fit_input_sha256": _sha(input_path), "fit_rows": rows}
    return {"root_source_pass_sha256": root_pass_sha, "cutoff_states": result}


def verify_source_ready(*, write_receipt: bool = False) -> dict[str, Any]:
    freeze, freeze_sha = _signed_json(pilot.FREEZE)
    preflight, preflight_sha = _signed_json(pilot.PREFLIGHT)
    synthetic, synthetic_sha = _signed_json(pilot.SYNTHETIC_QA)
    _need(freeze.get("status") ==
          "IDENTITY_FROZEN_PENDING_NON_TARGET_PREFLIGHT_AND_ROOT_REVIEW_R2",
          "Prefit freeze is not pending root review")
    _need(preflight.get("status") ==
          "PASS_NON_TARGET_IDENTITY_PREFLIGHT_PENDING_ROOT_SOURCE_PASS_R2",
          "Non-target identity preflight is not in pending-root state")
    _need(preflight.get("prefit_freeze_sha256") == freeze_sha,
          "Preflight does not bind the prefit freeze")
    _need(synthetic.get("status") == "PASS_SYNTHETIC_SOURCE_AND_RECEIPT_E2E",
          "Synthetic CLI/receipt E2E has not passed")
    _need(all(bool(value) for key, value in synthetic["checks"].items()
              if key not in {"official_lp_starts", "target_values_loaded",
                             "test_labels_read", "csv_created", "kaggle_accessed"}),
          "Synthetic checks contain a failure")
    _need(not pilot.ROOT_SOURCE_PASS.exists(),
          "Root source pass exists; source-ready-only verifier must be rerun by root")
    _need(not pilot.LEDGER.exists(), "Official fit ledger exists before root source pass")
    _need(freeze.get("official_lp_starts") == 0
          and preflight.get("official_lp_starts") == 0,
          "A pre-source-pass LP start was recorded")
    _need(preflight.get("target_reader_called") is False
          and preflight.get("train_target_values_loaded") is False
          and preflight.get("score_target_values_loaded") is False,
          "Target values crossed the non-target preflight boundary")
    _need(preflight.get("test_labels_read") is False
          and preflight.get("test_or_sample_accessed") is False
          and preflight.get("kaggle_accessed") is False,
          "Test/sample or Kaggle access is recorded")
    _need(freeze.get("cache_assignment", {}).get("R06_fit71_used") is False,
          "Pilot source assignment drifted to the E71 R06 route")
    _need(freeze.get("prior_slot_accounting", {}).get("old_r4_lp_starts_already_consumed") == 3
          and freeze.get("prior_slot_accounting", {}).get("old_ledger_reset_or_rewritten") is False,
          "Old R4 budget accounting is incomplete")
    for group in (freeze["source_hashes"],):
        for name, expected in group.items():
            path = Path(pilot.SOURCE_FILES[name])
            _need(_sha(path) == expected, f"Source changed after freeze: {name}")
    for name, record in freeze["input_hashes"].items():
        path = Path(record["path"])
        _need(path.is_file() and _sha(path) == record["sha256"],
              f"Frozen input changed: {name}")
    ancestor = subprocess.run(
        ["git", "-C", str(pilot.ROOT), "merge-base", "--is-ancestor",
         str(freeze["source_commit"]), "HEAD"], capture_output=True, text=True, check=False)
    _need(ancestor.returncode == 0,
          "Frozen source commit is not an ancestor of the current checkout")
    comparator_audit = _check_r4_comparator_lineage(freeze)
    row_audit = _check_row_cache_alignment(freeze, preflight)
    _need(row_audit["r06_prediction_sha256"] == preflight["array_hashes"]["r06_oof_prediction"],
          "Independent R06 prediction hash differs from preflight")
    resources = preflight.get("resource_snapshots", [])
    _need(len(resources) >= 2 and all(item.get("status") == "PASS" for item in resources),
          "Resource preflight did not pass at each recorded boundary")
    body = {
        "schema_version": 1, "experiment_id": pilot.EXPERIMENT_ID,
        "status": "PASS_SOURCE_READY_PENDING_ROOT_RELEASE",
        "prefit_freeze_sha256": freeze_sha,
        "identity_preflight_sha256": preflight_sha,
        "synthetic_qa_sha256": synthetic_sha,
        "historical_comparator_audit": comparator_audit,
        "row_identity_audit": row_audit,
        "official_lp_starts": 0,
        "root_source_pass_exists": False,
        "test_labels_read": False, "csv_created": False, "kaggle_accessed": False,
        "review_scope": "source and input preparation only; no official fit executed",
    }
    if write_receipt:
        body["signature"] = pilot.canonical_sha(body)
        pilot.write_once(pilot.VERIFICATION, body)
        return {"status": body["status"], "path": str(pilot.VERIFICATION.resolve()),
                "sha256": _sha(pilot.VERIFICATION), "fit_starts": 0}
    return body


def main() -> int:
    try:
        result = verify_source_ready(write_receipt=True)
        print(json.dumps(result, sort_keys=True, ensure_ascii=False))
        return 0
    except (VerificationError, pilot.GateError) as exc:
        print(json.dumps({"status": "FAIL_SOURCE_READY", "error": str(exc)},
                         ensure_ascii=False), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
