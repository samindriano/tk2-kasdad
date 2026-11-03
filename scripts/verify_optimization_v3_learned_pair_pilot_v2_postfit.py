"""Independent byte/identity and frozen-metric verification for FM V2 postfit repair."""
from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys
import zipfile
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import optimization_v3_learned_pair_pilot_v2 as pilot
import optimization_v3_positional_bag_offset_v1 as offset_inputs
import verify_optimization_v3_learned_pair_pilot_v2 as frozen_verifier

AMENDMENT_ID = "LEARNED_PAIR_PILOT_V2_POSTFIT_SERIALIZATION"
AMENDMENT_ROOT_DEFAULT = ROOT / "experiments/runs/optimization_v3_research_next_v1/LEARNED_PAIR_PILOT_V2_POSTFIT"
AUTH_DOC = pilot.REPORT_ROOT / "POSTFIT_SERIALIZATION_AUTHORIZATION.md"
BLOCKER = pilot.RUN_ROOT / "POSTFIT_BLOCKER.json"
SOURCE_FILES = (
    "scripts/optimization_v3_learned_pair_pilot_v2_postfit.py",
    "scripts/verify_optimization_v3_learned_pair_pilot_v2_postfit.py",
    "tests/test_learned_pair_pilot_v2_postfit.py",
)
EXPECTED_TRAINING_COMMIT = "2e5418c5181e08c77c6c4c0e135835d5f4fc84a2"
EXPECTED_AUTHORIZATION_COMMIT = "626eb4298021f6f2cbd262241ec85bf4faa9cdd7"
EXPECTED_LEDGER_SHA256 = "ccf18357d2d7fcb513e9f36b8fb2ec39688d28ab3d43471b5db345c35fce6b14"
EXPECTED_BLOCKER_SHA256 = "561dda827f000bf7e681026e1504731249fa339db08689e5f1e1e95a6483f323"


class VerificationError(RuntimeError):
    pass


def _sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _sha256_file(path: Path) -> str:
    return pilot._sha256_file(path)


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"),
                      allow_nan=False, default=lambda x: x.item() if isinstance(x, np.generic) else str(x)).encode("utf-8")


def _canonical_sha256(value: Any) -> str:
    return _sha256_bytes(_canonical(value))


def _normalized_worktree_bytes(raw: bytes) -> bytes:
    return raw.replace(b"\r\n", b"\n")


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _git_blob(commit: str, rel: str) -> bytes:
    proc = subprocess.run(["git", "-C", str(ROOT), "show", f"{commit}:{rel}"],
                          check=True, capture_output=True)
    return proc.stdout


def _member_header(raw: bytes) -> tuple[tuple[int, ...], bool, np.dtype[Any]]:
    from numpy.lib.format import read_magic, read_array_header_1_0, read_array_header_2_0
    with io.BytesIO(raw) as stream:
        version = read_magic(stream)
        if version == (1, 0):
            shape, fortran_order, dtype = read_array_header_1_0(stream)
        elif version in ((2, 0), (3, 0)):
            shape, fortran_order, dtype = read_array_header_2_0(stream)
        else:
            raise VerificationError(f"Unsupported NPY version: {version}")
    return tuple(shape), bool(fortran_order), np.dtype(dtype)


def _archive_members(path: Path) -> dict[str, bytes]:
    with zipfile.ZipFile(path, "r") as archive:
        names = archive.namelist()
        if len(names) != len(set(names)) or any(not name.endswith(".npy") for name in names):
            raise VerificationError(f"NPZ member namespace invalid: {path}")
        return {name: archive.read(name) for name in names}


def _safe_member(path: Path, name: str) -> np.ndarray:
    with np.load(path, allow_pickle=False) as archive:
        if name not in archive.files:
            raise VerificationError(f"Missing NPZ member {name}: {path}")
        return np.asarray(archive[name]).copy()


def _member_records(members: dict[str, bytes], *, allow_object_id: bool) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    for name, raw in members.items():
        shape, fortran_order, dtype = _member_header(raw)
        if dtype.hasobject:
            if not (allow_object_id and name == "score_record_ids.npy"):
                raise VerificationError(f"Unexpected object dtype member: {name}")
            records[name] = {"raw_npy_sha256": _sha256_bytes(raw), "shape": list(shape),
                             "dtype": dtype.str, "fortran_order": fortran_order, "values_read": False}
            continue
        array = np.asarray(np.load(io.BytesIO(raw), allow_pickle=False)).copy()
        if tuple(array.shape) != shape or array.dtype != dtype:
            raise VerificationError(f"NPY header/data mismatch: {name}")
        records[name] = {"raw_npy_sha256": _sha256_bytes(raw), "shape": list(shape),
                         "dtype": dtype.str, "fortran_order": fortran_order, "values_read": True,
                         "value_sha256": offset_inputs.sha256_array(array)}
    return records


def _verify_receipt_signature(receipt: dict[str, Any]) -> dict[str, Any]:
    body = dict(receipt)
    signature = body.pop("receipt_sha256", None)
    if signature != _canonical_sha256(body):
        raise VerificationError("Postfit amendment receipt signature mismatch")
    return body


def _verify_member_pair(original: dict[str, bytes], derived: dict[str, bytes]) -> tuple[dict[str, Any], dict[str, Any]]:
    if list(original) != list(derived) or "score_record_ids.npy" not in original:
        raise VerificationError("Original and derived NPZ members/order differ")
    old_records = _member_records(original, allow_object_id=True)
    new_records = _member_records(derived, allow_object_id=False)
    if any(original[name] != derived[name] for name in original if name != "score_record_ids.npy"):
        raise VerificationError("Safe member NPY bytes changed during serialization amendment")
    old_shape, old_fortran, old_dtype = _member_header(original["score_record_ids.npy"])
    new_shape, new_fortran, new_dtype = _member_header(derived["score_record_ids.npy"])
    if not old_dtype.hasobject or new_dtype.kind != "U" or old_shape != new_shape:
        raise VerificationError("score_record_ids was not converted from object to same-shape Unicode")
    if old_fortran != new_fortran:
        raise VerificationError("score_record_ids array order changed")
    return old_records, new_records


def _verify_identity_vectors(rows: np.ndarray, ids: np.ndarray, cycles: np.ndarray,
                             expected_rows: np.ndarray, expected_ids: np.ndarray,
                             expected_cycles: np.ndarray) -> None:
    row_values = np.asarray(rows, dtype=np.int64)
    id_values = np.asarray(ids).astype(str)
    cycle_values = np.asarray(cycles, dtype=np.int64)
    if (row_values.ndim != 1 or id_values.shape != row_values.shape or cycle_values.shape != row_values.shape
            or not np.array_equal(row_values, np.asarray(expected_rows, dtype=np.int64))
            or not np.array_equal(id_values, np.asarray(expected_ids).astype(str))
            or not np.array_equal(cycle_values, np.asarray(expected_cycles, dtype=np.int64))):
        raise VerificationError("Postfit score row/ID/cycle identity mismatch")


def _validate_committed_source_version(commit: str, expected_hashes: dict[str, str],
                                       expected_blob_hashes: dict[str, str], source_files: tuple[str, ...]) -> None:
    if set(expected_hashes) != set(source_files):
        raise VerificationError("Committed amendment source list differs from its receipt")
    for rel, expected in expected_hashes.items():
        blob = _git_blob(commit, rel)
        checkout_hashes = {_sha256_bytes(blob), _sha256_bytes(blob.replace(b"\n", b"\r\n"))}
        if (_sha256_bytes(blob) != expected_blob_hashes.get(rel) or expected not in checkout_hashes):
            raise VerificationError(f"Amendment source commit blob changed: {rel}")


def _validate_current_verification_source() -> dict[str, Any]:
    source_commit = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"],
                                   check=True, capture_output=True, text=True).stdout.strip()
    hashes: dict[str, str] = {}
    blob_hashes: dict[str, str] = {}
    for rel in SOURCE_FILES:
        path = ROOT / rel
        raw = path.read_bytes()
        blob = _git_blob(source_commit, rel)
        if _normalized_worktree_bytes(raw) != blob:
            raise VerificationError(f"Current amendment verifier source is not committed: {rel}")
        hashes[rel] = _sha256_bytes(raw)
        blob_hashes[rel] = _sha256_bytes(blob)
    return {"source_commit": source_commit, "source_hashes": hashes,
            "source_git_blob_sha256": blob_hashes}


def verify_export(run_root: Path) -> dict[str, Any]:
    run_root = Path(run_root)
    receipt_path = run_root / "AMENDMENT_RECEIPT.json"
    receipt = _read_json(receipt_path)
    receipt_body = _verify_receipt_signature(receipt)
    if (receipt.get("status") != "EXPORTED_ZERO_FIT_SERIALIZATION_AMENDMENT"
            or receipt.get("amendment_id") != AMENDMENT_ID
            or receipt.get("card_id") != pilot.CARD_ID
            or receipt.get("training_source_commit") != EXPECTED_TRAINING_COMMIT
            or receipt.get("original_blocker_sha256") != EXPECTED_BLOCKER_SHA256
            or receipt.get("original_ledger_sha256") != EXPECTED_LEDGER_SHA256
            or receipt.get("authorization_sha256") != _sha256_file(AUTH_DOC)):
        raise VerificationError("Postfit amendment authorization/lineage binding mismatch")
    auth_blob = _git_blob(EXPECTED_AUTHORIZATION_COMMIT,
                          "reports/optimization_v3/learned_pair_pilot_v2/POSTFIT_SERIALIZATION_AUTHORIZATION.md")
    if (_sha256_bytes(auth_blob) != receipt.get("authorization_git_blob_sha256")
            or _normalized_worktree_bytes(AUTH_DOC.read_bytes()) != auth_blob):
        raise VerificationError("Reviewed postfit authorization does not match its committed Git blob")
    training_freeze_path = pilot.RUN_ROOT / "FREEZE.json"
    original_ready_path = pilot.RUN_ROOT / "PREDICTIONS_READY.json"
    original_blocker_path = BLOCKER
    freeze_raw = training_freeze_path.read_bytes()
    original_ready_raw = original_ready_path.read_bytes()
    original_ready = json.loads(original_ready_raw)
    frozen = json.loads(freeze_raw)
    freeze_body = dict(frozen)
    freeze_sig = freeze_body.pop("freeze_signature", None)
    if (_canonical_sha256(freeze_body) != freeze_sig
            or _sha256_bytes(freeze_raw) != receipt.get("original_freeze_sha256")
            or _sha256_bytes(original_ready_raw) != receipt.get("original_ready_sha256")
            or _sha256_file(original_blocker_path) != receipt.get("original_blocker_sha256")
            or _sha256_file(pilot.RUN_ROOT / "SYNTHETIC_QA.json") != receipt.get("original_synthetic_qa_sha256")
            or frozen.get("freeze_signature") != receipt.get("freeze_signature")
            or frozen.get("source_commit") != EXPECTED_TRAINING_COMMIT
            or frozen.get("input_sha256") != receipt.get("input_sha256")):
        raise VerificationError("Original freeze/readiness/blocker preservation check failed")
    if _sha256_file(pilot.TRAIN) != receipt.get("input_sha256"):
        raise VerificationError("Official Train input hash changed")
    current_saved_inputs = offset_inputs.verify_saved_inputs(include_current_code=True)
    if (current_saved_inputs.get("input_sha256") != receipt.get("input_sha256")
            or _canonical_sha256(current_saved_inputs) != receipt.get("current_saved_inputs_validation_sha256")
            or current_saved_inputs != receipt.get("current_saved_inputs_validation")):
        raise VerificationError("Current positional/A02 saved-input validation differs from amendment receipt")
    for rel, expected in frozen["source_hashes"].items():
        blob = _git_blob(frozen["source_commit"], rel)
        if (_sha256_file(ROOT / rel) != expected
                or _sha256_bytes(blob) != receipt.get("training_source_git_blob_sha256", {}).get(rel)
                or _normalized_worktree_bytes((ROOT / rel).read_bytes()) != blob):
            raise VerificationError(f"Frozen training source current file/Git blob mismatch: {rel}")
    _validate_committed_source_version(receipt["amendment_source_commit"], receipt["amendment_source_hashes"],
                                       receipt["amendment_source_git_blob_sha256"], SOURCE_FILES)
    verification_source = _validate_current_verification_source()
    if original_ready.get("freeze_signature") != freeze_sig or original_ready.get("query_labels_opened") is not False:
        raise VerificationError("Original prediction readiness is not the frozen closed-label object")
    if ((run_root / "FREEZE.json").read_bytes() != freeze_raw
        or (run_root / "SYNTHETIC_QA.json").read_bytes() != (pilot.RUN_ROOT / "SYNTHETIC_QA.json").read_bytes()
        or (run_root / "SOURCE_PREDICTIONS_READY.json").read_bytes() != original_ready_raw
        or (run_root / "SOURCE_POSTFIT_BLOCKER.json").read_bytes() != original_blocker_path.read_bytes()):
        raise VerificationError("Verification-view preservation inputs differ from original artifacts")
    if _sha256_file(pilot.LEDGER) != EXPECTED_LEDGER_SHA256:
        raise VerificationError("Fit/prediction/prelabel stage modified the shared budget ledger")
    events = pilot._ledger_events()
    snapshot = offset_inputs.ledger_snapshot(events)
    expected_snapshot = {
        "global_cap": 117,
        "effective_category_caps": {"continuation": 38, "control_replay": 2, "pilot": 76, "technical_retry": 1},
        "active_reservations_by_category": {"continuation": 38, "control_replay": 2, "pilot": 73, "technical_retry": 1},
        "active_reservations": 114, "fit_starts": 114, "fit_saved": 110,
        "fit_failed": 4, "fit_pending": 0, "event_rows": 291,
    }
    if snapshot != expected_snapshot:
        raise VerificationError(f"Prelabel budget snapshot differs: {snapshot}")
    for name in ("QUERY_LABEL_ACCESS.json", "QUERY_LABEL_READ.json", "RESULTS.json", "PRELABEL_VERIFICATION.json"):
        if (pilot.RUN_ROOT / name).exists():
            raise VerificationError(f"Original frozen run root was modified with {name}")
    ready_path = run_root / "PREDICTIONS_READY.json"
    ready = _read_json(ready_path)
    if (ready.get("freeze_signature") != freeze_sig
            or ready.get("query_labels_opened") is not False
            or ready.get("postfit_amendment", {}).get("receipt_sha256") != _sha256_file(receipt_path)
            or ready.get("postfit_amendment", {}).get("original_ready_sha256") != receipt.get("original_ready_sha256")
            or set(ready.get("predictions", {})) != {"E67", "E75"}):
        raise VerificationError("Amended readiness does not preserve frozen source boundary")
    train_identity = offset_inputs._read_identity_frame()
    train_ids_all = train_identity["record_id"].astype(str).to_numpy(dtype=str)
    cycles_all = train_identity["cycle_code"].to_numpy(dtype=np.int64)
    a02_freeze = frozen_verifier._verify_a02_freeze()
    origins_out: dict[str, Any] = {}
    for origin in (67, 75):
        key = f"E{origin}"
        old_item = original_ready["predictions"][key]
        new_item = ready["predictions"][key]
        old_path = ROOT / old_item["path"]
        new_path = ROOT / new_item["path"]
        origin_receipt = receipt["origins"][key]
        if (new_item.get("path") != origin_receipt.get("derived_prediction_path")
                or new_item.get("sha256") != origin_receipt.get("derived_prediction_sha256")
                or new_item.get("fit_receipts") != old_item.get("fit_receipts")
                or _sha256_file(old_path) != old_item.get("sha256")
                or _sha256_file(new_path) != new_item.get("sha256")
                or _sha256_file(old_path) != origin_receipt.get("original_prediction_sha256")):
            raise VerificationError(f"E{origin} original/derived prediction hashes or ready bindings differ")
        old_members, new_members = _archive_members(old_path), _archive_members(new_path)
        try:
            old_records, new_records = _verify_member_pair(old_members, new_members)
        except VerificationError as exc:
            raise VerificationError(f"E{origin} member pair invalid: {exc}") from exc
        if old_records != origin_receipt.get("member_audit", {}).get("original_members") or new_records != origin_receipt.get("member_audit", {}).get("derived_members"):
            raise VerificationError(f"E{origin} per-member receipt hash/dtype/shape/value audit differs")
        if any(old_members[name] != new_members[name] for name in old_members if name != "score_record_ids.npy"):
            raise VerificationError(f"E{origin} safe raw member bytes changed during ID repair")
        if _sha256_bytes(old_members["score_record_ids.npy"]) != origin_receipt["member_audit"]["original_members"]["score_record_ids.npy"]["raw_npy_sha256"]:
            raise VerificationError(f"E{origin} old object-ID member hash changed")
        old_id_shape, old_id_order, old_id_dtype = _member_header(old_members["score_record_ids.npy"])
        new_id_shape, new_id_order, new_id_dtype = _member_header(new_members["score_record_ids.npy"])
        if (not old_id_dtype.hasobject or old_id_shape != new_id_shape
                or new_id_dtype.kind != "U" or new_id_shape != old_id_shape or old_id_order != new_id_order):
            raise VerificationError(f"E{origin} object-to-Unicode ID member conversion is invalid")
        rows = _safe_member(new_path, "score_rows").astype(np.int64)
        ids = _safe_member(new_path, "score_record_ids").astype(str)
        cycles = _safe_member(new_path, "score_cycles").astype(np.int64)
        if (len(rows) != len(np.unique(rows)) or not np.all(np.diff(rows) > 0)
                or np.any(rows < 0) or np.any(rows >= len(train_identity))):
            raise VerificationError(f"E{origin} derived score IDs/cycles do not match official Train rows")
        try:
            _verify_identity_vectors(rows, ids, cycles, rows, train_ids_all[rows], cycles_all[rows])
        except VerificationError as exc:
            raise VerificationError(f"E{origin} derived score IDs/cycles do not match official Train rows") from exc
        historical_path = pilot.PILOT_ROOT / f"E{origin}" / f"predictions_E{origin}.npz"
        historical = {name: _safe_member(historical_path, name)
                      for name in ("fit_rows", "score_rows", "score_record_ids", "score_cycles")}
        historical_ids = historical["score_record_ids"].astype(str)
        try:
            _verify_identity_vectors(rows, ids, cycles, historical["score_rows"], historical_ids,
                                     historical["score_cycles"])
        except VerificationError as exc:
            raise VerificationError(f"E{origin} derived identities differ from verified historical positional predictions") from exc
        a02_meta = frozen["saved_inputs"]["origins"][key]["a02_observed_reference"]
        a02_probability = frozen_verifier._verify_a02_observed(
            origin, historical["fit_rows"].astype(np.int64), rows, ids, cycles, a02_meta, a02_freeze)
        if not np.array_equal(_safe_member(new_path, "prob_a02_observed").astype(np.float64), a02_probability):
            raise VerificationError(f"E{origin} derived A02 probability differs from verified reference")
        if (offset_inputs.sha256_array(rows) != origin_receipt.get("score_rows_sha256")
                or offset_inputs.sha256_array(ids) != origin_receipt.get("score_record_ids_sha256")
                or offset_inputs.sha256_array(cycles) != origin_receipt.get("score_cycles_sha256")
                or origin_receipt.get("historical_identity_exact") is not True
                or origin_receipt.get("official_train_identity_exact") is not True
                or origin_receipt.get("a02_identity_and_probability_verified") is not True):
            raise VerificationError(f"E{origin} new identity receipt does not bind source identities")
        origins_out[key] = {
            "rows": len(rows), "ids_sha256": offset_inputs.sha256_array(ids),
            "cycles_sha256": offset_inputs.sha256_array(cycles),
            "old_archive_sha256": _sha256_file(old_path), "derived_archive_sha256": _sha256_file(new_path),
            "safe_member_count": len(old_members) - 1,
            "safe_member_raw_byte_identical": True,
            "object_id_values_deserialized": False,
            "derived_unicode_dtype": new_id_dtype.str,
            "historical_and_a02_identity_alignment": True,
        }
    return {
        "status": "PASS_IDENTITY_AMENDMENT", "amendment_id": AMENDMENT_ID,
        "amendment_receipt_sha256": _sha256_file(receipt_path),
        "verification_source": verification_source,
        "identity_verification": origins_out,
        "original_freeze_ready_blocker_and_outputs_preserved": True,
        "query_labels_opened": False, "ledger_sha256_unchanged": EXPECTED_LEDGER_SHA256,
        "fit_calls": 0,
    }


def verify_prelabel(run_root: Path) -> dict[str, Any]:
    identity = verify_export(run_root)
    original_verifier_path = ROOT / "scripts/verify_optimization_v3_learned_pair_pilot_v2.py"
    proc = subprocess.run([sys.executable, str(original_verifier_path), "--official", "--run-root", str(run_root)],
                          cwd=ROOT, check=False, capture_output=True, text=True)
    if proc.returncode != 0:
        raise VerificationError(f"Frozen full pre-label verifier failed: {proc.stdout or proc.stderr}")
    original = json.loads(proc.stdout)
    if (original.get("status") != "PASS" or original.get("query_labels_opened") is not False
            or original.get("freeze_signature") != _read_json(run_root / "FREEZE.json").get("freeze_signature")
            or _sha256_file(pilot.LEDGER) != EXPECTED_LEDGER_SHA256):
        raise VerificationError("Frozen full prelabel verifier did not pass with query labels closed")
    body = {
        "status": "PASS", "amendment_id": AMENDMENT_ID, "card_id": pilot.CARD_ID,
        "source_commit": original["source_commit"], "freeze_signature": original["freeze_signature"],
        "input_sha256": original["input_sha256"], "query_labels_opened": False,
        "amendment_receipt_sha256": _sha256_file(run_root / "AMENDMENT_RECEIPT.json"),
        "identity_verification_sha256": _sha256_file(run_root / "IDENTITY_VERIFICATION.json"),
        "predictions_ready_sha256": _sha256_file(run_root / "PREDICTIONS_READY.json"),
        "prediction_output_sha256s": {key: _sha256_file(ROOT / item["path"])
                                      for key, item in _read_json(run_root / "PREDICTIONS_READY.json")["predictions"].items()},
        "budget_ledger_sha256": _sha256_file(pilot.LEDGER),
        "original_independent_prelabel_verification": original,
        "verifier": "independent postfit identity audit plus unchanged original V2 official verifier in an isolated verification view",
    }
    body["verification_signature"] = _canonical_sha256(body)
    return body


def verify_scored(run_root: Path) -> dict[str, Any]:
    result_path = run_root / "RESULTS.json"
    result = _read_json(result_path)
    scored = frozen_verifier.verify_scored(run_root)
    receipt = _read_json(run_root / "AMENDMENT_RECEIPT.json")
    if (result.get("postfit_amendment_id") != AMENDMENT_ID
            or result.get("postfit_amendment_receipt_sha256") != _sha256_file(run_root / "AMENDMENT_RECEIPT.json")
            or result.get("export_source_commit") != receipt.get("amendment_source_commit")
            or result.get("source_commit") != EXPECTED_TRAINING_COMMIT
            or scored.get("status") != "PASS_INDEPENDENT_METRICS_AND_GATES"):
        raise VerificationError("Scored metrics do not bind the authorized postfit amendment lineage")
    raw_ledger = pilot.LEDGER.read_bytes()
    lines = raw_ledger.splitlines(keepends=True)
    events = [json.loads(line) for line in lines]
    completions = [item for item in events if item.get("event") == "CARD_COMPLETED" and item.get("card_id") == pilot.CARD_ID]
    if _sha256_bytes(raw_ledger) == EXPECTED_LEDGER_SHA256:
        if completions:
            raise VerificationError("Pre-completion ledger unexpectedly already contains CARD_COMPLETED")
    else:
        if len(lines) != 292 or _sha256_bytes(b"".join(lines[:-1])) != EXPECTED_LEDGER_SHA256:
            raise VerificationError("Ledger does not contain exactly one append after the prelabel baseline")
        event = json.loads(lines[-1])
        if (len(completions) != 1 or event.get("event") != "CARD_COMPLETED"
                or event.get("card_id") != pilot.CARD_ID or event.get("amendment_id") != AMENDMENT_ID
                or event.get("result_sha256") != _sha256_file(result_path)
                or event.get("scored_verification_sha256") != _sha256_file(run_root / "SCORED_VERIFICATION.json")
                or event.get("amendment_receipt_sha256") != _sha256_file(run_root / "AMENDMENT_RECEIPT.json")):
            raise VerificationError("Single normal completion event is not bound to the amended result")
    return scored


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--export", action="store_true")
    group.add_argument("--prelabel", action="store_true")
    group.add_argument("--scored", action="store_true")
    parser.add_argument("--run-root", type=Path, required=True)
    args = parser.parse_args()
    try:
        if args.export:
            result = verify_export(args.run_root)
        elif args.prelabel:
            result = verify_prelabel(args.run_root)
        else:
            result = verify_scored(args.run_root)
        print(json.dumps(result, sort_keys=True, ensure_ascii=False, allow_nan=False))
        return 0
    except Exception as exc:
        print(json.dumps({"status": "FAIL", "error_type": type(exc).__name__, "error": str(exc)},
                         sort_keys=True, ensure_ascii=False, allow_nan=False))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
