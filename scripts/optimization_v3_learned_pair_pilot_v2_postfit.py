"""Zero-fit serialization repair and evaluation adapter for learned-pair V2."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import zipfile
from typing import Any, Mapping

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import optimization_v3_learned_pair_pilot_v2 as pilot
import optimization_v3_positional_bag_offset_v1 as offset_inputs

CARD_ID = "LEARNED_PAIR_PILOT_V2"
AMENDMENT_ID = "LEARNED_PAIR_PILOT_V2_POSTFIT_SERIALIZATION"
ORIGINAL_RUN_ROOT = pilot.RUN_ROOT
AMENDMENT_ROOT = ROOT / "experiments/runs/optimization_v3_research_next_v1/LEARNED_PAIR_PILOT_V2_POSTFIT"
REPORT_ROOT = pilot.REPORT_ROOT
AUTH_DOC = REPORT_ROOT / "POSTFIT_SERIALIZATION_AUTHORIZATION.md"
POSTFIT_BLOCKER = ORIGINAL_RUN_ROOT / "POSTFIT_BLOCKER.json"
AMENDMENT_SOURCE_FILES = (
    "scripts/optimization_v3_learned_pair_pilot_v2_postfit.py",
    "scripts/verify_optimization_v3_learned_pair_pilot_v2_postfit.py",
    "tests/test_learned_pair_pilot_v2_postfit.py",
)
EXPECTED_TRAINING_COMMIT = "2e5418c5181e08c77c6c4c0e135835d5f4fc84a2"
EXPECTED_AUTHORIZATION_COMMIT = "626eb4298021f6f2cbd262241ec85bf4faa9cdd7"
EXPECTED_LEDGER_SHA256 = "ccf18357d2d7fcb513e9f36b8fb2ec39688d28ab3d43471b5db345c35fce6b14"
EXPECTED_BLOCKER_SHA256 = "561dda827f000bf7e681026e1504731249fa339db08689e5f1e1e95a6483f323"
RECEIPT_NAME = "AMENDMENT_RECEIPT.json"
READY_NAME = "PREDICTIONS_READY.json"
PRELABEL_NAME = "PRELABEL_VERIFICATION.json"
RESULT_NAME = "RESULTS.json"
SCORED_VERIFICATION_NAME = "SCORED_VERIFICATION.json"


class PostfitError(RuntimeError):
    pass


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"),
                      allow_nan=False, default=lambda x: x.item() if isinstance(x, np.generic) else str(x)).encode("utf-8")


def canonical_sha256(value: Any) -> str:
    return sha256_bytes(canonical_bytes(value))


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _json_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, indent=2, allow_nan=False,
                      default=lambda x: x.item() if isinstance(x, np.generic) else str(x)).encode("utf-8") + b"\n"


def _write_create_once(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != data:
            raise PostfitError(f"Create-once amendment artifact differs on resume: {path}")
        return
    fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    try:
        with os.fdopen(fd, "wb", closefd=False) as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
    finally:
        os.close(fd)


def _npy_header(raw: bytes) -> tuple[tuple[int, ...], bool, np.dtype[Any]]:
    with io.BytesIO(raw) as stream:
        version = np.lib.format.read_magic(stream)
        if version == (1, 0):
            shape, fortran_order, dtype = np.lib.format.read_array_header_1_0(stream)
        elif version in ((2, 0), (3, 0)):
            shape, fortran_order, dtype = np.lib.format.read_array_header_2_0(stream)
        else:
            raise PostfitError(f"Unsupported NPY format version: {version}")
    return tuple(shape), bool(fortran_order), np.dtype(dtype)


def npy_bytes(array: np.ndarray) -> bytes:
    stream = io.BytesIO()
    np.save(stream, array, allow_pickle=False)
    return stream.getvalue()


def archive_members(path: Path) -> dict[str, bytes]:
    with zipfile.ZipFile(path, "r") as archive:
        names = archive.namelist()
        if len(names) != len(set(names)) or any(not name.endswith(".npy") for name in names):
            raise PostfitError(f"Unexpected or duplicate NPZ members: {path}")
        return {name: archive.read(name) for name in names}


def read_safe_member(path: Path, key: str) -> np.ndarray:
    with np.load(path, allow_pickle=False) as archive:
        if key not in archive.files:
            raise PostfitError(f"Missing safe NPZ array {key}: {path}")
        return np.asarray(archive[key]).copy()


def _member_manifest(members: Mapping[str, bytes], *, skip_object_values: bool) -> dict[str, Any]:
    manifest: dict[str, Any] = {}
    for name, raw in members.items():
        shape, fortran_order, dtype = _npy_header(raw)
        entry: dict[str, Any] = {
            "raw_npy_sha256": sha256_bytes(raw), "shape": list(shape),
            "dtype": dtype.str, "fortran_order": fortran_order,
        }
        if dtype.hasobject:
            if not skip_object_values:
                raise PostfitError(f"Object dtype is forbidden in this archive member: {name}")
            entry["values_read"] = False
        else:
            entry["values_read"] = True
            array = np.asarray(np.load(io.BytesIO(raw), allow_pickle=False)).copy()
            if list(array.shape) != list(shape) or array.dtype != dtype:
                raise PostfitError(f"NPY member header/read mismatch: {name}")
            entry["value_sha256"] = pilot.offset_inputs.sha256_array(array)
        manifest[name] = entry
    return manifest


def rebuild_npz_preserving_members(original_path: Path, derived_path: Path,
                                   score_ids: np.ndarray) -> dict[str, Any]:
    """Replace only score_record_ids.npy; never deserialize the old object member."""
    members = archive_members(original_path)
    id_member = "score_record_ids.npy"
    if id_member not in members:
        raise PostfitError(f"Original prediction archive has no {id_member}")
    original_manifest = _member_manifest(members, skip_object_values=True)
    id_header = _npy_header(members[id_member])
    if not id_header[2].hasobject or id_header[0] != tuple(np.asarray(score_ids).shape):
        raise PostfitError("Original score ID member is not the expected object array/shape")
    if any(name != id_member and entry["dtype"] == "|O" for name, entry in original_manifest.items()):
        raise PostfitError("Object dtype exists outside the authorized score ID member")
    ids = np.asarray(score_ids)
    if ids.ndim != 1 or ids.dtype.kind != "U":
        raise PostfitError("Replacement score IDs must be a one-dimensional Unicode array")
    max_width = max((len(item) for item in ids.tolist()), default=1)
    fixed_ids = ids.astype(f"<U{max_width}", copy=False)
    if not np.array_equal(fixed_ids.astype(str), ids.astype(str)):
        raise PostfitError("Replacement score IDs would be truncated")
    replacement = npy_bytes(fixed_ids)
    new_members = {name: (replacement if name == id_member else raw) for name, raw in members.items()}
    derived_manifest = _member_manifest(new_members, skip_object_values=False)
    for name in members:
        if name == id_member:
            if derived_manifest[name]["dtype"].split("<U")[-1] == "" or not derived_manifest[name]["dtype"].endswith("U" + str(max_width)):
                raise PostfitError("Replacement score IDs are not the expected fixed-width Unicode dtype")
            continue
        if members[name] != new_members[name] or original_manifest[name] != derived_manifest[name]:
            raise PostfitError(f"Unchanged NPZ member failed byte/value identity: {name}")
    if derived_path.exists():
        existing = archive_members(derived_path)
        if list(existing) != list(new_members) or any(existing[name] != new_members[name] for name in new_members):
            raise PostfitError(f"Existing derived archive differs; refusing overwrite: {derived_path}")
    else:
        derived_path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(prefix=".postfit-", suffix=".npz", dir=derived_path.parent, delete=False) as temp:
            temp_path = Path(temp.name)
        try:
            with zipfile.ZipFile(temp_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
                for name, raw in new_members.items():
                    archive.writestr(name, raw)
            with zipfile.ZipFile(temp_path, "r") as archive:
                if archive.namelist() != list(members):
                    raise PostfitError("Derived NPZ member ordering differs from the frozen output")
            os.link(temp_path, derived_path)
        finally:
            temp_path.unlink(missing_ok=True)
    return {
        "original_archive_sha256": pilot._sha256_file(original_path),
        "derived_archive_sha256": pilot._sha256_file(derived_path),
        "member_order": list(members), "original_members": original_manifest,
        "derived_members": derived_manifest,
        "changed_member": id_member,
        "changed_member_reason": "fixed-width Unicode identity rebuilt from official Train row IDs and verified historical/A02 identities; original object payload was not deserialized",
    }


def _git_output(*args: str) -> bytes:
    process = subprocess.run(["git", "-C", str(ROOT), *args], check=True,
                             stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return process.stdout


def _normalized_worktree_bytes(raw: bytes) -> bytes:
    return raw.replace(b"\r\n", b"\n")


def _file_hashes(paths: tuple[str, ...]) -> dict[str, str]:
    return {rel: pilot._sha256_file(ROOT / rel) for rel in paths}


def _validate_training_lineage(frozen: Mapping[str, Any]) -> dict[str, Any]:
    if frozen.get("source_commit") != EXPECTED_TRAINING_COMMIT:
        raise PostfitError("Frozen training commit differs from the reviewed V2 source commit")
    training_blob_hashes: dict[str, str] = {}
    for rel, expected in frozen["source_hashes"].items():
        path = ROOT / rel
        if not path.is_file() or pilot._sha256_file(path) != expected:
            raise PostfitError(f"Frozen training source hash changed: {rel}")
        blob = _git_output("show", f"{frozen['source_commit']}:{rel}")
        if _normalized_worktree_bytes(path.read_bytes()) != blob:
            raise PostfitError(f"Original training checkout is not an EOL-only conversion of its Git blob: {rel}")
        training_blob_hashes[rel] = sha256_bytes(blob)
    if pilot._sha256_file(pilot.TRAIN) != frozen["input_sha256"]:
        raise PostfitError("Official Classification Train hash changed")
    if pilot._sha256_file(ORIGINAL_RUN_ROOT / "SYNTHETIC_QA.json") != frozen["synthetic_qa_sha256"]:
        raise PostfitError("Original source-bound synthetic QA receipt changed")
    if pilot._sha256_file(POSTFIT_BLOCKER) != EXPECTED_BLOCKER_SHA256:
        raise PostfitError("Original post-fit failure diagnostic changed")
    if not AUTH_DOC.is_file():
        raise PostfitError("Reviewed post-fit authorization document is missing")
    auth_blob = _git_output(
        "show", f"{EXPECTED_AUTHORIZATION_COMMIT}:reports/optimization_v3/learned_pair_pilot_v2/POSTFIT_SERIALIZATION_AUTHORIZATION.md")
    if _normalized_worktree_bytes(AUTH_DOC.read_bytes()) != auth_blob:
        raise PostfitError("Reviewed post-fit authorization differs from its committed Git blob")
    saved_inputs = offset_inputs.verify_saved_inputs(include_current_code=True)
    if saved_inputs.get("input_sha256") != frozen["input_sha256"]:
        raise PostfitError("Current saved positional/A02 inputs do not bind to the frozen official Train")
    source_hashes = _file_hashes(AMENDMENT_SOURCE_FILES)
    amendment_blob_hashes: dict[str, str] = {}
    for rel, expected in source_hashes.items():
        blob = _git_output("show", f"HEAD:{rel}")
        if _normalized_worktree_bytes((ROOT / rel).read_bytes()) != blob:
            raise PostfitError(f"Amendment source checkout differs from its committed Git blob: {rel}")
        amendment_blob_hashes[rel] = sha256_bytes(blob)
    if pilot._git("status", "--porcelain"):
        raise PostfitError("Amendment export requires committed clean adapter/verifier/tests")
    return {
        "amendment_source_hashes": source_hashes,
        "amendment_source_git_blob_sha256": amendment_blob_hashes,
        "training_source_git_blob_sha256": training_blob_hashes,
        "authorization_git_blob_sha256": sha256_bytes(auth_blob),
    }


def _ensure_copy(source: Path, destination: Path) -> None:
    raw = source.read_bytes()
    if destination.exists():
        if destination.read_bytes() != raw:
            raise PostfitError(f"Preserved source artifact differs on resume: {destination}")
        return
    _write_create_once(destination, raw)


def _ensure_hardlink(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if pilot._sha256_file(source) != pilot._sha256_file(destination):
            raise PostfitError(f"Verification-view hardlink/copy differs from original: {destination}")
        return
    os.link(source, destination)


def _read_archive_safe_subset(path: Path, names: tuple[str, ...]) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        result = {}
        for name in names:
            if name not in archive.files:
                raise PostfitError(f"Required source identity member missing: {name}")
            result[name] = np.asarray(archive[name]).copy()
    return result


def require_identity_alignment(rows: np.ndarray, ids: np.ndarray, cycles: np.ndarray,
                               expected_rows: np.ndarray, expected_ids: np.ndarray,
                               expected_cycles: np.ndarray, *, label: str) -> None:
    rows = np.asarray(rows, dtype=np.int64)
    ids = np.asarray(ids).astype(str)
    cycles = np.asarray(cycles, dtype=np.int64)
    expected_rows = np.asarray(expected_rows, dtype=np.int64)
    expected_ids = np.asarray(expected_ids).astype(str)
    expected_cycles = np.asarray(expected_cycles, dtype=np.int64)
    if (rows.ndim != 1 or ids.shape != rows.shape or cycles.shape != rows.shape
            or not np.array_equal(rows, expected_rows)
            or not np.array_equal(ids, expected_ids)
            or not np.array_equal(cycles, expected_cycles)):
        raise PostfitError(f"{label} row/ID/cycle identity alignment failed")


def _identity_for_origin(origin: int, frozen: Mapping[str, Any],
                         original_output: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    historical_path = pilot.PILOT_ROOT / f"E{origin}" / f"predictions_E{origin}.npz"
    historical = _read_archive_safe_subset(
        historical_path, ("fit_rows", "score_rows", "score_record_ids", "score_cycles"))
    score_rows = np.asarray(historical["score_rows"], dtype=np.int64)
    cycles_hist = np.asarray(historical["score_cycles"], dtype=np.int64)
    hist_ids = np.asarray(historical["score_record_ids"]).astype(str)
    if len(score_rows) != len(np.unique(score_rows)) or not np.all(np.diff(score_rows) > 0):
        raise PostfitError(f"E{origin} historical score rows are not unique/in source order")
    train_identity = offset_inputs._read_identity_frame()
    train_ids_all = train_identity["record_id"].astype(str).to_numpy(dtype=str)
    train_cycles_all = train_identity["cycle_code"].to_numpy(dtype=np.int64)
    if np.any(score_rows < 0) or np.any(score_rows >= len(train_identity)):
        raise PostfitError(f"E{origin} score rows exceed official Train identity bounds")
    ids = train_ids_all[score_rows]
    cycles = train_cycles_all[score_rows]
    with np.load(original_output, allow_pickle=False) as archive:
        output_rows = np.asarray(archive["score_rows"], dtype=np.int64)
        output_cycles = np.asarray(archive["score_cycles"], dtype=np.int64)
        a02_output = np.asarray(archive["prob_a02_observed"], dtype=np.float64)
        present = np.asarray(archive["present_mask"], dtype=bool)
        missing_fallback = np.asarray(archive["prob_missing_hgb"], dtype=np.float64)
        bag = np.asarray(archive["prob_bag"], dtype=np.float64)
        fm = np.asarray(archive["prob_fm"], dtype=np.float64)
        no_pair = np.asarray(archive["prob_nopair"], dtype=np.float64)
        raw_bag = np.asarray(archive["prob_bag_raw"], dtype=np.float64)
    require_identity_alignment(output_rows, ids, output_cycles, score_rows, hist_ids, cycles_hist,
                               label=f"E{origin} output/historical")
    require_identity_alignment(score_rows, ids, cycles, score_rows, hist_ids, cycles_hist,
                               label=f"E{origin} Train/historical")
    if (any(array.shape != score_rows.shape for array in (a02_output, present, missing_fallback, bag, fm, no_pair, raw_bag))
            or any(not np.isfinite(array).all() for array in (a02_output, missing_fallback, bag, fm, no_pair, raw_bag))
            or any(np.any((array < 0) | (array > 1)) for array in (a02_output, missing_fallback, bag, fm, no_pair, raw_bag))):
        raise PostfitError(f"E{origin} official Train/historical/output identities or safe arrays are misaligned")
    if not np.array_equal(bag[~present], missing_fallback[~present]):
        raise PostfitError(f"E{origin} cached C2_B missing-stream route changed")
    a02_freeze = pilot._verify_a02_freeze()
    meta = frozen["saved_inputs"]["origins"][f"E{origin}"]["a02_observed_reference"]
    a02_probability = pilot_verifier._verify_a02_observed(
        origin, np.asarray(historical["fit_rows"], dtype=np.int64), score_rows, ids,
        cycles, meta, a02_freeze)
    if not np.array_equal(a02_probability, a02_output):
        raise PostfitError(f"E{origin} A02 observed probabilities differ from original output")
    identity = {
        "origin": origin, "score_rows": len(score_rows),
        "score_rows_sha256": offset_inputs.sha256_array(score_rows),
        "score_record_ids_sha256": offset_inputs.sha256_array(ids),
        "score_cycles_sha256": offset_inputs.sha256_array(cycles),
        "historical_prediction_path": historical_path.relative_to(ROOT).as_posix(),
        "historical_prediction_sha256": pilot._sha256_file(historical_path),
        "historical_identity_exact": True,
        "official_train_identity_exact": True,
        "a02_reference": meta,
        "a02_identity_and_probability_verified": True,
        "a02_probability_sha256": offset_inputs.sha256_array(a02_probability),
        "missing_route_bit_exact": True,
    }
    width = max((len(item) for item in ids.tolist()), default=1)
    return score_rows, cycles, ids.astype(f"<U{width}", copy=False), identity


# Import independent verifier only after shared paths/constants are defined.
import verify_optimization_v3_learned_pair_pilot_v2 as pilot_verifier


def _validate_ledger_and_fit_receipts(frozen: Mapping[str, Any]) -> dict[str, Any]:
    ledger_sha = pilot._sha256_file(pilot.LEDGER)
    if ledger_sha != EXPECTED_LEDGER_SHA256:
        raise PostfitError(f"Live budget ledger changed before score completion: {ledger_sha}")
    events = pilot._ledger_events()
    snapshot = offset_inputs.ledger_snapshot(events)
    expected = {
        "global_cap": 117,
        "effective_category_caps": {"continuation": 38, "control_replay": 2, "pilot": 76, "technical_retry": 1},
        "active_reservations_by_category": {"continuation": 38, "control_replay": 2, "pilot": 73, "technical_retry": 1},
        "active_reservations": 114, "fit_starts": 114, "fit_saved": 110,
        "fit_failed": 4, "fit_pending": 0, "event_rows": 291,
    }
    if snapshot != expected:
        raise PostfitError(f"Ledger differs from reviewed pre-score snapshot: {snapshot}")
    own = pilot._own_events(events)
    if [event["fit_id"] for event in own["FIT_STARTED"]] != list(pilot.FIT_IDS):
        raise PostfitError("Four original V2 fit starts are missing, duplicate, or out of order")
    if {event["fit_id"] for event in own["FIT_SAVED"]} != set(pilot.FIT_IDS) or own["FIT_FAILED"]:
        raise PostfitError("V2 fit ledger is not four saved/zero failed")
    completions = [event for event in events if event.get("event") == "CARD_COMPLETED" and event.get("card_id") == CARD_ID]
    if completions:
        raise PostfitError("A V2 completion event already exists; do not score twice")
    receipts: dict[str, Any] = {}
    for fit_id in pilot.FIT_IDS:
        artifact, receipt_path = pilot._fit_paths(fit_id)
        receipt = pilot._validate_receipt(fit_id, frozen, artifact_path=artifact, receipt_path=receipt_path)
        saved = [event for event in own["FIT_SAVED"] if event["fit_id"] == fit_id]
        if len(saved) != 1 or saved[0].get("receipt_sha256") != pilot._sha256_file(receipt_path):
            raise PostfitError(f"V2 receipt is not bound by exactly one FIT_SAVED event: {fit_id}")
        receipts[fit_id] = {"artifact_sha256": pilot._sha256_file(artifact),
                            "receipt_sha256": pilot._sha256_file(receipt_path),
                            "optimizer_converged": receipt.get("optimizer_converged"),
                            "fit_population": receipt.get("fit_population")}
    return {"sha256": ledger_sha, "snapshot": snapshot, "fit_receipts": receipts}


def validate_baseline() -> tuple[dict[str, Any], dict[str, Any], dict[str, str]]:
    freeze_path = ORIGINAL_RUN_ROOT / "FREEZE.json"
    frozen_raw = freeze_path.read_bytes()
    frozen = json.loads(frozen_raw)
    body = dict(frozen)
    signature = body.pop("freeze_signature", None)
    if signature != canonical_sha256(body):
        raise PostfitError("Original V2 freeze signature is invalid")
    if _file_hashes(tuple(frozen["source_hashes"])) != frozen["source_hashes"]:
        raise PostfitError("Frozen source files no longer match original V2 source hashes")
    amend_sources = _validate_training_lineage(frozen)
    fit_accounting = _validate_ledger_and_fit_receipts(frozen)
    ready_path = ORIGINAL_RUN_ROOT / READY_NAME
    ready_raw = ready_path.read_bytes()
    ready = json.loads(ready_raw)
    if (ready.get("source_commit") != EXPECTED_TRAINING_COMMIT
            or ready.get("freeze_signature") != signature
            or ready.get("query_labels_opened") is not False
            or ready.get("fit_ids") != list(pilot.FIT_IDS)
            or set(ready.get("predictions", {})) != {"E67", "E75"}):
        raise PostfitError("Original readiness metadata differs from reviewed V2 state")
    blocker = _read_json(POSTFIT_BLOCKER)
    if blocker.get("query_labels_opened") is not False or blocker.get("results_created") is not False:
        raise PostfitError("Original blocker does not prove labels remained closed")
    if (ORIGINAL_RUN_ROOT / "QUERY_LABEL_ACCESS.json").exists() or (ORIGINAL_RUN_ROOT / "QUERY_LABEL_READ.json").exists():
        raise PostfitError("Original V2 root already contains query-label access/read receipts")
    for key, value in ready["predictions"].items():
        path = ROOT / value["path"]
        if not path.is_file() or pilot._sha256_file(path) != value["sha256"]:
            raise PostfitError(f"Original malformed prediction archive changed: {key}")
    original = {
        "freeze_path": freeze_path, "freeze_sha256": pilot._sha256_file(freeze_path),
        "frozen": frozen, "ready_path": ready_path, "ready_sha256": pilot._sha256_file(ready_path),
        "ready": ready, "blocker_sha256": pilot._sha256_file(POSTFIT_BLOCKER),
        "synthetic_qa_sha256": pilot._sha256_file(ORIGINAL_RUN_ROOT / "SYNTHETIC_QA.json"),
        "current_saved_inputs_validation": saved_inputs,
        "ledger": fit_accounting,
    }
    return original, fit_accounting, amend_sources


def _create_verification_view(original: Mapping[str, Any]) -> None:
    AMENDMENT_ROOT.mkdir(parents=True, exist_ok=True)
    for name, source in (("FREEZE.json", ORIGINAL_RUN_ROOT / "FREEZE.json"),
                         ("SYNTHETIC_QA.json", ORIGINAL_RUN_ROOT / "SYNTHETIC_QA.json"),
                         ("SOURCE_PREDICTIONS_READY.json", ORIGINAL_RUN_ROOT / READY_NAME),
                         ("SOURCE_POSTFIT_BLOCKER.json", POSTFIT_BLOCKER)):
        _ensure_copy(source, AMENDMENT_ROOT / name)
    for subdir, original_subdir in (("units", ORIGINAL_RUN_ROOT / "units"),
                                    ("receipts", ORIGINAL_RUN_ROOT / "receipts")):
        for fit_id in pilot.FIT_IDS:
            suffix = ".npz" if subdir == "units" else ".json"
            _ensure_hardlink(original_subdir / f"{fit_id}{suffix}", AMENDMENT_ROOT / subdir / f"{fit_id}{suffix}")


def _expected_ready_from_receipt(original_ready: Mapping[str, Any], receipt: Mapping[str, Any]) -> dict[str, Any]:
    ready = dict(original_ready)
    predictions = {}
    for key, old_item in original_ready["predictions"].items():
        origin = receipt["origins"][key]
        predictions[key] = {**old_item,
                            "path": origin["derived_prediction_path"],
                            "sha256": origin["derived_prediction_sha256"],
                            "postfit_identity_amendment": AMENDMENT_ID}
    ready["predictions"] = predictions
    ready["postfit_amendment"] = {
        "amendment_id": AMENDMENT_ID,
        "receipt_sha256": pilot._sha256_file(AMENDMENT_ROOT / RECEIPT_NAME),
        "original_ready_sha256": pilot._sha256_file(ORIGINAL_RUN_ROOT / READY_NAME),
        "export_source_commit": receipt["amendment_source_commit"],
        "query_labels_opened": False,
    }
    return ready


def _resume_export_metadata(original: Mapping[str, Any] | None = None) -> dict[str, Any]:
    frozen = _read_json(ORIGINAL_RUN_ROOT / "FREEZE.json")
    original_ready = _read_json(ORIGINAL_RUN_ROOT / READY_NAME)
    if original is not None:
        _create_verification_view(original)
    else:
        _create_verification_view({})
    receipt = _read_json(AMENDMENT_ROOT / RECEIPT_NAME)
    expected_ready = _expected_ready_from_receipt(original_ready, receipt)
    _write_create_once(AMENDMENT_ROOT / READY_NAME, _json_bytes(expected_ready))
    command = [sys.executable, str(ROOT / "scripts/verify_optimization_v3_learned_pair_pilot_v2_postfit.py"),
               "--export", "--run-root", str(AMENDMENT_ROOT)]
    proc = subprocess.run(command, cwd=ROOT, check=False, capture_output=True, text=True)
    if proc.returncode != 0:
        raise PostfitError(f"Independent resume export verification failed: {proc.stderr or proc.stdout}")
    verified = json.loads(proc.stdout)
    _write_create_once(AMENDMENT_ROOT / "IDENTITY_VERIFICATION.json", _json_bytes(verified))
    return {"status": verified["status"], "amendment_receipt_sha256": pilot._sha256_file(AMENDMENT_ROOT / RECEIPT_NAME),
            "identity_verification": verified, "ready_sha256": pilot._sha256_file(AMENDMENT_ROOT / READY_NAME)}


def export() -> dict[str, Any]:
    if (AMENDMENT_ROOT / RECEIPT_NAME).is_file():
        return _resume_export_metadata()
    original, fit_accounting, amendment_sources = validate_baseline()
    for name in (READY_NAME, PRELABEL_NAME, "IDENTITY_VERIFICATION.json", RESULT_NAME,
                 SCORED_VERIFICATION_NAME, "QUERY_LABEL_ACCESS.json", "QUERY_LABEL_READ.json"):
        if (AMENDMENT_ROOT / name).exists():
            raise PostfitError(f"Partial amendment state exists without a receipt; preserve for review: {name}")
    _create_verification_view(original)
    origins: dict[str, Any] = {}
    output_meta: dict[str, Any] = {}
    for origin in (67, 75):
        key = f"E{origin}"
        old_item = original["ready"]["predictions"][key]
        old_path = ROOT / old_item["path"]
        rows, cycles, ids, identity = _identity_for_origin(origin, original["frozen"], old_path)
        derived_path = AMENDMENT_ROOT / key / f"predictions_{key}_unicode.npz"
        member_audit = rebuild_npz_preserving_members(old_path, derived_path, ids)
        relative = derived_path.relative_to(ROOT).as_posix()
        origins[key] = {
            **identity,
            "original_prediction_path": old_item["path"],
            "original_prediction_sha256": old_item["sha256"],
            "derived_prediction_path": relative,
            "derived_prediction_sha256": member_audit["derived_archive_sha256"],
            "member_audit": member_audit,
            "record_id_unicode_dtype": _npy_header(archive_members(derived_path)["score_record_ids.npy"])[2].str,
            "score_rows_first_last": [int(rows[0]), int(rows[-1])],
            "score_cycles_first_last": [int(cycles[0]), int(cycles[-1])],
        }
        output_meta[key] = {**old_item, "path": relative,
                            "sha256": member_audit["derived_archive_sha256"],
                            "postfit_identity_amendment": AMENDMENT_ID}
    receipt_body = {
        "schema_version": 1, "status": "EXPORTED_ZERO_FIT_SERIALIZATION_AMENDMENT",
        "amendment_id": AMENDMENT_ID, "card_id": CARD_ID,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "authorization_sha256": pilot._sha256_file(AUTH_DOC),
        "authorization_git_blob_sha256": amendment_sources["authorization_git_blob_sha256"],
        "original_blocker_sha256": original["blocker_sha256"],
        "training_source_commit": original["frozen"]["source_commit"],
        "training_source_hashes": original["frozen"]["source_hashes"],
        "training_source_git_blob_sha256": amendment_sources["training_source_git_blob_sha256"],
        "freeze_signature": original["frozen"]["freeze_signature"],
        "input_sha256": original["frozen"]["input_sha256"],
        "current_saved_inputs_validation": original["current_saved_inputs_validation"],
        "current_saved_inputs_validation_sha256": canonical_sha256(original["current_saved_inputs_validation"]),
        "original_freeze_sha256": original["freeze_sha256"],
        "original_ready_sha256": original["ready_sha256"],
        "original_synthetic_qa_sha256": original["synthetic_qa_sha256"],
        "original_ledger_sha256": fit_accounting["sha256"],
        "original_ledger_snapshot": fit_accounting["snapshot"],
        "fit_receipts": fit_accounting["fit_receipts"],
        "amendment_source_commit": pilot._git("rev-parse", "HEAD"),
        "amendment_source_hashes": amendment_sources["amendment_source_hashes"],
        "amendment_source_git_blob_sha256": amendment_sources["amendment_source_git_blob_sha256"],
        "origins": origins,
        "preservation": {
            "original_run_root_unchanged": True,
            "original_freeze_and_readiness_preserved": True,
            "fit_receipts_and_artifacts_reused_without_refit": True,
            "old_object_id_values_deserialized": False,
            "numeric_and_safe_member_npy_bytes_preserved": True,
        },
        "query_labels_opened": False,
        "fit_count_added": 0,
        "ledger_writes": 0,
    }
    receipt = {**receipt_body, "receipt_sha256": canonical_sha256(receipt_body)}
    _write_create_once(AMENDMENT_ROOT / RECEIPT_NAME, _json_bytes(receipt))
    ready = dict(original["ready"])
    ready["predictions"] = output_meta
    ready["postfit_amendment"] = {
        "amendment_id": AMENDMENT_ID,
        "receipt_sha256": pilot._sha256_file(AMENDMENT_ROOT / RECEIPT_NAME),
        "original_ready_sha256": original["ready_sha256"],
        "export_source_commit": receipt_body["amendment_source_commit"],
        "query_labels_opened": False,
    }
    _write_create_once(AMENDMENT_ROOT / READY_NAME, _json_bytes(ready))
    identity_verification = subprocess.run(
        [sys.executable, str(ROOT / "scripts/verify_optimization_v3_learned_pair_pilot_v2_postfit.py"),
         "--export", "--run-root", str(AMENDMENT_ROOT)],
        cwd=ROOT, check=False, capture_output=True, text=True)
    if identity_verification.returncode != 0:
        raise PostfitError(f"Independent amendment export verification failed: {identity_verification.stderr or identity_verification.stdout}")
    verified = json.loads(identity_verification.stdout)
    _write_create_once(AMENDMENT_ROOT / "IDENTITY_VERIFICATION.json", _json_bytes(verified))
    return {"status": verified["status"], "amendment_receipt_sha256": receipt["receipt_sha256"],
            "identity_verification": verified, "ready_sha256": pilot._sha256_file(AMENDMENT_ROOT / READY_NAME)}


def verify_existing_export() -> dict[str, Any]:
    command = [sys.executable, str(ROOT / "scripts/verify_optimization_v3_learned_pair_pilot_v2_postfit.py"),
               "--export", "--run-root", str(AMENDMENT_ROOT)]
    proc = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
    if proc.returncode != 0:
        raise PostfitError(f"Existing amendment export no longer verifies: {proc.stderr or proc.stdout}")
    result = json.loads(proc.stdout)
    recorded = _read_json(AMENDMENT_ROOT / "IDENTITY_VERIFICATION.json")
    if canonical_sha256(recorded) != canonical_sha256(result):
        raise PostfitError("Saved identity verification differs from current independent recomputation")
    return {"status": result["status"], "amendment_receipt_sha256": pilot._sha256_file(AMENDMENT_ROOT / RECEIPT_NAME),
            "identity_verification": result, "ready_sha256": pilot._sha256_file(AMENDMENT_ROOT / READY_NAME)}


def verify_prelabel() -> dict[str, Any]:
    command = [sys.executable, str(ROOT / "scripts/verify_optimization_v3_learned_pair_pilot_v2_postfit.py"),
               "--prelabel", "--run-root", str(AMENDMENT_ROOT)]
    proc = subprocess.run(command, cwd=ROOT, check=False, capture_output=True, text=True)
    if proc.returncode != 0:
        raise PostfitError(f"Independent amendment pre-label verification failed: {proc.stderr or proc.stdout}")
    verified = json.loads(proc.stdout)
    if verified.get("status") != "PASS" or verified.get("query_labels_opened") is not False:
        raise PostfitError("Amendment pre-label verifier did not return a closed-label PASS")
    _write_create_once(AMENDMENT_ROOT / PRELABEL_NAME, _json_bytes(verified))
    return verified


def _assert_prelabel() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    export_result = verify_existing_export()
    ready = _read_json(AMENDMENT_ROOT / READY_NAME)
    prelabel_path = AMENDMENT_ROOT / PRELABEL_NAME
    prelabel = _read_json(prelabel_path)
    signature_body = dict(prelabel)
    signature = signature_body.pop("verification_signature", None)
    if (prelabel.get("status") != "PASS" or prelabel.get("query_labels_opened") is not False
            or signature != canonical_sha256(signature_body)
            or prelabel.get("freeze_signature") != ready.get("freeze_signature")
            or prelabel.get("predictions_ready_sha256") != pilot._sha256_file(AMENDMENT_ROOT / READY_NAME)
            or prelabel.get("amendment_receipt_sha256") != pilot._sha256_file(AMENDMENT_ROOT / RECEIPT_NAME)
            or prelabel.get("identity_verification_sha256") != pilot._sha256_file(AMENDMENT_ROOT / "IDENTITY_VERIFICATION.json")):
        raise PostfitError("Saved post-fit prelabel PASS is missing, stale, or misbound")
    refreshed = verify_prelabel()
    if canonical_sha256(refreshed) != canonical_sha256(prelabel):
        raise PostfitError("Full independent prelabel recheck differs from saved PASS")
    return export_result, ready, prelabel


def _score_metrics(train: pd.DataFrame, ready: Mapping[str, Any], frozen: Mapping[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    origins: dict[str, Any] = {}
    all_gates: list[dict[str, Any]] = []
    for origin in (67, 75):
        key = f"E{origin}"
        prediction_path = ROOT / ready["predictions"][key]["path"]
        with np.load(prediction_path, allow_pickle=False) as archive:
            pred = {name: np.asarray(archive[name]).copy() for name in archive.files}
        rows = np.asarray(pred["score_rows"], dtype=np.int64)
        sub = train.iloc[rows]
        ids = sub["record_id"].astype(str).to_numpy(dtype=str)
        cycles = pd.to_numeric(sub["cycle_code"], errors="raise").to_numpy(dtype=np.int64)
        if not np.array_equal(ids, pred["score_record_ids"].astype(str)) or not np.array_equal(cycles, pred["score_cycles"].astype(np.int64)):
            raise PostfitError(f"E{origin} amended output identities changed after label barrier")
        y = pd.to_numeric(sub["artifact_signal"], errors="raise").to_numpy(dtype=np.int8)
        if not np.isin(y, (0, 1)).all():
            raise PostfitError(f"E{origin} evaluation labels are not binary")
        arms = {"bag": pred["prob_bag"], "nopair": pred["prob_nopair"],
                "a02_observed": pred["prob_a02_observed"], "fm": pred["prob_fm"]}
        if any(np.asarray(p).shape != y.shape or not np.isfinite(p).all() or np.any((p < 0) | (p > 1)) for p in arms.values()):
            raise PostfitError(f"E{origin} evaluation probabilities are invalid")
        views_map = pilot._view_masks(origin, cycles)
        present = np.asarray(pred["present_mask"], dtype=bool)
        text_load = pd.to_numeric(sub["text_load"], errors="coerce").to_numpy(dtype=np.float64)
        manifest = pilot._read_json(pilot.PILOT_ROOT / f"MATRIX_MANIFEST_E{origin}.json")
        q75 = float(manifest["text_load_q75_fit_only"])
        q4 = np.isfinite(text_load) & (text_load >= q75)
        group_masks: list[tuple[str, str, np.ndarray]] = [
            ("availability", "present", present), ("availability", "missing_or_blank", ~present),
            ("text_load", "Q4", q4), ("text_load", "below_Q4", np.isfinite(text_load) & ~q4),
            ("text_load", "missing", ~np.isfinite(text_load)),
        ]
        friction = sub["entry_friction"].to_numpy(dtype=object)
        for value in sorted({pilot._label(item) for item in friction}):
            group_masks.append(("entry_friction", value,
                                np.asarray([pilot._label(item) == value for item in friction], dtype=bool)))
        view_metrics: list[dict[str, Any]] = []
        subgroup_metrics: list[dict[str, Any]] = []
        per_cycle: list[dict[str, Any]] = []
        for view, view_mask in views_map.items():
            for arm, probability in arms.items():
                view_metrics.append({"view": view, "arm": arm, **pilot._metric(y[view_mask], np.asarray(probability)[view_mask])})
            for group, value, group_mask in group_masks:
                for arm, probability in arms.items():
                    subgroup_metrics.append({"view": view, "group": group, "group_value": value,
                                             "arm": arm, **pilot._metric(y[view_mask & group_mask], np.asarray(probability)[view_mask & group_mask])})
        for cycle in sorted(np.unique(cycles).tolist()):
            cycle_mask = cycles == cycle
            each = {arm: pilot._metric(y[cycle_mask], np.asarray(probability)[cycle_mask]) for arm, probability in arms.items()}
            per_cycle.append({"cycle": int(cycle), "n": int(cycle_mask.sum()), "metrics": each,
                              "fm_minus_bag_macro_f1": each["fm"]["macro_f1"] - each["bag"]["macro_f1"],
                              "fm_minus_nopair_macro_f1": each["fm"]["macro_f1"] - each["nopair"]["macro_f1"],
                              "fm_minus_a02_observed_macro_f1": each["fm"]["macro_f1"] - each["a02_observed"]["macro_f1"]})
        gates = pilot._gates_for_origin(origin, view_metrics, subgroup_metrics)
        all_gates.extend({"origin": origin, **gate} for gate in gates)
        origins[key] = {
            "score_rows": len(rows), "present_rows": int(present.sum()), "missing_or_blank_rows": int((~present).sum()),
            "fit_defined_text_load_q75": q75, "views": view_metrics, "subgroups": subgroup_metrics,
            "per_cycle": per_cycle, "gates": gates,
            "per_cycle_positive_gain_counts": {
                "fm_vs_bag": sum(row["fm_minus_bag_macro_f1"] > 0 for row in per_cycle),
                "fm_vs_nopair": sum(row["fm_minus_nopair_macro_f1"] > 0 for row in per_cycle),
                "fm_vs_a02_observed": sum(row["fm_minus_a02_observed_macro_f1"] > 0 for row in per_cycle),
                "cycles": len(per_cycle),
            },
            "per_cycle_gain_sign_counts": {
                comparator: {
                    "positive": int(sum(row[f"fm_minus_{comparator}_macro_f1"] > 0 for row in per_cycle)),
                    "negative": int(sum(row[f"fm_minus_{comparator}_macro_f1"] < 0 for row in per_cycle)),
                    "tie": int(sum(row[f"fm_minus_{comparator}_macro_f1"] == 0 for row in per_cycle)),
                } for comparator in ("bag", "nopair", "a02_observed")
            },
            "a02_observed_reference": frozen["saved_inputs"]["origins"][key]["a02_observed_reference"],
        }
    return origins, all_gates


def score() -> dict[str, Any]:
    lock_path = AMENDMENT_ROOT / "POSTFIT_SCORE.lock"
    with offset_inputs._fit_run_lock(lock_path):
        if (AMENDMENT_ROOT / RESULT_NAME).is_file():
            return verify_score_and_complete(append_completion=True)
        _, ready, prelabel = _assert_prelabel()
        frozen = _read_json(AMENDMENT_ROOT / "FREEZE.json")
        ledger = _validate_ledger_and_fit_receipts(frozen)
        if ledger["sha256"] != EXPECTED_LEDGER_SHA256:
            raise PostfitError("Live budget ledger changed after the mandatory pre-label PASS")
        if (AMENDMENT_ROOT / "QUERY_LABEL_READ.json").exists():
            raise PostfitError("Outer targets were already read without a durable score; stop for review")
        ready_path = AMENDMENT_ROOT / READY_NAME
        scored_identity: dict[str, Any] = {}
        # Load and validate all prediction members before opening any evaluation labels.
        for key, item in ready["predictions"].items():
            archive_path = ROOT / item["path"]
            with np.load(archive_path, allow_pickle=False) as archive:
                rows = np.asarray(archive["score_rows"], dtype=np.int64)
                ids = np.asarray(archive["score_record_ids"]).astype(str)
                cycles = np.asarray(archive["score_cycles"], dtype=np.int64)
                if any(not np.isfinite(np.asarray(archive[name], dtype=np.float64)).all()
                       for name in ("prob_bag", "prob_nopair", "prob_a02_observed", "prob_fm")):
                    raise PostfitError(f"Non-finite probability found before score access for {key}")
            scored_identity[key] = {
                "score_rows_sha256": offset_inputs.sha256_array(rows),
                "score_ids_sha256": offset_inputs.sha256_array(ids),
                "score_cycles_sha256": offset_inputs.sha256_array(cycles),
            }
        access_payload = {
            "authorized_at_utc": datetime.now(timezone.utc).isoformat(),
            "purpose": "evaluation only after full independent postfit prelabel PASS",
            "prediction_ready_sha256": pilot._sha256_file(ready_path),
            "prelabel_verification_sha256": pilot._sha256_file(AMENDMENT_ROOT / PRELABEL_NAME),
            "train_sha256": frozen["input_sha256"],
            "columns_opened": ["record_id", "cycle_code", "artifact_signal", "text_load", "entry_friction"],
            "scored_populations": scored_identity,
            "amendment_receipt_sha256": pilot._sha256_file(AMENDMENT_ROOT / RECEIPT_NAME),
        }
        access_path = AMENDMENT_ROOT / "QUERY_LABEL_ACCESS.json"
        _write_create_once(access_path, _json_bytes(access_payload))
        train = pd.read_csv(pilot.TRAIN,
                            usecols=["record_id", "cycle_code", "artifact_signal", "text_load", "entry_friction"],
                            dtype={"record_id": "string"})
        read_payload = {
            "read_at_utc": datetime.now(timezone.utc).isoformat(),
            "query_label_access_sha256": pilot._sha256_file(access_path),
            "train_sha256": frozen["input_sha256"], "rows_read": int(len(train)),
            "columns_read": ["record_id", "cycle_code", "artifact_signal", "text_load", "entry_friction"],
        }
        _write_create_once(AMENDMENT_ROOT / "QUERY_LABEL_READ.json", _json_bytes(read_payload))
        origins, all_gates = _score_metrics(train, ready, frozen)
        fit_receipts = {}
        optimizer_pass = True
        for fit_id in pilot.FIT_IDS:
            artifact, receipt_path = pilot._fit_paths(fit_id)
            receipt = pilot._validate_receipt(fit_id, frozen, artifact_path=artifact, receipt_path=receipt_path)
            fit_receipts[fit_id] = {"receipt_sha256": pilot._sha256_file(receipt_path),
                                    "artifact_sha256": pilot._sha256_file(artifact),
                                    "optimizer": receipt["optimizer"]}
            optimizer_pass &= bool(receipt["optimizer_converged"])
        unsupported = [gate for gate in all_gates if gate.get("status") == "UNKNOWN_UNSUPPORTED"]
        mechanism_gates = [gate for gate in all_gates if gate.get("track") == "mechanism"]
        competition_gates = [gate for gate in all_gates if gate.get("track") == "competition"]
        mechanism_pass = optimizer_pass and all(gate["pass"] is True for gate in mechanism_gates)
        competition_pass = mechanism_pass and all(gate["pass"] is True for gate in competition_gates)
        result = {
            "schema_version": 1, "card_id": CARD_ID, "status": "COMPLETED_DIAGNOSTIC_ONLY",
            "verdicts": {"mechanism": "MECHANISM_PASS" if mechanism_pass else "MECHANISM_FAIL",
                         "competition": "COMPETITION_PASS" if competition_pass else "COMPETITION_FAIL"},
            "adaptive_selection": True, "mechanism_pilot": True,
            "qualified_candidate": False, "champion_change": False, "upload_authorized": False,
            "source_commit": frozen["source_commit"], "freeze_signature": frozen["freeze_signature"],
            "input_sha256": frozen["input_sha256"],
            "query_labels_opened_after_prelabel_verification": True,
            "query_label_access_receipt_sha256": pilot._sha256_file(access_path),
            "query_label_read_receipt_sha256": pilot._sha256_file(AMENDMENT_ROOT / "QUERY_LABEL_READ.json"),
            "prelabel_verification_sha256": pilot._sha256_file(AMENDMENT_ROOT / PRELABEL_NAME),
            "predictions_ready_sha256": pilot._sha256_file(ready_path),
            "ledger_snapshot": ledger["snapshot"], "ledger_sha256": ledger["sha256"],
            "fit_receipts": fit_receipts, "optimizer_all_success": optimizer_pass,
            "gate_checks": all_gates, "mechanism_gate_checks": mechanism_gates,
            "competition_gate_checks": competition_gates,
            "mechanism_gates_pass": mechanism_pass, "competition_gates_pass": competition_pass,
            "evaluable_gates_pass": mechanism_pass, "unsupported_group_checks": unsupported,
            "unsupported_group_check_count": len(unsupported), "origins": origins,
            "postfit_amendment_id": AMENDMENT_ID,
            "postfit_amendment_receipt_sha256": pilot._sha256_file(AMENDMENT_ROOT / RECEIPT_NAME),
            "postfit_prelabel_verification_sha256": pilot._sha256_file(AMENDMENT_ROOT / PRELABEL_NAME),
            "export_source_commit": _read_json(AMENDMENT_ROOT / RECEIPT_NAME)["amendment_source_commit"],
            "interpretation": "Diagnostic-only, adaptively selected two-origin learned-factor pilot. E75 W6 overlaps E67 W6 and is history-sensitivity evidence, not independent replication. No verdict changes a champion or authorizes continuation, full-Train refit, Test inference, or upload.",
        }
        _write_create_once(AMENDMENT_ROOT / RESULT_NAME, _json_bytes(result))
        report = pilot._make_report(result)
        report += (
            "\n## Integritas amendment pasca-fit\n\n"
            "Empat fit asli digunakan kembali tanpa refit. Hanya `score_record_ids` yang diekspor ulang sebagai Unicode tetap; payload object lama tidak dideserialisasi. Byte NPY semua member lain dipertahankan identik, dan identitas baris dicocokkan terhadap official Train, prediksi historis positional, serta referensi A02 per origin. Amendment ini tidak mengubah probabilitas, threshold, route, comparator, metric, atau gate.\n"
        )
        _write_create_once(REPORT_ROOT / "POSTFIT_RESULTS.json", _json_bytes(result))
        _write_create_once(REPORT_ROOT / "POSTFIT_RESULTS.md", report.encode("utf-8"))
        return verify_score_and_complete(append_completion=True)


def verify_score_and_complete(*, append_completion: bool) -> dict[str, Any]:
    proc = subprocess.run(
        [sys.executable, str(ROOT / "scripts/verify_optimization_v3_learned_pair_pilot_v2_postfit.py"),
         "--scored", "--run-root", str(AMENDMENT_ROOT)],
        cwd=ROOT, check=False, capture_output=True, text=True)
    if proc.returncode != 0:
        raise PostfitError(f"Independent postfit score verification failed: {proc.stderr or proc.stdout}")
    verified = json.loads(proc.stdout)
    if verified.get("status") != "PASS_INDEPENDENT_METRICS_AND_GATES":
        raise PostfitError("Independent postfit score verifier did not pass")
    _write_create_once(AMENDMENT_ROOT / SCORED_VERIFICATION_NAME, _json_bytes(verified))
    _write_create_once(REPORT_ROOT / "POSTFIT_SCORED_VERIFICATION.json", _json_bytes(verified))
    result_path = AMENDMENT_ROOT / RESULT_NAME
    result = _read_json(result_path)
    completion_path = AMENDMENT_ROOT / "COMPLETION_RECEIPT.json"
    prior_events = [event for event in pilot._ledger_events()
                    if event.get("event") == "CARD_COMPLETED" and event.get("card_id") == CARD_ID]
    if prior_events:
        prior = prior_events[0]
        if (len(prior_events) != 1 or prior.get("amendment_id") != AMENDMENT_ID
                or prior.get("result_sha256") != pilot._sha256_file(result_path)
                or prior.get("scored_verification_sha256") != pilot._sha256_file(AMENDMENT_ROOT / SCORED_VERIFICATION_NAME)
                or prior.get("amendment_receipt_sha256") != pilot._sha256_file(AMENDMENT_ROOT / RECEIPT_NAME)):
            raise PostfitError("Existing completion event conflicts with amended result; preserve and stop")
        completion = {
            "status": "COMPLETED_DIAGNOSTIC_ONLY", "completion_event": prior,
            "completion_event_sha256": sha256_bytes(canonical_bytes(prior) + b"\n"),
            "ledger_sha256_after_completion": pilot._sha256_file(pilot.LEDGER),
            "result_sha256": pilot._sha256_file(result_path),
            "scored_verification_sha256": pilot._sha256_file(AMENDMENT_ROOT / SCORED_VERIFICATION_NAME),
            "amendment_receipt_sha256": pilot._sha256_file(AMENDMENT_ROOT / RECEIPT_NAME),
            "query_labels_opened_only_after_prelabel_pass": True,
        }
        _write_create_once(completion_path, _json_bytes(completion))
        _write_create_once(REPORT_ROOT / "POSTFIT_COMPLETION_RECEIPT.json", _json_bytes(completion))
        return result
    if not append_completion:
        return verified
    if pilot._sha256_file(pilot.LEDGER) != EXPECTED_LEDGER_SHA256:
        raise PostfitError("Live ledger changed before the single normal completion event")
    event = {
        "event": "CARD_COMPLETED", "batch_id": "optimization_v3_research_next_v1",
        "card_id": CARD_ID, "stage": pilot.STAGE, "amendment_id": AMENDMENT_ID,
        "freeze_signature": result["freeze_signature"], "status": result["status"],
        "fit_ids": list(pilot.FIT_IDS), "result_sha256": pilot._sha256_file(result_path),
        "scored_verification_sha256": pilot._sha256_file(AMENDMENT_ROOT / SCORED_VERIFICATION_NAME),
        "amendment_receipt_sha256": pilot._sha256_file(AMENDMENT_ROOT / RECEIPT_NAME),
        "prelabel_verification_sha256": pilot._sha256_file(AMENDMENT_ROOT / PRELABEL_NAME),
    }
    completion_event = pilot._append_ledger(event, path=pilot.LEDGER, official=True)
    postcheck = subprocess.run(
        [sys.executable, str(ROOT / "scripts/verify_optimization_v3_learned_pair_pilot_v2_postfit.py"),
         "--scored", "--run-root", str(AMENDMENT_ROOT)],
        cwd=ROOT, check=False, capture_output=True, text=True)
    if postcheck.returncode != 0 or canonical_sha256(json.loads(postcheck.stdout)) != canonical_sha256(verified):
        raise PostfitError(f"Completion event post-check failed: {postcheck.stderr or postcheck.stdout}")
    receipt = {
        "status": "COMPLETED_DIAGNOSTIC_ONLY", "completion_event": completion_event,
        "completion_event_sha256": sha256_bytes(canonical_bytes(completion_event) + b"\n"),
        "ledger_sha256_after_completion": pilot._sha256_file(pilot.LEDGER),
        "result_sha256": pilot._sha256_file(result_path),
        "scored_verification_sha256": pilot._sha256_file(AMENDMENT_ROOT / SCORED_VERIFICATION_NAME),
        "amendment_receipt_sha256": pilot._sha256_file(AMENDMENT_ROOT / RECEIPT_NAME),
        "query_labels_opened_only_after_prelabel_pass": True,
    }
    _write_create_once(completion_path, _json_bytes(receipt))
    _write_create_once(REPORT_ROOT / "POSTFIT_COMPLETION_RECEIPT.json", _json_bytes(receipt))
    return result


def run_synthetic_qa(temp_root: Path) -> dict[str, Any]:
    temp_root = Path(temp_root).resolve()
    if temp_root == ROOT or ROOT in temp_root.parents:
        raise PostfitError("Synthetic amendment QA root must be outside the repository")
    temp_root.mkdir(parents=True, exist_ok=True)
    before_ledger = pilot._sha256_file(pilot.LEDGER)
    ids = np.asarray(["rid-1", "rid-2-long-unicode-東京"], dtype="<U64")
    score_rows = np.asarray([1, 3], dtype=np.int64)
    cycles = np.asarray([48, 49], dtype=np.int64)
    numeric = np.asarray([0.125, 0.875], dtype=np.float64)
    old = temp_root / "old.npz"
    derived = temp_root / "derived.npz"
    np.savez_compressed(old, score_rows=score_rows, score_record_ids=np.asarray(["unsafe-old-1", "unsafe-old-2"], dtype=object),
                        score_cycles=cycles, probabilities=numeric, raw_logits=np.asarray([-1.0, 1.0]))
    old_header = _npy_header(archive_members(old)["score_record_ids.npy"])
    if not old_header[2].hasobject or old_header[0] != (2,):
        raise PostfitError("Synthetic original object-ID fixture was not built")
    # Do not deserialize the synthetic object member; assert its normal safe load is rejected.
    object_rejected = False
    try:
        read_safe_member(old, "score_record_ids")
    except ValueError as exc:
        object_rejected = "Object arrays cannot be loaded" in str(exc)
    if not object_rejected:
        raise PostfitError("Synthetic object-ID archive did not fail closed under allow_pickle=False")
    audit = rebuild_npz_preserving_members(old, derived, ids)
    resumed_audit = rebuild_npz_preserving_members(old, derived, ids)
    if resumed_audit["derived_archive_sha256"] != audit["derived_archive_sha256"]:
        raise PostfitError("Synthetic idempotent resume changed the derived archive")
    new_ids = read_safe_member(derived, "score_record_ids")
    if new_ids.dtype.kind != "U" or new_ids.tolist() != ids.tolist():
        raise PostfitError("Unicode round-trip failed, including long non-ASCII ID")
    for key in ("score_rows", "score_cycles", "probabilities", "raw_logits"):
        if not np.array_equal(read_safe_member(old, key), read_safe_member(derived, key)):
            raise PostfitError(f"Synthetic safe member changed: {key}")
    identity_ok = np.array_equal(ids, np.asarray(["rid-1", "rid-2-long-unicode-東京"], dtype=str))
    if not identity_ok or not np.array_equal(cycles, np.asarray([48, 49], dtype=np.int64)):
        raise PostfitError("Synthetic same-origin identity alignment failed")
    receipt_body = {"original_sha256": audit["original_archive_sha256"],
                    "derived_sha256": audit["derived_archive_sha256"],
                    "source_training_commit": EXPECTED_TRAINING_COMMIT,
                    "amendment_source_commit": "synthetic-only", "member_audit": audit}
    receipt = {**receipt_body, "receipt_sha256": canonical_sha256(receipt_body)}
    if canonical_sha256({k: v for k, v in receipt.items() if k != "receipt_sha256"}) != receipt["receipt_sha256"]:
        raise PostfitError("Synthetic old/new receipt binding failed")
    tampered_numeric = temp_root / "tampered_numeric.npz"
    arrays = {key: read_safe_member(derived, key) for key in ("score_rows", "score_record_ids", "score_cycles", "probabilities", "raw_logits")}
    arrays["probabilities"][0] = 0.5
    np.savez_compressed(tampered_numeric, **arrays)
    if pilot._sha256_file(tampered_numeric) == audit["derived_archive_sha256"]:
        raise PostfitError("Synthetic tampered numeric archive escaped hash detection")
    tampered_id = temp_root / "tampered_id.npz"
    arrays["probabilities"] = read_safe_member(derived, "probabilities")
    arrays["score_record_ids"][0] = "altered"
    np.savez_compressed(tampered_id, **arrays)
    if pilot._sha256_file(tampered_id) == audit["derived_archive_sha256"]:
        raise PostfitError("Synthetic tampered ID archive escaped hash detection")
    tampered_receipt = dict(receipt)
    tampered_receipt["derived_sha256"] = "0" * 64
    if canonical_sha256({k: v for k, v in tampered_receipt.items() if k != "receipt_sha256"}) == receipt["receipt_sha256"]:
        raise PostfitError("Synthetic receipt binding did not detect a derived hash change")
    # All frozen gate paths are reused unchanged; test exact row parity on synthetic gate inputs.
    gate_parity: dict[str, int] = {}
    for origin, views in ((67, ("W4", "W5", "W6", "H12_E67")), (75, ("W6",))):
        view_rows = []
        for view in views:
            for arm, score in (("bag", .70), ("nopair", .69), ("a02_observed", .68), ("fm", .71)):
                view_rows.append({"view": view, "arm": arm, "macro_f1": score,
                                  "positive_recall": score, "n": 300, "positive": 150, "negative": 150})
        group_rows = []
        for view in views:
            for arm, score in (("bag", .70), ("nopair", .69), ("a02_observed", .68), ("fm", .71)):
                group_rows.append({"view": view, "group": "availability", "group_value": "present",
                                   "arm": arm, "macro_f1": score, "positive_recall": score,
                                   "n": 300, "positive": 150, "negative": 150})
        regular = [{"origin": origin, **item} for item in pilot._gates_for_origin(origin, view_rows, group_rows)]
        independent = pilot_verifier._gates_independent(origin, view_rows, group_rows)
        if canonical_sha256(regular) != canonical_sha256(independent):
            raise PostfitError(f"Synthetic original vs independent frozen gate parity failed at E{origin}")
        gate_parity[f"E{origin}"] = len(regular)
    writer_blocked = False
    try:
        pilot._append_ledger({"event": "SYNTHETIC_ONLY"}, path=pilot.LEDGER, official=False)
    except pilot.LearnedPairPilotError:
        writer_blocked = True
    if not writer_blocked or pilot._sha256_file(pilot.LEDGER) != before_ledger or before_ledger != EXPECTED_LEDGER_SHA256:
        raise PostfitError("Synthetic lifecycle touched or could write to the production ledger")
    result = {
        "status": "PASS_SYNTHETIC_POSTFIT_AMENDMENT_QA", "synthetic_archive_only": True,
        "object_id_rejected_without_deserialization": object_rejected,
        "unicode_roundtrip_including_long_non_ascii_id": True,
        "non_id_member_bytes_and_values_preserved": True,
        "same_origin_identity_alignment": identity_ok,
        "tampered_numeric_rejected": True, "tampered_id_rejected": True,
        "old_new_receipt_binding": True, "resume_requires_no_fit": True,
        "idempotent_resume_archive_sha256": resumed_audit["derived_archive_sha256"],
        "frozen_gate_parity_counts": gate_parity,
        "production_ledger_writer_blocked": writer_blocked,
        "production_ledger_sha256_before": before_ledger,
        "production_ledger_sha256_after": pilot._sha256_file(pilot.LEDGER),
        "temporary_root": str(temp_root), "fit_calls": 0,
    }
    _write_create_once(temp_root / "SYNTHETIC_QA.json", _json_bytes(result))
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("synthetic", "export", "verify-prelabel", "score"))
    parser.add_argument("--temp-root", type=Path)
    args = parser.parse_args()
    try:
        if args.command == "synthetic":
            if args.temp_root is None:
                parser.error("synthetic requires --temp-root outside the repository")
                return 2
            result = run_synthetic_qa(args.temp_root)
        elif args.command == "export":
            result = export()
        elif args.command == "verify-prelabel":
            result = verify_prelabel()
        else:
            result = score()
        print(json.dumps(result, sort_keys=True, ensure_ascii=False, allow_nan=False))
        return 0
    except Exception as exc:
        print(json.dumps({"status": "ERROR", "error_type": type(exc).__name__, "error": str(exc)},
                         sort_keys=True, ensure_ascii=False, allow_nan=False), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
