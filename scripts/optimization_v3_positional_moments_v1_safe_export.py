"""Read-only audit and safe Unicode-ID export for frozen POS_MOMENTS bundles.

Only the two frozen POS_MOMENTS prediction archives are whitelisted. Every
non-object NPY member is copied byte-for-byte. The serialized object-ID member
is never deserialized or member-hashed; IDs are reconstructed from official
Train record_id at frozen numeric score_rows, after checking the original
producer's logical object-array hash convention.
"""
from __future__ import annotations

import ast
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import struct
import sys
import tempfile
from typing import Any, Iterator, Mapping
import zipfile

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import optimization_v3_positional_moments_v1 as moments
from scripts import optimization_v3_positional_moments_v1_prelabel_adapter as parent_adapter
from scripts import verify_optimization_v3_positional_moments_v1 as verifier


ADAPTER_ID = "POS_MOMENTS_V1_SAFE_EXPORT_AND_READ_ADAPTER_V1"
EXPORT_DIR = moments.RUN_ROOT / "safe_exports_v1"
MANIFEST_REPORT = moments.REPORT_ROOT / "SAFE_EXPORT_MANIFEST_V1.json"
MANIFEST_RUN = moments.RUN_ROOT / "SAFE_EXPORT_MANIFEST_V1.json"
PRELABEL_FAILURE_REPORT = moments.REPORT_ROOT / "PRELABEL_SCHEMA_ADAPTER_FAILURE_V1.json"
PRELABEL_RUN_READY = moments.RUN_ROOT / "PREDICTIONS_READY.json"
PRELABEL_REPORT_READY = moments.REPORT_ROOT / "PREDICTIONS_READY.json"
PILOT_READY = moments.PILOT_RUN / "PREDICTIONS_READY.json"
SAFE_BUNDLE_KEYS = (
    "score_rows", "score_record_ids", "score_cycles", "present_mask", "entry_friction", "text_load_group",
    "prob_pP", "prob_a02", "prob_bag", "prob_q_real", "prob_candidate_real", "fit_receipt_sha256_real",
    "prob_q_shuffled", "prob_candidate_shuffled", "fit_receipt_sha256_shuffled", "prob_mix50",
)


class SafeExportError(RuntimeError):
    """The frozen bundle, ID lineage, or safe export failed closed."""


def _sha256_file(path: Path) -> str:
    return moments.sha256_file(path)


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _canonical_hash(payload: Mapping[str, Any]) -> str:
    return moments.canonical_sha256(payload)


def _read_npy_header(stream: Any) -> dict[str, Any]:
    """Read only an NPY header; never deserialize the array payload."""
    magic = stream.read(8)
    if len(magic) != 8 or magic[:6] != b"\x93NUMPY":
        raise SafeExportError("NPZ member is not a valid NPY stream")
    major, minor = magic[6], magic[7]
    if (major, minor) == (1, 0):
        header_size = struct.unpack("<H", stream.read(2))[0]
        encoding = "latin1"
    elif (major, minor) in ((2, 0), (3, 0)):
        header_size = struct.unpack("<I", stream.read(4))[0]
        encoding = "utf8" if major == 3 else "latin1"
    else:
        raise SafeExportError(f"Unsupported NPY format version {(major, minor)}")
    header_bytes = stream.read(header_size)
    if len(header_bytes) != header_size:
        raise SafeExportError("Truncated NPY header")
    header = ast.literal_eval(header_bytes.decode(encoding))
    if not isinstance(header, dict) or not {"descr", "shape", "fortran_order"} <= set(header):
        raise SafeExportError("NPY header fields are invalid")
    dtype = np.dtype(header["descr"])
    shape = tuple(int(dim) for dim in header["shape"])
    if any(dim < 0 for dim in shape):
        raise SafeExportError("NPY shape contains a negative dimension")
    return {"version": [major, minor], "dtype": dtype.str, "kind": dtype.kind,
            "shape": list(shape), "fortran_order": bool(header["fortran_order"])}


def _safe_relpath(path: str) -> str:
    normalized = path.replace("\\", "/")
    parsed = PurePosixPath(normalized)
    if parsed.is_absolute() or ".." in parsed.parts or parsed.as_posix() != normalized:
        raise SafeExportError(f"Non-canonical repository-relative path: {path!r}")
    return parsed.as_posix()


def _load_safe_member(archive: zipfile.ZipFile, member: str) -> tuple[np.ndarray, bytes, dict[str, Any]]:
    raw = archive.read(member)
    info = _read_npy_header(io.BytesIO(raw))
    if info["kind"] == "O":
        raise SafeExportError(f"Object member is not eligible for safe loading: {member}")
    try:
        value = np.load(io.BytesIO(raw), allow_pickle=False)
    except Exception as exc:
        raise SafeExportError(f"Safe NPY member rejected: {member}: {type(exc).__name__}: {exc}") from exc
    if value.dtype.kind == "O" or value.shape != tuple(info["shape"]):
        raise SafeExportError(f"Loaded safe member header mismatch: {member}")
    return value, raw, info


def _archive_member_names(archive: zipfile.ZipFile) -> list[str]:
    names = [item.filename for item in archive.infolist()]
    if len(names) != len(set(names)):
        raise SafeExportError("NPZ contains duplicate member names")
    expected = {f"{key}.npy" for key in SAFE_BUNDLE_KEYS}
    if set(names) != expected:
        raise SafeExportError(f"NPZ member set differs from whitelist: {names}")
    return names


def _assert_copied_member_identity(name: str, source_raw: bytes, derived_raw: bytes,
                                   record: Mapping[str, Any]) -> None:
    if (record.get("action") != "copied_npy_member_bytes_verbatim"
            or _sha256_bytes(source_raw) != record.get("original_member_sha256")
            or _sha256_bytes(derived_raw) != record.get("derived_member_sha256")
            or source_raw != derived_raw):
        raise SafeExportError(f"Safe member byte identity failed: {name}")


def _read_object_header(archive: zipfile.ZipFile, member: str) -> dict[str, Any]:
    with archive.open(member, "r") as stream:
        info = _read_npy_header(stream)
    if info["kind"] != "O":
        raise SafeExportError(f"Expected only score_record_ids to be object dtype; {member} is {info['dtype']}")
    return info


def _write_safe_npz(original: Path, derived: Path, reconstructed_ids: np.ndarray) -> dict[str, Any]:
    """Copy safe NPY members and replace only score_record_ids with Unicode."""
    original, derived = Path(original), Path(derived)
    if reconstructed_ids.dtype.kind != "U" or reconstructed_ids.ndim != 1:
        raise SafeExportError("Reconstructed score_record_ids must be a one-dimensional Unicode array")
    raw_members: dict[str, bytes] = {}
    member_records: dict[str, Any] = {}
    with zipfile.ZipFile(original, "r") as source:
        names = _archive_member_names(source)
        for name in names:
            if name == "score_record_ids.npy":
                info = _read_object_header(source, name)
                if info["shape"] != [len(reconstructed_ids)]:
                    raise SafeExportError("Serialized object ID array shape differs from frozen score_rows")
                member_records[name] = {
                    "action": "reconstructed_without_deserializing_or_member_hashing_original",
                    "original_header": info,
                    "original_member_sha256": None,
                }
            else:
                array, raw, info = _load_safe_member(source, name)
                if info["kind"] == "O":
                    raise SafeExportError(f"Unexpected object array in frozen NPZ: {name}")
                raw_members[name] = raw
                member_records[name] = {
                    "action": "copied_npy_member_bytes_verbatim",
                    "original_header": info,
                    "original_member_sha256": _sha256_bytes(raw),
                    "array_sha256": moments.sha256_array(array),
                }
    id_stream = io.BytesIO()
    np.save(id_stream, reconstructed_ids, allow_pickle=False)
    id_raw = id_stream.getvalue()
    id_info = _read_npy_header(io.BytesIO(id_raw))
    if id_info["kind"] != "U" or id_info["shape"] != [len(reconstructed_ids)]:
        raise SafeExportError("Safe Unicode ID replacement did not serialize as expected")
    raw_members["score_record_ids.npy"] = id_raw
    member_records["score_record_ids.npy"]["derived_header"] = id_info
    member_records["score_record_ids.npy"]["derived_member_sha256"] = _sha256_bytes(id_raw)

    derived.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(prefix="posmom_safe_", suffix=".npz", dir=derived.parent, delete=False) as tmp:
        temp_path = Path(tmp.name)
    try:
        with zipfile.ZipFile(temp_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as target:
            for name in names:
                info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
                info.compress_type = zipfile.ZIP_DEFLATED
                info.create_system = 0
                info.external_attr = 0
                target.writestr(info, raw_members[name])
        if derived.exists():
            if _sha256_file(derived) != _sha256_file(temp_path):
                raise SafeExportError(f"Refusing to replace a different existing derived export: {derived}")
            temp_path.unlink()
        else:
            os.replace(temp_path, derived)
    finally:
        if temp_path.exists():
            temp_path.unlink()

    # Verify zip readback with allow_pickle=False and exact raw-member identities.
    with zipfile.ZipFile(original, "r") as source, zipfile.ZipFile(derived, "r") as target:
        if _archive_member_names(source) != _archive_member_names(target):
            raise SafeExportError("Derived archive member order/identity differs from original")
        for name in _archive_member_names(source):
            if name == "score_record_ids.npy":
                continue
            source_raw, target_raw = source.read(name), target.read(name)
            if source_raw != target_raw:
                raise SafeExportError(f"Derived safe member bytes changed: {name}")
        with np.load(derived, allow_pickle=False) as safe_archive:
            if set(safe_archive.files) != {key[:-4] for key in _archive_member_names(target)}:
                raise SafeExportError("Derived archive does not load under allow_pickle=False")
            safe_ids = np.asarray(safe_archive["score_record_ids"])
            if safe_ids.dtype.kind != "U" or not np.array_equal(safe_ids, reconstructed_ids):
                raise SafeExportError("Derived Unicode IDs fail safe round-trip")
    for name in member_records:
        if name != "score_record_ids.npy":
            member_records[name]["derived_member_sha256"] = _sha256_bytes(raw_members[name])
            member_records[name]["byte_identical"] = True
    member_records["score_record_ids.npy"]["byte_identical"] = False
    return {"members": member_records, "derived_archive_sha256": _sha256_file(derived)}


def _validate_rows_and_ids(
    *, origin: int, score_rows: np.ndarray, score_cycles: np.ndarray,
    all_ids: np.ndarray, all_cycles: np.ndarray, declared_old_ids_sha256: str,
    expected_rows: np.ndarray, parent_rows: np.ndarray, parent_cycles: np.ndarray,
    parent_ids: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any]]:
    rows = np.asarray(score_rows, dtype=np.int64).reshape(-1)
    cycles = np.asarray(score_cycles, dtype=np.int64).reshape(-1)
    ids = np.asarray(all_ids, dtype=str).reshape(-1)
    all_cycle_values = np.asarray(all_cycles, dtype=np.int64).reshape(-1)
    if not 0 <= origin <= np.iinfo(np.int64).max:
        raise SafeExportError("Origin is not a valid cycle integer")
    if (len(rows) == 0 or len(rows) != len(cycles) or (rows < 0).any()
            or (rows >= len(ids)).any() or not np.all(rows[1:] > rows[:-1])
            or len(np.unique(rows)) != len(rows)):
        raise SafeExportError("score_rows are empty, out of range, duplicated, or not strictly ordered")
    if len(ids) != len(all_cycle_values) or np.any(ids == "") or len(np.unique(ids)) != len(ids):
        raise SafeExportError("Official Train IDs are missing or duplicated")
    if len(set(zip(ids.tolist(), all_cycle_values.tolist()))) != len(ids):
        raise SafeExportError("Official Train record_id/cycle pairs are not unique")
    expected = np.asarray(expected_rows, dtype=np.int64).reshape(-1)
    if not np.array_equal(rows, expected):
        raise SafeExportError(f"E{origin} score_rows differ from all Train rows after the frozen origin")
    if not np.array_equal(cycles, all_cycle_values[rows]) or np.any(cycles <= origin):
        raise SafeExportError(f"E{origin} score_cycles do not match official Train rows/origin")
    p_rows = np.asarray(parent_rows, dtype=np.int64).reshape(-1)
    p_cycles = np.asarray(parent_cycles, dtype=np.int64).reshape(-1)
    p_ids = np.asarray(parent_ids).astype(str).reshape(-1)
    reconstructed = np.asarray(ids[rows], dtype=np.str_)
    if (not np.array_equal(rows, p_rows) or not np.array_equal(cycles, p_cycles)
            or not np.array_equal(reconstructed, p_ids)):
        raise SafeExportError(f"E{origin} reconstructed IDs differ from verified parent rows/cycles/IDs")
    old_convention = moments.sha256_array(np.asarray(reconstructed, dtype=object))
    if old_convention != declared_old_ids_sha256:
        raise SafeExportError(f"E{origin} reconstructed object-dtype logical hash differs from producer receipt")
    new_convention = moments.sha256_array(reconstructed)
    return reconstructed, {
        "origin": origin, "rows": len(rows), "score_rows_sha256": moments.sha256_array(rows),
        "score_cycles_sha256": moments.sha256_array(cycles),
        "record_id_source": "official Train.csv record_id selected by frozen score_rows",
        "old_dtype": "object", "old_dtype_logical_sha256": old_convention,
        "producer_declared_old_dtype_sha256": declared_old_ids_sha256,
        "old_dtype_hash_matches": True, "new_dtype": str(reconstructed.dtype),
        "new_unicode_logical_sha256": new_convention,
        "id_count": len(reconstructed), "unique_ids": True,
        "query_labels_read": False,
    }


def _read_parent_identity(origin: int, expected: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    path = moments.ROOT / expected["path"]
    if _sha256_file(path) != expected["sha256"]:
        raise SafeExportError(f"E{origin} parent prediction archive hash differs from frozen lineage")
    with zipfile.ZipFile(path, "r") as archive:
        names = [item.filename for item in archive.infolist()]
        if len(names) != len(set(names)):
            raise SafeExportError(f"E{origin} parent NPZ contains duplicate members")
        required = {"score_rows.npy", "score_cycles.npy", "score_record_ids.npy"}
        if not required <= set(names):
            raise SafeExportError(f"E{origin} parent NPZ lacks frozen identity arrays")
        rows, _, _ = _load_safe_member(archive, "score_rows.npy")
        cycles, _, _ = _load_safe_member(archive, "score_cycles.npy")
        ids, _, ids_info = _load_safe_member(archive, "score_record_ids.npy")
        if ids_info["kind"] == "O" or ids.dtype.kind not in "US":
            raise SafeExportError(f"E{origin} frozen parent IDs are not safely readable Unicode/bytes")
    return rows, cycles, ids


def _train_identity(freeze_data: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    train_path = moments.TRAIN
    if _sha256_file(train_path) != freeze_data["input_sha256"]:
        raise SafeExportError("Official Train file hash differs from the frozen source")
    frame = pd.read_csv(train_path, usecols=["record_id", "cycle_code"], dtype={"record_id": "string"})
    if len(frame) != int(freeze_data["input_rows"]) or frame["record_id"].isna().any():
        raise SafeExportError("Official Train identity columns have wrong row count or missing IDs")
    cycle = pd.to_numeric(frame["cycle_code"], errors="raise").to_numpy(dtype=np.int64)
    ids = np.asarray(frame["record_id"].astype(str).tolist(), dtype=np.str_)
    return ids, cycle


def export_all() -> dict[str, Any]:
    freeze_data = moments._load_freeze()
    ledger_before = _sha256_file(moments.LEDGER)
    ledger_events, ledger_snapshot = moments._check_live_ledger(freeze_data)
    ready = moments._read_json(PRELABEL_RUN_READY)
    if ready != moments._read_json(PRELABEL_REPORT_READY):
        raise SafeExportError("Tracked/run PREDICTIONS_READY receipts differ")
    if (ready.get("status") != "PASS_PREDICTIONS_DURABLE_PRELABEL_CLOSED"
            or ready.get("query_labels_read") is not False or ready.get("test_or_sample_access") is not False
            or ready.get("freeze_signature") != freeze_data["freeze_signature"]
            or ready.get("fit_ids") != list(moments.FIT_IDS)):
        raise SafeExportError("Frozen current PREDICTIONS_READY identity/boundary is invalid")

    parent_ready = moments._read_json(PILOT_READY)
    if parent_ready.get("status") != "PREDICTIONS_READY_FOR_SCORE_LABEL_OPENING":
        raise SafeExportError("Historical parent receipt has an unexpected status")
    parent_expected = parent_adapter._frozen_parent_expected(freeze_data)
    parent_index_basic = parent_adapter.normalize_prediction_index(
        parent_ready, root=moments.ROOT, expected=parent_expected, require_parent_manifests=True
    )
    parent_index = {
        key: {
            **parent_index_basic[key],
            "matrix_manifest_path": parent_expected[key]["matrix_manifest_path"],
            "matrix_manifest_sha256": parent_expected[key]["matrix_manifest_sha256"],
        }
        for key in ("E67", "E75")
    }
    train_ids, train_cycles = _train_identity(freeze_data)
    origins: dict[str, Any] = {}
    for origin in (67, 75):
        key = f"E{origin}"
        original_info = ready["predictions"].get(key)
        if not isinstance(original_info, Mapping):
            raise SafeExportError(f"Current PREDICTIONS_READY lacks {key}")
        original_rel = _safe_relpath(original_info.get("path", ""))
        expected_rel = f"experiments/runs/optimization_v3_research_next_v1/POS_MOMENTS_V1/{key}/predictions_{key}.npz"
        if original_rel != expected_rel:
            raise SafeExportError(f"{key} original prediction path is not whitelisted")
        original_path = moments.ROOT / original_rel
        original_hash = _sha256_file(original_path)
        if original_hash != original_info.get("sha256"):
            raise SafeExportError(f"{key} original archive differs from PREDICTIONS_READY SHA")
        parent_expected_item = parent_expected[key]
        parent_rows, parent_cycles, parent_ids = _read_parent_identity(origin, parent_expected_item)

        safe_arrays: dict[str, np.ndarray] = {}
        original_headers: dict[str, Any] = {}
        original_member_hashes: dict[str, str] = {}
        with zipfile.ZipFile(original_path, "r") as archive:
            for member in _archive_member_names(archive):
                if member == "score_record_ids.npy":
                    header = _read_object_header(archive, member)
                    original_headers[member] = header
                    if header["shape"] != [int(original_info["rows"])]:
                        raise SafeExportError(f"{key} original object ID header row count differs")
                else:
                    array, raw, header = _load_safe_member(archive, member)
                    safe_arrays[member[:-4]] = array
                    original_headers[member] = header
                    original_member_hashes[member] = _sha256_bytes(raw)
        score_rows = np.asarray(safe_arrays["score_rows"], dtype=np.int64)
        score_cycles = np.asarray(safe_arrays["score_cycles"], dtype=np.int64)
        expected_rows = np.flatnonzero(train_cycles > origin).astype(np.int64)
        ids_unicode, identity_record = _validate_rows_and_ids(
            origin=origin, score_rows=score_rows, score_cycles=score_cycles,
            all_ids=train_ids, all_cycles=train_cycles,
            declared_old_ids_sha256=str(original_info.get("score_record_ids_sha256", "")),
            expected_rows=expected_rows, parent_rows=parent_rows,
            parent_cycles=parent_cycles, parent_ids=parent_ids,
        )
        if len(score_rows) != int(original_info.get("rows", -1)):
            raise SafeExportError(f"{key} row count differs from prediction receipt")
        if moments.sha256_array(score_cycles) != original_info.get("score_cycles_sha256"):
            raise SafeExportError(f"{key} score cycle hash differs from prediction receipt")
        present = np.asarray(safe_arrays["present_mask"], dtype=bool)
        if (int(present.sum()) != int(original_info.get("present_rows", -1))
                or int((~present).sum()) != int(original_info.get("missing_rows", -1))):
            raise SafeExportError(f"{key} stream availability counts differ from prediction receipt")

        derived_rel = f"experiments/runs/optimization_v3_research_next_v1/POS_MOMENTS_V1/safe_exports_v1/{key}/predictions_{key}_safe_v1.npz"
        derived_path = moments.ROOT / derived_rel
        archive_record = _write_safe_npz(original_path, derived_path, ids_unicode)
        member_record = archive_record["members"]
        for name, original_member_hash in original_member_hashes.items():
            if member_record[name]["original_member_sha256"] != original_member_hash:
                raise SafeExportError(f"{key} unchanged member manifest hash differs: {name}")
        origins[key] = {
            "origin": origin,
            "original_archive_path": original_rel,
            "original_archive_sha256": original_hash,
            "original_ready_declared_sha256": original_info["sha256"],
            "original_score_record_ids_header": original_headers["score_record_ids.npy"],
            "original_score_record_ids_member_status": "header_only; payload_not_deserialized_or_member_hashed",
            "members": member_record,
            "safe_export_path": derived_rel,
            "safe_export_sha256": archive_record["derived_archive_sha256"],
            "score_record_ids_identity": identity_record,
            "score_rows": len(score_rows),
            "score_rows_sha256": moments.sha256_array(score_rows),
        }

    body: dict[str, Any] = {
        "schema_version": 1, "adapter_id": ADAPTER_ID,
        "status": "PASS_SAFE_EXPORTS_READY_FOR_LABEL_CLOSED_VERIFICATION",
        "source_commit": freeze_data["source_commit"],
        "freeze_signature": freeze_data["freeze_signature"],
        "input_sha256": freeze_data["input_sha256"],
        "original_predictions_ready_sha256": _sha256_file(PRELABEL_RUN_READY),
        "original_parent_ready_sha256": _sha256_file(PILOT_READY),
        "adapter_source_path": Path(__file__).resolve().relative_to(moments.ROOT).as_posix(),
        "adapter_source_sha256": _sha256_file(Path(__file__)),
        "parent_schema_adapter_source_sha256": _sha256_file(Path(parent_adapter.__file__)),
        "parent_origins_verified_by": parent_index,
        "train_identity_columns_only": ["record_id", "cycle_code"],
        "query_labels_read": False, "test_or_sample_access": False,
        "original_object_member_policy": "Never deserialize or member-hash the original object member; verify archive SHA, header, Train-row ID logical hash, and derive a Unicode replacement.",
        "origins": origins,
        "ledger_sha256_before_export": ledger_before,
        "ledger_snapshot_before_export": ledger_snapshot,
        "ledger_events_before_export": len(ledger_events),
    }
    if _sha256_file(moments.LEDGER) != ledger_before:
        raise SafeExportError("Shared fit budget ledger changed during safe export")
    body["manifest_signature"] = _canonical_hash(body)
    _write_signed_manifest(body)
    return body


def _write_signed_manifest(body: Mapping[str, Any]) -> None:
    payload = json.dumps(body, sort_keys=True, indent=2, ensure_ascii=False) + "\n"
    for path in (MANIFEST_REPORT, MANIFEST_RUN):
        if path.exists():
            current = json.loads(path.read_text(encoding="utf-8"))
            if current != dict(body):
                raise SafeExportError(f"Refusing to overwrite different signed export manifest: {path}")
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            temp = path.with_suffix(path.suffix + ".tmp")
            temp.write_text(payload, encoding="utf-8")
            temp.replace(path)


def _load_signed_manifest() -> dict[str, Any]:
    first = json.loads(MANIFEST_REPORT.read_text(encoding="utf-8"))
    second = json.loads(MANIFEST_RUN.read_text(encoding="utf-8"))
    if first != second:
        raise SafeExportError("Tracked/run safe export manifests differ")
    signature = first.get("manifest_signature")
    unsigned = dict(first)
    unsigned.pop("manifest_signature", None)
    if signature != _canonical_hash(unsigned):
        raise SafeExportError("Safe export manifest signature mismatch")
    freeze_data = moments._load_freeze()
    if (first.get("adapter_id") != ADAPTER_ID
            or first.get("freeze_signature") != freeze_data["freeze_signature"]
            or first.get("input_sha256") != freeze_data["input_sha256"]
            or first.get("adapter_source_sha256") != _sha256_file(Path(__file__))):
        raise SafeExportError("Safe export manifest is not bound to current frozen source/input")
    return first


def _validate_archive_export(origin: int, ready: Mapping[str, Any], manifest: Mapping[str, Any],
                             train_ids: np.ndarray, train_cycles: np.ndarray) -> dict[str, Any]:
    key = f"E{origin}"
    item = manifest["origins"].get(key)
    ready_item = ready.get("predictions", {}).get(key)
    if not isinstance(item, Mapping) or not isinstance(ready_item, Mapping):
        raise SafeExportError(f"Safe export is missing whitelisted origin {key}")
    original_rel = _safe_relpath(item.get("original_archive_path", ""))
    allowed_original = f"experiments/runs/optimization_v3_research_next_v1/POS_MOMENTS_V1/{key}/predictions_{key}.npz"
    if original_rel != allowed_original or ready_item.get("path") != original_rel:
        raise SafeExportError(f"{key} original archive is outside the exact whitelist")
    original_path = moments.ROOT / original_rel
    if (_sha256_file(original_path) != item.get("original_archive_sha256")
            or ready_item.get("sha256") != item.get("original_archive_sha256")):
        raise SafeExportError(f"{key} original archive SHA differs from signed manifest/ready receipt")
    safe_rel = _safe_relpath(item.get("safe_export_path", ""))
    allowed_safe = f"experiments/runs/optimization_v3_research_next_v1/POS_MOMENTS_V1/safe_exports_v1/{key}/predictions_{key}_safe_v1.npz"
    if safe_rel != allowed_safe:
        raise SafeExportError(f"{key} safe export path is outside the exact whitelist")
    safe_path = moments.ROOT / safe_rel
    if _sha256_file(safe_path) != item.get("safe_export_sha256"):
        raise SafeExportError(f"{key} safe archive hash differs from signed export manifest")

    members = item.get("members", {})
    with zipfile.ZipFile(original_path, "r") as source, zipfile.ZipFile(safe_path, "r") as target:
        source_names = _archive_member_names(source)
        target_names = _archive_member_names(target)
        if source_names != target_names or set(members) != set(source_names):
            raise SafeExportError(f"{key} original/derived NPY member identity differs")
        for name in source_names:
            record = members[name]
            if name == "score_record_ids.npy":
                original_header = _read_object_header(source, name)
                if original_header != record.get("original_header") or record.get("original_member_sha256") is not None:
                    raise SafeExportError(f"{key} original object ID header/policy changed")
                safe_raw = target.read(name)
                safe_header = _read_npy_header(io.BytesIO(safe_raw))
                if (safe_header.get("kind") != "U"
                        or safe_header.get("shape") != original_header.get("shape")
                        or _sha256_bytes(safe_raw) != record.get("derived_member_sha256")):
                    raise SafeExportError(f"{key} Unicode replacement header/hash differs")
            else:
                src_raw, dst_raw = source.read(name), target.read(name)
                try:
                    _assert_copied_member_identity(name, src_raw, dst_raw, record)
                except SafeExportError as exc:
                    raise SafeExportError(f"{key} {exc}") from exc

    # Only the derived bundle is decoded; allow_pickle is explicitly false.
    with np.load(safe_path, allow_pickle=False) as bundle:
        if set(bundle.files) != {key_name[:-4] for key_name in source_names}:
            raise SafeExportError(f"{key} safe NPZ keys differ from signed manifest")
        arrays = {name: np.asarray(bundle[name]).copy() for name in bundle.files}
    rows = np.asarray(arrays["score_rows"], dtype=np.int64)
    cycles = np.asarray(arrays["score_cycles"], dtype=np.int64)
    ids = np.asarray(arrays["score_record_ids"])
    parent_expected = parent_adapter._frozen_parent_expected(moments._load_freeze())
    parent_rows, parent_cycles, parent_ids = _read_parent_identity(origin, parent_expected[key])
    expected_rows = np.flatnonzero(train_cycles > origin).astype(np.int64)
    reconstructed, identity = _validate_rows_and_ids(
        origin=origin, score_rows=rows, score_cycles=cycles, all_ids=train_ids,
        all_cycles=train_cycles, declared_old_ids_sha256=str(ready_item.get("score_record_ids_sha256", "")),
        expected_rows=expected_rows, parent_rows=parent_rows, parent_cycles=parent_cycles, parent_ids=parent_ids,
    )
    if (ids.dtype.kind != "U" or not np.array_equal(ids, reconstructed)
            or moments.sha256_array(ids) != item["score_record_ids_identity"]["new_unicode_logical_sha256"]):
        raise SafeExportError(f"{key} safe Unicode IDs differ from target-free Train reconstruction")
    if moments.sha256_array(cycles) != ready_item.get("score_cycles_sha256"):
        raise SafeExportError(f"{key} score cycle hash differs from original receipt")
    return {"origin": origin, "safe_path": safe_rel, "safe_sha256": item["safe_export_sha256"],
            "arrays": arrays, "identity": identity}


def validate_all_exports() -> tuple[dict[str, Any], dict[str, Any]]:
    manifest = _load_signed_manifest()
    freeze_data = moments._load_freeze()
    if _sha256_file(moments.LEDGER) != manifest.get("ledger_sha256_before_export"):
        raise SafeExportError("Shared ledger changed after the safe export freeze point")
    _, ledger_snapshot = moments._check_live_ledger(freeze_data)
    if ledger_snapshot != manifest.get("ledger_snapshot_before_export"):
        raise SafeExportError("Shared ledger snapshot changed after safe export")
    if _sha256_file(PRELABEL_RUN_READY) != manifest.get("original_predictions_ready_sha256"):
        raise SafeExportError("Original PREDICTIONS_READY bytes changed after safe export")
    if _sha256_file(PILOT_READY) != manifest.get("original_parent_ready_sha256"):
        raise SafeExportError("Historical parent PREDICTIONS_READY bytes changed after safe export")
    original_ready = moments._read_json(PRELABEL_RUN_READY)
    if original_ready != moments._read_json(PRELABEL_REPORT_READY):
        raise SafeExportError("Original tracked/run prediction receipts differ")
    train_ids, train_cycles = _train_identity(freeze_data)
    checked = {
        f"E{origin}": _validate_archive_export(origin, original_ready, manifest, train_ids, train_cycles)
        for origin in (67, 75)
    }
    return manifest, {key: {k: v for k, v in value.items() if k != "arrays"}
                      for key, value in checked.items()}


def _safe_ready(manifest: Mapping[str, Any]) -> dict[str, Any]:
    ready = moments._read_json(PRELABEL_RUN_READY)
    adapted = json.loads(json.dumps(ready))
    for key in ("E67", "E75"):
        origin = manifest["origins"][key]
        adapted["predictions"][key]["path"] = origin["safe_export_path"]
        adapted["predictions"][key]["sha256"] = origin["safe_export_sha256"]
    return adapted


@contextmanager
def _installed_read_adapters(safe_ready: Mapping[str, Any], freeze_data: Mapping[str, Any]) -> Iterator[None]:
    original_moments_reader = moments._read_json
    original_verifier_reader = verifier._read_json
    ready_paths = {PRELABEL_RUN_READY.resolve(), PRELABEL_REPORT_READY.resolve()}
    parent_ready_path = PILOT_READY.resolve()
    expected_parent = parent_adapter._frozen_parent_expected(freeze_data)
    parent_payload = original_moments_reader(PILOT_READY)
    normalized_parent = parent_adapter.normalize_prediction_index(
        parent_payload, root=moments.ROOT, expected=expected_parent, require_parent_manifests=True
    )
    adapted_parent = {**parent_payload, "predictions": normalized_parent}

    def _moments_reader(path: Path) -> Any:
        if Path(path).resolve() in ready_paths:
            return dict(safe_ready)
        return original_moments_reader(path)

    def _verifier_reader(path: Path) -> Any:
        resolved = Path(path).resolve()
        if resolved in ready_paths:
            return dict(safe_ready)
        if resolved == parent_ready_path:
            return adapted_parent
        return original_verifier_reader(path)

    moments._read_json = _moments_reader
    verifier._read_json = _verifier_reader
    try:
        yield
    finally:
        moments._read_json = original_moments_reader
        verifier._read_json = original_verifier_reader


def run_prelabel() -> dict[str, Any]:
    manifest, checks = validate_all_exports()
    freeze_data = moments._load_freeze()
    safe_ready = _safe_ready(manifest)
    with _installed_read_adapters(safe_ready, freeze_data):
        result = verifier.verify_prelabel_outputs(moments.ROOT, moments.RUN_ROOT, freeze_data)
    if result.get("status") != "PASS" or result.get("query_labels_read") is not False:
        raise SafeExportError("Unchanged frozen verifier did not PASS on signed safe exports")
    result.update({
        "verification_method": "unchanged frozen verifier through whitelist-only signed safe export reader",
        "adapter_id": ADAPTER_ID,
        "adapter_source_path": Path(__file__).resolve().relative_to(moments.ROOT).as_posix(),
        "adapter_source_sha256": _sha256_file(Path(__file__)),
        "safe_export_manifest_signature": manifest["manifest_signature"],
        "safe_export_manifest_sha256": _sha256_file(MANIFEST_REPORT),
        "safe_export_origins": checks,
        "original_frozen_prelabel_failure_preserved": _sha256_file(PRELABEL_FAILURE_REPORT),
        "query_labels_read": False, "test_or_sample_access": False,
    })
    return result


def run_score() -> dict[str, Any]:
    manifest, _ = validate_all_exports()
    freeze_data = moments._load_freeze()
    prelabel = moments._read_json(moments.RUN_ROOT / "PRELABEL_VERIFICATION.json")
    if (prelabel.get("status") != "PASS" or prelabel.get("query_labels_read") is not False
            or prelabel.get("freeze_signature") != freeze_data["freeze_signature"]
            or prelabel.get("adapter_id") != ADAPTER_ID
            or prelabel.get("safe_export_manifest_signature") != manifest["manifest_signature"]
            or prelabel.get("safe_export_manifest_sha256") != _sha256_file(MANIFEST_REPORT)):
        raise SafeExportError("Outcome metrics remain closed until safe-export prelabel PASS")
    safe_ready = _safe_ready(manifest)
    with _installed_read_adapters(safe_ready, freeze_data):
        result = moments.score()
    return result


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(payload, sort_keys=True, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temp.replace(path)


def main() -> int:
    if len(sys.argv) != 2 or sys.argv[1] not in {"export", "prelabel", "score"}:
        raise SystemExit("usage: optimization_v3_positional_moments_v1_safe_export.py {export|prelabel|score}")
    command = sys.argv[1]
    if command == "export":
        result = export_all()
        print(json.dumps(result, sort_keys=True, indent=2, ensure_ascii=False))
    elif command == "prelabel":
        try:
            result = run_prelabel()
        except Exception as exc:
            failure = {
                "schema_version": 1, "adapter_id": ADAPTER_ID,
                "status": "FAIL_PRELABEL; SCORE_REMAINS_CLOSED",
                "error_type": type(exc).__name__, "error": str(exc),
                "freeze_signature": moments._load_freeze()["freeze_signature"],
                "safe_export_manifest_signature": _load_signed_manifest().get("manifest_signature"),
                "query_labels_read": False, "test_or_sample_access": False,
            }
            _atomic_json(moments.REPORT_ROOT / "SAFE_EXPORT_PRELABEL_FAILURE_V1.json", failure)
            _atomic_json(moments.RUN_ROOT / "SAFE_EXPORT_PRELABEL_FAILURE_V1.json", failure)
            raise
        moments._atomic_json(moments.REPORT_ROOT / "PRELABEL_VERIFICATION.json", result)
        moments._atomic_json(moments.RUN_ROOT / "PRELABEL_VERIFICATION.json", result)
        _atomic_json(moments.REPORT_ROOT / "SAFE_EXPORT_PRELABEL_VERIFICATION_V1.json", result)
        _atomic_json(moments.RUN_ROOT / "SAFE_EXPORT_PRELABEL_VERIFICATION_V1.json", result)
        print(json.dumps(result, sort_keys=True, indent=2, ensure_ascii=False))
    else:
        result = run_score()
        print(json.dumps(result, sort_keys=True, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
