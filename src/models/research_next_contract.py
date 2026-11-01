"""Immutable, pickle-free artifact contract for the Research Next experiments.

Identity is intentionally explicit: callers pass only stable experiment identity
fields (recipe, source hashes, split, etc.). This module hashes every identity
field it receives and does not silently remove timestamps, paths, or other
dynamic context.

The bundle directory is published with one same-volume directory rename. Run a
single publisher for a destination at a time. Failed staging directories are
left in place for diagnosis; an existing destination is never replaced.
"""
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from io import BytesIO
import json
import os
from pathlib import Path
import re
import tempfile
from typing import Any, Mapping

import numpy as np


SCHEMA_VERSION = 1
_ARRAY_KEY = re.compile(r"^[A-Za-z][A-Za-z0-9_]*$")
_REQUIRED_ARRAYS = frozenset(
    {"score_rows", "score_record_ids", "prediction", "baseline_prediction"}
)
_TASK_RANGES = {"classification": (0.0, 1.0), "regression": (0.0, 100.0)}


class BundleContractError(RuntimeError):
    """A payload or saved bundle violates the immutable output contract."""


@dataclass(frozen=True)
class ResearchBundle:
    """Validated arrays and JSON receipt returned by read/resume operations."""

    arrays: dict[str, np.ndarray]
    receipt: dict[str, Any]


def normalize_json(value: Any, *, _path: str = "$") -> Any:
    """Return strict JSON data; tuples and NumPy values become JSON lists/scalars.

    Mapping keys must already be strings. Nonfinite numbers, byte strings,
    arbitrary Python objects, and sets are rejected rather than stringified.
    """
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, np.generic):
        return normalize_json(value.item(), _path=_path)
    if isinstance(value, np.ndarray):
        return normalize_json(value.tolist(), _path=_path)
    if isinstance(value, (float, np.floating)):
        number = float(value)
        if not np.isfinite(number):
            raise BundleContractError(f"Nonfinite JSON number at {_path}")
        return number
    if isinstance(value, Mapping):
        normalized: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise BundleContractError(f"JSON object key at {_path} must be a string")
            normalized[key] = normalize_json(item, _path=f"{_path}.{key}")
        return normalized
    if isinstance(value, (list, tuple)):
        return [normalize_json(item, _path=f"{_path}[{index}]") for index, item in enumerate(value)]
    raise BundleContractError(f"Unsupported JSON value at {_path}: {type(value).__name__}")


def canonical_json_bytes(value: Any) -> bytes:
    """Serialize strict JSON with deterministic key ordering and separators."""
    try:
        normalized = normalize_json(value)
        return json.dumps(
            normalized,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        if isinstance(exc, BundleContractError):
            raise
        raise BundleContractError(f"Could not encode canonical JSON: {exc}") from exc


def canonical_sha256(value: Any) -> str:
    return sha256(canonical_json_bytes(value)).hexdigest()


def file_sha256(path: str | Path) -> str:
    digest = sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _record_id_unicode(value: Any, *, field: str) -> np.ndarray:
    """Convert only the explicitly declared record-ID field to Unicode."""
    if isinstance(value, np.ndarray):
        if value.ndim != 1:
            raise BundleContractError(f"{field} must be a one-dimensional string array")
        items = value.tolist()
    elif isinstance(value, (list, tuple)):
        items = list(value)
    else:
        raise BundleContractError(f"{field} must be a one-dimensional string sequence")
    if not all(isinstance(item, (str, np.str_)) for item in items):
        raise BundleContractError(f"{field} may contain only strings")
    return np.asarray(items, dtype=str)


def partition_payload(
    payload: Mapping[str, Any], *, record_id_field: str = "score_record_ids"
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Split one payload into safe NPZ arrays and JSON diagnostics.

    Dictionaries and non-array diagnostic values go to JSON metadata. Numeric,
    boolean, and already-Unicode ndarrays are retained as arrays. Python string
    sequences are converted to Unicode only for ``record_id_field``; other
    string sequences remain JSON metadata. A 0-D object ndarray is accepted
    only when it contains a Mapping, which is normalized into metadata. Other
    object ndarrays fail closed (except a string-only record-ID field).
    """
    if not isinstance(payload, Mapping):
        raise BundleContractError("Payload must be a mapping")
    if not isinstance(record_id_field, str) or not record_id_field:
        raise BundleContractError("record_id_field must be a nonempty string")

    arrays: dict[str, np.ndarray] = {}
    metadata: dict[str, Any] = {}
    for name, value in payload.items():
        if not isinstance(name, str) or not _ARRAY_KEY.fullmatch(name):
            raise BundleContractError(f"Invalid payload key: {name!r}")

        if isinstance(value, Mapping):
            metadata[name] = normalize_json(value, _path=f"$.{name}")
            continue

        if isinstance(value, np.ndarray) and value.dtype.kind == "O":
            if value.ndim == 0 and isinstance(value.item(), Mapping):
                metadata[name] = normalize_json(value.item(), _path=f"$.{name}")
                continue
            if name == record_id_field:
                arrays[name] = _record_id_unicode(value, field=name)
                continue
            raise BundleContractError(f"Object array is prohibited for payload field {name!r}")

        if name == record_id_field and isinstance(value, (list, tuple)):
            arrays[name] = _record_id_unicode(value, field=name)
            continue

        if isinstance(value, (list, tuple)):
            try:
                candidate = np.asarray(value)
            except (TypeError, ValueError):
                candidate = np.asarray(value, dtype=object)
            if candidate.dtype.kind == "O":
                metadata[name] = normalize_json(value, _path=f"$.{name}")
                continue
            if candidate.dtype.kind in "US" and name != record_id_field:
                metadata[name] = normalize_json(value, _path=f"$.{name}")
                continue
            value = candidate

        if isinstance(value, np.ndarray) or isinstance(value, (np.number, bool, int, float)):
            array = np.asarray(value)
            if array.dtype.kind == "O":
                raise BundleContractError(f"Object array is prohibited for payload field {name!r}")
            if array.dtype.kind not in "biufU":
                raise BundleContractError(
                    f"Only numeric, boolean, or Unicode arrays are allowed ({name}: {array.dtype})"
                )
            if array.dtype.kind == "f" and not np.isfinite(array).all():
                raise BundleContractError(f"Nonfinite values in array {name!r}")
            arrays[name] = np.array(array, copy=True)
            continue

        metadata[name] = normalize_json(value, _path=f"$.{name}")

    return arrays, metadata


def _require_vector(arrays: Mapping[str, np.ndarray], key: str, *, dtype_kinds: str) -> np.ndarray:
    array = arrays[key]
    if array.ndim != 1 or array.dtype.kind not in dtype_kinds:
        raise BundleContractError(f"{key} must be a one-dimensional {dtype_kinds}-kind array")
    return array


def validate_arrays(arrays: Mapping[str, np.ndarray], *, task: str) -> None:
    """Validate the shared row/prediction contract for a classification or regression bundle."""
    task_name = task.casefold() if isinstance(task, str) else ""
    if task_name not in _TASK_RANGES:
        raise BundleContractError(f"Unsupported task {task!r}; expected classification or regression")
    if not isinstance(arrays, Mapping):
        raise BundleContractError("Arrays must be a mapping")
    names = set(arrays)
    missing = _REQUIRED_ARRAYS - names
    if missing:
        raise BundleContractError(f"Missing required arrays: {sorted(missing)}")

    safe: dict[str, np.ndarray] = {}
    for name, values in arrays.items():
        if not isinstance(name, str) or not _ARRAY_KEY.fullmatch(name):
            raise BundleContractError(f"Invalid NPZ array name: {name!r}")
        array = np.asarray(values)
        if array.dtype.kind not in "biufU":
            raise BundleContractError(f"Unsafe NPZ dtype for {name}: {array.dtype}")
        if array.dtype.kind == "f" and not np.isfinite(array).all():
            raise BundleContractError(f"Nonfinite values in NPZ array {name}")
        safe[name] = array

    rows = _require_vector(safe, "score_rows", dtype_kinds="iu")
    if len(rows) == 0:
        raise BundleContractError("score_rows must not be empty")
    if len(np.unique(rows)) != len(rows):
        raise BundleContractError("score_rows must be unique")
    if np.any(rows < 0):
        raise BundleContractError("score_rows must be nonnegative")

    record_ids = _require_vector(safe, "score_record_ids", dtype_kinds="U")
    if len(record_ids) != len(rows):
        raise BundleContractError("score_record_ids and score_rows lengths differ")
    if np.any(record_ids == ""):
        raise BundleContractError("score_record_ids may not contain empty strings")
    if len(np.unique(record_ids)) != len(record_ids):
        raise BundleContractError("score_record_ids must be unique")

    prediction = _require_vector(safe, "prediction", dtype_kinds="iuf")
    baseline = _require_vector(safe, "baseline_prediction", dtype_kinds="iuf")
    if prediction.shape != rows.shape or baseline.shape != rows.shape:
        raise BundleContractError("prediction and baseline_prediction must align with score_rows")
    lower, upper = _TASK_RANGES[task_name]
    for name, values in (("prediction", prediction), ("baseline_prediction", baseline)):
        if np.any(values < lower) or np.any(values > upper):
            raise BundleContractError(f"{name} is outside the {task_name} range [{lower}, {upper}]")

    for name, values in safe.items():
        if name.startswith("raw_"):
            raw = _require_vector(safe, name, dtype_kinds="iuf")
            if raw.shape != rows.shape:
                raise BundleContractError(f"{name} must align with score_rows")


def _normalize_identity(identity: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(identity, Mapping):
        raise BundleContractError("Identity must be a mapping of stable experiment fields")
    value = normalize_json(identity, _path="$.identity")
    assert isinstance(value, dict)
    if not value:
        raise BundleContractError("Identity must contain at least one stable field")
    return value


def _normalize_metadata(metadata: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(metadata, Mapping):
        raise BundleContractError("Metadata must be a mapping")
    value = normalize_json(metadata, _path="$.metadata")
    assert isinstance(value, dict)
    return value


def _npz_bytes(arrays: Mapping[str, np.ndarray]) -> bytes:
    buffer = BytesIO()
    np.savez_compressed(buffer, **arrays)
    return buffer.getvalue()


def _write_synced(path: Path, data: bytes) -> None:
    with path.open("xb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())


def publish_bundle(
    destination: str | Path,
    payload: Mapping[str, Any],
    identity: Mapping[str, Any],
    *,
    task: str,
) -> dict[str, Any]:
    """Atomically publish one immutable ``arrays.npz`` + ``receipt.json`` bundle.

    All JSON and array validation runs before a staging directory is created.
    The final destination must not exist. If writing or publication fails after
    staging starts, the stage remains available for diagnosis.
    """
    destination = Path(destination)
    if not destination.name or destination.name in {".", ".."}:
        raise BundleContractError("Destination must be a named bundle directory")
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite immutable bundle: {destination}")

    arrays, metadata = partition_payload(payload)
    validate_arrays(arrays, task=task)
    stable_identity = _normalize_identity(identity)
    metadata = _normalize_metadata(metadata)
    # Materialize every canonical JSON object before touching disk.
    identity_bytes = canonical_json_bytes(stable_identity)
    metadata_bytes = canonical_json_bytes(metadata)
    array_bytes = _npz_bytes(arrays)
    receipt: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "task": task.casefold(),
        "identity": stable_identity,
        "identity_sha256": sha256(identity_bytes).hexdigest(),
        "metadata": metadata,
        "metadata_sha256": sha256(metadata_bytes).hexdigest(),
        "array_names": sorted(arrays),
        "npz_sha256": sha256(array_bytes).hexdigest(),
    }
    receipt["receipt_sha256"] = canonical_sha256(receipt)
    receipt_bytes = json.dumps(
        receipt, sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False
    ).encode("utf-8") + b"\n"

    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{destination.name}.staging-", dir=destination.parent))
    try:
        _write_synced(staging / "arrays.npz", array_bytes)
        _write_synced(staging / "receipt.json", receipt_bytes)
        os.rename(staging, destination)
    except Exception as exc:
        raise BundleContractError(
            f"Bundle publication failed; diagnostic staging directory preserved at {staging}: {exc}"
        ) from exc
    return receipt


def _reject_json_constant(value: str) -> None:
    raise BundleContractError(f"Nonfinite JSON token in receipt: {value}")


def read_bundle(
    destination: str | Path,
    expected_identity: Mapping[str, Any],
    *,
    task: str,
) -> ResearchBundle:
    """Read and verify a bundle, including receipt, NPZ, identity, and array contract."""
    destination = Path(destination)
    npz_path = destination / "arrays.npz"
    receipt_path = destination / "receipt.json"
    if not destination.is_dir() or not npz_path.is_file() or not receipt_path.is_file():
        raise BundleContractError(f"Incomplete research bundle: {destination}")
    try:
        receipt = json.loads(
            receipt_path.read_text(encoding="utf-8"), parse_constant=_reject_json_constant
        )
    except (OSError, json.JSONDecodeError) as exc:
        raise BundleContractError(f"Could not read bundle receipt at {receipt_path}: {exc}") from exc
    if not isinstance(receipt, dict) or receipt.get("schema_version") != SCHEMA_VERSION:
        raise BundleContractError("Unexpected or invalid bundle receipt schema")
    integrity = {key: value for key, value in receipt.items() if key != "receipt_sha256"}
    if receipt.get("receipt_sha256") != canonical_sha256(integrity):
        raise BundleContractError("Receipt integrity hash mismatch")

    stable_identity = _normalize_identity(expected_identity)
    if receipt.get("identity_sha256") != canonical_sha256(stable_identity):
        raise BundleContractError("Bundle identity hash does not match expected identity")
    if receipt.get("identity") != stable_identity:
        raise BundleContractError("Bundle identity does not match expected identity")

    metadata = _normalize_metadata(receipt.get("metadata"))
    if receipt.get("metadata_sha256") != canonical_sha256(metadata):
        raise BundleContractError("Bundle metadata hash mismatch")
    task_name = task.casefold() if isinstance(task, str) else ""
    if task_name not in _TASK_RANGES or receipt.get("task") != task_name:
        raise BundleContractError("Bundle task does not match expected task")
    if receipt.get("npz_sha256") != file_sha256(npz_path):
        raise BundleContractError("NPZ hash differs from receipt")

    try:
        with np.load(npz_path, allow_pickle=False) as archive:
            arrays = {key: archive[key].copy() for key in archive.files}
    except Exception as exc:
        raise BundleContractError(f"Could not safely load bundle NPZ: {exc}") from exc
    if receipt.get("array_names") != sorted(arrays):
        raise BundleContractError("NPZ array names differ from the receipt")
    validate_arrays(arrays, task=task_name)
    return ResearchBundle(arrays=arrays, receipt=receipt)


def resume_or_publish_bundle(
    destination: str | Path,
    payload: Mapping[str, Any],
    identity: Mapping[str, Any],
    *,
    task: str,
) -> ResearchBundle:
    """Resume an exact existing bundle, or publish a new one; never overwrite.

    Existing bundle arrays, identity, task, and normalized JSON metadata must
    match the requested inputs exactly. A mismatch is reported without mutation.
    """
    destination = Path(destination)
    arrays, metadata = partition_payload(payload)
    validate_arrays(arrays, task=task)
    stable_identity = _normalize_identity(identity)
    metadata = _normalize_metadata(metadata)
    if destination.exists():
        saved = read_bundle(destination, stable_identity, task=task)
        if saved.receipt.get("metadata") != metadata:
            raise BundleContractError("Existing bundle metadata differs; refusing resume")
        if set(saved.arrays) != set(arrays):
            raise BundleContractError("Existing bundle array names differ; refusing resume")
        for name in arrays:
            if saved.arrays[name].dtype != arrays[name].dtype or not np.array_equal(
                saved.arrays[name], arrays[name]
            ):
                raise BundleContractError(f"Existing bundle array {name!r} differs; refusing resume")
        return saved

    publish_bundle(destination, payload, stable_identity, task=task)
    return read_bundle(destination, stable_identity, task=task)

