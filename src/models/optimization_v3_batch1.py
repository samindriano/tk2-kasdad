"""Reusable, fail-closed primitives for the V3 batch1 research queue.

The runner deliberately receives only an explicit feature allowlist and fit
targets. It never opens Test or passes validation targets to a model adapter.
Card-specific feature construction and policy reconstruction live in frozen,
versioned adapters/verifiers rather than in this module.
"""
from __future__ import annotations

from datetime import datetime, timezone
from contextlib import contextmanager
from hashlib import sha256
from importlib import metadata as importlib_metadata
from io import BytesIO
import json
import os
from pathlib import Path
import tempfile
import time
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits


PROTECTED_FEATURES = frozenset({"record_id", "artifact_signal", "threat_rank"})
SCHEMA_VERSION = 1


class BatchIntegrityError(RuntimeError):
    """Raised when a frozen input or persisted result cannot be trusted."""


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        if value.dtype == object:
            raise TypeError("Object arrays are not allowed in batch artifacts")
        return [_jsonable(v) for v in value.tolist()]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        result = float(value)
        if not np.isfinite(result):
            raise ValueError("Nonfinite JSON number in batch metadata")
        return result
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if value is None or isinstance(value, (str, int)):
        return value
    raise TypeError(f"Unsupported batch JSON value: {type(value).__name__}")


def canonical_json(value: Any) -> bytes:
    return json.dumps(
        _jsonable(value), sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def json_sha256(value: Any) -> str:
    return sha256(canonical_json(value)).hexdigest()


def file_sha256(path: str | Path) -> str:
    digest = sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_queue(path: str | Path) -> dict[str, Any]:
    queue = json.loads(Path(path).read_text(encoding="utf-8"))
    cards = queue.get("cards")
    order = queue.get("experiment_order")
    if queue.get("research_batch_name") != "optimization_v3_batch1":
        raise BatchIntegrityError("Unexpected queue namespace")
    if not isinstance(cards, list) or len(cards) != 45 or not isinstance(order, list):
        raise BatchIntegrityError("Queue must contain the declared 45 cards and order")
    ids = [card.get("id") for card in cards]
    if len(set(ids)) != 45 or set(ids) != set(order):
        raise BatchIntegrityError("Queue card IDs and experiment order disagree")
    if [card["id"] for card in sorted(cards, key=lambda x: x["queue_order"])] != order:
        raise BatchIntegrityError("Queue order does not match card queue_order")
    return queue


def package_versions(names: Sequence[str]) -> dict[str, str]:
    return {name: importlib_metadata.version(name) for name in names}


def _write_link_exclusive(path: Path, payload: bytes) -> None:
    """Publish a complete file atomically without replacing an existing path."""
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".partial", dir=path.parent)
    temp_path = Path(temp_name)
    try:
        with os.fdopen(handle, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        # Same-directory hard-link publication is atomic and fails if path exists.
        os.link(temp_path, path)
    finally:
        try:
            temp_path.unlink()
        except FileNotFoundError:
            pass


def write_json_exclusive(path: str | Path, value: Any) -> None:
    payload = json.dumps(_jsonable(value), sort_keys=True, indent=2, allow_nan=False).encode("utf-8") + b"\n"
    _write_link_exclusive(Path(path), payload)


def _npz_bytes(arrays: Mapping[str, np.ndarray]) -> bytes:
    buffer = BytesIO()
    safe: dict[str, np.ndarray] = {}
    for name, values in arrays.items():
        arr = np.asarray(values)
        if arr.dtype == object:
            raise TypeError(f"Object dtype prohibited in NPZ field {name}")
        safe[name] = arr
    np.savez_compressed(buffer, **safe)
    return buffer.getvalue()


def write_bundle_exclusive(
    npz_path: str | Path,
    arrays: Mapping[str, np.ndarray],
    identity: Mapping[str, Any],
    metadata: Mapping[str, Any],
) -> dict[str, Any]:
    """Write an NPZ then its hash-bound sidecar; never overwrite either file."""
    npz_path = Path(npz_path)
    sidecar_path = npz_path.with_suffix(".json")
    if npz_path.exists() or sidecar_path.exists():
        raise FileExistsError(f"Refusing to overwrite or complete existing bundle: {npz_path}")
    payload = _npz_bytes(arrays)
    sidecar = {
        "schema_version": SCHEMA_VERSION,
        "identity": _jsonable(identity),
        "identity_sha256": json_sha256(identity),
        "npz_sha256": sha256(payload).hexdigest(),
        "metadata": _jsonable(metadata),
    }
    sidecar["integrity_sha256"] = json_sha256(sidecar)
    _write_link_exclusive(npz_path, payload)
    # If this publication fails, the orphaned NPZ is intentionally preserved.
    _write_link_exclusive(sidecar_path, json.dumps(sidecar, sort_keys=True, indent=2, allow_nan=False).encode("utf-8") + b"\n")
    return sidecar


def read_bundle(
    npz_path: str | Path, expected_identity: Mapping[str, Any]
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    npz_path = Path(npz_path)
    sidecar_path = npz_path.with_suffix(".json")
    if not npz_path.is_file() or not sidecar_path.is_file():
        raise BatchIntegrityError(f"Incomplete artifact pair: {npz_path}")
    sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    if sidecar.get("schema_version") != SCHEMA_VERSION:
        raise BatchIntegrityError("Unexpected artifact schema version")
    integrity_record = {k: v for k, v in sidecar.items() if k != "integrity_sha256"}
    if sidecar.get("integrity_sha256") != json_sha256(integrity_record):
        raise BatchIntegrityError("Sidecar metadata integrity check failed")
    if sidecar.get("identity_sha256") != json_sha256(expected_identity) or sidecar.get("identity") != _jsonable(expected_identity):
        raise BatchIntegrityError("Artifact identity does not match the frozen recipe")
    if sidecar.get("npz_sha256") != file_sha256(npz_path):
        raise BatchIntegrityError("NPZ hash differs from its sidecar")
    with np.load(npz_path, allow_pickle=False) as archive:
        arrays = {key: archive[key] for key in archive.files}
    return arrays, sidecar


def append_event(path: str | Path, event: Mapping[str, Any]) -> None:
    """Append one complete audit line; callers serialize training jobs."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(_jsonable(event), sort_keys=True, allow_nan=False).encode("utf-8") + b"\n"
    fd = os.open(path, os.O_CREAT | os.O_APPEND | os.O_WRONLY, 0o600)
    try:
        os.write(fd, line)
        os.fsync(fd)
    finally:
        os.close(fd)


@contextmanager
def exclusive_training_slot(path: str | Path):
    """Acquire a process-owned lock; the OS releases it on crash or exit."""
    lock_path = Path(path)
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    stream = lock_path.open("a+b")
    try:
        stream.seek(0, os.SEEK_END)
        if stream.tell() == 0:
            stream.write(b"\0")
            stream.flush()
        stream.seek(0)
        if os.name == "nt":
            import msvcrt
            try:
                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            except OSError as exc:
                raise BatchIntegrityError("Another V3 batch1 training job holds the exclusive slot") from exc
            unlock = lambda: msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl
            try:
                fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError as exc:
                raise BatchIntegrityError("Another V3 batch1 training job holds the exclusive slot") from exc
            unlock = lambda: fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
        try:
            yield
        finally:
            stream.seek(0)
            unlock()
    finally:
        stream.close()


def reserve_card_budget(path: str | Path, *, budget_key: str, card_id: str,
                        predictor_fits: int, cap: int = 180) -> dict[str, Any]:
    """Reserve the full declared card budget before its first candidate fit."""
    ledger = Path(path)
    reservations: dict[str, tuple[int, str | None]] = {}
    if ledger.exists():
        for line_number, line in enumerate(ledger.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError as exc:
                raise BatchIntegrityError(f"Invalid batch ledger line {line_number}") from exc
            if event.get("event") == "BUDGET_RESERVED":
                key = str(event["budget_key"])
                count = int(event["predictor_fits"])
                card = str(event["card_id"]) if event.get("card_id") is not None else None
                reservation = (count, card)
                if key in reservations:
                    if reservations[key] != reservation:
                        raise BatchIntegrityError(f"Conflicting existing budget reservation for {key}")
                    raise BatchIntegrityError(f"Duplicate existing budget reservation for {key}")
                reservations[key] = reservation
    expected = int(predictor_fits)
    if expected < 0:
        raise BatchIntegrityError("Negative predictor fit reservation")
    if budget_key in reservations:
        existing_count, existing_card = reservations[budget_key]
        if existing_count != expected or existing_card != str(card_id):
            raise BatchIntegrityError(f"Existing reservation differs for {budget_key}")
        return {"status": "EXISTING_RESERVATION", "reserved_total": sum(value[0] for value in reservations.values()),
                "card_reservation": expected, "budget_key": budget_key}
    reserved_total = sum(value[0] for value in reservations.values())
    total = reserved_total + expected
    if total > int(cap):
        raise BatchIntegrityError(f"Fit budget blocked before card: reserved={reserved_total}, card={expected}, cap={cap}")
    append_event(ledger, {
        "event": "BUDGET_RESERVED", "at_utc": datetime.now(timezone.utc).isoformat(),
        "budget_key": budget_key, "card_id": card_id, "predictor_fits": expected,
        "cap": int(cap), "reserved_total_after": int(total),
    })
    return {"status": "RESERVED", "reserved_total": int(total),
            "card_reservation": expected, "budget_key": budget_key}


def make_freeze_identity(
    *, card: Mapping[str, Any], feature_names: Sequence[str],
    input_files: Mapping[str, str | Path], source_files: Sequence[str | Path],
    evaluation_views: Mapping[str, Any], packages: Sequence[str],
) -> dict[str, Any]:
    features = list(feature_names)
    if not features or len(set(features)) != len(features):
        raise BatchIntegrityError("Feature allowlist is empty or has duplicates")
    forbidden = sorted(set(features) & PROTECTED_FEATURES)
    if forbidden:
        raise BatchIntegrityError(f"Target/identifier leakage in feature allowlist: {forbidden}")
    missing = [str(path) for path in [*input_files.values(), *source_files] if not Path(path).is_file()]
    if missing:
        raise FileNotFoundError(f"Freeze inputs are missing: {missing}")
    return {
        "schema_version": SCHEMA_VERSION,
        "card": _jsonable(card),
        "feature_names": features,
        "input_sha256": {name: file_sha256(path) for name, path in sorted(input_files.items())},
        "source_sha256": {str(Path(path)): file_sha256(path) for path in sorted(source_files, key=str)},
        "package_versions": package_versions(packages),
        "evaluation_views": _jsonable(evaluation_views),
        "validation_labels_passed_to_adapter": False,
        "test_data_used": False,
    }


def ensure_frozen(path: str | Path, identity: Mapping[str, Any]) -> str:
    """Create the pre-fit freeze once, or require a byte-equivalent identity."""
    path = Path(path)
    signature = json_sha256(identity)
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing.get("identity") != _jsonable(identity) or existing.get("signature") != signature:
            raise BatchIntegrityError("Existing freeze differs from current source/input identity")
        return signature
    write_json_exclusive(path, {
        "identity": _jsonable(identity), "signature": signature,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    })
    return signature


def temporal_rows(
    frame: pd.DataFrame, cutoff: int, *, horizon: int = 12,
    cycle_column: str = "cycle_code",
) -> tuple[np.ndarray, np.ndarray]:
    cycles = pd.to_numeric(frame[cycle_column], errors="raise").to_numpy(dtype=int)
    fit_rows = np.flatnonzero(cycles <= cutoff)
    score_rows = np.flatnonzero((cycles > cutoff) & (cycles <= cutoff + horizon))
    if not len(fit_rows) or not len(score_rows):
        raise BatchIntegrityError(f"Unsupported temporal origin: cutoff={cutoff}")
    if cycles[fit_rows].max() > cutoff or cycles[score_rows].min() <= cutoff:
        raise BatchIntegrityError("Temporal row boundary failed")
    if len(np.unique(fit_rows)) != len(fit_rows) or len(np.unique(score_rows)) != len(score_rows):
        raise BatchIntegrityError("Duplicate row positions")
    return fit_rows, score_rows


def evaluation_views_for_card(queue: Mapping[str, Any], card_id: str) -> dict[str, list[tuple[int, int, int]]]:
    """Derive the canonical row views from the frozen queue profile."""
    cards = {str(card["id"]): card for card in queue["cards"]}
    if card_id not in cards:
        raise BatchIntegrityError(f"Unknown queue card: {card_id}")
    profile = str(cards[card_id]["evaluation"])
    if profile == "STANDARD":
        origins = [int(x) for x in queue["evaluation"]["STANDARD"]["pilot_cutoffs"]]
        return {
            "W1": [(origins[0], origins[0] + 1, origins[0] + 4)],
            "W4": [(origins[1], origins[1] + 1, origins[1] + 4)],
            "primary": [(origins[0], origins[0] + 1, origins[0] + 4),
                        (origins[1], origins[1] + 1, origins[1] + 4)],
            "H12": [(origins[0], origins[0] + 1, origins[0] + 12),
                    (origins[1], origins[1] + 1, origins[1] + 12)],
        }
    if profile == "LATE":
        origins = [int(x) for x in queue["evaluation"]["LATE"]["pilot_cutoffs"]]
        late_window = [int(x) for x in queue["evaluation"]["LATE"]["H12_late_window"]]
        return {
            "W4": [(origins[0], origins[0] + 1, origins[0] + 4)],
            "W6": [(origins[1], origins[1] + 1, origins[1] + 4)],
            "primary": [(origins[0], origins[0] + 1, origins[0] + 4),
                        (origins[1], origins[1] + 1, origins[1] + 4)],
            "H12_L": [(origins[0], late_window[0], late_window[1])],
        }
    if profile == "CACHE_SIX":
        origins = [int(x) for x in queue["evaluation"]["CACHE_SIX"]["primary_cutoffs"]]
        primary = [(origin, origin + 1, origin + 4) for origin in origins]
        h12_origins = [int(x) for x in queue["evaluation"]["CACHE_SIX"]["H12_cutoffs"]]
        return {
            **{f"W{index + 1}": [segment] for index, segment in enumerate(primary)},
            "primary": primary,
            "H12_E": [(h12_origins[0], h12_origins[0] + 1, h12_origins[0] + 12)],
            "H12_L": [(h12_origins[1], h12_origins[1] + 1, h12_origins[1] + 12)],
            "H12": [(h12_origins[0], h12_origins[0] + 1, h12_origins[0] + 12),
                    (h12_origins[1], h12_origins[1] + 1, h12_origins[1] + 12)],
        }
    raise BatchIntegrityError(f"Unknown evaluation profile {profile} for {card_id}")


def _score_rows_for_views(frame: pd.DataFrame, cutoff: int,
                          views: Mapping[str, Sequence[tuple[int, int, int]]]) -> np.ndarray:
    cycles = pd.to_numeric(frame["cycle_code"], errors="raise").to_numpy(dtype=int)
    allowed: set[int] = set()
    for segments in views.values():
        for origin, low, high in segments:
            if int(origin) != cutoff:
                continue
            if int(low) <= cutoff or int(high) < int(low):
                raise BatchIntegrityError("Evaluation view crosses its fit cutoff or has invalid bounds")
            allowed.update(range(int(low), int(high) + 1))
    if not allowed:
        raise BatchIntegrityError(f"No declared score cycles for cutoff {cutoff}")
    rows = np.flatnonzero(np.isin(cycles, np.array(sorted(allowed), dtype=int)))
    if not len(rows) or not set(cycles[rows]).issubset(allowed):
        raise BatchIntegrityError(f"Declared view cycles have no rows at cutoff {cutoff}")
    return rows


def metric(task: str, y_true: np.ndarray, prediction: np.ndarray) -> dict[str, Any]:
    y = np.asarray(y_true)
    p = np.asarray(prediction)
    if y.ndim != 1 or p.shape != y.shape or len(y) == 0:
        raise BatchIntegrityError("Metric arrays must be nonempty, one-dimensional, and aligned")
    if not np.isfinite(y.astype(float)).all() or not np.isfinite(p.astype(float)).all():
        raise BatchIntegrityError("Nonfinite metric inputs")
    if task == "classification":
        if not np.isin(y, [0, 1]).all() or np.any((p.astype(float) < 0) | (p.astype(float) > 1)):
            raise BatchIntegrityError("Classification metric expects binary labels and probabilities in [0,1]")
        p = (p.astype(float) >= 0.5).astype(np.int64)
        f1 = []
        for positive in (0, 1):
            tp = int(np.sum((y == positive) & (p == positive)))
            fp = int(np.sum((y != positive) & (p == positive)))
            fn = int(np.sum((y == positive) & (p != positive)))
            denom = 2 * tp + fp + fn
            f1.append(0.0 if denom == 0 else 2.0 * tp / denom)
        return {"n": int(len(y)), "macro_f1": float(sum(f1) / 2.0)}
    if task == "regression":
        return {"n": int(len(y)), "mae": float(np.mean(np.abs(p.astype(float) - y.astype(float))))}
    raise ValueError(f"Unsupported task: {task}")


def fit_only_quantile_bins(train_values: Sequence[Any], score_values: Sequence[Any],
                           *, n_bins: int = 5) -> tuple[np.ndarray, np.ndarray, list[float]]:
    """Fit equal-frequency cut points on Train only; tied cuts remain collapsed."""
    if n_bins < 1:
        raise ValueError("n_bins must be positive")
    train = pd.to_numeric(pd.Series(train_values), errors="coerce").to_numpy(dtype=float)
    score = pd.to_numeric(pd.Series(score_values), errors="coerce").to_numpy(dtype=float)
    finite = train[np.isfinite(train)]
    if not len(finite):
        raise BatchIntegrityError("Cannot fit quantile bins without finite Train support")
    edges = np.unique(np.quantile(finite, np.linspace(0.0, 1.0, n_bins + 1)))

    def transform(values: np.ndarray) -> np.ndarray:
        result = np.full(len(values), -1, dtype=np.int16)
        valid = np.isfinite(values)
        if len(edges) == 1:
            result[valid] = 0
        else:
            result[valid] = np.searchsorted(edges[1:-1], values[valid], side="left").astype(np.int16)
        return result

    return transform(train), transform(score), [float(x) for x in edges]


def _category_key(value: Any) -> str | None:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    text = str(value).strip()
    return text if text else None


def category_novelty_flags(train_categories: Sequence[Any], score_categories: Sequence[Any]) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Return deterministic vocabulary and score-time missing/unseen flags."""
    vocabulary = sorted({key for value in train_categories if (key := _category_key(value)) is not None})
    vocab = set(vocabulary)
    missing = np.array([_category_key(value) is None for value in score_categories], dtype=bool)
    unseen = np.array([(_category_key(value) is not None and _category_key(value) not in vocab)
                       for value in score_categories], dtype=bool)
    return missing, unseen, vocabulary


def stream_missing_mask(values: Sequence[Any]) -> np.ndarray:
    """Use the project rule: null, blank, or whitespace-only streams are missing."""
    output = []
    for value in values:
        if value is None or (isinstance(value, float) and np.isnan(value)):
            output.append(True)
        else:
            output.append(not str(value).strip())
    return np.asarray(output, dtype=bool)


def stable_unique_tokens(documents: Sequence[str]) -> list[str]:
    """Hash-seed-independent token identity order for deterministic adapters."""
    return sorted({token for document in documents for token in str(document).split()})


def past_cycle_smoothed_target_rates(
    train_categories: Sequence[Any], train_targets: Sequence[float], train_cycles: Sequence[int],
    score_categories: Sequence[Any], score_cycles: Sequence[int], *,
    prior: float = 0.5, smoothing: float = 30.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Encode each row from strictly earlier-cycle Train outcomes only."""
    cat = np.array([_category_key(x) for x in train_categories], dtype=object)
    y = np.asarray(train_targets, dtype=np.float64)
    cycles = np.asarray(train_cycles, dtype=np.int64)
    val_cat = np.array([_category_key(x) for x in score_categories], dtype=object)
    val_cycles = np.asarray(score_cycles, dtype=np.int64)
    if not (cat.shape == y.shape == cycles.shape) or val_cat.shape != val_cycles.shape:
        raise BatchIntegrityError("Target-rate inputs are not aligned")
    if not np.isfinite(y).all() or not np.isfinite(prior) or smoothing <= 0:
        raise BatchIntegrityError("Target-rate values or smoothing are invalid")
    train_rate = np.full(len(y), prior, dtype=np.float64)
    score_rate = np.full(len(val_cat), prior, dtype=np.float64)
    past_sum: dict[str, float] = {}
    past_n: dict[str, int] = {}
    for cycle in sorted(set(cycles.tolist()) | set(val_cycles.tolist())):
        train_mask = cycles == cycle
        score_mask = val_cycles == cycle
        for indices, keys, output in (
            (np.flatnonzero(train_mask), cat, train_rate),
            (np.flatnonzero(score_mask), val_cat, score_rate),
        ):
            for idx in indices:
                key = keys[idx]
                if key is not None and past_n.get(key, 0):
                    n = past_n[key]
                    output[idx] = (past_sum[key] + smoothing * prior) / (n + smoothing)
        for key in sorted({x for x in cat[train_mask].tolist() if x is not None}):
            selected = train_mask & (cat == key)
            past_sum[key] = past_sum.get(key, 0.0) + float(y[selected].sum(dtype=np.float64))
            past_n[key] = past_n.get(key, 0) + int(selected.sum())
    return train_rate, score_rate


def supported_group_metric(task: str, y_true: np.ndarray, prediction: np.ndarray,
                           minimum_support: int) -> dict[str, Any]:
    """Represent empty/undersupported groups explicitly, never as zero evidence."""
    y = np.asarray(y_true)
    p = np.asarray(prediction)
    n = int(len(y))
    supported = n >= int(minimum_support)
    return {"n": n, "supported": supported,
            "metric": metric(task, y, p) if supported else None}


def _run_pilot_locked(
    *, frame: pd.DataFrame, target: np.ndarray, card: Mapping[str, Any],
    feature_names: Sequence[str], adapter: Callable[..., Mapping[str, Any]],
    control_loader: Callable[[int, np.ndarray], Mapping[str, np.ndarray]],
    freeze_identity: Mapping[str, Any], freeze_path: str | Path,
    artifact_dir: str | Path, event_path: str | Path,
    evaluation_views: Mapping[str, Sequence[tuple[int, int, int]]],
    threads: int = 2, report_path: str | Path | None = None,
) -> dict[str, Any]:
    """Execute declared temporal fits only; all outcomes remain development data."""
    task = str(card["task"])
    if task not in {"classification", "regression"}:
        raise BatchIntegrityError(f"Invalid task on {card.get('id')}")
    features = list(feature_names)
    if set(features) & PROTECTED_FEATURES:
        raise BatchIntegrityError("Protected field passed as model input")
    absent = sorted(set(features) - set(frame.columns))
    if absent:
        raise BatchIntegrityError(f"Feature columns missing from Train: {absent}")
    y = np.asarray(target)
    if y.shape != (len(frame),):
        raise BatchIntegrityError("Target is not row-aligned with Train")
    if "record_id" not in frame or frame["record_id"].isna().any() or frame["record_id"].astype(str).str.strip().eq("").any():
        raise BatchIntegrityError("Train record_id is missing or blank")
    if frame["record_id"].astype(str).duplicated().any():
        raise BatchIntegrityError("Train record_id is not unique")
    cutoffs = sorted({int(origin) for view in evaluation_views.values() for origin, _, _ in view})
    if not cutoffs:
        raise BatchIntegrityError("No temporal origins supplied")
    signature = ensure_frozen(freeze_path, freeze_identity)
    card_id = str(card["id"])
    task_root = Path(artifact_dir)
    per_origin: dict[int, dict[str, Any]] = {}
    attempted_predictor_fits = 0
    for cutoff in cutoffs:
        fit_rows, _ = temporal_rows(frame, cutoff, horizon=12)
        score_rows = _score_rows_for_views(frame, cutoff, evaluation_views)
        if np.intersect1d(fit_rows, score_rows).size:
            raise BatchIntegrityError("Temporal fit and score rows overlap")
        fit_x = frame.iloc[fit_rows].loc[:, features].copy()
        score_x = frame.iloc[score_rows].loc[:, features].copy()
        controls = {k: np.asarray(v) for k, v in control_loader(cutoff, score_rows).items()}
        if "baseline_prediction" not in controls:
            raise BatchIntegrityError("Control bridge must return baseline_prediction")
        for name, values in controls.items():
            if values.shape[0] != len(score_rows):
                raise BatchIntegrityError(f"Control {name} is not aligned to score rows")
        npz_path = task_root / f"{card_id}_fit{cutoff}.npz"
        bundle_identity = {"freeze_signature": signature, "card_id": card_id, "cutoff": cutoff, "score_rows": score_rows.tolist()}
        if npz_path.exists() or npz_path.with_suffix(".json").exists():
            arrays, sidecar = read_bundle(npz_path, bundle_identity)
            if not np.array_equal(arrays.get("score_rows"), score_rows):
                raise BatchIntegrityError("Resumed bundle row positions changed")
            expected_ids = frame.iloc[score_rows]["record_id"].astype(str).to_numpy(dtype=str)
            if not np.array_equal(arrays.get("score_record_ids"), expected_ids):
                raise BatchIntegrityError("Resumed bundle record_id alignment changed")
            row = {"arrays": arrays, "metadata": sidecar["metadata"], "fit_predictor_count": int(sidecar["metadata"]["fit_predictor_count"])}
            append_event(event_path, {"at_utc": datetime.now(timezone.utc).isoformat(), "card_id": card_id, "cutoff": cutoff, "event": "RESUMED_VERIFIED", "fit_predictor_count": row["fit_predictor_count"]})
        else:
            append_event(event_path, {"at_utc": datetime.now(timezone.utc).isoformat(), "card_id": card_id, "cutoff": cutoff, "event": "FIT_STARTED", "freeze_signature": signature, "fit_rows": len(fit_rows), "score_rows": len(score_rows)})
            started = time.perf_counter()
            try:
                with threadpool_limits(limits=threads):
                    result = dict(adapter(fit_x, y[fit_rows].copy(), score_x,
                                          controls=controls, card_id=card_id, cutoff=cutoff))
                if "prediction" not in result:
                    raise BatchIntegrityError("Adapter omitted final policy prediction")
                prediction = np.asarray(result["prediction"], dtype=np.float64)
                if prediction.shape != (len(score_rows),) or not np.isfinite(prediction).all():
                    raise BatchIntegrityError("Adapter prediction is invalid or misaligned")
                if task == "classification" and (np.any((prediction < 0) | (prediction > 1))):
                    raise BatchIntegrityError("Classification adapter must return probability in [0,1]")
                if task == "regression" and (np.any((prediction < 0) | (prediction > 100))):
                    raise BatchIntegrityError("Regression adapter prediction must be clipped to [0,100]")
                score_ids = frame.iloc[score_rows]["record_id"].astype(str).to_numpy(dtype=str)
                if len(np.unique(score_ids)) != len(score_ids):
                    raise BatchIntegrityError("Score record_id values are duplicated")
                arrays = {"score_rows": score_rows, "score_record_ids": score_ids,
                          "baseline_prediction": controls["baseline_prediction"], "prediction": prediction}
                if "raw_candidate" in result:
                    raw = np.asarray(result["raw_candidate"], dtype=np.float64)
                    if raw.shape != prediction.shape or not np.isfinite(raw).all():
                        raise BatchIntegrityError("Adapter raw_candidate is invalid")
                    arrays["raw_candidate"] = raw
                for name, values in controls.items():
                    if name != "baseline_prediction":
                        arrays[f"control__{name}"] = values
                for name, values in dict(result.get("audit_arrays", {})).items():
                    arr = np.asarray(values)
                    if arr.shape[0] != len(score_rows) or arr.dtype == object:
                        raise BatchIntegrityError(f"Adapter audit array {name} invalid")
                    arrays[f"audit__{name}"] = arr
                fit_count = int(result.get("fit_predictor_count", 1))
                if fit_count < 0:
                    raise BatchIntegrityError("fit_predictor_count cannot be negative")
                metadata = {
                    "card_id": card_id, "cutoff": cutoff, "fit_predictor_count": fit_count,
                    "fit_rows": int(len(fit_rows)), "score_rows": int(len(score_rows)),
                    "fit_seconds": float(time.perf_counter() - started),
                    "fit_metadata": _jsonable(result.get("fit_metadata", {})),
                    "state": _jsonable(result.get("state", {})),
                    "adapter_independent_reconstruction_required": True,
                }
                write_bundle_exclusive(npz_path, arrays, bundle_identity, metadata)
                append_event(event_path, {"at_utc": datetime.now(timezone.utc).isoformat(), "card_id": card_id, "cutoff": cutoff, "event": "FIT_SAVED", "fit_predictor_count": fit_count, "fit_seconds": metadata["fit_seconds"]})
                row = {"arrays": arrays, "metadata": metadata, "fit_predictor_count": fit_count}
            except Exception as exc:
                append_event(event_path, {"at_utc": datetime.now(timezone.utc).isoformat(), "card_id": card_id, "cutoff": cutoff, "event": "FIT_FAILED", "error_type": type(exc).__name__, "error": str(exc)[:1000], "fit_seconds": float(time.perf_counter() - started)})
                raise
        attempted_predictor_fits += row["fit_predictor_count"]
        per_origin[cutoff] = row

    report_views: dict[str, Any] = {}
    cycles = pd.to_numeric(frame["cycle_code"], errors="raise").to_numpy(dtype=int)
    for view_name, segments in evaluation_views.items():
        view_rows: list[np.ndarray] = []
        view_predictions: list[np.ndarray] = []
        view_baselines: list[np.ndarray] = []
        for cutoff, low_cycle, high_cycle in segments:
            result = per_origin[int(cutoff)]["arrays"]
            rows = np.asarray(result["score_rows"], dtype=int)
            mask = (cycles[rows] >= int(low_cycle)) & (cycles[rows] <= int(high_cycle))
            selected = rows[mask]
            if not len(selected):
                raise BatchIntegrityError(f"View {view_name} has no rows for cutoff {cutoff}")
            view_rows.append(selected)
            view_predictions.append(np.asarray(result["prediction"])[mask])
            view_baselines.append(np.asarray(result["baseline_prediction"])[mask])
        pooled_rows = np.concatenate(view_rows)
        if len(np.unique(pooled_rows)) != len(pooled_rows):
            raise BatchIntegrityError(f"View {view_name} includes duplicate Train rows")
        prediction = np.concatenate(view_predictions)
        baseline = np.concatenate(view_baselines)
        m_candidate = metric(task, y[pooled_rows], prediction)
        m_baseline = metric(task, y[pooled_rows], baseline)
        delta = (m_candidate["macro_f1"] - m_baseline["macro_f1"]) if task == "classification" else (m_baseline["mae"] - m_candidate["mae"])
        report_views[view_name] = {"rows": int(len(pooled_rows)), "candidate": m_candidate, "baseline": m_baseline, "gain_positive_is_better": float(delta)}

    expected_fits = int(card.get("pilot_predictor_fits", attempted_predictor_fits))
    if attempted_predictor_fits != expected_fits:
        append_event(event_path, {"at_utc": datetime.now(timezone.utc).isoformat(), "card_id": card_id,
            "event": "FIT_COUNT_MISMATCH_BLOCKED", "actual": attempted_predictor_fits, "expected": expected_fits})
        raise BatchIntegrityError(f"Card fit count mismatch: actual={attempted_predictor_fits}, expected={expected_fits}")
    report = {
        "schema_version": SCHEMA_VERSION, "card_id": card_id, "task": task,
        "status": "PILOT_SAVED_AWAITING_INDEPENDENT_QA", "freeze_signature": signature,
        "cutoffs": cutoffs, "fit_predictor_count": attempted_predictor_fits,
        "expected_fit_count_from_card": expected_fits,
        "fit_count_matches_card": bool(attempted_predictor_fits == expected_fits),
        "views": report_views,
        "outcome_reuse_warning": "Temporal development outcomes are repeatedly reused; no untouched holdout claim.",
        "test_data_used": False,
    }
    if report_path is not None:
        report_path = Path(report_path)
        if report_path.exists():
            existing = json.loads(report_path.read_text(encoding="utf-8"))
            if existing != _jsonable(report):
                raise BatchIntegrityError(f"Existing final report differs: {report_path}")
        else:
            write_json_exclusive(report_path, report)
    return report


def run_pilot(*, card: Mapping[str, Any], artifact_dir: str | Path,
              budget_ledger_path: str | Path | None = None,
              max_successful_fits: int = 180, **kwargs: Any) -> dict[str, Any]:
    """Serialize candidate jobs and reserve the complete declared card budget."""
    artifact_dir = Path(artifact_dir)
    lock_path = artifact_dir.parent / ".training.lock"
    ledger_path = Path(budget_ledger_path) if budget_ledger_path is not None else artifact_dir.parent / "batch_ledger.jsonl"
    card_id = str(card["id"])
    reservation_key = f"{card_id}:PILOT"
    with exclusive_training_slot(lock_path):
        ensure_frozen(kwargs["freeze_path"], kwargs["freeze_identity"])
        reserve_card_budget(
            ledger_path, budget_key=reservation_key, card_id=card_id,
            predictor_fits=int(card.get("pilot_predictor_fits", 0)),
            cap=int(max_successful_fits),
        )
        return _run_pilot_locked(card=card, artifact_dir=artifact_dir, **kwargs)


__all__ = [
    "BatchIntegrityError", "append_event", "canonical_json", "ensure_frozen",
    "exclusive_training_slot",
    "file_sha256", "json_sha256", "load_queue", "make_freeze_identity", "metric",
    "category_novelty_flags", "evaluation_views_for_card", "fit_only_quantile_bins",
    "past_cycle_smoothed_target_rates", "read_bundle", "run_pilot", "stable_unique_tokens",
    "reserve_card_budget", "stream_missing_mask", "supported_group_metric", "temporal_rows", "write_bundle_exclusive",
    "write_json_exclusive",
]
