"""Bounded Train-only runner for RNBR_TARGET_MEDIAN_V1.

Only four memory constructions are authorized. This runner consumes donor
Regression targets after durable FIT_STARTED events, never reads query targets,
and keeps all outputs in the new target-median namespace.
"""
from __future__ import annotations

import os
for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_name] = "2"

import argparse
from datetime import datetime, timezone
import hashlib
import io
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import uuid
from typing import Any, Mapping

import numpy as np
import psutil
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import optimization_v3_rnbr_impact_match_v2 as impact_runner  # noqa: E402
from scripts import optimization_v3_rnbr_official as official  # noqa: E402
from src.models.regression_neighbor_rnbr import (  # noqa: E402
    EXPECTED_SOURCE_CUTOFFS,
    MAX_NEIGHBORS,
    MIN_NEIGHBORS,
    RnbrProtocolError,
    shuffle_residuals_within_cycle_decile,
    validate_saved_geometry,
)
from src.models.regression_neighbor_rnbr_target_median import (  # noqa: E402
    ARMS,
    MIN_PRESENT_SUPPORT_COVERAGE,
    MIN_SHUFFLE_SOURCE_MOVEMENT,
    ORIGINS,
    PILOT_ID,
    TargetMedianError,
    predict_from_target_median,
    shuffle_targets_within_cycle_decile,
)

_THREAD_LIMITER = threadpool_limits(limits=2)

PROTOCOL_PATH = ROOT / "experiments/optimization_v3/rnbr_target_median_v1/protocol.json"
BASE_FREEZE_PATH = ROOT / "experiments/optimization_v3/rnbr_pilot_v1/official_freeze.json"
BASE_RECOVERY_PATH = ROOT / "reports/optimization_v3/rnbr_pilot_v1/official_execution_v1/SERIALIZATION_RECOVERY_V2.json"
BASE_VERIFICATION_PATH = ROOT / "reports/optimization_v3/rnbr_pilot_v1/official_execution_v1/INDEPENDENT_VERIFICATION.json"
PREFLIGHT_PATH = ROOT / "reports/optimization_v3/rnbr_target_median_v1/targetfree_preflight.json"
SYNTHETIC_PATH = ROOT / "reports/optimization_v3/rnbr_target_median_v1/synthetic_qa.json"
FREEZE_PATH = ROOT / "experiments/optimization_v3/rnbr_target_median_v1/freeze.json"
REPORT_ROOT = ROOT / "reports/optimization_v3/rnbr_target_median_v1"
RUN_ROOT = ROOT / "experiments/runs/optimization_v3_research_next_v1/RNBR_TARGET_MEDIAN_V1"
LEDGER_PATH = ROOT / "experiments/runs/optimization_v3_research_next_v1/budget.jsonl"

CARD_ID = "REGRESSION_RNBR_TARGET_MEDIAN_V1"
STAGE = "FOUR_FIT_TARGET_MEDIAN_MEMORY_PILOT"
AMENDMENT_ID = "RNBR_TARGET_MEDIAN_CAP87_V1_2026_11_03"
RESERVATION_KEY = f"{CARD_ID}:{STAGE}"
FIT_IDS = tuple(f"{PILOT_ID}_E{origin}_{arm.upper()}" for origin in ORIGINS for arm in ARMS)
EXPECTED_BASE_FREEZE_SIGNATURE = "b4467f22077287c94f18e81f4b5e306307cd9d81646241f487ebf2537d3cab91"
EXPECTED_TRAIN_SHA256 = "756523ff8156a9cd776e52249f7e2a8f23dcec432b8d10570e55ff6db0e0f082"
PREVIOUS_LEDGER_SHA256 = "26745b805f7fca86b1d154e7f3f42a7004e3aec6a32c5001fa7abb505aadea04"
PREVIOUS_LEDGER_SNAPSHOT = {
    "global_cap": 83,
    "effective_category_caps": {"pilot": 42, "continuation": 38, "control_replay": 2, "technical_retry": 1},
    "active_reservations_by_category": {"pilot": 42, "continuation": 38, "control_replay": 2, "technical_retry": 1},
    "active_reservations": 83,
    "fit_starts": 83,
    "fit_saved": 80,
    "fit_failed": 3,
    "fit_pending": 0,
}
AMENDED_CAP_SNAPSHOT = {
    "global_cap": 87,
    "effective_category_caps": {"pilot": 46, "continuation": 38, "control_replay": 2, "technical_retry": 1},
}
RESOURCE_MAX_RSS_BYTES = 6 * 1024**3
RESOURCE_MIN_FREE_BYTES = 2 * 1024**3

SOURCE_PATHS = (
    "experiments/optimization_v3/rnbr_target_median_v1/protocol.json",
    "src/models/regression_neighbor_rnbr_target_median.py",
    "src/models/regression_neighbor_rnbr.py",
    "scripts/optimization_v3_rnbr_target_median_v1.py",
    "scripts/optimization_v3_rnbr_target_median_v1_verify.py",
    "scripts/optimization_v3_rnbr_target_median_v1_synthetic.py",
    "scripts/optimization_v3_rnbr_official.py",
    "scripts/optimization_v3_rnbr_impact_match_v2.py",
    "src/models/regression_rnbr_impact_match.py",
    "tests/test_optimization_v3_rnbr_target_median_v1.py",
)


class TargetMedianRunError(RuntimeError):
    """Frozen inputs, accounting, resource, or persistence invariant failed."""


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
                      allow_nan=False).encode("utf-8")


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return [_jsonable(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return _jsonable(value.item())
    if isinstance(value, Path):
        return value.as_posix()
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    return repr(value)


def _sha_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha_file(path: Path) -> str:
    return official.sha256_file(path)


def _sha_array(value: Any, dtype: str | None = None) -> str:
    return official.sha256_array(value, dtype)


def _git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def _write_once(path: Path, content: bytes) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    digest = _sha_bytes(content)
    if path.exists():
        if path.read_bytes() != content:
            raise TargetMedianRunError(f"Refusing to overwrite immutable artifact: {path}")
        return digest
    staging = path.with_name(path.name + f".{uuid.uuid4().hex}.staging")
    try:
        with staging.open("xb") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        if path.exists():
            if path.read_bytes() != content:
                raise TargetMedianRunError(f"Immutable artifact appeared with different content: {path}")
            staging.unlink()
            return digest
        os.replace(staging, path)
    finally:
        if staging.exists():
            staging.unlink()
    return digest


def _write_json_once(path: Path, payload: Mapping[str, Any]) -> str:
    return _write_once(path, (json.dumps(_jsonable(dict(payload)), indent=2,
                                         ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8"))


def _source_hashes() -> dict[str, str]:
    hashes: dict[str, str] = {}
    for relative in SOURCE_PATHS:
        path = ROOT / relative
        if not path.is_file():
            raise TargetMedianRunError(f"Required frozen source file is missing: {relative}")
        hashes[relative] = _sha_file(path)
    return hashes


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TargetMedianRunError(f"Expected JSON object: {path}")
    return value


def _load_numeric_archive(path: Path, fields: tuple[str, ...]) -> dict[str, np.ndarray]:
    if not path.is_file():
        raise TargetMedianRunError(f"Missing numeric bundle: {path}")
    result: dict[str, np.ndarray] = {}
    with np.load(path, allow_pickle=False) as archive:
        missing = set(fields) - set(archive.files)
        if missing:
            raise TargetMedianRunError(f"Bundle lacks required numeric arrays: {sorted(missing)}")
        for field in fields:
            arr = np.asarray(archive[field]).copy()
            if arr.dtype.kind == "O":
                raise TargetMedianRunError(f"Object array is prohibited in safe numeric bundle access: {field}")
            result[field] = arr
    return result


def _ledger_events(ledger_path: Path) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    if not ledger_path.is_file():
        raise TargetMedianRunError(f"Shared fit ledger is missing: {ledger_path}")
    for line_no, line in enumerate(ledger_path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError as exc:
            raise TargetMedianRunError(f"Malformed fit ledger at line {line_no}") from exc
        if not isinstance(event, dict) or not isinstance(event.get("event"), str):
            raise TargetMedianRunError(f"Invalid fit ledger event at line {line_no}")
        events.append(event)
    return events


def ledger_snapshot(events: list[dict[str, Any]]) -> dict[str, Any]:
    authorization = [event for event in events if event.get("event") == "BATCH_ALLOWANCE_AUTHORIZED"]
    if len(authorization) != 1:
        raise TargetMedianRunError("Shared Research Next ledger must contain one original authorization")
    global_cap = int(authorization[0]["cap"])
    caps = {str(key): int(value) for key, value in authorization[0]["category_caps"].items()}
    for event in events:
        if event.get("event") in {"BUDGET_CAP_AMENDMENT", "BUDGET_CATEGORY_AMENDMENT"}:
            if event.get("global_cap") is not None:
                global_cap = int(event["global_cap"])
            if event.get("effective_caps") is not None:
                caps = {str(key): int(value) for key, value in event["effective_caps"].items()}
    reservations = [event for event in events if event.get("event") == "CARD_FITS_RESERVED"]
    released = {str(event.get("reservation_key")): int(event.get("unused_fits", 0))
                for event in events if event.get("event") == "RESERVATION_UNUSED_RELEASED"}
    active = {category: 0 for category in caps}
    for event in reservations:
        category = str(event.get("category"))
        if category not in active:
            raise TargetMedianRunError(f"Unknown reserved fit category {category!r}")
        active[category] += int(event["fits"]) - released.get(str(event["reservation_key"]), 0)
    starts = [event for event in events if event.get("event") == "FIT_STARTED"]
    saved = [event for event in events if event.get("event") == "FIT_SAVED"]
    failed = [event for event in events if event.get("event") == "FIT_FAILED"]
    terminal_ids = {str(event.get("fit_id")) for event in events
                    if event.get("event") in {"FIT_SAVED", "FIT_FAILED"}}
    pending = [event for event in starts if str(event.get("fit_id")) not in terminal_ids]
    return {
        "event_rows": len(events),
        "global_cap": global_cap,
        "effective_category_caps": caps,
        "active_reservations_by_category": active,
        "active_reservations": sum(active.values()),
        "fit_starts": len(starts),
        "fit_saved": len(saved),
        "fit_failed": len(failed),
        "fit_pending": len(pending),
    }


def _append_ledger_event(ledger_path: Path, event: Mapping[str, Any]) -> dict[str, Any]:
    payload = dict(event)
    payload.setdefault("at_utc", datetime.now(timezone.utc).isoformat())
    encoded = _canonical(payload) + b"\n"
    ledger_path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(ledger_path, os.O_CREAT | os.O_APPEND | os.O_WRONLY, 0o600)
    try:
        with os.fdopen(fd, "ab", closefd=False) as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
    finally:
        os.close(fd)
    return payload


def _fit_ids_for(arm: str) -> tuple[str, ...]:
    return tuple(f"{PILOT_ID}_E{origin}_{arm.upper()}" for origin in ORIGINS)


def _validate_pre_amendment_snapshot(snapshot: Mapping[str, Any], ledger_sha256: str,
                                     expected_sha256: str) -> None:
    if ledger_sha256 != expected_sha256:
        raise TargetMedianRunError(
            f"Live ledger changed after the frozen preflight: expected {expected_sha256}, got {ledger_sha256}"
        )
    for key, expected in PREVIOUS_LEDGER_SNAPSHOT.items():
        if snapshot.get(key) != expected:
            raise TargetMedianRunError(f"Pre-amendment ledger field {key} differs: {snapshot.get(key)}")


def _apply_cap_amendment_and_reserve(
    ledger_path: Path,
    freeze: Mapping[str, Any],
    *,
    amendment_id: str = AMENDMENT_ID,
    fit_ids: tuple[str, ...] = FIT_IDS,
) -> dict[str, Any]:
    if len(fit_ids) != 4 or len(set(fit_ids)) != 4:
        raise TargetMedianRunError("The amendment must reserve exactly four unique fit IDs")
    events = _ledger_events(ledger_path)
    current_hash = _sha_file(ledger_path)
    snapshot = ledger_snapshot(events)
    amendments = [event for event in events if event.get("event") == "BUDGET_CAP_AMENDMENT"
                  and event.get("amendment_id") == amendment_id]
    if len(amendments) > 1:
        raise TargetMedianRunError("Duplicate target-median budget amendment; refusing to proceed")
    expected_amendment = {
        "event": "BUDGET_CAP_AMENDMENT",
        "batch_id": "optimization_v3_research_next_v1",
        "amendment_id": amendment_id,
        "card_id": CARD_ID,
        "authorized_by": "Samuel-authorized successor scope relayed by Sol; exactly four target-median memory fits",
        "prior_global_cap": 83,
        "global_cap": 87,
        "prior_effective_caps": {"pilot": 42, "continuation": 38, "control_replay": 2, "technical_retry": 1},
        "effective_caps": {"pilot": 46, "continuation": 38, "control_replay": 2, "technical_retry": 1},
        "active_reservations_before": {"pilot": 42, "continuation": 38, "control_replay": 2, "technical_retry": 1},
        "active_total_before": 83,
        "historical_started_saved_failed_before": {"started": 83, "saved": 80, "failed": 3},
        "pre_amendment_event_rows": 216,
        "pre_amendment_ledger_sha256": str(freeze["ledger_sha_before"]),
        "source_commit": str(freeze["source_commit"]),
        "freeze_signature": str(freeze["freeze_signature"]),
        "reason": "Four measured RNBR target-median pilot memory fits only; continuation, replay, and retry caps unchanged.",
    }
    if not amendments:
        _validate_pre_amendment_snapshot(snapshot, current_hash, str(freeze["ledger_sha_before"]))
        _append_ledger_event(ledger_path, expected_amendment)
        events = _ledger_events(ledger_path)
        snapshot = ledger_snapshot(events)
    else:
        existing = amendments[0]
        for key, expected in expected_amendment.items():
            if existing.get(key) != expected:
                raise TargetMedianRunError(f"Existing target-median cap amendment differs at {key}")
    if snapshot.get("global_cap") != 87 or snapshot.get("effective_category_caps") != AMENDED_CAP_SNAPSHOT["effective_category_caps"]:
        raise TargetMedianRunError(f"Effective cap after target-median amendment differs: {snapshot}")
    reservations = [event for event in events if event.get("event") == "CARD_FITS_RESERVED"
                    and event.get("reservation_key") == RESERVATION_KEY]
    if len(reservations) > 1:
        raise TargetMedianRunError("Duplicate target-median reservation; refusing to proceed")
    expected_reservation = {
        "event": "CARD_FITS_RESERVED",
        "batch_id": "optimization_v3_research_next_v1",
        "reservation_key": RESERVATION_KEY,
        "card_id": CARD_ID,
        "stage": STAGE,
        "category": "pilot",
        "fits": 4,
        "fit_ids": list(fit_ids),
        "amendment_id": amendment_id,
        "global_cap": 87,
        "effective_category_cap": 46,
        "freeze_signature": str(freeze["freeze_signature"]),
        "source_commit": str(freeze["source_commit"]),
    }
    if not reservations:
        prior_starts = [event for event in events if event.get("event") == "FIT_STARTED"
                        and event.get("fit_id") in set(fit_ids)]
        if prior_starts:
            raise TargetMedianRunError("Fit starts exist without the exact reservation")
        _append_ledger_event(ledger_path, expected_reservation)
        events = _ledger_events(ledger_path)
        snapshot = ledger_snapshot(events)
    else:
        existing = reservations[0]
        for key, expected in expected_reservation.items():
            if existing.get(key) != expected:
                raise TargetMedianRunError(f"Existing target-median reservation differs at {key}")
    if (snapshot.get("global_cap") != 87 or snapshot.get("active_reservations") != 87
            or snapshot.get("active_reservations_by_category") != AMENDED_CAP_SNAPSHOT["effective_category_caps"]
            or snapshot.get("fit_starts", -1) < 83 or snapshot.get("fit_starts", -1) > 87
            or snapshot.get("fit_saved", -1) < 80 or snapshot.get("fit_saved", -1) > 84
            or snapshot.get("fit_failed") != 3 or snapshot.get("fit_pending", -1) not in {0, 1}):
        raise TargetMedianRunError(f"Post-reservation ledger accounting differs: {snapshot}")
    target_starts = [event for event in events if event.get("event") == "FIT_STARTED"
                     and event.get("fit_id") in set(fit_ids)]
    target_saved = [event for event in events if event.get("event") == "FIT_SAVED"
                    and event.get("fit_id") in set(fit_ids)]
    target_failed = [event for event in events if event.get("event") == "FIT_FAILED"
                     and event.get("fit_id") in set(fit_ids)]
    target_ids = [str(event["fit_id"]) for event in target_starts]
    if (len(target_ids) != len(set(target_ids)) or len(target_saved) > len(target_starts)
            or target_failed or snapshot["fit_starts"] != 83 + len(target_starts)
            or snapshot["fit_saved"] != 80 + len(target_saved)
            or snapshot["fit_pending"] != len(target_starts) - len(target_saved)):
        raise TargetMedianRunError(f"Target-median fit accounting is inconsistent: {snapshot}")
    return {"snapshot": snapshot, "ledger_sha256": _sha_file(ledger_path),
            "amendment_id": amendment_id, "reservation_key": RESERVATION_KEY,
            "fit_ids": list(fit_ids)}


def _source_ids_for_neighbors(neighbor_rows: np.ndarray, counts: np.ndarray,
                              train_ids: np.ndarray) -> np.ndarray:
    ids = np.full(neighbor_rows.shape, "", dtype="U64")
    for i, count in enumerate(counts):
        k = int(count)
        if k:
            active = neighbor_rows[i, :k]
            if np.any((active < 0) | (active >= len(train_ids))):
                raise TargetMedianRunError("Neighbor row position is outside official Train")
            ids[i, :k] = np.asarray([f"str_:{value}" for value in train_ids[active]], dtype="U64")
    return ids


def _load_original_numeric_cache(origin: int, original: Mapping[str, Any],
                                 train: Mapping[str, np.ndarray], geometry: Mapping[str, np.ndarray]
                                 ) -> dict[str, np.ndarray]:
    path = ROOT / str(original["bundle_path"])
    fields = ("score_rows", "score_cycles", "prediction", "baseline_prediction", "query_missing",
              "neighbor_count", "neighbor_source_rows", "neighbor_similarities", "donor_rows",
              "donor_cycles", "donor_source_cutoffs")
    arrays = _load_numeric_archive(path, fields)
    rows = np.asarray(arrays["score_rows"], dtype=np.int64)
    low, high = ((68, 79) if origin == 67 else (76, 79))
    expected_rows = np.flatnonzero((train["cycles"] >= low) & (train["cycles"] <= high)).astype(np.int64)
    if (not np.array_equal(rows, expected_rows)
            or not np.array_equal(arrays["score_cycles"].astype(np.int64), train["cycles"][rows])
            or not np.array_equal(arrays["neighbor_count"].astype(np.int64), geometry["neighbor_count"])
            or not np.array_equal(arrays["neighbor_source_rows"].astype(np.int64), geometry["neighbor_source_rows"])
            or not np.array_equal(arrays["neighbor_similarities"].astype(np.float64), geometry["neighbor_similarities"])):
        raise TargetMedianRunError(f"Original R-NBR E{origin} cache does not match frozen text geometry")
    expected_donors = np.flatnonzero((train["cycles"] >= 56) & (train["cycles"] <= origin)
                                     & train["present"]).astype(np.int64)
    if (not np.array_equal(arrays["donor_rows"].astype(np.int64), expected_donors)
            or not np.array_equal(arrays["donor_cycles"].astype(np.int64), train["cycles"][expected_donors])
            or not np.array_equal(arrays["donor_source_cutoffs"].astype(np.int64),
                                  official.donor_source_cutoffs(train["cycles"][expected_donors]))):
        raise TargetMedianRunError(f"Original R-NBR E{origin} donor row identity differs from official Train")
    if (not np.isfinite(arrays["prediction"]).all() or not np.isfinite(arrays["baseline_prediction"]).all()
            or np.any((arrays["prediction"] < 0.0) | (arrays["prediction"] > 100.0))
            or np.any((arrays["baseline_prediction"] < 0.0) | (arrays["baseline_prediction"] > 100.0))):
        raise TargetMedianRunError(f"Original R-NBR E{origin} outputs are invalid")
    return arrays


def _build_targetfree_inputs() -> dict[str, Any]:
    if not PROTOCOL_PATH.is_file() or _sha_file(PROTOCOL_PATH) == "":
        raise TargetMedianRunError("New target-median protocol file is missing")
    protocol = _read_json(PROTOCOL_PATH)
    if protocol.get("protocol_id") != PILOT_ID or protocol.get("fit_count") != 4:
        raise TargetMedianRunError("Protocol identity or fit count differs")
    base_freeze = official.load_freeze(require_clean=False)
    if base_freeze.get("freeze_signature") != EXPECTED_BASE_FREEZE_SIGNATURE:
        raise TargetMedianRunError("Historical official R-NBR freeze signature differs")
    base_recovery = _read_json(BASE_RECOVERY_PATH)
    base_verification = _read_json(BASE_VERIFICATION_PATH)
    if (base_recovery.get("status") != "FOUR_FIT_PILOT_COMPLETE_DIAGNOSTIC_ONLY"
            or base_recovery.get("freeze_signature") != EXPECTED_BASE_FREEZE_SIGNATURE
            or base_verification.get("status") != "PASS"
            or base_verification.get("freeze_signature") != EXPECTED_BASE_FREEZE_SIGNATURE
            or base_verification.get("source_boundaries", {}).get("classification_labels_loaded") is not False
            or base_verification.get("source_boundaries", {}).get("test_or_sample_submission_opened") is not False):
        raise TargetMedianRunError("Original R-NBR recovery/independent verification is not PASS")
    train = official.load_targetfree_train()
    train_sha = _sha_file(official.REG_TRAIN)
    if train_sha != EXPECTED_TRAIN_SHA256:
        raise TargetMedianRunError(f"Official Regression Train hash differs: {train_sha}")
    r06 = official.load_r06_predictions(train, base_freeze)
    input_hashes: dict[str, Any] = {
        "protocol": _sha_file(PROTOCOL_PATH),
        "official_regression_train": train_sha,
        "base_rnbr_freeze": _sha_file(BASE_FREEZE_PATH),
        "base_rnbr_recovery": _sha_file(BASE_RECOVERY_PATH),
        "base_rnbr_independent_verification": _sha_file(BASE_VERIFICATION_PATH),
        "r06_cache_inventory": {},
        "a02_text_geometry_inventory": {},
        "original_rnbr_observed_bundles": {},
    }
    for cutoff_text, spec in sorted(base_freeze["r06_lineage"]["cache_inventory"].items(), key=lambda item: int(item[0])):
        input_hashes["r06_cache_inventory"][cutoff_text] = {
            "npz": str(spec["npz"]), "npz_sha256": _sha_file(ROOT / spec["npz"]),
            "sidecar": str(spec["json"]), "sidecar_sha256": _sha_file(ROOT / spec["json"]),
        }

    origins_payload: dict[str, Any] = {}
    origin_inputs: dict[int, dict[str, np.ndarray]] = {}
    geometry_hashes: dict[str, Any] = {}
    for origin in ORIGINS:
        geometry = official.load_geometry(origin, train, base_freeze)
        query_rows = geometry["score_rows"].astype(np.int64)
        query_ids = train["ids"][query_rows].astype(str)
        query_cycles = train["cycles"][query_rows].astype(np.int64)
        query_missing = ~train["present"][query_rows]
        donor_rows = np.flatnonzero((train["cycles"] >= 56) & (train["cycles"] <= origin)
                                    & train["present"]).astype(np.int64)
        donor_ids = train["ids"][donor_rows].astype(str)
        donor_cycles = train["cycles"][donor_rows].astype(np.int64)
        donor_cutoffs = official.donor_source_cutoffs(donor_cycles)
        donor_r06 = official.aligned_r06_predictions(donor_rows, donor_cycles, r06)
        base_query_lookup = {int(row): float(value) for row, value in
                             zip(r06[origin]["rows"], r06[origin]["prediction"])}
        try:
            query_baseline = np.asarray([base_query_lookup[int(row)] for row in query_rows], dtype=np.float64)
        except KeyError as exc:
            raise TargetMedianRunError(f"R06 origin E{origin} cache misses query row {exc.args[0]}") from exc
        nr = geometry["neighbor_source_rows"].astype(np.int64)
        counts = geometry["neighbor_count"].astype(np.int64)
        cosine = geometry["neighbor_similarities"].astype(np.float64)
        neighbor_ids = _source_ids_for_neighbors(nr, counts, train["ids"])
        geometry_validation = validate_saved_geometry(
            origin=origin,
            donor_rows=donor_rows, donor_ids=donor_ids, donor_cycles=donor_cycles,
            donor_source_cutoffs=donor_cutoffs, donor_present=np.ones(len(donor_rows), dtype=bool),
            query_rows=query_rows, query_ids=query_ids, query_cycles=query_cycles,
            query_missing=query_missing, neighbor_rows=nr, neighbor_ids=neighbor_ids,
            neighbor_cosines=cosine, neighbor_counts=counts,
        )
        original = impact_runner._load_original_rnbr_predictions(origin, base_freeze, train)
        old_numeric = _load_original_numeric_cache(origin, original, train, geometry)
        if (not np.array_equal(old_numeric["baseline_prediction"].astype(np.float64), query_baseline)
                or not np.array_equal(old_numeric["query_missing"].astype(bool), query_missing)):
            raise TargetMedianRunError(f"Original R-NBR E{origin} R06 baseline or missing route differs")
        fallback = query_missing | (counts < MIN_NEIGHBORS)
        if not np.array_equal(old_numeric["prediction"][fallback], query_baseline[fallback]):
            raise TargetMedianRunError(f"Original R-NBR E{origin} exact R06 fallback check failed")

        movement_source = shuffle_residuals_within_cycle_decile(
            np.arange(len(donor_rows), dtype=np.float64), donor_cycles, donor_r06,
            donor_rows, seed=42,
        ).astype(np.int64)
        moved = int(np.sum(movement_source != np.arange(len(donor_rows), dtype=np.int64)))
        movement = moved / len(donor_rows) if len(donor_rows) else 0.0

        def view_stats(name: str, low: int, high: int) -> dict[str, Any]:
            mask = (query_cycles >= low) & (query_cycles <= high)
            present = ~query_missing[mask]
            present_counts = counts[mask][present]
            n_present = int(present.sum())
            n_supported = int(np.sum(present_counts >= MIN_NEIGHBORS))
            coverage = n_supported / n_present if n_present else 0.0
            return {
                "view": name, "rows": int(mask.sum()), "present_denominator": n_present,
                "supported_present_ge5": n_supported, "coverage": coverage,
                "missing_rows_exact_r06": int(np.sum(query_missing[mask])),
                "present_rows_below_five_neighbors": int(np.sum(present_counts < MIN_NEIGHBORS)),
                "gate_coverage_ge_0_95": coverage >= MIN_PRESENT_SUPPORT_COVERAGE,
            }

        if origin == 67:
            views = {"W4": view_stats("W4", 68, 71), "H12": view_stats("H12", 68, 79)}
        else:
            views = {"W6": view_stats("W6", 76, 79)}
        if movement < MIN_SHUFFLE_SOURCE_MOVEMENT:
            raise TargetMedianRunError(f"E{origin} fixed-cycle-decile shuffle moved only {movement:.6f} of donors")
        if any(not item["gate_coverage_ge_0_95"] for item in views.values()):
            raise TargetMedianRunError(f"E{origin} original text geometry failed supported coverage: {views}")
        if any(cutoff >= cycle or EXPECTED_SOURCE_CUTOFFS.get(int(cycle)) != int(cutoff)
               for cycle, cutoff in zip(donor_cycles, donor_cutoffs)):
            raise TargetMedianRunError(f"E{origin} donor source cutoff boundary failed")

        accepted_npz_path = ROOT / original["bundle_path"]
        accepted_receipt_path = ROOT / original["receipt_path"]
        input_hashes["original_rnbr_observed_bundles"][str(origin)] = {
            "fit_id": f"RNBR_PILOT_E{origin}_OBSERVED",
            "bundle": str(original["bundle_path"]), "bundle_sha256": _sha_file(accepted_npz_path),
            "receipt": str(original["receipt_path"]), "receipt_sha256": _sha_file(accepted_receipt_path),
            "receipt_internal_sha256": original["receipt_sha256"],
        }
        for arm_name, spec in sorted(base_freeze["a02_saved_geometry"][str(origin)].items()):
            paths: dict[str, str] = {}
            for key, value in spec.items():
                if isinstance(value, str) and value.lower().endswith((".npz", ".json", ".joblib")):
                    artifact = ROOT / value
                    if not artifact.is_file():
                        raise TargetMedianRunError(f"Missing frozen A02 geometry reference {value}")
                    paths[value] = _sha_file(artifact)
            input_hashes["a02_text_geometry_inventory"].setdefault(str(origin), {})[arm_name] = paths

        geometry_hashes[str(origin)] = {
            "query_rows": _sha_array(query_rows), "query_ids_from_train_rows": _sha_array(query_ids),
            "query_cycles": _sha_array(query_cycles), "query_missing": _sha_array(query_missing),
            "donor_rows": _sha_array(donor_rows), "donor_ids_from_train_rows": _sha_array(donor_ids),
            "donor_cycles": _sha_array(donor_cycles), "donor_source_cutoffs": _sha_array(donor_cutoffs),
            "donor_r06_oof": _sha_array(donor_r06), "query_r06_origin": _sha_array(query_baseline),
            "neighbor_rows": _sha_array(nr), "neighbor_ids_from_train_rows": _sha_array(neighbor_ids),
            "neighbor_cosines": _sha_array(cosine), "neighbor_counts": _sha_array(counts),
            "original_rnbr_prediction": _sha_array(old_numeric["prediction"]),
            "original_geometry_npz_sha256": input_hashes["original_rnbr_observed_bundles"][str(origin)]["bundle_sha256"],
        }
        origins_payload[str(origin)] = {
            "donor_rows": int(len(donor_rows)), "query_rows": int(len(query_rows)),
            "query_cycle_range": [int(query_cycles.min()), int(query_cycles.max())],
            "row_identity_validation": geometry_validation,
            "neighbor_links_checked": int(np.sum(counts)),
            "views": views,
            "fixed_cycle_decile_shuffle_source_movement": {
                "moved_rows": moved, "donor_rows": int(len(donor_rows)),
                "fraction_moved": movement, "gate_ge_0_80": movement >= MIN_SHUFFLE_SOURCE_MOVEMENT,
            },
            "original_comparator_loaded": True,
            "original_exact_fallback_verified": True,
            "accepted_original_bundle": input_hashes["original_rnbr_observed_bundles"][str(origin)],
        }
        origin_inputs[origin] = {
            "donor_rows": donor_rows, "donor_ids": donor_ids, "donor_cycles": donor_cycles,
            "donor_source_cutoffs": donor_cutoffs, "donor_r06_predictions": donor_r06,
            "donor_r06_oof": donor_r06,
            "query_rows": query_rows, "query_ids": query_ids, "query_cycles": query_cycles,
            "query_missing": query_missing, "query_baseline": query_baseline,
            "neighbor_rows": nr, "neighbor_ids": neighbor_ids, "neighbor_cosines": cosine,
            "neighbor_counts": counts, "original_prediction": old_numeric["prediction"].astype(np.float64),
            "original_bundle_sha256": np.asarray([input_hashes["original_rnbr_observed_bundles"][str(origin)]["bundle_sha256"]], dtype="U64"),
        }

    input_hashes["base_r06_cache_inventory"] = input_hashes.pop("r06_cache_inventory")
    return {
        "status": "PASS_TARGET_FREE_PREFLIGHT",
        "pilot_id": PILOT_ID,
        "base_freeze_signature": base_freeze["freeze_signature"],
        "train_rows": int(len(train["rows"])),
        "train_cycle_range": [int(train["cycles"].min()), int(train["cycles"].max())],
        "present_stream_rows": int(train["present"].sum()),
        "input_hashes": input_hashes,
        "geometry_hashes": geometry_hashes,
        "origins": origins_payload,
        "budget_checkpoint": {
            "ledger_sha256": _sha_file(LEDGER_PATH),
            "snapshot": ledger_snapshot(_ledger_events(LEDGER_PATH)),
        },
        "source_boundaries": {
            "official_train_only": True, "query_target_values_read": False,
            "regression_target_values_read": False, "classification_labels_or_probabilities_read": False,
            "full_a02_model_unpickle": False, "test_or_sample_submission_opened": False,
            "kaggle_accessed": False,
        },
        "resource_snapshot": {
            "available_ram_bytes": int(psutil.virtual_memory().available),
            "current_process_rss_bytes": int(psutil.Process().memory_info().rss),
            "max_fit_rss_bytes": RESOURCE_MAX_RSS_BYTES,
            "minimum_free_ram_bytes": RESOURCE_MIN_FREE_BYTES,
        },
        "target_values_opened": False,
    }, {"train": train, "r06": r06, "origins": origin_inputs, "base_freeze": base_freeze}


def write_targetfree_preflight() -> dict[str, Any]:
    payload, _ = _build_targetfree_inputs()
    _write_json_once(PREFLIGHT_PATH, payload)
    return payload


def _source_and_input_hashes_match(freeze: Mapping[str, Any], preflight: Mapping[str, Any]) -> None:
    if _source_hashes() != freeze.get("source_hashes"):
        raise TargetMedianRunError("Frozen source code hash differs; no fit may start")
    if _sha_file(PROTOCOL_PATH) != freeze.get("protocol_sha256"):
        raise TargetMedianRunError("Frozen protocol changed; no fit may start")
    for key in ("input_hashes", "geometry_hashes"):
        if preflight.get(key) != freeze.get(key):
            raise TargetMedianRunError(f"Frozen target-free {key} changed; no fit may start")
    if preflight.get("status") != "PASS_TARGET_FREE_PREFLIGHT":
        raise TargetMedianRunError("Target-free preflight is not PASS")


def make_freeze() -> dict[str, Any]:
    preflight = _read_json(PREFLIGHT_PATH)
    synthetic = _read_json(SYNTHETIC_PATH)
    if (preflight.get("status") != "PASS_TARGET_FREE_PREFLIGHT"
            or synthetic.get("status") != "PASS_SYNTHETIC_END_TO_END"):
        raise TargetMedianRunError("Target-free preflight and synthetic lifecycle must both PASS before freeze")
    source_hashes = _source_hashes()
    expected = {
        "schema_version": 1,
        "pilot_id": PILOT_ID,
        "status": "FROZEN_BEFORE_OFFICIAL_MEMORY_FITS",
        "source_commit": _git("rev-parse", "HEAD"),
        "source_hashes": source_hashes,
        "protocol_sha256": _sha_file(PROTOCOL_PATH),
        "preflight_sha256": _sha_file(PREFLIGHT_PATH),
        "synthetic_sha256": _sha_file(SYNTHETIC_PATH),
        "base_freeze_signature": EXPECTED_BASE_FREEZE_SIGNATURE,
        "ledger_sha_before": preflight["budget_checkpoint"]["ledger_sha256"],
        "ledger_snapshot_before": preflight["budget_checkpoint"]["snapshot"],
        "input_hashes": preflight["input_hashes"],
        "geometry_hashes": preflight["geometry_hashes"],
        "origins": ORIGINS,
        "arms": ARMS,
        "fit_ids": FIT_IDS,
        "fit_count": 4,
        "candidate_formula": "clip(R06_origin + clip(0.25*(weighted_median(target_donor, cosine^2)-R06_origin),-5,+5),0,100)",
        "shuffle_formula": "same target-median functional after seed42 shuffle within cycle x fixed R06 OOF prediction decile",
        "query_targets_read_before_all_four_saved": False,
        "test_or_sample_submission_opened": False,
        "kaggle_accessed": False,
    }
    expected["freeze_signature"] = _sha_bytes(_canonical(expected))
    _write_json_once(FREEZE_PATH, expected)
    return expected


def load_freeze() -> dict[str, Any]:
    freeze = _read_json(FREEZE_PATH)
    signature = freeze.get("freeze_signature")
    body = {key: value for key, value in freeze.items() if key != "freeze_signature"}
    if signature != _sha_bytes(_canonical(body)):
        raise TargetMedianRunError("Target-median freeze signature is invalid")
    if freeze.get("status") != "FROZEN_BEFORE_OFFICIAL_MEMORY_FITS" or freeze.get("fit_ids") != list(FIT_IDS):
        raise TargetMedianRunError("Target-median freeze identity or fit IDs differ")
    preflight = _read_json(PREFLIGHT_PATH)
    synthetic = _read_json(SYNTHETIC_PATH)
    if (_sha_file(PREFLIGHT_PATH) != freeze.get("preflight_sha256")
            or _sha_file(SYNTHETIC_PATH) != freeze.get("synthetic_sha256")
            or synthetic.get("status") != "PASS_SYNTHETIC_END_TO_END"):
        raise TargetMedianRunError("Committed preparation evidence differs from freeze")
    if _source_hashes() != freeze.get("source_hashes") or _sha_file(PROTOCOL_PATH) != freeze.get("protocol_sha256"):
        raise TargetMedianRunError("Frozen source code or protocol hash changed")
    _source_and_input_hashes_match(freeze, preflight)
    return freeze


def _bundle_dir(fit_id: str, run_root: Path = RUN_ROOT) -> Path:
    if fit_id not in FIT_IDS and not fit_id.startswith("SYNTH_RNBR_TARGET_MEDIAN_V1_"):
        raise TargetMedianRunError(f"Unknown fit ID {fit_id}")
    return run_root / "bundles" / fit_id


def _array_hashes(arrays: Mapping[str, np.ndarray]) -> dict[str, str]:
    return {key: _sha_array(np.asarray(value)) for key, value in sorted(arrays.items())}


def _persist_bundle(fit_id: str, arrays: Mapping[str, np.ndarray], identity: Mapping[str, Any],
                    *, run_root: Path = RUN_ROOT) -> dict[str, Any]:
    directory = _bundle_dir(fit_id, run_root)
    npz_path, receipt_path = directory / "arrays.npz", directory / "receipt.json"
    directory.mkdir(parents=True, exist_ok=True)
    clean_arrays = {str(key): np.asarray(value) for key, value in arrays.items()}
    if any(value.dtype.kind == "O" for value in clean_arrays.values()):
        raise TargetMedianRunError("Object arrays are forbidden in target-median bundles")
    array_hashes = _array_hashes(clean_arrays)
    buffer = io.BytesIO()
    np.savez_compressed(buffer, **clean_arrays)
    npz_bytes = buffer.getvalue()
    npz_hash = _sha_bytes(npz_bytes)
    if npz_path.exists():
        if _sha_file(npz_path) != npz_hash:
            raise TargetMedianRunError(f"Refusing to overwrite a different saved fit bundle: {fit_id}")
    else:
        _write_once(npz_path, npz_bytes)
    receipt_body = {
        "schema_version": 1,
        "pilot_id": PILOT_ID,
        "fit_id": fit_id,
        "freeze_signature": identity["freeze_signature"],
        "source_commit": identity["source_commit"],
        "origin": int(identity["origin"]),
        "arm": str(identity["arm"]),
        "fit_kind": "TARGET_MEDIAN_MEMORY_CONSTRUCTION",
        "formula": identity["formula"],
        "target_read_scope": identity["target_read_scope"],
        "query_targets_read": False,
        "classification_labels_or_probabilities_read": False,
        "full_a02_model_unpickle": False,
        "test_or_sample_submission_opened": False,
        "kaggle_accessed": False,
        "npz_sha256": npz_hash,
        "array_hashes": array_hashes,
        "input_hashes": identity["input_hashes"],
        "geometry_hashes": identity["geometry_hashes"],
        "target_audit": identity["target_audit"],
        "resource_snapshot": identity["resource_snapshot"],
    }
    receipt = dict(receipt_body)
    receipt["receipt_payload_sha256"] = _sha_bytes(_canonical(receipt_body))
    receipt_bytes = (json.dumps(receipt, indent=2, ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8")
    _write_once(receipt_path, receipt_bytes)
    return {
        "receipt": receipt,
        "npz_path": npz_path,
        "receipt_path": receipt_path,
        "npz_sha256": npz_hash,
        "receipt_file_sha256": _sha_bytes(receipt_bytes),
    }


def _load_bundle(fit_id: str, freeze: Mapping[str, Any], *, run_root: Path = RUN_ROOT
                 ) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    directory = _bundle_dir(fit_id, run_root)
    npz_path, receipt_path = directory / "arrays.npz", directory / "receipt.json"
    if not npz_path.is_file() or not receipt_path.is_file():
        raise TargetMedianRunError(f"Fit {fit_id} lacks a complete durable bundle and receipt")
    receipt = _read_json(receipt_path)
    body = {key: value for key, value in receipt.items() if key != "receipt_payload_sha256"}
    if (receipt.get("receipt_payload_sha256") != _sha_bytes(_canonical(body))
            or receipt.get("npz_sha256") != _sha_file(npz_path)
            or receipt.get("freeze_signature") != freeze.get("freeze_signature")
            or receipt.get("source_commit") != freeze.get("source_commit")
            or receipt.get("fit_id") != fit_id or receipt.get("pilot_id") != PILOT_ID
            or receipt.get("query_targets_read") is not False
            or receipt.get("classification_labels_or_probabilities_read") is not False
            or receipt.get("full_a02_model_unpickle") is not False
            or receipt.get("test_or_sample_submission_opened") is not False
            or receipt.get("kaggle_accessed") is not False):
        raise TargetMedianRunError(f"Fit {fit_id} receipt or boundary attestation failed")
    arrays = _load_numeric_archive(npz_path, tuple(receipt["array_hashes"].keys()))
    if _array_hashes(arrays) != receipt["array_hashes"]:
        raise TargetMedianRunError(f"Fit {fit_id} persisted array hash mismatch")
    return arrays, receipt


def _assert_resource_budget() -> dict[str, Any]:
    own_pid = os.getpid()
    competing = []
    active_markers = (
        "optimization_v3_batch1_", "optimization_v3_research_next_n",
        "optimization_v3_final_continuation_", "optimization_v3_rnbr_official.py --run",
        "optimization_v3_rnbr_target_median_v1.py --run",
    )
    for process in psutil.process_iter(["pid", "name", "cmdline"]):
        if process.info.get("pid") == own_pid:
            continue
        command = " ".join(process.info.get("cmdline") or []).lower()
        name = str(process.info.get("name") or "").lower()
        if name.startswith("python") and any(marker in command for marker in active_markers):
            competing.append({"pid": process.info.get("pid"), "command": command[:240]})
    if competing:
        raise TargetMedianRunError(f"Another project fit process is active: {competing}")
    rss = int(psutil.Process().memory_info().rss)
    available = int(psutil.virtual_memory().available)
    if rss + 512 * 1024**2 > RESOURCE_MAX_RSS_BYTES or available < RESOURCE_MIN_FREE_BYTES:
        raise TargetMedianRunError(f"Resource budget blocked fit start: RSS={rss}, available={available}")
    return {
        "jobs": 1, "threads": 2, "process_rss_before_fit_bytes": rss,
        "available_ram_before_fit_bytes": available,
        "max_process_rss_bytes": RESOURCE_MAX_RSS_BYTES,
        "minimum_free_ram_bytes": RESOURCE_MIN_FREE_BYTES,
        "estimated_incremental_memory_bytes": 512 * 1024**2,
    }


def _fit_events(ledger_path: Path, fit_id: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    events = _ledger_events(ledger_path)
    starts = [event for event in events if event.get("event") == "FIT_STARTED" and event.get("fit_id") == fit_id]
    terminal = [event for event in events if event.get("fit_id") == fit_id
                and event.get("event") in {"FIT_SAVED", "FIT_FAILED"}]
    return starts, terminal


def _verify_memory_arrays(
    fit_id: str,
    arrays: Mapping[str, np.ndarray],
    origin_input: Mapping[str, np.ndarray],
    *,
    expected_target_actual: np.ndarray | None = None,
) -> dict[str, Any]:
    origin = int(arrays["origin"][0])
    arm = str(arrays["arm"][0])
    if fit_id != f"{PILOT_ID}_E{origin}_{arm.upper()}" or arm not in ARMS or origin not in ORIGINS:
        raise TargetMedianRunError(f"Bundle {fit_id} identity fields differ")
    for field in ("donor_rows", "donor_ids", "donor_cycles", "donor_source_cutoffs",
                  "donor_r06_oof", "query_rows", "query_ids", "query_cycles", "query_missing",
                  "query_baseline", "neighbor_rows", "neighbor_ids", "neighbor_cosines", "neighbor_counts"):
        if not np.array_equal(arrays[field], origin_input[field]):
            raise TargetMedianRunError(f"Bundle {fit_id} differs from frozen input {field}")
    target_actual = np.asarray(arrays["donor_targets_actual"], dtype=np.float64)
    target_used = np.asarray(arrays["donor_targets_used"], dtype=np.float64)
    target_source_rows = np.asarray(arrays["target_source_rows"], dtype=np.int64)
    if expected_target_actual is not None and not np.array_equal(target_actual, expected_target_actual):
        raise TargetMedianRunError(f"Bundle {fit_id} donor targets do not match official Train")
    if arm == "observed":
        expected_used = target_actual
        expected_sources = origin_input["donor_rows"]
        expected_positions = np.arange(len(target_actual), dtype=np.int64)
        if not np.array_equal(arrays["target_source_positions"], expected_positions):
            raise TargetMedianRunError(f"Bundle {fit_id} observed target positions are not identity order")
    else:
        expected_used, expected_sources, shuffle_audit = shuffle_targets_within_cycle_decile(
            target_actual, origin_input["donor_cycles"], origin_input["donor_r06_predictions"],
            origin_input["donor_rows"], seed=42,
        )
        if shuffle_audit["fraction_moved"] < MIN_SHUFFLE_SOURCE_MOVEMENT:
            raise TargetMedianRunError(f"Bundle {fit_id} shuffle source movement is below 0.80")
        if not np.array_equal(arrays["target_source_positions"], shuffle_audit["source_positions"]):
            raise TargetMedianRunError(f"Bundle {fit_id} shuffle position mapping differs")
    if not np.array_equal(target_used, expected_used) or not np.array_equal(target_source_rows, expected_sources):
        raise TargetMedianRunError(f"Bundle {fit_id} target memory/shuffle mapping differs")
    expected = predict_from_target_median(
        donor_rows=origin_input["donor_rows"], donor_targets=target_used,
        query_r06_predictions=origin_input["query_baseline"], query_missing=origin_input["query_missing"],
        neighbor_rows=origin_input["neighbor_rows"], neighbor_cosines=origin_input["neighbor_cosines"],
        neighbor_counts=origin_input["neighbor_counts"],
    )
    for field in ("weighted_median_target", "median_applied", "delta", "prediction", "neighbor_count_used"):
        if not np.array_equal(arrays[field], expected[field]):
            raise TargetMedianRunError(f"Bundle {fit_id} persisted prediction field differs: {field}")
    if np.any(arrays["query_missing"].astype(bool) & (arrays["prediction"] != arrays["query_baseline"])):
        raise TargetMedianRunError(f"Bundle {fit_id} changed a missing-stream prediction")
    return {"fit_id": fit_id, "status": "PASS", "donor_rows": int(len(target_actual)),
            "query_rows": int(len(arrays["query_rows"])),
            "supported_queries": int(np.sum(arrays["median_applied"])),
            "prediction_sha256": _sha_array(arrays["prediction"]),
            "target_memory_sha256": _sha_array(target_used)}


def _start_fit(ledger_path: Path, freeze: Mapping[str, Any], fit_id: str,
               origin: int, arm: str, resource: Mapping[str, Any], *,
               amendment_id: str = AMENDMENT_ID) -> None:
    starts, terminal = _fit_events(ledger_path, fit_id)
    if starts or terminal:
        raise TargetMedianRunError(f"Fit ID {fit_id} has an existing ledger event; resume path must verify it")
    events = _ledger_events(ledger_path)
    reservation = [event for event in events if event.get("event") == "CARD_FITS_RESERVED"
                   and event.get("reservation_key") == RESERVATION_KEY]
    if (len(reservation) != 1 or fit_id not in reservation[0].get("fit_ids", [])
            or reservation[0].get("freeze_signature") != freeze.get("freeze_signature")):
        raise TargetMedianRunError(f"Fit {fit_id} is outside the exact frozen reservation")
    snapshot = ledger_snapshot(events)
    if snapshot.get("fit_pending") != 0 or snapshot.get("fit_starts") >= snapshot.get("global_cap"):
        raise TargetMedianRunError(f"Shared ledger has a pending fit or exhausted cap: {snapshot}")
    _append_ledger_event(ledger_path, {
        "event": "FIT_STARTED", "batch_id": "optimization_v3_research_next_v1",
        "reservation_key": RESERVATION_KEY, "card_id": CARD_ID, "stage": STAGE,
        "fit_id": fit_id, "cutoff": int(origin), "component": f"{arm}_target_median_memory",
        "fit_kind": "TARGET_MEDIAN_MEMORY_CONSTRUCTION", "fit_consumed_at_start": True,
        "amendment_id": amendment_id, "freeze_signature": freeze["freeze_signature"],
        "source_commit": freeze["source_commit"], "resource_snapshot": dict(resource),
        "query_targets_read": False, "classification_labels_or_probabilities_read": False,
    })


def _finish_fit(ledger_path: Path, fit_id: str, status: str,
                *, receipt_sha256: str | None = None, error: str | None = None) -> None:
    if status not in {"FIT_SAVED", "FIT_FAILED"}:
        raise TargetMedianRunError("Invalid fit terminal event")
    starts, terminal = _fit_events(ledger_path, fit_id)
    if len(starts) != 1:
        raise TargetMedianRunError(f"Fit {fit_id} must have one durable FIT_STARTED")
    if terminal:
        if len(terminal) != 1 or terminal[0].get("event") != status:
            raise TargetMedianRunError(f"Fit {fit_id} already has a conflicting terminal event")
        if status == "FIT_SAVED" and terminal[0].get("receipt_sha256") != receipt_sha256:
            raise TargetMedianRunError(f"Fit {fit_id} saved receipt hash differs")
        return
    _append_ledger_event(ledger_path, {
        "event": status, "batch_id": "optimization_v3_research_next_v1",
        "reservation_key": RESERVATION_KEY, "card_id": CARD_ID, "stage": STAGE,
        "fit_id": fit_id, "cutoff": starts[0]["cutoff"],
        "receipt_sha256": receipt_sha256, "error": error,
    })


def _target_payload(origin_input: Mapping[str, np.ndarray], *, purpose: str) -> dict[str, np.ndarray]:
    payload = official.read_regression_targets(origin_input["donor_rows"], purpose=purpose)
    if (not np.array_equal(payload["ids"].astype(str), origin_input["donor_ids"].astype(str))
            or not np.array_equal(payload["cycles"].astype(np.int64), origin_input["donor_cycles"].astype(np.int64))):
        raise TargetMedianRunError("Regression donor target read is not aligned to frozen official Train rows")
    return payload


def _build_fit_bundle(
    fit_id: str,
    origin: int,
    arm: str,
    freeze: Mapping[str, Any],
    origin_input: Mapping[str, np.ndarray],
    *,
    target_payload: Mapping[str, np.ndarray],
    resource: Mapping[str, Any],
    run_root: Path = RUN_ROOT,
) -> dict[str, Any]:
    actual = np.asarray(target_payload["target"], dtype=np.float64)
    if arm == "observed":
        used = actual.copy()
        source_rows = origin_input["donor_rows"].copy()
        positions = np.arange(len(actual), dtype=np.int64)
        target_shuffle_audit = {"fraction_moved": 0.0, "seed": 42, "identity_arm": True}
    else:
        used, source_rows, shuffle_audit = shuffle_targets_within_cycle_decile(
            actual, origin_input["donor_cycles"], origin_input["donor_r06_predictions"],
            origin_input["donor_rows"], seed=42,
        )
        positions = np.asarray(shuffle_audit["source_positions"], dtype=np.int64)
        target_shuffle_audit = {
            "fraction_moved": float(shuffle_audit["fraction_moved"]),
            "moved_rows": int(shuffle_audit["moved_rows"]), "seed": 42,
            "fixed_decile_definition": "min(9,floor(clip(R06_OOF_donor,0,100)/10))",
        }
    predicted = predict_from_target_median(
        donor_rows=origin_input["donor_rows"], donor_targets=used,
        query_r06_predictions=origin_input["query_baseline"],
        query_missing=origin_input["query_missing"],
        neighbor_rows=origin_input["neighbor_rows"],
        neighbor_cosines=origin_input["neighbor_cosines"],
        neighbor_counts=origin_input["neighbor_counts"],
    )
    arrays: dict[str, np.ndarray] = {
        "origin": np.asarray([origin], dtype=np.int16),
        "arm": np.asarray([arm], dtype="U16"),
        "donor_rows": np.asarray(origin_input["donor_rows"], dtype=np.int64),
        "donor_ids": np.asarray(origin_input["donor_ids"], dtype="U64"),
        "donor_cycles": np.asarray(origin_input["donor_cycles"], dtype=np.int64),
        "donor_source_cutoffs": np.asarray(origin_input["donor_source_cutoffs"], dtype=np.int64),
        "donor_r06_oof": np.asarray(origin_input["donor_r06_predictions"], dtype=np.float64),
        "donor_targets_actual": actual.copy(), "donor_targets_used": np.asarray(used, dtype=np.float64),
        "target_source_rows": np.asarray(source_rows, dtype=np.int64),
        "target_source_positions": positions,
        "target_fixed_deciles": np.minimum(
            9, np.floor(np.clip(origin_input["donor_r06_predictions"], 0, 100) / 10)
        ).astype(np.int8),
        "query_rows": np.asarray(origin_input["query_rows"], dtype=np.int64),
        "query_ids": np.asarray(origin_input["query_ids"], dtype="U64"),
        "query_cycles": np.asarray(origin_input["query_cycles"], dtype=np.int64),
        "query_missing": np.asarray(origin_input["query_missing"], dtype=bool),
        "query_baseline": np.asarray(origin_input["query_baseline"], dtype=np.float64),
        "neighbor_rows": np.asarray(origin_input["neighbor_rows"], dtype=np.int64),
        "neighbor_ids": np.asarray(origin_input["neighbor_ids"], dtype="U64"),
        "neighbor_cosines": np.asarray(origin_input["neighbor_cosines"], dtype=np.float64),
        "neighbor_counts": np.asarray(origin_input["neighbor_counts"], dtype=np.int16),
        "weighted_median_target": predicted["weighted_median_target"],
        "median_applied": predicted["median_applied"],
        "delta": predicted["delta"], "prediction": predicted["prediction"],
        "neighbor_count_used": predicted["neighbor_count_used"],
    }
    actual_sha, used_sha = _sha_array(actual), _sha_array(used)
    identity = {
        "freeze_signature": freeze["freeze_signature"],
        "source_commit": freeze["source_commit"],
        "origin": origin,
        "arm": arm,
        "formula": "clip(R06_origin+clip(0.25*(weighted_median(donor threat_rank, cosine^2)-R06_origin),-5,+5),0,100)",
        "target_read_scope": "official Regression Train threat_rank on signal_stream-present donor rows cycle 56 through origin only",
        "input_hashes": freeze["input_hashes"],
        "geometry_hashes": freeze["geometry_hashes"][str(origin)],
        "target_audit": {
            "donor_target_values_count": int(len(actual)),
            "donor_target_values_sha256": actual_sha,
            "donor_targets_used_sha256": used_sha,
            "target_source_rows_sha256": _sha_array(source_rows),
            "query_target_values_read": False,
            "target_shuffle": target_shuffle_audit,
        },
        "resource_snapshot": dict(resource),
    }
    saved = _persist_bundle(fit_id, arrays, identity, run_root=run_root)
    saved["arrays"] = arrays
    return saved


def _verify_saved_bundle_against_input(fit_id: str, freeze: Mapping[str, Any],
                                       origin_input: Mapping[str, np.ndarray], *,
                                       expected_target_actual: np.ndarray | None = None,
                                       run_root: Path = RUN_ROOT) -> dict[str, Any]:
    arrays, receipt = _load_bundle(fit_id, freeze, run_root=run_root)
    if (receipt.get("input_hashes") != freeze.get("input_hashes")
            or receipt.get("geometry_hashes") != freeze["geometry_hashes"][str(int(arrays["origin"][0]))]):
        raise TargetMedianRunError(f"Fit {fit_id} lineage hash differs from freeze")
    return _verify_memory_arrays(fit_id, arrays, origin_input,
                                 expected_target_actual=expected_target_actual)


def _run_fit(ledger_path: Path, freeze: Mapping[str, Any], origin: int, arm: str,
             origin_input: Mapping[str, np.ndarray], *, run_root: Path = RUN_ROOT) -> dict[str, Any]:
    fit_id = f"{PILOT_ID}_E{origin}_{arm.upper()}"
    starts, terminal = _fit_events(ledger_path, fit_id)
    if terminal:
        if len(starts) != 1 or len(terminal) != 1:
            raise TargetMedianRunError(f"Fit {fit_id} has duplicate or incomplete terminal ledger events")
        if terminal[0]["event"] == "FIT_FAILED":
            raise TargetMedianRunError(f"Fit {fit_id} previously failed; no retry is authorized")
        donor_payload = _target_payload(origin_input, purpose=f"{PILOT_ID}_resume_verify_donor_only")
        result = _verify_saved_bundle_against_input(
            fit_id, freeze, origin_input,
            expected_target_actual=donor_payload["target"], run_root=run_root,
        )
        _finish_fit(ledger_path, fit_id, "FIT_SAVED", receipt_sha256=_sha_file(_bundle_dir(fit_id, run_root) / "receipt.json"))
        return {**result, "resumed": True}
    if starts:
        if len(starts) != 1:
            raise TargetMedianRunError(f"Fit {fit_id} has duplicate FIT_STARTED events")
        # A durable complete bundle after an interrupted process can be verified and reconciled.
        target_payload = _target_payload(origin_input, purpose=f"{PILOT_ID}_resume_verify_donor_only")
        arrays, receipt = _load_bundle(fit_id, freeze, run_root=run_root)
        result = _verify_memory_arrays(fit_id, arrays, origin_input,
                                       expected_target_actual=target_payload["target"])
        _finish_fit(ledger_path, fit_id, "FIT_SAVED", receipt_sha256=_sha_file(_bundle_dir(fit_id, run_root) / "receipt.json"))
        return {**result, "resumed": True, "reconciled_started_output": True,
                "receipt_payload_sha256": receipt["receipt_payload_sha256"]}
    resource = _assert_resource_budget()
    _start_fit(ledger_path, freeze, fit_id, origin, arm, resource)
    try:
        donor_payload = _target_payload(origin_input, purpose=f"{PILOT_ID}_{arm}_donor_only")
        saved = _build_fit_bundle(fit_id, origin, arm, freeze, origin_input,
                                  target_payload=donor_payload, resource=resource, run_root=run_root)
        result = _verify_memory_arrays(fit_id, saved["arrays"], origin_input,
                                       expected_target_actual=donor_payload["target"])
    except Exception as exc:
        _finish_fit(ledger_path, fit_id, "FIT_FAILED", error=f"{type(exc).__name__}: {str(exc)[:400]}")
        raise
    receipt_sha = _sha_file(saved["receipt_path"])
    _finish_fit(ledger_path, fit_id, "FIT_SAVED", receipt_sha256=receipt_sha)
    return {**result, "resumed": False, "receipt_file_sha256": receipt_sha,
            "npz_sha256": saved["npz_sha256"]}


def run_official() -> dict[str, Any]:
    freeze = load_freeze()
    current, context = _build_targetfree_inputs()
    _source_and_input_hashes_match(freeze, current)
    if current["budget_checkpoint"]["ledger_sha256"] != freeze["ledger_sha_before"]:
        # An existing, exact amendment/reservation is the only allowed live-ledger change on resume.
        events = _ledger_events(LEDGER_PATH)
        if not any(event.get("event") == "BUDGET_CAP_AMENDMENT"
                   and event.get("amendment_id") == AMENDMENT_ID for event in events):
            raise TargetMedianRunError("Shared ledger changed after freeze; no fit or reservation started")
    # Idempotently records cap 83->87 and reserves this exact four-fit card.
    budget = _apply_cap_amendment_and_reserve(LEDGER_PATH, freeze)
    results = []
    for origin in ORIGINS:
        for arm in ARMS:
            results.append(_run_fit(LEDGER_PATH, freeze, origin, arm,
                                    context["origins"][origin], run_root=RUN_ROOT))
    ledger_end = ledger_snapshot(_ledger_events(LEDGER_PATH))
    if (ledger_end.get("global_cap") != 87 or ledger_end.get("fit_starts") != 87
            or ledger_end.get("fit_pending") != 0):
        raise TargetMedianRunError(f"Four-fit run did not finish with exact expected ledger state: {ledger_end}")
    run_receipt = {
        "status": "FOUR_FITS_SAVED_AWAITING_INDEPENDENT_VERIFICATION",
        "pilot_id": PILOT_ID, "freeze_signature": freeze["freeze_signature"],
        "fit_results": results, "ledger_snapshot": ledger_end,
        "ledger_sha256": _sha_file(LEDGER_PATH),
        "query_target_values_read": False, "test_or_sample_submission_opened": False,
        "kaggle_accessed": False,
    }
    _write_json_once(RUN_ROOT / "run_receipt.json", run_receipt)
    verifier = ROOT / "scripts/optimization_v3_rnbr_target_median_v1_verify.py"
    completed = subprocess.run([sys.executable, str(verifier), "--verify"], cwd=ROOT,
                               text=True, capture_output=True, check=False)
    if completed.returncode != 0:
        raise TargetMedianRunError(
            f"Independent target-median verifier failed: {completed.stdout[-2000:]}\n{completed.stderr[-2000:]}"
        )
    return {"run_receipt": run_receipt, "verification_stdout": completed.stdout[-3000:],
            "budget": budget}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--preflight", action="store_true", help="Target-free integrity/support check only")
    group.add_argument("--freeze", action="store_true", help="Freeze only after preflight and synthetic QA PASS")
    group.add_argument("--run", action="store_true", help="Run exactly four authorized memory fits and verify")
    args = parser.parse_args()
    try:
        if args.preflight:
            result = write_targetfree_preflight()
        elif args.freeze:
            result = make_freeze()
        else:
            result = run_official()
        print(json.dumps(_jsonable(result), ensure_ascii=False, indent=2))
    except (TargetMedianRunError, TargetMedianError, RnbrProtocolError, official.OfficialRnbrError,
            impact_runner.ImpactMatchRunError) as exc:
        print(f"BLOCKED: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc


if __name__ == "__main__":
    main()
