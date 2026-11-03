"""Run the fixed two-origin POSITIONAL_TEXT_PILOT_V1 safely and resumably."""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
from typing import Any, Callable, Mapping

import joblib
import numpy as np
import pandas as pd
import psutil
from scipy import sparse
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.models.classification_models import (  # noqa: E402
    _ENCODER_PARAMS,
    _HGB_PARAMS,
    _LR_PARAMS,
    _TFIDF_PARAMS,
    _text_values,
)
from src.models.optimization_v3_positional_text_pilot_v1 import (  # noqa: E402
    ALL_ARMS,
    LR_ARMS,
    ORIGINS,
    PILOT_ID,
    THRESHOLD,
    _jsonable,
    availability_route,
    build_c2b_arm_matrices,
    canonical_sha256,
    fit_c1b_structured_state,
    fit_c2b_ohe,
    fit_c2b_scaler,
    fit_c2b_structured_state,
    fit_c2b_tfidf,
    fit_logistic,
    fit_missing_hgb,
    positive_probability,
    sha256_array,
    sha256_csr,
    sha256_bytes,
    structured_state_sha256,
    transform_c1b_structured,
    transform_c2b_text,
    validate_feature_frame,
)
from src.models.round1_models import EXPANDED_FEATURES  # noqa: E402


TRAIN = ROOT / "data/raw/classification/Train.csv"
TRAIN_SHA256 = "875552a68a3d24b0f3e1966d6ddb5671e6f54305a1c526e2f0a76747942ed91c"
REPORT_ROOT = ROOT / "reports/optimization_v3/positional_text_pilot_v1"
RECIPE_PATH = REPORT_ROOT / "RECIPE.json"
AUTH_PATH = ROOT / "reports/optimization_v3/positional_text_preflight_v1/SOL_PILOT_AUTHORIZATION.md"
RUN_ROOT = ROOT / "experiments/runs/optimization_v3_research_next_v1/POSITIONAL_TEXT_PILOT_V1"
LEDGER = ROOT / "experiments/runs/optimization_v3_research_next_v1/budget.jsonl"
FREEZE_PATH = RUN_ROOT / "FREEZE.json"
INITIAL_GLOBAL_CAP = 87
INITIAL_CAPS = {"pilot": 46, "continuation": 38, "control_replay": 2, "technical_retry": 1}
BASELINE_FIT_COUNTS = {"fit_starts": 87, "fit_saved": 84, "fit_failed": 3, "fit_pending": 0}
PREVIOUS_SNAPSHOT = {
    "global_cap": INITIAL_GLOBAL_CAP,
    "effective_category_caps": INITIAL_CAPS,
    "active_reservations_by_category": INITIAL_CAPS,
    "active_reservations": INITIAL_GLOBAL_CAP,
    **BASELINE_FIT_COUNTS,
}
AMENDMENT_ID = "POSITIONAL_CLASSIFICATION_PILOT_CAP105_2026_11_03"
CARD_ID = "POSITIONAL_CLASSIFICATION_PILOT_V1"
STAGE = "TWO_ORIGIN_18_UNIT_MECHANISM_PILOT"
RESERVATION_KEY = f"{CARD_ID}:{STAGE}"
AMENDED_CAPS = {"pilot": 64, "continuation": 38, "control_replay": 2, "technical_retry": 1}
CAP_AFTER = 105
FIT_IDS = tuple(
    f"{PILOT_ID}_E{origin}_{component}"
    for origin in ORIGINS
    for component in (
        "C2B_STRUCTURED", "C2B_SCALER", "C2B_OHE", "C2B_TFIDF", "C1B_STRUCTURED",
        "BAG_LR", "POSITION_LR", "SHUFFLED_POSITION_LR", "MISSING_HGB",
    )
)
UNIT_PLAN = {
    "C2B_STRUCTURED": ("PREPROCESSING", 512 * 1024**2),
    "C2B_SCALER": ("PREPROCESSING", 512 * 1024**2),
    "C2B_OHE": ("PREPROCESSING", 768 * 1024**2),
    "C2B_TFIDF": ("PREPROCESSING", 1536 * 1024**2),
    "C1B_STRUCTURED": ("PREPROCESSING", 768 * 1024**2),
    "BAG_LR": ("PREDICTOR", 1536 * 1024**2),
    "POSITION_LR": ("PREDICTOR", 2048 * 1024**2),
    "SHUFFLED_POSITION_LR": ("PREDICTOR", 2048 * 1024**2),
    "MISSING_HGB": ("PREDICTOR", 1024 * 1024**2),
}
RSS_LIMIT = 6 * 1024**3
INITIAL_FREE_MIN = 4 * 1024**3
FREE_RESERVE = 2 * 1024**3
SOURCE_PATHS = (
    "scripts/optimization_v3_positional_text_pilot_v1.py",
    "scripts/optimization_v3_positional_text_pilot_v1_verify.py",
    "scripts/optimization_v3_positional_text_pilot_v1_synthetic.py",
    "src/models/optimization_v3_positional_text_pilot_v1.py",
    "src/models/positional_text_features.py",
    "src/models/classification_models.py",
    "src/models/round1_models.py",
    "tests/test_optimization_v3_positional_text_pilot_v1.py",
    "reports/optimization_v3/positional_text_pilot_v1/RECIPE.json",
    "reports/optimization_v3/positional_text_pilot_v1/REPORT.md",
    "reports/optimization_v3/positional_text_pilot_v1/SOL_PILOT_AUTHORIZATION.md",
    "reports/optimization_v3/positional_text_pilot_v1/INTEGRITY_INCIDENT.md",
    "reports/optimization_v3/positional_text_preflight_v1/REPORT.md",
)
FEATURE_COLUMNS = list(dict.fromkeys(["record_id", "cycle_code", *EXPANDED_FEATURES, "signal_stream"]))


class PilotError(RuntimeError):
    pass


def _sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True, encoding="utf-8").strip()


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(_jsonable(value), sort_keys=True, ensure_ascii=False, indent=2,
                         allow_nan=False).encode("utf-8") + b"\n"
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _atomic_joblib(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    os.close(fd)
    temporary = Path(temporary_name)
    try:
        joblib.dump(value, temporary, compress=0)
        with temporary.open("rb+") as stream:
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_sparse(path: Path, matrix: sparse.spmatrix) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as stream:
        sparse.save_npz(stream, sparse.csr_matrix(matrix), compressed=True)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _atomic_npz(path: Path, **arrays: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _canonical_line(event: Mapping[str, Any]) -> bytes:
    payload = dict(event)
    payload.setdefault("at_utc", datetime.now(timezone.utc).isoformat())
    return json.dumps(_jsonable(payload), sort_keys=True, ensure_ascii=False,
                      allow_nan=False, separators=(",", ":")).encode("utf-8") + b"\n"


def read_ledger(*, path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise PilotError(f"Shared fit ledger is missing: {path}")
    events = []
    for line_no, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not raw.strip():
            continue
        event = json.loads(raw)
        if not isinstance(event, dict) or not isinstance(event.get("event"), str):
            raise PilotError(f"Invalid ledger event on line {line_no}")
        events.append(event)
    return events


def ledger_snapshot(events: list[dict[str, Any]]) -> dict[str, Any]:
    authorizations = [e for e in events if e.get("event") == "BATCH_ALLOWANCE_AUTHORIZED"]
    if len(authorizations) != 1:
        raise PilotError("Expected exactly one Research Next base authorization")
    global_cap = int(authorizations[0]["cap"])
    caps = {str(k): int(v) for k, v in authorizations[0]["category_caps"].items()}
    for event in events:
        if event.get("event") == "BUDGET_CAP_AMENDMENT":
            global_cap = int(event["global_cap"])
            caps = {str(k): int(v) for k, v in event["effective_caps"].items()}
        elif event.get("event") == "BUDGET_CATEGORY_AMENDMENT":
            if event.get("global_cap") is not None:
                global_cap = int(event["global_cap"])
            if event.get("effective_caps") is not None:
                caps = {str(k): int(v) for k, v in event["effective_caps"].items()}
    released = {str(e["reservation_key"]): int(e.get("unused_fits", 0))
                for e in events if e.get("event") == "RESERVATION_UNUSED_RELEASED"}
    active = {key: 0 for key in caps}
    for event in events:
        if event.get("event") == "CARD_FITS_RESERVED":
            category = str(event.get("category"))
            if category not in active:
                raise PilotError(f"Unknown ledger reservation category: {category}")
            active[category] += int(event["fits"]) - released.get(str(event["reservation_key"]), 0)
    starts = [e for e in events if e.get("event") == "FIT_STARTED"]
    saves = [e for e in events if e.get("event") == "FIT_SAVED"]
    failures = [e for e in events if e.get("event") == "FIT_FAILED"]
    terminal_ids = {str(e.get("fit_id")) for e in events
                    if e.get("event") in {"FIT_SAVED", "FIT_FAILED"}}
    pending = [e for e in starts if str(e.get("fit_id")) not in terminal_ids]
    return {
        "event_rows": len(events), "global_cap": global_cap,
        "effective_category_caps": caps, "active_reservations_by_category": active,
        "active_reservations": sum(active.values()), "fit_starts": len(starts),
        "fit_saved": len(saves), "fit_failed": len(failures), "fit_pending": len(pending),
    }


def append_ledger(event: Mapping[str, Any], *, path: Path) -> dict[str, Any]:
    return _append_official_ledger(event, path=path)


def _append_official_ledger(event: Mapping[str, Any], *, path: Path) -> dict[str, Any]:
    payload = dict(event)
    payload.setdefault("at_utc", datetime.now(timezone.utc).isoformat())
    encoded = _canonical_line(payload)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_CREAT | os.O_APPEND | os.O_WRONLY, 0o600)
    try:
        with os.fdopen(fd, "ab", closefd=False) as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
    finally:
        os.close(fd)
    return payload


def resource_snapshot(estimate_bytes: int = 0, *, require_initial_free: bool = False) -> dict[str, int]:
    process = psutil.Process()
    rss = int(process.memory_info().rss)
    available = int(psutil.virtual_memory().available)
    projected_rss = rss + int(estimate_bytes)
    projected_free = available - int(estimate_bytes)
    result = {
        "jobs": 1, "threads": 2, "process_rss_bytes": rss,
        "available_ram_bytes": available,
        "estimated_incremental_memory_bytes": int(estimate_bytes),
        "projected_rss_bytes": projected_rss,
        "projected_available_ram_bytes": projected_free,
        "max_rss_bytes": RSS_LIMIT,
        "initial_available_ram_min_bytes": INITIAL_FREE_MIN,
        "minimum_free_ram_bytes": FREE_RESERVE,
    }
    if require_initial_free and available < INITIAL_FREE_MIN:
        raise PilotError(f"Initial free RAM below 4 GiB: {available}")
    if projected_rss > RSS_LIMIT:
        raise PilotError(f"Projected process RSS exceeds 6 GiB: {projected_rss}")
    if projected_free < FREE_RESERVE:
        raise PilotError(f"Projected free RAM below 2 GiB reserve: {projected_free}")
    return result


def source_hashes() -> dict[str, str]:
    missing = [rel for rel in SOURCE_PATHS if not (ROOT / rel).is_file()]
    if missing:
        raise PilotError(f"Required source files missing: {missing}")
    return {rel: _sha_file(ROOT / rel) for rel in SOURCE_PATHS}


def load_features() -> pd.DataFrame:
    if _sha_file(TRAIN) != TRAIN_SHA256:
        raise PilotError("Official Classification Train SHA-256 differs from frozen source")
    frame = pd.read_csv(TRAIN, usecols=FEATURE_COLUMNS, dtype={"record_id": "string"})
    if (len(frame) != 50_608 or frame["record_id"].isna().any()
            or frame["record_id"].duplicated().any()):
        raise PilotError("Official Classification Train row/record_id integrity failed")
    frame["cycle_code"] = pd.to_numeric(frame["cycle_code"], errors="raise").astype(np.int64)
    if set(frame["cycle_code"].unique()) - set(range(47, 80)):
        raise PilotError("Unexpected cycle_code outside official train cycles 47-79")
    return frame.reset_index(drop=True)


def _fold_rows(frame: pd.DataFrame, origin: int) -> tuple[np.ndarray, np.ndarray, int, int]:
    cycles = frame["cycle_code"].to_numpy(dtype=np.int64)
    fit_rows = np.flatnonzero((cycles >= 47) & (cycles <= origin)).astype(np.int64)
    score_low, score_high = ((68, 79) if origin == 67 else (76, 79))
    score_rows = np.flatnonzero((cycles >= score_low) & (cycles <= score_high)).astype(np.int64)
    expected = {67: (29_705, 20_903), 75: (42_117, 8_491)}[origin]
    if (len(fit_rows), len(score_rows)) != expected:
        raise PilotError(f"Unexpected E{origin} fit/score rows: {(len(fit_rows), len(score_rows))}")
    return fit_rows, score_rows, score_low, score_high


def target_values_for_rows(rows: np.ndarray, expected_cycles: np.ndarray, *, purpose: str) -> np.ndarray:
    """Parse only requested label cells; callers enforce fit/query ordering."""
    if purpose not in {"fit-prefix", "outer-score-after-predictions"}:
        raise PilotError(f"Invalid target-read purpose: {purpose}")
    requested = np.asarray(rows, dtype=np.int64)
    cycles_expected = np.asarray(expected_cycles, dtype=np.int64)
    if requested.ndim != 1 or cycles_expected.shape != requested.shape or len(set(requested.tolist())) != len(requested):
        raise PilotError("Requested target rows/cycles are invalid or duplicated")
    wanted = set(requested.tolist())
    found: dict[int, int] = {}
    with TRAIN.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.reader(stream)
        header = next(reader)
        cycle_col = header.index("cycle_code")
        target_col = header.index("artifact_signal")
        for row_number, record in enumerate(reader):
            if row_number not in wanted:
                continue
            cycle = int(record[cycle_col])
            if cycle != int(cycles_expected[len(found)]):
                raise PilotError("Target source row cycle does not align to requested feature row")
            value = int(record[target_col])
            if value not in (0, 1):
                raise PilotError("artifact_signal must be integer 0/1")
            found[row_number] = value
    if set(found) != wanted:
        raise PilotError("Target source did not return every requested official Train row")
    return np.asarray([found[int(row)] for row in requested], dtype=np.int8)


def _fit_row_metadata(frame: pd.DataFrame, rows: np.ndarray) -> dict[str, Any]:
    return {
        "rows": int(len(rows)),
        "row_indices_sha256": sha256_array(rows),
        "record_ids_sha256": sha256_array(frame.iloc[rows]["record_id"].astype(str).to_numpy()),
        "cycles_sha256": sha256_array(frame.iloc[rows]["cycle_code"].to_numpy(dtype=np.int64)),
        "cycle_min": int(frame.iloc[rows]["cycle_code"].min()),
        "cycle_max": int(frame.iloc[rows]["cycle_code"].max()),
    }


def _text_load_q75(frame: pd.DataFrame) -> float:
    values = pd.to_numeric(frame["text_load"], errors="coerce").to_numpy(dtype=np.float64)
    finite = values[np.isfinite(values)]
    if not len(finite):
        raise PilotError("No finite fit-prefix text_load values to define Q4")
    return float(np.quantile(finite, 0.75, method="linear"))


def _category_encoder_sha(preprocessor: Any) -> str:
    encoder = preprocessor.encoder
    values = [[str(value) for value in categories] for categories in encoder.categories_]
    return canonical_sha256(values)


def _id_for(origin: int, component: str) -> str:
    value = f"{PILOT_ID}_E{origin}_{component}"
    if value not in FIT_IDS:
        raise PilotError(f"Unregistered fit unit {value}")
    return value


def _unit_paths(fit_id: str) -> tuple[Path, Path]:
    safe = fit_id.replace("/", "_")
    return RUN_ROOT / "units" / f"{safe}.joblib", RUN_ROOT / "receipts" / f"{safe}.json"


def _matrix_manifest_path(origin: int) -> Path:
    return RUN_ROOT / f"MATRIX_MANIFEST_E{origin}.json"


def _prediction_verification_path(origin: int) -> Path:
    return RUN_ROOT / f"PREDICTION_VERIFICATION_E{origin}.json"


def _preflight_path() -> Path:
    return RUN_ROOT / "PREFIT_RESOURCE_PREFLIGHT.json"


def _readiness_path() -> Path:
    return RUN_ROOT / "PREDICTIONS_READY.json"


def _results_path() -> Path:
    return RUN_ROOT / "RESULTS.json"


def _without_event_rows(snapshot: Mapping[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in snapshot.items() if key != "event_rows"}


def _fit_events(fit_id: str, events: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    starts = [e for e in events if e.get("event") == "FIT_STARTED" and e.get("fit_id") == fit_id]
    terminal = [e for e in events if e.get("event") in {"FIT_SAVED", "FIT_FAILED"}
                and e.get("fit_id") == fit_id]
    return starts, terminal


def _check_expected_budget(events: list[dict[str, Any]], *, before_unit: bool) -> None:
    snapshot = ledger_snapshot(events)
    amendment = [e for e in events if e.get("event") == "BUDGET_CAP_AMENDMENT"
                 and e.get("amendment_id") == AMENDMENT_ID]
    reservation = [e for e in events if e.get("event") == "CARD_FITS_RESERVED"
                   and e.get("reservation_key") == RESERVATION_KEY]
    if len(amendment) != 1 or len(reservation) != 1:
        raise PilotError("Exact positional-pilot budget amendment/reservation missing")
    starts, saves, failed = [], [], []
    target_set = set(FIT_IDS)
    for event in events:
        if event.get("fit_id") not in target_set:
            continue
        if event.get("event") == "FIT_STARTED":
            starts.append(event)
        elif event.get("event") == "FIT_SAVED":
            saves.append(event)
        elif event.get("event") == "FIT_FAILED":
            failed.append(event)
    if failed:
        raise PilotError("A positional-pilot fit unit has failed; retries are prohibited")
    if len(starts) != len({e["fit_id"] for e in starts}) or len(saves) != len({e["fit_id"] for e in saves}):
        raise PilotError("Duplicate positional-pilot fit events detected")
    expected_starts = BASELINE_FIT_COUNTS["fit_starts"] + len(starts)
    expected_saves = BASELINE_FIT_COUNTS["fit_saved"] + len(saves)
    expected_pending = len(starts) - len(saves)
    if (snapshot["global_cap"] != CAP_AFTER or snapshot["effective_category_caps"] != AMENDED_CAPS
            or snapshot["active_reservations_by_category"] != AMENDED_CAPS
            or snapshot["active_reservations"] != CAP_AFTER
            or snapshot["fit_starts"] != expected_starts
            or snapshot["fit_saved"] != expected_saves
            or snapshot["fit_failed"] != BASELINE_FIT_COUNTS["fit_failed"]
            or snapshot["fit_pending"] != expected_pending):
        raise PilotError(f"Shared ledger accounting changed unexpectedly: {snapshot}")
    if before_unit and snapshot["fit_pending"] != 0:
        raise PilotError("Shared ledger has a pending fit; do not start another fit")


def reserve_pilot(freeze: Mapping[str, Any]) -> dict[str, Any]:
    events = read_ledger(path=LEDGER)
    current_sha = _sha_file(LEDGER)
    snapshot = ledger_snapshot(events)
    amendments = [e for e in events if e.get("event") == "BUDGET_CAP_AMENDMENT"
                  and e.get("amendment_id") == AMENDMENT_ID]
    reservations = [e for e in events if e.get("event") == "CARD_FITS_RESERVED"
                    and e.get("reservation_key") == RESERVATION_KEY]
    if len(amendments) > 1 or len(reservations) > 1:
        raise PilotError("Duplicate positional pilot amendment or reservation")
    if not amendments and not reservations:
        if (current_sha != freeze["pre_reservation_ledger_sha256"]
                or snapshot != freeze["pre_reservation_ledger_snapshot"]
                or _without_event_rows(snapshot) != PREVIOUS_SNAPSHOT):
            raise PilotError("Live ledger changed since frozen prefit snapshot; refuse reservation")
        amendment = {
            "event": "BUDGET_CAP_AMENDMENT", "batch_id": "optimization_v3_research_next_v1",
            "amendment_id": AMENDMENT_ID, "card_id": CARD_ID,
            "authorized_by": "Sam via Sol; exactly 18 positional Classification pilot fit units",
            "prior_global_cap": INITIAL_GLOBAL_CAP, "global_cap": CAP_AFTER,
            "prior_effective_caps": dict(INITIAL_CAPS),
            "effective_caps": AMENDED_CAPS,
            "active_reservations_before": dict(INITIAL_CAPS),
            "active_total_before": sum(INITIAL_CAPS.values()),
            "historical_started_saved_failed_before": {
                "started": BASELINE_FIT_COUNTS["fit_starts"],
                "saved": BASELINE_FIT_COUNTS["fit_saved"],
                "failed": BASELINE_FIT_COUNTS["fit_failed"],
            },
            "pre_amendment_event_rows": len(events),
            "pre_amendment_ledger_sha256": current_sha,
            "source_commit": freeze["source_commit"],
            "freeze_signature": freeze["freeze_signature"],
            "reason": "Authorized two-origin positional Classification pilot; 10 preprocessing and 8 predictor fits; all other category caps unchanged.",
        }
        append_ledger(amendment, path=LEDGER)
        events = read_ledger(path=LEDGER)
        snapshot = ledger_snapshot(events)
    expected_amendment = {
        "global_cap": CAP_AFTER,
        "effective_caps": AMENDED_CAPS,
        "prior_global_cap": INITIAL_GLOBAL_CAP,
        "prior_effective_caps": INITIAL_CAPS,
        "freeze_signature": freeze["freeze_signature"],
        "source_commit": freeze["source_commit"],
        "card_id": CARD_ID,
    }
    if amendments:
        existing = amendments[0]
        if any(existing.get(key) != value for key, value in expected_amendment.items()):
            raise PilotError("Existing budget amendment does not match frozen positional pilot")
    else:
        amendments = [e for e in events if e.get("event") == "BUDGET_CAP_AMENDMENT"
                      and e.get("amendment_id") == AMENDMENT_ID]
        if len(amendments) != 1 or any(amendments[0].get(k) != v for k, v in expected_amendment.items()):
            raise PilotError("Budget amendment failed independent post-append verification")
    reservation_body = {
        "event": "CARD_FITS_RESERVED", "batch_id": "optimization_v3_research_next_v1",
        "reservation_key": RESERVATION_KEY, "card_id": CARD_ID, "stage": STAGE,
        "category": "pilot", "fits": len(FIT_IDS), "fit_ids": list(FIT_IDS),
        "amendment_id": AMENDMENT_ID, "global_cap": CAP_AFTER,
        "effective_category_cap": AMENDED_CAPS["pilot"],
        "freeze_signature": freeze["freeze_signature"], "source_commit": freeze["source_commit"],
    }
    if not reservations:
        wanted = {
            "global_cap": CAP_AFTER, "effective_category_caps": AMENDED_CAPS,
            "active_reservations_by_category": INITIAL_CAPS,
            "active_reservations": sum(INITIAL_CAPS.values()),
            **BASELINE_FIT_COUNTS,
        }
        if _without_event_rows(snapshot) != wanted:
            raise PilotError(f"Post-amendment pre-reservation budget differs: {snapshot}")
        append_ledger(reservation_body, path=LEDGER)
        events = read_ledger(path=LEDGER)
    else:
        if any(reservations[0].get(k) != v for k, v in reservation_body.items()):
            raise PilotError("Existing positional pilot reservation differs from the frozen 18-unit plan")
    reservation_events = [e for e in events if e.get("event") == "CARD_FITS_RESERVED"
                          and e.get("reservation_key") == RESERVATION_KEY]
    if len(reservation_events) != 1 or any(reservation_events[0].get(k) != v
                                          for k, v in reservation_body.items()):
        raise PilotError("Post-reservation ledger verification failed")
    snapshot = ledger_snapshot(events)
    expected_snapshot = {
        "global_cap": CAP_AFTER, "effective_category_caps": AMENDED_CAPS,
        "active_reservations_by_category": AMENDED_CAPS, "active_reservations": CAP_AFTER,
        **BASELINE_FIT_COUNTS,
    }
    if _without_event_rows(snapshot) != expected_snapshot:
        raise PilotError(f"Ledger after 18-unit reservation differs: {snapshot}")
    return {"reservation": reservation_events[0], "snapshot": snapshot,
            "ledger_sha256": _sha_file(LEDGER)}


def _start_fit(freeze: Mapping[str, Any], origin: int, component: str,
               fit_kind: str, estimate: int, unit_metadata: Mapping[str, Any]) -> dict[str, Any]:
    fit_id = _id_for(origin, component)
    events = read_ledger(path=LEDGER)
    _check_expected_budget(events, before_unit=True)
    starts, terminal = _fit_events(fit_id, events)
    if starts or terminal:
        raise PilotError(f"Fit unit {fit_id} has an existing non-resumable start; no retry")
    reservations = [e for e in events if e.get("event") == "CARD_FITS_RESERVED"
                    and e.get("reservation_key") == RESERVATION_KEY]
    if len(reservations) != 1 or fit_id not in reservations[0]["fit_ids"]:
        raise PilotError(f"Fit unit {fit_id} is not in the exact reservation")
    resource = resource_snapshot(
        estimate,
        require_initial_free=ledger_snapshot(events)["fit_starts"] == BASELINE_FIT_COUNTS["fit_starts"],
    )
    event = {
        "event": "FIT_STARTED", "batch_id": "optimization_v3_research_next_v1",
        "reservation_key": RESERVATION_KEY, "card_id": CARD_ID, "stage": STAGE,
        "fit_id": fit_id, "cutoff": int(origin), "component": component.lower(),
        "fit_kind": fit_kind, "fit_consumed_at_start": True,
        "amendment_id": AMENDMENT_ID, "freeze_signature": freeze["freeze_signature"],
        "source_commit": freeze["source_commit"], "resource_snapshot": resource,
        "fit_population": dict(unit_metadata),
    }
    append_ledger(event, path=LEDGER)
    return {"fit_id": fit_id, "started_at_utc": datetime.now(timezone.utc).isoformat(),
            "resource_snapshot": resource}


def _finish_fit(fit_id: str, receipt_sha256: str) -> None:
    events = read_ledger(path=LEDGER)
    starts, terminal = _fit_events(fit_id, events)
    if len(starts) != 1 or terminal:
        raise PilotError(f"Cannot save fit unit {fit_id}: start/terminal accounting mismatch")
    append_ledger({
        "event": "FIT_SAVED", "batch_id": "optimization_v3_research_next_v1",
        "reservation_key": RESERVATION_KEY, "card_id": CARD_ID, "stage": STAGE,
        "fit_id": fit_id, "cutoff": int(starts[0]["cutoff"]),
        "fit_kind": starts[0]["fit_kind"], "receipt_sha256": receipt_sha256,
        "amendment_id": AMENDMENT_ID,
    }, path=LEDGER)
    _check_expected_budget(read_ledger(path=LEDGER), before_unit=False)


def _run_unit(freeze: Mapping[str, Any], origin: int, component: str,
              unit_metadata: Mapping[str, Any], fit_func: Callable[[], tuple[Any, dict[str, Any]]]) -> tuple[Any, dict[str, Any]]:
    fit_id = _id_for(origin, component)
    artifact_path, receipt_path = _unit_paths(fit_id)
    fit_kind, estimate = UNIT_PLAN[component]
    events = read_ledger(path=LEDGER)
    starts, terminal = _fit_events(fit_id, events)
    if terminal:
        if len(starts) != 1 or len(terminal) != 1 or terminal[0]["event"] != "FIT_SAVED":
            raise PilotError(f"Fit unit {fit_id} is failed or inconsistent; no retry")
        if not artifact_path.is_file() or not receipt_path.is_file():
            raise PilotError(f"Saved fit unit {fit_id} is missing its artifact/receipt")
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        if (receipt.get("fit_id") != fit_id or receipt.get("freeze_signature") != freeze["freeze_signature"]
                or receipt.get("artifact_sha256") != _sha_file(artifact_path)
                or terminal[0].get("receipt_sha256") != _sha_file(receipt_path)):
            raise PilotError(f"Saved fit unit {fit_id} artifact or receipt hash mismatch")
        return joblib.load(artifact_path), receipt
    if starts:
        raise PilotError(f"Fit unit {fit_id} was started but not saved; retries are prohibited")
    started = _start_fit(freeze, origin, component, fit_kind, estimate, unit_metadata)
    start_time = time.perf_counter()
    try:
        value, fit_metadata = fit_func()
        _atomic_joblib(artifact_path, value)
        after = resource_snapshot(0)
        receipt = {
            "schema_version": 1, "fit_id": fit_id, "pilot_id": PILOT_ID,
            "origin": origin, "component": component, "fit_kind": fit_kind,
            "freeze_signature": freeze["freeze_signature"],
            "source_commit": freeze["source_commit"],
            "source_hashes": freeze["source_hashes"],
            "input_sha256": freeze["input_sha256"],
            "fit_population": dict(unit_metadata),
            "resource_before": started["resource_snapshot"], "resource_after": after,
            "elapsed_seconds": float(time.perf_counter() - start_time),
            "artifact_path": artifact_path.relative_to(ROOT).as_posix(),
            "artifact_sha256": _sha_file(artifact_path),
            "fit_metadata": fit_metadata,
            "convergence_status": "PASS_NO_CONVERGENCE_WARNING" if fit_kind == "PREDICTOR" else "NOT_APPLICABLE",
        }
        _atomic_json(receipt_path, receipt)
        if receipt["artifact_sha256"] != _sha_file(artifact_path):
            raise PilotError(f"Artifact hash changed while saving {fit_id}")
        receipt_sha = _sha_file(receipt_path)
        _finish_fit(fit_id, receipt_sha)
        return value, receipt
    except Exception as exc:
        events_now = read_ledger(path=LEDGER)
        current_starts, current_terminal = _fit_events(fit_id, events_now)
        if current_starts and not current_terminal:
            append_ledger({
                "event": "FIT_FAILED", "batch_id": "optimization_v3_research_next_v1",
                "reservation_key": RESERVATION_KEY, "card_id": CARD_ID, "stage": STAGE,
                "fit_id": fit_id, "cutoff": int(origin), "fit_kind": fit_kind,
                "amendment_id": AMENDMENT_ID, "error": f"{type(exc).__name__}: {exc}",
            }, path=LEDGER)
        raise


def _fit_metadata(frame: pd.DataFrame, fit_rows: np.ndarray) -> dict[str, Any]:
    return {**_fit_row_metadata(frame, fit_rows), "features_sha256": sha256_csr(sparse.csr_matrix((0, 0))) if False else None}


def _save_matrix_set(origin: int, freeze: Mapping[str, Any], frame: pd.DataFrame,
                     fit_rows: np.ndarray, score_rows: np.ndarray,
                     matrices: Mapping[str, Any], hgb_fit: np.ndarray, hgb_score: np.ndarray,
                     preprocessing_sha256: Mapping[str, str], q75: float) -> dict[str, Any]:
    matrix_dir = RUN_ROOT / f"E{origin}" / "matrices"
    manifest_path = _matrix_manifest_path(origin)
    metadata = {
        "schema_version": 1, "pilot_id": PILOT_ID, "origin": origin,
        "project_root": str(ROOT),
        "freeze_signature": freeze["freeze_signature"], "input_sha256": freeze["input_sha256"],
        "fit_row_metadata": _fit_row_metadata(frame, fit_rows),
        "score_row_metadata": _fit_row_metadata(frame, score_rows),
        "text_load_q75_fit_only": float(q75),
        "shared_preprocessing_sha256": dict(preprocessing_sha256),
        "position_audit": matrices["position_audit"],
        "matrices": {},
    }
    matrix_objects: dict[str, Any] = {k: matrices[k] for k in (
        "fit_base", "score_base", "fit_position", "score_position",
        "fit_shuffled_position", "score_shuffled_position",
    )}
    matrix_objects["fit_c1b"] = np.asarray(hgb_fit, dtype=np.float64)
    matrix_objects["score_c1b"] = np.asarray(hgb_score, dtype=np.float64)
    for name, matrix in matrix_objects.items():
        if sparse.issparse(matrix):
            if not np.isfinite(matrix.data).all():
                raise PilotError(f"Nonfinite values in {name}")
            path = matrix_dir / f"{name}.npz"
            logical_hash = sha256_csr(matrix)
            shape = list(matrix.shape)
            _atomic_sparse(path, matrix)
        else:
            if not np.isfinite(matrix).all():
                raise PilotError(f"Nonfinite values in {name}")
            path = matrix_dir / f"{name}.npy"
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary = path.with_suffix(".npy.tmp")
            with temporary.open("wb") as stream:
                np.save(stream, matrix, allow_pickle=False)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, path)
            logical_hash = sha256_array(matrix)
            shape = list(matrix.shape)
        metadata["matrices"][name] = {
            "path": path.relative_to(ROOT).as_posix(), "sha256": _sha_file(path),
            "logical_sha256": logical_hash, "shape": shape,
            "dtype": str(matrix.dtype if sparse.issparse(matrix) else matrix.dtype),
        }
    metadata["matrix_set_sha256"] = canonical_sha256(metadata["matrices"])
    _atomic_json(manifest_path, metadata)
    return metadata


def _load_matrix_set(manifest: Mapping[str, Any]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    root = Path(str(manifest.get("project_root", ROOT))).resolve()
    for name, item in manifest["matrices"].items():
        path = Path(item["path"])
        if not path.is_absolute():
            path = root / path
        if not path.is_file() or _sha_file(path) != item["sha256"]:
            raise PilotError(f"Matrix cache file hash mismatch for {name}")
        if path.suffix == ".npz":
            matrix = sparse.load_npz(path).tocsr()
            logical = sha256_csr(matrix)
        else:
            matrix = np.load(path, allow_pickle=False)
            logical = sha256_array(matrix)
        if list(matrix.shape) != item["shape"] or logical != item["logical_sha256"]:
            raise PilotError(f"Matrix cache shape/content mismatch for {name}")
        output[name] = matrix
    if canonical_sha256(manifest["matrices"]) != manifest["matrix_set_sha256"]:
        raise PilotError("Matrix manifest aggregate hash mismatch")
    return output


def _load_or_build_matrices(origin: int, freeze: Mapping[str, Any], frame: pd.DataFrame,
                            fit_rows: np.ndarray, score_rows: np.ndarray,
                            c2b_struct: Mapping[str, Any], scaler: Any, ohe: Any, vectorizer: Any,
                            c1b_struct: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    manifest_path = _matrix_manifest_path(origin)
    expected_shared = {
        "structured_state": structured_state_sha256(c2b_struct),
        "scaler_mean": sha256_array(scaler.mean_),
        "scaler_scale": sha256_array(scaler.scale_),
        "ohe_categories": canonical_sha256([[str(value) for value in values]
                                             for values in ohe.categories_]),
        "tfidf_vocabulary_idf": canonical_sha256({
            "vocabulary": sorted((str(key), int(value)) for key, value in vectorizer.vocabulary_.items()),
            "idf_sha256": sha256_array(vectorizer.idf_),
        }),
        "c1b_structured_state": structured_state_sha256(c1b_struct),
    }
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (manifest.get("freeze_signature") != freeze["freeze_signature"]
                or manifest.get("input_sha256") != freeze["input_sha256"]
                or manifest.get("fit_row_metadata") != _fit_row_metadata(frame, fit_rows)
                or manifest.get("score_row_metadata") != _fit_row_metadata(frame, score_rows)
                or manifest.get("shared_preprocessing_sha256") != expected_shared):
            raise PilotError("Existing matrix manifest differs from frozen data rows")
        loaded = _load_matrix_set(manifest)
        expected_shapes = {
            "fit_base": (len(fit_rows), None), "score_base": (len(score_rows), None),
            "fit_position": (len(fit_rows), None), "score_position": (len(score_rows), None),
            "fit_shuffled_position": (len(fit_rows), None),
            "score_shuffled_position": (len(score_rows), None),
            "fit_c1b": (len(fit_rows), None), "score_c1b": (len(score_rows), None),
        }
        if set(loaded) != set(expected_shapes):
            raise PilotError("Cached matrix set has unexpected or missing matrices")
        for name, matrix in loaded.items():
            rows_expected = expected_shapes[name][0]
            if matrix.shape[0] != rows_expected:
                raise PilotError(f"Cached matrix row count mismatch for {name}")
        if (loaded["fit_position"].shape != loaded["fit_shuffled_position"].shape
                or loaded["score_position"].shape != loaded["score_shuffled_position"].shape):
            raise PilotError("Cached observed/shuffled position matrices have different shapes")
        return loaded, manifest

    fit_frame = validate_feature_frame(frame.iloc[fit_rows])
    score_frame = validate_feature_frame(frame.iloc[score_rows])
    matrices = build_c2b_arm_matrices(c2b_struct, scaler, ohe, vectorizer, fit_frame, score_frame)
    hgb_fit = transform_c1b_structured(c1b_struct, fit_frame)
    hgb_score = transform_c1b_structured(c1b_struct, score_frame)
    q75 = float(c2b_struct["text_load_q75_fit_only"])
    shared = {**matrices["shared_preprocessing_sha256"],
              "c1b_structured_state": expected_shared["c1b_structured_state"]}
    manifest = _save_matrix_set(origin, freeze, frame, fit_rows, score_rows,
                                matrices, hgb_fit, hgb_score, shared, q75)
    return _load_matrix_set(manifest), manifest


def _train_labels(frame: pd.DataFrame, fit_rows: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    cycles = frame.iloc[fit_rows]["cycle_code"].to_numpy(dtype=np.int64)
    target = target_values_for_rows(fit_rows, cycles, purpose="fit-prefix")
    meta = {"target_read_purpose": "fit-prefix", "fit_target_sha256": sha256_array(target),
            "positive_count": int(target.sum()), "negative_count": int((target == 0).sum())}
    return target, meta


def _verify_prediction_bundle(frame: pd.DataFrame, origin: int, path: Path,
                              matrix_manifest: Mapping[str, Any]) -> dict[str, Any]:
    if not path.is_file():
        raise PilotError(f"Prediction bundle is missing: {path}")
    fit_rows, expected_score_rows, score_low, score_high = _fold_rows(frame, origin)
    with np.load(path, allow_pickle=False) as archive:
        content = {key: archive[key] for key in archive.files}
    score_ids = frame.iloc[expected_score_rows]["record_id"].astype(str).to_numpy()
    score_cycles = frame.iloc[expected_score_rows]["cycle_code"].to_numpy(dtype=np.int64)
    if (not np.array_equal(content.get("score_rows"), expected_score_rows)
            or not np.array_equal(content.get("score_record_ids"), score_ids)
            or not np.array_equal(content.get("score_cycles"), score_cycles)):
        raise PilotError(f"Prediction row/ID/cycle alignment failed for E{origin}")
    if content.get("target") is not None or any("label" in key.casefold() or key == "y" for key in content):
        raise PilotError("Prediction bundle must not contain score labels")
    present = ~_text_values(frame.iloc[expected_score_rows]["signal_stream"])[1]
    for arm in (*LR_ARMS, "missing_hgb", "route_bag", "route_position", "route_shuffled_position"):
        key = f"prob_{arm}"
        if key not in content:
            raise PilotError(f"Prediction bundle missing {key}")
        values = np.asarray(content[key], dtype=np.float64)
        if values.shape != (len(expected_score_rows),) or not np.isfinite(values).all() or np.any((values < 0) | (values > 1)):
            raise PilotError(f"Invalid probability vector in {key}")
    for name in ("bag", "position", "shuffled_position"):
        route = content[f"prob_route_{name}"]
        if not np.array_equal(route[~present], content["prob_missing_hgb"][~present]):
            raise PilotError(f"Missing-stream fallback differs for route {name}")
    if content.get("matrix_manifest_sha256") is None or str(content["matrix_manifest_sha256"].item()) != _sha_file(
        _matrix_manifest_path(origin)
    ):
        raise PilotError("Prediction bundle matrix manifest hash mismatch")
    return {
        "status": "PASS_PREDICTION_INTEGRITY",
        "origin": origin,
        "score_rows": len(expected_score_rows),
        "score_cycle_min": score_low,
        "score_cycle_max": score_high,
        "present_rows": int(present.sum()),
        "missing_rows": int((~present).sum()),
        "prediction_file_sha256": _sha_file(path),
        "matrix_manifest_sha256": _sha_file(_matrix_manifest_path(origin)),
        "fit_rows_hash": sha256_array(fit_rows),
        "score_rows_hash": sha256_array(expected_score_rows),
    }


def _fit_origin(freeze: Mapping[str, Any], origin: int, frame: pd.DataFrame) -> dict[str, Any]:
    fit_rows, score_rows, score_low, score_high = _fold_rows(frame, origin)
    fit_frame = validate_feature_frame(frame.iloc[fit_rows])
    score_frame = validate_feature_frame(frame.iloc[score_rows])
    fit_population = _fit_row_metadata(frame, fit_rows)
    y_fit, target_meta = _train_labels(frame, fit_rows)
    if len(np.unique(y_fit)) != 2:
        raise PilotError(f"E{origin} fit prefix lacks both target classes")

    def prep(component: str, fn: Callable[[], Any], metadata: Mapping[str, Any]):
        fit_id = _id_for(origin, component)
        if any(e.get("event") == "FIT_SAVED" and e.get("fit_id") == fit_id
               for e in read_ledger(path=LEDGER)):
            value, receipt = _run_unit(freeze, origin, component, metadata, lambda: (None, {}))
            return value
        value, _ = _run_unit(freeze, origin, component, metadata, lambda: fn())
        return value

    def fit_c2b_structured_unit():
        state = fit_c2b_structured_state(fit_frame)
        state["text_load_q75_fit_only"] = _text_load_q75(fit_frame)
        audit = state["preprocessor_audit_state"]
        return state, {
            "underlying_fit_calls": ["OrdinalEncoder.fit x1 (nested in HGBPreprocessor.fit_transform)"],
            "learned_numeric_medians": audit["numeric_imputer_medians"],
            "numeric_medians_sha256": canonical_sha256(audit["numeric_imputer_medians"]),
            "learned_category_counts": audit["category_counts"],
            "ordinal_encoder_categories_sha256": _category_encoder_sha(state["preprocessor"]),
            "preprocessor_audit_sha256": canonical_sha256(audit),
            "fit_matrix_sha256": state["fit_matrix_sha256"],
            "fit_defined_text_load_q75": state["text_load_q75_fit_only"],
            "fit_rows": int(len(fit_frame)),
        }

    c2b_struct = prep("C2B_STRUCTURED", fit_c2b_structured_unit, fit_population)

    def fit_scaler_unit():
        value = fit_c2b_scaler(c2b_struct, fit_frame)
        return value, {"underlying_fit_calls": ["StandardScaler.fit x1"],
                       "mean_sha256": sha256_array(value.mean_),
                       "scale_sha256": sha256_array(value.scale_),
                       "feature_count": int(len(value.mean_))}

    scaler = prep("C2B_SCALER", fit_scaler_unit, fit_population)

    def fit_ohe_unit():
        value = fit_c2b_ohe(c2b_struct, fit_frame)
        categories = [[str(item) for item in values] for values in value.categories_]
        infrequent = getattr(value, "infrequent_categories_", None)
        infrequent_payload = [None if values is None else [str(item) for item in values]
                              for values in infrequent]
        return value, {"underlying_fit_calls": ["OneHotEncoder.fit x1"],
                       "categories_sha256": canonical_sha256(categories),
                       "infrequent_categories_sha256": canonical_sha256(infrequent_payload),
                       "category_counts": [len(values) for values in value.categories_]}

    ohe = prep("C2B_OHE", fit_ohe_unit, fit_population)

    def fit_tfidf_unit():
        value = fit_c2b_tfidf(fit_frame)
        vocabulary = sorted((str(token), int(index)) for token, index in value.vocabulary_.items())
        return value, {"underlying_fit_calls": ["TfidfVectorizer.fit x1"],
                       "vocabulary_size": len(vocabulary),
                       "vocabulary_sha256": canonical_sha256(vocabulary),
                       "idf_sha256": sha256_array(value.idf_),
                       "feature_count": int(len(value.idf_))}

    vectorizer = prep("C2B_TFIDF", fit_tfidf_unit, fit_population)

    def fit_c1b_structured_unit():
        state = fit_c1b_structured_state(fit_frame)
        audit = state["preprocessor_audit_state"]
        return state, {
            "underlying_fit_calls": ["OrdinalEncoder.fit x1 (nested in HGBPreprocessor.fit_transform)"],
            "learned_numeric_medians": audit["numeric_imputer_medians"],
            "numeric_medians_sha256": canonical_sha256(audit["numeric_imputer_medians"]),
            "learned_category_counts": audit["category_counts"],
            "ordinal_encoder_categories_sha256": _category_encoder_sha(state["preprocessor"]),
            "preprocessor_audit_sha256": canonical_sha256(audit),
            "categorical_mask": state["categorical_mask"],
            "fit_matrix_sha256": state["fit_matrix_sha256"],
            "fit_rows": int(len(fit_frame)),
        }

    c1b_struct = prep("C1B_STRUCTURED", fit_c1b_structured_unit, fit_population)
    matrices, matrix_manifest = _load_or_build_matrices(
        origin, freeze, frame, fit_rows, score_rows, c2b_struct, scaler, ohe, vectorizer, c1b_struct
    )
    y_meta = {**fit_population, **target_meta, "matrix_manifest_sha256": _sha_file(
        _matrix_manifest_path(origin)
    )}
    arms = {
        "bag": ("fit_base", "score_base", "BAG_LR"),
        "position": ("fit_position", "score_position", "POSITION_LR"),
        "shuffled_position": ("fit_shuffled_position", "score_shuffled_position", "SHUFFLED_POSITION_LR"),
    }
    model_probs: dict[str, np.ndarray] = {}
    for arm, (fit_key, score_key, component) in arms.items():
        model_value = prep(component, lambda fit_key=fit_key, score_key=score_key, arm=arm: _fit_lr_unit(
            matrices[fit_key], matrices[score_key], y_fit
        ), y_meta)
        model_probs[arm] = np.asarray(model_value["probability"], dtype=np.float64)

    hgb_value = prep("MISSING_HGB", lambda: _fit_hgb_unit(
        c1b_struct, matrices["fit_c1b"], matrices["score_c1b"], y_fit
    ), y_meta)
    model_probs["missing_hgb"] = np.asarray(hgb_value["probability"], dtype=np.float64)
    present = ~_text_values(score_frame["signal_stream"])[1]
    routes = availability_route(present, model_probs["bag"], model_probs["position"],
                                model_probs["shuffled_position"], model_probs["missing_hgb"])
    prediction_path = RUN_ROOT / f"E{origin}" / f"predictions_E{origin}.npz"
    score_ids = frame.iloc[score_rows]["record_id"].astype(str).to_numpy(dtype=str)
    score_cycles = frame.iloc[score_rows]["cycle_code"].to_numpy(dtype=np.int64)
    payload: dict[str, Any] = {
        "origin": np.asarray([origin], dtype=np.int64),
        "fit_rows": fit_rows,
        "score_rows": score_rows,
        "score_record_ids": score_ids,
        "score_cycles": score_cycles,
        "present_mask": present,
        "matrix_manifest_sha256": np.asarray([_sha_file(_matrix_manifest_path(origin))]),
    }
    for arm, probability in model_probs.items():
        payload[f"prob_{arm}"] = probability
    for arm, probability in routes.items():
        payload[f"prob_route_{arm}"] = probability
    _atomic_npz(prediction_path, **payload)
    verify = _verify_prediction_bundle(frame, origin, prediction_path, matrix_manifest)
    verification_path = _prediction_verification_path(origin)
    _atomic_json(verification_path, verify)
    return {
        "origin": origin,
        "prediction_path": prediction_path.relative_to(ROOT).as_posix(),
        "prediction_sha256": _sha_file(prediction_path),
        "prediction_verification_path": verification_path.relative_to(ROOT).as_posix(),
        "prediction_verification_sha256": _sha_file(verification_path),
        "matrix_manifest_path": _matrix_manifest_path(origin).relative_to(ROOT).as_posix(),
        "matrix_manifest_sha256": _sha_file(_matrix_manifest_path(origin)),
        "score_rows": int(len(score_rows)), "score_cycles": [score_low, score_high],
        "status": "PREDICTIONS_DURABLE_AND_VERIFIED_NO_SCORE_LABELS_OPENED",
    }


def _fit_lr_unit(fit_matrix: Any, score_matrix: Any, y_fit: np.ndarray) -> tuple[dict[str, Any], dict[str, Any]]:
    with threadpool_limits(limits=2):
        estimator, metadata = fit_logistic(fit_matrix, y_fit)
        probability = positive_probability(estimator, score_matrix)
    return {"estimator": estimator, "probability": probability}, {
        **metadata, "fit_matrix_sha256": sha256_csr(fit_matrix),
        "score_matrix_sha256": sha256_csr(score_matrix),
        "predicted_rows": int(len(probability)), "probability_sha256": sha256_array(probability),
        "preprocessing_reused_from": "same-origin shared C2_B structured/scaler/OHE/TFIDF artifacts",
    }


def _fit_hgb_unit(state: Mapping[str, Any], fit_matrix: np.ndarray,
                  score_matrix: np.ndarray, y_fit: np.ndarray) -> tuple[dict[str, Any], dict[str, Any]]:
    with threadpool_limits(limits=2):
        estimator, metadata = fit_missing_hgb(state, fit_matrix, y_fit)
        probability = positive_probability(estimator, score_matrix)
    return {"estimator": estimator, "probability": probability}, {
        **metadata, "fit_matrix_sha256": sha256_array(fit_matrix),
        "score_matrix_sha256": sha256_array(score_matrix),
        "predicted_rows": int(len(probability)), "probability_sha256": sha256_array(probability),
    }


def create_target_free_preflight() -> dict[str, Any]:
    if _git("status", "--porcelain"):
        raise PilotError("Preflight/freeze requires a clean committed source tree")
    frame = load_features()
    folds = {}
    for origin in ORIGINS:
        fit_rows, score_rows, low, high = _fold_rows(frame, origin)
        folds[str(origin)] = {
            "fit_rows": len(fit_rows), "fit_cycle_min_max": [47, origin],
            "fit_ids_sha256": sha256_array(frame.iloc[fit_rows]["record_id"].astype(str).to_numpy()),
            "score_rows": len(score_rows), "score_cycle_min_max": [low, high],
            "score_ids_sha256": sha256_array(frame.iloc[score_rows]["record_id"].astype(str).to_numpy()),
        }
    events = read_ledger(path=LEDGER)
    snapshot = ledger_snapshot(events)
    if _without_event_rows(snapshot) != PREVIOUS_SNAPSHOT:
        raise PilotError(f"Live ledger differs from exact expected 87-start checkpoint: {snapshot}")
    resource = resource_snapshot(0, require_initial_free=True)
    result = {
        "status": "PASS_TARGET_FREE_PREFIT_RESOURCE_PREFLIGHT",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_commit": _git("rev-parse", "HEAD"),
        "source_hashes": source_hashes(),
        "input_path": TRAIN.relative_to(ROOT).as_posix(),
        "input_sha256": _sha_file(TRAIN), "rows": len(frame),
        "columns_read": FEATURE_COLUMNS,
        "target_column_read": False, "test_or_sample_submission_opened": False,
        "cycle_range": [int(frame.cycle_code.min()), int(frame.cycle_code.max())],
        "folds": folds,
        "ledger_sha256": _sha_file(LEDGER), "ledger_snapshot": snapshot,
        "resource_preflight": resource,
        "python_version": sys.version,
        "numpy_version": np.__version__, "pandas_version": pd.__version__,
        "sklearn_version": __import__("sklearn").__version__,
        "joblib_version": joblib.__version__,
        "fit_plan": {"total_units": 18, "preprocessing_units": 10, "predictor_fits": 8},
    }
    path = _preflight_path()
    _atomic_json(path, result)
    return result


def create_freeze() -> dict[str, Any]:
    preflight_path = _preflight_path()
    if not preflight_path.is_file():
        raise PilotError("Run --preflight after committing source code before freeze")
    preflight = json.loads(preflight_path.read_text(encoding="utf-8"))
    if preflight.get("status") != "PASS_TARGET_FREE_PREFIT_RESOURCE_PREFLIGHT":
        raise PilotError("Target-free prefit/resource preflight is not PASS")
    if _git("status", "--porcelain"):
        raise PilotError("Freeze requires a clean committed source tree")
    hashes = source_hashes()
    if hashes != preflight["source_hashes"]:
        raise PilotError("Source hash changed after target-free preflight")
    if _sha_file(TRAIN) != TRAIN_SHA256 or _sha_file(LEDGER) != preflight["ledger_sha256"]:
        raise PilotError("Input or live ledger changed after target-free preflight")
    if _without_event_rows(ledger_snapshot(read_ledger(path=LEDGER))) != PREVIOUS_SNAPSHOT:
        raise PilotError("Live ledger no longer matches prefit snapshot")
    recipe = json.loads(RECIPE_PATH.read_text(encoding="utf-8"))
    payload: dict[str, Any] = {
        "schema_version": 1, "pilot_id": PILOT_ID,
        "source_commit": preflight["source_commit"],
        "source_hashes": hashes,
        "recipe_path": RECIPE_PATH.relative_to(ROOT).as_posix(),
        "recipe_sha256": _sha_file(RECIPE_PATH),
        "input_path": TRAIN.relative_to(ROOT).as_posix(),
        "input_sha256": _sha_file(TRAIN),
        "preflight_path": preflight_path.relative_to(ROOT).as_posix(),
        "preflight_sha256": _sha_file(preflight_path),
        "pre_reservation_ledger_sha256": _sha_file(LEDGER),
        "pre_reservation_ledger_snapshot": ledger_snapshot(read_ledger(path=LEDGER)),
        "fit_ids": list(FIT_IDS), "fit_plan": recipe["shared_preprocessing"],
        "resource_policy": recipe["resource_policy"],
        "test_or_sample_submission_opened": False,
        "score_labels_opened": False,
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    payload["freeze_signature"] = canonical_sha256(payload)
    RUN_ROOT.mkdir(parents=True, exist_ok=True)
    _atomic_json(FREEZE_PATH, payload)
    return payload


def load_freeze() -> dict[str, Any]:
    if not FREEZE_PATH.is_file():
        raise PilotError("Frozen pilot manifest is missing; run --freeze before any reservation or fit")
    freeze = json.loads(FREEZE_PATH.read_text(encoding="utf-8"))
    signature = freeze.pop("freeze_signature", None)
    if signature != canonical_sha256(freeze):
        raise PilotError("Freeze signature mismatch")
    freeze["freeze_signature"] = signature
    if freeze["pilot_id"] != PILOT_ID or freeze["input_sha256"] != TRAIN_SHA256:
        raise PilotError("Freeze belongs to a different pilot or source input")
    if source_hashes() != freeze["source_hashes"] or _sha_file(TRAIN) != freeze["input_sha256"]:
        raise PilotError("Frozen code/input hash changed")
    if _git("status", "--porcelain"):
        raise PilotError("Working tree must remain clean while frozen fits run")
    return freeze


def run_predictions() -> dict[str, Any]:
    RUN_ROOT.mkdir(parents=True, exist_ok=True)
    lock_path = RUN_ROOT / "RUNNING.lock"
    try:
        fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError as exc:
        raise PilotError(f"Another positional pilot run or an unresolved run lock exists: {lock_path}") from exc
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            stream.write(json.dumps({"pid": os.getpid(), "started_at_utc": datetime.now(timezone.utc).isoformat()}))
            stream.flush()
            os.fsync(stream.fileno())
        return _run_predictions_under_lock()
    finally:
        lock_path.unlink(missing_ok=True)


def _run_predictions_under_lock() -> dict[str, Any]:
    freeze = load_freeze()
    resource_snapshot(0, require_initial_free=True)
    reservation = reserve_pilot(freeze)
    frame = load_features()
    if len(frame) != 50_608:
        raise PilotError("Official Train features changed after freeze")
    origins = []
    for origin in ORIGINS:
        origins.append(_fit_origin(freeze, origin, frame))
    events = read_ledger(path=LEDGER)
    _check_expected_budget(events, before_unit=False)
    snapshot = ledger_snapshot(events)
    expected_final = {
        "fit_starts": BASELINE_FIT_COUNTS["fit_starts"] + len(FIT_IDS),
        "fit_saved": BASELINE_FIT_COUNTS["fit_saved"] + len(FIT_IDS),
        "fit_failed": BASELINE_FIT_COUNTS["fit_failed"],
        "fit_pending": 0,
    }
    if any(snapshot[key] != value for key, value in expected_final.items()):
        raise PilotError(f"Expected all 18 reserved units saved: {snapshot}")
    result = {
        "status": "PREDICTIONS_READY_FOR_SCORE_LABEL_OPENING",
        "origin_predictions": origins,
        "ledger_reservation": reservation,
        "ledger_snapshot": snapshot,
        "ledger_sha256": _sha_file(LEDGER),
        "query_labels_opened": False,
    }
    _atomic_json(_readiness_path(), result)
    return result


def score_predictions() -> dict[str, Any]:
    freeze = load_freeze()
    frame = load_features()
    readiness_path = _readiness_path()
    if not readiness_path.is_file():
        raise PilotError("Both origins must have durable predictions before score labels can be read")
    readiness = json.loads(readiness_path.read_text(encoding="utf-8"))
    if readiness.get("status") != "PREDICTIONS_READY_FOR_SCORE_LABEL_OPENING" or readiness.get("query_labels_opened") is not False:
        raise PilotError("Prediction readiness report is invalid")
    events = read_ledger(path=LEDGER)
    _check_expected_budget(events, before_unit=False)
    snapshot = ledger_snapshot(events)
    expected_final = {
        "fit_starts": BASELINE_FIT_COUNTS["fit_starts"] + len(FIT_IDS),
        "fit_saved": BASELINE_FIT_COUNTS["fit_saved"] + len(FIT_IDS),
        "fit_failed": BASELINE_FIT_COUNTS["fit_failed"],
        "fit_pending": 0,
    }
    if (any(snapshot[key] != value for key, value in expected_final.items())
            or readiness.get("ledger_sha256") != _sha_file(LEDGER)):
        raise PilotError("Completed pilot ledger/readiness receipt failed before score-label access")
    saved_fit_ids = {event.get("fit_id") for event in events if event.get("event") == "FIT_SAVED"}
    if not set(FIT_IDS).issubset(saved_fit_ids):
        raise PilotError("One or more reserved fit units are not durably saved")
    for origin in ORIGINS:
        manifest_path = _matrix_manifest_path(origin)
        prediction_path = RUN_ROOT / f"E{origin}" / f"predictions_E{origin}.npz"
        if not manifest_path.is_file() or not prediction_path.is_file():
            raise PilotError("Missing matrix or prediction output before query label access")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        _load_matrix_set(manifest)
        verification = _verify_prediction_bundle(frame, origin, prediction_path, manifest)
        saved_verification = _prediction_verification_path(origin)
        if (not saved_verification.is_file()
                or json.loads(saved_verification.read_text(encoding="utf-8")) != verification):
            raise PilotError(f"Independent prediction verification no longer matches E{origin}")
    from scripts.optimization_v3_positional_text_pilot_v1_verify import (
        score_origin,
        verify_metrics_independently,
        verify_prediction_bundle,
    )

    independent_prelabel = []
    for origin in ORIGINS:
        prediction_path = RUN_ROOT / f"E{origin}" / f"predictions_E{origin}.npz"
        manifest_path = _matrix_manifest_path(origin)
        independent_prelabel.append(verify_prediction_bundle(
            frame, origin, prediction_path, manifest_path
        ))
    if len(independent_prelabel) != len(ORIGINS) or any(
        item.get("status") != "PASS_INDEPENDENT_PREDICTION_INTEGRITY"
        or item.get("query_labels_read") is not False
        or item.get("fit_or_transform_fitted") is not False
        for item in independent_prelabel
    ):
        raise PilotError("Independent prelabel verification failed; do not open score labels")

    results = []
    for origin in ORIGINS:
        _, score_rows, low, high = _fold_rows(frame, origin)
        score_cycles = frame.iloc[score_rows]["cycle_code"].to_numpy(dtype=np.int64)
        y_score = target_values_for_rows(score_rows, score_cycles,
                                        purpose="outer-score-after-predictions")
        prediction_path = RUN_ROOT / f"E{origin}" / f"predictions_E{origin}.npz"
        q75 = float(json.loads(_matrix_manifest_path(origin).read_text(
            encoding="utf-8"))["text_load_q75_fit_only"])
        result = score_origin(frame, origin, prediction_path, y_score, low, high, fit_text_q75=q75)
        independent = verify_metrics_independently(result, frame, origin, prediction_path, y_score, q75)
        results.append({"result": result, "independent_metric_verification": independent})
    result = {
        "status": "SCORED_AFTER_ALL_ORIGIN_PREDICTIONS_VERIFIED",
        "pilot_id": PILOT_ID, "freeze_signature": freeze["freeze_signature"],
        "origins": results,
        "independent_prelabel_verification": independent_prelabel,
        "query_label_reads": {
            "E67": [68, 79], "E75": [76, 79],
            "opened_after_all_predictions_durable": True,
        },
        "ledger_snapshot": ledger_snapshot(read_ledger(path=LEDGER)),
        "ledger_sha256": _sha_file(LEDGER),
    }
    _atomic_json(_results_path(), result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--preflight", action="store_true")
    group.add_argument("--freeze", action="store_true")
    group.add_argument("--run-predictions", action="store_true")
    group.add_argument("--score", action="store_true")
    args = parser.parse_args()
    try:
        if args.preflight:
            result = create_target_free_preflight()
        elif args.freeze:
            result = create_freeze()
        elif args.run_predictions:
            result = run_predictions()
        else:
            result = score_predictions()
        print(json.dumps(_jsonable(result), ensure_ascii=False, indent=2, allow_nan=False))
        return 0
    except Exception as exc:
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
