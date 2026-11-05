"""Source-gated frequency-adaptive positional-ridge pilot.

The preflight and synthetic commands never read target values. Official fit
execution requires a committed ROOT_SOURCE_PASS bound to this exact source.
This runner has no Test, SampleSubmission, score, or Kaggle code path.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import time
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import joblib
import numpy as np
import pandas as pd
import psutil
from scipy import sparse
from scipy.special import expit
from threadpoolctl import threadpool_limits

from src.models.pos_frequency_ridge_v1 import (
    FrequencyRidgeError,
    canonical_sha256,
    enforce_fit_cap,
    fit_resume_decision,
    fit_weighted_offset,
    make_penalty_maps,
    repeat_token_weights,
    sha256_array,
    union_document_frequency,
)
from src.models.optimization_v3_positional_text_pilot_v1 import sha256_csr


CAMPAIGN_ID = "CLS_POS_FREQUENCY_RIDGE_V1"
FIT_IDS = ("E67_ALIGNED", "E67_PERMUTED", "E75_ALIGNED", "E75_PERMUTED")
ORIGINS = (67, 75)
TRAIN = ROOT / "data/raw/classification/Train.csv"
PROTOCOL = ROOT / "experiments/optimization_v3/cls_pos_frequency_ridge_v1/protocol.json"
REPORT_ROOT = ROOT / "reports/optimization_v3/cls_pos_frequency_ridge_v1"
ROOT_SOURCE_PASS = REPORT_ROOT / "ROOT_SOURCE_PASS.json"
PILOT_ROOT = ROOT / "experiments/runs/optimization_v3_research_next_v1/POSITIONAL_TEXT_PILOT_V1"
OFFSET_ROOT = ROOT / "experiments/runs/optimization_v3_research_next_v1/POS_BAG_OFFSET_V1"
PILOT_ACCOUNTING = ROOT / "reports/optimization_v3/positional_text_pilot_v1/FIT_ACCOUNTING.json"
SHARED_LEDGER = ROOT / "experiments/runs/optimization_v3_research_next_v1/budget.jsonl"
RUN_ROOT = ROOT / "experiments/runs/optimization_v3_cls_pos_frequency_ridge_v1"
LEDGER = RUN_ROOT / "BUDGET_LEDGER.jsonl"
SOURCE_PREFLIGHT = REPORT_ROOT / "SOURCE_PREFLIGHT.json"
SOURCE_FILES = (
    "experiments/optimization_v3/cls_pos_frequency_ridge_v1/protocol.json",
    "src/models/pos_frequency_ridge_v1.py",
    "scripts/optimization_v3_cls_pos_frequency_ridge_v1.py",
    "scripts/verify_optimization_v3_cls_pos_frequency_ridge_v1.py",
    "tests/test_pos_frequency_ridge_v1.py",
)
POS_BAG_FREEZE_SHA256 = "5d754b99cf1beb789baec9707652cd9c3842ea95933c0ecc21be515f62eaef9d"
POS_BAG_FREEZE_SIGNATURE = "ba58b1e446a6be8546789c5a56596fbdbf075efc0ee19fcc1f41e02af7b419d7"
PILOT_FREEZE_SHA256 = "4e362e41b36b28a79be33deae881cfcad353d68bc044129c3f969860733e23b1"
PILOT_FREEZE_SIGNATURE = "f310d0340ea4402677ca9e86aa29c43b1536ded902c0f6a6f52cb53bd333c292"
ACCOUNTING_SHA256 = "c7e70dde6f97ec5a391221e4721be726878927200800388eaafcbcb6ae13b76b"
TRAIN_SHA256 = "875552a68a3d24b0f3e1966d6ddb5671e6f54305a1c526e2f0a76747942ed91c"
FREEZE_ORIGINS = {
    67: {
        "rows": 29705,
        "score_rows": 20903,
        "manifest_sha256": "d8d672c26771944bc2bb6eee907e9bef903959fbc62d9f5df3aaab7f7cb28028",
        "prediction_sha256": "2f22c68aca959baad6cd4244e05f96b702b4b89dfdc83f82c8edde6cf465d444",
    },
    75: {
        "rows": 42117,
        "score_rows": 8491,
        "manifest_sha256": "67ec3f9bd376f5f4dad2afbd5e13ee5fd8316126fa3894ef676fa40077166fc5",
        "prediction_sha256": "83e5beb9b10be4d395df4a2292cb464625f356e5e77a76a232b92f007c1b1306",
    },
}
INITIAL_FREE_MIN = 4 * 1024**3
FREE_RESERVE = 2 * 1024**3
RSS_LIMIT = 6 * 1024**3
FIT_LOCK = threading.Lock()


class FrequencyRidgeRunnerError(RuntimeError):
    """A source, identity, resource, ledger, or fit gate failed closed."""


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, allow_nan=False,
                      sort_keys=True, separators=(",", ":")).encode("utf-8")


def _sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _atomic_bytes(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f".{os.getpid()}.partial")
    with temp.open("wb") as stream:
        stream.write(content)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temp, path)


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    _atomic_bytes(path, _canonical_bytes(dict(payload)) + b"\n")


def _append_event(path: Path, event: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("ab") as stream:
        stream.write(_canonical_bytes(dict(event)) + b"\n")
        stream.flush()
        os.fsync(stream.fileno())


def _events(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    result: list[dict[str, Any]] = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise FrequencyRidgeRunnerError(f"Invalid ledger JSON at line {line_no}") from exc
        if not isinstance(value, dict):
            raise FrequencyRidgeRunnerError(f"Ledger row {line_no} is not an object")
        result.append(value)
    return result


def _git(*args: str) -> str:
    result = subprocess.run(["git", *args], cwd=ROOT, text=True, capture_output=True, check=False)
    if result.returncode:
        raise FrequencyRidgeRunnerError(f"git {' '.join(args)} failed: {result.stderr.strip()}")
    return result.stdout.strip()


def _source_hashes() -> dict[str, str]:
    hashes: dict[str, str] = {}
    for relative in SOURCE_FILES:
        path = ROOT / relative
        if not path.is_file():
            raise FrequencyRidgeRunnerError(f"Required source file is missing: {relative}")
        hashes[relative] = _sha_file(path)
    return hashes


def _resolve_repo_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def _verify_historical_freezes() -> dict[str, Any]:
    pos_freeze_path = OFFSET_ROOT / "FREEZE.json"
    pilot_freeze_path = PILOT_ROOT / "FREEZE.json"
    accounting_path = PILOT_ACCOUNTING
    for path, expected in ((pos_freeze_path, POS_BAG_FREEZE_SHA256),
                           (pilot_freeze_path, PILOT_FREEZE_SHA256),
                           (accounting_path, ACCOUNTING_SHA256)):
        if not path.is_file() or _sha_file(path) != expected:
            raise FrequencyRidgeRunnerError(f"Pinned historical freeze/accounting hash mismatch: {path}")
    pos_freeze = _json(pos_freeze_path)
    pilot_freeze = _json(pilot_freeze_path)
    accounting = _json(accounting_path)
    if (pos_freeze.get("freeze_signature") != POS_BAG_FREEZE_SIGNATURE
            or pilot_freeze.get("freeze_signature") != PILOT_FREEZE_SIGNATURE
            or pos_freeze.get("input_sha256") != TRAIN_SHA256
            or pilot_freeze.get("input_sha256") != TRAIN_SHA256
            or accounting.get("train_sha256") != TRAIN_SHA256):
        raise FrequencyRidgeRunnerError("Historical freeze signature or official Train lineage mismatch")
    checked_source_hashes: dict[str, str] = {}
    for freeze in (pos_freeze, pilot_freeze):
        for raw_path, expected in freeze.get("source_hashes", {}).items():
            path = _resolve_repo_path(raw_path)
            if not path.is_file() or _sha_file(path) != expected:
                raise FrequencyRidgeRunnerError(f"Historical source pin mismatch: {raw_path}")
            checked_source_hashes[Path(raw_path).name + ":" + str(expected)] = expected
    return {
        "pos_bag_freeze_sha256": _sha_file(pos_freeze_path),
        "pos_bag_freeze_signature": pos_freeze["freeze_signature"],
        "position_pilot_freeze_sha256": _sha_file(pilot_freeze_path),
        "position_pilot_freeze_signature": pilot_freeze["freeze_signature"],
        "fit_accounting_sha256": _sha_file(accounting_path),
        "fit_accounting_train_sha256": accounting["train_sha256"],
        "historical_source_files_checked": len(checked_source_hashes),
        "input_sha256": pos_freeze.get("input_sha256"),
        "prior_source_commit": pos_freeze.get("source_commit"),
    }


def _read_identity_frame() -> pd.DataFrame:
    frame = pd.read_csv(TRAIN, usecols=["record_id", "cycle_code"], dtype={"record_id": "string"})
    if len(frame) != 50608 or frame["record_id"].isna().any() or frame["record_id"].duplicated().any():
        raise FrequencyRidgeRunnerError("Official Train identity count/uniqueness check failed")
    frame["cycle_code"] = pd.to_numeric(frame["cycle_code"], errors="raise").astype(np.int64)
    if frame["cycle_code"].min() != 47 or frame["cycle_code"].max() != 79:
        raise FrequencyRidgeRunnerError("Official Train cycle coverage differs from frozen 47-79")
    return frame


def _row_metadata(frame: pd.DataFrame, rows: np.ndarray) -> dict[str, Any]:
    selected = frame.iloc[np.asarray(rows, dtype=np.int64)]
    ids = selected["record_id"].astype(str).to_numpy()
    cycles = selected["cycle_code"].to_numpy(dtype=np.int64)
    if len(rows) == 0:
        raise FrequencyRidgeRunnerError("A frozen origin population is unexpectedly empty")
    return {
        "rows": int(len(rows)),
        "cycle_min": int(cycles.min()),
        "cycle_max": int(cycles.max()),
        "row_indices_sha256": sha256_array(np.asarray(rows, dtype=np.int64)),
        "record_ids_sha256": sha256_array(ids),
        "cycles_sha256": sha256_array(cycles),
    }


def _verify_origin_identity_and_files(frame: pd.DataFrame, origin: int) -> dict[str, Any]:
    manifest_path = PILOT_ROOT / f"MATRIX_MANIFEST_E{origin}.json"
    prediction_path = PILOT_ROOT / f"E{origin}" / f"predictions_E{origin}.npz"
    manifest = _json(manifest_path)
    expected = FREEZE_ORIGINS[origin]
    if _sha_file(manifest_path) != expected["manifest_sha256"]:
        raise FrequencyRidgeRunnerError(f"E{origin} matrix manifest hash mismatch")
    if not prediction_path.is_file() or _sha_file(prediction_path) != expected["prediction_sha256"]:
        raise FrequencyRidgeRunnerError(f"E{origin} prediction bundle hash mismatch")
    fit_rows = np.flatnonzero(frame["cycle_code"].to_numpy(dtype=np.int64) <= origin)
    score_rows = np.flatnonzero(frame["cycle_code"].to_numpy(dtype=np.int64) > origin)
    if (len(fit_rows) != expected["rows"] or len(score_rows) != expected["score_rows"]
            or _row_metadata(frame, fit_rows) != manifest.get("fit_row_metadata")
            or _row_metadata(frame, score_rows) != manifest.get("score_row_metadata")):
        raise FrequencyRidgeRunnerError(f"E{origin} exact Train row identity/cycle map changed")
    with np.load(prediction_path, allow_pickle=False) as archive:
        saved_fit_rows = np.asarray(archive["fit_rows"], dtype=np.int64)
        saved_score_rows = np.asarray(archive["score_rows"], dtype=np.int64)
        saved_score_ids = np.asarray(archive["score_record_ids"]).astype(str)
        saved_score_cycles = np.asarray(archive["score_cycles"], dtype=np.int64)
    if (not np.array_equal(saved_fit_rows, fit_rows) or not np.array_equal(saved_score_rows, score_rows)
            or not np.array_equal(saved_score_ids, frame.iloc[score_rows]["record_id"].astype(str).to_numpy())
            or not np.array_equal(saved_score_cycles, frame.iloc[score_rows]["cycle_code"].to_numpy(dtype=np.int64))):
        raise FrequencyRidgeRunnerError(f"E{origin} cached prediction row IDs/cycles differ from Train identity")
    matrix_hashes: dict[str, str] = {}
    frozen = _json(OFFSET_ROOT / "FREEZE.json")["origins"][f"E{origin}"]
    for name, item in manifest["matrices"].items():
        path = _resolve_repo_path(str(item["path"]))
        actual_hash = _sha_file(path)
        if actual_hash != frozen["matrix_file_hashes"].get(name):
            raise FrequencyRidgeRunnerError(f"E{origin} cached matrix byte hash mismatch: {name}")
        matrix_hashes[name] = actual_hash
    bag_id = f"POSITIONAL_TEXT_PILOT_V1_E{origin}_BAG_LR"
    accounting = _json(PILOT_ACCOUNTING)
    bag_entry = next((item for item in accounting["fit_units"] if item.get("fit_id") == bag_id), None)
    if bag_entry is None:
        raise FrequencyRidgeRunnerError(f"Missing frozen bag fit accounting: {bag_id}")
    for path, key in ((PILOT_ROOT / "units" / f"{bag_id}.joblib", "artifact_sha256"),
                      (PILOT_ROOT / "receipts" / f"{bag_id}.json", "receipt_sha256")):
        if not path.is_file() or _sha_file(path) != bag_entry[key]:
            raise FrequencyRidgeRunnerError(f"E{origin} frozen bag state/receipt mismatch: {path.name}")
    bag_receipt = _json(PILOT_ROOT / "receipts" / f"{bag_id}.json")
    if (bag_receipt.get("fit_id") != bag_id
            or bag_receipt.get("artifact_sha256") != bag_entry["artifact_sha256"]
            or bag_receipt.get("fit_population") != bag_entry["fit_population"]):
        raise FrequencyRidgeRunnerError(f"E{origin} frozen bag receipt lineage mismatch")
    return {
        "origin": origin,
        "fit_rows": len(fit_rows),
        "score_rows": len(score_rows),
        "fit_row_metadata": manifest["fit_row_metadata"],
        "score_row_metadata": manifest["score_row_metadata"],
        "matrix_manifest_sha256": _sha_file(manifest_path),
        "prediction_bundle_sha256": _sha_file(prediction_path),
        "matrix_file_hashes": matrix_hashes,
        "bag_artifact_sha256": bag_entry["artifact_sha256"],
        "bag_receipt_sha256": bag_entry["receipt_sha256"],
    }


def resource_snapshot(estimated_additional_bytes: int = 2 * 1024**3) -> dict[str, Any]:
    memory = psutil.virtual_memory()
    current_rss = psutil.Process().memory_info().rss
    projected_free = max(0, int(memory.available) - int(estimated_additional_bytes))
    return {
        "available_bytes": int(memory.available),
        "total_bytes": int(memory.total),
        "process_rss_bytes": int(current_rss),
        "estimated_additional_bytes": int(estimated_additional_bytes),
        "projected_free_bytes": projected_free,
        "initial_min_bytes": INITIAL_FREE_MIN,
        "free_reserve_bytes": FREE_RESERVE,
        "rss_limit_bytes": RSS_LIMIT,
        "threads": 2,
        "jobs": 1,
        "initial_ram_pass": int(memory.available) >= INITIAL_FREE_MIN,
        "projected_reserve_pass": projected_free >= FREE_RESERVE,
        "rss_pass": int(current_rss) + int(estimated_additional_bytes) <= RSS_LIMIT,
    }


def run_source_preflight() -> dict[str, Any]:
    if not TRAIN.is_file() or _sha_file(TRAIN) != TRAIN_SHA256:
        raise FrequencyRidgeRunnerError("Official Classification Train file hash differs from frozen source")
    historical = _verify_historical_freezes()
    frame = _read_identity_frame()
    origins = {f"E{origin}": _verify_origin_identity_and_files(frame, origin) for origin in ORIGINS}
    source_hashes = _source_hashes()
    shared_ledger_hash = _sha_file(SHARED_LEDGER) if SHARED_LEDGER.is_file() else None
    protocol = _json(PROTOCOL)
    expected_ledger_hash = protocol["historical_evidence"]["prior_shared_ledger_sha256_at_checkpoint"]
    resource = resource_snapshot()
    resource_ok = all(resource[key] for key in ("initial_ram_pass", "projected_reserve_pass", "rss_pass"))
    result = {
        "schema_version": 1,
        "campaign_id": CAMPAIGN_ID,
        "status": "PASS_SOURCE_AND_IDENTITY_NO_GO_CURRENT_RESOURCE" if not resource_ok else "PASS_SOURCE_AND_IDENTITY_RESOURCE_READY",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_commit": _git("rev-parse", "HEAD"),
        "source_hashes": source_hashes,
        "train_sha256": TRAIN_SHA256,
        "train_rows": len(frame),
        "train_cycle_min": int(frame["cycle_code"].min()),
        "train_cycle_max": int(frame["cycle_code"].max()),
        "target_values_opened": False,
        "test_or_sample_submission_accessed": False,
        "kaggle_accessed": False,
        "official_fit_started": False,
        "historical_checks": historical,
        "origin_identity_and_cache_checks": origins,
        "prior_shared_ledger_reference": {
            "expected_sha256": expected_ledger_hash,
            "current_sha256": shared_ledger_hash,
            "matches_checkpoint": shared_ledger_hash == expected_ledger_hash,
            "modified": False,
        },
        "resource": resource,
        "root_source_pass_present": ROOT_SOURCE_PASS.is_file(),
        "e79_user_authorization": "Ya, otorisasi satu benchmark E79 (not executed in this source-only pass)",
    }
    _atomic_json(SOURCE_PREFLIGHT, result)
    return result


def _require_root_source_pass() -> dict[str, Any]:
    if not ROOT_SOURCE_PASS.is_file():
        raise FrequencyRidgeRunnerError("Official fit blocked: committed ROOT_SOURCE_PASS.json is absent")
    tracked = subprocess.run(["git", "ls-files", "--error-unmatch",
                              str(ROOT_SOURCE_PASS.relative_to(ROOT).as_posix())],
                             cwd=ROOT, text=True, capture_output=True)
    if tracked.returncode != 0:
        raise FrequencyRidgeRunnerError("ROOT_SOURCE_PASS.json must be committed before an official fit")
    passed = _json(ROOT_SOURCE_PASS)
    current_hashes = _source_hashes()
    if passed.get("status") != "ROOT_SOURCE_PASS" or passed.get("source_hashes") != current_hashes:
        raise FrequencyRidgeRunnerError("ROOT_SOURCE_PASS status/source hashes do not match executable sources")
    commit = str(passed.get("source_commit", ""))
    if not commit:
        raise FrequencyRidgeRunnerError("ROOT_SOURCE_PASS is missing source_commit")
    is_ancestor = subprocess.run(["git", "merge-base", "--is-ancestor", commit, "HEAD"], cwd=ROOT)
    if is_ancestor.returncode != 0:
        raise FrequencyRidgeRunnerError("ROOT_SOURCE_PASS source_commit is not an ancestor of current HEAD")
    if passed.get("source_preflight_sha256") != _sha_file(SOURCE_PREFLIGHT):
        raise FrequencyRidgeRunnerError("ROOT_SOURCE_PASS does not bind the current target-free source preflight")
    preflight_tracked = subprocess.run(["git", "ls-files", "--error-unmatch",
                                        str(SOURCE_PREFLIGHT.relative_to(ROOT).as_posix())],
                                       cwd=ROOT, text=True, capture_output=True)
    if preflight_tracked.returncode != 0:
        raise FrequencyRidgeRunnerError("SOURCE_PREFLIGHT.json must be committed before an official fit")
    if _git("diff", "--quiet") != "" or _git("diff", "--cached", "--quiet") != "":
        raise FrequencyRidgeRunnerError("Tracked source tree changed after ROOT_SOURCE_PASS")
    return passed


def _read_fit_labels(fit_rows: np.ndarray, expected_ids: np.ndarray, expected_target_sha256: str) -> np.ndarray:
    if _sha_file(TRAIN) != TRAIN_SHA256:
        raise FrequencyRidgeRunnerError("Official Train changed before fit-target access")
    fit_rows = np.asarray(fit_rows, dtype=np.int64)
    ids_expected = np.asarray(expected_ids).astype(str)
    if len(fit_rows) != len(ids_expected) or len(np.unique(fit_rows)) != len(fit_rows):
        raise FrequencyRidgeRunnerError("Frozen fit row indices/IDs are invalid")
    position = {int(row): index for index, row in enumerate(fit_rows.tolist())}
    y = np.empty(len(fit_rows), dtype=np.int8)
    seen = np.zeros(len(fit_rows), dtype=bool)
    with TRAIN.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = __import__("csv").reader(stream)
        header = next(reader, None)
        if not header or "record_id" not in header or "artifact_signal" not in header:
            raise FrequencyRidgeRunnerError("Official Train CSV header is invalid")
        id_col, target_col = header.index("record_id"), header.index("artifact_signal")
        for row_index, record in enumerate(reader):
            out_idx = position.get(row_index)
            if out_idx is None:
                continue
            if max(id_col, target_col) >= len(record) or record[id_col] != ids_expected[out_idx]:
                raise FrequencyRidgeRunnerError("Fit target row ID differs from frozen prefix identity")
            y[out_idx] = int(record[target_col])
            seen[out_idx] = True
    if not seen.all() or not np.isin(y, (0, 1)).all() or sha256_array(y) != expected_target_sha256:
        raise FrequencyRidgeRunnerError("Official fit-prefix labels differ from frozen target hash")
    return y


def _load_csr_from_manifest(manifest: Mapping[str, Any], name: str) -> sparse.csr_matrix:
    item = manifest["matrices"][name]
    path = _resolve_repo_path(str(item["path"]))
    if _sha_file(path) != item["sha256"]:
        raise FrequencyRidgeRunnerError(f"Frozen matrix byte hash mismatch: {name}")
    matrix = sparse.load_npz(path).tocsr()
    if (list(matrix.shape) != item["shape"] or str(matrix.dtype) != item["dtype"]
            or sha256_csr(matrix) != item["logical_sha256"]):
        raise FrequencyRidgeRunnerError(f"Frozen matrix logical shape/hash mismatch: {name}")
    return matrix


def _append_campaign_reservation(root_pass: Mapping[str, Any], prior_ledger_hash: str | None) -> None:
    existing = _events(LEDGER)
    reservations = [e for e in existing if e.get("event") == "CAMPAIGN_RESERVED"]
    if reservations:
        if (len(reservations) != 1 or reservations[0].get("fit_ids") != list(FIT_IDS)
                or reservations[0].get("source_commit") != root_pass.get("source_commit")
                or reservations[0].get("source_hashes") != root_pass.get("source_hashes")
                or reservations[0].get("prior_shared_ledger_sha256") != prior_ledger_hash):
            raise FrequencyRidgeRunnerError("Existing campaign ledger reservation conflicts with exact cap-4 plan")
        return
    _append_event(LEDGER, {
        "event": "CAMPAIGN_RESERVED",
        "campaign_id": CAMPAIGN_ID,
        "reservation_id": "CLS_POS_FREQUENCY_RIDGE_V1_CAP4_R1",
        "fit_ids": list(FIT_IDS),
        "fit_cap": 4,
        "new_preprocessing_maps": ["E67_PREFIX_DF", "E75_PREFIX_DF"],
        "source_commit": root_pass["source_commit"],
        "source_hashes": root_pass["source_hashes"],
        "source_preflight_sha256": _sha_file(SOURCE_PREFLIGHT),
        "prior_shared_ledger_sha256": prior_ledger_hash,
        "prior_campaign_checkpoint_commit": "a92e5df980d85ff7f4c4e402929d1c7761010b46",
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
    })


def _ensure_penalty_map(origin: int, x_fit: sparse.csr_matrix, vocabulary_size: int) -> dict[str, Any]:
    artifact = RUN_ROOT / "maps" / f"penalty_map_E{origin}.npz"
    receipt_path = RUN_ROOT / "maps" / f"penalty_map_E{origin}.json"
    events = _events(LEDGER)
    saved = [e for e in events if e.get("event") == "PREPROCESSING_MAP_SAVED" and e.get("origin") == origin]
    if saved:
        if len(saved) != 1 or not artifact.is_file() or not receipt_path.is_file():
            raise FrequencyRidgeRunnerError(f"E{origin} penalty map has a conflicting durable receipt")
        receipt = _json(receipt_path)
        if (receipt.get("artifact_sha256") != _sha_file(artifact)
                or saved[0].get("receipt_sha256") != _sha_file(receipt_path)):
            raise FrequencyRidgeRunnerError(f"E{origin} penalty map receipt/hash mismatch")
        with np.load(artifact, allow_pickle=False) as loaded:
            maps = {name: loaded[name].copy() for name in loaded.files}
        if (set(maps) != {"document_frequency", "aligned_token_weights", "permuted_token_weights", "permutation"}
                or len(maps["document_frequency"]) != int(receipt.get("vocabulary_size", -1))
                or sha256_array(maps["document_frequency"]) != receipt.get("df_sha256")
                or sha256_array(maps["aligned_token_weights"]) != receipt.get("aligned_sha256")
                or sha256_array(maps["permuted_token_weights"]) != receipt.get("permuted_sha256")
                or sha256_array(maps["permutation"]) != receipt.get("permutation_sha256")
                or not np.isclose(float(maps["aligned_token_weights"].mean()), 1.0, rtol=0, atol=1e-14)):
            raise FrequencyRidgeRunnerError(f"E{origin} saved penalty map arrays do not match their receipt")
        maps["receipt"] = receipt
        return maps
    if any(e.get("event") == "PREPROCESSING_MAP_STARTED" and e.get("origin") == origin for e in events):
        raise FrequencyRidgeRunnerError(f"E{origin} map computation has an incomplete start; no silent retry")
    _append_event(LEDGER, {"event": "PREPROCESSING_MAP_STARTED", "campaign_id": CAMPAIGN_ID,
                           "origin": origin, "fit_rows": int(x_fit.shape[0]),
                           "matrix_sha256": sha256_csr(x_fit),
                           "recorded_at_utc": datetime.now(timezone.utc).isoformat()})
    df = union_document_frequency(x_fit, vocabulary_size)
    result = make_penalty_maps(df)
    if result["aligned_equals_permuted"]:
        raise FrequencyRidgeRunnerError(f"E{origin} alignment diagnostic is degenerate; do not replace seed")
    artifact.parent.mkdir(parents=True, exist_ok=True)
    temp = artifact.with_name(artifact.name + f".{os.getpid()}.partial")
    with temp.open("wb") as stream:
        np.savez_compressed(stream,
                            document_frequency=result["document_frequency"],
                            aligned_token_weights=result["aligned_token_weights"],
                            permuted_token_weights=result["permuted_token_weights"],
                            permutation=result["permutation"])
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temp, artifact)
    receipt = {
        "schema_version": 1,
        "campaign_id": CAMPAIGN_ID,
        "origin": origin,
        "fit_rows": int(x_fit.shape[0]),
        "vocabulary_size": vocabulary_size,
        "fit_position_sha256": sha256_csr(x_fit),
        "df_sha256": result["df_sha256"],
        "aligned_sha256": result["aligned_sha256"],
        "permuted_sha256": result["permuted_sha256"],
        "permutation_sha256": result["permutation_sha256"],
        "clip_low_count": result["clip_low_count"],
        "clip_high_count": result["clip_high_count"],
        "mean_aligned_weight": float(np.mean(result["aligned_token_weights"])),
        "seed": 42,
        "formula": "clip(sqrt(median(df)/df_j),0.5,2.0), normalize mean 1",
        "artifact_path": artifact.relative_to(ROOT).as_posix(),
        "artifact_sha256": _sha_file(artifact),
    }
    _atomic_json(receipt_path, receipt)
    _append_event(LEDGER, {"event": "PREPROCESSING_MAP_SAVED", "campaign_id": CAMPAIGN_ID,
                           "origin": origin, "artifact_sha256": receipt["artifact_sha256"],
                           "receipt_sha256": _sha_file(receipt_path),
                           "recorded_at_utc": datetime.now(timezone.utc).isoformat()})
    return {**result, "receipt": receipt}


def _fit_paths(fit_id: str) -> tuple[Path, Path]:
    return RUN_ROOT / "units" / f"{fit_id}.npz", RUN_ROOT / "receipts" / f"{fit_id}.json"


def _run_one_fit(fit_id: str) -> dict[str, Any]:
    if fit_id not in FIT_IDS:
        raise FrequencyRidgeRunnerError(f"Fit ID is outside the exact four-fit allowance: {fit_id}")
    root_pass = _require_root_source_pass()
    preflight = _json(SOURCE_PREFLIGHT)
    if (preflight.get("train_sha256") != TRAIN_SHA256
            or preflight.get("target_values_opened") is not False
            or preflight.get("source_hashes") != _source_hashes()):
        raise FrequencyRidgeRunnerError("Current target-free preflight does not bind the approved source/input")
        if not preflight.get("prior_shared_ledger_reference", {}).get("matches_checkpoint"):
            raise FrequencyRidgeRunnerError("Historical shared-ledger reference changed after source preflight")
    protocol = _json(PROTOCOL)
    prior_ledger_hash = _sha_file(SHARED_LEDGER) if SHARED_LEDGER.is_file() else None
    expected_ledger_hash = protocol["historical_evidence"]["prior_shared_ledger_sha256_at_checkpoint"]
    if prior_ledger_hash != expected_ledger_hash:
        raise FrequencyRidgeRunnerError("Historical shared ledger changed after source preflight")
    if _sha_file(TRAIN) != TRAIN_SHA256:
        raise FrequencyRidgeRunnerError("Official Train file hash changed after root source review")
    _verify_historical_freezes()
    with _fit_run_lock(RUN_ROOT / "FIT_RUN.lock"):
        _append_campaign_reservation(root_pass, prior_ledger_hash)
        frame = _read_identity_frame()
        origin = 67 if fit_id.startswith("E67_") else 75
        arm = "ALIGNED" if fit_id.endswith("ALIGNED") else "PERMUTED"
        expected_rows = FREEZE_ORIGINS[origin]["rows"]
        identity_cycles = frame["cycle_code"].to_numpy(dtype=np.int64)
        fit_rows = np.flatnonzero(identity_cycles <= origin)
        if len(fit_rows) != expected_rows:
            raise FrequencyRidgeRunnerError(f"E{origin} fit population changed before matrix load")
        manifest_path = PILOT_ROOT / f"MATRIX_MANIFEST_E{origin}.json"
        manifest = _json(manifest_path)
        _verify_origin_identity_and_files(frame, origin)
        resource = resource_snapshot(estimated_additional_bytes=2 * 1024**3)
        if not resource["initial_ram_pass"] or not resource["projected_reserve_pass"] or not resource["rss_pass"]:
            raise FrequencyRidgeRunnerError(f"Resource gate failed before matrix load/target access: {resource}")
        base_fit = _load_csr_from_manifest(manifest, "fit_base")
        augmented_fit = _load_csr_from_manifest(manifest, "fit_position")
        x_fit = augmented_fit[:, base_fit.shape[1]:].tocsr()
        if x_fit.shape[0] != len(fit_rows) or x_fit.shape[1] % 3:
            raise FrequencyRidgeRunnerError("Frozen positional block dimensions do not match fit population")
        vocabulary_size = x_fit.shape[1] // 3
        if manifest["position_audit"]["fit_original"]["unigram_vocabulary_size"] != vocabulary_size:
            raise FrequencyRidgeRunnerError("Frozen unigram block order/width differs from matrix audit")
        if (augmented_fit[:, :base_fit.shape[1]] != base_fit).nnz:
            raise FrequencyRidgeRunnerError("Augmented positional matrix base columns differ from frozen C2_B base")
        penalty_map = _ensure_penalty_map(origin, x_fit, vocabulary_size)
        weights = penalty_map["aligned_token_weights"] if arm == "ALIGNED" else penalty_map["permuted_token_weights"]
        repeated_weights = repeat_token_weights(weights)
        if len(repeated_weights) != x_fit.shape[1]:
            raise FrequencyRidgeRunnerError("Repeated token penalties do not match positional columns")
        events = _events(LEDGER)
        if any(event.get("event") == "FIT_FAILED" for event in events):
            raise FrequencyRidgeRunnerError("A previous predictor fit failed; stop the campaign for root review")
        decision = fit_resume_decision(events, fit_id, FIT_IDS)
        artifact_path, receipt_path = _fit_paths(fit_id)
        if decision == "RESUME_SAVED":
            if not artifact_path.is_file() or not receipt_path.is_file():
                raise FrequencyRidgeRunnerError("Ledger says fit is saved but its durable files are missing")
            receipt = _json(receipt_path)
            if receipt.get("artifact_sha256") != _sha_file(artifact_path):
                raise FrequencyRidgeRunnerError("Saved fit receipt does not match its gamma artifact")
            return {"fit_id": fit_id, "status": "ALREADY_SAVED", "receipt_sha256": _sha_file(receipt_path)}
        enforce_fit_cap(events, FIT_IDS)
        terminal_ids = {event.get("fit_id") for event in events
                        if event.get("event") in {"FIT_SAVED", "FIT_FAILED"}}
        pending_ids = {event.get("fit_id") for event in events if event.get("event") == "FIT_STARTED"} - terminal_ids
        if pending_ids:
            raise FrequencyRidgeRunnerError(f"Another predictor fit is still pending: {sorted(pending_ids)}")
        bag_id = f"POSITIONAL_TEXT_PILOT_V1_E{origin}_BAG_LR"
        accounting = _json(PILOT_ACCOUNTING)
        bag_acc = next(item for item in accounting["fit_units"] if item.get("fit_id") == bag_id)
        expected_ids = frame.iloc[fit_rows]["record_id"].astype(str).to_numpy()
        if _row_metadata(frame, fit_rows) != manifest["fit_row_metadata"]:
            raise FrequencyRidgeRunnerError("Fit row IDs/cycles differ from frozen matrix manifest before labels")
        bag_path = PILOT_ROOT / "units" / f"{bag_id}.joblib"
        bag_unit = joblib.load(bag_path)
        bag = bag_unit.get("estimator")
        if bag is None or list(np.asarray(bag.classes_).tolist()) != [0, 1]:
            raise FrequencyRidgeRunnerError("Frozen C2_B bag estimator/class order invalid")
        if np.asarray(bag.coef_).shape != (1, base_fit.shape[1]) or len(np.asarray(bag.intercept_).reshape(-1)) != 1:
            raise FrequencyRidgeRunnerError("Frozen C2_B bag coefficient width differs from saved base matrix")
        z0_fit = np.asarray(base_fit @ bag.coef_[0], dtype=np.float64).reshape(-1) + float(bag.intercept_[0])
        if not np.isfinite(z0_fit).all():
            raise FrequencyRidgeRunnerError("Frozen C2_B bag fit-prefix logits are non-finite")
        y = _read_fit_labels(fit_rows, expected_ids, bag_acc["fit_population"]["fit_target_sha256"])
        resource = resource_snapshot(estimated_additional_bytes=1024**3)
        if not resource["initial_ram_pass"] or not resource["projected_reserve_pass"] or not resource["rss_pass"]:
            raise FrequencyRidgeRunnerError(f"Resource gate failed immediately before predictor fit: {resource}")
        _append_event(LEDGER, {
            "event": "FIT_STARTED", "campaign_id": CAMPAIGN_ID, "fit_id": fit_id,
            "origin": origin, "arm": arm, "fit_consumed_at_start": True,
            "source_commit": root_pass["source_commit"],
            "source_hashes": root_pass["source_hashes"],
            "train_sha256": TRAIN_SHA256,
            "fit_rows": len(fit_rows), "fit_row_indices_sha256": sha256_array(fit_rows),
            "fit_ids_sha256": sha256_array(expected_ids), "fit_target_sha256": sha256_array(y),
            "fit_position_sha256": sha256_csr(x_fit), "fit_base_sha256": sha256_csr(base_fit),
            "penalty_map_receipt_sha256": _sha_file(RUN_ROOT / "maps" / f"penalty_map_E{origin}.json"),
            "resource": resource, "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
        })
        started = time.perf_counter()
        try:
            with threadpool_limits(limits=2):
                gamma, optimizer = fit_weighted_offset(x_fit, z0_fit, y, repeated_weights)
            artifact_path.parent.mkdir(parents=True, exist_ok=True)
            temp = artifact_path.with_name(artifact_path.name + f".{os.getpid()}.partial")
            with temp.open("wb") as stream:
                np.savez_compressed(stream, gamma=gamma)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temp, artifact_path)
            receipt = {
                "schema_version": 1, "campaign_id": CAMPAIGN_ID, "fit_id": fit_id,
                "origin": origin, "arm": arm, "fit_kind": "PREDICTOR",
                "source_commit": root_pass["source_commit"], "source_hashes": root_pass["source_hashes"],
                "root_source_pass_sha256": _sha_file(ROOT_SOURCE_PASS),
                "source_preflight_sha256": _sha_file(SOURCE_PREFLIGHT),
                "train_sha256": TRAIN_SHA256, "fit_rows": len(fit_rows),
                "fit_row_indices_sha256": sha256_array(fit_rows),
                "fit_ids_sha256": sha256_array(expected_ids),
                "fit_target_sha256": sha256_array(y),
                "fit_position_sha256": sha256_csr(x_fit),
                "fit_base_sha256": sha256_csr(base_fit),
                "fixed_bag_artifact_sha256": bag_acc["artifact_sha256"],
                "fixed_bag_receipt_sha256": bag_acc["receipt_sha256"],
                "fixed_bag_offset_sha256": sha256_array(z0_fit),
                "penalty_map_receipt_sha256": _sha_file(RUN_ROOT / "maps" / f"penalty_map_E{origin}.json"),
                "penalty_vector_sha256": sha256_array(repeated_weights),
                "optimizer": optimizer,
                "gamma_sha256": sha256_array(gamma),
                "artifact_path": artifact_path.relative_to(ROOT).as_posix(),
                "artifact_sha256": _sha_file(artifact_path),
                "elapsed_seconds": float(time.perf_counter() - started),
                "target_access": "official fit-prefix only; no query/Test labels opened",
            }
            _atomic_json(receipt_path, receipt)
            _append_event(LEDGER, {"event": "FIT_SAVED", "campaign_id": CAMPAIGN_ID,
                                   "fit_id": fit_id, "receipt_sha256": _sha_file(receipt_path),
                                   "artifact_sha256": receipt["artifact_sha256"],
                                   "recorded_at_utc": datetime.now(timezone.utc).isoformat()})
            return {"fit_id": fit_id, "status": "SAVED", "optimizer": optimizer,
                    "receipt_sha256": _sha_file(receipt_path)}
        except Exception as exc:
            _append_event(LEDGER, {"event": "FIT_FAILED", "campaign_id": CAMPAIGN_ID,
                                   "fit_id": fit_id, "error_type": type(exc).__name__,
                                   "error": str(exc),
                                   "recorded_at_utc": datetime.now(timezone.utc).isoformat()})
            raise


@contextmanager
def _fit_run_lock(path: Path):
    if not FIT_LOCK.acquire(blocking=False):
        raise FrequencyRidgeRunnerError("Another frequency-ridge operation is active in this process")
    path.parent.mkdir(parents=True, exist_ok=True)
    stream = None
    unlock = None
    try:
        stream = path.open("a+b")
        stream.seek(0)
        if stream.read(1) == b"":
            stream.write(b"\0")
            stream.flush()
        stream.seek(0)
        if os.name == "nt":
            import msvcrt
            try:
                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            except OSError as exc:
                raise FrequencyRidgeRunnerError("Another process owns the frequency-ridge OS lock") from exc
            unlock = lambda: msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl
            try:
                fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError as exc:
                raise FrequencyRidgeRunnerError("Another process owns the frequency-ridge OS lock") from exc
            unlock = lambda: fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
        stream.seek(0)
        payload = _canonical_bytes({"pid": os.getpid(), "started_at_utc": datetime.now(timezone.utc).isoformat()})
        stream.write(payload)
        stream.truncate()
        stream.flush()
        os.fsync(stream.fileno())
        yield
    finally:
        if unlock is not None and stream is not None:
            try:
                stream.seek(0)
                unlock()
            except OSError:
                pass
        if stream is not None:
            stream.close()
        FIT_LOCK.release()


def run_synthetic_qa() -> dict[str, Any]:
    rng = np.random.default_rng(20261105)
    x = sparse.csr_matrix(rng.normal(size=(90, 12)))
    offset = rng.normal(size=90)
    p = expit(offset + np.asarray(x @ rng.normal(scale=0.12, size=12)).reshape(-1))
    y = rng.binomial(1, p).astype(np.int8)
    df = np.arange(1, 5, dtype=np.int64)
    maps = make_penalty_maps(df)
    weights = repeat_token_weights(maps["aligned_token_weights"])
    gamma, metadata = fit_weighted_offset(x, offset, y, weights)
    if not metadata["success"] or not np.isfinite(gamma).all():
        raise FrequencyRidgeRunnerError("Synthetic optimizer failed")
    fallback = np.asarray([0.125, 0.75, 0.333333333333])
    probability = expit(np.asarray(offset[:3] + x[:3, :] @ gamma).reshape(-1))
    routed = fallback.copy()
    present = np.asarray([True, False, False])
    routed[present] = probability[present]
    if not np.array_equal(routed[~present], fallback[~present]):
        raise FrequencyRidgeRunnerError("Synthetic missing fallback changed")
    with tempfile.TemporaryDirectory(prefix="cls-pos-frequency-ridge-v1-") as temporary:
        temp = Path(temporary)
        allowed = ("SYNTH_FIT",)
        ledger = [{"event": "FIT_STARTED", "fit_id": "SYNTH_FIT"}]
        artifact = temp / "gamma.npz"
        with artifact.open("wb") as stream:
            np.savez_compressed(stream, gamma=gamma)
        digest = _sha_file(artifact)
        receipt = {"fit_id": "SYNTH_FIT", "artifact_sha256": digest}
        _atomic_json(temp / "receipt.json", receipt)
        ledger.append({"event": "FIT_SAVED", "fit_id": "SYNTH_FIT"})
        if fit_resume_decision(ledger, "SYNTH_FIT", allowed) != "RESUME_SAVED":
            raise FrequencyRidgeRunnerError("Synthetic restart failed to resume saved fit")
        if len([event for event in ledger if event.get("event") == "FIT_STARTED"]) != 1:
            raise FrequencyRidgeRunnerError("Synthetic resume duplicated a predictor fit")
        enforce_fit_cap(ledger, allowed)
    return {
        "status": "PASS",
        "synthetic_rows": 90,
        "synthetic_position_columns": 12,
        "fit_success": metadata["success"],
        "loss_nonincreasing": metadata["loss_final"] <= metadata["loss_gamma0"] + 1e-8,
        "fallback_exact": True,
        "durable_resume_without_duplicate_fit": True,
        "official_inputs_opened": False,
        "official_labels_opened": False,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--synthetic-qa", action="store_true")
    group.add_argument("--preflight", action="store_true")
    group.add_argument("--run-next-fit", choices=FIT_IDS)
    args = parser.parse_args(argv)
    try:
        if args.synthetic_qa:
            result = run_synthetic_qa()
        elif args.preflight:
            result = run_source_preflight()
        else:
            result = _run_one_fit(args.run_next_fit)
        print(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False))
        return 0
    except (FrequencyRidgeRunnerError, FrequencyRidgeError, OSError, ValueError) as exc:
        print(json.dumps({"status": "BLOCKED_OR_FAILED_CLOSED", "error": f"{type(exc).__name__}: {exc}"},
                         ensure_ascii=False, sort_keys=True, indent=2))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
