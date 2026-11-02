"""Run the frozen six-predictor A02 continuation and gated final preparation."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gc
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any

import joblib
import numpy as np
import pandas as pd
import psutil
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import average_precision_score, log_loss, roc_auc_score
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.models.classification_a02_neighbors import (  # noqa: E402
    ResidualNeighborPredictor, TFIDF_PARAMETERS, fit_text_space,
    permute_residuals_within_cycle_probability_decile,
)
from src.models.optimization_v3_a02_continuation_budget import (  # noqa: E402
    A02ContinuationBudgetError, apply_cap_amendment, reserve, start_fit,
)
from src.models.research_next_budget import finish_fit, read_events  # noqa: E402
from src.models.research_next_contract import canonical_sha256, file_sha256  # noqa: E402
from scripts import optimization_v3_a02_neighbors as pilot  # noqa: E402


PROTOCOL_PATH = ROOT / "experiments/optimization_v3/a02_continuation_v1/protocol.json"
FREEZE_PATH = ROOT / "experiments/optimization_v3/a02_continuation_v1/freeze.json"
TRAIN_PATH = ROOT / "data/raw/classification/Train.csv"
LEDGER_PATH = ROOT / "experiments/runs/optimization_v3_research_next_v1/budget.jsonl"
RUN_ROOT = ROOT / "experiments/runs/optimization_v3_a02_continuation_v1"
REPORT_ROOT = ROOT / "reports/optimization_v3/a02_continuation_v1"
SYNTHETIC_QA_PATH = REPORT_ROOT / "synthetic_qa.json"
PREFLIGHT_PATH = REPORT_ROOT / "preflight.json"
CARD_ID = "CLASSIFICATION_A02_NEIGHBOR_RESIDUAL_CONTINUATION"
STAGE = "A02_SIX_FIT_CONTINUATION"
THREADS = 2
JOB_CAP_BYTES = 6 * 1024**3
MIN_FREE_RAM_BYTES = 4 * 1024**3
EXPECTED_TRAIN_SHA256 = "875552a68a3d24b0f3e1966d6ddb5671e6f54305a1c526e2f0a76747942ed91c"
SOURCE_FILES = [
    "src/models/classification_a02_neighbors.py",
    "src/models/classification_models.py",
    "src/models/research_next_budget.py",
    "src/models/research_next_contract.py",
    "src/models/optimization_v3_budget_amendment.py",
    "src/models/optimization_v3_a02_continuation_budget.py",
    "scripts/optimization_v3_a02_neighbors.py",
    "scripts/optimization_v3_a02_neighbors_verify.py",
    "scripts/optimization_v3_a02_continuation.py",
    "scripts/optimization_v3_a02_continuation_verify.py",
    "scripts/optimization_v3_c19_benchmark.py",
    "tests/test_optimization_v3_a02_continuation.py",
]
CACHE_PATHS = pilot.CACHE_PATHS
FIT_IDS = {
    (59, "observed"): "A02_CONT_E59_OBSERVED",
    (59, "shuffled"): "A02_CONT_E59_SHUFFLED",
    (63, "observed"): "A02_CONT_E63_OBSERVED",
    (63, "shuffled"): "A02_CONT_E63_SHUFFLED",
    (71, "observed"): "A02_CONT_E71_OBSERVED",
    (71, "shuffled"): "A02_CONT_E71_SHUFFLED",
}
PILOT_PREDICTION_ROOT = ROOT / "experiments/runs/optimization_v3_a02_neighbors_v1"
PILOT_REPORT_ROOT = ROOT / "reports/optimization_v3/a02_neighbors_v1"


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def _write_once(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    if path.exists():
        if path.read_text(encoding="utf-8") != encoded:
            raise RuntimeError(f"Immutable A02 continuation artifact differs: {path}")
        return
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(encoded)
        stream.flush()
        os.fsync(stream.fileno())


def _read_protocol() -> dict[str, Any]:
    protocol = _json(PROTOCOL_PATH)
    if FREEZE_PATH.exists() and canonical_sha256(protocol) != _json(FREEZE_PATH).get("protocol_sha256"):
        raise RuntimeError("A02 continuation protocol differs from freeze")
    return protocol


def _source_commit() -> str:
    result = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, check=True,
                            capture_output=True, text=True, encoding="utf-8")
    return result.stdout.strip()


def _active_accounting(events: list[dict[str, Any]]) -> tuple[dict[str, int], int, int, int, int]:
    reservations = [e for e in events if e.get("event") == "CARD_FITS_RESERVED"]
    releases = {e.get("reservation_key"): int(e.get("unused_fits", 0))
                for e in events if e.get("event") == "RESERVATION_UNUSED_RELEASED"}
    active: dict[str, int] = {}
    gross = released = 0
    for event in reservations:
        fits = int(event["fits"])
        unused = releases.get(event.get("reservation_key"), 0)
        if unused > fits or unused < 0:
            raise RuntimeError("Budget release exceeds its reservation")
        category = str(event.get("category"))
        active[category] = active.get(category, 0) + fits - unused
        gross += fits
        released += unused
    starts = [e for e in events if e.get("event") == "FIT_STARTED"]
    saved = [e for e in events if e.get("event") == "FIT_SAVED"]
    failed = [e for e in events if e.get("event") == "FIT_FAILED"]
    return active, sum(active.values()), gross, len(starts), len(saved) + len(failed)


def _input_files() -> list[str]:
    paths = [
        "experiments/optimization_v3/c19_a02_v2/A02/protocol.json",
        "experiments/optimization_v3/c19_a02_v2/A02/freeze.json",
        "data/raw/classification/Train.csv",
        "reports/optimization_v3/a02_neighbors_v1/verification.json",
        "reports/optimization_v3/a02_neighbors_v1/metrics_reconstructed.json",
        "reports/optimization_v3/a02_neighbors_v1/probability_metrics_from_saved_predictions.json",
        "reports/optimization_v3/a02_neighbors_v1/logloss_endpoint_diagnostic.json",
        "reports/optimization_v3/a02_neighbors_v1/technical_resume_receipt.json",
    ]
    for cutoff, path in sorted(CACHE_PATHS.items()):
        paths.extend([path.relative_to(ROOT).as_posix(), path.with_suffix(".json").relative_to(ROOT).as_posix()])
    for origin in (67, 75):
        for arm in ("observed", "shuffled"):
            path = PILOT_PREDICTION_ROOT / f"E{origin}/{arm}/predictions.npz"
            paths.extend([path.relative_to(ROOT).as_posix(), path.with_suffix(".json").relative_to(ROOT).as_posix(),
                          (path.parent / "fit_receipt.json").relative_to(ROOT).as_posix()])
    return paths


def build_freeze() -> dict[str, Any]:
    """Create an immutable recipe/source/data freeze before new-origin scores."""
    if FREEZE_PATH.exists():
        return verify_freeze()
    protocol = _json(PROTOCOL_PATH)
    parent_protocol, parent_freeze = pilot._load_frozen()
    if parent_freeze.get("freeze_signature") != "cdac164df87d4f559e36795066cc0c3a91034e5917b0b1fd72cebd1051dae08b":
        raise RuntimeError("Active A02 pilot freeze identity differs")
    if canonical_sha256(parent_protocol) != parent_freeze.get("protocol_sha256"):
        raise RuntimeError("Parent A02 pilot protocol signature differs")
    for cutoff, path in CACHE_PATHS.items():
        if file_sha256(path) != parent_freeze["input_sha256"][path.relative_to(ROOT).as_posix()]:
            raise RuntimeError(f"C16 OOF cache {cutoff} differs from frozen parent inputs")
        side = path.with_suffix(".json")
        if file_sha256(side) != parent_freeze["input_sha256"][side.relative_to(ROOT).as_posix()]:
            raise RuntimeError(f"C16 OOF sidecar {cutoff} differs from frozen parent inputs")
    if file_sha256(TRAIN_PATH) != EXPECTED_TRAIN_SHA256:
        raise RuntimeError("Official Classification Train hash differs")
    source_hashes = {relative: file_sha256(ROOT / relative) for relative in SOURCE_FILES}
    input_hashes = {relative: file_sha256(ROOT / relative) for relative in _input_files()}
    ledger_events = read_events(LEDGER_PATH)
    active, active_total, gross, started, terminal = _active_accounting(ledger_events)
    saved = sum(e.get("event") == "FIT_SAVED" for e in ledger_events)
    failed = sum(e.get("event") == "FIT_FAILED" for e in ledger_events)
    released = sum(int(e.get("unused_fits", 0)) for e in ledger_events
                   if e.get("event") == "RESERVATION_UNUSED_RELEASED")
    checkpoint = {
        "event_rows": len(ledger_events), "ledger_sha256": file_sha256(LEDGER_PATH),
        "gross_reserved": gross, "released_unused": released, "active_reserved": active_total,
        "active_reservations_by_category": {key: active.get(key, 0) for key in
            ("pilot", "continuation", "control_replay", "technical_retry")},
        "started": started, "saved": saved, "failed": failed, "terminal": terminal,
        "pending_started_fits": [e.get("fit_id") for e in ledger_events if e.get("event") == "FIT_STARTED"
            and not any(t.get("fit_id") == e.get("fit_id") and t.get("event") in {"FIT_SAVED", "FIT_FAILED"}
                        for t in ledger_events)],
    }
    if (checkpoint["event_rows"] != 155 or checkpoint["active_reserved"] != 59
            or checkpoint["started"] != 59 or checkpoint["saved"] != 57
            or checkpoint["failed"] != 2 or checkpoint["released_unused"] != 4
            or checkpoint["active_reservations_by_category"] != {
                "pilot": 34, "continuation": 24, "control_replay": 0, "technical_retry": 1}
            or checkpoint["pending_started_fits"]):
        raise RuntimeError(f"Budget checkpoint differs from the authorized amendment precondition: {checkpoint}")
    if not SYNTHETIC_QA_PATH.is_file() or _json(SYNTHETIC_QA_PATH).get("status") != "PASS":
        raise RuntimeError("Synthetic E2E/persistence-resume QA must pass before freeze")
    if protocol.get("origins") != [59, 63, 71] or protocol.get("task_fit_count") != 6:
        raise RuntimeError("A02 continuation protocol origins/fit count differ")
    identity = {
        "candidate_id": protocol["candidate_id"], "source_commit": _source_commit(),
        "parent_freeze_signature": parent_freeze["freeze_signature"],
        "parent_protocol_sha256": parent_freeze["protocol_sha256"],
        "protocol_sha256": canonical_sha256(protocol),
        "source_sha256": source_hashes, "input_sha256": input_hashes,
        "synthetic_qa_sha256": {SYNTHETIC_QA_PATH.relative_to(ROOT).as_posix(): file_sha256(SYNTHETIC_QA_PATH)},
        "budget_checkpoint": checkpoint,
        "frozen_gate_sha256": canonical_sha256(protocol["scientific_gates"]),
        "fit_origins": [59, 63, 71], "fit_ids": protocol["fit_ids"],
        "task_fit_count": 6, "conditional_final_fit_count": 1,
    }
    freeze = {
        "schema_version": 1, "candidate_id": protocol["candidate_id"],
        "protocol_sha256": canonical_sha256(protocol), "identity": identity,
        "source_sha256": source_hashes, "input_sha256": input_hashes,
        "freeze_signature": canonical_sha256(identity),
    }
    _write_once(FREEZE_PATH, freeze)
    return freeze


def verify_freeze() -> dict[str, Any]:
    freeze = _json(FREEZE_PATH)
    protocol = _json(PROTOCOL_PATH)
    if canonical_sha256(protocol) != freeze.get("protocol_sha256"):
        raise RuntimeError("A02 continuation protocol hash mismatch")
    identity = freeze.get("identity", {})
    if canonical_sha256(identity) != freeze.get("freeze_signature"):
        raise RuntimeError("A02 continuation freeze signature mismatch")
    ancestor = subprocess.run(["git", "merge-base", "--is-ancestor", str(identity.get("source_commit", "")), _source_commit()],
                              cwd=ROOT, check=False, capture_output=True)
    if ancestor.returncode != 0:
        raise RuntimeError("A02 continuation source commit is not an ancestor of the current checkout")
    if identity.get("frozen_gate_sha256") != canonical_sha256(protocol["scientific_gates"]):
        raise RuntimeError("A02 continuation gate signature differs from freeze")
    for relative, expected in freeze.get("source_sha256", {}).items():
        if file_sha256(ROOT / relative) != expected:
            raise RuntimeError(f"A02 continuation source hash mismatch: {relative}")
    for relative, expected in freeze.get("input_sha256", {}).items():
        if file_sha256(ROOT / relative) != expected:
            raise RuntimeError(f"A02 continuation input hash mismatch: {relative}")
    for relative, expected in identity.get("synthetic_qa_sha256", {}).items():
        if file_sha256(ROOT / relative) != expected:
            raise RuntimeError("A02 continuation synthetic QA receipt hash mismatch")
    if file_sha256(TRAIN_PATH) != EXPECTED_TRAIN_SHA256:
        raise RuntimeError("A02 continuation official Train hash mismatch")
    parent_protocol, parent_freeze = pilot._load_frozen()
    if canonical_sha256(parent_protocol) != parent_freeze.get("protocol_sha256"):
        raise RuntimeError("A02 parent pilot freeze no longer validates")
    return freeze


def _bank_blocks(origin: int, protocol: dict[str, Any]) -> list[tuple[int, int, int]]:
    return [tuple(map(int, block)) for block in protocol["oof_bank_blocks"][str(origin)]]


def _build_bank(origin: int, caches: dict[int, dict[str, np.ndarray]],
                train: pd.DataFrame, protocol: dict[str, Any]) -> dict[str, np.ndarray]:
    parts = [pilot._cache_subset(caches[cutoff], train, low, high)
             for cutoff, low, high in _bank_blocks(origin, protocol)]
    keys = ("rows", "ids", "cycles", "baseline", "missing")
    joined = {key: np.concatenate([part[key] for part in parts]) for key in keys}
    order = np.argsort(joined["rows"], kind="stable")
    joined = {key: value[order] for key, value in joined.items()}
    expected = np.flatnonzero(train.cycle_code.between(56, origin).to_numpy())
    if not np.array_equal(joined["rows"], expected):
        raise RuntimeError(f"A02 continuation E{origin} bank does not match exact prior OOF blocks")
    if len(np.unique(joined["rows"])) != len(joined["rows"]):
        raise RuntimeError(f"A02 continuation E{origin} bank has overlapping rows")
    if np.any(joined["cycles"] > origin):
        raise RuntimeError(f"A02 continuation E{origin} bank contains future cycles")
    if not np.array_equal(joined["ids"].astype(str), train.record_id.to_numpy(dtype=str)[joined["rows"]]):
        raise RuntimeError(f"A02 continuation E{origin} bank IDs do not align to official Train")
    return joined


def _load_structural_inputs(*, include_target: bool) -> tuple[pd.DataFrame, dict[int, dict[str, np.ndarray]],
                                                              dict[int, dict[str, np.ndarray]],
                                                              dict[int, dict[str, np.ndarray]]]:
    train = pilot._load_train(include_target=include_target)
    caches = {cutoff: pilot._load_cache(cutoff, train) for cutoff in sorted(CACHE_PATHS)}
    protocol = _json(PROTOCOL_PATH)
    banks = {origin: _build_bank(origin, caches, train, protocol) for origin in (59, 63, 71)}
    queries = {origin: pilot._cache_subset(caches[origin], train, origin + 1, origin + 4)
               for origin in (59, 63, 71)}
    return train, caches, banks, queries


def preflight() -> dict[str, Any]:
    freeze = verify_freeze()
    protocol = _json(PROTOCOL_PATH)
    synthetic = _json(SYNTHETIC_QA_PATH)
    if synthetic.get("status") != "PASS" or synthetic.get("fit_type") != "SYNTHETIC_ONLY":
        raise RuntimeError("A02 synthetic end-to-end and recovery QA is not a passing synthetic-only receipt")
    train, caches, banks, queries = _load_structural_inputs(include_target=False)
    cycle = train.cycle_code.to_numpy(dtype=np.int64)
    expected_windows = {"W1": (56, 59), "W2": (60, 63), "W3": (64, 67),
                        "W4": (68, 71), "W5": (72, 75), "W6": (76, 79)}
    row_counts = {name: int(np.sum((cycle >= low) & (cycle <= high)))
                  for name, (low, high) in expected_windows.items()}
    if row_counts != {name: int(protocol["windows"][name]["rows"]) for name in expected_windows}:
        raise RuntimeError(f"A02 continuation official Train window counts differ: {row_counts}")
    if sum(row_counts.values()) != int(protocol["windows"]["full_six_primary"]["unique_rows"]):
        raise RuntimeError("A02 continuation full-six denominator differs")
    max_bank_present = max(int((~bank["missing"]).sum()) for bank in banks.values())
    max_query_rows = max(len(query["rows"]) for query in queries.values())
    analyzer = TfidfVectorizer(**dict(TFIDF_PARAMETERS)).build_analyzer()
    docs = train.signal_stream.fillna("").astype(str).tolist()
    # Conservative per-job upper bound: two prefix CSR copies, one bank CSR,
    # one 128-row dense cosine block, plus 512 MiB for Python/allocator state.
    token_nnz = sum(len(analyzer(doc)) for doc in docs)
    csr_upper = token_nnz * 12 + (len(train) + 1) * 4
    bank_csr_upper = max_bank_present * 160 * 12 + (max_bank_present + 1) * 4
    query_dense = min(128, max_query_rows) * max_bank_present * 8
    estimate = 2 * csr_upper + bank_csr_upper + query_dense + 512 * 1024**2
    virtual = psutil.virtual_memory()
    rss = psutil.Process().memory_info().rss
    disk = __import__("shutil").disk_usage(RUN_ROOT.parent if RUN_ROOT.parent.exists() else ROOT)
    if virtual.available < MIN_FREE_RAM_BYTES or rss > JOB_CAP_BYTES or estimate > JOB_CAP_BYTES or disk.free < 4 * 1024**3:
        raise RuntimeError(
            f"A02 continuation resource preflight failed: available={virtual.available}, "
            f"rss={rss}, estimate={estimate}, disk_free={disk.free}"
        )
    pilot_verification = _json(PILOT_REPORT_ROOT / "verification.json")
    if pilot_verification.get("status") != "PASS":
        raise RuntimeError("Saved A02 pilot verification is not PASS")
    report = {
        "status": "A02_CONTINUATION_PREFLIGHT_PASS", "freeze_signature": freeze["freeze_signature"],
        "protocol_sha256": freeze["protocol_sha256"], "source_commit": freeze["identity"]["source_commit"],
        "official_train_rows": len(train), "official_train_sha256": file_sha256(TRAIN_PATH),
        "target_column_loaded": False, "test_or_sample_loaded": False,
        "verified_c16_cache_sha256": {str(k): file_sha256(v) for k, v in CACHE_PATHS.items()},
        "verified_c16_sidecar_sha256": {str(k): file_sha256(v.with_suffix(".json")) for k, v in CACHE_PATHS.items()},
        "new_origin_bank_rows": {str(k): len(v["rows"]) for k, v in banks.items()},
        "new_origin_bank_present_rows": {str(k): int((~v["missing"]).sum()) for k, v in banks.items()},
        "new_origin_query_rows": {str(k): len(v["rows"]) for k, v in queries.items()},
        "new_origin_query_present_rows": {str(k): int((~v["missing"]).sum()) for k, v in queries.items()},
        "window_rows": row_counts, "new_origin_aggregate_rows": row_counts["W2"] + row_counts["W3"] + row_counts["W5"],
        "full_six_rows": sum(row_counts.values()), "late_rows": row_counts["W5"] + row_counts["W6"],
        "pilot_verification_status": pilot_verification.get("status"),
        "resource_preflight": {
            "one_job_at_a_time": True, "threads": THREADS,
            "per_job_rss_cap_bytes": JOB_CAP_BYTES, "required_free_ram_bytes": MIN_FREE_RAM_BYTES,
            "available_ram_bytes": int(virtual.available), "process_rss_bytes": int(rss),
            "max_bank_present_rows": max_bank_present, "max_query_rows": max_query_rows,
            "tfidf_analyzer_nnz": int(token_nnz), "prefix_csr_upper_bytes": int(csr_upper),
            "bank_csr_upper_bytes": int(bank_csr_upper), "query_dense_block_bytes": int(query_dense),
            "overhead_reserve_bytes": 512 * 1024**2, "estimated_peak_bytes": int(estimate),
            "disk_free_bytes": int(disk.free), "status": "PASS",
        },
        "no_full_query_similarity_matrix": True,
        "similarity_block_rows_max": 128,
    }
    if PREFLIGHT_PATH.exists():
        previous = _json(PREFLIGHT_PATH)
        stable = ("status", "freeze_signature", "protocol_sha256", "source_commit", "official_train_rows",
                  "official_train_sha256", "target_column_loaded", "test_or_sample_loaded",
                  "verified_c16_cache_sha256", "verified_c16_sidecar_sha256", "new_origin_bank_rows",
                  "new_origin_bank_present_rows", "new_origin_query_rows", "new_origin_query_present_rows",
                  "window_rows", "new_origin_aggregate_rows", "full_six_rows", "late_rows",
                  "pilot_verification_status", "no_full_query_similarity_matrix", "similarity_block_rows_max")
        if any(previous.get(key) != report.get(key) for key in stable):
            raise RuntimeError("A02 continuation preflight stable inputs changed")
        return previous
    _write_once(PREFLIGHT_PATH, report)
    return report


def _fit_events(fit_id: str, ledger_path: Path = LEDGER_PATH) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
    events = read_events(ledger_path)
    starts = [e for e in events if e.get("event") == "FIT_STARTED" and e.get("fit_id") == fit_id]
    terminal = [e for e in events if e.get("event") in {"FIT_SAVED", "FIT_FAILED"} and e.get("fit_id") == fit_id]
    if len(starts) > 1 or len(terminal) > 1:
        raise RuntimeError(f"Duplicate A02 continuation fit events for {fit_id}")
    return starts, terminal[0] if terminal else None


def _atomic_joblib(model: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    joblib.dump(model, temporary, compress=3)
    with temporary.open("rb+") as stream:
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def fit_or_resume_model(
    *, origin: int, arm: str, text_space: Any, bank: dict[str, np.ndarray], train: pd.DataFrame,
    freeze: dict[str, Any], folder: Path, ledger_path: Path = LEDGER_PATH,
    fit_id: str | None = None, card_id: str = CARD_ID, stage: str = STAGE,
) -> tuple[ResidualNeighborPredictor, dict[str, Any]]:
    """Fit one counted predictor, or recover its durable state without refitting."""
    fit_id = fit_id or FIT_IDS[(origin, arm)]
    model_path, state_path, receipt_path = (folder / "model.joblib", folder / "fit_state.json",
                                             folder / "fit_receipt.json")
    started, terminal = _fit_events(fit_id, ledger_path)
    if terminal and terminal.get("event") == "FIT_FAILED":
        raise RuntimeError(f"{fit_id} has a consumed failed fit; automatic retry is prohibited")
    bank_mask = ~bank["missing"]
    bank_rows = bank["rows"][bank_mask]
    bank_cycles = bank["cycles"][bank_mask]
    bank_ids = bank["ids"][bank_mask].astype(str)
    bank_probability = bank["baseline"][bank_mask]
    bank_residual = train.artifact_signal.to_numpy(dtype=np.int8)[bank_rows].astype(np.float64) - bank_probability
    if model_path.exists():
        if not started:
            raise RuntimeError(f"{fit_id} has a model without a start event; refusing unaccounted fit state")
        model = joblib.load(model_path)
        state = model.audit_state()
        expected_text_state = text_space.audit_state()
        expected_residual = train.artifact_signal.to_numpy(dtype=np.int8)[bank_rows].astype(np.float64) - bank_probability
        if arm == "shuffled":
            expected_residual, expected_assignment, _ = permute_residuals_within_cycle_probability_decile(
                expected_residual, bank_cycles, bank_probability, bank_rows, seed=42,
            )
        else:
            expected_assignment = np.arange(len(bank_rows), dtype=np.int64)
        if (state.get("origin_cycle") != origin or state.get("residual_arm") != arm
                or state.get("bank_rows") != len(bank_rows)
                or model.text_space.audit_state() != expected_text_state
                or not np.array_equal(model.bank_row_positions, bank_rows)
                or [str(value).split(":", 1)[-1] for value in model.bank_ids] != bank_ids.tolist()
                or not np.array_equal(model.bank_cycles, bank_cycles)
                or not np.array_equal(model.bank_oof_probabilities, bank_probability)
                or not np.array_equal(model.residual_assignment, expected_assignment)
                or not np.array_equal(model.bank_residuals, expected_residual)):
            raise RuntimeError(f"{fit_id} durable model identity mismatch")
        if state_path.exists() and _json(state_path) != state:
            raise RuntimeError(f"{fit_id} model state differs from saved audit state")
        _write_once(state_path, state)
        if receipt_path.exists():
            receipt = _json(receipt_path)
        else:
            receipt = _fit_receipt(fit_id, origin, arm, model_path, state, bank_rows,
                                   bank_cycles, text_space, freeze, None, True)
            _write_once(receipt_path, receipt)
        if receipt.get("model_sha256") != file_sha256(model_path) or receipt.get("state_sha256") != canonical_sha256(state):
            raise RuntimeError(f"{fit_id} durable model receipt mismatch")
        check = dict(receipt)
        receipt_hash = check.pop("receipt_sha256", None)
        if receipt_hash != canonical_sha256(check):
            raise RuntimeError(f"{fit_id} receipt signature mismatch")
        if terminal is None:
            finish_fit(ledger_path, fit_id=fit_id, status="FIT_SAVED", receipt_sha256=receipt_hash)
        return model, receipt
    if started or terminal is not None:
        raise RuntimeError(f"{fit_id} has a consumed fit without durable model; no retry is authorized")
    if text_space.fit_rows != int(np.sum(train.cycle_code.to_numpy(dtype=np.int64) <= origin)):
        raise RuntimeError(f"{fit_id} TF-IDF prefix population differs from the frozen origin")
    virtual = psutil.virtual_memory()
    process_rss = psutil.Process().memory_info().rss
    if virtual.available < MIN_FREE_RAM_BYTES or process_rss > JOB_CAP_BYTES:
        raise RuntimeError(f"Live memory guard failed before {fit_id}: available={virtual.available}, rss={process_rss}")
    start_fit(ledger_path, fit_id=fit_id, card_id=card_id, stage=stage,
              cutoff=origin, component=f"TEXT_NEIGHBOR_RESIDUAL_{arm.upper()}",
              freeze_signature=freeze["freeze_signature"])
    model = ResidualNeighborPredictor(origin_cycle=origin, residual_arm=arm)
    try:
        with threadpool_limits(limits=THREADS), pilot._PeakRSS() as monitor:
            model.fit(
                text_space=text_space,
                bank_texts=train.loc[bank_rows, "signal_stream"].reset_index(drop=True),
                bank_record_ids=bank_ids, bank_row_positions=bank_rows,
                bank_cycles=bank_cycles, bank_oof_probabilities=bank_probability,
                bank_oof_residuals=bank_residual,
            )
        state = model.audit_state()
        expected_residual = bank_residual
        if arm == "shuffled":
            expected_residual, expected_assignment, _ = permute_residuals_within_cycle_probability_decile(
                bank_residual, bank_cycles, bank_probability, bank_rows, seed=42,
            )
        else:
            expected_assignment = np.arange(len(bank_rows), dtype=np.int64)
        if (state.get("bank_rows") != len(bank_rows) or monitor.peak > JOB_CAP_BYTES
                or state.get("origin_cycle") != origin or state.get("residual_arm") != arm
                or model.text_space.audit_state() != text_space.audit_state()
                or not np.array_equal(model.bank_row_positions, bank_rows)
                or [str(value).split(":", 1)[-1] for value in model.bank_ids] != bank_ids.tolist()
                or not np.array_equal(model.bank_cycles, bank_cycles)
                or not np.array_equal(model.bank_oof_probabilities, bank_probability)
                or not np.array_equal(model.residual_assignment, expected_assignment)
                or not np.array_equal(model.bank_residuals, expected_residual)):
            raise RuntimeError(f"{fit_id} fitted state or RSS cap failed")
        _atomic_joblib(model, model_path)
        _write_once(state_path, state)
        receipt = _fit_receipt(fit_id, origin, arm, model_path, state, bank_rows,
                               bank_cycles, text_space, freeze, monitor.peak, False)
        _write_once(receipt_path, receipt)
        finish_fit(ledger_path, fit_id=fit_id, status="FIT_SAVED",
                   receipt_sha256=receipt["receipt_sha256"])
        return model, receipt
    except Exception as exc:
        current_started, current_terminal = _fit_events(fit_id, ledger_path)
        if current_started and current_terminal is None and not model_path.exists():
            finish_fit(ledger_path, fit_id=fit_id, status="FIT_FAILED", error=type(exc).__name__)
        raise


def _fit_receipt(fit_id: str, origin: int, arm: str, model_path: Path,
                 state: dict[str, Any], bank_rows: np.ndarray, bank_cycles: np.ndarray,
                 text_space: Any, freeze: dict[str, Any], peak_rss: int | None,
                 recovered: bool) -> dict[str, Any]:
    receipt: dict[str, Any] = {
        "fit_id": fit_id, "candidate_id": CARD_ID, "origin_cycle": int(origin),
        "residual_arm": arm, "freeze_signature": freeze["freeze_signature"],
        "model_sha256": file_sha256(model_path), "state_sha256": canonical_sha256(state),
        "bank_rows": int(len(bank_rows)), "bank_present_rows": int(len(bank_rows)),
        "bank_cycle_min": int(bank_cycles.min()), "bank_cycle_max": int(bank_cycles.max()),
        "bank_row_positions_sha256": state["bank_row_positions_sha256"],
        "text_prefix_rows": int(text_space.fit_rows),
        "text_prefix_row_positions_sha256": text_space.fit_row_positions_sha256,
        "observed_peak_rss_bytes": int(peak_rss) if peak_rss is not None else None,
        "task_estimator_fit_count": 1, "persistence_recovered_without_refit": bool(recovered),
        "preprocessing_fit_population": f"official Train feature-text prefix cycles 47-{origin}; no target values",
        "residual_fit_population": f"present-stream C16 baseline OOF residuals from declared four-cycle blocks through {origin}",
        "query_labels_passed_to_fit": False, "test_data_used": False,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    receipt["receipt_sha256"] = canonical_sha256(receipt)
    return receipt


def _prediction_arrays(model: ResidualNeighborPredictor, query: dict[str, np.ndarray],
                       train: pd.DataFrame, origin: int) -> tuple[dict[str, np.ndarray], list[float], dict[str, Any]]:
    rows = query["rows"]
    text = train.loc[rows, "signal_stream"].reset_index(drop=True)
    groups, edges = pilot._text_load_groups(train, origin, rows)
    with threadpool_limits(limits=THREADS):
        output = model.predict(query_texts=text, query_record_ids=query["ids"].astype(str),
                               query_row_positions=rows, query_cycles=query["cycles"],
                               baseline_probabilities=query["baseline"], missing_mask=query["missing"])
    probability = output["candidate_probability"].astype(np.float64)
    if (len(probability) != len(rows) or not np.isfinite(probability).all()
            or np.any((probability < 0.0) | (probability > 1.0))
            or not np.array_equal(probability[query["missing"]], query["baseline"][query["missing"]])):
        raise RuntimeError(f"A02 E{origin}/{model.residual_arm} probability route failed")
    arrays = {
        "score_rows": rows.astype(np.int64),
        "score_record_ids": np.asarray(query["ids"], dtype=str),
        "score_cycles": query["cycles"].astype(np.int16),
        "baseline_prediction": query["baseline"].astype(np.float64),
        "prediction": probability,
        "labels": (probability >= 0.5).astype(np.int8),
        "missing_mask": query["missing"].astype(bool),
        "entry_friction": train.loc[rows, "entry_friction"].to_numpy(dtype=np.int8),
        "text_load_group": groups.astype(np.int8),
        "neighbor_count": output["neighbor_count"].astype(np.int16),
        "neighbor_source_rows": output["neighbor_source_rows"].astype(np.int64),
        "neighbor_ids": output["neighbor_ids"].astype(str),
        "neighbor_similarities": output["neighbor_similarities"].astype(np.float64),
        "correction": output["correction"].astype(np.float64),
        "correction_unclipped": output["correction_unclipped"].astype(np.float64),
    }
    return arrays, edges, output["diagnostics"]


def _arrays_equal(left: dict[str, np.ndarray], right: dict[str, np.ndarray]) -> bool:
    return set(left) == set(right) and all(np.array_equal(left[key], right[key]) for key in left)


def write_predictions_once(origin: int, arm: str, model: ResidualNeighborPredictor,
                           query: dict[str, np.ndarray], train: pd.DataFrame,
                           receipt: dict[str, Any], freeze: dict[str, Any],
                           *, output_root: Path = RUN_ROOT) -> Path:
    folder = output_root / f"E{origin}" / arm
    path, side_path = folder / "predictions.npz", folder / "predictions.json"
    folder.mkdir(parents=True, exist_ok=True)
    arrays, edges, diagnostics = _prediction_arrays(model, query, train, origin)
    if side_path.exists() and not path.exists():
        raise RuntimeError(f"A02 E{origin}/{arm} sidecar exists without NPZ; refusing rewrite")
    if path.exists():
        if side_path.exists() and _json(side_path).get("npz_sha256") != file_sha256(path):
            raise RuntimeError(f"A02 E{origin}/{arm} NPZ bytes disagree with sidecar")
        with np.load(path, allow_pickle=False) as loaded:
            prior = {key: np.asarray(loaded[key]).copy() for key in loaded.files}
        if not _arrays_equal(prior, arrays):
            raise RuntimeError(f"A02 E{origin}/{arm} saved predictions differ from deterministic reconstruction")
    else:
        temporary = path.with_name(path.name + ".tmp")
        with temporary.open("wb") as stream:
            np.savez_compressed(stream, **arrays)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    if not side_path.exists():
        ids_payload = json.dumps(arrays["score_record_ids"].tolist(), separators=(",", ":"),
                                 ensure_ascii=False).encode("utf-8")
        sidecar = {
            "fit_id": receipt["fit_id"], "fit_receipt_sha256": receipt["receipt_sha256"],
            "origin_cycle": int(origin), "residual_arm": arm,
            "freeze_signature": freeze["freeze_signature"], "npz_sha256": file_sha256(path),
            "score_rows": int(len(arrays["score_rows"])),
            "score_row_positions_sha256": hashlib.sha256(arrays["score_rows"].astype("<i8").tobytes()).hexdigest(),
            "score_record_ids_sha256": hashlib.sha256(ids_payload).hexdigest(),
            "text_load_quartile_fit_prefix_edges": edges, "diagnostics": diagnostics,
            "query_labels_passed_to_prediction": False, "test_data_used": False,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
        }
        _write_once(side_path, sidecar)
    return path


def run_continuation() -> dict[str, Any]:
    freeze = verify_freeze()
    prep = preflight()
    if prep.get("status") != "A02_CONTINUATION_PREFLIGHT_PASS":
        raise RuntimeError("A02 continuation preflight is not PASS")
    apply_cap_amendment(LEDGER_PATH, freeze=freeze,
                        source_commit=freeze["identity"]["source_commit"])
    reserve(LEDGER_PATH, card_id=CARD_ID, stage=STAGE, fits=6,
            freeze_signature=freeze["freeze_signature"])
    train, caches, banks, queries = _load_structural_inputs(include_target=True)
    cycle_all = train.cycle_code.to_numpy(dtype=np.int64)
    for origin in (59, 63, 71):
        prefix_rows = np.flatnonzero(cycle_all <= origin)
        text_space, prefix_matrix, _ = fit_text_space(
            train.loc[prefix_rows, "signal_stream"].reset_index(drop=True),
            prefix_rows, cycle_all[prefix_rows], origin_cycle=origin,
        )
        del prefix_matrix
        for arm in ("observed", "shuffled"):
            model, receipt = fit_or_resume_model(
                origin=origin, arm=arm, text_space=text_space, bank=banks[origin],
                train=train, freeze=freeze, folder=RUN_ROOT / f"E{origin}" / arm,
            )
            write_predictions_once(origin, arm, model, queries[origin], train, receipt, freeze)
            del model
            gc.collect()
        del text_space
        gc.collect()
    verifier = __import__("scripts.optimization_v3_a02_continuation_verify", fromlist=["verify"])
    result = verifier.verify(write_outputs=True)
    return result


def run_final_inference() -> dict[str, Any]:
    """One gated full-Train fit and official-Test inference; never called before PASS."""
    freeze = verify_freeze()
    result_path = REPORT_ROOT / "results.json"
    verification_path = REPORT_ROOT / "verification.json"
    if not result_path.is_file() or not verification_path.is_file():
        raise RuntimeError("A02 final inference requires complete continuation evaluation/verification")
    results, verification = _json(result_path), _json(verification_path)
    gates = results.get("gates", {})
    if (verification.get("status") != "PASS" or gates.get("scientific_status") != "PASS"
            or gates.get("risk_review_hold") is not False):
        raise RuntimeError("A02 continuation gates/risk checks do not authorize final inference")
    final_card, final_stage = CARD_ID, "A02_CONTINUATION_FINAL_FULLTRAIN"
    final_fit_id = "A02_CONT_E79_OBSERVED_FULLTRAIN"
    # Only now load official Test features/IDs and SampleSubmission for this one inference.
    from scripts import optimization_v3_c19_benchmark as c19
    train, test, sample = c19._load_train_test_sample()
    baseline = c19._load_baseline_cache(test, sample)
    train, caches, banks, _ = _load_structural_inputs(include_target=True)
    # Six complete OOF blocks, only through origin 75, are used as the residual bank.
    blocks = [(55, 56, 59), (59, 60, 63), (63, 64, 67),
              (67, 68, 71), (71, 72, 75), (75, 76, 79)]
    pieces = [pilot._cache_subset(caches[cutoff], train, low, high)
              for cutoff, low, high in blocks]
    joined = {key: np.concatenate([p[key] for p in pieces])
              for key in ("rows", "ids", "cycles", "baseline", "missing")}
    order = np.argsort(joined["rows"], kind="stable")
    bank = {key: value[order] for key, value in joined.items()}
    expected_rows = np.flatnonzero(train.cycle_code.between(56, 79).to_numpy())
    if not np.array_equal(bank["rows"], expected_rows) or np.any(bank["cycles"] > 79):
        raise RuntimeError("A02 final residual bank is not the six complete C16 OOF blocks")
    virtual = psutil.virtual_memory()
    if virtual.available < MIN_FREE_RAM_BYTES or psutil.Process().memory_info().rss > JOB_CAP_BYTES:
        raise RuntimeError("A02 final conditional fit resource guard failed")
    prefix_rows = np.arange(len(train), dtype=np.int64)
    cycle_all = train.cycle_code.to_numpy(dtype=np.int64)
    text_space, prefix_matrix, _ = fit_text_space(
        train.signal_stream.reset_index(drop=True), prefix_rows, cycle_all, origin_cycle=79,
    )
    del prefix_matrix
    virtual = psutil.virtual_memory()
    if virtual.available < MIN_FREE_RAM_BYTES or psutil.Process().memory_info().rss > JOB_CAP_BYTES:
        raise RuntimeError("A02 final post-preprocessing resource guard failed")
    reserve(LEDGER_PATH, card_id=final_card, stage=final_stage, fits=1,
            freeze_signature=freeze["freeze_signature"])
    folder = RUN_ROOT / "E79/fulltrain_observed"
    model, receipt = fit_or_resume_model(
        origin=79, arm="observed", text_space=text_space, bank=bank, train=train,
        freeze=freeze, folder=folder, fit_id=final_fit_id, card_id=final_card, stage=final_stage,
    )
    test_cycle = test.cycle_code.to_numpy(dtype=np.int64)
    test_rows = np.arange(len(test), dtype=np.int64) + len(train)
    query = {
        "rows": test_rows, "ids": test.record_id.astype(str).to_numpy(dtype=str),
        "cycles": test_cycle, "baseline": baseline["probability"].astype(np.float64),
        "missing": baseline["missing_mask"].astype(bool),
    }
    output = model.predict(query_texts=test.signal_stream.reset_index(drop=True),
                           query_record_ids=query["ids"], query_row_positions=query["rows"],
                           query_cycles=query["cycles"], baseline_probabilities=query["baseline"],
                           missing_mask=query["missing"])
    probability = output["candidate_probability"].astype(np.float64)
    labels = (probability >= 0.5).astype(np.int8)
    if (len(labels) != 29421 or sample.record_id.astype(str).tolist() != test.record_id.astype(str).tolist()
            or not np.isfinite(probability).all() or np.any((probability < 0) | (probability > 1))
            or not np.array_equal(probability[query["missing"]], query["baseline"][query["missing"]])):
        raise RuntimeError("A02 official Test route or ordered IDs failed; do not refit")
    pred_path = folder / "test_predictions.npz"
    pred_receipt_path = folder / "test_predictions.json"
    arrays = {"record_ids": query["ids"], "baseline_probability": query["baseline"],
              "missing_mask": query["missing"], "candidate_probability": probability,
              "labels": labels, "neighbor_count": output["neighbor_count"],
              "neighbor_source_rows": output["neighbor_source_rows"],
              "neighbor_ids": output["neighbor_ids"], "neighbor_similarities": output["neighbor_similarities"],
              "correction": output["correction"], "correction_unclipped": output["correction_unclipped"]}
    if pred_path.exists() or pred_receipt_path.exists():
        if not pred_path.exists():
            raise RuntimeError("A02 final prediction receipt exists without NPZ; no overwrite/refit")
        with np.load(pred_path, allow_pickle=False) as saved:
            if not _arrays_equal({k: np.asarray(saved[k]).copy() for k in saved.files}, arrays):
                raise RuntimeError("A02 final prediction bundle differs; no overwrite/refit")
        if pred_receipt_path.exists():
            saved_receipt = _json(pred_receipt_path)
            if (saved_receipt.get("npz_sha256") != file_sha256(pred_path)
                    or saved_receipt.get("fit_receipt_sha256") != receipt["receipt_sha256"]
                    or saved_receipt.get("freeze_signature") != freeze["freeze_signature"]):
                raise RuntimeError("A02 final prediction receipt differs; no overwrite/refit")
        else:
            _write_once(pred_receipt_path, {
                "fit_receipt_sha256": receipt["receipt_sha256"], "freeze_signature": freeze["freeze_signature"],
                "npz_sha256": file_sha256(pred_path), "rows": len(labels),
                "official_test_targets_loaded": False, "sample_submission_used_for_inference": False,
            })
    else:
        folder.mkdir(parents=True, exist_ok=True)
        temporary = pred_path.with_name(pred_path.name + ".tmp")
        with temporary.open("wb") as stream:
            np.savez_compressed(stream, **arrays)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, pred_path)
        _write_once(pred_receipt_path, {
            "fit_receipt_sha256": receipt["receipt_sha256"], "freeze_signature": freeze["freeze_signature"],
            "npz_sha256": file_sha256(pred_path), "rows": len(labels),
            "official_test_targets_loaded": False, "sample_submission_used_for_inference": False,
        })
    csv_path = ROOT / "submissions/optimization_v3/a02_continuation_v1/a02_continuation.csv"
    manifest_path = REPORT_ROOT / "submission_manifest.json"
    expected_frame = pd.DataFrame({"record_id": query["ids"], "artifact_signal": labels})
    if csv_path.exists():
        prior = pd.read_csv(csv_path, dtype={"record_id": str})
        if (prior.columns.tolist() != expected_frame.columns.tolist()
                or prior.record_id.astype(str).tolist() != expected_frame.record_id.astype(str).tolist()
                or not np.array_equal(prior.artifact_signal.to_numpy(dtype=np.int8), labels)):
            raise RuntimeError("A02 continuation CSV exists but differs from deterministic inference")
    else:
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        temporary_csv = csv_path.with_name(csv_path.name + ".tmp")
        expected_frame.to_csv(temporary_csv, index=False)
        with temporary_csv.open("rb+") as stream:
            os.fsync(stream.fileno())
        os.replace(temporary_csv, csv_path)
    check = pd.read_csv(csv_path, dtype={"record_id": str})
    if (check.columns.tolist() != sample.columns.tolist() or len(check) != 29421
            or check.record_id.astype(str).tolist() != sample.record_id.astype(str).tolist()
            or check.record_id.isna().any() or check.record_id.duplicated().any()
            or not np.isin(check.artifact_signal.to_numpy(), [0, 1]).all()):
        raise RuntimeError("A02 continuation official CSV verification failed; no refit")
    manifest = {
        "status": "PREPARED_VALIDATED_NOT_SUBMITTED", "candidate_id": "A02_LEXICAL_NEIGHBOR_RESIDUAL_CONTINUATION",
        "submission_authorized": False, "source_commit": freeze["identity"]["source_commit"],
        "protocol_sha256": freeze["protocol_sha256"], "freeze_signature": freeze["freeze_signature"],
        "official_train_rows": len(train), "official_train_sha256": file_sha256(TRAIN_PATH),
        "fit_id": final_fit_id, "fit_receipt_sha256": receipt["receipt_sha256"],
        "model_sha256": file_sha256(folder / "model.joblib"), "fit_state_sha256": canonical_sha256(model.audit_state()),
        "OOF_bank_blocks": blocks, "OOF_bank_rows": len(bank["rows"]),
        "OOF_bank_present_rows": int((~bank["missing"]).sum()), "OOF_bank_target_residual_only": True,
        "base_cache_sha256": file_sha256(c19.BASE_CACHE), "base_manifest_sha256": file_sha256(c19.BASE_MANIFEST),
        "base_route": "C16 baseline_probability from verified full-Train sparse25 cache; no C16 refit",
        "test_rows": len(test), "test_feature_sha256": file_sha256(c19.TEST_PATH),
        "sample_submission_sha256": file_sha256(c19.SAMPLE_PATH),
        "test_target_loaded": False, "sample_submission_used_for_schema_only": True,
        "route": "threshold 0.50; exact baseline on missing stream; present = clipped baseline + A02 correction",
        "csv_path": csv_path.relative_to(ROOT).as_posix(), "csv_sha256": file_sha256(csv_path),
        "prediction_bundle_sha256": file_sha256(pred_path), "rows": len(check),
        "columns": check.columns.tolist(),
        "label_counts": {str(int(k)): int(v) for k, v in check.artifact_signal.value_counts().sort_index().items()},
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    if manifest_path.exists():
        prior_manifest = _json(manifest_path)
        for key, value in manifest.items():
            if key != "created_at_utc" and prior_manifest.get(key) != value:
                raise RuntimeError(f"A02 continuation existing manifest differs: {key}")
        manifest = prior_manifest
    else:
        _write_once(manifest_path, manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--synthetic-qa", action="store_true")
    parser.add_argument("--freeze", action="store_true")
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--final-inference", action="store_true")
    args = parser.parse_args()
    selected = sum((args.synthetic_qa, args.freeze, args.preflight, args.run, args.final_inference))
    if selected != 1:
        parser.error("choose exactly one mode")
    if args.synthetic_qa:
        from scripts.optimization_v3_a02_continuation_verify import run_synthetic_qa
        value = run_synthetic_qa()
    elif args.freeze:
        value = build_freeze()
    elif args.preflight:
        value = preflight()
    elif args.run:
        value = run_continuation()
    else:
        value = run_final_inference()
    print(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
