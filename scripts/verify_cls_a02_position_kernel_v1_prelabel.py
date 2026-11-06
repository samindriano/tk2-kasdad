from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
from typing import Any

import joblib
import numpy as np
import pandas as pd
from scipy import sparse
from threadpoolctl import threadpool_limits

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.models.a02_position_kernel_v1 import load_npz_allowlisted


OFFICIAL_ROOT = Path(os.environ.get("KASDAD_TP2_SOURCE_ROOT", REPO_ROOT.parent / "TP 2")).resolve()
REPORT_ROOT = REPO_ROOT / "reports/simplification_v1/CLS_POST_BAG2_NEXT_V1"
RUN_ROOT = REPO_ROOT / "experiments/runs/simplification_v1/CLS_POST_BAG2_NEXT_V1"
FREEZE_PATH = REPORT_ROOT / "FREEZE.json"
LEDGER_PATH = REPORT_ROOT / "BUDGET_LEDGER.jsonl"
OUTPUT_PATH = REPORT_ROOT / "PRELABEL_VERIFICATION.json"
TRAIN_PATH = OFFICIAL_ROOT / "data/raw/classification/Train.csv"
POS_ROOT = OFFICIAL_ROOT / "experiments/runs/optimization_v3_research_next_v1/POSITIONAL_TEXT_PILOT_V1"
ORIGINS = (67, 75)
ARMS = ("REAL", "SHUFFLED")
FIT_IDS = tuple(f"E{o}_{a}" for o in ORIGINS for a in ARMS)
K = 32
FLOOR = 0.5
SHRINKAGE = 4.0
CORRECTION_LIMIT = 0.05
THREADS = 2
INITIAL_RAM_MIN = 1024**3
RESERVE_BYTES = 512 * 1024**2
BLOCK_ROWS = 16

A02_MEMBERS = (
    "score_rows", "score_record_ids", "score_cycles", "baseline_prediction", "prediction",
    "missing_mask", "entry_friction", "text_load_group", "neighbor_count",
    "neighbor_source_rows", "neighbor_ids", "neighbor_similarities", "correction",
    "correction_unclipped",
)
POS_MEMBERS = ("score_rows", "score_record_ids", "score_cycles", "present_mask", "prob_position")
OUTPUT_MEMBERS = (
    "score_rows", "score_record_ids", "score_cycles", "missing_mask", "entry_friction",
    "text_load_group", "p_sparse25", "p_a02_original", "p_pos_original",
    "p_a02_candidate", "champion_score", "candidate_score", "champion_class",
    "candidate_class", "m0", "neighbor_count", "neighbor_source_rows", "neighbor_ids",
    "neighbor_kernel_similarity", "neighbor_weights", "position_kernel_positive_mass",
    "correction_raw", "correction", "mass_error", "neighbor_overlap_fraction",
)


class PrelabelError(RuntimeError):
    pass


def sha_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _committed_and_clean(path: Path) -> bool:
    try:
        rel = path.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return False
    tracked = subprocess.run(["git", "ls-files", "--error-unmatch", rel], cwd=REPO_ROOT,
                             text=True, capture_output=True).returncode == 0
    clean = subprocess.run(["git", "diff", "--quiet", "HEAD", "--", rel], cwd=REPO_ROOT,
                           text=True, capture_output=True).returncode == 0
    return tracked and clean


def _lock_wait() -> None:
    import msvcrt
    lock = Path(tempfile.gettempdir()) / "kasdad_single_heavy_job.lock"
    lock.parent.mkdir(parents=True, exist_ok=True)
    with lock.open("a+b") as stream:
        stream.seek(0, os.SEEK_END)
        if stream.tell() == 0:
            stream.write(b"\0")
            stream.flush()
        while True:
            stream.seek(0)
            try:
                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
                break
            except OSError:
                print("Waiting for KASDAD shared lock before prelabel reconstruction", flush=True)
                time.sleep(5)
        try:
            yield
        finally:
            stream.seek(0)
            msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)


def _shared_lock_context():
    # Windows runner is the target host for this scoped local pilot.
    from contextlib import contextmanager
    return contextmanager(_lock_wait)()


def _npz_output(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as z:
        names = set(z.files)
        if names != set(OUTPUT_MEMBERS):
            raise PrelabelError(f"Output schema/target-member check failed for {path}: {sorted(names ^ set(OUTPUT_MEMBERS))}")
        if names & {"labels", "artifact_signal", "y_true", "target", "threat_rank"}:
            raise PrelabelError(f"Candidate prediction output contains a target member: {path}")
        return {name: z[name].copy() for name in OUTPUT_MEMBERS}


def _top32(sims: np.ndarray, source_rows: np.ndarray) -> np.ndarray:
    positive = np.flatnonzero(sims > 0.0)
    if not len(positive):
        return np.empty(0, dtype=np.int64)
    take = min(K, len(positive))
    values = sims[positive]
    if len(positive) > take:
        threshold = np.partition(values, len(values) - take)[len(values) - take]
        above = positive[values > threshold]
        tied = positive[values == threshold]
        need = take - len(above)
        tied = tied[np.argsort(source_rows[tied], kind="stable")]
        chosen = np.concatenate([above, tied[:need]])
    else:
        chosen = positive
    return chosen[np.lexsort((source_rows[chosen], -sims[chosen]))].astype(np.int64)


def _route(p_pos: np.ndarray, p_a02: np.ndarray, missing: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    probability = p_a02.copy()
    present = ~missing
    probability[present] = 0.5 * p_pos[present] + 0.5 * p_a02[present]
    predicted = np.zeros(len(probability), dtype=np.int8)
    predicted[present] = (probability[present] >= 0.50).astype(np.int8)
    predicted[missing] = (p_a02[missing] >= 0.53230976129477).astype(np.int8)
    return probability, predicted


def _verify_source_hashes(freeze: dict[str, Any]) -> None:
    for path_text, expected in freeze["source_hashes"].items():
        if path_text.startswith("OFFICIAL::"):
            path = OFFICIAL_ROOT / path_text.removeprefix("OFFICIAL::")
        else:
            path = REPO_ROOT / path_text
        if not path.is_file() or sha_file(path) != expected:
            raise PrelabelError(f"Frozen source changed: {path_text}")
    for path_text, expected in freeze["input_hashes"].items():
        if not path_text.startswith("OFFICIAL::"):
            raise PrelabelError(f"Unexpected nonofficial input path in freeze: {path_text}")
        path = OFFICIAL_ROOT / path_text.removeprefix("OFFICIAL::")
        if not path.is_file() or sha_file(path) != expected:
            raise PrelabelError(f"Frozen input changed: {path_text}")


def _verify_unit(origin: int, arm: str, freeze: dict[str, Any], frame: pd.DataFrame) -> dict[str, Any]:
    from src.models.classification_a02_neighbors import _documents
    from src.models.optimization_v3_positional_text_pilot_v1 import build_position_block

    unit_id = f"E{origin}_{arm}"
    bundle_path = RUN_ROOT / unit_id / "predictions.npz"
    receipt_path = RUN_ROOT / unit_id / "receipt.json"
    if not bundle_path.is_file() or not receipt_path.is_file():
        raise PrelabelError(f"Durable output missing for {unit_id}")
    receipt = load_json(receipt_path)
    arrays = _npz_output(bundle_path)
    if receipt.get("fit_id") != unit_id or receipt.get("artifact_sha256") != sha_file(bundle_path):
        raise PrelabelError(f"Output receipt mismatch for {unit_id}")
    if (receipt.get("lineage", {}).get("source_commit") != freeze.get("source_commit")
            or receipt.get("lineage", {}).get("official_source_commit") != freeze.get("official_source_commit")
            or receipt.get("lineage", {}).get("input_hashes") != freeze.get("input_hashes")
            or receipt.get("lineage", {}).get("source_hashes") != freeze.get("source_hashes")
            or receipt.get("lineage", {}).get("origin") != origin
            or receipt.get("lineage", {}).get("arm") != arm):
        raise PrelabelError(f"Output lineage does not match freeze for {unit_id}")
    if (receipt.get("query_target_included") is not False
            or receipt.get("estimator_fits") != 0
            or receipt.get("preprocessing_fits") != 0
            or receipt.get("output_members") != sorted(OUTPUT_MEMBERS)):
        raise PrelabelError(f"Output receipt contract differs from target-free predictor construction for {unit_id}")

    base = OFFICIAL_ROOT / f"experiments/runs/optimization_v3_a02_neighbors_v1/E{origin}/observed"
    a02_manifest = load_json(base / "predictions.json")
    fit_state = load_json(base / "fit_state.json")
    fit_receipt = load_json(base / "fit_receipt.json")
    pos_verify = load_json(POS_ROOT / f"PREDICTION_VERIFICATION_E{origin}.json")
    a02_npz = load_npz_allowlisted(base / "predictions.npz", allowed_members=A02_MEMBERS)
    pos_path = POS_ROOT / f"E{origin}/predictions_E{origin}.npz"
    if pos_verify.get("prediction_file_sha256") != sha_file(pos_path):
        raise PrelabelError(f"Original POS prediction file hash mismatch for {unit_id}")
    pos_npz = load_npz_allowlisted(
        pos_path, allowed_members=POS_MEMBERS,
        expected_sha256=pos_verify["prediction_file_sha256"], allow_closed_extras=True,
    )
    model = joblib.load(base / "model.joblib")
    if model.audit_state() != fit_state or model.origin_cycle != origin:
        raise PrelabelError(f"Historical A02 saved state mismatch for {unit_id}")
    if fit_receipt.get("model_sha256") != sha_file(base / "model.joblib"):
        raise PrelabelError(f"Historical A02 model hash mismatch for {unit_id}")
    if a02_manifest.get("npz_sha256") != sha_file(base / "predictions.npz"):
        raise PrelabelError(f"Historical A02 prediction hash mismatch for {unit_id}")
    if pos_verify.get("status") != "PASS_PREDICTION_INTEGRITY":
        raise PrelabelError(f"Original POS cache did not pass its frozen verification for {unit_id}")

    qrows = np.asarray(a02_npz["score_rows"], dtype=np.int64)
    qids = a02_npz["score_record_ids"].astype(str)
    qcycles = np.asarray(a02_npz["score_cycles"], dtype=np.int64)
    if not np.array_equal(qrows, pos_npz["score_rows"]) or not np.array_equal(qids, pos_npz["score_record_ids"].astype(str)):
        raise PrelabelError(f"A02/POS cache row identity mismatch for {unit_id}")
    if not np.array_equal(qcycles, pos_npz["score_cycles"].astype(np.int64)):
        raise PrelabelError(f"A02/POS cache cycle identity mismatch for {unit_id}")
    if not np.array_equal(arrays["score_rows"], qrows) or not np.array_equal(arrays["score_record_ids"].astype(str), qids):
        raise PrelabelError(f"Candidate query identity mismatch for {unit_id}")
    if not np.array_equal(arrays["score_cycles"].astype(np.int64), qcycles):
        raise PrelabelError(f"Candidate query cycles mismatch for {unit_id}")
    if not np.array_equal(arrays["missing_mask"], a02_npz["missing_mask"]):
        raise PrelabelError(f"Candidate missing route differs for {unit_id}")
    if not np.array_equal(arrays["entry_friction"], a02_npz["entry_friction"]):
        raise PrelabelError(f"entry_friction subgroup binding changed for {unit_id}")
    if not np.array_equal(arrays["text_load_group"], a02_npz["text_load_group"]):
        raise PrelabelError(f"fit-defined text_load subgroup binding changed for {unit_id}")
    if not np.array_equal(pos_npz["present_mask"], ~arrays["missing_mask"]):
        raise PrelabelError(f"Original POS availability routing changed for {unit_id}")

    bank_rows = np.asarray(model.bank_row_positions, dtype=np.int64)
    bank_cycles = np.asarray(model.bank_cycles, dtype=np.int64)
    bank_raw = frame.iloc[bank_rows]
    query_raw = frame.iloc[qrows]
    if tuple(f"str:{str(x)}" for x in bank_raw["record_id"].tolist()) != tuple(model.bank_ids):
        raise PrelabelError(f"Historical bank ID-to-Train row binding failed for {unit_id}")
    if not np.array_equal(bank_raw["cycle_code"].to_numpy(dtype=np.int64), bank_cycles):
        raise PrelabelError(f"Historical bank cycle-to-Train binding failed for {unit_id}")
    if not np.array_equal(query_raw["record_id"].astype("string").fillna("").astype(str).to_numpy().astype(qids.dtype), qids):
        raise PrelabelError(f"Query Train ID-to-row binding failed for {unit_id}")
    if not np.array_equal(query_raw["cycle_code"].to_numpy(dtype=np.int64), qcycles):
        raise PrelabelError(f"Query Train cycle-to-row binding failed for {unit_id}")
    if np.any(bank_cycles > origin) or np.any(qcycles <= origin):
        raise PrelabelError(f"Future/self donor cycle condition failed for {unit_id}")

    bank_docs, bank_missing = _documents(["" if pd.isna(x) else str(x) for x in bank_raw["signal_stream"].tolist()])
    query_docs, query_missing = _documents(["" if pd.isna(x) else str(x) for x in query_raw["signal_stream"].tolist()])
    missing = np.asarray(a02_npz["missing_mask"], dtype=bool)
    if np.any(bank_missing) or not np.array_equal(query_missing, missing):
        raise PrelabelError(f"Train text availability differs from frozen A02 rows for {unit_id}")
    q_bag, _ = model.text_space.transform(query_docs)
    b_bag = model.bank_matrix
    b_pos, bpos_meta = build_position_block(bank_docs, model.text_space.vectorizer,
                                            shuffled=(arm == "SHUFFLED"), seed=42)
    q_pos, qpos_meta = build_position_block(query_docs, model.text_space.vectorizer,
                                            shuffled=(arm == "SHUFFLED"), seed=42)
    if arm == "SHUFFLED":
        if (bpos_meta["bag_preserved_rows"] != len(bank_docs)
                or bpos_meta["segment_sizes_preserved_rows"] != len(bank_docs)
                or qpos_meta["bag_preserved_rows"] != len(query_docs)
                or qpos_meta["segment_sizes_preserved_rows"] != len(query_docs)):
            raise PrelabelError(f"Within-document shuffle invariants fail for {unit_id}")

    old_sims = np.asarray(a02_npz["neighbor_similarities"], dtype=np.float64)
    m0 = np.sum(old_sims * old_sims, axis=1, dtype=np.float64)
    if not np.allclose(arrays["m0"], m0, rtol=0.0, atol=1e-12):
        raise PrelabelError(f"Original A02 correction-mass reconstruction failed for {unit_id}")
    original_rows = np.asarray(a02_npz["neighbor_source_rows"], dtype=np.int64)
    baseline = np.asarray(a02_npz["baseline_prediction"], dtype=np.float64)
    original = np.asarray(a02_npz["prediction"], dtype=np.float64)
    p_pos = np.asarray(pos_npz["prob_position"], dtype=np.float64)
    if not np.allclose(arrays["p_sparse25"], baseline, rtol=0.0, atol=1e-12):
        raise PrelabelError(f"Sparse25 query baseline binding failed for {unit_id}")
    if not np.allclose(arrays["p_a02_original"], original, rtol=0.0, atol=1e-12):
        raise PrelabelError(f"Original A02 query probability binding failed for {unit_id}")
    if not np.allclose(arrays["p_pos_original"], p_pos, rtol=0.0, atol=1e-12):
        raise PrelabelError(f"Original POS probability binding failed for {unit_id}")

    expected_rows = np.full((len(qrows), K), -1, dtype=np.int64)
    expected_ids = np.full((len(qrows), K), "", dtype=arrays["neighbor_ids"].dtype)
    expected_sims = np.zeros((len(qrows), K), dtype=np.float64)
    expected_weights = np.zeros((len(qrows), K), dtype=np.float64)
    expected_counts = np.zeros(len(qrows), dtype=np.int16)
    expected_raw = np.zeros(len(qrows), dtype=np.float64)
    expected_mass = np.zeros(len(qrows), dtype=np.float64)
    supported = (~missing) & (m0 > 0.0)
    for start in range(0, len(qrows), BLOCK_ROWS):
        stop = min(start + BLOCK_ROWS, len(qrows))
        bag_dense = (q_bag[start:stop] @ b_bag.T).toarray()
        pos_dense = (q_pos[start:stop] @ b_pos.T).toarray()
        kernel = bag_dense * (FLOOR + (1.0 - FLOOR) * pos_dense)
        for local, qi in enumerate(range(start, stop)):
            if not supported[qi]:
                continue
            chosen = _top32(kernel[local], bank_rows)
            if not len(chosen):
                raise PrelabelError(f"Positive original mass has no positive kernel donor for {unit_id} row {qi}")
            sims = kernel[local, chosen]
            squared = sims * sims
            mass = float(np.sum(squared, dtype=np.float64))
            w = squared * (float(m0[qi]) / mass)
            expected_raw[qi] = float(np.dot(w, model.bank_residuals[chosen])) / (SHRINKAGE + float(m0[qi]))
            expected_mass[qi] = mass
            expected_rows[qi, :len(chosen)] = bank_rows[chosen]
            expected_ids[qi, :len(chosen)] = np.asarray(model.bank_ids, dtype=expected_ids.dtype)[chosen]
            expected_sims[qi, :len(chosen)] = sims
            expected_weights[qi, :len(chosen)] = w
            expected_counts[qi] = len(chosen)
    expected_correction = np.clip(expected_raw, -CORRECTION_LIMIT, CORRECTION_LIMIT)
    expected_correction[missing | (m0 == 0)] = 0.0
    expected_a02 = np.clip(baseline + expected_correction, 0.0, 1.0)
    expected_a02[missing | (m0 == 0)] = baseline[missing | (m0 == 0)]
    expected_champion_score, expected_champion_class = _route(p_pos, original, missing)
    expected_candidate_score, expected_candidate_class = _route(p_pos, expected_a02, missing)

    exact = {
        "neighbor_count": expected_counts,
        "neighbor_source_rows": expected_rows,
        "neighbor_ids": expected_ids,
        "neighbor_kernel_similarity": expected_sims,
        "neighbor_weights": expected_weights,
    }
    for name, expected in exact.items():
        if not np.array_equal(arrays[name], expected):
            raise PrelabelError(f"Independent full-bank retrieval mismatch in {name} for {unit_id}")
    numeric = {
        "position_kernel_positive_mass": expected_mass,
        "correction_raw": expected_raw,
        "correction": expected_correction,
        "p_a02_candidate": expected_a02,
        "champion_score": expected_champion_score,
        "candidate_score": expected_candidate_score,
        "mass_error": np.sum(expected_weights, axis=1) - m0,
    }
    for name, expected in numeric.items():
        if not np.allclose(arrays[name], expected, rtol=0.0, atol=1e-12):
            raise PrelabelError(f"Independent formula reconstruction mismatch in {name} for {unit_id}")
    if not np.array_equal(arrays["champion_class"], expected_champion_class):
        raise PrelabelError(f"Full-champion class route mismatch for {unit_id}")
    if not np.array_equal(arrays["candidate_class"], expected_candidate_class):
        raise PrelabelError(f"Candidate class route mismatch for {unit_id}")
    if not np.array_equal(arrays["p_a02_candidate"][missing], original[missing]):
        raise PrelabelError(f"Missing-stream A02 output changed for {unit_id}")
    if float(np.max(np.abs(arrays["mass_error"]))) > 1e-12:
        raise PrelabelError(f"Mass identity exceeds tolerance for {unit_id}")
    if np.any((m0 > 0.0) & (arrays["neighbor_count"] == 0) & ~missing):
        raise PrelabelError(f"Supported present rows lost donors for {unit_id}")
    if np.any((m0 == 0.0) & (arrays["neighbor_count"] > 0)):
        raise PrelabelError(f"Zero-support rows gained donors for {unit_id}")

    supported_orig = original_rows >= 0
    overlap = np.zeros(len(qrows), dtype=np.float64)
    for qi in np.flatnonzero(arrays["neighbor_count"] > 0):
        old = set(original_rows[qi][supported_orig[qi]].tolist())
        new = set(arrays["neighbor_source_rows"][qi][arrays["neighbor_source_rows"][qi] >= 0].tolist())
        overlap[qi] = len(old & new) / max(1, len(old))
    if not np.allclose(arrays["neighbor_overlap_fraction"], overlap, rtol=0.0, atol=1e-12):
        raise PrelabelError(f"Original/candidate neighbor overlap mismatch for {unit_id}")

    donors_seen = arrays["neighbor_source_rows"][arrays["neighbor_source_rows"] >= 0]
    if not set(donors_seen.tolist()).issubset(set(bank_rows.tolist())):
        raise PrelabelError(f"Selected donor fell outside the original full bank for {unit_id}")
    return {
        "fit_id": unit_id,
        "artifact_sha256": sha_file(bundle_path),
        "receipt_sha256": sha_file(receipt_path),
        "pos_archive_unused_member_names": list(pos_npz.unused_member_names),
        "rows": int(len(qrows)),
        "missing_rows": int(missing.sum()),
        "bank_rows": int(len(bank_rows)),
        "positive_mass_rows": int(supported.sum()),
        "selected_donor_references": int(np.sum(arrays["neighbor_count"])),
        "max_mass_error": float(np.max(np.abs(arrays["mass_error"]))),
        "mean_neighbor_overlap": float(np.mean(overlap[supported])) if np.any(supported) else 0.0,
        "clip_count": int(np.sum(np.abs(expected_raw) > CORRECTION_LIMIT)),
        "query_labels_accessed": False,
        "test_or_sample_accessed": False,
    }


def verify() -> dict[str, Any]:
    if OUTPUT_PATH.exists():
        raise PrelabelError("Prelabel verification output already exists; do not overwrite")
    freeze = load_json(FREEZE_PATH)
    if not _committed_and_clean(FREEZE_PATH):
        raise PrelabelError("Freeze must be committed and unchanged before prelabel verification")
    if freeze.get("original_prelabel_boundary_compliant") is not False or freeze.get("scientific_qualification") is not False:
        raise PrelabelError("Frozen deviation record was lost")
    _verify_source_hashes(freeze)
    events = [json.loads(line) for line in LEDGER_PATH.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not _committed_and_clean(LEDGER_PATH):
        raise PrelabelError("Four completed-unit ledger must be committed and unchanged before prelabel verification")
    cap_events = [event for event in events if event.get("event") == "CAP_SET"]
    if len(cap_events) != 1 or cap_events[0].get("cap") != 4 or cap_events[0].get("units") != list(FIT_IDS):
        raise PrelabelError("Fresh lane-local cap4 ledger is missing, duplicated, or changed")
    for event_name in ("PREDICTOR_RESERVED", "PREDICTOR_STARTED", "PREDICTOR_SAVED"):
        ids = [event.get("fit_id") for event in events if event.get("event") == event_name]
        if sorted(ids) != sorted(FIT_IDS) or len(ids) != len(FIT_IDS):
            raise PrelabelError(f"Budget ledger event count/IDs invalid for {event_name}: {ids}")
    event_positions: dict[tuple[str, str], int] = {}
    for index, event in enumerate(events):
        event_name = event.get("event")
        unit_id = event.get("fit_id")
        if event_name in {"PREDICTOR_RESERVED", "PREDICTOR_STARTED", "PREDICTOR_SAVED"}:
            if event.get("pilot_id") != "CLS_A02_POSITION_KERNEL_V1":
                raise PrelabelError(f"Budget event pilot identity mismatch: {event_name}/{unit_id}")
            if event_name == "PREDICTOR_RESERVED" and event.get("source_commit") != freeze.get("source_commit"):
                raise PrelabelError(f"Reservation source commit differs from freeze: {unit_id}")
            event_positions[(str(unit_id), str(event_name))] = index
    for unit_id in FIT_IDS:
        reserve_i = event_positions[(unit_id, "PREDICTOR_RESERVED")]
        start_i = event_positions[(unit_id, "PREDICTOR_STARTED")]
        saved_i = event_positions[(unit_id, "PREDICTOR_SAVED")]
        if not reserve_i < start_i < saved_i:
            raise PrelabelError(f"Budget lifecycle order is invalid for {unit_id}")
    frame = pd.read_csv(TRAIN_PATH, usecols=["record_id", "cycle_code", "signal_stream"],
                         dtype={"record_id": "string"})
    reports = {}
    with _shared_lock_context():
        import psutil
        available = int(psutil.virtual_memory().available)
        if available < INITIAL_RAM_MIN or available - 512 * 1024**2 < RESERVE_BYTES:
            raise PrelabelError(f"Resource gate failed before independent verification: free={available}")
        with threadpool_limits(limits=THREADS):
            for origin in ORIGINS:
                for arm in ARMS:
                    reports[f"E{origin}_{arm}"] = _verify_unit(origin, arm, freeze, frame)

    e67 = _npz_output(RUN_ROOT / "E67_REAL/predictions.npz")
    e75 = _npz_output(RUN_ROOT / "E75_REAL/predictions.npz")
    m67 = (e67["score_cycles"] >= 76) & (e67["score_cycles"] <= 79)
    m75 = (e75["score_cycles"] >= 76) & (e75["score_cycles"] <= 79)
    if set(e67["score_record_ids"][m67].astype(str)) != set(e75["score_record_ids"][m75].astype(str)):
        raise PrelabelError("E67/E75 overlapping W6 IDs differ")
    unique_ids = set(e67["score_record_ids"][(e67["score_cycles"] >= 68) & (e67["score_cycles"] <= 75)].astype(str))
    unique_ids.update(e75["score_record_ids"][(e75["score_cycles"] >= 76) & (e75["score_cycles"] <= 79)].astype(str))
    if len(unique_ids) != 20903:
        raise PrelabelError(f"Unique primary pool count differs from frozen 20,903: {len(unique_ids)}")
    result = {
        "status": "PASS_AMENDED_PRELABEL_VERIFICATION",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "pilot_id": "CLS_A02_POSITION_KERNEL_V1",
        "source_commit": freeze["source_commit"],
        "official_source_commit": freeze["official_source_commit"],
        "freeze_sha256": sha_file(FREEZE_PATH),
        "budget_ledger_sha256": sha_file(LEDGER_PATH),
        "four_unit_receipts_and_outputs": reports,
        "closed_pos_archive_members_left_unread": {
            f"E{origin}": reports[f"E{origin}_REAL"]["pos_archive_unused_member_names"]
            for origin in ORIGINS
        },
        "unique_primary_rows": len(unique_ids),
        "overlap_e67_e75_w6_rows": int(m67.sum()),
        "all_four_outputs_target_free": True,
        "all_four_independently_reconstructed": True,
        "model_construction_training_boundary_verified": True,
        "original_prelabel_boundary_compliant": False,
        "scientific_qualification": False,
        "query_labels_accessed_by_this_verifier": False,
        "test_or_sample_submission_accessed": False,
        "kaggle_accessed": False,
        "incident_reference": "reports/simplification_v1/CLS_POST_BAG2_NEXT_V1/INCIDENT_01.json",
    }
    with OUTPUT_PATH.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(result, stream, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    return result


if __name__ == "__main__":
    try:
        print(json.dumps(verify(), sort_keys=True, ensure_ascii=False))
    except Exception as exc:
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        raise SystemExit(2)
