from __future__ import annotations

import os
for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_name] = "2"

import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from typing import Any, Mapping

import joblib
import numpy as np
import pandas as pd
import psutil
from scipy import sparse
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import optimization_v3_rnbr_official as official  # noqa: E402
from scripts import optimization_v3_rnbr_continuation as ledger_prior  # noqa: E402
from src.models.classification_a02_neighbors import _documents  # noqa: E402
from src.models.regression_neighbor_rnbr import (  # noqa: E402
    EXPECTED_SOURCE_CUTOFFS,
    MAX_NEIGHBORS,
    MIN_NEIGHBORS,
    predict_from_saved_neighbors,
)
from src.models import regression_rnbr_impact_match as impact  # noqa: E402

_LIMITER = threadpool_limits(limits=2)

BASE_HEAD = "ccc787cfe1538d7426cb952d6334892b9d818e31"
BASE_FREEZE_PATH = ROOT / "experiments/optimization_v3/rnbr_pilot_v1/official_freeze.json"
BASE_REPORT = ROOT / "reports/optimization_v3/rnbr_pilot_v1/official_execution_v1"
BASE_RECOVERY = BASE_REPORT / "SERIALIZATION_RECOVERY_V2.json"
BASE_VERIFICATION = BASE_REPORT / "INDEPENDENT_VERIFICATION.json"
TRAIN = ROOT / "data/raw/regression/Train.csv"
LEDGER = ROOT / "experiments/runs/optimization_v3_research_next_v1/budget.jsonl"

CARD_ID = "REGRESSION_RNBR_IMPACT_MATCH_V1"
STAGE = "FOUR_FIT_EXACT_IMPACT_MEMORY_PILOT"
AMENDMENT_ID = "RNBR_IMPACT_MATCH_CAP81_2026_11_03"
RESERVATION_KEY = f"{CARD_ID}:{STAGE}"
FIT_IDS = (
    "RNBR_IMPACT_MATCH_E67_OBSERVED",
    "RNBR_IMPACT_MATCH_E67_SHUFFLED",
    "RNBR_IMPACT_MATCH_E75_OBSERVED",
    "RNBR_IMPACT_MATCH_E75_SHUFFLED",
)
ORIGINS = (67, 75)
ARMS = ("observed", "shuffled")
QUERY_CYCLES = {67: (68, 79), 75: (76, 79)}
VIEWS = {"W4": (68, 71), "W6": (76, 79), "H12": (68, 79)}
MIN_SUPPORT_COVERAGE = 0.95
MIN_SHUFFLE_MOVEMENT = 0.80
MAX_RSS_BYTES = 6 * 1024**3
MIN_FREE_RAM_BYTES = 2 * 1024**3

PROTOCOL_PATH = ROOT / "experiments/optimization_v3/rnbr_impact_match_v1/protocol.json"
PREFLIGHT_PATH = ROOT / "reports/optimization_v3/rnbr_impact_match_v1/targetfree_preflight.json"
SYNTHETIC_PATH = ROOT / "reports/optimization_v3/rnbr_impact_match_v1/synthetic_qa.json"
FREEZE_PATH = ROOT / "experiments/optimization_v3/rnbr_impact_match_v1/freeze.json"
REPORT_DIR = ROOT / "reports/optimization_v3/rnbr_impact_match_v1"
RUN_ROOT = ROOT / "experiments/runs/optimization_v3_research_next_v1/RNBR_IMPACT_MATCH_V1"
GEOMETRY_DIR = RUN_ROOT / "targetfree_geometry"


class ImpactMatchRunError(RuntimeError):
    pass


def _canonical(value: Any) -> bytes:
    return impact.canonical_bytes(value)


def _sha_file(path: Path) -> str:
    return impact.sha256_file(path)


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ImpactMatchRunError(f"Expected JSON object: {path}")
    return value


def _write_once(path: Path, value: Mapping[str, Any]) -> None:
    rendered = json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_text(encoding="utf-8") != rendered:
            raise ImpactMatchRunError(f"Refusing to overwrite immutable artifact: {path}")
        return
    staging = path.with_name(path.name + ".staging")
    with staging.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(rendered)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(staging, path)


def _git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def _source_paths() -> tuple[str, ...]:
    return (
        "scripts/optimization_v3_rnbr_impact_match.py",
        "scripts/optimization_v3_rnbr_impact_match_verify.py",
        "scripts/optimization_v3_rnbr_official.py",
        "scripts/optimization_v3_rnbr_official_verify.py",
        "scripts/optimization_v3_rnbr_continuation.py",
        "src/models/regression_rnbr_impact_match.py",
        "src/models/regression_neighbor_rnbr.py",
        "src/models/classification_a02_neighbors.py",
        "src/models/research_next_budget.py",
    )


def _protocol() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "protocol_id": "RNBR_IMPACT_MATCH_V1",
        "status": "FROZEN_BEFORE_TASK_FITS",
        "authorized_base_head": BASE_HEAD,
        "origins": [67, 75],
        "fit_ids": list(FIT_IDS),
        "fit_count": 4,
        "fit_semantics": "one residual-memory construction for each origin x observed/shuffled arm; no estimator refit",
        "target_source": "official Regression Train only",
        "donors": "present-stream rows, cycles 56 through origin, finite exact float64 impact_load",
        "queries": {"E67_H12": [68, 79], "E67_W4": [68, 71], "E75_W6": [76, 79]},
        "retrieval": {
            "text": "reuse frozen A02 target-free TF-IDF vectorizer at the same origin; extract vectorizer state only",
            "matching": "exact equality of parsed float64 impact_load; no binning, rounding, or quantile edges",
            "geometry": "search all same-impact donors for positive cosine top32, stable descending cosine then source row",
            "minimum_neighbors": MIN_NEIGHBORS,
            "maximum_neighbors": MAX_NEIGHBORS,
        },
        "residual": "threat_rank minus final-mapped R06 OOF prediction using the frozen strictly-earlier source cutoff",
        "shuffle": {
            "strata": "exact donor cycle x exact impact_load x fixed R06 OOF prediction decile",
            "seed": 42,
            "rng_reset_each_origin": True,
            "traversal": "ascending cycle, impact_load, decile, and source row",
        },
        "formula": "clip(R06_origin + clip(0.25 * cosine_squared_weighted_median(residual), -5, +5), 0, 100)",
        "fallback": "exact R06 for missing stream, missing impact, or fewer than five positive-cosine same-impact neighbors",
        "preflight": {
            "positive_cosine_support_coverage_minimum": MIN_SUPPORT_COVERAGE,
            "coverage_denominator": "present-stream queries in each declared W4, W6, and E67 H12 view",
            "shuffle_source_position_movement_minimum": MIN_SHUFFLE_MOVEMENT,
        },
        "gates": {
            "W4_gain_vs_R06_min": 0.0,
            "W6_gain_vs_R06_min": 0.0,
            "pooled_gain_vs_R06_min": 0.03,
            "H12_gain_vs_R06_min": 0.0,
            "pooled_gain_vs_matched_shuffle_min": 0.01,
            "H12_gain_vs_matched_shuffle_min": 0.0,
            "pooled_gain_vs_original_RNBR_min": 0.03,
            "H12_gain_vs_original_RNBR_min": 0.0,
            "tail_subgroup_harm_vs_R06_max_if_n_ge_100": 0.25,
        },
        "source_boundaries": {
            "classification_labels_or_probabilities": "never read",
            "official Test or SampleSubmission": "never read",
            "query targets": "read only after all four durable bundles and independent integrity checks",
            "full_Train_refit": False,
            "Kaggle": "not accessed",
        },
        "execution": {"jobs": 1, "threads": 2, "max_rss_bytes": MAX_RSS_BYTES,
                      "minimum_free_ram_bytes": MIN_FREE_RAM_BYTES},
        "no_rescue": "preflight failure skips all task fits; no threshold, bin, seed, or recipe search",
    }


def _check_base_freeze(*, train: Mapping[str, np.ndarray]) -> dict[str, Any]:
    base = _read_json(BASE_FREEZE_PATH)
    body = {key: value for key, value in base.items() if key != "freeze_signature"}
    if base.get("freeze_signature") != _sha256(_canonical(body)):
        raise ImpactMatchRunError("Historical RNBR official freeze signature is invalid")
    if _sha_file(TRAIN) != base["official_train"]["sha256"]:
        raise ImpactMatchRunError("Official Regression Train hash differs from the frozen source")
    for rel, expected in base.get("code_sha256", {}).items():
        if _sha_file(ROOT / rel) != expected:
            raise ImpactMatchRunError(f"Historical frozen RNBR source differs: {rel}")
    r06_source = base["r06_lineage"]
    if (_sha_file(ROOT / r06_source["source_path"]) != r06_source["source_sha256"]
            or _sha_file(ROOT / r06_source["recipe_path"]) != r06_source["recipe_sha256"]):
        raise ImpactMatchRunError("R06 frozen code or recipe hash differs")
    # This loader parses only record_id, cycle_code, and signal_stream.
    recomputed = official.load_targetfree_train()
    if any(not np.array_equal(train[key], recomputed[key]) for key in ("rows", "cycles", "ids", "present")):
        raise ImpactMatchRunError("Regression target-free Train metadata changed while loading")
    r06_cache = official.load_r06_predictions(train, base)
    for origin in ORIGINS:
        # This checks saved A02 geometry, fit receipt and fit_state only; it
        # reads no Classification labels or probabilities.
        official.load_geometry(origin, train, base)
    return {"freeze": base, "r06_cache": r06_cache}


def _load_targetfree_features(train: Mapping[str, np.ndarray]) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    frame = pd.read_csv(TRAIN, usecols=["record_id", "cycle_code", "signal_stream", "impact_load"],
                        dtype={"record_id": str, "impact_load": "float64"})
    if (len(frame) != len(train["rows"])
            or not np.array_equal(frame["record_id"].astype(str).to_numpy(), train["ids"])
            or not np.array_equal(pd.to_numeric(frame["cycle_code"], errors="raise").to_numpy(dtype=np.int64), train["cycles"])):
        raise ImpactMatchRunError("Target-free feature rows/IDs/cycles do not align to official Train")
    impacts = frame["impact_load"].to_numpy(dtype=np.float64)
    if np.isinf(impacts).any():
        raise ImpactMatchRunError("Official Train impact_load contains infinity")
    docs, missing = _documents(frame["signal_stream"])
    if not np.array_equal(missing, ~train["present"]):
        raise ImpactMatchRunError("Normalized text availability differs from official target-free Train")
    return frame, np.asarray(docs, dtype=object), impacts


def _verify_a02_text_state(origin: int, base_freeze: Mapping[str, Any],
                           train: Mapping[str, np.ndarray], docs: np.ndarray):
    spec = base_freeze["a02_saved_geometry"][str(origin)]["observed"]
    model_path = ROOT / spec["model"]
    if _sha_file(model_path) != spec["model_sha256"]:
        raise ImpactMatchRunError(f"A02 E{origin} fitted state hash differs")
    receipt_path = ROOT / spec["receipt_json"]
    receipt = _read_json(receipt_path)
    receipt_body = {key: value for key, value in receipt.items() if key != "receipt_sha256"}
    state_path = ROOT / spec["state_json"]
    state = _read_json(state_path)
    state_body = {key: value for key, value in state.items() if key != "state_sha256"}
    if (receipt.get("receipt_sha256") != _sha256(_canonical(receipt_body))
            or receipt.get("fit_id") != f"A02_E{origin}_OBSERVED"
            or receipt.get("origin_cycle") != origin
            or receipt.get("freeze_signature") != spec["freeze_signature"]
            or receipt.get("model_sha256") != spec["model_sha256"]
            or receipt.get("state_sha256") != _sha256(_canonical(state))
            or receipt.get("query_labels_passed_to_fit") is not False
            or receipt.get("test_data_used") is not False):
        raise ImpactMatchRunError(f"A02 E{origin} text-only fit receipt does not verify")
    if (state.get("state_sha256") != _sha256(_canonical(state_body))
            or state.get("state_sha256") != spec["state_sha256"]):
        raise ImpactMatchRunError(f"A02 E{origin} residual-state receipt hash differs")
    model = joblib.load(model_path)
    text_space = getattr(model, "text_space", None)
    vectorizer = getattr(text_space, "vectorizer", None)
    if vectorizer is None or not hasattr(text_space, "audit_state"):
        raise ImpactMatchRunError(f"A02 E{origin} model lacks its fitted target-free text space")
    audit = text_space.audit_state()
    prefix_rows = np.flatnonzero(train["cycles"] <= origin).astype(np.int64)
    prefix_docs = docs[prefix_rows].tolist()
    if (audit.get("origin_cycle") != origin or audit.get("fit_rows") != len(prefix_rows)
            or audit.get("fit_row_positions_sha256") != official.sha256_array(prefix_rows, "<i8")
            or audit.get("fit_documents_sha256") != official.sha256_bytes(official.canonical_bytes(prefix_docs))
            or audit.get("text_only_cosine") is not True
            or receipt.get("text_prefix_rows") != len(prefix_rows)
            or receipt.get("text_prefix_row_positions_sha256") != audit.get("fit_row_positions_sha256")):
        raise ImpactMatchRunError(f"A02 E{origin} vectorizer audit does not match the frozen train prefix")
    return vectorizer, audit


def _load_original_rnbr_predictions(origin: int, base_freeze: Mapping[str, Any],
                                    train: Mapping[str, np.ndarray]) -> dict[str, Any]:
    recovery = _read_json(BASE_RECOVERY)
    independent = _read_json(BASE_VERIFICATION)
    if (recovery.get("status") != "FOUR_FIT_PILOT_COMPLETE_DIAGNOSTIC_ONLY"
            or recovery.get("freeze_signature") != base_freeze.get("freeze_signature")
            or independent.get("status") != "PASS"
            or independent.get("freeze_signature") != base_freeze.get("freeze_signature")
            or independent.get("source_boundaries", {}).get("classification_labels_loaded") is not False
            or independent.get("source_boundaries", {}).get("test_or_sample_submission_opened") is not False):
        raise ImpactMatchRunError("Existing original-RNBR pilot recovery/verification is not PASS")
    fit_id = f"RNBR_PILOT_E{origin}_OBSERVED"
    fit_record = next((item for item in recovery["fit_receipts"] if item["fit_id"] == fit_id), None)
    if fit_record is None:
        raise ImpactMatchRunError(f"Original observed R-NBR receipt is missing for E{origin}")
    if fit_id == "RNBR_PILOT_E67_OBSERVED":
        bundle_rel = recovery["serialization_recovery_detail"]["recovery_bundle_path"]
        receipt_rel = recovery["serialization_recovery_detail"]["recovery_receipt_path"]
    else:
        folder = official._bundle_dir(fit_id)
        bundle_rel = folder.relative_to(ROOT).as_posix() + "/arrays.npz"
        receipt_rel = folder.relative_to(ROOT).as_posix() + "/receipt.json"
    npz_path, receipt_path = ROOT / bundle_rel, ROOT / receipt_rel
    receipt = _read_json(receipt_path)
    receipt_body = {key: value for key, value in receipt.items() if key != "integrity_sha256"}
    if (_sha_file(npz_path) != fit_record["bundle_npz_sha256"]
            or _sha_file(receipt_path) != fit_record["receipt_sha256"]
            or receipt.get("integrity_sha256") != official.sha256_bytes(official.canonical_bytes(receipt_body))
            or receipt.get("npz_sha256") != fit_record["bundle_npz_sha256"]
            or receipt.get("identity", {}).get("fit_id") != fit_id
            or receipt.get("identity", {}).get("freeze_signature") != base_freeze["freeze_signature"]
            or receipt.get("identity", {}).get("query_labels_read") is not False
            or receipt.get("identity", {}).get("test_or_sample_submission_opened") is not False
            or receipt.get("identity", {}).get("classification_labels_used") is not False):
        raise ImpactMatchRunError(f"Accepted original RNBR E{origin} observed bundle does not verify")
    # Load only numeric query rows/predictions/baseline. Some historical string
    # members use object dtype; they are deliberately never deserialized.
    with np.load(npz_path, allow_pickle=False) as archive:
        rows = np.asarray(archive["score_rows"], dtype=np.int64).copy()
        prediction = np.asarray(archive["prediction"], dtype=np.float64).copy()
        baseline = np.asarray(archive["baseline_prediction"], dtype=np.float64).copy()
    expected_low, expected_high = QUERY_CYCLES[origin]
    expected = np.flatnonzero((train["cycles"] >= expected_low) & (train["cycles"] <= expected_high))
    if (not np.array_equal(rows, expected) or prediction.shape != rows.shape or baseline.shape != rows.shape
            or not np.isfinite(prediction).all() or np.any((prediction < 0) | (prediction > 100))):
        raise ImpactMatchRunError(f"Accepted original RNBR E{origin} rows/predictions are invalid")
    return {"rows": rows, "prediction": prediction, "baseline": baseline,
            "npz_sha256": fit_record["bundle_npz_sha256"],
            "receipt_sha256": fit_record["receipt_sha256"],
            "bundle_path": bundle_rel, "receipt_path": receipt_rel}


def _baseline_for_origin(cache: Mapping[int, Mapping[str, np.ndarray]], origin: int,
                         rows: np.ndarray, ids: np.ndarray) -> np.ndarray:
    source = cache[origin]
    lookup = {int(row): i for i, row in enumerate(source["rows"])}
    values = np.empty(len(rows), dtype=np.float64)
    for i, (row, record_id) in enumerate(zip(rows, ids)):
        j = lookup.get(int(row))
        if j is None or str(source["ids"][j]) != str(record_id):
            raise ImpactMatchRunError(f"R06 origin E{origin} query cache misses row/ID {row}")
        values[i] = float(source["prediction"][j])
    if not np.isfinite(values).all() or np.any((values < 0.0) | (values > 100.0)):
        raise ImpactMatchRunError(f"R06 E{origin} query baseline is invalid")
    return values


def _build_targetfree_geometry(origin: int, train: Mapping[str, np.ndarray],
                               frame: pd.DataFrame, docs: np.ndarray, impacts: np.ndarray,
                               cache: Mapping[int, Mapping[str, np.ndarray]], vectorizer: Any
                               ) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    cycles, present = train["cycles"], train["present"]
    donor_rows = np.flatnonzero((cycles >= 56) & (cycles <= origin) & present & np.isfinite(impacts)).astype(np.int64)
    query_low, query_high = QUERY_CYCLES[origin]
    query_rows = np.flatnonzero((cycles >= query_low) & (cycles <= query_high)).astype(np.int64)
    if not len(donor_rows) or not len(query_rows):
        raise ImpactMatchRunError(f"E{origin} donor or query population is empty")
    donor_docs = docs[donor_rows].tolist()
    query_docs = docs[query_rows].tolist()
    donor_matrix = vectorizer.transform(donor_docs).tocsr()
    query_matrix = vectorizer.transform(query_docs).tocsr()
    query_missing = ~present[query_rows]
    donor_impact = impacts[donor_rows]
    query_impact = impacts[query_rows]
    geometry = impact.exact_impact_top32(
        query_matrix, donor_matrix, query_impact=query_impact, donor_impact=donor_impact,
        donor_rows=donor_rows, query_stream_missing=query_missing,
    )
    donor_ids = train["ids"][donor_rows].astype(str)
    query_ids = train["ids"][query_rows].astype(str)
    donor_cycle = cycles[donor_rows].astype(np.int64)
    query_cycle = cycles[query_rows].astype(np.int64)
    neighbor_ids = np.full(geometry["neighbor_rows"].shape, "", dtype="U64")
    donor_lookup = {int(row): j for j, row in enumerate(donor_rows)}
    for i, count in enumerate(geometry["neighbor_count"]):
        for j, row in enumerate(geometry["neighbor_rows"][i, :int(count)]):
            neighbor_ids[i, j] = donor_ids[donor_lookup[int(row)]]
    baseline = _baseline_for_origin(cache, origin, query_rows, query_ids)
    effective_missing = query_missing | ~np.isfinite(query_impact)
    impact.validate_same_impact_geometry(
        query_rows=query_rows, query_cycles=query_cycle, query_impact=query_impact, query_ids=query_ids,
        query_stream_missing=query_missing, donor_rows=donor_rows, donor_cycles=donor_cycle,
        donor_impact=donor_impact, donor_ids=donor_ids, origin=origin,
        neighbor_rows=geometry["neighbor_rows"], neighbor_ids=neighbor_ids,
        neighbor_similarities=geometry["neighbor_similarities"], neighbor_count=geometry["neighbor_count"],
    )
    arrays = {
        "origin": np.asarray([origin], dtype=np.int16),
        "donor_rows": donor_rows, "donor_ids": donor_ids, "donor_cycles": donor_cycle,
        "donor_impact_load": donor_impact,
        "query_rows": query_rows, "query_ids": query_ids, "query_cycles": query_cycle,
        "query_impact_load": query_impact,
        "query_stream_missing": query_missing,
        "query_impact_missing": ~np.isfinite(query_impact),
        "query_effective_missing": effective_missing,
        "baseline_prediction": baseline,
        "neighbor_rows": geometry["neighbor_rows"],
        "neighbor_ids": neighbor_ids,
        "neighbor_similarities": geometry["neighbor_similarities"],
        "neighbor_count": geometry["neighbor_count"],
    }
    present_query = ~query_missing
    view_stats: dict[str, Any] = {}
    for view, (low, high) in (("H12_E67", (68, 79)), ("W4_E67", (68, 71)), ("W6_E75", (76, 79))):
        if (origin == 67 and view == "W6_E75") or (origin == 75 and view != "W6_E75"):
            continue
        selected = (query_cycle >= low) & (query_cycle <= high)
        denominator = int(np.sum(selected & present_query))
        supported = int(np.sum(selected & present_query & (geometry["neighbor_count"] >= MIN_NEIGHBORS)))
        impact_missing = int(np.sum(selected & present_query & ~np.isfinite(query_impact)))
        view_stats[view] = {
            "total_rows": int(np.sum(selected)),
            "present_stream_denominator": denominator,
            "supported_positive_cosine_rows_ge5": supported,
            "coverage": supported / denominator if denominator else 0.0,
            "stream_missing_rows": int(np.sum(selected & query_missing)),
            "impact_missing_present_rows": impact_missing,
            "exact_r06_fallback_rows": int(np.sum(selected & (geometry["neighbor_count"] < MIN_NEIGHBORS) | selected & query_missing)),
        }
    # A unique synthetic label per source position measures the exact frozen
    # permutation's donor-position movement without opening any target values.
    _, _, shuffle_audit = impact.shuffle_residuals_within_exact_cells(
        np.arange(len(donor_rows), dtype=np.float64), donor_cycle, donor_impact,
        official.aligned_r06_predictions(donor_rows, donor_cycle, cache), donor_rows, seed=42,
    )
    meta = {
        "origin": origin,
        "donor_rows": len(donor_rows),
        "present_queries": int(present_query.sum()),
        "query_rows": len(query_rows),
        "tfidf_features": int(donor_matrix.shape[1]),
        "donor_matrix_nnz": int(donor_matrix.nnz),
        "query_matrix_nnz": int(query_matrix.nnz),
        "views": view_stats,
        "matched_shuffle_source_position_audit": shuffle_audit,
        "geometry_array_sha256": {key: impact.sha256_array(value) for key, value in arrays.items()},
    }
    return arrays, meta


def _synthetic_payload() -> dict[str, np.ndarray]:
    donor_rows = np.asarray([3, 8, 13, 21, 34, 55], dtype=np.int64)
    donor_cycles = np.asarray([56, 56, 57, 58, 59, 59], dtype=np.int64)
    donor_impact = np.asarray([1.0, 1.0, 2.0, 1.0, 1.0, 2.0], dtype=np.float64)
    donor_ids = np.asarray(["d-α", "d-東京", "d-3", "d-4", "d-5", "d-6"], dtype="U8")
    donor_matrix = sparse.csr_matrix(np.asarray([
        [1., 0., 0.], [1., 0., 0.], [0., 1., 0.],
        [1., 0., 0.], [0., 0., 1.], [0., 1., 0.],
    ]))
    query_matrix = sparse.csr_matrix(np.asarray([[1., 0., 0.], [0., 1., 0.], [1., 0., 0.], [0., 0., 0.]]))
    query_rows = np.asarray([101, 102, 103, 104], dtype=np.int64)
    query_cycles = np.asarray([60, 60, 60, 60], dtype=np.int64)
    query_impact = np.asarray([1.0, 2.0, 1.0, np.nan], dtype=np.float64)
    query_ids = np.asarray(["q1", "q2", "q3", "q4"], dtype="U2")
    stream_missing = np.asarray([False, False, True, False])
    result = impact.exact_impact_top32(
        query_matrix, donor_matrix, query_impact=query_impact, donor_impact=donor_impact,
        donor_rows=donor_rows, query_stream_missing=stream_missing,
    )
    neighbor_ids = np.full(result["neighbor_rows"].shape, "", dtype="U8")
    lookup = {int(row): j for j, row in enumerate(donor_rows)}
    for i, count in enumerate(result["neighbor_count"]):
        for j, row in enumerate(result["neighbor_rows"][i, :int(count)]):
            neighbor_ids[i, j] = donor_ids[lookup[int(row)]]
    baseline = np.asarray([51.0, 62.0, 12.0, 95.0])
    residual = np.asarray([-10.0, -2.0, 3.0, 8.0, 13.0, 20.0])
    residual_shuffled, source_rows, shuffle_audit = impact.shuffle_residuals_within_exact_cells(
        residual, donor_cycles, donor_impact, np.asarray([20., 20., 30., 30., 40., 40.]), donor_rows,
    )
    impact.validate_same_impact_geometry(
        query_rows=query_rows, query_cycles=query_cycles, query_impact=query_impact, query_ids=query_ids,
        query_stream_missing=stream_missing, donor_rows=donor_rows, donor_cycles=donor_cycles,
        donor_impact=donor_impact, donor_ids=donor_ids, origin=59,
        neighbor_rows=result["neighbor_rows"], neighbor_ids=neighbor_ids,
        neighbor_similarities=result["neighbor_similarities"], neighbor_count=result["neighbor_count"],
    )
    predicted = predict_from_saved_neighbors(
        donor_rows=donor_rows, donor_residuals=residual,
        query_r06_predictions=baseline, query_missing=stream_missing | ~np.isfinite(query_impact),
        neighbor_rows=result["neighbor_rows"], neighbor_cosines=result["neighbor_similarities"],
        neighbor_counts=result["neighbor_count"],
    )
    control = predict_from_saved_neighbors(
        donor_rows=donor_rows, donor_residuals=residual_shuffled,
        query_r06_predictions=baseline, query_missing=stream_missing | ~np.isfinite(query_impact),
        neighbor_rows=result["neighbor_rows"], neighbor_cosines=result["neighbor_similarities"],
        neighbor_counts=result["neighbor_count"],
    )
    return {
        "query_rows": query_rows, "query_ids": query_ids, "query_cycles": query_cycles,
        "query_impact_load": query_impact, "query_stream_missing": stream_missing,
        "donor_rows": donor_rows, "donor_ids": donor_ids, "donor_cycles": donor_cycles,
        "donor_impact_load": donor_impact, "donor_residuals": residual,
        "neighbor_rows": result["neighbor_rows"], "neighbor_ids": neighbor_ids,
        "neighbor_similarities": result["neighbor_similarities"],
        "neighbor_count": result["neighbor_count"], "shuffle_source_rows": source_rows,
        "shuffle_residuals": residual_shuffled,
        "observed_prediction": predicted["prediction"], "shuffled_prediction": control["prediction"],
        "baseline_prediction": baseline,
        "shuffle_source_movement": np.asarray([shuffle_audit["source_position_movement_rate"]]),
    }


def synthetic_qa() -> dict[str, Any]:
    payload = _synthetic_payload()
    with tempfile.TemporaryDirectory(prefix="rnbr-impact-match-synthetic-") as temp:
        folder = Path(temp) / "fit"
        receipt = impact.persist_bundle_once(
            folder, fit_id="RNBR_IMPACT_MATCH_SYNTHETIC", freeze_signature="a" * 64,
            arrays=payload, metadata={"synthetic_only": True, "task_fit": False},
        )
        loaded, read_receipt = impact.load_bundle(
            folder, fit_id="RNBR_IMPACT_MATCH_SYNTHETIC", freeze_signature="a" * 64
        )
        if receipt != read_receipt or any(not impact.arrays_equal(loaded[key], value) for key, value in payload.items()):
            raise ImpactMatchRunError("Synthetic impact-match persistence/readback failed")
        original_hash = _sha_file(folder / "arrays.npz")
        impact.persist_bundle_once(
            folder, fit_id="RNBR_IMPACT_MATCH_SYNTHETIC", freeze_signature="a" * 64,
            arrays=payload, metadata={"synthetic_only": True, "task_fit": False},
        )
        if _sha_file(folder / "arrays.npz") != original_hash:
            raise ImpactMatchRunError("Synthetic impact-match resume changed the durable NPZ")
        try:
            changed = dict(payload)
            changed["observed_prediction"] = payload["observed_prediction"] + 1.0
            impact.persist_bundle_once(
                folder, fit_id="RNBR_IMPACT_MATCH_SYNTHETIC", freeze_signature="a" * 64,
                arrays=changed, metadata={"synthetic_only": True, "task_fit": False},
            )
        except impact.ImpactMatchError:
            reject_overwrite = True
        else:
            reject_overwrite = False
        if not reject_overwrite:
            raise ImpactMatchRunError("Synthetic altered-output resume was not rejected")
    if (payload["neighbor_rows"][0, 0] not in payload["donor_rows"]
            or payload["neighbor_count"][2] != 0 or payload["neighbor_count"][3] != 0
            or payload["observed_prediction"][2] != payload["baseline_prediction"][2]
            or payload["observed_prediction"][3] != payload["baseline_prediction"][3]):
        raise ImpactMatchRunError("Synthetic exact-impact or R06 fallback assertions failed")
    return {
        "status": "PASS",
        "fit_type": "SYNTHETIC_ONLY",
        "actual_same_impact_builder": True,
        "same_impact_full_bank_search": True,
        "strict_time_and_geometry_validation": True,
        "weighted_median_formula_observed_and_control": True,
        "missing_stream_and_missing_impact_exact_R06_fallback": True,
        "pickle_free_unicode_persistence_readback": True,
        "idempotent_resume": True,
        "changed_output_rejected": True,
        "synthetic_donor_rows": len(payload["donor_rows"]),
        "synthetic_query_rows": len(payload["query_rows"]),
    }


def _source_hashes() -> dict[str, str]:
    return {rel: _sha_file(ROOT / rel) for rel in _source_paths()}


def _base_artifact_hashes(base_freeze: Mapping[str, Any]) -> dict[str, str]:
    result: dict[str, str] = {
        "experiments/optimization_v3/rnbr_pilot_v1/official_freeze.json": _sha_file(BASE_FREEZE_PATH),
        "reports/optimization_v3/rnbr_pilot_v1/official_execution_v1/SERIALIZATION_RECOVERY_V2.json": _sha_file(BASE_RECOVERY),
        "reports/optimization_v3/rnbr_pilot_v1/official_execution_v1/INDEPENDENT_VERIFICATION.json": _sha_file(BASE_VERIFICATION),
        "data/raw/regression/Train.csv": _sha_file(TRAIN),
        base_freeze["r06_lineage"]["source_path"]: _sha_file(ROOT / base_freeze["r06_lineage"]["source_path"]),
        base_freeze["r06_lineage"]["recipe_path"]: _sha_file(ROOT / base_freeze["r06_lineage"]["recipe_path"]),
    }
    for cutoff, item in base_freeze["r06_lineage"]["cache_inventory"].items():
        result[item["npz"]] = _sha_file(ROOT / item["npz"])
        result[item["json"]] = _sha_file(ROOT / item["json"])
    for origin in ORIGINS:
        for arm, item in base_freeze["a02_saved_geometry"][str(origin)].items():
            for key in ("npz", "predictions_json", "receipt_json", "state_json", "model"):
                rel = item[key]
                result[rel] = _sha_file(ROOT / rel)
    old_receipts = _read_json(BASE_RECOVERY)["fit_receipts"]
    for origin in ORIGINS:
        fit_id = f"RNBR_PILOT_E{origin}_OBSERVED"
        if fit_id == "RNBR_PILOT_E67_OBSERVED":
            record = _read_json(BASE_RECOVERY)["serialization_recovery_detail"]["recovery_event"]
            for key in ("recovery_bundle_path", "recovery_receipt_path"):
                rel = record[key]
                result[rel] = _sha_file(ROOT / rel)
        else:
            fit_id = f"RNBR_PILOT_E{origin}_OBSERVED"
            folder = official._bundle_dir(fit_id)
            result[folder.relative_to(ROOT).as_posix() + "/arrays.npz"] = _sha_file(folder / "arrays.npz")
            result[folder.relative_to(ROOT).as_posix() + "/receipt.json"] = _sha_file(folder / "receipt.json")
    return result


def _load_prepare_inputs():
    if _git("merge-base", "--is-ancestor", BASE_HEAD, "HEAD") == "":
        pass
    # Explicitly check ancestry with exit status; _git raises on an invalid base.
    subprocess.run(["git", "merge-base", "--is-ancestor", BASE_HEAD, "HEAD"], cwd=ROOT, check=True)
    targetfree = official.load_targetfree_train()
    frame, docs, impacts = _load_targetfree_features(targetfree)
    base = _check_base_freeze(train=targetfree)
    return targetfree, frame, docs, impacts, base


def prepare() -> dict[str, Any]:
    """Build target-free geometry, run synthetic QA, and freeze the pilot."""
    protocol = _protocol()
    _write_once(PROTOCOL_PATH, protocol)
    synthetic = synthetic_qa()
    _write_once(SYNTHETIC_PATH, synthetic)
    train, frame, docs, impacts, base = _load_prepare_inputs()
    origin_info: dict[str, Any] = {}
    all_views: dict[str, Any] = {}
    for origin in ORIGINS:
        vectorizer, text_audit = _verify_a02_text_state(origin, base["freeze"], train, docs)
        arrays, meta = _build_targetfree_geometry(
            origin, train, frame, docs, impacts, base["r06_cache"], vectorizer
        )
        geometry_path = GEOMETRY_DIR / f"E{origin}_geometry.npz"
        geometry_hash = impact.save_npz_once(geometry_path, **arrays)
        meta["geometry_path_ignored"] = geometry_path.relative_to(ROOT).as_posix()
        meta["geometry_file_sha256"] = geometry_hash
        meta["a02_text_audit"] = text_audit
        origin_info[str(origin)] = meta
        all_views.update(meta["views"])
        del vectorizer
    coverage_checks = {
        view: {"coverage": stats["coverage"], "minimum": MIN_SUPPORT_COVERAGE,
               "pass": stats["coverage"] >= MIN_SUPPORT_COVERAGE}
        for view, stats in all_views.items()
    }
    shuffle_checks = {
        str(origin): {
            "source_position_movement_rate": origin_info[str(origin)]["matched_shuffle_source_position_audit"]["source_position_movement_rate"],
            "minimum": MIN_SHUFFLE_MOVEMENT,
            "pass": origin_info[str(origin)]["matched_shuffle_source_position_audit"]["source_position_movement_rate"] >= MIN_SHUFFLE_MOVEMENT,
        } for origin in ORIGINS
    }
    support_pass = bool(coverage_checks) and all(item["pass"] for item in coverage_checks.values())
    shuffle_pass = all(item["pass"] for item in shuffle_checks.values())
    status = "PASS" if support_pass and shuffle_pass and synthetic.get("status") == "PASS" else "SKIPPED_SCIENTIFIC_PREFLIGHT"
    report = {
        "status": status,
        "pilot_id": "RNBR_IMPACT_MATCH_V1",
        "official_task_fit_started": False,
        "query_targets_read": False,
        "classification_labels_or_probabilities_read": False,
        "test_or_sample_submission_opened": False,
        "kaggle_accessed": False,
        "train_sha256": _sha_file(TRAIN),
        "train_rows": len(train["rows"]),
        "impact_load_missing_rows": int(np.isnan(impacts).sum()),
        "impact_load_finite_unique_values": int(len(np.unique(impacts[np.isfinite(impacts)]))),
        "origin_support": origin_info,
        "support_gate_checks": coverage_checks,
        "shuffle_source_position_checks": shuffle_checks,
        "synthetic_qa_sha256": _sha_file(SYNTHETIC_PATH),
        "minimum_present_query_support_coverage": MIN_SUPPORT_COVERAGE,
        "minimum_shuffle_source_position_movement": MIN_SHUFFLE_MOVEMENT,
    }
    _write_once(PREFLIGHT_PATH, report)
    freeze_body = _build_freeze_body(base["freeze"], report)
    freeze = {**freeze_body, "freeze_signature": _sha256(_canonical(freeze_body))}
    _write_once(FREEZE_PATH, freeze)
    return {"status": status, "freeze_signature": freeze["freeze_signature"],
            "preflight": report, "freeze_path": FREEZE_PATH.relative_to(ROOT).as_posix()}


def _build_freeze_body(base_freeze: Mapping[str, Any], preflight: Mapping[str, Any]) -> dict[str, Any]:
    if _read_json(PROTOCOL_PATH) != _protocol():
        raise ImpactMatchRunError("Impact-match protocol differs from its fixed recipe")
    if _read_json(SYNTHETIC_PATH).get("status") != "PASS":
        raise ImpactMatchRunError("Synthetic wrapper/persistence QA is not PASS")
    geometry_hashes = {}
    for origin in ORIGINS:
        rel = preflight["origin_support"][str(origin)]["geometry_path_ignored"]
        path = ROOT / rel
        if not path.is_file() or _sha_file(path) != preflight["origin_support"][str(origin)]["geometry_file_sha256"]:
            raise ImpactMatchRunError(f"Frozen target-free E{origin} geometry file/hash is absent or changed")
        geometry_hashes[str(origin)] = _sha_file(path)
    return {
        "schema_version": 1,
        "protocol_id": "RNBR_IMPACT_MATCH_V1",
        "authorized_base_head": BASE_HEAD,
        "base_rnbr_freeze_signature": base_freeze["freeze_signature"],
        "protocol_sha256": _sha_file(PROTOCOL_PATH),
        "synthetic_qa_sha256": _sha_file(SYNTHETIC_PATH),
        "targetfree_preflight_sha256": _sha_file(PREFLIGHT_PATH),
        "source_sha256": _source_hashes(),
        "input_sha256": _base_artifact_hashes(base_freeze),
        "official_train": {"path": "data/raw/regression/Train.csv", "rows": 50608,
                           "sha256": base_freeze["official_train"]["sha256"],
                           "target_policy": "donors only after their fit starts; query targets only after all four bundles verify"},
        "origins": [67, 75],
        "fit_ids": list(FIT_IDS),
        "fit_count": 4,
        "geometry_file_sha256": geometry_hashes,
        "preflight_status": preflight["status"],
        "preflight_gates": {
            "support_coverage_minimum": MIN_SUPPORT_COVERAGE,
            "shuffle_source_movement_minimum": MIN_SHUFFLE_MOVEMENT,
        },
        "source_boundaries": {
            "classification_labels_or_probabilities_read": False,
            "official_Test_or_SampleSubmission_opened": False,
            "Kaggle_accessed": False,
            "full_Train_refit": False,
            "preprocessing_refit": False,
        },
        "resource_policy": {"jobs": 1, "threads": 2, "max_rss_bytes": MAX_RSS_BYTES,
                            "minimum_free_ram_bytes": MIN_FREE_RAM_BYTES},
    }


def _current_freeze() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    freeze = _read_json(FREEZE_PATH)
    body = {key: value for key, value in freeze.items() if key != "freeze_signature"}
    if freeze.get("freeze_signature") != _sha256(_canonical(body)):
        raise ImpactMatchRunError("Impact-match freeze self-signature is invalid")
    if _git("status", "--porcelain", "--untracked-files=all"):
        raise ImpactMatchRunError("Pilot fits require committed preparation and a clean working tree")
    if freeze.get("authorized_base_head") != BASE_HEAD:
        raise ImpactMatchRunError("Impact-match freeze is not anchored to the authorized HEAD")
    for rel, expected in freeze["source_sha256"].items():
        if _sha_file(ROOT / rel) != expected:
            raise ImpactMatchRunError(f"Impact-match frozen source changed: {rel}")
    for rel, expected in freeze["input_sha256"].items():
        if _sha_file(ROOT / rel) != expected:
            raise ImpactMatchRunError(f"Impact-match frozen input/lineage changed: {rel}")
    protocol = _read_json(PROTOCOL_PATH)
    preflight = _read_json(PREFLIGHT_PATH)
    if (protocol != _protocol() or _sha_file(PROTOCOL_PATH) != freeze["protocol_sha256"]
            or _sha_file(PREFLIGHT_PATH) != freeze["targetfree_preflight_sha256"]
            or _sha_file(SYNTHETIC_PATH) != freeze["synthetic_qa_sha256"]
            or _build_freeze_body(_read_json(BASE_FREEZE_PATH), preflight) != body):
        raise ImpactMatchRunError("Impact-match protocol, preflight, geometry, or freeze content changed")
    return freeze, protocol, preflight


def _resource_check() -> dict[str, Any]:
    available = int(psutil.virtual_memory().available)
    rss = int(psutil.Process().memory_info().rss)
    if available < MIN_FREE_RAM_BYTES:
        raise ImpactMatchRunError(f"System free RAM is below the 2 GiB reserve: {available}")
    if rss > MAX_RSS_BYTES:
        raise ImpactMatchRunError(f"Pilot process RSS exceeds the 6 GiB limit: {rss}")
    return {"available_ram_bytes": available, "process_rss_bytes": rss,
            "jobs": 1, "threads": 2, "max_rss_bytes": MAX_RSS_BYTES}


def _ledger_events() -> list[dict[str, Any]]:
    return ledger_prior._read_events()


def _ledger_snapshot(events: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    return ledger_prior._snapshot(_ledger_events() if events is None else events)


def _append_ledger(event: Mapping[str, Any]) -> dict[str, Any]:
    return ledger_prior._append_event(event)


def amend_and_reserve(freeze: Mapping[str, Any], source_commit: str) -> dict[str, Any]:
    events = _ledger_events()
    snapshot = _ledger_snapshot(events)
    prior_caps = {"pilot": 38, "continuation": 38, "control_replay": 0, "technical_retry": 1}
    new_caps = {"pilot": 42, "continuation": 38, "control_replay": 0, "technical_retry": 1}
    prior_amendments = [event for event in events if event.get("event") == "BUDGET_CAP_AMENDMENT"
                        and event.get("amendment_id") == AMENDMENT_ID]
    if len(prior_amendments) > 1:
        raise ImpactMatchRunError("Duplicate impact-match cap amendment ID")
    if not prior_amendments:
        expected = {"events": 201, "global_cap": 77, "caps": prior_caps,
                    "active_by_category": prior_caps, "active_reserved": 77,
                    "started": 77, "saved": 74, "failed": 3, "pending": 0}
        if snapshot != expected or _sha_file(LEDGER) != "35ca73c9adc64df8b794db29cc8cd39109362bea865a8bc6b1d8caf0c7703ff8":
            raise ImpactMatchRunError(f"Live fit ledger differs from the reviewed E79 checkpoint: {snapshot}")
        existing_ids = {event.get("fit_id") for event in events if event.get("fit_id")}
        if existing_ids.intersection(FIT_IDS):
            raise ImpactMatchRunError("A new impact-match fit ID already exists; no retry or replay")
        _append_ledger({
            "event": "BUDGET_CAP_AMENDMENT", "batch_id": "optimization_v3_research_next_v1",
            "amendment_id": AMENDMENT_ID, "card_id": CARD_ID,
            "authorized_by": "Sol decision relaying Sam approval for exactly four diagnostic fits",
            "prior_global_cap": 77, "global_cap": 81,
            "prior_effective_caps": prior_caps, "effective_caps": new_caps,
            "active_reservations_before": snapshot["active_by_category"],
            "active_total_before": snapshot["active_reserved"],
            "pre_amendment_event_rows": snapshot["events"],
            "pre_amendment_ledger_sha256": _sha_file(LEDGER),
            "freeze_signature": freeze["freeze_signature"], "source_commit": source_commit,
            "reason": "Append exactly four user-authorized RNBR exact-impact pilot memory fits; other categories unchanged.",
        })
        events = _ledger_events()
    amendment = [event for event in events if event.get("event") == "BUDGET_CAP_AMENDMENT"
                 and event.get("amendment_id") == AMENDMENT_ID]
    if (len(amendment) != 1 or amendment[0].get("prior_global_cap") != 77
            or amendment[0].get("global_cap") != 81 or amendment[0].get("effective_caps") != new_caps
            or amendment[0].get("freeze_signature") != freeze["freeze_signature"]):
        raise ImpactMatchRunError("Existing impact-match budget amendment differs from the approved exact cap")
    snapshot = _ledger_snapshot(events)
    reservations = [event for event in events if event.get("event") == "CARD_FITS_RESERVED"
                    and event.get("reservation_key") == RESERVATION_KEY]
    if len(reservations) > 1:
        raise ImpactMatchRunError("Duplicate impact-match reservation key")
    if not reservations:
        if (snapshot["global_cap"] != 81 or snapshot["caps"] != new_caps
                or snapshot["active_reserved"] != 77 or snapshot["started"] != 77
                or snapshot["pending"] != 0):
            raise ImpactMatchRunError(f"Post-amendment pre-reservation ledger differs: {snapshot}")
        _append_ledger({
            "event": "CARD_FITS_RESERVED", "batch_id": "optimization_v3_research_next_v1",
            "reservation_key": RESERVATION_KEY, "card_id": CARD_ID, "stage": STAGE,
            "category": "pilot", "fits": 4, "fit_ids": list(FIT_IDS),
            "amendment_id": AMENDMENT_ID, "global_cap": 81, "effective_category_cap": 42,
            "freeze_signature": freeze["freeze_signature"], "source_commit": source_commit,
        })
        events = _ledger_events()
    reservation = [event for event in events if event.get("event") == "CARD_FITS_RESERVED"
                   and event.get("reservation_key") == RESERVATION_KEY]
    snapshot = _ledger_snapshot(events)
    starts = [event for event in events if event.get("event") == "FIT_STARTED"
              and event.get("reservation_key") == RESERVATION_KEY]
    if (len(reservation) != 1 or reservation[0].get("fits") != 4
            or reservation[0].get("fit_ids") != list(FIT_IDS)
            or reservation[0].get("freeze_signature") != freeze["freeze_signature"]
            or snapshot["global_cap"] != 81 or snapshot["caps"] != new_caps
            or snapshot["active_reserved"] != 81 or snapshot["started"] != 77 + len(starts)
            or len({event.get("fit_id") for event in starts}) != len(starts)
            or len(starts) > 4 or snapshot["pending"] != 0):
        raise ImpactMatchRunError(f"Post-reservation impact-match ledger differs: {snapshot}")
    return {"amendment": amendment[0], "reservation": reservation[0],
            "snapshot": snapshot, "ledger_sha256": _sha_file(LEDGER)}


def _start_fit(fit_id: str, origin: int, arm: str, freeze: Mapping[str, Any], source_commit: str) -> dict[str, Any]:
    events = _ledger_events()
    existing = [event for event in events if event.get("event") == "FIT_STARTED" and event.get("fit_id") == fit_id]
    terminal = [event for event in events if event.get("fit_id") == fit_id
                and event.get("event") in {"FIT_SAVED", "FIT_FAILED"}]
    if existing or terminal:
        raise ImpactMatchRunError(f"Fit {fit_id} already has ledger history; use durable resume verification, never refit")
    snapshot = _ledger_snapshot(events)
    if snapshot["pending"] != 0:
        raise ImpactMatchRunError("A fit is already pending in the shared ledger")
    reservation = [event for event in events if event.get("event") == "CARD_FITS_RESERVED"
                   and event.get("reservation_key") == RESERVATION_KEY]
    starts = [event for event in events if event.get("event") == "FIT_STARTED"
              and event.get("reservation_key") == RESERVATION_KEY]
    if len(reservation) != 1 or reservation[0].get("fit_ids") != list(FIT_IDS) or len(starts) >= 4:
        raise ImpactMatchRunError("Four-fit impact-match reservation is missing or exhausted")
    resource = _resource_check()
    event = {
        "event": "FIT_STARTED", "batch_id": "optimization_v3_research_next_v1",
        "reservation_key": RESERVATION_KEY, "card_id": CARD_ID, "stage": STAGE,
        "fit_id": fit_id, "cutoff": origin, "component": f"exact_impact_{arm}_residual_memory",
        "fit_consumed_at_start": True, "amendment_id": AMENDMENT_ID,
        "freeze_signature": freeze["freeze_signature"], "source_commit": source_commit,
        "resource_snapshot": resource,
    }
    return _append_ledger(event)


def _target_bundle_inputs(origin: int, freeze: Mapping[str, Any],
                          geometry: Mapping[str, np.ndarray], train: Mapping[str, np.ndarray],
                          cache: Mapping[int, Mapping[str, np.ndarray]]) -> dict[str, np.ndarray]:
    donor_rows = geometry["donor_rows"].astype(np.int64)
    donor_cycles = geometry["donor_cycles"].astype(np.int64)
    donor_ids = geometry["donor_ids"].astype(str)
    targets = official.read_regression_targets(donor_rows, purpose=f"RNBR_IMPACT_MATCH_E{origin}_DONOR_TARGETS")
    if (not np.array_equal(targets["ids"].astype(str), donor_ids)
            or not np.array_equal(targets["cycles"].astype(np.int64), donor_cycles)):
        raise ImpactMatchRunError(f"E{origin} donor target IDs/cycles are not aligned")
    donor_y = targets["target"].astype(np.float64)
    donor_r06 = official.aligned_r06_predictions(donor_rows, donor_cycles, cache)
    cutoffs = np.asarray([EXPECTED_SOURCE_CUTOFFS[int(cycle)] for cycle in donor_cycles], dtype=np.int64)
    if np.any(cutoffs >= donor_cycles):
        raise ImpactMatchRunError(f"E{origin} donor R06 source cutoff is not strictly earlier")
    residuals = donor_y - donor_r06
    if not np.isfinite(residuals).all():
        raise ImpactMatchRunError(f"E{origin} donor residuals are nonfinite")
    return {"donor_rows": donor_rows, "donor_ids": donor_ids, "donor_cycles": donor_cycles,
            "donor_impact_load": geometry["donor_impact_load"].astype(np.float64),
            "donor_source_cutoffs": cutoffs, "donor_targets": donor_y,
            "donor_r06_predictions": donor_r06, "donor_residuals": residuals}


def _fit_arrays(origin: int, arm: str, geometry: Mapping[str, np.ndarray],
                donor: Mapping[str, np.ndarray], source_commit: str) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    residuals = donor["donor_residuals"]
    if arm == "observed":
        residuals_used = residuals.copy()
        shuffle_source_rows = donor["donor_rows"].copy()
        shuffle_audit = {
            "cells": 0, "permutable_cells": 0, "permutable_donors": 0,
            "source_positions_moved": 0, "source_position_movement_rate": 0.0,
            "residual_values_moved": 0, "residual_value_movement_rate": 0.0,
        }
    elif arm == "shuffled":
        residuals_used, shuffle_source_rows, shuffle_audit = impact.shuffle_residuals_within_exact_cells(
            residuals, donor["donor_cycles"], donor["donor_impact_load"],
            donor["donor_r06_predictions"], donor["donor_rows"], seed=42,
        )
    else:
        raise ImpactMatchRunError(f"Unknown impact-match residual arm {arm}")
    if arm == "shuffled" and shuffle_audit["source_position_movement_rate"] < MIN_SHUFFLE_MOVEMENT:
        raise ImpactMatchRunError(f"E{origin} actual shuffle movement fell below the frozen 80% gate")
    predictions = predict_from_saved_neighbors(
        donor_rows=donor["donor_rows"], donor_residuals=residuals_used,
        query_r06_predictions=geometry["baseline_prediction"],
        query_missing=geometry["query_effective_missing"],
        neighbor_rows=geometry["neighbor_rows"],
        neighbor_cosines=geometry["neighbor_similarities"],
        neighbor_counts=geometry["neighbor_count"],
    )
    arrays = {
        **{key: value.copy() for key, value in donor.items() if key != "donor_targets"},
        "score_rows": geometry["query_rows"].astype(np.int64).copy(),
        "score_record_ids": geometry["query_ids"].astype(str).copy(),
        "score_cycles": geometry["query_cycles"].astype(np.int64).copy(),
        "query_impact_load": geometry["query_impact_load"].astype(np.float64).copy(),
        "query_stream_missing": geometry["query_stream_missing"].astype(bool).copy(),
        "query_impact_missing": geometry["query_impact_missing"].astype(bool).copy(),
        "query_effective_missing": geometry["query_effective_missing"].astype(bool).copy(),
        "baseline_prediction": geometry["baseline_prediction"].astype(np.float64).copy(),
        "neighbor_rows": geometry["neighbor_rows"].astype(np.int64).copy(),
        "neighbor_ids": geometry["neighbor_ids"].astype(str).copy(),
        "neighbor_similarities": geometry["neighbor_similarities"].astype(np.float64).copy(),
        "neighbor_count": geometry["neighbor_count"].astype(np.int16).copy(),
        "shuffle_source_rows": shuffle_source_rows.astype(np.int64).copy(),
        "donor_residuals_used": residuals_used.astype(np.float64).copy(),
        "delta": predictions["delta"].astype(np.float64).copy(),
        "prediction": predictions["prediction"].astype(np.float64).copy(),
        "neighbor_count_used": predictions["neighbor_count_used"].astype(np.int16).copy(),
    }
    metadata = {
        "origin": origin, "arm": arm,
        "source_commit": source_commit,
        "freeze_signature": _read_json(FREEZE_PATH)["freeze_signature"],
        "donor_rows": len(donor["donor_rows"]), "query_rows": len(geometry["query_rows"]),
        "donor_rows_sha256": impact.sha256_array(donor["donor_rows"], "<i8"),
        "donor_target_sha256": impact.sha256_array(donor["donor_targets"], "<f8"),
        "donor_r06_oof_sha256": impact.sha256_array(donor["donor_r06_predictions"], "<f8"),
        "donor_residual_sha256": impact.sha256_array(residuals, "<f8"),
        "residuals_used_sha256": impact.sha256_array(residuals_used, "<f8"),
        "shuffle_source_rows_sha256": impact.sha256_array(shuffle_source_rows, "<i8"),
        "shuffle_audit": shuffle_audit,
        "query_targets_read": False,
        "classification_labels_or_probabilities_read": False,
        "test_or_sample_submission_opened": False,
        "fit_recipe": "RNBR_IMPACT_MATCH_V1 frozen protocol",
        "new_estimator_fit_count": 0,
        "memory_construction_fit_count": 1,
    }
    return arrays, metadata


def _fit_dir(fit_id: str) -> Path:
    if fit_id not in FIT_IDS:
        raise ImpactMatchRunError(f"Unknown impact-match fit ID: {fit_id}")
    return RUN_ROOT / "bundles" / fit_id


def _fit_saved_event(fit_id: str, events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [event for event in events if event.get("fit_id") == fit_id and event.get("event") == "FIT_SAVED"]


def _resume_saved_fit(fit_id: str, freeze: Mapping[str, Any], events: list[dict[str, Any]]) -> bool:
    starts = [event for event in events if event.get("fit_id") == fit_id and event.get("event") == "FIT_STARTED"]
    terminal = [event for event in events if event.get("fit_id") == fit_id
                and event.get("event") in {"FIT_SAVED", "FIT_FAILED"}]
    saved = _fit_saved_event(fit_id, events)
    folder = _fit_dir(fit_id)
    if saved:
        if len(starts) != 1 or len(saved) != 1 or terminal != saved:
            raise ImpactMatchRunError(f"Saved fit {fit_id} has duplicate/conflicting accounting")
        arrays, receipt = impact.load_bundle(folder, fit_id=fit_id, freeze_signature=freeze["freeze_signature"])
        digest = _sha_file(folder / "receipt.json")
        if saved[0].get("receipt_sha256") != digest:
            raise ImpactMatchRunError(f"Saved fit {fit_id} ledger receipt hash differs")
        return True
    if terminal:
        raise ImpactMatchRunError(f"Fit {fit_id} is already FIT_FAILED; retry is forbidden")
    if starts:
        if len(starts) != 1 or not folder.is_dir():
            raise ImpactMatchRunError(f"Fit {fit_id} is pending without a durable complete bundle; no retry")
        arrays, receipt = impact.load_bundle(folder, fit_id=fit_id, freeze_signature=freeze["freeze_signature"])
        receipt_hash = _sha_file(folder / "receipt.json")
        _append_ledger({
            "event": "FIT_SAVED", "batch_id": "optimization_v3_research_next_v1",
            "reservation_key": RESERVATION_KEY, "card_id": CARD_ID, "stage": STAGE,
            "fit_id": fit_id, "cutoff": int(receipt["metadata"]["origin"]),
            "receipt_sha256": receipt_hash, "recovered_without_refit": True,
        })
        return True
    if folder.exists():
        raise ImpactMatchRunError(f"Unstarted fit {fit_id} already has outputs; no reuse without lineage")
    return False


def _commit_fit_saved(fit_id: str, origin: int, receipt: Mapping[str, Any]) -> None:
    _append_ledger({
        "event": "FIT_SAVED", "batch_id": "optimization_v3_research_next_v1",
        "reservation_key": RESERVATION_KEY, "card_id": CARD_ID, "stage": STAGE,
        "fit_id": fit_id, "cutoff": origin,
        "receipt_sha256": _sha_file(_fit_dir(fit_id) / "receipt.json"),
    })


def execute() -> dict[str, Any]:
    freeze, protocol, preflight = _current_freeze()
    if preflight.get("status") != "PASS" or freeze.get("preflight_status") != "PASS":
        raise ImpactMatchRunError("Scientific preflight did not pass; no official task fits are allowed")
    source_commit = _git("rev-parse", "HEAD")
    resources = _resource_check()
    ledger = amend_and_reserve(freeze, source_commit)
    train = official.load_targetfree_train()
    frame, docs, impacts = _load_targetfree_features(train)
    base = _check_base_freeze(train=train)
    geometries: dict[int, dict[str, np.ndarray]] = {}
    vectorizers: dict[int, Any] = {}
    for origin in ORIGINS:
        folder = GEOMETRY_DIR / f"E{origin}_geometry.npz"
        with np.load(folder, allow_pickle=False) as archive:
            geom = {key: np.asarray(archive[key]).copy() for key in archive.files}
        vectorizer, audit = _verify_a02_text_state(origin, base["freeze"], train, docs)
        geometries[origin] = geom
        vectorizers[origin] = vectorizer
    donor_cache: dict[int, dict[str, np.ndarray]] = {}
    results: list[dict[str, Any]] = []
    for origin, arm in ((67, "observed"), (67, "shuffled"), (75, "observed"), (75, "shuffled")):
        fit_id = f"RNBR_IMPACT_MATCH_E{origin}_{arm.upper()}"
        events = _ledger_events()
        if _resume_saved_fit(fit_id, freeze, events):
            results.append({"fit_id": fit_id, "status": "FIT_SAVED_VERIFIED_RESUME"})
            if arm == "observed":
                loaded, _ = impact.load_bundle(_fit_dir(fit_id), fit_id=fit_id,
                                               freeze_signature=freeze["freeze_signature"])
                donor_cache[origin] = {key: loaded[key] for key in (
                    "donor_rows", "donor_ids", "donor_cycles", "donor_impact_load",
                    "donor_source_cutoffs", "donor_r06_predictions", "donor_residuals",
                    "donor_residuals_used", "donor_targets") if key in loaded}
            continue
        _resource_check()
        _start_fit(fit_id, origin, arm, freeze, source_commit)
        try:
            if origin not in donor_cache:
                donor_cache[origin] = _target_bundle_inputs(
                    origin, freeze, geometries[origin], train, base["r06_cache"]
                )
            elif "donor_targets" not in donor_cache[origin]:
                donor_cache[origin] = _target_bundle_inputs(
                    origin, freeze, geometries[origin], train, base["r06_cache"]
                )
            arrays, metadata = _fit_arrays(origin, arm, geometries[origin], donor_cache[origin], source_commit)
            metadata["resource_snapshot"] = _resource_check()
            metadata["preflight_coverage"] = preflight["origin_support"][str(origin)]["views"]
            receipt = impact.persist_bundle_once(
                _fit_dir(fit_id), fit_id=fit_id, freeze_signature=freeze["freeze_signature"],
                arrays=arrays, metadata=metadata,
            )
            _commit_fit_saved(fit_id, origin, receipt)
            results.append({"fit_id": fit_id, "status": "FIT_SAVED",
                            "receipt_sha256": _sha_file(_fit_dir(fit_id) / "receipt.json"),
                            "npz_sha256": _sha_file(_fit_dir(fit_id) / "arrays.npz"),
                            "resource_snapshot": metadata["resource_snapshot"]})
        except Exception as exc:
            events = _ledger_events()
            terminal = [event for event in events if event.get("fit_id") == fit_id
                        and event.get("event") in {"FIT_SAVED", "FIT_FAILED"}]
            if not terminal:
                _append_ledger({
                    "event": "FIT_FAILED", "batch_id": "optimization_v3_research_next_v1",
                    "reservation_key": RESERVATION_KEY, "card_id": CARD_ID, "stage": STAGE,
                    "fit_id": fit_id, "cutoff": origin, "receipt_sha256": None,
                    "error": f"{type(exc).__name__}: {exc}",
                })
            raise
        if origin == 67 and arm == "shuffled":
            del vectorizers[origin]
        if origin == 75 and arm == "shuffled":
            del vectorizers[origin]
    if len(results) != 4:
        raise ImpactMatchRunError(f"Expected exactly four saved/resumed fit outputs, got {len(results)}")
    from scripts import optimization_v3_rnbr_impact_match_verify as verify  # noqa: PLC0415
    verification = verify.verify_all()
    return {"preflight_status": preflight["status"], "freeze_signature": freeze["freeze_signature"],
            "source_commit": source_commit, "pre_fit_resources": resources, "ledger": _ledger_snapshot(),
            "fit_results": results, "verification": verification}


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="RNBR exact-impact-match four-fit diagnostic pilot")
    parser.add_argument("--prepare", action="store_true", help="Build target-free preflight and frozen geometry")
    parser.add_argument("--run", action="store_true", help="Run/resume the four authorized fits")
    parser.add_argument("--verify", action="store_true", help="Independently verify saved fit bundles and evaluate")
    parser.add_argument("--synthetic", action="store_true", help="Run only synthetic QA")
    args = parser.parse_args()
    if sum((args.prepare, args.run, args.verify, args.synthetic)) != 1:
        parser.error("select exactly one of --prepare, --run, --verify, or --synthetic")
    if args.synthetic:
        result = synthetic_qa()
    elif args.prepare:
        result = prepare()
    elif args.verify:
        from scripts import optimization_v3_rnbr_impact_match_verify as verify
        result = verify.verify_all()
    else:
        result = execute()
    print(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()

