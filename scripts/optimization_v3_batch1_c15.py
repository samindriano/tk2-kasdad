"""Frozen C15 LATE pilot using genuinely past C2_B and sparse OOF scores."""
from __future__ import annotations

import argparse
import json
import platform
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
import psutil

from scripts.optimization_v3_batch1_verify import verify_report
from scripts.optimization_v3_sparse_classification import (
    CACHE as SPARSE_PILOT_CACHE,
    CONFIG as SPARSE_PILOT_CONFIG,
    check as check_sparse_pilot,
)
from scripts.optimization_v3_sparse_classification_fullsix import (
    CACHE as SPARSE_FULL_CACHE,
    CONFIG as SPARSE_FULL_CONFIG,
    CB_CACHE,
    check_freeze as check_sparse_fullsix,
    control_record,
    pilot_record,
)
from scripts.research_campaign_classification_stage2 import (
    ROUND1_AUDITS, ROUND1_CACHE, _load_round1_control,
)
from src.models.optimization_v3_batch1 import (
    BatchIntegrityError,
    append_event,
    ensure_frozen,
    evaluation_views_for_card,
    file_sha256,
    json_sha256,
    load_queue,
    make_freeze_identity,
    read_bundle,
    run_pilot,
    write_json_exclusive,
)
from src.models.optimization_v3_batch1_c15 import (
    CARD_ID, EARLY_FALLBACK_CUTOFF, LOGIT_EPSILON, PARAMETERS,
    fit_predict,
)
from src.models.classification_round2_models import stream_missing_mask
from src.models.round1_evaluation import array_digest
from src.utils.paths import task_files


ROOT = Path(__file__).resolve().parents[1]
CARD_PATH = ROOT / "reports/optimization_v3/research_backlog/cards/C15.md"
QUEUE_PATH = ROOT / "experiments/optimization_v3/research_backlog/queue.json"
PREFIT_REVISION_01 = ROOT / "experiments/optimization_v3/batch1/C15"
EXPERIMENT_DIR = PREFIT_REVISION_01 / "revision_02"
REPORT_DIR = ROOT / "reports/optimization_v3/batch1/C15/revision_02"
RUN_DIR = ROOT / "experiments/runs/optimization_v3_batch1/C15"
RECIPE_PATH = EXPERIMENT_DIR / "recipe.json"
FREEZE_PATH = EXPERIMENT_DIR / "freeze.json"
ENGINE_REPORT_PATH = RUN_DIR / "pilot_engine_report.json"
RESULTS_PATH = REPORT_DIR / "results.json"
VERIFICATION_PATH = REPORT_DIR / "verification.json"
GATE_PATH = REPORT_DIR / "gate.json"
SUMMARY_PATH = REPORT_DIR / "summary.json"
LEDGER_PATH = RUN_DIR.parent / "batch_ledger.jsonl"
FULLSIX_JSON = ROOT / "reports/optimization_v3/classification_sparse_full/fullsix.json"
FULLSIX_VERIFICATION = ROOT / "reports/optimization_v3/classification_sparse_full/verification.json"
FULLSIX_WRAPPER = ROOT / "reports/optimization_v3/classification_sparse_full/verification_wrapper_run.json"
BASELINE_MANIFEST = ROOT / "reports/optimization_v3/kaggle/v3_cls_sparse25.json"
BASELINE_DECISION = ROOT / "experiments/optimization_v3/decisions/v3_cls_sparse25.json"
CHAMPION_CHECKPOINT = ROOT / "experiments/optimization_v3/champions_checkpoint1.json"

TRAIN_SHA256 = "875552a68a3d24b0f3e1966d6ddb5671e6f54305a1c526e2f0a76747942ed91c"
MANIFEST_SHA256 = "6e7c2cb50fe6d0ee4767a44c40f152b6538b933fab6cf7082659b16d791f7d1e"
FEATURE_NAMES = ("cycle_code", "signal_stream")
OOF_BLOCKS = ((55, 56, 59), (59, 60, 63), (63, 64, 67), (67, 68, 71), (71, 72, 75))
SCORE_CUTOFFS = (55, 67, 75)
THREADS = 2
MIN_FREE_RAM_BYTES = 4 * 1024**3
FREE_RAM_RESERVE_BYTES = 2 * 1024**3
JOB_MEMORY_CAP_BYTES = 6 * 1024**3
ESTIMATED_ADDITIONAL_BYTES = 256 * 1024**2
PACKAGES = ("numpy", "pandas", "scikit-learn", "threadpoolctl", "psutil")
GATE_DEFAULTS = {
    "pilot_primary_gain_min": 0.0005,
    "pilot_H12_harm_max": 0.001,
    "individual_pilot_view_harm_max": 0.005,
    "supported_group_n_min": 200,
    "supported_group_F1_harm_max": 0.01,
    "positive_class_recall_harm_max": 0.02,
    "positive_and_negative_per_group_min": 30,
}


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_or_match(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if _json(path) != dict(value):
            raise BatchIntegrityError(f"Refusing to overwrite differing C15 artifact: {path}")
    else:
        write_json_exclusive(path, dict(value))


def _npz_rows(path: Path, key: str) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(path)
    with np.load(path, allow_pickle=False) as archive:
        if key not in archive.files:
            raise BatchIntegrityError(f"{path} lacks row key {key}")
        rows = np.asarray(archive[key], dtype=np.int64)
    if rows.ndim != 1 or not len(rows) or len(np.unique(rows)) != len(rows):
        raise BatchIntegrityError(f"Invalid source row positions in {path}")
    return rows


def _cache_paths(origin: int) -> tuple[Path, Path, str]:
    c2b = ROUND1_CACHE / f"fit{origin}__C2_B.npz"
    if origin in (55, 67):
        sparse = SPARSE_PILOT_CACHE / f"fit{origin}.npz"
        sparse_row_key = "rows"
    else:
        sparse = SPARSE_FULL_CACHE / f"fit{origin}.npz"
        sparse_row_key = "rows"
    return c2b, sparse, sparse_row_key


def _expected_cache_horizon(origin: int) -> int:
    return 67 if origin == 55 else 79 if origin == 67 else origin + 4


def _source_rows(frame: pd.DataFrame) -> dict[int, dict[str, Any]]:
    cycles = pd.to_numeric(frame["cycle_code"], errors="raise").to_numpy(dtype=np.int64)
    output: dict[int, dict[str, Any]] = {}
    for origin in (55, 59, 63, 67, 71, 75):
        c2b_path, sparse_path, sparse_key = _cache_paths(origin)
        c2b_rows = _npz_rows(c2b_path, "outer_rows")
        sparse_rows = _npz_rows(sparse_path, sparse_key)
        expected = np.flatnonzero((cycles > origin) & (cycles <= _expected_cache_horizon(origin)))
        if not np.array_equal(c2b_rows, expected) or not np.array_equal(sparse_rows, expected):
            raise BatchIntegrityError(f"C15 source cache rows do not match Train cycles at origin {origin}")
        sidecar = _json(sparse_path.with_suffix(".json"))
        if sidecar.get("npz_sha256") != file_sha256(sparse_path):
            raise BatchIntegrityError(f"C15 sparse source checksum mismatch at origin {origin}")
        output[origin] = {"rows": expected, "c2b_path": c2b_path,
                          "sparse_path": sparse_path, "sparse_sidecar": sparse_path.with_suffix(".json")}
    for origin in (55, 67):
        if not (ROUND1_AUDITS / f"fit{origin}.json").is_file():
            raise FileNotFoundError(ROUND1_AUDITS / f"fit{origin}.json")
    return output


def _oof_row_sets(frame: pd.DataFrame, sources: Mapping[int, Mapping[str, Any]]) -> dict[int, dict[str, Any]]:
    cycles = pd.to_numeric(frame["cycle_code"], errors="raise").to_numpy(dtype=np.int64)
    missing = stream_missing_mask(frame["signal_stream"])
    labels = pd.to_numeric(frame["artifact_signal"], errors="raise").to_numpy(dtype=np.int8)
    selected: dict[int, list[np.ndarray]] = {67: [], 75: []}
    provenance: dict[int, list[dict[str, Any]]] = {67: [], 75: []}
    for origin, low, high in OOF_BLOCKS:
        expected = np.flatnonzero((cycles >= low) & (cycles <= high))
        rows = np.asarray(sources[origin]["rows"], dtype=np.int64)
        block = rows[(cycles[rows] >= low) & (cycles[rows] <= high)]
        if not np.array_equal(block, expected) or np.any(cycles[block] <= origin):
            raise BatchIntegrityError(f"C15 OOF block {origin}->{low}-{high} is not strictly prequential")
        for cutoff in (67, 75):
            if high <= cutoff:
                selected[cutoff].append(block)
                provenance[cutoff].append({"origin_cutoff": origin, "cycle_start": low,
                                           "cycle_end": high, "rows": int(len(block))})
    result: dict[int, dict[str, Any]] = {}
    for cutoff, parts in selected.items():
        rows = np.concatenate(parts)
        if len(np.unique(rows)) != len(rows):
            raise BatchIntegrityError(f"C15 OOF rows overlap at cutoff {cutoff}")
        present = ~missing[rows]
        class_counts = {str(int(label)): int(np.sum(labels[rows][present] == label)) for label in (0, 1)}
        result[cutoff] = {
            "rows": rows,
            "total_rows": int(len(rows)),
            "present_rows": int(present.sum()),
            "present_class_counts": class_counts,
            "cycle_min": int(cycles[rows].min()),
            "cycle_max": int(cycles[rows].max()),
            "blocks": provenance[cutoff],
            "support_pass": bool(len(rows) >= 5_000 and min(class_counts.values()) >= 1_000),
        }
    return result


def _confusion_metric(cm: Any) -> dict[str, Any]:
    matrix = np.asarray(cm, dtype=np.int64)
    if matrix.shape != (2, 2) or np.any(matrix < 0):
        raise BatchIntegrityError("Stored binary confusion matrix is invalid")
    tn, fp = (int(matrix[0, 0]), int(matrix[0, 1]))
    fn, tp = (int(matrix[1, 0]), int(matrix[1, 1]))
    f0 = 0.0 if 2 * tn + fp + fn == 0 else 2.0 * tn / (2 * tn + fp + fn)
    f1 = 0.0 if 2 * tp + fp + fn == 0 else 2.0 * tp / (2 * tp + fp + fn)
    return {"n": int(matrix.sum()), "confusion_matrix": matrix.tolist(),
            "macro_f1": float((f0 + f1) / 2.0), "class0_f1": float(f0),
            "class1_f1": float(f1), "positive_recall": float(tp / (tp + fn)) if tp + fn else 0.0}


def _row_matched_baseline() -> dict[str, Any]:
    report = _json(FULLSIX_JSON)
    verification = _json(FULLSIX_VERIFICATION)
    manifest = _json(BASELINE_MANIFEST)
    checkpoint = _json(CHAMPION_CHECKPOINT)
    wrapper = _json(FULLSIX_WRAPPER)
    if (report.get("test_read") is not False or report.get("kaggle_access") is not False
            or verification.get("status") != "PASS" or wrapper.get("status") != "PASS"
            or manifest.get("status") != "PREPARED_AND_QA_PASS"
            or manifest.get("candidate_id") != "v3_cls_sparse25"
            or checkpoint.get("classification", {}).get("manifest") != "reports/optimization_v3/kaggle/v3_cls_sparse25.json"):
        raise BatchIntegrityError("C15 accepted-control lineage is not independently verified")
    recipe = manifest.get("recipe", {})
    expected_recipe = {
        "kind": "availability_present_blend",
        "present": [{"model": "C2_B", "weight": 0.75}, {"model": "SPARSE_BINARY", "weight": 0.25}],
        "missing": "CB_MISSING", "threshold": 0.5,
    }
    if recipe != expected_recipe:
        raise BatchIntegrityError("C15 current champion manifest has a changed route")
    if (manifest.get("input_sha256", {}).get("train") != TRAIN_SHA256
            or manifest.get("fit_preprocessing_only_train") is not True
            or manifest.get("test_labels_used") is not False
            or file_sha256(BASELINE_DECISION) != manifest.get("decision_sha256")):
        raise BatchIntegrityError("C15 accepted champion manifest or decision lineage changed")
    results = report["results"]
    w4 = _confusion_metric(results["windows"]["W4"]["models"]["blend25"]["confusion_matrix"])
    w6 = _confusion_metric(results["windows"]["W6"]["models"]["blend25"]["confusion_matrix"])
    pooled = (np.asarray(w4["confusion_matrix"], dtype=np.int64)
              + np.asarray(w6["confusion_matrix"], dtype=np.int64))
    primary = _confusion_metric(pooled)
    h12 = _confusion_metric(results["H12_late"]["models"]["blend25"]["confusion_matrix"])
    if (w4["n"] != 4_950 or w6["n"] != 8_491 or primary["n"] != 13_441
            or h12["n"] != 20_903
            or not np.isclose(primary["macro_f1"], 0.828146974270008, atol=1e-15, rtol=0.0)
            or not np.isclose(h12["macro_f1"], 0.8089613432399282, atol=1e-15, rtol=0.0)):
        raise BatchIntegrityError("C15 row-matched control differs from the accepted late view")
    return {
        "accepted_manifest": "reports/optimization_v3/kaggle/v3_cls_sparse25.json",
        "accepted_manifest_sha256": file_sha256(BASELINE_MANIFEST),
        "champion_checkpoint_sha256": file_sha256(CHAMPION_CHECKPOINT),
        "fullsix_report_sha256": file_sha256(FULLSIX_JSON),
        "fullsix_verification_sha256": file_sha256(FULLSIX_VERIFICATION),
        "fullsix_wrapper_sha256": file_sha256(FULLSIX_WRAPPER),
        "route": expected_recipe,
        "W4": w4,
        "W6": w6,
        "primary_W4_plus_W6": primary,
        "H12_L": h12,
        "provenance_note": "W4/W6 confusion matrices were summed before computing Macro F1; controls use the accepted v3_cls_sparse25 full-six report.",
    }


def _q_edges(frame: pd.DataFrame) -> list[float]:
    cycles = pd.to_numeric(frame["cycle_code"], errors="raise").to_numpy(dtype=np.int64)
    values = pd.to_numeric(frame.loc[(cycles >= 47) & (cycles <= 55), "text_load"], errors="coerce").to_numpy(dtype=float)
    values = values[np.isfinite(values)]
    edges = np.quantile(values, [0.25, 0.50, 0.75]).astype(float).tolist()
    if not (edges[0] < edges[1] < edges[2]):
        raise BatchIntegrityError("C15 fit-defined text_load quartile edges are tied or unordered")
    return edges


def _group_bins(frame: pd.DataFrame, rows: np.ndarray, edges: list[float]) -> np.ndarray:
    value = pd.to_numeric(frame.iloc[rows]["text_load"], errors="coerce").to_numpy(dtype=float)
    result = np.full(len(rows), -1, dtype=np.int8)
    finite = np.isfinite(value)
    e1, e2, e3 = edges
    result[finite & (value <= e1)] = 0
    result[finite & (value > e1) & (value <= e2)] = 1
    result[finite & (value > e2) & (value <= e3)] = 2
    result[finite & (value > e3)] = 3
    return result


def _all_input_files(train_path: Path, card: Mapping[str, Any]) -> dict[str, Path]:
    files: dict[str, Path] = {
        "official_classification_train": train_path,
        "canonical_queue": QUEUE_PATH,
        "canonical_card": CARD_PATH,
        "accepted_sparse25_manifest": BASELINE_MANIFEST,
        "accepted_sparse25_decision": BASELINE_DECISION,
        "champion_checkpoint": CHAMPION_CHECKPOINT,
        "accepted_fullsix_report": FULLSIX_JSON,
        "accepted_fullsix_verification": FULLSIX_VERIFICATION,
        "accepted_fullsix_wrapper": FULLSIX_WRAPPER,
        "sparse_pilot_protocol": SPARSE_PILOT_CONFIG / "protocol.json",
        "sparse_pilot_freeze": SPARSE_PILOT_CONFIG / "freeze.json",
        "sparse_pilot_report": ROOT / "reports/optimization_v3/classification_sparse/pilot.json",
        "sparse_pilot_verification": ROOT / "reports/optimization_v3/classification_sparse/verification.json",
        "sparse_fullsix_protocol": SPARSE_FULL_CONFIG / "protocol.json",
        "sparse_fullsix_freeze": SPARSE_FULL_CONFIG / "freeze.json",
        "superseded_prefit_recipe_revision_01": PREFIT_REVISION_01 / "recipe.json",
        "superseded_prefit_freeze_revision_01": PREFIT_REVISION_01 / "freeze.json",
    }
    for origin in (55, 59, 63, 67, 71, 75):
        c2b_path, sparse_path, _ = _cache_paths(origin)
        files[f"round1_c2b_{origin}"] = c2b_path
        files[f"round1_c2b_audit_{origin}"] = ROUND1_AUDITS / f"fit{origin}.json"
        files[f"sparse_oof_{origin}"] = sparse_path
        files[f"sparse_oof_sidecar_{origin}"] = sparse_path.with_suffix(".json")
    for cutoff in SCORE_CUTOFFS:
        files[f"catboost_missing_{cutoff}"] = CB_CACHE / f"fit{cutoff}.npz"
        files[f"catboost_missing_sidecar_{cutoff}"] = CB_CACHE / f"fit{cutoff}.json"
    return files


def _source_files() -> list[Path]:
    return [
        Path(__file__), ROOT / "src/models/optimization_v3_batch1_c15.py",
        ROOT / "src/models/optimization_v3_batch1.py",
        ROOT / "scripts/optimization_v3_batch1_verify.py",
        ROOT / "scripts/optimization_v3_batch1.py",
        ROOT / "scripts/optimization_v3_sparse_classification.py",
        ROOT / "scripts/optimization_v3_sparse_classification_fullsix.py",
        ROOT / "scripts/optimization_v3_verify_sparse_classification.py",
        ROOT / "scripts/optimization_v3_verify_sparse_classification_fullsix.py",
        ROOT / "scripts/research_campaign_classification_stage2.py",
        ROOT / "src/models/optimization_v3_sparse_classification.py",
        ROOT / "src/models/classification_round2_models.py",
        ROOT / "src/models/classification_models.py",
        ROOT / "src/models/round1_models.py",
        ROOT / "src/models/round1_evaluation.py",
        ROOT / "src/utils/paths.py",
        ROOT / "tests/test_optimization_v3_batch1_c15.py",
    ]


def _load_official_train() -> tuple[Path, pd.DataFrame, np.ndarray, np.ndarray]:
    train_path = task_files("classification").get("train")
    if train_path is None or not Path(train_path).is_file():
        raise FileNotFoundError("Official Classification Train was not found through task_files")
    path = Path(train_path)
    frame = pd.read_csv(path)
    required = {"record_id", "cycle_code", "signal_stream", "text_load", "artifact_signal"}
    if not required.issubset(frame.columns) or frame["record_id"].isna().any() or frame["record_id"].astype(str).duplicated().any():
        raise BatchIntegrityError("Official Train schema or record_id identity is invalid")
    target = pd.to_numeric(frame["artifact_signal"], errors="raise").to_numpy(dtype=np.int8)
    cycles = pd.to_numeric(frame["cycle_code"], errors="raise").to_numpy(dtype=np.int64)
    if (file_sha256(path) != TRAIN_SHA256 or len(frame) != 50_608
            or not np.array_equal(np.unique(cycles), np.arange(47, 80))
            or not np.isin(target, [0, 1]).all()):
        raise BatchIntegrityError("Official Classification Train differs from frozen source identity")
    return path, frame, target, cycles


def _validate_card(card: Mapping[str, Any], queue: Mapping[str, Any]) -> None:
    canonical = next((item for item in queue.get("cards", []) if item.get("id") == CARD_ID), None)
    if canonical is None or dict(card) != dict(canonical):
        raise BatchIntegrityError("C15 card is not the canonical queue object")
    expected = {"id": "C15", "task": "classification", "family": "score_meta",
                "policy": "C_META", "pilot_predictor_fits": 2, "evaluation": "LATE",
                "condition": "No additional dependency"}
    if any(card.get(key) != value for key, value in expected.items()):
        raise BatchIntegrityError("C15 card identity or declared pilot contract changed")
    if not CARD_PATH.is_file() or "C15 — Two-score prequential meta-head" not in CARD_PATH.read_text(encoding="utf-8"):
        raise BatchIntegrityError("C15 human-readable card is missing or changed")
    gate = queue.get("evaluation", {}).get("classification_gate", {})
    if any(gate.get(k) != v for k, v in GATE_DEFAULTS.items()):
        raise BatchIntegrityError("C15 canonical classification gates changed")
    if (int(queue.get("execution_preset", {}).get("concurrent_fit_jobs", -1)) != 1
            or int(queue.get("execution_preset", {}).get("threads_per_job", -1)) != THREADS
            or float(queue.get("execution_preset", {}).get("max_job_RSS_GiB", -1)) != 6.0
            or float(queue.get("execution_preset", {}).get("minimum_free_RAM_GiB_before_start", -1)) != 4.0):
        raise BatchIntegrityError("C15 shared execution resource policy changed")


def _resource_preflight() -> dict[str, Any]:
    available = int(psutil.virtual_memory().available)
    free_gib = available / 1024**3
    extra_allowed = min(JOB_MEMORY_CAP_BYTES, available - FREE_RAM_RESERVE_BYTES)
    status = "PASS" if (available >= MIN_FREE_RAM_BYTES
                        and extra_allowed >= ESTIMATED_ADDITIONAL_BYTES) else "BLOCKED_RESOURCE"
    return {
        "status": status,
        "available_bytes": available,
        "available_gib": float(free_gib),
        "minimum_free_bytes": MIN_FREE_RAM_BYTES,
        "reserve_bytes": FREE_RAM_RESERVE_BYTES,
        "estimated_additional_bytes": ESTIMATED_ADDITIONAL_BYTES,
        "allowed_additional_bytes": int(max(0, extra_allowed)),
        "job_cap_bytes": JOB_MEMORY_CAP_BYTES,
        "estimate_is_observed_peak": False,
        "threads": THREADS,
        "concurrent_fit_jobs": 1,
    }


def _budget_preflight(card: Mapping[str, Any], queue: Mapping[str, Any]) -> dict[str, Any]:
    events = []
    if LEDGER_PATH.exists():
        events = [json.loads(line) for line in LEDGER_PATH.read_text(encoding="utf-8").splitlines() if line.strip()]
    reservations = [event for event in events if event.get("event") == "BUDGET_RESERVED"]
    existing = [event for event in reservations if event.get("budget_key") == "C15:PILOT"]
    reserved = sum(int(event.get("predictor_fits", 0)) for event in reservations)
    cap = int(queue["execution_preset"]["max_successful_new_predictor_fits"])
    expected = int(card["pilot_predictor_fits"])
    if len(existing) > 1 or (existing and int(existing[0].get("predictor_fits", -1)) != expected):
        raise BatchIntegrityError("Existing C15 fit budget reservation is inconsistent")
    total = reserved if existing else reserved + expected
    return {"status": "PASS" if total <= cap else "BLOCKED_BUDGET",
            "reserved_total": int(reserved), "card_fits": expected,
            "next_total": int(total), "cap": cap,
            "existing_reservation": existing[0] if existing else None}


def _prepare(card: Mapping[str, Any], queue: Mapping[str, Any]) -> dict[str, Any]:
    _validate_card(card, queue)
    baseline = _row_matched_baseline()
    train_path, frame, target, cycles = _load_official_train()
    sources = _source_rows(frame)
    oof_support = _oof_row_sets(frame, sources)
    if not all(item["support_pass"] for item in oof_support.values()):
        raise BatchIntegrityError(f"C15 OOF class/row support gate failed: {oof_support}")
    views = evaluation_views_for_card(queue, CARD_ID)
    views["H12_E"] = [(EARLY_FALLBACK_CUTOFF, 56, 67)]
    expected_views = {"W4": [(67, 68, 71)], "W6": [(75, 76, 79)],
                      "primary": [(67, 68, 71), (75, 76, 79)],
                      "H12_L": [(67, 68, 79)], "H12_E": [(55, 56, 67)]}
    if views != expected_views:
        raise BatchIntegrityError(f"C15 fixed LATE evaluation views changed: {views}")
    expected_rows = {"W4": 4_950, "W6": 8_491, "primary": 13_441,
                     "H12_L": 20_903, "H12_E": 17_685}
    actual_rows = {
        name: int(sum(np.sum(np.isin(cycles, np.arange(low, high + 1))) for _origin, low, high in segments))
        for name, segments in views.items()
    }
    if actual_rows != expected_rows:
        raise BatchIntegrityError(f"C15 declared score denominators differ from official Train: {actual_rows}")
    if (oof_support[67]["total_rows"] != 17_685 or oof_support[67]["present_rows"] != 16_482
            or oof_support[75]["total_rows"] != 30_097 or oof_support[75]["present_rows"] != 28_045):
        raise BatchIntegrityError("C15 observed OOF support differs from the verified preflight snapshot")

    fit_rows = {cutoff: np.flatnonzero(cycles <= cutoff) for cutoff in (55, 67, 75)}
    score_rows = {
        cutoff: np.flatnonzero(np.isin(cycles, np.asarray(sorted({c for segments in views.values()
                           for origin, low, high in segments if origin == cutoff
                           for c in range(low, high + 1)}), dtype=np.int64)))
        for cutoff in (55, 67, 75)
    }
    for cutoff, expected_n in {55: 12_020, 67: 29_705, 75: 42_117}.items():
        if len(fit_rows[cutoff]) != expected_n:
            raise BatchIntegrityError(f"C15 outer fit population differs at cutoff {cutoff}")
    edges = _q_edges(frame)
    inputs = _all_input_files(train_path, card)
    missing_paths = [str(path) for path in inputs.values() if not path.is_file()]
    if missing_paths:
        raise FileNotFoundError(f"C15 frozen source inputs are missing: {missing_paths}")
    source_files = _source_files()
    identity = make_freeze_identity(card=card, feature_names=FEATURE_NAMES,
                                   input_files=inputs, source_files=source_files,
                                   evaluation_views=views, packages=PACKAGES)
    recipe = {
        "card_id": CARD_ID,
        "revision": "revision_02; supersedes one prefit-only revision_01 after stricter cache/source audits were added; no candidate prediction values were read before this revision.",
        "queue_recipe": {key: card[key] for key in ("id", "task", "name", "family", "change", "policy",
                        "pilot_predictor_fits", "evaluation", "condition")},
        "meta_model": {
            "class": "sklearn.linear_model.LogisticRegression",
            "parameters": {**PARAMETERS, "penalty": "l2 (scikit-learn default)"},
            "input_features": ["logit(C2_B probability)", "logit(raw sparse-binary probability)"],
            "clip_epsilon": LOGIT_EPSILON,
            "threshold": 0.50,
        },
        "oof_design": {
            "blocks": [dict(origin_cutoff=o, cycle_start=lo, cycle_end=hi,
                            rows=int(len(sources[o]["rows"][(cycles[sources[o]["rows"]] >= lo)
                                                          & (cycles[sources[o]["rows"]] <= hi)])))
                       for o, lo, hi in OOF_BLOCKS],
            "cutoff67_support": {k: v for k, v in oof_support[67].items() if k != "rows"},
            "cutoff75_support": {k: v for k, v in oof_support[75].items() if k != "rows"},
            "fit_population": "present-stream official Train labels from exact disjoint OOF blocks strictly before each outer cutoff; rows from cycles 47-55 excluded",
        },
        "policy": {
            "cutoff55": "exact current champion fallback because authorized OOF support is unavailable",
            "cutoff67_and_75_present": "C15 logistic regression fitted only on present-stream OOF rows",
            "missing_stream": "exact accepted CatBoost probability",
            "baseline": "v3_cls_sparse25; 0.75*C2_B + 0.25*raw sparse probability when present, CatBoost when missing",
            "threshold": 0.50,
        },
        "views": views,
        "expected_rows": expected_rows,
        "row_matched_control": baseline,
        "fit_rows_by_cutoff": {str(k): int(len(v)) for k, v in fit_rows.items()},
        "subgroup_edges": {"definition": "text_load quartiles from unlabeled feature values in cycles 47-55, matching accepted full-six report",
                           "edges": edges},
        "gate": GATE_DEFAULTS,
        "resource_policy": {"minimum_free_bytes": MIN_FREE_RAM_BYTES,
                            "reserve_bytes": FREE_RAM_RESERVE_BYTES,
                            "estimated_additional_bytes": ESTIMATED_ADDITIONAL_BYTES,
                            "job_memory_cap_bytes": JOB_MEMORY_CAP_BYTES,
                            "threads": THREADS},
        "no_test_sample_or_kaggle": True,
    }
    identity.update({"recipe": recipe, "row_matched_controls": baseline,
                     "oof_support": {str(k): {x: y for x, y in v.items() if x != "rows"}
                                     for k, v in oof_support.items()},
                     "expected_rows": expected_rows,
                     "resource_policy": {"minimum_free_bytes": MIN_FREE_RAM_BYTES,
                                         "reserve_bytes": FREE_RAM_RESERVE_BYTES,
                                         "estimated_additional_bytes": ESTIMATED_ADDITIONAL_BYTES,
                                         "job_memory_cap_bytes": JOB_MEMORY_CAP_BYTES,
                                         "threads": THREADS}})
    identity["recipe_artifact_sha256"] = json_sha256(recipe)
    resource = _resource_preflight()
    budget = _budget_preflight(card, queue)
    freeze_valid = (not RECIPE_PATH.exists() or _json(RECIPE_PATH) == recipe)
    if FREEZE_PATH.exists():
        frozen = _json(FREEZE_PATH)
        freeze_valid = freeze_valid and frozen.get("signature") == json_sha256(identity) and frozen.get("identity") == identity
    status = "PREFLIGHT_PASS" if resource["status"] == "PASS" and budget["status"] == "PASS" and freeze_valid else "BLOCKED_RESOURCE_OR_BUDGET_OR_FREEZE"
    input_hashes = {name: file_sha256(path) for name, path in sorted(inputs.items())}
    source_hashes = {str(path.relative_to(ROOT).as_posix()): file_sha256(path) for path in source_files}
    public = {
        "card_id": CARD_ID, "status": status,
        "scope": "official Classification Train only; no candidate prediction values read during preflight",
        "official_train_sha256": TRAIN_SHA256,
        "input_sha256": input_hashes,
        "source_sha256": source_hashes,
        "freeze_identity_sha256": json_sha256(identity),
        "support": {str(k): {name: value for name, value in v.items() if name != "rows"} for k, v in oof_support.items()},
        "row_matched_control": baseline,
        "score_rows": expected_rows,
        "fit_rows": {str(k): int(len(v)) for k, v in fit_rows.items()},
        "text_load_quartile_edges": edges,
        "resource_preflight": resource,
        "budget_preflight": budget,
        "candidate_fit_or_metric_computed": False,
        "oof_prediction_values_read": False,
        "test_sample_or_kaggle_access": False,
        "python_version": platform.python_version(),
    }
    return {"public": public, "identity": identity, "recipe": recipe, "frame": frame,
            "target": target, "cycles": cycles, "views": views,
            "sources": sources, "oof_support": oof_support, "score_rows_by_cutoff": score_rows,
            "fit_rows_by_cutoff": fit_rows, "edges": edges}


def _load_oof_probabilities(context: Mapping[str, Any]) -> dict[int, dict[str, np.ndarray]]:
    frame = context["frame"]
    target = context["target"]
    cycles = context["cycles"]
    all_rows: list[int] = []
    all_c2b: list[float] = []
    all_sparse: list[float] = []
    all_labels: list[int] = []
    all_cycles: list[int] = []
    all_origins: list[int] = []
    all_stream: list[Any] = []
    for origin, low, high in OOF_BLOCKS:
        source = context["sources"][origin]
        source_rows = np.asarray(source["rows"], dtype=np.int64)
        block_rows = source_rows[(cycles[source_rows] >= low) & (cycles[source_rows] <= high)]
        horizon = _expected_cache_horizon(origin)
        round1_rows, c2b_full, _audit = _load_round1_control(frame, origin, horizon, "C2_B")
        if origin in (55, 67):
            with np.load(source["sparse_path"], allow_pickle=False) as archive:
                sparse_rows = np.asarray(archive["rows"], dtype=np.int64)
                sparse_full = np.asarray(archive["raw_candidate"], dtype=np.float64)
            sidecar = _json(source["sparse_sidecar"])
            if (sidecar.get("npz_sha256") != file_sha256(source["sparse_path"])
                    or sidecar.get("freeze_signature") != _json(SPARSE_PILOT_CONFIG / "freeze.json").get("signature")
                    or not np.array_equal(sparse_rows, source_rows)
                    or sparse_full.shape != source_rows.shape
                    or not np.isfinite(sparse_full).all()
                    or np.any((sparse_full < 0) | (sparse_full > 1))):
                raise BatchIntegrityError(f"C15 sparse pilot OOF source failed integrity at {origin}")
        else:
            with np.load(source["sparse_path"], allow_pickle=False) as archive:
                sparse_rows = np.asarray(archive["rows"], dtype=np.int64)
                sparse_full = np.asarray(archive["raw_candidate"], dtype=np.float64)
            sidecar = _json(source["sparse_sidecar"])
            if (sidecar.get("npz_sha256") != file_sha256(source["sparse_path"])
                    or int(sidecar.get("cutoff", -1)) != origin
                    or int(sidecar.get("horizon", -1)) != _expected_cache_horizon(origin)
                    or not np.array_equal(sparse_rows, source_rows)
                    or sparse_full.shape != source_rows.shape
                    or not np.isfinite(sparse_full).all()
                    or np.any((sparse_full < 0) | (sparse_full > 1))
                    or sidecar.get("raw_prediction_sha256") != array_digest(sparse_full)):
                raise BatchIntegrityError(f"C15 sparse full-six OOF source failed integrity at {origin}")
        c2b_lookup = {int(row): float(value) for row, value in zip(round1_rows, c2b_full)}
        sparse_lookup = {int(row): float(value) for row, value in zip(sparse_rows, sparse_full)}
        if not np.array_equal(round1_rows, sparse_rows):
            raise BatchIntegrityError(f"C15 C2_B and sparse OOF rows differ at origin {origin}")
        all_rows.extend(block_rows.tolist())
        all_c2b.extend(c2b_lookup[int(row)] for row in block_rows)
        all_sparse.extend(sparse_lookup[int(row)] for row in block_rows)
        all_labels.extend(target[block_rows].tolist())
        all_cycles.extend(cycles[block_rows].tolist())
        all_origins.extend([origin] * len(block_rows))
        all_stream.extend(frame.iloc[block_rows]["signal_stream"].tolist())

    rows = np.asarray(all_rows, dtype=np.int64)
    if len(np.unique(rows)) != len(rows) or not np.array_equal(np.sort(rows), rows):
        raise BatchIntegrityError("C15 assembled OOF rows are not sorted, unique Train positions")
    arrays = {
        "rows": rows,
        "c2b_probability": np.asarray(all_c2b, dtype=np.float64),
        "sparse_probability": np.asarray(all_sparse, dtype=np.float64),
        "labels": np.asarray(all_labels, dtype=np.int8),
        "cycles": np.asarray(all_cycles, dtype=np.int64),
        "origin_cutoff": np.asarray(all_origins, dtype=np.int64),
        "stream_missing": stream_missing_mask(pd.Series(all_stream)),
        "signal_stream": np.asarray(all_stream, dtype=object),
    }
    output: dict[int, dict[str, np.ndarray]] = {}
    for cutoff in (67, 75):
        mask = arrays["cycles"] <= cutoff
        output[cutoff] = {key: value[mask] for key, value in arrays.items()}
    return output


def _load_controls(frame: pd.DataFrame, target: np.ndarray, edges: list[float], sources: Mapping[int, Any]):
    check_sparse_pilot()
    check_sparse_fullsix()

    def load(cutoff: int, score_rows: np.ndarray) -> dict[str, np.ndarray]:
        record = control_record(frame, target, int(cutoff))
        if not np.array_equal(record["rows"], score_rows):
            raise BatchIntegrityError(f"C15 accepted C2_B/CB route is not row-aligned at {cutoff}")
        if cutoff in (55, 67):
            sparse_record = pilot_record(frame, record)
            raw_sparse = np.asarray(sparse_record["raw_candidate"], dtype=np.float64)
            baseline = np.asarray(sparse_record["blend25"], dtype=np.float64)
        elif cutoff == 75:
            path = sources[75]["sparse_path"]
            with np.load(path, allow_pickle=False) as archive:
                sparse_rows = np.asarray(archive["rows"], dtype=np.int64)
                raw_sparse = np.asarray(archive["raw_candidate"], dtype=np.float64)
                baseline = np.asarray(archive["blend25"], dtype=np.float64)
            sidecar = _json(path.with_suffix(".json"))
            if (sidecar.get("npz_sha256") != file_sha256(path)
                    or sidecar.get("raw_prediction_sha256") != array_digest(raw_sparse)
                    or sidecar.get("blend_prediction_sha256") != array_digest(baseline)
                    or not np.array_equal(sparse_rows, score_rows)):
                raise BatchIntegrityError("C15 cutoff75 sparse incumbent rows/hash differ")
        else:
            raise BatchIntegrityError(f"Unauthorized C15 control cutoff: {cutoff}")
        missing = np.asarray(record["missing"], dtype=bool)
        c2b = np.asarray(record["c2b"], dtype=np.float64)
        cb = np.asarray(record["cb_missing"], dtype=np.float64)
        expected = np.where(missing, cb, 0.75 * c2b + 0.25 * raw_sparse)
        if not np.array_equal(expected, baseline):
            raise BatchIntegrityError(f"C15 accepted baseline fails exact route reconstruction at {cutoff}")
        return {
            "baseline_prediction": baseline,
            "baseline_sparse_raw": raw_sparse,
            "c2b_probability": c2b,
            "cb_probability": cb,
            "stream_missing": missing,
            "control_rows": np.asarray(score_rows, dtype=np.int64),
            "text_load_bin": _group_bins(frame, score_rows, edges),
        }
    return load


def _metric_independent(y: np.ndarray, probability: np.ndarray) -> dict[str, Any]:
    actual = np.asarray(y, dtype=np.int8)
    pred = np.asarray(probability, dtype=np.float64) >= 0.50
    if actual.ndim != 1 or pred.shape != actual.shape or not len(actual):
        raise BatchIntegrityError("C15 independent metric inputs are empty or misaligned")
    tn = int(np.sum((actual == 0) & (~pred)))
    fp = int(np.sum((actual == 0) & pred))
    fn = int(np.sum((actual == 1) & (~pred)))
    tp = int(np.sum((actual == 1) & pred))
    f0 = 2 * tn / (2 * tn + fp + fn) if (2 * tn + fp + fn) else 0.0
    f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0
    return {"n": int(len(actual)), "macro_f1": float((f0 + f1) / 2.0),
            "confusion_matrix": [[tn, fp], [fn, tp]],
            "positive_recall": float(tp / (tp + fn)) if tp + fn else 0.0,
            "class_counts": {"0": int(np.sum(actual == 0)), "1": int(np.sum(actual == 1))}}


def _independent_qa(context: Mapping[str, Any], engine: Mapping[str, Any], oof_support: Mapping[int, Any]) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    frame, target, cycles, views = context["frame"], context["target"], context["cycles"], context["views"]
    cutoffs = sorted({int(origin) for segments in views.values() for origin, _, _ in segments})
    by_cutoff: dict[int, dict[str, np.ndarray]] = {}
    for cutoff in cutoffs:
        score_rows = context["score_rows_by_cutoff"][cutoff]
        identity = {"freeze_signature": engine["freeze_signature"], "card_id": CARD_ID,
                    "cutoff": cutoff, "score_rows": score_rows.tolist()}
        arrays, sidecar = read_bundle(RUN_DIR / f"C15_fit{cutoff}.npz", identity)
        if not np.array_equal(arrays["score_rows"], score_rows):
            raise BatchIntegrityError(f"C15 saved score rows changed at {cutoff}")
        if not np.array_equal(arrays["score_record_ids"], frame.iloc[score_rows]["record_id"].astype(str).to_numpy()):
            raise BatchIntegrityError(f"C15 saved record_id ordering changed at {cutoff}")
        baseline = np.asarray(arrays["baseline_prediction"], dtype=np.float64)
        prediction = np.asarray(arrays["prediction"], dtype=np.float64)
        raw = np.asarray(arrays["raw_candidate"], dtype=np.float64)
        missing = np.asarray(arrays["control__stream_missing"], dtype=bool)
        cb = np.asarray(arrays["control__cb_probability"], dtype=np.float64)
        c2b = np.asarray(arrays["control__c2b_probability"], dtype=np.float64)
        old_sparse = np.asarray(arrays["control__baseline_sparse_raw"], dtype=np.float64)
        if (not np.isfinite(prediction).all() or not np.isfinite(raw).all()
                or np.any((prediction < 0) | (prediction > 1))
                or not np.array_equal(baseline, np.where(missing, cb, 0.75 * c2b + 0.25 * old_sparse))):
            raise BatchIntegrityError(f"C15 saved probabilities or baseline route invalid at {cutoff}")
        expected_prediction = np.where(missing, cb, raw)
        if cutoff == EARLY_FALLBACK_CUTOFF:
            if (int(sidecar["metadata"].get("fit_predictor_count", -1)) != 0
                    or not np.array_equal(prediction, baseline) or not np.array_equal(raw, baseline)):
                raise BatchIntegrityError("C15 early horizon did not use exact current-route fallback")
        else:
            metadata = sidecar["metadata"]
            state = metadata.get("state", {})
            expected_support = oof_support[cutoff]
            expected_origins = {str(block["origin_cutoff"]): int(block["rows"])
                                for block in expected_support["blocks"]}
            if (int(metadata.get("fit_predictor_count", -1)) != 1
                    or int(metadata.get("fit_rows", -1)) != len(context["fit_rows_by_cutoff"][cutoff])
                    or state.get("id") != "C15_TWO_SCORE_PREQUENTIAL_LOGISTIC_REGRESSION"
                    or int(state.get("cutoff", -1)) != cutoff
                    or int(state.get("oof_fit_rows", -1)) != int(expected_support["total_rows"])
                    or int(state.get("oof_present_fit_rows", -1)) != int(expected_support["present_rows"])
                    or state.get("oof_present_class_counts") != expected_support["present_class_counts"]
                    or state.get("oof_origin_cutoffs") != expected_origins
                    or int(state.get("oof_cycle_max", -1)) > cutoff
                    or state.get("score_labels_passed_to_fit") is not False
                    or state.get("test_data_used") is not False):
                raise BatchIntegrityError(f"C15 cutoff {cutoff} fit count is not one")
            if not np.array_equal(prediction[missing], cb[missing]):
                raise BatchIntegrityError(f"C15 CatBoost missing route changed at {cutoff}")
        if not np.array_equal(prediction, expected_prediction):
            raise BatchIntegrityError(f"C15 candidate routing does not reconstruct at {cutoff}")
        by_cutoff[cutoff] = {"rows": score_rows, "prediction": prediction,
                              "baseline": baseline, "stream_missing": missing,
                              "text_load_bin": np.asarray(arrays["control__text_load_bin"], dtype=np.int8),
                              "metadata": sidecar["metadata"]}

    report_views: dict[str, Any] = {}
    subgroup_failures: list[dict[str, Any]] = []
    gate = GATE_DEFAULTS
    for name, segments in views.items():
        row_parts, prediction_parts, baseline_parts, miss_parts, bin_parts = [], [], [], [], []
        for cutoff, low, high in segments:
            source = by_cutoff[int(cutoff)]
            mask = (cycles[source["rows"]] >= low) & (cycles[source["rows"]] <= high)
            selected = source["rows"][mask]
            row_parts.append(selected)
            prediction_parts.append(source["prediction"][mask])
            baseline_parts.append(source["baseline"][mask])
            miss_parts.append(source["stream_missing"][mask])
            bin_parts.append(source["text_load_bin"][mask])
        rows = np.concatenate(row_parts)
        if len(np.unique(rows)) != len(rows):
            raise BatchIntegrityError(f"C15 rows overlap in {name}")
        pred = np.concatenate(prediction_parts)
        base = np.concatenate(baseline_parts)
        miss = np.concatenate(miss_parts)
        bins = np.concatenate(bin_parts)
        candidate_metric = _metric_independent(target[rows], pred)
        baseline_metric = _metric_independent(target[rows], base)
        delta = float(candidate_metric["macro_f1"] - baseline_metric["macro_f1"])
        groups: dict[str, np.ndarray] = {"stream_present": ~miss, "stream_missing": miss}
        for value, label in enumerate(("text_load_q1_low_tail", "text_load_q2", "text_load_q3", "text_load_q4_high_tail")):
            groups[label] = bins == value
        subgroup_results: dict[str, Any] = {}
        for group_name, mask in groups.items():
            if not mask.any():
                subgroup_results[group_name] = {"n": 0, "status": "UNSUPPORTED_EMPTY"}
                continue
            y_group = target[rows][mask]
            cand_group = _metric_independent(y_group, pred[mask])
            base_group = _metric_independent(y_group, base[mask])
            support = cand_group["class_counts"]
            supported = (cand_group["n"] >= int(gate["supported_group_n_min"])
                         and min(support.values()) >= int(gate["positive_and_negative_per_group_min"]))
            group_delta = cand_group["macro_f1"] - base_group["macro_f1"]
            recall_delta = cand_group["positive_recall"] - base_group["positive_recall"]
            subgroup_results[group_name] = {
                "status": "SUPPORTED" if supported else "DESCRIPTIVE_UNSUPPORTED",
                "n": cand_group["n"], "class_counts": support,
                "baseline_macro_f1": base_group["macro_f1"],
                "candidate_macro_f1": cand_group["macro_f1"],
                "delta_macro_f1": float(group_delta),
                "baseline_positive_recall": base_group["positive_recall"],
                "candidate_positive_recall": cand_group["positive_recall"],
                "delta_positive_recall": float(recall_delta),
            }
            if supported:
                if group_delta < -float(gate["supported_group_F1_harm_max"]):
                    subgroup_failures.append({"view": name, "group": group_name,
                                              "guard": "macro_f1_harm", "delta": float(group_delta)})
                if recall_delta < -float(gate["positive_class_recall_harm_max"]):
                    subgroup_failures.append({"view": name, "group": group_name,
                                              "guard": "positive_recall_harm", "delta": float(recall_delta)})
        report_views[name] = {
            "rows": int(len(rows)), "candidate": candidate_metric,
            "baseline": baseline_metric, "gain_positive_is_better": delta,
            "subgroups": subgroup_results,
        }

    p = report_views["primary"]["gain_positive_is_better"]
    h = report_views["H12_L"]["gain_positive_is_better"]
    w4 = report_views["W4"]["gain_positive_is_better"]
    w6 = report_views["W6"]["gain_positive_is_better"]
    checks = {
        "primary_gain_at_least_0_0005": bool(p >= float(gate["pilot_primary_gain_min"])),
        "H12_L_harm_at_most_0_001": bool(h >= -float(gate["pilot_H12_harm_max"])),
        "W4_harm_at_most_0_005": bool(w4 >= -float(gate["individual_pilot_view_harm_max"])),
        "W6_harm_at_most_0_005": bool(w6 >= -float(gate["individual_pilot_view_harm_max"])),
        "supported_subgroup_guards": not subgroup_failures,
        "OOF_support_and_chronology": all(bool(v["support_pass"]) for v in oof_support.values()),
    }
    decision = "ADVANCE" if all(checks.values()) else "RETIRE"
    verification = {
        "card_id": CARD_ID, "status": "INDEPENDENT_QA_PASS", "decision": decision,
        "views": report_views, "gate_checks": checks,
        "subgroup_failures": subgroup_failures,
        "fit_cutoffs": [67, 75], "early_fallback_cutoff": 55,
        "test_data_used": False, "sample_submission_used": False, "kaggle_access": False,
        "candidate_fit_or_metric_computed_before_freeze": False,
        "reconstruction_method": "Rebuilt policy and Macro F1 from hash-verified saved predictions, controls, and Train labels; subgroup metrics recomputed from saved masks independently of runner metric helper.",
    }
    gate_report = {
        "card_id": CARD_ID, "decision": decision,
        "status": "PASS" if all(checks.values()) else "FAIL_VALID_EVIDENCE",
        "thresholds": gate, "checks": checks,
        "subgroup_failures": subgroup_failures,
        "decision_basis": "Frozen LATE W4+W6/H12_L gates with exact v3_cls_sparse25 row-matched controls; early H12_E remains exact fallback.",
    }
    results = {
        **dict(engine), "status": f"PILOT_COMPLETE_{decision}",
        "row_matched_control_provenance": context["public"]["row_matched_control"],
        "oof_support": context["public"]["support"],
        "independent_views": report_views,
        "gate": gate_report,
        "test_data_used": False, "sample_submission_used": False, "kaggle_access": False,
    }
    return results, verification, gate_report


def run_card(card: Mapping[str, Any], queue: Mapping[str, Any], *,
             preflight_only: bool = False, freeze_only: bool = False,
             run_frozen_pilot: bool = False) -> dict[str, Any]:
    if sum((preflight_only, freeze_only, run_frozen_pilot)) != 1:
        raise BatchIntegrityError("Choose exactly one C15 mode")
    context = _prepare(card, queue)
    if preflight_only:
        return context["public"]
    if context["public"]["status"] != "PREFLIGHT_PASS":
        raise BatchIntegrityError("C15 blocked by RAM, budget, or freeze preflight; no freeze or fit may start")
    if freeze_only:
        EXPERIMENT_DIR.mkdir(parents=True, exist_ok=True)
        _write_or_match(RECIPE_PATH, context["recipe"])
        signature = ensure_frozen(FREEZE_PATH, context["identity"])
        result = {**context["public"], "status": "FROZEN_READY",
                  "recipe_path": str(RECIPE_PATH.relative_to(ROOT)),
                  "freeze_path": str(FREEZE_PATH.relative_to(ROOT)),
                  "freeze_signature": signature,
                  "candidate_fit_or_metric_computed": False,
                  "oof_prediction_values_read": False}
        return result

    if (not RECIPE_PATH.is_file() or not FREEZE_PATH.is_file()
            or _json(RECIPE_PATH) != context["recipe"]
            or _json(FREEZE_PATH).get("identity") != context["identity"]
            or _json(FREEZE_PATH).get("signature") != json_sha256(context["identity"])):
        raise BatchIntegrityError("C15 run requires the exact committed prefit recipe and freeze")
    frame, target, views = context["frame"], context["target"], context["views"]
    oof_by_cutoff = _load_oof_probabilities(context)
    controls_loader = _load_controls(frame, target, context["edges"], context["sources"])
    prefit_resource = _resource_preflight()
    if prefit_resource["status"] != "PASS":
        raise BatchIntegrityError(f"C15 fit blocked by fresh post-cache-load resource check: {prefit_resource}")
    adapter = lambda fit_x, fit_y, score_x, **kwargs: fit_predict(
        fit_x, fit_y, score_x, oof=oof_by_cutoff.get(int(kwargs["cutoff"]), {}), **kwargs)
    # One locked job reserves both predictor fits atomically; the cutoff-55 fallback counts zero.
    engine = run_pilot(
        card=card, artifact_dir=RUN_DIR, budget_ledger_path=LEDGER_PATH,
        max_successful_fits=int(queue["execution_preset"]["max_successful_new_predictor_fits"]),
        task="classification", frame=frame, target=target, features=FEATURE_NAMES,
        evaluation_views=views, adapter=adapter, control_loader=controls_loader,
        freeze_path=FREEZE_PATH, freeze_identity=context["identity"],
        report_path=ENGINE_REPORT_PATH, threads=THREADS,
    )
    results, verification, gate = _independent_qa(context, engine, context["oof_support"])
    # Generic verifier independently checks the shared runner's view metrics and fit count.
    runner_verification = verify_report(
        report=engine, frame=frame, target=target, artifact_dir=RUN_DIR,
        freeze_signature=engine["freeze_signature"], evaluation_views=views,
    )
    if runner_verification.get("status") != "PASS":
        raise BatchIntegrityError("C15 shared-engine report did not pass its independent generic verifier")
    accounting = {
        "budget_reserved_predictor_fits": 2,
        "successful_predictor_fits": 2,
        "cutoff_fits": {str(cutoff): _json((RUN_DIR / f"C15_fit{cutoff}.json"))["metadata"]
                         for cutoff in (55, 67, 75)},
        "resource_preflight_after_cache_load": prefit_resource,
        "resource_peak_observed": False,
    }
    results["fit_accounting"] = accounting
    verification["shared_runner_verification"] = runner_verification
    _write_or_match(RESULTS_PATH, results)
    _write_or_match(VERIFICATION_PATH, verification)
    _write_or_match(GATE_PATH, gate)
    summary = {
        "card_id": CARD_ID, "decision": gate["decision"],
        "primary_rows": results["independent_views"]["primary"]["rows"],
        "primary_baseline_macro_f1": results["independent_views"]["primary"]["baseline"]["macro_f1"],
        "primary_candidate_macro_f1": results["independent_views"]["primary"]["candidate"]["macro_f1"],
        "primary_gain": results["independent_views"]["primary"]["gain_positive_is_better"],
        "H12_L_rows": results["independent_views"]["H12_L"]["rows"],
        "H12_L_baseline_macro_f1": results["independent_views"]["H12_L"]["baseline"]["macro_f1"],
        "H12_L_candidate_macro_f1": results["independent_views"]["H12_L"]["candidate"]["macro_f1"],
        "H12_L_gain": results["independent_views"]["H12_L"]["gain_positive_is_better"],
        "W4_gain": results["independent_views"]["W4"]["gain_positive_is_better"],
        "W6_gain": results["independent_views"]["W6"]["gain_positive_is_better"],
        "supported_subgroup_failures": len(gate["subgroup_failures"]),
        "fit_predictor_count": 2,
        "test_sample_kaggle_used": False,
        "details": ["results.json", "verification.json", "gate.json"],
    }
    _write_or_match(SUMMARY_PATH, summary)
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--preflight-only", action="store_true")
    modes.add_argument("--freeze-only", action="store_true")
    modes.add_argument("--run-frozen-pilot", action="store_true")
    args = parser.parse_args(argv)
    queue = load_queue(QUEUE_PATH)
    card = next(item for item in queue["cards"] if item.get("id") == CARD_ID)
    result = run_card(card, queue, preflight_only=args.preflight_only,
                      freeze_only=args.freeze_only, run_frozen_pilot=args.run_frozen_pilot)
    print(json.dumps(result, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["run_card", "main"]
