"""R06 character TF-IDF sparse L1 regression pilot on frozen V3 controls."""
from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import subprocess
from typing import Any, Mapping

import numpy as np
import pandas as pd
import psutil
from sklearn.feature_extraction.text import TfidfVectorizer

from scripts import optimization_v3_batch1_r02 as r02
from scripts.optimization_v3_batch1_verify import verify_report
from src.models.optimization_v3_batch1 import (
    BatchIntegrityError,
    append_event,
    ensure_frozen,
    evaluation_views_for_card,
    file_sha256,
    json_sha256,
    make_freeze_identity,
    package_versions,
    read_bundle,
    run_pilot,
    write_json_exclusive,
)
from src.models.optimization_v3_batch1_r06 import (
    CHAR_TFIDF_PARAMETERS,
    FEATURE_NAMES,
    SPARSE_LGB_PARAMETERS,
    fit_predict,
)
from src.models.regression_research_text import _text_documents
from src.utils.paths import task_files


ROOT = Path(__file__).resolve().parents[1]
QUEUE_PATH = ROOT / "experiments/optimization_v3/research_backlog/queue.json"
CARD_PATH = ROOT / "reports/optimization_v3/research_backlog/cards/R06.md"
EXPERIMENT_DIR = ROOT / "experiments/optimization_v3/batch1/R06"
RUN_DIR = ROOT / "experiments/runs/optimization_v3_batch1/R06"
REPORT_DIR = ROOT / "reports/optimization_v3/batch1/R06"
RECIPE_PATH = EXPERIMENT_DIR / "recipe.json"
FREEZE_PATH = EXPERIMENT_DIR / "freeze.json"
PREFLIGHT_PATH = REPORT_DIR / "preflight.json"
ENGINE_REPORT_PATH = RUN_DIR / "pilot.json"
RESULTS_PATH = REPORT_DIR / "results.json"
VERIFICATION_PATH = REPORT_DIR / "verification.json"
SUBGROUP_PATH = REPORT_DIR / "subgroups.json"
GATE_PATH = REPORT_DIR / "gate.json"
SUMMARY_PATH = REPORT_DIR / "summary.md"
ATTEMPT_PATH = REPORT_DIR / "attempts.jsonl"
LEDGER_PATH = RUN_DIR.parent / "batch_ledger.jsonl"

EXPECTED_TRAIN_SHA256 = "756523ff8156a9cd776e52249f7e2a8f23dcec432b8d10570e55ff6db0e0f082"
EXPECTED_QUEUE_SHA256 = "822dcd20fa28fd0e9ddc841a8c3709b372dacaca305b94b39b11a9203418fec7"
EXPECTED_CARD_SHA256 = "abc21ce6bdb5c4537a119f7edd394bb649c6e0df88f6eaecacc5e28e69bcf253"
EXPECTED_TRAIN_ROWS = 50_608
EXPECTED_CYCLES = np.arange(47, 80, dtype=np.int64)
EXPECTED_VIEW_ROWS = {"W1": 4_114, "W4": 4_950, "primary": 9_064, "H12": 38_588}
EXPECTED_VIEWS = {
    "W1": [(55, 56, 59)],
    "W4": [(67, 68, 71)],
    "primary": [(55, 56, 59), (67, 68, 71)],
    "H12": [(55, 56, 67), (67, 68, 79)],
}
THREADS = 2
JOB_MEMORY_CAP_BYTES = 6 * 1024**3
FREE_MEMORY_RESERVE_BYTES = 2 * 1024**3
MIN_FREE_MEMORY_BYTES = 4 * 1024**3
LIGHTGBM_MEMORY_OVERHEAD_BYTES = 512 * 1024**2
STRUCTURED_NNZ_UPPER_PER_ROW = 64
STRUCTURED_WIDTH_UPPER = 1_000
EXPECTED_CARD = {
    "id": "R06",
    "task": "regression",
    "name": "Character sparse L1 boosting",
    "family": "character_boost",
    "priority": "P3",
    "change": "char_wb(3,5)/min_df3/cap30000/sublinear TF-IDF plus original structured/OHE; original sparse L1 model. Different learner from failed character Ridge; low-confidence hash fragments.",
    "policy": "R_AUX25",
    "pilot_predictor_fits": 2,
    "evaluation": "STANDARD",
    "condition": "No additional dependency",
    "status": "PLANNED_NOT_RUN",
    "implementation": "NOT_IMPLEMENTED",
    "card_path": "reports/optimization_v3/research_backlog/cards/R06.md",
}
PACKAGES = ["numpy", "pandas", "scipy", "scikit-learn", "lightgbm", "threadpoolctl", "psutil"]


def _json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise BatchIntegrityError(f"Cannot read valid JSON from {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise BatchIntegrityError(f"Expected a JSON object in {path}")
    return value


def _write_or_match(path: Path, payload: Mapping[str, Any]) -> None:
    if path.exists():
        if _json(path) != dict(payload):
            raise BatchIntegrityError(f"Existing R06 artifact differs; refusing overwrite: {path}")
    else:
        write_json_exclusive(path, dict(payload))


def _write_text_or_match(path: Path, payload: str) -> None:
    if path.exists():
        if path.read_text(encoding="utf-8") != payload:
            raise BatchIntegrityError(f"Existing R06 summary differs; refusing overwrite: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(payload)


def _head_commit() -> str:
    result = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, check=True,
                            capture_output=True, text=True)
    return result.stdout.strip()


def _source_files() -> list[Path]:
    return sorted({
        *r02._source_files(),
        Path(__file__),
        ROOT / "src/models/optimization_v3_batch1_r06.py",
        ROOT / "tests/test_optimization_v3_batch1_r06_model.py",
        ROOT / "tests/test_optimization_v3_batch1_r06_runner.py",
    }, key=str)


def _input_files(train_path: Path, card: Mapping[str, Any], queue: Mapping[str, Any]) -> dict[str, Path]:
    files = r02._input_files(train_path, card, queue)
    files["canonical_queue"] = QUEUE_PATH
    files["canonical_card"] = CARD_PATH
    return files


def _stable_resource_identity(resource: Mapping[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in resource.items()
            if key not in {"available_memory_bytes", "allowed_memory_bytes"}}


def _resource_preflight(frame: pd.DataFrame, cycles: np.ndarray) -> dict[str, Any]:
    params = dict(CHAR_TFIDF_PARAMETERS)
    analyzer = TfidfVectorizer(**params).build_analyzer()
    min_df = int(params["min_df"])
    feature_cap = int(params["max_features"])
    cutoffs: dict[str, Any] = {}
    available = int(psutil.virtual_memory().available)
    allowed = min(JOB_MEMORY_CAP_BYTES, max(0, available - FREE_MEMORY_RESERVE_BYTES))
    for cutoff in (55, 67):
        fit_mask = cycles <= cutoff
        documents, missing = _text_documents(frame.loc[fit_mask, "signal_stream"])
        document_frequency: Counter[str] = Counter()
        unique_ngrams = 0
        analyzed_occurrences = 0
        nonempty_documents = 0
        for document in documents:
            terms = analyzer(document)
            analyzed_occurrences += len(terms)
            unique = set(terms)
            unique_ngrams += len(unique)
            if unique:
                nonempty_documents += 1
                document_frequency.update(unique)
        eligible = sum(count >= min_df for count in document_frequency.values())
        fit_rows = int(fit_mask.sum())
        char_nnz_upper = min(unique_ngrams, fit_rows * feature_cap)
        structured_nnz_upper = fit_rows * STRUCTURED_NNZ_UPPER_PER_ROW
        combined_nnz_upper = char_nnz_upper + structured_nnz_upper + fit_rows
        csr_bytes_upper = combined_nnz_upper * 12 + (fit_rows + 1) * 8
        width_upper = min(eligible, feature_cap) + STRUCTURED_WIDTH_UPPER + 1
        histogram_bytes_upper = width_upper * 256 * 16 * THREADS * 2
        estimated_peak = (3 * csr_bytes_upper + histogram_bytes_upper
                          + LIGHTGBM_MEMORY_OVERHEAD_BYTES)
        support_status = "SKIPPED_SUPPORT" if eligible == 0 else "PASS"
        status = support_status if support_status == "SKIPPED_SUPPORT" else (
            "PASS" if available >= MIN_FREE_MEMORY_BYTES and estimated_peak <= allowed
            else "BLOCKED_RESOURCE"
        )
        cutoffs[str(cutoff)] = {
            "status": status,
            "fit_rows": fit_rows,
            "fit_missing_stream_rows": int(missing.sum()),
            "nonempty_documents_with_char_ngrams": nonempty_documents,
            "eligible_char_ngrams_at_min_df": int(eligible),
            "feature_cap": feature_cap,
            "analyzed_char_ngram_occurrences_upper_bound": int(analyzed_occurrences),
            "char_matrix_nnz_upper_bound": int(char_nnz_upper),
            "structured_nnz_upper_bound": int(structured_nnz_upper),
            "csr_bytes_upper_bound": int(csr_bytes_upper),
            "histogram_bytes_upper_bound": int(histogram_bytes_upper),
            "estimated_peak_bytes": int(estimated_peak),
            "estimated_peak_is_observed": False,
        }
    return {
        "status": "SKIPPED_SUPPORT" if any(x["status"] == "SKIPPED_SUPPORT" for x in cutoffs.values())
        else "PASS" if all(x["status"] == "PASS" for x in cutoffs.values()) else "BLOCKED_RESOURCE",
        "cutoffs": cutoffs,
        "available_memory_bytes": available,
        "free_memory_reserve_bytes": FREE_MEMORY_RESERVE_BYTES,
        "allowed_memory_bytes": int(allowed),
        "minimum_free_memory_bytes": MIN_FREE_MEMORY_BYTES,
        "job_memory_cap_bytes": JOB_MEMORY_CAP_BYTES,
        "threads": THREADS,
        "concurrent_training_jobs": 1,
        "sparse_only": True,
        "dense_full_text_matrix": False,
    }


def _control_views(
    frame: pd.DataFrame,
    target: np.ndarray,
    cycles: np.ndarray,
    views: Mapping[str, list[tuple[int, int, int]]],
    controls: Mapping[int, Mapping[str, np.ndarray]],
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for name, segments in views.items():
        rows_parts: list[np.ndarray] = []
        pred_parts: list[np.ndarray] = []
        for cutoff, low, high in segments:
            record = controls[int(cutoff)]
            rows = np.asarray(record["rows"], dtype=np.int64)
            mask = (cycles[rows] >= int(low)) & (cycles[rows] <= int(high))
            rows_parts.append(rows[mask])
            pred_parts.append(np.asarray(record["baseline_prediction"], dtype=np.float64)[mask])
        rows = np.concatenate(rows_parts)
        predictions = np.concatenate(pred_parts)
        if len(np.unique(rows)) != len(rows):
            raise BatchIntegrityError(f"R06 control view {name} contains duplicate rows")
        expected_rows = np.concatenate([
            np.flatnonzero((cycles >= int(low)) & (cycles <= int(high)))
            for _, low, high in segments
        ])
        if not np.array_equal(rows, expected_rows):
            raise BatchIntegrityError(f"R06 control view {name} does not match exact Train row order")
        metric = float(np.mean(np.abs(predictions - target[rows])))
        result[name] = {"rows": int(len(rows)), "mae": metric}
    if "H12" in result:
        if result["H12"]["rows"] != EXPECTED_VIEW_ROWS["H12"]:
            raise BatchIntegrityError("R06 H12 control denominator differs from frozen queue")
        expected_h12 = float(json.loads((ROOT / "experiments/optimization_v3/research_backlog/queue.json").read_text(encoding="utf-8"))["baseline"]["regression"]["H12"])
        if not np.isclose(result["H12"]["mae"], expected_h12, atol=1e-10, rtol=0.0):
            raise BatchIntegrityError(f"R06 three-component H12 control differs: {result['H12']['mae']}")
    return result


def _control_preflight(
    frame: pd.DataFrame,
    queue: Mapping[str, Any],
    inputs: Mapping[str, Path],
    views: Mapping[str, list[tuple[int, int, int]]],
    target: np.ndarray,
    cycles: np.ndarray,
) -> tuple[dict[int, dict[str, np.ndarray]], dict[str, Any]]:
    cache_hashes = r02._validated_cache_hashes(queue, inputs)
    manifest = _json(inputs["accepted_regression_manifest"])
    if manifest.get("task") != "regression" or manifest.get("status") != "PREPARED_AND_QA_PASS":
        raise BatchIntegrityError("R06 accepted regression manifest is not QA-passed")
    controls = r02._load_control_rows(frame, views)
    metrics = _control_views(frame, target, cycles, views, controls)
    control_identity = {
        "control_source": "R02 independently reconstructed frozen three-component champion",
        "cache_hashes": cache_hashes,
        "accepted_manifest_sha256": file_sha256(inputs["accepted_regression_manifest"]),
        "views": metrics,
        "evaluation_views": views,
        "test_data_read": False,
    }
    return controls, control_identity


def _candidate_recipe() -> dict[str, Any]:
    params: dict[str, Any] = {}
    for key, value in TfidfVectorizer(**CHAR_TFIDF_PARAMETERS).get_params(deep=False).items():
        if isinstance(value, tuple):
            params[key] = list(value)
        elif isinstance(value, type):
            params[key] = value.__name__
        else:
            params[key] = value
    return {
        "card_id": "R06",
        "model": "LightGBM regression_l1; sparse structured/OHE + char_wb TF-IDF + missing indicator",
        "feature_allowlist": list(FEATURE_NAMES),
        "text_vectorizer": params,
        "structured_preprocessing": "RegressionTextModel fit-only numeric imputation/scaling and OneHotEncoder; original R02 structured design",
        "missing_stream_indicator": True,
        "sparse_lgb_parameters": dict(SPARSE_LGB_PARAMETERS),
        "policy": "clip(0.75*current_three_component_champion + 0.25*rank50(raw R06),0,100)",
        "rank50": "within each complete score cycle; average ties; percentile=100*rank/n; rank50=0.5*raw+0.5*percentile; clip[0,100]",
        "views": EXPECTED_VIEWS,
        "profile": "STANDARD",
        "fit_rows": {"55": 12_020, "67": 29_705},
        "score_rows": {"55": 17_685, "67": 20_903},
        "predictor_fits": 2,
        "preprocessing_fit_calls_per_cutoff": {"character_tfidf": 1, "structured_processor_scaler_ohe": 1},
        "test_data_used": False,
        "semantic_interpretation": "Character n-grams are anonymous text fragments; no semantic interpretation is made.",
    }


def _source_commit_for_recipe() -> str:
    if RECIPE_PATH.is_file():
        recipe_record = _json(RECIPE_PATH)
        source_commit = recipe_record.get("source_commit")
        if not isinstance(source_commit, str) or not source_commit:
            raise BatchIntegrityError("Existing R06 recipe lacks a source commit")
        return source_commit
    return _head_commit()


def _prepare(card: Mapping[str, Any], queue: Mapping[str, Any]) -> dict[str, Any]:
    mismatches = {key: (card.get(key), value) for key, value in EXPECTED_CARD.items()
                  if card.get(key) != value}
    if mismatches:
        raise BatchIntegrityError(f"R06 canonical card assertions failed: {mismatches}")
    if queue.get("campaign_V3_complete") is not False:
        raise BatchIntegrityError("R06 runs only while V3 campaign remains unfinished")
    queue_hash = file_sha256(QUEUE_PATH)
    if queue_hash != EXPECTED_QUEUE_SHA256:
        raise BatchIntegrityError(f"R06 canonical queue hash differs: {queue_hash}")
    card_hash = file_sha256(CARD_PATH)
    if card_hash != EXPECTED_CARD_SHA256:
        raise BatchIntegrityError(f"R06 canonical card hash differs: {card_hash}")

    train_path = task_files("regression").get("train")
    if train_path is None or not train_path.is_file():
        raise FileNotFoundError("Official Regression Train is missing")
    train_hash = file_sha256(train_path)
    if train_hash != EXPECTED_TRAIN_SHA256:
        raise BatchIntegrityError("Official Regression Train hash differs from R06 card")
    frame = pd.read_csv(train_path)
    if len(frame) != EXPECTED_TRAIN_ROWS:
        raise BatchIntegrityError(f"R06 expected {EXPECTED_TRAIN_ROWS} official Train rows")
    required = set(FEATURE_NAMES) | {"record_id", "cycle_code", "threat_rank"}
    absent = sorted(required - set(frame.columns))
    if absent:
        raise BatchIntegrityError(f"Official Regression Train is missing fields: {absent}")
    forbidden = set(FEATURE_NAMES) & {"record_id", "artifact_signal", "threat_rank"}
    if forbidden:
        raise BatchIntegrityError(f"R06 feature allowlist includes protected fields: {sorted(forbidden)}")
    if (frame["record_id"].isna().any()
            or frame["record_id"].astype(str).str.strip().eq("").any()
            or not frame["record_id"].astype(str).is_unique):
        raise BatchIntegrityError("Regression Train record_id values are blank or duplicated")
    cycles = pd.to_numeric(frame["cycle_code"], errors="raise").to_numpy(dtype=np.int64)
    if not np.array_equal(np.unique(cycles), EXPECTED_CYCLES):
        raise BatchIntegrityError("Regression Train cycles differ from expected 47–79")
    target = pd.to_numeric(frame["threat_rank"], errors="raise").to_numpy(dtype=np.float64)
    if (target.shape != (len(frame),) or not np.isfinite(target).all()
            or np.any((target < 0) | (target > 100))):
        raise BatchIntegrityError("Regression threat_rank is invalid")

    views = evaluation_views_for_card(queue, "R06")
    if views != EXPECTED_VIEWS:
        raise BatchIntegrityError(f"R06 temporal evaluation views changed: {views}")
    inputs = _input_files(train_path, card, queue)
    controls, control_identity = _control_preflight(frame, queue, inputs, views, target, cycles)
    control_metrics = control_identity["views"]
    expected_rows = {name: int(sum(np.sum((cycles >= low) & (cycles <= high))
                                   for _, low, high in segments))
                     for name, segments in views.items()}
    if expected_rows != EXPECTED_VIEW_ROWS:
        raise BatchIntegrityError(f"R06 Train row counts differ from frozen views: {expected_rows}")

    resource = _resource_preflight(frame, cycles)
    recipe = _candidate_recipe()
    recipe_record = {
        "recipe": recipe,
        "queue_sha256": queue_hash,
        "card_sha256": card_hash,
        "source_commit": _source_commit_for_recipe(),
    }
    if RECIPE_PATH.exists() and _json(RECIPE_PATH) != recipe_record:
        raise BatchIntegrityError("Existing R06 recipe differs; refusing to overwrite")
    identity = make_freeze_identity(
        card=card,
        feature_names=FEATURE_NAMES,
        input_files=inputs,
        source_files=_source_files(),
        evaluation_views=views,
        packages=PACKAGES,
    )
    identity.update({
        "candidate_recipe": recipe,
        "candidate_recipe_sha256": json_sha256(recipe),
        "recipe_record_sha256": json_sha256(recipe_record),
        "source_commit": recipe_record["source_commit"],
        "control_preflight_identity_sha256": json_sha256(control_identity),
        "control_cache_hashes": control_identity["cache_hashes"],
        "resource_preflight": _stable_resource_identity(resource),
        "python_version": platform.python_version(),
    })
    baseline = queue["baseline"]["regression"]
    summary_preflight = {
        "status": "PREFLIGHT_PASS" if resource["status"] == "PASS" else resource["status"],
        "card_id": "R06",
        "official_train_sha256": train_hash,
        "train_rows": int(len(frame)),
        "fit_rows": {"55": int(np.sum(cycles <= 55)), "67": int(np.sum(cycles <= 67))},
        "score_rows": {"55": int(np.sum((cycles >= 56) & (cycles <= 67))),
                       "67": int(np.sum((cycles >= 68) & (cycles <= 79)))},
        "view_rows": expected_rows,
        "baseline_canonical_fullsix_primary_mae": float(baseline["primary"]),
        "baseline_pilot_control_metrics": control_metrics,
        "accepted_cache_sha256": control_identity["cache_hashes"],
        "resource": resource,
        "freeze_identity_sha256": json_sha256(identity),
        "source_sha256": identity["source_sha256"],
        "input_sha256": identity["input_sha256"],
        "official_test_read": False,
        "candidate_fit_performed": False,
    }
    return {
        "card": card,
        "queue": queue,
        "frame": frame,
        "target": target,
        "cycles": cycles,
        "train_path": train_path,
        "views": views,
        "inputs": inputs,
        "controls": controls,
        "recipe": recipe,
        "recipe_record": recipe_record,
        "identity": identity,
        "preflight": summary_preflight,
        "resource": resource,
        "control_metrics": control_metrics,
    }


def _independent_rank50(values: np.ndarray, cycles: np.ndarray) -> np.ndarray:
    raw = np.asarray(values, dtype=np.float64)
    cycle_codes = np.asarray(cycles, dtype=np.int64)
    if raw.ndim != 1 or raw.shape != cycle_codes.shape or not len(raw) or not np.isfinite(raw).all():
        raise BatchIntegrityError("R06 rank50 verifier inputs must be finite aligned vectors")
    mapped = np.empty(len(raw), dtype=np.float64)
    for cycle in np.unique(cycle_codes):
        mask = cycle_codes == cycle
        group = raw[mask]
        order = np.argsort(group, kind="mergesort")
        ranks = np.empty(len(group), dtype=np.float64)
        start = 0
        while start < len(order):
            stop = start + 1
            while stop < len(order) and group[order[stop]] == group[order[start]]:
                stop += 1
            ranks[order[start:stop]] = ((start + 1) + stop) / 2.0
            start = stop
        mapped[mask] = 0.5 * group + 0.5 * (100.0 * ranks / len(group))
    return np.clip(mapped, 0.0, 100.0)


def reconstruct_r06(arrays: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    raw = np.asarray(arrays["raw_candidate"], dtype=np.float64)
    sparse_raw = np.asarray(arrays["control__baseline_sparse_raw"], dtype=np.float64)
    ridge25 = np.asarray(arrays["control__ridge25_prediction"], dtype=np.float64)
    cycles = np.asarray(arrays["control__cycle_code"], dtype=np.int64)
    saved_baseline = np.asarray(arrays["baseline_prediction"], dtype=np.float64)
    if len({len(raw), len(sparse_raw), len(ridge25), len(cycles), len(saved_baseline)}) != 1:
        raise BatchIntegrityError("R06 reconstruction inputs are not aligned")
    baseline = np.clip(0.75 * ridge25 + 0.25 * _independent_rank50(sparse_raw, cycles), 0.0, 100.0)
    if not np.allclose(saved_baseline, baseline, atol=1e-10, rtol=0.0):
        raise BatchIntegrityError("R06 saved three-component baseline does not reconstruct from R02 cache components")
    candidate = np.clip(0.75 * baseline + 0.25 * _independent_rank50(raw, cycles), 0.0, 100.0)
    return candidate, baseline


def _subgroup_diagnostics(
    report: Mapping[str, Any], frame: pd.DataFrame, target: np.ndarray,
    views: Mapping[str, list[tuple[int, int, int]]],
) -> dict[str, Any]:
    cycles = pd.to_numeric(frame["cycle_code"], errors="raise").to_numpy(dtype=np.int64)
    bundles: dict[int, dict[str, np.ndarray]] = {}
    for cutoff in report["cutoffs"]:
        npz_path = RUN_DIR / f"R06_fit{int(cutoff)}.npz"
        sidecar = _json(npz_path.with_suffix(".json"))
        arrays, _ = read_bundle(npz_path, sidecar["identity"])
        bundles[int(cutoff)] = {
            "rows": np.asarray(arrays["score_rows"], dtype=np.int64),
            "candidate": np.asarray(arrays["prediction"], dtype=np.float64),
            "baseline": np.asarray(arrays["baseline_prediction"], dtype=np.float64),
        }
    output: dict[str, Any] = {}
    failures: list[dict[str, Any]] = []
    for view in ("primary", "H12"):
        row_parts: list[np.ndarray] = []
        pred_parts: list[np.ndarray] = []
        base_parts: list[np.ndarray] = []
        for cutoff, low, high in views[view]:
            block = bundles[int(cutoff)]
            mask = (cycles[block["rows"]] >= int(low)) & (cycles[block["rows"]] <= int(high))
            row_parts.append(block["rows"][mask])
            pred_parts.append(block["candidate"][mask])
            base_parts.append(block["baseline"][mask])
        rows = np.concatenate(row_parts)
        prediction = np.concatenate(pred_parts)
        baseline = np.concatenate(base_parts)
        if len(np.unique(rows)) != len(rows):
            raise BatchIntegrityError(f"R06 subgroup QA found duplicate rows in {view}")
        y = target[rows]
        local = frame.iloc[rows]
        stream = local["signal_stream"]
        missing = stream.isna().to_numpy(dtype=bool) | stream.fillna("").astype(str).str.strip().eq("").to_numpy(dtype=bool)
        friction = pd.to_numeric(local["entry_friction"], errors="coerce").to_numpy(dtype=np.float64)
        groups = {
            "entry_friction_0": friction == 0.0,
            "entry_friction_1": friction == 1.0,
            "stream_present": ~missing,
            "stream_missing": missing,
            "target_le_5": y <= 5.0,
            "target_ge_95": y >= 95.0,
        }
        view_result: dict[str, Any] = {}
        for name, mask in groups.items():
            n = int(mask.sum())
            supported = n >= 100
            harm = (float(np.mean(np.abs(prediction[mask] - y[mask]))
                          - np.mean(np.abs(baseline[mask] - y[mask]))) if n else None)
            guard_applies = name in {"stream_missing", "target_le_5", "target_ge_95"} and supported
            record = {
                "n": n,
                "supported": supported,
                "evidence_status": "SUPPORTED" if supported else "DESCRIPTIVE_UNSUPPORTED",
                "mae_harm_candidate_minus_control": harm,
                "guard_applies": guard_applies,
                "guard_pass": None if not guard_applies else bool(harm <= 0.25),
            }
            view_result[name] = record
            if guard_applies and harm > 0.25:
                failures.append({"view": view, "group": name, **record})
        output[view] = view_result
    return {
        "minimum_support": 100,
        "supported_tail_or_missing_MAE_harm_max": 0.25,
        "views": output,
        "supported_guard_failures": failures,
    }


def _attempt_accounting() -> dict[str, Any]:
    events = [json.loads(line) for line in ATTEMPT_PATH.read_text(encoding="utf-8").splitlines()
              if line.strip()]
    own = [event for event in events if event.get("card_id") == "R06"]
    started = [event for event in own if event.get("event") == "FIT_STARTED"]
    saved = [event for event in own if event.get("event") == "FIT_SAVED"]
    failed = [event for event in own if event.get("event") == "FIT_FAILED"]
    preflight = [event for event in own if event.get("event") == "PREFLIGHT_PASS"]
    times = [float(event["fit_seconds"]) for event in saved]
    model_fit_calls = 0
    for cutoff in (55, 67):
        sidecar_path = RUN_DIR / f"R06_fit{cutoff}.json"
        if sidecar_path.is_file():
            sidecar = _json(sidecar_path)
            model_fit_calls += int(sidecar.get("metadata", {}).get("fit_metadata", {}).get("predictor_fit_calls", 0))
    return {
        "preflight_events": len(preflight),
        "fit_attempts_started": len(started),
        "fit_saved_events": len(saved),
        "fit_failed_events": len(failed),
        "successful_predictor_fits": sum(int(event.get("fit_predictor_count", 0)) for event in saved),
        "predictor_fit_calls": model_fit_calls,
        "fit_seconds_by_cutoff": {str(event["cutoff"]): float(event["fit_seconds"]) for event in saved},
        "total_fit_seconds": float(sum(times)),
        "technical_retries": len(failed),
        "attempt_accounting_complete": len(preflight) == 1 and len(started) == 2 and len(saved) == 2
        and len(failed) == 0 and model_fit_calls == 2,
    }


def _render_summary(results: Mapping[str, Any], gate: Mapping[str, Any]) -> str:
    views = results["pilot"]["views"]
    lines = [
        "# R06 — Character sparse L1 boosting",
        "",
        f"**Verdict: {gate['status']}** — fixed STANDARD temporal Train pilot; no Test/Kaggle access.",
        "",
        "Recipe: `char_wb` TF-IDF 3–5 grams, `min_df=3`, cap 30,000, sublinear TF; fit-only structured/OHE plus missing-stream indicator; LightGBM `regression_l1`.",
        "Policy: 0.75 × frozen three-component champion + 0.25 × within-cycle rank50(raw R06), clipped to [0,100].",
        "",
        "| View | n | Control MAE | R06 MAE | Gain (control − R06) |",
        "|---|---:|---:|---:|---:|",
    ]
    for name in ("W1", "W4", "primary", "H12"):
        item = views[name]
        lines.append(f"| {name} | {item['rows']:,} | {item['baseline']['mae']:.6f} | {item['candidate']['mae']:.6f} | {item['gain_positive_is_better']:+.6f} |")
    lines.extend([
        "",
        "Control and candidate use the same exact row positions per view. Vocabulary, IDF, numeric scaling, and categorical encoding are fit separately from the past-only rows at each cutoff. Anonymous character fragments are not interpreted semantically.",
        "",
        "See `results.json`, `verification.json`, `subgroups.json`, `gate.json`, and `preflight.json` for exact provenance, support, fit accounting, memory estimate, and diagnostics.",
        "",
    ])
    return "\n".join(lines)


def _append_event_once(path: Path, event: Mapping[str, Any]) -> None:
    key_fields = ("card_id", "event", "budget_key", "freeze_signature")
    if path.exists():
        matches = []
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            current = json.loads(line)
            if all(current.get(key) == event.get(key) for key in key_fields):
                matches.append(current)
        if matches:
            if len(matches) != 1 or any(matches[0].get(k) != v for k, v in event.items()):
                raise BatchIntegrityError(f"Conflicting existing R06 ledger event: {event['event']}")
            return
    append_event(path, event)


def _reconcile_budget(results: Mapping[str, Any], gate: Mapping[str, Any]) -> dict[str, Any]:
    ledger = [json.loads(line) for line in LEDGER_PATH.read_text(encoding="utf-8").splitlines()
              if line.strip()]
    reservations = [event for event in ledger if event.get("event") == "BUDGET_RESERVED"
                    and event.get("budget_key") == "R06:PILOT"]
    if len(reservations) != 1 or int(reservations[0].get("predictor_fits", 0)) != 2:
        raise BatchIntegrityError("R06 global fit reservation is missing or invalid")
    totals = [int(event["reserved_total_after"]) for event in ledger
              if event.get("event") == "BUDGET_RESERVED" and "reserved_total_after" in event]
    reconciliation = {
        "event": "FIT_BUDGET_RECONCILED",
        "card_id": "R06",
        "budget_key": "R06:PILOT",
        "reserved_predictor_fits": 2,
        "successful_predictor_fits": int(results["fit_accounting"]["successful_predictor_fits"]),
        "fit_started_events": int(results["fit_accounting"]["fit_attempts_started"]),
        "fit_saved_events": int(results["fit_accounting"]["fit_saved_events"]),
        "fit_failed_events": int(results["fit_accounting"]["fit_failed_events"]),
        "attempt_accounting_complete": bool(results["fit_accounting"]["attempt_accounting_complete"]),
        "technical_retries": int(results["fit_accounting"]["technical_retries"]),
        "decision": gate["status"],
        "freeze_signature": results["freeze_signature"],
        "reserved_global_total_after": max(totals) if totals else None,
        "results_sha256": file_sha256(RESULTS_PATH),
        "gate_sha256": file_sha256(GATE_PATH),
        "verification_sha256": file_sha256(VERIFICATION_PATH),
    }
    _append_event_once(LEDGER_PATH, reconciliation)
    return reconciliation


def _execute(context: Mapping[str, Any]) -> dict[str, Any]:
    preflight = context["preflight"]
    if preflight["status"] == "SKIPPED_SUPPORT":
        return {"card_id": "R06", "status": "SKIPPED_SUPPORT", "preflight": preflight,
                "candidate_fit_performed": False, "test_data_read": False}
    if preflight["status"] != "PREFLIGHT_PASS":
        return {"card_id": "R06", "status": "SKIPPED_RESOURCE", "preflight": preflight,
                "candidate_fit_performed": False, "test_data_read": False}
    return {"card_id": "R06", "status": "PREFLIGHT_PASS", "preflight": preflight,
            "candidate_fit_performed": False, "test_data_read": False}


def run_card(card: dict[str, Any], queue: dict[str, Any], *,
             preflight_only: bool = False, freeze_only: bool = False,
             run_frozen_pilot: bool = False) -> dict[str, Any]:
    if sum((preflight_only, freeze_only, run_frozen_pilot)) != 1:
        raise BatchIntegrityError("Choose exactly one R06 execution mode")
    context = _prepare(card, queue)
    state = _execute(context)
    if preflight_only or state["status"] in {"SKIPPED_SUPPORT", "SKIPPED_RESOURCE"}:
        return state
    if freeze_only:
        for directory in (EXPERIMENT_DIR, RUN_DIR, REPORT_DIR):
            directory.mkdir(parents=True, exist_ok=True)
        _write_or_match(RECIPE_PATH, context["recipe_record"])
        _write_or_match(PREFLIGHT_PATH, context["preflight"])
        signature = ensure_frozen(FREEZE_PATH, context["identity"])
        return {"card_id": "R06", "status": "FROZEN_NO_FIT", "freeze_signature": signature,
                "recipe_path": str(RECIPE_PATH), "freeze_path": str(FREEZE_PATH),
                "fit_performed": False, "test_data_read": False}

    if not all(path.is_file() for path in (RECIPE_PATH, PREFLIGHT_PATH, FREEZE_PATH)):
        raise BatchIntegrityError("R06 requires committed recipe, preflight, and freeze before fitting")
    if _json(RECIPE_PATH) != context["recipe_record"]:
        raise BatchIntegrityError("R06 recipe differs from frozen source identity")
    frozen_preflight = _json(PREFLIGHT_PATH)
    if frozen_preflight.get("freeze_identity_sha256") != json_sha256(context["identity"]):
        raise BatchIntegrityError("R06 saved preflight identity differs from current source/input identity")
    frozen = _json(FREEZE_PATH)
    if (frozen.get("signature") != json_sha256(context["identity"])
            or frozen.get("identity") != context["identity"]):
        raise BatchIntegrityError("R06 source/input/recipe differs from committed freeze")
    if file_sha256(context["train_path"]) != EXPECTED_TRAIN_SHA256:
        raise BatchIntegrityError("Official Regression Train changed after R06 freeze")

    preflight_event = {
        "card_id": "R06",
        "event": "PREFLIGHT_PASS",
        "freeze_signature": frozen["signature"],
        "resource": _stable_resource_identity(context["preflight"]["resource"]),
        "control_metrics": context["control_metrics"],
        "fit_predictor_count_reserved": 2,
    }
    _append_event_once(ATTEMPT_PATH, preflight_event)
    pilot = run_pilot(
        card=card,
        artifact_dir=RUN_DIR,
        budget_ledger_path=LEDGER_PATH,
        max_successful_fits=int(queue["execution_preset"]["max_successful_new_predictor_fits"]),
        frame=context["frame"],
        target=context["target"],
        feature_names=FEATURE_NAMES,
        adapter=fit_predict,
        control_loader=r02._control_loader(context["controls"]),
        freeze_identity=context["identity"],
        freeze_path=FREEZE_PATH,
        event_path=ATTEMPT_PATH,
        evaluation_views=context["views"],
        threads=THREADS,
        report_path=ENGINE_REPORT_PATH,
    )
    verification = verify_report(
        report=pilot,
        frame=context["frame"],
        target=context["target"],
        artifact_dir=RUN_DIR,
        freeze_signature=pilot["freeze_signature"],
        evaluation_views=context["views"],
        prediction_reconstructors={"R06": reconstruct_r06},
    )
    subgroup = _subgroup_diagnostics(pilot, context["frame"], context["target"], context["views"])
    accounting = _attempt_accounting()
    accounting["fit_accounting_matches_card"] = (
        accounting["preflight_events"] == 1
        and accounting["successful_predictor_fits"] == 2
        and accounting["fit_saved_events"] == 2
        and accounting["fit_failed_events"] == 0
        and accounting["predictor_fit_calls"] == 2
        and all((RUN_DIR / f"R06_fit{cutoff}.npz").is_file()
                and (RUN_DIR / f"R06_fit{cutoff}.json").is_file() for cutoff in (55, 67))
    )
    if not accounting["fit_accounting_matches_card"]:
        raise BatchIntegrityError(f"R06 fit attempt accounting differs from its 2-fit card: {accounting}")

    gate_config = queue["evaluation"]["regression_gate"]
    gains = pilot["views"]
    checks = {
        "pilot_primary_MAE_gain_at_least_0_03": gains["primary"]["gain_positive_is_better"] >= float(gate_config["pilot_primary_MAE_gain_min"]),
        "pilot_H12_MAE_harm_at_most_0_05": gains["H12"]["gain_positive_is_better"] >= -float(gate_config["pilot_H12_MAE_harm_max"]),
        "W1_MAE_harm_at_most_0_5": gains["W1"]["gain_positive_is_better"] >= -float(gate_config["individual_pilot_view_MAE_harm_max"]),
        "W4_MAE_harm_at_most_0_5": gains["W4"]["gain_positive_is_better"] >= -float(gate_config["individual_pilot_view_MAE_harm_max"]),
        "supported_tail_or_missing_group_harms": len(subgroup["supported_guard_failures"]) == 0,
        "independent_verification_pass": verification.get("status") in {"PASS", "INDEPENDENT_QA_PASS"},
        "fit_attempt_accounting_pass": accounting["fit_accounting_matches_card"],
    }
    gate = {
        "card_id": "R06",
        "profile": card["evaluation"],
        "status": "ADVANCE" if all(checks.values()) else "RETIRE",
        "checks": {key: bool(value) for key, value in checks.items()},
        "thresholds": gate_config,
        "subgroup_diagnostics": subgroup,
        "decision_basis": "Frozen W1+W4/H12 Train pilot, independent reconstruction, and declared regression gates only; no Test/Kaggle evidence.",
    }
    results = {
        "card_id": "R06",
        "status": f"PILOT_COMPLETE_{gate['status']}",
        "source_commit": context["recipe_record"]["source_commit"],
        "freeze_signature": pilot["freeze_signature"],
        "input_sha256": context["identity"]["input_sha256"],
        "source_sha256": context["identity"]["source_sha256"],
        "preflight": context["preflight"],
        "pilot": pilot,
        "fit_accounting": accounting,
        "independent_verification_status": verification.get("status"),
        "subgroups": subgroup,
        "gate": gate,
        "official_test_read": False,
        "kaggle_accessed": False,
    }
    _write_or_match(VERIFICATION_PATH, verification)
    _write_or_match(SUBGROUP_PATH, subgroup)
    _write_or_match(GATE_PATH, gate)
    _write_or_match(RESULTS_PATH, results)
    _write_text_or_match(SUMMARY_PATH, _render_summary(results, gate))
    reconciliation = _reconcile_budget(results, gate)
    return {"results": results, "verification": verification, "gate": gate,
            "budget_reconciliation": reconciliation}


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--preflight-only", action="store_true")
    modes.add_argument("--freeze-only", action="store_true")
    modes.add_argument("--run-frozen-pilot", action="store_true")
    args = parser.parse_args()
    queue = json.loads(QUEUE_PATH.read_text(encoding="utf-8"))
    card = next(item for item in queue["cards"] if item.get("id") == "R06")
    result = run_card(card, queue, preflight_only=args.preflight_only,
                      freeze_only=args.freeze_only, run_frozen_pilot=args.run_frozen_pilot)
    print(json.dumps(result, indent=2, allow_nan=False))
