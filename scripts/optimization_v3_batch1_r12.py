"""Preflight, freeze, and execute the R12 cycle-grouped LambdaRank pilot.

Run each stage explicitly: ``--preflight-only`` is read-only,
``--freeze-only`` writes the immutable recipe identity, and
``--run-frozen-pilot`` is the only stage that fits candidate predictors.
Regression Test and SampleSubmission are never opened.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
import threading
import time
from typing import Any, Mapping

import numpy as np
import pandas as pd
import psutil
from sklearn.feature_extraction.text import TfidfVectorizer
from lightgbm import LGBMRanker

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import optimization_v3_batch1_r01 as r01
from scripts.optimization_v3_batch1_verify import verify_report
from src.models.optimization_v3_batch1 import (
    BatchIntegrityError,
    evaluation_views_for_card,
    ensure_frozen,
    file_sha256,
    json_sha256,
    load_queue,
    make_freeze_identity,
    read_bundle,
    run_pilot,
    write_json_exclusive,
)
from src.models.optimization_v3_batch1_r12 import (
    CARD_ID,
    HYBRID_FEATURES,
    RANKER_BASE_PARAMETERS,
    blend_rank_percentile,
    cycle_average_rank_percentile,
    fit_predict,
    reconstruct_r12,
)
from src.models.regression_research_text import RegressionTextModel, _TFIDF_PARAMS, _text_documents


QUEUE_PATH = ROOT / "experiments/optimization_v3/research_backlog/queue.json"
CARD_PATH = ROOT / "reports/optimization_v3/research_backlog/cards/R12.md"
EXPERIMENT_DIR = ROOT / "experiments/optimization_v3/batch1/R12"
FREEZE_PATH = EXPERIMENT_DIR / "freeze.json"
ARTIFACT_DIR = ROOT / "experiments/runs/optimization_v3_batch1/R12"
REPORT_DIR = ROOT / "reports/optimization_v3/batch1/R12"
TRAIN_SHA256 = r01.EXPECTED_TRAIN_SHA256
QUEUE_SHA256 = r01.EXPECTED_QUEUE_SHA256
CARD_SHA256 = "e858f45c14a50ae0ceee186766fadce2037fba39c44a61a9ff84b50b9e704557"
EXPECTED_CARD: dict[str, Any] = {
    "id": "R12",
    "task": "regression",
    "name": "Cycle-grouped LambdaRank",
    "family": "ranking",
    "priority": "P1",
    "change": "Original sparse word+structured design, one contiguous query per fit cycle; integer grades clip(floor(y/10),0,9), label_gain0..9. LGBMRanker lambdarank/200trees/15leaves/min_child40/lambda3/lr0.05/seed42/2threads, truncation=max fit-query size. Convert scores only to100*average_rank/n over entire future cycle. Do not rank50-mix unbounded ranker score units.",
    "policy": "R_RANK25",
    "pilot_predictor_fits": 2,
    "evaluation": "STANDARD",
    "condition": "No additional dependency",
    "status": "PLANNED_NOT_RUN",
    "card_path": "reports/optimization_v3/research_backlog/cards/R12.md",
}
PACKAGES = ["numpy", "pandas", "scipy", "scikit-learn", "threadpoolctl", "psutil", "lightgbm"]
EXPECTED_VIEWS = {
    "W1": [(55, 56, 59)],
    "W4": [(67, 68, 71)],
    "primary": [(55, 56, 59), (67, 68, 71)],
    "H12": [(55, 56, 67), (67, 68, 79)],
}


def _write_or_match(path: Path, payload: dict[str, Any]) -> None:
    if path.is_file():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing != payload:
            raise BatchIntegrityError(f"Existing R12 artifact differs; refusing overwrite: {path}")
    else:
        write_json_exclusive(path, payload)


def _read_controls() -> tuple[dict[str, Any], Path, pd.DataFrame, np.ndarray, np.ndarray,
                              dict[str, Any], dict[int, dict[str, np.ndarray]], Any]:
    if file_sha256(QUEUE_PATH) != QUEUE_SHA256:
        raise BatchIntegrityError("Canonical V3 queue hash differs from the frozen identity")
    queue = load_queue(QUEUE_PATH)
    cards = {str(card["id"]): card for card in queue["cards"]}
    queue_card = cards.get(CARD_ID)
    if queue_card is None or {key: queue_card.get(key) for key in EXPECTED_CARD} != EXPECTED_CARD:
        raise BatchIntegrityError("Canonical R12 card differs from its frozen definition")
    if file_sha256(CARD_PATH) != CARD_SHA256:
        raise BatchIntegrityError("R12 card document hash differs from its frozen identity")
    train_path, frame, target, cycles = r01._check_official_train()
    if tuple(r01.FEATURE_NAMES) != tuple(HYBRID_FEATURES):
        raise BatchIntegrityError("R12 sparse feature order differs from the verified hybrid allowlist")
    manifest = r01._validate_manifest(queue)
    views = evaluation_views_for_card(queue, CARD_ID)
    if views != EXPECTED_VIEWS:
        raise BatchIntegrityError(f"R12 STANDARD views differ from the frozen protocol: {views}")

    base = r01._read_npz(r01.BASE_CACHE)
    sparse_cache = r01._read_npz(r01.SPARSE_CACHE)
    diagnostic = r01._read_npz(r01.DIAGNOSTIC_CACHE)
    control_loader_base = r01._load_controls(frame, base, sparse_cache, diagnostic)
    control_check = r01._validate_views_and_controls(
        frame, target, cycles, views, control_loader_base, queue, diagnostic
    )
    cycle_counts = {int(cycle): int(np.sum(cycles == cycle)) for cycle in np.unique(cycles)}

    def control_loader(cutoff: int, score_rows: np.ndarray) -> dict[str, np.ndarray]:
        controls = control_loader_base(cutoff, score_rows)
        score_cycles = cycles[np.asarray(score_rows, dtype=np.int64)]
        expected_counts = np.asarray([cycle_counts[int(cycle)] for cycle in score_cycles], dtype=np.int64)
        for cycle in np.unique(score_cycles):
            observed_counts = np.unique(expected_counts[score_cycles == cycle])
            if not np.array_equal(observed_counts, np.asarray([cycle_counts[int(cycle)]])):
                raise BatchIntegrityError("R12 full-cycle count bridge is not aligned")
        return {**controls, "expected_score_cycle_count": expected_counts}

    return queue, train_path, frame, target, cycles, control_check, {
        55: {"rows": np.flatnonzero(cycles <= 55), "counts": {c: cycle_counts[c] for c in range(56, 68)}},
        67: {"rows": np.flatnonzero(cycles <= 67), "counts": {c: cycle_counts[c] for c in range(68, 80)}},
    }, control_loader


def _ledger_status(queue: Mapping[str, Any]) -> dict[str, int]:
    ledger = REPORT_DIR.parent / "batch_ledger.jsonl"
    reservations: dict[str, int] = {}
    if ledger.is_file():
        for line_number, line in enumerate(ledger.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError as exc:
                raise BatchIntegrityError(f"Invalid batch fit ledger line {line_number}") from exc
            if event.get("event") == "BUDGET_RESERVED":
                key, count = str(event["budget_key"]), int(event["predictor_fits"])
                if key in reservations and reservations[key] != count:
                    raise BatchIntegrityError(f"Conflicting fit reservation for {key}")
                reservations[key] = count
    key = f"{CARD_ID}:PILOT"
    expected = int(EXPECTED_CARD["pilot_predictor_fits"])
    if key in reservations and reservations[key] != expected:
        raise BatchIntegrityError("Existing R12 fit reservation differs from the canonical card")
    total_before = int(sum(reservations.values()))
    total_after = total_before + (0 if key in reservations else expected)
    cap = int(queue["execution_preset"]["max_successful_new_predictor_fits"])
    if total_after > cap:
        raise BatchIntegrityError(f"R12 fit budget blocked: reserved_after={total_after}, cap={cap}")
    return {"reserved_fit_total_before_R12": total_before, "reserved_fit_total_after_R12": total_after,
            "card_predictor_fits": expected, "fit_budget_cap": cap}


def _resource_preflight(frame: pd.DataFrame, cycles: np.ndarray, queue: Mapping[str, Any]) -> dict[str, Any]:
    preset = queue["execution_preset"]
    largest_rows = np.flatnonzero(cycles <= 67)
    documents, _ = _text_documents(frame.iloc[largest_rows]["signal_stream"])
    analyzer = TfidfVectorizer(**_TFIDF_PARAMS).build_analyzer()
    # Fit-free upper bound: all analyzed word unigrams and bigrams, before pruning.
    nnz_upper = int(sum(len(analyzer(document)) for document in documents))
    sparse_upper_bytes = int(nnz_upper * 16 + (len(largest_rows) + 1) * 8)
    sparse_upper_gib = float(sparse_upper_bytes / (1024 ** 3))
    # Includes a fixed process/LightGBM histogram allowance plus three sparse
    # working copies; this is a conservative screen, not a measured peak.
    estimated_peak_gib = float(2.25 + 3.0 * sparse_upper_gib)
    available_gib = float(psutil.virtual_memory().available / (1024 ** 3))
    cap_gib = float(preset["max_job_RSS_GiB"])
    reserve_gib = 2.0
    allowed_gib = min(cap_gib, available_gib - reserve_gib)
    minimum_free_gib = float(preset["minimum_free_RAM_GiB_before_start"])
    if int(preset["concurrent_fit_jobs"]) != 1 or int(preset["threads_per_job"]) != 2:
        raise BatchIntegrityError("R12 requires one serialized fit job with two threads")
    if available_gib < minimum_free_gib or estimated_peak_gib > allowed_gib:
        raise BatchIntegrityError(
            f"R12 resource gate blocked: available={available_gib:.2f}GiB, "
            f"estimated_peak={estimated_peak_gib:.2f}GiB, allowed={allowed_gib:.2f}GiB"
        )
    result = {
        "fit_rows_at_largest_cutoff": int(len(largest_rows)),
        "word_ngram_nnz_upper_bound": nnz_upper,
        "sparse_design_upper_gib": sparse_upper_gib,
        "estimated_peak_job_gib": estimated_peak_gib,
        "estimate_method": "2.25GiB fixed allowance + 3 x fit-free sparse word ngram upper bound; ranker pair-buffer RSS not directly measured",
        "available_ram_gib_observed": available_gib,
        "allowed_job_ram_gib": allowed_gib,
        "max_job_ram_gib": cap_gib,
        "minimum_free_ram_gib": minimum_free_gib,
        "threads": 2,
        "concurrent_fit_jobs": 1,
        "no_dense_full_word_matrix": True,
        "estimator_fits_performed": 0,
        "peak_RSS_measured": False,
    }
    return result


def _input_files(train_path: Path) -> dict[str, Path]:
    files = dict(r01._input_files(train_path))
    files["r12_card"] = CARD_PATH
    return files


def _source_files() -> list[Path]:
    sources = list(r01._source_files())
    sources.extend([
        Path(__file__),
        ROOT / "src/models/optimization_v3_batch1_r12.py",
    ])
    return list(dict.fromkeys(sources))


def _recipe(queue: Mapping[str, Any], control_check: Mapping[str, Any], views: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "card_id": CARD_ID,
        "status": "FROZEN_RECIPE_BEFORE_PILOT",
        "card_sha256": CARD_SHA256,
        "canonical_queue_sha256": QUEUE_SHA256,
        "accepted_control": {
            "manifest": str(r01.ACCEPTED_MANIFEST.relative_to(ROOT)),
            "manifest_sha256": r01.EXPECTED_MANIFEST_SHA256,
            "route": ".5625 rank50 LightGBM SVD32 + .1875 rank75 Hybrid Ridge + .25 rank50 sparse LGB",
            "primary_mae": 11.559390078665364,
            "H12_mae": 14.04753258299653,
            "control_reconstruction": dict(control_check),
        },
        "features": list(HYBRID_FEATURES),
        "preprocessing": {
            "structured": "RegressionTextModel._structured_fit/_structured_transform; fit-prefix numeric imputation/scaling and rare-category OHE",
            "categorical_min_frequency": 30,
            "categorical_max_categories": 254,
            "text": RegressionTextModel._text_state(None, None)["parameters"],
            "text_vectorizer_fit_scope": "fit rows at each cutoff only",
            "missing_stream_indicator": True,
            "sparse_design": True,
            "svd": False,
            "full_word_design_densified": False,
        },
        "estimator": {
            "class": "lightgbm.LGBMRanker",
            "parameters_base": dict(RANKER_BASE_PARAMETERS),
            "parameters_per_fit": {
                "label_gain": list(range(10)),
                "lambdarank_truncation_level": "maximum group size among fit cycles",
            },
            "target_grade": "clip(floor(threat_rank / 10), 0, 9)",
            "query_definition": "one stable contiguous query per exact fit cycle",
            "group_order": "stable cycle sort; labels and CSR rows sorted identically",
            "score_inverse_mapping": "stable-sort score rows by cycle then inverse-map to original row positions",
        },
        "policy": "clip(0.75 * exact three-component champion + 0.25 * (100 * average rank / complete future-cycle n), 0, 100)",
        "evaluation_views": {key: [list(segment) for segment in value] for key, value in views.items()},
        "support": {
            "pilot_primary_rows": int(queue["evaluation"]["STANDARD"]["pilot_primary_rows"]),
            "H12_rows": int(queue["evaluation"]["STANDARD"]["H12_rows"]),
            "supported_group_n_min": int(queue["evaluation"]["regression_gate"]["supported_group_n_min"]),
        },
        "pilot_predictor_fit_count": int(EXPECTED_CARD["pilot_predictor_fits"]),
        "metric": "MAE; positive gain = exact champion MAE - candidate MAE",
        "gates": {
            key: queue["evaluation"]["regression_gate"][key]
            for key in ("pilot_primary_MAE_gain_min", "pilot_H12_MAE_harm_max",
                        "individual_pilot_view_MAE_harm_max", "supported_group_n_min",
                        "supported_tail_or_missing_MAE_harm_max")
        },
        "test_data_used": False,
    }


def _freeze_identity(queue: Mapping[str, Any], train_path: Path, control_check: Mapping[str, Any],
                     views: Mapping[str, Any], fit_support: Mapping[int, Any]) -> dict[str, Any]:
    identity = make_freeze_identity(
        card=EXPECTED_CARD,
        feature_names=HYBRID_FEATURES,
        input_files=_input_files(train_path),
        source_files=_source_files(),
        evaluation_views=views,
        packages=PACKAGES,
    )
    identity.update({
        "accepted_manifest_sha256": r01.EXPECTED_MANIFEST_SHA256,
        "exact_control_cache_sha256": {
            "paired_blends": r01.EXPECTED_BASE_CACHE_SHA256,
            "sparse_pilot": r01.EXPECTED_SPARSE_CACHE_SHA256,
            "independent_champion": r01.EXPECTED_DIAGNOSTIC_CACHE_SHA256,
        },
        "control_reconstruction": dict(control_check),
        "recipe_sha256": json_sha256(_recipe(queue, control_check, views)),
        "fit_support": {
            str(cutoff): {
                "fit_rows": int(len(values["rows"])),
                "fit_cycles": list(range(47, cutoff + 1)),
                "score_cycle_counts": {str(k): int(v) for k, v in values["counts"].items()},
            }
            for cutoff, values in fit_support.items()
        },
        "resource_preflight_recipe": {
            "max_job_RSS_GiB": int(queue["execution_preset"]["max_job_RSS_GiB"]),
            "minimum_free_RAM_GiB_before_start": int(queue["execution_preset"]["minimum_free_RAM_GiB_before_start"]),
            "reserve_GiB": 2.0,
            "threads": 2,
            "concurrent_fit_jobs": 1,
            "no_dense_full_word_matrix": True,
        },
        "python_version": platform.python_version(),
        "validation_labels_passed_to_adapter": False,
        "test_data_used": False,
    })
    return identity


def preflight() -> dict[str, Any]:
    queue, train_path, frame, target, cycles, control_check, support, control_loader = _read_controls()
    del control_loader
    views = evaluation_views_for_card(queue, CARD_ID)
    resources = _resource_preflight(frame, cycles, queue)
    ledger = _ledger_status(queue)
    fit_support = {
        str(cutoff): {
            "fit_rows": int(len(values["rows"])),
            "fit_cycles": list(range(47, cutoff + 1)),
            "max_query_rows": int(max(np.sum(cycles[values["rows"]] == c) for c in range(47, cutoff + 1))),
            "grade_counts": np.bincount(np.clip(np.floor(target[values["rows"]] / 10), 0, 9).astype(int), minlength=10).tolist(),
            "score_cycle_counts": {str(k): int(v) for k, v in values["counts"].items()},
        }
        for cutoff, values in support.items()
    }
    if any(len(item["grade_counts"]) != 10 or any(value == 0 for value in item["grade_counts"])
           for item in fit_support.values()):
        raise BatchIntegrityError("R12 fit prefix lacks one or more relevance grades")
    identity = _freeze_identity(queue, train_path, control_check, views, support)
    constructor_checks: dict[str, Any] = {}
    for cutoff, item in fit_support.items():
        max_query = int(item["max_query_rows"])
        requested = {
            **RANKER_BASE_PARAMETERS,
            "label_gain": list(range(10)),
            "lambdarank_truncation_level": max_query,
        }
        estimator = LGBMRanker(**requested)
        effective = estimator.get_params(deep=False)
        if any(effective.get(key) != value for key, value in requested.items()):
            raise BatchIntegrityError(f"LightGBM Ranker constructor did not retain card parameters at cutoff {cutoff}")
        constructor_checks[str(cutoff)] = {
            "status": "PASS_CONSTRUCTOR_ONLY_NO_FIT",
            "fit_cycle_count": int(len(item["fit_cycles"])),
            "max_query_size": max_query,
            "label_gain": list(range(10)),
            "lambdarank_truncation_level": max_query,
        }
    return {
        "status": "PREFLIGHT_PASS_NO_FREEZE_NO_FIT",
        "card_id": CARD_ID,
        "train_sha256": TRAIN_SHA256,
        "train_rows": int(len(frame)),
        "target_min": float(np.min(target)),
        "target_max": float(np.max(target)),
        "control_reconstruction": control_check,
        "fit_support": fit_support,
        "evaluation_views": {key: [list(segment) for segment in value] for key, value in views.items()},
        "resource_preflight": resources,
        "ranker_constructor_checks": constructor_checks,
        "fit_budget": ledger,
        "feature_count": len(HYBRID_FEATURES),
        "freeze_identity_sha256": json_sha256(identity),
        "official_test_read": False,
        "model_fits_performed": 0,
        "freeze_or_output_written": False,
    }


class _PeakRssSampler:
    def __init__(self, interval_seconds: float = 0.25):
        self.interval_seconds = interval_seconds
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.peak_rss_bytes = 0
        self.sample_count = 0
        self.process = psutil.Process(os.getpid())

    def _sample(self) -> None:
        while not self._stop.wait(self.interval_seconds):
            try:
                rss = int(self.process.memory_info().rss)
            except psutil.Error:
                return
            self.peak_rss_bytes = max(self.peak_rss_bytes, rss)
            self.sample_count += 1

    def __enter__(self) -> "_PeakRssSampler":
        self.peak_rss_bytes = int(self.process.memory_info().rss)
        self._thread = threading.Thread(target=self._sample, name="r12-rss-sampler", daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *_: Any) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        try:
            self.peak_rss_bytes = max(self.peak_rss_bytes, int(self.process.memory_info().rss))
        except psutil.Error:
            pass


def _view_data(frame: pd.DataFrame, target: np.ndarray, views: Mapping[str, Any],
               freeze_signature: str, name: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    cycles = pd.to_numeric(frame["cycle_code"], errors="raise").to_numpy(dtype=np.int64)
    rows_out: list[np.ndarray] = []
    y_out: list[np.ndarray] = []
    pred_out: list[np.ndarray] = []
    baseline_out: list[np.ndarray] = []
    for cutoff, low, high in views[name]:
        path = ARTIFACT_DIR / f"{CARD_ID}_fit{cutoff}.npz"
        sidecar = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
        identity = sidecar.get("identity", {})
        if identity.get("freeze_signature") != freeze_signature:
            raise BatchIntegrityError(f"R12 bundle has the wrong freeze at cutoff {cutoff}")
        arrays, _ = read_bundle(path, identity)
        rows = np.asarray(arrays["score_rows"], dtype=np.int64)
        mask = (cycles[rows] >= int(low)) & (cycles[rows] <= int(high))
        selected = rows[mask]
        rows_out.append(selected)
        y_out.append(target[selected])
        pred_out.append(np.asarray(arrays["prediction"], dtype=np.float64)[mask])
        baseline_out.append(np.asarray(arrays["baseline_prediction"], dtype=np.float64)[mask])
    rows = np.concatenate(rows_out)
    if len(np.unique(rows)) != len(rows):
        raise BatchIntegrityError(f"R12 persisted view {name} contains duplicate Train rows")
    return rows, np.concatenate(y_out), np.concatenate(pred_out), np.concatenate(baseline_out)


def run_frozen_pilot() -> dict[str, Any]:
    queue, train_path, frame, target, cycles, control_check, fit_support, control_loader = _read_controls()
    views = evaluation_views_for_card(queue, CARD_ID)
    resources = _resource_preflight(frame, cycles, queue)
    identity = _freeze_identity(queue, train_path, control_check, views, fit_support)
    if not FREEZE_PATH.is_file():
        raise BatchIntegrityError("R12 freeze is absent; run --freeze-only before fitting")
    frozen = json.loads(FREEZE_PATH.read_text(encoding="utf-8"))
    if frozen.get("identity") != identity or frozen.get("signature") != json_sha256(identity):
        raise BatchIntegrityError("R12 source/input/fold identity changed after freeze")
    recipe_path = REPORT_DIR / "recipe.json"
    if not recipe_path.is_file():
        raise BatchIntegrityError("R12 frozen recipe report is missing")
    if not ARTIFACT_DIR.exists():
        ARTIFACT_DIR.mkdir(parents=True)
    event_path = REPORT_DIR / "attempts.jsonl"
    failure_events = []
    if event_path.is_file():
        failure_events = [
            json.loads(line) for line in event_path.read_text(encoding="utf-8").splitlines()
            if line.strip() and json.loads(line).get("card_id") == CARD_ID
            and json.loads(line).get("event") == "FIT_FAILED"
        ]
    if failure_events:
        raise BatchIntegrityError("R12 has an earlier failed fit attempt; refusing an automatic retry")

    with _PeakRssSampler() as sampler:
        report = run_pilot(
            frame=frame,
            target=target,
            card=EXPECTED_CARD,
            feature_names=HYBRID_FEATURES,
            adapter=fit_predict,
            control_loader=control_loader,
            freeze_identity=identity,
            freeze_path=FREEZE_PATH,
            artifact_dir=ARTIFACT_DIR,
            event_path=event_path,
            evaluation_views=views,
            threads=2,
            report_path=REPORT_DIR / "pilot.json",
            budget_ledger_path=REPORT_DIR.parent / "batch_ledger.jsonl",
            max_successful_fits=int(queue["execution_preset"]["max_successful_new_predictor_fits"]),
        )

    verification = verify_report(
        report=report,
        frame=frame,
        target=target,
        artifact_dir=ARTIFACT_DIR,
        freeze_signature=report["freeze_signature"],
        evaluation_views=views,
        prediction_reconstructors={CARD_ID: reconstruct_r12},
    )
    view_details: dict[str, Any] = {}
    subgroup_failures: list[dict[str, Any]] = []
    for name in views:
        rows, y, prediction, baseline = _view_data(frame, target, views, report["freeze_signature"], name)
        subgroups = r01._subgroup_diagnostics(frame, target, rows, y, prediction, baseline)
        view_details[name] = {"rows": int(len(rows)), "subgroups": subgroups}
        for field in ("stream_missing", "target_tail"):
            for group_name, record in subgroups["groups"][field].items():
                if (record["supported"] and record["candidate_minus_baseline_mae"] is not None
                        and float(record["candidate_minus_baseline_mae"]) > 0.25):
                    subgroup_failures.append({"view": name, "field": field, "group": group_name, **record})

    gates = queue["evaluation"]["regression_gate"]
    checks = {
        "pilot_primary_MAE_gain_min": report["views"]["primary"]["gain_positive_is_better"] >= float(gates["pilot_primary_MAE_gain_min"]),
        "pilot_H12_MAE_harm_max": report["views"]["H12"]["gain_positive_is_better"] >= -float(gates["pilot_H12_MAE_harm_max"]),
        "W1_MAE_harm_max": report["views"]["W1"]["gain_positive_is_better"] >= -float(gates["individual_pilot_view_MAE_harm_max"]),
        "W4_MAE_harm_max": report["views"]["W4"]["gain_positive_is_better"] >= -float(gates["individual_pilot_view_MAE_harm_max"]),
        "supported_tail_or_missing_harm_max": len(subgroup_failures) == 0,
        "independent_metric_and_policy_reconstruction": verification.get("status") == "PASS",
    }
    gate = {
        "card_id": CARD_ID,
        "profile": "STANDARD",
        "status": "ADVANCE" if all(checks.values()) else "RETIRE",
        "checks": {key: bool(value) for key, value in checks.items()},
        "thresholds": {key: gates[key] for key in (
            "pilot_primary_MAE_gain_min", "pilot_H12_MAE_harm_max",
            "individual_pilot_view_MAE_harm_max", "supported_group_n_min",
            "supported_tail_or_missing_MAE_harm_max")},
        "supported_tail_or_missing_failures": subgroup_failures,
        "view_subgroups": view_details,
        "decision_basis": "Predeclared temporal pilot metrics and supported subgroup guards; no Test or Kaggle result used.",
    }
    fit_events = [
        json.loads(line) for line in event_path.read_text(encoding="utf-8").splitlines()
        if line.strip() and json.loads(line).get("card_id") == CARD_ID
    ] if event_path.is_file() else []
    verification.update({
        "fit_accounting": {
            "attempted_predictor_fits": int(sum(e.get("event") == "FIT_STARTED" for e in fit_events)),
            "successful_predictor_fits": int(sum(e.get("event") == "FIT_SAVED" for e in fit_events)),
            "fit_failed_events": int(sum(e.get("event") == "FIT_FAILED" for e in fit_events)),
            "expected_card_predictor_fits": 2,
            "fit_seconds_by_cutoff": {
                str(e["cutoff"]): float(e["fit_seconds"])
                for e in fit_events if e.get("event") == "FIT_SAVED"
            },
        },
        "resource_observation": {
            "sampled_peak_process_RSS_GiB": float(sampler.peak_rss_bytes / (1024 ** 3)),
            "RSS_sample_count": int(sampler.sample_count),
            "preflight_estimated_peak_job_GiB": float(resources["estimated_peak_job_gib"]),
            "queue_job_cap_GiB": float(resources["max_job_ram_gib"]),
            "resource_estimate_limitation": resources["estimate_method"],
        },
        "official_test_read": False,
        "test_labels_used": False,
        "control_reconstruction": control_check,
    })
    _write_or_match(REPORT_DIR / "verification.json", verification)
    _write_or_match(REPORT_DIR / "gate.json", gate)
    primary = report["views"]["primary"]
    h12 = report["views"]["H12"]
    summary = (
        "# R12 Pilot Summary\n\n"
        f"- Decision: **{gate['status']}**.\n"
        f"- Primary MAE: {primary['baseline']['mae']:.6f} champion → {primary['candidate']['mae']:.6f} candidate; "
        f"gain {primary['gain_positive_is_better']:+.6f} on {primary['rows']:,} rows.\n"
        f"- H12 MAE: {h12['baseline']['mae']:.6f} champion → {h12['candidate']['mae']:.6f} candidate; "
        f"gain {h12['gain_positive_is_better']:+.6f} on {h12['rows']:,} rows.\n"
        f"- W1/W4 gains: {report['views']['W1']['gain_positive_is_better']:+.6f} / "
        f"{report['views']['W4']['gain_positive_is_better']:+.6f}.\n"
        f"- Sampled peak process RSS: {verification['resource_observation']['sampled_peak_process_RSS_GiB']:.3f} GiB "
        f"({verification['resource_observation']['RSS_sample_count']} monitor samples).\n"
        "- Candidate scores were mapped by average rank within each complete future cycle, then blended 25% with the exact champion.\n"
        "- Temporal development outcomes are reused; this is not untouched holdout evidence.\n"
        "- Regression Test, SampleSubmission, and Kaggle were not accessed.\n"
    )
    summary_path = REPORT_DIR / "summary.md"
    if summary_path.exists():
        if summary_path.read_text(encoding="utf-8") != summary:
            raise BatchIntegrityError("Existing R12 summary differs; refusing overwrite")
    else:
        with summary_path.open("x", encoding="utf-8", newline="\n") as stream:
            stream.write(summary)
    return {"pilot": report, "verification": verification, "gate": gate, "resources": resources}


def freeze_only() -> dict[str, Any]:
    result = preflight()
    queue, train_path, frame, target, cycles, control_check, support, _ = _read_controls()
    del frame, target, cycles
    views = evaluation_views_for_card(queue, CARD_ID)
    identity = _freeze_identity(queue, train_path, control_check, views, support)
    recipe = _recipe(queue, control_check, views)
    EXPERIMENT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    ensure_frozen(FREEZE_PATH, identity)
    _write_or_match(REPORT_DIR / "recipe.json", recipe)
    result["status"] = "FROZEN_BEFORE_FIT"
    result["freeze_signature"] = json_sha256(identity)
    result["freeze_path"] = str(FREEZE_PATH.relative_to(ROOT))
    result["recipe_path"] = str((REPORT_DIR / "recipe.json").relative_to(ROOT))
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--preflight-only", action="store_true")
    modes.add_argument("--freeze-only", action="store_true")
    modes.add_argument("--run-frozen-pilot", action="store_true")
    args = parser.parse_args(argv)
    if args.preflight_only:
        result = preflight()
    elif args.freeze_only:
        result = freeze_only()
    else:
        result = run_frozen_pilot()
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
