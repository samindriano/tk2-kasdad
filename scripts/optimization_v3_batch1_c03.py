"""Frozen C03 execution bridge: official Train, verified controls, and outputs."""
from __future__ import annotations

import json
from pathlib import Path
import platform
import re
from typing import Any

import numpy as np
import pandas as pd
import psutil

from scripts.optimization_v3_batch1_verify import reconstruct_c03, verify_report
from scripts.optimization_v3_sparse_classification import CACHE as SPARSE_PILOT_CACHE
from scripts.optimization_v3_sparse_classification import CONFIG as SPARSE_PILOT_CONFIG
from scripts.optimization_v3_sparse_classification import check as check_sparse_pilot
from scripts.optimization_v3_sparse_classification_fullsix import CB_CACHE, control_record, pilot_record
from src.models.optimization_v3_batch1 import (
    BatchIntegrityError, append_event, evaluation_views_for_card, file_sha256, json_sha256,
    load_queue, make_freeze_identity, run_pilot, write_json_exclusive,
)
from src.models.optimization_v3_batch1_c03 import FEATURE_NAMES, fit_predict
from src.utils.paths import task_files


ROOT = Path(__file__).resolve().parents[1]
QUEUE_PATH = ROOT / "experiments/optimization_v3/research_backlog/queue.json"
FREEZE_PATH = ROOT / "experiments/optimization_v3/batch1/C03/freeze.json"
ARTIFACT_DIR = ROOT / "experiments/runs/optimization_v3_batch1/C03"
REPORT_DIR = ROOT / "reports/optimization_v3/batch1/C03"
PACKAGES = ["numpy", "pandas", "scipy", "scikit-learn", "lightgbm", "threadpoolctl", "psutil"]


def _preflight(frame: pd.DataFrame, target: np.ndarray, train_path: Path, expected_sha: str) -> dict[str, Any]:
    digest = file_sha256(train_path)
    if digest != expected_sha:
        raise BatchIntegrityError("Classification Train hash differs from frozen V3 baseline")
    cycles = pd.to_numeric(frame.cycle_code, errors="raise").to_numpy(dtype=int)
    if len(frame) != 50608 or not np.array_equal(np.unique(cycles), np.arange(47, 80)):
        raise BatchIntegrityError("Expected official Classification Train cycles 47-79 and 50,608 rows")
    if frame.record_id.isna().any() or not frame.record_id.astype(str).is_unique:
        raise BatchIntegrityError("Train record_id is missing or duplicated")
    if target.shape != (len(frame),) or not np.isin(target, [0, 1]).all():
        raise BatchIntegrityError("Train artifact_signal is invalid")
    fit_rows = int(np.sum(cycles <= 67))
    texts = frame.loc[cycles <= 67, "signal_stream"].fillna("").astype(str)
    token_counts = np.fromiter(
        (len(re.findall(r"(?u)\b\w\w+\b", value.lower())) for value in texts),
        dtype=np.int64, count=len(texts))
    nnz_upper = int(np.sum(token_counts + np.maximum(token_counts - 1, 0), dtype=np.int64))
    sparse_design_upper_bytes = int(nnz_upper * 16 + (fit_rows + 1) * 8)
    estimated_peak_gib = float(1.5 + 3.0 * sparse_design_upper_bytes / (1024 ** 3))
    available_gib = float(psutil.virtual_memory().available / (1024 ** 3))
    allowed_gib = min(6.0, available_gib - 2.0)
    if available_gib < 4.0 or estimated_peak_gib > allowed_gib:
        raise BatchIntegrityError(
            f"Resource preflight blocked: free={available_gib:.2f}GiB, "
            f"estimate={estimated_peak_gib:.2f}GiB, allowed={allowed_gib:.2f}GiB")
    return {
        "train_sha256": digest, "train_rows": int(len(frame)), "fit_rows_at_cutoff67": fit_rows,
        "upper_estimated_nnz": nnz_upper, "upper_sparse_design_gib": sparse_design_upper_bytes / (1024 ** 3),
        "estimated_peak_job_gib": estimated_peak_gib, "available_ram_gib_observed": available_gib,
        "allowed_job_ram_gib": allowed_gib, "single_training_job": True, "threads": 2,
    }


def _control_input_files(train_path: Path, card_path: Path, views: dict[str, list[tuple[int, int, int]]]) -> dict[str, Path]:
    ends = sorted({int(cutoff) for segments in views.values() for cutoff, _, _ in segments})
    inputs: dict[str, Path] = {
        "official_classification_train": train_path,
        "queue": QUEUE_PATH,
        "card": card_path,
        "accepted_classification_manifest": ROOT / "reports/optimization_v3/kaggle/v3_cls_sparse25.json",
        "frozen_sparse_pilot_protocol": SPARSE_PILOT_CONFIG / "protocol.json",
        "frozen_sparse_pilot_freeze": SPARSE_PILOT_CONFIG / "freeze.json",
    }
    for end in ends:
        inputs[f"sparse_pilot_{end}_npz"] = SPARSE_PILOT_CACHE / f"fit{end}.npz"
        inputs[f"sparse_pilot_{end}_sidecar"] = SPARSE_PILOT_CACHE / f"fit{end}.json"
        inputs[f"catboost_{end}_npz"] = CB_CACHE / f"fit{end}.npz"
        inputs[f"catboost_{end}_sidecar"] = CB_CACHE / f"fit{end}.json"
        inputs[f"round1_c2b_{end}"] = ROOT / f"experiments/runs/classification_round1_v1/fit{end}__C2_B.npz"
        inputs[f"round1_audit_{end}"] = ROOT / f"reports/classification_round1/fit_audits/fit{end}.json"
    return inputs


def _source_files() -> list[Path]:
    return [
        Path(__file__), ROOT / "scripts/optimization_v3_batch1_verify.py",
        ROOT / "scripts/optimization_v3_batch1.py",
        ROOT / "src/models/optimization_v3_batch1.py", ROOT / "src/models/optimization_v3_batch1_c03.py",
        ROOT / "scripts/optimization_v3_sparse_classification.py",
        ROOT / "scripts/optimization_v3_sparse_classification_fullsix.py",
        ROOT / "scripts/optimization_v3_verify_sparse_classification.py",
        ROOT / "src/models/optimization_v3_sparse_classification.py",
        ROOT / "scripts/research_campaign_classification_stage2.py",
        ROOT / "scripts/research_campaign_classification.py",
        ROOT / "scripts/optimization_v2_classification.py",
        ROOT / "src/models/round1_evaluation.py",
        ROOT / "src/data/audit.py",
        ROOT / "src/models/classification_metrics.py",
        ROOT / "src/models/classification_round2_models.py",
        ROOT / "src/models/classification_models.py",
        ROOT / "src/models/round1_models.py",
        ROOT / "src/models/regression_research_text.py",
        ROOT / "src/models/regression_round2_models.py",
        ROOT / "src/utils/paths.py",
    ]


def _load_controls(frame: pd.DataFrame, y: np.ndarray):
    check_sparse_pilot()

    def load(cutoff: int, score_rows: np.ndarray) -> dict[str, np.ndarray]:
        control = control_record(frame, y, int(cutoff))
        champion = pilot_record(frame, control)
        np.testing.assert_array_equal(control["rows"], score_rows)
        np.testing.assert_array_equal(champion["rows"], score_rows)
        return {
            "baseline_prediction": champion["blend25"],
            "baseline_sparse_raw": champion["raw_candidate"],
            "c2b_probability": control["c2b"],
            "cb_probability": control["cb_missing"],
            "stream_missing": control["missing"],
        }
    return load


def run_card(card: dict[str, Any], queue: dict[str, Any], *, preflight_only: bool = False) -> dict[str, Any]:
    if card.get("id") != "C03" or card.get("task") != "classification":
        raise BatchIntegrityError("C03 runner received a different queue card")
    train_path = task_files("classification").get("train")
    if train_path is None or not train_path.is_file():
        raise FileNotFoundError("Official Classification Train is missing")
    frame = pd.read_csv(train_path)
    target = pd.to_numeric(frame["artifact_signal"], errors="raise").to_numpy(dtype=np.int8)
    views = evaluation_views_for_card(queue, "C03")
    expected_sha = queue["baseline"]["classification"]["Train_sha256"]
    manifest_path = ROOT / queue["baseline"]["classification"]["manifest"]
    if file_sha256(manifest_path) != queue["baseline"]["classification"]["manifest_sha256"]:
        raise BatchIntegrityError("Accepted V3 classification manifest hash differs from the queue")
    preflight = _preflight(frame, target, train_path, expected_sha)
    controls = _load_controls(frame, target)
    control_files = _control_input_files(train_path, ROOT / card["card_path"], views)
    identity = make_freeze_identity(card=card, feature_names=FEATURE_NAMES,
        input_files=control_files, source_files=_source_files(),
        evaluation_views=views, packages=PACKAGES)
    identity["accepted_baseline_manifest_sha256"] = queue["baseline"]["classification"]["manifest_sha256"]
    identity["python_version"] = platform.python_version()
    identity["v3_sparse_pilot_freeze_signature"] = json.loads(
        (SPARSE_PILOT_CONFIG / "freeze.json").read_text(encoding="utf-8"))["signature"]
    identity["resource_preflight_recipe"] = {
        "estimated_peak_job_gib": preflight["estimated_peak_job_gib"],
        "cap_gib": 6.0, "minimum_free_ram_gib": 4.0, "reserve_gib": 2.0,
        "estimate_method": "train-only token upper bound, sparse matrix bytes x3 plus 1.5GiB fixed",
    }
    cycles = pd.to_numeric(frame.cycle_code, errors="raise").to_numpy(dtype=int)
    for cutoff in sorted({origin for segments in views.values() for origin, _, _ in segments}):
        score_cycles = {
            cycle
            for segments in views.values()
            for origin, low, high in segments if origin == cutoff
            for cycle in range(low, high + 1)
        }
        score_rows = np.flatnonzero(np.isin(cycles, np.asarray(sorted(score_cycles), dtype=int)))
        bridged = controls(int(cutoff), score_rows)
        if np.asarray(bridged["baseline_prediction"]).shape != (len(score_rows),):
            raise BatchIntegrityError(f"Control bridge output shape mismatch at cutoff {cutoff}")
        if np.asarray(bridged["stream_missing"]).shape != (len(score_rows),):
            raise BatchIntegrityError(f"Missing-stream route mask shape mismatch at cutoff {cutoff}")
        for name, values in bridged.items():
            if np.asarray(values).shape[0] != len(score_rows):
                raise BatchIntegrityError(f"Control {name} is not aligned at cutoff {cutoff}")
    if preflight_only:
        return {
            "status": "PREFLIGHT_PASS_NO_FREEZE_NO_FIT",
            "card_id": "C03", "preflight": preflight,
            "source_files": len(_source_files()), "control_files": len(control_files),
            "evaluation_views": views,
            "freeze_identity_sha256": json_sha256(identity),
            "official_test_read": False, "model_fit_performed": False,
        }
    append_event(REPORT_DIR / "attempts.jsonl", {
        "event": "PREFLIGHT_PASS", **preflight,
    })
    report_path = REPORT_DIR / "pilot.json"
    budget_ledger_path = REPORT_DIR.parent / "batch_ledger.jsonl"
    report = run_pilot(frame=frame, target=target, card=card,
        feature_names=FEATURE_NAMES, adapter=fit_predict, control_loader=controls,
        freeze_identity=identity, freeze_path=FREEZE_PATH, artifact_dir=ARTIFACT_DIR,
        event_path=REPORT_DIR / "attempts.jsonl", evaluation_views=views,
        threads=2, report_path=report_path, budget_ledger_path=budget_ledger_path,
        max_successful_fits=int(queue["execution_preset"]["max_successful_new_predictor_fits"]))
    verification = verify_report(report=report, frame=frame, target=target,
        artifact_dir=ARTIFACT_DIR, freeze_signature=report["freeze_signature"],
        evaluation_views=views, prediction_reconstructors={"C03": reconstruct_c03})
    gate_config = queue["evaluation"]["classification_gate"]
    subgroup_failures = verification["views"]["primary"]["subgroup_diagnostics"]["supported_guard_failures"]
    checks = {
        "primary_gain_at_least_0_0005": report["views"]["primary"]["gain_positive_is_better"] >= float(gate_config["pilot_primary_gain_min"]),
        "H12_harm_at_most_0_001": report["views"]["H12"]["gain_positive_is_better"] >= -float(gate_config["pilot_H12_harm_max"]),
        "W1_harm_at_most_0_005": report["views"]["W1"]["gain_positive_is_better"] >= -float(gate_config["individual_pilot_view_harm_max"]),
        "W4_harm_at_most_0_005": report["views"]["W4"]["gain_positive_is_better"] >= -float(gate_config["individual_pilot_view_harm_max"]),
        "supported_subgroup_guards": len(subgroup_failures) == 0,
    }
    gate = {
        "card_id": "C03", "profile": card["evaluation"],
        "status": "ADVANCE" if all(checks.values()) else "RETIRE",
        "checks": {name: bool(value) for name, value in checks.items()},
        "thresholds": gate_config,
        "supported_subgroup_failures": subgroup_failures,
        "decision_basis": "Predeclared temporal pilot gates only; no Test or Kaggle result used.",
    }
    gate_path = REPORT_DIR / "gate.json"
    if gate_path.exists():
        existing_gate = json.loads(gate_path.read_text(encoding="utf-8"))
        if existing_gate != gate:
            raise BatchIntegrityError("Existing pilot gate differs from independently verified evidence")
    else:
        write_json_exclusive(gate_path, gate)
    verification_path = REPORT_DIR / "verification.json"
    if verification_path.exists():
        existing = json.loads(verification_path.read_text(encoding="utf-8"))
        if existing != verification:
            raise BatchIntegrityError("Existing independent verification differs")
    else:
        write_json_exclusive(verification_path, verification)
    return {"pilot": report, "verification": verification, "gate": gate, "preflight": preflight}


__all__ = ["run_card"]
