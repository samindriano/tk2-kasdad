"""Rebuild the Stage 2 report from frozen score caches; never fits models."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import importlib.metadata
import json
from pathlib import Path
import platform
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from src.data.audit import sha256 as file_sha256, write_json
from src.models.classification_metrics import binary_metrics
from src.models.classification_models import _fit_fingerprint, _json_digest
from src.models.classification_round2_models import stream_missing_mask
from src.models.round1_evaluation import array_digest, code_digest, digest_json
from src.utils.paths import task_files
from scripts.research_campaign_classification_stage2 import (
    ALL_FIT_ENDS,
    CANDIDATES,
    NEW_FIT_ENDS,
    PRIMARY_WINDOWS,
    RUN_DIR as STAGE2_RUN_DIR,
    STAGE1_RUN_DIR,
    STAGE1_VERIFICATION_PATH,
    AMENDMENT_FREEZE_PATH,
    STAGE2_FREEZE_PATH,
    STAGE2_PROTOCOL_PATH,
    _expected_rows,
    _load_round1_control,
    blend_gate,
    complementarity_counts,
    load_train,
    verify_freeze as verify_stage2_freeze,
)
from scripts.research_campaign_classification import (
    FAILED_ATTEMPT_PATH as STAGE1_FAILURE_PATH,
    verify_amendment,
)


RECOVERY_PROTOCOL_PATH = ROOT / "experiments/dual_campaign/classification/stage2_report_recovery_protocol.json"
RECOVERY_FREEZE_PATH = ROOT / "experiments/dual_campaign/classification/stage2_report_recovery_freeze.json"
RECOVERY_NOTE_PATH = ROOT / "reports/dual_campaign/classification/stage2_report_recovery_note.json"
REPORT_DIR = ROOT / "reports/dual_campaign/classification/stage2"
RESULT_PATH = REPORT_DIR / "full_six_results.json"
ROUND1_CACHE = ROOT / "experiments/runs/classification_round1_v1"
ROUND1_AUDITS = ROOT / "reports/classification_round1/fit_audits"
STAGE1_REPORT_DIR = ROOT / "reports/dual_campaign/classification/amendment_01"
CODE_FILES = [
    "experiments/dual_campaign/classification/protocol.json",
    "experiments/dual_campaign/classification/amendment_01.json",
    "experiments/dual_campaign/classification/amendment_01_freeze.json",
    "experiments/dual_campaign/classification/stage2_protocol.json",
    "experiments/dual_campaign/classification/stage2_freeze.json",
    "experiments/dual_campaign/classification/stage2_report_recovery_protocol.json",
    "scripts/research_campaign_classification.py",
    "scripts/verify_campaign_classification.py",
    "scripts/research_campaign_classification_stage2.py",
    "scripts/verify_campaign_classification_stage2.py",
    "scripts/assemble_campaign_classification_stage2.py",
    "scripts/verify_campaign_classification_stage2_recovery.py",
    "tests/test_campaign_classification.py",
    "tests/test_campaign_classification_stage2.py",
    "tests/test_campaign_classification_stage2_recovery.py",
    "src/models/campaign_classification.py",
    "src/models/classification_models.py",
    "src/models/classification_round2_models.py",
    "src/models/classification_metrics.py",
    "src/models/round1_models.py",
    "src/models/round1_evaluation.py",
    "src/utils/paths.py",
    "src/data/audit.py",
]


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _new_fit_input_hashes() -> dict[str, dict[str, str]]:
    result = {}
    for end in NEW_FIT_ENDS:
        for recipe in CANDIDATES:
            for suffix in ("npz", "json"):
                path = STAGE2_RUN_DIR / f"fit{end}__{recipe}.{suffix}"
                if not path.is_file():
                    raise FileNotFoundError(f"Completed Stage 2 fit artifact missing: {path}")
                result[f"fit{end}/{recipe}/{suffix}"] = {"path": path.relative_to(ROOT).as_posix(),
                                                           "sha256": file_sha256(path)}
    if len(result) != 24:
        raise RuntimeError(f"Expected 24 Stage 2 fit/cache files, found {len(result)}")
    return result


def _stage1_input_hashes() -> dict[str, str]:
    result = {}
    for end in (55, 67):
        for recipe in CANDIDATES:
            for suffix in ("npz", "json"):
                path = STAGE1_RUN_DIR / f"fit{end}__{recipe}.{suffix}"
                if not path.is_file():
                    raise FileNotFoundError(f"Reusable Stage 1 fit artifact missing: {path}")
                result[f"fit{end}/{recipe}/{suffix}"] = file_sha256(path)
    return result


def recovery_identity(data_dir: Path | None = None) -> dict[str, Any]:
    stage2_freeze = verify_stage2_freeze(data_dir)
    stage1_freeze = verify_amendment(data_dir)
    verification = _read_json(STAGE1_VERIFICATION_PATH)
    if verification.get("status") != "PASS" or verification.get("freeze_signature") != stage1_freeze["signature"]:
        raise RuntimeError("Stage 1 must remain independently verified before report reassembly")
    note = _read_json(RECOVERY_NOTE_PATH)
    if (note.get("parent_stage2_freeze_signature") != stage2_freeze["signature"]
            or note.get("successful_score_report_written") is not False):
        raise RuntimeError("Report-recovery failure receipt does not match frozen Stage 2")
    if RESULT_PATH.exists():
        raise FileExistsError("Refusing to overwrite an existing Stage 2 results report")
    train = task_files("classification", data_dir)["train"]
    if train is None:
        raise FileNotFoundError("Official Classification Train.csv was not found")
    return {
        "protocol_sha256": digest_json(_read_json(RECOVERY_PROTOCOL_PATH)),
        "code_sha256": {name: code_digest(ROOT / name) for name in CODE_FILES},
        "parent_stage2_freeze_signature": stage2_freeze["signature"],
        "parent_stage2_freeze_sha256": file_sha256(STAGE2_FREEZE_PATH),
        "stage1_amendment_freeze_sha256": file_sha256(AMENDMENT_FREEZE_PATH),
        "stage1_verification_sha256": file_sha256(STAGE1_VERIFICATION_PATH),
        "stage1_failure_note_sha256": file_sha256(STAGE1_FAILURE_PATH),
        "report_failure_note_sha256": file_sha256(RECOVERY_NOTE_PATH),
        "stage1_reused_fit_artifacts": _stage1_input_hashes(),
        "stage2_completed_fit_artifacts": _new_fit_input_hashes(),
        "classification_train_sha256": file_sha256(train),
        "round1_controls": {str(end): {
            model: {"cache_sha256": file_sha256(ROUND1_CACHE / f"fit{end}__{model}.npz"),
                    "audit_sha256": file_sha256(ROUND1_AUDITS / f"fit{end}.json")}
            for model in ("C1_B", "C2_B")}
            for end in ALL_FIT_ENDS},
        "python": platform.python_version(),
        "packages": {name: importlib.metadata.version(name)
                     for name in ["numpy", "pandas", "scipy", "scikit-learn", "threadpoolctl", "matplotlib"]},
    }


def freeze_recovery(data_dir: Path | None = None) -> dict[str, Any]:
    current = recovery_identity(data_dir)
    if RECOVERY_FREEZE_PATH.exists():
        receipt = _read_json(RECOVERY_FREEZE_PATH)
        if receipt.get("identity") != current:
            raise RuntimeError("Existing report-recovery freeze differs; preserve it and version a new recovery")
        return receipt
    stage2 = _read_json(STAGE2_FREEZE_PATH)
    receipt = {
        "status": "REPORT_ONLY_RECOVERY_FROZEN_NO_MODEL_FITS",
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_commit": stage2["source_commit"],
        "parent_stage2_freeze_signature": stage2["signature"],
        "identity": current,
        "signature": digest_json(current),
        "operation_boundary": "Saved predictions and model fit receipts are validated and scored; no model is fitted and no new prediction is generated.",
    }
    write_json(RECOVERY_FREEZE_PATH, receipt)
    return receipt


def verify_recovery_freeze(data_dir: Path | None = None) -> dict[str, Any]:
    if not RECOVERY_FREEZE_PATH.is_file():
        raise RuntimeError("Freeze report-only recovery before assembly")
    receipt = _read_json(RECOVERY_FREEZE_PATH)
    current = recovery_identity(data_dir)
    if receipt.get("identity") != current or receipt.get("signature") != digest_json(current):
        raise RuntimeError("Recovery protocol, scores, Train, fit receipts, code, or runtime changed")
    return receipt


def _record_from_stage1(frame: pd.DataFrame, recipe: str, fit_end: int,
                        start: int, end: int) -> dict[str, Any]:
    cache = STAGE1_RUN_DIR / f"fit{fit_end}__{recipe}.npz"
    with np.load(cache, allow_pickle=False) as archive:
        rows = archive["outer_rows"].astype(np.int64, copy=True)
        candidate = archive["candidate_probability"].astype(float, copy=True)
        routed = archive["routed_probability"].astype(float, copy=True)
        c4 = archive["c4_probability"].astype(float, copy=True)
        missing = archive["missing_stream_mask"].astype(bool, copy=True)
    cycle = pd.to_numeric(frame.cycle_code).to_numpy(dtype=int)[rows]
    chosen = (cycle >= start) & (cycle <= end)
    selected_rows = rows[chosen]
    if not np.array_equal(selected_rows, _expected_rows(frame, start - 1, end)):
        raise RuntimeError(f"Stage 1 reused window rows mismatch for {recipe}, fit{fit_end}")
    return {"rows": selected_rows, "candidate_probability": candidate[chosen],
            "routed_probability": routed[chosen], "c4_probability": c4[chosen],
            "missing_stream_mask": missing[chosen], "source": "stage1_reused_without_refit"}


def _record_from_stage2(frame: pd.DataFrame, y: np.ndarray, recipe: str, fit_end: int,
                        start: int, end: int) -> dict[str, Any]:
    cache_path = STAGE2_RUN_DIR / f"fit{fit_end}__{recipe}.npz"
    meta_path = STAGE2_RUN_DIR / f"fit{fit_end}__{recipe}.json"
    meta = _read_json(meta_path)
    if meta.get("cache_sha256") != file_sha256(cache_path):
        raise RuntimeError(f"Stage 2 cache hash mismatch for fit{fit_end}/{recipe}")
    cycle = pd.to_numeric(frame.cycle_code, errors="raise").to_numpy(dtype=int)
    fit_rows = np.flatnonzero((cycle >= 47) & (cycle <= fit_end))
    rows_expected = _expected_rows(frame, fit_end, end)
    features = meta["features"]
    fit_data = frame.iloc[fit_rows][features].copy()
    fit_target = y[fit_rows]
    fingerprint = _fit_fingerprint(fit_data, tuple(features), fit_target)
    state = meta["fit_state"]
    if (state["fit_data_sha256"] != fingerprint or state["fit_rows"] != len(fit_rows)
            or state["input_feature_names"] != features
            or meta.get("outer_labels_used_for_fit_or_selection") is not False):
        raise RuntimeError(f"Stage 2 fit-only audit failed for fit{fit_end}/{recipe}")
    with np.load(cache_path, allow_pickle=False) as archive:
        rows = archive["outer_rows"].astype(np.int64, copy=True)
        candidate = archive["candidate_probability"].astype(float, copy=True)
        routed = archive["routed_probability"].astype(float, copy=True)
        c4 = archive["c4_probability"].astype(float, copy=True)
        missing = archive["missing_stream_mask"].astype(bool, copy=True)
        margin = archive["decision_margin"].astype(float, copy=True) if "decision_margin" in archive else None
    if not np.array_equal(rows, rows_expected) or meta["validation_rows"] != len(rows):
        raise RuntimeError(f"Stage 2 validation rows invalid for fit{fit_end}/{recipe}")
    live_missing = stream_missing_mask(frame.iloc[rows])
    if not np.array_equal(missing, live_missing):
        raise RuntimeError(f"Stage 2 availability mask mismatch for fit{fit_end}/{recipe}")
    c1_rows, c1_p, _ = _load_round1_control(frame, fit_end, end, "C1_B")
    c2_rows, c2_p, _ = _load_round1_control(frame, fit_end, end, "C2_B")
    if not np.array_equal(c1_rows, rows) or not np.array_equal(c2_rows, rows):
        raise RuntimeError(f"Round 1 control rows mismatched for fit{fit_end}")
    c4_expected = np.where(missing, c1_p, c2_p)
    route_expected = np.where(missing, c1_p, candidate)
    if not np.array_equal(c4, c4_expected) or not np.array_equal(routed, route_expected):
        raise RuntimeError(f"Candidate/C4 availability route invalid for fit{fit_end}/{recipe}")
    if (array_digest(candidate) != meta["candidate_probability_sha256"]
            or array_digest(routed) != meta["routed_probability_sha256"]
            or array_digest(c4) != meta["C4_HGB_probability_sha256"]):
        raise RuntimeError(f"Probability digest mismatch for fit{fit_end}/{recipe}")
    if recipe == "DC_SVC":
        if margin is None or not np.isfinite(margin).all() or not np.array_equal(margin >= 0, candidate >= 0.50):
            raise RuntimeError(f"Fixed SVC margin rule invalid for fit{fit_end}")
    return {"rows": rows, "candidate_probability": candidate,
            "routed_probability": routed, "c4_probability": c4,
            "missing_stream_mask": missing, "fit_meta": meta,
            "source": "stage2_saved_fit"}


def _score(frame: pd.DataFrame, y: np.ndarray, rows: np.ndarray, p: np.ndarray) -> dict[str, Any]:
    actual = y[rows]
    base = binary_metrics(actual, p, threshold=0.50)
    cycle = pd.to_numeric(frame.cycle_code).to_numpy(dtype=int)[rows]
    per_cycle = []
    for value in sorted(np.unique(cycle)):
        mask = cycle == value
        metrics = binary_metrics(actual[mask], p[mask], threshold=0.50)
        per_cycle.append({"cycle": int(value), "n": metrics["n"], "macro_f1": metrics["macro_f1"],
                          "confusion_matrix": metrics["confusion_matrix"], "per_class": metrics["per_class"],
                          "positive_rate": metrics["positive_rate"],
                          "positive_prediction_rate": metrics["positive_prediction_rate"],
                          "average_precision": metrics["average_precision"],
                          "brier_score": metrics["brier_score"], "threshold": 0.50})
    return {
        "n": base["n"],
        "macro_f1": base["macro_f1"],
        "confusion_matrix": base["confusion_matrix"],
        "per_class": base["per_class"],
        "positive_rate": base["positive_rate"],
        "positive_prediction_rate": base["positive_prediction_rate"],
        "average_precision": float(average_precision_score(actual, p)) if len(np.unique(actual)) == 2 else None,
        "brier_score": base["brier_score"],
        "false_positive": base["false_positive"],
        "false_negative": base["false_negative"],
        "threshold": 0.50,
        "cycles": [int(value) for value in sorted(np.unique(cycle))],
        "per_cycle": per_cycle,
        "equal_cycle_macro_f1": float(np.mean([item["macro_f1"] for item in per_cycle])) if per_cycle else None,
    }


def _plot(report: dict[str, Any]) -> list[Path]:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    colors = {"C4_HGB": "#333333", "DC_SVC": "#4472C4", "DC_NBSVM": "#70AD47", "DC_WORD_CHAR": "#ED7D31"}
    names = list(PRIMARY_WINDOWS)
    x = np.arange(len(names))
    p1 = REPORT_DIR / "full_six_window_macro_f1.png"
    fig, axis = plt.subplots(figsize=(9.4, 4.7))
    for model_id in ["C4_HGB", *CANDIDATES]:
        vals = [report["window_metrics"][name][model_id]["macro_f1"] for name in names]
        axis.plot(x, vals, marker="o", linewidth=1.8, label=model_id, color=colors[model_id])
    axis.set_xticks(x, names)
    axis.set_ylim(0.55, 0.90)
    axis.set_ylabel("Macro F1 · threshold 0.50")
    axis.set_title("Six expanding windows · candidate routes vs C4_HGB")
    axis.grid(axis="y", alpha=0.25)
    axis.legend(frameon=False, ncol=2)
    fig.tight_layout()
    fig.savefig(p1, dpi=180, bbox_inches="tight")
    plt.close(fig)

    p2 = REPORT_DIR / "paired_present_errors.png"
    fig, axis = plt.subplots(figsize=(8.2, 4.5))
    position, width = np.arange(len(CANDIDATES)), 0.36
    candidate_only = [report["complementarity"][recipe]["candidate_only_correct"] for recipe in CANDIDATES]
    c4_only = [report["complementarity"][recipe]["C4_only_correct"] for recipe in CANDIDATES]
    axis.bar(position - width / 2, candidate_only, width, color="#70AD47", label="Candidate-only correct")
    axis.bar(position + width / 2, c4_only, width, color="#777777", label="C4-only correct")
    axis.set_xticks(position, CANDIDATES, rotation=15)
    axis.set_ylabel("Stream-present validation rows")
    axis.set_title("Paired corrections on the same validation rows")
    axis.grid(axis="y", alpha=0.22)
    axis.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(p2, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return [p1, p2]


def assemble(data_dir: Path | None = None) -> dict[str, Any]:
    recovery_freeze = verify_recovery_freeze(data_dir)
    stage2_freeze = verify_stage2_freeze(data_dir)
    frame, y, train = load_train(data_dir)
    if file_sha256(train) != recovery_freeze["identity"]["classification_train_sha256"]:
        raise RuntimeError("Official Train hash differs from report recovery freeze")
    # Re-validate all Stage 1 reused artifacts before combining the full-six windows.
    from scripts.verify_campaign_classification import verify as verify_stage1_outputs
    stage1_verification = verify_stage1_outputs(data_dir)
    if stage1_verification["status"] != "PASS":
        raise RuntimeError("Stage 1 independent verification failed")

    windows: dict[str, dict[str, dict[str, Any]]] = {name: {} for name in PRIMARY_WINDOWS}
    fit_artifacts = []
    for name, (fit_end, start, end) in PRIMARY_WINDOWS.items():
        for recipe in CANDIDATES:
            if fit_end in (55, 67):
                record = _record_from_stage1(frame, recipe, fit_end, start, end)
            else:
                record = _record_from_stage2(frame, y, recipe, fit_end, start, end)
                fit_artifacts.append(f"fit{fit_end}__{recipe}")
            windows[name][recipe] = record

    report: dict[str, Any] = {
        "status": "FULL_SIX_COMPLETE_REVIEW_REQUIRED",
        "protocol_id": _read_json(STAGE2_PROTOCOL_PATH)["protocol_id"],
        "freeze_signature": stage2_freeze["signature"],
        "report_recovery_freeze_signature": recovery_freeze["signature"],
        "source_commit": stage2_freeze["source_commit"],
        "parent_amendment_freeze_signature": stage2_freeze["parent_amendment_freeze_signature"],
        "train_sha256": file_sha256(train),
        "train_rows": int(len(frame)),
        "cycle_range": [int(pd.to_numeric(frame.cycle_code).min()), int(pd.to_numeric(frame.cycle_code).max())],
        "validation_rows": int(sum(len(windows[name][CANDIDATES[0]]["rows"]) for name in PRIMARY_WINDOWS)),
        "test_read": False,
        "kaggle_submission": False,
        "new_fit_count": len(fit_artifacts),
        "new_fit_endpoints": NEW_FIT_ENDS,
        "candidates": CANDIDATES,
        "SVC_probability_note": "DC_SVC expit(margin) is an uncalibrated monotone score. AP is valid for ranking; Brier is descriptive, not calibration evidence.",
        "adaptation_warning": _read_json(STAGE2_PROTOCOL_PATH)["adaptation_warning"],
        "report_recovery_note": _read_json(RECOVERY_NOTE_PATH),
        "stage1_verification": stage1_verification,
    }
    report["window_metrics"] = {}
    for name, records in windows.items():
        report["window_metrics"][name] = {}
        for recipe in CANDIDATES:
            item = records[recipe]
            rows = item["rows"]
            candidate_metrics = _score(frame, y, rows, item["routed_probability"])
            c4_metrics = _score(frame, y, rows, item["c4_probability"])
            report["window_metrics"][name][recipe] = {
                "candidate": candidate_metrics,
                "C4_HGB": c4_metrics,
                "macro_f1_delta": float(candidate_metrics["macro_f1"] - c4_metrics["macro_f1"]),
                "equal_cycle_macro_f1_delta": float(candidate_metrics["equal_cycle_macro_f1"] - c4_metrics["equal_cycle_macro_f1"]),
                "stream_present_rows": int((~item["missing_stream_mask"]).sum()),
                "stream_missing_rows": int(item["missing_stream_mask"].sum()),
                "source": item["source"],
            }
    report["full_primary"] = {}
    report["subgroups"] = {}
    report["complementarity"] = {}
    report["candidate_gate"] = {}
    report["h12"] = {}
    full_vectors = {}
    for recipe in CANDIDATES:
        pieces = [windows[name][recipe] for name in PRIMARY_WINDOWS]
        rows = np.concatenate([piece["rows"] for piece in pieces]).astype(np.int64)
        candidate = np.concatenate([piece["candidate_probability"] for piece in pieces]).astype(float)
        routed = np.concatenate([piece["routed_probability"] for piece in pieces]).astype(float)
        c4 = np.concatenate([piece["c4_probability"] for piece in pieces]).astype(float)
        missing = np.concatenate([piece["missing_stream_mask"] for piece in pieces]).astype(bool)
        if len(np.unique(rows)) != len(rows) or not np.array_equal(
                np.sort(rows), np.flatnonzero(pd.to_numeric(frame.cycle_code).between(56, 79).to_numpy())):
            raise RuntimeError(f"{recipe}: full primary validation rows do not cover cycles 56-79 exactly once")
        full_vectors[recipe] = {"rows": rows, "candidate_probability": candidate,
                                "routed_probability": routed, "c4_probability": c4,
                                "missing_stream_mask": missing}
        candidate_metric = _score(frame, y, rows, routed)
        c4_metric = _score(frame, y, rows, c4)
        report["full_primary"][recipe] = {
            "candidate": candidate_metric,
            "C4_HGB": c4_metric,
            "macro_f1_delta": float(candidate_metric["macro_f1"] - c4_metric["macro_f1"]),
            "equal_cycle_macro_f1_delta": float(candidate_metric["equal_cycle_macro_f1"] - c4_metric["equal_cycle_macro_f1"]),
            "windows": list(PRIMARY_WINDOWS),
        }
        for subgroup, mask in (("stream_present", ~missing), ("stream_missing", missing)):
            report["subgroups"].setdefault(recipe, {})[subgroup] = {
                "candidate": _score(frame, y, rows[mask], routed[mask]),
                "C4_HGB": _score(frame, y, rows[mask], c4[mask]),
                "n": int(mask.sum()),
            }
        present = ~missing
        paired = complementarity_counts(y[rows[present]], candidate[present], c4[present])
        paired["population"] = "stream_present only"
        paired["candidate_full_primary_macro_f1"] = candidate_metric["macro_f1"]
        paired["C4_full_primary_macro_f1"] = c4_metric["macro_f1"]
        paired["candidate_average_precision"] = candidate_metric["average_precision"]
        paired["C4_average_precision"] = c4_metric["average_precision"]
        report["complementarity"][recipe] = paired
        report["candidate_gate"][recipe] = {
            "blend_gate_passed": blend_gate(paired),
            "rule": "candidate_only_correct >= 25; C4_only_correct >= 25; net candidate correct-count delta >= 0",
        }
        for label, fit_end, start, end in (("H12_E", 55, 56, 67), ("H12_L", 67, 68, 79)):
            hrecord = _record_from_stage1(frame, recipe, fit_end, start, end)
            report["h12"].setdefault(recipe, {})[label] = {
                "candidate": _score(frame, y, hrecord["rows"], hrecord["routed_probability"]),
                "C4_HGB": _score(frame, y, hrecord["rows"], hrecord["c4_probability"]),
            }
            report["h12"][recipe][label]["macro_f1_delta"] = float(
                report["h12"][recipe][label]["candidate"]["macro_f1"] - report["h12"][recipe][label]["C4_HGB"]["macro_f1"]
            )
    best = max(CANDIDATES, key=lambda recipe: report["full_primary"][recipe]["candidate"]["macro_f1"])
    report["fixed_50_50_blend"] = {"evaluated": False, "best_candidate": best,
                                     "blend_weight_candidate": 0.50, "threshold": 0.50}
    if report["candidate_gate"][best]["blend_gate_passed"]:
        vec = full_vectors[best]
        rows = vec["rows"]
        present = ~vec["missing_stream_mask"]
        blended = np.where(present,
                           0.50 * vec["candidate_probability"] + 0.50 * vec["c4_probability"],
                           vec["c4_probability"])
        metrics = _score(frame, y, rows, blended)
        c4_metrics = _score(frame, y, rows, vec["c4_probability"])
        candidate_metrics = _score(frame, y, rows, vec["routed_probability"])
        report["fixed_50_50_blend"] = {
            "evaluated": True,
            "best_candidate": best,
            "blend_weight_candidate": 0.50,
            "threshold": 0.50,
            "population": "Full cycles 56-79; only present-stream scores blended; missing rows retain C1_B",
            "metrics": metrics,
            "C4_HGB_metrics": c4_metrics,
            "candidate_metrics": candidate_metrics,
            "macro_f1_delta_vs_C4": float(metrics["macro_f1"] - c4_metrics["macro_f1"]),
        }
    report["fit_audit_summary"] = {
        "new_fit_count": len(fit_artifacts),
        "new_fit_artifacts": fit_artifacts,
        "new_outer_labels_used_for_fit_or_selection": False,
        "stage1_fit_55_67_reused_without_refit": True,
        "missing_stream_C1_B_branch_reused": True,
        "all_new_fit_receipts_validated": True,
    }
    charts = _plot(report)
    report["charts"] = {path.name: file_sha256(path) for path in charts}
    report["assembled_at_utc"] = datetime.now(timezone.utc).isoformat()
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    write_json(RESULT_PATH, report)
    comparison = []
    for window, entries in report["window_metrics"].items():
        for recipe in CANDIDATES:
            item = entries[recipe]
            comparison.append({
                "window": window,
                "recipe": recipe,
                "candidate_macro_f1": item["candidate"]["macro_f1"],
                "C4_macro_f1": item["C4_HGB"]["macro_f1"],
                "delta_macro_f1": item["macro_f1_delta"],
                "candidate_equal_cycle_macro_f1": item["candidate"]["equal_cycle_macro_f1"],
                "C4_equal_cycle_macro_f1": item["C4_HGB"]["equal_cycle_macro_f1"],
                "candidate_average_precision": item["candidate"]["average_precision"],
                "C4_average_precision": item["C4_HGB"]["average_precision"],
                "candidate_brier": item["candidate"]["brier_score"],
                "C4_brier": item["C4_HGB"]["brier_score"],
                "stream_present_rows": item["stream_present_rows"],
                "stream_missing_rows": item["stream_missing_rows"],
            })
    pd.DataFrame(comparison).to_csv(REPORT_DIR / "full_six_window_comparison.csv", index=False)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--freeze-only", action="store_true")
    parser.add_argument("--assemble", action="store_true")
    args = parser.parse_args()
    if args.freeze_only == args.assemble:
        parser.error("Choose exactly one of --freeze-only or --assemble")
    if args.freeze_only:
        receipt = freeze_recovery(args.data_dir)
        print(receipt["status"], receipt["signature"])
    else:
        report = assemble(args.data_dir)
        print(report["status"], report["fit_audit_summary"]["new_fit_count"], "verified saved fits")


if __name__ == "__main__":
    main()
