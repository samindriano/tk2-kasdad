"""Execute the frozen full-six classification comparison for three candidates."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import subprocess
import sys
from time import perf_counter
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
from threadpoolctl import threadpool_limits

from src.data.audit import sha256 as file_sha256, write_json
from src.models.campaign_classification import build_campaign_classifier
from src.models.classification_metrics import binary_metrics
from src.models.classification_round2_models import stream_missing_mask
from src.models.round1_evaluation import array_digest, code_digest, digest_json
from src.utils.paths import PROJECT_ROOT, task_files
from scripts.research_campaign_classification import (
    AMENDMENT_FREEZE_PATH,
    AMENDMENT_PATH,
    CANDIDATES,
    FAILED_ATTEMPT_PATH,
    PROTOCOL_PATH as STAGE1_PROTOCOL_PATH,
    REPORT_DIR as STAGE1_REPORT_DIR,
    ROUND1_AUDITS,
    ROUND1_CACHE,
    RUN_DIR as STAGE1_RUN_DIR,
    amendment_identity,
    load_train,
    verify_amendment,
)


STAGE2_PROTOCOL_PATH = ROOT / "experiments/dual_campaign/classification/stage2_protocol.json"
STAGE2_FREEZE_PATH = ROOT / "experiments/dual_campaign/classification/stage2_freeze.json"
RUN_DIR = ROOT / "experiments/runs/dual_cls_v1_stage2"
REPORT_DIR = ROOT / "reports/dual_campaign/classification/stage2"
STAGE1_RESULTS_PATH = STAGE1_REPORT_DIR / "stage1_results.json"
STAGE1_VERIFICATION_PATH = STAGE1_REPORT_DIR / "stage1_verification.json"
ALL_FIT_ENDS = [55, 59, 63, 67, 71, 75]
NEW_FIT_ENDS = [59, 63, 71, 75]
PRIMARY_WINDOWS = {
    "W1": (55, 56, 59),
    "W2": (59, 60, 63),
    "W3": (63, 64, 67),
    "W4": (67, 68, 71),
    "W5": (71, 72, 75),
    "W6": (75, 76, 79),
}
CODE_FILES = [
    "experiments/dual_campaign/classification/protocol.json",
    "experiments/dual_campaign/classification/amendment_01.json",
    "experiments/dual_campaign/classification/amendment_01_freeze.json",
    "experiments/dual_campaign/classification/stage2_protocol.json",
    "scripts/research_campaign_classification.py",
    "scripts/verify_campaign_classification.py",
    "scripts/research_campaign_classification_stage2.py",
    "scripts/verify_campaign_classification_stage2.py",
    "tests/test_campaign_classification.py",
    "tests/test_campaign_classification_stage2.py",
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


def _train_file(data_dir: Path | None = None) -> Path:
    path = task_files("classification", data_dir)["train"]
    if path is None:
        raise FileNotFoundError("Official Classification Train.csv was not found")
    return path.resolve()


def _control_identity(fit_end: int) -> dict[str, dict[str, str]]:
    records = {}
    for model_id in ("C1_B", "C2_B"):
        cache = ROUND1_CACHE / f"fit{fit_end}__{model_id}.npz"
        audit = ROUND1_AUDITS / f"fit{fit_end}.json"
        if not cache.is_file() or not audit.is_file():
            raise FileNotFoundError(f"Round 1 {model_id} cache/audit missing for fit{fit_end}")
        records[model_id] = {"cache_sha256": file_sha256(cache), "audit_sha256": file_sha256(audit)}
    return records


def _stage1_artifact_identity() -> dict[str, Any]:
    stage1_freeze = _read_json(AMENDMENT_FREEZE_PATH)
    results = _read_json(STAGE1_RESULTS_PATH)
    verification = _read_json(STAGE1_VERIFICATION_PATH)
    if (verification.get("status") != "PASS"
            or results.get("freeze_signature") != stage1_freeze.get("signature")
            or verification.get("freeze_signature") != stage1_freeze.get("signature")):
        raise RuntimeError("Stage 1 outputs must pass independent verification before Stage 2 freeze")
    used = {}
    for end in (55, 67):
        for recipe in CANDIDATES:
            for suffix, path in (("cache", STAGE1_RUN_DIR / f"fit{end}__{recipe}.npz"),
                                 ("meta", STAGE1_RUN_DIR / f"fit{end}__{recipe}.json")):
                if not path.is_file():
                    raise FileNotFoundError(f"Stage 1 reusable fit artifact is missing: {path}")
                used[f"{end}/{recipe}/{suffix}"] = file_sha256(path)
    return {
        "stage1_amendment_freeze_sha256": file_sha256(AMENDMENT_FREEZE_PATH),
        "stage1_results_sha256": file_sha256(STAGE1_RESULTS_PATH),
        "stage1_verification_sha256": file_sha256(STAGE1_VERIFICATION_PATH),
        "stage1_reused_fit_artifacts": used,
    }


def identity(data_dir: Path | None = None) -> dict[str, Any]:
    verify_amendment(data_dir)
    amendment_id = amendment_identity(data_dir)
    train = _train_file(data_dir)
    return {
        "stage2_protocol_sha256": digest_json(_read_json(STAGE2_PROTOCOL_PATH)),
        "code_sha256": {name: code_digest(ROOT / name) for name in CODE_FILES},
        "classification_train_sha256": file_sha256(train),
        "stage1_inputs": _stage1_artifact_identity(),
        "stage1_amendment_identity_sha256": digest_json(amendment_id),
        "round1_control_inputs": {str(end): _control_identity(end) for end in ALL_FIT_ENDS},
        "python": platform.python_version(),
        "packages": {name: importlib.metadata.version(name)
                     for name in ["numpy", "pandas", "scipy", "scikit-learn", "threadpoolctl"]},
    }


def freeze_protocol(data_dir: Path | None = None) -> dict[str, Any]:
    current = identity(data_dir)
    if STAGE2_FREEZE_PATH.exists():
        receipt = _read_json(STAGE2_FREEZE_PATH)
        if receipt.get("identity") != current:
            raise RuntimeError("Existing Stage 2 freeze differs; preserve it and create a new protocol before fitting")
        return receipt
    parent = verify_amendment(data_dir)
    receipt = {
        "status": "FULL_SIX_STAGE_FROZEN_BEFORE_NEW_FITS",
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_commit": parent["source_commit"],
        "parent_amendment_freeze_signature": parent["signature"],
        "identity": current,
        "signature": digest_json(current),
        "hash_convention": "Canonical protocol JSON; source UTF-8 normalized to LF; official Train, controls, and reused Stage 1 artifacts byte-exact.",
        "source_boundary": "Official Classification Train only; Test was not opened.",
    }
    write_json(STAGE2_FREEZE_PATH, receipt)
    return receipt


def verify_freeze(data_dir: Path | None = None) -> dict[str, Any]:
    if not STAGE2_FREEZE_PATH.is_file():
        raise RuntimeError("Freeze full-six Stage 2 before any additional candidate fit")
    receipt = _read_json(STAGE2_FREEZE_PATH)
    current = identity(data_dir)
    if receipt.get("identity") != current or receipt.get("signature") != digest_json(current):
        raise RuntimeError("Stage 2 protocol, Train, controls, reuse artifacts, code, or runtime changed")
    if receipt.get("parent_amendment_freeze_signature") != _read_json(AMENDMENT_FREEZE_PATH).get("signature"):
        raise RuntimeError("Stage 2 freeze parent amendment signature mismatch")
    return receipt


def _expected_rows(frame: pd.DataFrame, fit_end: int, horizon: int) -> np.ndarray:
    cycle = pd.to_numeric(frame.cycle_code, errors="raise").to_numpy(dtype=int)
    return np.flatnonzero((cycle > fit_end) & (cycle <= horizon))


def _load_round1_control(frame: pd.DataFrame, fit_end: int, horizon: int,
                         model_id: str) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    cache = ROUND1_CACHE / f"fit{fit_end}__{model_id}.npz"
    audit_path = ROUND1_AUDITS / f"fit{fit_end}.json"
    audit = _read_json(audit_path).get(model_id)
    if not isinstance(audit, dict) or audit.get("model") != model_id:
        raise ValueError(f"Round 1 audit identity mismatch: fit{fit_end}/{model_id}")
    with np.load(cache, allow_pickle=False) as archive:
        rows = archive["outer_rows"].astype(np.int64, copy=True)
        probability = archive["probability"].astype(float, copy=True)
    expected = _expected_rows(frame, fit_end, horizon)
    if not np.array_equal(rows, expected) or probability.shape != rows.shape:
        raise ValueError(f"Round 1 control row identity mismatch: fit{fit_end}/{model_id}")
    if not np.isfinite(probability).all() or ((probability < 0) | (probability > 1)).any():
        raise ValueError(f"Round 1 control scores invalid: fit{fit_end}/{model_id}")
    if array_digest(probability) != audit.get("probability_sha256"):
        raise ValueError(f"Round 1 control score hash mismatch: fit{fit_end}/{model_id}")
    expected_fit = int(pd.to_numeric(frame.cycle_code).between(47, fit_end).sum())
    if audit.get("model") != model_id or int(audit.get("fit_rows", -1)) != expected_fit:
        raise ValueError(f"Round 1 control fit audit mismatch: fit{fit_end}/{model_id}")
    return rows, probability, {
        "cache_sha256": file_sha256(cache),
        "audit_sha256": file_sha256(audit_path),
        "probability_sha256": array_digest(probability),
        "fit_rows": expected_fit,
    }


def compact_metrics(y: np.ndarray, probability: np.ndarray) -> dict[str, Any]:
    actual = np.asarray(y, dtype=np.int8)
    p = np.asarray(probability, dtype=float)
    base = binary_metrics(actual, p, threshold=0.50)
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
    }


def score_rows(frame: pd.DataFrame, y: np.ndarray, rows: np.ndarray,
               probability: np.ndarray) -> dict[str, Any]:
    cycles = pd.to_numeric(frame.cycle_code).to_numpy(dtype=int)[rows]
    actual = y[rows]
    result = compact_metrics(actual, probability)
    cycle_results = []
    for cycle in sorted(np.unique(cycles)):
        mask = cycles == cycle
        cycle_results.append({"cycle": int(cycle), **compact_metrics(actual[mask], probability[mask])})
    result["per_cycle"] = cycle_results
    result["equal_cycle_macro_f1"] = float(np.mean([item["macro_f1"] for item in cycle_results]))
    result["cycles"] = [int(value) for value in sorted(np.unique(cycles))]
    return result


def complementarity_counts(y: np.ndarray, candidate_probability: np.ndarray,
                           control_probability: np.ndarray) -> dict[str, int]:
    actual = np.asarray(y, dtype=np.int8)
    candidate = np.asarray(candidate_probability, dtype=float) >= 0.50
    control = np.asarray(control_probability, dtype=float) >= 0.50
    if actual.ndim != 1 or candidate.shape != actual.shape or control.shape != actual.shape:
        raise ValueError("Complementarity vectors must be aligned")
    candidate_error = candidate != actual
    control_error = control != actual
    candidate_only = control_error & ~candidate_error
    control_only = candidate_error & ~control_error
    return {
        "n": int(len(actual)),
        "both_correct": int((~candidate_error & ~control_error).sum()),
        "both_wrong": int((candidate_error & control_error).sum()),
        "candidate_only_correct": int(candidate_only.sum()),
        "C4_only_correct": int(control_only.sum()),
        "net_candidate_correct_count_delta": int(candidate_only.sum() - control_only.sum()),
        "disagreements": int(np.count_nonzero(candidate != control)),
    }


def blend_gate(metrics: dict[str, int]) -> bool:
    return (metrics["candidate_only_correct"] >= 25
            and metrics["C4_only_correct"] >= 25
            and metrics["net_candidate_correct_count_delta"] >= 0)


def _stage1_window(frame: pd.DataFrame, recipe: str, fit_end: int,
                   start: int, end: int) -> dict[str, np.ndarray]:
    cache_path = STAGE1_RUN_DIR / f"fit{fit_end}__{recipe}.npz"
    with np.load(cache_path, allow_pickle=False) as archive:
        rows = archive["outer_rows"].astype(np.int64, copy=True)
        candidate = archive["candidate_probability"].astype(float, copy=True)
        routed = archive["routed_probability"].astype(float, copy=True)
        c4 = archive["c4_probability"].astype(float, copy=True)
        missing = archive["missing_stream_mask"].astype(bool, copy=True)
    cycle = pd.to_numeric(frame.cycle_code).to_numpy(dtype=int)[rows]
    chosen = (cycle >= start) & (cycle <= end)
    if not chosen.any() or not np.array_equal(rows[chosen], _expected_rows(frame, start - 1, end)):
        raise ValueError(f"Reused Stage 1 {recipe} fit{fit_end} rows do not match primary window {start}-{end}")
    return {
        "rows": rows[chosen],
        "candidate_probability": candidate[chosen],
        "routed_probability": routed[chosen],
        "c4_probability": c4[chosen],
        "missing_stream_mask": missing[chosen],
        "source": "stage1_reused_without_refit",
    }


def _fit_new_window(frame: pd.DataFrame, y: np.ndarray, recipe: str,
                    fit_end: int) -> dict[str, Any]:
    window = next(name for name, data in PRIMARY_WINDOWS.items() if data[0] == fit_end)
    _, start, end = PRIMARY_WINDOWS[window]
    cycle = pd.to_numeric(frame.cycle_code, errors="raise").to_numpy(dtype=int)
    fit_rows = np.flatnonzero((cycle >= 47) & (cycle <= fit_end))
    rows = _expected_rows(frame, fit_end, end)
    features = build_campaign_classifier(recipe).input_feature_names
    X_fit = frame.iloc[fit_rows][features].copy()
    y_fit = y[fit_rows]
    X_outer = frame.iloc[rows][features].copy()
    c1_rows, c1_probability, c1_identity = _load_round1_control(frame, fit_end, end, "C1_B")
    c2_rows, c2_probability, c2_identity = _load_round1_control(frame, fit_end, end, "C2_B")
    if not np.array_equal(rows, c1_rows) or not np.array_equal(rows, c2_rows):
        raise ValueError(f"fit{fit_end}: Round 1 controls do not align to {window}")
    missing = stream_missing_mask(frame.iloc[rows])
    c4 = np.where(missing, c1_probability, c2_probability)

    cache_path = RUN_DIR / f"fit{fit_end}__{recipe}.npz"
    meta_path = RUN_DIR / f"fit{fit_end}__{recipe}.json"
    if cache_path.exists() or meta_path.exists():
        raise FileExistsError(f"Refusing to overwrite Stage 2 artifact {cache_path}")
    model = build_campaign_classifier(recipe)
    started = perf_counter()
    with threadpool_limits(limits=2):
        model.fit(X_fit, y_fit)
        state = model.audit_state()
        candidate = np.asarray(model.predict_proba(X_outer), dtype=float)
        diagnostics = model.prediction_diagnostics()
        margin = model.predict_margin(X_outer) if recipe == "DC_SVC" else np.empty(0, dtype=float)
    if candidate.shape != rows.shape or not np.isfinite(candidate).all() or ((candidate < 0) | (candidate > 1)).any():
        raise RuntimeError(f"{recipe} returned invalid scores for fit{fit_end}")
    if digest_json(state) != digest_json(model.audit_state()):
        raise RuntimeError(f"{recipe} fit state changed during predictions at fit{fit_end}")
    routed = np.where(missing, c1_probability, candidate)
    if not np.array_equal(routed[missing], c1_probability[missing]):
        raise RuntimeError("Stage 2 changed the archived C1_B missing-stream branch")
    meta = {
        "recipe_id": recipe,
        "primary_window": window,
        "fit_end": fit_end,
        "validation_cycles": [start, end],
        "fit_rows": int(len(fit_rows)),
        "validation_rows": int(len(rows)),
        "features": list(features),
        "fit_data_sha256": state["fit_data_sha256"],
        "fit_state": state,
        "prediction_diagnostics": diagnostics,
        "threshold": 0.50,
        "outer_labels_used_for_fit_or_selection": False,
        "source_round1_controls": {"C1_B": c1_identity, "C2_B": c2_identity},
        "runtime_seconds": perf_counter() - started,
        "candidate_probability_sha256": array_digest(candidate),
        "routed_probability_sha256": array_digest(routed),
        "C4_HGB_probability_sha256": array_digest(c4),
        "margin_sha256": array_digest(margin) if recipe == "DC_SVC" else None,
    }
    np.savez_compressed(cache_path, outer_rows=rows.astype(np.int64), candidate_probability=candidate,
                        routed_probability=routed, c4_probability=c4,
                        missing_stream_mask=missing.astype(np.uint8),
                        **({"decision_margin": margin} if recipe == "DC_SVC" else {}))
    meta["cache_sha256"] = file_sha256(cache_path)
    write_json(meta_path, meta)
    return {
        "rows": rows,
        "candidate_probability": candidate,
        "routed_probability": routed,
        "c4_probability": c4,
        "missing_stream_mask": missing,
        "meta": meta,
        "source": "stage2_new_fit",
    }


def _collect_all_windows(frame: pd.DataFrame, y: np.ndarray) -> dict[str, dict[str, dict[str, Any]]]:
    windows: dict[str, dict[str, dict[str, Any]]] = {}
    for name, (fit_end, start, end) in PRIMARY_WINDOWS.items():
        windows[name] = {}
        for recipe in CANDIDATES:
            if fit_end in (55, 67):
                record = _stage1_window(frame, recipe, fit_end, start, end)
            else:
                record = _fit_new_window(frame, y, recipe, fit_end)
            windows[name][recipe] = record
    return windows


def _flatten_candidate(windows: dict[str, dict[str, dict[str, Any]]],
                       recipe: str) -> dict[str, np.ndarray]:
    ordered = [windows[name][recipe] for name in PRIMARY_WINDOWS]
    rows = np.concatenate([item["rows"] for item in ordered]).astype(np.int64)
    candidate = np.concatenate([item["candidate_probability"] for item in ordered]).astype(float)
    routed = np.concatenate([item["routed_probability"] for item in ordered]).astype(float)
    c4 = np.concatenate([item["c4_probability"] for item in ordered]).astype(float)
    missing = np.concatenate([item["missing_stream_mask"] for item in ordered]).astype(bool)
    if len(np.unique(rows)) != len(rows):
        raise ValueError(f"{recipe} full-six outer rows overlap")
    return {"rows": rows, "candidate_probability": candidate, "routed_probability": routed,
            "c4_probability": c4, "missing_stream_mask": missing}


def _make_plot(report: dict[str, Any]) -> list[Path]:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    colors = {"C4_HGB": "#333333", "DC_SVC": "#4472C4", "DC_NBSVM": "#70AD47", "DC_WORD_CHAR": "#ED7D31"}
    windows = list(PRIMARY_WINDOWS)
    path1 = REPORT_DIR / "full_six_window_macro_f1.png"
    fig, ax = plt.subplots(figsize=(9.4, 4.8))
    x = np.arange(len(windows))
    for recipe in ["C4_HGB", *CANDIDATES]:
        ys = [report["window_metrics"][name][recipe]["macro_f1"] for name in windows]
        ax.plot(x, ys, marker="o", linewidth=1.8, label=recipe, color=colors[recipe])
    ax.set_xticks(x, windows)
    ax.set_ylim(0.55, 0.90)
    ax.set_ylabel("Macro F1 · fixed threshold 0.50")
    ax.set_title("Six expanding-window primary comparisons (cycles 56–79)")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False, ncol=2)
    fig.tight_layout()
    fig.savefig(path1, dpi=180, bbox_inches="tight")
    plt.close(fig)

    path2 = REPORT_DIR / "paired_present_errors.png"
    fig, ax = plt.subplots(figsize=(8.2, 4.5))
    positions = np.arange(len(CANDIDATES))
    width = 0.36
    candidate_only = [report["complementarity"][recipe]["candidate_only_correct"] for recipe in CANDIDATES]
    c4_only = [report["complementarity"][recipe]["C4_only_correct"] for recipe in CANDIDATES]
    ax.bar(positions - width / 2, candidate_only, width, label="Candidate-only correct", color="#70AD47")
    ax.bar(positions + width / 2, c4_only, width, label="C4-only correct", color="#777777")
    ax.set_xticks(positions, CANDIDATES, rotation=15)
    ax.set_ylabel("Stream-present validation rows")
    ax.set_title("Paired error corrections vs C4_HGB · cycles 56–79")
    ax.grid(axis="y", alpha=0.22)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path2, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return [path1, path2]


def run(data_dir: Path | None = None) -> dict[str, Any]:
    frozen = verify_freeze(data_dir)
    frame, y, train = load_train(data_dir)
    if file_sha256(train) != frozen["identity"]["classification_train_sha256"]:
        raise RuntimeError("Official Train hash differs from Stage 2 freeze")
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    for fit_end in NEW_FIT_ENDS:
        for recipe in CANDIDATES:
            for suffix in ("json", "npz"):
                path = RUN_DIR / f"fit{fit_end}__{recipe}.{suffix}"
                if path.exists():
                    raise FileExistsError(f"Stage 2 refuses output overwrite: {path}")
    started = perf_counter()
    windows = _collect_all_windows(frame, y)
    full = {recipe: _flatten_candidate(windows, recipe) for recipe in CANDIDATES}
    all_rows = full[CANDIDATES[0]]["rows"]
    expected_primary_rows = np.flatnonzero(pd.to_numeric(frame.cycle_code).between(56, 79).to_numpy())
    if not np.array_equal(np.sort(all_rows), expected_primary_rows):
        raise RuntimeError("Full-six validation rows do not cover cycles 56-79 exactly once")
    report: dict[str, Any] = {
        "status": "FULL_SIX_COMPLETE_REVIEW_REQUIRED",
        "protocol_id": _read_json(STAGE2_PROTOCOL_PATH)["protocol_id"],
        "freeze_signature": frozen["signature"],
        "source_commit": frozen["source_commit"],
        "parent_amendment_freeze_signature": frozen["parent_amendment_freeze_signature"],
        "train_sha256": file_sha256(train),
        "train_rows": int(len(frame)),
        "cycle_range": [int(pd.to_numeric(frame.cycle_code).min()), int(pd.to_numeric(frame.cycle_code).max())],
        "validation_rows": int(len(all_rows)),
        "test_read": False,
        "kaggle_submission": False,
        "new_fit_count": len(NEW_FIT_ENDS) * len(CANDIDATES),
        "new_fit_endpoints": NEW_FIT_ENDS,
        "candidates": CANDIDATES,
        "SVC_probability_note": "DC_SVC expit(margin) is an uncalibrated monotone score. Its AP is a ranking metric; its Brier score is descriptive only, not calibration evidence.",
        "adaptation_warning": _read_json(STAGE2_PROTOCOL_PATH)["adaptation_warning"],
    }
    report["window_metrics"] = {}
    cycle_vector = pd.to_numeric(frame.cycle_code).to_numpy(dtype=int)
    for name, (fit_end, start, end) in PRIMARY_WINDOWS.items():
        report["window_metrics"][name] = {}
        for recipe in CANDIDATES:
            record = windows[name][recipe]
            rows = record["rows"]
            candidate_metrics = score_rows(frame, y, rows, record["routed_probability"])
            c4_metrics = score_rows(frame, y, rows, record["c4_probability"])
            report["window_metrics"][name][recipe] = {
                "candidate": candidate_metrics,
                "C4_HGB": c4_metrics,
                "macro_f1_delta": float(candidate_metrics["macro_f1"] - c4_metrics["macro_f1"]),
                "equal_cycle_macro_f1_delta": float(candidate_metrics["equal_cycle_macro_f1"] - c4_metrics["equal_cycle_macro_f1"]),
                "stream_present_rows": int((~record["missing_stream_mask"]).sum()),
                "stream_missing_rows": int(record["missing_stream_mask"].sum()),
                "source": record["source"],
            }
    report["full_primary"] = {}
    report["h12"] = {}
    report["subgroups"] = {}
    report["complementarity"] = {}
    report["candidate_gate"] = {}
    for recipe in CANDIDATES:
        record = full[recipe]
        rows = record["rows"]
        actual = y[rows]
        candidate_route = record["routed_probability"]
        c4 = record["c4_probability"]
        report["full_primary"][recipe] = {
            "candidate": score_rows(frame, y, rows, candidate_route),
            "C4_HGB": score_rows(frame, y, rows, c4),
        }
        report["full_primary"][recipe]["macro_f1_delta"] = float(
            report["full_primary"][recipe]["candidate"]["macro_f1"] - report["full_primary"][recipe]["C4_HGB"]["macro_f1"]
        )
        report["full_primary"][recipe]["equal_cycle_macro_f1_delta"] = float(
            report["full_primary"][recipe]["candidate"]["equal_cycle_macro_f1"] - report["full_primary"][recipe]["C4_HGB"]["equal_cycle_macro_f1"]
        )
        report["full_primary"][recipe]["windows"] = list(PRIMARY_WINDOWS)
        for label, mask in (
            ("stream_present", ~record["missing_stream_mask"]),
            ("stream_missing", record["missing_stream_mask"]),
        ):
            report["subgroups"].setdefault(recipe, {})[label] = {
                "candidate": score_rows(frame, y, rows[mask], candidate_route[mask]),
                "C4_HGB": score_rows(frame, y, rows[mask], c4[mask]),
                "n": int(mask.sum()),
            }
        present = ~record["missing_stream_mask"]
        paired = complementarity_counts(
            y[rows[present]], record["candidate_probability"][present], c4[present]
        )
        paired["population"] = "stream_present only"
        paired["candidate_full_primary_macro_f1"] = report["full_primary"][recipe]["candidate"]["macro_f1"]
        paired["C4_full_primary_macro_f1"] = report["full_primary"][recipe]["C4_HGB"]["macro_f1"]
        paired["candidate_average_precision"] = report["full_primary"][recipe]["candidate"]["average_precision"]
        paired["C4_average_precision"] = report["full_primary"][recipe]["C4_HGB"]["average_precision"]
        report["complementarity"][recipe] = paired
        report["candidate_gate"][recipe] = {
            "blend_gate_passed": blend_gate(paired),
            "rule": "candidate_only_correct >= 25; C4_only_correct >= 25; net candidate correct-count delta >= 0",
        }
    # Re-score H12_E and H12_L from the already-frozen Stage 1 predictions.
    for recipe in CANDIDATES:
        h12_records = {}
        for label, fit_end, start, end in (("H12_E", 55, 56, 67), ("H12_L", 67, 68, 79)):
            path = STAGE1_RUN_DIR / f"fit{fit_end}__{recipe}.npz"
            with np.load(path, allow_pickle=False) as archive:
                rows = archive["outer_rows"].astype(np.int64, copy=True)
                routed = archive["routed_probability"].astype(float, copy=True)
                c4 = archive["c4_probability"].astype(float, copy=True)
            cy = cycle_vector[rows]
            selected = (cy >= start) & (cy <= end)
            selected_rows = rows[selected]
            h12_records[label] = {
                "candidate": score_rows(frame, y, selected_rows, routed[selected]),
                "C4_HGB": score_rows(frame, y, selected_rows, c4[selected]),
            }
            h12_records[label]["macro_f1_delta"] = float(
                h12_records[label]["candidate"]["macro_f1"] - h12_records[label]["C4_HGB"]["macro_f1"]
            )
        report["h12"][recipe] = h12_records

    best = max(CANDIDATES, key=lambda recipe: report["full_primary"][recipe]["candidate"]["macro_f1"])
    report["fixed_50_50_blend"] = {"evaluated": False, "best_candidate": best,
                                     "blend_weight_candidate": 0.50, "threshold": 0.50}
    if report["candidate_gate"][best]["blend_gate_passed"]:
        record = full[best]
        rows = record["rows"]
        present = ~record["missing_stream_mask"]
        blend = np.where(present,
                         0.50 * record["candidate_probability"] + 0.50 * record["c4_probability"],
                         record["c4_probability"])
        report["fixed_50_50_blend"] = {
            "evaluated": True,
            "best_candidate": best,
            "blend_weight_candidate": 0.50,
            "threshold": 0.50,
            "population": "Full cycles 56-79; only stream-present scores blended; missing rows retain C1_B",
            "metrics": score_rows(frame, y, rows, blend),
            "C4_HGB_metrics": score_rows(frame, y, rows, record["c4_probability"]),
            "candidate_metrics": score_rows(frame, y, rows, record["routed_probability"]),
            "macro_f1_delta_vs_C4": float(score_rows(frame, y, rows, blend)["macro_f1"]
                                                   - score_rows(frame, y, rows, record["c4_probability"])["macro_f1"]),
        }
    report["runtime_seconds"] = perf_counter() - started
    report["fit_audit_summary"] = {
        "new_fit_count": len(NEW_FIT_ENDS) * len(CANDIDATES),
        "new_fit_artifacts": [f"fit{end}__{recipe}" for end in NEW_FIT_ENDS for recipe in CANDIDATES],
        "new_outer_labels_used_for_fit_or_selection": False,
        "stage1_fit_55_67_reused_without_refit": True,
        "missing_stream_C1_B_branch_reused": True,
    }
    chart_paths = _make_plot(report)
    report["charts"] = {path.name: file_sha256(path) for path in chart_paths}
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    write_json(REPORT_DIR / "full_six_results.json", report)
    rows_table = []
    for window, records in report["window_metrics"].items():
        for recipe in CANDIDATES:
            current = records[recipe]
            rows_table.append({"window": window, "recipe": recipe,
                               "candidate_macro_f1": current["candidate"]["macro_f1"],
                               "C4_macro_f1": current["C4_HGB"]["macro_f1"],
                               "delta_macro_f1": current["macro_f1_delta"],
                               "candidate_equal_cycle_macro_f1": current["candidate"]["equal_cycle_macro_f1"],
                               "C4_equal_cycle_macro_f1": current["C4_HGB"]["equal_cycle_macro_f1"],
                               "candidate_average_precision": current["candidate"]["average_precision"],
                               "C4_average_precision": current["C4_HGB"]["average_precision"],
                               "candidate_brier": current["candidate"]["brier_score"],
                               "C4_brier": current["C4_HGB"]["brier_score"],
                               "stream_present_rows": current["stream_present_rows"],
                               "stream_missing_rows": current["stream_missing_rows"]})
    pd.DataFrame(rows_table).to_csv(REPORT_DIR / "full_six_window_comparison.csv", index=False)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--freeze-only", action="store_true", help="Freeze this protocol, full source, and all inputs before fitting.")
    parser.add_argument("--run", action="store_true", help="Run only the 12 new folds at cutoffs 59, 63, 71, and 75.")
    args = parser.parse_args()
    if args.freeze_only == args.run:
        parser.error("Choose exactly one of --freeze-only or --run")
    if args.freeze_only:
        receipt = freeze_protocol(args.data_dir)
        print(receipt["status"], receipt["signature"])
    else:
        report = run(args.data_dir)
        print(report["status"], report["new_fit_count"], "new fits", REPORT_DIR / "full_six_results.json")


if __name__ == "__main__":
    main()
