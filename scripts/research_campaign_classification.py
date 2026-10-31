"""Freeze and run the bounded Stage 1 dual-campaign classification screen.

This runner reads official Classification Train only. It never reads Test,
chooses a threshold from outer labels, or writes into an older experiment.
"""
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
from threadpoolctl import threadpool_limits

from src.data.audit import sha256 as file_sha256, write_json
from src.models.campaign_classification import build_campaign_classifier
from src.models.classification_metrics import binary_metrics
from src.models.classification_models import _fit_fingerprint
from src.models.classification_round2_models import stream_missing_mask
from src.models.round1_evaluation import array_digest, code_digest, digest_json
from src.utils.paths import PROJECT_ROOT, task_files


PROTOCOL_PATH = ROOT / "experiments/dual_campaign/classification/protocol.json"
FREEZE_PATH = ROOT / "experiments/dual_campaign/classification/freeze.json"
AMENDMENT_PATH = ROOT / "experiments/dual_campaign/classification/amendment_01.json"
AMENDMENT_FREEZE_PATH = ROOT / "experiments/dual_campaign/classification/amendment_01_freeze.json"
FAILED_ATTEMPT_PATH = ROOT / "reports/dual_campaign/classification/stage1_attempt.json"
RUN_DIR = ROOT / "experiments/runs/dual_cls_v1a"
REPORT_DIR = ROOT / "reports/dual_campaign/classification/amendment_01"
ROUND1_CACHE = ROOT / "experiments/runs/classification_round1_v1"
ROUND1_AUDITS = ROOT / "reports/classification_round1/fit_audits"
CODE_FILES = [
    "experiments/dual_campaign/classification/protocol.json",
    "experiments/dual_campaign/classification/amendment_01.json",
    "src/models/campaign_classification.py",
    "scripts/research_campaign_classification.py",
    "scripts/verify_campaign_classification.py",
    "tests/test_campaign_classification.py",
    "src/models/classification_models.py",
    "src/models/classification_round2_models.py",
    "src/models/classification_metrics.py",
    "src/models/round1_models.py",
    "src/models/round1_evaluation.py",
    "src/utils/paths.py",
    "src/data/audit.py",
    "experiments/classification_round1/protocol.json",
    "experiments/classification_round1/freeze.json",
    "scripts/run_classification_round1.py",
]
CANDIDATES = ["DC_SVC", "DC_NBSVM", "DC_WORD_CHAR"]
EARLY_ENDS = [55, 67]
AMENDED_SOURCE_PATHS = {
    "src/models/campaign_classification.py",
    "scripts/research_campaign_classification.py",
    "scripts/verify_campaign_classification.py",
}


def train_path(data_dir: Path | None = None) -> Path:
    path = task_files("classification", data_dir)["train"]
    if path is None:
        raise FileNotFoundError("Official Classification Train.csv was not found")
    return path.resolve()


def control_identity(fit_end: int) -> dict[str, str]:
    identity = {}
    for model_id in ("C1_B", "C2_B"):
        cache = ROUND1_CACHE / f"fit{fit_end}__{model_id}.npz"
        audit = ROUND1_AUDITS / f"fit{fit_end}.json"
        if not cache.is_file() or not audit.is_file():
            raise FileNotFoundError(f"Verified Round 1 {model_id} control is absent for fit{fit_end}")
        identity[model_id] = {"cache_sha256": file_sha256(cache), "audit_sha256": file_sha256(audit)}
    return identity


def identity(data_dir: Path | None = None) -> dict[str, Any]:
    train = train_path(data_dir)
    return {
        "protocol_sha256": digest_json(json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))),
        "code_sha256": {name: code_digest(ROOT / name) for name in CODE_FILES},
        "classification_train_sha256": file_sha256(train),
        "round1_control_inputs": {str(end): control_identity(end) for end in EARLY_ENDS},
        "python": platform.python_version(),
        "packages": {name: importlib.metadata.version(name)
                     for name in ["numpy", "pandas", "scipy", "scikit-learn", "threadpoolctl"]},
    }


def freeze_protocol(data_dir: Path | None = None) -> dict[str, Any]:
    current = identity(data_dir)
    if FREEZE_PATH.exists():
        existing = json.loads(FREEZE_PATH.read_text(encoding="utf-8"))
        if existing.get("identity") != current:
            raise RuntimeError("Existing dual-campaign freeze differs; preserve it and issue a reviewed amendment before fitting.")
        return existing
    head = subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip()
    receipt = {
        "status": "INITIAL_SCREEN_FROZEN_BEFORE_NEW_FITS",
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_commit": head,
        "identity": current,
        "signature": digest_json(current),
        "hash_convention": "Canonical protocol JSON; repository source normalized to UTF-8 LF; official Train and archived control files hashed byte-for-byte.",
        "source_boundary": "Only official Classification Train was read for model fitting. Official Test was not opened.",
    }
    write_json(FREEZE_PATH, receipt)
    return receipt


def verify_freeze(data_dir: Path | None = None) -> dict[str, Any]:
    if not FREEZE_PATH.exists():
        raise RuntimeError("Freeze the initial protocol before any campaign fit.")
    receipt = json.loads(FREEZE_PATH.read_text(encoding="utf-8"))
    current = identity(data_dir)
    if receipt.get("identity") != current or receipt.get("signature") != digest_json(current):
        raise RuntimeError("Frozen protocol, Train, control cache, code, or runtime changed; refusing fit/cache reuse.")
    return receipt


def verify_parent_freeze(data_dir: Path | None = None) -> dict[str, Any]:
    """Validate the pre-amendment receipt, allowing only recorded source edits."""
    if not FREEZE_PATH.exists():
        raise RuntimeError("Initial dual-campaign freeze is missing")
    parent = json.loads(FREEZE_PATH.read_text(encoding="utf-8"))
    identity_before = parent.get("identity")
    if not isinstance(identity_before, dict) or parent.get("signature") != digest_json(identity_before):
        raise RuntimeError("Initial dual-campaign freeze receipt is corrupted")
    initial_protocol_hash = digest_json(json.loads(PROTOCOL_PATH.read_text(encoding="utf-8")))
    if initial_protocol_hash != identity_before.get("protocol_sha256"):
        raise RuntimeError("The original frozen protocol changed; amendment cannot proceed")
    train = train_path(data_dir)
    if file_sha256(train) != identity_before.get("classification_train_sha256"):
        raise RuntimeError("Official Classification Train changed after the initial freeze")
    current_controls = {str(end): control_identity(end) for end in EARLY_ENDS}
    if current_controls != identity_before.get("round1_control_inputs"):
        raise RuntimeError("Archived Round 1 C1_B/C2_B baseline artifacts changed")
    if platform.python_version() != identity_before.get("python"):
        raise RuntimeError("Python runtime changed since the initial freeze")
    packages = {name: importlib.metadata.version(name)
                for name in ["numpy", "pandas", "scipy", "scikit-learn", "threadpoolctl"]}
    if packages != identity_before.get("packages"):
        raise RuntimeError("Package versions changed since the initial freeze")
    for relative, expected in identity_before.get("code_sha256", {}).items():
        if relative in AMENDED_SOURCE_PATHS:
            continue
        if code_digest(ROOT / relative) != expected:
            raise RuntimeError(f"Unapproved source change since the initial freeze: {relative}")
    if not FAILED_ATTEMPT_PATH.is_file():
        raise RuntimeError("Recorded pre-score SVC convergence failure is missing")
    failure = json.loads(FAILED_ATTEMPT_PATH.read_text(encoding="utf-8"))
    if (failure.get("freeze_signature") != parent.get("signature")
            or failure.get("status") != "FIT_FAILED_NO_OUTER_PREDICTIONS"
            or failure.get("outer_prediction_rows_written") != 0):
        raise RuntimeError("Pre-score SVC failure receipt does not match the original freeze")
    return parent


def amendment_identity(data_dir: Path | None = None) -> dict[str, Any]:
    parent = verify_parent_freeze(data_dir)
    train = train_path(data_dir)
    return {
        "parent_freeze_signature": parent["signature"],
        "initial_protocol_sha256": digest_json(json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))),
        "amendment_sha256": digest_json(json.loads(AMENDMENT_PATH.read_text(encoding="utf-8"))),
        "failed_attempt_sha256": file_sha256(FAILED_ATTEMPT_PATH),
        "code_sha256": {name: code_digest(ROOT / name) for name in CODE_FILES},
        "classification_train_sha256": file_sha256(train),
        "round1_control_inputs": {str(end): control_identity(end) for end in EARLY_ENDS},
        "python": platform.python_version(),
        "packages": {name: importlib.metadata.version(name)
                     for name in ["numpy", "pandas", "scipy", "scikit-learn", "threadpoolctl"]},
    }


def freeze_amendment(data_dir: Path | None = None) -> dict[str, Any]:
    current = amendment_identity(data_dir)
    if AMENDMENT_FREEZE_PATH.exists():
        existing = json.loads(AMENDMENT_FREEZE_PATH.read_text(encoding="utf-8"))
        if existing.get("identity") != current:
            raise RuntimeError("Existing amendment freeze differs; preserve it and create a new numbered amendment before fitting.")
        return existing
    parent = verify_parent_freeze(data_dir)
    receipt = {
        "status": "CONVERGENCE_BUDGET_AMENDMENT_FROZEN_BEFORE_SCORE",
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_commit": parent["source_commit"],
        "identity": current,
        "signature": digest_json(current),
        "parent_freeze_signature": parent["signature"],
        "change_summary": "Only DC_SVC LinearSVC max_iter changed from 5000 to 20000 after a failed fit and before any score; all other recipe definitions remain fixed.",
        "hash_convention": "Canonical JSON; repository source normalized to UTF-8 LF; official Train and archived controls hashed byte-for-byte.",
    }
    write_json(AMENDMENT_FREEZE_PATH, receipt)
    return receipt


def verify_amendment(data_dir: Path | None = None) -> dict[str, Any]:
    if not AMENDMENT_FREEZE_PATH.is_file():
        raise RuntimeError("Freeze amendment 01 before any successful campaign fit")
    receipt = json.loads(AMENDMENT_FREEZE_PATH.read_text(encoding="utf-8"))
    current = amendment_identity(data_dir)
    if receipt.get("identity") != current or receipt.get("signature") != digest_json(current):
        raise RuntimeError("Amendment, Train, archived control, code, or runtime changed; refusing fit/cache reuse")
    if receipt.get("parent_freeze_signature") != current.get("parent_freeze_signature"):
        raise RuntimeError("Amendment does not descend from the preserved original freeze")
    return receipt


def load_train(data_dir: Path | None) -> tuple[pd.DataFrame, np.ndarray, Path]:
    path = train_path(data_dir)
    frame = pd.read_csv(path)
    required = {"record_id", "cycle_code", "signal_stream", "artifact_signal"}
    if not required.issubset(frame.columns):
        raise ValueError(f"Classification Train is missing columns: {sorted(required - set(frame.columns))}")
    if frame.record_id.isna().any() or not frame.record_id.is_unique:
        raise ValueError("Train record_id must be present and unique")
    cycle = pd.to_numeric(frame.cycle_code, errors="raise").to_numpy(dtype=int)
    y_raw = pd.to_numeric(frame.artifact_signal, errors="raise").to_numpy(dtype=float)
    if not np.isin(cycle, np.arange(47, 80)).all() or not np.isin(y_raw, [0, 1]).all():
        raise ValueError("Train must contain cycles 47-79 and binary artifact_signal labels only")
    return frame, y_raw.astype(np.int8), path


def _expected_rows(frame: pd.DataFrame, fit_end: int, horizon: int) -> np.ndarray:
    cycle = pd.to_numeric(frame.cycle_code, errors="raise").to_numpy(dtype=int)
    return np.flatnonzero((cycle > fit_end) & (cycle <= horizon))


def load_c1b(frame: pd.DataFrame, fit_end: int) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    cache = ROUND1_CACHE / f"fit{fit_end}__C1_B.npz"
    audit_path = ROUND1_AUDITS / f"fit{fit_end}.json"
    expected_rows = _expected_rows(frame, fit_end, 79 if fit_end == 67 else 67)
    audit_root = json.loads(audit_path.read_text(encoding="utf-8"))
    audit = audit_root.get("C1_B")
    if not isinstance(audit, dict) or audit.get("model") != "C1_B":
        raise ValueError(f"Round 1 C1_B fit audit identity mismatch at fit{fit_end}")
    with np.load(cache, allow_pickle=False) as archive:
        rows = archive["outer_rows"].astype(np.int64, copy=True)
        probability = archive["probability"].astype(float, copy=True)
    if not np.array_equal(rows, expected_rows) or probability.shape != rows.shape:
        raise ValueError(f"Round 1 C1_B cache rows differ from expected fit{fit_end} validation rows")
    if not np.isfinite(probability).all() or ((probability < 0) | (probability > 1)).any():
        raise ValueError(f"Round 1 C1_B probabilities are invalid at fit{fit_end}")
    if array_digest(probability) != audit.get("probability_sha256"):
        raise ValueError(f"Round 1 C1_B probability hash mismatch at fit{fit_end}")
    expected_fit_rows = int(pd.to_numeric(frame.cycle_code).between(47, fit_end).sum())
    if int(audit.get("fit_rows", -1)) != expected_fit_rows:
        raise ValueError(f"Round 1 C1_B fit-row count mismatch at fit{fit_end}")
    return rows, probability, {
        "model": "C1_B",
        "fit_rows": expected_fit_rows,
        "cache_sha256": file_sha256(cache),
        "audit_sha256": file_sha256(audit_path),
        "probability_sha256": array_digest(probability),
    }


def load_c2b(frame: pd.DataFrame, fit_end: int) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    cache = ROUND1_CACHE / f"fit{fit_end}__C2_B.npz"
    audit_path = ROUND1_AUDITS / f"fit{fit_end}.json"
    expected_rows = _expected_rows(frame, fit_end, 79 if fit_end == 67 else 67)
    audit_root = json.loads(audit_path.read_text(encoding="utf-8"))
    audit = audit_root.get("C2_B")
    if not isinstance(audit, dict) or audit.get("model") != "C2_B":
        raise ValueError(f"Round 1 C2_B fit audit identity mismatch at fit{fit_end}")
    with np.load(cache, allow_pickle=False) as archive:
        rows = archive["outer_rows"].astype(np.int64, copy=True)
        probability = archive["probability"].astype(float, copy=True)
    if not np.array_equal(rows, expected_rows) or probability.shape != rows.shape:
        raise ValueError(f"Round 1 C2_B cache rows differ from expected fit{fit_end} validation rows")
    if not np.isfinite(probability).all() or ((probability < 0) | (probability > 1)).any():
        raise ValueError(f"Round 1 C2_B probabilities are invalid at fit{fit_end}")
    if array_digest(probability) != audit.get("probability_sha256"):
        raise ValueError(f"Round 1 C2_B probability hash mismatch at fit{fit_end}")
    expected_fit_rows = int(pd.to_numeric(frame.cycle_code).between(47, fit_end).sum())
    if int(audit.get("fit_rows", -1)) != expected_fit_rows:
        raise ValueError(f"Round 1 C2_B fit-row count mismatch at fit{fit_end}")
    return rows, probability, {
        "model": "C2_B",
        "fit_rows": expected_fit_rows,
        "cache_sha256": file_sha256(cache),
        "audit_sha256": file_sha256(audit_path),
        "probability_sha256": array_digest(probability),
    }


def compact_metrics(y: np.ndarray, probability: np.ndarray) -> dict[str, Any]:
    details = binary_metrics(y, probability, threshold=0.50)
    return {
        "n": details["n"],
        "macro_f1": details["macro_f1"],
        "confusion_matrix": details["confusion_matrix"],
        "per_class": details["per_class"],
        "positive_rate": details["positive_rate"],
        "positive_prediction_rate": details["positive_prediction_rate"],
        "false_positive": details["false_positive"],
        "false_negative": details["false_negative"],
        "threshold": 0.50,
    }


def evaluate_slice(frame: pd.DataFrame, y: np.ndarray, rows: np.ndarray,
                   probability: np.ndarray) -> dict[str, Any]:
    cycle = pd.to_numeric(frame.cycle_code).to_numpy(dtype=int)[rows]
    actual = y[rows]
    result = compact_metrics(actual, probability)
    result["equal_cycle_macro_f1"] = float(np.mean([
        compact_metrics(actual[cycle == value], probability[cycle == value])["macro_f1"]
        for value in sorted(np.unique(cycle))
    ]))
    result["cycles"] = [int(value) for value in sorted(np.unique(cycle))]
    result["per_cycle"] = [
        {"cycle": int(value), **compact_metrics(actual[cycle == value], probability[cycle == value])}
        for value in sorted(np.unique(cycle))
    ]
    return result


def _save_group(frame: pd.DataFrame, y: np.ndarray, fit_end: int,
                frozen: dict[str, Any], data_sha: str) -> dict[str, Any]:
    horizon = 67 if fit_end == 55 else 79
    cycle = pd.to_numeric(frame.cycle_code, errors="raise").to_numpy(dtype=int)
    fit_rows = np.flatnonzero((cycle >= 47) & (cycle <= fit_end))
    outer_rows = _expected_rows(frame, fit_end, horizon)
    features = build_campaign_classifier(CANDIDATES[0]).input_feature_names
    fit_frame = frame.iloc[fit_rows][features].copy()
    y_fit = y[fit_rows]
    X_outer = frame.iloc[outer_rows][features].copy()

    c1_rows, c1_p, c1_meta = load_c1b(frame, fit_end)
    c2_rows, c2_p, c2_meta = load_c2b(frame, fit_end)
    if not np.array_equal(c1_rows, outer_rows) or not np.array_equal(c2_rows, outer_rows):
        raise ValueError(f"Cached control row identity mismatch at fit{fit_end}")
    missing = stream_missing_mask(frame.iloc[outer_rows])
    c4_probability = np.where(missing, c1_p, c2_p)

    fit_group = {
        "fit_end": fit_end,
        "fit_rows": int(len(fit_rows)),
        "fit_cycle_range": [int(cycle[fit_rows].min()), int(cycle[fit_rows].max())],
        "outer_rows": int(len(outer_rows)),
        "outer_cycles": [int(cycle[outer_rows].min()), int(cycle[outer_rows].max())],
        "fit_record_ids_sha256": hashlib.sha256(
            "\n".join(frame.iloc[fit_rows].record_id.astype(str)).encode("utf-8")
        ).hexdigest(),
        "fit_labels_sha256": array_digest(y_fit),
        "fit_features": list(features),
        "outer_labels_used_only_for_evaluation": True,
        "control_provenance": {"C1_B": c1_meta, "C2_B": c2_meta},
        "C4_HGB_probability_sha256": array_digest(c4_probability),
    }

    outputs = {}
    for recipe_id in CANDIDATES:
        cache_path = RUN_DIR / f"fit{fit_end}__{recipe_id}.npz"
        meta_path = RUN_DIR / f"fit{fit_end}__{recipe_id}.json"
        if cache_path.exists() or meta_path.exists():
            raise FileExistsError(f"Campaign output already exists; refuse overwrite: fit{fit_end}/{recipe_id}")
        model = build_campaign_classifier(recipe_id)
        started = perf_counter()
        with threadpool_limits(limits=2):
            model.fit(fit_frame, y_fit)
            fitted_state = model.audit_state()
            candidate_p = np.asarray(model.predict_proba(X_outer), dtype=float)
            diagnostics = model.prediction_diagnostics()
            if recipe_id == "DC_SVC":
                margin = model.predict_margin(X_outer)
                diagnostics = model.prediction_diagnostics()
            else:
                margin = np.empty(0, dtype=float)
        if candidate_p.shape != outer_rows.shape or not np.isfinite(candidate_p).all() or ((candidate_p < 0) | (candidate_p > 1)).any():
            raise RuntimeError(f"{recipe_id} returned invalid fit{fit_end} probabilities")
        if digest_json(fitted_state) != digest_json(model.audit_state()):
            raise RuntimeError(f"{recipe_id} model audit state changed during prediction")
        routed = np.where(missing, c1_p, candidate_p)
        if not np.array_equal(routed[missing], c1_p[missing]):
            raise RuntimeError(f"{recipe_id} changed the frozen C1_B missing-stream route")
        meta = {
            "recipe_id": recipe_id,
            "fit_end": fit_end,
            "fit_rows": int(len(fit_rows)),
            "outer_rows": int(len(outer_rows)),
            "fit_feature_names": list(model.input_feature_names),
            "fit_data_sha256": fitted_state["fit_data_sha256"],
            "fit_state": fitted_state,
            "prediction_diagnostics": diagnostics,
            "threshold": 0.50,
            "decision_rule": "margin >= 0" if recipe_id == "DC_SVC" else "predict_proba >= 0.50",
            "outer_labels_used_for_fit_or_selection": False,
            "runtime_seconds": perf_counter() - started,
            "candidate_probability_sha256": array_digest(candidate_p),
            "routed_probability_sha256": array_digest(routed),
            "missing_route_exact_C1_B_match": True,
            "candidate_margin_sha256": array_digest(margin) if recipe_id == "DC_SVC" else None,
        }
        np.savez_compressed(cache_path, outer_rows=outer_rows.astype(np.int64),
                            candidate_probability=candidate_p,
                            routed_probability=routed,
                            c4_probability=c4_probability,
                            missing_stream_mask=missing.astype(np.uint8),
                            **({"decision_margin": margin} if recipe_id == "DC_SVC" else {}))
        meta["cache_sha256"] = file_sha256(cache_path)
        write_json(meta_path, meta)
        candidate_prob = np.where(missing, c1_p, candidate_p)
        outputs[recipe_id] = {
            "meta": meta,
            "candidate_probability": candidate_p,
            "routed_probability": candidate_prob,
        }
    fit_group["candidates"] = {}
    slices = {
        "primary_W1_or_W4": (56, 59) if fit_end == 55 else (68, 71),
        "H12_E_or_H12_L": (56, 67) if fit_end == 55 else (68, 79),
    }
    for recipe_id, record in outputs.items():
        fit_group["candidates"][recipe_id] = {"fit": record["meta"], "scores": {}}
        for name, (start, end) in slices.items():
            selected = np.flatnonzero((cycle[outer_rows] >= start) & (cycle[outer_rows] <= end))
            selected_rows = outer_rows[selected]
            current = evaluate_slice(frame, y, selected_rows, record["routed_probability"][selected])
            base = evaluate_slice(frame, y, selected_rows, c4_probability[selected])
            fit_group["candidates"][recipe_id]["scores"][name] = {
                "candidate": current,
                "C4_HGB": base,
                "delta_macro_f1": float(current["macro_f1"] - base["macro_f1"]),
                "delta_equal_cycle_macro_f1": float(current["equal_cycle_macro_f1"] - base["equal_cycle_macro_f1"]),
                "stream_present_rows": int((~missing[selected]).sum()),
                "stream_missing_rows": int(missing[selected].sum()),
            }
    fit_group["control_metrics"] = {}
    for name, (start, end) in slices.items():
        selected = np.flatnonzero((cycle[outer_rows] >= start) & (cycle[outer_rows] <= end))
        fit_group["control_metrics"][name] = evaluate_slice(frame, y, outer_rows[selected], c4_probability[selected])
    return fit_group


def plot_stage1(results: dict[str, Any]) -> Path:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    labels = ["DC_SVC", "DC_NBSVM", "DC_WORD_CHAR"]
    colors = ["#4472C4", "#70AD47", "#ED7D31"]
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.3), sharey=True)
    panels = [("55", "H12_E · fit cycles 47–55, validate 56–67", "H12_E_or_H12_L"),
              ("67", "H12_L · fit cycles 47–67, validate 68–79", "H12_E_or_H12_L")]
    for axis, (fit_key, title, score_key) in zip(axes, panels):
        fit = results["fits"][fit_key]
        control = fit["control_metrics"][score_key]["macro_f1"]
        values = [fit["candidates"][recipe]["scores"][score_key]["candidate"]["macro_f1"]
                  for recipe in labels]
        bars = axis.bar(labels, values, color=colors, width=0.66)
        axis.axhline(control, color="#333333", linestyle="--", linewidth=1.3,
                     label=f"C4_HGB {control:.3f}")
        axis.set_title(title, fontsize=10)
        axis.set_ylim(0.55, 0.90)
        axis.grid(axis="y", alpha=0.22)
        axis.tick_params(axis="x", labelrotation=20)
        for bar, value in zip(bars, values):
            axis.text(bar.get_x() + bar.get_width() / 2, value + 0.004,
                      f"{value:.3f}\nΔ {value-control:+.3f}", ha="center", va="bottom", fontsize=8)
        axis.legend(frameon=False, loc="lower right", fontsize=8)
    axes[0].set_ylabel("Macro F1 · fixed 0.50 route")
    fig.suptitle("Stage 1: text-representation candidates against C4_HGB", fontsize=12)
    fig.tight_layout()
    path = REPORT_DIR / "stage1_h12_macro_f1.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path


def run_stage1(data_dir: Path | None = None) -> dict[str, Any]:
    frozen = verify_amendment(data_dir)
    frame, y, train = load_train(data_dir)
    if file_sha256(train) != frozen["identity"]["classification_train_sha256"]:
        raise RuntimeError("Train hash differs from freeze")
    cycle = pd.to_numeric(frame.cycle_code).to_numpy(dtype=int)
    results = {
        "status": "STAGE1_COMPLETE_REVIEW_REQUIRED",
        "protocol_id": json.loads(AMENDMENT_PATH.read_text(encoding="utf-8"))["amendment_id"],
        "parent_protocol_id": json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))["protocol_id"],
        "freeze_signature": frozen["signature"],
        "source_commit": frozen["source_commit"],
        "train_sha256": file_sha256(train),
        "train_rows": int(len(frame)),
        "cycle_range": [int(cycle.min()), int(cycle.max())],
        "target_prevalence": float(y.mean()),
        "test_read": False,
        "kaggle_submission": False,
        "adaptation_warning": "Cycles 56-79 were used by earlier experiments; Stage 1 is development evidence only.",
        "fits": {},
    }
    started = perf_counter()
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    for end in EARLY_ENDS:
        for recipe_id in CANDIDATES:
            expected = (RUN_DIR / f"fit{end}__{recipe_id}.npz", RUN_DIR / f"fit{end}__{recipe_id}.json")
            if any(path.exists() for path in expected):
                raise FileExistsError(f"Stage 1 refuses to overwrite an existing output: {expected}")
        results["fits"][str(end)] = _save_group(frame, y, end, frozen, file_sha256(train))
    results["runtime_seconds"] = perf_counter() - started
    results["candidate_gate"] = {
        recipe: {
            "H12_E_delta": results["fits"]["55"]["candidates"][recipe]["scores"]["H12_E_or_H12_L"]["delta_macro_f1"],
            "H12_L_delta": results["fits"]["67"]["candidates"][recipe]["scores"]["H12_E_or_H12_L"]["delta_macro_f1"],
            "retire_only_if_both_deltas_at_most_minus_0.02": (
                results["fits"]["55"]["candidates"][recipe]["scores"]["H12_E_or_H12_L"]["delta_macro_f1"] <= -0.02
                and results["fits"]["67"]["candidates"][recipe]["scores"]["H12_E_or_H12_L"]["delta_macro_f1"] <= -0.02
            ),
        } for recipe in CANDIDATES
    }
    results["chart_sha256"] = file_sha256(plot_stage1(results))
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    write_json(REPORT_DIR / "stage1_results.json", results)
    rows = []
    for fit_end, fit in results["fits"].items():
        for recipe in CANDIDATES:
            for score_name, score in fit["candidates"][recipe]["scores"].items():
                rows.append({"fit_end": int(fit_end), "validation": score_name, "recipe": recipe,
                             "n": score["candidate"]["n"], "candidate_macro_f1": score["candidate"]["macro_f1"],
                             "C4_macro_f1": score["C4_HGB"]["macro_f1"],
                             "delta_macro_f1": score["delta_macro_f1"],
                             "candidate_equal_cycle_macro_f1": score["candidate"]["equal_cycle_macro_f1"],
                             "C4_equal_cycle_macro_f1": score["C4_HGB"]["equal_cycle_macro_f1"],
                             "delta_equal_cycle_macro_f1": score["delta_equal_cycle_macro_f1"],
                             "stream_present_rows": score["stream_present_rows"],
                             "stream_missing_rows": score["stream_missing_rows"]})
    pd.DataFrame(rows).to_csv(REPORT_DIR / "stage1_comparison.csv", index=False)
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, help="Root containing classification/Train.csv; defaults to TP2_DATA_DIR/repository data.")
    parser.add_argument("--freeze-amendment-only", action="store_true", help="Freeze convergence-budget amendment, code, official Train, and archived-control hashes; do not fit.")
    parser.add_argument("--stage1", action="store_true", help="Run the frozen early diagnostic fits 55 and 67 for all three candidates.")
    args = parser.parse_args()
    if args.freeze_amendment_only == args.stage1:
        parser.error("Choose exactly one of --freeze-amendment-only or --stage1")
    if args.freeze_amendment_only:
        receipt = freeze_amendment(args.data_dir)
        print(receipt["status"], receipt["signature"])
    else:
        result = run_stage1(args.data_dir)
        print(result["status"], "fits", sum(len(fit["candidates"]) for fit in result["fits"].values()),
              "report", (REPORT_DIR / "stage1_results.json"))


if __name__ == "__main__":
    main()
