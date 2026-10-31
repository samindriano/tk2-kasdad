"""Frozen six-window continuation of the V3 Classification OOF residual head A.

This lane trains only four new correction heads (59, 63, 71, 75), reuses the
already verified 55/67 pilot outputs, and compares row-exactly against the
current C2_B+CB_MISSING champion and the independently verified sparse25
challenger. It never reads official Test or Kaggle artifacts.
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
import time
from typing import Any

import numpy as np
import pandas as pd
from scipy.special import expit, logit
from sklearn.metrics import average_precision_score
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import optimization_v3_cls_residual as pilot
from scripts.research_campaign_classification_stage2 import (
    PRIMARY_WINDOWS, _load_round1_control,
)
from src.data.audit import sha256 as file_sha256
from src.models.classification_round2_models import stream_missing_mask
from src.models.optimization_v3_residual import OOFResidualHead
from src.models.round1_evaluation import array_digest, digest_json

CONFIG = ROOT / "experiments/optimization_v3/classification_residual_fullsix"
PROTOCOL = CONFIG / "protocol.json"
FREEZE = CONFIG / "freeze.json"
REPORT_DIR = ROOT / "reports/optimization_v3/classification_residual_fullsix"
RUN_DIR = ROOT / "experiments/runs/optimization_v3_cls_residual_fullsix"
PILOT_CONFIG = ROOT / "experiments/optimization_v3/classification_residual"
PILOT_REPORT = ROOT / "reports/optimization_v3/classification_residual"
PILOT_RUN = ROOT / "experiments/runs/optimization_v3_cls_residual"
SPARSE_CONFIG = ROOT / "experiments/optimization_v3/classification_sparse_full"
SPARSE_PILOT_CONFIG = ROOT / "experiments/optimization_v3/classification_sparse"
SPARSE_REPORT = ROOT / "reports/optimization_v3/classification_sparse_full"
SPARSE_PILOT_REPORT = ROOT / "reports/optimization_v3/classification_sparse"
SPARSE_RUN = ROOT / "experiments/runs/optimization_v3_classification_sparse_full"
SPARSE_PILOT_RUN = ROOT / "experiments/runs/optimization_v3_classification_sparse"
ROUND1 = ROOT / "experiments/runs/classification_round1_v1"
ROUND1_AUDITS = ROOT / "reports/classification_round1/fit_audits"
CB_RUN = ROOT / "experiments/runs/dual_cls_catboost_v1"

ENDS = [55, 59, 63, 67, 71, 75]
NEW_ENDS = [59, 63, 71, 75]
REUSED_ENDS = [55, 67]
THRESHOLD = 0.5
HEAD_NAME = "A_base_structured"
SCORE_NAMES = ("C2_B",)
ALL_FIT_CODES = [
    "scripts/optimization_v3_cls_residual_fullsix.py",
    "scripts/optimization_v3_verify_cls_residual_fullsix.py",
    "tests/test_optimization_v3_cls_residual_fullsix.py",
    "scripts/optimization_v3_cls_residual.py",
    "scripts/optimization_v3_verify_cls_residual_v3.py",
    "src/models/optimization_v3_residual.py",
    "src/models/round1_models.py",
    "src/models/round1_evaluation.py",
    "src/models/classification_models.py",
    "src/models/classification_round2_models.py",
    "scripts/research_campaign_classification_stage2.py",
    "src/data/audit.py",
    "src/utils/paths.py",
]
SPARSE_CODES = [
    "scripts/optimization_v3_sparse_classification.py",
    "scripts/optimization_v3_verify_sparse_classification.py",
    "scripts/optimization_v3_sparse_classification_fullsix.py",
    "scripts/optimization_v3_verify_sparse_classification_fullsix.py",
    "src/models/optimization_v3_sparse_classification.py",
]


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json_new(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as f:
        json.dump(value, f, indent=2, sort_keys=True, allow_nan=False)
        f.write("\n")


def load_train():
    frame, y, cycle, missing, train = pilot.load_train()
    expected = "875552a68a3d24b0f3e1966d6ddb5671e6f54305a1c526e2f0a76747942ed91c"
    if file_sha256(train) != expected:
        raise RuntimeError("Official Classification Train hash differs from the assigned input")
    return frame, y, cycle, missing, train


def sparse_artifact_paths() -> list[Path]:
    paths = [
        SPARSE_CONFIG / "protocol.json", SPARSE_CONFIG / "freeze.json",
        SPARSE_CONFIG / "verification_amendment.json",
        SPARSE_PILOT_CONFIG / "protocol.json", SPARSE_PILOT_CONFIG / "freeze.json",
        SPARSE_PILOT_REPORT / "pilot.json", SPARSE_PILOT_REPORT / "verification.json",
        SPARSE_REPORT / "fullsix.json", SPARSE_REPORT / "verification.json",
        SPARSE_RUN / "fit59.npz", SPARSE_RUN / "fit59.json",
        SPARSE_RUN / "fit63.npz", SPARSE_RUN / "fit63.json",
        SPARSE_RUN / "fit71.npz", SPARSE_RUN / "fit71.json",
        SPARSE_RUN / "fit75.npz", SPARSE_RUN / "fit75.json",
    ]
    for end in REUSED_ENDS:
        paths += [SPARSE_PILOT_RUN / f"fit{end}.npz",
                  SPARSE_PILOT_RUN / f"fit{end}.json"]
    return paths


def upstream_paths(train: Path) -> list[Path]:
    paths = [
        train,
        PILOT_CONFIG / "protocol.json", PILOT_CONFIG / "freeze.json",
        PILOT_CONFIG / "freeze_amendment_01.json",
        PILOT_REPORT / "results.json", PILOT_REPORT / "verification_v3.json",
        PILOT_REPORT / "verification_freeze_v3_amendment_01.json",
        PILOT_REPORT / "verification_amendment_02.json",
        ROOT / "experiments/runs/optimization_v2_classification_calibration/inner51.npz",
        ROOT / "experiments/runs/optimization_v2_classification_calibration/inner51.json",
        ROOT / "experiments/runs/optimization_v2_classification_stack/cb51.npz",
        ROOT / "experiments/runs/optimization_v2_classification_stack/cb51.json",
        ROOT / "experiments/optimization_v2/classification_calibration/freeze.json",
        ROOT / "experiments/optimization_v2/classification_stack/freeze.json",
        ROOT / "experiments/dual_campaign/classification/catboost_freeze.json",
    ]
    for end in ENDS:
        paths += [ROUND1 / f"fit{end}__C2_B.npz", ROUND1_AUDITS / f"fit{end}.json",
                  CB_RUN / f"fit{end}.npz", CB_RUN / f"fit{end}.json"]
    for end in REUSED_ENDS:
        paths += [PILOT_RUN / f"fit{end}_{HEAD_NAME}.npz",
                  PILOT_RUN / f"fit{end}_{HEAD_NAME}.txt"]
    return paths + sparse_artifact_paths()


def identity() -> dict[str, Any]:
    frame, y, cycle, missing, train = load_train()
    sparse_ver = read_json(SPARSE_REPORT / "verification.json")
    sparse_result = read_json(SPARSE_REPORT / "fullsix.json")
    sparse_pilot_ver = read_json(SPARSE_PILOT_REPORT / "verification.json")
    residual_ver = read_json(PILOT_REPORT / "verification_v3.json")
    if sparse_ver.get("status") != "PASS" or sparse_result.get("status") != "READY_FOR_INDEPENDENT_VERIFICATION":
        raise RuntimeError("Sparse challenger must first have its own independent PASS")
    if sparse_pilot_ver.get("status") != "PASS" or residual_ver.get("status") != "PASS":
        raise RuntimeError("Reused pilot outputs must be independently verified PASS")
    source_files = ALL_FIT_CODES + SPARSE_CODES
    missing_sources = [p for p in source_files if not (ROOT / p).is_file()]
    if missing_sources:
        raise FileNotFoundError(f"Frozen source files missing: {missing_sources}")
    inputs = upstream_paths(train)
    missing_inputs = [str(p) for p in inputs if not p.is_file()]
    if missing_inputs:
        raise FileNotFoundError(f"Frozen upstream inputs missing: {missing_inputs}")
    oof = {}
    for end in NEW_ENDS:
        prepared = pilot.prepare_oof(frame, y, cycle, missing, end, HEAD_NAME)
        oof[str(end)] = {
            "input_sha256": prepared["raw_input_sha256"],
            "rows_all": int(len(prepared["all_rows"])),
            "rows_present": int(len(prepared["present_rows"])),
            "block_summary": prepared["block_summary"],
        }
    return {
        "protocol_sha256": file_sha256(PROTOCOL),
        "source_sha256": {p: file_sha256(ROOT / p) for p in source_files},
        "input_sha256": {p.resolve().relative_to(ROOT).as_posix(): file_sha256(p) for p in inputs},
        "official_train_sha256": file_sha256(train),
        "raw_oof_identities": oof,
        "champion_cutoffs": ENDS,
        "new_head_cutoffs": NEW_ENDS,
        "reused_pilot_cutoffs": REUSED_ENDS,
        "root_sparse_freeze_signature": read_json(SPARSE_CONFIG / "freeze.json")["signature"],
        "root_sparse_verification_signature": sparse_ver["freeze_signature"],
        "root_sparse_primary_n": sparse_result["results"]["primary_six_window_pooled"]["n"],
        "root_sparse_h12_n": sparse_result["results"]["H12_fixed_origin_pooled"]["n"],
        "pilot_residual_freeze_signature": read_json(PILOT_CONFIG / "freeze_amendment_01.json")["signature"],
        "pilot_residual_verification_signature": residual_ver["verification_freeze_signature"],
        "python": platform.python_version(),
        "packages": {n: importlib.metadata.version(n) for n in ["numpy", "pandas", "scipy", "scikit-learn", "lightgbm", "threadpoolctl"]},
        "test_read": False,
        "kaggle_access": False,
    }


def freeze_only() -> dict[str, Any]:
    current = identity()
    sig = digest_json(current)
    if FREEZE.exists():
        old = read_json(FREEZE)
        if old.get("identity") != current or old.get("signature") != sig:
            raise RuntimeError("Existing fullsix freeze differs; preserve it and create a reviewed new version")
        return old
    if RUN_DIR.exists() and any(RUN_DIR.iterdir()):
        raise FileExistsError("Fullsix run cache is nonempty before freeze")
    receipt = {"status": "FROZEN_BEFORE_FOUR_HEAD_A_FITS", "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
               "identity": current, "signature": sig,
               "source_boundary": "Official Classification Train and frozen OOF/outer probability caches only; no Test or Kaggle access.",
               "source_commit_context": subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip()}
    write_json_new(FREEZE, receipt)
    return receipt


def check_freeze() -> dict[str, Any]:
    if not FREEZE.is_file():
        raise RuntimeError("Freeze exact source/data/cache hashes with --freeze-only before any head fit")
    frozen = read_json(FREEZE)
    current = identity()
    if current != frozen.get("identity") or digest_json(current) != frozen.get("signature"):
        raise RuntimeError("Fullsix source, Train, OOF or comparator identity changed after freeze")
    return frozen


def control_window(frame: pd.DataFrame, cycle: np.ndarray, missing: np.ndarray,
                   cutoff: int) -> dict[str, Any]:
    hi = cutoff + 4
    rows, base, base_identity = _load_round1_control(frame, cutoff, hi, "C2_B")
    expected = np.flatnonzero((cycle > cutoff) & (cycle <= hi))
    if not np.array_equal(rows, expected):
        raise AssertionError(f"C2_B outer rows differ at cutoff {cutoff}")
    path = CB_RUN / f"fit{cutoff}.npz"
    meta = read_json(path.with_suffix(".json"))
    if meta.get("npz_sha256") != file_sha256(path):
        raise AssertionError(f"CatBoost outer cache hash mismatch at cutoff {cutoff}")
    with np.load(path, allow_pickle=False) as z:
        cb_rows = z["rows"].astype(np.int64, copy=True)
        cb = z["candidate"].astype(float, copy=True)
    if not np.array_equal(cb_rows, rows) or meta.get("prediction_sha256") != array_digest(cb):
        raise AssertionError(f"CB_MISSING cache row/prediction identity mismatch at cutoff {cutoff}")
    m = missing[rows]
    champion = np.where(m, cb, base)
    sparse = load_sparse_prediction(cutoff, rows, champion)
    return {"rows": rows, "base": base, "cb": cb, "champion": champion,
            "missing": m, "sparse25": sparse,
            "cache_hashes": {"C2_B": base_identity,
                             "CB_MISSING_npz": file_sha256(path),
                             "CB_MISSING_json": file_sha256(path.with_suffix(".json"))}}


def load_sparse_prediction(cutoff: int, rows: np.ndarray, champion: np.ndarray) -> np.ndarray:
    root = SPARSE_PILOT_RUN if cutoff in REUSED_ENDS else SPARSE_RUN
    path = root / f"fit{cutoff}.npz"
    with np.load(path, allow_pickle=False) as z:
        srows = z["rows"].astype(np.int64, copy=True)
        blend = z["blend25"].astype(float, copy=True)
        current = z["champion"].astype(float, copy=True)
    if not np.array_equal(srows, rows) or not np.array_equal(current, champion):
        raise AssertionError(f"Sparse challenger is not row-exact with champion at {cutoff}")
    if not np.isfinite(blend).all() or np.any((blend < 0) | (blend > 1)):
        raise AssertionError(f"Sparse challenger probabilities invalid at {cutoff}")
    return blend


def reuse_pilot(cutoff: int) -> dict[str, Any]:
    path = PILOT_RUN / f"fit{cutoff}_{HEAD_NAME}.npz"
    report = read_json(PILOT_REPORT / "results.json")
    meta = report["head_metadata"][f"{cutoff}/{HEAD_NAME}"]
    if file_sha256(path) != meta["prediction_sha256"]:
        raise AssertionError(f"Pilot prediction changed at cutoff {cutoff}")
    with np.load(path, allow_pickle=False) as z:
        rows = z["outer_rows"].astype(np.int64, copy=True)
        miss = z["missing_mask"].astype(bool, copy=True)
        champion = z["champion_probability"].astype(float, copy=True)
        candidate = z["candidate_probability"].astype(float, copy=True)
        base = z["base_probability"].astype(float, copy=True)
        cb = z["cb_probability"].astype(float, copy=True)
    if rows.size != np.unique(rows).size or not np.array_equal(candidate[miss], champion[miss]):
        raise AssertionError(f"Pilot reuse rows/route invalid at cutoff {cutoff}")
    sparse = load_sparse_prediction(cutoff, rows, champion)
    return {"rows": rows, "missing": miss, "champion": champion,
            "candidate": candidate, "base": base, "cb": cb,
            "sparse25": sparse, "pilot_cache_sha256": file_sha256(path),
            "head_model_sha256": meta["model_sha256"]}


def train_new(frame: pd.DataFrame, y: np.ndarray, cycle: np.ndarray,
              missing: np.ndarray, cutoff: int, freeze_signature: str) -> dict[str, Any]:
    fit_data = pilot.prepare_oof(frame, y, cycle, missing, cutoff, HEAD_NAME)
    head = OOFResidualHead("classification", SCORE_NAMES)
    with threadpool_limits(limits=2):
        head.fit(fit_data["X"], fit_data["scores"], fit_data["y"])
    before = head.audit_state()
    outer = control_window(frame, cycle, missing, cutoff)
    rows = outer["rows"]
    present = ~outer["missing"]
    X_outer = frame.iloc[rows[present]][list(pilot.EXPANDED_FEATURES)].copy()
    scores = outer["base"][present, None]
    design, _, _ = head._design(X_outer, scores, False)
    raw = head.model.booster_.predict(design, raw_score=True, num_threads=2)
    with np.errstate(divide="raise", invalid="raise"):
        corrected_present = expit(logit(np.clip(outer["base"][present], 1e-6, 1 - 1e-6)) + 0.5 * raw)
    candidate = outer["champion"].copy()
    candidate[present] = corrected_present
    if not np.isfinite(candidate).all() or np.any((candidate < 0) | (candidate > 1)):
        raise AssertionError("Residual probabilities invalid")
    if not np.array_equal(candidate[outer["missing"]], outer["champion"][outer["missing"]]):
        raise AssertionError("Missing-stream routing changed")
    if head.audit_state() != before:
        raise AssertionError("Residual head state changed during outer prediction")
    pred_path = RUN_DIR / f"fit{cutoff}_{HEAD_NAME}.npz"
    model_path = RUN_DIR / f"fit{cutoff}_{HEAD_NAME}.txt"
    meta_path = RUN_DIR / f"fit{cutoff}_{HEAD_NAME}.json"
    for p in (pred_path, model_path, meta_path):
        if p.exists():
            raise FileExistsError(f"Refusing to overwrite fullsix artifact: {p}")
    model_text = head.model.booster_.model_to_string()
    np.savez_compressed(pred_path, outer_rows=rows.astype(np.int64),
                        missing_mask=outer["missing"].astype(np.uint8),
                        base_probability=outer["base"], cb_probability=outer["cb"],
                        champion_probability=outer["champion"],
                        sparse25_probability=outer["sparse25"],
                        candidate_probability=candidate,
                        raw_correction_present=raw,
                        outer_design_present=design.to_numpy(dtype=float),
                        oof_present_rows=fit_data["present_rows"].astype(np.int64),
                        oof_scores=fit_data["scores"].astype(float),
                        oof_target=fit_data["y"].astype(np.int8))
    model_path.write_text(model_text, encoding="utf-8")
    meta = {"freeze_signature": freeze_signature, "cutoff": cutoff,
            "outer_cycles": [cutoff + 1, cutoff + 4],
            "oof_raw_input_sha256": fit_data["raw_input_sha256"],
            "oof_rows_all": len(fit_data["all_rows"]),
            "oof_rows_present": len(fit_data["present_rows"]),
            "oof_blocks": fit_data["block_summary"], "state": before,
            "outer_rows": len(rows), "outer_present_rows": int(present.sum()),
            "outer_missing_rows": int((~present).sum()),
            "outer_cache_hashes": outer["cache_hashes"],
            "model_path": str(model_path.relative_to(ROOT)),
            "model_sha256": file_sha256(model_path),
            "prediction_path": str(pred_path.relative_to(ROOT)),
            "prediction_sha256": file_sha256(pred_path),
            "outer_design_sha256": array_digest(design.to_numpy(dtype=float)),
            "raw_correction_sha256": array_digest(raw),
            "candidate_probability_sha256": array_digest(candidate),
            "formula": "expit(logit(clip(C2_B,1e-6,1-1e-6)) + 0.5 * raw_correction)",
            "missing_route_exact_match": True, "eval_set_used": False,
            "outer_labels_used_for_fit": False,
            "seconds": None}
    write_json_new(meta_path, meta)
    return {"rows": rows, "missing": outer["missing"],
            "champion": outer["champion"], "candidate": candidate,
            "sparse25": outer["sparse25"], "base": outer["base"],
            "cb": outer["cb"], "metadata": meta, "cache_sha256": file_sha256(pred_path)}


def metric(y: np.ndarray, p: np.ndarray) -> dict[str, Any]:
    yy = np.asarray(y, dtype=np.int8)
    pp = np.asarray(p, dtype=float)
    if yy.ndim != 1 or pp.shape != yy.shape or not len(yy) or not np.isfinite(pp).all() or ((pp < 0) | (pp > 1)).any():
        raise ValueError("Metric vectors must be nonempty, aligned, finite probabilities")
    pred = pp >= THRESHOLD
    tp = int(np.sum(pred & (yy == 1))); fp = int(np.sum(pred & (yy == 0)))
    tn = int(np.sum(~pred & (yy == 0))); fn = int(np.sum(~pred & (yy == 1)))
    p1 = tp / (tp + fp) if tp + fp else 0.0
    r1 = tp / (tp + fn) if tp + fn else 0.0
    p0 = tn / (tn + fn) if tn + fn else 0.0
    r0 = tn / (tn + fp) if tn + fp else 0.0
    f1 = 2 * p1 * r1 / (p1 + r1) if p1 + r1 else 0.0
    f0 = 2 * p0 * r0 / (p0 + r0) if p0 + r0 else 0.0
    return {"n": int(len(yy)), "macro_f1": float((f0 + f1) / 2),
            "f1_negative": float(f0), "f1_positive": float(f1),
            "positive_prevalence": float(np.mean(yy)),
            "positive_prediction_rate": float(np.mean(pred)),
            "tp": tp, "tn": tn, "fp": fp, "fn": fn,
            "error_count": fp + fn,
            "average_precision": float(average_precision_score(yy, pp)) if np.unique(yy).size == 2 else None,
            "brier": float(np.mean((pp - yy) ** 2)), "threshold": THRESHOLD}


def train_edges(frame: pd.DataFrame, cutoff: int, feature: str) -> list[float]:
    cycles = pd.to_numeric(frame.cycle_code, errors="raise").to_numpy(dtype=int)
    values = pd.to_numeric(frame.loc[(cycles >= 47) & (cycles <= cutoff), feature], errors="coerce").to_numpy(dtype=float)
    values = values[np.isfinite(values)]
    edges = np.quantile(values, [0.25, 0.5, 0.75]).astype(float).tolist()
    if len(values) == 0 or not np.isfinite(edges).all() or not edges[0] < edges[1] < edges[2]:
        raise ValueError(f"Fit-only quartile edges for {feature} are not strictly increasing at cutoff {cutoff}")
    return edges


def evaluate_vectors(frame: pd.DataFrame, y: np.ndarray, cycle: np.ndarray,
                     rows: np.ndarray, missing: np.ndarray,
                     vectors: dict[str, np.ndarray], cutoff_by_row: np.ndarray) -> dict[str, Any]:
    yy = y[rows]
    result: dict[str, Any] = {"n": int(len(rows)), "models": {}, "subgroups": {}, "paired_errors": {}}
    for name, p in vectors.items():
        if len(p) != len(rows):
            raise AssertionError("Evaluation vector row count mismatch")
        parts = {"all": np.ones(len(rows), dtype=bool), "stream_present": ~missing,
                 "stream_missing": missing}
        model = {k: metric(yy[m], p[m]) for k, m in parts.items() if m.any()}
        model["by_cycle"] = {str(int(c)): metric(yy[cycle[rows] == c], p[cycle[rows] == c])
                             for c in sorted(np.unique(cycle[rows]))}
        model["mean_cycle_macro_f1"] = float(np.mean([m["macro_f1"] for m in model["by_cycle"].values()]))
        result["models"][name] = model
    base_pred = vectors["champion"] >= THRESHOLD
    for name in ("residual_A", "sparse25"):
        candidate_pred = vectors[name] >= THRESHOLD
        base_err = base_pred != yy
        cand_err = candidate_pred != yy
        result["paired_errors"][name] = {
            "champion_errors_fixed": int(np.sum(base_err & ~cand_err)),
            "champion_correct_broken": int(np.sum(~base_err & cand_err)),
            "candidate_errors": int(cand_err.sum()),
            "champion_errors": int(base_err.sum()),
            "prediction_disagreements": int(np.sum(base_pred != candidate_pred)),
        }
    cycles_here = cycle[rows]
    for feature in ("entry_friction", "text_load"):
        bins = np.full(len(rows), -1, dtype=int)
        edge_record = {}
        for cutoff in sorted(np.unique(cutoff_by_row)):
            take = cutoff_by_row == cutoff
            edges = train_edges(frame, int(cutoff), feature)
            vals = pd.to_numeric(frame.iloc[rows[take]][feature], errors="coerce").to_numpy(dtype=float)
            valid = np.isfinite(vals)
            bins_local = np.digitize(vals, edges, right=True)
            loc = np.flatnonzero(take)
            bins[loc[valid]] = bins_local[valid]
            edge_record[str(int(cutoff))] = edges
        groups = {}
        labels = ["Q1_low", "Q2", "Q3", "Q4_high"]
        for idx, label in enumerate(labels):
            mask = bins == idx
            groups[label] = {"n": int(mask.sum()),
                             "models": {name: metric(yy[mask], p[mask]) for name, p in vectors.items()} if mask.any() else None}
        groups["missing_feature"] = {"n": int(np.sum(bins < 0)),
                                     "models": {name: metric(yy[bins < 0], p[bins < 0]) for name, p in vectors.items()} if np.any(bins < 0) else None}
        result["subgroups"][feature] = {"fit_only_quartile_edges_by_cutoff": edge_record, "groups": groups}
    result["per_class_metrics"] = {name: {"class_0_f1": item["all"]["f1_negative"],
                                          "class_1_f1": item["all"]["f1_positive"],
                                          "class_0_recall": item["all"]["tn"] / max(1, item["all"]["tn"] + item["all"]["fp"]),
                                          "class_1_recall": item["all"]["tp"] / max(1, item["all"]["tp"] + item["all"]["fn"])}
                                 for name, item in result["models"].items()}
    return result


def aggregate(frame: pd.DataFrame, y: np.ndarray, cycle: np.ndarray,
              records: dict[int, dict[str, Any]]) -> dict[str, Any]:
    slices: dict[str, list[tuple[int, int, int]]] = {
        name: [(end, lo, hi)] for name, (end, lo, hi) in PRIMARY_WINDOWS.items()
    }
    slices["primary_six_window_pooled"] = [(end, lo, hi) for end, lo, hi in PRIMARY_WINDOWS.values()]
    slices["H12_E"] = [(55, 56, 67)]; slices["H12_L"] = [(67, 68, 79)]
    slices["H12_fixed_origin_pooled"] = slices["H12_E"] + slices["H12_L"]
    out: dict[str, Any] = {}
    for view, parts in slices.items():
        rows_list, missing_list, cutoff_list = [], [], []
        model_lists = {name: [] for name in ("champion", "residual_A", "sparse25")}
        for end, lo, hi in parts:
            rec = records[end]
            keep = (cycle[rec["rows"]] >= lo) & (cycle[rec["rows"]] <= hi)
            rows = rec["rows"][keep]
            rows_list.append(rows); missing_list.append(rec["missing"][keep])
            cutoff_list.append(np.full(len(rows), end, dtype=np.int16))
            for name, key in (("champion", "champion"), ("residual_A", "candidate"), ("sparse25", "sparse25")):
                model_lists[name].append(rec[key][keep])
        rows = np.concatenate(rows_list)
        if rows.size != np.unique(rows).size:
            raise AssertionError(f"Evaluation view {view} contains duplicate rows")
        cutoff_rows = np.concatenate(cutoff_list)
        miss = np.concatenate(missing_list).astype(bool)
        vectors = {k: np.concatenate(v) for k, v in model_lists.items()}
        out[view] = evaluate_vectors(frame, y, cycle, rows, miss, vectors, cutoff_rows)
        out[view]["row_indices_sha256"] = array_digest(rows)
        out[view]["row_ids_sha256"] = array_digest(pd.to_numeric(frame.iloc[rows].record_id, errors="raise").to_numpy())
        # Candidate minus champion deltas and candidate-vs-sparse disagreements.
        out[view]["delta_vs_champion"] = {
            name: {metric_name: out[view]["models"][name]["all"][metric_name]
                   - out[view]["models"]["champion"]["all"][metric_name]
                   for metric_name in ("macro_f1", "average_precision", "brier")}
            for name in ("residual_A", "sparse25")}
        out[view]["residual_vs_sparse"] = {
            "probability_mae": float(np.mean(np.abs(vectors["residual_A"] - vectors["sparse25"]))),
            "probability_correlation": float(np.corrcoef(vectors["residual_A"], vectors["sparse25"])[0, 1]),
            "prediction_disagreement_n": int(np.sum((vectors["residual_A"] >= 0.5) != (vectors["sparse25"] >= 0.5))),
        }
        # Explicit, non-tuned complementarity counts against the same label vector.
        ea = (vectors["residual_A"] >= 0.5) != y[rows]
        es = (vectors["sparse25"] >= 0.5) != y[rows]
        out[view]["residual_vs_sparse"]["residual_only_correct_sparse_wrong"] = int(np.sum(~ea & es))
        out[view]["residual_vs_sparse"]["sparse_only_correct_residual_wrong"] = int(np.sum(ea & ~es))
    return out


def report_markdown(result: dict[str, Any]) -> str:
    r = result["evaluation"]
    lines = [
        "# V3 Classification — OOF residual Head A, six-window continuation", "",
        "## Frozen scope", "",
        "Only Head A (C2_B probability plus expanded structured covariates) was continued. Cutoffs 59, 63, 71, and 75 are four newly fitted heads; already verified cutoffs 55 and 67 are reused byte-for-byte. Head B was not fitted again. The comparison is against the current C2_B-present / CB_MISSING-missing champion and the separately verified fixed sparse25 challenger. Threshold and head shrinkage remain 0.50; no weight, route, threshold, or feature selection used these outer labels.", "",
        "The correction heads train only on stream-present OOF rows from cycles 52 through each cutoff, using exact nonoverlapping four-cycle blocks scored by a base fit that ends before those rows. The missing-stream branch keeps the cached CatBoost prediction exactly. Official Test and Kaggle were not read.", "",
        "## Primary and fixed-origin comparison", "",
        "| View | Rows | Champion F1 | Residual A F1 (Δ) | Sparse25 F1 (Δ) | Residual A Brier | Sparse25 Brier |", "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for view in ["W1", "W2", "W3", "W4", "W5", "W6", "primary_six_window_pooled", "H12_E", "H12_L", "H12_fixed_origin_pooled"]:
        d = r[view]
        ch = d["models"]["champion"]["all"]["macro_f1"]
        a = d["models"]["residual_A"]["all"]["macro_f1"]
        s = d["models"]["sparse25"]["all"]["macro_f1"]
        lines.append(f"| {view} | {d['n']} | {ch:.6f} | {a:.6f} ({a-ch:+.6f}) | {s:.6f} ({s-ch:+.6f}) | {d['models']['residual_A']['all']['brier']:.6f} | {d['models']['sparse25']['all']['brier']:.6f} |")
    pooled = r["primary_six_window_pooled"]; h12 = r["H12_fixed_origin_pooled"]
    lines += ["", "## Error movement and subgroup robustness", "",
              "Each view reports per-class precision-relevant counts, per-cycle metrics, stream-present/missing metrics, fit-only entry_friction and text_load quartiles, and paired error transitions. Quartile edges are computed separately at each prediction cutoff using feature values from cycles 47 through that cutoff; labels are not used to define subgroup boundaries.", "",
              f"In the pooled primary view, residual A fixes {pooled['paired_errors']['residual_A']['champion_errors_fixed']:,} champion errors and breaks {pooled['paired_errors']['residual_A']['champion_correct_broken']:,} correct champion predictions. Sparse25 fixes {pooled['paired_errors']['sparse25']['champion_errors_fixed']:,} and breaks {pooled['paired_errors']['sparse25']['champion_correct_broken']:,}. Residual A and sparse25 disagree on {pooled['residual_vs_sparse']['prediction_disagreement_n']:,} pooled-primary predictions.", "",
              f"H12 uses fixed pilot origins 55 (cycles 56–67) and 67 (cycles 68–79); it overlaps the six-window validation chronology. It is a stability view, not an independent holdout. Primary pooled coverage is {pooled['n']:,}; H12 is {h12['n']:,}.", "",
              "## Interpretation limits", "",
              "These temporal validation outcomes are development evidence and overlap prior campaign work. A six-window or H12 gain does not establish generalization to the official Test. Subgroup contrasts are descriptive and can have small support. The residual head is a prespecified fixed recipe, not a tuned final model. No final route or model replacement is selected in this report.", ""]
    return "\n".join(lines)


def run() -> dict[str, Any]:
    frozen = check_freeze()
    frame, y, cycle, missing, _ = load_train()
    records: dict[int, dict[str, Any]] = {}
    for end in ENDS:
        if end in REUSED_ENDS:
            rec = reuse_pilot(end)
            # Fold-level comparator row binding uses the original pilot horizons.
            records[end] = rec
        else:
            records[end] = train_new(frame, y, cycle, missing, end, frozen["signature"])
    evaluation = aggregate(frame, y, cycle, records)
    result = {
        "status": "READY_FOR_INDEPENDENT_VERIFICATION",
        "freeze_signature": frozen["signature"],
        "freeze_identity": frozen["identity"],
        "new_fits": len(NEW_ENDS), "new_fit_cutoffs": NEW_ENDS,
        "reused_cutoffs": REUSED_ENDS, "head": HEAD_NAME,
        "head_recipe": {"utility": "src.models.optimization_v3_residual.OOFResidualHead",
                        "objective": "binary LightGBM raw-margin correction",
                        "params": {"n_estimators": 200, "num_leaves": 7,
                                   "min_child_samples": 80, "reg_lambda": 10.0,
                                   "learning_rate": 0.03, "random_state": 42,
                                   "n_jobs": 2},
                        "initial_margin": "logit(C2_B OOF probability)",
                        "shrinkage": 0.5, "threshold": 0.5},
        "test_read": False, "kaggle_access": False,
        "outer_labels_used_for_fit_or_selection": False,
        "missing_route_unchanged": True,
        "fit_metadata": {str(end): (records[end]["metadata"] if end in NEW_ENDS else {
            "prediction_cache_sha256": records[end]["pilot_cache_sha256"],
            "model_sha256": records[end]["head_model_sha256"], "reused": True}) for end in ENDS},
        "evaluation": evaluation,
        "interpretation_limits": [
            "Primary six-window and H12 metrics are development evidence, not an untouched holdout.",
            "H12 fixed-origin views overlap the chronological region represented by the primary windows.",
            "Entry-friction and text-load cut points use feature values from each model's training period only.",
            "No model, threshold, or route is selected from these comparisons in this lane.",
        ],
    }
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    result_path = REPORT_DIR / "fullsix.json"
    if result_path.exists() or (REPORT_DIR / "FULLSIX.md").exists():
        raise FileExistsError("Refusing to overwrite existing fullsix report")
    write_json_new(result_path, result)
    (REPORT_DIR / "FULLSIX.md").write_text(report_markdown(result), encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--freeze-only", action="store_true")
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()
    if args.freeze_only == args.run:
        parser.error("Choose exactly one of --freeze-only or --run")
    if args.freeze_only:
        x = freeze_only()
        print(json.dumps({"status": x["status"], "signature": x["signature"],
                          "new_head_cutoffs": x["identity"]["new_head_cutoffs"],
                          "oof_rows_present": {k: v["rows_present"] for k, v in x["identity"]["raw_oof_identities"].items()}}, indent=2))
    else:
        x = run()
        print(json.dumps({"status": x["status"], "new_fits": x["new_fits"],
                          "primary": x["evaluation"]["primary_six_window_pooled"]["delta_vs_champion"],
                          "h12": x["evaluation"]["H12_fixed_origin_pooled"]["delta_vs_champion"]}, indent=2))


if __name__ == "__main__":
    main()
