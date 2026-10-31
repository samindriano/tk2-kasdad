"""Frozen two-head Classification OOF residual pilot; never reads Test/Kaggle."""
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
from typing import Any

import numpy as np
import pandas as pd
from scipy.special import expit, logit

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.audit import sha256 as file_sha256
from src.models.classification_models import _fit_fingerprint
from src.models.classification_round2_models import stream_missing_mask
from src.models.optimization_v3_residual import OOFResidualHead
from src.models.round1_evaluation import array_digest, digest_json
from src.models.round1_models import EXPANDED_FEATURES
from src.utils.paths import task_files

CONFIG = ROOT / "experiments/optimization_v3/classification_residual"
PROTOCOL_PATH = CONFIG / "protocol.json"
ORIGINAL_FREEZE_PATH = CONFIG / "freeze.json"
AMENDMENT_PATH = CONFIG / "amendment_01.json"
FREEZE_PATH = CONFIG / "freeze_amendment_01.json"
REPORT_DIR = ROOT / "reports/optimization_v3/classification_residual"
RUN_DIR = ROOT / "experiments/runs/optimization_v3_cls_residual"
ROUND1 = ROOT / "experiments/runs/classification_round1_v1"
ROUND1_AUDITS = ROOT / "reports/classification_round1/fit_audits"
CB_RUNS = ROOT / "experiments/runs/dual_cls_catboost_v1"
V2_CAL_RUNS = ROOT / "experiments/runs/optimization_v2_classification_calibration"
V2_STACK_RUNS = ROOT / "experiments/runs/optimization_v2_classification_stack"
V2_CAL_FREEZE = ROOT / "experiments/optimization_v2/classification_calibration/freeze.json"
V2_STACK_FREEZE = ROOT / "experiments/optimization_v2/classification_stack/freeze.json"
CB_FREEZE = ROOT / "experiments/dual_campaign/classification/catboost_freeze.json"

OUTER = {55: {"horizon": 67, "primary": [56, 59], "h12": "H12_E"},
         67: {"horizon": 79, "primary": [68, 71], "h12": "H12_L"}}
HEADS = {"A_base_structured": ("C2_B",),
         "B_base_structured_plus_CB_ablation": ("C2_B", "CB_MISSING")}
BASE_BLOCKS = {55: (51, 52, 55), 59: (55, 56, 59), 63: (59, 60, 63), 67: (63, 64, 67)}
CODE_FILES = [
    "scripts/optimization_v3_cls_residual.py",
    "scripts/optimization_v3_verify_cls_residual.py",
    "tests/test_optimization_v3_cls_residual.py",
    "src/models/optimization_v3_residual.py",
    "src/models/round1_models.py",
    "src/models/round1_evaluation.py",
    "src/models/regression_round2_models.py",
    "src/models/classification_models.py",
    "src/models/classification_round2_models.py",
    "src/models/campaign_classification_catboost.py",
    "src/utils/paths.py",
    "src/data/audit.py",
]


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def path_hashes(paths: list[Path]) -> dict[str, str]:
    result = {}
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(f"Frozen input missing: {path}")
        result[str(path.resolve().relative_to(ROOT))] = file_sha256(path)
    return result


def train_file() -> Path:
    path = task_files("classification")["train"]
    if path is None:
        raise FileNotFoundError("Official Classification Train.csv not found")
    return path.resolve()


def load_train() -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray, Path]:
    path = train_file()
    frame = pd.read_csv(path)
    if not set(EXPANDED_FEATURES).issubset(frame.columns) or not {"cycle_code", "artifact_signal", "signal_stream"}.issubset(frame.columns):
        raise ValueError("Official Train schema does not match the frozen residual protocol")
    cycle = pd.to_numeric(frame["cycle_code"], errors="raise").to_numpy(dtype=int)
    y = pd.to_numeric(frame["artifact_signal"], errors="raise").to_numpy(dtype=np.int8)
    if not np.isin(y, [0, 1]).all() or len(frame) != 50608 or (cycle.min(), cycle.max()) != (47, 79):
        raise ValueError("Official Train row count, target, or cycle span changed")
    missing = stream_missing_mask(frame).astype(bool)
    return frame, y, cycle, missing, path


def expected_rows(cycle: np.ndarray, lo: int, hi: int) -> np.ndarray:
    return np.flatnonzero((cycle >= lo) & (cycle <= hi))


def validate_oof_blocks(blocks: list[dict[str, Any]], cycle: np.ndarray,
                        outer_cutoff: int) -> np.ndarray:
    """Reject nonconsecutive, duplicated, shifted, or future OOF rows."""
    expected_cutoffs = list(range(51, outer_cutoff, 4))
    if [int(b["fit_cutoff"]) for b in blocks] != expected_cutoffs:
        raise ValueError("OOF blocks do not use the exact frozen sequence of base-fit cutoffs")
    all_rows: list[np.ndarray] = []
    for block in blocks:
        cutoff = int(block["fit_cutoff"])
        rows = np.asarray(block["rows"], dtype=np.int64)
        if rows.ndim != 1 or rows.size == 0 or (rows < 0).any() or (rows >= cycle.size).any():
            raise ValueError("Invalid OOF block row indices")
        expected = expected_rows(cycle, cutoff + 1, cutoff + 4)
        if not np.array_equal(rows, expected):
            raise ValueError(f"OOF rows for cutoff {cutoff} are not exactly its next four cycles")
        if cutoff + 4 > outer_cutoff:
            raise ValueError("An OOF base fit or OOF block reaches beyond the correction-head cutoff")
        all_rows.append(rows)
    joined = np.concatenate(all_rows)
    if np.unique(joined).size != joined.size:
        raise ValueError("OOF blocks contain duplicate row IDs")
    expected = expected_rows(cycle, 52, outer_cutoff)
    if not np.array_equal(np.sort(joined), expected):
        raise ValueError("OOF blocks do not exactly cover cycles 52 through the outer cutoff")
    return joined


def training_input_hash(rows: np.ndarray, X: pd.DataFrame, y: np.ndarray,
                        scores: np.ndarray) -> str:
    h = hashlib.sha256()
    h.update(json.dumps({"rows": len(rows), "features": list(X.columns),
                         "score_shape": list(scores.shape)}, sort_keys=True).encode())
    h.update(np.asarray(rows, dtype="<i8").tobytes())
    h.update(pd.util.hash_pandas_object(X.reset_index(drop=True), index=False).to_numpy(dtype="<u8").tobytes())
    h.update(array_digest(y).encode())
    h.update(array_digest(scores).encode())
    return h.hexdigest()


def all_input_paths(train: Path) -> list[Path]:
    paths = [train, PROTOCOL_PATH, V2_CAL_FREEZE, V2_STACK_FREEZE, CB_FREEZE]
    paths += [ROUND1 / f"fit{end}__C2_B.npz" for end in (55, 59, 63, 67)]
    paths += [ROUND1_AUDITS / f"fit{end}.json" for end in (55, 59, 63, 67)]
    paths += [CB_RUNS / f"fit{end}.{suffix}" for end in (55, 59, 63, 67) for suffix in ("npz", "json")]
    paths += [V2_CAL_RUNS / f"inner51.{suffix}" for suffix in ("npz", "json")]
    paths += [V2_STACK_RUNS / f"cb51.{suffix}" for suffix in ("npz", "json")]
    return paths


def _verify_fit_fingerprint(frame: pd.DataFrame, y: np.ndarray, cycle: np.ndarray,
                            cutoff: int, features: list[str], state: dict[str, Any],
                            expected_rows_count: int) -> None:
    fit_rows = expected_rows(cycle, 47, cutoff)
    X = frame.iloc[fit_rows][features]
    fit_y = y[fit_rows]
    if len(fit_rows) != expected_rows_count or state.get("fit_rows") != len(fit_rows):
        raise ValueError(f"Cached model fit support mismatch at cutoff {cutoff}")
    if state.get("fit_data_sha256") != _fit_fingerprint(X, tuple(features), fit_y):
        raise ValueError(f"Cached model training fingerprint mismatch at cutoff {cutoff}")


def load_block(frame: pd.DataFrame, y: np.ndarray, cycle: np.ndarray,
               fit_cutoff: int) -> dict[str, Any]:
    lo, hi = fit_cutoff + 1, fit_cutoff + 4
    rows = expected_rows(cycle, lo, hi)
    if fit_cutoff == 51:
        with np.load(V2_CAL_RUNS / "inner51.npz", allow_pickle=False) as z:
            base_rows = z["rows"].astype(np.int64, copy=True)
            base_scores = z["scores"].astype(float, copy=True)[:, 0]
        cal_meta = read_json(V2_CAL_RUNS / "inner51.json")
        cal_freeze = read_json(V2_CAL_FREEZE)
        if cal_meta.get("freeze_signature") != cal_freeze.get("signature") or cal_meta.get("npz_sha256") != file_sha256(V2_CAL_RUNS / "inner51.npz"):
            raise ValueError("V2 C2_B fit51 cache no longer matches its frozen receipt")
        states = cal_meta.get("states", [])
        if not states or states[0].get("id") != "C2_B":
            raise ValueError("V2 inner51 score column zero is not the archived C2_B model")
        expected_fit_n = int(np.count_nonzero((cycle >= 47) & (cycle <= 51)))
        _verify_fit_fingerprint(frame, y, cycle, 51, states[0]["input_feature_names"], states[0], expected_fit_n)
        with np.load(V2_STACK_RUNS / "cb51.npz", allow_pickle=False) as z:
            cb_rows = z["rows"].astype(np.int64, copy=True)
            cb = z["probability"].astype(float, copy=True)
        cb_meta = read_json(V2_STACK_RUNS / "cb51.json")
        cb_freeze = read_json(V2_STACK_FREEZE)
        cb_state = cb_meta.get("state", {})
        if cb_meta.get("freeze_signature") != cb_freeze.get("signature") or cb_meta.get("npz_sha256") != file_sha256(V2_STACK_RUNS / "cb51.npz"):
            raise ValueError("V2 CB fit51 cache no longer matches its frozen receipt")
        if cb_state.get("id") != "CB_MISSING":
            raise ValueError("V2 cb51 score is not the structured CB_MISSING score")
        _verify_fit_fingerprint(frame, y, cycle, 51, cb_state["input_feature_names"], cb_state, expected_fit_n)
        identities = {"C2_B": {"npz": file_sha256(V2_CAL_RUNS / "inner51.npz"),
                                 "json": file_sha256(V2_CAL_RUNS / "inner51.json")},
                      "CB_MISSING": {"npz": file_sha256(V2_STACK_RUNS / "cb51.npz"),
                                     "json": file_sha256(V2_STACK_RUNS / "cb51.json")}}
    else:
        base_path = ROUND1 / f"fit{fit_cutoff}__C2_B.npz"
        audit_path = ROUND1_AUDITS / f"fit{fit_cutoff}.json"
        with np.load(base_path, allow_pickle=False) as z:
            base_rows = z["outer_rows"].astype(np.int64, copy=True)
            base_all = z["probability"].astype(float, copy=True)
        audit = read_json(audit_path).get("C2_B")
        if not isinstance(audit, dict) or audit.get("model") != "C2_B":
            raise ValueError(f"C2_B audit identity missing at fit{fit_cutoff}")
        if array_digest(base_all) != audit.get("probability_sha256"):
            raise ValueError(f"C2_B probability hash mismatch at fit{fit_cutoff}")
        state = audit["fitted_state"]
        _verify_fit_fingerprint(frame, y, cycle, fit_cutoff, state["input_feature_names"], state,
                                int(np.count_nonzero((cycle >= 47) & (cycle <= fit_cutoff))))
        cb_path = CB_RUNS / f"fit{fit_cutoff}.npz"
        cb_meta_path = cb_path.with_suffix(".json")
        with np.load(cb_path, allow_pickle=False) as z:
            cb_rows = z["rows"].astype(np.int64, copy=True)
            cb_all = z["candidate"].astype(float, copy=True)
        cb_meta = read_json(cb_meta_path)
        cb_state = cb_meta.get("state", {})
        cb_freeze = read_json(CB_FREEZE)
        if cb_meta.get("freeze_signature") != cb_freeze.get("signature") or cb_meta.get("npz_sha256") != file_sha256(cb_path):
            raise ValueError(f"CB_MISSING cache no longer matches its frozen receipt at fit{fit_cutoff}")
        if cb_state.get("id") != "CB_MISSING" or array_digest(cb_all) != cb_meta.get("prediction_sha256"):
            raise ValueError(f"CB_MISSING identity/probability hash mismatch at fit{fit_cutoff}")
        _verify_fit_fingerprint(frame, y, cycle, fit_cutoff, cb_state["input_feature_names"], cb_state,
                                int(np.count_nonzero((cycle >= 47) & (cycle <= fit_cutoff))))
        identities = {"C2_B": {"npz": file_sha256(base_path), "audit": file_sha256(audit_path)},
                      "CB_MISSING": {"npz": file_sha256(cb_path), "json": file_sha256(cb_meta_path)}}
        # Round1 predictions are aligned to the same four-cycle cache row set as
        # the CatBoost predictions; fit 55 is allowed to cover its H12_E horizon.
        if fit_cutoff == 55:
            pass
        bpos = np.searchsorted(base_rows, rows)
        cpos = np.searchsorted(cb_rows, rows)
        if (bpos >= len(base_rows)).any() or (cpos >= len(cb_rows)).any() or not np.array_equal(base_rows[bpos], rows) or not np.array_equal(cb_rows[cpos], rows):
            raise ValueError(f"C2_B/CB rows do not align for OOF block fit{fit_cutoff}")
        base_scores = base_all[bpos]
        cb = cb_all[cpos]
    if fit_cutoff == 51:
        if not np.array_equal(base_rows, rows) or not np.array_equal(cb_rows, rows):
            raise ValueError("fit51 cached OOF row IDs do not equal cycles 52-55")
    if base_scores.shape != rows.shape or cb.shape != rows.shape:
        raise ValueError(f"Score shape mismatch for OOF block fit{fit_cutoff}")
    for name, p in (("C2_B", base_scores), ("CB_MISSING", cb)):
        if not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
            raise ValueError(f"Invalid {name} probabilities in OOF block fit{fit_cutoff}")
    return {"fit_cutoff": fit_cutoff, "rows": rows, "scores": np.column_stack([base_scores, cb]),
            "source_hashes": identities}


def prepare_oof(frame: pd.DataFrame, y: np.ndarray, cycle: np.ndarray,
                missing: np.ndarray, cutoff: int, head_name: str) -> dict[str, Any]:
    blocks = [load_block(frame, y, cycle, base_end)
              for base_end in range(51, cutoff, 4)]
    all_rows = validate_oof_blocks(blocks, cycle, cutoff)
    all_scores = np.concatenate([b["scores"] for b in blocks], axis=0)
    if all_scores.shape[0] != all_rows.size:
        raise ValueError("OOF score rows are not aligned to exact OOF block rows")
    present = ~missing[all_rows]
    rows = all_rows[present]
    X = frame.iloc[rows][list(EXPANDED_FEATURES)].copy()
    yy = y[rows]
    col_indices = [0] if head_name == "A_base_structured" else [0, 1]
    scores = all_scores[present][:, col_indices]
    if rows.size != np.unique(rows).size or not np.isfinite(scores).all() or not np.isin(yy, [0, 1]).all():
        raise ValueError("Invalid present-only OOF meta-training input")
    score_names = HEADS[head_name]
    raw_hash = training_input_hash(rows, X, yy, scores)
    return {"blocks": blocks, "all_rows": all_rows, "present_rows": rows,
            "X": X, "y": yy, "scores": scores, "score_names": score_names,
            "raw_input_sha256": raw_hash,
            "block_summary": [{"fit_cutoff": b["fit_cutoff"],
                               "cycles": [b["fit_cutoff"] + 1, b["fit_cutoff"] + 4],
                               "all_rows": int(len(b["rows"])),
                               "present_rows": int((~missing[b["rows"]]).sum()),
                               "row_sha256": array_digest(b["rows"]),
                               "source_hashes": b["source_hashes"]} for b in blocks]}


def outer_inputs(frame: pd.DataFrame, y: np.ndarray, cycle: np.ndarray,
                 missing: np.ndarray, cutoff: int) -> dict[str, Any]:
    horizon = OUTER[cutoff]["horizon"]
    rows = expected_rows(cycle, cutoff + 1, horizon)
    base_path = ROUND1 / f"fit{cutoff}__C2_B.npz"
    with np.load(base_path, allow_pickle=False) as z:
        base_rows = z["outer_rows"].astype(np.int64, copy=True)
        base_all = z["probability"].astype(float, copy=True)
    audit = read_json(ROUND1_AUDITS / f"fit{cutoff}.json")["C2_B"]
    if not np.array_equal(base_rows, expected_rows(cycle, cutoff + 1, horizon)) or array_digest(base_all) != audit.get("probability_sha256"):
        raise ValueError(f"Outer C2_B cache mismatch at fit{cutoff}")
    cb_path = CB_RUNS / f"fit{cutoff}.npz"
    with np.load(cb_path, allow_pickle=False) as z:
        cb_rows = z["rows"].astype(np.int64, copy=True)
        cb_all = z["candidate"].astype(float, copy=True)
    cb_meta = read_json(cb_path.with_suffix(".json"))
    if not np.array_equal(cb_rows, rows) or not np.array_equal(base_rows, rows) or array_digest(cb_all) != cb_meta.get("prediction_sha256"):
        raise ValueError(f"Outer C2_B and CB_MISSING caches do not align exactly at fit{cutoff}")
    if not np.isfinite(base_all).all() or not np.isfinite(cb_all).all():
        raise ValueError("Outer control probabilities are not finite")
    return {"rows": rows, "base": base_all, "cb": cb_all,
            "champion": np.where(missing[rows], cb_all, base_all),
            "input_hashes": {"C2_B_npz": file_sha256(base_path),
                             "C2_B_audit": file_sha256(ROUND1_AUDITS / f"fit{cutoff}.json"),
                             "CB_MISSING_npz": file_sha256(cb_path),
                             "CB_MISSING_meta": file_sha256(cb_path.with_suffix(".json"))}}


def identity() -> dict[str, Any]:
    protocol = read_json(PROTOCOL_PATH)
    frame, y, cycle, missing, train = load_train()
    identities = {}
    for cutoff in OUTER:
        identities[str(cutoff)] = {}
        for head_name in HEADS:
            oof = prepare_oof(frame, y, cycle, missing, cutoff, head_name)
            outer = outer_inputs(frame, y, cycle, missing, cutoff)
            identities[str(cutoff)][head_name] = {
                "oof_cycles": [52, cutoff], "oof_rows_all": int(len(oof["all_rows"])),
                "oof_rows_present": int(len(oof["present_rows"])),
                "oof_input_sha256": oof["raw_input_sha256"],
                "oof_blocks": oof["block_summary"],
                "outer_rows_sha256": array_digest(outer["rows"]),
                "outer_cache_hashes": outer["input_hashes"],
            }
    return {
        "protocol_sha256": file_sha256(PROTOCOL_PATH),
        "engineering_amendment_sha256": file_sha256(AMENDMENT_PATH),
        "code_sha256": {path: file_sha256(ROOT / path) for path in CODE_FILES},
        "official_train_sha256": file_sha256(train),
        "source_cache_sha256": path_hashes(all_input_paths(train)),
        "oof_training_input_identities": identities,
        "feature_allowlist": list(EXPANDED_FEATURES),
        "head_score_names": {k: list(v) for k, v in HEADS.items()},
        "outer_fits": {str(k): v for k, v in OUTER.items()},
        "python": platform.python_version(),
        "packages": {name: importlib.metadata.version(name)
                     for name in ["numpy", "pandas", "scipy", "scikit-learn", "lightgbm"]},
        "source_commit_context": subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip(),
    }


def freeze_only() -> dict[str, Any]:
    current = identity()
    signature = digest_json(current)
    if FREEZE_PATH.exists():
        old = read_json(FREEZE_PATH)
        if old.get("identity") != current or old.get("signature") != signature:
            raise RuntimeError("Existing residual freeze differs; preserve it and start a reviewed new protocol.")
        return old
    if RUN_DIR.exists() and any(RUN_DIR.iterdir()):
        raise FileExistsError("Residual run directory is not empty before freeze")
    prior = read_json(ORIGINAL_FREEZE_PATH)
    amendment = read_json(AMENDMENT_PATH)
    if (prior.get("signature") != amendment.get("parent_freeze_signature")
            or prior.get("status") != "FROZEN_BEFORE_RESIDUAL_HEAD_FITS"
            or amendment.get("model_fit_or_prediction_occurred_before_correction") is not False):
        raise RuntimeError("Engineering freeze amendment does not bind the preserved pre-fit receipt")
    receipt = {"status": "FROZEN_BEFORE_RESIDUAL_HEAD_FITS_AMENDMENT_01", "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
               "identity": current, "signature": signature,
               "source_commit_context": current["source_commit_context"],
               "supersedes_freeze_signature": prior["signature"],
               "engineering_amendment_sha256": file_sha256(AMENDMENT_PATH),
               "source_boundary": "Official Classification Train and exact archived OOF/outer score caches only; no Test read."}
    CONFIG.mkdir(parents=True, exist_ok=True)
    FREEZE_PATH.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return receipt


def check_freeze() -> dict[str, Any]:
    if not FREEZE_PATH.is_file():
        raise RuntimeError("Freeze protocol and OOF hashes with --freeze-only before any head fit")
    receipt = read_json(FREEZE_PATH)
    current = identity()
    if digest_json(current) != receipt.get("signature") or digest_json(receipt.get("identity")) != receipt.get("signature"):
        raise RuntimeError("Residual protocol, source, Train, OOF scores, or raw OOF inputs changed after freeze")
    return receipt


def binary_metrics(y: np.ndarray, p: np.ndarray) -> dict[str, Any]:
    yy = np.asarray(y, dtype=np.int8)
    pp = np.asarray(p, dtype=float)
    if pp.shape != yy.shape or not np.isfinite(pp).all() or ((pp < 0) | (pp > 1)).any():
        raise ValueError("Evaluation probabilities must be finite and aligned")
    pred = pp >= 0.5
    pos = yy == 1
    tp = int(np.count_nonzero(pred & pos)); tn = int(np.count_nonzero(~pred & ~pos))
    fp = int(np.count_nonzero(pred & ~pos)); fn = int(np.count_nonzero(~pred & pos))
    ppos = tp / (tp + fp) if tp + fp else 0.0
    rpos = tp / (tp + fn) if tp + fn else 0.0
    pneg = tn / (tn + fn) if tn + fn else 0.0
    rneg = tn / (tn + fp) if tn + fp else 0.0
    fpos = 2 * ppos * rpos / (ppos + rpos) if ppos + rpos else 0.0
    fneg = 2 * pneg * rneg / (pneg + rneg) if pneg + rneg else 0.0
    return {"n": int(len(yy)), "macro_f1": float((fpos + fneg) / 2),
            "f1_positive": float(fpos), "f1_negative": float(fneg),
            "positive_prevalence": float(pos.mean()) if len(yy) else None,
            "positive_prediction_rate": float(pred.mean()) if len(yy) else None,
            "tp": tp, "tn": tn, "fp": fp, "fn": fn,
            "error_count": fp + fn, "brier": float(np.mean((pp - yy) ** 2)),
            "threshold": 0.5}


def compare_views(frame: pd.DataFrame, y: np.ndarray, cycle: np.ndarray,
                  records: dict[int, dict[str, dict[str, Any]]]) -> dict[str, Any]:
    sections: dict[str, list[tuple[int, np.ndarray]]] = {"W1": [], "W4": [], "H12_E": [], "H12_L": []}
    for cutoff, record in records.items():
        rows = record["A_base_structured"]["outer_rows"]
        if cutoff == 55:
            sections["W1"].append((cutoff, np.flatnonzero((cycle[rows] >= 56) & (cycle[rows] <= 59))))
            sections["H12_E"].append((cutoff, np.arange(len(rows))))
        else:
            sections["W4"].append((cutoff, np.flatnonzero((cycle[rows] >= 68) & (cycle[rows] <= 71))))
            sections["H12_L"].append((cutoff, np.arange(len(rows))))
    sections["W1_plus_W4_pilot_slice"] = sections["W1"] + sections["W4"]
    sections["H12_pooled"] = sections["H12_E"] + sections["H12_L"]
    result = {}
    for section, parts in sections.items():
        result[section] = {}
        for head_name in HEADS:
            labels, champion, candidate = [], [], []
            for cutoff, local in parts:
                rec = records[cutoff][head_name]
                rows = rec["outer_rows"][local]
                labels.append(y[rows]); champion.append(rec["champion_probability"][local])
                candidate.append(rec["candidate_probability"][local])
            yy = np.concatenate(labels); pc = np.concatenate(champion); ph = np.concatenate(candidate)
            m = np.concatenate([records[cutoff][head_name]["missing_mask"][local] for cutoff, local in parts]).astype(bool)
            present = ~m
            current_all = binary_metrics(yy, pc); head_all = binary_metrics(yy, ph)
            current_present = binary_metrics(yy[present], pc[present]); head_present = binary_metrics(yy[present], ph[present])
            current_missing = binary_metrics(yy[m], pc[m]); head_missing = binary_metrics(yy[m], ph[m])
            result[section][head_name] = {
                "all_current": current_all, "all_candidate": head_all,
                "all_delta_macro_f1": head_all["macro_f1"] - current_all["macro_f1"],
                "present_current": current_present, "present_candidate": head_present,
                "present_delta_macro_f1": head_present["macro_f1"] - current_present["macro_f1"],
                "missing_current": current_missing, "missing_candidate": head_missing,
                "missing_exact_match": bool(np.array_equal(ph[m], pc[m])),
            }
    return result


def write_report(results: dict[str, Any]) -> None:
    sections = results["evaluation"]
    lines = [
        "# V3 Classification — Nonlinear OOF residual pilot",
        "",
        f"**Status:** `{results['status']}`. Two heads, four outer fits total; no new base model was fitted. Official Test and Kaggle were not read or used.",
        "",
        "The correction heads were trained only on stream-present, past-only OOF blocks: cutoff 55 used cycles 52–55 scored by base fit 51; cutoff 67 used cycles 52–67 scored by base fits 51, 55, 59, and 63. Every block is exactly four next cycles, disjoint, and ends no later than its correction cutoff. The head preprocessor fits only on these OOF rows. Outer cutoffs 55 and 67 reuse cached C2_B and CB_MISSING predictions; missing rows retain CB_MISSING byte-for-byte.",
        "",
        "Head A uses C2_B score + expanded structured covariates. Head B adds the weak structured CB score as a fixed ablation. Both use the same frozen LightGBM correction settings, logit(C2_B) init margin, 0.5 shrinkage, and fixed 0.50 threshold. No outer outcomes choose inputs, weights, thresholds, or route.",
        "",
        "## Held-out views",
        "",
        "The pilot covers W1 and W4 plus their existing H12_E/H12_L views. The pooled primary pilot slice combines W1 and W4 only; it is not the full six-window primary metric. H12 views reuse those same two cutoffs and remain adaptive development evidence.",
        "",
        "| View | Head | Present F1 current → candidate (Δ) | All-row F1 current → candidate (Δ) | Missing route exact? |",
        "|---|---|---:|---:|---|",
    ]
    order = ["W1", "W4", "W1_plus_W4_pilot_slice", "H12_E", "H12_L", "H12_pooled"]
    for view in order:
        for head in HEADS:
            d = sections[view][head]
            lines.append(f"| {view} | {head} | {d['present_current']['macro_f1']:.4f} → {d['present_candidate']['macro_f1']:.4f} ({d['present_delta_macro_f1']:+.4f}) | {d['all_current']['macro_f1']:.4f} → {d['all_candidate']['macro_f1']:.4f} ({d['all_delta_macro_f1']:+.4f}) | {d['missing_exact_match']} |")
    harms = [sections[v][h]["present_delta_macro_f1"] for v in ("W1", "W4", "H12_E", "H12_L") for h in HEADS]
    pooled_gain = any(sections[v][h]["present_delta_macro_f1"] > 0 for v in ("W1_plus_W4_pilot_slice", "H12_pooled") for h in HEADS)
    gate = all(x >= -0.01 for x in harms) and pooled_gain and all(sections[v][h]["missing_exact_match"] for v in order for h in HEADS)
    lines += ["", f"**Frozen pilot gate:** `{'PASS' if gate else 'FAIL'}`. It requires at least one pooled primary pilot-slice or H12 present-stream F1 gain, no individual W1/W4/H12_E/H12_L present-stream loss below −0.01, and exact missing-route preservation. This gate is a continuation screen, not a model-selection guarantee.", "",
              "## Provenance and limits", "",
              f"Official Train SHA-256: `{results['freeze_identity']['official_train_sha256']}`. Source/protocol/OOF inputs were frozen before fitting with signature `{results['freeze_signature']}`. The full raw input hashes, OOF row IDs and block denominators are in `results.json`; the ignored run directory contains outer vectors, transformed designs and LightGBM text models for independent reconstruction.", "",
              "Only cycles 56–59 and 68–71 are represented in the pooled primary pilot slice; no claim about W2/W3/W5/W6 or the full six-window primary score is made. The H12 pooled view overlaps its early and late constituent slices and uses the same chronological dataset already studied. If both heads pass, Head B is not preferred automatically: the weak CB score is an ablation, and a small difference must be weighed against the pre-existing negative conditional diagnostics.", "",
              "Reproduce with `python scripts/optimization_v3_cls_residual.py --freeze-only` before fits, then `python scripts/optimization_v3_cls_residual.py --run`, then `python scripts/optimization_v3_verify_cls_residual.py`. No Kaggle submission or Git operation is part of this pilot.", ""]
    (REPORT_DIR / "FINDINGS.md").write_text("\n".join(lines), encoding="utf-8")


def run() -> dict[str, Any]:
    freeze = check_freeze()
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    frame, y, cycle, missing, _ = load_train()
    records: dict[int, dict[str, dict[str, Any]]] = {}
    states = {}
    for cutoff in OUTER:
        outer = outer_inputs(frame, y, cycle, missing, cutoff)
        recs: dict[str, dict[str, Any]] = {}
        for head_name, score_names in HEADS.items():
            fit_data = prepare_oof(frame, y, cycle, missing, cutoff, head_name)
            model = OOFResidualHead("classification", score_names)
            model.fit(fit_data["X"], fit_data["scores"], fit_data["y"])
            if model.audit_state() != model.audit_state():
                raise RuntimeError("Head state changed during fit")
            outer_rows = outer["rows"]
            present = ~missing[outer_rows]
            X_outer = frame.iloc[outer_rows[present]][list(EXPANDED_FEATURES)].copy()
            score_indices = [0] if head_name == "A_base_structured" else [0, 1]
            outer_scores_all = np.column_stack([outer["base"], outer["cb"]])[:, score_indices]
            outer_scores_present = outer_scores_all[present]
            design, _, _ = model._design(X_outer, outer_scores_present, False)
            design_array = design.to_numpy(dtype=float)
            raw_correction = model.model.booster_.predict(design, raw_score=True, num_threads=2)
            independently_composed = expit(logit(np.clip(outer["base"][present], 1e-6, 1 - 1e-6)) + 0.5 * raw_correction)
            model_prediction = model.predict(X_outer, outer_scores_present)
            if not np.allclose(independently_composed, model_prediction, rtol=0, atol=1e-12):
                raise RuntimeError("Residual prediction does not equal fixed raw-margin + base-score composition")
            if not np.isfinite(fit_data["scores"]).all() or not np.isfinite(independently_composed).all() or ((independently_composed < 0) | (independently_composed > 1)).any():
                raise RuntimeError("Nonfinite present meta or corrected probabilities")
            candidate = outer["champion"].copy()
            candidate[present] = independently_composed
            if not np.array_equal(candidate[missing[outer_rows]], outer["champion"][missing[outer_rows]]):
                raise RuntimeError("Residual head changed the missing-stream route")
            model_path = RUN_DIR / f"fit{cutoff}_{head_name}.txt"
            pred_path = RUN_DIR / f"fit{cutoff}_{head_name}.npz"
            if model_path.exists() or pred_path.exists():
                raise FileExistsError(f"Refusing to overwrite residual run artifact: {pred_path}")
            model_path.write_text(model.model.booster_.model_to_string(), encoding="utf-8")
            np.savez_compressed(pred_path,
                                outer_rows=outer_rows.astype(np.int64),
                                missing_mask=missing[outer_rows].astype(np.uint8),
                                base_probability=outer["base"],
                                cb_probability=outer["cb"],
                                champion_probability=outer["champion"],
                                candidate_probability=candidate,
                                raw_correction_present=raw_correction,
                                outer_design_present=design_array,
                                oof_present_rows=fit_data["present_rows"].astype(np.int64),
                                oof_scores=fit_data["scores"].astype(float),
                                oof_target=fit_data["y"].astype(np.int8))
            recs[head_name] = {"outer_rows": outer_rows,
                               "champion_probability": outer["champion"],
                               "candidate_probability": candidate,
                               "missing_mask": missing[outer_rows].astype(bool)}
            state = model.audit_state()
            states[f"{cutoff}/{head_name}"] = {
                "state": state,
                "oof_rows_all": int(len(fit_data["all_rows"])),
                "oof_rows_present": int(len(fit_data["present_rows"])),
                "oof_raw_input_sha256": fit_data["raw_input_sha256"],
                "oof_blocks": fit_data["block_summary"],
                "outer_rows": int(len(outer_rows)),
                "outer_present_rows": int(present.sum()),
                "outer_missing_rows": int((~present).sum()),
                "outer_input_hashes": outer["input_hashes"],
                "score_names": list(score_names),
                "model_path": str(model_path.relative_to(ROOT)),
                "model_sha256": file_sha256(model_path),
                "prediction_path": str(pred_path.relative_to(ROOT)),
                "prediction_sha256": file_sha256(pred_path),
                "outer_design_sha256": array_digest(design_array),
                "raw_correction_sha256": array_digest(raw_correction),
                "candidate_present_probability_sha256": array_digest(candidate[present]),
                "missing_route_exact_match": True,
                "formula": "expit(logit(clip(C2_B,1e-6,1-1e-6)) + 0.5 * raw_correction)",
            }
        records[cutoff] = recs
    evaluation = compare_views(frame, y, cycle, records)
    harms = [evaluation[v][h]["present_delta_macro_f1"]
             for v in ("W1", "W4", "H12_E", "H12_L") for h in HEADS]
    pooled_gain = any(evaluation[v][h]["present_delta_macro_f1"] > 0
                      for v in ("W1_plus_W4_pilot_slice", "H12_pooled") for h in HEADS)
    missing_exact = all(evaluation[v][h]["missing_exact_match"] for v in evaluation for h in HEADS)
    gate = all(value >= -0.01 for value in harms) and pooled_gain and missing_exact
    result = {"status": "PASS_CONTINUATION_GATE" if gate else "FAIL_CONTINUATION_GATE",
              "freeze_signature": freeze["signature"], "freeze_identity": freeze["identity"],
              "outer_fits": [55, 67], "head_fits": 4,
              "test_read": False, "kaggle_read_or_submit": False,
              "outer_labels_used_for_fit_or_selection": False,
              "head_metadata": states, "evaluation": evaluation,
              "gate": {"pooled_primary_slice_or_H12_improved": pooled_gain,
                       "no_view_harm_over_0_01": bool(all(value >= -0.01 for value in harms)),
                       "missing_route_exact_match": missing_exact,
                       "overall_pass": gate},
              "interpretation": "Two-cutoff pilot only. W1+W4 is a slice of primary windows, not a full-six primary metric. H12_E/L share chronology and overlap the primary slice."}
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    result_path = REPORT_DIR / "results.json"
    if result_path.exists():
        raise FileExistsError(f"Refusing to overwrite pilot result: {result_path}")
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_report(result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--freeze-only", action="store_true")
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()
    if args.freeze_only == args.run:
        parser.error("Choose exactly one: --freeze-only or --run")
    if args.freeze_only:
        receipt = freeze_only()
        print(json.dumps({"status": receipt["status"], "signature": receipt["signature"],
                          "official_train_sha256": receipt["identity"]["official_train_sha256"],
                          "oof_rows": {cutoff: {name: data["oof_rows_present"]
                                                for name, data in heads.items()}
                                       for cutoff, heads in receipt["identity"]["oof_training_input_identities"].items()}}, indent=2))
    else:
        result = run()
        print(json.dumps({"status": result["status"], "freeze_signature": result["freeze_signature"],
                          "gate": result["gate"],
                          "evaluation": {view: {head: {"present_delta_macro_f1": metrics["present_delta_macro_f1"],
                                                       "present_current_f1": metrics["present_current"]["macro_f1"],
                                                       "present_candidate_f1": metrics["present_candidate"]["macro_f1"]}
                                                for head, metrics in heads.items()}
                                        for view, heads in result["evaluation"].items()}}, indent=2))


if __name__ == "__main__":
    main()
