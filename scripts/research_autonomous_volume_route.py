"""Post-hoc q80 route ablation using only previously fitted branch vectors.

Run ``--freeze`` before ``--score``. Freezing reads official Classification
Train inputs without reading its target and verifies existing prediction
caches. Scoring composes those same vectors; it performs zero new fits.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.models.classification_metrics import binary_metrics
from src.models.classification_models import _array_digest
from src.models.classification_round2_models import stream_missing_mask
from src.utils.paths import task_files


PROTOCOL_DIR = ROOT / "experiments" / "autonomous_eda" / "classification_volume"
PROTOCOL_PATH = PROTOCOL_DIR / "protocol.json"
FREEZE_PATH = PROTOCOL_DIR / "freeze.json"
REPORT_DIR = ROOT / "reports" / "autonomous_eda" / "classification_volume"
RUN_DIR = ROOT / "experiments" / "runs" / "autonomous_cls_volume_v1"
SOURCE_EXPERIMENT = ROOT / "experiments" / "autonomous_eda" / "classification"
SOURCE_REPORT = ROOT / "reports" / "autonomous_eda" / "classification"
SOURCE_RUN = ROOT / "experiments" / "runs" / "autonomous_cls_v1"
ROUND1_RUN = ROOT / "experiments" / "runs" / "classification_round1_v1"
ROUND1_AUDITS = ROOT / "reports" / "classification_round1" / "fit_audits"
ROUND2_PROTOCOL = ROOT / "experiments" / "classification_round2" / "protocol.json"
FIT_ENDS = [55, 59, 63, 67, 71, 75]
PRIMARY = [
    {"id": "W1", "fit": [47, 55], "validation": [56, 59]},
    {"id": "W2", "fit": [47, 59], "validation": [60, 63]},
    {"id": "W3", "fit": [47, 63], "validation": [64, 67]},
    {"id": "W4", "fit": [47, 67], "validation": [68, 71]},
    {"id": "W5", "fit": [47, 71], "validation": [72, 75]},
    {"id": "W6", "fit": [47, 75], "validation": [76, 79]},
]
H12 = [
    {"id": "H12_E", "fit": [47, 55], "validation": [56, 67]},
    {"id": "H12_L", "fit": [47, 67], "validation": [68, 79]},
]
MODELS = ["C4_HGB", "present_specialist", "present_q80_route"]
VOLUME_LABELS = ["Q1_low", "Q2", "Q3", "Q4", "Q5_high"]
CODE_FILES = [
    "src/models/autonomous_volume_route.py",
    "scripts/research_autonomous_volume_route.py",
    "scripts/verify_autonomous_volume_route.py",
    "tests/test_autonomous_volume_route.py",
]


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_sha256(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False,
                         separators=(",", ":")).encode("utf-8", errors="surrogatepass")
    return hashlib.sha256(payload).hexdigest()


def array_sha256(array: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(array, dtype="<f8").tobytes(order="C")).hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)


def train_path() -> Path:
    path = task_files("classification")["train"]
    if path is None:
        raise FileNotFoundError("Official Classification Train.csv is missing")
    return path.resolve()


def prior_freeze() -> dict[str, Any]:
    path = SOURCE_EXPERIMENT / "freeze.json"
    data = read_json(path)
    unsigned = {key: value for key, value in data.items() if key != "freeze_signature"}
    if json_sha256(unsigned) != data.get("freeze_signature"):
        raise ValueError("Prior Classification freeze signature does not reconcile")
    prior_verification = read_json(SOURCE_REPORT / "verification.json")
    if prior_verification.get("status") != "PASS" or \
       prior_verification.get("freeze_signature") != data.get("freeze_signature"):
        raise ValueError("Prior Classification cache verification is not PASS")
    for relative, expected in data["source_code_sha256"].items():
        if file_sha256(ROOT / relative) != expected:
            raise ValueError(f"Prior frozen source changed: {relative}")
    if file_sha256(train_path()) != data["train"]["sha256"]:
        raise ValueError("Official Classification Train differs from the verified source")
    return data


def feature_frame(*, include_target: bool) -> tuple[pd.DataFrame, np.ndarray | None, Path]:
    path = train_path()
    frame = pd.read_csv(path) if include_target else pd.read_csv(
        path, usecols=["record_id", "cycle_code", "signal_stream", "text_load"]
    )
    if not {"record_id", "cycle_code", "signal_stream", "text_load"}.issubset(frame.columns):
        raise ValueError("Official Train is missing required routing columns")
    if frame["record_id"].isna().any() or not frame["record_id"].is_unique:
        raise ValueError("record_id must be unique and nonmissing")
    target = None
    if include_target:
        target = pd.to_numeric(frame["artifact_signal"], errors="raise").to_numpy(dtype=np.int8)
        if not np.isin(target, [0, 1]).all():
            raise ValueError("artifact_signal must contain only 0/1")
    return frame, target, path


def expected_horizon(fit_end: int) -> int:
    windows = [window for window in [*PRIMARY, *H12] if int(window["fit"][1]) == fit_end]
    if not windows:
        raise ValueError(f"No frozen windows at cutoff {fit_end}")
    return max(int(window["validation"][1]) for window in windows)


def _array_from_cache(cache: Path, key: str) -> np.ndarray:
    with np.load(cache, allow_pickle=False) as archive:
        return archive[key].copy()


def _read_branch_cache(fit_end: int, frame: pd.DataFrame) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    cache_path = SOURCE_RUN / f"fit{fit_end}.npz"
    audit_path = SOURCE_RUN / f"fit{fit_end}_audit.json"
    if not cache_path.exists() or not audit_path.exists():
        raise FileNotFoundError(f"Prior verified branch cache missing: {cache_path}")
    audit = read_json(audit_path)
    if file_sha256(cache_path) != audit.get("cache_sha256"):
        raise ValueError(f"Prior branch cache hash mismatch at fit{fit_end}")
    with np.load(cache_path, allow_pickle=False) as archive:
        needed = ["rows", "C1_B_all_probability", "C2_B_all_probability",
                  "C2_B_present_rows", "C2_B_present_probability"]
        if not set(needed).issubset(archive.files):
            raise ValueError(f"Prior branch cache missing required vectors at fit{fit_end}")
        arrays = {key: archive[key].copy() for key in needed}
    rows = arrays["rows"].astype(np.int64)
    cycle = pd.to_numeric(frame["cycle_code"], errors="raise").to_numpy(dtype=int)
    horizon = expected_horizon(fit_end)
    expected_rows = np.flatnonzero((cycle > fit_end) & (cycle <= horizon))
    if not np.array_equal(rows, expected_rows):
        raise ValueError(f"Cached outer row order or cycle boundary differs at fit{fit_end}")
    missing = stream_missing_mask(frame.iloc[rows])
    present_rows = rows[~missing]
    if not np.array_equal(arrays["C2_B_present_rows"].astype(np.int64), present_rows):
        raise ValueError(f"Cached present expert prediction rows mismatch at fit{fit_end}")
    if arrays["C1_B_all_probability"].shape != rows.shape or arrays["C2_B_all_probability"].shape != rows.shape:
        raise ValueError(f"Cached all-row control shape mismatch at fit{fit_end}")
    if arrays["C2_B_present_probability"].shape != present_rows.shape:
        raise ValueError(f"Cached present-specialist prediction shape mismatch at fit{fit_end}")
    for key in ("C1_B_all_probability", "C2_B_all_probability", "C2_B_present_probability"):
        if not np.isfinite(arrays[key]).all() or ((arrays[key] < 0) | (arrays[key] > 1)).any():
            raise ValueError(f"Invalid cached probability values: fit{fit_end}/{key}")
    if array_sha256(arrays["C1_B_all_probability"]) != audit["baseline_control_c1_b_probability_sha256"]:
        raise ValueError(f"C1_B all-row probability hash mismatch at fit{fit_end}")
    if array_sha256(arrays["C2_B_all_probability"]) != audit["baseline_control_c2_b_probability_sha256"]:
        raise ValueError(f"C2_B all-row probability hash mismatch at fit{fit_end}")
    if array_sha256(arrays["C2_B_present_probability"]) != audit["branch_prediction_sha256"]["C2_B_present_only"]:
        raise ValueError(f"C2_B present-specialist probability hash mismatch at fit{fit_end}")
    # Reconcile the original C1_B/C2_B controls to their Round 1 artifacts, read-only.
    round1_audit = read_json(ROUND1_AUDITS / f"fit{fit_end}.json")
    identities = {}
    for model_id, key in (("C1_B", "C1_B_all_probability"), ("C2_B", "C2_B_all_probability")):
        path = ROUND1_RUN / f"fit{fit_end}__{model_id}.npz"
        expected_model = round1_audit[model_id]
        if not path.exists():
            raise FileNotFoundError(path)
        with np.load(path, allow_pickle=False) as original:
            original_rows = original["outer_rows"].astype(np.int64)
            original_p = original["probability"].astype(float)
        if not np.array_equal(original_rows, rows) or not np.array_equal(original_p, arrays[key]):
            raise ValueError(f"{model_id} all-row cache differs from Round 1 at fit{fit_end}")
        if array_sha256(original_p) != expected_model.get("probability_sha256"):
            raise ValueError(f"{model_id} vector no longer matches Round 1 audit at fit{fit_end}")
        identities[model_id] = {
            "cache_sha256": file_sha256(path),
            "audit_probability_sha256": expected_model["probability_sha256"],
        }
    return arrays, {
        "cache_sha256": file_sha256(cache_path),
        "audit_sha256": file_sha256(audit_path),
        "audit": audit,
        "round1_control_sources": identities,
        "outer_rows": int(len(rows)),
        "outer_cycle_min": int(cycle[rows].min()),
        "outer_cycle_max": int(cycle[rows].max()),
    }


def code_hashes() -> dict[str, str]:
    return {relative: file_sha256(ROOT / relative) for relative in CODE_FILES}


def verify_windows_against_round2() -> None:
    source = read_json(ROUND2_PROTOCOL)
    if source.get("primary_windows") != PRIMARY or source.get("secondary_windows") != H12:
        raise ValueError("Volume route windows differ from frozen Round 2 windows")


def freeze() -> None:
    if FREEZE_PATH.exists():
        raise FileExistsError(f"Volume route extension already frozen: {FREEZE_PATH}")
    if RUN_DIR.exists() and any(RUN_DIR.rglob("*.npz")):
        raise RuntimeError("Volume route predictions already exist; cannot freeze after scoring")
    verify_windows_against_round2()
    base = prior_freeze()
    protocol = read_json(PROTOCOL_PATH)
    if protocol.get("source_development_freeze_signature") != base["freeze_signature"]:
        raise ValueError("Volume route protocol references a different source freeze")
    frame, _target, source = feature_frame(include_target=False)
    if file_sha256(source) != base["train"]["sha256"]:
        raise ValueError("Input SHA does not match the verified official Train")
    volume = pd.to_numeric(frame["text_load"], errors="coerce").to_numpy(dtype=float)
    cycle = pd.to_numeric(frame["cycle_code"], errors="raise").to_numpy(dtype=int)
    if not np.isfinite(volume).all():
        raise ValueError("text_load must be finite for q80 routing")
    missing_all = stream_missing_mask(frame)
    prior_fit_support = pd.read_csv(SOURCE_REPORT / "fit_support.csv")
    folds: dict[str, Any] = {}
    caches: dict[str, Any] = {}
    for fit_end in FIT_ENDS:
        fit_mask = cycle <= fit_end
        q80 = float(np.quantile(volume[fit_mask], 0.80))
        prior = prior_fit_support[(prior_fit_support["fit_end_cycle"] == fit_end) &
                                  (prior_fit_support["availability"] == "present")]
        if len(prior) != 1:
            raise ValueError(f"Missing previous fit-defined volume edges for fit{fit_end}")
        prior_edges = np.asarray(json.loads(prior.iloc[0]["quantile_edges_for_validation"]), dtype=float)
        if len(prior_edges) != 4 or not np.isclose(q80, prior_edges[3], atol=1e-12, rtol=0):
            raise ValueError(f"q80 differs from previously frozen fit-only quintile edge at fit{fit_end}")
        arrays, cache_info = _read_branch_cache(fit_end, frame)
        rows = arrays["rows"].astype(np.int64)
        missing = missing_all[rows]
        row_volume = volume[rows]
        high = (~missing) & (row_volume > q80)
        low = (~missing) & ~high
        folds[str(fit_end)] = {
            "q80": q80,
            "q80_source_rows": int(fit_mask.sum()),
            "q80_fit_text_load_sha256": array_sha256(volume[fit_mask]),
            "q80_fit_cycle_range": [int(cycle[fit_mask].min()), int(cycle[fit_mask].max())],
            "validation_rows": int(len(rows)),
            "validation_missing_rows": int(missing.sum()),
            "validation_present_not_above_q80_rows": int(low.sum()),
            "validation_present_above_q80_rows": int(high.sum()),
            "validation_present_q80_ties_including_missing": int((row_volume == q80).sum()),
        }
        caches[str(fit_end)] = {
            **{key: value for key, value in cache_info.items() if key != "audit"},
            "source_branch_state_sha256": {
                "C2_B_present_only": cache_info["audit"]["new_fits"]["C2_B_present_only"]["model_state_sha256"],
                "C1_B_missing_only": cache_info["audit"]["new_fits"]["C1_B_missing_only"]["model_state_sha256"],
            },
            "source_present_branch_fit_rows": cache_info["audit"]["new_fits"]["C2_B_present_only"]["fit_rows"],
            "source_present_branch_fit_data_sha256": cache_info["audit"]["new_fits"]["C2_B_present_only"]["fit_data_sha256"],
        }
    tracked_inputs = [
        SOURCE_EXPERIMENT / "protocol.json",
        SOURCE_EXPERIMENT / "freeze.json",
        SOURCE_REPORT / "FINDINGS.md",
        SOURCE_REPORT / "results.json",
        SOURCE_REPORT / "verification.json",
        SOURCE_REPORT / "summary.csv",
        SOURCE_REPORT / "windows.csv",
        SOURCE_REPORT / "per_cycle.csv",
        SOURCE_REPORT / "fit_support.csv",
        SOURCE_REPORT / "subgroups.csv",
        ROUND2_PROTOCOL,
    ]
    tracked_inputs.extend(ROOT / relative for relative in base["source_code_sha256"])
    for fit_end in FIT_ENDS:
        tracked_inputs.extend([SOURCE_RUN / f"fit{fit_end}.npz", SOURCE_RUN / f"fit{fit_end}_audit.json"])
        tracked_inputs.extend([ROUND1_RUN / f"fit{fit_end}__C1_B.npz", ROUND1_RUN / f"fit{fit_end}__C2_B.npz",
                               ROUND1_AUDITS / f"fit{fit_end}.json"])
    identity = {
        "status": "FROZEN_BEFORE_VOLUME_ROUTE_SCORING",
        "frozen_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "source_head": subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, check=True,
                                       capture_output=True, text=True).stdout.strip(),
        "protocol_sha256": file_sha256(PROTOCOL_PATH),
        "source_code_sha256": code_hashes(),
        "official_train": {"sha256": file_sha256(source), "rows": int(len(frame)),
                           "record_id_sha256": hashlib.sha256("\n".join(frame.record_id.astype(str)).encode("utf-8")).hexdigest(),
                           "target_read_during_freeze": False},
        "source_development_experiment": {
            "freeze_signature": base["freeze_signature"],
            "verification_status": "PASS",
            "verification_sha256": file_sha256(SOURCE_REPORT / "verification.json"),
            "source_protocol_sha256": file_sha256(SOURCE_EXPERIMENT / "protocol.json"),
            "source_results_sha256": file_sha256(SOURCE_REPORT / "results.json"),
            "source_subgroup_sha256": file_sha256(SOURCE_REPORT / "subgroups.csv"),
        },
        "round2_protocol_sha256": file_sha256(ROUND2_PROTOCOL),
        "folds": folds,
        "read_only_cache_inputs": caches,
        "read_only_cache_files_sha256": {str(path.relative_to(ROOT)): file_sha256(path) for path in tracked_inputs},
        "new_temporal_model_fits": 0,
        "official_test_values_read": False,
    }
    identity["freeze_signature"] = json_sha256(identity)
    write_json(FREEZE_PATH, identity)
    print(json.dumps({"status": identity["status"], "freeze_signature": identity["freeze_signature"],
                      "q80_by_fit": {key: {"q80": value["q80"],
                                           "fit_rows": value["q80_source_rows"],
                                           "present_high_validation": value["validation_present_above_q80_rows"],
                                           "present_low_validation": value["validation_present_not_above_q80_rows"],
                                           "missing_validation": value["validation_missing_rows"]}
                                     for key, value in folds.items()},
                      "new_model_fits": 0, "target_read_during_freeze": False},
                     indent=2, ensure_ascii=False))


def verify_freeze(frame: pd.DataFrame, source: Path) -> dict[str, Any]:
    if not FREEZE_PATH.exists():
        raise FileNotFoundError("Run --freeze before scoring")
    frozen = read_json(FREEZE_PATH)
    unsigned = {key: value for key, value in frozen.items() if key != "freeze_signature"}
    if json_sha256(unsigned) != frozen.get("freeze_signature"):
        raise ValueError("Volume route freeze signature does not reconcile")
    if frozen["protocol_sha256"] != file_sha256(PROTOCOL_PATH):
        raise ValueError("Volume route protocol changed after freeze")
    if frozen["source_code_sha256"] != code_hashes():
        raise ValueError("Volume route source code changed after freeze")
    if frozen["official_train"]["sha256"] != file_sha256(source):
        raise ValueError("Official Train bytes changed after volume route freeze")
    if frozen["source_development_experiment"]["freeze_signature"] != read_json(SOURCE_EXPERIMENT / "freeze.json")["freeze_signature"]:
        raise ValueError("Source model freeze was changed")
    for relative, expected in frozen["read_only_cache_files_sha256"].items():
        if file_sha256(ROOT / relative) != expected:
            raise ValueError(f"A read-only source file/cache changed after freeze: {relative}")
    return frozen


def read_fold_vectors(fit_end: int, frame: pd.DataFrame, frozen: dict[str, Any]) -> dict[str, Any]:
    arrays, info = _read_branch_cache(fit_end, frame)
    if info["cache_sha256"] != frozen["read_only_cache_inputs"][str(fit_end)]["cache_sha256"]:
        raise ValueError(f"Previously frozen branch cache hash changed at fit{fit_end}")
    rows = arrays["rows"].astype(np.int64)
    row_missing = stream_missing_mask(frame.iloc[rows])
    present_specialist_lookup = {int(row): float(p) for row, p in
                                 zip(arrays["C2_B_present_rows"], arrays["C2_B_present_probability"])}
    specialist_full = np.asarray([present_specialist_lookup.get(int(row), np.nan) for row in rows], dtype=float)
    if not np.isfinite(specialist_full[~row_missing]).all() or np.isfinite(specialist_full[row_missing]).any():
        raise ValueError(f"Present-specialist cache alignment invalid at fit{fit_end}")
    q80 = float(frozen["folds"][str(fit_end)]["q80"])
    volume = pd.to_numeric(frame.iloc[rows]["text_load"], errors="coerce").to_numpy(dtype=float)
    high = (~row_missing) & (volume > q80)
    c1_all = arrays["C1_B_all_probability"].astype(float)
    c2_all = arrays["C2_B_all_probability"].astype(float)
    c4 = np.where(row_missing, c1_all, c2_all)
    uniform = np.where(row_missing, c1_all, specialist_full)
    q80_route = np.where(row_missing, c1_all, np.where(high, specialist_full, c2_all))
    if not np.array_equal(q80_route[~high], c4[~high]):
        raise ValueError(f"q80 route changed an unaffected row at fit{fit_end}")
    if not np.array_equal(q80_route[high], uniform[high]):
        raise ValueError(f"q80 route differs from present specialist on high-volume rows at fit{fit_end}")
    return {"rows": rows, "C4_HGB": c4, "present_specialist": uniform,
            "present_q80_route": q80_route, "high_present": high,
            "present_specialist_full": specialist_full,
            "cache_sha256": info["cache_sha256"]}


def _volume_edges_fit_only(frame: pd.DataFrame, fit_end: int) -> np.ndarray:
    cycle = frame["cycle_code"].to_numpy(dtype=int)
    volume = pd.to_numeric(frame["text_load"], errors="raise").to_numpy(dtype=float)
    return np.quantile(volume[cycle <= fit_end], [0.2, 0.4, 0.6, 0.8])


def evaluate(frame: pd.DataFrame, y: np.ndarray, folds: dict[int, dict[str, Any]]) -> dict[str, Any]:
    cycle = frame["cycle_code"].to_numpy(dtype=int)
    volume = pd.to_numeric(frame["text_load"], errors="raise").to_numpy(dtype=float)
    missing_all = stream_missing_mask(frame)
    all_windows = {"primary": PRIMARY, "h12": H12}
    aggregate: dict[str, Any] = {}
    windows_out: list[dict[str, Any]] = []
    cycles_out: list[dict[str, Any]] = []
    subgroups_out: list[dict[str, Any]] = []
    summary: dict[str, Any] = {}
    for family, windows in all_windows.items():
        values = {name: {"y": [], "p": [], "rows": [], "bins": [], "q80_high": []}
                  for name in MODELS}
        for window in windows:
            fit_end = int(window["fit"][1])
            lo, hi = map(int, window["validation"])
            fold = folds[fit_end]
            rows = fold["rows"]
            select = (cycle[rows] >= lo) & (cycle[rows] <= hi)
            selected_rows = rows[select]
            if len(selected_rows) == 0:
                raise ValueError(f"Empty validation rows: {window['id']}")
            yy = y[selected_rows]
            bins = np.searchsorted(_volume_edges_fit_only(frame, fit_end), volume[selected_rows], side="right")
            high = fold["high_present"][select]
            for name in MODELS:
                pp = fold[name][select]
                metric = binary_metrics(yy, pp, 0.50)
                windows_out.append({
                    "family": family, "window": window["id"], "fit_end_cycle": fit_end,
                    "validation_cycles": [lo, hi], "model": name, "n": len(selected_rows),
                    "macro_f1": metric["macro_f1"], "positive_rate": metric["positive_rate"],
                    "positive_prediction_rate": metric["positive_prediction_rate"],
                    "false_positive": metric["false_positive"], "false_negative": metric["false_negative"],
                    "confusion_matrix": metric["confusion_matrix"], "per_class": metric["per_class"],
                })
                values[name]["y"].append(yy)
                values[name]["p"].append(pp)
                values[name]["rows"].append(selected_rows)
                values[name]["bins"].append(bins)
                values[name]["q80_high"].append(high)
                for cycle_value in range(lo, hi + 1):
                    mask_cycle = cycle[selected_rows] == cycle_value
                    if not mask_cycle.any():
                        continue
                    cm = binary_metrics(yy[mask_cycle], pp[mask_cycle], 0.50)
                    cycles_out.append({"family": family, "window": window["id"], "cycle_code": cycle_value,
                                       "model": name, "n": cm["n"], "macro_f1": cm["macro_f1"],
                                       "positive_rate": cm["positive_rate"],
                                       "false_positive": cm["false_positive"], "false_negative": cm["false_negative"]})
        aggregate[family] = {}
        for name in MODELS:
            yy = np.concatenate(values[name]["y"])
            pp = np.concatenate(values[name]["p"])
            rows = np.concatenate(values[name]["rows"])
            bins = np.concatenate(values[name]["bins"])
            q80_high = np.concatenate(values[name]["q80_high"])
            cycles = sorted(np.unique(cycle[rows]))
            equal_cycle = float(np.mean([
                binary_metrics(yy[cycle[rows] == cycle_value], pp[cycle[rows] == cycle_value], 0.50)["macro_f1"]
                for cycle_value in cycles
            ]))
            masks = {
                "stream_missing": missing_all[rows],
                "present_not_above_q80": (~missing_all[rows]) & ~q80_high,
                "present_above_q80": q80_high,
                "all_rows_q80_volume_bin": bins == 4,
            }
            for index, label in enumerate(VOLUME_LABELS):
                masks[f"text_load_{label}"] = bins == index
            group_metrics = {}
            for group, mask in masks.items():
                metric = binary_metrics(yy[mask], pp[mask], 0.50)
                group_metrics[group] = metric
                subgroups_out.append({"family": family, "model": name, "group": group,
                                      "n": metric["n"], "positive_rate": metric["positive_rate"],
                                      "macro_f1": metric["macro_f1"],
                                      "false_positive": metric["false_positive"],
                                      "false_negative": metric["false_negative"],
                                      "per_class": metric["per_class"]})
            aggregate[family][name] = {
                "metrics": binary_metrics(yy, pp, 0.50),
                "equal_cycle_macro_f1": equal_cycle,
                "per_cycle_count": len(cycles),
                "subgroups": group_metrics,
            }
    for name in MODELS:
        primary = aggregate["primary"][name]
        h12 = aggregate["h12"][name]
        c4_primary = aggregate["primary"]["C4_HGB"]
        c4_h12 = aggregate["h12"]["C4_HGB"]
        uniform_primary = aggregate["primary"]["present_specialist"]
        uniform_h12 = aggregate["h12"]["present_specialist"]
        summary[name] = {
            "primary_macro_f1": primary["metrics"]["macro_f1"],
            "primary_equal_cycle_macro_f1": primary["equal_cycle_macro_f1"],
            "h12_macro_f1": h12["metrics"]["macro_f1"],
            "delta_primary_vs_C4": primary["metrics"]["macro_f1"] - c4_primary["metrics"]["macro_f1"],
            "delta_equal_cycle_vs_C4": primary["equal_cycle_macro_f1"] - c4_primary["equal_cycle_macro_f1"],
            "delta_h12_vs_C4": h12["metrics"]["macro_f1"] - c4_h12["metrics"]["macro_f1"],
            "delta_primary_vs_uniform_present": primary["metrics"]["macro_f1"] - uniform_primary["metrics"]["macro_f1"],
            "delta_h12_vs_uniform_present": h12["metrics"]["macro_f1"] - uniform_h12["metrics"]["macro_f1"],
        }
    window_baseline = {(row["family"], row["window"]): row["macro_f1"] for row in windows_out
                       if row["model"] == "C4_HGB"}
    window_uniform = {(row["family"], row["window"]): row["macro_f1"] for row in windows_out
                      if row["model"] == "present_specialist"}
    for row in windows_out:
        if row["model"] != "C4_HGB":
            row["delta_vs_C4"] = row["macro_f1"] - window_baseline[(row["family"], row["window"])]
            row["delta_vs_present_specialist"] = row["macro_f1"] - window_uniform[(row["family"], row["window"])]
        else:
            row["delta_vs_C4"] = 0.0
            row["delta_vs_present_specialist"] = row["macro_f1"] - window_uniform[(row["family"], row["window"])]
    return {"aggregate": aggregate, "summary": summary, "windows": windows_out,
            "cycles": cycles_out, "subgroups": subgroups_out}


def reconcile_parent_results(evaluation: dict[str, Any]) -> dict[str, Any]:
    prior = read_json(SOURCE_REPORT / "results.json")
    comparisons = []
    for family, prior_family in (("primary", "primary"), ("h12", "h12")):
        for local_name, prior_name in (("C4_HGB", "C4_HGB"), ("present_specialist", "present_specialist")):
            expected = prior["aggregate"][prior_family][prior_name]
            actual = evaluation["aggregate"][family][local_name]
            if not np.isclose(actual["metrics"]["macro_f1"], expected["metrics"]["macro_f1"], atol=1e-12, rtol=0):
                raise ValueError(f"Reused {local_name}/{family} does not match the original report")
            if not np.isclose(actual["equal_cycle_macro_f1"], expected["equal_cycle_macro_f1"], atol=1e-12, rtol=0):
                raise ValueError(f"Reused equal-cycle {local_name}/{family} does not match the original report")
            comparisons.append({"family": family, "model": local_name,
                                "macro_f1": actual["metrics"]["macro_f1"],
                                "equal_cycle_macro_f1": actual["equal_cycle_macro_f1"],
                                "matches_original_report": True})
    return {"status": "PASS", "comparisons": comparisons}


def make_figure(evaluation: dict[str, Any], path: Path) -> None:
    window_rows = [row for row in evaluation["windows"] if row["family"] == "primary"]
    order = ["W1", "W2", "W3", "W4", "W5", "W6"]
    colors = {"C4_HGB": "#666666", "present_specialist": "#4C91C6", "present_q80_route": "#D6763B"}
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2), constrained_layout=True)
    for name in ("present_specialist", "present_q80_route"):
        selected = {row["window"]: row for row in window_rows if row["model"] == name}
        values = [100 * selected[window]["delta_vs_C4"] for window in order]
        label = name.replace("_", " ") + " Δ vs C4"
        axes[0].plot(order, values, marker="o", linewidth=1.8, color=colors[name], label=label)
    axes[0].axhline(0, color="#444444", linewidth=1)
    axes[0].set_ylabel("Macro F1 change vs C4 (percentage points)")
    axes[0].set_title("Primary-window deltas")
    axes[0].grid(axis="y", alpha=0.25)
    axes[0].legend(frameon=False, fontsize=8)
    labels = ["Primary pooled", "Equal-cycle", "H12 pooled"]
    x = np.arange(len(labels))
    for name, offset in (("present_specialist", -0.10), ("present_q80_route", 0.10)):
        summary = evaluation["summary"][name]
        deltas = [summary["delta_primary_vs_C4"], summary["delta_equal_cycle_vs_C4"], summary["delta_h12_vs_C4"]]
        axes[1].plot(x + offset, 100 * np.asarray(deltas), linestyle="none", marker="o", markersize=7,
                     color=colors[name], label=name.replace("_", " "))
    axes[1].axhline(0, color="#444444", linewidth=1)
    axes[1].set_xticks(x, labels)
    axes[1].set_ylabel("Macro F1 change vs C4 (percentage points)")
    axes[1].set_title("Does routing only high-volume present rows help?")
    axes[1].grid(axis="y", alpha=0.25)
    axes[1].legend(frameon=False, fontsize=8)
    fig.suptitle("Fit-only q80 route composed from verified cached experts", fontsize=12.5)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def _format_metric(metric: dict[str, Any]) -> str:
    return f"{metric['macro_f1']:.6f} (n={metric['n']:,})"


def render_findings(results: dict[str, Any], freeze: dict[str, Any]) -> str:
    a = results["aggregate"]
    summary = results["summary"]
    lines = [
        "# Classification follow-up: fit-only q80 route for present streams",
        "",
        "## Adaptive question and fixed rule",
        "",
        "The earlier uniform present-specialist was close to C4 overall, with localized gains in the highest fit-defined text_load quintile. This limited follow-up asks whether routing only present-stream rows above the fit-only 80th percentile to that specialist can retain those gains while leaving other rows on the established C4 experts.",
        "",
        "For each temporal fit cutoff, q80 is `np.quantile(text_load[cycle_code <= fit_end], 0.80)`. Stream-missing rows route first to all-row C1_B. Present rows strictly above q80 route to present-only C2_B; equality and lower values use all-row C2_B. The quantile and route were fixed before scoring. This is an adaptive follow-up on already viewed Train folds, not an independent test.",
        "",
        "The comparison only composes three previously audited prediction vectors. It performs zero new temporal model fits and leaves all original fit caches and receipts read-only.",
        "",
        "## Pooled results",
        "",
        "| Route | Primary Macro F1 | Δ vs C4 | Equal-cycle | H12 Macro F1 | Δ H12 vs C4 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name in MODELS:
        s = summary[name]
        lines.append(f"| {name} | {s['primary_macro_f1']:.6f} | {s['delta_primary_vs_C4']:+.6f} | "
                     f"{s['primary_equal_cycle_macro_f1']:.6f} ({a['primary'][name]['per_cycle_count']} cycles) | "
                     f"{s['h12_macro_f1']:.6f} | {s['delta_h12_vs_C4']:+.6f} |")
    lines += ["", "![Temporal result](figures/q80_route_effects.png)", "",
              "Per-window chronology and class metrics are in `windows.csv`; cycle-level results are in `per_cycle.csv`.",
              "", "## Route and subgroup checks", "",
              "The q80 route is exactly identical to C4 on every missing-stream row and every present row at or below the cutoff. On high-volume present rows it is exactly identical to the uniform present-specialist. Denominators and full per-class precision/recall/F1, positive rates, false positives, and false negatives for stream-missing, present-low, present-high, and text_load quintiles are in `subgroups.csv`.",
              "",
              "The fit-only q80 thresholds and train/validation branch counts are frozen in `freeze.json`. All cutoffs used thresholds computed only from rows available by that cutoff; q80 was not selected using validation labels.",
              "", "## Interpretation", "",
              "This route is an adaptive ablation prompted by the earlier Q5 result. Any improvement would localize to the present/high-volume branch by construction; it would not establish that high text volume causes better predictions. Validation records and public Kaggle outcomes did not influence this route. Primary and H12 reuse development records, so the results are not independent confirmation.",
              "",
              f"Input Train SHA-256 `{freeze['official_train']['sha256']}`. Extension freeze `{freeze['freeze_signature']}`. New model fits: 0; official Test read: no; Kaggle submissions: 0.",
              "",
              "MAIN retains all candidate selection and external evaluation decisions.", ""]
    return "\n".join(lines)


def score() -> None:
    verify_windows_against_round2()
    frame, target, source = feature_frame(include_target=True)
    assert target is not None
    frozen = verify_freeze(frame, source)
    if RUN_DIR.exists() and any(RUN_DIR.rglob("*.npz")):
        raise FileExistsError(f"Refusing to overwrite volume-route predictions: {RUN_DIR}")
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    folds = {fit_end: read_fold_vectors(fit_end, frame, frozen) for fit_end in FIT_ENDS}
    for fit_end, fold in folds.items():
        np.savez_compressed(RUN_DIR / f"fit{fit_end}.npz", rows=fold["rows"],
                             probability=fold["present_q80_route"])
    evaluation = evaluate(frame, target, folds)
    source_reconciliation = reconcile_parent_results(evaluation)
    write_json(REPORT_DIR / "results.json", {
        "status": "SCORED_FROM_VERIFIED_CACHED_BRANCH_VECTORS",
        "protocol_id": read_json(PROTOCOL_PATH)["protocol_id"],
        "freeze_signature": frozen["freeze_signature"],
        "input_sha256": frozen["official_train"]["sha256"],
        "new_temporal_model_fits": 0,
        "official_test_values_read": False,
        "kaggle_submissions": 0,
        "source_baseline_reconciliation": source_reconciliation,
        **evaluation,
        "prediction_cache_sha256": {
            str(fit_end): file_sha256(RUN_DIR / f"fit{fit_end}.npz") for fit_end in FIT_ENDS
        },
    })
    write_csv(REPORT_DIR / "summary.csv", [{"model": name, **value}
                                            for name, value in evaluation["summary"].items()])
    write_csv(REPORT_DIR / "windows.csv", evaluation["windows"])
    write_csv(REPORT_DIR / "per_cycle.csv", evaluation["cycles"])
    write_csv(REPORT_DIR / "subgroups.csv", evaluation["subgroups"])
    make_figure(evaluation, REPORT_DIR / "figures" / "q80_route_effects.png")
    results = read_json(REPORT_DIR / "results.json")
    (REPORT_DIR / "FINDINGS.md").write_text(render_findings(results, frozen), encoding="utf-8")
    print(json.dumps({"status": results["status"], "new_model_fits": 0,
                      "summary": results["summary"], "source_reconciliation": source_reconciliation},
                     indent=2, ensure_ascii=False))


def main() -> None:
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--freeze", action="store_true", help="freeze the q80 composition before scoring")
    group.add_argument("--score", action="store_true", help="score the frozen composition from old caches")
    args = parser.parse_args()
    freeze() if args.freeze else score()


if __name__ == "__main__":
    main()
