"""Four-fit, Train-only pilot for ordered target-stat token summaries."""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import argparse
import hashlib
import importlib.metadata
import json
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import (average_precision_score, brier_score_loss,
                             confusion_matrix, f1_score,
                             precision_recall_fscore_support)
from threadpoolctl import threadpool_limits

from scripts import optimization_v3_character as reg_pilot
from scripts import optimization_v3_distributional as reg_parent
from scripts import optimization_v3_reg_diagnostics as reg_diag
from scripts import optimization_v3_sparse_classification as cls_sparse
from scripts.optimization_v2_classification import load_train as load_cls_train
from src.models.classification_round2_models import stream_missing_mask
from src.models.optimization_v3_supervised import OrderedStatsBoost
from src.models.regression_research_text import HYBRID_FEATURES
from src.models.round1_evaluation import array_digest, code_digest, digest_json
from src.models.round1_models import _fit_fingerprint

CONFIG = ROOT / "experiments/optimization_v3/supervised_pilot"
OUT = ROOT / "reports/optimization_v3/supervised_pilot"
CACHE = ROOT / "experiments/runs/optimization_v3_supervised_pilot"
ENDPOINTS = [55, 67]
TASKS = ["classification", "regression"]
THRESHOLD = 0.50
BLEND_NEW_WEIGHT = 0.25
VIEW_SPECS = {
    "W1": (55, 56, 59),
    "W4": (67, 68, 71),
    "H12_E": (55, 56, 67),
    "H12_L": (67, 68, 79),
}
GROUPS = ["stream_missing", "stream_present"]
CODE_FILES = [
    "src/models/optimization_v3_token_stats.py",
    "src/models/optimization_v3_supervised.py",
    "scripts/optimization_v3_supervised_pilot.py",
    "scripts/optimization_v3_verify_supervised_pilot.py",
    "tests/test_optimization_v3_supervised_pilot.py",
    "src/models/optimization_v3_sparse_classification.py",
    "scripts/optimization_v3_sparse_classification.py",
    "scripts/optimization_v3_verify_sparse_classification.py",
    "scripts/optimization_v3_sparse_classification_fullsix.py",
    "scripts/optimization_v3_verify_sparse_classification_fullsix.py",
    "scripts/optimization_v3_verify_sparse_classification_fullsix_wrapper.py",
    "src/models/classification_models.py",
    "src/models/classification_round2_models.py",
    "scripts/optimization_v2_classification.py",
    "scripts/research_campaign_classification_stage2.py",
    "scripts/optimization_v3_character.py",
    "scripts/optimization_v3_distributional.py",
    "scripts/optimization_v3_reg_diagnostics.py",
    "src/models/optimization_v3_character.py",
    "src/models/optimization_v3_distributional.py",
    "src/models/regression_research_common.py",
    "src/models/regression_research_text.py",
    "src/models/regression_round2_models.py",
    "src/models/regression_round2_metrics.py",
    "src/models/round1_models.py",
    "src/models/round1_evaluation.py",
    "src/utils/paths.py",
]


def sha(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(path: Path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_new(path: Path, payload) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=True, allow_nan=False)
        handle.write("\n")


def head_contextual() -> str:
    # Context only. Source code hashes, not HEAD, govern the frozen identity.
    return subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip()


def identity_input_paths(class_train: Path, reg_train: Path) -> list[Path]:
    class_pilot = cls_sparse.CONFIG
    class_full_config = ROOT / "experiments/optimization_v3/classification_sparse_full"
    class_full_report = ROOT / "reports/optimization_v3/classification_sparse_full"
    class_full_cache = ROOT / "experiments/runs/optimization_v3_classification_sparse_full"
    paths = [
        class_train, reg_train,
        class_pilot / "protocol.json", class_pilot / "freeze.json",
        cls_sparse.OUT / "pilot.json", cls_sparse.OUT / "verification.json",
        class_full_config / "protocol.json", class_full_config / "freeze.json",
        class_full_report / "fullsix.json", class_full_report / "verification.json",
        class_full_report / "verification_wrapper_run.json",
        reg_pilot.CONFIG / "protocol.json", reg_pilot.CONFIG / "freeze.json",
        reg_pilot.OUT / "pilot.json", reg_pilot.OUT / "verification.json",
        reg_diag.OUT_DIR / "diagnostics.json", reg_diag.RUN_DIR / "oof_residuals.npz",
        reg_diag.PRIMARY_CACHE, reg_diag.PILOT_CACHE, reg_diag.FULL_CACHE,
        CONFIG / "protocol.json", CONFIG / "dry_run.json",
    ]
    for end in ENDPOINTS:
        paths += [
            cls_sparse.CACHE / f"fit{end}.npz", cls_sparse.CACHE / f"fit{end}.json",
            reg_pilot.CACHE / f"fit{end}.npz", reg_pilot.CACHE / f"fit{end}.json",
        ]
    for end in [59, 63, 71, 75]:
        paths += [class_full_cache / f"fit{end}.npz", class_full_cache / f"fit{end}.json"]
    return paths


def load_official_data():
    class_frame, class_y, class_train_path = load_cls_train(None)
    reg_frame, reg_train_path = reg_parent.load()
    if len(class_frame) != 50608 or len(class_y) != 50608:
        raise AssertionError("Official Classification Train must have 50,608 rows")
    if len(reg_frame) != 50608:
        raise AssertionError("Official Regression Train must have 50,608 rows")
    if set(pd.to_numeric(class_frame.cycle_code, errors="raise").astype(int)) != set(range(47, 80)):
        raise AssertionError("Classification Train cycles must be 47-79")
    if set(pd.to_numeric(reg_frame.cycle_code, errors="raise").astype(int)) != set(range(47, 80)):
        raise AssertionError("Regression Train cycles must be 47-79")
    return class_frame, np.asarray(class_y, dtype=np.int8), class_train_path, reg_frame, reg_train_path


def make_identity() -> dict:
    class_frame, class_y, class_train, reg_frame, reg_train = load_official_data()
    cls_freeze = cls_sparse.check()
    cls_pilot_report = read_json(cls_sparse.OUT / "pilot.json")
    cls_verify = read_json(cls_sparse.OUT / "verification.json")
    cls_full_verification = read_json(ROOT / "reports/optimization_v3/classification_sparse_full/verification_wrapper_run.json")
    reg_freeze = reg_pilot.check()
    reg_pilot_report = read_json(reg_pilot.OUT / "pilot.json")
    reg_verify = read_json(reg_pilot.OUT / "verification.json")
    reg_diagnostics = read_json(reg_diag.OUT_DIR / "diagnostics.json")
    if (cls_verify.get("status") != "PASS" or cls_verify.get("newfits") != 0 or
            cls_pilot_report.get("Test_read") is not False or
            cls_full_verification.get("status") != "PASS" or
            reg_verify.get("status") != "PASS" or reg_verify.get("newfits") != 0 or
            reg_pilot_report.get("Test_read") is not False or
            reg_diagnostics.get("scope", {}).get("test_read") is not False):
        raise AssertionError("Required existing Classification/Regression control receipts must pass")
    paths = identity_input_paths(class_train, reg_train)
    absent = [str(path) for path in paths if not path.is_file()]
    if absent:
        raise FileNotFoundError(f"Supervised pilot frozen inputs missing: {absent}")
    return {
        "protocol_sha256_exact_bytes": sha(CONFIG / "protocol.json"),
        "source_code_sha256_lf_utf8": {path: code_digest(ROOT / path) for path in CODE_FILES},
        "input_sha256_exact_bytes": {
            path.resolve().relative_to(ROOT).as_posix(): sha(path) for path in paths
        },
        "classification_official_train_sha256": sha(class_train),
        "regression_official_train_sha256": sha(reg_train),
        "classification_rows": len(class_frame),
        "regression_rows": len(reg_frame),
        "classification_sparse_pilot_signature": cls_freeze["signature"],
        "classification_sparse_fullsix_verification_status": cls_full_verification["status"],
        "regression_character_pilot_signature": reg_freeze["signature"],
        "regression_character_pilot_verification_status": reg_verify["status"],
        "packages": {
            name: importlib.metadata.version(name)
            for name in ["numpy", "pandas", "scipy", "scikit-learn", "lightgbm", "threadpoolctl"]
        },
        "test_read": False,
        "kaggle_access": False,
    }


def _synthetic_frame(n: int = 120) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    i = np.arange(n)
    texts = np.array([
        f"shared vulnerability alpha token{i[j] % 7} evidence pattern{i[j] % 5}"
        if j % 13 else ""
        for j in range(n)
    ], dtype=object)
    texts[9] = None
    frame = pd.DataFrame({
        "impact_load": 20.0 + (i % 17),
        "entry_friction": (i % 2).astype(float),
        "cycle_code": 47 + (i // 20),
        "scope_pressure": 40.0 + (i % 19),
        "text_load": 15.0 + (i % 11),
        "scope_trace": [f"{j % 4} nodes" for j in i],
        "org_spread": [str(j % 6) for j in i],
        "module_spread": [f"{j % 5} nodes" for j in i],
        "telemetry_flag": ["active" if j % 2 else "inactive" for j in i],
        "org_code": [f"org-{j % 6}" for j in i],
        "module_code": [f"module-{j % 5}" for j in i],
        "pattern_code": [f"pattern-{j % 4}" for j in i],
        "signal_stream": texts,
    })
    class_y = (i % 3 == 0).astype(np.int8)
    reg_y = ((i * 37) % 101).astype(float)
    return frame, class_y, reg_y


def classification_fit_mask(frame: pd.DataFrame, end: int) -> np.ndarray:
    cycles = pd.to_numeric(frame.cycle_code, errors="raise").to_numpy(dtype=int)
    return (cycles >= 47) & (cycles <= int(end)) & ~stream_missing_mask(frame)


def token_boundary_preflight() -> dict:
    """Independent perturbation fixture: labels from cycle c cannot affect c's encodings."""
    from src.models.optimization_v3_token_stats import OrderedTokenStats

    texts = ["shared alpha token"] * 12
    cycles = np.repeat([47, 48, 49], 4)
    target = np.array([0., 0., 1., 1., 1., 1., 0., 0., 0., 0., 0., 0.])
    first = OrderedTokenStats()
    original = first.fit_transform(texts, cycles, target)
    changed_target = target.copy()
    changed_target[4:8] = np.array([0., 0., 0., 0.])
    second = OrderedTokenStats()
    changed = second.fit_transform(texts, cycles, changed_target)
    np.testing.assert_array_equal(original[:8], changed[:8])
    if np.array_equal(original[8:], changed[8:]):
        raise AssertionError("Perturbed cycle labels should alter a later cycle's supported-token features")
    expected_seen = {str(int(c)): int(np.sum(cycles < c)) for c in np.unique(cycles)}
    for lineage in first.audit_state()["lineage"]:
        if lineage["labels_previously_seen"] != expected_seen[str(int(lineage["cycle"]))]:
            raise AssertionError("Token lineage does not equal the strict-earlier-cycle fit population")
    return {
        "status": "PASS",
        "same_cycle_label_perturbation_leaves_current_and_past_features_unchanged": True,
        "later_cycle_supported_token_features_respond_to_past_label_perturbation": True,
        "labels_previously_seen_matches_manual_cycle_counts": True,
        "manual_labels_previously_seen": expected_seen,
    }


def synthetic_model_preflight() -> dict:
    frame, class_y, reg_y = _synthetic_frame()
    train = frame.cycle_code <= 51
    hold = ~train
    result = {"status": "PASS", "official_data_used": False, "test_read": False, "tasks": {}}
    for task, target in [("classification", class_y), ("regression", reg_y)]:
        model = OrderedStatsBoost(task)
        fields = list(model.input_feature_names)
        task_train = train & (~stream_missing_mask(frame) if task == "classification" else True)
        model.fit(frame.loc[task_train, fields], target[task_train])
        before = model.audit_state()
        output = model.predict_proba(frame.loc[hold, fields]) if task == "classification" else model.predict(frame.loc[hold, fields])
        after = model.audit_state()
        if before != after or output.shape != (int(hold.sum()),) or not np.isfinite(output).all():
            raise AssertionError(f"Synthetic {task} model schema/preprocessing preflight failed")
        if task == "classification" and np.any((output < 0) | (output > 1)):
            raise AssertionError("Synthetic classification preflight returned invalid probabilities")
        line = before["ordered_tokens"]["lineage"]
        cycles = frame.loc[task_train, "cycle_code"].to_numpy(dtype=int)
        if any(item["labels_previously_seen"] != int((cycles < item["cycle"]).sum()) for item in line):
            raise AssertionError(f"Synthetic {task} encoder lineage boundary failed")
        result["tasks"][task] = {
            "fit_rows": int(task_train.sum()), "validation_rows": int(hold.sum()),
            "feature_count": int(len(fields)), "output_finite": True,
            "fit_state_immutable_during_predict": True,
            "fit_population": "present_only" if task == "classification" else "all_rows",
            "strict_earlier_cycle_lineage": True,
        }
    # Exercise both policy controllers on synthetic aligned rows.
    missing = np.array([False, True, False, True])
    raw_prob = np.array([0.2, 0.9, 0.7, 0.1])
    c2b = np.array([0.4, 0.2, 0.6, 0.4])
    latest = np.array([0.35, 0.1, 0.8, 0.2])
    cb = np.array([0.5, 0.5, 0.5, 0.5])
    routed = classification_routes(missing, raw_prob, c2b, latest, cb)
    if not np.array_equal(routed["blend25"][missing], cb[missing]):
        raise AssertionError("Synthetic missing classification branch changed")
    raw = np.array([1.0, 10.0, 20.0, 70.0])
    cycles = np.array([60, 60, 61, 61])
    ranked = rank50_within_cycle(raw, cycles)
    mixture = regression_candidates(np.array([5.0, 15.0, 25.0, 75.0]), raw, cycles)
    if not np.array_equal(mixture["rank50"], ranked) or np.any((mixture["blend25"] < 0) | (mixture["blend25"] > 100)):
        raise AssertionError("Synthetic regression rank/mixture schema failed")
    result["route_and_mixture_schemas"] = "PASS"
    result["ordered_token_boundary"] = token_boundary_preflight()
    return result


def dry_run() -> dict:
    if CACHE.exists() or (CONFIG / "freeze.json").exists():
        raise AssertionError("Dry-run must occur before official fit freeze/cache")
    result = synthetic_model_preflight()
    path = CONFIG / "dry_run.json"
    if path.exists():
        if read_json(path) != result:
            raise AssertionError("Existing prefit dry-run receipt differs; preserve it and amend under a new name")
    else:
        write_new(path, result)
    return result


def freeze() -> dict:
    path = CONFIG / "freeze.json"
    if not (CONFIG / "dry_run.json").is_file() or read_json(CONFIG / "dry_run.json").get("status") != "PASS":
        raise AssertionError("Run the fixture dry-run before freezing official fits")
    identity = make_identity()
    if path.exists():
        old = read_json(path)
        if old.get("identity") != identity or old.get("signature") != digest_json(identity):
            raise AssertionError("Existing supervised pilot freeze differs")
        return old
    if CACHE.exists():
        raise AssertionError("Official prediction cache must not exist before freeze")
    frozen = {
        "status": "FROZEN_BEFORE_FOUR_OFFICIAL_FITS",
        "at_utc": datetime.now(timezone.utc).isoformat(),
        "source_head_contextual": head_contextual(),
        "identity": identity,
        "signature": digest_json(identity),
        "prefit_dry_run_sha256": sha(CONFIG / "dry_run.json"),
        "prefit_dry_run_status": "PASS",
    }
    write_new(path, frozen)
    return frozen


def check_freeze() -> dict:
    frozen = read_json(CONFIG / "freeze.json")
    identity = make_identity()
    if frozen.get("identity") != identity or frozen.get("signature") != digest_json(identity):
        raise AssertionError("Supervised pilot frozen source/data/runtime identity changed")
    if frozen.get("prefit_dry_run_status") != "PASS" or frozen.get("prefit_dry_run_sha256") != sha(CONFIG / "dry_run.json"):
        raise AssertionError("Prefit schema dry-run receipt mismatch")
    return frozen


def classification_routes(missing: np.ndarray, raw: np.ndarray, c2b: np.ndarray,
                          latest: np.ndarray, cb: np.ndarray) -> dict[str, np.ndarray]:
    missing = np.asarray(missing, dtype=bool)
    raw, c2b, latest, cb = [np.asarray(value, dtype=float) for value in [raw, c2b, latest, cb]]
    if missing.ndim != 1 or any(array.shape != missing.shape for array in [raw, c2b, latest, cb]):
        raise ValueError("Classification route vectors must be aligned")
    if any(not np.isfinite(array).all() for array in [raw, c2b, latest, cb]):
        raise ValueError("Classification route vectors must be finite")
    direct = np.where(missing, cb, raw)
    blended = np.where(missing, cb, 0.75 * latest + BLEND_NEW_WEIGHT * raw)
    return {
        "initial_control": np.where(missing, cb, c2b),
        "latest_sparse25": np.where(missing, cb, latest),
        "direct": direct,
        "blend25": blended,
    }


def rank50_within_cycle(raw: np.ndarray, cycles: np.ndarray) -> np.ndarray:
    raw = np.asarray(raw, dtype=float)
    cycles = np.asarray(cycles)
    if raw.ndim != 1 or cycles.shape != raw.shape or not np.isfinite(raw).all():
        raise ValueError("Rank mapping requires aligned finite predictions and cycles")
    percentile = np.empty(len(raw), dtype=float)
    for cycle in np.unique(cycles):
        mask = cycles == cycle
        percentile[mask] = 100.0 * rankdata(raw[mask], method="average") / int(mask.sum())
    return np.clip(0.5 * raw + 0.5 * percentile, 0.0, 100.0)


def regression_candidates(champion: np.ndarray, raw: np.ndarray, cycles: np.ndarray) -> dict[str, np.ndarray]:
    champion = np.asarray(champion, dtype=float)
    raw = np.asarray(raw, dtype=float)
    cycles = np.asarray(cycles)
    if champion.ndim != 1 or raw.shape != champion.shape or cycles.shape != champion.shape:
        raise ValueError("Regression candidate, champion, and cycle vectors must align")
    if not np.isfinite(champion).all() or not np.isfinite(raw).all():
        raise ValueError("Regression prediction arrays must be finite")
    rank = rank50_within_cycle(raw, cycles)
    blend = np.clip(0.75 * champion + 0.25 * rank, 0.0, 100.0)
    return {"raw": raw, "rank50": rank, "blend25": blend}


def get_class_controls(frame: pd.DataFrame, target: np.ndarray) -> dict[int, dict]:
    receipt = cls_sparse.check()
    report = read_json(cls_sparse.OUT / "pilot.json")
    verification = read_json(cls_sparse.OUT / "verification.json")
    if report.get("freeze_signature") != receipt["signature"] or verification.get("status") != "PASS":
        raise AssertionError("Sparse pilot controls lack a matching independent verification")
    records = {}
    for end in ENDPOINTS:
        rows, controls, cb = cls_sparse.existing(frame, target, end)
        path = cls_sparse.CACHE / f"fit{end}.npz"
        meta = read_json(path.with_suffix(".json"))
        if meta.get("freeze_signature") != receipt["signature"] or meta.get("npz_sha256") != cls_sparse.sha(path):
            raise AssertionError(f"Classification sparse OOF control sidecar mismatch at {end}")
        with np.load(path, allow_pickle=False) as archive:
            pilot_arrays = {key: archive[key].copy() for key in archive.files}
        np.testing.assert_array_equal(pilot_arrays["rows"], rows)
        missing = cls_sparse.stream_missing_mask(frame.iloc[rows])
        initial = np.where(missing, cb, controls[:, 0])
        np.testing.assert_array_equal(initial, pilot_arrays["champion"])
        latest = pilot_arrays["blend25"].astype(float)
        np.testing.assert_array_equal(latest[missing], cb[missing])
        if not np.isfinite(latest).all() or np.any((latest < 0) | (latest > 1)):
            raise AssertionError(f"Invalid latest sparse25 control probability at {end}")
        records[end] = {
            "rows": rows,
            "c2b": controls[:, 0].astype(float),
            "cb": cb.astype(float),
            "missing": missing,
            "initial_control": initial.astype(float),
            "latest_control": latest,
            "pilot_state": meta["state"],
            "pilot_npz_sha256": cls_sparse.sha(path),
        }
    return records


def get_regression_views(frame: pd.DataFrame):
    with np.load(reg_diag.PRIMARY_CACHE, allow_pickle=False) as base, \
         np.load(reg_diag.PILOT_CACHE, allow_pickle=False) as pilot_cache, \
         np.load(reg_diag.FULL_CACHE, allow_pickle=False) as full_cache:
        primary, primary_audit = reg_diag._build_view(
            frame, base, pilot_cache, full_cache,
            [(55, 56, 59), (59, 60, 63), (63, 64, 67), (67, 68, 71), (71, 72, 75), (75, 76, 79)],
        )
        h12, h12_audit = reg_diag._build_view(frame, base, pilot_cache, full_cache, [(55, 56, 67), (67, 68, 79)])
    if primary.row.duplicated().any() or h12.row.duplicated().any():
        raise AssertionError("Regression current three-component controls have overlapping row ids")
    expected = np.flatnonzero(frame.cycle_code.between(56, 79).to_numpy())
    np.testing.assert_array_equal(np.sort(primary.row.to_numpy(dtype=np.int64)), expected)
    np.testing.assert_array_equal(np.sort(h12.row.to_numpy(dtype=np.int64)), expected)
    return primary, h12, {"primary": primary_audit, "H12": h12_audit}


def _fit_metadata(task: str, end: int, state: dict, seconds: float,
                  freeze_signature: str, arrays_path: Path, outputs: dict) -> dict:
    target_name = "artifact_signal" if task == "classification" else "threat_rank"
    return {
        "freeze_signature": freeze_signature,
        "task": task,
        "cutoff": end,
        "fit_cycles": [47, end],
        "validation_cycles": [end + 1, 79 if end == 67 else 67],
        "fit_rows": int(state["fit_rows"]),
        "fit_data_sha256": state["fit_data_sha256"],
        "target_column": target_name,
        "target_unit_for_token_stats": "raw binary target" if task == "classification" else "threat_rank/100",
        "ordered_token_state": state["ordered_tokens"],
        "full_model_state": state,
        "model_state_sha256": state["model_sha256"],
        "npz_sha256": sha(arrays_path),
        "array_sha256": {key: array_digest(value) for key, value in outputs.items()
                         if key not in {"rows", "missing"}},
        "seconds": float(seconds),
        "outer_labels_used_for_fit": False,
        "test_read": False,
    }


def _classification_fit(end: int, frame: pd.DataFrame, target: np.ndarray,
                        control: dict, freeze_signature: str) -> dict:
    fit = classification_fit_mask(frame, end)
    if int(fit.sum()) < 100 or len(np.unique(target[fit])) != 2:
        raise AssertionError(f"Classification present-only fit population invalid at cutoff {end}")
    features = list(HYBRID_FEATURES)
    model = OrderedStatsBoost("classification")
    started = time.perf_counter()
    with threadpool_limits(limits=2):
        model.fit(frame.loc[fit, features], target[fit])
        state = model.audit_state()
        raw = np.zeros(len(control["rows"]), dtype=float)
        present = ~control["missing"]
        raw[present] = model.predict_proba(frame.iloc[control["rows"][present]][features])
    if state != model.audit_state():
        raise AssertionError(f"Classification model audit state changed at cutoff {end}")
    if state["fit_data_sha256"] != _fit_fingerprint(frame.loc[fit, features], tuple(features), target[fit]):
        raise AssertionError(f"Classification fit fingerprint mismatch at cutoff {end}")
    if state["fit_rows"] != int(fit.sum()):
        raise AssertionError(f"Classification fit support mismatch at cutoff {end}")
    fit_cycles = pd.to_numeric(frame.loc[fit, "cycle_code"], errors="raise").to_numpy(dtype=int)
    expected_seen = {int(c): int(np.sum(fit_cycles < c)) for c in np.unique(fit_cycles)}
    for lineage in state["ordered_tokens"]["lineage"]:
        if lineage["labels_previously_seen"] != expected_seen[int(lineage["cycle"])]:
            raise AssertionError(f"Classification present-only token lineage mismatch at cutoff {end}")
    routes = classification_routes(control["missing"], raw, control["c2b"], control["latest_control"], control["cb"])
    cycles = frame.iloc[control["rows"]].cycle_code.to_numpy(dtype=int)
    if not np.array_equal(control["rows"], np.flatnonzero(frame.cycle_code.between(end + 1, 67 if end == 55 else 79).to_numpy())):
        raise AssertionError(f"Classification validation horizon misaligned at cutoff {end}")
    outputs = {
        "rows": control["rows"],
        "raw_probability": np.asarray(raw, dtype=float),
        "initial_control": routes["initial_control"],
        "latest_control": routes["latest_sparse25"],
        "direct": routes["direct"],
        "blend25": routes["blend25"],
        "c2b_present": control["c2b"],
        "cb_missing": control["cb"],
        "missing": control["missing"].astype(np.uint8),
        "cycles": cycles.astype(np.int16),
    }
    path = CACHE / f"classification_fit{end}.npz"
    CACHE.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **outputs)
    seconds = time.perf_counter() - started
    metadata = _fit_metadata("classification", end, state, seconds, freeze_signature, path, outputs)
    write_new(path.with_suffix(".json"), metadata)
    print(f"FITTED supervised classification cutoff={end} rows={len(control['rows'])} seconds={seconds:.1f}", flush=True)
    return {"end": end, "rows": control["rows"], "cycles": cycles, "raw_probability": np.asarray(raw, float),
            **routes, "missing": control["missing"], "state": state, "seconds": seconds,
            "cache_sha256": sha(path), "source": "new_fit"}


def _regression_fit(end: int, frame: pd.DataFrame, target: np.ndarray,
                    control_view: pd.DataFrame, freeze_signature: str) -> dict:
    rows = control_view.row.to_numpy(dtype=np.int64)
    champion = control_view.prediction.to_numpy(dtype=float)
    fit = frame.cycle_code.between(47, end).to_numpy()
    features = list(HYBRID_FEATURES)
    model = OrderedStatsBoost("regression")
    started = time.perf_counter()
    with threadpool_limits(limits=2):
        model.fit(frame.loc[fit, features], target[fit])
        state = model.audit_state()
        raw = model.predict(frame.iloc[rows][features])
    if state != model.audit_state():
        raise AssertionError(f"Regression model audit state changed at cutoff {end}")
    if state["fit_data_sha256"] != _fit_fingerprint(frame.loc[fit, features], tuple(features), target[fit]):
        raise AssertionError(f"Regression fit fingerprint mismatch at cutoff {end}")
    cycles = frame.iloc[rows].cycle_code.to_numpy(dtype=int)
    predictions = regression_candidates(champion, raw, cycles)
    horizon = 67 if end == 55 else 79
    if not np.array_equal(rows, np.flatnonzero(frame.cycle_code.between(end + 1, horizon).to_numpy())):
        raise AssertionError(f"Regression validation horizon misaligned at cutoff {end}")
    outputs = {
        "rows": rows,
        "raw": np.asarray(raw, dtype=float),
        "rank50": predictions["rank50"],
        "blend25": predictions["blend25"],
        "champion": champion,
        "cycles": cycles.astype(np.int16),
    }
    path = CACHE / f"regression_fit{end}.npz"
    CACHE.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **outputs)
    seconds = time.perf_counter() - started
    metadata = _fit_metadata("regression", end, state, seconds, freeze_signature, path, outputs)
    write_new(path.with_suffix(".json"), metadata)
    print(f"FITTED supervised regression cutoff={end} rows={len(rows)} seconds={seconds:.1f}", flush=True)
    return {"end": end, "rows": rows, "cycles": cycles, **predictions, "champion": champion,
            "state": state, "seconds": seconds, "cache_sha256": sha(path), "source": "new_fit"}


def prepare_output_directories_before_fits() -> None:
    expected = [
        CACHE / f"{task}_fit{end}.{suffix}"
        for task in TASKS for end in ENDPOINTS for suffix in ["npz", "json"]
    ] + [OUT / "pilot.json", OUT / "FULLPILOT.md", OUT / "verification.json"]
    existing = [str(path) for path in expected if path.exists()]
    if existing:
        raise FileExistsError(
            "Refusing to fit while prior official outputs already exist; preserve them and use a separately versioned continuation: "
            + ", ".join(existing)
        )
    CACHE.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)


def class_metrics(actual: np.ndarray, probability: np.ndarray) -> dict:
    y = np.asarray(actual, dtype=np.int8)
    p = np.asarray(probability, dtype=float)
    if y.ndim != 1 or p.shape != y.shape or len(y) == 0 or not np.isfinite(p).all():
        raise ValueError("Classification metric vectors must be aligned and finite")
    if np.any((p < 0) | (p > 1)):
        raise ValueError("Classification probabilities must be in [0,1]")
    pred = p >= THRESHOLD
    cm = confusion_matrix(y, pred, labels=[0, 1])
    precision, recall, f1, support = precision_recall_fscore_support(y, pred, labels=[0, 1], zero_division=0)
    return {
        "n": int(len(y)),
        "macro_f1": float(f1_score(y, pred, average="macro", zero_division=0)),
        "confusion_matrix": cm.astype(int).tolist(),
        "per_class": [{"label": label, "support": int(support[label]), "precision": float(precision[label]),
                       "recall": float(recall[label]), "f1": float(f1[label])} for label in [0, 1]],
        "positive_rate": float(y.mean()),
        "positive_prediction_rate": float(pred.mean()),
        "average_precision": float(average_precision_score(y, p)) if len(np.unique(y)) == 2 else None,
        "brier_score": float(brier_score_loss(y, p)),
        "threshold": THRESHOLD,
    }


def compare_class_view(frame: pd.DataFrame, target: np.ndarray, rows: np.ndarray,
                      policies: dict[str, np.ndarray]) -> dict:
    rows = np.asarray(rows, dtype=np.int64)
    missing = stream_missing_mask(frame.iloc[rows])
    groups = {"all": np.ones(len(rows), dtype=bool), "stream_present": ~missing, "stream_missing": missing}
    result = {"n": int(len(rows)), "metrics": {}, "subgroups": {}}
    for policy, p in policies.items():
        result["metrics"][policy] = class_metrics(target[rows], p)
    for group_name, mask in groups.items():
        result["subgroups"][group_name] = {
            "n": int(mask.sum()),
            "metrics": {policy: class_metrics(target[rows][mask], p[mask])
                        for policy, p in policies.items()},
        }
    result["comparisons"] = {}
    for candidate in ["direct", "blend25"]:
        for control in ["initial_control", "latest_sparse25"]:
            a = result["metrics"][control]
            b = result["metrics"][candidate]
            result["comparisons"][f"{candidate}_vs_{control}"] = {
                "macro_f1_delta": b["macro_f1"] - a["macro_f1"],
                "average_precision_delta": b["average_precision"] - a["average_precision"],
                "brier_delta_candidate_minus_control": b["brier_score"] - a["brier_score"],
            }
    return result


def regression_error(target: np.ndarray, prediction: np.ndarray) -> dict:
    y = np.asarray(target, dtype=float)
    p = np.asarray(prediction, dtype=float)
    if y.ndim != 1 or p.shape != y.shape or len(y) == 0 or not np.isfinite(p).all():
        raise ValueError("Regression metric vectors must be aligned and finite")
    err = p - y
    ae = np.abs(err)
    return {
        "n": int(len(y)),
        "mae": float(ae.mean()),
        "rmse": float(np.sqrt(np.mean(err ** 2))),
        "bias_prediction_minus_target": float(err.mean()),
        "median_signed_residual": float(np.median(err)),
        "p90_absolute_error": float(np.quantile(ae, 0.90)),
        "p95_absolute_error": float(np.quantile(ae, 0.95)),
        "over_rate": float(np.mean(err > 0)),
        "under_rate": float(np.mean(err < 0)),
        "tie_rate": float(np.mean(err == 0)),
    }


def compare_reg_view(frame: pd.DataFrame, target: np.ndarray, rows: np.ndarray,
                     predictions: dict[str, np.ndarray]) -> dict:
    rows = np.asarray(rows, dtype=np.int64)
    y = target[rows]
    cycles = pd.to_numeric(frame.iloc[rows].cycle_code, errors="raise").to_numpy(dtype=int)
    entry = pd.to_numeric(frame.iloc[rows].entry_friction, errors="coerce").to_numpy(dtype=float)
    stream = frame.iloc[rows].signal_stream
    missing = stream.isna().to_numpy() | stream.fillna("").astype(str).str.strip().eq("").to_numpy()
    group_masks = {
        "EF0": entry == 0,
        "EF1": entry == 1,
        "tail_le5": y <= 5,
        "tail_ge95": y >= 95,
        "stream_missing": missing,
        "stream_present": ~missing,
    }
    result = {"n": int(len(rows)), "metrics": {}, "per_cycle": {}, "subgroups": {}}
    for name, p in predictions.items():
        overall = regression_error(y, p)
        per_cycle = {str(int(c)): regression_error(y[cycles == c], p[cycles == c]) for c in np.unique(cycles)}
        overall["equal_cycle_mae"] = float(np.mean([r["mae"] for r in per_cycle.values()]))
        result["metrics"][name] = overall
        result["per_cycle"][name] = per_cycle
    champion = result["metrics"]["champion"]
    result["delta_vs_champion"] = {}
    for name in ["raw", "rank50", "blend25"]:
        metric = result["metrics"][name]
        result["delta_vs_champion"][name] = {
            "mae_gain_champion_minus_candidate": champion["mae"] - metric["mae"],
            "rmse_gain_champion_minus_candidate": champion["rmse"] - metric["rmse"],
            "bias_shift_candidate_minus_champion": metric["bias_prediction_minus_target"] - champion["bias_prediction_minus_target"],
            "equal_cycle_mae_gain": champion["equal_cycle_mae"] - metric["equal_cycle_mae"],
        }
    for group, mask in group_masks.items():
        result["subgroups"][group] = {
            "n": int(mask.sum()),
            "metrics": {name: regression_error(y[mask], p[mask]) for name, p in predictions.items()},
            "mae_gain_vs_champion": {
                name: regression_error(y[mask], predictions["champion"])["mae"] -
                      regression_error(y[mask], predictions[name])["mae"]
                for name in predictions if name != "champion"
            },
        }
    return result


def build_class_views(frame: pd.DataFrame, target: np.ndarray,
                      records: dict[int, dict]) -> dict:
    cycles = pd.to_numeric(frame.cycle_code, errors="raise").to_numpy(dtype=int)
    views = {}
    primary_parts = []
    h12_parts = []
    for end, early_name, late_name in [(55, "W1", "H12_E"), (67, "W4", "H12_L")]:
        record = records[end]
        cycle = cycles[record["rows"]]
        wstart, wstop = (56, 59) if end == 55 else (68, 71)
        wmask = (cycle >= wstart) & (cycle <= wstop)
        part_rows = record["rows"][wmask]
        part_policies = {name: values[wmask] for name, values in record["policies"].items()}
        expected = np.flatnonzero(frame.cycle_code.between(wstart, wstop).to_numpy())
        np.testing.assert_array_equal(part_rows, expected)
        views[early_name] = compare_class_view(frame, target, part_rows, part_policies)
        primary_parts.append((part_rows, part_policies))
        if end == 55:
            e_start, e_stop = 56, 67
        else:
            e_start, e_stop = 68, 79
        emask = (cycle >= e_start) & (cycle <= e_stop)
        hrows = record["rows"][emask]
        hpol = {name: value[emask] for name, value in record["policies"].items()}
        views[late_name] = compare_class_view(frame, target, hrows, hpol)
        h12_parts.append((hrows, hpol))
    p_rows = np.concatenate([part[0] for part in primary_parts])
    if len(np.unique(p_rows)) != len(p_rows):
        raise AssertionError("Classification W1/W4 pilot rows overlap")
    p_policies = {name: np.concatenate([part[1][name] for part in primary_parts])
                  for name in primary_parts[0][1]}
    views["primary_pilot"] = compare_class_view(frame, target, p_rows, p_policies)
    h_rows = np.concatenate([part[0] for part in h12_parts])
    if len(np.unique(h_rows)) != len(h_rows):
        raise AssertionError("Classification H12 pilot rows overlap")
    h_policies = {name: np.concatenate([part[1][name] for part in h12_parts]) for name in h12_parts[0][1]}
    views["H12"] = compare_class_view(frame, target, h_rows, h_policies)
    return views


def build_reg_views(frame: pd.DataFrame, target: np.ndarray,
                    records: dict[int, dict], primary: pd.DataFrame,
                    h12: pd.DataFrame) -> dict:
    views = {}
    primary_parts = []
    for name, end, low, high in [("W1", 55, 56, 59), ("W4", 67, 68, 71)]:
        base = primary.loc[(primary.cutoff == end) & primary.cycle.between(low, high)].copy()
        record = records[end]
        cycle = record["cycles"]
        mask = (cycle >= low) & (cycle <= high)
        rows = record["rows"][mask]
        np.testing.assert_array_equal(rows, base.row.to_numpy(dtype=np.int64))
        predictions = {
            "champion": record["champion"][mask],
            "raw": record["raw"][mask],
            "rank50": record["rank50"][mask],
            "blend25": record["blend25"][mask],
        }
        views[name] = compare_reg_view(frame, target, rows, predictions)
        primary_parts.append((rows, predictions))
    primary_rows = np.concatenate([v[0] for v in primary_parts])
    if len(np.unique(primary_rows)) != len(primary_rows):
        raise AssertionError("Regression W1/W4 pilot rows overlap")
    primary_predictions = {key: np.concatenate([v[1][key] for v in primary_parts])
                           for key in primary_parts[0][1]}
    views["primary_pilot"] = compare_reg_view(frame, target, primary_rows, primary_predictions)

    hparts = []
    for name, end, low, high in [("H12_E", 55, 56, 67), ("H12_L", 67, 68, 79)]:
        base = h12.loc[(h12.cutoff == end) & h12.cycle.between(low, high)].copy()
        record = records[end]
        mask = (record["cycles"] >= low) & (record["cycles"] <= high)
        rows = record["rows"][mask]
        np.testing.assert_array_equal(rows, base.row.to_numpy(dtype=np.int64))
        predictions = {key: record[key][mask] for key in ["champion", "raw", "rank50", "blend25"]}
        views[name] = compare_reg_view(frame, target, rows, predictions)
        hparts.append((rows, predictions))
    hrows = np.concatenate([v[0] for v in hparts])
    if len(np.unique(hrows)) != len(hrows):
        raise AssertionError("Regression H12 early/late rows overlap")
    hpred = {key: np.concatenate([v[1][key] for v in hparts]) for key in hparts[0][1]}
    views["H12"] = compare_reg_view(frame, target, hrows, hpred)
    return views


def run() -> dict:
    frozen = check_freeze()
    prepare_output_directories_before_fits()
    class_frame, class_y, _, reg_frame, _ = load_official_data()
    reg_y = pd.to_numeric(reg_frame.threat_rank, errors="raise").to_numpy(dtype=float)
    class_controls = get_class_controls(class_frame, class_y)
    reg_primary, reg_h12, group_audits = get_regression_views(reg_frame)
    class_records, reg_records = {}, {}
    for end in ENDPOINTS:
        class_records[end] = _classification_fit(end, class_frame, class_y, class_controls[end], frozen["signature"])
        reg_window = reg_h12.loc[reg_h12.cutoff.eq(end)].copy()
        reg_records[end] = _regression_fit(end, reg_frame, reg_y, reg_window, frozen["signature"])
    class_views = build_class_views(class_frame, class_y, class_records)
    reg_views = build_reg_views(reg_frame, reg_y, reg_records, reg_primary, reg_h12)
    class_window_deltas = [
        class_views[name]["comparisons"]["blend25_vs_latest_sparse25"]["macro_f1_delta"]
        for name in ["W1", "W4", "H12_E", "H12_L"]
    ]
    class_primary_gain = class_views["primary_pilot"]["comparisons"]["blend25_vs_latest_sparse25"]["macro_f1_delta"]
    class_h12_gain = class_views["H12"]["comparisons"]["blend25_vs_latest_sparse25"]["macro_f1_delta"]
    cls_gate = {
        "primary_or_h12_improves": bool(class_primary_gain > 0 or class_h12_gain > 0),
        "no_view_harm_over_0_01": bool(min(class_window_deltas) >= -0.01),
        "primary_gain_vs_latest_sparse25": float(class_primary_gain),
        "h12_gain_vs_latest_sparse25": float(class_h12_gain),
        "minimum_view_delta_vs_latest_sparse25": float(min(class_window_deltas)),
        "views_nonnegative_vs_latest_sparse25": int(sum(d >= 0 for d in class_window_deltas)),
    }
    reg_window_harms = [
        -reg_views[name]["delta_vs_champion"]["blend25"]["mae_gain_champion_minus_candidate"]
        for name in ["W1", "W4", "H12_E", "H12_L"]
    ]
    reg_primary_gain = reg_views["primary_pilot"]["delta_vs_champion"]["blend25"]["mae_gain_champion_minus_candidate"]
    reg_h12_gain = reg_views["H12"]["delta_vs_champion"]["blend25"]["mae_gain_champion_minus_candidate"]
    reg_gate = {
        "primary_or_h12_improves": bool(reg_primary_gain > 0 or reg_h12_gain > 0),
        "max_view_harm_at_most_0_50": bool(max(reg_window_harms) <= 0.50),
        "primary_gain_vs_three_component_champion": float(reg_primary_gain),
        "h12_gain_vs_three_component_champion": float(reg_h12_gain),
        "maximum_W1_W4_H12_view_harm": float(max(reg_window_harms)),
        "view_harms": {name: float(value) for name, value in zip(["W1", "W4", "H12_E", "H12_L"], reg_window_harms)},
    }
    model_audit = {}
    for task, records in [("classification", class_records), ("regression", reg_records)]:
        for end, record in records.items():
            stats = record["state"]["ordered_tokens"]
            model_audit[f"{task}_{end}"] = {
                "fit_rows": int(record["state"]["fit_rows"]),
                "fit_population": "present_stream_only" if task == "classification" else "all_train_rows",
                "fit_data_sha256": record["state"]["fit_data_sha256"],
                "fit_target_sum": float(stats["fit_target_sum"]),
                "dictionary_size": int(stats["dictionary_size"]),
                "dictionary_sha256": stats["dictionary_sha256"],
                "training_features_sha256": stats["training_features_sha256"],
                "lineage": stats["lineage"],
                "validation_rows": int(len(record["rows"])),
                "fit_seconds": float(record["seconds"]),
                "cache_sha256": record["cache_sha256"],
            }
    result = {
        "status": "READY_FOR_INDEPENDENT_VERIFICATION",
        "freeze_signature": frozen["signature"],
        "source_head_contextual": frozen["source_head_contextual"],
        "classification_train_sha256": frozen["identity"]["classification_official_train_sha256"],
        "regression_train_sha256": frozen["identity"]["regression_official_train_sha256"],
        "official_train_rows_per_task": 50608,
        "new_official_fits": 4,
        "fit_plan": [f"{task}@{end}" for end in ENDPOINTS for task in TASKS],
        "test_read": False,
        "kaggle_access": False,
        "outer_labels_used_for_fit": False,
        "classification_threshold": THRESHOLD,
        "classification_fixed_new_blend_weight_present": BLEND_NEW_WEIGHT,
        "regression_rank_weight": 0.50,
        "regression_fixed_rank50_blend_weight": 0.25,
        "classification": {"views": class_views, "advance_gate_vs_latest_sparse25": cls_gate},
        "regression": {"views": reg_views, "advance_gate_vs_current_champion": reg_gate,
                       "fit_defined_control_group_audits": group_audits},
        "fit_audits": model_audit,
        "interpretation_limits": [
            "Training features encode each row using labels from strictly earlier cycles; validation rows use all labels available through their fit cutoff.",
            "This encodes observed past target-token associations, not semantic token meaning or causation.",
            "The temporal views are reused known Train outcomes and not an untouched holdout.",
            "Regression target-tail groups are defined after scoring and are descriptive only.",
            "The current classification external public score is contextual and was not used to select parameters, thresholds, or weights.",
        ],
    }
    write_new(OUT / "pilot.json", result)
    write_markdown(result)
    print(json.dumps({
        "status": result["status"],
        "classification_gate": cls_gate,
        "regression_gate": reg_gate,
        "new_official_fits": 4,
    }, indent=2, allow_nan=False), flush=True)
    return result


def write_markdown(result: dict) -> None:
    cls = result["classification"]["views"]
    reg = result["regression"]["views"]
    lines = [
        "# Ordered supervised token-statistics pilot", "",
        "## Question", "",
        "Do compact summaries of token-level target associations learned strictly from earlier cycles add predictive value beyond the existing representations? Each report is encoded before any labels from its own cycle enter the token dictionary. These statistics do not reveal anonymous token meaning and do not establish causality.", "",
        "Exactly four official-Train fits were run: Classification and Regression at cutoffs 55 and 67. The model, token support (3), smoothing (30), LightGBM parameters (400 trees, 15 leaves, minimum child 40, lambda 3), rank rule, blend weights, and thresholds were fixed before these fits.", "",
        "## Classification", "",
        "The initial C2_B-present/CB-missing control and the latest fixed sparse25-present/CB-missing control are both reported. The direct candidate replaces only the present branch. The fixed candidate blends 25% new supervised probability with 75% of the latest sparse25 present expert; the CB missing branch is preserved exactly.", "",
        "| View | n | Latest Macro F1 | Direct Macro F1 | Direct Δ | Fixed blend Macro F1 | Blend Δ vs latest | AP Δ vs latest | Brier Δ vs latest |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for name in ["W1", "W4", "H12_E", "H12_L", "primary_pilot", "H12"]:
        view = cls[name]
        latest = view["metrics"]["latest_sparse25"]
        direct = view["metrics"]["direct"]
        blend = view["metrics"]["blend25"]
        direct_delta = view["comparisons"]["direct_vs_latest_sparse25"]
        blend_delta = view["comparisons"]["blend25_vs_latest_sparse25"]
        lines.append(f"| {name} | {view['n']:,} | {latest['macro_f1']:.6f} | {direct['macro_f1']:.6f} | {direct_delta['macro_f1_delta']:+.6f} | {blend['macro_f1']:.6f} | {blend_delta['macro_f1_delta']:+.6f} | {blend_delta['average_precision_delta']:+.6f} | {blend_delta['brier_delta_candidate_minus_control']:+.6f} |")
    gate = result["classification"]["advance_gate_vs_latest_sparse25"]
    lines += ["", f"Classification advance gate versus latest sparse25: **{'PASS' if gate['primary_or_h12_improves'] and gate['no_view_harm_over_0_01'] else 'FAIL'}**; primary Δ={gate['primary_gain_vs_latest_sparse25']:+.6f}, H12 Δ={gate['h12_gain_vs_latest_sparse25']:+.6f}, minimum individual-view Δ={gate['minimum_view_delta_vs_latest_sparse25']:+.6f}.",
              "Per-class precision/recall/F1, confusion matrices, present/missing groups, and both initial/latest control comparisons are in `pilot.json`.", "",
              "## Regression", "",
              "Regression compares raw supervised predictions, fixed within-cycle rank50, and the exact 25% rank50 + 75% three-component champion blend. It reports MAE, RMSE, signed bias, equal-cycle MAE, per-cycle metrics, tails, EF0/EF1, and signal-stream availability.", "",
              "| View | n | Champion MAE | Raw MAE | Rank50 MAE | Fixed blend MAE | Blend MAE gain | Bias shift |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for name in ["W1", "W4", "H12_E", "H12_L", "primary_pilot", "H12"]:
        view = reg[name]
        lines.append(f"| {name} | {view['n']:,} | {view['metrics']['champion']['mae']:.4f} | {view['metrics']['raw']['mae']:.4f} | {view['metrics']['rank50']['mae']:.4f} | {view['metrics']['blend25']['mae']:.4f} | {view['delta_vs_champion']['blend25']['mae_gain_champion_minus_candidate']:+.4f} | {view['delta_vs_champion']['blend25']['bias_shift_candidate_minus_champion']:+.3f} |")
    rgate = result["regression"]["advance_gate_vs_current_champion"]
    lines += ["", f"Regression fixed-blend advance gate: **{'PASS' if rgate['primary_or_h12_improves'] and rgate['max_view_harm_at_most_0_50'] else 'FAIL'}**; primary gain={rgate['primary_gain_vs_three_component_champion']:+.4f}, H12 gain={rgate['h12_gain_vs_three_component_champion']:+.4f}, maximum W1/W4/H12 view harm={rgate['maximum_W1_W4_H12_view_harm']:+.4f}.",
              "", "## Leakage and scope checks", "",
              "- All four full audit states record per-cycle `labels_previously_seen` counts; each equals the number of fit rows with a strictly earlier cycle.",
              "- The synthetic same-cycle perturbation check confirms current-cycle labels do not alter features for their own cycle, but do alter later-cycle features when shared tokens have support.",
              "- Fit fingerprints bind only Train rows through each cutoff; transformations are fit past-only.",
              "- No Test/Kaggle read or call; these known Train-cycle views are development evidence, not an untouched holdout.",
              "- Anonymous target-stat token values are evidence of past target association, not semantic interpretation or causal effects.",
              "", "Full per-class, per-cycle, and subgroup results and fit lineages are in `pilot.json`; independent checks are in `verification.json`.", ""]
    path = OUT / "FULLPILOT.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write("\n".join(lines))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--dry-run", action="store_true")
    group.add_argument("--freeze", action="store_true")
    group.add_argument("--run", action="store_true")
    args = parser.parse_args()
    if args.dry_run:
        print(json.dumps(dry_run(), indent=2, allow_nan=False))
    elif args.freeze:
        frozen = freeze()
        print(frozen["signature"])
    else:
        run()
