"""Frozen two-origin sparse MLP pilot for Classification Research Next V1.

Commands:
  python -m scripts.optimization_v3_classification_sparse_mlp_v1 prepare
  python -m scripts.optimization_v3_classification_sparse_mlp_v1 fit 55
  python -m scripts.optimization_v3_classification_sparse_mlp_v1 fit 67
  python -m scripts.optimization_v3_classification_sparse_mlp_v1 summarize

Only official Classification Train labels enter this runner. Test and
SampleSubmission are not opened by any command.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import tempfile
from time import perf_counter
from typing import Any, Mapping
import warnings

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import joblib
import numpy as np
import pandas as pd
import psutil
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import (average_precision_score, confusion_matrix, f1_score,
                            log_loss, precision_score, recall_score, roc_auc_score)
from threadpoolctl import threadpool_limits

from scripts.research_campaign_classification import load_train
from scripts.optimization_v3_classification_next_v1 import _load_baseline, _load_c19_pilot
from src.data.audit import sha256 as file_sha256
from src.models.classification_models import _array_digest, _json_digest, _number_array
from src.models.classification_round2_models import stream_missing_mask
from src.models.round1_models import _to_builtin
from src.models.classification_sparse_mlp_v1 import (
    PROTOCOL, ROOT as MODEL_ROOT, build_fit_design, estimate_peak_bytes,
    make_estimator, read_protocol, route_probabilities, transform_score_design,
)
from src.models.research_next_budget import (finish_fit, ledger_summary, read_events,
                                            release_unused, reserve_card, start_fit)
from src.models.research_next_contract import (canonical_sha256, file_sha256 as contract_sha,
                                               publish_bundle, read_bundle)
from src.utils.paths import task_files


CARD_ID = "CLASSIFICATION_SPARSE_MLP_V1"
STAGE = "MLP_NEW_PILOT"
ORIGINS = (55, 67)
THRESHOLD = 0.50
THREADS = 2
JOB_CAP_BYTES = 6 * 1024**3
FREE_RESERVE_BYTES = 2 * 1024**3
CONFIG_DIR = ROOT / "experiments/optimization_v3/classification_mlp_v1"
RUN_ROOT = ROOT / "experiments/runs/optimization_v3_classification_mlp_v1"
REPORT_ROOT = ROOT / "reports/optimization_v3/classification_mlp_v1"
FREEZE = CONFIG_DIR / "freeze.json"
PREFLIGHT = REPORT_ROOT / "preflight.json"
SYNTHETIC_REPORT = REPORT_ROOT / "synthetic_preflight.json"
LEDGER = ROOT / "experiments/runs/optimization_v3_research_next_v1/budget.jsonl"
TRAIN_EXPECTED_SHA256 = "875552a68a3d24b0f3e1966d6ddb5671e6f54305a1c526e2f0a76747942ed91c"
CODE_FILES = (
    "src/models/classification_sparse_mlp_v1.py",
    "scripts/optimization_v3_classification_sparse_mlp_v1.py",
    "scripts/optimization_v3_classification_sparse_mlp_v1_verify.py",
    "tests/test_optimization_v3_classification_sparse_mlp_v1.py",
    "src/models/classification_models.py",
    "src/models/research_next_n02.py",
    "src/models/round1_models.py",
    "src/models/classification_round2_models.py",
    "src/models/classification_metrics.py",
    "src/models/research_next_budget.py",
    "src/models/research_next_contract.py",
    "scripts/research_campaign_classification.py",
    "scripts/optimization_v3_classification_next_v1.py",
    "scripts/optimization_v3_research_next_n01.py",
    "src/utils/paths.py",
)
STATIC_INPUTS = (
    "experiments/optimization_v3/classification_mlp_v1/protocol.json",
    "reports/optimization_v3/classification_mlp_v1/DECISION.md",
    "reports/optimization_v3/classification_next_v1/FINAL_SUMMARY.md",
    "reports/optimization_v3/classification_diagnostics/FINDINGS.md",
    "experiments/optimization_v3/classification_sparse/protocol.json",
    "experiments/optimization_v3/classification_sparse/freeze.json",
    "reports/optimization_v3/classification_sparse/pilot.json",
)


def _sha(path: Path) -> str:
    return file_sha256(Path(path))


def _json(path: Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _write_once(path: Path, value: Any) -> None:
    path = Path(path)
    payload = json.dumps(value, sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_text(encoding="utf-8") != payload:
            raise RuntimeError(f"Immutable output differs; refusing overwrite: {path}")
        return
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())


def _git_head() -> str:
    return subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip()


def _git_is_ancestor(commit: str) -> bool:
    return subprocess.run(["git", "-C", str(ROOT), "merge-base", "--is-ancestor", commit, "HEAD"],
                          check=False, capture_output=True).returncode == 0


def _protocol_expected_hashes() -> dict[str, str]:
    protocol = read_protocol()
    return {str(key): str(value) for key, value in protocol.get("input_sha256", {}).items()}


def _paths_for_inputs(train_path: Path) -> list[Path]:
    paths = [train_path, *(ROOT / name for name in STATIC_INPUTS)]
    for cutoff in ORIGINS:
        paths.extend([
            ROOT / f"experiments/runs/optimization_v3_classification_sparse/fit{cutoff}.npz",
            ROOT / f"experiments/runs/optimization_v3_classification_sparse/fit{cutoff}.json",
            ROOT / f"experiments/runs/classification_round1_v1/fit{cutoff}__C2_B.npz",
            ROOT / f"reports/classification_round1/fit_audits/fit{cutoff}.json",
            ROOT / f"experiments/runs/dual_cls_catboost_v1/fit{cutoff}.npz",
            ROOT / f"experiments/runs/dual_cls_catboost_v1/fit{cutoff}.json",
            ROOT / f"experiments/runs/optimization_v3_batch1/C19/C19_fit{cutoff}.npz",
            ROOT / f"experiments/runs/optimization_v3_batch1/C19/C19_fit{cutoff}.json",
        ])
    return paths


def make_identity(*, source_commit: str | None = None) -> dict[str, Any]:
    protocol = read_protocol()
    train_path = Path(task_files("classification")["train"]).resolve()
    if not train_path.is_file() or _sha(train_path) != TRAIN_EXPECTED_SHA256:
        raise RuntimeError("Official Classification Train missing or SHA256 differs from frozen protocol")
    missing = [str(path) for path in [*(ROOT / p for p in CODE_FILES), *_paths_for_inputs(train_path)] if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Required frozen MLP input/source files are missing: {missing}")
    expected = _protocol_expected_hashes()
    for relative, digest in expected.items():
        path = ROOT / relative
        if not path.is_file() or _sha(path) != digest:
            raise RuntimeError(f"Protocol-pinned source hash mismatch: {relative}")
    inputs = _paths_for_inputs(train_path)
    source_files = [ROOT / name for name in CODE_FILES]
    packages = {}
    for package in ("numpy", "pandas", "scipy", "scikit-learn", "joblib", "threadpoolctl", "psutil"):
        packages[package] = importlib.metadata.version(package)
    return {
        "campaign": "CLASSIFICATION_SPARSE_MLP_V1",
        "source_commit": source_commit or _git_head(),
        "protocol_sha256": _sha(PROTOCOL),
        "code_sha256": {p.resolve().relative_to(ROOT).as_posix(): _sha(p) for p in source_files},
        "input_sha256": {p.resolve().relative_to(ROOT).as_posix(): _sha(p) for p in inputs},
        "packages": packages,
        "estimator": dict(protocol["model"]),
        "origins": list(ORIGINS),
        "test_read": False,
        "sample_submission_read": False,
        "kaggle_access": False,
    }


def frozen_identity() -> tuple[dict[str, Any], str]:
    if not FREEZE.is_file():
        raise RuntimeError("MLP freeze is missing; run the preparation command first")
    frozen = _json(FREEZE)
    identity = frozen.get("identity")
    if not isinstance(identity, dict) or not identity.get("source_commit"):
        raise RuntimeError("MLP freeze identity is invalid")
    if not _git_is_ancestor(str(identity["source_commit"])):
        raise RuntimeError("Current HEAD does not descend from frozen preparation commit")
    current = make_identity(source_commit=str(identity["source_commit"]))
    if current != identity:
        raise RuntimeError("Frozen MLP code, protocol, raw Train, or cache input hash changed")
    signature = hashlib.sha256(json.dumps(identity, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
    if frozen.get("signature") != signature:
        raise RuntimeError("MLP freeze signature mismatch")
    return identity, signature


def _load_context() -> dict[str, Any]:
    frame, target, train_path = load_train(None)
    if len(frame) != 50_608 or _sha(train_path) != TRAIN_EXPECTED_SHA256:
        raise RuntimeError("Official Classification Train row count/hash differs")
    cycles = pd.to_numeric(frame["cycle_code"], errors="raise").to_numpy(dtype=np.int64)
    if not np.array_equal(np.unique(cycles), np.arange(47, 80)):
        raise RuntimeError("Official Train cycle coverage differs from 47-79")
    baselines: dict[int, dict[str, Any]] = {}
    for cutoff in ORIGINS:
        baseline = _load_baseline(frame, target, cutoff)
        historical = _load_c19_pilot(frame, target, cutoff, baseline)
        if not np.array_equal(historical["baseline_prediction"], baseline["baseline"]):
            raise RuntimeError(f"Historical sparse25 baseline differs from reconstructed cutoff {cutoff}")
        expected_rows = np.flatnonzero((cycles > cutoff) & (cycles <= cutoff + 12))
        if not np.array_equal(baseline["rows"], expected_rows):
            raise RuntimeError(f"Frozen baseline row boundary differs at cutoff {cutoff}")
        baselines[cutoff] = baseline
    return {"frame": frame, "target": target, "cycles": cycles,
            "train_path": Path(train_path), "baselines": baselines}


def _design_for_origin(context: Mapping[str, Any], cutoff: int) -> tuple[Any, Any, np.ndarray, np.ndarray, np.ndarray]:
    frame = context["frame"]
    cycles = context["cycles"]
    fit_rows = np.flatnonzero((cycles >= 47) & (cycles <= cutoff))
    score_rows = np.asarray(context["baselines"][cutoff]["rows"], dtype=np.int64)
    feature_names = [name for name in frame.columns if name not in {"record_id", "artifact_signal"}]
    X_fit = frame.iloc[fit_rows].loc[:, feature_names]
    y_fit = np.asarray(context["target"])[fit_rows]
    base, fit_matrix, built_y = build_fit_design(X_fit, y_fit)
    if not np.array_equal(built_y, y_fit):
        raise RuntimeError("C2_B fit target changed during design construction")
    base._mlp_c2b_preprocessing_parity = _verify_c2b_preprocessing(base, cutoff)
    if base._mlp_c2b_preprocessing_parity["status"] != "PASS":
        raise RuntimeError(f"Rebuilt fit-only C2_B preprocessing differs from saved audit at {cutoff}")
    X_score = frame.iloc[score_rows].loc[:, feature_names]
    score_matrix = transform_score_design(base, X_score)
    if fit_matrix.shape[1] != score_matrix.shape[1]:
        raise RuntimeError("C2_B train and score design schemas differ")
    return base, fit_matrix, built_y, score_matrix, score_rows


def _verify_c2b_preprocessing(base: Any, cutoff: int) -> dict[str, Any]:
    """Compare rebuilt official preprocessing state with the archived C2_B audit."""
    audit_path = ROOT / f"reports/classification_round1/fit_audits/fit{cutoff}.json"
    cached = _json(audit_path)["C2_B"]["fitted_state"]
    categories = base._linear_encoder.categories_
    infrequent = base._linear_encoder.infrequent_categories_
    transformed = {
        "numeric_names": list(base._linear_numeric_names),
        "input_category_names": list(base._category_names),
        "numeric_count": int(len(base._linear_numeric_names)),
        "encoded_category_counts": [int(len(values)) for values in categories],
        "encoded_category_sha256": _json_digest([[str(value) for value in values] for values in categories]),
        "infrequent_category_sha256": _json_digest([
            None if values is None else [str(value) for value in values] for values in infrequent
        ]),
    }
    text_state = _to_builtin(base._text_state())
    scaler_state = {"mean_sha256": _array_digest(base._linear_scaler.mean_),
                    "scale_sha256": _array_digest(base._linear_scaler.scale_)}
    checks = {
        "fit_rows": int(base._fit_rows) == int(cached["fit_rows"]),
        "fit_fingerprint": base._fit_fingerprint == cached["fit_data_sha256"],
        "input_features": list(base.input_feature_names) == cached["input_feature_names"],
        "structured_preprocessor": base._structured_preprocessor.audit_state() == cached["preprocessor"],
        "scaler": scaler_state == cached["scaler"],
        "transformed_features": transformed == cached["transformed_features"],
        "text_vectorizer": text_state == cached["text"],
    }
    return {"status": "PASS" if all(checks.values()) else "FAIL", "cutoff": int(cutoff),
            "fit_rows": int(base._fit_rows), "checks": checks,
            "audit_sha256": _sha(audit_path), "rebuilt_preprocessor": base._structured_preprocessor.audit_state(),
            "rebuilt_text_state": text_state, "rebuilt_scaler": scaler_state,
            "rebuilt_transformed_features": transformed}


def _runtime_check(memory: Mapping[str, int]) -> dict[str, Any]:
    available = int(psutil.virtual_memory().available)
    estimated = int(memory["estimated_additional_peak_bytes"])
    allowed = int(JOB_CAP_BYTES)
    free_reserve = int(FREE_RESERVE_BYTES)
    status = "PASS" if estimated <= allowed and available >= estimated + free_reserve else "BLOCKED_MEMORY"
    return {"status": status, "available_ram_bytes": available,
            "estimated_additional_peak_bytes": estimated, "job_cap_bytes": allowed,
            "required_free_reserve_bytes": free_reserve,
            "minimum_available_bytes": estimated + free_reserve,
            "memory_components": dict(memory), "threads": THREADS}


def _synthetic_frame(n: int = 192) -> tuple[pd.DataFrame, np.ndarray]:
    i = np.arange(n)
    frame = pd.DataFrame({
        "impact_load": (i % 47).astype(float) + 1,
        "entry_friction": (i % 2).astype(float),
        "cycle_code": 47 + (i % 8),
        "scope_pressure": (i % 31).astype(float) + 0.5,
        "text_load": (i % 9).astype(float),
        "scope_trace": np.where(i % 4 == 0, "2+", "2"),
        "org_spread": np.where(i % 5 == 0, "3+", "2"),
        "module_spread": np.where(i % 3 == 0, "2+", "1"),
        "telemetry_flag": np.where(i % 4 == 0, "true", "false"),
        "org_code": pd.Series([None if j % 6 == 0 else f"org{j % 7}" for j in i], dtype=object),
        "module_code": pd.Series([None if j % 5 == 0 else f"mod{j % 9}" for j in i], dtype=object),
        "pattern_code": pd.Series([None if j % 7 == 0 else f"pat{j % 5}" for j in i], dtype=object),
        "signal_stream": pd.Series([None if j % 13 == 0 else
            f"token{j % 11} common{j % 4} artifact{int(j % 2)}" for j in i], dtype=object),
    })
    y = ((i % 5 == 0) | (i % 7 == 0)).astype(np.int8)
    return frame, y


class _InjectedPostDurableFailure(RuntimeError):
    pass


def run_synthetic_e2e() -> dict[str, Any]:
    """Exercise builder, fit, persistence, failure/resume, route and metrics."""
    from scipy import sparse
    from src.models.classification_models import ClassificationModel, _text_values
    from src.models.research_next_contract import publish_bundle, read_bundle

    train_x, y = _synthetic_frame()
    base, fit_matrix, fit_y = build_fit_design(train_x, y)
    # Independent synthetic reference: invoke the production C2_B transforms
    # directly, without fitting its LogisticRegression head.
    ref = ClassificationModel({"id": "C2_B"})
    selected = ref._select(train_x).copy()
    ref._fit_rows = len(y)
    ref._fit_fingerprint = "synthetic-reference-only"
    ref._positive_rate = float(y.mean())
    ref._estimator = None
    ref._structured_preprocessor = None
    ref._text_vectorizer = None
    ref._linear_scaler = None
    ref._linear_encoder = None
    ref._lookup_rates = {}
    ref._lookup_counts = {}
    ref._fallback_all_empty = False
    ref._fallback_vocab_empty = False
    ref._transform_state = {}
    ref._hgb_categorical_mask = []
    ref._native_vocabulary_hashes = {}
    structured = ref._fit_structured(selected)
    text = ref._fit_text(selected["signal_stream"])
    _docs, missing = _text_values(selected["signal_stream"])
    reference_matrix = sparse.hstack((structured, text,
        sparse.csr_matrix(missing.astype(np.float64).reshape(-1, 1))), format="csr", dtype=np.float64)
    if fit_matrix.shape != reference_matrix.shape or (fit_matrix != reference_matrix).nnz:
        raise RuntimeError("Synthetic C2_B fit matrix differs from independent reference")

    score_x = train_x.iloc[:48].copy()
    score_matrix = transform_score_design(base, score_x)
    reference_score = sparse.hstack((base._transform_structured(base._select(score_x)),
        base._transform_text(score_x["signal_stream"]),
        sparse.csr_matrix(_text_values(score_x["signal_stream"])[1].astype(np.float64).reshape(-1, 1))),
        format="csr", dtype=np.float64)
    if score_matrix.shape != reference_score.shape or (score_matrix != reference_score).nnz:
        raise RuntimeError("Synthetic C2_B score transform differs from independent reference")

    params = read_protocol()["model"]["parameters"]
    estimator = make_estimator()
    caught: list[str] = []
    with warnings.catch_warnings(record=True) as found, threadpool_limits(limits=THREADS):
        warnings.simplefilter("always")
        estimator.fit(fit_matrix, fit_y)
        caught = [f"{w.category.__name__}: {w.message}" for w in found]
    if estimator.n_iter_ != 30 or len(estimator.loss_curve_) != 30:
        raise RuntimeError("Synthetic MLP did not execute the fixed 30 epochs")
    loss = np.asarray(estimator.loss_curve_, dtype=np.float64)
    raw = estimator.predict_proba(score_matrix)[:, 1]
    if not np.isfinite(loss).all() or not np.isfinite(raw).all() or np.any((raw < 0) | (raw > 1)):
        raise RuntimeError("Synthetic MLP loss or score probabilities are invalid")

    n_score = len(score_x)
    baseline = np.linspace(0.15, 0.85, n_score)
    c2b = np.linspace(0.25, 0.75, n_score)
    cb = np.full(n_score, 0.37)
    missing_rows = _text_values(score_x["signal_stream"])[1]
    candidate = route_probabilities("candidate", baseline, raw, c2b, cb, missing_rows)
    matched = route_probabilities("matched_control", baseline, raw, c2b, cb, missing_rows)
    y_score = y[:n_score]
    import joblib as _joblib
    from src.models.classification_sparse_mlp_v1 import route_probabilities as _route
    with tempfile.TemporaryDirectory(prefix="kasdad-mlp-synthetic-") as temporary:
        temp = Path(temporary)
        state_path = temp / "model.joblib"
        temp_state = temp / "model.joblib.tmp"
        _joblib.dump({"base": base, "estimator": estimator}, temp_state, compress=0)
        os.replace(temp_state, state_path)
        state_sha = _sha(state_path)
        try:
            raise _InjectedPostDurableFailure("synthetic failure after fitted state is durable")
        except _InjectedPostDurableFailure:
            resumed = _joblib.load(state_path)
        resumed_matrix = transform_score_design(resumed["base"], score_x)
        resumed_raw = resumed["estimator"].predict_proba(resumed_matrix)[:, 1]
        if not np.array_equal(raw, resumed_raw):
            raise RuntimeError("Synthetic resume prediction differs from original without refit")
        resumed_candidate = _route("candidate", baseline, resumed_raw, c2b, cb, missing_rows)
        bundle_path = temp / "bundle"
        identity = {"campaign": "synthetic", "model_sha256": state_sha, "fit_origin": 47}
        payload = {
            "score_rows": np.arange(n_score, dtype=np.int64),
            "score_record_ids": np.asarray([f"synthetic-{i:04d}" for i in range(n_score)], dtype=str),
            "cycle_code": score_x["cycle_code"].to_numpy(dtype=np.int16),
            "prediction": resumed_candidate,
            "baseline_prediction": baseline,
            "raw_mlp_probability": resumed_raw,
            "matched_prediction": matched,
            "control_c2b_probability": c2b,
            "control_cb_probability": cb,
            "control_stream_missing": missing_rows,
            "payload_meta": {"probability_semantics": "synthetic routed positive-class probabilities"},
        }
        receipt = publish_bundle(bundle_path, payload, identity, task="classification")
        saved = read_bundle(bundle_path, identity, task="classification")
        if saved.receipt["receipt_sha256"] != receipt["receipt_sha256"]:
            raise RuntimeError("Synthetic persistence receipt read-back mismatch")
        rebuilt = _route("candidate", saved.arrays["baseline_prediction"], saved.arrays["raw_mlp_probability"],
                         saved.arrays["control_c2b_probability"], saved.arrays["control_cb_probability"],
                         saved.arrays["control_stream_missing"])
        if not np.array_equal(rebuilt, saved.arrays["prediction"]):
            raise RuntimeError("Synthetic bundle route reconstruction mismatch")
        # Execute the same metric and group guard engine across the frozen view names.
        synthetic_outputs = {}
        synthetic_frame = train_x.iloc[:n_score].copy()
        synthetic_frame["cycle_code"] = np.where(np.arange(n_score) < n_score // 2, 56, 68)
        for origin, select in ((55, np.arange(n_score) < n_score // 2),
                               (67, np.arange(n_score) >= n_score // 2)):
            rows = np.flatnonzero(select)
            synthetic_outputs[origin] = {
                "score_rows": rows,
                "score_record_ids": np.asarray([f"synthetic-{i:04d}" for i in rows]),
                "cycle_code": synthetic_frame["cycle_code"].to_numpy(dtype=np.int64)[rows],
                "prediction": resumed_candidate[rows],
                "baseline_prediction": baseline[rows],
                "matched_prediction": matched[rows],
                "c2b_diagnostic": _route("c2b_diagnostic", baseline[rows], resumed_raw[rows], c2b[rows], cb[rows], missing_rows[rows]),
                "origin": np.full(len(rows), origin, dtype=np.int64),
            }
        metrics, groups, cycles, gates = evaluate_views(synthetic_outputs, synthetic_frame, y_score,
                                                        fit_edges={55: [], 67: []}, definitions=view_definitions())
        if set(metrics) != set(view_definitions()) or not groups or not cycles:
            raise RuntimeError("Synthetic all-view metric/group/per-cycle path did not execute")
        report = {
            "status": "PASS",
            "synthetic_only": True,
            "official_fits": 0,
            "official_train_read": False,
            "test_read": False,
            "sample_submission_read": False,
            "c2b_fit_matrix_reference_parity": True,
            "c2b_score_matrix_reference_parity": True,
            "fit_rows": int(fit_matrix.shape[0]),
            "fit_columns": int(fit_matrix.shape[1]),
            "fit_nnz": int(fit_matrix.nnz),
            "score_rows": int(score_matrix.shape[0]),
            "n_iter": int(estimator.n_iter_),
            "loss_curve": loss.tolist(),
            "initial_loss": float(loss[0]),
            "final_loss": float(loss[-1]),
            "convergence_warnings": caught,
            "model_state_sha256": state_sha,
            "failure_injection_after_durable_state": True,
            "resume_without_refit": True,
            "bundle_publish_readback": True,
            "route_reconstruction": True,
            "all_six_views_metrics_and_group_guards": True,
            "synthetic_views": metrics,
            "synthetic_group_count": len(groups),
            "synthetic_per_cycle_rows": len(cycles),
            "synthetic_gate_checks": gates,
        }
    return report


def view_definitions() -> dict[str, list[tuple[int, int, int]]]:
    return {
        "primary": [(55, 56, 59), (67, 68, 71)],
        "W1": [(55, 56, 59)],
        "W4": [(67, 68, 71)],
        "H12_55": [(55, 56, 67)],
        "H12_67": [(67, 68, 79)],
        "H12_pooled": [(55, 56, 67), (67, 68, 79)],
    }


def _metric(y: np.ndarray, probability: np.ndarray) -> dict[str, Any]:
    y = np.asarray(y, dtype=np.int8)
    p = np.asarray(probability, dtype=np.float64)
    if y.ndim != 1 or p.shape != y.shape or not len(y) or not np.isin(y, [0, 1]).all():
        raise ValueError("Metric inputs must be aligned nonempty binary vectors")
    if not np.isfinite(p).all() or np.any((p < 0) | (p > 1)):
        raise ValueError("Metric probabilities must be finite and in [0, 1]")
    pred = (p >= THRESHOLD).astype(np.int8)
    cm = confusion_matrix(y, pred, labels=[0, 1])
    both = len(np.unique(y)) == 2
    return {
        "n": int(len(y)),
        "macro_f1": float(f1_score(y, pred, average="macro", labels=[0, 1], zero_division=0)),
        "roc_auc": float(roc_auc_score(y, p)) if both else None,
        "average_precision": float(average_precision_score(y, p)) if both else None,
        "log_loss": float(log_loss(y, p, labels=[0, 1])),
        "brier_score": float(np.mean(np.square(p - y))),
        "precision_positive": float(precision_score(y, pred, pos_label=1, zero_division=0)),
        "recall_positive": float(recall_score(y, pred, pos_label=1, zero_division=0)),
        "precision_negative": float(precision_score(y, pred, pos_label=0, zero_division=0)),
        "recall_negative": float(recall_score(y, pred, pos_label=0, zero_division=0)),
        "positive_rate": float(y.mean()),
        "predicted_positive_rate": float(pred.mean()),
        "confusion_matrix_labels_0_1": cm.tolist(),
    }


def _origin_bins(frame: pd.DataFrame, cutoff: int) -> tuple[np.ndarray, list[float]]:
    parsed, _missing, _invalid = _number_array(frame["text_load"])
    cycles = pd.to_numeric(frame["cycle_code"], errors="raise").to_numpy(dtype=np.int64)
    fit = cycles <= cutoff
    finite = parsed[fit & np.isfinite(parsed)]
    if not len(finite):
        return np.full(len(frame), -1, dtype=np.int64), []
    edges = np.unique(np.quantile(finite, [0.25, 0.50, 0.75]))
    labels = np.full(len(frame), -1, dtype=np.int64)
    valid = np.isfinite(parsed)
    labels[valid] = np.searchsorted(edges, parsed[valid], side="right").astype(np.int64)
    return labels, edges.astype(float).tolist()


def evaluate_views(outputs: Mapping[int, Mapping[str, Any]], frame: pd.DataFrame, target: np.ndarray,
                   *, fit_edges: Mapping[int, list[float]] | None = None,
                   definitions: Mapping[str, list[tuple[int, int, int]]] | None = None
                   ) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Build metrics, fit-defined protected groups, per-cycle deltas and frozen gates."""
    definitions = view_definitions() if definitions is None else definitions
    target = np.asarray(target, dtype=np.int8)
    cycles_all = pd.to_numeric(frame["cycle_code"], errors="raise").to_numpy(dtype=np.int64)
    bins_by_origin: dict[int, np.ndarray] = {}
    if fit_edges is None:
        for cutoff in outputs:
            bins_by_origin[int(cutoff)] = _origin_bins(frame, int(cutoff))[0]
    else:
        parsed, _missing, _invalid = _number_array(frame["text_load"])
        for cutoff, edges in fit_edges.items():
            labels = np.full(len(frame), -1, dtype=np.int64)
            valid = np.isfinite(parsed)
            labels[valid] = np.searchsorted(np.asarray(edges, dtype=np.float64), parsed[valid], side="right")
            bins_by_origin[int(cutoff)] = labels
    metrics: dict[str, Any] = {}
    groups: list[dict[str, Any]] = []
    per_cycle: list[dict[str, Any]] = []
    for view_name, blocks in definitions.items():
        parts: list[dict[str, np.ndarray]] = []
        for cutoff, low, high in blocks:
            record = outputs[int(cutoff)]
            select = (np.asarray(record["cycle_code"]) >= int(low)) & (np.asarray(record["cycle_code"]) <= int(high))
            parts.append({key: np.asarray(record[key])[select] for key in (
                "score_rows", "prediction", "baseline_prediction", "matched_prediction", "c2b_diagnostic", "origin")})
        rows = np.concatenate([part["score_rows"] for part in parts]).astype(np.int64)
        if len(np.unique(rows)) != len(rows):
            raise RuntimeError(f"Evaluation view {view_name} contains duplicate Train rows")
        predictions = {key: np.concatenate([part[key] for part in parts]).astype(np.float64) for key in (
            "prediction", "baseline_prediction", "matched_prediction", "c2b_diagnostic")}
        origins = np.concatenate([part["origin"] for part in parts]).astype(np.int64)
        y = target[rows]
        model_metrics = {"candidate": _metric(y, predictions["prediction"]),
                         "sparse25": _metric(y, predictions["baseline_prediction"]),
                         "matched_control": _metric(y, predictions["matched_prediction"]),
                         "C2_B_diagnostic": _metric(y, predictions["c2b_diagnostic"])}
        gains = {}
        for comparator in ("sparse25", "matched_control", "C2_B_diagnostic"):
            gains[f"candidate_minus_{comparator}"] = {
                key: (None if model_metrics["candidate"][key] is None or model_metrics[comparator][key] is None
                      else float(model_metrics["candidate"][key] - model_metrics[comparator][key]))
                for key in ("macro_f1", "roc_auc", "average_precision", "log_loss", "brier_score",
                            "precision_positive", "recall_positive", "predicted_positive_rate")
            }
        metrics[view_name] = {"n": int(len(rows)), "models": model_metrics, "gains": gains,
                              "origins": sorted(set(origins.tolist()))}
        for cycle_value in sorted(np.unique(cycles_all[rows]).tolist()):
            take = cycles_all[rows] == cycle_value
            per_cycle.append({"view": view_name, "cycle": int(cycle_value), "n": int(take.sum()),
                "candidate_macro_f1": _metric(y[take], predictions["prediction"][take])["macro_f1"],
                "sparse25_macro_f1": _metric(y[take], predictions["baseline_prediction"][take])["macro_f1"],
                "matched_macro_f1": _metric(y[take], predictions["matched_prediction"][take])["macro_f1"],
                "candidate_minus_sparse25_macro_f1": float(_metric(y[take], predictions["prediction"][take])["macro_f1"]-
                                                               _metric(y[take], predictions["baseline_prediction"][take])["macro_f1"]),
            })
        row_frame = frame.iloc[rows]
        missing = stream_missing_mask(row_frame)
        quartile = np.asarray([bins_by_origin[int(origin)][int(row)] for row, origin in zip(rows, origins)])
        grouping = {
            "fit_defined_text_load_quartile": np.asarray(["missing" if value < 0 else f"Q{min(int(value)+1,4)}" for value in quartile]),
            "stream_availability": np.where(missing, "missing", "present"),
            "entry_friction": row_frame["entry_friction"].astype(str).to_numpy(),
        }
        for field, labels in grouping.items():
            for label in sorted(np.unique(labels).tolist()):
                take = labels == label
                gy = y[take]
                n0, n1 = int(np.sum(gy == 0)), int(np.sum(gy == 1))
                supported = n0 >= 30 and n1 >= 30 and int(take.sum()) >= 200
                candidate_metric = _metric(gy, predictions["prediction"][take])
                sparse_metric = _metric(gy, predictions["baseline_prediction"][take])
                f1_harm = float(sparse_metric["macro_f1"] - candidate_metric["macro_f1"])
                recall_harm = float(sparse_metric["recall_positive"] - candidate_metric["recall_positive"])
                groups.append({"view": view_name, "field": field, "group": str(label),
                    "n": int(take.sum()), "class_0_n": n0, "class_1_n": n1, "supported": bool(supported),
                    "candidate_macro_f1": candidate_metric["macro_f1"], "sparse25_macro_f1": sparse_metric["macro_f1"],
                    "f1_harm": f1_harm, "candidate_positive_recall": candidate_metric["recall_positive"],
                    "sparse25_positive_recall": sparse_metric["recall_positive"],
                    "positive_recall_harm": recall_harm,
                    "fit_origins": sorted(set(origins[take].tolist())),
                    "fails_0_01_f1": bool(supported and f1_harm > 0.01),
                    "fails_0_03_recall": bool(supported and recall_harm > 0.03),
                    "fails_0_02_recall": bool(supported and recall_harm > 0.02),
                })
    gains = {name: float(metrics[name]["gains"]["candidate_minus_sparse25"]["macro_f1"])
             for name in ("primary", "W1", "W4", "H12_55", "H12_67", "H12_pooled")}
    primary_matched = float(metrics["primary"]["gains"]["candidate_minus_matched_control"]["macro_f1"])
    supported = [row for row in groups if row["supported"]]
    checks = {
        "primary_gain_vs_sparse25_min_0_0005": gains["primary"] >= 0.0005,
        "primary_gain_vs_matched_min_0_0002": primary_matched >= 0.0002,
        "H12_pooled_gain_vs_sparse25_nonnegative": gains["H12_pooled"] >= 0.0,
        "H12_origin67_harm_max_0_0005": gains["H12_67"] >= -0.0005,
        "W1_W4_harm_max_0_003": min(gains["W1"], gains["W4"]) >= -0.003,
        "supported_group_F1_harm_max_0_01": not any(row["fails_0_01_f1"] for row in supported),
        "supported_group_recall_harm_max_0_03": not any(row["fails_0_03_recall"] for row in supported),
    }
    gate = {"status": "PASS_ALL_FROZEN_GATES" if all(checks.values()) else "NO_GO_GATE_FAILED",
        "checks": checks,
        "thresholds": {"primary_gain_vs_sparse25_min": 0.0005, "primary_gain_vs_matched_min": 0.0002,
            "H12_pooled_gain_min": 0.0, "H12_origin67_harm_max": 0.0005,
            "W1_W4_harm_max": 0.003, "group_min_n": 200, "group_each_class_min": 30,
            "group_F1_harm_max": 0.01, "group_recall_harm_max": 0.03},
        "view_macro_f1_gains_vs_sparse25": gains,
        "primary_gain_vs_matched_control": primary_matched,
        "supported_groups": int(len(supported)), "unsupported_groups": int(len(groups)-len(supported)),
        "supported_group_failures": [row for row in supported if row["fails_0_01_f1"] or row["fails_0_03_recall"]],
        "original_0_02_recall_failures_reported": [row for row in supported if row["fails_0_02_recall"]],
        "original_0_02_recall_guard_pass": not any(row["fails_0_02_recall"] for row in supported),
    }
    return metrics, groups, per_cycle, gate


def _atomic_joblib(value: Any, path: Path) -> str:
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite durable fitted state: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    joblib.dump(value, temporary, compress=0)
    with temporary.open("rb+") as stream:
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    return _sha(path)


def _fit_state(base: Any, estimator: Any, fit_matrix: Any, target: np.ndarray, cutoff: int,
               warnings_seen: list[str], seconds: float, model_sha: str) -> dict[str, Any]:
    vectorizer = base._text_vectorizer
    vocab = {} if vectorizer is None else vectorizer.vocabulary_
    vocab_sha = hashlib.sha256(json.dumps(vocab, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()).hexdigest()
    idf_sha = hashlib.sha256(np.asarray([], dtype="<f8").tobytes() if vectorizer is None else
                             np.asarray(vectorizer.idf_, dtype="<f8").tobytes()).hexdigest()
    matrix_sha = hashlib.sha256(fit_matrix.indptr.tobytes() + fit_matrix.indices.tobytes() + fit_matrix.data.tobytes()).hexdigest()
    target_sha = hashlib.sha256(np.asarray(target, dtype=np.int8).tobytes()).hexdigest()
    return {
        "status": "MODEL_STATE_DURABLE",
        "card_id": CARD_ID, "stage": STAGE, "cutoff": int(cutoff),
        "fit_rows": int(len(target)), "fit_cycle_min": 47, "fit_cycle_max": int(cutoff),
        "training_design_shape": [int(x) for x in fit_matrix.shape], "training_design_nnz": int(fit_matrix.nnz),
        "training_design_sha256": matrix_sha, "target_sha256": target_sha,
        "preprocessing_state": {
            "recipe": "exact C2_B structured numeric imputation/scaling + OHE + TF-IDF 1-2 grams + stream missing indicator",
            "fit_rows_only": int(len(target)), "feature_names": list(base.input_feature_names),
            "structured_feature_count_including_missing_indicator": int(base._transform_state["structured_columns"]),
            "tfidf_feature_count": int(base._transform_state["text_columns"]),
            "tfidf_vocabulary_size": int(len(vocab)), "tfidf_vocabulary_sha256": vocab_sha,
            "tfidf_idf_sha256": idf_sha,
            "structured_preprocessor": base._structured_preprocessor.audit_state(),
            "c2b_historical_preprocessing_parity": base._mlp_c2b_preprocessing_parity,
        },
        "estimator_parameters": estimator.get_params(deep=True),
        "n_iter": int(estimator.n_iter_), "loss_curve": [float(x) for x in estimator.loss_curve_],
        "initial_training_loss": float(estimator.loss_curve_[0]),
        "final_training_loss": float(estimator.loss_curve_[-1]),
        "convergence_warnings": warnings_seen, "convergence_certified": False,
        "fit_seconds": float(seconds), "model_joblib_sha256": model_sha,
        "score_labels_passed_to_fit": False, "test_used": False,
    }


def _check_preflight() -> dict[str, Any]:
    identity, signature = frozen_identity()
    if not PREFLIGHT.is_file() or not SYNTHETIC_REPORT.is_file():
        raise RuntimeError("Preflight or synthetic E2E report is missing")
    preflight = _json(PREFLIGHT)
    synthetic = _json(SYNTHETIC_REPORT)
    if preflight.get("freeze_signature") != signature or preflight.get("status") != "PASS":
        raise RuntimeError("Frozen memory/input preflight did not PASS for current identity")
    if synthetic.get("status") != "PASS" or preflight.get("synthetic_report_sha256") != _sha(SYNTHETIC_REPORT):
        raise RuntimeError("Synthetic actual-payload E2E did not PASS or its hash changed")
    return {"identity": identity, "signature": signature, "preflight": preflight, "synthetic": synthetic}


def prepare() -> dict[str, Any]:
    """Freeze source/cache identity and complete synthetic plus real transform preflight."""
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_ROOT.mkdir(parents=True, exist_ok=True)
    create_freeze = False
    if FREEZE.exists():
        identity, signature = frozen_identity()
    else:
        identity = make_identity()
        signature = hashlib.sha256(json.dumps(identity, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
        frozen = {"schema": 1, "id": CARD_ID, "identity": identity, "signature": signature,
                  "frozen_at_local": pd.Timestamp.now(tz="Asia/Jakarta").isoformat(),
                  "outcome_blind_to_official_mlp_fits": True}
        create_freeze = True
    if PREFLIGHT.is_file() and SYNTHETIC_REPORT.is_file():
        previous = _json(PREFLIGHT)
        if previous.get("freeze_signature") != signature or previous.get("synthetic_report_sha256") != _sha(SYNTHETIC_REPORT):
            raise RuntimeError("Existing MLP preflight no longer matches its frozen identity")
        return previous
    synthetic = run_synthetic_e2e()
    _write_once(SYNTHETIC_REPORT, synthetic)
    if create_freeze:
        _write_once(FREEZE, frozen)
    context = _load_context()
    resource: dict[str, Any] = {}
    for cutoff in ORIGINS:
        base, fit_matrix, y_fit, score_matrix, rows = _design_for_origin(context, cutoff)
        estimate = estimate_peak_bytes(fit_matrix, score_matrix)
        current = _runtime_check(estimate)
        resource[str(cutoff)] = {
            **current,
            "fit_rows": int(len(y_fit)), "score_rows": int(len(rows)),
            "fit_matrix_shape": [int(x) for x in fit_matrix.shape],
            "fit_matrix_nnz": int(fit_matrix.nnz),
            "score_matrix_shape": [int(x) for x in score_matrix.shape],
            "score_matrix_nnz": int(score_matrix.nnz),
            "preprocessing_fit_rows_only": True,
            "c2b_historical_preprocessing_parity": base._mlp_c2b_preprocessing_parity,
        }
        del base, fit_matrix, score_matrix, y_fit
    status = "PASS" if all(v["status"] == "PASS" for v in resource.values()) and synthetic["status"] == "PASS" else "BLOCKED_PREFLIGHT"
    result = {
        "status": status, "campaign": CARD_ID, "freeze_signature": signature,
        "source_commit": identity["source_commit"], "official_estimator_fits": 0,
        "official_cutoffs_fit": [], "official_train_sha256": identity["input_sha256"]["data/raw/classification/Train.csv"],
        "test_read": False, "sample_submission_read": False, "kaggle_access": False,
        "synthetic_report_sha256": _sha(SYNTHETIC_REPORT), "synthetic_status": synthetic["status"],
        "resource_by_origin": resource,
        "budget_before_fit": ledger_summary(LEDGER),
        "budget_fits_authorized_by_protocol": 2,
        "checks": {
            "historical_baseline_and_controls_verified": True,
            "fit_only_C2_B_transforms_built_for_both_origins": True,
            "actual_design_resource_estimate_with_reserve": all(v["status"] == "PASS" for v in resource.values()),
            "synthetic_e2e_persistence_failure_resume_and_all_views": synthetic["status"] == "PASS",
        },
    }
    _write_once(PREFLIGHT, result)
    return result


def _bundle_identity(identity: Mapping[str, Any], signature: str, cutoff: int, model_sha: str,
                     state_sha: str) -> dict[str, Any]:
    return {"campaign": CARD_ID, "stage": STAGE, "cutoff": int(cutoff),
        "fit_cycle_min": 47, "fit_cycle_max": int(cutoff), "source_commit": identity["source_commit"],
        "freeze_signature": signature, "protocol_sha256": identity["protocol_sha256"],
        "model_joblib_sha256": model_sha, "fit_state_sha256": state_sha,
        "threshold": THRESHOLD, "candidate_policy": "present=.75*sparse25+.25*MLP; missing=frozen-CB; threshold=.50"}


def _fit_event_state(fit_id: str) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    events = read_events(LEDGER)
    starts = [e for e in events if e.get("event") == "FIT_STARTED" and e.get("fit_id") == fit_id]
    terminals = [e for e in events if e.get("event") in {"FIT_SAVED", "FIT_FAILED"} and e.get("fit_id") == fit_id]
    if len(starts) > 1 or len(terminals) > 1:
        raise RuntimeError(f"Duplicate budget ledger events for {fit_id}")
    return starts[0] if starts else None, terminals[0] if terminals else None


def _acquire_lock() -> Path:
    RUN_ROOT.mkdir(parents=True, exist_ok=True)
    lock = RUN_ROOT / ".active_fit.lock"
    try:
        with lock.open("x", encoding="utf-8") as stream:
            stream.write(json.dumps({"pid": os.getpid(), "at_utc": pd.Timestamp.now(tz="UTC").isoformat()}))
            stream.flush()
            os.fsync(stream.fileno())
    except FileExistsError as exc:
        raise RuntimeError(f"A MLP fit lock already exists; inspect before resuming: {lock}") from exc
    return lock


def fit_origin(cutoff: int) -> dict[str, Any]:
    if cutoff not in ORIGINS:
        raise ValueError(f"Only frozen origins {ORIGINS} are allowed")
    if cutoff == 67:
        events = read_events(LEDGER)
        first_terminal = any(event.get("event") in {"FIT_SAVED", "FIT_FAILED"} and
                             event.get("fit_id") == "CLASSIFICATION_SPARSE_MLP_V1_55" for event in events)
        if not first_terminal:
            raise RuntimeError("Frozen continuation order requires cutoff 55 to be terminal before cutoff 67")
    frozen = _check_preflight()
    identity = frozen["identity"]
    signature = frozen["signature"]
    context = _load_context()
    base, fit_matrix, y_fit, score_matrix, score_rows = _design_for_origin(context, cutoff)
    resource = _runtime_check(estimate_peak_bytes(fit_matrix, score_matrix))
    if resource["status"] != "PASS":
        raise RuntimeError(f"BLOCKED_MEMORY before fit start at cutoff {cutoff}: {resource}")
    if fit_matrix.shape[0] != int(read_protocol()["fit_rows_expected"][str(cutoff)]):
        raise RuntimeError(f"Fit row count differs from frozen protocol at cutoff {cutoff}")
    baseline = context["baselines"][cutoff]
    cycles = context["cycles"][score_rows]
    score_ids = context["frame"].iloc[score_rows]["record_id"].astype(str).to_numpy()
    np.testing.assert_array_equal(score_rows, baseline["rows"])
    np.testing.assert_array_equal(score_ids, context["frame"].iloc[score_rows]["record_id"].astype(str).to_numpy())
    fit_id = f"CLASSIFICATION_SPARSE_MLP_V1_{cutoff}"
    fit_dir = RUN_ROOT / f"fit{cutoff}"
    model_path = fit_dir / "fitted_state.joblib"
    state_path = fit_dir / "fit_state.json"
    bundle_path = fit_dir / "bundle"
    starts, terminal = _fit_event_state(fit_id)
    if starts and terminal and terminal.get("event") == "FIT_FAILED":
        raise RuntimeError(f"Frozen fit {fit_id} already failed; no implicit retry")

    lock = _acquire_lock()
    started_here = False
    try:
        fit_dir.mkdir(parents=True, exist_ok=True)
        if not model_path.exists():
            if starts:
                raise RuntimeError(f"FIT_STARTED exists without durable model state for {fit_id}; refusing refit")
            reserve_card(LEDGER, card_id=CARD_ID, stage=STAGE, fits=2, category="continuation")
            start_fit(LEDGER, fit_id=fit_id, card_id=CARD_ID, stage=STAGE,
                      cutoff=cutoff, component="MLPClassifier32")
            started_here = True
            estimator = make_estimator()
            caught: list[str] = []
            start_time = perf_counter()
            try:
                with warnings.catch_warnings(record=True) as found, threadpool_limits(limits=THREADS):
                    warnings.simplefilter("always")
                    estimator.fit(fit_matrix, y_fit)
                    caught = [f"{w.category.__name__}: {w.message}" for w in found]
                seconds = perf_counter() - start_time
                if int(estimator.n_iter_) != 30 or len(estimator.loss_curve_) != 30:
                    raise RuntimeError(f"Fixed MLP iteration count differs at cutoff {cutoff}: {estimator.n_iter_}")
                loss = np.asarray(estimator.loss_curve_, dtype=np.float64)
                if not np.isfinite(loss).all():
                    raise FloatingPointError("Official MLP loss curve contains nonfinite values")
                # Persist the fitted estimator and preprocessing before scoring
                # or report construction so interruption can resume without refit.
                model_sha = _atomic_joblib({"base": base, "estimator": estimator}, model_path)
                state = _fit_state(base, estimator, fit_matrix, y_fit, cutoff, caught, seconds, model_sha)
                state["freeze_signature"] = signature
                state["input_hashes"] = identity["input_sha256"]
                _write_once(state_path, state)
            except Exception as exc:
                _finish_failure_if_needed(fit_id, exc)
                raise
        else:
            if not starts:
                raise RuntimeError(f"Durable model state exists without FIT_STARTED event: {fit_id}")
            if not state_path.is_file():
                raise RuntimeError(f"Durable model exists without state receipt: {fit_id}")
            state = _json(state_path)
            if state.get("freeze_signature") != signature or state.get("model_joblib_sha256") != _sha(model_path):
                raise RuntimeError(f"Durable fitted state does not match frozen identity for {fit_id}")
            stored = joblib.load(model_path)
            base, estimator = stored["base"], stored["estimator"]
        if state_path.is_file():
            state = _json(state_path)
            model_sha = _sha(model_path)
            if state.get("model_joblib_sha256") != model_sha:
                raise RuntimeError("Fitted model binary hash differs from fit-state receipt")
        else:
            raise RuntimeError("Fitted state receipt was not durably written")

        if bundle_path.exists():
            bundle_id = _bundle_identity(identity, signature, cutoff, model_sha, _sha(state_path))
            saved = read_bundle(bundle_path, bundle_id, task="classification")
            if terminal is None:
                finish_fit(LEDGER, fit_id=fit_id, status="FIT_SAVED", receipt_sha256=saved.receipt["receipt_sha256"])
            elif terminal.get("event") != "FIT_SAVED" or terminal.get("receipt_sha256") != saved.receipt["receipt_sha256"]:
                raise RuntimeError("Existing bundle conflicts with terminal budget event")
            return {"status": "RESUMED_SAVED_BUNDLE", "cutoff": cutoff,
                    "receipt_sha256": saved.receipt["receipt_sha256"], "model_sha256": model_sha}
        if terminal and terminal.get("event") == "FIT_SAVED":
            raise RuntimeError(f"Ledger claims FIT_SAVED but bundle missing for {fit_id}")

        score_proba = estimator.predict_proba(score_matrix)
        if not np.array_equal(np.asarray(estimator.classes_), np.asarray([0, 1])):
            raise RuntimeError("MLP class order does not match frozen positive-class convention")
        raw_mlp = np.asarray(score_proba[:, 1], dtype=np.float64)
        base_probability = np.asarray(baseline["baseline"], dtype=np.float64)
        c2b = np.asarray(baseline["c2b"], dtype=np.float64)
        cb = np.asarray(baseline["cb"], dtype=np.float64)
        missing = np.asarray(baseline["missing"], dtype=bool)
        candidate = route_probabilities("candidate", base_probability, raw_mlp, c2b, cb, missing)
        matched = route_probabilities("matched_control", base_probability, raw_mlp, c2b, cb, missing)
        ids = np.asarray(context["frame"].iloc[score_rows]["record_id"].astype(str), dtype=str)
        payload = {
            "score_rows": score_rows.astype(np.int64), "score_record_ids": ids,
            "cycle_code": cycles.astype(np.int16), "prediction": candidate,
            "baseline_prediction": base_probability, "raw_mlp_probability": raw_mlp,
            "matched_prediction": matched, "control_c2b_probability": c2b,
            "control_cb_probability": cb, "control_stream_missing": missing,
            "payload_meta": {
                "card_id": CARD_ID, "stage": STAGE, "cutoff": int(cutoff),
                "probability_semantics": "positive-class probability; prediction is final routed probability",
                "baseline_probability_semantics": "frozen sparse25 final routed probability",
                "candidate_route": "present=.75*sparse25+.25*raw_MLP; missing=frozen-CB; threshold=.50",
                "matched_control_route": "present=.75*sparse25+.25*C2_B; missing=frozen-CB; threshold=.50",
                "test_used": False, "score_labels_passed_to_fit": False,
                "model_joblib_sha256": model_sha, "fit_state_sha256": _sha(state_path),
            },
        }
        bundle_id = _bundle_identity(identity, signature, cutoff, model_sha, _sha(state_path))
        receipt = publish_bundle(bundle_path, payload, bundle_id, task="classification")
        saved = read_bundle(bundle_path, bundle_id, task="classification")
        rebuilt = route_probabilities("candidate", saved.arrays["baseline_prediction"],
            saved.arrays["raw_mlp_probability"], saved.arrays["control_c2b_probability"],
            saved.arrays["control_cb_probability"], saved.arrays["control_stream_missing"])
        np.testing.assert_array_equal(rebuilt, saved.arrays["prediction"])
        if saved.receipt["receipt_sha256"] != receipt["receipt_sha256"]:
            raise RuntimeError("Published bundle receipt did not read back exactly")
        if started_here or terminal is None:
            finish_fit(LEDGER, fit_id=fit_id, status="FIT_SAVED", receipt_sha256=receipt["receipt_sha256"])
        return {"status": "FIT_SAVED", "cutoff": cutoff,
                "receipt_sha256": receipt["receipt_sha256"], "model_sha256": model_sha,
                "fit_state_sha256": _sha(state_path), "fit_rows": int(len(y_fit)),
                "score_rows": int(len(score_rows)), "resource": resource}
    finally:
        try:
            lock.unlink(missing_ok=True)
        except OSError:
            pass


def _finish_failure_if_needed(fit_id: str, exc: Exception) -> None:
    starts, terminal = _fit_event_state(fit_id)
    if starts and terminal is None:
        finish_fit(LEDGER, fit_id=fit_id, status="FIT_FAILED", error=f"{type(exc).__name__}: {exc}")


def _load_bundle_output(cutoff: int, context: Mapping[str, Any], identity: Mapping[str, Any], signature: str) -> dict[str, Any]:
    fit_dir = RUN_ROOT / f"fit{cutoff}"
    model_path, state_path = fit_dir / "fitted_state.joblib", fit_dir / "fit_state.json"
    if not model_path.is_file() or not state_path.is_file():
        raise RuntimeError(f"Fitted MLP state missing for cutoff {cutoff}")
    state = _json(state_path)
    bundle_id = _bundle_identity(identity, signature, cutoff, _sha(model_path), _sha(state_path))
    bundle = read_bundle(fit_dir / "bundle", bundle_id, task="classification")
    arrays = bundle.arrays
    cycles = context["cycles"][arrays["score_rows"]]
    if not np.array_equal(arrays["score_record_ids"].astype(str), context["frame"].iloc[arrays["score_rows"]]["record_id"].astype(str).to_numpy()):
        raise RuntimeError(f"Saved MLP bundle record IDs differ from official Train at cutoff {cutoff}")
    result = {key: arrays[key] for key in arrays} | {
        "origin": np.full(len(arrays["score_rows"]), cutoff, dtype=np.int64),
        "cycle_code": cycles, "loss_curve": state["loss_curve"],
        "fit_state": state, "receipt": bundle.receipt,
    }
    result["c2b_diagnostic"] = np.where(
        arrays["control_stream_missing"].astype(bool),
        arrays["control_cb_probability"], arrays["control_c2b_probability"])
    return result


def summarize() -> dict[str, Any]:
    frozen = _check_preflight()
    identity, signature = frozen["identity"], frozen["signature"]
    context = _load_context()
    if (RUN_ROOT / ".active_fit.lock").exists():
        raise RuntimeError("Cannot summarize while an official MLP fit lock is active")
    events = read_events(LEDGER)
    fit_status: dict[str, Any] = {}
    outputs: dict[int, dict[str, Any]] = {}
    pending = False
    for cutoff in ORIGINS:
        fit_id = f"{CARD_ID}_{cutoff}"
        started = [event for event in events if event.get("event") == "FIT_STARTED" and event.get("fit_id") == fit_id]
        terminal = [event for event in events if event.get("event") in {"FIT_SAVED", "FIT_FAILED"} and event.get("fit_id") == fit_id]
        if len(started) > 1 or len(terminal) > 1:
            raise RuntimeError(f"Duplicate ledger events for fit {fit_id}")
        status = "NOT_STARTED" if not started else "PENDING" if not terminal else terminal[0]["event"]
        fit_status[str(cutoff)] = {"fit_id": fit_id, "status": status,
            "cutoff": cutoff, "error": terminal[0].get("error") if terminal else None,
            "receipt_sha256": terminal[0].get("receipt_sha256") if terminal else None}
        pending = pending or status == "PENDING"
        bundle_exists = (RUN_ROOT / f"fit{cutoff}/bundle/receipt.json").is_file()
        if status == "FIT_SAVED" and bundle_exists:
            outputs[cutoff] = _load_bundle_output(cutoff, context, identity, signature)
        elif status == "FIT_SAVED" or bundle_exists:
            raise RuntimeError(f"Ledger and durable bundle state disagree for cutoff {cutoff}")
    complete = len(outputs) == len(ORIGINS) and all(fit_status[str(c)]["status"] == "FIT_SAVED" for c in ORIGINS)
    all_terminal = all(fit_status[str(c)]["status"] in {"FIT_SAVED", "FIT_FAILED"} for c in ORIGINS)
    metrics: dict[str, Any] | None = None
    groups: list[dict[str, Any]] = []
    per_cycle: list[dict[str, Any]] = []
    if complete:
        metrics, groups, per_cycle, gate = evaluate_views(outputs, context["frame"], context["target"])
    else:
        gate = {"status": "NOT_EVALUATED_INCOMPLETE_FITS", "checks": {}, "thresholds": {}}
    release_event = _release_if_reserved() if all_terminal and not pending else None
    budget_snapshot_path = REPORT_ROOT / "budget_snapshot.json"
    if all_terminal:
        if budget_snapshot_path.exists():
            budget_snapshot = _json(budget_snapshot_path)
        else:
            budget_snapshot = {"stage": STAGE, "release_event": release_event,
                               "ledger_summary": ledger_summary(LEDGER)}
            _write_once(budget_snapshot_path, budget_snapshot)
        budget_result = budget_snapshot["ledger_summary"]
        budget_snapshot_sha = _sha(budget_snapshot_path)
    else:
        budget_result = ledger_summary(LEDGER)
        budget_snapshot_sha = None
    fits = {str(cutoff): ({"fit_rows": outputs[cutoff]["fit_state"]["fit_rows"],
        "score_rows": len(outputs[cutoff]["score_rows"]),
        "model_joblib_sha256": outputs[cutoff]["fit_state"]["model_joblib_sha256"],
        "fit_state_sha256": _sha(RUN_ROOT / f"fit{cutoff}/fit_state.json"),
        "bundle_receipt_sha256": outputs[cutoff]["receipt"]["receipt_sha256"],
        "n_iter": outputs[cutoff]["fit_state"]["n_iter"],
        "loss_curve": outputs[cutoff]["loss_curve"],
        "convergence_warnings": outputs[cutoff]["fit_state"]["convergence_warnings"]} if cutoff in outputs else fit_status[str(cutoff)])
        for cutoff in ORIGINS}
    result = {
        "status": ("COMPLETE_REVIEW_REQUIRED" if complete else
                   "TERMINAL_FAILURE_REVIEW_REQUIRED" if all_terminal else "INCOMPLETE_CHECKPOINT"),
        "scientific_verdict": gate["status"] if complete else
            "FIT_FAILURE_NOT_SCIENTIFICALLY_EVALUATED" if all_terminal else "NOT_EVALUATED_INCOMPLETE_FITS",
        "campaign": CARD_ID, "freeze_signature": signature,
        "source_commit": identity["source_commit"],
        "official_train_sha256": _sha(context["train_path"]),
        "official_train_rows": int(len(context["frame"])),
        "official_fits": 2, "fit_origins": fits,
        "threshold": THRESHOLD, "metrics_by_view": metrics,
        "protected_groups": groups, "per_cycle": per_cycle,
        "fit_status_by_origin": fit_status,
        "frozen_gate": gate,
        "comparators": {"sparse25": "frozen incumbent routed probability",
            "matched_control": "present=.75*sparse25+.25*C2_B; missing=frozen-CB; zero new estimator fits",
            "C2_B_diagnostic": "raw C2_B present probability; frozen-CB missing branch"},
        "interpretation_limits": [
            "This is a frozen two-origin pilot and does not establish an untouched holdout result.",
            "The matched allocation control distinguishes a new head from simply reducing sparse25 allocation; it does not causally isolate nonlinear capacity from optimizer/model differences.",
            "Thirty epochs are fixed by protocol. ConvergenceWarning is reported; no convergence is claimed and no epoch rescue was attempted.",
            "The diagnostic is limited to anonymous token features and these temporal origins.",
            "No Test, SampleSubmission, full-Train refit, CSV creation, Kaggle access, or upload occurred.",
        ],
        "budget": budget_result,
        "budget_snapshot_sha256": budget_snapshot_sha, "test_read": False,
        "sample_submission_read": False, "kaggle_access": False,
    }
    if all_terminal:
        _write_once(REPORT_ROOT / "results.json", result)
    if complete:
        _write_csv_once(REPORT_ROOT / "protected_groups.csv", groups)
        _write_csv_once(REPORT_ROOT / "per_cycle.csv", per_cycle)
    return result


def _write_csv_once(path: Path, rows: list[dict[str, Any]]) -> None:
    import csv
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = list(dict.fromkeys(key for row in rows for key in row))
    if path.exists():
        with path.open("r", encoding="utf-8", newline="") as stream:
            previous = list(csv.DictReader(stream))
        expected = [{key: str(row.get(key, "")) for key in columns} for row in rows]
        if previous != expected:
            raise RuntimeError(f"Immutable CSV differs; refusing overwrite: {path}")
        return
    with path.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in columns})
        stream.flush()
        os.fsync(stream.fileno())


def _release_if_reserved() -> dict[str, Any] | None:
    events = read_events(LEDGER)
    reservation = [event for event in events if event.get("event") == "CARD_FITS_RESERVED"
                   and event.get("reservation_key") == f"{CARD_ID}:{STAGE}"]
    if not reservation:
        return None
    starts = [event for event in events if event.get("event") == "FIT_STARTED"
              and event.get("reservation_key") == f"{CARD_ID}:{STAGE}"]
    terminal_ids = {event.get("fit_id") for event in events if event.get("event") in {"FIT_SAVED", "FIT_FAILED"}}
    pending = [event for event in starts if event.get("fit_id") not in terminal_ids]
    if pending:
        raise RuntimeError("Cannot release MLP fit slots with an unfinished started fit")
    return release_unused(LEDGER, card_id=CARD_ID, stage=STAGE)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("prepare", "fit", "summarize"))
    parser.add_argument("cutoff", nargs="?", type=int)
    args = parser.parse_args(argv)
    if args.command == "prepare":
        result = prepare()
    elif args.command == "fit":
        if args.cutoff is None:
            parser.error("fit requires cutoff 55 or 67")
        result = fit_origin(args.cutoff)
    elif args.command == "summarize":
        result = summarize()
    else:
        result = summarize()
    print(json.dumps(result, ensure_ascii=False, allow_nan=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
