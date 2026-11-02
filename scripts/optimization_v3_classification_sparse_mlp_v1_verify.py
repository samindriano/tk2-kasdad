"""Independent zero-fit verifier for the frozen sparse MLP pilot."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (average_precision_score, confusion_matrix, f1_score,
                            log_loss, precision_score, recall_score, roc_auc_score)

from scripts.research_campaign_classification import load_train
from src.data.audit import sha256 as file_sha256
from src.models.classification_models import _number_array
from src.models.classification_round2_models import stream_missing_mask
from src.models.classification_sparse_mlp_v1 import transform_score_design
from src.models.research_next_contract import read_bundle
from src.models.research_next_budget import ledger_summary, read_events


CARD_ID = "CLASSIFICATION_SPARSE_MLP_V1"
STAGE = "MLP_NEW_PILOT"
ORIGINS = (55, 67)
THRESHOLD = 0.50
CONFIG = ROOT / "experiments/optimization_v3/classification_mlp_v1"
RUN_ROOT = ROOT / "experiments/runs/optimization_v3_classification_mlp_v1"
REPORT_ROOT = ROOT / "reports/optimization_v3/classification_mlp_v1"
FREEZE = CONFIG / "freeze.json"
LEDGER = ROOT / "experiments/runs/optimization_v3_research_next_v1/budget.jsonl"


def _sha(path: Path) -> str:
    return file_sha256(Path(path))


def _json(path: Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _array_sha(value: Any) -> str:
    array = np.asarray(value, dtype="<f8")
    return hashlib.sha256(array.tobytes(order="C")).hexdigest()


def _same(left: Any, right: Any, *, atol: float = 1e-12) -> bool:
    if isinstance(left, dict) and isinstance(right, dict):
        return set(left) == set(right) and all(_same(left[key], right[key], atol=atol) for key in left)
    if isinstance(left, list) and isinstance(right, list):
        return len(left) == len(right) and all(_same(a, b, atol=atol) for a, b in zip(left, right))
    if isinstance(left, (int, float)) and isinstance(right, (int, float)) and not isinstance(left, bool) and not isinstance(right, bool):
        return bool(np.isclose(float(left), float(right), atol=atol, rtol=0.0, equal_nan=False))
    return left == right


def _identity_audit() -> tuple[dict[str, Any], str, list[str]]:
    frozen = _json(FREEZE)
    identity = frozen["identity"]
    expected_signature = hashlib.sha256(json.dumps(identity, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
    problems: list[str] = []
    if frozen.get("signature") != expected_signature:
        problems.append("freeze signature mismatch")
    protocol = ROOT / "experiments/optimization_v3/classification_mlp_v1/protocol.json"
    if _sha(protocol) != identity.get("protocol_sha256"):
        problems.append("protocol hash mismatch")
    for relative, expected in identity.get("code_sha256", {}).items():
        path = ROOT / relative
        if not path.is_file() or _sha(path) != expected:
            problems.append(f"code hash mismatch: {relative}")
    for relative, expected in identity.get("input_sha256", {}).items():
        path = ROOT / relative
        if not path.is_file() or _sha(path) != expected:
            problems.append(f"input hash mismatch: {relative}")
    if identity.get("test_read") is not False or identity.get("sample_submission_read") is not False or identity.get("kaggle_access") is not False:
        problems.append("frozen source boundary flags are not false")
    return identity, expected_signature, problems


def _reconstruct_baseline(frame: pd.DataFrame, target: np.ndarray, cutoff: int) -> dict[str, Any]:
    cycles = pd.to_numeric(frame["cycle_code"], errors="raise").to_numpy(dtype=np.int64)
    rows_expected = np.flatnonzero((cycles > cutoff) & (cycles <= cutoff + 12))
    fit_rows = np.flatnonzero((cycles >= 47) & (cycles <= cutoff))
    round1_path = ROOT / f"experiments/runs/classification_round1_v1/fit{cutoff}__C2_B.npz"
    round1_audit_path = ROOT / f"reports/classification_round1/fit_audits/fit{cutoff}.json"
    cb_path = ROOT / f"experiments/runs/dual_cls_catboost_v1/fit{cutoff}.npz"
    cb_meta_path = cb_path.with_suffix(".json")
    sparse_path = ROOT / f"experiments/runs/optimization_v3_classification_sparse/fit{cutoff}.npz"
    sparse_meta_path = sparse_path.with_suffix(".json")
    with np.load(round1_path, allow_pickle=False) as archive:
        c2b_rows = archive["outer_rows"].astype(np.int64, copy=True)
        c2b = archive["probability"].astype(np.float64, copy=True)
    c2b_audit_root = _json(round1_audit_path)
    c2b_audit = c2b_audit_root["C2_B"]
    cb_meta = _json(cb_meta_path)
    with np.load(cb_path, allow_pickle=False) as archive:
        cb_rows = archive["rows"].astype(np.int64, copy=True)
        cb = archive["candidate"].astype(np.float64, copy=True)
    sparse_meta = _json(sparse_meta_path)
    with np.load(sparse_path, allow_pickle=False) as archive:
        sparse_rows = archive["rows"].astype(np.int64, copy=True)
        sparse_raw = archive["raw_candidate"].astype(np.float64, copy=True)
    if not (np.array_equal(c2b_rows, rows_expected) and np.array_equal(cb_rows, rows_expected) and np.array_equal(sparse_rows, rows_expected)):
        raise RuntimeError(f"Control cache row identities differ at cutoff {cutoff}")
    if not (int(c2b_audit.get("fit_rows", -1)) == len(fit_rows) and int(cb_meta.get("fit_rows", -1)) == len(fit_rows)):
        raise RuntimeError(f"Control cache fit population mismatch at cutoff {cutoff}")
    if _array_sha(c2b) != c2b_audit.get("probability_sha256"):
        raise RuntimeError(f"C2_B probability digest mismatch at cutoff {cutoff}")
    if _sha(cb_path) != cb_meta.get("npz_sha256") or _array_sha(cb) != cb_meta.get("prediction_sha256"):
        raise RuntimeError(f"CB cache digest mismatch at cutoff {cutoff}")
    if _sha(sparse_path) != sparse_meta.get("npz_sha256"):
        raise RuntimeError(f"Sparse25 cache digest mismatch at cutoff {cutoff}")
    if not all(np.isfinite(v).all() and np.all((v >= 0) & (v <= 1)) for v in (c2b, cb, sparse_raw)):
        raise RuntimeError(f"Control probabilities invalid at cutoff {cutoff}")
    missing = stream_missing_mask(frame.iloc[rows_expected])
    baseline = np.where(missing, cb, 0.75 * c2b + 0.25 * sparse_raw)
    c19_path = ROOT / f"experiments/runs/optimization_v3_batch1/C19/C19_fit{cutoff}.npz"
    c19_sidecar = _json(c19_path.with_suffix(".json"))
    with np.load(c19_path, allow_pickle=False) as archive:
        if not np.array_equal(archive["score_rows"], rows_expected):
            raise RuntimeError(f"C19 historical cache row mismatch at cutoff {cutoff}")
        if not np.array_equal(archive["baseline_prediction"], baseline):
            raise RuntimeError(f"C19 historical sparse25 baseline differs at cutoff {cutoff}")
        if not np.array_equal(archive["control__c2b_probability"], c2b):
            raise RuntimeError(f"C19 C2_B control differs at cutoff {cutoff}")
        if not np.array_equal(archive["control__cb_probability"], cb):
            raise RuntimeError(f"C19 CB control differs at cutoff {cutoff}")
    if _sha(c19_path) != c19_sidecar.get("npz_sha256"):
        raise RuntimeError(f"C19 historical sidecar hash mismatch at cutoff {cutoff}")
    return {"rows": rows_expected, "fit_rows": fit_rows, "c2b": c2b, "cb": cb,
        "sparse_raw": sparse_raw, "baseline": baseline, "missing": missing,
        "hashes": {str(p.relative_to(ROOT).as_posix()): _sha(p) for p in (
            round1_path, round1_audit_path, cb_path, cb_meta_path, sparse_path, sparse_meta_path, c19_path, c19_path.with_suffix(".json"))}}


def _metric(y: np.ndarray, p: np.ndarray) -> dict[str, Any]:
    y = np.asarray(y, dtype=np.int8)
    p = np.asarray(p, dtype=np.float64)
    pred = (p >= THRESHOLD).astype(np.int8)
    cm = confusion_matrix(y, pred, labels=[0, 1])
    both = len(np.unique(y)) == 2
    return {
        "n": int(len(y)),
        "macro_f1": float(f1_score(y, pred, average="macro", labels=[0, 1], zero_division=0)),
        "roc_auc": float(roc_auc_score(y, p)) if both else None,
        "average_precision": float(average_precision_score(y, p)) if both else None,
        "log_loss": float(log_loss(y, p, labels=[0, 1])),
        "brier_score": float(np.mean((p-y) ** 2)),
        "precision_positive": float(precision_score(y, pred, pos_label=1, zero_division=0)),
        "recall_positive": float(recall_score(y, pred, pos_label=1, zero_division=0)),
        "precision_negative": float(precision_score(y, pred, pos_label=0, zero_division=0)),
        "recall_negative": float(recall_score(y, pred, pos_label=0, zero_division=0)),
        "positive_rate": float(y.mean()), "predicted_positive_rate": float(pred.mean()),
        "confusion_matrix_labels_0_1": cm.tolist(),
    }


def _views(outputs: dict[int, dict[str, Any]], frame: pd.DataFrame, target: np.ndarray
           ) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    definitions = {
        "primary": [(55, 56, 59), (67, 68, 71)], "W1": [(55, 56, 59)], "W4": [(67, 68, 71)],
        "H12_55": [(55, 56, 67)], "H12_67": [(67, 68, 79)],
        "H12_pooled": [(55, 56, 67), (67, 68, 79)],
    }
    parsed, _missing, _invalid = _number_array(frame["text_load"])
    bins: dict[int, np.ndarray] = {}
    for cutoff in outputs:
        cycle = pd.to_numeric(frame["cycle_code"], errors="raise").to_numpy(dtype=np.int64)
        finite = parsed[(cycle >= 47) & (cycle <= cutoff) & np.isfinite(parsed)]
        edges = np.unique(np.quantile(finite, [0.25, 0.50, 0.75]))
        labels = np.full(len(frame), -1, dtype=np.int64)
        valid = np.isfinite(parsed)
        labels[valid] = np.searchsorted(edges, parsed[valid], side="right")
        bins[cutoff] = labels
    cycles_all = pd.to_numeric(frame["cycle_code"], errors="raise").to_numpy(dtype=np.int64)
    metrics: dict[str, Any] = {}
    groups: list[dict[str, Any]] = []
    per_cycle: list[dict[str, Any]] = []
    for name, blocks in definitions.items():
        picked = []
        for cutoff, low, high in blocks:
            record = outputs[cutoff]
            take = (record["cycle_code"] >= low) & (record["cycle_code"] <= high)
            picked.append({key: record[key][take] for key in (
                "score_rows", "prediction", "baseline_prediction", "matched_prediction", "c2b_diagnostic", "origin")})
        rows = np.concatenate([part["score_rows"] for part in picked]).astype(np.int64)
        if len(np.unique(rows)) != len(rows):
            raise RuntimeError(f"Duplicate Train rows in verification view {name}")
        origins = np.concatenate([part["origin"] for part in picked]).astype(np.int64)
        y = target[rows]
        predictions = {key: np.concatenate([part[key] for part in picked]).astype(np.float64) for key in (
            "prediction", "baseline_prediction", "matched_prediction", "c2b_diagnostic")}
        models = {"candidate": _metric(y, predictions["prediction"]),
                  "sparse25": _metric(y, predictions["baseline_prediction"]),
                  "matched_control": _metric(y, predictions["matched_prediction"]),
                  "C2_B_diagnostic": _metric(y, predictions["c2b_diagnostic"])}
        gains = {}
        for other in ("sparse25", "matched_control", "C2_B_diagnostic"):
            gains[f"candidate_minus_{other}"] = {
                field: (None if models["candidate"][field] is None or models[other][field] is None else
                        float(models["candidate"][field] - models[other][field]))
                for field in ("macro_f1", "roc_auc", "average_precision", "log_loss", "brier_score",
                              "precision_positive", "recall_positive", "predicted_positive_rate")}
        metrics[name] = {"n": len(rows), "models": models, "gains": gains,
                         "origins": sorted(set(origins.tolist()))}
        for value in sorted(np.unique(cycles_all[rows]).tolist()):
            take = cycles_all[rows] == value
            candidate_f1 = _metric(y[take], predictions["prediction"][take])["macro_f1"]
            sparse_f1 = _metric(y[take], predictions["baseline_prediction"][take])["macro_f1"]
            per_cycle.append({"view": name, "cycle": int(value), "n": int(take.sum()),
                "candidate_macro_f1": candidate_f1, "sparse25_macro_f1": sparse_f1,
                "matched_macro_f1": _metric(y[take], predictions["matched_prediction"][take])["macro_f1"],
                "candidate_minus_sparse25_macro_f1": float(candidate_f1-sparse_f1)})
        row_frame = frame.iloc[rows]
        missing = stream_missing_mask(row_frame)
        quartile = np.asarray([bins[int(origin)][int(row)] for row, origin in zip(rows, origins)])
        group_maps = {
            "fit_defined_text_load_quartile": np.asarray(["missing" if value < 0 else f"Q{min(int(value)+1,4)}" for value in quartile]),
            "stream_availability": np.where(missing, "missing", "present"),
            "entry_friction": row_frame["entry_friction"].astype(str).to_numpy(),
        }
        for field, labels in group_maps.items():
            for label in sorted(np.unique(labels).tolist()):
                take = labels == label
                gy = y[take]
                n0, n1 = int(np.sum(gy == 0)), int(np.sum(gy == 1))
                supported = bool(int(take.sum()) >= 200 and min(n0, n1) >= 30)
                c = _metric(gy, predictions["prediction"][take])
                b = _metric(gy, predictions["baseline_prediction"][take])
                f1_harm = float(b["macro_f1"] - c["macro_f1"])
                recall_harm = float(b["recall_positive"] - c["recall_positive"])
                groups.append({"view": name, "field": field, "group": str(label),
                    "n": int(take.sum()), "class_0_n": n0, "class_1_n": n1, "supported": supported,
                    "candidate_macro_f1": c["macro_f1"], "sparse25_macro_f1": b["macro_f1"],
                    "f1_harm": f1_harm, "candidate_positive_recall": c["recall_positive"],
                    "sparse25_positive_recall": b["recall_positive"], "positive_recall_harm": recall_harm,
                    "fit_origins": sorted(set(origins[take].tolist())),
                    "fails_0_01_f1": bool(supported and f1_harm > 0.01),
                    "fails_0_03_recall": bool(supported and recall_harm > 0.03),
                    "fails_0_02_recall": bool(supported and recall_harm > 0.02)})
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
        "view_macro_f1_gains_vs_sparse25": gains, "primary_gain_vs_matched_control": primary_matched,
        "supported_groups": len(supported), "unsupported_groups": len(groups)-len(supported),
        "supported_group_failures": [row for row in supported if row["fails_0_01_f1"] or row["fails_0_03_recall"]],
        "original_0_02_recall_failures_reported": [row for row in supported if row["fails_0_02_recall"]],
        "original_0_02_recall_guard_pass": not any(row["fails_0_02_recall"] for row in supported)}
    return metrics, groups, per_cycle, gate


def verify() -> dict[str, Any]:
    identity, signature, problems = _identity_audit()
    frame, target, train_path = load_train(None)
    if len(frame) != 50_608 or _sha(train_path) != identity["input_sha256"].get("data/raw/classification/Train.csv"):
        problems.append("official Classification Train identity mismatch")
    if _sha(train_path) != "875552a68a3d24b0f3e1966d6ddb5671e6f54305a1c526e2f0a76747942ed91c":
        problems.append("official Train raw SHA256 differs from protocol")
    cycles = pd.to_numeric(frame["cycle_code"], errors="raise").to_numpy(dtype=np.int64)
    baselines = {cutoff: _reconstruct_baseline(frame, target, cutoff) for cutoff in ORIGINS}
    outputs: dict[int, dict[str, Any]] = {}
    fit_checks: dict[str, Any] = {}
    fit_events = read_events(LEDGER)
    for cutoff in ORIGINS:
        fit_id = f"{CARD_ID}_{cutoff}"
        starts = [event for event in fit_events if event.get("event") == "FIT_STARTED" and event.get("fit_id") == fit_id]
        terminals = [event for event in fit_events if event.get("event") in {"FIT_SAVED", "FIT_FAILED"} and event.get("fit_id") == fit_id]
        fit_dir = RUN_ROOT / f"fit{cutoff}"
        model_path, state_path = fit_dir / "fitted_state.joblib", fit_dir / "fit_state.json"
        if len(starts) != 1 or len(terminals) != 1 or terminals[0].get("event") != "FIT_SAVED":
            problems.append(f"budget ledger lacks one FIT_SAVED for cutoff {cutoff}")
        if not model_path.is_file() or not state_path.is_file():
            problems.append(f"durable fitted state missing for cutoff {cutoff}")
            continue
        state = _json(state_path)
        model_sha = _sha(model_path)
        if state.get("model_joblib_sha256") != model_sha:
            problems.append(f"model binary hash mismatch at cutoff {cutoff}")
        if state.get("freeze_signature") != signature or state.get("input_hashes") != identity["input_sha256"]:
            problems.append(f"fit state identity mismatch at cutoff {cutoff}")
        expected_fit = np.flatnonzero((cycles >= 47) & (cycles <= cutoff))
        if len(expected_fit) != int(state.get("fit_rows", -1)) or not np.array_equal(expected_fit, baselines[cutoff]["fit_rows"]):
            problems.append(f"fit population mismatch at cutoff {cutoff}")
        stored = joblib.load(model_path)
        base, estimator = stored["base"], stored["estimator"]
        feature_names = list(base.input_feature_names)
        y_fit = target[expected_fit]
        X_fit = frame.iloc[expected_fit].loc[:, feature_names]
        X_score = frame.iloc[baselines[cutoff]["rows"]].loc[:, feature_names]
        from src.models.round1_models import _fit_fingerprint
        if base._fit_fingerprint != _fit_fingerprint(X_fit, tuple(feature_names), y_fit):
            problems.append(f"model preprocessing fit fingerprint mismatch at cutoff {cutoff}")
        archived_c2b = _json(ROOT / f"reports/classification_round1/fit_audits/fit{cutoff}.json")["C2_B"]["fitted_state"]
        if base._fit_fingerprint != archived_c2b.get("fit_data_sha256"):
            problems.append(f"rebuilt MLP preprocessing input differs from archived C2_B fit at cutoff {cutoff}")
        params = estimator.get_params(deep=True)
        expected_params = _json(ROOT / "experiments/optimization_v3/classification_mlp_v1/protocol.json")["model"]["parameters"]
        normalized_params = {key: list(value) if isinstance(value, tuple) else value for key, value in params.items()}
        if set(normalized_params) != set(expected_params) or any(normalized_params[key] != expected_params[key] for key in normalized_params):
            problems.append(f"MLP estimator parameters differ from frozen recipe at cutoff {cutoff}")
        if int(estimator.n_iter_) != 30 or len(estimator.loss_curve_) != 30 or not np.isfinite(estimator.loss_curve_).all():
            problems.append(f"MLP fit schedule/loss curve invalid at cutoff {cutoff}")
        expected_rows = baselines[cutoff]["rows"]
        bundle_identity = {"campaign": CARD_ID, "stage": STAGE, "cutoff": int(cutoff),
            "fit_cycle_min": 47, "fit_cycle_max": int(cutoff), "source_commit": identity["source_commit"],
            "freeze_signature": signature, "protocol_sha256": identity["protocol_sha256"],
            "model_joblib_sha256": model_sha, "fit_state_sha256": _sha(state_path),
            "threshold": THRESHOLD,
            "candidate_policy": "present=.75*sparse25+.25*MLP; missing=frozen-CB; threshold=.50"}
        bundle = read_bundle(fit_dir / "bundle", bundle_identity, task="classification")
        arr = bundle.arrays
        ids = frame.iloc[expected_rows]["record_id"].astype(str).to_numpy()
        expected_cycles = cycles[expected_rows]
        for field, expected in (("score_rows", expected_rows), ("score_record_ids", ids), ("cycle_code", expected_cycles)):
            if not np.array_equal(arr[field], expected):
                problems.append(f"bundle {field} differs from official Train at cutoff {cutoff}")
        baseline = baselines[cutoff]
        if not np.array_equal(arr["baseline_prediction"], baseline["baseline"]):
            problems.append(f"bundle sparse25 baseline differs from independently reconstructed route at cutoff {cutoff}")
        if not np.array_equal(arr["control_c2b_probability"], baseline["c2b"]):
            problems.append(f"C2_B component differs at cutoff {cutoff}")
        if not np.array_equal(arr["control_cb_probability"], baseline["cb"]):
            problems.append(f"CB component differs at cutoff {cutoff}")
        if not np.array_equal(arr["control_stream_missing"].astype(bool), baseline["missing"]):
            problems.append(f"availability route mask differs at cutoff {cutoff}")
        score_matrix = transform_score_design(base, X_score)
        rebuilt_raw = estimator.predict_proba(score_matrix)[:, 1].astype(np.float64)
        expected_candidate = np.where(baseline["missing"], baseline["cb"],
                                      0.75 * baseline["baseline"] + 0.25 * rebuilt_raw)
        expected_matched = np.where(baseline["missing"], baseline["cb"],
                                    0.75 * baseline["baseline"] + 0.25 * baseline["c2b"])
        if not np.array_equal(arr["raw_mlp_probability"], rebuilt_raw):
            problems.append(f"raw MLP probabilities do not reconstruct at cutoff {cutoff}")
        if not np.array_equal(arr["prediction"], expected_candidate):
            problems.append(f"candidate routing mismatch at cutoff {cutoff}")
        if not np.array_equal(arr["matched_prediction"], expected_matched):
            problems.append(f"matched policy routing mismatch at cutoff {cutoff}")
        if not np.isfinite(arr["prediction"]).all() or np.any((arr["prediction"] < 0) | (arr["prediction"] > 1)):
            problems.append(f"invalid candidate probability at cutoff {cutoff}")
        outputs[cutoff] = {
            "score_rows": arr["score_rows"].astype(np.int64), "score_record_ids": arr["score_record_ids"].astype(str),
            "cycle_code": expected_cycles, "prediction": arr["prediction"].astype(np.float64),
            "baseline_prediction": arr["baseline_prediction"].astype(np.float64),
            "matched_prediction": arr["matched_prediction"].astype(np.float64),
            "c2b_diagnostic": np.where(baseline["missing"], baseline["cb"], baseline["c2b"]),
            "origin": np.full(len(expected_rows), cutoff, dtype=np.int64),
        }
        fit_checks[str(cutoff)] = {"fit_rows": int(len(expected_fit)), "score_rows": int(len(expected_rows)),
            "fit_cycle_min": 47, "fit_cycle_max": cutoff, "n_iter": int(estimator.n_iter_),
            "model_joblib_sha256": model_sha, "fit_state_sha256": _sha(state_path),
            "bundle_receipt_sha256": bundle.receipt["receipt_sha256"],
            "preprocessing_fit_only": state.get("preprocessing_state", {}).get("fit_rows_only") == len(expected_fit),
            "target_fingerprint_recomputed": base._fit_fingerprint == _fit_fingerprint(X_fit, tuple(feature_names), y_fit),
            "loss_curve": [float(value) for value in estimator.loss_curve_],
            "convergence_warnings": state.get("convergence_warnings", [])}
    if set(outputs) != set(ORIGINS):
        problems.append("not all frozen origins yielded valid outputs")
    recomputed_metrics: dict[str, Any] = {}
    recomputed_groups: list[dict[str, Any]] = []
    recomputed_cycles: list[dict[str, Any]] = []
    recomputed_gate: dict[str, Any] = {}
    if len(outputs) == len(ORIGINS):
        recomputed_metrics, recomputed_groups, recomputed_cycles, recomputed_gate = _views(outputs, frame, target)
        result_path = REPORT_ROOT / "results.json"
        if not result_path.is_file():
            problems.append("research results.json missing")
        else:
            saved_results = _json(result_path)
            if not _same(saved_results.get("metrics_by_view"), recomputed_metrics):
                problems.append("saved metrics do not match independent metric recomputation")
            if not _same(saved_results.get("protected_groups"), recomputed_groups):
                problems.append("saved protected groups do not match independent recomputation")
            if not _same(saved_results.get("per_cycle"), recomputed_cycles):
                problems.append("saved per-cycle results do not match independent recomputation")
            if not _same(saved_results.get("frozen_gate"), recomputed_gate):
                problems.append("saved frozen gate does not match independent gate recomputation")
            if saved_results.get("scientific_verdict") != recomputed_gate.get("status"):
                problems.append("scientific verdict label differs from independently recomputed gate")
    result = {
        "status": "PASS" if not problems else "FAIL",
        "campaign": CARD_ID, "freeze_signature": signature,
        "checks": {
            "identity_and_all_input_hashes": not any("hash mismatch" in p or "signature mismatch" in p for p in problems),
            "official_train_rows_hash_and_cycles": _sha(train_path) == "875552a68a3d24b0f3e1966d6ddb5671e6f54305a1c526e2f0a76747942ed91c" and len(frame) == 50_608 and np.array_equal(np.unique(cycles), np.arange(47, 80)),
            "historical_control_cache_and_sparse25_route_rebuilt": len(baselines) == 2,
            "both_official_origins_and_fit_populations": len(outputs) == 2,
            "candidate_and_matched_routes_rebuilt": len(outputs) == 2,
            "all_metrics_groups_per_cycle_and_gate_recomputed": len(recomputed_metrics) == 6,
            "test_sample_submission_and_kaggle_not_used": identity.get("test_read") is False and identity.get("sample_submission_read") is False and identity.get("kaggle_access") is False,
        },
        "fit_audits": fit_checks,
        "independent_metrics_by_view": recomputed_metrics,
        "independent_protected_groups": recomputed_groups,
        "independent_per_cycle": recomputed_cycles,
        "independent_frozen_gate": recomputed_gate,
        "budget": ledger_summary(LEDGER),
        "test_read": False, "sample_submission_read": False, "kaggle_access": False,
        "problems": problems,
    }
    path = REPORT_ROOT / "independent_verification.json"
    encoded = json.dumps(result, sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.read_text(encoding="utf-8") != encoded:
        raise RuntimeError(f"Immutable verifier output differs; refusing overwrite: {path}")
    if not path.exists():
        path.write_text(encoded, encoding="utf-8", newline="\n")
    return result


if __name__ == "__main__":
    verification = verify()
    print(json.dumps(verification, ensure_ascii=False, allow_nan=False, indent=2))
