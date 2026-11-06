"""Independent fit, prediction, score, and gate verifier for CLS_POS_RECENCY_V1."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.special import expit
from sklearn.metrics import (average_precision_score, f1_score, log_loss,
                            precision_score, recall_score, roc_auc_score)


def _canonical_json(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
                      allow_nan=False).encode("utf-8")


def _hash_array(value: Any) -> str:
    array = np.asarray(value)
    header = hashlib.sha256(_canonical_json({"shape": list(array.shape), "dtype": str(array.dtype)})).hexdigest()
    digest = hashlib.sha256(header.encode("ascii"))
    if array.dtype.kind in "OUS":
        digest.update(json.dumps(array.astype(str).tolist(), ensure_ascii=False,
                                 separators=(",", ":")).encode("utf-8"))
    else:
        digest.update(np.ascontiguousarray(array).tobytes())
    return digest.hexdigest()


def _hash_csr(value: sparse.spmatrix) -> str:
    x = sparse.csr_matrix(value, dtype=np.float64, copy=True)
    x.sum_duplicates(); x.eliminate_zeros(); x.sort_indices()
    digest = hashlib.sha256()
    digest.update(_canonical_json({"shape": list(x.shape), "dtype": str(x.dtype), "format": "csr"}))
    for part in (x.indptr, x.indices, x.data):
        digest.update(np.ascontiguousarray(part).tobytes())
    return digest.hexdigest()


def _base_weights(cycles: Sequence[int], origin: int) -> np.ndarray:
    cycle = np.asarray(cycles, dtype=np.int64).reshape(-1)
    if len(cycle) == 0 or np.any(cycle > origin):
        raise RuntimeError("Independent recency cycles escape the prefix")
    return np.maximum(0.25, np.exp2(-(origin - cycle).astype(np.float64) / 8.0))


def _class_weights(base: Sequence[float], target: Sequence[int]) -> np.ndarray:
    value = np.asarray(base, dtype=np.float64).reshape(-1)
    y = np.asarray(target, dtype=np.int8).reshape(-1)
    if len(value) != len(y) or len(value) == 0 or not np.isfinite(value).all() or np.any(value <= 0):
        raise RuntimeError("Independent class-normalized weights have invalid inputs")
    out = np.zeros_like(value)
    for cls in (0, 1):
        mask = y == cls
        if not mask.any():
            raise RuntimeError(f"Fit prefix lacks target class {cls}")
        out[mask] = value[mask] / value[mask].mean()
    return out


def _shuffle_class_weights(weights: Sequence[float], target: Sequence[int]) -> tuple[np.ndarray, np.ndarray]:
    value = np.asarray(weights, dtype=np.float64).reshape(-1)
    y = np.asarray(target, dtype=np.int8).reshape(-1)
    rng = np.random.default_rng(42)
    result = np.empty_like(value)
    permutation = np.arange(len(y), dtype=np.int64)
    for cls in (0, 1):
        idx = np.flatnonzero(y == cls)
        order = rng.permutation(len(idx))
        result[idx] = value[idx[order]]
        permutation[idx] = idx[order]
    return result, permutation


def _objective_gradient(
    gamma: Sequence[float], x_position: sparse.spmatrix, z_bag: Sequence[float],
    target: Sequence[int], weights: Sequence[float],
) -> tuple[float, np.ndarray]:
    x = sparse.csr_matrix(x_position, dtype=np.float64)
    coef = np.asarray(gamma, dtype=np.float64).reshape(-1)
    offset = np.asarray(z_bag, dtype=np.float64).reshape(-1)
    y = np.asarray(target, dtype=np.float64).reshape(-1)
    w = np.asarray(weights, dtype=np.float64).reshape(-1)
    if x.shape != (len(y), len(coef)) or len(offset) != len(y) or len(w) != len(y):
        raise RuntimeError("Independent objective inputs do not align")
    logits = offset + np.asarray(x @ coef, dtype=np.float64).reshape(-1)
    loss = float(np.sum(w * np.logaddexp(0.0, -(2.0 * y - 1.0) * logits)) + 0.5 * np.dot(coef, coef))
    gradient = np.asarray(x.T @ (w * (expit(logits) - y)), dtype=np.float64).reshape(-1) + coef
    if not np.isfinite(loss) or not np.isfinite(gradient).all():
        raise RuntimeError("Independent objective reconstruction is non-finite")
    return loss, gradient


def _read_fit_targets_independent(train_path: Path, rows: np.ndarray, ids: np.ndarray) -> np.ndarray:
    wanted = np.asarray(rows, dtype=np.int64).reshape(-1)
    identities = np.asarray(ids).astype(str).reshape(-1)
    selected = set(int(row) for row in wanted.tolist())
    frame = pd.read_csv(train_path, usecols=["record_id", "artifact_signal"],
        dtype={"record_id": "string", "artifact_signal": "int8"}, keep_default_na=False,
        skiprows=lambda line: line > 0 and (line - 1) not in selected)
    order = np.argsort(wanted, kind="stable")
    if len(frame) != len(wanted) or not np.array_equal(frame["record_id"].astype(str).to_numpy(), identities[order]):
        raise RuntimeError("Independent fit target read does not match the frozen fit IDs")
    y_sorted = frame["artifact_signal"].to_numpy(dtype=np.int8)
    y = np.empty_like(y_sorted); y[order] = y_sorted
    if not np.isin(y, (0, 1)).all():
        raise RuntimeError("Independent fit targets are not binary")
    return y


def _independent_metric(y: Sequence[int], probability: Sequence[float], label: Sequence[int]) -> dict[str, Any]:
    target = np.asarray(y, dtype=np.int8).reshape(-1)
    p = np.asarray(probability, dtype=np.float64).reshape(-1)
    pred = np.asarray(label, dtype=np.int8).reshape(-1)
    if len(target) == 0 or not (len(target) == len(p) == len(pred)):
        raise ValueError("Independent metric vectors are empty or unaligned")
    if not np.isin(target, (0, 1)).all() or not np.isin(pred, (0, 1)).all():
        raise ValueError("Independent metric targets and decisions must be binary")
    if not np.isfinite(p).all() or np.any((p < 0) | (p > 1)):
        raise ValueError("Independent metric probabilities invalid")
    tn = int(np.count_nonzero((target == 0) & (pred == 0)))
    fp = int(np.count_nonzero((target == 0) & (pred == 1)))
    fn = int(np.count_nonzero((target == 1) & (pred == 0)))
    tp = int(np.count_nonzero((target == 1) & (pred == 1)))
    both = np.unique(target).size == 2
    clipped = np.clip(p, np.finfo(np.float64).eps, 1.0 - np.finfo(np.float64).eps)
    return {
        "n": int(len(target)), "positive": int(target.sum()), "negative": int(len(target) - target.sum()),
        "macro_f1": float(f1_score(target, pred, average="macro", labels=[0, 1], zero_division=0)),
        "positive_precision": float(precision_score(target, pred, pos_label=1, zero_division=0)),
        "positive_recall": float(recall_score(target, pred, pos_label=1, zero_division=0)),
        "roc_auc": float(roc_auc_score(target, p)) if both else None,
        "average_precision": float(average_precision_score(target, p)) if both else None,
        "log_loss": float(log_loss(target, clipped, labels=[0, 1])),
        "brier": float(np.mean((p - target) ** 2)),
        "confusion_matrix_labels_0_1": [[tn, fp], [fn, tp]],
    }


def independent_metric_comparison(
    y: Sequence[int], p_a: Sequence[float], p_b: Sequence[float],
    yhat_a: Sequence[int], yhat_b: Sequence[int],
) -> dict[str, Any]:
    a = _independent_metric(y, p_a, yhat_a)
    b = _independent_metric(y, p_b, yhat_b)
    return {
        "support": {"n": b["n"], "positive": b["positive"], "negative": b["negative"]},
        "reference": a, "candidate": b,
        "gain_macro_f1": b["macro_f1"] - a["macro_f1"],
        "gain_positive_recall": b["positive_recall"] - a["positive_recall"],
        "gain_positive_precision": b["positive_precision"] - a["positive_precision"],
        "gain_roc_auc": None if a["roc_auc"] is None or b["roc_auc"] is None else b["roc_auc"] - a["roc_auc"],
        "gain_average_precision": None if a["average_precision"] is None or b["average_precision"] is None else b["average_precision"] - a["average_precision"],
        "gain_log_loss": a["log_loss"] - b["log_loss"],
        "gain_brier": a["brier"] - b["brier"],
    }


def compare_metrics(expected: Mapping[str, Any], observed: Mapping[str, Any], tolerance: float = 5e-12) -> dict[str, Any]:
    differences: list[dict[str, Any]] = []
    def walk(a: Any, b: Any, path: str = "") -> None:
        if isinstance(a, Mapping) and isinstance(b, Mapping):
            if set(a) != set(b):
                differences.append({"path": path, "expected_keys": sorted(a), "observed_keys": sorted(b)})
                return
            for key in sorted(a):
                walk(a[key], b[key], f"{path}/{key}")
        elif a is None or b is None:
            if a is not b:
                differences.append({"path": path, "expected": a, "observed": b})
        elif isinstance(a, (int, np.integer)) and isinstance(b, (int, np.integer)):
            if int(a) != int(b):
                differences.append({"path": path, "expected": int(a), "observed": int(b)})
        elif isinstance(a, (int, float, np.integer, np.floating)) and isinstance(b, (int, float, np.integer, np.floating)):
            delta = abs(float(a) - float(b))
            if not np.isfinite(delta) or delta > tolerance:
                differences.append({"path": path, "expected": float(a), "observed": float(b), "abs_diff": delta})
        elif a != b:
            differences.append({"path": path, "expected": str(a), "observed": str(b)})
    walk(expected, observed)
    return {"status": "PASS" if not differences else "FAIL",
            "difference_count": len(differences),
            "max_abs_difference": max((d.get("abs_diff", 0.0) for d in differences), default=0.0),
            "differences": differences[:20]}


def build_metric_report(views: Mapping[str, Any], checks: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not checks or any(item.get("status") != "PASS" for item in checks):
        raise RuntimeError("Independent metric comparison set is empty or failed")
    body = {
        "status": "PASS_INDEPENDENT_METRIC_RECALC",
        "comparison_count": int(len(checks)),
        "checks": [dict(item) for item in checks],
        "views_sha256": hashlib.sha256(_canonical_json(views)).hexdigest(),
        "query_labels_reopened": False,
        "method": "Separate confusion/F1/precision/recall/AUC/AP/log-loss/Brier arithmetic on the same one-pass target vector",
    }
    body["verification_signature"] = hashlib.sha256(_canonical_json(body)).hexdigest()
    return body


def _supported(mask: np.ndarray, y: np.ndarray) -> dict[str, Any]:
    chosen = np.asarray(mask, dtype=bool).reshape(-1)
    target = np.asarray(y, dtype=np.int8).reshape(-1)
    n = int(chosen.sum()); pos = int(target[chosen].sum()); neg = n - pos
    return {"n": n, "positive": pos, "negative": neg,
            "supported": n >= 200 and pos >= 30 and neg >= 30,
            "minimum_n": 200, "minimum_per_class": 30}


def verify_fit_artifact(official_root: Path, fit_id: str, *, write_report: bool = True) -> dict[str, Any]:
    from scripts import cls_pos_recency_v1 as runner

    freeze = runner.verify_freeze(official_root, require_committed_freeze=True)
    origin, arm = runner.core.fit_id_parts(fit_id)
    context = runner._parent_context(official_root, origin)
    rows_all = np.asarray(context["fit_rows"], dtype=np.int64)
    ids_all = np.asarray(context["fit_ids"]).astype(str)
    identity = context["old_runner"]._read_identity_frame()
    cycles_all = identity.iloc[rows_all]["cycle_code"].to_numpy(dtype=np.int64)
    present = runner._read_selected_stream_presence(
        Path(official_root) / "data/raw/classification/Train.csv", rows_all, ids_all)
    rows, ids, cycles = rows_all[present], ids_all[present], cycles_all[present]
    runner.append_event({"event": "FIT_VERIFY_PREFIX_LABELS_OPENED", "fit_id": fit_id,
        "origin": origin, "rows": int(len(rows)), "rows_sha256": _hash_array(rows),
        "ids_sha256": _hash_array(ids), "query_labels_read": False})
    y = _read_fit_targets_independent(Path(official_root) / "data/raw/classification/Train.csv", rows, ids)
    runner.append_event({"event": "FIT_VERIFY_PREFIX_LABELS_READ", "fit_id": fit_id,
        "origin": origin, "rows": int(len(rows)), "rows_sha256": _hash_array(rows),
        "target_sha256": _hash_array(y), "query_labels_read": False})
    x = sparse.csr_matrix(context["x_fit"][present], dtype=np.float64)
    z = np.asarray(context["z0_fit"], dtype=np.float64)[present]
    base = _base_weights(cycles, origin)
    recency = _class_weights(base, y)
    shuffled, permutation = _shuffle_class_weights(recency, y)
    weights = recency if arm == "RECENCY" else shuffled
    recorded_permutation = permutation if arm == "SHUFFLED_WITHIN_CLASS" else np.arange(len(y), dtype=np.int64)
    artifact_path, receipt_path, verification_path = runner._fit_paths(fit_id)
    if not artifact_path.is_file() or not receipt_path.is_file():
        raise RuntimeError(f"Fit artifact/receipt missing: {fit_id}")
    receipt = runner.read_json(receipt_path)
    with np.load(artifact_path, allow_pickle=False) as archive:
        if archive.files != ["gamma"]:
            raise RuntimeError(f"Unexpected fit artifact schema: {fit_id}")
        gamma = np.asarray(archive["gamma"], dtype=np.float64).copy()
    loss, gradient = _objective_gradient(gamma, x, z, y, weights)
    loss_zero, gradient_zero = _objective_gradient(np.zeros_like(gamma), x, z, y, weights)
    if (receipt.get("candidate_id") != runner.core.CANDIDATE_ID or receipt.get("fit_id") != fit_id
        or receipt.get("freeze_signature") != freeze["freeze_signature"]
        or receipt.get("arm") != arm or receipt.get("origin") != origin
        or receipt.get("fit_rows") != len(rows)
        or receipt.get("fit_rows_sha256") != _hash_array(rows)
        or receipt.get("fit_ids_sha256") != _hash_array(ids)
        or receipt.get("fit_cycles_sha256") != _hash_array(cycles)
        or receipt.get("fit_target_sha256") != _hash_array(y)
        or receipt.get("fit_position_sha256") != _hash_csr(x)
        or receipt.get("z_bag_sha256") != _hash_array(z)
        or receipt.get("base_weight_sha256") != _hash_array(base)
        or receipt.get("weight_sha256") != _hash_array(weights)
        or receipt.get("weight_permutation_sha256") != _hash_array(recorded_permutation)
        or receipt.get("gamma_sha256") != _hash_array(gamma)
        or receipt.get("artifact_sha256") != runner.sha256_file(artifact_path)
        or receipt.get("optimizer", {}).get("success") is not True
        or abs(loss_zero - float(receipt.get("optimizer", {}).get("loss_gamma0", float("inf")))) > 1e-8
        or not np.isfinite(gamma).all() or not np.isfinite(loss)
        or loss > receipt.get("optimizer", {}).get("loss_gamma0", float("inf")) + 1e-8):
        raise RuntimeError(f"Independent fit receipt/objective verification failed: {fit_id}")
    tol = 1e-8
    require_gradient = bool(np.isfinite(gradient).all()
        and abs(loss - float(receipt["optimizer"]["loss_final"])) <= tol
        and abs(float(np.max(np.abs(gradient))) - float(receipt["optimizer"]["gradient_inf_norm"])) <= 1e-8
        and np.isfinite(gradient_zero).all())
    if not require_gradient:
        raise RuntimeError(f"Independent final objective/gradient differs: {fit_id}")
    body = {
        "schema_version": 1,
        "candidate_id": runner.core.CANDIDATE_ID,
        "status": "PASS_INDEPENDENT_FIT_RECONSTRUCTION",
        "fit_id": fit_id,
        "origin": origin,
        "arm": arm,
        "freeze_signature": freeze["freeze_signature"],
        "artifact_sha256": runner.sha256_file(artifact_path),
        "receipt_sha256": runner.sha256_file(receipt_path),
        "fit_rows": int(len(rows)),
        "fit_present_rows": int(len(rows)),
        "fit_target_sha256": _hash_array(y),
        "recency_weight_sha256": _hash_array(recency),
        "shuffled_weight_sha256": _hash_array(shuffled),
        "permutation_sha256": _hash_array(permutation),
        "loss_reconstructed": loss,
        "gradient_inf_norm_reconstructed": float(np.max(np.abs(gradient))),
        "prefix_labels_read_for_verification": True,
        "query_labels_read": False,
        "optimizer_refits": 0,
        "preprocessing_fits": 0,
    }
    body["verification_signature"] = runner.canonical_sha256(body)
    if write_report:
        runner.write_once(verification_path, body)
    return body


def _independent_probabilities(context: Mapping[str, Any], gamma: np.ndarray) -> np.ndarray:
    logits = np.asarray(context["z0_score"] + context["x_score"] @ gamma, dtype=np.float64).reshape(-1)
    return np.asarray(expit(logits), dtype=np.float64)


def verify_prediction_bundle(official_root: Path, origin: int) -> dict[str, Any]:
    from scripts import cls_pos_recency_v1 as runner

    freeze = runner.verify_freeze(official_root, require_committed_freeze=True)
    context = runner._parent_context(official_root, origin)
    runner._validate_origin_context(official_root, origin, context)
    present = np.asarray(context["present"], dtype=bool)
    p_a02 = np.asarray(context["p_a02"], dtype=np.float64)
    original = np.asarray(context["p_pos_original"], dtype=np.float64)
    champion = p_a02.copy()
    champion[present] = 0.5 * original[present] + 0.5 * p_a02[present]
    vectors: dict[str, np.ndarray] = {}
    for arm in runner.core.ARMS:
        fit_id = f"CLS_POS_RECENCY_V1_E{origin}_{arm}"
        gamma, receipt, _ = runner.verify_fit_binding(freeze, fit_id)
        p_pos = _independent_probabilities(context, gamma)
        candidate = p_a02.copy()
        candidate[present] = 0.5 * p_pos[present] + 0.5 * p_a02[present]
        decision = np.where(present, candidate >= 0.5, candidate >= 0.53230976129477).astype(np.int8)
        vectors[arm] = candidate
        vectors[f"label_{arm}"] = decision
        vectors[f"p_pos_{arm}"] = p_pos
        vectors[f"gamma_sha256_{arm}"] = np.asarray([receipt["gamma_sha256"]])
    label_champion = np.where(present, champion >= 0.5, champion >= 0.53230976129477).astype(np.int8)
    expected = {
        "origin": np.asarray([origin], dtype=np.int64),
        "row_positions": np.asarray(context["score_rows"], dtype=np.int64),
        "record_ids": np.asarray(context["score_ids"]).astype(str),
        "cycles": np.asarray(context["cycles"], dtype=np.int64),
        "present": present,
        "ef1": np.asarray(context["entry_friction"], dtype=np.int8),
        "text_load_group": np.asarray(context["text_load_group"]).astype(str),
        "p_a02": p_a02,
        "p_pos_original": original,
        "p_champion": champion,
        "label_champion": label_champion,
    }
    for arm in runner.core.ARMS:
        expected[f"p_{arm.lower()}"] = vectors[arm]
        expected[f"label_{arm.lower()}"] = vectors[f"label_{arm}"]
        expected[f"p_pos_{arm.lower()}"] = vectors[f"p_pos_{arm}"]
        expected[f"gamma_sha256_{arm.lower()}"] = vectors[f"gamma_sha256_{arm}"]
    bundle_path = runner.RUN_ROOT / f"E{origin}" / f"predictions_E{origin}.npz"
    if not bundle_path.is_file():
        raise RuntimeError(f"Prediction bundle missing: E{origin}")
    with np.load(bundle_path, allow_pickle=False) as archive:
        if archive.files != list(expected):
            raise RuntimeError(f"Prediction bundle schema mismatch: E{origin}")
        saved = {key: archive[key].copy() for key in archive.files}
    mismatches = [key for key, value in expected.items() if not np.array_equal(saved[key], value)]
    if mismatches:
        raise RuntimeError(f"Independent prediction replay failed E{origin}: {mismatches}")
    missing = ~present
    for arm in runner.core.ARMS:
        if not (np.array_equal(saved[f"p_{arm.lower()}"][missing].view(np.uint64), p_a02[missing].view(np.uint64))
                and np.array_equal(saved[f"label_{arm.lower()}"][missing], label_champion[missing])):
            raise RuntimeError(f"Missing-stream fallback differs for E{origin}/{arm}")
    return {
        "status": "PASS_TARGET_FREE_PREDICTION_REPLAY",
        "origin": origin,
        "rows": int(len(present)),
        "present_rows": int(present.sum()),
        "missing_rows": int((~present).sum()),
        "prediction_bundle_sha256": runner.sha256_file(bundle_path),
        "row_ids_sha256": _hash_array(expected["record_ids"]),
        "row_positions_sha256": _hash_array(expected["row_positions"]),
        "missing_route_exact": True,
        "query_labels_read": False,
        "test_or_sample_access": False,
    }


def verify_prelabel_outputs(official_root: Path, *, write_report: bool = True) -> dict[str, Any]:
    from scripts import cls_pos_recency_v1 as runner

    freeze = runner.verify_freeze(official_root, require_committed_freeze=True)
    ready = runner.read_json(runner.READY_PATH)
    signature = ready.get("ready_signature")
    ready_body = dict(ready); ready_body.pop("ready_signature", None)
    if (signature != runner.canonical_sha256(ready_body)
        or ready.get("candidate_id") != runner.core.CANDIDATE_ID
        or ready.get("freeze_signature") != freeze["freeze_signature"]
        or ready.get("query_labels_read") is not False
        or ready.get("test_or_sample_access") is not False):
        raise RuntimeError("Target-free prediction readiness signature/boundary failed")
    fit_checks = {}
    for fit_id in runner.core.FIT_IDS:
        _, receipt, verification = runner.verify_fit_binding(freeze, fit_id)
        fit_checks[fit_id] = {"receipt_sha256": runner.sha256_file(runner._fit_paths(fit_id)[1]),
                              "verification_sha256": runner.sha256_file(runner._fit_paths(fit_id)[2]),
                              "gamma_sha256": receipt["gamma_sha256"],
                              "fit_verification_status": verification["status"]}
    predictions = {}
    for origin in runner.core.ORIGINS:
        result = verify_prediction_bundle(official_root, origin)
        item = ready["predictions"][f"E{origin}"]
        bundle_path = runner.AUDIT_ROOT / item["path"]
        if (runner.sha256_file(bundle_path) != item["sha256"]
            or result["prediction_bundle_sha256"] != item["sha256"]
            or result["row_ids_sha256"] != item["ordered_ids_sha256"]):
            raise RuntimeError(f"Prediction readiness hash mismatch: E{origin}")
        predictions[f"E{origin}"] = result
    e67 = np.load(runner.RUN_ROOT / "E67/predictions_E67.npz", allow_pickle=False)
    e75 = np.load(runner.RUN_ROOT / "E75/predictions_E75.npz", allow_pickle=False)
    try:
        rows67 = np.asarray(e67["row_positions"], dtype=np.int64)
        cyc67 = np.asarray(e67["cycles"], dtype=np.int64)
        ids67 = np.asarray(e67["record_ids"]).astype(str)
        mask = (cyc67 >= 76) & (cyc67 <= 79)
        if (int(mask.sum()) != 8491
            or not np.array_equal(np.asarray(e75["row_positions"], dtype=np.int64), rows67[mask])
            or not np.array_equal(np.asarray(e75["record_ids"]).astype(str), ids67[mask])):
            raise RuntimeError("E75 W6 rows are not the exact E67 cycles 76-79 subset")
    finally:
        e67.close(); e75.close()
    body = {
        "schema_version": 1,
        "candidate_id": runner.core.CANDIDATE_ID,
        "status": "PASS_TARGET_FREE_PRELABEL",
        "freeze_signature": freeze["freeze_signature"],
        "source_commit": freeze["audit_source_commit"],
        "prediction_readiness_sha256": runner.sha256_file(runner.READY_PATH),
        "prediction_hashes": {key: value["prediction_bundle_sha256"] for key, value in predictions.items()},
        "fit_checks": fit_checks,
        "predictions": predictions,
        "E75_exact_subset_of_E67_W6": True,
        "optimizer_refits": 0,
        "preprocessing_fits": 0,
        "prefix_labels_read_during_prelabel": 0,
        "query_labels_read": False,
        "test_or_sample_access": False,
    }
    body["verification_signature"] = runner.canonical_sha256(body)
    if write_report:
        runner.write_once(runner.PRELABEL_PATH, body)
    return body


def _compare_recursive(expected: Any, observed: Any, path: str = "", tol: float = 5e-12) -> list[dict[str, Any]]:
    diffs: list[dict[str, Any]] = []
    if isinstance(expected, Mapping) and isinstance(observed, Mapping):
        if set(expected) != set(observed):
            return [{"path": path, "expected_keys": sorted(expected), "observed_keys": sorted(observed)}]
        for key in sorted(expected):
            diffs.extend(_compare_recursive(expected[key], observed[key], f"{path}/{key}", tol))
    elif expected is None or observed is None:
        if expected is not observed:
            diffs.append({"path": path, "expected": expected, "observed": observed})
    elif isinstance(expected, (float, int, np.floating, np.integer)) and isinstance(observed, (float, int, np.floating, np.integer)):
        if isinstance(expected, (int, np.integer)) and isinstance(observed, (int, np.integer)):
            if int(expected) != int(observed):
                diffs.append({"path": path, "expected": int(expected), "observed": int(observed)})
        else:
            delta = abs(float(expected) - float(observed))
            if not np.isfinite(delta) or delta > tol:
                diffs.append({"path": path, "expected": float(expected), "observed": float(observed), "abs_diff": delta})
    elif expected != observed:
        diffs.append({"path": path, "expected": str(expected), "observed": str(observed)})
    return diffs


def _rebuild_gate(score: Mapping[str, Any], prelabel: Mapping[str, Any]) -> dict[str, Any]:
    views = score["views"]
    primary = views["UNIQUE_PRIMARY"]["UNIQUE_PRIMARY"]
    e67 = views["E67"]["H12_E67"]
    e75 = views["E75"]["W6"]
    unique_gain = primary["recency_vs_champion"]["gain_macro_f1"]
    h12_gain = e67["recency_vs_champion"]["gain_macro_f1"]
    w6_gain = e75["recency_vs_champion"]["gain_macro_f1"]
    unique_control_gain = primary["recency_vs_shuffled"]["gain_macro_f1"]
    w6_control_gain = e75["recency_vs_shuffled"]["gain_macro_f1"]
    all_groups = []
    for view in views.values():
        for item in view.values():
            if isinstance(item, Mapping) and "subgroups" in item:
                all_groups.extend(item["subgroups"])
    supported = [g for g in all_groups if g["support"]["supported"]]
    legacy_ok = all(g["recency_vs_champion"]["gain_macro_f1"] >= -0.01
                    and g["recency_vs_champion"]["gain_positive_recall"] >= -0.02 for g in supported)
    exploratory_group_ok = all(g["recency_vs_champion"]["gain_macro_f1"] >= -0.02
                               and g["recency_vs_champion"]["gain_positive_recall"] >= -0.03 for g in supported)
    missing_exact = all(x["missing_route_exact"] for x in prelabel["predictions"].values())
    metric_qa = score.get("independent_metric_qa", {}).get("status") == "PASS"
    resource_ok = score.get("resource_policy", {}).get("post_score_reserve_pass") is True
    scientific = (unique_gain >= 0.0005 and h12_gain >= 0.0 and w6_gain >= 0.0
                  and unique_control_gain >= 0.0002 and w6_control_gain >= 0.0
                  and legacy_ok and missing_exact and metric_qa and resource_ok)
    exploratory = ((unique_gain > 0.0 and w6_gain >= -0.0005)
                   or (w6_gain >= 0.0005 and unique_gain >= -0.0005))
    exploratory = exploratory and exploratory_group_ok and missing_exact and metric_qa and resource_ok
    exploratory_signal = ((unique_gain > 0.0 and w6_gain >= -0.0005)
                           or (w6_gain >= 0.0005 and unique_gain >= -0.0005))
    return {
        "scientific_qualified": bool(scientific),
        "exploratory_eligible": bool(exploratory),
        "exploratory_signal_ok": bool(exploratory_signal),
        "legacy_supported_subgroup_guard_ok": bool(legacy_ok),
        "exploratory_supported_subgroup_guard_ok": bool(exploratory_group_ok),
        "missing_route_exact": bool(missing_exact),
        "independent_metric_qa": bool(metric_qa),
        "post_score_resource_reserve_ok": bool(resource_ok),
        "thresholds": {
            "unique_macro_f1_gain_min": 0.0005,
            "E67_H12_macro_f1_gain_min": 0.0,
            "E75_W6_macro_f1_gain_min": 0.0,
            "unique_recency_minus_shuffled_min": 0.0002,
            "E75_W6_recency_minus_shuffled_min": 0.0,
            "legacy_supported_group_harm_max": {"macro_f1": 0.01, "positive_recall": 0.02},
            "exploratory_supported_group_harm_max": {"macro_f1": 0.02, "positive_recall": 0.03},
        },
    }


def verify_score_outputs(official_root: Path, *, write_report: bool = True) -> dict[str, Any]:
    from scripts import cls_pos_recency_v1 as runner

    freeze = runner.verify_freeze(official_root, require_committed_freeze=True)
    runner.require_committed("reports/simplification_v1/CLS_POS_RECENCY_V1/PREDICTIONS_READY.json")
    runner.require_committed("reports/simplification_v1/CLS_POS_RECENCY_V1/PRELABEL_VERIFICATION.json")
    score = runner.read_json(runner.SCORE_PATH)
    metric = runner.read_json(runner.METRIC_PATH)
    prelabel = runner.read_json(runner.PRELABEL_PATH)
    score_sig = score.get("score_signature")
    score_body = dict(score); score_body.pop("score_signature", None)
    score_metric = dict(score.get("metric_verification", {}))
    score_metric_sig = score_metric.pop("verification_signature", None)
    metric_sig = metric.get("verification_signature")
    metric_body = dict(metric); metric_body.pop("verification_signature", None)
    prelabel_sig = prelabel.get("verification_signature")
    prelabel_body = dict(prelabel); prelabel_body.pop("verification_signature", None)
    if (score_sig != runner.canonical_sha256(score_body)
        or score_metric_sig != runner.canonical_sha256(score_metric)
        or metric_sig != runner.canonical_sha256(metric_body)):
        raise RuntimeError("Score or independent metric report signature invalid")
    if prelabel_sig != runner.canonical_sha256(prelabel_body):
        raise RuntimeError("Committed target-free prelabel signature invalid")
    if (score.get("candidate_id") != runner.core.CANDIDATE_ID
        or score.get("freeze_signature") != freeze["freeze_signature"]
        or score.get("prelabel_sha256") != runner.sha256_file(runner.PRELABEL_PATH)
        or score.get("query_labels_opened") is not True
        or score.get("test_or_sample_access") is not False
        or metric.get("score_sha256") != runner.sha256_file(runner.SCORE_PATH)
        or metric.get("status") != "PASS_INDEPENDENT_METRIC_RECALC"):
        raise RuntimeError("Score/metric report lineage or boundary mismatch")
    checks = metric.get("checks", [])
    if not checks or any(c.get("status") != "PASS" for c in checks):
        raise RuntimeError("Independent metric comparison set is incomplete or failed")
    if (metric.get("comparison_count") != len(checks)
        or metric.get("checks") != score.get("metric_verification", {}).get("checks")
        or metric.get("views_sha256") != score.get("metric_verification", {}).get("views_sha256")):
        raise RuntimeError("Saved independent metric report differs from the score-bound checks")
    actual_views_sha256 = hashlib.sha256(_canonical_json(score.get("views"))).hexdigest()
    if metric.get("views_sha256") != actual_views_sha256:
        raise RuntimeError("Independent metric report views hash does not bind the saved score views")
    if len({c["metric_id"] for c in checks}) != len(checks):
        raise RuntimeError("Duplicate metric IDs in independent metric report")
    events = runner.events()
    opened = [e for e in events if e.get("event") == "QUERY_LABELS_OPENED"]
    read = [e for e in events if e.get("event") == "QUERY_LABELS_READ"]
    score_start = [e for e in events if e.get("event") == "SCORE_STARTED"]
    resource = [e for e in events if e.get("event") == "SCORE_RESOURCE_SNAPSHOT"]
    if (len(opened) != 1 or len(read) != 1
        or len(score_start) != 1 or len(resource) != 1
        or opened[0].get("rows") != 20903
        or opened[0].get("rows_sha256") != read[0].get("rows_sha256")
        or opened[0].get("ids_sha256") != read[0].get("ids_sha256")
        or read[0].get("target_sha256") != score.get("query_target_sha256")
        or opened[0].get("prelabel_sha256") != runner.sha256_file(runner.PRELABEL_PATH)
        or score_start[0].get("resource_before", {}).get("available_bytes") < runner.MIN_SCORE_AVAILABLE
        or resource[0].get("available_after_metrics_bytes") != score.get("resource_policy", {}).get("available_after_metrics_bytes")
        or resource[0].get("post_score_reserve_pass") != score.get("resource_policy", {}).get("post_score_reserve_pass")):
        raise RuntimeError("One-pass query-label ledger does not match score rows and prelabel")
    gate = _rebuild_gate(score, prelabel)
    differences = _compare_recursive(gate, score.get("gates", {}))
    if differences:
        raise RuntimeError(f"Independent gate reconstruction differs: {differences[:10]}")
    report = {
        "schema_version": 1,
        "candidate_id": runner.core.CANDIDATE_ID,
        "status": "PASS_INDEPENDENT_METRIC_AND_GATE_VERIFICATION",
        "freeze_signature": freeze["freeze_signature"],
        "score_sha256": runner.sha256_file(runner.SCORE_PATH),
        "metric_report_sha256": runner.sha256_file(runner.METRIC_PATH),
        "independent_metric_comparison_count": int(len(checks)),
        "query_label_reads_for_scoring": 1,
        "query_label_reads_during_verification": 0,
        "total_query_label_open_events": 1,
        "test_or_sample_access": False,
        "kaggle_access": False,
        "gate_reconstruction": gate,
    }
    report["verification_signature"] = runner.canonical_sha256(report)
    if write_report:
        runner.write_once(runner.SCORE_VERIFY_PATH, report)
    return report
