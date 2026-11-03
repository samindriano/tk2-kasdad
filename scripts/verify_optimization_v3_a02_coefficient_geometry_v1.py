"""Independent label-free candidate verifier and post-label metric verifier."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Mapping

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import normalize

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from src.models import classification_a02_neighbors as a02
import optimization_v3_positional_text_pilot_v1 as positional
import optimization_v3_a02_coefficient_geometry_v1 as runner


ATOL = 1e-12


class VerificationError(RuntimeError):
    pass


def _json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise VerificationError(f"Required verification input is missing: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _npz(path: Path) -> dict[str, np.ndarray]:
    if not path.is_file():
        raise VerificationError(f"Candidate output NPZ is missing: {path}")
    with np.load(path, allow_pickle=False) as archive:
        if "artifact_signal" in archive.files or "y_true" in archive.files or "target" in archive.files:
            raise VerificationError("Candidate NPZ unexpectedly stores ground-truth labels")
        result = {}
        for key in archive.files:
            values = archive[key]
            if values.dtype.kind == "O":
                raise VerificationError(f"Object dtype is prohibited in stored output {key}")
            result[key] = values.copy()
    return result


def _hash_array(values: Any) -> str:
    return positional.sha256_array(values)


def _independent_top32(sim: np.ndarray, source_rows: np.ndarray) -> np.ndarray:
    values = np.asarray(sim, dtype=np.float64)
    rows = np.asarray(source_rows, dtype=np.int64)
    pos = np.flatnonzero(values > 0.0)
    k = min(a02.NEIGHBORS_MAX, len(pos))
    if k == 0:
        return np.empty(0, dtype=np.int64)
    if len(pos) > k:
        threshold = np.partition(values[pos], len(pos) - k)[len(pos) - k]
        above = pos[values[pos] > threshold]
        ties = pos[values[pos] == threshold]
        order = np.argsort(rows[ties], kind="stable")
        chosen = np.concatenate((above, ties[order[:k - len(above)]]))
    else:
        chosen = pos
    return chosen[np.lexsort((rows[chosen], -values[chosen]))].astype(np.int64)


def _reconstruct_origin(frame: pd.DataFrame, origin: int, arm: str) -> dict[str, Any]:
    source = runner.load_origin_sources(frame, origin)
    model = source["model"]
    estimator = source["estimator"]
    c2b_vectorizer = source["vectorizer"]
    a02_vectorizer = model.text_space.vectorizer
    if (not np.array_equal(estimator.classes_, np.asarray([0, 1]))
            or estimator.coef_.ndim != 2 or estimator.coef_.shape[0] != 1):
        raise VerificationError(f"C2_B E{origin} classifier class order or coefficient shape mismatch")
    if (c2b_vectorizer.vocabulary_ != a02_vectorizer.vocabulary_
            or c2b_vectorizer.get_params(deep=False) != a02_vectorizer.get_params(deep=False)
            or not np.array_equal(c2b_vectorizer.idf_, a02_vectorizer.idf_)):
        raise VerificationError(f"C2_B/A02 E{origin} text mapping is no longer exact")
    beta = np.empty(len(a02_vectorizer.vocabulary_), dtype=np.float64)
    for token, a02_col in a02_vectorizer.vocabulary_.items():
        c2b_col = c2b_vectorizer.vocabulary_.get(token)
        if c2b_col is None:
            raise VerificationError(f"No C2_B coefficient column exists for A02 token {token!r}")
        beta[int(a02_col)] = estimator.coef_[0, source["structured_width"] + int(c2b_col)]
    if not np.isfinite(beta).all():
        raise VerificationError("C2_B token coefficient mapping contains non-finite values")
    d = np.sqrt(1.0 + np.abs(beta) / (1.0 + np.abs(beta)))
    if arm == "permuted":
        d = np.random.default_rng(42).permutation(d)
    elif arm != "aligned":
        raise VerificationError(f"Unknown CGEOM arm: {arm}")

    bank = sparse_weight_normalize(model.bank_matrix, d)
    q_raw, q_missing = model.text_space.transform(frame.iloc[source["score_rows"]]["signal_stream"])
    query = sparse_weight_normalize(q_raw, d)
    cache = source["cache"]
    n = len(source["score_rows"])
    old_rows = cache["neighbor_source_rows"].astype(np.int64)
    old_sims = cache["neighbor_similarities"].astype(np.float64)
    m0 = np.sum(old_sims * old_sims, axis=1, dtype=np.float64)
    if not np.array_equal(q_missing, cache["missing_mask"]):
        raise VerificationError("Independent query normalization changed the missing-stream route")

    out_rows = np.full((n, a02.NEIGHBORS_MAX), -1, dtype=np.int64)
    max_id_len = max((len(str(value)) for value in model.bank_ids), default=1)
    out_ids = np.full((n, a02.NEIGHBORS_MAX), "", dtype=f"<U{max_id_len}")
    out_sims = np.zeros((n, a02.NEIGHBORS_MAX), dtype=np.float64)
    out_weights = np.zeros((n, a02.NEIGHBORS_MAX), dtype=np.float64)
    counts = np.zeros(n, dtype=np.int16)
    new_mass = np.zeros(n, dtype=np.float64)
    raw_correction = np.zeros(n, dtype=np.float64)
    bank_row_to_idx = {int(row): idx for idx, row in enumerate(model.bank_row_positions)}

    for start in range(0, n, a02.QUERY_BLOCK_ROWS):
        stop = min(start + a02.QUERY_BLOCK_ROWS, n)
        similarities = (query[start:stop] @ bank.T).toarray()
        for local, row_index in enumerate(range(start, stop)):
            if q_missing[row_index] or m0[row_index] == 0.0:
                continue
            selected = _independent_top32(similarities[local], model.bank_row_positions)
            if not len(selected):
                raise VerificationError("Positive original M0 has no support in independent full-bank search")
            sim = similarities[local, selected]
            sq = sim * sim
            mass = float(np.sum(sq, dtype=np.float64))
            weight = sq * (m0[row_index] / mass)
            if abs(float(np.sum(weight)) - m0[row_index]) > ATOL * max(1.0, abs(float(m0[row_index]))):
                raise VerificationError("Independent reconstructed weights fail the exact original-mass constraint")
            indices = selected
            raw_correction[row_index] = float(np.dot(weight, model.bank_residuals[indices])) / (a02.SHRINKAGE + m0[row_index])
            k = len(selected)
            out_rows[row_index, :k] = model.bank_row_positions[selected]
            out_ids[row_index, :k] = [model.bank_ids[idx] for idx in selected]
            out_sims[row_index, :k] = sim
            out_weights[row_index, :k] = weight
            counts[row_index] = k
            new_mass[row_index] = mass

    correction = np.clip(raw_correction, -a02.CORRECTION_LIMIT, a02.CORRECTION_LIMIT)
    fallback = q_missing | (m0 == 0.0)
    correction[fallback] = 0.0
    prediction = np.clip(cache["baseline_prediction"] + correction, 0.0, 1.0)
    prediction[fallback] = cache["baseline_prediction"][fallback]

    # Independently bind the source rows and IDs that entered the weight sum.
    for row_idx in range(n):
        k = int(counts[row_idx])
        if k == 0:
            if np.any(out_rows[row_idx] != -1) or np.any(out_sims[row_idx] != 0.0):
                raise VerificationError("Independent empty-neighbor padding is inconsistent")
            continue
        rows = out_rows[row_idx, :k]
        if len(set(rows.tolist())) != k or any(int(row) not in bank_row_to_idx for row in rows):
            raise VerificationError("Independent donor rows are duplicate or outside the cached bank")
        indices = np.asarray([bank_row_to_idx[int(row)] for row in rows], dtype=np.int64)
        if np.any(model.bank_cycles[indices] > source["score_cycles"][row_idx]):
            raise VerificationError("Independent weighted donor includes a future cycle")
        if not np.array_equal(out_ids[row_idx, :k], np.asarray([model.bank_ids[idx] for idx in indices], dtype=str)):
            raise VerificationError("Independent donor IDs do not map to the selected official rows")

    path = runner._unit_paths(f"CGEOM_V1_E{origin}_{arm.upper()}")[0]
    stored = _npz(path)
    expected = {
        "candidate_probability": prediction,
        "baseline_probability": cache["baseline_prediction"],
        "original_a02_probability": cache["prediction"],
        "m0": m0,
        "neighbor_count": counts,
        "neighbor_source_rows": out_rows,
        "neighbor_ids": out_ids,
        "neighbor_similarities": out_sims,
        "neighbor_weights": out_weights,
        "new_cosine_squared_mass": new_mass,
        "correction_raw": raw_correction,
        "correction": correction,
        "missing_mask": q_missing,
        "query_rows": source["score_rows"],
        "query_cycles": source["score_cycles"],
        "query_record_ids": source["score_ids"],
    }
    if set(stored) != set(expected):
        raise VerificationError(f"Stored production output schema differs from required fields at {path}")
    for key, values in expected.items():
        if stored[key].shape != values.shape or stored[key].dtype != values.dtype:
            raise VerificationError(f"Stored production output shape/dtype mismatch at E{origin}/{arm}/{key}")
        if values.dtype.kind in "f":
            if not np.allclose(stored[key], values, atol=ATOL, rtol=0, equal_nan=False):
                raise VerificationError(f"Independent numerical reconstruction differs at E{origin}/{arm}/{key}")
        elif not np.array_equal(stored[key], values):
            raise VerificationError(f"Independent array reconstruction differs at E{origin}/{arm}/{key}")
    receipt_path = runner._unit_paths(f"CGEOM_V1_E{origin}_{arm.upper()}")[1]
    receipt = _json(receipt_path)
    if (receipt.get("artifact_sha256") != runner.sha_file(path)
            or receipt.get("freeze_signature") != runner.load_freeze()["freeze_signature"]
            or receipt.get("fit_id") != f"CGEOM_V1_E{origin}_{arm.upper()}"
            or receipt.get("query_labels_opened") is not False
            or receipt.get("array_sha256") != {key: _hash_array(value) for key, value in stored.items()}):
        raise VerificationError(f"Unit receipt is not bound to independent stored arrays at E{origin}/{arm}")
    return {
        "origin": origin, "arm": arm, "fit_id": receipt["fit_id"],
        "artifact_sha256": runner.sha_file(path), "receipt_sha256": runner.sha_file(receipt_path),
        "query_rows": int(n), "neighbor_search": "full original bank, blockwise, positive cosine, stable source-row ties",
        "candidate_probability_max_abs_error": float(np.max(np.abs(stored["candidate_probability"] - prediction))),
        "m0_max_abs_error": float(np.max(np.abs(stored["m0"] - m0))),
        "mass_constraint_max_abs_error": float(np.max(np.abs(stored["neighbor_weights"].sum(axis=1) - m0))),
        "all_members_read_allow_pickle_false": True,
        "query_labels_opened": False,
    }


def sparse_weight_normalize(matrix: Any, multipliers: np.ndarray):
    weighted = matrix.multiply(multipliers).tocsr()
    weighted.sum_duplicates()
    weighted.sort_indices()
    return normalize(weighted, norm="l2", axis=1, copy=False).tocsr()


def verify_prelabel() -> dict[str, Any]:
    ledger_sha_before = runner.sha_file(runner.LEDGER)
    ledger_snapshot_before = runner._budget_snapshot(runner._ledger_events())
    frozen = runner.load_freeze()
    preflight = _json(runner.PREFLIGHT_PATH)
    synthetic = _json(runner.SYNTHETIC_QA_PATH)
    if (preflight.get("status") != "PASS_TARGET_FREE_PREFIT_RESOURCE_AND_LINEAGE_PREFLIGHT"
            or preflight.get("source_commit") != frozen["source_commit"]
            or preflight.get("source_hashes") != frozen["source_hashes"]
            or synthetic.get("source_commit") != frozen["source_commit"]
            or synthetic.get("source_hashes") != frozen["source_hashes"]
            or preflight.get("synthetic_qa_sha256") != runner.sha_file(runner.SYNTHETIC_QA_PATH)):
        raise VerificationError("Prelabel source, preflight, or synthetic-QA binding failed")
    frame = positional.load_features()
    if "artifact_signal" in frame or "threat_rank" in frame:
        raise VerificationError("Independent prelabel verifier must not load target columns")
    identities = []
    for origin in runner.ORIGINS:
        for arm in runner.ARMS:
            identities.append(_reconstruct_origin(frame, origin, arm))
    events = runner._ledger_events()
    snapshot = runner._assert_budget(events, phase="after_all_fits")
    own_events = runner._own_events(events)
    expected_fit_ids = set(runner.FIT_IDS)
    if (snapshot["fit_pending"] != 0 or snapshot["fit_starts"] != 118
            or snapshot["fit_saved"] != 114 or snapshot["fit_failed"] != 4
            or {event["fit_id"] for event in own_events["FIT_STARTED"]} != expected_fit_ids
            or {event["fit_id"] for event in own_events["FIT_SAVED"]} != expected_fit_ids
            or own_events["FIT_FAILED"]
            or len(identities) != 4 or len({item["fit_id"] for item in identities}) != 4):
        raise VerificationError(f"Independent prelabel fit accounting failed: {snapshot}")
    ledger_sha_after = runner.sha_file(runner.LEDGER)
    if ledger_sha_after != ledger_sha_before or snapshot != ledger_snapshot_before:
        raise VerificationError("Shared budget ledger changed during prelabel verification")
    artifact_hashes = {item["fit_id"]: item["artifact_sha256"] for item in identities}
    receipt_hashes = {item["fit_id"]: item["receipt_sha256"] for item in identities}
    result = {
        "status": "PASS_INDEPENDENT_NUMERICAL_AND_BOUNDARY_VERIFICATION",
        "pilot_id": runner.PILOT_ID, "source_commit": frozen["source_commit"],
        "source_hashes": frozen["source_hashes"], "freeze_signature": frozen["freeze_signature"],
        "input_sha256": runner.TRAIN_SHA256,
        "preflight_sha256": runner.sha_file(runner.PREFLIGHT_PATH),
        "synthetic_qa_sha256": runner.sha_file(runner.SYNTHETIC_QA_PATH),
        "ledger_sha256_before": ledger_sha_before, "ledger_sha256": ledger_sha_after,
        "ledger_snapshot": snapshot,
        "independent_unit_reconstructions": identities,
        "unit_artifact_sha256": artifact_hashes, "unit_receipt_sha256": receipt_hashes,
        "labels_opened": False, "test_or_sample_submission_opened": False,
        "verified_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    runner.write_create_once(runner.PRELABEL_PATH, runner.json_bytes(result))
    return result


def _metrics_independent(y: np.ndarray, p: np.ndarray) -> dict[str, Any]:
    actual = np.asarray(y, dtype=np.int8)
    probability = np.asarray(p, dtype=np.float64)
    pred = probability >= 0.50
    tn = int(np.sum((actual == 0) & ~pred))
    fp = int(np.sum((actual == 0) & pred))
    fn = int(np.sum((actual == 1) & ~pred))
    tp = int(np.sum((actual == 1) & pred))

    def f1_for(label: int) -> float:
        truth = actual == label
        guessed = pred if label == 1 else ~pred
        true_positive = int(np.sum(truth & guessed))
        false_positive = int(np.sum((~truth) & guessed))
        false_negative = int(np.sum(truth & (~guessed)))
        precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 0.0
        recall = true_positive / (true_positive + false_negative) if true_positive + false_negative else 0.0
        return 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0

    clipped = np.minimum(np.maximum(probability, 1e-15), 1.0 - 1e-15)
    return {
        "n": int(len(actual)), "positive": int(np.sum(actual == 1)), "negative": int(np.sum(actual == 0)),
        "macro_f1": float((f1_for(0) + f1_for(1)) / 2.0),
        "positive_recall": float(tp / (tp + fn)) if tp + fn else 0.0,
        "precision": float(tp / (tp + fp)) if tp + fp else 0.0,
        "roc_auc": float(roc_auc_score(actual, probability)) if len(np.unique(actual)) == 2 else None,
        "average_precision": float(average_precision_score(actual, probability)) if len(np.unique(actual)) == 2 else None,
        "log_loss_clip_1e-15": float(-np.mean(actual * np.log(clipped) + (1 - actual) * np.log(1.0 - clipped))),
        "brier": float(np.mean((probability - actual) ** 2)),
        "confusion_matrix": [[tn, fp], [fn, tp]], "threshold": 0.50,
        "log_loss_probability_clip_epsilon": 1e-15, "zero_denominator_precision_recall_f1": 0.0,
    }


def _metric_tree_close(actual: Any, expected: Any, path: str = "root") -> None:
    if isinstance(expected, dict):
        if not isinstance(actual, dict) or set(actual) != set(expected):
            raise VerificationError(f"Result metric fields differ at {path}")
        for key in expected:
            _metric_tree_close(actual[key], expected[key], f"{path}.{key}")
    elif isinstance(expected, list):
        if not isinstance(actual, list) or len(actual) != len(expected):
            raise VerificationError(f"Result metric list differs at {path}")
        for i, (left, right) in enumerate(zip(actual, expected)):
            _metric_tree_close(left, right, f"{path}[{i}]")
    elif isinstance(expected, (int, float)) and not isinstance(expected, bool):
        if not isinstance(actual, (int, float)) or isinstance(actual, bool) or abs(float(actual) - float(expected)) > ATOL:
            raise VerificationError(f"Numeric result differs at {path}: {actual!r} != {expected!r}")
    elif actual != expected:
        raise VerificationError(f"Result differs at {path}: {actual!r} != {expected!r}")


def verify_scored_metrics() -> dict[str, Any]:
    frozen = runner.load_freeze()
    prelabel = _json(runner.PRELABEL_PATH)
    result = _json(runner.RESULT_PATH)
    if (prelabel.get("status") != "PASS_INDEPENDENT_NUMERICAL_AND_BOUNDARY_VERIFICATION"
            or prelabel.get("labels_opened") is not False
            or result.get("prelabel_verification_sha256") != runner.sha_file(runner.PRELABEL_PATH)
            or result.get("freeze_signature") != frozen["freeze_signature"]
            or result.get("source_commit") != frozen["source_commit"]
            or result.get("input_sha256") != runner.TRAIN_SHA256):
        raise VerificationError("Scored verification is not authorized by a verified prelabel checkpoint")
    frame = positional.load_features()
    cycles_all = frame["cycle_code"].to_numpy(dtype=np.int64)
    rows67 = np.flatnonzero((cycles_all >= 68) & (cycles_all <= 79)).astype(np.int64)
    y67 = positional.target_values_for_rows(rows67, cycles_all[rows67], purpose="outer-score-after-predictions")
    rows75 = np.flatnonzero((cycles_all >= 76) & (cycles_all <= 79)).astype(np.int64)
    lookup = {int(row): i for i, row in enumerate(rows67)}
    if any(int(row) not in lookup for row in rows75):
        raise VerificationError("E75 rows are not a subset of E67 score rows")
    idx75 = np.asarray([lookup[int(row)] for row in rows75], dtype=np.int64)
    y75 = y67[idx75]
    independently_computed: dict[str, Any] = {"origins": {}, "pooled_unique_metrics": {}}
    cache_values: dict[int, dict[str, Any]] = {}
    for origin, rows, y in ((67, rows67, y67), (75, rows75, y75)):
        source = runner.load_origin_sources(frame, origin)
        arm_arrays = {}
        for arm in runner.ARMS:
            fit_id = f"CGEOM_V1_E{origin}_{arm.upper()}"
            path, receipt_path = runner._unit_paths(fit_id)
            if prelabel.get("unit_artifact_sha256", {}).get(fit_id) != runner.sha_file(path):
                raise VerificationError(f"Output artifact changed after prelabel verification: {fit_id}")
            arm_arrays[arm] = _npz(path)
            if _json(receipt_path).get("artifact_sha256") != runner.sha_file(path):
                raise VerificationError(f"Output receipt no longer binds the score file: {fit_id}")
        probabilities = {
            "aligned": arm_arrays["aligned"]["candidate_probability"],
            "permuted": arm_arrays["permuted"]["candidate_probability"],
            "a02": source["cache"]["prediction"],
            "sparse25_baseline": source["cache"]["baseline_prediction"],
        }
        mask_map = {"E67_H12": np.ones(len(rows), dtype=bool),
                    "E67_W4": (source["score_cycles"] >= 68) & (source["score_cycles"] <= 71),
                    "E67_W5": (source["score_cycles"] >= 72) & (source["score_cycles"] <= 75),
                    "E67_W6": (source["score_cycles"] >= 76) & (source["score_cycles"] <= 79)} if origin == 67 else {
                        "E75_W6": (source["score_cycles"] >= 76) & (source["score_cycles"] <= 79)}
        metrics = {}
        for name, mask in mask_map.items():
            metrics[name] = {arm: _metrics_independent(y[mask], value[mask]) for arm, value in probabilities.items()}
        cycles = {}
        for cycle in sorted(set(source["score_cycles"].tolist())):
            mask = source["score_cycles"] == cycle
            cycles[str(int(cycle))] = {arm: _metrics_independent(y[mask], value[mask]) for arm, value in probabilities.items()}
        independently_computed["origins"][str(origin)] = {
            "fit_prefix_rows": source["fit_prefix_rows"], "query_rows": source["query_rows"],
            "present_rows": source["present_rows"], "missing_rows": source["missing_rows"],
            "metrics": metrics, "per_cycle": cycles,
        }
        cache_values[origin] = {"source": source, "arms": arm_arrays, "probabilities": probabilities,
                                "cycles": source["score_cycles"]}
    pooled67 = (cache_values[67]["cycles"] >= 68) & (cache_values[67]["cycles"] <= 71)
    pooled75 = (cache_values[75]["cycles"] >= 76) & (cache_values[75]["cycles"] <= 79)
    yp = np.concatenate((y67[pooled67], y75[pooled75]))
    for arm in ("aligned", "permuted", "a02", "sparse25_baseline"):
        p67 = cache_values[67]["probabilities"][arm][pooled67]
        p75 = cache_values[75]["probabilities"][arm][pooled75]
        independently_computed["pooled_unique_metrics"][arm] = _metrics_independent(yp, np.concatenate((p67, p75)))
    _metric_tree_close(result["origins"], independently_computed["origins"], "origins")
    _metric_tree_close(result["pooled_unique_metrics"], independently_computed["pooled_unique_metrics"], "pooled_unique_metrics")
    _verify_gate_tables(result, independently_computed, cache_values, {67: y67, 75: y75}, yp, pooled67, pooled75)
    verified = {
        "status": "PASS_INDEPENDENT_METRICS_AND_GATES",
        "pilot_id": runner.PILOT_ID, "source_commit": frozen["source_commit"],
        "freeze_signature": frozen["freeze_signature"], "input_sha256": runner.TRAIN_SHA256,
        "result_sha256": runner.sha_file(runner.RESULT_PATH),
        "prelabel_verification_sha256": runner.sha_file(runner.PRELABEL_PATH),
        "query_label_access_receipt_sha256": runner.sha_file(runner.RUN_ROOT / "QUERY_LABEL_ACCESS.json"),
        "ledger_sha256": runner.sha_file(runner.LEDGER),
        "ledger_snapshot": runner._budget_snapshot(runner._ledger_events()),
        "metric_comparison_tolerance": ATOL,
        "labels_opened_only_after_prelabel_pass": True,
        "independent_metric_and_gate_reconstruction": "PASS",
        "gate_counts": result["gate_counts"], "status_verification": result["status"],
        "verified_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    runner.write_create_once(runner.SCORED_VERIFICATION_PATH, runner.json_bytes(verified))
    return verified


def _verify_gate_tables(result: Mapping[str, Any], computed: Mapping[str, Any], caches: Mapping[int, Any],
                        labels: Mapping[int, np.ndarray], pooled_y: np.ndarray,
                        pooled67: np.ndarray, pooled75: np.ndarray) -> None:
    expected_gates = []
    e67 = computed["origins"]["67"]["metrics"]
    e75 = computed["origins"]["75"]["metrics"]["E75_W6"]
    def gate(name: str, gain: float, threshold: float, context: dict[str, Any]) -> dict[str, Any]:
        return {"gate": name, "gain": float(gain), "required_minimum_gain": float(threshold),
                "tolerance": 1e-12, "status": "PASS" if gain + 1e-12 >= threshold else "FAIL", **context}
    expected_gates.extend([
        gate("E67_H12_aligned_vs_A02", e67["E67_H12"]["aligned"]["macro_f1"]-e67["E67_H12"]["a02"]["macro_f1"], .0005, {"view":"E67_H12","comparator":"a02"}),
        gate("E75_W6_aligned_vs_A02", e75["aligned"]["macro_f1"]-e75["a02"]["macro_f1"], .0005, {"view":"E75_W6","comparator":"a02"}),
        gate("E67_H12_aligned_vs_permuted", e67["E67_H12"]["aligned"]["macro_f1"]-e67["E67_H12"]["permuted"]["macro_f1"], .0002, {"view":"E67_H12","comparator":"permuted"}),
        gate("E75_W6_aligned_vs_permuted", e75["aligned"]["macro_f1"]-e75["permuted"]["macro_f1"], 0.0, {"view":"E75_W6","comparator":"permuted"}),
    ])
    for view in ("E67_W4", "E67_W5", "E67_W6"):
        for comparator in ("a02", "permuted"):
            expected_gates.append(gate(f"{view}_harm_vs_{comparator}", e67[view]["aligned"]["macro_f1"]-e67[view][comparator]["macro_f1"], -.005, {"view":view,"comparator":comparator}))
    subgroup_checks = []
    for view, origin, mask in (("E67_H12",67,np.ones(len(caches[67]["cycles"]),dtype=bool)),
                               ("E67_W4",67,(caches[67]["cycles"]>=68)&(caches[67]["cycles"]<=71)),
                               ("E67_W5",67,(caches[67]["cycles"]>=72)&(caches[67]["cycles"]<=75)),
                               ("E67_W6",67,(caches[67]["cycles"]>=76)&(caches[67]["cycles"]<=79)),
                               ("E75_W6",75,(caches[75]["cycles"]>=76)&(caches[75]["cycles"]<=79))):
        source = caches[origin]["source"]
        groups = {
            "availability": np.where(source["cache"]["missing_mask"],"missing","present"),
            "text_load_quartile": np.asarray([f"Q{int(x)+1}" for x in source["cache"]["text_load_group"]]),
            "entry_friction": np.asarray([str(int(x)) for x in source["cache"]["entry_friction"]]),
        }
        for group, values in groups.items():
            for value in sorted(set(values[mask].tolist())):
                gm=mask&(values==value); yg=labels[origin][gm]; n=len(yg)
                pos=int(np.sum(yg==1)); neg=int(np.sum(yg==0))
                base={"view":view,"origin":origin,"group":group,"group_value":str(value),"n":n,"positive":pos,"negative":neg,
                      "minimum_support":{"n":200,"each_class":30}}
                if n<200 or pos<30 or neg<30:
                    subgroup_checks.append({**base,"status":"UNKNOWN_UNSUPPORTED","aligned_metrics":None,"a02_metrics":None,"permuted_metrics":None,"gains":None})
                    continue
                aligned=_metrics_independent(yg,caches[origin]["arms"]["aligned"]["candidate_probability"][gm])
                a02m=_metrics_independent(yg,source["cache"]["prediction"][gm])
                perm=_metrics_independent(yg,caches[origin]["arms"]["permuted"]["candidate_probability"][gm])
                gains={"a02":{"macro_f1":aligned["macro_f1"]-a02m["macro_f1"],"positive_recall":aligned["positive_recall"]-a02m["positive_recall"]},
                       "permuted":{"macro_f1":aligned["macro_f1"]-perm["macro_f1"],"positive_recall":aligned["positive_recall"]-perm["positive_recall"]}}
                passed=all(gains[c]["macro_f1"]+1e-12>=-.01 and gains[c]["positive_recall"]+1e-12>=-.02 for c in ("a02","permuted"))
                subgroup_checks.append({**base,"status":"PASS" if passed else "FAIL","aligned_metrics":aligned,"a02_metrics":a02m,"permuted_metrics":perm,"gains":gains,
                                        "thresholds":{"macro_f1_min_gain":-.01,"positive_recall_min_gain":-.02}})
    pooled_groups_by_origin = {}
    for origin, mask in ((67, pooled67), (75, pooled75)):
        source = caches[origin]["source"]
        pooled_groups_by_origin[origin] = {
            "availability": np.where(source["cache"]["missing_mask"],"missing","present"),
            "text_load_quartile": np.asarray([f"E{origin}_Q{int(x)+1}" for x in source["cache"]["text_load_group"]]),
            "entry_friction": np.asarray([str(int(x)) for x in source["cache"]["entry_friction"]]),
        }
    pooled_labels = np.concatenate((labels[67][pooled67], labels[75][pooled75]))
    pooled_probabilities = {}
    for arm in ("aligned", "a02", "permuted"):
        chunks = []
        for origin, mask in ((67, pooled67), (75, pooled75)):
            if arm in {"aligned", "permuted"}:
                p = caches[origin]["arms"][arm]["candidate_probability"]
            else:
                p = caches[origin]["source"]["cache"]["prediction"]
            chunks.append(p[mask])
        pooled_probabilities[arm] = np.concatenate(chunks)
    for group in ("availability", "text_load_quartile", "entry_friction"):
        values = np.concatenate([pooled_groups_by_origin[o][group][m] for o,m in ((67,pooled67),(75,pooled75))])
        for value in sorted(set(values.tolist())):
            gm=values==value; yg=pooled_labels[gm]; n=len(yg)
            pos=int(np.sum(yg==1)); neg=int(np.sum(yg==0))
            base={"view":"POOLED_E67_W4_PLUS_E75_W6","origin":"67+75","group":group,"group_value":str(value),
                  "n":n,"positive":pos,"negative":neg,"minimum_support":{"n":200,"each_class":30}}
            if n<200 or pos<30 or neg<30:
                subgroup_checks.append({**base,"status":"UNKNOWN_UNSUPPORTED","aligned_metrics":None,"a02_metrics":None,"permuted_metrics":None,"gains":None})
                continue
            aligned=_metrics_independent(yg,pooled_probabilities["aligned"][gm])
            a02m=_metrics_independent(yg,pooled_probabilities["a02"][gm])
            perm=_metrics_independent(yg,pooled_probabilities["permuted"][gm])
            gains={"a02":{"macro_f1":aligned["macro_f1"]-a02m["macro_f1"],"positive_recall":aligned["positive_recall"]-a02m["positive_recall"]},
                   "permuted":{"macro_f1":aligned["macro_f1"]-perm["macro_f1"],"positive_recall":aligned["positive_recall"]-perm["positive_recall"]}}
            passed=all(gains[c]["macro_f1"]+1e-12>=-.01 and gains[c]["positive_recall"]+1e-12>=-.02 for c in ("a02","permuted"))
            subgroup_checks.append({**base,"status":"PASS" if passed else "FAIL","aligned_metrics":aligned,"a02_metrics":a02m,"permuted_metrics":perm,"gains":gains,
                                    "thresholds":{"macro_f1_min_gain":-.01,"positive_recall_min_gain":-.02}})
    pooled_gate_checks = expected_gates + [
        {"gate":f"subgroup:{x['view']}:{x['group']}:{x['group_value']}","status":x["status"],"view":x["view"],"group":x["group"],"group_value":x["group_value"],
         "support":{"n":x["n"],"positive":x["positive"],"negative":x["negative"]},"gains":x["gains"],"thresholds":x.get("thresholds")}
        for x in subgroup_checks]
    _metric_tree_close(result["subgroup_checks"], subgroup_checks, "subgroup_checks")
    _metric_tree_close(result["gate_checks"], pooled_gate_checks, "gate_checks")
    false=sum(item["status"]=="FAIL" for item in pooled_gate_checks)
    unknown=sum(item["status"]=="UNKNOWN_UNSUPPORTED" for item in pooled_gate_checks)
    counts={"total":len(pooled_gate_checks),"fail":false,"unknown":unknown,"pass":sum(item["status"]=="PASS" for item in pooled_gate_checks)}
    if result["gate_counts"] != counts:
        raise VerificationError(f"Gate counts differ from independent reconstruction: {result['gate_counts']} != {counts}")
    status="CLOSED_GATE_FAIL" if false else ("NOT_ADVANCE_UNSUPPORTED" if unknown else "ADVANCE_TO_SOL_REVIEW")
    if result["status"] != status:
        raise VerificationError(f"CGEOM overall status differs from independently reconstructed gates: {result['status']} != {status}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", required=True, choices=("prelabel", "scored"))
    args = parser.parse_args(argv)
    if args.stage == "prelabel":
        value = verify_prelabel()
    else:
        value = verify_scored_metrics()
    print(json.dumps(value, sort_keys=True, ensure_ascii=False, indent=2, allow_nan=False, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
