"""Independent verifier for A02 continuation persistence, neighbors, metrics, and gates."""
from __future__ import annotations

import hashlib
import os
import json
from pathlib import Path
import tempfile
from typing import Any
from unittest import mock

import joblib
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import average_precision_score, log_loss, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
import sys
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.models.classification_a02_neighbors import (  # noqa: E402
    ResidualNeighborPredictor, TFIDF_PARAMETERS,
)
from src.models.research_next_budget import finish_fit, read_events  # noqa: E402
from src.models.research_next_contract import canonical_sha256, file_sha256  # noqa: E402
from scripts import optimization_v3_a02_neighbors as pilot  # noqa: E402
from scripts import optimization_v3_a02_continuation as runner  # noqa: E402


TRAIN_PATH = ROOT / "data/raw/classification/Train.csv"
LEDGER_PATH = ROOT / "experiments/runs/optimization_v3_research_next_v1/budget.jsonl"
REPORT_ROOT = ROOT / "reports/optimization_v3/a02_continuation_v1"
RUN_ROOT = ROOT / "experiments/runs/optimization_v3_a02_continuation_v1"
CONTINUATION_CARD = runner.CARD_ID
CONTINUATION_STAGE = runner.STAGE
EXPECTED_CACHE_WINDOWS = {
    "55": [56, 67], "59": [60, 63], "63": [64, 67],
    "67": [68, 79], "71": [72, 75], "75": [76, 79],
}


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def _ids_sha(ids: list[str]) -> str:
    return hashlib.sha256(json.dumps(ids, separators=(",", ":"), ensure_ascii=False).encode("utf-8")).hexdigest()


def _metric(y: np.ndarray, probability: np.ndarray) -> dict[str, Any]:
    truth = np.asarray(y, dtype=np.int8)
    p = np.asarray(probability, dtype=np.float64)
    pred = (p >= 0.5).astype(np.int8)
    tn = int(np.sum((truth == 0) & (pred == 0)))
    fp = int(np.sum((truth == 0) & (pred == 1)))
    fn = int(np.sum((truth == 1) & (pred == 0)))
    tp = int(np.sum((truth == 1) & (pred == 1)))
    f0 = 2 * tn / (2 * tn + fp + fn) if (2 * tn + fp + fn) else 0.0
    f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0
    return {
        "rows": len(truth), "macro_f1": float((f0 + f1) / 2),
        "positive_recall": float(tp / (tp + fn)) if tp + fn else None,
        "precision": float(tp / (tp + fp)) if tp + fp else None,
        "positive_predictions": int(pred.sum()),
        "confusion": {"tn": tn, "fp": fp, "fn": fn, "tp": tp},
    }


def _probability_metrics(y: np.ndarray, probability: np.ndarray,
                         baseline: np.ndarray) -> dict[str, Any]:
    truth = np.asarray(y, dtype=np.int8)
    p = np.asarray(probability, dtype=np.float64)
    base = np.asarray(baseline, dtype=np.float64)
    clipped = np.clip(p, 1e-15, 1 - 1e-15)
    clipped_base = np.clip(base, 1e-15, 1 - 1e-15)
    loss = -(truth * np.log(clipped) + (1 - truth) * np.log1p(-clipped))
    base_loss = -(truth * np.log(clipped_base) + (1 - truth) * np.log1p(-clipped_base))
    wrong_endpoint = ((p == 0) & (truth == 1)) | ((p == 1) & (truth == 0))
    endpoint = np.isin(p, [0.0, 1.0])
    return {
        "roc_auc": float(roc_auc_score(truth, p)),
        "average_precision": float(average_precision_score(truth, p)),
        "brier": float(np.mean((p - truth) ** 2)),
        "log_loss_manual_eps_1e_15": float(loss.mean()),
        "log_loss_sklearn_default_dtype_epsilon": float(log_loss(truth, p, labels=[0, 1])),
        "sklearn_version": __import__("sklearn").__version__,
        "sklearn_float64_epsilon": float(np.finfo(np.float64).eps),
        "endpoint_rows_exact_0_or_1": int(endpoint.sum()),
        "wrong_endpoint_rows": int(wrong_endpoint.sum()),
        "wrong_endpoint_breakdown": {
            "p_0_y_1": int(((p == 0) & (truth == 1)).sum()),
            "p_1_y_0": int(((p == 1) & (truth == 0)).sum()),
        },
        "wrong_endpoint_delta_contribution_to_full_mean": float((loss[wrong_endpoint] - base_loss[wrong_endpoint]).sum() / len(truth)),
        "all_other_delta_contribution_to_full_mean": float((loss[~wrong_endpoint] - base_loss[~wrong_endpoint]).sum() / len(truth)),
        "candidate_minus_baseline_log_loss_manual_eps_1e_15": float((loss - base_loss).mean()),
    }


def _independent_cache(cutoff: int, train: pd.DataFrame,
                       protocol: dict[str, Any]) -> dict[str, np.ndarray]:
    path = pilot.CACHE_PATHS[cutoff]
    side_path = path.with_suffix(".json")
    side = _json(side_path)
    if file_sha256(path) != side.get("npz_sha256"):
        raise RuntimeError(f"C16 cache {cutoff} does not match sidecar byte hash")
    side_js = lambda value: hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                            allow_nan=False).encode("utf-8")).hexdigest()
    if (side_js(side.get("identity")) != side.get("identity_sha256")
            or side_js({k: v for k, v in side.items() if k != "integrity_sha256"}) != side.get("integrity_sha256")):
        raise RuntimeError(f"C16 cache {cutoff} sidecar signature failed")
    if side.get("identity", {}).get("card_id") != "C16" or int(side["identity"].get("cutoff", -1)) != cutoff:
        raise RuntimeError(f"C16 cache {cutoff} identity mismatch")
    with np.load(path, allow_pickle=False) as data:
        rows = np.asarray(data["score_rows"]).astype(np.int64, copy=True)
        ids = np.asarray(data["score_record_ids"]).astype(str, copy=True)
        base = np.asarray(data["baseline_prediction"]).astype(np.float64, copy=True)
        raw_sparse = np.asarray(data["control__baseline_sparse_raw"]).astype(np.float64, copy=True)
        c2b = np.asarray(data["control__c2b_probability"]).astype(np.float64, copy=True)
        cb = np.asarray(data["control__cb_probability"]).astype(np.float64, copy=True)
        missing = np.asarray(data["control__stream_missing"]).astype(bool, copy=True)
        control_rows = np.asarray(data["control__control_rows"]).astype(np.int64, copy=True)
    if (len(np.unique(rows)) != len(rows) or np.any(np.diff(rows) <= 0)
            or ids.tolist() != train.record_id.astype(str).to_numpy()[rows].tolist()
            or not np.array_equal(control_rows, rows)):
        raise RuntimeError(f"C16 cache {cutoff} source row/ID join failed")
    mask = train.signal_stream.fillna("").astype(str).str.strip().eq("").to_numpy()[rows]
    if not np.array_equal(missing, mask):
        raise RuntimeError(f"C16 cache {cutoff} missing-stream mask mismatch")
    reconstructed = np.where(mask, cb, 0.75 * c2b + 0.25 * raw_sparse)
    if not np.array_equal(reconstructed, base):
        raise RuntimeError(f"C16 cache {cutoff} baseline does not reconstruct exact C16 route")
    cycles = train.cycle_code.to_numpy(dtype=np.int64)[rows]
    if protocol.get("cache_score_cycle_windows") != EXPECTED_CACHE_WINDOWS:
        raise RuntimeError("A02 frozen cache score windows differ")
    expected_window = EXPECTED_CACHE_WINDOWS[str(cutoff)]
    if (not np.array_equal(np.unique(cycles), np.arange(expected_window[0], expected_window[1] + 1))
            or protocol["cache_score_cycle_windows"][str(cutoff)] != expected_window):
        raise RuntimeError(f"C16 cache {cutoff} complete cycle window mismatch")
    for value in np.unique(cycles):
        if int(np.sum(cycles == value)) != int(np.sum(train.cycle_code.to_numpy() == value)):
            raise RuntimeError(f"C16 cache {cutoff} cycle {value} is incomplete")
    return {"rows": rows, "ids": ids, "cycles": cycles, "baseline": base, "missing": mask}


def _select_cache(cache: dict[str, np.ndarray], train: pd.DataFrame,
                  low: int, high: int) -> dict[str, np.ndarray]:
    keep = (cache["cycles"] >= low) & (cache["cycles"] <= high)
    rows = cache["rows"][keep]
    expected = np.flatnonzero(train.cycle_code.between(low, high).to_numpy())
    if not np.array_equal(rows, expected):
        raise RuntimeError(f"C16 OOF block {low}-{high} is not complete and source-row ordered")
    return {key: cache[key][keep] for key in ("rows", "ids", "cycles", "baseline", "missing")}


def _independent_bank(origin: int, caches: dict[int, dict[str, np.ndarray]],
                      train: pd.DataFrame, protocol: dict[str, Any]) -> dict[str, np.ndarray]:
    blocks = protocol["oof_bank_blocks"].get(str(origin))
    if blocks is None:
        blocks = [[55, 56, 59], [59, 60, 63], [63, 64, 67], [67, 68, 71], [71, 72, 75], [75, 76, 79]]
    pieces = [_select_cache(caches[int(cutoff)], train, int(low), int(high))
              for cutoff, low, high in blocks]
    joined = {key: np.concatenate([piece[key] for piece in pieces])
              for key in ("rows", "ids", "cycles", "baseline", "missing")}
    order = np.argsort(joined["rows"], kind="stable")
    joined = {key: value[order] for key, value in joined.items()}
    expected = np.flatnonzero(train.cycle_code.between(56, origin).to_numpy())
    if (not np.array_equal(joined["rows"], expected)
            or len(np.unique(joined["rows"])) != len(joined["rows"])
            or np.any(joined["cycles"] > origin)):
        raise RuntimeError(f"A02 E{origin} independent bank join or temporal boundary failed")
    if joined["ids"].astype(str).tolist() != train.record_id.astype(str).to_numpy()[joined["rows"]].tolist():
        raise RuntimeError(f"A02 E{origin} bank IDs failed Train join")
    return joined


def _model_bank_check(model: ResidualNeighborPredictor, bank: dict[str, np.ndarray],
                      train: pd.DataFrame, arm: str) -> np.ndarray:
    present = ~bank["missing"]
    rows = bank["rows"][present]
    cycles = bank["cycles"][present]
    probability = bank["baseline"][present]
    expected_ids = bank["ids"][present].astype(str).tolist()
    model_id_values = [str(value).split(":", 1)[-1] for value in model.bank_ids]
    if (model.origin_cycle < int(cycles.max()) or model.residual_arm != arm
            or not np.array_equal(model.bank_row_positions, rows)
            or model_id_values != expected_ids
            or not np.array_equal(model.bank_cycles, cycles)
            or not np.array_equal(model.bank_oof_probabilities, probability)):
        raise RuntimeError("Fitted model does not contain the exact verified OOF bank")
    y = train.artifact_signal.to_numpy(dtype=np.int8)
    residual = y[rows].astype(np.float64) - probability
    assignment = np.arange(len(rows), dtype=np.int64)
    if arm == "shuffled":
        deciles = np.minimum(9, np.floor(10.0 * probability).astype(np.int8))
        rng = np.random.default_rng(42)
        for cycle_value in np.unique(cycles):
            for decile in range(10):
                indices = np.flatnonzero((cycles == cycle_value) & (deciles == decile))
                if len(indices) > 1:
                    assignment[indices] = rng.permutation(indices)
    if (not np.array_equal(model.residual_assignment, assignment)
            or not np.array_equal(model.bank_residuals, residual[assignment])):
        raise RuntimeError("Observed/shuffled OOF residual assignment does not reconstruct")
    return residual[assignment]


def _independent_output(origin: int, arm: str, train: pd.DataFrame,
                        bank: dict[str, np.ndarray], query: dict[str, np.ndarray],
                        freeze: dict[str, Any]) -> dict[str, np.ndarray]:
    folder = RUN_ROOT / f"E{origin}/{arm}"
    model_path, state_path, receipt_path = folder / "model.joblib", folder / "fit_state.json", folder / "fit_receipt.json"
    pred_path, side_path = folder / "predictions.npz", folder / "predictions.json"
    model, state, receipt, sidecar = joblib.load(model_path), _json(state_path), _json(receipt_path), _json(side_path)
    if (canonical_sha256(model.audit_state()) != canonical_sha256(state)
            or receipt.get("state_sha256") != canonical_sha256(state)
            or receipt.get("model_sha256") != file_sha256(model_path)
            or receipt.get("freeze_signature") != freeze["freeze_signature"]
            or sidecar.get("freeze_signature") != freeze["freeze_signature"]
            or sidecar.get("npz_sha256") != file_sha256(pred_path)
            or sidecar.get("fit_receipt_sha256") != receipt.get("receipt_sha256")):
        raise RuntimeError(f"A02 E{origin}/{arm} persisted model/receipt/output hashes differ")
    receipt0 = dict(receipt)
    signature = receipt0.pop("receipt_sha256", None)
    if signature != canonical_sha256(receipt0):
        raise RuntimeError(f"A02 E{origin}/{arm} fit receipt canonical hash failed")
    fit_id = runner.FIT_IDS[(origin, arm)]
    if (receipt.get("fit_id") != fit_id or receipt.get("task_estimator_fit_count") != 1
            or receipt.get("query_labels_passed_to_fit") is not False
            or receipt.get("test_data_used") is not False):
        raise RuntimeError(f"A02 E{origin}/{arm} fit receipt lineage/boundary mismatch")
    shuffled_or_observed_residual = _model_bank_check(model, bank, train, arm)
    with np.load(pred_path, allow_pickle=False) as archive:
        out = {key: np.asarray(archive[key]).copy() for key in archive.files}
    required = {"score_rows", "score_record_ids", "score_cycles", "baseline_prediction", "prediction",
                "labels", "missing_mask", "entry_friction", "text_load_group", "neighbor_count",
                "neighbor_source_rows", "neighbor_ids", "neighbor_similarities", "correction", "correction_unclipped"}
    if set(out) != required:
        raise RuntimeError(f"A02 E{origin}/{arm} prediction schema changed")
    rows = out["score_rows"].astype(np.int64)
    if (not np.array_equal(rows, query["rows"])
            or out["score_record_ids"].astype(str).tolist() != query["ids"].astype(str).tolist()
            or not np.array_equal(out["score_cycles"].astype(np.int64), query["cycles"])
            or not np.array_equal(out["baseline_prediction"], query["baseline"])
            or not np.array_equal(out["missing_mask"], query["missing"])):
        raise RuntimeError(f"A02 E{origin}/{arm} prediction IDs/rows/cycles/baseline differ from C16 query")
    if (not np.array_equal(out["entry_friction"], train.entry_friction.to_numpy(dtype=np.int8)[rows])
            or out["score_record_ids"].astype(str).tolist() != train.record_id.astype(str).to_numpy()[rows].tolist()):
        raise RuntimeError(f"A02 E{origin}/{arm} saved query features fail official Train join")
    if set(out["score_record_ids"].astype(str).tolist()) & set(bank["ids"][~bank["missing"]].astype(str).tolist()):
        raise RuntimeError(f"A02 E{origin}/{arm} query IDs overlap residual bank")
    if np.intersect1d(rows, bank["rows"]).size or np.any(bank["cycles"] > origin) or np.any(query["cycles"] <= origin):
        raise RuntimeError(f"A02 E{origin}/{arm} source row/cycle temporal isolation failed")

    # Independent lexical geometry: instantiate a new vectorizer and choose
    # neighbors via full stable sort, not the training predictor's top-k path.
    cycle = train.cycle_code.to_numpy(dtype=np.int64)
    prefix_rows = np.flatnonzero(cycle <= origin)
    docs = train.signal_stream.fillna("").astype(str).tolist()
    params = dict(TFIDF_PARAMETERS)
    vectorizer = TfidfVectorizer(**params)
    prefix_matrix = vectorizer.fit_transform([docs[i] for i in prefix_rows]).tocsr()
    bank_present = ~bank["missing"]
    bank_rows = bank["rows"][bank_present]
    bank_matrix = vectorizer.transform(train.signal_stream.iloc[bank_rows].fillna("").astype(str).tolist()).tocsr()
    query_matrix = vectorizer.transform(train.signal_stream.iloc[rows].fillna("").astype(str).tolist()).tocsr()
    # Reconstruct exact lexical inputs as a guard against vocabulary/IDF drift.
    model_vocabulary = model.text_space.vectorizer.vocabulary_
    if model_vocabulary != vectorizer.vocabulary_ or not np.array_equal(model.text_space.vectorizer.idf_, vectorizer.idf_):
        raise RuntimeError(f"A02 E{origin}/{arm} TF-IDF prefix vocabulary/IDF differs")
    del prefix_matrix
    count_out = out["neighbor_count"].astype(np.int64)
    positions_out = out["neighbor_source_rows"].astype(np.int64)
    ids_out = out["neighbor_ids"].astype(str)
    sims_out = out["neighbor_similarities"].astype(np.float64)
    raw_out = out["correction_unclipped"].astype(np.float64)
    correction_out = out["correction"].astype(np.float64)
    prediction_out = out["prediction"].astype(np.float64)
    residual_by_row = {int(row): float(value) for row, value in zip(bank_rows, shuffled_or_observed_residual)}
    bank_id_by_row = {int(row): str(identifier) for row, identifier in zip(bank_rows, model.bank_ids)}
    expected_counts = np.zeros(len(rows), dtype=np.int64)
    expected_positions = np.full(positions_out.shape, -1, dtype=np.int64)
    expected_ids = np.full(ids_out.shape, "", dtype=ids_out.dtype)
    expected_sims = np.zeros_like(sims_out)
    expected_raw = np.zeros(len(rows), dtype=np.float64)
    for start in range(0, len(rows), 128):
        stop = min(start + 128, len(rows))
        sim_block = (query_matrix[start:stop] @ bank_matrix.T).toarray()
        for local, qi in enumerate(range(start, stop)):
            if query["missing"][qi]:
                continue
            positive = np.flatnonzero(sim_block[local] > 0.0)
            if not len(positive):
                continue
            ordered = positive[np.lexsort((bank_rows[positive], -sim_block[local, positive]))]
            chosen = ordered[:min(32, len(ordered))]
            sim = sim_block[local, chosen]
            selected_rows = bank_rows[chosen]
            if np.any(bank["cycles"][bank_present][chosen] >= query["cycles"][qi]):
                raise RuntimeError("A02 neighbor cycle is not strictly earlier than its query")
            if len(set(selected_rows.tolist())) != len(chosen):
                raise RuntimeError("A02 duplicate lexical neighbor source rows")
            n = len(chosen)
            expected_counts[qi] = n
            expected_positions[qi, :n] = selected_rows
            expected_ids[qi, :n] = [bank_id_by_row[int(row)] for row in selected_rows]
            expected_sims[qi, :n] = sim
            weights = sim * sim
            expected_raw[qi] = float(np.dot(weights, [residual_by_row[int(row)] for row in selected_rows]) /
                                     (4.0 + weights.sum()))
    if (not np.array_equal(count_out, expected_counts)
            or not np.array_equal(positions_out, expected_positions)
            or not np.array_equal(ids_out, expected_ids)
            or not np.allclose(sims_out, expected_sims, rtol=0, atol=1e-15)
            or not np.allclose(raw_out, expected_raw, rtol=0, atol=1e-15)):
        raise RuntimeError(f"A02 E{origin}/{arm} independent cosine top-32/tie/correction reconstruction failed")
    expected_correction = np.clip(expected_raw, -0.05, 0.05)
    expected_correction[query["missing"]] = 0.0
    expected_probability = np.clip(query["baseline"] + expected_correction, 0.0, 1.0)
    expected_probability[query["missing"]] = query["baseline"][query["missing"]]
    if (not np.array_equal(correction_out, expected_correction)
            or not np.array_equal(prediction_out, expected_probability)
            or not np.array_equal(out["labels"], (expected_probability >= 0.5).astype(np.int8))):
        raise RuntimeError(f"A02 E{origin}/{arm} route/threshold reconstruction failed")
    edges = np.quantile(train.text_load.to_numpy(dtype=np.float64)[prefix_rows], [0.25, 0.5, 0.75], method="linear")
    groups = np.searchsorted(edges, train.text_load.to_numpy(dtype=np.float64)[rows], side="right").astype(np.int8)
    if (not np.array_equal(out["text_load_group"], groups)
            or not np.array_equal(np.asarray(sidecar["text_load_quartile_fit_prefix_edges"]), edges)):
        raise RuntimeError(f"A02 E{origin}/{arm} fit-prefix subgroup edges changed")
    return {
        "rows": rows, "ids": out["score_record_ids"].astype(str), "cycles": out["score_cycles"].astype(np.int64),
        "baseline": query["baseline"], "prediction": prediction_out, "labels": out["labels"].astype(np.int8),
        "missing": query["missing"], "entry_friction": out["entry_friction"].astype(np.int8),
        "text_load_group": out["text_load_group"].astype(np.int8), "neighbor_count": count_out,
        "neighbor_source_rows": positions_out, "neighbor_similarities": sims_out,
        "correction": correction_out, "correction_unclipped": raw_out,
        "model_sha256": receipt["model_sha256"], "receipt_sha256": receipt["receipt_sha256"],
        "output_sha256": file_sha256(pred_path), "diagnostics": sidecar.get("diagnostics", {}),
    }


def _pilot_sample(origin: int, arm: str, train: pd.DataFrame,
                  expected_low: int, expected_high: int) -> dict[str, np.ndarray]:
    path = runner.PILOT_PREDICTION_ROOT / f"E{origin}/{arm}/predictions.npz"
    side = _json(path.with_suffix(".json"))
    with np.load(path, allow_pickle=False) as archive:
        values = {key: np.asarray(archive[key]).copy() for key in archive.files}
    if side.get("npz_sha256") != file_sha256(path):
        raise RuntimeError(f"Inherited pilot E{origin}/{arm} prediction hash mismatch")
    rows = values["score_rows"].astype(np.int64)
    keep = (values["score_cycles"] >= expected_low) & (values["score_cycles"] <= expected_high)
    selected = {key: value[keep] for key, value in values.items()}
    rows = selected["score_rows"].astype(np.int64)
    if (selected["score_record_ids"].astype(str).tolist() != train.record_id.astype(str).to_numpy()[rows].tolist()
            or not np.array_equal(selected["score_cycles"].astype(np.int64), train.cycle_code.to_numpy()[rows])):
        raise RuntimeError(f"Inherited pilot E{origin}/{arm} row/ID/cycle join failed")
    return {
        "rows": rows, "ids": selected["score_record_ids"].astype(str),
        "cycles": selected["score_cycles"].astype(np.int64),
        "baseline": selected["baseline_prediction"].astype(np.float64),
        "prediction": selected["prediction"].astype(np.float64),
        "labels": selected["labels"].astype(np.int8), "missing": selected["missing_mask"].astype(bool),
        "entry_friction": selected["entry_friction"].astype(np.int8),
        "text_load_group": selected["text_load_group"].astype(np.int8),
        "neighbor_count": selected["neighbor_count"].astype(np.int64),
        "correction": selected["correction"].astype(np.float64),
        "correction_unclipped": selected["correction_unclipped"].astype(np.float64),
        "neighbor_source_rows": selected["neighbor_source_rows"].astype(np.int64),
        "neighbor_similarities": selected["neighbor_similarities"].astype(np.float64),
        "model_sha256": None, "receipt_sha256": side.get("fit_receipt_sha256"),
        "output_sha256": file_sha256(path), "diagnostics": side.get("diagnostics", {}),
    }


def _slice(sample: dict[str, np.ndarray], mask: np.ndarray) -> dict[str, np.ndarray]:
    return {key: value[mask] for key, value in sample.items()
            if isinstance(value, np.ndarray) and value.ndim == 1 and len(value) == len(sample["rows"])}


def _concat(pieces: list[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    result = {key: np.concatenate([piece[key] for piece in pieces])
              for key in pieces[0] if isinstance(pieces[0][key], np.ndarray) and pieces[0][key].ndim == 1}
    order = np.argsort(result["rows"], kind="stable")
    result = {key: values[order] for key, values in result.items()}
    if len(np.unique(result["rows"])) != len(result["rows"]):
        raise RuntimeError("Evaluation view contains duplicate Train rows")
    return result


def _evaluate(samples: dict[str, dict[str, dict[str, np.ndarray]]],
              train: pd.DataFrame, protocol: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    y_all = train.artifact_signal.to_numpy(dtype=np.int8)
    definitions = {
        **{f"W{i}": [f"W{i}"] for i in range(1, 7)},
        "new_origin_aggregate": ["W2", "W3", "W5"],
        "full_six_primary": ["W1", "W2", "W3", "W4", "W5", "W6"],
        "late": ["W5", "W6"],
    }
    results: dict[str, Any] = {"views": {}, "per_cycle": [], "subgroups": {}, "diagnostics": {}}
    groups_rows: list[dict[str, Any]] = []
    view_data: dict[str, dict[str, dict[str, np.ndarray]]] = {}
    for view_name, window_names in definitions.items():
        view_data[view_name] = {arm: _concat([samples[w][arm] for w in window_names])
                                for arm in ("baseline", "observed", "shuffled")}
        ref = view_data[view_name]["baseline"]
        obs = view_data[view_name]["observed"]
        shuf = view_data[view_name]["shuffled"]
        if not (np.array_equal(ref["rows"], obs["rows"]) and np.array_equal(ref["rows"], shuf["rows"])):
            raise RuntimeError(f"A02 view {view_name} arm row sets differ")
        y = y_all[ref["rows"]]
        base_m, obs_m, shuf_m = _metric(y, ref["prediction"]), _metric(y, obs["prediction"]), _metric(y, shuf["prediction"])
        results["views"][view_name] = {
            "rows": len(y), "baseline": base_m, "observed": obs_m, "shuffled": shuf_m,
            "observed_gain_vs_baseline": obs_m["macro_f1"] - base_m["macro_f1"],
            "observed_gain_vs_matched_shuffle": obs_m["macro_f1"] - shuf_m["macro_f1"],
            "shuffled_gain_vs_baseline": shuf_m["macro_f1"] - base_m["macro_f1"],
            "probability": {
                "baseline": _probability_metrics(y, ref["prediction"], ref["prediction"]),
                "observed": _probability_metrics(y, obs["prediction"], ref["prediction"]),
                "shuffled": _probability_metrics(y, shuf["prediction"], ref["prediction"]),
            },
        }
        for arm, sample in (("baseline", ref), ("observed", obs), ("shuffled", shuf)):
            fields = {
                "stream_availability": np.where(sample["missing"], "missing", "present"),
                "entry_friction": train.entry_friction.to_numpy(dtype=np.int64)[sample["rows"]].astype(str),
                "fit_prefix_defined_text_load_quartile": np.array([f"Q{int(x)+1}" for x in sample["text_load_group"]]),
            }
            results["subgroups"].setdefault(view_name, {}).setdefault(arm, {})
            for field, values in fields.items():
                results["subgroups"][view_name][arm][field] = {}
                for label in sorted(set(values.tolist())):
                    mask = values == label
                    gy = y[mask]
                    c0, c1 = int(np.sum(gy == 0)), int(np.sum(gy == 1))
                    supported = int(mask.sum()) >= 200 and c0 >= 30 and c1 >= 30
                    group = {"rows": int(mask.sum()), "class_0_rows": c0, "class_1_rows": c1,
                             "supported": bool(supported)}
                    if supported:
                        gm = _metric(gy, sample["prediction"][mask])
                        group["metrics"] = gm
                        if arm != "baseline":
                            bm = _metric(gy, ref["prediction"][mask])
                            f1_harm = bm["macro_f1"] - gm["macro_f1"]
                            recall_harm = (bm["positive_recall"] or 0.0) - (gm["positive_recall"] or 0.0)
                            group["harm_vs_baseline"] = {"macro_f1": f1_harm, "positive_recall": recall_harm}
                            group["historical_flags"] = {
                                "f1_gt_0_01": f1_harm > 0.01,
                                "recall_gt_0_02": recall_harm > 0.02,
                                "recall_gt_0_03": recall_harm > 0.03,
                            }
                            group["risk_review_hold"] = f1_harm > 0.03 or recall_harm > 0.05
                    results["subgroups"][view_name][arm][field][label] = group
                    if arm != "baseline":
                        groups_rows.append({"view": view_name, "arm": arm, "field": field, "group": label,
                                            "rows": int(mask.sum()), "class_0_rows": c0, "class_1_rows": c1,
                                            "supported": bool(supported),
                                            "macro_f1": group.get("metrics", {}).get("macro_f1"),
                                            "positive_recall": group.get("metrics", {}).get("positive_recall"),
                                            "f1_harm_vs_baseline": group.get("harm_vs_baseline", {}).get("macro_f1"),
                                            "recall_harm_vs_baseline": group.get("harm_vs_baseline", {}).get("positive_recall"),
                                            "risk_review_hold": group.get("risk_review_hold", False)})
        # Per-cycle metrics use the origin-specific prediction for each four-cycle block.
        for window_name in window_names:
            for cycle_value in np.unique(samples[window_name]["observed"]["cycles"]):
                window_sample = {arm: samples[window_name][arm] for arm in ("baseline", "observed", "shuffled")}
                mask = window_sample["baseline"]["cycles"] == cycle_value
                cy = y_all[window_sample["baseline"]["rows"][mask]]
                bm = _metric(cy, window_sample["baseline"]["prediction"][mask])
                om = _metric(cy, window_sample["observed"]["prediction"][mask])
                sm = _metric(cy, window_sample["shuffled"]["prediction"][mask])
                results["per_cycle"].append({"window": window_name, "cycle": int(cycle_value), "rows": len(cy),
                                              "baseline_macro_f1": bm["macro_f1"], "observed_macro_f1": om["macro_f1"],
                                              "shuffled_macro_f1": sm["macro_f1"],
                                              "observed_gain_vs_baseline": om["macro_f1"] - bm["macro_f1"],
                                              "observed_gain_vs_matched_shuffle": om["macro_f1"] - sm["macro_f1"]})
    # Inherited H12-L uses the verified E67 observed/shuffled bundles. There is
    # no persisted "baseline" arm file; reconstruct that arm from each bundle's
    # verified baseline_prediction array.
    h12_observed = _pilot_sample(67, "observed", train, 68, 79)
    h12_shuffled = _pilot_sample(67, "shuffled", train, 68, 79)
    h12_data = {
        "baseline": {**h12_observed, "prediction": h12_observed["baseline"].copy(),
                     "labels": (h12_observed["baseline"] >= 0.5).astype(np.int8)},
        "observed": h12_observed,
        "shuffled": h12_shuffled,
    }
    if not (np.array_equal(h12_data["baseline"]["rows"], h12_data["observed"]["rows"])
            and np.array_equal(h12_data["baseline"]["rows"], h12_data["shuffled"]["rows"])):
        raise RuntimeError("Inherited H12-L pilot row sets differ")
    hy = y_all[h12_data["baseline"]["rows"]]
    hb, ho, hs = (_metric(hy, h12_data[a]["prediction"]) for a in ("baseline", "observed", "shuffled"))
    results["views"]["inherited_H12_L"] = {
        "rows": len(hy), "baseline": hb, "observed": ho, "shuffled": hs,
        "observed_gain_vs_baseline": ho["macro_f1"] - hb["macro_f1"],
        "observed_gain_vs_matched_shuffle": ho["macro_f1"] - hs["macro_f1"],
        "shuffled_gain_vs_baseline": hs["macro_f1"] - hb["macro_f1"],
        "inherited_pilot_evidence": True,
        "independent_of_new_origin_aggregate": False,
    }
    # Compute H12 subgroup guards with E67 fit-prefix group labels and same pilot rows.
    h12_proxy = {arm: h12_data[arm] for arm in ("baseline", "observed", "shuffled")}
    results["subgroups"]["inherited_H12_L"] = {}
    # Translate through same evaluation logic with the saved fit-defined Q groups.
    for arm in ("baseline", "observed", "shuffled"):
        sample = h12_proxy[arm]
        fields = {
            "stream_availability": np.where(sample["missing"], "missing", "present"),
            "entry_friction": train.entry_friction.to_numpy(dtype=np.int64)[sample["rows"]].astype(str),
            "fit_prefix_defined_text_load_quartile": np.array([f"Q{int(x)+1}" for x in sample["text_load_group"]]),
        }
        results["subgroups"]["inherited_H12_L"][arm] = {}
        for field, values in fields.items():
            results["subgroups"]["inherited_H12_L"][arm][field] = {}
            for label in sorted(set(values.tolist())):
                mask = values == label
                gy = hy[mask]
                c0, c1 = int(np.sum(gy == 0)), int(np.sum(gy == 1))
                supported = int(mask.sum()) >= 200 and c0 >= 30 and c1 >= 30
                group = {"rows": int(mask.sum()), "class_0_rows": c0, "class_1_rows": c1,
                         "supported": bool(supported)}
                if supported:
                    gm = _metric(gy, sample["prediction"][mask])
                    group["metrics"] = gm
                    if arm != "baseline":
                        bm = _metric(gy, h12_data["baseline"]["prediction"][mask])
                        fh = bm["macro_f1"] - gm["macro_f1"]
                        rh = (bm["positive_recall"] or 0.0) - (gm["positive_recall"] or 0.0)
                        group["harm_vs_baseline"] = {"macro_f1": fh, "positive_recall": rh}
                        group["historical_flags"] = {"f1_gt_0_01": fh > .01,
                            "recall_gt_0_02": rh > .02, "recall_gt_0_03": rh > .03}
                        group["risk_review_hold"] = fh > .03 or rh > .05
                results["subgroups"]["inherited_H12_L"][arm][field][label] = group
                if arm != "baseline":
                    groups_rows.append({"view": "inherited_H12_L", "arm": arm, "field": field, "group": label,
                                        "rows": int(mask.sum()), "class_0_rows": c0, "class_1_rows": c1,
                                        "supported": bool(supported),
                                        "macro_f1": group.get("metrics", {}).get("macro_f1"),
                                        "positive_recall": group.get("metrics", {}).get("positive_recall"),
                                        "f1_harm_vs_baseline": group.get("harm_vs_baseline", {}).get("macro_f1"),
                                        "recall_harm_vs_baseline": group.get("harm_vs_baseline", {}).get("positive_recall"),
                                        "risk_review_hold": group.get("risk_review_hold", False)})
    for arm in ("observed", "shuffled"):
        source = h12_data[arm]
        y = hy
        results["views"]["inherited_H12_L"].setdefault("probability", {})
        results["views"]["inherited_H12_L"]["probability"][arm] = _probability_metrics(
            y, source["prediction"], h12_data["baseline"]["prediction"])
    results["views"]["inherited_H12_L"].setdefault("probability", {})
    results["views"]["inherited_H12_L"]["probability"]["baseline"] = _probability_metrics(
        hy, h12_data["baseline"]["prediction"], h12_data["baseline"]["prediction"])

    historical_flags = {"f1_gt_0_01": 0, "recall_gt_0_02": 0, "recall_gt_0_03": 0}
    risk_flags = []
    for row in groups_rows:
        if row["supported"]:
            view = results["subgroups"][row["view"]][row["arm"]][row["field"]][row["group"]]
            for flag in historical_flags:
                historical_flags[flag] += int(view.get("historical_flags", {}).get(flag, False))
            if view.get("risk_review_hold"):
                risk_flags.append({**row})
    results["guard_summary"] = {
        "minimum_support": {"n": 200, "each_class": 30},
        "supported_group_view_arm_rows": sum(bool(row["supported"]) for row in groups_rows),
        "historical_harm_flags_fired": historical_flags,
        "risk_review_hold_flags": risk_flags,
        "risk_review_hold": bool(risk_flags),
    }
    results["source_boundary"] = {
        "official_train_only": True, "official_test_or_sample_loaded": False,
        "validation_labels_passed_to_fit": False,
        "npz_labels_used_as_y_true": False,
        "evaluation_cycles_reused_from_prior_research": True,
        "new_predictions_only_at_origins_59_63_71": True,
    }
    return results, results["per_cycle"], groups_rows


def _gate_results(results: dict[str, Any], protocol: dict[str, Any]) -> dict[str, Any]:
    v = results["views"]
    g = protocol["scientific_gates"]
    windows = [v[f"W{i}"] for i in (2, 3, 5)]
    new = v["new_origin_aggregate"]
    full = v["full_six_primary"]
    late = v["late"]
    h12 = v["inherited_H12_L"]
    checks = {
        "new_aggregate_vs_baseline": new["observed_gain_vs_baseline"] >= g["new_origin_aggregate_gain_vs_baseline_min"],
        "new_aggregate_vs_matched_shuffle": new["observed_gain_vs_matched_shuffle"] >= g["new_origin_aggregate_gain_vs_matched_shuffle_min"],
        "at_least_two_new_windows_nonnegative": sum(w["observed_gain_vs_baseline"] >= 0 for w in windows) >= g["new_windows_nonnegative_vs_baseline_min_count"],
        "no_new_window_harm_gt_0_003": all(w["observed_gain_vs_baseline"] >= -g["new_window_harm_vs_baseline_max"] for w in windows),
        "full_six_vs_baseline_nonnegative": full["observed_gain_vs_baseline"] >= g["full_six_primary_gain_vs_baseline_min"],
        "full_six_vs_matched_shuffle_nonnegative": full["observed_gain_vs_matched_shuffle"] >= g["full_six_primary_gain_vs_matched_shuffle_min"],
        "late_harm_at_most_0_001": late["observed_gain_vs_baseline"] >= -g["late_W5_W6_harm_vs_baseline_max"],
        "inherited_H12_L_vs_baseline": h12["observed_gain_vs_baseline"] >= g["inherited_H12_L_gain_vs_baseline_min"],
        "inherited_H12_L_vs_matched_shuffle": h12["observed_gain_vs_matched_shuffle"] >= g["inherited_H12_L_gain_vs_matched_shuffle_min"],
    }
    risk = results["guard_summary"]["risk_review_hold"]
    return {"checks": checks, "scientific_status": "PASS" if all(checks.values()) else "FAIL",
            "risk_review_hold": bool(risk), "conditional_final_eligible": all(checks.values()) and not risk,
            "failed_checks": [name for name, passed in checks.items() if not passed]}


def _write_outputs(results: dict[str, Any], cycles: list[dict[str, Any]],
                   groups: list[dict[str, Any]], freeze: dict[str, Any],
                   protocol: dict[str, Any]) -> None:
    gates = _gate_results(results, protocol)
    output = {
        "status": "A02_CONTINUATION_EVALUATED", "candidate_id": protocol["candidate_id"],
        "source_commit": freeze["identity"]["source_commit"],
        "protocol_sha256": freeze["protocol_sha256"], "freeze_signature": freeze["freeze_signature"],
        "budget_cap_amendment_id": "A02_CONTINUATION_CAP66_2026_11_02",
        "fit_ids": list(protocol["fit_ids"].values()), "new_task_fit_count": 6,
        "views": results["views"], "gates": gates, "guard_summary": results["guard_summary"],
        "source_boundary": results["source_boundary"], "interpretation": {
            "predictive_association_not_causation": True,
            "evaluation_rows_previously_seen_in_research": True,
            "new_origin_predictions_only": [59, 63, 71],
            "fullsix_origin_route": "W1 exact C16 fallback; W2/W3/W5 new; W4/W6 inherited pilot",
            "h12_l_is_inherited_pilot_evidence_and_not_independent": True,
        },
        "endpoint_diagnostic_convention": {
            "manual_clipping": 1e-15,
            "library_default": f"scikit-learn {__import__('sklearn').__version__} log_loss default dtype epsilon {np.finfo(np.float64).eps}",
            "probabilities_are_never_modified_for_scoring": True,
        },
    }
    runner._write_once(REPORT_ROOT / "results.json", output)
    _write_csv_once(REPORT_ROOT / "per_cycle.csv", pd.DataFrame(cycles))
    _write_csv_once(REPORT_ROOT / "groups.csv", pd.DataFrame(groups))
    lines = [
        "# A02 continuation — result checkpoint", "",
        f"**Scientific gates:** {gates['scientific_status']}  ",
        f"**Risk review:** {'HOLD' if gates['risk_review_hold'] else 'no hold flag'}  ",
        f"**Conditional final eligible:** {gates['conditional_final_eligible']}", "",
        "The three continuation origins (59, 63, 71) produce 21,033 previously used development rows. "
        "They add new origin-specific predictions, not an untouched holdout. The 38,588-row full-six "
        "view includes an exact C16 fallback at W1 and the already verified A02 pilot predictions at W4/W6.", "",
        "| View | Rows | Baseline Macro F1 | Observed Macro F1 | Δ vs baseline | Δ vs matched shuffle |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for view in ("W1", "W2", "W3", "W4", "W5", "W6", "new_origin_aggregate", "full_six_primary", "late", "inherited_H12_L"):
        row = results["views"][view]
        lines.append(f"| {view} | {row['rows']:,} | {row['baseline']['macro_f1']:.6f} | {row['observed']['macro_f1']:.6f} | "
                     f"{row['observed_gain_vs_baseline']:+.6f} | {row['observed_gain_vs_matched_shuffle']:+.6f} |")
    lines.extend(["", "H12-L is inherited pilot evidence and reuses development rows. Full per-cycle, subgroup, "
                  "confusion, precision/recall, probability, and endpoint diagnostics are in `results.json`, `per_cycle.csv`, and `groups.csv`.", "",
                  "The endpoint log-loss diagnostic reports manual clipping at 1e-15 separately from scikit-learn's default dtype epsilon. "
                  "Predictions were not clipped or recalibrated for analysis. The route remains a predictive association; it is not causal.", "",
                  "No Test/SampleSubmission was accessed in the continuation evaluation. A final origin-79 fit/inference is allowed only if all frozen gates pass and no risk-review hold is present.", ""])
    md_path = REPORT_ROOT / "SUMMARY.md"
    md_text = "\n".join(lines)
    if md_path.exists() and md_path.read_text(encoding="utf-8") != md_text:
        raise RuntimeError("A02 continuation summary is immutable and differs")
    if not md_path.exists():
        md_path.parent.mkdir(parents=True, exist_ok=True)
        md_path.write_text(md_text, encoding="utf-8", newline="\n")


def _write_csv_once(path: Path, frame: pd.DataFrame) -> None:
    text = frame.to_csv(index=False, lineterminator="\n")
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_text(encoding="utf-8") != text:
            raise RuntimeError(f"Immutable A02 continuation CSV differs: {path}")
        return
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(text)
        stream.flush()
        os.fsync(stream.fileno())


def verify(*, write_outputs: bool = True) -> dict[str, Any]:
    freeze = runner.verify_freeze()
    protocol = _json(runner.PROTOCOL_PATH)
    # Independent cache loading does not call the continuation runner's cache reader.
    train = pd.read_csv(TRAIN_PATH, usecols=["record_id", "cycle_code", "signal_stream", "entry_friction", "text_load", "artifact_signal"], dtype={"record_id": str})
    if len(train) != 50608 or not train.record_id.is_unique:
        raise RuntimeError("Official Classification Train schema/identity mismatch")
    caches = {cutoff: _independent_cache(cutoff, train, protocol) for cutoff in sorted(pilot.CACHE_PATHS)}
    new_samples: dict[tuple[int, str], dict[str, np.ndarray]] = {}
    banks = {origin: _independent_bank(origin, caches, train, protocol) for origin in (59, 63, 71)}
    for origin in (59, 63, 71):
        query = _select_cache(caches[origin], train, origin + 1, origin + 4)
        bank = banks[origin]
        for arm in ("observed", "shuffled"):
            new_samples[(origin, arm)] = _independent_output(origin, arm, train, bank, query, freeze)
        for key in ("neighbor_count", "neighbor_source_rows", "neighbor_similarities"):
            if not np.array_equal(new_samples[(origin, "observed")][key], new_samples[(origin, "shuffled")][key]):
                raise RuntimeError(f"Matched shuffle changed geometry for E{origin}/{key}")

    events = read_events(LEDGER_PATH)
    started = [e for e in events if e.get("event") == "FIT_STARTED" and e.get("card_id") == CONTINUATION_CARD]
    saved = [e for e in events if e.get("event") == "FIT_SAVED" and e.get("card_id") == CONTINUATION_CARD]
    failed = [e for e in events if e.get("event") == "FIT_FAILED" and e.get("card_id") == CONTINUATION_CARD]
    expected_ids = sorted(runner.FIT_IDS.values())
    if (sorted(e.get("fit_id") for e in started) != expected_ids
            or sorted(e.get("fit_id") for e in saved) != expected_ids or failed):
        raise RuntimeError("A02 continuation ledger does not show exactly six saved/no-failure fits")
    reservation = [e for e in events if e.get("event") == "CARD_FITS_RESERVED"
                   and e.get("card_id") == CONTINUATION_CARD and e.get("stage") == CONTINUATION_STAGE]
    if (len(reservation) != 1 or reservation[0].get("fits") != 6
            or reservation[0].get("freeze_signature") != freeze["freeze_signature"]):
        raise RuntimeError("A02 continuation six-fit reservation does not match freeze")
    for event in started:
        if (event.get("reservation_key") != f"{CONTINUATION_CARD}:{CONTINUATION_STAGE}"
                or event.get("freeze_signature") != freeze["freeze_signature"]):
            raise RuntimeError(f"A02 continuation fit {event.get('fit_id')} budget lineage mismatch")
    # Build W1 exact fallback without touching Test; no predictor was trained there.
    w1 = _select_cache(caches[55], train, 56, 59)
    edges55 = np.quantile(train.text_load.to_numpy(dtype=np.float64)[train.cycle_code.to_numpy(dtype=np.int64) <= 55],
                          [0.25, 0.5, 0.75], method="linear")
    w1_groups = np.searchsorted(edges55, train.text_load.to_numpy(dtype=np.float64)[w1["rows"]], side="right").astype(np.int8)
    fallback = {"rows": w1["rows"], "ids": w1["ids"].astype(str), "cycles": w1["cycles"],
                "baseline": w1["baseline"], "prediction": w1["baseline"], "labels": (w1["baseline"] >= .5).astype(np.int8),
                "missing": w1["missing"], "entry_friction": train.entry_friction.to_numpy(dtype=np.int8)[w1["rows"]],
                "text_load_group": w1_groups, "neighbor_count": np.zeros(len(w1["rows"]), dtype=np.int64),
                "correction": np.zeros(len(w1["rows"])), "correction_unclipped": np.zeros(len(w1["rows"])),
                "neighbor_source_rows": np.empty((len(w1["rows"]), 0), dtype=np.int64),
                "neighbor_similarities": np.empty((len(w1["rows"]), 0), dtype=np.float64)}
    samples: dict[str, dict[str, dict[str, np.ndarray]]] = {}
    samples["W1"] = {arm: fallback for arm in ("baseline", "observed", "shuffled")}
    for origin, wname in ((59, "W2"), (63, "W3"), (67, "W4"), (71, "W5"), (75, "W6")):
        if origin in (59, 63, 71):
            observed = new_samples[(origin, "observed")]
            shuffled = new_samples[(origin, "shuffled")]
            baseline = {**observed, "prediction": observed["baseline"], "labels": (observed["baseline"] >= .5).astype(np.int8)}
        else:
            low, high = (68, 71) if origin == 67 else (76, 79)
            observed = _pilot_sample(origin, "observed", train, low, high)
            shuffled = _pilot_sample(origin, "shuffled", train, low, high)
            baseline = {**observed, "prediction": observed["baseline"], "labels": (observed["baseline"] >= .5).astype(np.int8)}
            if (not np.array_equal(observed["rows"], shuffled["rows"])
                    or not np.array_equal(observed["baseline"], shuffled["baseline"])):
                raise RuntimeError(f"Inherited pilot E{origin} observed/shuffled row or baseline mismatch")
            pilot_verification = _json(runner.PILOT_REPORT_ROOT / "verification.json")
            if pilot_verification.get("status") != "PASS":
                raise RuntimeError("Inherited A02 pilot verification is not PASS")
        samples[wname] = {"baseline": baseline, "observed": observed, "shuffled": shuffled}
    results, per_cycle, groups = _evaluate(samples, train, protocol)
    gate = _gate_results(results, protocol)
    results["gates"] = gate
    # Add primary pilot comparison as historical context; the gate uses full-six as frozen.
    pilot_primary = _concat([samples["W4"]["observed"], samples["W6"]["observed"]])
    pilot_baseline = _concat([samples["W4"]["baseline"], samples["W6"]["baseline"]])
    pilot_shuffle = _concat([samples["W4"]["shuffled"], samples["W6"]["shuffled"]])
    y = train.artifact_signal.to_numpy(dtype=np.int8)[pilot_baseline["rows"]]
    pb, po, ps = (_metric(y, s["prediction"]) for s in (pilot_baseline, pilot_primary, pilot_shuffle))
    results["views"]["inherited_pilot_primary_W4_W6"] = {
        "rows": len(y), "baseline": pb, "observed": po, "shuffled": ps,
        "observed_gain_vs_baseline": po["macro_f1"] - pb["macro_f1"],
        "observed_gain_vs_matched_shuffle": po["macro_f1"] - ps["macro_f1"],
        "inherited_pilot_evidence": True,
    }
    accounting = __import__("src.models.optimization_v3_a02_continuation_budget", fromlist=["verify_accounting"]).verify_accounting(
        LEDGER_PATH, freeze_signature=freeze["freeze_signature"])
    if accounting["started"] > 66 or accounting["active_reserved"] > 66:
        raise RuntimeError("A02 continuation amended budget exceeded cap 66")
    output = {
        "status": "PASS", "candidate_id": protocol["candidate_id"],
        "source_commit": freeze["identity"]["source_commit"], "protocol_sha256": freeze["protocol_sha256"],
        "freeze_signature": freeze["freeze_signature"], "checks": {
            "freeze_sources_inputs_and_budget_amendment": True,
            "all_six_oof_caches_and_route_reconstructed": True,
            "three_new_origin_banks_and_temporal_bounds": True,
            "six_model_states_receipts_and_prediction_hashes": True,
            "independent_tfidf_cosine_top32_ties_residuals_and_route": True,
            "observed_and_shuffled_neighbor_geometry_identical": True,
            "official_train_labels_joined_by_score_rows": True,
            "npz_labels_not_used_as_ground_truth": True,
            "full_six_fallback_and_pilot_reuse_verified": True,
            "conditional_final_not_started_during_continuation": not any(e.get("event") == "FIT_STARTED" and e.get("fit_id") == "A02_CONT_E79_OBSERVED_FULLTRAIN" for e in events),
        },
        "new_continuation_fit_ids": expected_ids,
        "new_continuation_fits_started_saved_failed": {"started": len(started), "saved": len(saved), "failed": len(failed)},
        "budget_accounting": accounting,
        "gate_summary": gate,
    }
    if not output["checks"]["conditional_final_not_started_during_continuation"]:
        raise RuntimeError("Conditional final fit started before the gate review")
    if write_outputs:
        _write_outputs(results, per_cycle, groups, freeze, protocol)
        runner._write_once(REPORT_ROOT / "verification.json", output)
    return output


def run_synthetic_qa() -> dict[str, Any]:
    """Run a small synthetic model/persistence/resume/failure-injection E2E only."""
    from src.models.classification_a02_neighbors import fit_text_space
    from src.models import optimization_v3_a02_continuation_budget as budget
    from scripts import optimization_v3_a02_continuation as run

    synthetic = pd.DataFrame({
        "record_id": [f"syn_{i}" for i in range(8)],
        "cycle_code": [49, 49, 49, 50, 50, 50, 51, 52],
        "signal_stream": ["anchor amber alpha", "anchor amber beta", "anchor amber gamma",
                          "anchor cobalt delta", "anchor cobalt epsilon", "anchor cobalt zeta",
                          "anchor amber query", "  "],
        "entry_friction": [0, 0, 1, 0, 1, 1, 0, 1],
        "text_load": [10., 20., 30., 40., 50., 60., 70., 80.],
        "artifact_signal": [0, 1, 0, 1, 0, 1, 1, 0],
    })
    prefix = np.arange(6, dtype=np.int64)
    space, prefix_matrix, _ = fit_text_space(synthetic.signal_stream.iloc[:6], prefix,
        synthetic.cycle_code.to_numpy(dtype=np.int64)[:6], origin_cycle=50)
    del prefix_matrix
    baseline = np.array([0.15, 0.18, 0.17, 0.85, 0.88, 0.86], dtype=np.float64)
    bank = {
        "rows": prefix.copy(), "ids": synthetic.record_id.astype(str).to_numpy()[:6],
        "cycles": synthetic.cycle_code.to_numpy(dtype=np.int64)[:6], "baseline": baseline,
        "missing": np.zeros(6, dtype=bool),
    }
    query = {
        "rows": np.array([6, 7], dtype=np.int64), "ids": synthetic.record_id.astype(str).to_numpy()[6:],
        "cycles": synthetic.cycle_code.to_numpy(dtype=np.int64)[6:],
        "baseline": np.array([0.42, 0.63], dtype=np.float64), "missing": np.array([False, True]),
    }
    models = {}
    outputs = {}
    residuals = synthetic.artifact_signal.to_numpy(dtype=np.int8)[:6].astype(np.float64) - baseline
    for arm in ("observed", "shuffled"):
        model = ResidualNeighborPredictor(origin_cycle=50, residual_arm=arm)
        model.fit(space, synthetic.signal_stream.iloc[:6], bank["ids"], bank["rows"], bank["cycles"], baseline, residuals)
        out = model.predict(synthetic.signal_stream.iloc[6:], query["ids"], query["rows"], query["cycles"],
                            query["baseline"], missing_mask=query["missing"])
        if not np.array_equal(out["candidate_probability"][query["missing"]], query["baseline"][query["missing"]]):
            raise RuntimeError("Synthetic exact missing-stream fallback failed")
        models[arm], outputs[arm] = model, out
    for key in ("neighbor_count", "neighbor_source_rows", "neighbor_similarities"):
        if not np.array_equal(outputs["observed"][key], outputs["shuffled"][key]):
            raise RuntimeError("Synthetic matched shuffle changed lexical geometry")
    if models["shuffled"].residual_assignment.shape != (6,) or np.array_equal(
            models["shuffled"].bank_residuals, residuals):
        raise RuntimeError("Synthetic shuffled residual assignment did not exercise the control")

    with tempfile.TemporaryDirectory(prefix="a02_cont_synthetic_") as tmp:
        temp = Path(tmp)
        fit_folder = temp / "fit"
        ledger = temp / "ledger.jsonl"
        fit_id = "SYNTH_A02_RESUME_ONLY"
        ledger.write_text(json.dumps({"event": "FIT_STARTED", "fit_id": fit_id,
            "batch_id": "synthetic_optimization_v3_research_next_v1",
            "reservation_key": "SYNTHETIC_ONLY:SYNTHETIC",
            "card_id": "SYNTHETIC_ONLY", "stage": "SYNTHETIC", "cutoff": 50,
            "component": "synthetic", "fit_consumed_at_start": True}) + "\n", encoding="utf-8")
        run._atomic_joblib(models["observed"], fit_folder / "model.joblib")
        # Inject a crash after the durable model but before state/receipt/terminal event.
        freeze = {"freeze_signature": "synthetic-only-freeze"}
        with mock.patch.object(ResidualNeighborPredictor, "fit", side_effect=AssertionError("resume must not refit")):
            resumed, receipt = run.fit_or_resume_model(
                origin=50, arm="observed", text_space=space, bank=bank, train=synthetic,
                freeze=freeze, folder=fit_folder, ledger_path=ledger, fit_id=fit_id,
                card_id="SYNTHETIC_ONLY", stage="SYNTHETIC",
            )
            resumed_again, receipt_again = run.fit_or_resume_model(
                origin=50, arm="observed", text_space=space, bank=bank, train=synthetic,
                freeze=freeze, folder=fit_folder, ledger_path=ledger, fit_id=fit_id,
                card_id="SYNTHETIC_ONLY", stage="SYNTHETIC",
            )
        events = read_events(ledger)
        if (receipt["receipt_sha256"] != receipt_again["receipt_sha256"]
                or not np.array_equal(resumed.audit_state()["bank_cycles_sha256"], resumed_again.audit_state()["bank_cycles_sha256"])
                or sum(e.get("event") == "FIT_STARTED" for e in events) != 1
                or sum(e.get("event") == "FIT_SAVED" for e in events) != 1
                or receipt.get("persistence_recovered_without_refit") is not True):
            raise RuntimeError("Synthetic persistence/resume did not recover exact saved state")
        outroot = temp / "outputs"
        first_hash = None
        original_write_once = run._write_once
        def injected_write(path: Path, value: Any) -> None:
            if path.name == "predictions.json":
                raise OSError("synthetic injected stop after NPZ persistence")
            original_write_once(path, value)
        try:
            with mock.patch.object(run, "_write_once", side_effect=injected_write):
                run.write_predictions_once(50, "observed", resumed, query, synthetic,
                    receipt, freeze, output_root=outroot)
        except OSError as exc:
            if "synthetic injected" not in str(exc):
                raise
        pred_path = outroot / "E50/observed/predictions.npz"
        if not pred_path.is_file() or (outroot / "E50/observed/predictions.json").exists():
            raise RuntimeError("Synthetic output failure injection did not land after NPZ atomic replace")
        first_hash = file_sha256(pred_path)
        run.write_predictions_once(50, "observed", resumed, query, synthetic, receipt, freeze, output_root=outroot)
        run.write_predictions_once(50, "observed", resumed, query, synthetic, receipt, freeze, output_root=outroot)
        if first_hash != file_sha256(pred_path):
            raise RuntimeError("Synthetic output resume changed saved NPZ bytes")
        with np.load(pred_path, allow_pickle=False) as archive:
            saved = {key: np.asarray(archive[key]).copy() for key in archive.files}
        if not np.array_equal(saved["prediction"], outputs["observed"]["candidate_probability"]):
            raise RuntimeError("Synthetic E2E predictions changed on persistence roundtrip")

    receipt = {
        "status": "PASS", "fit_type": "SYNTHETIC_ONLY",
        "no_official_data_loaded": True, "synthetic_core_predictor_fits": 2,
        "official_fit_budget_consumed": 0,
        "checks": {
            "observed_and_shuffled_core_paths": True,
            "shuffle_assignment_nonidentity": True,
            "observed_shuffled_geometry_equal": True,
            "missing_route_bit_exact": True,
            "model_write_before_resume": True,
            "missing_state_and_receipt_recovered_without_refit": True,
            "duplicate_fit_start_or_terminal_not_added": True,
            "failure_injected_after_prediction_npz_before_sidecar": True,
            "prediction_npz_exact_readback_and_resume": True,
            "npz_bytes_unchanged_on_resume": True,
        },
    }
    path = REPORT_ROOT / "synthetic_qa.json"
    runner._write_once(path, receipt)
    return receipt


__all__ = ["run_synthetic_qa", "verify"]
