"""Independent A02 verifier: cache lineage, neighbor math, metrics, guards, and gates."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.models.optimization_v3_budget_amendment import AMENDMENT_ID  # noqa: E402
from src.models.research_next_budget import read_events  # noqa: E402
from src.models.research_next_contract import canonical_sha256, file_sha256  # noqa: E402
from scripts.optimization_v3_a02_neighbors import (  # noqa: E402
    CACHE_PATHS, EXPECTED_ROWS, EXPECTED_TRAIN_SHA256, FIT_IDS, FREEZE_PATH,
    LEDGER_PATH, REPORT_ROOT, RUN_ROOT, TRAIN_PATH, _read_json,
)


def _metric(y: np.ndarray, probability: np.ndarray) -> dict[str, Any]:
    truth = np.asarray(y, dtype=np.int8)
    pred = (np.asarray(probability, dtype=np.float64) >= 0.5).astype(np.int8)
    tn = int(np.sum((truth == 0) & (pred == 0)))
    fp = int(np.sum((truth == 0) & (pred == 1)))
    fn = int(np.sum((truth == 1) & (pred == 0)))
    tp = int(np.sum((truth == 1) & (pred == 1)))
    f1_0 = 2 * tn / (2 * tn + fp + fn) if (2 * tn + fp + fn) else 0.0
    f1_1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0
    return {
        "rows": int(len(truth)),
        "macro_f1": float((f1_0 + f1_1) / 2.0),
        "positive_recall": float(tp / (tp + fn)) if (tp + fn) else None,
        "confusion": {"tn": tn, "fp": fp, "fn": fn, "tp": tp},
        "positive_predictions": int(pred.sum()),
    }


def _hash_ids(values: list[str]) -> str:
    encoded = json.dumps(values, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _check_frozen_inputs(freeze: dict[str, Any], protocol: dict[str, Any]) -> dict[str, Any]:
    if canonical_sha256(protocol) != freeze.get("protocol_sha256"):
        raise RuntimeError("A02 verifier: protocol signature mismatch")
    if canonical_sha256(freeze.get("identity")) != freeze.get("freeze_signature"):
        raise RuntimeError("A02 verifier: freeze identity signature mismatch")
    checks = 0
    for relative, expected in freeze["source_sha256"].items():
        if file_sha256(ROOT / relative) != expected:
            raise RuntimeError(f"A02 verifier: source changed {relative}")
        checks += 1
    for relative, expected in freeze["input_sha256"].items():
        if file_sha256(ROOT / relative) != expected:
            raise RuntimeError(f"A02 verifier: input changed {relative}")
        checks += 1
    for relative, expected in freeze["identity"].get("synthetic_qa_sha256", {}).items():
        if file_sha256(ROOT / relative) != expected:
            raise RuntimeError(f"A02 verifier: synthetic QA receipt changed {relative}")
        checks += 1
    if not freeze.get("identity", {}).get("source_commit"):
        raise RuntimeError("A02 verifier: frozen source commit is absent")
    if file_sha256(TRAIN_PATH) != EXPECTED_TRAIN_SHA256:
        raise RuntimeError("A02 verifier: official Train SHA mismatch")
    amendment = [event for event in read_events(LEDGER_PATH)
                 if event.get("event") == "BUDGET_CATEGORY_AMENDMENT"
                 and event.get("amendment_id") == AMENDMENT_ID]
    if len(amendment) != 1 or canonical_sha256(amendment[0]) != freeze.get("budget_amendment_event_sha256"):
        raise RuntimeError("A02 verifier: budget amendment event differs from freeze")
    return {"checks": checks + 4, "source_hashes_pass": True,
            "input_hashes_pass": True, "synthetic_qa_hash_pass": True,
            "source_commit": freeze["identity"]["source_commit"],
            "protocol_sha256": freeze["protocol_sha256"],
            "freeze_signature": freeze["freeze_signature"],
            "budget_amendment_event_sha256": freeze["budget_amendment_event_sha256"],
            "budget_event_pass": True}


def _read_source_cache_independently(cutoff: int, train: pd.DataFrame) -> dict[str, np.ndarray]:
    path = CACHE_PATHS[cutoff]
    side = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
    if hashlib.sha256(path.read_bytes()).hexdigest() != side.get("npz_sha256"):
        raise RuntimeError(f"A02 verifier: C16 cache byte hash mismatch at {cutoff}")
    def js_sha(obj: Any) -> str:
        return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(",", ":"),
                                        allow_nan=False).encode("utf-8")).hexdigest()
    if (js_sha(side.get("identity")) != side.get("identity_sha256")
            or js_sha({k: v for k, v in side.items() if k != "integrity_sha256"}) != side.get("integrity_sha256")):
        raise RuntimeError(f"A02 verifier: C16 cache sidecar signature mismatch at {cutoff}")
    identity = side.get("identity", {})
    if identity.get("card_id") != "C16" or int(identity.get("cutoff", -1)) != cutoff:
        raise RuntimeError(f"A02 verifier: wrong C16 cutoff identity {cutoff}")
    with np.load(path, allow_pickle=False) as archive:
        rows = archive["score_rows"].astype(np.int64, copy=True)
        ids = archive["score_record_ids"].astype(str, copy=True)
        base = archive["baseline_prediction"].astype(np.float64, copy=True)
        sparse = archive["control__baseline_sparse_raw"].astype(np.float64, copy=True)
        c2b = archive["control__c2b_probability"].astype(np.float64, copy=True)
        cb = archive["control__cb_probability"].astype(np.float64, copy=True)
        mask = archive["control__stream_missing"].astype(bool, copy=True)
        control_rows = archive["control__control_rows"].astype(np.int64, copy=True)
    if (len(np.unique(rows)) != len(rows) or np.any(np.diff(rows) <= 0)
            or np.any(rows >= len(train)) or ids.tolist() != train.record_id.to_numpy()[rows].tolist()
            or not np.array_equal(control_rows, rows)):
        raise RuntimeError(f"A02 verifier: C16 cache row join failed at {cutoff}")
    source_mask = train.signal_stream.fillna("").astype(str).str.strip().eq("").to_numpy()[rows]
    if not np.array_equal(mask, source_mask):
        raise RuntimeError(f"A02 verifier: C16 missing mask differs at {cutoff}")
    independently_routed = np.where(mask, cb, 0.75 * c2b + 0.25 * sparse)
    if not np.array_equal(independently_routed, base):
        raise RuntimeError(f"A02 verifier: C16 baseline_prediction is not sparse25 at {cutoff}")
    cycles = train.cycle_code.to_numpy(dtype=np.int64)[rows]
    if not np.array_equal(np.unique(cycles), np.arange(cutoff + 1, min(cutoff + 12, 79) + 1)):
        raise RuntimeError(f"A02 verifier: C16 cache cycles are incomplete at {cutoff}")
    for cycle in np.unique(cycles):
        if np.sum(cycles == cycle) != np.sum(train.cycle_code.to_numpy() == cycle):
            raise RuntimeError(f"A02 verifier: incomplete source cycle {cycle} at {cutoff}")
    return {"rows": rows, "ids": ids, "cycles": cycles, "baseline": base, "missing": mask}


def _source_bank(origin: int, caches: dict[int, dict[str, np.ndarray]], train: pd.DataFrame) -> dict[str, np.ndarray]:
    blocks = ([(55, 56, 59), (59, 60, 63), (63, 64, 67)] if origin == 67 else
              [(55, 56, 59), (59, 60, 63), (63, 64, 67), (67, 68, 71), (71, 72, 75)])
    parts = []
    for cutoff, low, high in blocks:
        cache = caches[cutoff]
        keep = (cache["cycles"] >= low) & (cache["cycles"] <= high)
        parts.append({key: cache[key][keep] for key in ("rows", "ids", "cycles", "baseline", "missing")})
    joined = {key: np.concatenate([part[key] for part in parts]) for key in parts[0]}
    order = np.argsort(joined["rows"], kind="stable")
    joined = {key: value[order] for key, value in joined.items()}
    if not np.array_equal(joined["rows"], np.flatnonzero(train.cycle_code.between(56, origin).to_numpy())):
        raise RuntimeError(f"A02 verifier: bank rows fail exact source block join for origin {origin}")
    return joined


def _load_arm(origin: int, arm: str, train: pd.DataFrame, bank: dict[str, np.ndarray],
              freeze: dict[str, Any]) -> dict[str, Any]:
    folder = RUN_ROOT / f"E{origin}" / arm
    model_path = folder / "model.joblib"
    state_path = folder / "fit_state.json"
    receipt_path = folder / "fit_receipt.json"
    output_path = folder / "predictions.npz"
    sidecar_path = output_path.with_suffix(".json")
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    state = json.loads(state_path.read_text(encoding="utf-8"))
    sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    if (hashlib.sha256(model_path.read_bytes()).hexdigest() != receipt.get("model_sha256")
            or canonical_sha256(state) != receipt.get("state_sha256")
            or canonical_sha256(state) != canonical_sha256(joblib.load(model_path).audit_state())
            or receipt.get("freeze_signature") != freeze["freeze_signature"]
            or sidecar.get("npz_sha256") != hashlib.sha256(output_path.read_bytes()).hexdigest()
            or sidecar.get("fit_receipt_sha256") != receipt.get("receipt_sha256")
            or sidecar.get("freeze_signature") != freeze["freeze_signature"]):
        raise RuntimeError(f"A02 verifier: durable fit/output receipt mismatch E{origin}/{arm}")
    receipt_copy = dict(receipt)
    receipt_digest = receipt_copy.pop("receipt_sha256", None)
    if receipt_digest != canonical_sha256(receipt_copy):
        raise RuntimeError(f"A02 verifier: fit receipt signature mismatch E{origin}/{arm}")
    expected_fit_rows = int((train.cycle_code.to_numpy(dtype=np.int64) <= origin).sum())
    if (receipt.get("fit_id") != FIT_IDS[(origin, arm)]
            or receipt.get("origin_cycle") != origin
            or receipt.get("residual_arm") != arm
            or receipt.get("text_prefix_rows") != expected_fit_rows
            or receipt.get("query_labels_passed_to_fit") is not False
            or receipt.get("test_data_used") is not False
            or receipt.get("task_estimator_fit_count") != 1):
        raise RuntimeError(f"A02 verifier: fit receipt identity/population mismatch E{origin}/{arm}")

    model = joblib.load(model_path)
    if (model.origin_cycle != origin or model.residual_arm != arm
            or model.bank_row_positions.shape != (int((~bank["missing"]).sum()),)
            or not np.array_equal(model.bank_row_positions, bank["rows"][~bank["missing"]])
            or model.bank_ids != tuple(f"str:{value}" for value in bank["ids"][~bank["missing"]].astype(str))
            or not np.array_equal(model.bank_cycles, bank["cycles"][~bank["missing"]])
            or not np.array_equal(model.bank_oof_probabilities, bank["baseline"][~bank["missing"]])):
        raise RuntimeError(f"A02 verifier: model bank rows/IDs/cycles/probabilities differ E{origin}/{arm}")

    # Independent shuffle-control assignment reconstruction.
    bank_rows = bank["rows"][~bank["missing"]]
    bank_cycles = bank["cycles"][~bank["missing"]]
    bank_prob = bank["baseline"][~bank["missing"]]
    y = train.artifact_signal.to_numpy(dtype=np.int8)
    original_residual = y[bank_rows].astype(np.float64) - bank_prob
    if arm == "observed":
        expected_assignment = np.arange(len(bank_rows), dtype=np.int64)
    else:
        bins = np.minimum(9, np.floor(10.0 * bank_prob).astype(np.int8))
        expected_assignment = np.arange(len(bank_rows), dtype=np.int64)
        rng = np.random.default_rng(42)
        for cycle in np.unique(bank_cycles):
            for decile in range(10):
                indices = np.flatnonzero((bank_cycles == cycle) & (bins == decile))
                if len(indices) > 1:
                    expected_assignment[indices] = rng.permutation(indices)
    if (not np.array_equal(model.residual_assignment, expected_assignment)
            or not np.array_equal(model.bank_residuals, original_residual[expected_assignment])):
        raise RuntimeError(f"A02 verifier: residual assignment does not match frozen arm E{origin}/{arm}")

    with np.load(output_path, allow_pickle=False) as archive:
        out = {key: np.asarray(archive[key]).copy() for key in archive.files}
    required = {"score_rows", "score_record_ids", "score_cycles", "baseline_prediction", "prediction",
                "labels", "missing_mask", "entry_friction", "text_load_group", "neighbor_count",
                "neighbor_source_rows", "neighbor_ids", "neighbor_similarities", "correction",
                "correction_unclipped"}
    if set(out) != required:
        raise RuntimeError(f"A02 verifier: output schema mismatch E{origin}/{arm}")
    rows = out["score_rows"].astype(np.int64)
    expected_rows = np.flatnonzero(
        (train.cycle_code.to_numpy(dtype=np.int64) >= origin + 1)
        & (train.cycle_code.to_numpy(dtype=np.int64) <= min(origin + 12, 79))
    )
    if (rows.ndim != 1 or len(np.unique(rows)) != len(rows) or np.any(np.diff(rows) <= 0)
            or not np.array_equal(rows, expected_rows)
            or out["score_record_ids"].astype(str).tolist() != train.record_id.to_numpy()[rows].tolist()
            or not np.array_equal(out["score_cycles"], train.cycle_code.to_numpy(dtype=np.int16)[rows])
            or not np.array_equal(out["missing_mask"], train.signal_stream.fillna("").astype(str).str.strip().eq("").to_numpy()[rows])
            or not np.array_equal(out["entry_friction"], train.entry_friction.to_numpy(dtype=np.int8)[rows])):
        raise RuntimeError(f"A02 verifier: output rows or feature masks mismatch E{origin}/{arm}")
    if sidecar.get("score_rows") != len(rows) or sidecar.get("score_row_positions_sha256") != hashlib.sha256(rows.astype("<i8").tobytes()).hexdigest():
        raise RuntimeError(f"A02 verifier: output sidecar row digest mismatch E{origin}/{arm}")
    if sidecar.get("score_record_ids_sha256") != _hash_ids(out["score_record_ids"].astype(str).tolist()):
        raise RuntimeError(f"A02 verifier: output sidecar ID digest mismatch E{origin}/{arm}")
    if (sidecar.get("query_labels_passed_to_prediction") is not False
            or sidecar.get("origin_cycle") != origin
            or sidecar.get("residual_arm") != arm):
        raise RuntimeError(f"A02 verifier: output sidecar boundary/identity mismatch E{origin}/{arm}")

    # Independently recompute top-neighbor source identity and residual correction.
    count = out["neighbor_count"].astype(np.int64)
    positions = out["neighbor_source_rows"].astype(np.int64)
    neighbor_ids = out["neighbor_ids"].astype(str)
    sims = out["neighbor_similarities"].astype(np.float64)
    correction = out["correction"].astype(np.float64)
    raw_correction = out["correction_unclipped"].astype(np.float64)
    baseline = out["baseline_prediction"].astype(np.float64)
    prediction = out["prediction"].astype(np.float64)
    missing = out["missing_mask"].astype(bool)
    if (positions.ndim != 2 or neighbor_ids.shape != positions.shape or sims.shape != positions.shape
            or len(count) != len(rows)
            or not np.isfinite(sims).all()
            or np.any(sims < 0.0)):
        raise RuntimeError("A02 verifier: neighbor output array shape/value integrity failed")
    bank_position_to_index = {int(row): index for index, row in enumerate(model.bank_row_positions)}
    for i, n_neighbors in enumerate(count):
        if n_neighbors < 0 or n_neighbors > min(32, len(model.bank_row_positions)):
            raise RuntimeError("A02 verifier: invalid neighbor count")
        valid_rows = positions[i, :n_neighbors]
        valid_sims = sims[i, :n_neighbors]
        valid_ids = neighbor_ids[i, :n_neighbors]
        if n_neighbors:
            if (np.any(valid_sims <= 0.0) or not np.isfinite(valid_sims).all()
                    or len(np.unique(valid_rows)) != n_neighbors
                    or any(int(row) not in bank_position_to_index for row in valid_rows)
                    or np.any(valid_rows >= rows[i])
                    or np.any(model.bank_cycles[[bank_position_to_index[int(row)] for row in valid_rows]] >= out["score_cycles"][i])):
                raise RuntimeError("A02 verifier: neighbor identity violates bank/past boundaries")
            expected_ids = [model.bank_ids[bank_position_to_index[int(row)]] for row in valid_rows]
            if valid_ids.tolist() != expected_ids:
                raise RuntimeError("A02 verifier: neighbor IDs do not match bank row positions")
            order_key = [(-float(sim), int(row)) for sim, row in zip(valid_sims, valid_rows)]
            if order_key != sorted(order_key):
                raise RuntimeError("A02 verifier: neighbors are not sorted by similarity then source row")
            indices = np.array([bank_position_to_index[int(row)] for row in valid_rows], dtype=np.int64)
            weights = valid_sims * valid_sims
            expected_raw = float(np.dot(weights, model.bank_residuals[indices]) / (4.0 + weights.sum()))
        else:
            expected_raw = 0.0
        if (np.any(positions[i, n_neighbors:] != -1)
                or np.any(neighbor_ids[i, n_neighbors:] != "")
                or np.any(sims[i, n_neighbors:] != 0.0)):
            raise RuntimeError("A02 verifier: padding arrays are not canonical")
        if not np.allclose(raw_correction[i], expected_raw, rtol=0.0, atol=1e-15):
            raise RuntimeError("A02 verifier: raw neighbor residual correction mismatch")
        expected_correction = float(np.clip(expected_raw, -0.05, 0.05))
        if missing[i]:
            expected_correction = 0.0
        if correction[i] != expected_correction:
            raise RuntimeError("A02 verifier: clipped/missing-stream correction mismatch")
    expected_prediction = np.clip(baseline + correction, 0.0, 1.0)
    expected_prediction[missing] = baseline[missing]
    if (not np.array_equal(expected_prediction, prediction)
            or not np.array_equal((prediction >= 0.5).astype(np.int8), out["labels"])
            or not np.array_equal(prediction[missing], baseline[missing])):
        raise RuntimeError(f"A02 verifier: candidate output route/threshold mismatch E{origin}/{arm}")

    # Independent fit-defined quartile assignment and no future prefix.
    prefix_mask = train.cycle_code.to_numpy(dtype=np.int64) <= origin
    prefix_text_load = train.loc[prefix_mask, "text_load"].to_numpy(dtype=np.float64)
    edges = np.quantile(prefix_text_load, [0.25, 0.5, 0.75], method="linear")
    if (not np.allclose(edges, sidecar["text_load_quartile_fit_prefix_edges"], rtol=0, atol=0)
            or not np.array_equal(np.searchsorted(edges, train.text_load.to_numpy(dtype=np.float64)[rows], side="right").astype(np.int8),
                                  out["text_load_group"])):
        raise RuntimeError(f"A02 verifier: fit-prefix text_load groups mismatch E{origin}/{arm}")
    return {"rows": rows, "ids": out["score_record_ids"].astype(str),
            "cycles": out["score_cycles"].astype(np.int64), "baseline": baseline,
            "prediction": prediction, "labels": out["labels"].astype(np.int8),
            "shuffle": None, "missing": missing,
            "entry_friction": out["entry_friction"].astype(np.int8),
            "text_load_group": out["text_load_group"].astype(np.int8),
            "neighbor_count": count, "correction": correction,
            "raw_correction": raw_correction, "n": len(rows),
            "neighbor_source_rows": positions, "neighbor_ids": neighbor_ids,
            "neighbor_similarities": sims,
            "fit_receipt": receipt, "state": state, "output_sidecar": sidecar,
            "model": model}


def _select(sample: dict[str, Any], mask: np.ndarray) -> dict[str, Any]:
    return {key: value[mask] for key, value in sample.items()
            if isinstance(value, np.ndarray) and value.ndim == 1 and len(value) == len(sample["rows"])}


def _view(samples: dict[tuple[int, str], dict[str, Any]], train: pd.DataFrame,
          view: str, arm: str | None = None) -> dict[str, np.ndarray]:
    selected: list[dict[str, Any]] = []
    if view == "primary":
        definitions = [(67, 68, 71), (75, 76, 79)]
    elif view == "W4":
        definitions = [(67, 68, 71)]
    elif view == "W6":
        definitions = [(75, 76, 79)]
    elif view == "H12_L":
        definitions = [(67, 68, 79)]
    else:
        raise ValueError(view)
    for origin, low, high in definitions:
        sample = samples[(origin, arm or "observed")]
        mask = (sample["cycles"] >= low) & (sample["cycles"] <= high)
        selected.append(_select(sample, mask))
    merged = {key: np.concatenate([piece[key] for piece in selected])
              for key in selected[0]}
    order = np.argsort(merged["rows"], kind="stable")
    merged = {key: value[order] for key, value in merged.items()}
    if len(np.unique(merged["rows"])) != len(merged["rows"]):
        raise RuntimeError(f"A02 verifier: duplicate row in evaluation view {view}")
    y = train.artifact_signal.to_numpy(dtype=np.int8)[merged["rows"]]
    merged["y"] = y
    return merged


def _score_report(samples: dict[tuple[int, str], dict[str, Any]], train: pd.DataFrame,
                  protocol: dict[str, Any]) -> dict[str, Any]:
    metrics: dict[str, Any] = {}
    view_data: dict[str, dict[str, np.ndarray]] = {}
    for view in ("W4", "W6", "primary", "H12_L"):
        observed = _view(samples, train, view, "observed")
        shuffled = _view(samples, train, view, "shuffled")
        if not np.array_equal(observed["rows"], shuffled["rows"]):
            raise RuntimeError(f"A02 observed/shuffled row sets differ for {view}")
        if not np.array_equal(observed["baseline"], shuffled["baseline"]):
            raise RuntimeError(f"A02 observed/shuffled baseline differs for {view}")
        base_metric = _metric(observed["y"], observed["baseline"])
        observed_metric = _metric(observed["y"], observed["prediction"])
        shuffled_metric = _metric(shuffled["y"], shuffled["prediction"])
        metrics[view] = {
            "n": len(observed["rows"]),
            "sparse25": base_metric,
            "observed": observed_metric,
            "shuffled": shuffled_metric,
            "observed_gain_vs_sparse25": observed_metric["macro_f1"] - base_metric["macro_f1"],
            "observed_gain_vs_shuffled": observed_metric["macro_f1"] - shuffled_metric["macro_f1"],
            "shuffled_gain_vs_sparse25": shuffled_metric["macro_f1"] - base_metric["macro_f1"],
            "observed_harm_vs_sparse25": base_metric["macro_f1"] - observed_metric["macro_f1"],
        }
        view_data[view] = observed

    per_cycle: list[dict[str, Any]] = []
    for origin in (67, 75):
        observed = samples[(origin, "observed")]
        shuffled = samples[(origin, "shuffled")]
        for cycle in np.unique(observed["cycles"]):
            mask = observed["cycles"] == cycle
            shuffle_mask = shuffled["cycles"] == cycle
            y = train.artifact_signal.to_numpy(dtype=np.int8)[observed["rows"][mask]]
            base_m = _metric(y, observed["baseline"][mask])
            obs_m = _metric(y, observed["prediction"][mask])
            shuf_m = _metric(y, shuffled["prediction"][shuffle_mask])
            per_cycle.append({"origin": origin, "cycle": int(cycle), "rows": int(mask.sum()),
                              "sparse25_macro_f1": base_m["macro_f1"],
                              "observed_macro_f1": obs_m["macro_f1"],
                              "shuffled_macro_f1": shuf_m["macro_f1"],
                              "observed_gain_vs_sparse25": obs_m["macro_f1"] - base_m["macro_f1"],
                              "observed_gain_vs_shuffled": obs_m["macro_f1"] - shuf_m["macro_f1"]})

    subgroup_report: dict[str, Any] = {}
    risk_flags: list[dict[str, Any]] = []
    for view, sample in view_data.items():
        if view == "primary":
            shuffled = _view(samples, train, "primary", "shuffled")
        else:
            shuffled = _view(samples, train, view, "shuffled")
        subgroup_report[view] = {}
        fields = {
            "stream_availability": np.where(sample["missing"], "missing", "present"),
            "entry_friction": sample["entry_friction"].astype(str),
            "text_load_quartile": np.array([f"Q{value + 1}" for value in sample["text_load_group"]]),
        }
        for field, groups in fields.items():
            subgroup_report[view][field] = {}
            for label in sorted(set(groups.tolist())):
                mask = groups == label
                ys = sample["y"][mask]
                class0 = int(np.sum(ys == 0))
                class1 = int(np.sum(ys == 1))
                supported = int(mask.sum()) >= 200 and class0 >= 30 and class1 >= 30
                group_value: dict[str, Any] = {
                    "rows": int(mask.sum()), "class_0_rows": class0, "class_1_rows": class1,
                    "supported": bool(supported),
                }
                if supported:
                    base_m = _metric(ys, sample["baseline"][mask])
                    candidate_m = _metric(ys, sample["prediction"][mask])
                    shuffle_m = _metric(ys, shuffled["prediction"][mask])
                    f1_harm = base_m["macro_f1"] - candidate_m["macro_f1"]
                    recall_harm = base_m["positive_recall"] - candidate_m["positive_recall"]
                    group_value.update({
                        "sparse25": base_m,
                        "observed": candidate_m,
                        "shuffled": shuffle_m,
                        "observed_f1_harm_vs_sparse25": f1_harm,
                        "observed_f1_gain_vs_shuffled": candidate_m["macro_f1"] - shuffle_m["macro_f1"],
                        "observed_positive_recall_harm_vs_sparse25": recall_harm,
                        "historical_f1_harm_flags": {str(limit): bool(f1_harm > limit) for limit in (0.01, 0.02, 0.03)},
                        "historical_recall_harm_flags": {str(limit): bool(recall_harm > limit) for limit in (0.02, 0.03, 0.05)},
                    })
                    if f1_harm > 0.03 or recall_harm > 0.05:
                        risk_flags.append({"view": view, "field": field, "group": label,
                                           "macro_f1_harm": f1_harm,
                                           "positive_recall_harm": recall_harm})
                else:
                    group_value.update({"sparse25": None, "observed": None, "shuffled": None,
                                        "observed_f1_harm_vs_sparse25": None,
                                        "observed_positive_recall_harm_vs_sparse25": None,
                                        "historical_f1_harm_flags": None,
                                        "historical_recall_harm_flags": None})
                subgroup_report[view][field][str(label)] = group_value

    gates = protocol["scientific_gate"]
    h12 = metrics["H12_L"]
    primary = metrics["primary"]
    checks = {
        "H12_L_gain_vs_sparse25": h12["observed_gain_vs_sparse25"] >= gates["H12_L_gain_vs_sparse25_at_least"],
        "H12_L_gain_vs_shuffled": h12["observed_gain_vs_shuffled"] >= gates["H12_L_gain_vs_shuffled_at_least"],
        "primary_gain_vs_sparse25": primary["observed_gain_vs_sparse25"] >= gates["primary_gain_vs_sparse25_at_least"],
        "primary_gain_vs_shuffled": primary["observed_gain_vs_shuffled"] >= gates["primary_gain_vs_shuffled_at_least"],
        "W4_harm_limit": metrics["W4"]["observed_harm_vs_sparse25"] <= gates["W4_and_W6_max_harm_vs_sparse25"],
        "W6_harm_limit": metrics["W6"]["observed_harm_vs_sparse25"] <= gates["W4_and_W6_max_harm_vs_sparse25"],
        "H12_L_harm_limit": h12["observed_harm_vs_sparse25"] <= gates["H12_L_max_harm_vs_sparse25"],
    }
    scientific_pass = all(checks.values())
    return {
        "candidate_id": "A02_LEXICAL_NEIGHBOR_RESIDUAL",
        "status": "RESULTS_RECONSTRUCTED_BY_INDEPENDENT_VERIFIER",
        "metrics": metrics,
        "per_cycle": per_cycle,
        "subgroups": subgroup_report,
        "gate_checks": checks,
        "scientific_verdict": "PILOT_PASS" if scientific_pass else "NO_GO_GATE_FAILED",
        "risk_review_status": "RISK_REVIEW_HOLD" if risk_flags else "NO_RISK_HOLD_TRIGGERED",
        "risk_flags": risk_flags,
        "neighbor_diagnostics": {
            f"E{origin}_{arm}": {
                "query_rows": int(sample["n"]),
                "missing_rows": int(sample["missing"].sum()),
                "zero_neighbor_rows": int(np.sum(sample["neighbor_count"] == 0)),
                "positive_neighbor_rows": int(np.sum(sample["neighbor_count"] > 0)),
                "mean_neighbor_count": float(sample["neighbor_count"].mean()),
                "nonzero_correction_rows": int(np.sum(sample["correction"] != 0.0)),
                "clip_rows": int(np.sum(np.abs(sample["raw_correction"]) > 0.05)),
                "correction_min": float(sample["correction"].min()),
                "correction_max": float(sample["correction"].max()),
            }
            for (origin, arm), sample in sorted(samples.items())
        },
        "interpretation": {
            "estimand": "incremental association/predictive signal in past-only lexical-neighbor residual assignments versus frozen sparse25 and matched shuffled assignment",
            "not_causal": True,
            "H12_L_is_not_independent_of_primary": True,
            "A02_is_pilot_only_no_continuation": True,
            "R06_or_Kaggle_used_for_recipe_selection": False,
        },
    }


def verify(*, write_outputs: bool = True) -> dict[str, Any]:
    protocol_path = ROOT / "experiments/optimization_v3/c19_a02_v1/A02/protocol.json"
    protocol = _read_json(protocol_path)
    freeze = _read_json(FREEZE_PATH)
    hashes = _check_frozen_inputs(freeze, protocol)
    if file_sha256(TRAIN_PATH) != EXPECTED_TRAIN_SHA256:
        raise RuntimeError("A02 official Train SHA-256 differs")
    train = pd.read_csv(TRAIN_PATH,
                        usecols=["record_id", "cycle_code", "signal_stream", "entry_friction", "text_load", "artifact_signal"],
                        dtype={"record_id": str})
    if len(train) != EXPECTED_ROWS or not train.record_id.is_unique:
        raise RuntimeError("A02 verifier: Train ID/row count mismatch")
    caches = {cutoff: _read_source_cache_independently(cutoff, train) for cutoff in sorted(CACHE_PATHS)}
    banks = {origin: _source_bank(origin, caches, train) for origin in (67, 75)}
    samples = {}
    for origin in (67, 75):
        for arm in ("observed", "shuffled"):
            samples[(origin, arm)] = _load_arm(origin, arm, train, banks[origin], freeze)
        observed = samples[(origin, "observed")]
        shuffled = samples[(origin, "shuffled")]
        for key in ("neighbor_count", "neighbor_source_rows", "neighbor_ids", "neighbor_similarities"):
            if not np.array_equal(observed[key], shuffled[key]):
                raise RuntimeError(f"A02 verifier: observed/shuffled neighbor geometry differs E{origin}/{key}")

    ledger = read_events(LEDGER_PATH)
    starts = [event for event in ledger if event.get("event") == "FIT_STARTED" and event.get("card_id") == "CLASSIFICATION_A02_NEIGHBOR_RESIDUAL"]
    terminal = [event for event in ledger if event.get("event") in {"FIT_SAVED", "FIT_FAILED"}
                and event.get("card_id") == "CLASSIFICATION_A02_NEIGHBOR_RESIDUAL"]
    started_ids = [event.get("fit_id") for event in starts]
    saved_ids = [event.get("fit_id") for event in terminal if event.get("event") == "FIT_SAVED"]
    failed_ids = [event.get("fit_id") for event in terminal if event.get("event") == "FIT_FAILED"]
    expected_ids = sorted(FIT_IDS.values())
    if sorted(started_ids) != expected_ids or sorted(saved_ids) != expected_ids or failed_ids:
        raise RuntimeError("A02 verifier: exactly four starts/saves and zero failures required")
    receipts = {sample["fit_receipt"]["fit_id"]: sample["fit_receipt"]
                for sample in samples.values()}
    for fit_id in expected_ids:
        fit_starts = [event for event in starts if event.get("fit_id") == fit_id]
        fit_saves = [event for event in terminal if event.get("fit_id") == fit_id
                     and event.get("event") == "FIT_SAVED"]
        expected_key = "CLASSIFICATION_A02_NEIGHBOR_RESIDUAL:A02_FOUR_FIT_PILOT"
        if (len(fit_starts) != 1 or len(fit_saves) != 1
                or fit_starts[0].get("reservation_key") != expected_key
                or fit_starts[0].get("stage") != "A02_FOUR_FIT_PILOT"
                or fit_saves[0].get("reservation_key") != expected_key
                or fit_saves[0].get("stage") != "A02_FOUR_FIT_PILOT"
                or fit_saves[0].get("receipt_sha256") != receipts[fit_id]["receipt_sha256"]):
            raise RuntimeError(f"A02 verifier: start/save receipt lineage mismatch for {fit_id}")
    reservation = [event for event in ledger if event.get("event") == "CARD_FITS_RESERVED"
                   and event.get("card_id") == "CLASSIFICATION_A02_NEIGHBOR_RESIDUAL"
                   and event.get("stage") == "A02_FOUR_FIT_PILOT"]
    if (len(reservation) != 1 or reservation[0].get("fits") != 4
            or reservation[0].get("amendment_id") != AMENDMENT_ID
            or reservation[0].get("freeze_signature") != freeze["freeze_signature"]):
        raise RuntimeError("A02 verifier: four-fit amended reservation mismatch")

    result = _score_report(samples, train, protocol)
    result["provenance"] = {
        "source_commit": freeze["identity"]["source_commit"],
        "protocol_sha256": freeze["protocol_sha256"],
        "freeze_signature": freeze["freeze_signature"],
        "source_sha256": freeze["source_sha256"],
        "input_sha256": freeze["input_sha256"],
        "synthetic_qa_sha256": freeze["identity"].get("synthetic_qa_sha256", {}),
        "budget_amendment_id": AMENDMENT_ID,
        "budget_amendment_event_sha256": freeze["budget_amendment_event_sha256"],
        "reservation": reservation[0],
        "fit_receipts": {fit_id: receipts[fit_id]["receipt_sha256"] for fit_id in expected_ids},
        "actual_task_fit_count": len(started_ids),
        "saved_task_fit_count": len(saved_ids),
        "fit_failure_count": len(failed_ids),
        "test_or_sample_used": False,
    }
    metrics_path = REPORT_ROOT / "metrics_reconstructed.json"
    encoded_metrics = json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    if write_outputs:
        if metrics_path.exists():
            if metrics_path.read_text(encoding="utf-8") != encoded_metrics:
                raise RuntimeError("A02 reconstructed metrics exist with different contents")
        else:
            metrics_path.parent.mkdir(parents=True, exist_ok=True)
            with metrics_path.open("x", encoding="utf-8", newline="\n") as stream:
                stream.write(encoded_metrics)
    checks = {
        **hashes,
        "official_train_rows": len(train),
        "cache_cutoffs_independently_joined": list(sorted(caches)),
        "baseline_route_bit_exact_for_all_six_caches": True,
        "bank_rows_E67": len(banks[67]["rows"]),
        "bank_rows_E75": len(banks[75]["rows"]),
        "four_task_fits_started_and_saved": 4,
        "fit_failures": 0,
        "four_output_files_validated": 4,
        "neighbor_math_and_self_future_guards": "PASS",
        "shuffle_assignment_reconstructed": "PASS",
        "independent_metric_and_subgroup_reconstruction": "PASS",
        "test_or_sample_access": False,
        "source_boundaries_pass": True,
        "scientific_verdict": result["scientific_verdict"],
    }
    verification = {"status": "PASS", "checks": checks,
                    "metrics_path": "reports/optimization_v3/a02_neighbors_v1/metrics_reconstructed.json",
                    "candidate_id": "A02_LEXICAL_NEIGHBOR_RESIDUAL"}
    verification_path = REPORT_ROOT / "verification.json"
    verification_encoded = json.dumps(verification, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    if write_outputs:
        if verification_path.exists():
            if verification_path.read_text(encoding="utf-8") != verification_encoded:
                raise RuntimeError("A02 verification exists with different contents")
        else:
            verification_path.parent.mkdir(parents=True, exist_ok=True)
            with verification_path.open("x", encoding="utf-8", newline="\n") as stream:
                stream.write(verification_encoded)
    return verification


if __name__ == "__main__":
    print(json.dumps(verify(write_outputs=True), indent=2, ensure_ascii=False))
