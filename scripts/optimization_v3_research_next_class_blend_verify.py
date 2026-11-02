"""Independent, zero-fit verification for the frozen Research Next class blend.

This script reads only the official Classification Train, the immutable N01/N02
pilot bundles named in the fixed review protocol, and the root evaluator outputs.
It intentionally does not import either experiment runner or its metric/gate code.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "experiments/optimization_v3/research_next_v1/review_class_blend/protocol.json"
RESULTS = ROOT / "reports/optimization_v3/research_next_v1/results/CLASS_BLEND_REVIEW/amendment_01/results.json"
GROUPS = ROOT / "reports/optimization_v3/research_next_v1/results/CLASS_BLEND_REVIEW/amendment_01/groups.csv"
DEFAULT_OUTPUT = ROOT / "reports/optimization_v3/research_next_v1/results/CLASS_BLEND_REVIEW/amendment_01/independent_verification.json"


class VerificationError(RuntimeError):
    """Raised when an immutable input or evaluator output fails verification."""


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
                      allow_nan=False).encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise VerificationError(f"Cannot read valid JSON at {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise VerificationError(f"Expected JSON object at {path}")
    return value


def _expect(condition: bool, message: str) -> None:
    if not condition:
        raise VerificationError(message)


def _close(a: float, b: float, *, atol: float = 2e-12) -> bool:
    return math.isfinite(float(a)) and math.isfinite(float(b)) and abs(float(a) - float(b)) <= atol


def _binary_metrics(y: np.ndarray, probability: np.ndarray, threshold: float) -> dict[str, Any]:
    y = np.asarray(y, dtype=np.int8)
    p = (np.asarray(probability, dtype=np.float64) >= threshold).astype(np.int8)
    _expect(y.ndim == p.ndim == 1 and len(y) == len(p), "Metric vectors do not align")
    _expect(np.isin(y, (0, 1)).all() and np.isin(p, (0, 1)).all(), "Metric inputs must be binary")
    tn = int(np.sum((y == 0) & (p == 0)))
    fp = int(np.sum((y == 0) & (p == 1)))
    fn = int(np.sum((y == 1) & (p == 0)))
    tp = int(np.sum((y == 1) & (p == 1)))
    f1_0 = 2 * tn / (2 * tn + fp + fn) if 2 * tn + fp + fn else 0.0
    f1_1 = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0
    return {
        "n": int(len(y)), "n0": int(np.sum(y == 0)), "n1": int(np.sum(y == 1)),
        "tn": tn, "fp": fp, "fn": fn, "tp": tp,
        "macro_f1": float((f1_0 + f1_1) / 2),
        "positive_recall": float(tp / (tp + fn)) if tp + fn else 0.0,
    }


def _load_protocol_once(path: Path) -> dict[str, Any]:
    """Load the frozen review protocol exactly once for this run."""
    protocol = _read_json(path)
    _expect(protocol.get("id") == "CLASS_BLEND_REVIEW_V1", "Unexpected review protocol id")
    _expect(protocol.get("status") == "FIXED_BEFORE_COMPUTING_NEW_BLEND_OUTCOMES",
            "Protocol was not fixed before evaluating this blend")
    _expect(protocol.get("official_new_fits") == 0, "Protocol does not prohibit new estimator fits")
    _expect(protocol.get("source_head") == "6139081307433beac43d5da0a04b94ef518d29a6",
            "Protocol source HEAD differs from the reviewed snapshot")
    recipe = protocol.get("recipe", {})
    _expect(recipe.get("candidate") == ".5 * routed N02 sketch + .5 * routed N01 class_only",
            "Candidate recipe differs from the fixed protocol")
    _expect(recipe.get("matched_control") == ".5 * routed N02 no_sketch + .5 * routed N01 class_only",
            "Matched-control recipe differs from the fixed protocol")
    _expect(float(recipe.get("threshold", -1)) == 0.5, "Decision threshold is not frozen at 0.50")
    _expect(recipe.get("stream_missing") == "exact same frozen CB fallback; verify both components identical",
            "Missing-stream routing contract differs from protocol")
    forbidden = set(recipe.get("forbidden", []))
    _expect({"weight search", "threshold search", "new fitting", "Test or Kaggle access",
             "changing gates after evaluation"} <= forbidden,
            "Protocol does not forbid all post hoc searches and data access")
    return protocol


def _validate_manifest(protocol: Mapping[str, Any]) -> dict[str, str]:
    manifest = protocol.get("input_sha256")
    _expect(isinstance(manifest, dict) and manifest, "Protocol input hash manifest is missing")
    actual: dict[str, str] = {}
    for relative, expected in manifest.items():
        path = (ROOT / Path(str(relative))).resolve()
        _expect(path.is_file(), f"Frozen input is missing: {relative}")
        digest = _file_sha256(path)
        _expect(digest == expected, f"Frozen input hash mismatch: {relative}")
        actual[str(relative)] = digest

    queue = _read_json(ROOT / "experiments/optimization_v3/research_next_v1/queue.json")
    _expect(queue.get("batch_id") == "optimization_v3_research_next_v1", "Research Next queue batch mismatch")
    for rel in (
        "experiments/optimization_v3/research_next_v1/frozen/N01/freeze.json",
        "experiments/optimization_v3/research_next_v1/frozen/N02/amendment_02/freeze.json",
    ):
        freeze = _read_json(ROOT / rel)
        identity = freeze.get("identity")
        _expect(isinstance(identity, dict), f"Frozen identity absent: {rel}")
        _expect(freeze.get("freeze_signature") == _sha256_bytes(_canonical(identity)),
                f"Frozen identity signature mismatch: {rel}")
        _expect(identity.get("batch_id") == "optimization_v3_research_next_v1",
                f"Wrong campaign in frozen identity: {rel}")
    ledger = protocol.get("budget_ledger")
    ledger_hash = protocol.get("ledger_sha256_before")
    _expect(isinstance(ledger, str) and isinstance(ledger_hash, str), "Protocol ledger identity is missing")
    ledger_path = (ROOT / ledger).resolve()
    _expect(ledger_path.is_file() and _file_sha256(ledger_path) == ledger_hash,
            "Research Next fit ledger changed after the blend review protocol was frozen")
    return actual


def _load_bundle(protocol: Mapping[str, Any], *, card: str, cutoff: int,
                 fit_kind: str, freeze: Mapping[str, Any]) -> dict[str, np.ndarray]:
    rel = (f"experiments/runs/optimization_v3_research_next_v1/{card}/pilot/"
           f"cutoff_{cutoff}/{fit_kind}")
    npz_rel, receipt_rel = f"{rel}/arrays.npz", f"{rel}/receipt.json"
    manifest = protocol["input_sha256"]
    npz_path, receipt_path = ROOT / npz_rel, ROOT / receipt_rel
    _expect(_file_sha256(npz_path) == manifest[npz_rel], f"Protocol NPZ hash mismatch: {npz_rel}")
    _expect(_file_sha256(receipt_path) == manifest[receipt_rel], f"Protocol receipt hash mismatch: {receipt_rel}")
    receipt = _read_json(receipt_path)
    integrity = {key: value for key, value in receipt.items() if key != "receipt_sha256"}
    _expect(receipt.get("receipt_sha256") == _sha256_bytes(_canonical(integrity)),
            f"Receipt integrity failure: {receipt_rel}")
    _expect(receipt.get("schema_version") == 1 and receipt.get("task") == "classification",
            f"Receipt task/schema failure: {receipt_rel}")
    identity = {
        "batch_id": "optimization_v3_research_next_v1", "card_id": card,
        "cutoff": int(cutoff), "fit_kind": fit_kind,
        "freeze_signature": freeze["freeze_signature"], "stage": "PILOT",
    }
    _expect(receipt.get("identity") == identity, f"Bundle identity mismatch: {receipt_rel}")
    _expect(receipt.get("identity_sha256") == _sha256_bytes(_canonical(identity)),
            f"Bundle identity hash mismatch: {receipt_rel}")
    _expect(receipt.get("npz_sha256") == _file_sha256(npz_path), f"NPZ receipt hash mismatch: {npz_rel}")
    metadata_outer = receipt.get("metadata", {})
    _expect(isinstance(metadata_outer, dict) and isinstance(metadata_outer.get("metadata"), dict),
            f"Bundle metadata envelope invalid: {receipt_rel}")
    _expect(receipt.get("metadata_sha256") == _sha256_bytes(_canonical(metadata_outer)),
            f"Bundle metadata hash mismatch: {receipt_rel}")
    metadata = metadata_outer["metadata"]
    _expect(metadata.get("task") == "classification" and metadata.get("cutoff") == cutoff
            and metadata.get("fit_kind") == fit_kind, f"Bundle fit metadata mismatch: {receipt_rel}")
    _expect(metadata.get("test_data_used") is False and metadata.get("sample_submission_used") is False
            and metadata.get("kaggle_calls") == 0 and metadata.get("score_labels_passed_to_fit") is False,
            f"Bundle provenance flags fail: {receipt_rel}")
    if card == "N02":
        fit_state = metadata.get("fit_state", {})
        _expect(fit_state.get("validation_rows_or_labels_passed_to_fit") is False,
                f"N02 fit received validation data: {receipt_rel}")
        _expect(fit_state.get("preprocessing_fit_rows_only") is True,
                f"N02 preprocessing was not fit-only: {receipt_rel}")
    with np.load(npz_path, allow_pickle=False) as archive:
        arrays = {name: archive[name].copy() for name in archive.files}
    _expect(sorted(arrays) == receipt.get("array_names"), f"NPZ names differ from receipt: {npz_rel}")
    required = {"score_rows", "score_record_ids", "cycle_code", "origin_cutoff", "prediction",
                "baseline_prediction", "raw_prediction", "stream_missing"}
    _expect(required <= set(arrays), f"Required arrays absent: {npz_rel}")
    n = len(arrays["score_rows"])
    _expect(n > 0 and all(arrays[key].shape == (n,) for key in required),
            f"Bundle arrays do not align: {npz_rel}")
    _expect(arrays["score_rows"].dtype.kind in "iu" and arrays["score_record_ids"].dtype.kind == "U",
            f"Bundle row identity dtypes invalid: {npz_rel}")
    _expect(len(np.unique(arrays["score_rows"])) == n and len(np.unique(arrays["score_record_ids"])) == n,
            f"Bundle row identities are not unique: {npz_rel}")
    for key in ("prediction", "baseline_prediction", "raw_prediction"):
        values = np.asarray(arrays[key], dtype=np.float64)
        _expect(np.isfinite(values).all() and ((values >= 0) & (values <= 1)).all(),
                f"Invalid probability array {key}: {npz_rel}")
    _expect(np.asarray(arrays["origin_cutoff"]).shape == (n,)
            and np.all(arrays["origin_cutoff"] == cutoff), f"Origin vector mismatch: {npz_rel}")
    return arrays


def _read_train(path: Path, expected_hash: str) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
    _expect(path.is_file() and _file_sha256(path) == expected_hash,
            "Official Classification Train hash differs from the protocol")
    frame = pd.read_csv(path, usecols=["record_id", "cycle_code", "artifact_signal",
                                      "entry_friction", "signal_stream", "text_load"])
    _expect(len(frame) == 50_608, "Official Classification Train row count is not 50,608")
    _expect(frame["record_id"].notna().all() and frame["record_id"].astype(str).is_unique,
            "Official Classification Train IDs are missing or duplicated")
    cycles = pd.to_numeric(frame["cycle_code"], errors="raise").to_numpy(dtype=np.int64)
    _expect(np.array_equal(np.unique(cycles), np.arange(47, 80)), "Train cycles are not exactly 47..79")
    raw_target = pd.to_numeric(frame["artifact_signal"], errors="raise").to_numpy()
    _expect(np.isin(raw_target, (0, 1)).all(), "Train target is not binary")
    target = raw_target.astype(np.int8)
    stream = frame["signal_stream"].to_numpy(dtype=object)
    missing = np.zeros(len(frame), dtype=bool)
    for i, value in enumerate(stream):
        is_na = pd.isna(value)
        text = "" if is_na else str(value).strip()
        missing[i] = bool(is_na) or not bool(text)
    return frame, target, cycles, missing


def _route_checks(bundles: Mapping[tuple[str, int, str], Mapping[str, np.ndarray]],
                  cutoffs: Sequence[int]) -> dict[int, dict[str, np.ndarray]]:
    result: dict[int, dict[str, np.ndarray]] = {}
    for cutoff in cutoffs:
        n01 = bundles[("N01", cutoff, "class_only")]
        sketch = bundles[("N02", cutoff, "sketch")]
        no_sketch = bundles[("N02", cutoff, "no_sketch")]
        for other, arrays in (("N02 sketch", sketch), ("N02 no_sketch", no_sketch)):
            for key in ("score_rows", "score_record_ids", "cycle_code", "origin_cutoff", "stream_missing"):
                _expect(np.array_equal(n01[key], arrays[key]),
                        f"N01 class-only and {other} row/route alignment differs at cutoff {cutoff}: {key}")
        missing = n01["stream_missing"].astype(bool)
        _expect("control__cb_probability" in n01 and "control__cb_probability" in sketch
                and "control__cb_probability" in no_sketch, f"CB control is absent at cutoff {cutoff}")
        cb = np.asarray(n01["control__cb_probability"], dtype=np.float64)
        _expect(np.array_equal(cb, np.asarray(sketch["control__cb_probability"], dtype=np.float64))
                and np.array_equal(cb, np.asarray(no_sketch["control__cb_probability"], dtype=np.float64)),
                f"Missing-stream CB fallback differs across components at cutoff {cutoff}")
        _expect(np.array_equal(missing, np.asarray(sketch["stream_missing"], dtype=bool))
                and np.array_equal(missing, np.asarray(no_sketch["stream_missing"], dtype=bool)),
                f"Missing-stream flags differ at cutoff {cutoff}")

        c2b = np.asarray(n01["control__c2b_probability"], dtype=np.float64)
        sparse_raw = np.asarray(n01["control__sparse_baseline_raw"], dtype=np.float64)
        sparse25 = np.where(missing, cb, 0.75 * c2b + 0.25 * sparse_raw)
        n01_class_only = np.where(missing, cb, 0.75 * c2b + 0.25 * n01["raw_prediction"])
        _expect(np.allclose(n01["baseline_prediction"], sparse25, rtol=0, atol=2e-12),
                f"N01 sparse25 baseline does not reconstruct at cutoff {cutoff}")
        _expect(np.allclose(n01["prediction"], n01_class_only, rtol=0, atol=2e-12),
                f"N01 class-only routed probability does not reconstruct at cutoff {cutoff}")

        current = np.asarray(sketch["control__current_route"], dtype=np.float64)
        _expect(np.allclose(current, sparse25, rtol=0, atol=2e-12)
                and np.allclose(no_sketch["control__current_route"], sparse25, rtol=0, atol=2e-12),
                f"N02 frozen current route does not equal C16 sparse25 at cutoff {cutoff}")
        expected_sketch = np.where(missing, cb, 0.75 * current + 0.25 * sketch["raw_prediction"])
        expected_no_sketch = np.where(missing, cb, 0.75 * current + 0.25 * no_sketch["raw_prediction"])
        _expect(np.allclose(sketch["prediction"], expected_sketch, rtol=0, atol=2e-12),
                f"N02 sketch probability does not reconstruct at cutoff {cutoff}")
        _expect(np.allclose(no_sketch["prediction"], expected_no_sketch, rtol=0, atol=2e-12),
                f"N02 no-sketch probability does not reconstruct at cutoff {cutoff}")
        candidate = np.where(missing, cb, 0.5 * (sketch["prediction"] + n01["prediction"]))
        matched = np.where(missing, cb, 0.5 * (no_sketch["prediction"] + n01["prediction"]))
        result[int(cutoff)] = {"rows": n01["score_rows"].astype(np.int64),
                               "candidate": candidate, "matched_combo": matched,
                               "sparse25": sparse25, "N01_class_only": n01_class_only,
                               "N02_sketch": np.asarray(sketch["prediction"], dtype=np.float64),
                               "N02_no_sketch": np.asarray(no_sketch["prediction"], dtype=np.float64),
                               "missing": missing, "cycles": n01["cycle_code"].astype(np.int64)}
    return result


def _view_rows(protocol: Mapping[str, Any], cycles: np.ndarray) -> dict[str, dict[int, np.ndarray]]:
    result: dict[str, dict[int, np.ndarray]] = {}
    for view, blocks in protocol["views"].items():
        by_origin: dict[int, np.ndarray] = {}
        for cutoff, lo, hi in blocks:
            mask = (cycles >= int(lo)) & (cycles <= int(hi))
            rows = np.flatnonzero(mask).astype(np.int64)
            _expect(len(rows) > 0 and int(cycles[rows].min()) == int(lo)
                    and int(cycles[rows].max()) == int(hi), f"Invalid cycle block in view {view}")
            _expect(int(cutoff) not in by_origin, f"Repeated origin block in view {view}")
            by_origin[int(cutoff)] = rows
        result[view] = by_origin
    return result


def _text_bins(frame: pd.DataFrame, cycles: np.ndarray, cutoff: int) -> tuple[np.ndarray, list[float]]:
    raw = frame["text_load"]
    missing_raw = raw.map(lambda value: isinstance(value, str) and not value.strip()).to_numpy(dtype=bool)
    parsed = pd.to_numeric(raw.mask(missing_raw), errors="coerce").to_numpy(dtype=np.float64)
    finite_fit = parsed[(cycles <= cutoff) & np.isfinite(parsed)]
    bins = np.full(len(frame), -1, dtype=np.int64)
    if len(finite_fit) == 0:
        return bins, []
    edges = np.unique(np.quantile(finite_fit, [0.25, 0.50, 0.75]))
    finite = np.isfinite(parsed)
    bins[finite] = np.searchsorted(edges, parsed[finite], side="right").astype(np.int64)
    return bins, [float(edge) for edge in edges]


def _independent_groups(protocol: Mapping[str, Any], frame: pd.DataFrame, target: np.ndarray,
                        cycles: np.ndarray, missing: np.ndarray,
                        predictions: Mapping[int, Mapping[str, np.ndarray]]) -> list[dict[str, Any]]:
    thresholds = protocol["gates"]
    minimum_n = int(thresholds["minimum_group_n"])
    minimum_class = int(thresholds["minimum_each_class_n"])
    f1_limit = float(thresholds["subgroup_f1_harm_max"])
    recall_limit = float(thresholds["subgroup_positive_recall_harm_max"])
    views = _view_rows(protocol, cycles)
    bins_by_cutoff = {cutoff: _text_bins(frame, cycles, cutoff)
                      for cutoff in sorted({int(block[0]) for blocks in protocol["views"].values() for block in blocks})}
    # Gather prediction vectors in source-row coordinates. Every official score row must map exactly once per origin.
    row_predictions: dict[int, dict[str, dict[int, float]]] = {}
    for cutoff, info in predictions.items():
        for arm in ("candidate", "matched_combo", "N01_class_only", "N02_sketch", "sparse25"):
            row_predictions.setdefault(cutoff, {}).setdefault(arm, {})
            for row, value in zip(info["rows"], info[arm]):
                row_predictions[cutoff][arm][int(row)] = float(value)

    output: list[dict[str, Any]] = []
    for view, origins in views.items():
        rows_by_origin: dict[int, np.ndarray] = {}
        for cutoff, expected_rows in origins.items():
            source_set = set(expected_rows.tolist())
            score_lookup = predictions[cutoff]["rows"].tolist()
            positions = [pos for pos, row in enumerate(score_lookup) if row in source_set]
            actual_rows = predictions[cutoff]["rows"][positions]
            _expect(set(actual_rows.tolist()) == source_set and len(actual_rows) == len(source_set),
                    f"Score coverage does not match source cycle block: {view}/{cutoff}")
            rows_by_origin[cutoff] = actual_rows
        flat_rows = np.concatenate(list(rows_by_origin.values())) if rows_by_origin else np.asarray([], dtype=np.int64)
        _expect(len(flat_rows) == len(np.unique(flat_rows)), f"Cycle blocks overlap in view {view}")
        _expect(len(flat_rows) == int(protocol["view_rows"][view]), f"Unexpected denominator for view {view}")

        pooled_labels: dict[str, dict[Any, list[tuple[int, int]]]] = {
            "entry_friction": {}, "stream_availability": {}, "fit_defined_text_load_quartile": {}}
        for cutoff, rows in rows_by_origin.items():
            ef = frame["entry_friction"].astype(str).to_numpy(dtype=object)[rows]
            availability = np.where(missing[rows], "missing", "present")
            bins, edges = bins_by_cutoff[cutoff]
            text_group = bins[rows]
            for local_pos, row in enumerate(rows):
                pooled_labels["entry_friction"].setdefault(str(ef[local_pos]), []).append((cutoff, int(row)))
                pooled_labels["stream_availability"].setdefault(str(availability[local_pos]), []).append((cutoff, int(row)))
                # Each row uses its own origin's fit-only edges; pooled groups compare relative quartile labels.
                group_label = str(int(text_group[local_pos]))
                pooled_labels["fit_defined_text_load_quartile"].setdefault(group_label, []).append((cutoff, int(row)))

        for field, label_map in pooled_labels.items():
            for group_label, pairs in label_map.items():
                group_rows = np.asarray([row for _cutoff, row in pairs], dtype=np.int64)
                group_origins = np.asarray([cutoff for cutoff, _row in pairs], dtype=np.int64)
                y = target[group_rows]
                n0, n1 = int(np.sum(y == 0)), int(np.sum(y == 1))
                supported = len(group_rows) >= minimum_n and min(n0, n1) >= minimum_class
                arms = ("candidate", "matched_combo", "N01_class_only", "N02_sketch", "sparse25")
                metrics: dict[str, dict[str, Any]] = {}
                # Preserve origin-specific predictions/edges in pooled groups while aggregating for non-text groups.
                for arm in arms:
                    probs = np.empty(len(group_rows), dtype=np.float64)
                    for pos, (cutoff, row) in enumerate(zip(group_origins, group_rows)):
                        probs[pos] = row_predictions[int(cutoff)][arm][int(row)]
                    metrics[arm] = _binary_metrics(y, probs, float(protocol["recipe"]["threshold"]))
                for arm in arms:
                    metric = metrics[arm]
                    f1_harm = metrics["sparse25"]["macro_f1"] - metric["macro_f1"]
                    recall_harm = metrics["sparse25"]["positive_recall"] - metric["positive_recall"]
                    guard_pass = (not supported) or (f1_harm <= f1_limit and recall_harm <= recall_limit)
                    output.append({
                        "view": view, "field": field, "group": str(group_label), "arm": arm,
                        "n": int(len(group_rows)), "n0": n0, "n1": n1, "supported": bool(supported),
                        "macro_f1": metric["macro_f1"], "positive_recall": metric["positive_recall"],
                        "f1_harm": float(f1_harm), "recall_harm": float(recall_harm),
                        "guard_pass": bool(guard_pass),
                    })
    return output


def _independent_results(protocol: Mapping[str, Any], frame: pd.DataFrame, target: np.ndarray,
                         cycles: np.ndarray, predictions: Mapping[int, Mapping[str, np.ndarray]]) -> dict[str, Any]:
    arms = ("sparse25", "N01_class_only", "N02_sketch", "matched_combo", "candidate")
    view_rows = _view_rows(protocol, cycles)
    metrics: dict[str, dict[str, dict[str, Any]]] = {}
    gains: dict[str, dict[str, float]] = {}
    for view, origins in view_rows.items():
        rows = np.concatenate([origins[cutoff] for cutoff in origins])
        _expect(len(rows) == len(np.unique(rows)) == int(protocol["view_rows"][view]),
                f"View denominator or overlap invalid: {view}")
        labels = target[rows]
        arms_rows: dict[str, np.ndarray] = {arm: np.empty(len(rows), dtype=np.float64) for arm in arms}
        row_to_view = {int(row): idx for idx, row in enumerate(rows)}
        for cutoff in origins:
            info = predictions[cutoff]
            for arm in arms:
                source_arm = arm
                if source_arm not in info:
                    continue
                for row, prob in zip(info["rows"], info[source_arm]):
                    index = row_to_view.get(int(row))
                    if index is not None:
                        arms_rows[arm][index] = float(prob)
        metrics[view] = {arm: _binary_metrics(labels, arms_rows[arm], float(protocol["recipe"]["threshold"]))
                         for arm in arms}
        base = metrics[view]["sparse25"]["macro_f1"]
        gains[view] = {arm: float(metrics[view][arm]["macro_f1"] - base)
                       for arm in arms if arm != "sparse25"}
    gates = protocol["gates"]
    primary = metrics["primary"]
    h12_names = gates["h12_scope"]
    checks = {
        "primary_vs_sparse25": gains["primary"]["candidate"] >= float(gates["primary_gain_vs_sparse25_min"]),
        "primary_vs_N01_class_only": metrics["primary"]["candidate"]["macro_f1"]
            - metrics["primary"]["N01_class_only"]["macro_f1"] >= float(gates["primary_gain_vs_each_component_min"]),
        "primary_vs_N02_sketch": metrics["primary"]["candidate"]["macro_f1"]
            - metrics["primary"]["N02_sketch"]["macro_f1"] >= float(gates["primary_gain_vs_each_component_min"]),
        "primary_vs_matched_combo": metrics["primary"]["candidate"]["macro_f1"]
            - metrics["primary"]["matched_combo"]["macro_f1"] >= float(gates["primary_gain_vs_matched_combo_min"]),
        "H12_vs_sparse25": all(
            metrics[name]["candidate"]["macro_f1"] - metrics[name]["sparse25"]["macro_f1"]
            >= -float(gates["h12_harm_vs_sparse25_max"]) for name in h12_names),
        "W1_harm_vs_sparse25": metrics["W1"]["candidate"]["macro_f1"]
            - metrics["W1"]["sparse25"]["macro_f1"] >= -float(gates["individual_primary_window_harm_max"]),
        "W4_harm_vs_sparse25": metrics["W4"]["candidate"]["macro_f1"]
            - metrics["W4"]["sparse25"]["macro_f1"] >= -float(gates["individual_primary_window_harm_max"]),
    }
    for h12_name in h12_names:
        checks[f"{h12_name}_vs_sparse25"] = (
            metrics[h12_name]["candidate"]["macro_f1"] - metrics[h12_name]["sparse25"]["macro_f1"]
            >= -float(gates["h12_harm_vs_sparse25_max"])
        )
    return {"views": metrics, "gains_vs_sparse25": gains, "checks": checks}


def _compare_root_views(independent: Mapping[str, Any], root_results: Mapping[str, Any]) -> dict[str, Any]:
    root_views = root_results.get("views")
    _expect(isinstance(root_views, dict), "Root results.json has no views object")
    check_arms = {"sparse25": "sparse25", "N01_class_only": "N01_class_only", "N02_sketch": "N02_sketch",
                  "matched_combo": "matched_combo", "candidate": "candidate"}
    mismatches: list[str] = []
    comparisons = 0
    for view, arms in independent["views"].items():
        root_view = root_views.get(view)
        _expect(isinstance(root_view, dict) and isinstance(root_view.get("metrics"), dict),
                f"Root results missing metrics for view {view}")
        for local_arm, external_arm in check_arms.items():
            root_metric = root_view["metrics"].get(external_arm)
            _expect(isinstance(root_metric, dict), f"Root results missing {view}/{external_arm}")
            for key in ("n", "tn", "fp", "fn", "tp"):
                if key in root_metric:
                    comparisons += 1
                    if int(root_metric[key]) != int(arms[local_arm][key]):
                        mismatches.append(f"{view}/{external_arm}/{key}")
            for key in ("macro_f1", "positive_recall"):
                if key in root_metric:
                    comparisons += 1
                    if not _close(float(root_metric[key]), float(arms[local_arm][key])):
                        mismatches.append(f"{view}/{external_arm}/{key}")
    _expect(comparisons >= 60, "Root results did not expose sufficient per-view/arm metrics for independent comparison")
    root_gate = root_results.get("gate", {}).get("checks", {})
    independent_gate = {
        "primary_vs_sparse25": independent["checks"]["primary_vs_sparse25"],
        "primary_vs_class_only": independent["checks"]["primary_vs_N01_class_only"],
        "primary_vs_sketch": independent["checks"]["primary_vs_N02_sketch"],
        "primary_vs_matched_combo": independent["checks"]["primary_vs_matched_combo"],
        "H12_55_aggregate_harm": independent["checks"]["H12_55_vs_sparse25"],
        "H12_67_aggregate_harm": independent["checks"]["H12_67_vs_sparse25"],
        "H12_pooled_aggregate_harm": independent["checks"]["H12_pooled_vs_sparse25"],
        "W1_aggregate_harm": independent["checks"]["W1_harm_vs_sparse25"],
        "W4_aggregate_harm": independent["checks"]["W4_harm_vs_sparse25"],
        "all_view_supported_subgroup_guards": independent["checks"].get("all_view_supported_subgroup_guards"),
    }
    gate_mismatches = {key: {"independent": value, "root": root_gate.get(key)}
                       for key, value in independent_gate.items()
                       if value is not None and root_gate.get(key) != value}
    expected_decision = ("CONDITIONAL_CONTINUATION_ELIGIBLE" if all(independent["checks"].values())
                         else "NO_GO_FURTHER_FITS_FOR_THIS_BLEND")
    root_decision = root_results.get("gate", {}).get("decision")
    if root_decision != expected_decision:
        gate_mismatches["decision"] = {"independent": expected_decision, "root": root_decision}
    return {"compared_fields": comparisons, "mismatches": mismatches,
            "gate_mismatches": gate_mismatches, "pass": not mismatches and not gate_mismatches}


def _compare_root_groups(independent_rows: Sequence[Mapping[str, Any]], root_path: Path) -> dict[str, Any]:
    with root_path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        root_rows = list(reader)
        fieldnames = set(reader.fieldnames or [])
    required = {"view", "field", "group", "arm", "n", "n0", "n1", "supported",
                "macro_f1", "positive_recall", "f1_harm", "recall_harm", "guard_pass"}
    _expect(required <= fieldnames, f"Root groups.csv schema missing columns: {sorted(required - fieldnames)}")
    key_fields = ("view", "field", "group", "arm")
    def key(row: Mapping[str, Any]) -> tuple[str, ...]:
        return tuple(str(row[name]) for name in key_fields)
    expected_by_key = {key(row): row for row in independent_rows}
    actual_by_key = {key(row): row for row in root_rows}
    _expect(len(expected_by_key) == len(independent_rows), "Independent group keys are duplicated")
    _expect(len(actual_by_key) == len(root_rows), "Root group keys are duplicated")
    missing = sorted(set(expected_by_key) - set(actual_by_key))
    extra = sorted(set(actual_by_key) - set(expected_by_key))
    mismatches: list[str] = []
    comparisons = 0
    for item in set(expected_by_key) & set(actual_by_key):
        expected, actual = expected_by_key[item], actual_by_key[item]
        for name in ("n", "n0", "n1"):
            comparisons += 1
            if int(actual[name]) != int(expected[name]):
                mismatches.append(f"{item}/{name}")
        for name in ("supported", "guard_pass"):
            actual_bool = str(actual[name]).strip().casefold() in {"true", "1", "yes"}
            comparisons += 1
            if actual_bool != bool(expected[name]):
                mismatches.append(f"{item}/{name}")
        for name in ("macro_f1", "positive_recall", "f1_harm", "recall_harm"):
            comparisons += 1
            if not _close(float(actual[name]), float(expected[name]), atol=3e-11):
                mismatches.append(f"{item}/{name}")
    return {"expected_rows": len(independent_rows), "root_rows": len(root_rows),
            "missing_keys": [list(x) for x in missing], "extra_keys": [list(x) for x in extra],
            "compared_fields": comparisons, "mismatches": mismatches,
            "pass": not missing and not extra and not mismatches}


def _write_once(path: Path, value: Mapping[str, Any]) -> None:
    payload = json.dumps(value, sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    if path.exists():
        existing = _read_json(path)
        _expect(_canonical(existing) == _canonical(value),
                f"Refusing to replace different verifier output: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(payload)
        stream.flush()
        import os
        os.fsync(stream.fileno())


def verify(protocol_path: Path, root_results_path: Path, root_groups_path: Path) -> dict[str, Any]:
    protocol = _load_protocol_once(protocol_path)
    manifest = _validate_manifest(protocol)
    train_rel = "data/raw/classification/Train.csv"
    train_path = ROOT / train_rel
    _expect(protocol["input_sha256"].get(train_rel), "Official Train is absent from protocol hash manifest")
    frame, target, cycles, missing = _read_train(train_path, protocol["input_sha256"][train_rel])
    n01_freeze = _read_json(ROOT / "experiments/optimization_v3/research_next_v1/frozen/N01/freeze.json")
    n02_freeze = _read_json(ROOT / "experiments/optimization_v3/research_next_v1/frozen/N02/amendment_02/freeze.json")
    bundles: dict[tuple[str, int, str], dict[str, np.ndarray]] = {}
    for cutoff in protocol["origins"]:
        bundles[("N01", int(cutoff), "class_only")] = _load_bundle(
            protocol, card="N01", cutoff=int(cutoff), fit_kind="class_only", freeze=n01_freeze)
        for kind in ("sketch", "no_sketch"):
            bundles[("N02", int(cutoff), kind)] = _load_bundle(
                protocol, card="N02", cutoff=int(cutoff), fit_kind=kind, freeze=n02_freeze)

    # Validate all fit/score boundaries and rebuild row identities from official Train.
    score_audit: dict[int, dict[str, int]] = {}
    for cutoff in protocol["origins"]:
        cutoff = int(cutoff)
        expected_fit = np.flatnonzero(cycles <= cutoff)
        _expect(len(expected_fit) > 0 and int(cycles[expected_fit].max()) == cutoff,
                f"No exact fit rows through cutoff {cutoff}")
        for card, kind in (("N01", "class_only"), ("N02", "sketch"), ("N02", "no_sketch")):
            arrays = bundles[(card, cutoff, kind)]
            _expect(int(np.asarray(arrays["score_rows"]).min()) >= 0
                    and int(np.asarray(arrays["score_rows"]).max()) < len(frame),
                    f"Score row index outside Train bounds: {card}/{cutoff}/{kind}")
            _expect(np.array_equal(frame["record_id"].astype(str).to_numpy()[arrays["score_rows"]],
                                   arrays["score_record_ids"].astype(str)),
                    f"Official Train IDs do not match score rows: {card}/{cutoff}/{kind}")
            _expect(np.array_equal(cycles[arrays["score_rows"]], arrays["cycle_code"]),
                    f"Official Train cycle labels do not match score rows: {card}/{cutoff}/{kind}")
            fit_mask = cycles <= cutoff
            _expect(not np.any(fit_mask[arrays["score_rows"]]),
                    f"Fit and score rows overlap: {card}/{cutoff}/{kind}")
            metadata = _read_json(ROOT / f"experiments/runs/optimization_v3_research_next_v1/{card}/pilot/cutoff_{cutoff}/{kind}/receipt.json")["metadata"]["metadata"]
            _expect(int(metadata["fit_rows"]) == len(expected_fit), f"Fit row count mismatch: {card}/{cutoff}/{kind}")
            _expect(int(metadata["score_rows"]) == len(arrays["score_rows"]), f"Score row count mismatch: {card}/{cutoff}/{kind}")
        score_audit[cutoff] = {"fit_rows": int(len(expected_fit)),
                               "score_rows": int(len(bundles[("N01", cutoff, "class_only")]["score_rows"]))}
    predictions = _route_checks(bundles, [int(value) for value in protocol["origins"]])
    for cutoff, info in predictions.items():
        idx = info["rows"]
        _expect(np.array_equal(info["missing"], missing[idx]), f"Stream-missing mask does not reconstruct at cutoff {cutoff}")
    independent = _independent_results(protocol, frame, target, cycles, predictions)
    groups = _independent_groups(protocol, frame, target, cycles, missing, predictions)
    supported_candidate_groups = [row for row in groups if row["supported"] and row["arm"] == "candidate"]
    independent["checks"]["all_view_supported_subgroup_guards"] = all(
        row["guard_pass"] for row in supported_candidate_groups)

    root_results = _read_json(root_results_path)
    _expect(root_results.get("id") == protocol["id"], "Root result protocol id differs")
    _expect(root_results.get("protocol_sha256") == _file_sha256(protocol_path),
            "Root results do not identify the frozen protocol hash")
    _expect(root_results.get("official_estimator_fits") == 0
            and root_results.get("test_access") is False and root_results.get("kaggle_access") is False
            and root_results.get("ledger_unchanged") is True,
            "Root evaluator provenance flags are inconsistent with the zero-fit boundary")
    root_view_check = _compare_root_views(independent, root_results)
    root_group_check = _compare_root_groups(groups, root_groups_path)
    _expect(root_view_check["pass"], "Independent view metrics disagree with root evaluator")
    _expect(root_group_check["pass"], "Independent group guards disagree with root evaluator")

    output = {
        "verifier": "independent_zero_fit_class_blend_verifier",
        "protocol_id": protocol["id"], "protocol_sha256": _file_sha256(protocol_path),
        "protocol_source_head": protocol["source_head"],
        "verifier_source_sha256": _file_sha256(Path(__file__).resolve()),
        "root_evaluator_source_sha256": root_results.get("source_sha256"),
        "raw_train_sha256": manifest[train_rel], "raw_train_rows": int(len(frame)),
        "input_hashes_verified": len(manifest), "bundle_count_verified": len(bundles),
        "score_audit_by_cutoff": score_audit,
        "independent_views": independent["views"], "gains_vs_sparse25": independent["gains_vs_sparse25"],
        "gate_checks": independent["checks"],
        "subgroups": {"rows": len(groups),
                      "supported_candidate_groups": len(supported_candidate_groups),
                      "supported_candidate_failures": sum(not row["guard_pass"] for row in supported_candidate_groups),
                      "groups_by_view": {view: len({(row["field"], row["group"])
                                                     for row in groups if row["view"] == view
                                                     and row["arm"] == "candidate"})
                                         for view in sorted(protocol["views"])},
                      "all_six_views": sorted({row["view"] for row in groups})},
        "root_results_crosscheck": root_view_check,
        "root_groups_crosscheck": root_group_check,
        "decision": ("CONDITIONAL_CONTINUATION_ELIGIBLE" if all(independent["checks"].values())
                     else "NO_GO_FURTHER_FITS_FOR_THIS_BLEND"),
        "status": "PASS",
    }
    return output


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=PROTOCOL)
    parser.add_argument("--root-results", type=Path, default=RESULTS)
    parser.add_argument("--root-groups", type=Path, default=GROUPS)
    parser.add_argument("--output", type=Path, default=None,
                        help="Optional immutable JSON output path; differing existing output is refused")
    args = parser.parse_args(argv)
    output_path = args.output
    try:
        result = verify(args.protocol.resolve(), args.root_results.resolve(), args.root_groups.resolve())
        if output_path is not None:
            _write_once(output_path.resolve(), result)
        print(json.dumps(result, sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False))
        return 0
    except (VerificationError, OSError, KeyError, TypeError, ValueError) as exc:
        failure = {"status": "FAIL", "error_type": type(exc).__name__, "error": str(exc)}
        print(json.dumps(failure, sort_keys=True, indent=2, ensure_ascii=False))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
