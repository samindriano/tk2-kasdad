"""Frozen, resumable C19 and matched tree-TTR classification research."""
from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import importlib.metadata
import json
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np
import pandas as pd
import psutil
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (average_precision_score, brier_score_loss, f1_score,
                             log_loss, roc_auc_score)
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.models.research_next_budget import (  # noqa: E402
    authorize_new_phase, finish_fit, ledger_summary, read_events, release_unused,
    reserve_card, start_fit,
)
from src.models.research_next_contract import (  # noqa: E402
    canonical_sha256, file_sha256, publish_bundle, read_bundle,
)
from src.models.optimization_v3_batch1_c19 import (  # noqa: E402
    BLEND_WEIGHT as C19_WEIGHT, FEATURE_NAMES, THRESHOLD, TFIDF_PARAMETERS,
    XGBoostSVD32Classifier,
)
from src.models.regression_research_text import RegressionTextModel  # noqa: E402
from src.models.classification_models import _TFIDF_PARAMS, _text_values  # noqa: E402
from src.models.optimization_v3_sparse_classification import (  # noqa: E402
    SparseTextBoostClassifier, PARAMS as SPARSE_PARAMS,
)
from src.models.optimization_v3_batch1_c19 import XGB_PARAMETERS  # noqa: E402
from scripts.optimization_v3_sparse_classification_fullsix import (  # noqa: E402
    control_record, pilot_record, ROUND1_CACHE, ROUND1_AUDITS, CB_CACHE,
    PILOT_CACHE, CACHE as FULLSIX_CACHE,
)
from scripts.optimization_v3_research_next_n01 import _text_quartiles  # noqa: E402
from src.utils.paths import task_files  # noqa: E402

PROTOCOL = ROOT / "experiments/optimization_v3/classification_next_v1/protocol.json"
FREEZE = ROOT / "experiments/optimization_v3/classification_next_v1/freeze.json"
LEDGER = ROOT / "experiments/runs/optimization_v3_research_next_v1/budget.jsonl"
RUN_ROOT = ROOT / "experiments/runs/optimization_v3_classification_next_v1"
REPORT_ROOT = ROOT / "reports/optimization_v3/classification_next_v1"
CLASS_ONLY_AUTH = ROOT / "reports/optimization_v3/measured_benchmark_v1/CLASSIFICATION/CLASS_ONLY_UPLOAD_AUTHORIZATION.json"
CLASS_ONLY_MANIFEST = ROOT / "reports/optimization_v3/measured_benchmark_v1/CLASSIFICATION/CLASS_ONLY_SUBMISSION_MANIFEST.json"
CLASS_ONLY_RECEIPTS = REPORT_ROOT / "submission_receipts.jsonl"
CLASS_ONLY_CSV = ROOT / "submissions/optimization_v3/measured_benchmark_v1/class_only.csv"
TRAIN_PATH = ROOT / "data/raw/classification/Train.csv"
TRAIN_SHA256 = "875552a68a3d24b0f3e1966d6ddb5671e6f54305a1c526e2f0a76747942ed91c"
SIX_ORIGINS = (55, 59, 63, 67, 71, 75)
NEW_C19_ORIGINS = (59, 63, 71, 75)
TTR_ORIGINS = (55, 67)
WINDOWS = {
    "W1": [(55, 56, 59)], "W2": [(59, 60, 63)], "W3": [(63, 64, 67)],
    "W4": [(67, 68, 71)], "W5": [(71, 72, 75)], "W6": [(75, 76, 79)],
}
C19_VIEWS = {
    **WINDOWS,
    "primary_fullsix": [item for name in WINDOWS for item in WINDOWS[name]],
    "legacy_primary_W1_W4": WINDOWS["W1"] + WINDOWS["W4"],
    "late_W5_W6": WINDOWS["W5"] + WINDOWS["W6"],
    "H12_55": [(55, 56, 67)], "H12_67": [(67, 68, 79)],
    "H12_pooled": [(55, 56, 67), (67, 68, 79)],
}
TTR_VIEWS = {
    "W1": WINDOWS["W1"], "W4": WINDOWS["W4"],
    "primary_two_origin": WINDOWS["W1"] + WINDOWS["W4"],
    "H12_55": [(55, 56, 67)], "H12_67": [(67, 68, 79)],
    "H12_pooled": [(55, 56, 67), (67, 68, 79)],
}
THREADS = 2
JOB_CAP_BYTES = 6 * 1024**3
FREE_RESERVE_BYTES = 2 * 1024**3
MIN_FREE_BYTES = 4 * 1024**3
TOKEN_RE = re.compile(r"[A-Za-z0-9]+")


def _sha(path: Path) -> str:
    return file_sha256(path)


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_once(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    if path.exists():
        if path.read_text(encoding="utf-8") != encoded:
            raise RuntimeError(f"Immutable output differs; refusing overwrite: {path}")
        return
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(encoded)
        stream.flush()


def _write_csv_once(path: Path, rows: list[dict[str, Any]], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        with path.open("r", encoding="utf-8", newline="") as stream:
            existing = list(csv.DictReader(stream))
        expected = [{key: str(row.get(key, "")) for key in columns} for row in rows]
        if existing != expected:
            raise RuntimeError(f"Immutable CSV differs; refusing overwrite: {path}")
        return
    with path.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in columns})
        stream.flush()


def _stage_budget_snapshot(stage: str, destination: Path) -> dict[str, Any]:
    """Freeze the ledger view at stage completion so later fits cannot change its report."""
    path = destination / "budget_snapshot.json"
    if path.exists():
        saved = _json(path)
        if saved.get("stage") != stage:
            raise RuntimeError(f"Budget snapshot stage mismatch at {path}")
        return saved
    snapshot = {"stage": stage, "ledger_summary": ledger_summary(LEDGER)}
    _write_once(path, snapshot)
    return snapshot


def _git_head() -> str:
    return subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip()


def _git_blob_sha(commit: str, relpath: str) -> str | None:
    result = subprocess.run(["git", "-C", str(ROOT), "show", f"{commit}:{relpath}"],
                            capture_output=True, check=False)
    return hashlib.sha256(result.stdout).hexdigest() if result.returncode == 0 else None


def _train() -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    if not TRAIN_PATH.is_file() or _sha(TRAIN_PATH) != TRAIN_SHA256:
        raise RuntimeError("Official Classification Train is missing or its frozen SHA256 changed")
    frame = pd.read_csv(TRAIN_PATH)
    if len(frame) != 50_608 or frame["record_id"].isna().any() or not frame["record_id"].astype(str).is_unique:
        raise RuntimeError("Official Train row count or unique record_id check failed")
    cycle = pd.to_numeric(frame["cycle_code"], errors="raise").to_numpy(dtype=np.int64)
    target_raw = pd.to_numeric(frame["artifact_signal"], errors="raise").to_numpy()
    if not np.array_equal(np.unique(cycle), np.arange(47, 80)) or not np.isin(target_raw, [0, 1]).all():
        raise RuntimeError("Official Train cycle or target support differs from frozen protocol")
    return frame, target_raw.astype(np.int8), cycle


def _metric(y: np.ndarray, p: np.ndarray) -> dict[str, Any]:
    y = np.asarray(y, dtype=np.int8)
    p = np.asarray(p, dtype=np.float64)
    if y.ndim != 1 or p.shape != y.shape or not len(y) or not np.isfinite(p).all() or np.any((p < 0) | (p > 1)):
        raise ValueError("Metric inputs must be aligned finite binary labels and probabilities")
    labels = p >= 0.5
    tp = int(np.sum((y == 1) & labels)); fn = int(np.sum((y == 1) & ~labels))
    tn = int(np.sum((y == 0) & ~labels)); fp = int(np.sum((y == 0) & labels))
    if len(np.unique(y)) != 2:
        auc = ap = None
    else:
        auc = float(roc_auc_score(y, p)); ap = float(average_precision_score(y, p))
    return {
        "n": int(len(y)), "class_0_n": int(np.sum(y == 0)), "class_1_n": int(np.sum(y == 1)),
        "macro_f1": float(f1_score(y, labels, average="macro", zero_division=0)),
        "positive_recall": float(tp / (tp + fn)) if tp + fn else None,
        "roc_auc": auc, "average_precision": ap, "log_loss": float(log_loss(y, p, labels=[0, 1])),
        "brier": float(brier_score_loss(y, p)), "true_positive": tp, "false_positive": fp,
        "true_negative": tn, "false_negative": fn, "predicted_positive_rate": float(labels.mean()),
        "mean_probability": float(p.mean()), "threshold": 0.5,
    }


def _token_features(values: pd.Series | list[Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    counts: list[int] = []; uniques: list[int] = []
    for value in values:
        if value is None or (not isinstance(value, (str, bytes)) and pd.isna(value)):
            text = ""
        else:
            text = value.decode("utf-8", errors="replace") if isinstance(value, bytes) else str(value)
        tokens = TOKEN_RE.findall(text.lower())
        counts.append(len(tokens)); uniques.append(len(set(tokens)))
    count = np.asarray(counts, dtype=np.int32)
    unique = np.asarray(uniques, dtype=np.int32)
    ttr = np.full(len(count), np.nan, dtype=np.float64)
    valid = count > 0
    ttr[valid] = unique[valid] / count[valid]
    return count, unique, ttr


def _route(candidate_kind: str, baseline: np.ndarray, raw: np.ndarray,
           c2b: np.ndarray, cb: np.ndarray, missing: np.ndarray) -> np.ndarray:
    missing = np.asarray(missing, dtype=bool)
    if candidate_kind == "C19":
        present = 0.75 * np.asarray(baseline, dtype=np.float64) + 0.25 * np.asarray(raw, dtype=np.float64)
    elif candidate_kind in {"TTR_COUNT", "TTR_BOTH"}:
        present = 0.75 * np.asarray(c2b, dtype=np.float64) + 0.25 * np.asarray(raw, dtype=np.float64)
    else:
        raise ValueError(f"Unknown route kind: {candidate_kind}")
    out = np.where(missing, cb, present).astype(np.float64, copy=False)
    if out.shape != missing.shape or not np.isfinite(out).all() or np.any((out < 0) | (out > 1)):
        raise ValueError("Routed probabilities are invalid")
    return out


def _payload(*, candidate_kind: str, rows: np.ndarray, ids: np.ndarray, cycles: np.ndarray,
             baseline: np.ndarray, raw: np.ndarray, c2b: np.ndarray, cb: np.ndarray,
             missing: np.ndarray, count: np.ndarray | None = None,
             unique: np.ndarray | None = None, prediction: np.ndarray | None = None) -> dict[str, Any]:
    prediction = _route(candidate_kind, baseline, raw, c2b, cb, missing) if prediction is None else np.asarray(prediction, dtype=np.float64)
    arrays: dict[str, np.ndarray] = {
        "score_rows": np.asarray(rows, dtype=np.int64),
        "score_record_ids": np.asarray(ids, dtype=str),
        "cycle_code": np.asarray(cycles, dtype=np.int16),
        "prediction": prediction,
        "baseline_prediction": np.asarray(baseline, dtype=np.float64),
        "raw_candidate": np.asarray(raw, dtype=np.float64),
        "control_stream_missing": np.asarray(missing, dtype=bool),
        "control_c2b_probability": np.asarray(c2b, dtype=np.float64),
        "control_cb_probability": np.asarray(cb, dtype=np.float64),
    }
    if count is not None:
        arrays["token_count"] = np.asarray(count, dtype=np.int32)
        arrays["token_unique"] = np.asarray(unique, dtype=np.int32)
    return {**arrays, "candidate_kind": candidate_kind,
            "route": ("present=.75*currentSparse25+.25*C19; missing=CB; threshold=.50" if candidate_kind == "C19"
                      else "present=.75*C2_B+.25*armExpert; missing=CB; threshold=.50"),
            "probability_field": "prediction is routed probability; decision label is prediction >= 0.50"}


def _payload_arrays(payload: Mapping[str, Any]) -> dict[str, np.ndarray]:
    return {key: np.asarray(value) for key, value in payload.items() if isinstance(value, np.ndarray)}


def _identity(cutoff: int, stage: str, candidate_kind: str, freeze_signature: str) -> dict[str, Any]:
    protocol_sha = _sha(PROTOCOL)
    return {
        "campaign": "CLASSIFICATION_NEXT_V1", "stage": stage, "candidate_kind": candidate_kind,
        "cutoff": int(cutoff), "fit_cycle_min": 47, "fit_cycle_max": int(cutoff),
        "protocol_sha256": protocol_sha, "freeze_signature": freeze_signature,
        "feature_allowlist_sha256": canonical_sha256(list(FEATURE_NAMES)),
        "threshold": 0.5,
    }


def _persist_fit(bundle_path: Path, fit_id: str, card_id: str, stage: str, cutoff: int,
                 candidate_kind: str, identity: Mapping[str, Any],
                 fit_fn: Callable[[], tuple[dict[str, Any], dict[str, Any]]],
                 ledger_path: Path = LEDGER) -> dict[str, Any]:
    events = read_events(ledger_path)
    starts = [e for e in events if e.get("event") == "FIT_STARTED" and e.get("fit_id") == fit_id]
    terminal = [e for e in events if e.get("fit_id") == fit_id and e.get("event") in {"FIT_SAVED", "FIT_FAILED"}]
    if len(starts) > 1 or len(terminal) > 1:
        raise RuntimeError(f"Budget ledger has duplicate events for {fit_id}")
    if bundle_path.exists():
        if not starts:
            raise RuntimeError(f"Saved bundle exists without a FIT_STARTED ledger event: {bundle_path}")
        if terminal and terminal[0]["event"] != "FIT_SAVED":
            raise RuntimeError(f"Saved bundle conflicts with failure ledger event: {fit_id}")
        saved = read_bundle(bundle_path, identity, task="classification")
        rebuilt = _route(candidate_kind, saved.arrays["baseline_prediction"], saved.arrays["raw_candidate"],
                         saved.arrays["control_c2b_probability"], saved.arrays["control_cb_probability"],
                         saved.arrays["control_stream_missing"])
        if not np.array_equal(rebuilt, saved.arrays["prediction"]):
            raise RuntimeError(f"Cached prediction route reconstruction mismatch for {fit_id}")
        if not terminal:
            finish_fit(ledger_path, fit_id=fit_id, status="FIT_SAVED", receipt_sha256=saved.receipt["receipt_sha256"])
        elif terminal[0].get("receipt_sha256") != saved.receipt["receipt_sha256"]:
            raise RuntimeError(f"Cached bundle receipt differs from ledger for {fit_id}")
        return {"arrays": saved.arrays, "metadata": saved.receipt["metadata"], "receipt": saved.receipt,
                "resumed": True}
    if starts:
        raise RuntimeError(f"A prior FIT_STARTED exists without a durable bundle for {fit_id}; refusing implicit refit")
    start_fit(ledger_path, fit_id=fit_id, card_id=card_id, stage=stage, cutoff=cutoff, component=candidate_kind)
    try:
        payload, metadata = fit_fn()
        arrays = _payload_arrays(payload)
        receipt = _publish_payload(bundle_path, payload, identity, metadata)
        finish_fit(ledger_path, fit_id=fit_id, status="FIT_SAVED", receipt_sha256=receipt["receipt_sha256"])
        saved = read_bundle(bundle_path, identity, task="classification")
        return {"arrays": saved.arrays, "metadata": saved.receipt["metadata"], "receipt": saved.receipt,
                "resumed": False, "fit_metadata": metadata}
    except Exception as exc:
        try:
            current = read_events(ledger_path)
            terminal_now = any(e.get("fit_id") == fit_id and e.get("event") in {"FIT_SAVED", "FIT_FAILED"} for e in current)
            if not terminal_now:
                if bundle_path.exists():
                    durable = read_bundle(bundle_path, identity, task="classification")
                    rebuilt = _route(candidate_kind, durable.arrays["baseline_prediction"], durable.arrays["raw_candidate"],
                        durable.arrays["control_c2b_probability"], durable.arrays["control_cb_probability"],
                        durable.arrays["control_stream_missing"])
                    if not np.array_equal(rebuilt, durable.arrays["prediction"]):
                        raise RuntimeError("Durable bundle failed route reconstruction during recovery")
                    finish_fit(ledger_path, fit_id=fit_id, status="FIT_SAVED",
                               receipt_sha256=durable.receipt["receipt_sha256"])
                else:
                    finish_fit(ledger_path, fit_id=fit_id, status="FIT_FAILED", error=f"{type(exc).__name__}: {exc}")
        finally:
            raise


def _publish_payload(bundle_path: Path, payload: Mapping[str, Any], identity: Mapping[str, Any],
                     metadata: Mapping[str, Any]) -> dict[str, Any]:
    details = dict(payload)
    arrays = _payload_arrays(details)
    json_meta = {key: value for key, value in details.items() if not isinstance(value, np.ndarray)}
    json_meta.update(metadata)
    return publish_bundle(bundle_path, {**arrays, "payload_meta": json_meta}, identity, task="classification")


def _load_baseline(frame: pd.DataFrame, target: np.ndarray, cutoff: int) -> dict[str, Any]:
    control = control_record(frame, target, cutoff)
    if cutoff in (55, 67):
        record = pilot_record(frame, control)
        rows = record["rows"]
        sparse_raw = np.asarray(record["raw_candidate"], dtype=np.float64)
        baseline = np.asarray(record["blend25"], dtype=np.float64)
        source_path = PILOT_CACHE / f"fit{cutoff}.npz"
        source_sidecar = source_path.with_suffix(".json")
    else:
        source_path = FULLSIX_CACHE / f"fit{cutoff}.npz"
        source_sidecar = source_path.with_suffix(".json")
        sidecar = _json(source_sidecar)
        if sidecar.get("npz_sha256") != _sha(source_path):
            raise RuntimeError(f"Sparse25 full-six cache checksum mismatch at cutoff {cutoff}")
        with np.load(source_path, allow_pickle=False) as z:
            rows = z["rows"].astype(np.int64, copy=True)
            sparse_raw = z["raw_candidate"].astype(np.float64, copy=True)
            baseline = z["blend25"].astype(np.float64, copy=True)
            incumbent = z["champion"].astype(np.float64, copy=True)
        if not np.array_equal(incumbent, control["champion"]):
            raise RuntimeError(f"Cached sparse25 incumbent differs from C2_B/CB controls at cutoff {cutoff}")
    if not np.array_equal(rows, control["rows"]):
        raise RuntimeError(f"Sparse25 control rows do not align at cutoff {cutoff}")
    reconstructed = np.where(control["missing"], control["cb_missing"],
                             0.75 * control["c2b"] + 0.25 * sparse_raw)
    if not np.array_equal(baseline, reconstructed):
        raise RuntimeError(f"Frozen sparse25 route does not reconstruct at cutoff {cutoff}")
    return {
        "cutoff": cutoff, "rows": np.asarray(rows, dtype=np.int64), "baseline": baseline,
        "sparse_raw": sparse_raw, "c2b": np.asarray(control["c2b"], dtype=np.float64),
        "cb": np.asarray(control["cb_missing"], dtype=np.float64),
        "missing": np.asarray(control["missing"], dtype=bool),
        "source_npz_sha256": _sha(source_path), "source_sidecar_sha256": _sha(source_sidecar),
        "control_c2b_source_sha256": _sha(ROUND1_CACHE / f"fit{cutoff}__C2_B.npz"),
        "control_c2b_audit_sha256": _sha(ROUND1_AUDITS / f"fit{cutoff}.json"),
        "control_cb_npz_sha256": _sha(CB_CACHE / f"fit{cutoff}.npz"),
        "control_cb_sidecar_sha256": _sha(CB_CACHE / f"fit{cutoff}.json"),
    }


def _load_c19_pilot(frame: pd.DataFrame, target: np.ndarray, cutoff: int,
                    baseline: Mapping[str, Any]) -> dict[str, Any]:
    path = RUN_ROOT.parent / "optimization_v3_batch1" / "C19" / f"C19_fit{cutoff}.npz"
    side = _json(path.with_suffix(".json"))
    if side.get("npz_sha256") != _sha(path):
        raise RuntimeError(f"C19 historical pilot bundle hash mismatch: {path}")
    with np.load(path, allow_pickle=False) as z:
        arrays = {k: z[k].copy() for k in z.files}
    cycles = pd.to_numeric(frame.cycle_code, errors="raise").to_numpy(dtype=np.int64)
    rows = np.asarray(arrays["score_rows"], dtype=np.int64)
    expected = np.flatnonzero((cycles > cutoff) & (cycles <= cutoff + 12))
    np.testing.assert_array_equal(rows, expected)
    np.testing.assert_array_equal(arrays["score_record_ids"].astype(str), frame.record_id.astype(str).to_numpy()[rows])
    np.testing.assert_array_equal(rows, baseline["rows"])
    np.testing.assert_array_equal(arrays["baseline_prediction"], baseline["baseline"])
    np.testing.assert_array_equal(arrays["control__stream_missing"].astype(bool), baseline["missing"])
    np.testing.assert_array_equal(arrays["control__cb_probability"], baseline["cb"])
    np.testing.assert_array_equal(arrays["control__c2b_probability"], baseline["c2b"])
    np.testing.assert_array_equal(arrays["control__baseline_sparse_raw"], baseline["sparse_raw"])
    rebuilt = _route("C19", arrays["baseline_prediction"], arrays["raw_candidate"],
                     baseline["c2b"], baseline["cb"], baseline["missing"])
    np.testing.assert_array_equal(arrays["prediction"], rebuilt)
    return {
        "score_rows": rows, "score_record_ids": arrays["score_record_ids"].astype(str),
        "cycle_code": cycles[rows], "prediction": arrays["prediction"].astype(np.float64),
        "baseline_prediction": arrays["baseline_prediction"].astype(np.float64),
        "raw_candidate": arrays["raw_candidate"].astype(np.float64),
        "control_stream_missing": baseline["missing"], "control_c2b_probability": baseline["c2b"],
        "control_cb_probability": baseline["cb"], "origin": cutoff,
        "source_npz_sha256": _sha(path), "source_sidecar_sha256": _sha(path.with_suffix(".json")),
        "historical_receipt": side,
    }


def _views(outputs: Mapping[int, Mapping[str, Any]], frame: pd.DataFrame, target: np.ndarray,
           definitions: Mapping[str, list[tuple[int, int, int]]]) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    cycle = pd.to_numeric(frame.cycle_code, errors="raise").to_numpy(dtype=np.int64)
    bins_by_origin = {cutoff: _text_quartiles(frame, cutoff)[0] for cutoff in outputs}
    metrics: dict[str, Any] = {}; groups: list[dict[str, Any]] = []; cycles_out: list[dict[str, Any]] = []
    for name, blocks in definitions.items():
        parts = []
        for cutoff, low, high in blocks:
            result = outputs[int(cutoff)]
            selected = (result["cycle_code"] >= low) & (result["cycle_code"] <= high)
            parts.append({key: np.asarray(result[key])[selected] for key in (
                "score_rows", "prediction", "baseline_prediction", "origin")})
        rows = np.concatenate([part["score_rows"] for part in parts])
        if len(np.unique(rows)) != len(rows):
            raise RuntimeError(f"Duplicate Train rows in evaluation view {name}")
        pred = np.concatenate([part["prediction"] for part in parts])
        base = np.concatenate([part["baseline_prediction"] for part in parts])
        origins = np.concatenate([part["origin"] for part in parts]).astype(np.int64)
        y = target[rows]
        cand_metric = _metric(y, pred); base_metric = _metric(y, base)
        metrics[name] = {"n": int(len(rows)), "candidate": cand_metric, "sparse25": base_metric,
                         "gain_macro_f1": float(cand_metric["macro_f1"] - base_metric["macro_f1"])}
        for cycle_value in sorted(np.unique(cycle[rows]).tolist()):
            mask = cycle[rows] == cycle_value
            cm = _metric(y[mask], pred[mask]); bm = _metric(y[mask], base[mask])
            cycles_out.append({"view": name, "cycle": int(cycle_value), "n": int(mask.sum()),
                               "candidate_macro_f1": cm["macro_f1"], "sparse25_macro_f1": bm["macro_f1"],
                               "gain_macro_f1": cm["macro_f1"] - bm["macro_f1"]})
        quartile_labels = np.asarray([bins_by_origin[int(origin)][int(row)] for row, origin in zip(rows, origins)])
        for label in sorted(np.unique(quartile_labels).tolist()):
            mask = quartile_labels == label
            gy = y[mask]; n = int(mask.sum()); n0 = int(np.sum(gy == 0)); n1 = int(np.sum(gy == 1))
            cm = _metric(gy, pred[mask]); bm = _metric(gy, base[mask])
            supported = bool(n >= 200 and n0 >= 30 and n1 >= 30)
            f1_harm = float(bm["macro_f1"] - cm["macro_f1"])
            recall_harm = (float(bm["positive_recall"] - cm["positive_recall"])
                           if bm["positive_recall"] is not None and cm["positive_recall"] is not None else None)
            groups.append({"view": name, "field": "fit_defined_text_load_quartile",
                           "group": "missing" if int(label) < 0 else f"Q{min(int(label)+1,4)}",
                           "n": n, "class_0_n": n0, "class_1_n": n1, "supported": supported,
                           "candidate_macro_f1": cm["macro_f1"], "sparse25_macro_f1": bm["macro_f1"],
                           "f1_harm": f1_harm, "candidate_positive_recall": cm["positive_recall"],
                           "sparse25_positive_recall": bm["positive_recall"], "positive_recall_harm": recall_harm,
                           "fit_origins": sorted(set(origins[mask].astype(int).tolist())),
                           "fails_0_01_f1": bool(supported and f1_harm > 0.01),
                           "fails_0_03_recall": bool(supported and recall_harm is not None and recall_harm > 0.03),
                           "fails_0_02_recall": bool(supported and recall_harm is not None and recall_harm > 0.02)})
    return metrics, groups, cycles_out


def _c19_gate(metrics: Mapping[str, Any], groups: list[dict[str, Any]]) -> dict[str, Any]:
    gains = {name: float(metrics[name]["gain_macro_f1"]) for name in WINDOWS}
    primary = float(metrics["primary_fullsix"]["gain_macro_f1"])
    h12 = float(metrics["H12_pooled"]["gain_macro_f1"])
    h12_67 = float(metrics["H12_67"]["gain_macro_f1"])
    late = float(metrics["late_W5_W6"]["gain_macro_f1"])
    supported = [row for row in groups if row["supported"]]
    checks = {
        "primary_gain_min_0_0005": primary >= 0.0005,
        "H12_pooled_gain_min_0_002": h12 >= 0.002,
        "H12_origin67_gain_min_0_001": h12_67 >= 0.001,
        "late_W5W6_harm_max_0_0005": late >= -0.0005,
        "W6_harm_max_0_001": gains["W6"] >= -0.001,
        "at_least_4_of_6_windows_nonnegative": sum(value >= 0 for value in gains.values()) >= 4,
        "supported_text_load_groups_F1_harm_max_0_01": not any(row["fails_0_01_f1"] for row in supported),
        "supported_text_load_groups_recall_harm_max_0_03": not any(row["fails_0_03_recall"] for row in supported),
    }
    return {"status": "PASS_ALL_FROZEN_GATES" if all(checks.values()) else "NO_GO_GATE_FAILED",
            "checks": checks, "thresholds": {"primary_gain_min": 0.0005, "H12_pooled_gain_min": 0.002,
                "H12_origin67_gain_min": 0.001, "late_harm_max": 0.0005, "W6_harm_max": 0.001,
                "nonnegative_windows_min": 4, "minimum_group_n": 200, "minimum_each_class_n": 30,
                "group_F1_harm_max": 0.01, "positive_recall_harm_max": 0.03},
            "window_gains": gains, "supported_groups": len(supported),
            "unsupported_groups": sum(not row["supported"] for row in groups),
            "supported_group_failures": [row for row in supported if row["fails_0_01_f1"] or row["fails_0_03_recall"]],
            "original_0_02_recall_guard_failures_reported": True}


def _ttr_gate(metrics: Mapping[str, Any], matched: Mapping[str, Any], groups: list[dict[str, Any]]) -> dict[str, Any]:
    supported = [row for row in groups if row["supported"]]
    primary = metrics["primary_two_origin"]
    h12_names = ("H12_55", "H12_67", "H12_pooled")
    checks = {
        "primary_gain_vs_sparse25_min_0_0005": primary["gain_macro_f1"] >= 0.0005,
        "primary_gain_vs_COUNT_min_0_0002": matched["primary_two_origin"]["macro_f1"] >= 0.0002,
        "H12_each_and_pooled_harm_max_0_001": all(metrics[name]["gain_macro_f1"] >= -0.001 for name in h12_names),
        "W1_W4_harm_max_0_005": all(metrics[name]["gain_macro_f1"] >= -0.005 for name in ("W1", "W4")),
        "supported_groups_F1_harm_max_0_01": not any(row["fails_0_01_f1"] for row in supported),
        "supported_groups_recall_harm_max_0_02": not any(row["fails_0_02_recall"] for row in supported),
    }
    return {"status": "PASS_DIAGNOSTIC_GATE" if all(checks.values()) else "FAIL_DIAGNOSTIC_GATE",
            "checks": checks, "matched_primary_gain_BOTH_minus_COUNT": matched["primary_two_origin"]["macro_f1"],
            "supported_groups": len(supported), "unsupported_groups": sum(not row["supported"] for row in groups),
            "supported_group_failures": [row for row in supported if row["fails_0_01_f1"] or row["fails_0_02_recall"]],
            "thresholds": {"primary_gain_vs_sparse25_min": 0.0005, "matched_count_gain_min": 0.0002,
                "H12_harm_max": 0.001, "W1_W4_harm_max": 0.005, "group_F1_harm_max": 0.01,
                "positive_recall_harm_max": 0.02, "minimum_group_n": 200, "minimum_each_class_n": 30},
            "stop_after_pilot_review": True}


def _static_paths() -> tuple[list[Path], list[Path]]:
    protocol = _json(PROTOCOL)
    declared_inputs = [ROOT / key for key in protocol.get("input_sha256", {})]
    inputs = set(declared_inputs)
    inputs.update({TRAIN_PATH, CLASS_ONLY_AUTH, CLASS_ONLY_MANIFEST, CLASS_ONLY_CSV})
    for cutoff in SIX_ORIGINS:
        inputs.update({ROUND1_CACHE / f"fit{cutoff}__C2_B.npz", ROUND1_AUDITS / f"fit{cutoff}.json",
                       CB_CACHE / f"fit{cutoff}.npz", CB_CACHE / f"fit{cutoff}.json"})
        if cutoff in (55, 67):
            inputs.update({PILOT_CACHE / f"fit{cutoff}.npz", PILOT_CACHE / f"fit{cutoff}.json"})
        else:
            inputs.update({FULLSIX_CACHE / f"fit{cutoff}.npz", FULLSIX_CACHE / f"fit{cutoff}.json"})
    inputs.update({ROOT / "experiments/optimization_v3/classification_sparse_full/freeze.json",
                   ROOT / "reports/optimization_v3/classification_sparse_full/verification.json",
                   ROOT / "experiments/optimization_v3/classification_sparse/freeze.json",
                   ROOT / "reports/optimization_v3/classification_sparse/pilot.json"})
    sources = [ROOT / item for item in (
        "scripts/optimization_v3_classification_next_v1.py",
        "scripts/optimization_v3_classification_next_v1_verify.py",
        "tests/test_optimization_v3_classification_next_v1.py",
        "src/models/research_next_budget.py", "src/models/research_next_contract.py",
        "src/models/optimization_v3_batch1_c19.py", "scripts/optimization_v3_batch1_c19.py",
        "scripts/optimization_v3_sparse_classification_fullsix.py",
        "scripts/optimization_v3_sparse_classification.py",
        "src/models/optimization_v3_sparse_classification.py",
        "scripts/optimization_v3_research_next_n01.py",
        "src/models/regression_research_text.py", "src/models/classification_models.py",
        "src/models/round1_models.py", "src/utils/paths.py",
    )]
    return sorted(inputs, key=lambda p: p.as_posix()), sorted(set(sources), key=lambda p: p.as_posix())


def _reconcile_class_only() -> dict[str, Any]:
    auth = _json(CLASS_ONLY_AUTH)
    manifest = _json(CLASS_ONLY_MANIFEST)
    event = {
        "event_id": "CLASS_ONLY_KAGGLE_USER_REPORTED_COMPLETE_20261102",
        "event": "KAGGLE_SUBMISSION_USER_REPORTED_COMPLETE",
        "reported_on": "2026-11-02",
        "evidence_kind": "user-pasted Kaggle submissions-history row in chat KASDAD - TK2 (2)",
        "competition": "tp2-classification-cyber-vulnerability-kasdd-gasal-2026-2027",
        "submission_filename": "class_only.csv", "displayed_status": "Complete",
        "displayed_uploader": "Samuel Indriano", "displayed_age": "3h ago",
        "public_score": 0.82774, "submission_ref": None, "exact_timestamp_utc": None,
        "upload_method": "unknown", "server_file_sha256": None,
        "current_executor_uploaded": False,
        "authorization_file": CLASS_ONLY_AUTH.relative_to(ROOT).as_posix(),
        "authorization_file_sha256": _sha(CLASS_ONLY_AUTH),
        "prior_manifest_status": manifest.get("status"),
        "prior_manifest_kaggle_called": manifest.get("kaggle_called"),
        "local_csv_sha256_prepared_before_user_upload": manifest.get("csv", {}).get("sha256"),
        "local_csv_still_matches_prepared_manifest": _sha(CLASS_ONLY_CSV) == manifest.get("csv", {}).get("sha256"),
        "previous_sparse25_public_score": 0.82908,
        "score_delta_vs_sparse25": round(0.82774 - 0.82908, 8),
        "interpretation": "User-reported Kaggle history observation; no API receipt, reference, exact server timestamp, upload method, or server-side CSV hash inferred.",
    }
    event["event_sha256"] = canonical_sha256(event)
    CLASS_ONLY_RECEIPTS.parent.mkdir(parents=True, exist_ok=True)
    if CLASS_ONLY_RECEIPTS.exists():
        current = [json.loads(line) for line in CLASS_ONLY_RECEIPTS.read_text(encoding="utf-8").splitlines() if line.strip()]
        matches = [row for row in current if row.get("event_id") == event["event_id"]]
        if matches and matches != [event]:
            raise RuntimeError("Existing ClassOnly receipt event differs; refusing to rewrite append-only history")
        if not matches:
            with CLASS_ONLY_RECEIPTS.open("a", encoding="utf-8", newline="\n") as stream:
                stream.write(json.dumps(event, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n")
                stream.flush()
    else:
        with CLASS_ONLY_RECEIPTS.open("x", encoding="utf-8", newline="\n") as stream:
            stream.write(json.dumps(event, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n")
            stream.flush()
    return event


def _load_old_c19_outputs(frame: pd.DataFrame, target: np.ndarray,
                          controls: Mapping[int, Mapping[str, Any]]) -> dict[int, dict[str, Any]]:
    outputs = {}
    for cutoff in (55, 67):
        output = _load_c19_pilot(frame, target, cutoff, controls[cutoff])
        outputs[cutoff] = {key: output[key] for key in (
            "score_rows", "score_record_ids", "cycle_code", "prediction", "baseline_prediction")}
        outputs[cutoff]["origin"] = np.full(len(output["score_rows"]), cutoff, dtype=np.int16)
    return outputs


def _resource_preflight(frame: pd.DataFrame) -> dict[str, Any]:
    from scripts.optimization_v3_batch1_c19 import _memory_estimate as c19_memory
    from scripts.optimization_v3_research_next_n01 import _resource_estimate as ttr_memory
    available = int(psutil.virtual_memory().available)
    if available < MIN_FREE_BYTES:
        return {"status": "BLOCKED_RESOURCE", "available_bytes": available,
                "minimum_free_bytes": MIN_FREE_BYTES, "job_cap_bytes": JOB_CAP_BYTES,
                "reserve_bytes": FREE_RESERVE_BYTES, "fits": {}}
    results = {}
    cycles = pd.to_numeric(frame.cycle_code, errors="raise").to_numpy(dtype=np.int64)
    for cutoff in NEW_C19_ORIGINS:
        fit = frame.loc[cycles <= cutoff, list(FEATURE_NAMES)]
        score = frame.loc[(cycles > cutoff) & (cycles <= cutoff + 4), list(FEATURE_NAMES)]
        results[f"C19_{cutoff}"] = c19_memory(fit, score, available_bytes=available)
    for cutoff in TTR_ORIGINS:
        fit = frame.loc[cycles <= cutoff, list(FEATURE_NAMES)]
        results[f"TTR_{cutoff}"] = ttr_memory(fit, available_bytes=available)
    results["status"] = "PASS" if all(item.get("status") == "PASS" for key, item in results.items() if key != "status") else "BLOCKED_RESOURCE"
    results["available_bytes"] = available
    results["max_job_memory_bytes"] = JOB_CAP_BYTES
    results["minimum_free_bytes"] = MIN_FREE_BYTES
    results["free_reserve_bytes"] = FREE_RESERVE_BYTES
    results["threads"] = THREADS
    results["concurrent_fit_jobs"] = 1
    return results


def _stable_identity(frame: pd.DataFrame, train_sha: str, controls: Mapping[int, Mapping[str, Any]]) -> dict[str, Any]:
    inputs, sources = _static_paths()
    missing = [p.as_posix() for p in [*inputs, *sources] if not p.is_file()]
    if missing:
        raise FileNotFoundError(f"Frozen classification-next inputs are missing: {missing}")
    protocol = _json(PROTOCOL)
    for rel, expected in protocol["input_sha256"].items():
        if _sha(ROOT / rel) != expected:
            raise RuntimeError(f"Pinned protocol input hash changed: {rel}")
    return {
        "protocol_sha256": _sha(PROTOCOL), "official_train_sha256": train_sha,
        "inputs_sha256": {p.relative_to(ROOT).as_posix(): _sha(p) for p in inputs},
        "sources_sha256": {p.relative_to(ROOT).as_posix(): _sha(p) for p in sources},
        "control_cache_sha256": {str(cutoff): {key: value for key, value in controls[cutoff].items()
                                                 if key.endswith("sha256")} for cutoff in SIX_ORIGINS},
        "packages": {name: importlib.metadata.version(name) for name in
                     ("numpy", "pandas", "scipy", "scikit-learn", "xgboost", "lightgbm", "threadpoolctl")},
        "training_scope": "official Classification Train cycles 47-79 only",
        "test_labels_or_other_task_labels_used": False,
        "candidate_threshold": 0.5,
    }


def run_synthetic_e2e() -> dict[str, Any]:
    """Exercise the production payload, route, persistence, view and resume path with synthetic data."""
    with tempfile.TemporaryDirectory(prefix="classification-next-synthetic-") as folder:
        root = Path(folder); fit_count = {"value": 0}
        rows_per_cycle = 800
        frame = pd.DataFrame({"record_id": [f"syn_{i}" for i in range(33 * rows_per_cycle)],
                              "cycle_code": np.repeat(np.arange(47, 80), rows_per_cycle),
                              "text_load": np.tile(np.linspace(1, 100, rows_per_cycle), 33),
                              "artifact_signal": np.tile(np.resize(np.array([0, 1], dtype=np.int8), rows_per_cycle), 33)})
        target = frame.artifact_signal.to_numpy(dtype=np.int8); cycles = frame.cycle_code.to_numpy(dtype=np.int64)
        outputs: dict[int, dict[str, Any]] = {}
        synthetic_ledger = root / "synthetic_budget.jsonl"
        authorize_new_phase(synthetic_ledger, cap=6,
            category_caps={"pilot": 0, "continuation": 6, "control_replay": 0, "technical_retry": 0},
            inherited_old_budget_snapshot={"synthetic": True, "official_ledger_mutated": False})
        reserve_card(synthetic_ledger, card_id="SYNTHETIC_CLASS_NEXT", stage="E2E", fits=6, category="continuation")
        for cutoff in SIX_ORIGINS:
            high = cutoff + (12 if cutoff in (55, 67) else 4)
            score_rows = np.flatnonzero((cycles > cutoff) & (cycles <= high))
            baseline = np.full(len(score_rows), 0.45, dtype=np.float64)
            c2b = np.full(len(score_rows), 0.40, dtype=np.float64)
            cb = np.full(len(score_rows), 0.55, dtype=np.float64)
            missing = np.zeros(len(score_rows), dtype=bool)
            missing[::13] = True
            kind = "C19"
            identity = {"synthetic": True, "cutoff": cutoff, "kind": kind}
            destination = root / f"bundle_{cutoff}"
            fit_id = f"SYN_{cutoff}"

            def fit_fn() -> tuple[dict[str, Any], dict[str, Any]]:
                fit_count["value"] += 1
                fit_rows = np.flatnonzero(cycles <= cutoff)
                estimator = _SyntheticEstimator()
                estimator.fit(frame.iloc[fit_rows][["text_load"]], target[fit_rows])
                raw = estimator.predict_probability(frame.iloc[score_rows][["text_load"]])
                pred = _route(kind, baseline, raw, c2b, cb, missing)
                payload = _payload(candidate_kind=kind, rows=score_rows, ids=frame.record_id.to_numpy()[score_rows],
                    cycles=cycles[score_rows], baseline=baseline, raw=raw, c2b=c2b, cb=cb, missing=missing,
                    prediction=pred)
                return payload, {"synthetic_fit_count": fit_count["value"],
                                 "synthetic_training_labels_only": True,
                                 "synthetic_test_labels_passed_to_predict": False}

            saved = _persist_fit(destination, fit_id, "SYNTHETIC_CLASS_NEXT", "E2E", cutoff,
                                 kind, identity, fit_fn, ledger_path=synthetic_ledger)
            # Simulated crash occurs after the durable bundle and before optional report publication.
            if cutoff == 55:
                try:
                    raise RuntimeError("synthetic injected post-bundle report failure")
                except RuntimeError as exc:
                    if "injected" not in str(exc):
                        raise
            if cutoff == 55:
                def forbidden_refit() -> tuple[dict[str, Any], dict[str, Any]]:
                    raise AssertionError("resume attempted an estimator fit")
                resumed = _persist_fit(destination, fit_id, "SYNTHETIC_CLASS_NEXT", "E2E", cutoff,
                    kind, identity, forbidden_refit, ledger_path=synthetic_ledger)
                assert resumed["resumed"]
                assert fit_count["value"] == 1
                saved = resumed
            arrays = saved["arrays"]
            rebuilt = _route(kind, arrays["baseline_prediction"], arrays["raw_candidate"],
                             arrays["control_c2b_probability"], arrays["control_cb_probability"],
                             arrays["control_stream_missing"])
            np.testing.assert_array_equal(rebuilt, arrays["prediction"])
            outputs[cutoff] = {"score_rows": arrays["score_rows"], "score_record_ids": arrays["score_record_ids"],
                "cycle_code": arrays["cycle_code"], "prediction": arrays["prediction"],
                "baseline_prediction": arrays["baseline_prediction"],
                "origin": np.full(len(score_rows), cutoff, dtype=np.int16)}
        views, groups, _cycles = _views(outputs, frame, target, C19_VIEWS)
        gate = _c19_gate(views, groups)
        report = {"synthetic": True, "views": views, "gate": gate, "fit_count": fit_count["value"],
                  "saved_bundle_count": 6, "post_bundle_failure_recovery": "PASS_NO_REFIT",
                  "independent_route_reconstruction": "PASS", "all_view_guard_path": "PASS",
                  "lightgbm_sparse_native_nan_fit_predict": _synthetic_lightgbm_native_nan_smoke()}
        _write_once(root / "synthetic_report.json", report)
        assert fit_count["value"] == 6
        return report


def _synthetic_lightgbm_native_nan_smoke() -> dict[str, Any]:
    """Exercise LightGBM fit/predict on the exact CSR NaN representation used by the TTR arm."""
    from lightgbm import LGBMClassifier

    counts = np.asarray([0, 0, 1, 1, 2, 2, 3, 3, 0, 1, 2, 3], dtype=np.float64)
    ttr = np.asarray([np.nan, np.nan, 1.0, 0.5, 1.0, 0.5, 1.0, 1.0, np.nan, 1.0, 0.5, 1.0], dtype=np.float64)
    design = sparse.csr_matrix(np.column_stack([counts, ttr]), dtype=np.float64)
    explicit_nan = int(np.isnan(design.data).sum())
    if explicit_nan != 3:
        raise RuntimeError("Synthetic TTR CSR design failed to preserve explicit NaN values")
    target = np.asarray([0, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0], dtype=np.int8)
    with threadpool_limits(limits=THREADS):
        model = LGBMClassifier(objective="binary", n_estimators=5, num_leaves=3,
                               min_child_samples=1, n_jobs=THREADS, verbosity=-1,
                               random_state=42).fit(design, target)
        prediction = model.predict_proba(design)[:, 1]
    if prediction.shape != target.shape or not np.isfinite(prediction).all():
        raise RuntimeError("Synthetic LightGBM native-NaN predictions are invalid")
    return {"status": "PASS", "rows": len(target), "explicit_nan_values": explicit_nan,
            "fit_and_predict_completed": True, "finite_prediction_rows": int(np.isfinite(prediction).sum()),
            "official_fit_budget_consumed": 0}


class _SyntheticEstimator:
    """Small test-only fit/predict object used before any official estimator fit."""
    def fit(self, frame: pd.DataFrame, target: np.ndarray) -> "_SyntheticEstimator":
        values = pd.to_numeric(frame["text_load"], errors="raise").to_numpy(dtype=np.float64)
        if len(values) != len(target) or not len(values):
            raise ValueError("Synthetic training matrix is invalid")
        self.mean_ = float(np.mean(values)); self.spread_ = max(float(np.std(values)), 1.0)
        return self

    def predict_probability(self, frame: pd.DataFrame) -> np.ndarray:
        values = pd.to_numeric(frame["text_load"], errors="raise").to_numpy(dtype=np.float64)
        score = np.clip(0.5 + (values - self.mean_) / (10.0 * self.spread_), 0.05, 0.95)
        return score.astype(np.float64)


def _prepare_context(*, write_outputs: bool) -> dict[str, Any]:
    protocol = _json(PROTOCOL)
    if protocol.get("id") != "CLASSIFICATION_NEXT_V1" or _sha(PROTOCOL) != "45a053ab61f13369e6838e92e7a9b7683d0f04a5e505f828ac2a15fb07d27a5a":
        raise RuntimeError("CLASSIFICATION_NEXT_V1 protocol identity/hash changed")
    frame, target, cycle = _train()
    controls = {cutoff: _load_baseline(frame, target, cutoff) for cutoff in SIX_ORIGINS}
    old_outputs = _load_old_c19_outputs(frame, target, controls)
    old_metrics, old_groups, _ = _views(old_outputs, frame, target, {
        "W1": [(55, 56, 59)], "W2": [(55, 60, 63)], "W3": [(55, 64, 67)],
        "W4": [(67, 68, 71)], "W5": [(67, 72, 75)], "W6": [(67, 76, 79)],
        "primary_old_two_window": [(55, 56, 59), (67, 68, 71)],
        "H12_55": [(55, 56, 67)], "H12_67": [(67, 68, 79)],
        "H12_pooled": [(55, 56, 67), (67, 68, 79)],
    })
    old_safety = [row for row in old_groups if row["supported"] and
                  (row["fails_0_01_f1"] or row["fails_0_03_recall"])]
    # Preserve the old 2% recall failure as history even when the new diagnostic ceiling is 3%.
    old_02_failures = [row for row in old_groups if row["supported"] and row["positive_recall_harm"] is not None
                       and row["positive_recall_harm"] > 0.02]
    resources = _resource_preflight(frame)
    synthetic = run_synthetic_e2e()
    stable = _stable_identity(frame, TRAIN_SHA256, controls)
    freeze_signature = canonical_sha256(stable)
    source_commit = _git_head()
    freeze_payload = {"status": "FROZEN_BEFORE_CLASSIFICATION_NEXT_V1_FITS", "identity": stable,
                      "freeze_signature": freeze_signature, "source_commit": source_commit,
                      "source_commit_is_stable_identity": False,
                      "resource_measurement_is_stable_identity": False,
                      "created_at_utc": pd.Timestamp.now(tz="UTC").isoformat()}
    preflight = {
        "status": "PASS" if resources.get("status") == "PASS" and not old_safety and synthetic["post_bundle_failure_recovery"] == "PASS_NO_REFIT" else "NO_GO",
        "protocol_sha256": _sha(PROTOCOL), "freeze_signature": freeze_signature,
        "source_commit": source_commit, "official_train_sha256": TRAIN_SHA256,
        "official_train_rows": len(frame), "cycle_range": [int(cycle.min()), int(cycle.max())],
        "control_artifacts": {str(k): {key: v for key, v in val.items() if key.endswith("sha256")} for k, val in controls.items()},
        "historical_C19_original_verdict": _json(ROOT / "reports/optimization_v3/batch1/C19/gate.json")["status"],
        "historical_C19_original_fullsix_metrics": old_metrics,
        "historical_C19_all_view_supported_group_safety_failures_at_0_03": old_safety,
        "historical_C19_original_0_02_recall_failures_retained": old_02_failures,
        "resource_preflight": resources, "synthetic_end_to_end": synthetic,
        "budget_before_new_fits": ledger_summary(LEDGER),
        "Test_or_SampleSubmission_accessed": False, "Kaggle_calls": 0,
        "class_only_user_reported_receipt": _reconcile_class_only(),
        "historical_source_discrepancy": _historical_source_audit(),
    }
    if preflight["status"] != "PASS":
        if write_outputs:
            _write_once(REPORT_ROOT / "preflight.json", preflight)
        return {"preflight": preflight, "freeze": freeze_payload, "frame": frame, "target": target,
                "cycle": cycle, "controls": controls, "outputs": old_outputs, "identity": stable}
    if write_outputs:
        if FREEZE.exists():
            if _json(FREEZE).get("identity") != stable or _json(FREEZE).get("freeze_signature") != freeze_signature:
                raise RuntimeError("Existing classification-next freeze differs; refusing replacement")
        else:
            _write_once(FREEZE, freeze_payload)
        _write_once(REPORT_ROOT / "synthetic_preflight.json", synthetic)
        preflight_path = REPORT_ROOT / "preflight.json"
        if preflight_path.exists():
            if _json(preflight_path).get("freeze_signature") != freeze_signature:
                raise RuntimeError("Existing preflight refers to a different frozen identity")
        else:
            _write_once(preflight_path, preflight)
    elif FREEZE.exists() and _json(FREEZE).get("identity") != stable:
        raise RuntimeError("Current stable inputs differ from the saved classification-next freeze")
    return {"preflight": preflight, "freeze": freeze_payload, "frame": frame, "target": target,
            "cycle": cycle, "controls": controls, "outputs": old_outputs, "identity": stable}


def _historical_source_audit() -> dict[str, Any]:
    old = _json(ROOT / "experiments/optimization_v3/batch1/C19/freeze.json")
    result = _json(ROOT / "reports/optimization_v3/batch1/C19/results.json")
    identity = old.get("identity", {})
    source_hashes = identity.get("source_sha256", identity.get("sources_sha256", {}))
    mismatches = []
    for rel, expected in source_hashes.items():
        declared_path = Path(rel)
        if declared_path.is_absolute():
            try:
                git_path = declared_path.resolve().relative_to(ROOT.resolve()).as_posix()
            except ValueError:
                git_path = None
        else:
            git_path = declared_path.as_posix()
        path = ROOT / git_path if git_path is not None else declared_path
        actual = _sha(path) if path.is_file() else None
        if actual != expected:
            historical = (_git_blob_sha(str(result.get("source_commit_context", "")), git_path)
                          if git_path is not None else None)
            mismatches.append({"path": rel, "frozen_sha256": expected, "current_sha256": actual,
                               "historical_commit_blob_sha256": historical,
                               "historical_blob_matches": historical == expected,
                               "git_path_normalized": git_path is not None})
    return {"historical_source_commit": result.get("source_commit_context"),
            "current_source_mismatches": mismatches,
            "interpretation": "Historical source differences are reported without modifying the C19 freeze or cache."}


def prepare() -> dict[str, Any]:
    context = _prepare_context(write_outputs=True)
    if context["preflight"]["status"] != "PASS":
        return context["preflight"]
    return {"status": "PREFLIGHT_PASS", "preflight_path": str((REPORT_ROOT / "preflight.json").relative_to(ROOT)),
            "freeze_path": str(FREEZE.relative_to(ROOT)), "freeze_signature": context["freeze"]["freeze_signature"],
            "budget_before_new_fits": context["preflight"]["budget_before_new_fits"]}


class ContinuedC19Classifier(XGBoostSVD32Classifier):
    """C19's frozen implementation with the same method body and additional approved cutoffs."""
    def __init__(self, cutoff: int):
        if cutoff not in (*NEW_C19_ORIGINS, 79):
            raise ValueError("C19 continuation accepts 59, 63, 71, 75; benchmark refit uses cutoff 79")
        self.cutoff = int(cutoff); self._fitted = False; self._fit_warnings = []; self._state = {}


def _c19_fit_function(context: Mapping[str, Any], cutoff: int) -> Callable[[], tuple[dict[str, Any], dict[str, Any]]]:
    frame = context["frame"]; target = context["target"]; cycle = context["cycle"]
    control = context["controls"][cutoff]
    rows = np.asarray(control["rows"], dtype=np.int64)
    fit_rows = np.flatnonzero(cycle <= cutoff).astype(np.int64)
    if not np.array_equal(rows, np.flatnonzero((cycle > cutoff) & (cycle <= cutoff + 4))):
        raise RuntimeError(f"C19 cutoff {cutoff} has incorrect 4-cycle scoring rows")
    fit_features = frame.iloc[fit_rows].loc[:, list(FEATURE_NAMES)].copy()
    score_features = frame.iloc[rows].loc[:, list(FEATURE_NAMES)].copy()
    estimate = __import__("scripts.optimization_v3_batch1_c19", fromlist=["_memory_estimate"])._memory_estimate(
        fit_features, score_features)
    if estimate.get("status") != "PASS":
        raise RuntimeError(f"C19 memory preflight failed immediately before fit {cutoff}: {estimate}")

    def fit() -> tuple[dict[str, Any], dict[str, Any]]:
        model = ContinuedC19Classifier(cutoff)
        started = time.perf_counter()
        with threadpool_limits(limits=THREADS):
            model.fit(fit_features, target[fit_rows])
            state = model.audit_state()
            raw = model.predict_probability(score_features)
        if state != model.audit_state():
            raise RuntimeError("C19 model audit state changed during scoring")
        pred = _route("C19", control["baseline"], raw, control["c2b"], control["cb"], control["missing"])
        payload = _payload(candidate_kind="C19", rows=rows, ids=frame.record_id.astype(str).to_numpy()[rows],
            cycles=cycle[rows], baseline=control["baseline"], raw=raw, c2b=control["c2b"], cb=control["cb"],
            missing=control["missing"], prediction=pred)
        elapsed = time.perf_counter() - started
        metadata = {"model_state": state, "fit_seconds": elapsed, "fit_rows": int(len(fit_rows)),
                    "fit_row_positions_sha256": hashlib.sha256(fit_rows.astype("<i8").tobytes()).hexdigest(),
                    "source_commit": _git_head(), "resource_preflight": estimate,
                    "validation_labels_passed_to_fit": False, "test_data_used": False,
                    "threshold": 0.5, "policy": payload["route"]}
        return payload, metadata
    return fit


def _outputs_with_new_c19(context: Mapping[str, Any], *, execute: bool) -> dict[int, dict[str, Any]]:
    outputs = dict(context["outputs"])
    freeze_signature = _json(FREEZE)["freeze_signature"]
    if execute:
        reserve_card(LEDGER, card_id="CLASSIFICATION_NEXT_V1_C19", stage="FULLSIX_4CYCLE_CONTINUATION",
                     fits=4, category="continuation")
    for cutoff in NEW_C19_ORIGINS:
        if not execute:
            raise RuntimeError("C19 continuation output is not cached; execution flag required")
        identity = _identity(cutoff, "FULLSIX_4CYCLE_CONTINUATION", "C19", freeze_signature)
        destination = RUN_ROOT / "C19_CONTINUATION" / f"cutoff_{cutoff}"
        fit_id = f"CLASSIFICATION_NEXT_V1_C19_{cutoff}"
        saved = _persist_fit(destination, fit_id, "CLASSIFICATION_NEXT_V1_C19", "FULLSIX_4CYCLE_CONTINUATION",
                             cutoff, "C19", identity, _c19_fit_function(context, cutoff))
        arrays = saved["arrays"]
        outputs[cutoff] = {key: arrays[key] for key in (
            "score_rows", "score_record_ids", "cycle_code", "prediction", "baseline_prediction")}
        outputs[cutoff]["origin"] = np.full(len(arrays["score_rows"]), cutoff, dtype=np.int16)
        outputs[cutoff]["receipt_sha256"] = saved["receipt"]["receipt_sha256"]
    if execute:
        release_unused(LEDGER, card_id="CLASSIFICATION_NEXT_V1_C19", stage="FULLSIX_4CYCLE_CONTINUATION")
    return outputs


def run_c19() -> dict[str, Any]:
    context = _prepare_context(write_outputs=False)
    if context["preflight"]["status"] != "PASS":
        return {"status": "NO_GO_PRECHECK", "preflight": context["preflight"]}
    if not FREEZE.is_file() or _json(FREEZE).get("identity") != context["identity"]:
        raise RuntimeError("C19 fit requires the committed prefit input/source freeze")
    outputs = _outputs_with_new_c19(context, execute=True)
    metrics, groups, per_cycle = _views(outputs, context["frame"], context["target"], C19_VIEWS)
    gate = _c19_gate(metrics, groups)
    fit_receipts = {}
    for cutoff in NEW_C19_ORIGINS:
        receipt_path = RUN_ROOT / "C19_CONTINUATION" / f"cutoff_{cutoff}" / "receipt.json"
        fit_receipts[str(cutoff)] = {"receipt_sha256": _json(receipt_path)["receipt_sha256"],
                                     "npz_sha256": _sha(receipt_path.parent / "arrays.npz")}
    dest = REPORT_ROOT / "C19_CONTINUATION"
    budget_after_lane = _stage_budget_snapshot("C19_CONTINUATION", dest)["ledger_summary"]
    result = {
        "status": "C19_CONTINUATION_COMPLETE", "original_C19_verdict": "RETIRE",
        "original_verdict_changed": False, "new_continuation_status": gate["status"],
        "freeze_signature": _json(FREEZE)["freeze_signature"], "protocol_sha256": _sha(PROTOCOL),
        "official_train_sha256": TRAIN_SHA256, "official_train_rows": 50_608,
        "views": metrics, "gate": gate, "subgroup_support_rule": "n>=200 and each class count>=30",
        "subgroup_guard_risk_ceiling": {"macro_F1_harm": 0.01, "positive_recall_harm": 0.03},
        "groups_path": "reports/optimization_v3/classification_next_v1/C19_CONTINUATION/groups.csv",
        "per_cycle_path": "reports/optimization_v3/classification_next_v1/C19_CONTINUATION/per_cycle.csv",
        "fit_receipts": fit_receipts, "historical_pilot_55_67_reused": True,
        "new_origins_fit": list(NEW_C19_ORIGINS), "official_new_estimator_fits": 4,
        "threshold_or_weight_search": False, "Test_or_SampleSubmission_accessed": False,
        "Kaggle_calls": 0, "budget_after_lane": budget_after_lane,
    }
    _write_once(dest / "results.json", result)
    _write_csv_once(dest / "groups.csv", groups, ["view", "field", "group", "n", "class_0_n", "class_1_n", "supported",
        "candidate_macro_f1", "sparse25_macro_f1", "f1_harm", "candidate_positive_recall", "sparse25_positive_recall",
        "positive_recall_harm", "fit_origins", "fails_0_01_f1", "fails_0_03_recall"])
    _write_csv_once(dest / "per_cycle.csv", per_cycle, ["view", "cycle", "n", "candidate_macro_f1", "sparse25_macro_f1", "gain_macro_f1"])
    return result


def run_c19_fulltrain() -> dict[str, Any]:
    """Conditional one-fit C19 benchmark CSV; all Test access is behind verified frozen gates."""
    c19_path = REPORT_ROOT / "C19_CONTINUATION" / "results.json"
    verify_path = REPORT_ROOT / "C19_CONTINUATION" / "independent_verification.json"
    if not c19_path.is_file() or not verify_path.is_file():
        return {"status": "SKIPPED_GATE_OR_VERIFICATION_NOT_READY", "Test_or_SampleSubmission_accessed": False}
    c19 = _json(c19_path); verified = _json(verify_path)
    if c19.get("gate", {}).get("status") != "PASS_ALL_FROZEN_GATES" or verified.get("status") != "PASS":
        return {"status": "NO_GO_FULLTRAIN_GATES_NOT_ALL_PASS", "Test_or_SampleSubmission_accessed": False,
                "c19_gate": c19.get("gate", {}).get("status"), "independent_verification": verified.get("status")}
    context = _prepare_context(write_outputs=False)
    if context["preflight"]["status"] != "PASS" or not FREEZE.is_file() or _json(FREEZE).get("identity") != context["identity"]:
        raise RuntimeError("C19 full-Train benchmark requires the verified research freeze")

    task_paths = task_files("classification")
    test_path = Path(task_paths["test"]) if task_paths.get("test") else None
    sample_path = Path(task_paths["sample"]) if task_paths.get("sample") else None
    if test_path is None or sample_path is None or not test_path.is_file() or not sample_path.is_file():
        return {"status": "BLOCKED_OFFICIAL_TEST_OR_SAMPLE_MISSING", "Test_or_SampleSubmission_accessed": False}
    required = ["record_id", "cycle_code", *FEATURE_NAMES]
    # Explicit usecols keeps any target-like Test column out of memory and out of the fit path.
    test = pd.read_csv(test_path, usecols=list(dict.fromkeys(required)))
    sample = pd.read_csv(sample_path)
    expected_columns = ["record_id", "artifact_signal"]
    if list(sample.columns) != expected_columns or len(sample) != 29_421:
        raise RuntimeError("SampleSubmission columns or row count differ from the frozen task format")
    if len(test) != 29_421 or test.record_id.isna().any() or not test.record_id.astype(str).is_unique:
        raise RuntimeError("Official Test row count or unique record_id check failed")
    ids = test.record_id.astype(str).to_numpy()
    if not np.array_equal(ids, sample.record_id.astype(str).to_numpy()):
        raise RuntimeError("Test ordered IDs differ from SampleSubmission")
    test_cycles = pd.to_numeric(test.cycle_code, errors="raise").to_numpy(dtype=np.int64)
    if not np.array_equal(np.unique(test_cycles), np.arange(80, 92)):
        raise RuntimeError("Official Test cycle range differs from 80-91")

    base_manifest = _json(ROOT / "reports/optimization_v3/kaggle/v3_cls_sparse25.json")
    class_manifest = _json(CLASS_ONLY_MANIFEST)
    base_cache = ROOT / "experiments/runs/optimization_v3_inference/v3_cls_sparse25.npz"
    if _sha(base_cache) != class_manifest["reuse_audit"]["cache_sha256"]:
        raise RuntimeError("Existing full-Train sparse25 inference cache hash differs from verified manifest")
    with np.load(base_cache, allow_pickle=False) as archive:
        cache_keys = set(archive.files)
        probability_key = next((name for name in ("probability", "probabilities", "prediction_probability") if name in cache_keys), None)
        if probability_key is None:
            raise RuntimeError(f"Existing sparse25 inference cache lacks probability values: {sorted(cache_keys)}")
        current_sparse25 = np.asarray(archive[probability_key], dtype=np.float64)
        cache_ids = np.asarray(archive["record_id"], dtype=str) if "record_id" in cache_keys else None
    if current_sparse25.shape != (len(test),) or not np.isfinite(current_sparse25).all() or np.any((current_sparse25 < 0) | (current_sparse25 > 1)):
        raise RuntimeError("Existing full-Train sparse25 probabilities are invalid")
    from src.models.round1_evaluation import array_digest
    if array_digest(current_sparse25) != base_manifest["probability_sha256"]:
        raise RuntimeError("Existing sparse25 probability array hash differs from the accepted manifest")
    if cache_ids is not None and not np.array_equal(cache_ids, ids):
        raise RuntimeError("Existing sparse25 inference IDs are not aligned to official Test")

    bundle_path = ROOT / class_manifest["fit_bundle"]
    source_receipt = _json(bundle_path / "receipt.json")
    source_bundle = read_bundle(bundle_path, source_receipt["identity"], task="classification")
    source_arrays = source_bundle.arrays
    id_candidates = [name for name in ("score_record_ids", "test_record_ids", "record_ids") if name in source_arrays]
    if not id_candidates or not np.array_equal(source_arrays[id_candidates[0]].astype(str), ids):
        raise RuntimeError("Attested full-Train control bundle lacks exact official Test ID alignment")
    def locate(*names: str) -> np.ndarray:
        for name in names:
            if name in source_arrays:
                return np.asarray(source_arrays[name])
        raise RuntimeError(f"Attested full-Train controls lack any of {names}")
    c2b = locate("control_c2b_probability", "control__c2b_probability").astype(np.float64)
    cb = locate("control_cb_probability", "control__cb_probability").astype(np.float64)
    missing = locate("control_stream_missing", "control__stream_missing").astype(bool)
    sparse_raw = locate("control_baseline_sparse_raw", "control__baseline_sparse_raw").astype(np.float64)
    from src.models.classification_round2_models import stream_missing_mask
    observed_missing = stream_missing_mask(test).astype(bool)
    if not np.array_equal(missing, observed_missing):
        raise RuntimeError("Attested full-Train missing-stream mask differs from official Test text availability")
    baseline_reconstructed = np.where(missing, cb, 0.75 * c2b + 0.25 * sparse_raw)
    if baseline_reconstructed.shape != current_sparse25.shape or not np.allclose(
            baseline_reconstructed, current_sparse25, atol=1e-12, rtol=0):
        raise RuntimeError("Full-Train C2_B/CB/sparse controls do not reconstruct accepted sparse25 probabilities")

    fit_frame, target, train_cycle = _train()
    fit_features = fit_frame.loc[:, list(FEATURE_NAMES)].copy()
    score_features = test.loc[:, list(FEATURE_NAMES)].copy()
    resource = __import__("scripts.optimization_v3_batch1_c19", fromlist=["_memory_estimate"])._memory_estimate(
        fit_features, score_features)
    if resource.get("status") != "PASS":
        return {"status": "BLOCKED_RESOURCE", "resource_preflight": resource,
                "Test_or_SampleSubmission_accessed": True, "fulltrain_fit_started": False}
    test_sha = _sha(test_path); sample_sha = _sha(sample_path)
    inference_identity = {
        **_identity(79, "FULLTRAIN_C19_BENCHMARK", "C19", _json(FREEZE)["freeze_signature"]),
        "official_test_sha256": test_sha, "sample_submission_sha256": sample_sha,
        "sparse25_probability_cache_sha256": _sha(base_cache),
        "fulltrain_controls_bundle_receipt_sha256": source_bundle.receipt["receipt_sha256"],
        "fulltrain_controls_bundle_npz_sha256": _sha(bundle_path / "arrays.npz"),
        "ordered_test_ids_sha256": hashlib.sha256("\n".join(ids).encode("utf-8")).hexdigest(),
    }
    freeze_path = ROOT / "experiments/optimization_v3/classification_next_v1/C19_FULLTRAIN_FREEZE.json"
    freeze_obj = {"status": "FROZEN_AFTER_ALL_C19_GATES_AND_BEFORE_FULLTRAIN_FIT",
        "identity": inference_identity, "freeze_signature": canonical_sha256(inference_identity),
        "source_commit": _git_head(), "training_rows": int(len(target)), "test_rows": int(len(test)),
        "test_labels_read": False, "test_target_column_loaded": False, "test_preprocessing_fit": False}
    if freeze_path.exists():
        if _json(freeze_path).get("identity") != inference_identity:
            raise RuntimeError("Existing C19 full-Train inference freeze differs")
    else:
        _write_once(freeze_path, freeze_obj)

    reserve_card(LEDGER, card_id="CLASSIFICATION_NEXT_V1_C19", stage="FULLTRAIN_BENCHMARK",
                 fits=1, category="continuation")
    fit_id = "CLASSIFICATION_NEXT_V1_C19_FULLTRAIN"
    def full_fit() -> tuple[dict[str, Any], dict[str, Any]]:
        model = ContinuedC19Classifier(79)
        start = time.perf_counter()
        with threadpool_limits(limits=THREADS):
            model.fit(fit_features, target)
            state = model.audit_state()
            raw = model.predict_probability(score_features)
        prediction = _route("C19", baseline_reconstructed, raw, c2b, cb, missing)
        payload = _payload(candidate_kind="C19", rows=np.arange(len(test), dtype=np.int64), ids=ids,
            cycles=test_cycles, baseline=baseline_reconstructed, raw=raw, c2b=c2b, cb=cb,
            missing=missing, prediction=prediction)
        return payload, {"model_state": state, "fit_rows": int(len(target)),
            "fit_cycle_min": int(train_cycle.min()), "fit_cycle_max": int(train_cycle.max()),
            "fit_seconds": time.perf_counter() - start, "resource_preflight": resource,
            "source_commit": _git_head(), "validation_labels_passed_to_fit": False,
            "test_labels_read": False, "test_target_column_loaded": False,
            "learned_preprocessing_fit_only_on_train": True, "threshold": 0.5,
            "policy": "present=.75*currentSparse25+.25*C19; missing=attestedCB"}
    saved = _persist_fit(RUN_ROOT / "C19_FULLTRAIN", fit_id, "CLASSIFICATION_NEXT_V1_C19",
        "FULLTRAIN_BENCHMARK", 79, "C19", inference_identity, full_fit)
    predictions = saved["arrays"]["prediction"]
    labels = (predictions >= 0.5).astype(np.int8)
    csv_path = ROOT / "submissions/optimization_v3/classification_next_v1/C19.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    output = pd.DataFrame({"record_id": ids, "artifact_signal": labels})
    if list(output.columns) != expected_columns or not np.isin(output.artifact_signal, [0, 1]).all():
        raise RuntimeError("C19 submission output schema or labels are invalid")
    if csv_path.exists():
        old = pd.read_csv(csv_path)
        if not old.equals(output):
            raise RuntimeError("Existing C19 full-Train CSV differs; refusing overwrite")
    else:
        output.to_csv(csv_path, index=False)
    check = pd.read_csv(csv_path)
    if not check.equals(output) or len(check) != 29_421:
        raise RuntimeError("C19 output CSV byte round-trip QA failed")
    csv_sha = _sha(csv_path)
    manifest = {"status": "PREPARED_VALIDATED_NOT_SUBMITTED", "candidate": "C19_CONTINUATION",
        "competition": "tp2-classification-cyber-vulnerability-kasdd-gasal-2026-2027",
        "csv_path": csv_path.relative_to(ROOT).as_posix(), "csv_sha256": csv_sha,
        "rows": len(check), "columns": list(check.columns), "labels": {str(k): int(v) for k, v in check.artifact_signal.value_counts().sort_index().items()},
        "source_commit": _git_head(), "research_freeze_signature": _json(FREEZE)["freeze_signature"],
        "fulltrain_freeze_signature": freeze_obj["freeze_signature"],
        "recipe": "Exact C19 XGBoost binary logistic + structured/OHE + fit-only TF-IDF/SVD32; present .75 full-Train sparse25 probability + .25 C19; missing attested frozen CB; threshold .50",
        "fit_rows": len(target), "test_rows": len(test), "fit_receipt_sha256": saved["receipt"]["receipt_sha256"],
        "input_sha256": {"train": _sha(TRAIN_PATH), "test": test_sha, "sample_submission": sample_sha,
            "protocol": _sha(PROTOCOL), "sparse25_probability_cache": _sha(base_cache),
            "fulltrain_controls_bundle_npz": _sha(bundle_path / "arrays.npz"),
            "fulltrain_controls_bundle_receipt": _sha(bundle_path / "receipt.json")},
        "ordered_test_ids_match_sample": True, "duplicate_or_missing_test_ids": False,
        "labels_integer_0_1": True, "preprocessing_fit_only_on_train": True,
        "test_labels_used": False, "test_target_column_read": False,
        "kaggle_called": False, "submission_ref": None, "public_score": None}
    _write_once(REPORT_ROOT / "C19_CONTINUATION" / "submission_manifest.json", manifest)
    return manifest


class SparseTextTTRBoostClassifier:
    """Original unweighted sparse LightGBM design plus the frozen scalar arm."""
    input_feature_names = tuple(FEATURE_NAMES)

    def __init__(self, arm: str):
        if arm not in {"COUNT", "BOTH"}:
            raise ValueError("TTR arm must be COUNT or BOTH")
        self.arm = arm; self.fitted = False

    def _select(self, frame: pd.DataFrame) -> pd.DataFrame:
        if frame.columns.has_duplicates or set(frame.columns) != set(self.input_feature_names):
            raise ValueError("TTR tree requires the exact original feature allowlist")
        return frame.loc[:, list(self.input_feature_names)].copy()

    def _design(self, frame: pd.DataFrame, *, fit: bool) -> tuple[sparse.csr_matrix, np.ndarray, np.ndarray]:
        frame = self._select(frame)
        if fit:
            (structured, self.processor, self.scaler, self.encoder, self.numeric_positions,
             structured_state) = RegressionTextModel._structured_fit(frame)
            self.structured_state = structured_state
            docs, missing = _text_values(frame["signal_stream"])
            self.vectorizer = TfidfVectorizer(**dict(_TFIDF_PARAMS))
            text = self.vectorizer.fit_transform(docs).tocsr()
        else:
            structured, _unknown = RegressionTextModel._structured_transform(
                frame, self.processor, self.scaler, self.encoder, self.numeric_positions)
            docs, missing = _text_values(frame["signal_stream"])
            text = self.vectorizer.transform(docs).tocsr()
        token_count, token_unique, ttr = _token_features(frame["signal_stream"])
        missing_block = sparse.csr_matrix(np.asarray(missing, dtype=np.float64).reshape(-1, 1))
        base = sparse.hstack([structured, text, missing_block], format="csr", dtype=np.float64)
        base_hash = hashlib.sha256(base.data.tobytes() + base.indices.tobytes() + base.indptr.tobytes()).hexdigest()
        self.base_design_sha256 = base_hash
        self.base_design_shape = list(base.shape)
        scalar_columns = [token_count.astype(np.float64)]
        if self.arm == "BOTH":
            # Preserve undefined TTR as an explicit NaN in CSR for LightGBM's native missing handling.
            scalar_columns.append(ttr)
        scalar = sparse.csr_matrix(np.column_stack(scalar_columns), dtype=np.float64)
        design = sparse.hstack([base, scalar], format="csr", dtype=np.float64)
        if design.shape[0] != len(frame) or design.dtype != np.float64:
            raise RuntimeError("TTR sparse design is misaligned or has an unexpected dtype")
        if self.arm == "BOTH" and token_count.size:
            expected_undefined = token_count == 0
            actual_ttr = ttr
            if not np.array_equal(np.isnan(actual_ttr), expected_undefined):
                raise RuntimeError("TTR undefined mask differs from zero-token rows")
        return design, token_count, token_unique

    def fit(self, frame: pd.DataFrame, target: np.ndarray) -> "SparseTextTTRBoostClassifier":
        self.fitted = False
        selected = self._select(frame)
        y = np.asarray(target, dtype=np.int8)
        if y.shape != (len(selected),) or not np.isin(y, [0, 1]).all() or len(np.unique(y)) != 2:
            raise ValueError("TTR fit requires an aligned binary target with both classes")
        design, counts, uniques = self._design(selected, fit=True)
        explicit_zero = int(np.sum(counts == 0))
        self.model = SparseTextBoostClassifier  # immutable source marker; no call to its fit method
        from lightgbm import LGBMClassifier
        from src.models.classification_models import _array_digest
        from src.models.round1_models import _fit_fingerprint
        from src.models.optimization_v3_sparse_classification import jsonhash
        params = dict(SPARSE_PARAMS)
        started = time.perf_counter()
        with threadpool_limits(limits=THREADS):
            self.estimator = LGBMClassifier(**params).fit(design, y)
        ordered = [(str(token), int(index)) for token, index in sorted(self.vectorizer.vocabulary_.items(), key=lambda item: item[1])]
        self.fit_state = {
            "model": "unweighted original SparseTextBoostClassifier LightGBM parameters",
            "arm": self.arm, "parameters": params, "input_feature_names": list(self.input_feature_names),
            "fit_rows": int(len(y)), "fit_data_sha256": _fit_fingerprint(selected, self.input_feature_names, y),
            "target_sha256": _array_digest(y), "vocabulary_size": len(ordered),
            "vocabulary_sha256": jsonhash(ordered), "idf_sha256": _array_digest(self.vectorizer.idf_),
            "structured_state": self.structured_state, "base_sparse_matrix_shape": self.base_design_shape,
            "base_sparse_matrix_sha256": self.base_design_sha256,
            "zero_token_rows_in_fit": explicit_zero, "tokenizer_rule": "lowercase regex [A-Za-z0-9]+",
            "count_definition": "token occurrences; zero when no tokens",
            "TTR_definition": "unique token count / token count when count>0; NaN for count=0",
            "undefined_ttr_policy": "LightGBM native NaN handling; no imputation or extra indicator",
            "fit_duration_seconds": time.perf_counter() - started,
            "eval_set_used": False, "early_stopping_used": False, "class_weight_used": False,
            "sample_weight_used": False, "validation_labels_passed_to_fit": False,
        }
        self.fitted = True
        return self

    def predict_proba(self, frame: pd.DataFrame) -> np.ndarray:
        if not self.fitted:
            raise RuntimeError("TTR tree must be fitted before prediction")
        design, _, _ = self._design(frame, fit=False)
        probability = np.asarray(self.estimator.predict_proba(design)[:, 1], dtype=np.float64)
        if not np.isfinite(probability).all() or np.any((probability < 0) | (probability > 1)):
            raise RuntimeError("TTR tree returned invalid probabilities")
        return probability


def _ttr_fit_function(context: Mapping[str, Any], cutoff: int, arm: str) -> Callable[[], tuple[dict[str, Any], dict[str, Any]]]:
    frame = context["frame"]; target = context["target"]; cycle = context["cycle"]
    control = context["controls"][cutoff]
    rows = np.asarray(control["rows"], dtype=np.int64)
    if not np.array_equal(rows, np.flatnonzero((cycle > cutoff) & (cycle <= cutoff + 12))):
        raise RuntimeError(f"TTR origin {cutoff} must score its exact next 12 cycles")
    fit_rows = np.flatnonzero(cycle <= cutoff).astype(np.int64)
    fit_features = frame.iloc[fit_rows].loc[:, list(FEATURE_NAMES)].copy()
    score_features = frame.iloc[rows].loc[:, list(FEATURE_NAMES)].copy()

    def fit() -> tuple[dict[str, Any], dict[str, Any]]:
        model = SparseTextTTRBoostClassifier(arm)
        start = time.perf_counter()
        model.fit(fit_features, target[fit_rows])
        state = model.fit_state
        raw = model.predict_proba(score_features)
        pred = _route(f"TTR_{arm}", control["baseline"], raw, control["c2b"], control["cb"], control["missing"])
        counts, uniques, _ttr = _token_features(score_features["signal_stream"])
        payload = _payload(candidate_kind=f"TTR_{arm}", rows=rows,
            ids=frame.record_id.astype(str).to_numpy()[rows], cycles=cycle[rows], baseline=control["baseline"],
            raw=raw, c2b=control["c2b"], cb=control["cb"], missing=control["missing"],
            count=counts, unique=uniques, prediction=pred)
        return payload, {"model_state": state, "fit_seconds": time.perf_counter() - start,
            "fit_rows": int(len(fit_rows)), "fit_row_positions_sha256": hashlib.sha256(fit_rows.astype("<i8").tobytes()).hexdigest(),
            "source_commit": _git_head(), "validation_labels_passed_to_fit": False,
            "test_data_used": False, "threshold": 0.5, "policy": payload["route"]}
    return fit


def _outputs_for_ttr(context: Mapping[str, Any], *, execute: bool) -> dict[str, dict[int, dict[str, Any]]]:
    outputs: dict[str, dict[int, dict[str, Any]]] = {"COUNT": {}, "BOTH": {}}
    if not execute:
        raise RuntimeError("TTR pilot is not cached; execution flag required")
    reserve_card(LEDGER, card_id="CLASSIFICATION_NEXT_V1_TREE_TTR", stage="PILOT_REALLOCATED_CONTINUATION_SLOTS",
                 fits=4, category="continuation")
    freeze_signature = _json(FREEZE)["freeze_signature"]
    for arm in ("COUNT", "BOTH"):
        for cutoff in TTR_ORIGINS:
            identity = _identity(cutoff, "PILOT_REALLOCATED_CONTINUATION_SLOTS", f"TTR_{arm}", freeze_signature)
            destination = RUN_ROOT / "TREE_TTR_PILOT" / arm / f"cutoff_{cutoff}"
            fit_id = f"CLASSIFICATION_NEXT_V1_TTR_{arm}_{cutoff}"
            saved = _persist_fit(destination, fit_id, "CLASSIFICATION_NEXT_V1_TREE_TTR",
                "PILOT_REALLOCATED_CONTINUATION_SLOTS", cutoff, f"TTR_{arm}", identity,
                _ttr_fit_function(context, cutoff, arm))
            arrays = saved["arrays"]
            outputs[arm][cutoff] = {key: arrays[key] for key in (
                "score_rows", "score_record_ids", "cycle_code", "prediction", "baseline_prediction")}
            outputs[arm][cutoff]["origin"] = np.full(len(arrays["score_rows"]), cutoff, dtype=np.int16)
            outputs[arm][cutoff]["receipt_sha256"] = saved["receipt"]["receipt_sha256"]
    release_unused(LEDGER, card_id="CLASSIFICATION_NEXT_V1_TREE_TTR", stage="PILOT_REALLOCATED_CONTINUATION_SLOTS")
    return outputs


def _matched_ttr(outputs: Mapping[str, Mapping[int, Mapping[str, Any]]], frame: pd.DataFrame,
                 target: np.ndarray) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    all_metrics = {}; matched = {}; all_groups = []; all_cycles = []
    for arm in ("COUNT", "BOTH"):
        metrics, groups, cycles = _views(outputs[arm], frame, target, TTR_VIEWS)
        all_metrics[arm] = metrics
        all_groups.extend({**row, "arm": arm} for row in groups)
        all_cycles.extend({**row, "arm": arm} for row in cycles)
    for name in TTR_VIEWS:
        count_metric = all_metrics["COUNT"][name]["candidate"]
        both_metric = all_metrics["BOTH"][name]["candidate"]
        matched[name] = {
            metric: (None if both_metric[metric] is None or count_metric[metric] is None
                     else float(both_metric[metric] - count_metric[metric]))
            for metric in ("macro_f1", "positive_recall", "roc_auc", "average_precision", "log_loss",
                           "brier", "predicted_positive_rate", "mean_probability")
        }
    return all_metrics, matched, all_groups, all_cycles


def _matched_ttr_groups(groups: list[dict[str, Any]]) -> list[dict[str, Any]]:
    keyed = {(row["arm"], row["view"], row["field"], row["group"]): row for row in groups}
    paired = []
    for key in sorted((view, field, group) for arm, view, field, group in keyed if arm == "COUNT"):
        count = keyed[("COUNT", *key)]
        both = keyed[("BOTH", *key)]
        for field in ("n", "class_0_n", "class_1_n", "supported", "fit_origins"):
            if count[field] != both[field]:
                raise RuntimeError(f"Matched TTR group support differs between arms for {key}: {field}")
        recall_delta = (None if count["candidate_positive_recall"] is None or both["candidate_positive_recall"] is None
                        else float(both["candidate_positive_recall"] - count["candidate_positive_recall"]))
        paired.append({"view": key[0], "field": key[1], "group": key[2], "n": count["n"],
            "class_0_n": count["class_0_n"], "class_1_n": count["class_1_n"],
            "supported": count["supported"], "fit_origins": count["fit_origins"],
            "count_macro_f1": count["candidate_macro_f1"], "both_macro_f1": both["candidate_macro_f1"],
            "delta_macro_f1_both_minus_count": float(both["candidate_macro_f1"] - count["candidate_macro_f1"]),
            "count_positive_recall": count["candidate_positive_recall"],
            "both_positive_recall": both["candidate_positive_recall"],
            "delta_positive_recall_both_minus_count": recall_delta})
    if len(keyed) != 2 * len(paired):
        raise RuntimeError("Matched TTR arms do not have identical group-key support")
    return paired


def run_ttr() -> dict[str, Any]:
    context = _prepare_context(write_outputs=False)
    if context["preflight"]["status"] != "PASS" or not FREEZE.is_file() or _json(FREEZE).get("identity") != context["identity"]:
        raise RuntimeError("TTR pilot requires the passing frozen preflight")
    c19_result_path = REPORT_ROOT / "C19_CONTINUATION" / "results.json"
    c19_verify_path = REPORT_ROOT / "C19_CONTINUATION" / "independent_verification.json"
    if not c19_result_path.is_file():
        raise RuntimeError("C19 continuation lane must be completed before TTR pilot")
    # C19 scientific NO_GO is not a blocker for this independent, pre-authorized TTR pilot.
    outputs = _outputs_for_ttr(context, execute=True)
    all_metrics, matched, groups, per_cycle = _matched_ttr(outputs, context["frame"], context["target"])
    candidate_metrics = all_metrics["BOTH"]
    candidate_groups = [row for row in groups if row["arm"] == "BOTH"]
    gate = _ttr_gate(candidate_metrics, matched, candidate_groups)
    matched_groups = _matched_ttr_groups(groups)
    fit_receipts = {}
    for arm in ("COUNT", "BOTH"):
        for cutoff in TTR_ORIGINS:
            receipt_path = RUN_ROOT / "TREE_TTR_PILOT" / arm / f"cutoff_{cutoff}" / "receipt.json"
            fit_receipts[f"{arm}_{cutoff}"] = {"receipt_sha256": _json(receipt_path)["receipt_sha256"],
                "npz_sha256": _sha(receipt_path.parent / "arrays.npz")}
    c19_verify_status = _json(c19_verify_path).get("status") if c19_verify_path.exists() else "PENDING"
    dest = REPORT_ROOT / "TREE_TTR_PILOT"
    budget_after_lane = _stage_budget_snapshot("TREE_TTR_PILOT", dest)["ledger_summary"]
    result = {"status": "TTR_TREE_PILOT_COMPLETE", "diagnostic_decision": gate["status"],
        "pilot_stops_for_review": True, "automatic_continuation_or_fulltrain": False,
        "metrics_by_arm": all_metrics, "matched_gain_BOTH_minus_COUNT": matched, "gate": gate,
        "matched_group_rows": len(matched_groups), "matched_groups_path": "reports/optimization_v3/classification_next_v1/TREE_TTR_PILOT/matched_groups.csv",
        "matched_metric_delta_definition": "BOTH minus COUNT; positive log_loss/Brier delta means worse probability fit.",
        "fit_receipts": fit_receipts, "official_new_estimator_fits": 4,
        "origins": list(TTR_ORIGINS), "scoring_horizon_cycles_each": 12,
        "tokenizer": "lowercase regex [A-Za-z0-9]+", "fit_missing_or_zero_token_ttr": "NaN; LightGBM native missing handling",
        "threshold": 0.5, "class_weight_or_sample_weight": False,
        "Test_or_SampleSubmission_accessed": False, "Kaggle_calls": 0,
        "C19_independent_verification_status_when_TTR_started": c19_verify_status,
        "budget_after_lane": budget_after_lane}
    _write_once(dest / "results.json", result)
    _write_csv_once(dest / "groups.csv", groups, ["arm", "view", "field", "group", "n", "class_0_n", "class_1_n", "supported",
        "candidate_macro_f1", "sparse25_macro_f1", "f1_harm", "candidate_positive_recall", "sparse25_positive_recall",
        "positive_recall_harm", "fit_origins", "fails_0_01_f1", "fails_0_03_recall", "fails_0_02_recall"])
    _write_csv_once(dest / "per_cycle.csv", per_cycle, ["arm", "view", "cycle", "n", "candidate_macro_f1", "sparse25_macro_f1", "gain_macro_f1"])
    _write_csv_once(dest / "matched_groups.csv", matched_groups, ["view", "field", "group", "n", "class_0_n", "class_1_n", "supported",
        "fit_origins", "count_macro_f1", "both_macro_f1", "delta_macro_f1_both_minus_count",
        "count_positive_recall", "both_positive_recall", "delta_positive_recall_both_minus_count"])
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--prepare", action="store_true")
    modes.add_argument("--run-c19", action="store_true")
    modes.add_argument("--run-c19-fulltrain", action="store_true")
    modes.add_argument("--run-ttr", action="store_true")
    args = parser.parse_args()
    result = prepare() if args.prepare else run_c19() if args.run_c19 else run_c19_fulltrain() if args.run_c19_fulltrain else run_ttr()
    print(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
