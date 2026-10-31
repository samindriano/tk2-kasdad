"""Independent output/metric audit for full-six classification Stage 2."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, confusion_matrix, f1_score, precision_recall_fscore_support

from src.data.audit import sha256 as file_sha256, write_json
from src.models.classification_models import _fit_fingerprint
from src.models.classification_round2_models import stream_missing_mask
from src.models.round1_evaluation import array_digest
from scripts.research_campaign_classification_stage2 import (
    ALL_FIT_ENDS,
    CANDIDATES,
    NEW_FIT_ENDS,
    PRIMARY_WINDOWS,
    REPORT_DIR,
    RUN_DIR,
    STAGE1_REPORT_DIR,
    STAGE1_RUN_DIR,
    _expected_rows,
    _load_round1_control,
    blend_gate,
    compact_metrics,
    complementarity_counts,
    load_train,
    score_rows,
    verify_freeze,
)


def _close(actual: float, expected: float, label: str, atol: float = 1e-12) -> None:
    if not np.isclose(float(actual), float(expected), atol=atol, rtol=0):
        raise AssertionError(f"{label}: {actual:.16g} != {expected:.16g}")


def independent_metrics(y: np.ndarray, probability: np.ndarray) -> dict[str, Any]:
    actual = np.asarray(y, dtype=np.int8)
    p = np.asarray(probability, dtype=float)
    if actual.ndim != 1 or p.shape != actual.shape or not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise AssertionError("Invalid independent metric input")
    predicted = (p >= 0.50).astype(np.int8)
    precision, recall, f1, support = precision_recall_fscore_support(
        actual, predicted, labels=[0, 1], zero_division=0
    )
    return {
        "n": int(len(actual)),
        "macro_f1": float(f1_score(actual, predicted, labels=[0, 1], average="macro", zero_division=0)),
        "confusion_matrix": confusion_matrix(actual, predicted, labels=[0, 1]).tolist(),
        "per_class": [
            {"label": label, "support": int(support[i]), "precision": float(precision[i]),
             "recall": float(recall[i]), "f1": float(f1[i])}
            for i, label in enumerate([0, 1])
        ],
        "positive_rate": float(actual.mean()) if len(actual) else None,
        "positive_prediction_rate": float(predicted.mean()) if len(predicted) else None,
        "average_precision": float(average_precision_score(actual, p)) if len(np.unique(actual)) == 2 else None,
        "brier_score": float(np.mean((p - actual) ** 2)) if len(actual) else None,
        "false_positive": int(((actual == 0) & (predicted == 1)).sum()),
        "false_negative": int(((actual == 1) & (predicted == 0)).sum()),
    }


def _assert_metric(record: dict[str, Any], y: np.ndarray, p: np.ndarray, label: str) -> None:
    expected = independent_metrics(y, p)
    for field in ("n", "confusion_matrix", "false_positive", "false_negative"):
        if record[field] != expected[field]:
            raise AssertionError(f"{label}.{field} differs")
    for field in ("macro_f1", "positive_rate", "positive_prediction_rate", "average_precision", "brier_score"):
        if expected[field] is None:
            if record[field] is not None:
                raise AssertionError(f"{label}.{field} should be null")
        else:
            _close(record[field], expected[field], f"{label}.{field}")
    if len(record["per_class"]) != 2:
        raise AssertionError(f"{label}.per_class missing")
    for i, current in enumerate(record["per_class"]):
        reference = expected["per_class"][i]
        if current["label"] != reference["label"] or current["support"] != reference["support"]:
            raise AssertionError(f"{label}.class{i}.support differs")
        for field in ("precision", "recall", "f1"):
            _close(current[field], reference[field], f"{label}.class{i}.{field}")


def _load_stage1_window(frame: pd.DataFrame, recipe: str, fit_end: int,
                        start: int, end: int) -> dict[str, np.ndarray]:
    path = STAGE1_RUN_DIR / f"fit{fit_end}__{recipe}.npz"
    with np.load(path, allow_pickle=False) as archive:
        all_rows = archive["outer_rows"].astype(np.int64, copy=True)
        candidate = archive["candidate_probability"].astype(float, copy=True)
        routed = archive["routed_probability"].astype(float, copy=True)
        c4 = archive["c4_probability"].astype(float, copy=True)
        missing = archive["missing_stream_mask"].astype(bool, copy=True)
    cycles = pd.to_numeric(frame.cycle_code).to_numpy(dtype=int)[all_rows]
    selected = (cycles >= start) & (cycles <= end)
    return {"rows": all_rows[selected], "candidate_probability": candidate[selected],
            "routed_probability": routed[selected], "c4_probability": c4[selected],
            "missing_stream_mask": missing[selected]}


def _load_window(frame: pd.DataFrame, recipe: str, window: str,
                 fit_end: int, start: int, end: int) -> dict[str, np.ndarray]:
    if fit_end in (55, 67):
        return _load_stage1_window(frame, recipe, fit_end, start, end)
    path = RUN_DIR / f"fit{fit_end}__{recipe}.npz"
    with np.load(path, allow_pickle=False) as archive:
        rows = archive["outer_rows"].astype(np.int64, copy=True)
        candidate = archive["candidate_probability"].astype(float, copy=True)
        routed = archive["routed_probability"].astype(float, copy=True)
        c4 = archive["c4_probability"].astype(float, copy=True)
        missing = archive["missing_stream_mask"].astype(bool, copy=True)
    if not np.array_equal(rows, _expected_rows(frame, fit_end, end)):
        raise AssertionError(f"{window}/{recipe}: validation row identity differs")
    return {"rows": rows, "candidate_probability": candidate,
            "routed_probability": routed, "c4_probability": c4,
            "missing_stream_mask": missing}


def verify(data_dir: Path | None = None) -> dict[str, Any]:
    frozen = verify_freeze(data_dir)
    result_path = REPORT_DIR / "full_six_results.json"
    report = json.loads(result_path.read_text(encoding="utf-8"))
    if report.get("status") != "FULL_SIX_COMPLETE_REVIEW_REQUIRED" or report.get("freeze_signature") != frozen["signature"]:
        raise AssertionError("Stage 2 report status/freeze signature differs")
    if report.get("test_read") is not False or report.get("kaggle_submission") is not False:
        raise AssertionError("Stage 2 report violates Test/Kaggle boundary")
    if report.get("new_fit_count") != 12 or report.get("new_fit_endpoints") != NEW_FIT_ENDS:
        raise AssertionError("Stage 2 did not perform exactly the 12 registered new fits")
    frame, y, train = load_train(data_dir)
    if file_sha256(train) != frozen["identity"]["classification_train_sha256"]:
        raise AssertionError("Official Train hash mismatch")
    cycle = pd.to_numeric(frame.cycle_code).to_numpy(dtype=int)
    windows: dict[str, dict[str, dict[str, np.ndarray]]] = {name: {} for name in PRIMARY_WINDOWS}
    checked_artifacts = []
    for name, (fit_end, start, end) in PRIMARY_WINDOWS.items():
        for recipe in CANDIDATES:
            record = _load_window(frame, recipe, name, fit_end, start, end)
            rows = record["rows"]
            if not np.array_equal(rows, _expected_rows(frame, start - 1, end)):
                raise AssertionError(f"{name}/{recipe}: wrong primary validation rows")
            missing_live = stream_missing_mask(frame.iloc[rows])
            if not np.array_equal(missing_live, record["missing_stream_mask"]):
                raise AssertionError(f"{name}/{recipe}: missing-stream mask differs from Train")
            expected_route = np.where(missing_live, record["c4_probability"], record["candidate_probability"])
            if not np.array_equal(record["routed_probability"], expected_route):
                raise AssertionError(f"{name}/{recipe}: route is not candidate-present/C1_B-missing")
            if not np.array_equal(record["routed_probability"][missing_live], record["c4_probability"][missing_live]):
                raise AssertionError(f"{name}/{recipe}: missing-stream predictions differ from C4")
            windows[name][recipe] = record
            for label, values in (("candidate", record["routed_probability"]), ("C4_HGB", record["c4_probability"])):
                _assert_metric(report["window_metrics"][name][recipe][label], y[rows], values,
                               f"{name}/{recipe}/{label}")
            checked_artifacts.append(f"{name}/{recipe}")
            if fit_end in NEW_FIT_ENDS:
                meta_path = RUN_DIR / f"fit{fit_end}__{recipe}.json"
                cache_path = RUN_DIR / f"fit{fit_end}__{recipe}.npz"
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
                if meta.get("cache_sha256") != file_sha256(cache_path):
                    raise AssertionError(f"fit{fit_end}/{recipe}: archive checksum mismatch")
                fit_rows = np.flatnonzero((cycle >= 47) & (cycle <= fit_end))
                features = meta["features"]
                expected_fingerprint = _fit_fingerprint(frame.iloc[fit_rows][features].copy(), tuple(features), y[fit_rows])
                if meta["fit_state"]["fit_data_sha256"] != expected_fingerprint:
                    raise AssertionError(f"fit{fit_end}/{recipe}: fit-only data boundary mismatch")
                if meta["fit_state"]["fit_rows"] != len(fit_rows) or meta["outer_labels_used_for_fit_or_selection"] is not False:
                    raise AssertionError(f"fit{fit_end}/{recipe}: fit-row/label-boundary audit mismatch")
                with np.load(cache_path, allow_pickle=False) as archive:
                    if not np.array_equal(archive["outer_rows"], rows):
                        raise AssertionError(f"fit{fit_end}/{recipe}: cached row IDs mismatch")
                    for field, expected in (("candidate_probability", record["candidate_probability"]),
                                            ("routed_probability", record["routed_probability"]),
                                            ("c4_probability", record["c4_probability"])):
                        if not np.array_equal(archive[field], expected):
                            raise AssertionError(f"fit{fit_end}/{recipe}: cached {field} mismatch")
                    if recipe == "DC_SVC":
                        margin = archive["decision_margin"].astype(float, copy=True)
                        if not np.array_equal(margin >= 0, record["candidate_probability"] >= 0.5):
                            raise AssertionError(f"fit{fit_end}/{recipe}: fixed margin-zero rule differs")
                checked_artifacts.append(f"fit{fit_end}/{recipe}/audit")

    full_vectors = {}
    for recipe in CANDIDATES:
        pieces = [windows[name][recipe] for name in PRIMARY_WINDOWS]
        rows = np.concatenate([part["rows"] for part in pieces])
        candidate = np.concatenate([part["candidate_probability"] for part in pieces])
        routed = np.concatenate([part["routed_probability"] for part in pieces])
        c4 = np.concatenate([part["c4_probability"] for part in pieces])
        missing = np.concatenate([part["missing_stream_mask"] for part in pieces])
        if len(np.unique(rows)) != len(rows) or not np.array_equal(np.sort(rows), np.flatnonzero(pd.Series(cycle).between(56, 79).to_numpy())):
            raise AssertionError(f"{recipe}: full-six validation coverage differs")
        full_vectors[recipe] = {"rows": rows, "candidate_probability": candidate,
                                "routed_probability": routed, "c4_probability": c4,
                                "missing_stream_mask": missing}
        record = report["full_primary"][recipe]
        _assert_metric(record["candidate"], y[rows], routed, f"full/{recipe}/candidate")
        _assert_metric(record["C4_HGB"], y[rows], c4, f"full/{recipe}/C4_HGB")
        if not np.isclose(record["macro_f1_delta"], record["candidate"]["macro_f1"] - record["C4_HGB"]["macro_f1"], atol=1e-12, rtol=0):
            raise AssertionError(f"full/{recipe}: Macro F1 delta does not reconcile")
        for subgroup, mask in (("stream_present", ~missing), ("stream_missing", missing)):
            _assert_metric(report["subgroups"][recipe][subgroup]["candidate"], y[rows[mask]], routed[mask],
                           f"full/{recipe}/{subgroup}/candidate")
            _assert_metric(report["subgroups"][recipe][subgroup]["C4_HGB"], y[rows[mask]], c4[mask],
                           f"full/{recipe}/{subgroup}/C4_HGB")
        present = ~missing
        paired = complementarity_counts(y[rows[present]], candidate[present], c4[present])
        for key, value in paired.items():
            if report["complementarity"][recipe][key] != value:
                raise AssertionError(f"full/{recipe}: paired error count {key} differs")
        if blend_gate(paired) != report["candidate_gate"][recipe]["blend_gate_passed"]:
            raise AssertionError(f"full/{recipe}: fixed blend gate differs")

    # Independent recomputation of H12 from the immutable Stage 1 candidate caches.
    for recipe in CANDIDATES:
        for label, fit_end, start, end in (("H12_E", 55, 56, 67), ("H12_L", 67, 68, 79)):
            path = STAGE1_RUN_DIR / f"fit{fit_end}__{recipe}.npz"
            with np.load(path, allow_pickle=False) as archive:
                rows = archive["outer_rows"].astype(np.int64, copy=True)
                routed = archive["routed_probability"].astype(float, copy=True)
                c4 = archive["c4_probability"].astype(float, copy=True)
            selected = (cycle[rows] >= start) & (cycle[rows] <= end)
            current = report["h12"][recipe][label]
            _assert_metric(current["candidate"], y[rows[selected]], routed[selected], f"{recipe}/{label}/candidate")
            _assert_metric(current["C4_HGB"], y[rows[selected]], c4[selected], f"{recipe}/{label}/C4_HGB")

    blend = report["fixed_50_50_blend"]
    best = max(CANDIDATES, key=lambda recipe: report["full_primary"][recipe]["candidate"]["macro_f1"])
    if blend.get("best_candidate") != best:
        raise AssertionError("Blend diagnostic did not use the top full-six pooled Macro F1 candidate")
    gate_passed = report["candidate_gate"][best]["blend_gate_passed"]
    if blend.get("evaluated") != gate_passed:
        raise AssertionError("Blend execution does not match the fixed complementarity gate")
    if gate_passed:
        record = full_vectors[best]
        rows = record["rows"]
        present = ~record["missing_stream_mask"]
        probability = np.where(present, 0.5 * record["candidate_probability"] + 0.5 * record["c4_probability"],
                              record["c4_probability"])
        _assert_metric(blend["metrics"], y[rows], probability, "fixed_blend")

    chart_paths = [REPORT_DIR / name for name in ("full_six_window_macro_f1.png", "paired_present_errors.png")]
    for chart in chart_paths:
        if report["charts"].get(chart.name) != file_sha256(chart):
            raise AssertionError(f"Chart checksum mismatch: {chart.name}")
    verification = {
        "status": "PASS",
        "freeze_signature": frozen["signature"],
        "independent_metric_implementation": "sklearn confusion_matrix, fixed-label precision_recall_fscore_support/f1_score, average_precision_score, and direct Brier computation",
        "verified_window_candidate_pairs": len(checked_artifacts),
        "new_fit_artifacts_expected": 12,
        "verified_new_fit_ids": [f"fit{end}__{recipe}" for end in NEW_FIT_ENDS for recipe in CANDIDATES],
        "blend_gate_evaluated": gate_passed,
        "official_test_read": False,
        "kaggle_submission": False,
        "train_sha256": file_sha256(train),
    }
    write_json(REPORT_DIR / "full_six_verification.json", verification)
    return verification


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path)
    args = parser.parse_args()
    result = verify(args.data_dir)
    print(result["status"], result["verified_new_fit_ids"])


if __name__ == "__main__":
    main()
